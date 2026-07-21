"""
check_classifier_localization.py
=================================
Grad-CAM diagnostic: visualizes WHERE the classifier looks when deciding
"pneumonia" on real, known-positive images.

WHAT TO LOOK FOR IN THE OUTPUT:
  - GOOD (localized): heat concentrated in a specific lung region —
    consistent with an actual opacity/consolidation pattern.
  - BAD (diffuse/shortcut): heat spread across the whole thorax, edges,
    shoulders, or background — suggests the classifier is using a global
    shortcut (brightness/contrast/exposure) rather than a real lesion.

Works with either classifier (auto-detects the last Conv2d layer, so it
works for both the ResNet-18 and the XRVPneumoniaClassifier without
changes) — just point CLASSIFIER_MODE at whichever you want to check.

Usage:
    python check_classifier_localization.py
Then look at the saved PNGs in outputs/gradcam_check/.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

import config
from ingestion import build_dataset
from src.models import build_classifier

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CLASSIFIER_MODES = ["current", "xrv"]   # runs both automatically, no manual edits needed
N_IMAGES         = 6                     # how many positive-labeled images to check
OUT_DIR          = "outputs/gradcam_check"


def get_classifier(mode: str, device: str):
    if mode == "current":
        return build_classifier(device=device)
    elif mode == "xrv":
        from src.xrv_classifier import build_xrv_classifier
        return build_xrv_classifier(device=device)
    else:
        raise ValueError(f"Unknown CLASSIFIER_MODE '{mode}', expected 'current' or 'xrv'")


def find_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    """
    Find the best Grad-CAM hook target.

    FIX #2: for DenseNet-style architectures, a single internal dense-layer
    Conv2d (e.g. the 128->32 bottleneck conv inside the last dense block)
    only contributes a narrow slice (e.g. 32 of ~1024) of the channels
    that actually reach the classifier — DenseNet concatenates every dense
    layer's output onto a growing stack. Hooking one internal conv misses
    most of what the classifier actually uses, which produces a
    near-uniform gradient map REGARDLESS of prediction confidence —
    consistent with seeing flat maps at f(x)=0.991, 0.572, and 0.760 alike.

    The correct Grad-CAM target for DenseNet is the `features` submodule's
    final output (the full concatenated feature stack, right before
    global pooling + classifier) — this is standard practice for
    DenseNet/torchvision-style architectures. This function looks for a
    submodule literally named "features" first; only falls back to the
    old last-spatial-conv heuristic (e.g. for ResNet-18, which has no
    "features" submodule and works fine with layer4's conv) if none is
    found.
    """
    named = dict(model.named_modules())
    for name, m in named.items():
        if name.split(".")[-1] == "features" and isinstance(m, torch.nn.Module):
            print(f"[check] found dedicated feature-extractor submodule: '{name}' "
                  f"— hooking its full output (standard DenseNet Grad-CAM target)")
            return m

    print("[check] no 'features' submodule found — falling back to last "
          "spatial Conv2d heuristic (expected path for ResNet-18)")
    last_spatial_conv = None
    last_any_conv      = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            last_any_conv = m
            if m.kernel_size != (1, 1):
                last_spatial_conv = m

    if last_spatial_conv is not None:
        return last_spatial_conv
    if last_any_conv is not None:
        print("[check] WARNING: only found 1x1 conv layers — Grad-CAM will "
              "have poor/no spatial resolution.")
        return last_any_conv
    raise RuntimeError("No usable layer found for Grad-CAM.")


def grad_cam(model, target_layer, x: torch.Tensor) -> np.ndarray:
    """
    x : (1, C, H, W), already the exact input the model expects, requires_grad
    Returns a (H, W) numpy heatmap normalised to [0, 1].

    FIX: originally used register_full_backward_hook to capture gradients
    at target_layer's output. TorchXRayVision's DenseNet.forward() runs
    F.relu(features, inplace=True) immediately after computing
    self.features(x) — an in-place op on the exact tensor our backward
    hook wraps. PyTorch forbids this combination (a backward hook wraps
    the output in a custom autograd Function; an in-place op right after
    conflicts with that wrapping and raises a RuntimeError). This is
    internal to the torchxrayvision library, not something we can patch
    from outside.

    Fix: don't use a backward hook at all. Instead, use a forward hook to
    capture the activation tensor and call .retain_grad() on it directly
    — this makes PyTorch populate its .grad attribute after backward(),
    with no custom-Function wrapping involved, so subsequent in-place ops
    downstream (like torchxrayvision's ReLU) are handled by ordinary
    autograd versioning and don't conflict.
    """
    activations = {}

    def fwd_hook(module, inp, out):
        out.retain_grad()
        activations["value"] = out

    h1 = target_layer.register_forward_hook(fwd_hook)

    model.zero_grad()
    out = model(x)
    out.sum().backward()

    h1.remove()

    act_tensor = activations["value"]
    if act_tensor.grad is None:
        raise RuntimeError(
            "activations.grad is None after backward() — the hooked layer's "
            "output may not be part of the path from input to the model's "
            "final output. Check that target_layer is actually used in the "
            "forward pass for this input."
        )

    act  = act_tensor[0]        # (C, h, w) — note: reflects value AFTER any
                                  # downstream in-place ops (e.g. the ReLU),
                                  # which is standard/expected for Grad-CAM
    grad = act_tensor.grad[0]    # (C, h, w)
    weights = grad.mean(dim=(1, 2))  # (C,) — global-avg-pool the gradients

    cam = torch.relu((weights[:, None, None] * act).sum(dim=0))  # (h, w)
    cam = cam.detach().cpu().numpy()
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam


def main():
    device = config.DEVICE
    os.makedirs(OUT_DIR, exist_ok=True)

    print("[check] loading dataset to find known-positive images ...")
    dataset = build_dataset(img_size=config.IMG_SIZE)

    # collect N positive-labeled real images ONCE, reused for every mode
    # so all classifiers are compared on the exact same images
    positives = []
    for i in range(len(dataset)):
        x, label = dataset[i]
        if float(label.item() if torch.is_tensor(label) else label) == 1.0:
            positives.append(x)
        if len(positives) >= N_IMAGES:
            break
    print(f"[check] found {len(positives)} positive images to check "
          f"(same images will be reused across all classifier modes)")

    for mode in CLASSIFIER_MODES:
        print(f"\n{'='*70}\n[check] MODE = {mode}\n{'='*70}")
        print(f"[check] loading classifier (mode={mode}) ...")
        classifier = get_classifier(mode, device)
        classifier.eval()
        target_layer = find_target_layer(classifier)
        print(f"[check] hooking layer: {target_layer}")

        for idx, x in enumerate(positives):
            x_in = x.unsqueeze(0).to(device)
            x3   = x_in.repeat(1, 3, 1, 1) if x_in.shape[1] == 1 else x_in
            x3.requires_grad_(True)

            with torch.no_grad():
                pred = classifier(x3).item()

            cam = grad_cam(classifier, target_layer, x3)
            cam_resized = torch.nn.functional.interpolate(
                torch.tensor(cam)[None, None], size=x.shape[-2:], mode="bilinear", align_corners=False
            )[0, 0].numpy()

            img_np = x[0].detach().cpu().numpy()  # (H, W) grayscale

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(img_np, cmap="gray")
            axes[0].set_title(f"Original | f(x)={pred:.3f}")
            axes[0].axis("off")

            axes[1].imshow(img_np, cmap="gray")
            axes[1].imshow(cam_resized, cmap="jet", alpha=0.45)
            axes[1].set_title(f"Grad-CAM [{mode}] (where the classifier looks)")
            axes[1].axis("off")

            out_path = os.path.join(OUT_DIR, f"gradcam_{mode}_{idx}.png")
            plt.tight_layout()
            plt.savefig(out_path, dpi=120)
            plt.close()
            print(f"[check] saved {out_path}")

        # free GPU memory before loading the next classifier
        del classifier
        torch.cuda.empty_cache()

    print(f"\n[check] done. Open the PNGs in {OUT_DIR}/ — files are named "
          f"gradcam_<mode>_<idx>.png for direct current-vs-xrv comparison "
          f"on the same images (same idx = same underlying image).")


if __name__ == "__main__":
    main()
