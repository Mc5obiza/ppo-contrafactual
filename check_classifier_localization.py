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
# CONFIG — edit this line to switch classifiers
# ─────────────────────────────────────────────
CLASSIFIER_MODE = "current"   # "current" = your ResNet-18, "xrv" = the pretrained swap
N_IMAGES        = 6           # how many positive-labeled images to check
OUT_DIR         = "outputs/gradcam_check"


def get_classifier(mode: str, device: str):
    if mode == "current":
        return build_classifier(device=device)
    elif mode == "xrv":
        from src.xrv_classifier import build_xrv_classifier
        return build_xrv_classifier(device=device)
    else:
        raise ValueError(f"Unknown CLASSIFIER_MODE '{mode}', expected 'current' or 'xrv'")


def find_last_conv(model: torch.nn.Module) -> torch.nn.Conv2d:
    """Auto-detect the last Conv2d layer for Grad-CAM hooking — works for
    both ResNet-18 and the xrv DenseNet without hardcoding layer names."""
    last_conv = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            last_conv = m
    if last_conv is None:
        raise RuntimeError("No Conv2d layer found in model — can't run Grad-CAM.")
    return last_conv


def grad_cam(model, target_layer, x: torch.Tensor) -> np.ndarray:
    """
    x : (1, C, H, W), already the exact input the model expects, requires_grad
    Returns a (H, W) numpy heatmap normalised to [0, 1].
    """
    activations = {}
    gradients   = {}

    def fwd_hook(module, inp, out):
        activations["value"] = out

    def bwd_hook(module, grad_in, grad_out):
        gradients["value"] = grad_out[0]

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    model.zero_grad()
    out = model(x)
    # out is (1,1) probability — backprop directly (no need for a "class index"
    # since this is a single-scalar binary output)
    out.sum().backward()

    h1.remove()
    h2.remove()

    act  = activations["value"][0]        # (C, h, w)
    grad = gradients["value"][0]           # (C, h, w)
    weights = grad.mean(dim=(1, 2))        # (C,) — global-avg-pool the gradients

    cam = torch.relu((weights[:, None, None] * act).sum(dim=0))  # (h, w)
    cam = cam.detach().cpu().numpy()
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam


def main():
    device = config.DEVICE
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"[check] loading classifier (mode={CLASSIFIER_MODE}) ...")
    classifier = get_classifier(CLASSIFIER_MODE, device)
    classifier.eval()
    target_layer = find_last_conv(classifier)
    print(f"[check] hooking layer: {target_layer}")

    print("[check] loading dataset to find known-positive images ...")
    dataset = build_dataset(img_size=config.IMG_SIZE)

    # collect N positive-labeled real images
    positives = []
    for i in range(len(dataset)):
        x, label = dataset[i]
        if float(label.item() if torch.is_tensor(label) else label) == 1.0:
            positives.append(x)
        if len(positives) >= N_IMAGES:
            break
    print(f"[check] found {len(positives)} positive images to check")

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
        axes[1].set_title("Grad-CAM (where the classifier looks)")
        axes[1].axis("off")

        out_path = os.path.join(OUT_DIR, f"gradcam_{CLASSIFIER_MODE}_{idx}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=120)
        plt.close()
        print(f"[check] saved {out_path}")

    print(f"\n[check] done. Open the {N_IMAGES} PNGs in {OUT_DIR}/ and look for: "
          f"localized hotspot (good) vs diffuse whole-chest heat (shortcut concern).")


if __name__ == "__main__":
    main()
