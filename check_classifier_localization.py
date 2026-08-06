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
# find_target_layer / grad_cam now live in src/gradcam.py so that this
# diagnostic and the Grad-CAM action-space constraint in generator.py
# (compute_gradcam_latent_mask) share ONE implementation — otherwise this
# script stops being evidence about the mask PPO is actually constrained by.
from src.gradcam import find_target_layer, grad_cam

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

            # src.gradcam.grad_cam returns a torch Tensor (h, w) on device,
            # already min-max normalised to [0, 1].
            cam = grad_cam(classifier, target_layer, x3)
            cam_resized = F.interpolate(
                cam[None, None], size=x.shape[-2:], mode="bilinear", align_corners=False
            )[0, 0].cpu().numpy()

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
