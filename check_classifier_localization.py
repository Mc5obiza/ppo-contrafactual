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

MODE "ensemble" additionally diagnoses WHY USE_GRADCAM_MASK measurably
lowered lesion-targeting lift (config.py's ARM 2 finding): it exercises the
SAME src.gradcam.ensemble_grad_cam path compute_gradcam_latent_mask calls
during PPO, over a larger, full-probability-range sample, and tests two
questions directly against the data rather than by inspection:
  1. Does the rsna member degenerate more often than chex, concentrated near
     predicted probability ~0.5 (an all-negative-contribution ReLU zeroing,
     per the hypothesis in factory/ppo.py's degeneracy-summary comment) —
     or near 0/1 (output saturation, per gradcam.DegenerateCAMError's own
     docstring)?
  2. Does the averaged ensemble CAM localize to the annotated lesion box
     worse when partially degenerate than when clean, and does a finer
     target layer (features.denseblock3, via GRADCAM_TARGET_LAYER_OVERRIDE)
     change that?

Usage:
    python check_classifier_localization.py
    GRADCAM_TARGET_LAYER_OVERRIDE=features.denseblock3 python check_classifier_localization.py
Then look at the saved PNGs and degeneracy_log.json in outputs/gradcam_check/.
"""

import json
import os
import random

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
from src.gradcam import find_target_layer, grad_cam, ensemble_grad_cam, DegenerateCAMError

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CLASSIFIER_MODES  = ["current", "xrv"]   # runs both automatically, no manual edits needed
N_IMAGES          = 6                     # how many positive-labeled images to check
OUT_DIR           = "outputs/gradcam_check"

RUN_ENSEMBLE_DIAGNOSTIC = True   # the larger degeneracy/lesion-overlap probe, see module docstring
ENSEMBLE_N_IMAGES       = 150     # sampled across the FULL probability range, not just positives
GRADCAM_TARGET_LAYER    = os.environ.get(
    "GRADCAM_TARGET_LAYER_OVERRIDE", config.GRADCAM_TARGET_LAYER
)


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


def _cam_lesion_lift(cam_resized: np.ndarray, mask: torch.Tensor) -> float | None:
    """
    Same lift semantics as paired_arm_test.py's _per_image_lift (fraction of
    mass inside the box, divided by the null baseline = box area fraction),
    applied to a Grad-CAM heatmap instead of an edit delta. >1 means the CAM
    concentrates inside the annotated lesion more than chance; 1.0 is
    chance; <1 means it avoids the box. None when the box covers all or none
    of the image (undefined null), matching paired_arm_test.py's guard.
    """
    m = mask.squeeze(0).numpy().astype(np.float32)
    null = float(m.mean())
    if null <= 0.0 or null >= 1.0:
        return None
    total = float(cam_resized.sum())
    if total <= 0:
        return None
    return (float((cam_resized * m).sum()) / total) / null


def run_ensemble_diagnostic(device: str):
    """
    Larger-scale probe of the ensemble Grad-CAM path used during PPO
    training (USE_GRADCAM_MASK), testing the two hypotheses in this
    module's docstring: does degeneracy concentrate by member and by
    predicted probability, and does partial degeneracy correlate with worse
    lesion localisation? See the module docstring for why this matters
    (ARM 2 measurably LOWERED lesion-targeting lift, the opposite of its
    design intent, and this was never diagnosed).
    """
    from src.xrv_classifier import build_ensemble_classifier

    print(f"\n{'=' * 70}\n[check] MODE = ensemble diagnostic "
          f"(target_layer={GRADCAM_TARGET_LAYER})\n{'=' * 70}")

    dataset = build_dataset(img_size=config.IMG_SIZE)
    classifier = build_ensemble_classifier(device=device)
    classifier.eval()
    member_names = [w.replace("densenet121-res224-", "") for w in classifier.weights_list]

    # Full-probability-range sample for the degeneracy analysis: a random
    # cross-section of the dataset regardless of label. dataset.samples is
    # the cheap (patient_id, label) list — reading it avoids decoding every
    # DICOM in the dataset just to check labels (dataset[i] does that work).
    # The hypothesis under test is specifically about behaviour near
    # predicted probability ~0.5, which a positives-only sample under-
    # represents (positives skew toward high pneumonia probability).
    rng = random.Random(0)
    all_idx = list(range(len(dataset)))
    rng.shuffle(all_idx)
    sample_idx = all_idx[:ENSEMBLE_N_IMAGES]

    # Separate known-positive-with-box subset for the lesion-overlap arm —
    # negatives carry no annotation (ingestion.RSNACXRDataset.lesion_boxes).
    positive_idx_pool = [i for i in all_idx if dataset.samples[i][1] == 1.0]
    positive_boxed_idx = [
        i for i in positive_idx_pool if dataset.lesion_boxes(i)
    ][:ENSEMBLE_N_IMAGES]

    log = []
    lifts_clean, lifts_partial = [], []

    def _process(idx):
        x, _ = dataset[idx]
        x_in = x.unsqueeze(0).to(device)
        x3 = x_in.repeat(1, 3, 1, 1) if x_in.shape[1] == 1 else x_in

        with torch.no_grad():
            per_member = classifier.per_member(x3).squeeze(0).cpu().numpy()  # (M,)

        x3g = x3.clone().requires_grad_(True)
        cam_np = None
        with torch.enable_grad():
            try:
                cam, degenerate = ensemble_grad_cam(classifier, x3g, GRADCAM_TARGET_LAYER)
                cam_np = cam.detach().cpu().numpy()
            except DegenerateCAMError:
                degenerate = [(i, "ALL_DEGENERATE") for i in range(len(classifier.members))]

        degenerate_idx = {i for i, _ in degenerate}
        for m_i, name in enumerate(member_names):
            log.append({
                "idx": idx, "member": name, "prob": float(per_member[m_i]),
                "degenerate": m_i in degenerate_idx,
            })

        if cam_np is not None and dataset.lesion_boxes(idx):
            cam_t = torch.from_numpy(cam_np)[None, None]
            cam_resized = F.interpolate(
                cam_t, size=x.shape[-2:], mode="bilinear", align_corners=False
            )[0, 0].numpy()
            mask = dataset.lesion_mask(idx)
            lift = _cam_lesion_lift(cam_resized, mask)
            if lift is not None:
                (lifts_partial if degenerate_idx else lifts_clean).append(lift)

    print(f"[check] degeneracy probe: {len(sample_idx)} images, full probability range")
    for i, idx in enumerate(sample_idx):
        _process(idx)
        if (i + 1) % 50 == 0:
            print(f"[check]   {i + 1}/{len(sample_idx)}")

    already = set(sample_idx)
    remaining_boxed = [i for i in positive_boxed_idx if i not in already]
    print(f"[check] lesion-overlap probe: {len(positive_boxed_idx)} known-positive, "
          f"boxed images ({len(remaining_boxed)} not already covered above)")
    for idx in remaining_boxed:
        _process(idx)

    os.makedirs(OUT_DIR, exist_ok=True)
    log_path = os.path.join(OUT_DIR, "degeneracy_log.json")
    with open(log_path, "w") as f:
        json.dump({"target_layer": GRADCAM_TARGET_LAYER, "entries": log}, f, indent=2)
    print(f"[check] wrote {log_path} ({len(log)} member-image entries)")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}\n[check] SUMMARY (target_layer={GRADCAM_TARGET_LAYER})\n{'=' * 70}")

    print("  Degeneracy rate by member:")
    for name in member_names:
        entries = [e for e in log if e["member"] == name]
        n = len(entries)
        n_deg = sum(e["degenerate"] for e in entries)
        if n:
            print(f"    {name:8s}: {n_deg}/{n} ({100.0 * n_deg / n:.1f}%)")
        else:
            print(f"    {name}: no data")

    print("\n  Degeneracy rate by predicted probability bin (per member) — "
          "tests whether it clusters near 0.5 (all-negative-contribution) "
          "or near 0/1 (output saturation):")
    bins = [(0.0, 0.1), (0.1, 0.3), (0.3, 0.4), (0.4, 0.6), (0.6, 0.7), (0.7, 0.9), (0.9, 1.001)]
    for name in member_names:
        entries = [e for e in log if e["member"] == name]
        row = []
        for lo, hi in bins:
            in_bin = [e for e in entries if lo <= e["prob"] < hi]
            if in_bin:
                rate = 100.0 * sum(e["degenerate"] for e in in_bin) / len(in_bin)
                row.append(f"[{lo:.1f},{min(hi, 1.0):.1f})={rate:4.0f}%(n={len(in_bin)})")
            else:
                row.append(f"[{lo:.1f},{min(hi, 1.0):.1f})=  n/a")
        print(f"    {name:8s}: " + "  ".join(row))

    def _fmt_lift(vals):
        if not vals:
            return "n/a"
        arr = np.array(vals)
        sd = arr.std(ddof=1) if len(arr) > 1 else 0.0
        return f"mean={arr.mean():.3f} (n={len(arr)}, sd={sd:.3f})"

    print("\n  CAM-lesion lift (>1 = concentrates inside the annotated box "
          "more than chance; 1.0 = chance):")
    print(f"    clean episodes            : {_fmt_lift(lifts_clean)}")
    print(f"    partially-degenerate      : {_fmt_lift(lifts_partial)}")

    del classifier
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
    if RUN_ENSEMBLE_DIAGNOSTIC:
        run_ensemble_diagnostic(config.DEVICE)
