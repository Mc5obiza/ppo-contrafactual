"""
check_lesion_localization.py
============================
The gate that has to pass before any LLR / MSI number means anything.

Sibling of check_classifier_localization.py, and the one place ground-truth ROI
masks are used at evaluation time — legitimately, because this script indexes
the dataset directly instead of going through metrics.scan_groups (which
returns tensors with no indices, under a determinism contract three entry
points depend on).

FOUR CHECKS
-----------
1. ANATOMY MASK SANITY
   BreastAnatomySegmenter's breast mask must be non-degenerate on real images.
   Both failures are silent: an all-zero mask leaves the policy nothing it is
   allowed to edit and every episode no-ops, an all-one mask quietly disables
   LOCK_BACKGROUND and lets edits land in the black background.

2. LESION SEGMENTER DICE against ground truth.
   LLR and MSI are computed FROM this net, so its Dice bounds how much they can
   possibly mean.

3. MSI GROUND-TRUTH CALIBRATION — the important one.
   MSI claims to measure margin spiculation. CBIS-DDSM labels margins directly
   (`mass margins`: SPICULATED / ILL_DEFINED vs CIRCUMSCRIBED), so the claim is
   testable on REAL images with no counterfactuals involved: MSI must be higher
   on spiculated lesions than on circumscribed ones. If it cannot separate the
   two on ground truth, it cannot be evidence about counterfactuals, and any
   p-value it later produces is a property of the segmenter rather than a
   finding.

   This is exactly the check SCP never had. SCP shipped a saturated constant
   for several evaluation runs while dutifully emitting paired t-tests, and
   nothing caught it until someone read the standard deviation.

4. GRAD-CAM vs GROUND TRUTH for f(.).
   Fraction of classifier attention landing inside the true lesion. The
   mammography version of the shortcut check: a classifier keying on the
   pectoral edge, the skin line or overall density will score near zero here
   while still reporting a healthy AUC.

Usage
-----
    python check_lesion_localization.py
    python check_lesion_localization.py --n 200
Then read the printed verdicts; PNGs land in outputs/lesion_check/.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

import config
import domains

OUT_DIR = "outputs/lesion_check"

# Thresholds for the printed verdicts. Deliberately modest — these are "is this
# instrument connected to reality" gates, not quality targets.
MIN_DICE = 0.40
MIN_CAM_OVERLAP = 0.15


def _load_gt_mask(path: str, size: int) -> np.ndarray:
    from PIL import Image
    m = Image.open(path).convert("L").resize((size, size), Image.NEAREST)
    return (np.asarray(m).astype(np.float32) / 255.0 > 0.5).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────

def check_anatomy(dataset, seg_net, idxs, device) -> bool:
    from src.metrics import _seg_channel

    b_idx = _seg_channel(seg_net, "Breast")
    d_idx = _seg_channel(seg_net, "Dense Tissue")
    cov_b, cov_d = [], []
    for i in idxs:
        x, _ = dataset[i]
        with torch.no_grad():
            p = torch.sigmoid(seg_net(x.unsqueeze(0).to(device) * 2048.0 - 1024.0))
        cov_b.append(float((p[0, b_idx] >= 0.5).float().mean()))
        if d_idx is not None:
            cov_d.append(float((p[0, d_idx] >= 0.5).float().mean()))

    mb = float(np.mean(cov_b))
    print(f"\n[1] ANATOMY MASK")
    print(f"    breast coverage : mean {mb:.3f}  "
          f"[min {np.min(cov_b):.3f}, max {np.max(cov_b):.3f}]")
    if cov_d:
        print(f"    dense  coverage : mean {np.mean(cov_d):.3f}")

    ok = True
    n_empty = sum(1 for c in cov_b if c < 0.02)
    n_full  = sum(1 for c in cov_b if c > 0.98)
    if n_empty:
        print(f"    ✗ {n_empty} image(s) produced an essentially EMPTY breast "
              f"mask — the policy would have nothing it may edit there.")
        ok = False
    if n_full:
        print(f"    ✗ {n_full} image(s) produced an essentially FULL-FRAME mask "
              f"— LOCK_BACKGROUND is a no-op there and edits can land in the "
              f"black background.")
        ok = False
    if ok:
        print(f"    ✓ non-degenerate on all {len(idxs)} images")
    return ok


def check_lesion_dice(dataset, lesion_net, idxs, device) -> tuple[bool, list]:
    dices, keep = [], []
    for i in idxs:
        gt_path = dataset.roi_mask_path(i)
        if not gt_path or not os.path.exists(gt_path):
            continue
        x, _ = dataset[i]
        gt = _load_gt_mask(gt_path, x.shape[-1])
        with torch.no_grad():
            p = (torch.sigmoid(lesion_net(x.unsqueeze(0).to(device)))[0, 0] >= 0.5)
        p = p.float().cpu().numpy()
        denom = p.sum() + gt.sum()
        dices.append(1.0 if denom == 0 else float(2.0 * (p * gt).sum() / denom))
        keep.append(i)

    print(f"\n[2] LESION SEGMENTER (vs ground-truth ROI, n={len(dices)})")
    if not dices:
        print("    ✗ no ground-truth masks resolved — cannot evaluate.")
        return False, keep
    md = float(np.mean(dices))
    print(f"    mean Dice = {md:.3f}   [median {np.median(dices):.3f}]")
    ok = md >= MIN_DICE
    print(f"    {'✓' if ok else '✗'} {'above' if ok else 'BELOW'} the "
          f"{MIN_DICE:.2f} floor — LLR and MSI are derived from this net, so "
          f"below it they measure the segmenter, not the edits.")
    return ok, keep


def check_msi_calibration(dataset, seg_net, lesion_net, idxs, device) -> bool:
    """
    THE gate for MSI. See this module's docstring.

    Groups REAL images by the dataset's own `mass margins` label and compares
    MSI between them. No counterfactuals are involved: if MSI cannot tell a
    spiculated lesion from a circumscribed one on ground truth, it is not
    measuring spiculation.
    """
    from scipy import stats as scipy_stats
    from src.metrics import compute_msi_batch

    SPIC = ("SPICULATED", "ILL_DEFINED", "ILL-DEFINED")
    CIRC = ("CIRCUMSCRIBED",)

    groups: dict[str, list[int]] = {"spiculated": [], "circumscribed": []}
    for i in idxs:
        m = (dataset.margin_label(i) or "").upper()
        if any(s in m for s in SPIC):
            groups["spiculated"].append(i)
        elif any(s in m for s in CIRC):
            groups["circumscribed"].append(i)

    print(f"\n[3] MSI CALIBRATION against ground-truth `mass margins`")
    print(f"    spiculated/ill-defined: {len(groups['spiculated'])}  |  "
          f"circumscribed: {len(groups['circumscribed'])}")

    if len(groups["spiculated"]) < 5 or len(groups["circumscribed"]) < 5:
        print("    ✗ too few labelled lesions in one group to calibrate. MSI "
              "cannot be validated on this subset — increase --n.")
        return False

    vals = {}
    for name, gidx in groups.items():
        xs = [dataset[i][0].unsqueeze(0) for i in gidx]
        v = np.array(compute_msi_batch(xs, seg_net, lesion_net, device), dtype=float)
        vals[name] = v[~np.isnan(v)]

    s, c = vals["spiculated"], vals["circumscribed"]
    if len(s) < 5 or len(c) < 5:
        print("    ✗ too many NaNs (no lesion segmented) to calibrate.")
        return False

    t, p = scipy_stats.ttest_ind(s, c, equal_var=False)
    print(f"    MSI spiculated    = {s.mean():.3f} ± {s.std():.3f}  (n={len(s)})")
    print(f"    MSI circumscribed = {c.mean():.3f} ± {c.std():.3f}  (n={len(c)})")
    print(f"    Welch t = {t:.2f}, p = {p:.4g}")

    ok = bool(s.mean() > c.mean() and p < 0.05)
    if ok:
        print("    ✓ MSI separates the two on REAL images — it is measuring "
              "margin morphology, so it may be reported on counterfactuals.")
    else:
        print("    ✗ MSI does NOT separate spiculated from circumscribed on "
              "ground truth.")
        print("      It is therefore not measuring spiculation here, and any "
              "p-value it produces on")
        print("      counterfactuals is an artefact. Do NOT report MSI — drop "
              "it from")
        print("      DomainProfile.clinical_metrics, or fix the lesion "
              "segmenter and re-run.")
    return ok


def check_gradcam_overlap(dataset, classifier, idxs, device) -> bool:
    from src.gradcam import cached_target_layer, grad_cam

    model = getattr(classifier, "cam_member", classifier)
    layer = cached_target_layer(model, config.GRADCAM_TARGET_LAYER)

    overlaps = []
    for i in idxs:
        gt_path = dataset.roi_mask_path(i)
        if not gt_path or not os.path.exists(gt_path):
            continue
        x, _ = dataset[i]
        gt = _load_gt_mask(gt_path, x.shape[-1])
        if gt.sum() == 0:
            continue
        # grad_cam documents that x must arrive with requires_grad=True — it
        # backprops the score to the hooked activations, so a leaf without a
        # grad path yields an all-zero map rather than an error.
        xin = x.unsqueeze(0).to(device).repeat(1, 3, 1, 1).requires_grad_(True)
        cam = grad_cam(model, layer, xin)          # (h, w), detached, in [0,1]
        cam = F.interpolate(
            cam.view(1, 1, *cam.shape[-2:]).float(), size=gt.shape,
            mode="bilinear", align_corners=False,
        )[0, 0].cpu().numpy()
        total = cam.sum()
        if total > 0:
            overlaps.append(float((cam * gt).sum() / total))

    print(f"\n[4] GRAD-CAM vs ground-truth lesion (n={len(overlaps)})")
    if not overlaps:
        print("    ✗ could not compute — no usable ground-truth masks.")
        return False
    mo = float(np.mean(overlaps))
    # A lesion covers a small share of a mammogram, so a classifier attending
    # at random scores near that share; the floor is set above it.
    print(f"    mean attention mass inside the lesion = {mo:.3f}")
    ok = mo >= MIN_CAM_OVERLAP
    print(f"    {'✓' if ok else '✗'} {'above' if ok else 'BELOW'} the "
          f"{MIN_CAM_OVERLAP:.2f} floor.")
    if not ok:
        print("      f(.) is keying on something other than the lesion — the "
              "pectoral edge, the skin")
        print("      line, or overall density are the usual mammography "
              "shortcuts. A healthy AUC does")
        print("      not rule this out. See check_classifier_localization.py "
              "for per-image heatmaps.")
    return ok


# ─────────────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Validate the breast segmentation "
                                             "and metric stack against ground truth.")
    ap.add_argument("--n", type=int, default=120,
                    help="images to sample (default 120)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    device = config.DEVICE if torch.cuda.is_available() else "cpu"
    domain = domains.active()
    print(domain.summary())

    if domain.build_lesion_net is None:
        raise SystemExit(
            f"domain {domain.name!r} has no lesion segmenter, so there is "
            f"nothing here to check. This script is for cbis_ddsm_breast."
        )

    os.makedirs(OUT_DIR, exist_ok=True)
    dataset = domain.build_dataset(config.IMG_SIZE)

    rng = np.random.default_rng(args.seed)
    idxs = sorted(rng.choice(len(dataset), size=min(args.n, len(dataset)),
                             replace=False).tolist())

    seg_net, _ = domain.build_masker()
    lesion_net = domain.build_lesion_net()
    classifier = domain.build_guidance()

    results = {
        "anatomy mask":      check_anatomy(dataset, seg_net, idxs, device),
        "lesion Dice":       check_lesion_dice(dataset, lesion_net, idxs, device)[0],
        "MSI calibration":   check_msi_calibration(dataset, seg_net, lesion_net,
                                                   idxs, device),
        "Grad-CAM overlap":  check_gradcam_overlap(dataset, classifier, idxs, device),
    }

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")

    if not all(results.values()):
        print("\nAt least one check failed. Each one gates a specific claim:")
        print("  anatomy mask      -> whether LOCK_BACKGROUND constrains anything")
        print("  lesion Dice       -> whether LLR/MSI measure lesions at all")
        print("  MSI calibration   -> whether MSI may be reported, full stop")
        print("  Grad-CAM overlap  -> whether f(.) looks at lesions or a shortcut")
        print("Fix the failing component before quoting the metric it gates.")
        return 1

    print("\n✓ All checks passed — the metric stack is measuring what it claims to.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
