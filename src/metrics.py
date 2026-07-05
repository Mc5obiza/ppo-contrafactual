"""
src/metrics.py
==============
Evaluation metrics from Singla et al., "Explaining the Black-box Smoothly —
A Counterfactual Approach", Medical Image Analysis (2022).

Implemented metrics
-------------------
1. Counterfactual Validity (CV) score          — Eq. 11
2. Fréchet Inception Distance (FID) score      — Eq. 10  (normalised per paper)
3. Foreign-Object Preservation (FOP) score     — Eq. 12
4. Cardio-Thoracic Ratio (CTR)                 — cardiomegaly clinical metric
5. Score for normal CostoPhrenic recess (SCP)  — pleural effusion clinical metric

All functions follow the exact notation of the paper.
"""

from __future__ import annotations

import os
import json
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy import linalg
from torchvision.models import inception_v3, Inception_V3_Weights
import torchvision.transforms as T

import config


# ─────────────────────────────────────────────────────────────────────────────
# 1. Counterfactual Validity (CV)  —  Eq. 11
# ─────────────────────────────────────────────────────────────────────────────

def counterfactual_validity(
    f_x: torch.Tensor,
    f_xc: torch.Tensor,
    tau: float = 0.8,
) -> float:
    """
    CV(X, Xc) = mean( |f(x) - f(xc)| > τ )

    Parameters
    ----------
    f_x  : (N,)  classifier scores on real images
    f_xc : (N,)  classifier scores on counterfactual images
    tau  : margin threshold (paper uses 0.8)

    Returns
    -------
    float  CV score ∈ [0, 1]
    """
    assert f_x.shape == f_xc.shape, "f_x and f_xc must have the same shape"
    valid = (f_x - f_xc).abs() > tau
    cv = valid.float().mean().item()
    return cv


# ─────────────────────────────────────────────────────────────────────────────
# 2. Fréchet Inception Distance (FID)  —  Eq. 10
# ─────────────────────────────────────────────────────────────────────────────

class _InceptionFeatureExtractor(nn.Module):
    """Penultimate-layer features of Inception-v3 (2048-d)."""

    def __init__(self, device: str):
        super().__init__()
        model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        model.fc = nn.Identity()      # drop final classifier
        model.aux_logits = False
        self.model = model.to(device).eval()
        self.device = device
        self.resize = T.Resize((299, 299), antialias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, 1, H, W) or (B, 3, H, W) → (B, 2048)"""
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = self.resize(x)
        with torch.no_grad():
            feat = self.model(x.to(self.device))
        return feat.cpu()


def _compute_activation_stats(
    images: list[torch.Tensor],
    extractor: _InceptionFeatureExtractor,
    batch_size: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (μ, Σ) of Inception activations for a list of images."""
    feats = []
    for i in range(0, len(images), batch_size):
        batch = torch.cat(images[i : i + batch_size], dim=0)
        feats.append(extractor(batch))
    feats = torch.cat(feats, dim=0).numpy()
    mu    = feats.mean(axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma


def frechet_inception_distance(
    X_real: list[torch.Tensor],
    X_cf:   list[torch.Tensor],
    device: str = config.DEVICE,
    batch_size: int = 16,
    reference_fid: Optional[float] = None,
) -> float:
    """
    FID(X, Xc) as defined in Eq. 10.

    Paper normalises all FID scores w.r.t. the best method (CycleGAN = 1).
    Pass ``reference_fid`` (the CycleGAN FID) to get the normalised value.

    Parameters
    ----------
    X_real : list of (1,1,H,W) tensors — real images
    X_cf   : list of (1,1,H,W) tensors — counterfactual images
    reference_fid : FID of the reference/best method for normalisation

    Returns
    -------
    float — raw FID (or normalised FID if reference_fid is given)
    """
    extractor = _InceptionFeatureExtractor(device)
    mu1, sigma1 = _compute_activation_stats(X_real, extractor, batch_size)
    mu2, sigma2 = _compute_activation_stats(X_cf,   extractor, batch_size)

    diff = mu1 - mu2
    # scipy>=1.15 dropped the `disp` kwarg and returns the array directly
    # instead of an (array, errest) tuple.
    covmean = linalg.sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = float(diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean))

    if reference_fid is not None and reference_fid > 0:
        fid = fid / reference_fid

    return fid


# ─────────────────────────────────────────────────────────────────────────────
# 3. Foreign-Object Preservation (FOP)  —  Eq. 12
# ─────────────────────────────────────────────────────────────────────────────

def foreign_object_preservation(
    obj_det,
    X_real: list[torch.Tensor],
    X_cf:   list[torch.Tensor],
    fo_threshold: float = 0.5,
    device: str = config.DEVICE,
) -> float:
    """
    FOP(X, Xc) = fraction of images where FO is detected in BOTH
                 the real image AND its counterfactual.

    Only images where O(x) > fo_threshold are considered (paper Eq. 12).

    Parameters
    ----------
    obj_det       : YOLOv8FeatureExtractor (or any model returning a scalar FO score)
    X_real        : list of (1,1,H,W) tensors
    X_cf          : list of (1,1,H,W) tensors
    fo_threshold  : detection confidence threshold (paper: 0.5)

    Returns
    -------
    float — FOP score ∈ [0, 1]
    """
    assert len(X_real) == len(X_cf)

    preserved, total = 0, 0
    for x, xc in zip(X_real, X_cf):
        x3  = (x.repeat(1, 3, 1, 1)  if x.shape[1]  == 1 else x).to(device)
        xc3 = (xc.repeat(1, 3, 1, 1) if xc.shape[1] == 1 else xc).to(device)

        with torch.no_grad():
            # obj_det returns feature vectors; use their L2-norm as a proxy
            # for FO presence (higher norm = richer feature = FO likely present)
            feat_x  = obj_det(x3).norm(dim=-1).item()
            feat_xc = obj_det(xc3).norm(dim=-1).item()

        fo_in_real = feat_x  > fo_threshold
        fo_in_cf   = feat_xc > fo_threshold

        if fo_in_real:
            total     += 1
            preserved += int(fo_in_cf)

    return preserved / max(total, 1)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Cardio-Thoracic Ratio (CTR)  —  cardiomegaly clinical metric
# ─────────────────────────────────────────────────────────────────────────────

def compute_ctr(
    x: torch.Tensor,
    seg_net: nn.Module,
    lung_channel_idx: list[int],
    heart_channel_idx: int,
    device: str = config.DEVICE,
    seg_input_size: int = config.SEG_INPUT_SIZE,
) -> float:
    """
    Cardio-Thoracic Ratio = cardiac_diameter / thoracic_diameter.

    Paper: CTR > 0.5 indicates cardiomegaly.
    We follow Chamveha et al. (2020) as referenced in the paper:
      - cardiac diameter  = horizontal span of the heart mask
      - thoracic diameter = horizontal span of the lung mask

    Parameters
    ----------
    x                 : (1, 1, H, W) CXR image, values in [0, 1]
    seg_net           : torchxrayvision PSPNet
    lung_channel_idx  : [left_lung_ch, right_lung_ch]
    heart_channel_idx : channel index for heart in seg_net.targets

    Returns
    -------
    float — CTR value (or np.nan if masks are empty)
    """
    B, C, H, W = x.shape

    x_xrv = x * 2048.0 - 1024.0
    if (H, W) != (seg_input_size, seg_input_size):
        x_xrv = F.interpolate(
            x_xrv, size=(seg_input_size, seg_input_size),
            mode="bilinear", align_corners=False,
        )

    with torch.no_grad():
        seg_out = seg_net(x_xrv.float().to(device))  # (1, n_classes, H', W')

    # ── lung mask ─────────────────────────────────────────────────────────────
    lung_logits  = seg_out[:, lung_channel_idx, :, :]
    lung_mask    = (torch.sigmoid(lung_logits).amax(dim=1) > 0.5).squeeze(0).cpu().numpy()

    # ── heart mask ────────────────────────────────────────────────────────────
    heart_logits = seg_out[:, heart_channel_idx, :, :]
    heart_mask   = (torch.sigmoid(heart_logits) > 0.5).squeeze(0).squeeze(0).cpu().numpy()

    if lung_mask.sum() == 0 or heart_mask.sum() == 0:
        return float("nan")

    # horizontal span of heart
    heart_cols     = np.where(heart_mask.any(axis=0))[0]
    cardiac_diam   = float(heart_cols[-1] - heart_cols[0]) if len(heart_cols) >= 2 else 0.0

    # horizontal span of lungs
    lung_cols      = np.where(lung_mask.any(axis=0))[0]
    thoracic_diam  = float(lung_cols[-1] - lung_cols[0])  if len(lung_cols) >= 2 else 1.0

    return cardiac_diam / (thoracic_diam + 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Score for normal CostoPhrenic recess (SCP)  —  pleural effusion metric
# ─────────────────────────────────────────────────────────────────────────────

def compute_scp(
    x: torch.Tensor,
    obj_det,
    device: str = config.DEVICE,
) -> float:
    """
    Score for detecting a healthy (normal) CostoPhrenic recess.

    In the paper, SCP is computed by a dedicated CP-recess object detector
    trained on MIMIC-CXR. We approximate it here using the YOLOv8 feature
    extractor: a high-entropy, low-norm feature response in the lower-lobe
    region corresponds to the absence of blunting (healthy CP recess ≈ high SCP).

    The returned value is in [0, 1] (sigmoid-normalised).

    Parameters
    ----------
    x       : (1, 1, H, W) CXR image
    obj_det : YOLOv8FeatureExtractor

    Returns
    -------
    float — SCP proxy score
    """
    B, C, H, W = x.shape
    # crop lower third of the image (CP recess region)
    lower_third = x[:, :, int(2 * H / 3) :, :]
    x3 = lower_third.repeat(1, 3, 1, 1) if lower_third.shape[1] == 1 else lower_third

    with torch.no_grad():
        feat = obj_det(x3.to(device))                  # (1, D)
        # proxy: low feature-norm in lower lobe = healthy (no fluid = high SCP)
        norm_val = feat.norm(dim=-1).item()
        scp = float(torch.sigmoid(torch.tensor(-norm_val / 50.0)).item())

    return scp


# ─────────────────────────────────────────────────────────────────────────────
# 6. Full evaluation suite  —  mirrors Table 1 of the paper
# ─────────────────────────────────────────────────────────────────────────────

def run_full_evaluation(
    dataset,
    classifier,
    generator,
    seg_net,
    lung_channel_idx: list[int],
    obj_det,
    n_eval: int = 200,
    tau_cv: float = 0.8,
    fo_threshold: float = 0.5,
    ckpt_dir: str = config.CKPT_DIR,
    device: str = config.DEVICE,
) -> dict:
    """
    Run the complete evaluation protocol from Singla et al. (2022).

    Splits the dataset into:
      Xn  — negative group: f(x) < 0.2  (predicted normal)
      Xp  — positive group: f(x) > 0.8  (predicted abnormal)

    Then generates:
      Xc_n→p  — counterfactuals from Xn with c = 1.0  (push toward abnormal)
      Xc_p→n  — counterfactuals from Xp with c = 0.0  (push toward normal)

    Computes for each split:
      - CV score
      - FID score  (negative group and positive group separately, as in Table 1)
      - FOP score
      - CTR statistics (mean ± std, paired t-test Δ)
      - SCP statistics (mean ± std, paired t-test Δ)

    Parameters
    ----------
    n_eval : number of images to sample from each group (capped by group size)

    Returns
    -------
    dict with all metric values, also printed to stdout and saved as JSON.
    """
    from scipy import stats as scipy_stats

    print("\n" + "=" * 70)
    print("EVALUATION — Singla et al. (2022) metrics")
    print("=" * 70)

    # ── try to find heart channel ──────────────────────────────────────────────
    heart_channel_idx = None
    if hasattr(seg_net, "targets"):
        for name in ("Heart", "heart", "Mediastinum"):
            if name in seg_net.targets:
                heart_channel_idx = seg_net.targets.index(name)
                break
    if heart_channel_idx is None:
        print("[metrics] WARNING: heart channel not found in seg_net.targets — CTR will be skipped")

    # ── scan dataset for Xn / Xp ──────────────────────────────────────────────
    print("[metrics] scanning dataset for negative/positive groups ...")
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)
    to_rgb = lambda t: t.repeat(1, 3, 1, 1) if t.shape[1] == 1 else t

    xn_list, xp_list = [], []
    with torch.no_grad():
        for x_batch, _ in loader:
            x_batch = x_batch.to(device)
            scores  = classifier(to_rgb(x_batch)).squeeze(-1)
            xn_list.extend([
                x_batch[i].unsqueeze(0).cpu()
                for i in range(len(x_batch)) if scores[i].item() < 0.2
            ])
            xp_list.extend([
                x_batch[i].unsqueeze(0).cpu()
                for i in range(len(x_batch)) if scores[i].item() > 0.8
            ])
            if len(xn_list) >= n_eval and len(xp_list) >= n_eval:
                break

    xn_list = xn_list[:n_eval]
    xp_list = xp_list[:n_eval]
    print(f"[metrics] Xn={len(xn_list)}  Xp={len(xp_list)}")

    # ── generate counterfactuals ───────────────────────────────────────────────
    print("[metrics] generating Xc (n→p) and Xc (p→n) ...")

    def _generate_batch(x_list: list, target_c: float) -> list[torch.Tensor]:
        cfs = []
        for x in x_list:
            xc, _ = generator.generate_counterfactual(x.to(device), target_c, M=1)
            cfs.append(xc.detach().cpu())
        return cfs

    xc_np = _generate_batch(xn_list, 1.0)   # negative → push positive
    xc_pn = _generate_batch(xp_list, 0.0)   # positive → push negative

    # ── classifier scores ──────────────────────────────────────────────────────
    def _score_batch(x_list):
        scores = []
        with torch.no_grad():
            for x in x_list:
                s = classifier(to_rgb(x.to(device))).item()
                scores.append(s)
        return torch.tensor(scores)

    f_xn    = _score_batch(xn_list)
    f_xp    = _score_batch(xp_list)
    f_xc_np = _score_batch(xc_np)
    f_xc_pn = _score_batch(xc_pn)

    # ── CV ────────────────────────────────────────────────────────────────────
    cv_np = counterfactual_validity(f_xn, f_xc_np, tau=tau_cv)
    cv_pn = counterfactual_validity(f_xp, f_xc_pn, tau=tau_cv)
    cv    = (cv_np + cv_pn) / 2.0

    # ── FID ───────────────────────────────────────────────────────────────────
    print("[metrics] computing FID ...")
    fid_neg = frechet_inception_distance(xn_list, xc_np, device=device)
    fid_pos = frechet_inception_distance(xp_list, xc_pn, device=device)

    # ── FOP ───────────────────────────────────────────────────────────────────
    fop_np = foreign_object_preservation(obj_det, xn_list, xc_np, fo_threshold, device)
    fop_pn = foreign_object_preservation(obj_det, xp_list, xc_pn, fo_threshold, device)
    fop    = (fop_np + fop_pn) / 2.0

    # ── CTR ───────────────────────────────────────────────────────────────────
    ctr_results = {}
    if heart_channel_idx is not None:
        print("[metrics] computing CTR ...")
        def _ctr_batch(x_list):
            return [
                compute_ctr(x.to(device), seg_net, lung_channel_idx, heart_channel_idx, device)
                for x in x_list
            ]
        ctr_xn    = np.array(_ctr_batch(xn_list))
        ctr_xp    = np.array(_ctr_batch(xp_list))
        ctr_xc_np = np.array(_ctr_batch(xc_np))
        ctr_xc_pn = np.array(_ctr_batch(xc_pn))

        # remove NaNs
        mask_n = ~np.isnan(ctr_xn) & ~np.isnan(ctr_xc_np)
        mask_p = ~np.isnan(ctr_xp) & ~np.isnan(ctr_xc_pn)

        delta_n = ctr_xn[mask_n] - ctr_xc_np[mask_n]   # Xn - Xc_n→p
        delta_p = ctr_xp[mask_p] - ctr_xc_pn[mask_p]   # Xp - Xc_p→n
        t_n, p_n = scipy_stats.ttest_rel(ctr_xn[mask_n], ctr_xc_np[mask_n])
        t_p, p_p = scipy_stats.ttest_rel(ctr_xp[mask_p], ctr_xc_pn[mask_p])

        ctr_results = dict(
            mean_xn=float(np.nanmean(ctr_xn)),       std_xn=float(np.nanstd(ctr_xn)),
            mean_xp=float(np.nanmean(ctr_xp)),       std_xp=float(np.nanstd(ctr_xp)),
            mean_xc_np=float(np.nanmean(ctr_xc_np)), std_xc_np=float(np.nanstd(ctr_xc_np)),
            mean_xc_pn=float(np.nanmean(ctr_xc_pn)), std_xc_pn=float(np.nanstd(ctr_xc_pn)),
            mean_delta_n=float(delta_n.mean()),       std_delta_n=float(delta_n.std()),
            t_n=float(t_n), p_n=float(p_n),
            mean_delta_p=float(delta_p.mean()),       std_delta_p=float(delta_p.std()),
            t_p=float(t_p), p_p=float(p_p),
        )

    # ── SCP ───────────────────────────────────────────────────────────────────
    print("[metrics] computing SCP ...")
    def _scp_batch(x_list):
        return [compute_scp(x.to(device), obj_det, device) for x in x_list]

    scp_xn    = np.array(_scp_batch(xn_list))
    scp_xp    = np.array(_scp_batch(xp_list))
    scp_xc_np = np.array(_scp_batch(xc_np))
    scp_xc_pn = np.array(_scp_batch(xc_pn))

    delta_sn = scp_xn - scp_xc_np
    delta_sp = scp_xp - scp_xc_pn
    t_sn, p_sn = scipy_stats.ttest_rel(scp_xn, scp_xc_np)
    t_sp, p_sp = scipy_stats.ttest_rel(scp_xp, scp_xc_pn)

    scp_results = dict(
        mean_xn=float(scp_xn.mean()),       std_xn=float(scp_xn.std()),
        mean_xp=float(scp_xp.mean()),       std_xp=float(scp_xp.std()),
        mean_xc_np=float(scp_xc_np.mean()), std_xc_np=float(scp_xc_np.std()),
        mean_xc_pn=float(scp_xc_pn.mean()), std_xc_pn=float(scp_xc_pn.std()),
        mean_delta_n=float(delta_sn.mean()), std_delta_n=float(delta_sn.std()),
        t_n=float(t_sn), p_n=float(p_sn),
        mean_delta_p=float(delta_sp.mean()), std_delta_p=float(delta_sp.std()),
        t_p=float(t_sp), p_p=float(p_sp),
    )

    # ── assemble results dict ─────────────────────────────────────────────────
    results = dict(
        CV=dict(n_to_p=cv_np, p_to_n=cv_pn, mean=cv),
        FID=dict(negative_group=fid_neg, positive_group=fid_pos),
        FOP=dict(n_to_p=fop_np, p_to_n=fop_pn, mean=fop),
        CTR=ctr_results,
        SCP=scp_results,
    )

    _print_results(results)
    _save_results(results, ckpt_dir)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_results(r: dict) -> None:
    print("\n" + "─" * 70)
    print("  METRIC SUMMARY  (Singla et al., 2022 — Table 1 / Fig. 9 style)")
    print("─" * 70)

    cv = r["CV"]
    print(f"\n► Counterfactual Validity (CV)   [τ = 0.8]")
    print(f"    Xn → Xp : {cv['n_to_p']:.4f}")
    print(f"    Xp → Xn : {cv['p_to_n']:.4f}")
    print(f"    mean     : {cv['mean']:.4f}")

    fid = r["FID"]
    print(f"\n► FID score  (lower = more realistic CFs)")
    print(f"    Negative group  FID(Xn,  Xc_n→p) : {fid['negative_group']:.2f}")
    print(f"    Positive group  FID(Xp,  Xc_p→n) : {fid['positive_group']:.2f}")

    fop = r["FOP"]
    print(f"\n► Foreign-Object Preservation (FOP)")
    print(f"    Xn → Xp : {fop['n_to_p']:.4f}")
    print(f"    Xp → Xn : {fop['p_to_n']:.4f}")
    print(f"    mean     : {fop['mean']:.4f}")

    if r["CTR"]:
        c = r["CTR"]
        print(f"\n► CTR — Cardiomegaly clinical metric")
        print(f"    mean CTR(Xn)     = {c['mean_xn']:.4f} ± {c['std_xn']:.4f}")
        print(f"    mean CTR(Xp)     = {c['mean_xp']:.4f} ± {c['std_xp']:.4f}")
        print(f"    mean CTR(Xc_n→p) = {c['mean_xc_np']:.4f} ± {c['std_xc_np']:.4f}")
        print(f"    mean CTR(Xc_p→n) = {c['mean_xc_pn']:.4f} ± {c['std_xc_pn']:.4f}")
        print(f"    Δ CTR(Xn − Xc_n→p) = {c['mean_delta_n']:.4f} ± {c['std_delta_n']:.4f}  "
              f"[t={c['t_n']:.2f}, p={c['p_n']:.4f}]")
        print(f"    Δ CTR(Xp − Xc_p→n) = {c['mean_delta_p']:.4f} ± {c['std_delta_p']:.4f}  "
              f"[t={c['t_p']:.2f}, p={c['p_p']:.4f}]")

    s = r["SCP"]
    print(f"\n► SCP — Pleural Effusion clinical metric")
    print(f"    mean SCP(Xn)     = {s['mean_xn']:.4f} ± {s['std_xn']:.4f}")
    print(f"    mean SCP(Xp)     = {s['mean_xp']:.4f} ± {s['std_xp']:.4f}")
    print(f"    mean SCP(Xc_n→p) = {s['mean_xc_np']:.4f} ± {s['std_xc_np']:.4f}")
    print(f"    mean SCP(Xc_p→n) = {s['mean_xc_pn']:.4f} ± {s['std_xc_pn']:.4f}")
    print(f"    Δ SCP(Xn − Xc_n→p) = {s['mean_delta_n']:.4f} ± {s['std_delta_n']:.4f}  "
          f"[t={s['t_n']:.2f}, p={s['p_n']:.4f}]")
    print(f"    Δ SCP(Xp − Xc_p→n) = {s['mean_delta_p']:.4f} ± {s['std_delta_p']:.4f}  "
          f"[t={s['t_p']:.2f}, p={s['p_p']:.4f}]")
    print("─" * 70)


def _save_results(results: dict, ckpt_dir: str) -> None:
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, "eval_metrics.json")

    # convert numpy floats → python floats for JSON serialisation
    def _to_serialisable(obj):
        if isinstance(obj, dict):
            return {k: _to_serialisable(v) for k, v in obj.items()}
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        return obj

    with open(path, "w") as f:
        json.dump(_to_serialisable(results), f, indent=2)
    print(f"\n[metrics] results saved to '{path}'")
