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

CHANGE FROM ORIGINAL: foreign_object_preservation() was returning a
degenerate 1.0/1.0/1.0 always — see its docstring for the root cause
and fix.
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
import torchxrayvision as xrv

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
        # FIX (Bug 1): Inception-v3 was trained on ImageNet-normalised inputs.
        # Feeding raw [0,1] pixel values (with no mean/std normalisation)
        # pushes every activation off-distribution and inflates FID for
        # every comparison, including real-vs-real. Normalise exactly like
        # the torchvision pretrained-weights preprocessing expects.
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, 1, H, W) or (B, 3, H, W), values in [0, 1] → (B, 2048)"""
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = self.resize(x)
        x = self.normalize(x)
        with torch.no_grad():
            feat = self.model(x.to(self.device))
        return feat.cpu()


class _XRVFeatureExtractor(nn.Module):
    """
    Penultimate-layer features (1024-d) of TorchXRayVision's DenseNet-121.

    Default weights ("densenet121-res224-rsna") are trained specifically
    on the RSNA Pneumonia Detection Challenge dataset — the same source
    distribution as this project's cxr-data-set. This gives a feature
    space that's directly sensitive to pneumonia-relevant visual patterns,
    unlike ImageNet-pretrained Inception-v3 or the broader multi-dataset
    "-all" xrv weights.

    Other useful weight options (pass via `weights=`):
      "densenet121-res224-all"  -> joint NIH+CheXpert+PadChest+MIMIC+RSNA,
                                    broader but less pneumonia-specific
      "densenet121-res224-nih"  -> NIH ChestX-ray14 only
      "densenet121-res224-chex" -> CheXpert only

    Since this project already depends on torchxrayvision for the PSPNet
    segmenter, this adds no new dependency.
    """

    def __init__(self, device: str, weights: str = "densenet121-res224-rsna"):
        super().__init__()
        model = xrv.models.DenseNet(weights=weights)
        self.model = model.to(device).eval()
        self.device = device
        self.input_size = 224  # xrv DenseNet default input resolution

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 1, H, W) or (B, 3, H, W), values expected in [0, 1]
            (single-channel grayscale CXR — extra channels are averaged down,
            since xrv models are single-channel).
        Returns (B, 1024) penultimate-layer features.
        """
        if x.shape[1] == 3:
            x = x.mean(dim=1, keepdim=True)

        # xrv models expect inputs scaled to roughly [-1024, 1024] (their
        # "normalize" convention), not raw [0,1]. If your tensors are
        # already in [0,1] from the VAE decoder, rescale like this:
        x_xrv = x * 2048.0 - 1024.0

        if x_xrv.shape[-1] != self.input_size or x_xrv.shape[-2] != self.input_size:
            x_xrv = F.interpolate(
                x_xrv, size=(self.input_size, self.input_size),
                mode="bilinear", align_corners=False,
            )

        with torch.no_grad():
            feat = self.model.features(x_xrv.float().to(self.device))
            feat = F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)  # (B, 1024)

        return feat.cpu()


def _compute_activation_stats(
    images: list[torch.Tensor],
    extractor: nn.Module,
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
    eps: float = 1e-6,
    backbone: str = "xrv",
    extractor: Optional[nn.Module] = None,
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
    eps    : regularisation added to the covariance diagonals before
             sqrtm. FIX (Bug 2): with N samples < 2048 feature dims,
             the covariance matrices are rank-deficient/singular, which
             makes scipy.linalg.sqrtm numerically unstable (near-zero
             eigenvalues produce spurious imaginary/large real components).
             This is the same fix used by the standard pytorch-fid
             implementation.

    backbone : "xrv"       -> TorchXRayVision DenseNet-121 (1024-d),
                              pretrained on chest X-rays. Domain-appropriate,
                              recommended default for this project.
               "inception" -> ImageNet Inception-v3 (2048-d), the classic
                              FID backbone. Kept for comparability with
                              natural-image FID literature / papers that
                              report vanilla FID.
    extractor : pre-built feature extractor to reuse across multiple FID
                calls (e.g. one built once in run_full_evaluation and
                shared between the negative-group and positive-group FID
                calls, instead of reloading weights twice). If None, a
                fresh extractor is built internally using `backbone`.

    Note
    ----
    FID estimates are only stable with large sample sizes (rule of thumb:
    N >> feature dim). With N ~ 100-200 (typical after filtering into
    Xn/Xp groups) expect high variance in this number even after the fixes
    above — increase n_eval wherever your compute budget allows.
    """
    if extractor is None:
        if backbone == "xrv":
            extractor = _XRVFeatureExtractor(device)
        elif backbone == "inception":
            extractor = _InceptionFeatureExtractor(device)
        else:
            raise ValueError(f"Unknown backbone '{backbone}', expected 'xrv' or 'inception'")
    mu1, sigma1 = _compute_activation_stats(X_real, extractor, batch_size)
    mu2, sigma2 = _compute_activation_stats(X_cf,   extractor, batch_size)

    # FIX (Bug 2): regularise before sqrtm to avoid singular-matrix blowups
    sigma1 = sigma1 + np.eye(sigma1.shape[0]) * eps
    sigma2 = sigma2 + np.eye(sigma2.shape[0]) * eps

    diff = mu1 - mu2
    # scipy>=1.15 dropped the `disp` kwarg and returns the array directly
    # instead of an (array, errest) tuple.
    covmean = linalg.sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        # residual imaginary component should be ~0 after regularisation;
        # if it isn't, something's still off numerically — warn loudly
        max_imag = np.max(np.abs(covmean.imag))
        if max_imag > 1e-3:
            print(f"[metrics] WARNING: sqrtm returned large imaginary "
                  f"component ({max_imag:.4f}) — FID may be unreliable. "
                  f"Consider increasing n_eval or eps.")
        covmean = covmean.real

    fid = float(diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean))

    if reference_fid is not None and reference_fid > 0:
        fid = fid / reference_fid

    return fid


# ─────────────────────────────────────────────────────────────────────────────
# 3. Foreign-Object Preservation (FOP)  —  Eq. 12
# ─────────────────────────────────────────────────────────────────────────────

class _XRVSupportDeviceDetector(nn.Module):
    """
    Detects presence of support devices (pacemakers, ECG leads, chest
    tubes, central lines, etc.) using TorchXRayVision's DenseNet-121.

    "Support Devices" is one of CheXpert's 14 original pathology labels,
    and models trained on (or including) CheXpert expose it directly as
    an output head. This gives a real, properly-calibrated [0,1]
    presence probability, trained specifically on chest X-rays for
    exactly this purpose — unlike a COCO-pretrained object detector,
    which has never seen a support device in training.

    Requires zero additional fine-tuning or compute: torchxrayvision is
    already a dependency of this project (used for the PSPNet segmenter),
    so this just loads a second pretrained xrv model.

    weights options that include "Support Devices" (check
    model.pathologies if using a different weight set):
      "densenet121-res224-all"  -> joint NIH+CheXpert+PadChest+MIMIC+RSNA
                                    (recommended: broadest training data)
      "densenet121-res224-chex" -> CheXpert only (where the label originates)
    """

    def __init__(self, device: str, weights: str = "densenet121-res224-all"):
        super().__init__()
        model = xrv.models.DenseNet(weights=weights)
        self.model      = model.to(device).eval()
        self.device     = device
        self.input_size = 224

        if "Support Devices" not in model.pathologies:
            raise ValueError(
                f"'Support Devices' not found in pathologies for "
                f"weights='{weights}': {model.pathologies}. Try "
                f"'densenet121-res224-all' or 'densenet121-res224-chex'."
            )
        self.sd_idx = model.pathologies.index("Support Devices")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 1, H, W) or (B, 3, H, W), values in [0, 1]
        Returns (B,) support-device presence probability in [0, 1].
        """
        if x.shape[1] == 3:
            x = x.mean(dim=1, keepdim=True)

        x_xrv = x * 2048.0 - 1024.0
        if x_xrv.shape[-1] != self.input_size or x_xrv.shape[-2] != self.input_size:
            x_xrv = F.interpolate(
                x_xrv, size=(self.input_size, self.input_size),
                mode="bilinear", align_corners=False,
            )

        with torch.no_grad():
            # xrv models output already-calibrated [0,1] probabilities
            # per pathology (sigmoid applied internally), not raw logits.
            probs = self.model(x_xrv.float().to(self.device))
            sd_prob = probs[:, self.sd_idx]

        return sd_prob.cpu()


def foreign_object_preservation(
    fo_detector,
    X_real: list[torch.Tensor],
    X_cf:   list[torch.Tensor],
    fo_threshold: float = 0.5,
    device: str = config.DEVICE,
) -> float:
    """
    FOP(X, Xc) = fraction of images where FO is detected in BOTH
                 the real image AND its counterfactual.

    Only images where a detection probability > fo_threshold is
    considered (paper Eq. 12).

    HISTORY:
      v1 thresholded a raw YOLO backbone feature-vector L2 norm against
      fo_threshold=0.5 — norms are routinely tens-to-hundreds, so this
      was true for ~every image, giving a degenerate 1.0 always.
      v2 fixed the threshold-space bug by using the full YOLO detection
      head instead, but yolov8n.pt is COCO-pretrained (everyday objects),
      not trained on medical support devices at all.
      v3 (this version) replaces YOLO entirely with a TorchXRayVision
      DenseNet-121 classifier that has a dedicated, CheXpert-trained
      "Support Devices" output — a properly domain-matched detector,
      requiring no fine-tuning since it's already pretrained for exactly
      this label.

    Parameters
    ----------
    fo_detector   : _XRVSupportDeviceDetector (or any nn.Module returning
                    a (B,) or scalar [0,1] support-device probability per
                    image when called on a (1,C,H,W) tensor)
    X_real        : list of (1,1,H,W) tensors
    X_cf          : list of (1,1,H,W) tensors
    fo_threshold  : detection probability threshold (paper: 0.5)

    Returns
    -------
    float — FOP score ∈ [0, 1]
    """
    assert len(X_real) == len(X_cf)

    preserved, total = 0, 0
    for x, xc in zip(X_real, X_cf):
        with torch.no_grad():
            conf_real = fo_detector(x.to(device)).item()
            conf_cf   = fo_detector(xc.to(device)).item()

        fo_in_real = conf_real > fo_threshold
        fo_in_cf   = conf_cf   > fo_threshold

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
    fid_backbone: str = "xrv",
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
    n_eval : number of images to sample from each group (capped by group size).
             NOTE: FID is unstable at small N (see frechet_inception_distance
             docstring). Prefer n_eval >= 1000 if your generation budget allows.
    fid_backbone : "xrv" (default, recommended for CXR data) or "inception".
             See frechet_inception_distance docstring for details. Built
             once here and reused for both the negative-group and
             positive-group FID calls to avoid loading weights twice.

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

    if len(xn_list) < 200 or len(xp_list) < 200:
        print(f"[metrics] WARNING: group size < 200 — FID estimate will have "
              f"high variance and likely be biased upward. Consider lowering "
              f"the f(x)<0.2 / f(x)>0.8 thresholds or increasing n_eval.")

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
    fid_extractor = (
        _XRVFeatureExtractor(device) if fid_backbone == "xrv"
        else _InceptionFeatureExtractor(device)
    )
    fid_neg = frechet_inception_distance(xn_list, xc_np, device=device, extractor=fid_extractor)
    fid_pos = frechet_inception_distance(xp_list, xc_pn, device=device, extractor=fid_extractor)

    # ── FOP ───────────────────────────────────────────────────────────────────
    print("[metrics] computing FOP (TorchXRayVision Support Devices detector) ...")
    fo_detector = _XRVSupportDeviceDetector(device)
    fop_np = foreign_object_preservation(fo_detector, xn_list, xc_np, fo_threshold, device)
    fop_pn = foreign_object_preservation(fo_detector, xp_list, xc_pn, fo_threshold, device)
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