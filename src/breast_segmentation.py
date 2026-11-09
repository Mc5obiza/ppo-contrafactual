"""
src/breast_segmentation.py
==========================
The mammography replacement for torchxrayvision's PSPNet.

WHY TWO NETS AND NOT ONE
------------------------
PSPNet quietly plays TWO unrelated roles in this pipeline, and dropping it for
a non-chest domain means replacing both:

  ROLE 1 — the ORGAN MASK that constrains where the generator may edit.
           src/generator.py::_get_seg_mask takes the Left/Right Lung channels,
           dilates and feathers them (config.BG_MASK_*), and uses the result as
           the RePaint / blended-latent-diffusion composite weight when
           config.LOCK_BACKGROUND is on. This runs INSIDE the denoising loop,
           once per guided step.

  ROLE 2 — the MEASUREMENT SUBSTRATE for the clinical metrics. compute_ctr
           takes the same net's Heart channel and measures a cardiothoracic
           ratio on both x and x_cf. This runs only in the metric phase.

A breast has no lungs and no heart, so both roles need new machinery:
BreastAnatomySegmenter for role 1, LesionSegmenter for role 2. They are
separate classes because their constraints are genuinely different — see the
differentiability note below — not because it was tidier.

GRADIENTS: WHAT IS AND IS NOT REQUIRED
--------------------------------------
Worth stating because the opposite is easy to assume. The mask does NOT need to
be differentiable in x. generate_counterfactual calls _get_seg_mask inside the
`with torch.no_grad():` block that spans src/generator.py lines 461-513, and
src/metrics.py calls the segmenter under no_grad too. The mask is a
multiplicative BLEND WEIGHT on the latents, so what guidance needs is for the
weight to be non-zero where editing should happen — the gradient flows through
the edited latent it multiplies, not through the weight itself.

The thresholds below are still soft (sigmoid((u - thr) / tau) rather than
u > thr) for a different and smaller reason: it gives the region boundary a few
grey levels of ramp instead of a hard step, which composites more cleanly with
the dilation and Gaussian feathering _build_edit_mask applies on top. Nothing
breaks if a future edit makes them hard.

The failure mode that IS real here is a DEGENERATE mask, and it is silent in
both directions: all-zero means the policy has nothing it is allowed to edit
and every episode no-ops, all-one means LOCK_BACKGROUND is quietly disabled and
edits land in the black background. build_breast_masker checks neither (it has
no image), so check_lesion_localization.py checks coverage on real images.

LesionSegmenter is likewise only ever called under no_grad, in the metric
phase, so it is an ordinary trained U-Net.

WHY ANALYTIC FOR THE ANATOMY, TRAINED FOR THE LESION
----------------------------------------------------
Not a stylistic split. A mammogram's background is genuinely, exactly black —
the breast is the only object in the frame — so an intensity threshold is not
an approximation of the organ mask, it very nearly IS the organ mask, and it is
better behaved than PSPNet's lungs ever were. Training a net for that would add
a failure mode to solve a problem that does not exist.

Lesion boundaries are the opposite: invisible to thresholding, and CBIS-DDSM
ships pixel-accurate ROI masks for exactly this. So that one is learned.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

import config


# ═════════════════════════════════════════════════════════════════════════════
# ROLE 1 — anatomy: the organ mask that drives LOCK_BACKGROUND
# ═════════════════════════════════════════════════════════════════════════════

# Softness of every threshold in this module, in [0,1] intensity units. Small
# enough that the mask is effectively binary a few grey levels either side of
# the threshold; large enough that the sigmoid has a live gradient at the
# boundary, which is the entire reason it is a sigmoid.
_TAU = 0.02

TARGETS = ["Breast", "Pectoral", "Dense Tissue"]


def _otsu_threshold(img: torch.Tensor, bins: int = 256) -> float:
    """
    Otsu's threshold for one (H, W) image in [0, 1].

    Returned as a python float rather than a tensor so it enters the mask as a
    plain constant — the threshold is a property of the histogram, and keeping
    it off the graph makes that explicit.
    """
    hist = torch.histc(img, bins=bins, min=0.0, max=1.0)
    total = hist.sum()
    if total <= 0:
        return 0.5

    p = hist / total
    levels = torch.linspace(0.0, 1.0, bins, device=img.device, dtype=p.dtype)

    w0 = torch.cumsum(p, dim=0)                       # weight of class 0
    w1 = 1.0 - w0
    m0 = torch.cumsum(p * levels, dim=0)
    mt = m0[-1]

    # Between-class variance. Guard the degenerate ends where one class is
    # empty; those bins would otherwise produce 0/0 and win the argmax as NaN.
    denom = w0 * w1
    valid = denom > 1e-12
    var_b = torch.zeros_like(p)
    var_b[valid] = ((mt * w0[valid] - m0[valid]) ** 2) / denom[valid]

    if not bool(valid.any()):
        return 0.5
    return float(levels[int(torch.argmax(var_b))])


def _largest_component_gate(binary: np.ndarray) -> np.ndarray:
    """
    Keep only the largest connected component of a 2-D boolean array.

    Mammograms routinely carry a laterality marker, a technologist's label or
    detector-edge speckle — bright objects that clear the intensity threshold
    but are not breast. Without this, they join the "organ" and the generator
    is licensed to edit them.

    Returned as a plain 0/1 array for the caller to multiply in.
    """
    try:
        from skimage.measure import label as _label
    except ImportError as e:
        raise ImportError(
            "src/breast_segmentation.py needs scikit-image for connected "
            "components (and for the lesion-metric region properties). "
            "`pip install scikit-image` and add it to requirements.txt — note "
            "compute nodes have no internet, so do this on the login node."
        ) from e

    if not binary.any():
        return binary.astype(np.float32)
    lab = _label(binary, connectivity=2)
    counts = np.bincount(lab.ravel())
    counts[0] = 0                                    # background is not a candidate
    return (lab == int(counts.argmax())).astype(np.float32)


class BreastAnatomySegmenter(nn.Module):
    """
    Drop-in replacement for xrv's PSPNet in role 1.

    Matches the PSPNet CALL CONTRACT exactly, so src/generator.py::_get_seg_mask
    and src/metrics.py need no edit:

      * input  : (B, 1|3, H, W) already scaled x*2048-1024 by the caller
      * output : (B, 3, H, W) LOGITS (callers apply their own sigmoid)
      * .targets : ["Breast", "Pectoral", "Dense Tissue"]

    Channels
    --------
    0 Breast       the organ mask. Soft Otsu + largest-component gate. This is
                   the direct analogue of the lung field and the default
                   editable region.
    1 Pectoral     the pectoral muscle triangle in the upper chest-wall corner
                   of MLO views. SUBTRACTED from the editable region when
                   config.MASK_EXCLUDE_PECTORAL is on.
    2 Dense Tissue fibroglandular tissue: a high-percentile threshold within
                   the breast, minus pectoral. This is where lesions actually
                   live, so it is the tighter editable region worth ablating
                   (config.BREAST_EDIT_REGION = "dense").

    Returns logits rather than probabilities purely for PSPNet parity — every
    existing caller does torch.sigmoid(out[:, idx]) and must keep working.
    """

    targets = TARGETS

    def __init__(
        self,
        device: str = config.DEVICE,
        exclude_pectoral: bool | None = None,
        dense_percentile: float = 0.65,
    ):
        super().__init__()
        self.device = device
        self.exclude_pectoral = (
            config.MASK_EXCLUDE_PECTORAL if exclude_pectoral is None else exclude_pectoral
        )
        self.dense_percentile = dense_percentile
        # Registered so .to(device) and .eval() behave like any other module in
        # PipelineContext even though there is nothing to learn.
        self.register_buffer("_dummy", torch.zeros(1), persistent=False)

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _to_unit(x: torch.Tensor) -> torch.Tensor:
        """
        Undo the caller's xrv scaling: (x*2048 - 1024) -> [0, 1].

        Callers hand us xrv-convention input because that is what PSPNet wants
        and this class exists to be swapped in for PSPNet. Everything below
        reasons about intensities in [0,1], so convert once, here, rather than
        carrying two conventions through the file.
        """
        if x.shape[1] == 3:
            x = x.mean(dim=1, keepdim=True)
        return (x + 1024.0) / 2048.0

    def _pectoral_mask(self, u: torch.Tensor, breast: torch.Tensor) -> torch.Tensor:
        """
        Soft half-plane covering the pectoral muscle, (B,1,H,W) in [0,1].

        The pectoral muscle appears in MLO views as a bright, roughly triangular
        region against the chest wall in one upper corner. We find the chest
        wall as the image side the breast tissue actually touches, then cut a
        triangle from that upper corner whose extent is set by how far the
        bright (above-Otsu-of-the-breast) tissue reaches along the top row.

        LIMITATIONS, stated plainly. This is geometric, not learned: CBIS-DDSM
        ships no pectoral ground truth, so there is nothing to train against and
        nothing to score it with. It is also a no-op on CC views, where no
        pectoral muscle is in frame — correctly so, since the triangle it cuts
        collapses when no bright tissue reaches the top corner.

        It is only ever used to EXCLUDE area from editing, which bounds the
        damage: a loose fit costs the policy some editable breast, it never
        licenses an edit outside the organ. That asymmetry is why an approximate
        method is acceptable here and would not be for the breast mask itself.
        """
        B, _, H, W = u.shape
        out = torch.zeros_like(u)

        with torch.no_grad():
            for b in range(B):
                m = breast[b, 0] > 0.5
                if not bool(m.any()):
                    continue

                # Chest wall = the vertical edge the breast touches most. In a
                # mammogram the breast is flush against it by construction.
                cols = m.any(dim=0).nonzero().flatten()
                if cols.numel() == 0:
                    continue
                left_contact  = int(m[:, 0].sum())
                right_contact = int(m[:, -1].sum())
                on_left = left_contact >= right_contact

                # How far the bright tissue reaches along the TOP row, measured
                # from the chest wall. That run length is the triangle's base.
                top = u[b, 0, : max(1, H // 20), :].mean(dim=0)
                thr = float(top.max()) * 0.5
                bright = top > thr
                if not bool(bright.any()):
                    continue
                if on_left:
                    run = int((~bright).nonzero().flatten()[0]) if bool((~bright).any()) else W
                else:
                    rev = torch.flip(bright, dims=[0])
                    run = int((~rev).nonzero().flatten()[0]) if bool((~rev).any()) else W

                # No meaningful bright run at the top => almost certainly a CC
                # view. Leave the mask empty rather than inventing a triangle.
                if run < W // 12:
                    continue

                # Triangle: full `run` wide at the top row, tapering to zero at
                # row `height`. Aspect ratio 1.6 is the usual MLO obliquity.
                height = min(H, int(run * 1.6))
                yy = torch.arange(H, device=u.device, dtype=u.dtype).view(H, 1)
                xx = torch.arange(W, device=u.device, dtype=u.dtype).view(1, W)
                width_at_y = run * (1.0 - yy / max(height, 1)).clamp(min=0.0)
                dist = xx if on_left else (W - 1 - xx)
                out[b, 0] = (dist < width_at_y).to(u.dtype)

        return out

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 1|3, H, W), xrv-scaled (x_unit * 2048 - 1024)
        Returns (B, 3, H, W) logits over TARGETS.
        """
        u = self._to_unit(x.to(self.device)).clamp(0.0, 1.0)
        B, _, H, W = u.shape

        # ── channel 0: breast ────────────────────────────────────────────────
        with torch.no_grad():
            thrs = [_otsu_threshold(u[b, 0]) for b in range(B)]
        thr_t = torch.tensor(thrs, device=u.device, dtype=u.dtype).view(B, 1, 1, 1)

        breast_soft = torch.sigmoid((u - thr_t) / _TAU)

        with torch.no_grad():
            gate = np.stack([
                _largest_component_gate((breast_soft[b, 0] > 0.5).cpu().numpy())
                for b in range(B)
            ])
        gate_t = torch.from_numpy(gate).to(u.device, u.dtype).unsqueeze(1)
        breast = breast_soft * gate_t

        # ── channel 1: pectoral ──────────────────────────────────────────────
        if self.exclude_pectoral:
            pect = self._pectoral_mask(u, breast) * breast
        else:
            pect = torch.zeros_like(breast)

        # ── channel 2: dense (fibroglandular) tissue ─────────────────────────
        # Percentile taken over breast pixels only — over the whole frame the
        # black background dominates and any percentile lands in it.
        with torch.no_grad():
            dense_thrs = []
            for b in range(B):
                vals = u[b, 0][breast[b, 0] > 0.5]
                dense_thrs.append(
                    float(torch.quantile(vals, self.dense_percentile))
                    if vals.numel() > 0 else 1.0
                )
        dthr = torch.tensor(dense_thrs, device=u.device, dtype=u.dtype).view(B, 1, 1, 1)
        dense = torch.sigmoid((u - dthr) / _TAU) * breast * (1.0 - pect)

        probs = torch.cat([breast, pect, dense], dim=1).clamp(1e-6, 1.0 - 1e-6)
        # Back to logits for PSPNet parity: callers apply their own sigmoid.
        return torch.log(probs / (1.0 - probs))


def build_breast_masker(
    device: str = config.DEVICE,
) -> tuple[BreastAnatomySegmenter, list[int]]:
    """
    (net, channel_idx) matching build_segmentation_net()'s contract.

    config.BREAST_EDIT_REGION picks which channel becomes the editable region:
      "breast" -> channel 0, the whole organ. The permissive default and the
                  direct analogue of the chest domains' lung field.
      "dense"  -> channel 2, fibroglandular tissue only. A strictly tighter
                  constraint: lesions live there, so it should cost little
                  coverage while removing a lot of editable fat. Worth an
                  ablation, not assumed better.
    """
    net = BreastAnatomySegmenter(device=device).to(device).eval()
    for p in net.parameters():
        p.requires_grad_(False)

    region = getattr(config, "BREAST_EDIT_REGION", "breast")
    if region not in ("breast", "dense"):
        raise ValueError(
            f"config.BREAST_EDIT_REGION={region!r} is not one of "
            f"'breast' / 'dense'."
        )
    idx = [TARGETS.index("Breast" if region == "breast" else "Dense Tissue")]

    print(f"[models] loaded BreastAnatomySegmenter | targets={TARGETS} | "
          f"edit region={region!r} (channel {idx[0]}) | "
          f"exclude_pectoral={net.exclude_pectoral}")
    return net, idx


# ═════════════════════════════════════════════════════════════════════════════
# ROLE 2 — lesions: the measurement substrate for the clinical metrics
# ═════════════════════════════════════════════════════════════════════════════

class _DecoderBlock(nn.Module):
    """Upsample, concatenate the skip connection, two 3x3 convs."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            # Encoder strides can leave a one-pixel mismatch at odd input
            # sizes; align to the skip rather than assuming they match.
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = torch.cat([x, skip], dim=1)
        return self.block(x)


class LesionSegmenter(nn.Module):
    """
    U-Net with an ImageNet ResNet-18 encoder, trained on CBIS-DDSM ROI masks.

    Input  : (B, 1|3, H, W) in [0, 1]   — the SAME convention as f(.), i.e.
             plain image space, NOT the xrv scaling BreastAnatomySegmenter takes.
             The two differ because this one never stands in for PSPNet.
    Output : (B, 1, H, W) logits.

    ANTI-CIRCULARITY. This net must never enter the guidance loop. It exists to
    MEASURE the counterfactuals (LLR, MSI in src/metrics.py), and a segmenter
    that had also helped steer generation would make those metrics circular in
    exactly the way CV already is — the failure heldout_validation.py was
    written to expose. domains/base.DomainProfile documents the same rule for
    build_lesion_net.

    Consequently it is only ever called under no_grad and carries no
    differentiability constraint, unlike BreastAnatomySegmenter.
    """

    def __init__(self, device: str = config.DEVICE, pretrained_encoder: bool = False):
        super().__init__()
        enc = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT if pretrained_encoder else None)

        self.stem = nn.Sequential(enc.conv1, enc.bn1, enc.relu)   # /2,  64
        self.pool = enc.maxpool                                    # /4
        self.enc1, self.enc2 = enc.layer1, enc.layer2              # /4 64, /8 128
        self.enc3, self.enc4 = enc.layer3, enc.layer4              # /16 256, /32 512

        self.dec4 = _DecoderBlock(512, 256, 256)
        self.dec3 = _DecoderBlock(256, 128, 128)
        self.dec2 = _DecoderBlock(128, 64, 64)
        self.dec1 = _DecoderBlock(64, 64, 32)
        self.dec0 = _DecoderBlock(32, 0, 16)
        self.head = nn.Conv2d(16, 1, 1)

        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        s0 = self.stem(x)          # /2   64
        s1 = self.enc1(self.pool(s0))   # /4   64
        s2 = self.enc2(s1)         # /8  128
        s3 = self.enc3(s2)         # /16 256
        s4 = self.enc4(s3)         # /32 512

        d = self.dec4(s4, s3)
        d = self.dec3(d, s2)
        d = self.dec2(d, s1)
        d = self.dec1(d, s0)
        d = self.dec0(d, None)
        out = self.head(d)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return out

    @torch.no_grad()
    def predict_mask(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """(B, 1, H, W) float 0/1 mask. The form the metric code consumes."""
        return (torch.sigmoid(self(x.to(self.device))) > threshold).float()


LESION_CKPT_NAME = "lesion_unet.pt"


def build_lesion_segmenter(device: str = config.DEVICE) -> LesionSegmenter:
    """Load the trained lesion U-Net, frozen and in eval mode."""
    ckpt = os.path.join(config.BREAST_CKPT_DIR, LESION_CKPT_NAME)
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"no trained lesion segmenter at '{ckpt}'. The LLR and MSI clinical "
            f"metrics are computed from it, so evaluation cannot report them "
            f"without it. Train it first:\n"
            f"    sbatch run_lesion_segmenter.slurm\n"
            f"(or: python train_lesion_segmenter.py)"
        )

    net = LesionSegmenter(device=device, pretrained_encoder=False)
    state = torch.load(ckpt, map_location=device, weights_only=True)
    val_dice = float("nan")
    if isinstance(state, dict) and "state_dict" in state:
        val_dice = float(state.get("val_dice", float("nan")))
        state = state["state_dict"]
    net.load_state_dict(state, strict=True)

    net = net.to(device).eval()
    for p in net.parameters():
        p.requires_grad_(False)

    dice_s = "" if val_dice != val_dice else f" (val Dice={val_dice:.3f})"
    print(f"[models] loaded LesionSegmenter from '{ckpt}'{dice_s}")
    if val_dice == val_dice and val_dice < 0.4:
        # Not fatal — a weak segmenter is still worth plotting — but LLR/MSI
        # computed from it are noise, and a printed number is a number someone
        # will eventually quote.
        print(f"[models]   WARNING: val Dice {val_dice:.3f} < 0.40. LLR and MSI "
              f"derived from this segmenter are not trustworthy; fix the "
              f"segmenter before reporting them as findings.")
    return net
