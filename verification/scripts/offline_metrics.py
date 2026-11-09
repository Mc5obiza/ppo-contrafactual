"""
offline_metrics.py — E4 / E8 of the CVPR-2027 issue list, computed from saved
evaluation shards (no regeneration).

Per arm (shard dir) and direction:
  E4a  share of sum|x_c - x| inside the background-lock edit mask m
       (lung prob of the SOURCE image, latent 32^2, 5x5 max-pool dilation,
       Gaussian feather sigma=1 cell, clamp; exactly generator._build_edit_mask
       at config.BG_MASK_DILATION / BG_MASK_FEATHER), upsampled bilinearly to
       256^2. Reported three ways: m-weighted share sum(m*d)/sum(d), hard share
       inside m>0.5, and the same for the undilated lung mask S>0.5, each with
       its area-fraction null (what a spatially uniform edit would score).
  E4b  fraction of pixels changed > 0.01 relative to the VAE round trip
       D(E(x)) (posterior MEAN, SD1.5 VAE fp16 as in the generator), alongside
       the usual fraction relative to x and the VAE floor itself |D(E(x)) - x|.
  E8   KID (cubic polynomial kernel, unbiased MMD^2, 100 subsets of 100) and a
       bootstrap 95% CI for FID (500 resamples of both sets), on the same
       1024-d TorchXRayVision densenet121-res224-rsna features the paper's FID
       uses: n->p counterfactuals vs real X_p, p->n counterfactuals vs real X_n.

    python offline_metrics.py OUT_JSON SHARD_DIR[:TAG] [SHARD_DIR[:TAG] ...]
"""
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg

import config
from src.models import apply_torch_load_patch, revert_torch_load_patch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
VAE_DTYPE = config.DTYPE if DEV == "cuda" else torch.float32
torch.manual_seed(0)
RNG = np.random.default_rng(0)


# ── models ───────────────────────────────────────────────────────────────────
def load_models():
    apply_torch_load_patch()
    import torchxrayvision as xrv
    seg = xrv.baseline_models.chestx_det.PSPNet().to(DEV).eval()
    feat = xrv.models.DenseNet(weights="densenet121-res224-rsna").to(DEV).eval()
    revert_torch_load_patch()
    lung_idx = [seg.targets.index("Left Lung"), seg.targets.index("Right Lung")]
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(
        config.DIFFUSION_MODEL_ID, subfolder="vae", torch_dtype=VAE_DTYPE
    ).to(DEV).eval()
    return seg, lung_idx, feat, vae


@torch.no_grad()
def lung_prob(seg, lung_idx, x):            # x (B,1,256,256) in [0,1]
    x_xrv = x * 2048.0 - 1024.0
    x_xrv = F.interpolate(x_xrv, size=(config.SEG_INPUT_SIZE,) * 2,
                          mode="bilinear", align_corners=False)
    out = seg(x_xrv.float())
    m = torch.sigmoid(out[:, lung_idx]).amax(dim=1, keepdim=True)
    return F.interpolate(m, size=x.shape[-2:], mode="bilinear", align_corners=False)


def gaussian_blur(m, sigma):
    if sigma is None or sigma <= 0:
        return m
    r = max(1, int(3.0 * sigma))
    c = torch.arange(-r, r + 1, device=m.device, dtype=torch.float32)
    g = torch.exp(-(c ** 2) / (2 * sigma ** 2)); g = (g / g.sum()).to(m.dtype)
    m = F.conv2d(m, g.view(1, 1, 1, -1), padding=(0, r))
    return F.conv2d(m, g.view(1, 1, -1, 1), padding=(r, 0))


def edit_mask(S):                            # S (B,1,256,256) lung prob
    lm = F.interpolate(S, size=(config.LATENT_SIZE,) * 2, mode="bilinear", align_corners=False)
    d = config.BG_MASK_DILATION
    if d > 0:
        lm = F.max_pool2d(lm, kernel_size=2 * d + 1, stride=1, padding=d)
    lm = gaussian_blur(lm, config.BG_MASK_FEATHER).clamp(0, 1)
    return F.interpolate(lm, size=S.shape[-2:], mode="bilinear", align_corners=False).clamp(0, 1)


@torch.no_grad()
def vae_roundtrip(vae, x):
    xr = x.repeat(1, 3, 1, 1).to(VAE_DTYPE) * 2.0 - 1.0
    z = vae.encode(xr).latent_dist.mean
    y = vae.decode(z).sample.clamp(-1, 1)
    y = (y + 1.0) / 2.0
    y = 0.299 * y[:, 0:1] + 0.587 * y[:, 1:2] + 0.114 * y[:, 2:3]
    return y.float()


@torch.no_grad()
def densenet_feats(feat, x):
    x_xrv = F.interpolate(x * 2048.0 - 1024.0, size=(224, 224), mode="bilinear", align_corners=False)
    f = feat.features(x_xrv.float())
    return F.adaptive_avg_pool2d(f, (1, 1)).flatten(1).cpu().double().numpy()


# ── distribution metrics ─────────────────────────────────────────────────────
def fid(a, b, eps=1e-6):
    mu1, mu2 = a.mean(0), b.mean(0)
    s1 = np.cov(a, rowvar=False) + eps * np.eye(a.shape[1])
    s2 = np.cov(b, rowvar=False) + eps * np.eye(b.shape[1])
    cs = linalg.sqrtm(s1 @ s2)
    cs = cs[0] if isinstance(cs, tuple) else cs
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * cs.real))


def kid(a, b, n_sub=100, sub=100):
    d = a.shape[1]; vals = []
    sub = min(sub, len(a), len(b))
    for _ in range(n_sub):
        x = a[RNG.choice(len(a), sub, replace=False)]
        y = b[RNG.choice(len(b), sub, replace=False)]
        kxx = (x @ x.T / d + 1) ** 3; kyy = (y @ y.T / d + 1) ** 3; kxy = (x @ y.T / d + 1) ** 3
        m = sub
        vals.append((kxx.sum() - np.trace(kxx)) / (m * (m - 1))
                    + (kyy.sum() - np.trace(kyy)) / (m * (m - 1)) - 2 * kxy.mean())
    return float(np.mean(vals)), float(np.std(vals))


def fid_boot(a, b, n=int(os.environ.get("OFFLINE_NBOOT", "500"))):
    v = [fid(a[RNG.integers(0, len(a), len(a))], b[RNG.integers(0, len(b), len(b))]) for _ in range(n)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def mean_ci(v):
    v = np.asarray(v, dtype=float)
    se = v.std(ddof=1) / np.sqrt(len(v))
    return {"mean": float(v.mean()), "ci95": [float(v.mean() - 1.96 * se), float(v.mean() + 1.96 * se)], "n": int(len(v))}


# ── main ─────────────────────────────────────────────────────────────────────
def load_shards(d):
    fs = sorted(glob.glob(os.path.join(d, "cf_shard_*.pt")))
    assert fs, f"no shards in {d}"
    blobs = [torch.load(f, map_location="cpu", weights_only=False) for f in fs]
    cat = lambda k: torch.cat([b[k] for b in blobs], 0)
    n = int(os.environ.get("OFFLINE_SMOKE", "0")) or None
    return {k: cat(k)[:n] for k in ("xn", "xp", "xc_np", "xc_pn")}


def per_direction(models, x, xc, bs=16):
    seg, lung_idx, feat, vae = models
    rows = {k: [] for k in ("share_m_weighted", "null_m_weighted", "share_m_gt05", "null_m_gt05",
                            "share_lung_gt05", "null_lung_gt05", "frac_changed_vs_x",
                            "frac_changed_vs_vae", "frac_vae_floor", "share_outside_m_gt05_vs_vae")}
    for i in range(0, len(x), bs):
        xb, cb = x[i:i + bs].to(DEV), xc[i:i + bs].to(DEV)
        S = lung_prob(seg, lung_idx, xb); m = edit_mask(S)
        rec = vae_roundtrip(vae, xb)
        d = (cb - xb).abs(); dr = (cb - rec).abs()
        for j in range(len(xb)):
            dj, mj, Sj, drj = d[j, 0], m[j, 0], S[j, 0], dr[j, 0]
            tot = dj.sum().clamp_min(1e-12)
            rows["share_m_weighted"].append(float((mj * dj).sum() / tot))
            rows["null_m_weighted"].append(float(mj.mean()))
            hard = (mj > 0.5).float(); lung = (Sj > 0.5).float()
            rows["share_m_gt05"].append(float((hard * dj).sum() / tot))
            rows["null_m_gt05"].append(float(hard.mean()))
            rows["share_lung_gt05"].append(float((lung * dj).sum() / tot))
            rows["null_lung_gt05"].append(float(lung.mean()))
            rows["frac_changed_vs_x"].append(float((dj > 0.01).float().mean()))
            rows["frac_changed_vs_vae"].append(float((drj > 0.01).float().mean()))
            rows["frac_vae_floor"].append(float(((rec[j, 0] - xb[j, 0]).abs() > 0.01).float().mean()))
            out = 1 - hard
            rows["share_outside_m_gt05_vs_vae"].append(
                float((out * drj).sum() / drj.sum().clamp_min(1e-12)))
    return {k: mean_ci(v) for k, v in rows.items()}


def main():
    out_json, specs = sys.argv[1], sys.argv[2:]
    models = load_models()
    res = {"settings": {"BG_MASK_DILATION": config.BG_MASK_DILATION,
                        "BG_MASK_FEATHER": config.BG_MASK_FEATHER,
                        "vae": config.DIFFUSION_MODEL_ID, "vae_dtype": str(VAE_DTYPE),
                        "fid_features": "densenet121-res224-rsna features, GAP, 1024-d",
                        "kid": "cubic poly kernel, unbiased, 100 subsets x 100",
                        "fid_boot": "500 resamples of both sets, percentile 95%"},
           "arms": {}}
    feats_cache = {}
    for spec in specs:
        d, _, tag = spec.partition(":"); tag = tag or os.path.basename(os.path.dirname(d.rstrip("/")))
        print(f"[offline] {tag}: {d}", flush=True)
        s = load_shards(d)
        arm = {"n_to_p": per_direction(models, s["xn"], s["xc_np"]),
               "p_to_n": per_direction(models, s["xp"], s["xc_pn"])}
        f = {k: np.concatenate([densenet_feats(models[2], s[k][i:i + 32].to(DEV))
                                for i in range(0, len(s[k]), 32)]) for k in s}
        for direc, cf, real in (("n_to_p", "xc_np", "xp"), ("p_to_n", "xc_pn", "xn")):
            km, ks = kid(f[cf], f[real])
            arm[direc]["FID"] = fid(f[cf], f[real])
            arm[direc]["FID_boot_ci95"] = fid_boot(f[cf], f[real])
            arm[direc]["KID_x1e3"] = {"mean": km * 1e3, "std": ks * 1e3}
        res["arms"][tag] = arm
        print(json.dumps(arm, indent=1), flush=True)
        json.dump(res, open(out_json, "w"), indent=1)
    print(f"[offline] wrote {out_json}")


if __name__ == "__main__":
    main()
