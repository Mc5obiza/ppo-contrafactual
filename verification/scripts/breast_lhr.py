"""
breast_lhr.py - J22: Lesion Hit Rate on the CBIS-DDSM (mammography) counterfactuals,
in BOTH directions. Second review, item 6 ("n->p localization anchor"), exploratory.

CBIS-DDSM has no normal mammograms: every image carries at least one radiologist ROI
mask, so benign->malignant (n->p) AND malignant->benign (p->n) edits can both be scored
against ground truth, which RSNA cannot do for n->p.

Inputs (no regeneration, CPU):
  * shards  checkpoints/cbis_ddsm_breast/eval_shards  (test-cancer branch, Aug 9 2026)
  * indices checkpoints/cbis_ddsm_breast/eval_group_index.json (breast_shard_audit.py
            gate B; xn_idx / xp_idx in shard order). Every image is re-checked here:
            dataset[idx] must equal the shard source pixels, or the image is dropped.
  * ROI     union of ALL ROI masks of an image (the dataset's _meta keeps only the
            first; rebuilt here from the case tables with the domain's own resolvers),
            resized exactly like the image (torchvision Resize to 256^2), > 0.5.
            Masks whose size differs from the full mammogram, or that are not binary
            (> 1% of pixels in 21..234: a cropped patch filed under the ROI series), are dropped and counted.
  * region  B = Breast channel (> 0.5) of the analytic BreastAnatomySegmenter from
            test-cancer (breast_segmentation_testcancer.py), the organ mask that gated
            the background lock in this run.

Per image, d = |x_c - x|:
  lift_img      whole-image null (pointing-game convention)
  lift_breast   edit mass and null both restricted to B  (the main paper's in-organ LHR)
  lift_*_vae    same with d = |x_c - D(E(x))|  (SD1.5 VAE posterior mean, fp32 CPU)
  floor_*       d = |D(E(x)) - x|
  uni_B_img     lift_img of a uniform wash over B
  shuf_breast   in-breast lift against every OTHER image's ROI (same direction), mean over j

    python breast_lhr.py OUT_DIR
Run from the repository root ($PPO_CF_REPO) with
PYTHONPATH=$PWD:$HOME/verification/scripts.
"""
import glob
import json
import os
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from scipy import stats

import config  # project config first
config.DEVICE = "cpu"
from domains import cbis_ddsm_breast as cb
import breast_segmentation_testcancer as bseg

torch.set_grad_enabled(False)
REPO = os.environ['PPO_CF_REPO']
SHARDS = f"{REPO}/checkpoints/cbis_ddsm_breast/eval_shards"
INDEX = f"{REPO}/checkpoints/cbis_ddsm_breast/eval_group_index.json"
RNG = np.random.default_rng(0)
LIMIT = int(os.environ.get("LIMIT", "0")) or None


def wlift(d, A, W):
    wd = (W * d).sum(); wa = (W * A).sum(); w = W.sum()
    if wd <= 0 or wa <= 0 or w <= 0:
        return float("nan")
    return float(((W * A * d).sum() / wd) / (wa / w))


def summ(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) < 2:
        return {"n": int(len(v))}
    se = v.std(ddof=1) / np.sqrt(len(v))
    t, p = stats.ttest_1samp(v, 1.0)
    return {"n": int(len(v)), "mean": float(v.mean()), "ci95": [float(v.mean() - 1.96 * se), float(v.mean() + 1.96 * se)],
            "median": float(np.median(v)), "t_vs_1": float(t), "p_vs_1": float(p),
            "wilcoxon_p_vs_1": float(stats.wilcoxon(v - 1.0).pvalue)}


def paired(a, b):
    m = np.isfinite(a) & np.isfinite(b); a, b = a[m], b[m]
    dd = a - b
    boot = np.array([dd[RNG.integers(0, len(dd), len(dd))].mean() for _ in range(10000)])
    return {"n": int(len(a)), "diff": float(dd.mean()), "boot95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "t_p": float(stats.ttest_rel(a, b).pvalue), "wilcoxon_p": float(stats.wilcoxon(a, b).pvalue)}


def roi_union_index(root):
    """image path -> list of ALL resolved ROI paths, using the domain's own resolvers."""
    df = cb._load_case_table(root)
    uid = cb._build_uid_index(root, cb.SERIES_FULL)
    roi = cb._build_uid_index(root, cb.SERIES_ROI)
    df["_path"] = df[cb.COL_IMAGE].map(lambda p: cb._resolve(p, uid))
    df["_roi"] = df[cb.COL_ROI].map(lambda p: cb._resolve(p, roi))
    out = {}
    for path, g in df[df["_path"].notna()].groupby("_path"):
        out[path] = sorted({r for r in g["_roi"] if isinstance(r, str)})
    return out


def load_roi(paths, full_size, resize):
    """Union mask at 256^2, or (None, reason)."""
    acc = None
    for p in paths:
        im = Image.open(p).convert("L")
        if im.size != full_size:
            return None, "size_mismatch"
        arr = np.asarray(im)
        # JPEG masks carry ~16 grey levels of compression noise near 0 and 255; a
        # cropped patch filed under the ROI series instead has real mid-range content.
        if ((arr > 20) & (arr < 235)).mean() > 0.01:
            return None, "not_binary"
        m = T.ToTensor()(resize(Image.fromarray(((arr > 127) * 255).astype(np.uint8))))[0]
        acc = m if acc is None else torch.maximum(acc, m)
    if acc is None:
        return None, "no_roi"
    A = (acc > 0.5).float()
    if A.sum() == 0:
        return None, "empty_after_resize"
    return A, "ok"


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    root = config.CBIS_DDSM_ROOT
    ds = cb.CBISDDSMDataset(root=root, img_size=256)
    rois = roi_union_index(root)
    gi = json.load(open(INDEX))
    paths = sorted(glob.glob(os.path.join(SHARDS, "cf_shard_*.pt")),
                   key=lambda p: int(re.search(r"(\d+)\.pt$", p).group(1)))
    bl = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    cat = lambda k: torch.cat([b[k] for b in bl]).float()[:, 0]
    groups = {"n_to_p": (cat("xn"), cat("xc_np"), gi["xn_idx"]), "p_to_n": (cat("xp"), cat("xc_pn"), gi["xp_idx"])}

    seg = bseg.BreastAnatomySegmenter(device="cpu", exclude_pectoral=True).eval()
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(config.DIFFUSION_MODEL_ID, subfolder="vae", torch_dtype=torch.float32).eval()
    resize = T.Resize((256, 256))

    def vae_rt(x):  # (B,1,H,W) in [0,1]
        z = vae.encode(x.repeat(1, 3, 1, 1) * 2 - 1).latent_dist.mean
        y = ((vae.decode(z).sample.clamp(-1, 1) + 1) / 2)
        return (0.299 * y[:, 0:1] + 0.587 * y[:, 1:2] + 0.114 * y[:, 2:3])

    res = {"settings": {"region": "BreastAnatomySegmenter channel 0 > 0.5 (test-cancer)", "roi": "union of all ROI masks, Resize 256, > 0.5",
                        "vae": config.DIFFUSION_MODEL_ID + " posterior mean fp32 CPU", "limit": LIMIT}, "directions": {}}
    per = {}
    for direc, (X, XC, IDX) in groups.items():
        X, XC, IDX = X[:LIMIT], XC[:LIMIT], IDX[:LIMIT]
        drop = {}
        keep = []
        for k, i in enumerate(IDX):
            xi = ds[int(i)][0][0]
            if float((xi - X[k]).abs().max()) > 1e-6:
                drop["pixel_mismatch"] = drop.get("pixel_mismatch", 0) + 1; continue
            path = ds.samples[int(i)][0]
            full = Image.open(path).size
            A, why = load_roi(rois.get(path, []), full, resize)
            if A is None:
                drop[why] = drop.get(why, 0) + 1; continue
            keep.append((k, int(i), A))
        print(f"[J22] {direc}: {len(IDX)} images, {len(keep)} kept, dropped {drop}", flush=True)
        ks = [k for k, _, _ in keep]
        xb = X[ks][:, None]
        B = torch.sigmoid(seg(xb * 2048.0 - 1024.0))[:, 0]
        Bh = (B > 0.5).float()
        rec = torch.cat([vae_rt(xb[s:s + 8]) for s in range(0, len(xb), 8)])[:, 0]
        rows = {m: [] for m in ["idx", "label", "lift_img", "lift_breast", "lift_img_vae", "lift_breast_vae", "floor_img", "floor_breast",
                                "uni_B_img", "share_B", "area_A", "area_B", "A_in_B"]}
        D, Am = [], []
        for j, (k, i, A) in enumerate(keep):
            x, c, Bj, r = X[k], XC[k], Bh[j], rec[j]
            dx, dv, df = (c - x).abs(), (c - r).abs(), (r - x).abs()
            one = torch.ones_like(A)
            vals = dict(idx=i, label=ds.samples[i][1], lift_img=wlift(dx, A, one), lift_breast=wlift(dx, A, Bj),
                        lift_img_vae=wlift(dv, A, one), lift_breast_vae=wlift(dv, A, Bj),
                        floor_img=wlift(df, A, one), floor_breast=wlift(df, A, Bj), uni_B_img=wlift(Bj, A, one),
                        share_B=float((Bj * dx).sum() / dx.sum().clamp_min(1e-12)), area_A=float(A.mean()),
                        area_B=float(Bj.mean()), A_in_B=float((A * Bj).sum() / A.sum()))
            for m, v in vals.items():
                rows[m].append(v)
            D.append((dx * Bj).flatten().double()); Am.append(A.flatten().double())
        D, Am = torch.stack(D), torch.stack(Am)
        Bf = Bh.flatten(1).double()
        H = D @ Am.T                                   # sum d_i B_i A_j
        nullB = (Bf @ Am.T) / Bf.sum(1, keepdim=True)  # |A_j n B_i| / |B_i|
        L = (H / D.sum(1, keepdim=True)) / nullB
        L[nullB <= 0] = float("nan")
        off = ~torch.eye(len(D), dtype=torch.bool)
        rows["shuf_breast"] = [float(np.nanmean(L[r][off[r]].numpy())) for r in range(len(D))]
        arr = {m: np.asarray(v, float) for m, v in rows.items()}
        per[direc] = arr
        res["directions"][direc] = {"n_images": len(IDX), "n_kept": len(keep), "dropped": drop,
                                    **{m: summ(v) for m, v in arr.items() if m not in ("idx", "label")},
                                    "own_minus_shuf_breast": paired(arr["lift_breast"], arr["shuf_breast"]),
                                    "lift_img_minus_uni_B": paired(arr["lift_img"], arr["uni_B_img"])}
        a = res["directions"][direc]
        print(f"[J22] {direc}: lift_img {a['lift_img']['mean']:.3f}  breast {a['lift_breast']['mean']:.3f} "
              f"{[round(v, 3) for v in a['lift_breast']['ci95']]} p={a['lift_breast']['p_vs_1']:.2g}  vae {a['lift_breast_vae']['mean']:.3f}  "
              f"floor {a['floor_breast']['mean']:.3f}  uniB {a['uni_B_img']['mean']:.3f}  shuf {a['shuf_breast']['mean']:.3f}  "
              f"shareB {a['share_B']['mean']:.3f}", flush=True)
        json.dump(res, open(os.path.join(out_dir, "breast_lhr.json"), "w"), indent=1)
    np.savez(os.path.join(out_dir, "breast_lhr_per_image.npz"), **{f"{d}__{m}": v for d, a in per.items() for m, v in a.items()})
    print(f"[J22] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
