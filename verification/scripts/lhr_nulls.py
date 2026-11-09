"""
lhr_nulls.py - J19: does LHR lift survive nulls that know where the lungs are?

Second-review item 1. The paper's LHR null is the annotation's share of the
WHOLE image. RSNA boxes lie inside the lungs and the background lock keeps most
of the edit inside the lungs, so an edit that is uniform WITHIN the lung already
beats the whole-image null by about |image| / |lung|. This script recomputes, per
annotated p->n image and per arm, from saved shards (no regeneration):

  lift_img      (sum_A d / sum d) / (|A| / P)       the paper's metric (clearance check)
  lift_lung     (sum_{A,L} d / sum_L d) / (|A n L| / |L|),  L = lung prob > 0.5
  lift_lung_s   same with soft lung prob S as weight
  lift_m        same with the soft editable-region mask m (generator's lock mask)
  pred_lungshare  lift_img a lung-blind-within-lung edit would score given this
                edit's own in-lung share (uniform inside L, uniform outside L)
  ratio_obs_pred  lift_img / pred_lungshare
  shuf_img, shuf_lung  the same lifts with every OTHER annotated image's box
                (mean over j != i): does the edit target THIS lesion, or generic
                lesion-prone lung zones?
  with d = |x_c - x| (default), d = |x_c - D(E(x))| ("_vae": edit beyond the VAE
  round trip), and d = |D(E(x)) - x| ("floor_": the VAE round trip alone).
  uni_L_img, uni_S_img, uni_m_img: lift_img of a uniform wash over L, S, m.

D(E(x)) is the SD1.5 VAE posterior mean round trip in fp32 on CPU (offline_metrics.vae_roundtrip).
S and m come from offline_metrics.lung_prob / edit_mask (the generator's mask definition).

    python lhr_nulls.py OUT_DIR SHARD_DIR:TAG [SHARD_DIR:TAG ...]
Run from ~/verification/code/49e6050 with PYTHONPATH=$PWD:~/verification/scripts.
"""
import glob
import json
import os
import re
import sys

import numpy as np
import torch

import config  # noqa: F401  (must precede torchxrayvision, see workspace notes)
import ingestion
import offline_metrics as om

torch.set_grad_enabled(False)

PAIRS = [  # (a, b): paired tests on shared annotated indices
    ("J4", "J5_none"), ("J4", "J5c_fixed"), ("J4", "J12_fixed_tuned"), ("J4", "J9_seed1"),
    ("J4", "J16_single_common"), ("J4", "J15_latent_shift"), ("J4", "J10_cyclegan"),
    ("evalA_ens_default", "v1_single"), ("evalA_ens_default", "ens3"),
    ("evalA_ens_default", "sparsity"), ("evalA_ens_default", "gradcam"),
    ("v1_single", "v1_random"),
]
KEY_METRICS = ["lift_img", "lift_lung", "lift_lung_s", "lift_m", "ratio_obs_pred",
               "lift_img_vae", "lift_lung_vae", "shuf_img", "shuf_lung"]


def load_arm(d):
    paths = sorted(glob.glob(os.path.join(d, "cf_shard_*.pt")),
                   key=lambda p: int(re.search(r"cf_shard_(\d+)\.pt$", p).group(1)))
    assert paths, f"no shards in {d}"
    xs, cs, idx = [], [], []
    for p in paths:
        b = torch.load(p, map_location="cpu", weights_only=False)
        xs.append(b["xp"].float()); cs.append(b["xc_pn"].float()); idx += [int(i) for i in b["xp_idx"]]
    return torch.cat(xs)[:, 0], torch.cat(cs)[:, 0], idx


def wlift(d, A, W):
    """(sum W A d / sum W d) / (sum W A / sum W); NaN if undefined."""
    wd = (W * d).sum(); wa = (W * A).sum(); w = W.sum()
    if wd <= 0 or wa <= 0 or w <= 0:
        return float("nan")
    return float(((W * A * d).sum() / wd) / (wa / w))


def summ(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) < 2:
        return {"n": int(len(v))}
    from scipy import stats
    se = v.std(ddof=1) / np.sqrt(len(v))
    t, p = stats.ttest_1samp(v, 1.0)
    try:
        wp = float(stats.wilcoxon(v - 1.0).pvalue)
    except ValueError:
        wp = float("nan")
    return {"n": int(len(v)), "mean": float(v.mean()), "ci95": [float(v.mean() - 1.96 * se), float(v.mean() + 1.96 * se)],
            "median": float(np.median(v)), "t_vs_1": float(t), "p_vs_1": float(p), "wilcoxon_p_vs_1": wp}


def paired(a, b):
    from scipy import stats
    m = np.isfinite(a) & np.isfinite(b); a, b = a[m], b[m]
    if len(a) < 3:
        return {"n": int(len(a))}
    dd = a - b
    rng = np.random.default_rng(0)
    boot = np.array([dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(10000)])
    sd = dd.std(ddof=1)
    return {"n": int(len(a)), "mean_a": float(a.mean()), "mean_b": float(b.mean()), "diff": float(dd.mean()),
            "boot95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "t_p": float(stats.ttest_rel(a, b).pvalue), "wilcoxon_p": float(stats.wilcoxon(a, b).pvalue),
            "cohen_d": float(dd.mean() / sd) if sd > 0 else float("nan")}


def main():
    out_dir, specs = sys.argv[1], sys.argv[2:]
    os.makedirs(out_dir, exist_ok=True)
    dataset = ingestion.build_dataset(img_size=config.IMG_SIZE)
    seg, lung_idx, _, vae = load_models()
    cache = {}       # idx -> dict(x, A, S, L, m, rec)
    res = {"settings": {"lung_hard": "S > 0.5", "vae": "SD1.5 posterior mean, fp32 CPU",
                        "BG_MASK_DILATION": config.BG_MASK_DILATION, "BG_MASK_FEATHER": config.BG_MASK_FEATHER},
           "arms": {}, "paired": {}, "within_arm": {}}
    per_image = {}
    for spec in specs:
        d, _, tag = spec.partition(":")
        if not os.path.isdir(d):
            print(f"[J19] SKIP {tag}: {d} missing", flush=True); continue
        x, c, idx = load_arm(d)
        keep = []
        for k, i in enumerate(idx):
            A = dataset.lesion_mask(i).to(torch.float32)
            while A.dim() > 2:
                A = A.squeeze(0)
            if 0.0 < float(A.mean()) < 1.0:
                keep.append(k)
                if i not in cache:
                    cache[i] = {"x": x[k], "A": A}
        if os.environ.get("LIMIT"):
            keep = keep[:int(os.environ["LIMIT"])]
        todo = [i for i in {idx[k] for k in keep} if "S" not in cache[i]]
        for s in range(0, len(todo), 16):
            ii = todo[s:s + 16]
            xb = torch.stack([cache[i]["x"] for i in ii])[:, None].to(om.DEV)
            S = om.lung_prob(seg, lung_idx, xb); m = om.edit_mask(S); rec = om.vae_roundtrip(vae, xb)
            S, m, rec = S.float().cpu(), m.float().cpu(), rec.float().cpu()
            for j, i in enumerate(ii):
                cache[i].update(S=S[j, 0], L=(S[j, 0] > 0.5).float(), m=m[j, 0], rec=rec[j, 0])
        print(f"[J19] {tag}: {len(idx)} p->n, {len(keep)} annotated, {len(todo)} new masks/VAE", flush=True)

        rows = {k: [] for k in ["idx", "x_maxdiff_vs_cache", "lift_img", "lift_lung", "lift_lung_s", "lift_m",
                                "share_L", "pred_lungshare", "ratio_obs_pred", "lift_img_vae", "lift_lung_vae",
                                "lift_m_vae", "floor_img", "floor_lung", "uni_L_img", "uni_S_img", "uni_m_img",
                                "shuf_img", "shuf_lung", "area_A", "area_L", "A_in_L"]}
        D, Am, Lm = [], [], []
        for k in keep:
            i = idx[k]; e = cache[i]; A, L, S, m, rec = e["A"], e["L"], e["S"], e["m"], e["rec"]
            dx = (c[k] - x[k]).abs(); dv = (c[k] - rec).abs(); df = (rec - x[k]).abs()
            one = torch.ones_like(A)
            sL = float((L * dx).sum() / dx.sum().clamp_min(1e-12))
            aL, aN = float((A * L).sum() / L.sum().clamp_min(1)), float((A * (1 - L)).sum() / (1 - L).sum().clamp_min(1))
            pred = (sL * aL + (1 - sL) * aN) / float(A.mean())
            li = wlift(dx, A, one)
            vals = dict(idx=i, x_maxdiff_vs_cache=float((x[k] - e["x"]).abs().max()), lift_img=li,
                        lift_lung=wlift(dx, A, L), lift_lung_s=wlift(dx, A, S), lift_m=wlift(dx, A, m),
                        share_L=sL, pred_lungshare=pred, ratio_obs_pred=li / pred if pred > 0 else float("nan"),
                        lift_img_vae=wlift(dv, A, one), lift_lung_vae=wlift(dv, A, L), lift_m_vae=wlift(dv, A, m),
                        floor_img=wlift(df, A, one), floor_lung=wlift(df, A, L),
                        uni_L_img=wlift(L, A, one), uni_S_img=wlift(S, A, one), uni_m_img=wlift(m, A, one),
                        area_A=float(A.mean()), area_L=float(L.mean()), A_in_L=float((A * L).sum() / A.sum()))
            for kk, v in vals.items():
                rows[kk].append(v)
            D.append(dx.flatten().double()); Am.append(A.flatten().double()); Lm.append(L.flatten().double())
        # mismatched-box null: edit of image i scored against box of image j != i
        D, Am, Lm = torch.stack(D), torch.stack(Am), torch.stack(Lm)
        n = len(D)
        H = D @ Am.T                               # sum_p d_i A_j
        img = (H / D.sum(1, keepdim=True)) / Am.mean(1)[None, :]
        DL = D * Lm
        HL = DL @ Am.T                             # sum_p d_i L_i A_j
        nullL = (Lm @ Am.T) / Lm.sum(1, keepdim=True)
        lung = (HL / DL.sum(1, keepdim=True).clamp_min(1e-12)) / nullL
        lung[nullL <= 0] = float("nan")
        off = ~torch.eye(n, dtype=torch.bool)
        rows["shuf_img"] = [float(img[r][off[r]].mean()) for r in range(n)]
        rows["shuf_lung"] = [float(np.nanmean(lung[r][off[r]].numpy())) for r in range(n)]

        arr = {k: np.asarray(v, float) for k, v in rows.items()}
        per_image[tag] = arr
        res["arms"][tag] = {"shard_dir": d, "n_annotated": len(keep),
                            "x_maxdiff_vs_cache": float(arr["x_maxdiff_vs_cache"].max()),
                            **{k: summ(v) for k, v in arr.items() if k not in ("idx", "x_maxdiff_vs_cache")}}
        res["within_arm"][tag] = {
            "lift_img_minus_uni_L_img": paired(arr["lift_img"], arr["uni_L_img"]),
            "lift_img_minus_uni_m_img": paired(arr["lift_img"], arr["uni_m_img"]),
            "lift_img_minus_shuf_img": paired(arr["lift_img"], arr["shuf_img"]),
            "lift_lung_minus_shuf_lung": paired(arr["lift_lung"], arr["shuf_lung"]),
            "lift_img_vae_minus_floor_img": paired(arr["lift_img_vae"], arr["floor_img"]),
        }
        a = res["arms"][tag]
        print(f"[J19] {tag}: lift_img {a['lift_img'].get('mean', float('nan')):.3f}  lung {a['lift_lung'].get('mean', float('nan')):.3f}"
              f"  m {a['lift_m'].get('mean', float('nan')):.3f}  obs/pred {a['ratio_obs_pred'].get('mean', float('nan')):.3f}"
              f"  uniL {a['uni_L_img'].get('mean', float('nan')):.3f}  shuf_img {a['shuf_img'].get('mean', float('nan')):.3f}"
              f"  shuf_lung {a['shuf_lung'].get('mean', float('nan')):.3f}  floor {a['floor_img'].get('mean', float('nan')):.3f}", flush=True)
        json.dump(res, open(os.path.join(out_dir, "lhr_nulls.json"), "w"), indent=1)

    for a, b in PAIRS:
        if a not in per_image or b not in per_image:
            continue
        ia = {int(i): r for r, i in enumerate(per_image[a]["idx"])}
        ib = {int(i): r for r, i in enumerate(per_image[b]["idx"])}
        sh = sorted(set(ia) & set(ib))
        res["paired"][f"{a}__vs__{b}"] = {m: paired(per_image[a][m][[ia[i] for i in sh]],
                                                    per_image[b][m][[ib[i] for i in sh]]) for m in KEY_METRICS}
    json.dump(res, open(os.path.join(out_dir, "lhr_nulls.json"), "w"), indent=1)
    np.savez(os.path.join(out_dir, "lhr_nulls_per_image.npz"),
             **{f"{t}__{k}": v for t, arr in per_image.items() for k, v in arr.items()})
    print(f"[J19] wrote {out_dir}", flush=True)


def load_models():
    from src.models import apply_torch_load_patch, revert_torch_load_patch
    apply_torch_load_patch()
    import torchxrayvision as xrv
    seg = xrv.baseline_models.chestx_det.PSPNet().to(om.DEV).eval()
    revert_torch_load_patch()
    lung_idx = [seg.targets.index("Left Lung"), seg.targets.index("Right Lung")]
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(config.DIFFUSION_MODEL_ID, subfolder="vae",
                                        torch_dtype=om.VAE_DTYPE).to(om.DEV).eval()
    return seg, lung_idx, None, vae


if __name__ == "__main__":
    main()
