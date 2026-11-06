"""
Lung-restricted Lesion Hit Rate (p->n), for comparing PPO-CF against baselines that are
NOT background-locked (CycleGAN J10, Latent Shift J15/J15b). Issue D1, and review-2's
"LHR null" item.

Why: the standard LHR null is the box's share of the WHOLE image. An edit confined to the
lung beats that null just by staying in the lung (Supp. E6: in-lung share / lung-area null
~2.08 for the no-policy run vs its LHR 2.064). So a locked method beats an unlocked one on
raw LHR for free. Here both the edit mass and the null are restricted to a region R:

    hit_R  = sum(d * M * R) / sum(d * R)      d = |x_c - x|, M = radiologist box mask
    null_R = sum(M * R) / sum(R)              (what a uniform edit inside R scores)
    lift_R = hit_R / null_R

R = "lung": the undilated lung mask S > 0.5 of the SOURCE image (PSPNet, as in J14 E4a).
R = "lock": the hard background-lock region m > 0.5 (J14's edit_mask, i.e.
     generator._build_edit_mask at config dilation/feather).
R = "image": the whole frame. This reproduces src.metrics.lesion_hit_rate (sanity check).
Images whose box has no overlap with R, or with no edit mass in R, are skipped for that R.

Paired comparison against the FIRST arm on the images scorable in both: mean difference of
lift_R, 95% percentile-bootstrap CI (10,000 resamples, seed 0), paired t and Wilcoxon.
CPU. Run from ~/verification/code/49e6050 with PYTHONPATH=$PWD:$HOME/verification/scripts:
    python inlung_lhr.py OUT.json NAME=SHARD_DIR [NAME=SHARD_DIR ...]
"""
from __future__ import annotations

import json
import sys

import config  # before torchxrayvision
import ingestion
import numpy as np
import torch

import offline_metrics as om
from src.models import apply_torch_load_patch, revert_torch_load_patch

REGIONS = ("image", "lung", "lock")


def _load(d):
    import glob, os
    blobs = [torch.load(p, map_location="cpu", weights_only=False)
             for p in sorted(glob.glob(os.path.join(d, "cf_shard_*.pt")))]
    xp = torch.cat([b["xp"] for b in blobs]); xc = torch.cat([b["xc_pn"] for b in blobs])
    idx = sum((list(b["xp_idx"]) for b in blobs), [])
    return xp, xc, [int(i) for i in idx]


def main(out_path, specs):
    apply_torch_load_patch()
    import torchxrayvision as xrv
    seg = xrv.baseline_models.chestx_det.PSPNet().to(om.DEV).eval()
    revert_torch_load_patch()
    lung_idx = [seg.targets.index("Left Lung"), seg.targets.index("Right Lung")]
    dataset = ingestion.build_dataset(img_size=config.IMG_SIZE)

    regions_cache = {}   # source idx -> (lung, lock) masks; same sources for every arm
    per = {}
    for spec in specs:
        name, d = spec.split("=", 1)
        xp, xc, idx = _load(d)
        rows = {r: {} for r in REGIONS}
        for i in range(0, len(idx), 16):
            xb, cb, ib = xp[i:i + 16], xc[i:i + 16], idx[i:i + 16]
            need = [k for k, j in enumerate(ib) if j not in regions_cache]
            if need:
                S = om.lung_prob(seg, lung_idx, xb[need].to(om.DEV)).cpu()
                m = om.edit_mask(S)
                for k, s_, m_ in zip(need, S, m):
                    regions_cache[ib[k]] = ((s_ > 0.5).float()[0], (m_ > 0.5).float()[0])
            for x1, c1, j in zip(xb, cb, ib):
                M = dataset.lesion_mask(j).to(torch.float32)[0]
                if not (0.0 < float(M.mean()) < 1.0):
                    continue
                dd = (c1.float() - x1.float()).abs()
                while dd.dim() > 2:
                    dd = dd.mean(0) if dd.shape[0] > 1 else dd.squeeze(0)
                lung, lock = regions_cache[j]
                for r, R in (("image", torch.ones_like(M)), ("lung", lung), ("lock", lock)):
                    tot, area, box = float((dd * R).sum()), float(R.sum()), float((M * R).sum())
                    if tot <= 0 or box <= 0:
                        continue
                    hit, null = float((dd * M * R).sum()) / tot, box / area
                    rows[r][j] = {"hit": hit, "null": null, "lift": hit / null}
        per[name] = rows
        print(f"[inlung-lhr] {name}: " + "  ".join(
            f"{r}: lift {np.mean([v['lift'] for v in rows[r].values()]):.3f} "
            f"(null {np.mean([v['null'] for v in rows[r].values()]):.3f}, n={len(rows[r])})" for r in REGIONS))

    from scipy import stats
    names = list(per)
    ref = names[0]
    summary = {n: {r: {"lift": float(np.mean([v["lift"] for v in per[n][r].values()])),
                       "null": float(np.mean([v["null"] for v in per[n][r].values()])),
                       "hit": float(np.mean([v["hit"] for v in per[n][r].values()])),
                       "n": len(per[n][r])} for r in REGIONS} for n in names}
    paired = {}
    rng = np.random.default_rng(0)
    for n in names[1:]:
        paired[n] = {}
        for r in REGIONS:
            common = sorted(set(per[ref][r]) & set(per[n][r]))
            a = np.array([per[ref][r][j]["lift"] for j in common])
            b = np.array([per[n][r][j]["lift"] for j in common])
            dlt = a - b
            boot = dlt[rng.integers(0, len(dlt), (10000, len(dlt)))].mean(1)
            paired[n][r] = {"n": len(common), "mean_ref": float(a.mean()), "mean_arm": float(b.mean()),
                            "diff_ref_minus_arm": float(dlt.mean()),
                            "ci95": [float(x) for x in np.percentile(boot, [2.5, 97.5])],
                            "t_p": float(stats.ttest_rel(a, b).pvalue) if len(common) > 1 else None,
                            "wilcoxon_p": float(stats.wilcoxon(a, b).pvalue) if len(common) > 1 else None,
                            "ref_better": int((dlt > 0).sum())}
            p = paired[n][r]
            print(f"[inlung-lhr] {ref} - {n} [{r}]: {p['diff_ref_minus_arm']:+.3f} "
                  f"CI [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}]  t p={p['t_p']:.2g}  "
                  f"{ref} better {p['ref_better']}/{p['n']}")
    with open(out_path, "w") as fh:
        json.dump({"regions": REGIONS, "reference_arm": ref, "summary": summary, "paired_vs_ref": paired,
                   "per_image": {n: {r: {str(j): v for j, v in per[n][r].items()} for r in REGIONS}
                                 for n in names}}, fh, indent=1)
    print(f"[inlung-lhr] wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
