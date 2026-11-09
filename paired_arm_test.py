"""
paired_arm_test.py
==================
Paired significance test between two arms, on the images they share.

WHY THIS AND NOT THE SUMMARY TABLE
----------------------------------
compare_arms.py prints each arm's mean LHR lift. Reading "2.293 vs 2.204" off
that table and calling it a difference is exactly the mistake this project has
already made once: an unpaired eyeball comparison of two summary statistics, with
no interval and no test.

Every arm generated counterfactuals for the SAME dataset indices from the SAME
source images — only the sampling schedule differed. That makes this a paired
design, and a paired test is both the correct instrument and a far more sensitive
one: per-image lesion-box area varies enormously (the null baseline ranges over
an order of magnitude), and pairing cancels it exactly.

The question it answers is the Phase 2 gate:

    does the TRAINED policy edit closer to the radiologist's box than an
    UNTRAINED one with the same action bounds and the same step budget?

A null result here means 800 episodes of PPO bought nothing measurable, and the
honest presentation is gradient-guided diffusion without the RL claim.

    module load python/3.12.8 && source venv/bin/activate
    python paired_arm_test.py trained random
"""

from __future__ import annotations

import glob
import os
import re
import sys

import numpy as np
import torch

import config
import ingestion


def _shard_dir(arm: str) -> str:
    """Accept either an arm name ('trained', 'none', 'random') or a shard
    directory path, so the Phase 3a countermeasure arms — which live in tagged
    directories like eval_shards_none_cm1 — can be compared with the same
    paired machinery, rather than a second implementation that could drift."""
    if os.path.isdir(arm):
        return arm
    return os.path.join(
        config.CKPT_DIR,
        "eval_shards" if arm == "trained" else f"eval_shards_{arm}",
    )


def _load_arm(arm: str) -> tuple[list, list, list]:
    """Return (xp, xc_pn, xp_idx) concatenated in global shard order."""
    paths = sorted(glob.glob(os.path.join(_shard_dir(arm), "cf_shard_*.pt")))
    if not paths:
        raise FileNotFoundError(f"no shards for arm '{arm}' in {_shard_dir(arm)}")
    blobs = {}
    for p in paths:
        m = re.search(r"cf_shard_(\d+)\.pt$", p)
        blobs[int(m.group(1))] = torch.load(p, map_location="cpu")
    order = sorted(blobs)
    unstack = lambda t: [t[i : i + 1] for i in range(t.shape[0])]
    xp = [im for k in order for im in unstack(blobs[k]["xp"])]
    xc = [im for k in order for im in unstack(blobs[k]["xc_pn"])]
    idx = [i for k in order for i in blobs[k]["xp_idx"]]
    return xp, xc, idx


def _per_image_lift(xp, xc, idx, dataset) -> dict[int, float]:
    """hit_rate / null per dataset index — the same computation as
    metrics.lesion_hit_rate, kept per-image instead of averaged."""
    out = {}
    for x, c, i in zip(xp, xc, idx):
        mask = dataset.lesion_mask(int(i)).to(torch.float32)
        null = float(mask.mean().item())
        if null <= 0.0 or null >= 1.0:
            continue
        d = (c.float() - x.float()).abs()
        while d.dim() > 3:
            d = d.squeeze(0)
        if d.shape[0] > 1:
            d = d.mean(dim=0, keepdim=True)
        total = float(d.sum().item())
        if total <= 0:
            continue
        out[int(i)] = (float((d * mask).sum().item()) / total) / null
    return out


def main(argv: list[str]) -> int:
    a_name, b_name = (argv + ["trained", "random"])[:2]
    print(f"\nPAIRED ARM TEST — {a_name} vs {b_name}  (LHR lift, p→n)\n" + "=" * 62)

    dataset = ingestion.build_dataset(img_size=config.IMG_SIZE)

    a = _per_image_lift(*_load_arm(a_name), dataset)
    b = _per_image_lift(*_load_arm(b_name), dataset)

    shared = sorted(set(a) & set(b))
    if not shared:
        print("no shared annotated images between the two arms — cannot pair.")
        return 1

    va = np.array([a[i] for i in shared])
    vb = np.array([b[i] for i in shared])
    d = va - vb

    from scipy import stats

    t, p = stats.ttest_rel(va, vb)
    try:
        w_stat, w_p = stats.wilcoxon(va, vb)
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")

    # Bootstrap CI on the paired mean difference — the interval is what says
    # "indistinguishable" rather than merely "not significant".
    rng = np.random.default_rng(0)
    boot = np.array([
        d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)
    ])
    lo, hi = np.percentile(boot, [2.5, 97.5])

    cohen_d = d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else float("nan")

    print(f"paired images            : {len(shared)}")
    print(f"mean lift {a_name:<14s}: {va.mean():.4f}  (sd {va.std(ddof=1):.4f})")
    print(f"mean lift {b_name:<14s}: {vb.mean():.4f}  (sd {vb.std(ddof=1):.4f})")
    print(f"paired difference       : {d.mean():+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"paired t-test           : t={t:.3f}  p={p:.4g}")
    print(f"Wilcoxon signed-rank    : W={w_stat:.1f}  p={w_p:.4g}")
    print(f"Cohen's d (paired)      : {cohen_d:+.3f}")
    print(f"{a_name} better on       : {int((d > 0).sum())} / {len(d)} images")

    print("\nREADING")
    if p < 0.05:
        print(f"  The difference is statistically detectable (p={p:.3g}), but read "
              f"the\n  effect size before calling it meaningful: a lift gap of "
              f"{d.mean():+.3f}\n  against a between-image sd of {va.std(ddof=1):.3f}.")
    else:
        print(f"  No detectable difference (p={p:.3g}). With {len(shared)} paired "
              f"images the\n  95% CI on the gap is [{lo:+.4f}, {hi:+.4f}] — the "
              f"data are consistent with\n  the trained policy editing no closer "
              f"to the lesion than an untrained one.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
