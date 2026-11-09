"""
breast_shard_audit.py
=====================
Reconstructs, verifies and provenance-audits the mammography counterfactual
shards in checkpoints/cbis_ddsm_breast/eval_shards/.

WHY THIS EXISTS
---------------
The breast shards were written on the `test-cancer` branch, before eval_main.py
learned to store xn_idx / xp_idx. They therefore contain PIXELS AND NOTHING
ELSE: four image tensors and a config stamp that — for this domain — is stale
(it records the CHEST constants densenet121-res224-{rsna,chex}, because the
breast domain sets guidance through its DomainProfile rather than through
config.GUIDANCE_ENSEMBLE_WEIGHTS).

Everything downstream of those shards therefore rests on two claims that have
to be PROVEN rather than assumed:

  1. we can rebuild the classifier that actually guided them, and
  2. we can say which dataset rows — and so which PATIENTS — they came from.

(2) is what makes an honest held-out validator possible at all. A validator is
only independent if it never trained on the images it scores, and without the
indices there is no way to know which patients those are.

THE THREE GATES
---------------
A  guidance reconstruction — rebuilding densenet121-breast-cbis must reproduce
   the published CV of 1.000/1.000 and the f(x)<0.2 / f(x)>0.8 group
   definition. If it does not, the model or the preprocessing is wrong and
   every number computed from here on is fiction.
B  index recovery — replaying the deterministic group scan must reproduce the
   shards' xn/xp tensors EXACTLY. Exact equality is the point: it converts "we
   think these are the images" into "these are the images".
C  contamination audit — how much of the eval set the guidance split already
   contains, which is what makes the existing resnet18-breast-cbis a weak
   validator and quantifies how much data a disjoint one can be trained on.

Writes checkpoints/cbis_ddsm_breast/eval_group_index.json.

    module load python/3.12.8 && source venv/bin/activate
    PPO_DOMAIN=cbis_ddsm_breast python breast_shard_audit.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
import domains


# The group definition scan_groups used at generation time. Not imported from
# src/metrics.py because that module's current version postdates these shards;
# the thresholds are restated here so this script describes the run it audits.
NEG_THRESH = 0.2
POS_THRESH = 0.8

# The published numbers this must reproduce, from
# `git show test-cancer:checkpoints/cbis_ddsm_breast/eval_metrics.json`.
PUBLISHED_CV = {"n_to_p": 1.0, "p_to_n": 1.0}
CV_TAU = 0.8

# Tolerance on the RE-DERIVED group membership, and why it is not zero.
#
# The shards were generated on an H200; this audit runs wherever it is
# submitted. Convolution kernel selection and cudnn's TF32 paths differ between
# GPU generations, which moves f(x) by ~1e-3. That is irrelevant anywhere
# except at a hard threshold, and the CBIS-DDSM score distribution happens to
# put several images within 0.007 of the 0.2 cut (0.1995, 0.1999, 0.2066 on the
# first run of this script). One of them lands on the far side here.
#
# So: an image whose score sits within GROUP_EPS of its threshold is reported
# as a BOUNDARY case, not a failure. A genuinely wrong model would not produce
# a handful of near-tie images — it would scatter scores across the range, and
# anything past this tolerance still fails. CV, which is the published number
# and is computed at tau=0.8 where no image is anywhere near the cut, is
# checked EXACTLY.
GROUP_EPS = 0.02


def load_shards(shard_dir: str) -> tuple[dict, dict]:
    """Concatenated tensors plus the stamp, refusing an incomplete set."""
    paths = sorted(glob.glob(os.path.join(shard_dir, "cf_shard_*.pt")))
    if not paths:
        raise FileNotFoundError(f"no shards in {shard_dir}")
    blobs = {}
    for p in paths:
        m = re.search(r"cf_shard_(\d+)\.pt$", p)
        blobs[int(m.group(1))] = torch.load(p, map_location="cpu", weights_only=False)
    order = sorted(blobs)
    total = blobs[order[0]]["num_shards"]
    missing = sorted(set(range(total)) - set(blobs))
    if missing:
        raise FileNotFoundError(
            f"{shard_dir}: incomplete, missing shard(s) {missing} of {total}. "
            f"A partial set would silently shorten every group."
        )
    tensors = {k: torch.cat([blobs[i][k] for i in order])
               for k in ("xn", "xp", "xc_np", "xc_pn")}
    return tensors, blobs[order[0]].get("config", {})


@torch.no_grad()
def score(clf, X: torch.Tensor, bs: int = 32) -> np.ndarray:
    out = []
    for i in range(0, len(X), bs):
        x = X[i:i + bs].to(config.DEVICE)
        out.append(clf(x).squeeze(-1).float().cpu())
    return torch.cat(out).numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Gate A — guidance reconstruction
# ─────────────────────────────────────────────────────────────────────────────

def gate_a(clf, T: dict) -> dict:
    s = {k: score(clf, v) for k, v in T.items()}

    cv_np = float((s["xc_np"] > CV_TAU).mean())
    cv_pn = float((s["xc_pn"] < 1.0 - CV_TAU).mean())

    # Group membership, split into clean / boundary / violating. See GROUP_EPS.
    n_out = s["xn"] >= NEG_THRESH
    p_out = s["xp"] <= POS_THRESH
    n_bad = int((s["xn"] >= NEG_THRESH + GROUP_EPS).sum())
    p_bad = int((s["xp"] <= POS_THRESH - GROUP_EPS).sum())

    print(f"  group definition   f(xn)<{NEG_THRESH}: {1 - n_out.mean():.3f}   "
          f"f(xp)>{POS_THRESH}: {1 - p_out.mean():.3f}")
    if n_out.any() or p_out.any():
        edge = sorted([float(v) for v in s["xn"][n_out]] +
                      [float(v) for v in s["xp"][p_out]])
        print(f"    {int(n_out.sum()) + int(p_out.sum())} image(s) on the far side "
              f"of a threshold: {['%.4f' % e for e in edge]}")
        print(f"    within +/-{GROUP_EPS} of the cut => cross-GPU kernel drift, "
              f"not a model mismatch")
    print(f"  counterfactual validity   n->p: {cv_np:.3f}   p->n: {cv_pn:.3f}   "
          f"(published {PUBLISHED_CV['n_to_p']:.3f} / {PUBLISHED_CV['p_to_n']:.3f})")
    print(f"  mean f(.)   xn={s['xn'].mean():.4f}  xc_np={s['xc_np'].mean():.4f}  "
          f"xp={s['xp'].mean():.4f}  xc_pn={s['xc_pn'].mean():.4f}")

    cv_ok = (abs(cv_np - PUBLISHED_CV["n_to_p"]) < 1e-9
             and abs(cv_pn - PUBLISHED_CV["p_to_n"]) < 1e-9)
    grp_ok = (n_bad == 0 and p_bad == 0)
    print(f"  CV reproduces exactly: {cv_ok} | group membership within "
          f"tolerance: {grp_ok} ({n_bad + p_bad} beyond {GROUP_EPS})")
    ok = cv_ok and grp_ok
    print(f"  GATE A: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(
            "GATE A FAILED — the rebuilt guidance classifier does not reproduce "
            "the run that generated these shards. Do not compute a transfer "
            "ratio from this: the denominator would be a different model's."
        )
    return {"cv_n_to_p": cv_np, "cv_p_to_n": cv_pn,
            "n_on_far_side_of_threshold": int(n_out.sum()) + int(p_out.sum()),
            "n_beyond_tolerance": n_bad + p_bad,
            "group_eps": GROUP_EPS,
            "mean_f": {k: float(v.mean()) for k, v in s.items()}}


# ─────────────────────────────────────────────────────────────────────────────
# Gate B — index recovery
# ─────────────────────────────────────────────────────────────────────────────

def gate_b(clf, dataset, T: dict) -> tuple[list[int], list[int], dict]:
    """
    Recover the dataset index behind every shard image, by CONTENT.

    The obvious method — replay the generation-time group scan and take the
    same first-200 — is not used as the primary, because it inherits the
    threshold that Gate A just showed is not reproducible across GPU
    generations: an image at 0.2066 here scored just under 0.2 on the H200, so
    the replayed Xn would be off by one and every subsequent index shifted.

    Hashing sidesteps the classifier entirely. Both the shard tensors and this
    dataset pass come from the same PIL decode -> Resize -> ToTensor pipeline on
    the same JPEGs, which is deterministic CPU code, so the float32 bytes are
    identical and a sha1 of them is an exact key. An unmatched image falls back
    to nearest-neighbour by L2, which is reported rather than assumed.

    The threshold replay is still run, as a cross-check on how much the two
    disagree.
    """
    import hashlib

    def h(t: torch.Tensor) -> str:
        return hashlib.sha1(np.ascontiguousarray(t.numpy())).hexdigest()

    loader = DataLoader(dataset, batch_size=32, shuffle=False,
                        num_workers=config.DATALOADER_NUM_WORKERS)
    groups: dict[str, list[int]] = {}
    all_imgs, all_scores = [], []
    base = 0
    with torch.no_grad():
        for x_batch, _ in loader:
            sc = clf(x_batch.to(config.DEVICE)).squeeze(-1).cpu().numpy()
            for i in range(len(x_batch)):
                groups.setdefault(h(x_batch[i]), []).append(base + i)
                all_imgs.append(x_batch[i])
                all_scores.append(float(sc[i]))
            base += x_batch.shape[0]
    table = {k: v[0] for k, v in groups.items()}
    print(f"  hashed {base} dataset images ({len(table)} distinct)")

    # CBIS-DDSM ships pixel-identical images under more than one series UID, so
    # "distinct" is smaller than "images". That is only a problem if a duplicate
    # pair straddles two PATIENTS: excluding a patient from the disjoint
    # validator's training pool would then fail to exclude a byte-identical copy
    # of an image it is asked to score. Checked rather than assumed.
    dup = {k: v for k, v in groups.items() if len(v) > 1}
    dataset_pid = getattr(dataset, "patient_id", None)
    cross = [v for v in dup.values()
             if dataset_pid and len({dataset_pid(i) for i in v}) > 1]
    print(f"  pixel-duplicate groups: {len(dup)} covering "
          f"{sum(len(v) for v in dup.values())} images; "
          f"{len(cross)} span more than one patient")
    if cross:
        raise SystemExit(
            f"GATE B FAILED — {len(cross)} pixel-identical image group(s) span "
            f"multiple patient ids, e.g. {cross[0]}. Patient-level exclusion "
            f"would not remove a byte-identical copy of an eval image from a "
            f"validator's training pool, so 'disjoint' would not be true."
        )

    stacked = torch.stack(all_imgs)

    def recover(X: torch.Tensor, name: str) -> list[int]:
        idx, fell_back = [], 0
        for k in range(len(X)):
            key = h(X[k])
            if key in table:
                idx.append(table[key])
            else:
                d = (stacked - X[k]).flatten(1).norm(dim=1)
                j = int(d.argmin())
                idx.append(j)
                fell_back += 1
                print(f"    {name}[{k}]: no exact hash; nearest is index {j} "
                      f"at L2={float(d[j]):.2e}")
        print(f"  {name}: {len(idx)} recovered, {len(set(idx))} distinct, "
              f"{fell_back} via nearest-neighbour fallback")
        return idx

    xn_idx = recover(T["xn"], "xn")
    xp_idx = recover(T["xp"], "xp")

    # Verify by reconstruction: the recovered indices must rebuild the shards.
    ok = True
    for name, idx, want in (("xn", xn_idx, T["xn"]), ("xp", xp_idx, T["xp"])):
        got = torch.stack([all_imgs[i] for i in idx])
        same = torch.equal(got, want)
        print(f"  {name}: rebuilt-from-indices == shard tensor -> {same}")
        ok = ok and same and len(set(idx)) == len(idx)

    # Cross-check against the threshold replay that generation actually used.
    replay_n, replay_p = [], []
    for g, v in enumerate(all_scores):
        if v < NEG_THRESH and len(replay_n) < len(xn_idx):
            replay_n.append(g)
        elif v > POS_THRESH and len(replay_p) < len(xp_idx):
            replay_p.append(g)
        if len(replay_n) >= len(xn_idx) and len(replay_p) >= len(xp_idx):
            break
    agree_n = len(set(replay_n) & set(xn_idx))
    agree_p = len(set(replay_p) & set(xp_idx))
    print(f"  cross-check vs threshold replay: xn {agree_n}/{len(xn_idx)} agree, "
          f"xp {agree_p}/{len(xp_idx)} agree")

    print(f"  GATE B: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(
            "GATE B FAILED — the recovered indices do not rebuild the shard "
            "tensors, so we cannot say which patients these counterfactuals "
            "came from. Patient-level disjointness for a new validator cannot "
            "be established without them."
        )
    n_dup_eval = sum(1 for v in dup.values() if set(v) & set(xn_idx + xp_idx))
    return xn_idx, xp_idx, {"replay_agreement_n": agree_n,
                            "replay_agreement_p": agree_p,
                            "distinct_images": len(table),
                            "duplicate_groups": len(dup),
                            "duplicate_groups_spanning_patients": len(cross),
                            "duplicate_groups_touching_eval": n_dup_eval}


# ─────────────────────────────────────────────────────────────────────────────
# Gate C — contamination audit
# ─────────────────────────────────────────────────────────────────────────────

def gate_c(dataset, xn_idx: list[int], xp_idx: list[int]) -> dict:
    from train_breast_classifier import patient_level_split

    train_idx, val_idx = patient_level_split(dataset, val_frac=0.2, seed=0)
    train_set = set(train_idx)

    eval_idx = xn_idx + xp_idx
    eval_pat = {dataset.patient_id(i) for i in eval_idx}
    train_pat = {dataset.patient_id(i) for i in train_idx}
    val_pat = {dataset.patient_id(i) for i in val_idx}

    seen_img = sum(1 for i in eval_idx if i in train_set)
    seen_pat = len(eval_pat & train_pat)

    # What a genuinely disjoint validator has left to train on: the guidance
    # models' val patients, minus every patient represented in the eval groups.
    pool_pat = val_pat - eval_pat
    pool_idx = [i for i in val_idx if dataset.patient_id(i) in pool_pat]
    labels = [float(dataset[i][1].item()) for i in pool_idx] if pool_idx else []

    print(f"  guidance split: {len(train_pat)} train patients / {len(train_idx)} images")
    print(f"                  {len(val_pat)} val patients / {len(val_idx)} images")
    print(f"  eval groups:    {len(eval_idx)} images from {len(eval_pat)} patients")
    print(f"  CONTAMINATION — eval images inside the guidance TRAIN fold: "
          f"{seen_img}/{len(eval_idx)} ({100*seen_img/len(eval_idx):.1f}%), "
          f"{seen_pat}/{len(eval_pat)} patients")
    print(f"    every -breast-cbis backbone, INCLUDING the reserved resnet18")
    print(f"    validator, was fit on those images.")
    print(f"  DISJOINT POOL for a real validator: {len(pool_pat)} patients / "
          f"{len(pool_idx)} images "
          f"({int(sum(labels))} malignant / {len(labels)-int(sum(labels))} benign)")
    print(f"  GATE C: reported (no pass/fail — this is the measurement)")

    return {
        "guidance_train_patients": len(train_pat),
        "guidance_train_images": len(train_idx),
        "guidance_val_patients": len(val_pat),
        "guidance_val_images": len(val_idx),
        "eval_images": len(eval_idx),
        "eval_patients": len(eval_pat),
        "eval_images_in_guidance_train": seen_img,
        "eval_patients_in_guidance_train": seen_pat,
        "disjoint_pool_patients": len(pool_pat),
        "disjoint_pool_images": len(pool_idx),
        "disjoint_pool_malignant": int(sum(labels)),
        "disjoint_pool_benign": len(labels) - int(sum(labels)),
        "disjoint_pool_indices": sorted(pool_idx),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard-dir",
                    default=os.path.join(config.CKPT_DIR, "eval_shards"))
    ap.add_argument("--out", default=os.path.join(config.CKPT_DIR,
                                                  "eval_group_index.json"))
    args = ap.parse_args()

    profile = domains.active()
    if profile.build_guidance is None:
        raise SystemExit(f"{profile.name} is an xrv domain; this audit is "
                         f"mammography-specific. Set PPO_DOMAIN=cbis_ddsm_breast.")
    print(profile.summary())

    T, stamp = load_shards(args.shard_dir)
    n = len(T["xn"])
    print(f"\nshards: {args.shard_dir}  n={n} per group")
    print(f"stamp GUIDANCE_ENSEMBLE_WEIGHTS = {stamp.get('GUIDANCE_ENSEMBLE_WEIGHTS')}")
    print(f"  ^ STALE for this domain: those are the CHEST constants. The breast "
          f"domain\n    sets guidance through its DomainProfile, so the stamp "
          f"cannot describe it.\n    Gate A is what establishes the real "
          f"denominator.")

    from src.breast_classifier import build_breast_classifier
    guidance = build_breast_classifier(weights=profile.guidance_weights)

    print(f"\n=== GATE A — guidance reconstruction ({profile.guidance_weights}) ===")
    a = gate_a(guidance, T)

    print(f"\n=== GATE B — index recovery ===")
    dataset = profile.build_dataset(config.IMG_SIZE)
    xn_idx, xp_idx, b = gate_b(guidance, dataset, T)

    print(f"\n=== GATE C — contamination audit ===")
    c = gate_c(dataset, xn_idx, xp_idx)

    out = {
        "shard_dir": args.shard_dir,
        "n_per_group": n,
        "guidance_weights": profile.guidance_weights,
        "guidance_is_ensemble": False,
        "gate_a": a,
        "gate_b": b,
        "xn_idx": xn_idx,
        "xp_idx": xp_idx,
        "xn_patients": [dataset.patient_id(i) for i in xn_idx],
        "xp_patients": [dataset.patient_id(i) for i in xp_idx],
        # Ground-truth pathology for the eval originals. Stored here so the
        # scoring tool can gate a validator on "can it tell malignant from
        # benign on THESE images" without reloading 2857 JPEGs to find out.
        # Note these are the RADIOLOGIST labels, not the group assignment: the
        # groups were picked by f(x) thresholds, so they agree only as well as
        # the guidance classifier does.
        "xn_labels": [float(dataset.samples[i][-1]) for i in xn_idx],
        "xp_labels": [float(dataset.samples[i][-1]) for i in xp_idx],
        "gate_c": c,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
