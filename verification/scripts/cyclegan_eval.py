"""
Vanilla CycleGAN baseline scored under the committed evaluation protocol.

Run from ~/verification/code/49e6050 (the committed run's eval code), see
12_cyclegan_eval.sbatch. Writes only under --out.

What is reused, unchanged
  * The 400 source images: Xn/Xp tensors AND index lists are read from the
    committed run's shards (checkpoints/eval_shards_ens_default), not rescanned.
    Each tensor is checked against ctx.dataset[idx] and each index against the
    test fold, so the inputs are provably the committed run's.
  * Scoring: src.metrics.run_full_evaluation, PRECOMPUTED path (the one
    eval_main.py --stage reduce uses), same kwargs as stage_reduce. That path
    never touches the generator or the policies, so it takes any
    image->counterfactual map. CV is therefore scored by ctx.classifier, the
    rsna+chex ensemble that also selected the groups.
  * Output shards follow the committed cf_shard_*.pt schema, so
    arm_transfer.py, paired_arm_test.py and per_image_lhr.py read them as-is.

What is CycleGAN-specific
  * xc_np = G_AtoB(Xn), xc_pn = G_BtoA(Xp)  (A = label-normal, B = label-
    pneumonia, train_cyclegan_baseline._label_partition). One deterministic
    forward pass per image, so there are no seeds.
  * The shard stamp does NOT copy the committed stamp: the diffusion keys
    (BUDGET_T, alphas, lock, ...) do not describe this generator. It carries
    the grouping keys plus GENERATOR=cyclegan and USE_GUIDANCE_ENSEMBLE=None.
    Consequences: eval_main.py --stage reduce refuses these shards (intended:
    its _plot_diffs would draw diffusion edits under the CycleGAN label), and
    arm_transfer.py MUST be given --guidance explicitly (without it, it warns
    and falls back to the single rsna model).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import json
import os
import re

import numpy as np
import torch

import config
from pipeline import build_common
from src.metrics import current_cfg_stamp, run_full_evaluation, score_batch

GROUPING_KEYS = ("EVAL_GROUP_SOURCE", "USE_SPLIT", "SPLIT_SEED", "TEST_FRAC",
                 "USE_GUIDANCE_ENSEMBLE", "GUIDANCE_ENSEMBLE_WEIGHTS")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_committed(shard_dir: str):
    paths = sorted(glob.glob(os.path.join(shard_dir, "cf_shard_*.pt")))
    blobs = {int(re.search(r"cf_shard_(\d+)\.pt$", p).group(1)):
             torch.load(p, map_location="cpu", weights_only=False) for p in paths}
    if not blobs:
        raise FileNotFoundError(f"no shards in {shard_dir}")
    total = blobs[min(blobs)]["num_shards"]
    if set(blobs) != set(range(total)):
        raise FileNotFoundError(f"{shard_dir}: have {sorted(blobs)}, expected 0..{total - 1}")
    order = sorted(blobs)
    stamp = blobs[0]["config"]
    if any(blobs[k]["config"] != stamp or blobs[k]["n_eval"] != blobs[0]["n_eval"] for k in order):
        raise ValueError(f"{shard_dir}: shards disagree on config stamp or n_eval")
    cat = lambda k: torch.cat([blobs[i][k] for i in order])
    idx = lambda k: [int(i) for s in order for i in blobs[s][k]]
    return cat("xn"), cat("xp"), idx("xn_idx"), idx("xp_idx"), stamp, blobs[0]["n_eval"]


def _load_cyclegan(src_path: str, ckpt_dir: str, device: str):
    spec = importlib.util.spec_from_file_location("cyclegan_baseline", src_path)
    cg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cg)
    gens = {}
    for name in ("G_AtoB", "G_BtoA"):
        g = cg.ResnetGenerator().to(device)   # defaults = the trained hparams (ngf 64, 9 blocks)
        g.load_state_dict(torch.load(os.path.join(ckpt_dir, f"{name}.pt"),
                                     map_location=device, weights_only=True))  # strict
        gens[name] = g.eval()
    return cg, gens


def _translate(cg, g, x: torch.Tensor, device: str, bs: int) -> list[torch.Tensor]:
    out = []
    for i in range(0, x.shape[0], bs):
        xc = cg.translate(g, x[i:i + bs].to(device)).float().cpu()
        out.extend(xc[j:j + 1] for j in range(xc.shape[0]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--committed-shards", required=True)
    ap.add_argument("--cyclegan-ckpt", required=True, help="dir with G_AtoB.pt / G_BtoA.pt")
    ap.add_argument("--cyclegan-src", required=True, help="path to src/cyclegan.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()
    device = config.DEVICE
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    os.makedirs(os.path.join(args.out, "shards"), exist_ok=True)
    manifest: dict = {"code": os.getcwd(), "committed_shards": args.committed_shards}

    xn, xp, xn_idx, xp_idx, stamp, n_eval = _load_committed(args.committed_shards)
    print(f"[cyclegan] committed groups: Xn={len(xn_idx)} Xp={len(xp_idx)} n_eval={n_eval}")
    manifest["n"] = {"xn": len(xn_idx), "xp": len(xp_idx), "n_eval": n_eval}
    manifest["idx_sha256"] = hashlib.sha256(json.dumps([xn_idx, xp_idx]).encode()).hexdigest()

    # The live config must reproduce the committed grouping/CV classifier.
    live = current_cfg_stamp()
    drift = {k: (stamp.get(k), live.get(k)) for k in GROUPING_KEYS if stamp.get(k) != live.get(k)}
    if drift:
        raise ValueError(f"grouping config differs from the committed shards: {drift}")

    ctx = build_common()
    test = set(int(i) for i in ctx.eval_indices())
    outside = [i for i in xn_idx + xp_idx if i not in test]
    if outside:
        raise ValueError(f"{len(outside)} committed eval indices are not in the test fold")

    # Inputs: committed tensors must equal what the dataset yields today.
    diffs = [float((ctx.dataset[i][0].reshape(x.shape) - x).abs().max())
             for x, i in zip(list(xn) + list(xp), xn_idx + xp_idx)]
    manifest["input_max_abs_diff_vs_dataset"] = max(diffs)
    print(f"[cyclegan] committed tensors vs dataset[idx]: max |diff| = {max(diffs):.3g}")
    if max(diffs) > 1e-5:
        raise ValueError("committed shard tensors do not match the dataset; preprocessing drifted")

    # Group membership under today's ensemble (informational; GPU nondeterminism
    # can move a score sitting on a threshold).
    xn_list = [xn[i:i + 1] for i in range(xn.shape[0])]
    xp_list = [xp[i:i + 1] for i in range(xp.shape[0])]
    f_n, f_p = score_batch(xn_list, ctx.classifier, device), score_batch(xp_list, ctx.classifier, device)
    manifest["group_check"] = {"xn_not_below_0.2": int((f_n >= 0.2).sum()),
                               "xp_not_above_0.8": int((f_p <= 0.8).sum())}
    print(f"[cyclegan] group membership under live ensemble: {manifest['group_check']}")

    cg, gens = _load_cyclegan(args.cyclegan_src, args.cyclegan_ckpt, device)
    manifest["cyclegan"] = {
        "src": os.path.realpath(args.cyclegan_src), "src_sha256": _sha256(args.cyclegan_src),
        **{f"{k}_sha256": _sha256(os.path.join(args.cyclegan_ckpt, f"{k}.pt")) for k in gens},
    }
    tm = os.path.join(args.cyclegan_ckpt, "cyclegan_train_manifest.json")
    if os.path.exists(tm):
        manifest["cyclegan"]["train_manifest"] = json.load(open(tm))
    print(f"[cyclegan] {manifest['cyclegan']}")

    xc_np = _translate(cg, gens["G_AtoB"], xn, device, args.batch_size)   # n -> p
    xc_pn = _translate(cg, gens["G_BtoA"], xp, device, args.batch_size)   # p -> n

    # The transfer denominator for this arm (grouping-ensemble shift), with the
    # sign convention of arm_transfer._per_image_deltas. If its CI reaches 0 the
    # ratio is undefined and must be reported as such, not as a number.
    deltas = {"n_to_p": (score_batch(xc_np, ctx.classifier, device) - f_n).numpy(),
              "p_to_n": (f_p - score_batch(xc_pn, ctx.classifier, device)).numpy()}
    rng = np.random.default_rng(0)
    manifest["delta_grouping_ensemble"] = {}
    for d, v in deltas.items():
        boot = v[rng.integers(0, len(v), (10000, len(v)))].mean(axis=1)
        manifest["delta_grouping_ensemble"][d] = {
            "mean": float(v.mean()), "ci95": [float(x) for x in np.percentile(boot, [2.5, 97.5])],
            "frac_positive": float((v > 0).mean())}
    print(f"[cyclegan] grouping-ensemble shift (transfer denominator): "
          f"{manifest['delta_grouping_ensemble']}")

    shard_stamp = {k: stamp[k] for k in ("EVAL_GROUP_SOURCE", "USE_SPLIT", "SPLIT_SEED", "TEST_FRAC")}
    shard_stamp.update(GENERATOR="cyclegan", USE_GUIDANCE_ENSEMBLE=None,
                       GUIDANCE_ENSEMBLE_WEIGHTS=[],
                       GROUPING_CLASSIFIER=list(stamp["GUIDANCE_ENSEMBLE_WEIGHTS"]),
                       GROUPS_FROM=os.path.realpath(args.committed_shards))
    shard_path = os.path.join(args.out, "shards", "cf_shard_000.pt")
    torch.save({
        "shard": 0, "num_shards": 1, "n_eval": n_eval, "base_seed": 0,
        "n_slice": [0, len(xn_idx)], "p_slice": [0, len(xp_idx)],
        "xn": xn, "xp": xp, "xc_np": torch.cat(xc_np), "xc_pn": torch.cat(xc_pn),
        "xn_idx": xn_idx, "xp_idx": xp_idx, "config": shard_stamp,
    }, shard_path)
    print(f"[cyclegan] wrote {shard_path}")

    run_full_evaluation(
        dataset=ctx.dataset, classifier=ctx.classifier, generator=None,
        seg_net=ctx.seg_net, lung_channel_idx=ctx.lung_channel_idx, obj_det=ctx.obj_det,
        n_eval=n_eval, ckpt_dir=args.out, device=device,
        xn_list=xn_list, xp_list=xp_list, xc_np=xc_np, xc_pn=xc_pn,
        xn_idx=xn_idx, xp_idx=xp_idx, clinical_metrics=ctx.domain.clinical_metrics,
    )
    with open(os.path.join(args.out, "cyclegan_eval_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[cyclegan] wrote {args.out}/eval_metrics.json and cyclegan_eval_manifest.json")


if __name__ == "__main__":
    main()
