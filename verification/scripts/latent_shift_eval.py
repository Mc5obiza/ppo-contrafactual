"""
Latent Shift baseline (Gifsplanation; Cohen et al., MIDL 2021) scored under the
J4 protocol. Issue D1 of PPO-CF_CVPR2027_issues_to_fix.pdf.

Run from ~/verification/code/49e6050 (the J4 eval code), see 25_latent_shift.sbatch.
Writes only under --out.

What is reused, unchanged
  * The 400 source images: Xn/Xp tensors AND index lists are read from J4's
    shards (out/J4_perdir/shards; same indices as E_orig), not rescanned, with
    cyclegan_eval's checks: grouping config vs live config, indices in the test
    fold, tensors equal to dataset[idx].
  * Scoring: src.metrics.run_full_evaluation, PRECOMPUTED path (generator=None),
    as in cyclegan_eval. CV is scored by ctx.classifier, the rsna+chex ensemble
    that selected the groups AND guides Latent Shift here.
  * Output shard follows the cf_shard_*.pt schema, so arm_transfer.py,
    paired_arm_test.py and per_image_lhr.py read it as-is.

What is Latent-Shift-specific
  * Autoencoder: torchxrayvision ResNetAE("101-elastic") (224 px, xrv scale
    [-1024, 1024]; trained on nih, pc, rsna, mimic_ch, chex per its metadata).
    256-px [0,1] images are mapped x*2048-1024 and resized to 224, and decoded
    images are mapped back and resized to 256 (bilinear).
  * One gradient per image, as in Latent Shift: g = d f(D(E(x))) / dz at z = E(x),
    f = the guidance ensemble. Counterfactual x_lam = D(z + s * lam * ||z|| * g/||g||),
    s = +1 for n->p and -1 for p->n.
  * lam is searched on LAMBDA_GRID, fixed before any result was seen. Per image
    the smallest lam meeting the CV criterion of eq. (cv) (s*Delta >= 0.8 and a
    0.5 crossing, scored by f) is kept, mirroring the policy's stop-at-validity
    rule. If no lam meets it, the lam with the largest signed shift is kept and
    the image counts as invalid. Nothing is tuned on LHR, transfer or any
    held-out quantity.
  * Deterministic: the AE is not variational, so there are no seeds.
  * The counterfactual is the full decode D(z'), which carries the AE round-trip
    error everywhere, as PPO-CF's carries the VAE's. D(E(x)) is saved in
    latent_shift_aux.pt for reconstruction-relative metrics (supplement).
  * Shard stamp: grouping keys + GENERATOR=latent_shift, USE_GUIDANCE_ENSEMBLE=None,
    so arm_transfer.py MUST be given --guidance explicitly (as for the CycleGAN).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

# The project's config must be imported BEFORE torchxrayvision, which registers
# its own top-level `config` package (domains/__init__.py enforces this).
import config
from pipeline import build_common
from src.metrics import current_cfg_stamp, run_full_evaluation, score_batch

import cyclegan_eval as cge   # _load_committed, _sha256, GROUPING_KEYS
import torchxrayvision as xrv

# x ||z||. 3.0 and 4.0 were added after the 8-image smoke test (18537169), where 2/8 p->n
# images took the then-largest value 2.0: a guidance-only search diagnostic; no anchored
# metric (LHR, transfer) was used to set the grid.
LAMBDA_GRID = (0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0)
CV_MARGIN = 0.8   # tau in eq. (cv)
AE_SIZE = 224


def _to_ae(x01: torch.Tensor) -> torch.Tensor:
    return F.interpolate(x01 * 2048.0 - 1024.0, size=(AE_SIZE, AE_SIZE),
                         mode="bilinear", align_corners=False)


def _from_ae(y: torch.Tensor, size) -> torch.Tensor:
    y01 = (y.clamp(-1024.0, 1024.0) + 1024.0) / 2048.0
    return F.interpolate(y01, size=size, mode="bilinear", align_corners=False).clamp(0.0, 1.0)


def _norm(t: torch.Tensor) -> torch.Tensor:
    return t.flatten(1).norm(dim=1).view(-1, *([1] * (t.dim() - 1)))


def latent_shift(ae, clf, x: torch.Tensor, s: int, device: str) -> dict:
    """x: (B,1,H,W) in [0,1]. Returns per-image counterfactuals and search record (CPU)."""
    size = tuple(x.shape[-2:])
    x = x.to(device)
    with torch.no_grad():
        f0 = clf(x).squeeze(-1)
    z = ae.encode(_to_ae(x)).detach().requires_grad_(True)
    xr = _from_ae(ae.decode(z), size)
    fr = clf(xr).squeeze(-1)
    # Images are independent and every module is in eval mode, so the gradient
    # of the batch sum is the per-image gradient.
    (g,) = torch.autograd.grad(fr.sum(), z)
    z, g, xr, fr = z.detach(), g.detach(), xr.detach(), fr.detach()
    direction = s * g / _norm(g).clamp_min(1e-12)
    zn = _norm(z)

    B = x.shape[0]
    best_x = xr.clone()
    best_lam = torch.zeros(B, device=device)
    best_gain = torch.full((B,), -float("inf"), device=device)
    found = torch.zeros(B, dtype=torch.bool, device=device)
    traj = []
    with torch.no_grad():
        for lam in LAMBDA_GRID:
            xl = _from_ae(ae.decode(z + lam * zn * direction), size)
            fl = clf(xl).squeeze(-1)
            gain = s * (fl - f0)
            valid = (gain >= CV_MARGIN) & (s * (fl - 0.5) > 0)
            upd = (valid & ~found) | (~found & ~valid & (gain > best_gain))
            best_x[upd] = xl[upd]
            best_lam = torch.where(upd, torch.full_like(best_lam, lam), best_lam)
            best_gain = torch.where(upd, gain, best_gain)
            found |= valid
            traj.append(fl)
    return {"xc": best_x.cpu(), "xr": xr.cpu(), "lam": best_lam.cpu(), "found": found.cpu(),
            "f0": f0.cpu(), "f_rec": fr.cpu(), "traj": torch.stack(traj, 1).cpu()}


def _run_group(ae, clf, x: torch.Tensor, s: int, device: str, bs: int) -> dict:
    parts = [latent_shift(ae, clf, x[i:i + bs], s, device) for i in range(0, x.shape[0], bs)]
    return {k: torch.cat([p[k] for p in parts]) for k in parts[0]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--committed-shards", required=True, help="J4 shard dir (source images + indices)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N images per group")
    args = ap.parse_args()
    device = config.DEVICE
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    os.makedirs(os.path.join(args.out, "shards"), exist_ok=True)
    manifest: dict = {"code": os.getcwd(), "committed_shards": args.committed_shards,
                      "lambda_grid": list(LAMBDA_GRID), "cv_margin": CV_MARGIN}

    xn, xp, xn_idx, xp_idx, stamp, n_eval = cge._load_committed(args.committed_shards)
    if args.limit:
        xn, xp, xn_idx, xp_idx = xn[:args.limit], xp[:args.limit], xn_idx[:args.limit], xp_idx[:args.limit]
        n_eval = args.limit
        manifest["SMOKE_TEST_LIMIT"] = args.limit
    print(f"[latent-shift] groups from {args.committed_shards}: Xn={len(xn_idx)} Xp={len(xp_idx)} n_eval={n_eval}")
    manifest["n"] = {"xn": len(xn_idx), "xp": len(xp_idx), "n_eval": n_eval}
    manifest["idx_sha256"] = hashlib.sha256(json.dumps([xn_idx, xp_idx]).encode()).hexdigest()

    live = current_cfg_stamp()
    drift = {k: (stamp.get(k), live.get(k)) for k in cge.GROUPING_KEYS if stamp.get(k) != live.get(k)}
    if drift:
        raise ValueError(f"grouping config differs from the source shards: {drift}")

    ctx = build_common()
    test = set(int(i) for i in ctx.eval_indices())
    outside = [i for i in xn_idx + xp_idx if i not in test]
    if outside:
        raise ValueError(f"{len(outside)} source indices are not in the test fold")
    diffs = [float((ctx.dataset[i][0].reshape(x.shape) - x).abs().max())
             for x, i in zip(list(xn) + list(xp), xn_idx + xp_idx)]
    manifest["input_max_abs_diff_vs_dataset"] = max(diffs)
    print(f"[latent-shift] source tensors vs dataset[idx]: max |diff| = {max(diffs):.3g}")
    if max(diffs) > 1e-5:
        raise ValueError("source shard tensors do not match the dataset; preprocessing drifted")

    xn_list = [xn[i:i + 1] for i in range(xn.shape[0])]
    xp_list = [xp[i:i + 1] for i in range(xp.shape[0])]
    f_n, f_p = score_batch(xn_list, ctx.classifier, device), score_batch(xp_list, ctx.classifier, device)
    manifest["group_check"] = {"xn_not_below_0.2": int((f_n >= 0.2).sum()),
                               "xp_not_above_0.8": int((f_p <= 0.8).sum())}
    print(f"[latent-shift] group membership under live ensemble: {manifest['group_check']}")

    ae = xrv.autoencoders.ResNetAE(weights="101-elastic").to(device).eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    ae_file = os.path.expanduser(os.path.join(
        "~/.torchxrayvision/models_data", os.path.basename(ae.weights_metadata["weights_url"])))
    manifest["autoencoder"] = {"weights": "101-elastic", "file": ae_file, "sha256": cge._sha256(ae_file),
                               "metadata": ae.weights_metadata}
    print(f"[latent-shift] autoencoder {manifest['autoencoder']}")

    res = {"n_to_p": _run_group(ae, ctx.classifier, xn, +1, device, args.batch_size),
           "p_to_n": _run_group(ae, ctx.classifier, xp, -1, device, args.batch_size)}
    xc_np = [res["n_to_p"]["xc"][i:i + 1] for i in range(len(xn_idx))]
    xc_pn = [res["p_to_n"]["xc"][i:i + 1] for i in range(len(xp_idx))]
    manifest["search"] = {}
    for d, r in res.items():
        lam = r["lam"].numpy()
        manifest["search"][d] = {
            "valid_by_search": int(r["found"].sum()), "n": int(len(lam)),
            "lam_median": float(np.median(lam)), "lam_hist": {str(l): int((lam == l).sum()) for l in LAMBDA_GRID},
            "f_source_mean": float(r["f0"].mean()), "f_recon_mean": float(r["f_rec"].mean()),
        }
    print(f"[latent-shift] search: {json.dumps(manifest['search'])}")

    deltas = {"n_to_p": (score_batch(xc_np, ctx.classifier, device) - f_n).numpy(),
              "p_to_n": (f_p - score_batch(xc_pn, ctx.classifier, device)).numpy()}
    rng = np.random.default_rng(0)
    manifest["delta_grouping_ensemble"] = {}
    for d, v in deltas.items():
        boot = v[rng.integers(0, len(v), (10000, len(v)))].mean(axis=1)
        manifest["delta_grouping_ensemble"][d] = {
            "mean": float(v.mean()), "ci95": [float(x) for x in np.percentile(boot, [2.5, 97.5])],
            "frac_positive": float((v > 0).mean())}
    print(f"[latent-shift] guidance-ensemble shift (transfer denominator): "
          f"{manifest['delta_grouping_ensemble']}")

    shard_stamp = {k: stamp[k] for k in ("EVAL_GROUP_SOURCE", "USE_SPLIT", "SPLIT_SEED", "TEST_FRAC")}
    shard_stamp.update(GENERATOR="latent_shift", USE_GUIDANCE_ENSEMBLE=None,
                       GUIDANCE_ENSEMBLE_WEIGHTS=[],
                       GROUPING_CLASSIFIER=list(stamp["GUIDANCE_ENSEMBLE_WEIGHTS"]),
                       LATENT_SHIFT_CLASSIFIER=list(stamp["GUIDANCE_ENSEMBLE_WEIGHTS"]),
                       LAMBDA_GRID=list(LAMBDA_GRID),
                       GROUPS_FROM=os.path.realpath(args.committed_shards))
    shard_path = os.path.join(args.out, "shards", "cf_shard_000.pt")
    torch.save({
        "shard": 0, "num_shards": 1, "n_eval": n_eval, "base_seed": 0,
        "n_slice": [0, len(xn_idx)], "p_slice": [0, len(xp_idx)],
        "xn": xn, "xp": xp, "xc_np": torch.cat(xc_np), "xc_pn": torch.cat(xc_pn),
        "xn_idx": xn_idx, "xp_idx": xp_idx, "config": shard_stamp,
    }, shard_path)
    torch.save({d: {k: v for k, v in r.items() if k != "xc"} for d, r in res.items()}
               | {"xn_idx": xn_idx, "xp_idx": xp_idx, "lambda_grid": list(LAMBDA_GRID)},
               os.path.join(args.out, "latent_shift_aux.pt"))
    print(f"[latent-shift] wrote {shard_path} and latent_shift_aux.pt")

    run_full_evaluation(
        dataset=ctx.dataset, classifier=ctx.classifier, generator=None,
        seg_net=ctx.seg_net, lung_channel_idx=ctx.lung_channel_idx, obj_det=ctx.obj_det,
        n_eval=n_eval, ckpt_dir=args.out, device=device,
        xn_list=xn_list, xp_list=xp_list, xc_np=xc_np, xc_pn=xc_pn,
        xn_idx=xn_idx, xp_idx=xp_idx, clinical_metrics=ctx.domain.clinical_metrics,
    )
    with open(os.path.join(args.out, "latent_shift_eval_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[latent-shift] wrote {args.out}/eval_metrics.json and latent_shift_eval_manifest.json")


if __name__ == "__main__":
    main()
