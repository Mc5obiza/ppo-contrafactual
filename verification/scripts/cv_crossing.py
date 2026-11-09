"""
CV at several margins, including the plain boundary-crossing rate (tau = 0), for any
set of cf_shard_*.pt directories. Issue D1: the external baselines (CycleGAN J10,
Latent Shift J15/J15b) cannot be judged on CV at tau = 0.8 alone.

Scoring is the J4 metric code, unchanged: src.metrics.score_batch and
src.metrics.counterfactual_validity, with f = the config's guidance ensemble
(rsna+chex) built by src.xrv_classifier.build_ensemble_classifier. CPU is fine.
Sanity check: tau = 0.8 must reproduce each arm's eval_metrics.json CV (printed).

Run from ~/verification/code/49e6050 with PYTHONPATH=$PWD:
    python cv_crossing.py OUT.json NAME=SHARD_DIR [NAME=SHARD_DIR ...]
"""
from __future__ import annotations

import glob
import json
import os
import sys

import config  # before torchxrayvision (registers its own top-level `config`)
import numpy as np
import torch

from src.metrics import counterfactual_validity, score_batch
from src.xrv_classifier import build_ensemble_classifier

TAUS = (0.0, 0.2, 0.4, 0.6, 0.8)
N_BOOT = 10000


def _load(d: str) -> dict:
    blobs = [torch.load(p, map_location="cpu", weights_only=False)
             for p in sorted(glob.glob(os.path.join(d, "cf_shard_*.pt")))]
    if not blobs:
        raise FileNotFoundError(d)
    out = {k: torch.cat([b[k] for b in blobs]) for k in ("xn", "xp", "xc_np", "xc_pn")}
    out["xn_idx"] = sum((list(b["xn_idx"]) for b in blobs), [])
    out["xp_idx"] = sum((list(b["xp_idx"]) for b in blobs), [])
    return out


def _valid(f_x, f_xc, target, tau) -> np.ndarray:
    # per-image version of counterfactual_validity (checked against it below)
    if target == "positive":
        v = ((f_xc - f_x) >= tau) & (f_x <= 0.5) & (f_xc > 0.5)
    else:
        v = ((f_x - f_xc) >= tau) & (f_x >= 0.5) & (f_xc < 0.5)
    return v.numpy().astype(float)


def _ci(v: np.ndarray) -> list[float]:
    rng = np.random.default_rng(0)
    b = v[rng.integers(0, len(v), (N_BOOT, len(v)))].mean(1)
    return [float(x) for x in np.percentile(b, [2.5, 97.5])]


def main(out_path: str, specs: list[str]) -> None:
    clf = build_ensemble_classifier(device="cpu", weights_list=list(config.GUIDANCE_ENSEMBLE_WEIGHTS))
    res = {"classifier": list(config.GUIDANCE_ENSEMBLE_WEIGHTS), "taus": list(TAUS), "arms": {}}
    for spec in specs:
        name, d = spec.split("=", 1)
        s = _load(d)
        one = lambda t: [t[i:i + 1] for i in range(t.shape[0])]
        f = {k: score_batch(one(s[k]), clf, "cpu") for k in ("xn", "xp", "xc_np", "xc_pn")}
        arm = {"dir": os.path.realpath(d), "n": [len(s["xn_idx"]), len(s["xp_idx"])],
               "idx": [s["xn_idx"], s["xp_idx"]], "per_image_f": {k: v.tolist() for k, v in f.items()}}
        for dname, fx, fxc, tgt in (("n_to_p", f["xn"], f["xc_np"], "positive"),
                                    ("p_to_n", f["xp"], f["xc_pn"], "negative")):
            rows = {}
            for tau in TAUS:
                v = _valid(fx, fxc, tgt, tau)
                ref = counterfactual_validity(fx, fxc, target=tgt, tau=tau)
                assert abs(v.mean() - ref) < 1e-6, (name, dname, tau, v.mean(), ref)
                rows[str(tau)] = {"rate": float(v.mean()), "ci95": _ci(v)}
            sh = (fxc - fx) if tgt == "positive" else (fx - fxc)
            arm[dname] = {"by_tau": rows, "mean_shift": float(sh.mean()),
                          "mean_f_cf": float(fxc.mean())}
        mpath = os.path.join(os.path.dirname(os.path.normpath(d)), "eval_metrics.json")
        if os.path.exists(mpath):
            arm["eval_metrics_CV"] = json.load(open(mpath))["CV"]
        res["arms"][name] = arm
        print(f"[cv-crossing] {name}: n={arm['n']}  eval_metrics CV={arm.get('eval_metrics_CV')}")
        for dname in ("n_to_p", "p_to_n"):
            r = arm[dname]["by_tau"]
            print(f"   {dname}: " + "  ".join(f"tau={t}: {r[str(t)]['rate']:.3f}" for t in TAUS)
                  + f"   mean shift {arm[dname]['mean_shift']:.3f}")
    # index identity check against the first arm (all arms must use J4's images)
    ref = next(iter(res["arms"].values()))["idx"]
    for name, arm in res["arms"].items():
        k = min(len(arm["idx"][0]), len(ref[0]))
        arm["idx_match_first_arm"] = arm["idx"][0][:k] == ref[0][:k] and arm["idx"][1][:k] == ref[1][:k]
        print(f"[cv-crossing] {name}: indices match first arm (prefix {k}): {arm['idx_match_first_arm']}")
    with open(out_path, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"[cv-crossing] wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
