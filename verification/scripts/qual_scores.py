"""
Per-image guidance and held-out validator scores for one evaluation (default: J4), used to
select the qualitative / failure examples of the paper's figure by a stated rule instead
of by eye. CPU. Uses arm_transfer's own classifier builders and scoring loop, so the
per-image values are the ones whose means give transfer.json; the script checks that
(mean deltas and ratios vs transfer.json, tolerance 1e-3) and refuses to write if not.

Run from ~/verification/code/49e6050 with PYTHONPATH=$PWD:$HOME/verification/scripts:
    python qual_scores.py OUT.npz SHARD_DIR TRANSFER_JSON
"""
from __future__ import annotations

import json
import sys

import config  # noqa: F401  (before torchxrayvision)
import numpy as np
import torch

config.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # config hard-codes "cuda"

import arm_transfer as at
from src.xrv_classifier import build_ensemble_classifier, build_xrv_classifier

GUIDANCE = ["densenet121-res224-rsna", "densenet121-res224-chex"]
VALIDATORS = ["densenet121-res224-nih", "densenet121-res224-pc", "densenet121-res224-mimic_ch"]


def main(out_path: str, shard_dir: str, transfer_json: str) -> None:
    torch.set_grad_enabled(False)
    xn, xp, xc_np, xc_pn, stamp = at._load_shards(shard_dir)
    assert at._guidance_members(stamp) == GUIDANCE, at._guidance_members(stamp)
    out = {}
    models = {"guidance": build_ensemble_classifier(weights_list=GUIDANCE, pathology="Pneumonia")}
    for v in VALIDATORS:
        models[v.replace("densenet121-res224-", "")] = build_xrv_classifier(weights=v, pathology="Pneumonia")
    for name, clf in models.items():
        clf.eval()
        for tag, X in (("xn", xn), ("xc_np", xc_np), ("xp", xp), ("xc_pn", xc_pn)):
            out[f"{name}__{tag}"] = at._score(clf, X).numpy()
        print(f"[qual] scored {name}", flush=True)

    # self-check against the reduce's transfer.json
    t = json.load(open(transfer_json))
    arm = list(t["arms"].values())[0]
    g_np = (out["guidance__xc_np"] - out["guidance__xn"]).mean()
    g_pn = (out["guidance__xp"] - out["guidance__xc_pn"]).mean()
    checks = [("guidance n_to_p", g_np, arm["delta_guidance_n_to_p"]),
              ("guidance p_to_n", g_pn, arm["delta_guidance_p_to_n"])]
    for v in VALIDATORS:
        s = v.replace("densenet121-res224-", "")
        r_np = (out[f"{s}__xc_np"] - out[f"{s}__xn"]).mean() / g_np
        r_pn = (out[f"{s}__xp"] - out[f"{s}__xc_pn"]).mean() / g_pn
        checks += [(f"{s} ratio n_to_p", r_np, arm["validators"][v]["ratio_n_to_p"]),
                   (f"{s} ratio p_to_n", r_pn, arm["validators"][v]["ratio_p_to_n"])]
    bad = False
    for name, mine, ref in checks:
        ok = abs(mine - ref) < 1e-3
        bad |= not ok
        print(f"[qual] check {name:22s} mine {mine:.4f} ref {ref:.4f} {'OK' if ok else 'MISMATCH'}")
    if bad:
        raise SystemExit("per-image scores do not reproduce transfer.json; not writing")
    import glob, os
    blobs = [torch.load(p, map_location="cpu", weights_only=False)
             for p in sorted(glob.glob(os.path.join(shard_dir, "cf_shard_*.pt")),
                             key=lambda s: int(s.rsplit("_", 1)[1].split(".")[0]))]
    out["xn_idx"] = np.array(sum((list(b["xn_idx"]) for b in blobs), []))
    out["xp_idx"] = np.array(sum((list(b["xp_idx"]) for b in blobs), []))
    assert len(out["xn_idx"]) == len(xn) and len(out["xp_idx"]) == len(xp)
    np.savez(out_path, **out)
    print(f"[qual] wrote {out_path}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
