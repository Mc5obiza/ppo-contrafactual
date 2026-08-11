"""
arm_transfer.py
===============
Held-out transfer ratio for an eval ARM, computed straight from its shards.

WHY NOT heldout_validation.py
-----------------------------
That script regenerates counterfactuals with the TRAINED POLICIES
(generate_batch_with_policy), so its numbers describe the trained arm and it
cannot measure a `--arm none` run at all. The Phase 3a countermeasure arms are
no-policy runs by design — Phase 2 showed the policy is inert — and their
counterfactuals already exist in cf_shard_*.pt. Scoring those directly is both
correct and ~3h cheaper per arm than regenerating.

THE DENOMINATOR IS PER-ARM, AND THAT MATTERS
--------------------------------------------
transfer = Δ_validator / Δ_guidance, and "guidance" is not the same model in
every arm: ARM 1 and ARM 3 guide on the ensemble, ARM 0 and ARM 2 on the single
classifier. Using one fixed denominator across arms is exactly the bug that
inflated every previously published ratio (see _config_guidance_spec in
heldout_validation.py). Each shard carries current_cfg_stamp(), which now records
USE_GUIDANCE_ENSEMBLE, so the right denominator is read from the data rather than
assumed.

VALIDATORS ARE THE DOMAIN'S heldout_validators — {nih, pc, mimic_ch} for RSNA.
That set is disjoint from BOTH the single guidance model and the ensemble
(domains/base.py enforces it), so it is the only set comparable ACROSS arms.
chex is a legitimate held-out validator when guidance is single-classifier, but
including it would let the ensemble arms "improve" purely by dropping a
validator, so it is excluded here and reported separately if wanted.

    module load python/3.12.8 && source venv/bin/activate
    python arm_transfer.py checkpoints/eval_shards_none_cm0 [more dirs...]
"""

from __future__ import annotations

import glob
import os
import re
import sys

import torch

import config


def _load_shards(shard_dir: str):
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
    cat = lambda k: torch.cat([blobs[i][k] for i in order])
    return (cat("xn"), cat("xp"), cat("xc_np"), cat("xc_pn"),
            blobs[order[0]].get("config", {}))


def _guidance_members(stamp: dict) -> list[str]:
    """What actually guided this arm, from its own stamp."""
    use_ens = stamp.get("USE_GUIDANCE_ENSEMBLE")
    try:
        import domains
        d = domains.active()
        if use_ens is None:                     # pre-stamp shards
            print("  [warn] shards predate the USE_GUIDANCE_ENSEMBLE stamp; "
                  "assuming the single guidance model")
            use_ens = False
        return list(d.ensemble_weights) if use_ens else [d.guidance_weights]
    except Exception:
        return list(stamp.get("GUIDANCE_ENSEMBLE_WEIGHTS", []))[: 2 if use_ens else 1]


def _validators() -> list[str]:
    try:
        import domains
        return list(domains.active().heldout_validators)
    except Exception:
        return ["densenet121-res224-" + s for s in ("nih", "pc", "mimic_ch")]


def _score(clf, X, bs: int = 32) -> torch.Tensor:
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            x = X[i : i + bs].to(config.DEVICE)
            x = x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x
            out.append(clf(x).squeeze(-1).float().cpu())
    return torch.cat(out)


def _deltas(clf, xn, xp, xc_np, xc_pn) -> tuple[float, float]:
    """Mean movement TOWARD the target, per direction."""
    d_np = (_score(clf, xc_np) - _score(clf, xn)).mean().item()
    d_pn = (_score(clf, xp) - _score(clf, xc_pn)).mean().item()
    return d_np, d_pn


def _verdict(r: float) -> str:
    return "ADVERSARIAL" if r < 0.15 else ("partial" if r < 0.5 else "TRANSFERS")


def main(dirs: list[str]) -> int:
    from src.xrv_classifier import build_xrv_classifier, build_ensemble_classifier

    try:
        import domains
        pathology = domains.active().pathology
    except Exception:
        pathology = "Pneumonia"

    cache: dict[tuple, object] = {}

    def clf_for(members: list[str]):
        key = tuple(members)
        if key not in cache:
            cache[key] = (
                build_ensemble_classifier(weights_list=list(members), pathology=pathology)
                if len(members) > 1
                else build_xrv_classifier(weights=members[0], pathology=pathology)
            )
        return cache[key]

    vals = _validators()
    summary = {}

    for d in dirs:
        name = os.path.basename(d.rstrip("/"))
        print(f"\n{'=' * 74}\n{name}\n{'=' * 74}")
        xn, xp, xc_np, xc_pn, stamp = _load_shards(d)
        gm = _guidance_members(stamp)
        short = lambda w: w.replace("densenet121-res224-", "")
        print(f"  n={len(xn)} per group | guidance = {'+'.join(map(short, gm))} "
              f"| groups = {stamp.get('EVAL_GROUP_SOURCE', '?')} "
              f"| gradcam = {stamp.get('USE_GRADCAM_MASK', '?')}")

        g_np, g_pn = _deltas(clf_for(gm), xn, xp, xc_np, xc_pn)
        print(f"  Δ_guidance  n→p={g_np:.4f}  p→n={g_pn:.4f}")
        print(f"  {'validator':12s} {'Δ n→p':>8s} {'Δ p→n':>8s} "
              f"{'ratio n→p':>10s} {'ratio p→n':>10s}   verdict")

        ratios = []
        for v in vals:
            d_np, d_pn = _deltas(clf_for([v]), xn, xp, xc_np, xc_pn)
            r_np = d_np / g_np if g_np else float("nan")
            r_pn = d_pn / g_pn if g_pn else float("nan")
            ratios.append((r_np, r_pn))
            print(f"  {short(v):12s} {d_np:8.4f} {d_pn:8.4f} "
                  f"{r_np:10.3f} {r_pn:10.3f}   {_verdict(r_np)}/{_verdict(r_pn)}")

        m_np = sum(r for r, _ in ratios) / len(ratios)
        m_pn = sum(r for _, r in ratios) / len(ratios)
        summary[name] = (m_np, m_pn)
        print(f"  {'MEAN':12s} {'':>17s} {m_np:10.3f} {m_pn:10.3f}")

    if len(summary) > 1:
        print(f"\n{'=' * 74}\nTRANSFER SUMMARY  (higher = more real signal, "
              f"less adversarial)\n{'=' * 74}")
        print(f"  {'arm':28s} {'mean n→p':>10s} {'mean p→n':>10s}")
        base = next(iter(summary.values()))
        for k, (a, b) in summary.items():
            print(f"  {k:28s} {a:10.3f} {b:10.3f}"
                  f"   ({a - base[0]:+.3f} / {b - base[1]:+.3f} vs first)")
        print("\n  Gate: an arm only closes the adversarial gap if these rise "
              "materially.\n  CV is NOT the gate — it is circular by construction.")
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(args))
