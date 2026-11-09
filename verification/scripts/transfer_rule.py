"""
Transfer ratio with the definedness rule, for shards whose guidance model is
not fixed by construction (the CycleGAN baseline).

Rule: a ratio rho = mean(Δ_validator) / mean(Δ_denominator) is reported only if
the 95% percentile-bootstrap CI of mean(Δ_denominator) excludes 0. Otherwise
it is "undefined" and the raw validator shifts are reported with their CIs.

Everything that touches a model is arm_transfer.py's own code (shard loader,
classifier builders, per-image deltas, ratio bootstrap), so the deltas are the
ones arm_transfer prints. Added here: bootstrap CIs on each MEAN shift (10,000
resamples of image indices, seed 0, percentile), and the rule.

Run from ~/verification/code/HEAD with PPO_DOMAIN=rsna_pneumonia:
    python transfer_rule.py OUT.json SHARD_DIR [SHARD_DIR ...]
Denominators: rsna+chex ensemble (primary) and rsna alone (sensitivity).
"""
import json
import sys

import numpy as np

import arm_transfer as at

DENOMINATORS = {
    "ens2_rsna_chex": ["densenet121-res224-rsna", "densenet121-res224-chex"],
    "rsna": ["densenet121-res224-rsna"],
}


def _mean_ci(v: np.ndarray, n_boot: int = 10000, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    boot = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(axis=1)
    return [float(x) for x in np.percentile(boot, [2.5, 97.5])]


def main(out_path: str, dirs: list[str]) -> None:
    import domains
    single, ens = at._builders(domains.active().pathology)
    groups = at._validator_groups()
    cache = {}

    def clf(members):
        key = tuple(members)
        if key not in cache:
            real = at._expand(list(members), groups)
            cache[key] = ens(real) if len(real) > 1 else single(real[0])
        return cache[key]

    vals = at._validators(None)
    record = {"rule": "ratio reported only if the 95% CI of the denominator's mean shift excludes 0",
              "validators": vals, "arms": {}}
    for d in dirs:
        xn, xp, xc_np, xc_pn, _ = at._load_shards(d)
        deltas = {name: dict(zip(("n_to_p", "p_to_n"), at._per_image_deltas(clf(m), xn, xp, xc_np, xc_pn)))
                  for name, m in {**DENOMINATORS, **{v: [v] for v in vals}}.items()}
        arm = {"n_per_group": [int(len(xn)), int(len(xp))], "shifts": {}, "ratios": {}}
        for name, dd in deltas.items():
            arm["shifts"][name] = {k: {"mean": float(a.mean()), "ci95": _mean_ci(a)} for k, a in dd.items()}
        for den in DENOMINATORS:
            arm["ratios"][den] = {}
            for direction in ("n_to_p", "p_to_n"):
                g = deltas[den][direction]
                lo, hi = arm["shifts"][den][direction]["ci95"]
                defined = lo > 0 or hi < 0
                entry = {"denominator_mean": float(g.mean()), "denominator_ci95": [lo, hi],
                         "defined": defined}
                if defined:
                    per_v = {}
                    for v in vals:
                        r, rlo, rhi = at._bootstrap_ratio_ci(g, deltas[v][direction])
                        per_v[v] = {"ratio": r, "ci95": [rlo, rhi]}
                    entry["per_validator"] = per_v
                    entry["mean_over_validators"] = float(np.mean([x["ratio"] for x in per_v.values()]))
                else:
                    entry["ratio"] = "undefined"
                    entry["raw_validator_shifts"] = {v: arm["shifts"][v][direction] for v in vals}
                arm["ratios"][den][direction] = entry
        record["arms"][d] = arm
        print(f"\n=== {d}  (Xn={len(xn)}, Xp={len(xp)})")
        for name, s in arm["shifts"].items():
            print(f"  Δ {name:28s} " + "  ".join(
                f"{k} {v['mean']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]" for k, v in s.items()))
        for den, by_dir in arm["ratios"].items():
            for direction, e in by_dir.items():
                if e["defined"]:
                    cells = "  ".join(f"{v.replace('densenet121-res224-', '')} {x['ratio']:.3f} "
                                      f"[{x['ci95'][0]:.3f}, {x['ci95'][1]:.3f}]"
                                      for v, x in e["per_validator"].items())
                    print(f"  ρ/{den:15s} {direction}: {cells}  mean {e['mean_over_validators']:.3f}")
                else:
                    print(f"  ρ/{den:15s} {direction}: UNDEFINED (denominator CI "
                          f"[{e['denominator_ci95'][0]:+.4f}, {e['denominator_ci95'][1]:+.4f}] contains 0)")
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
