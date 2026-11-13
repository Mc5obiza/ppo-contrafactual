"""
Checks + side-by-side report for J10 (12_cyclegan_eval.sbatch). CPU only, reads JSON/text.

    python cyclegan_report.py ~/verification/out/J10_cyclegan

LHR CI: 95% percentile bootstrap of the mean per-image lift (10,000 resamples
of annotated images, seed 0), from per_image_lhr.json.
"""
import json
import os
import re
import sys

import numpy as np

R = os.environ['PPO_CF_REPO']
COMMITTED_SHARDS = f"{R}/checkpoints/eval_shards_ens_default"
COMMITTED_METRICS = f"{R}/checkpoints_ens_default/eval_metrics.json"


def ci(v, seed=0, n=10000):
    v = np.asarray(v)
    b = v[np.random.default_rng(seed).integers(0, len(v), (n, len(v)))].mean(axis=1)
    return np.percentile(b, [2.5, 97.5])


def main(out):
    cg = json.load(open(f"{out}/eval_metrics.json"))
    cm = json.load(open(COMMITTED_METRICS))
    man = json.load(open(f"{out}/cyclegan_eval_manifest.json"))
    lhr = json.load(open(f"{out}/per_image_lhr.json"))
    paired = open(f"{out}/paired_lhr_vs_committed.txt").read()
    rule = json.load(open(f"{out}/transfer_rule.json"))
    per = {"CycleGAN": lhr[f"{out}/shards"], "committed": lhr[COMMITTED_SHARDS]}

    m = re.search(r"paired images\s*:\s*(\d+)", paired)
    n_shared = int(m.group(1)) if m else None
    gc = man["group_check"]
    checks = [
        ("input max |diff| vs dataset == 0", man["input_max_abs_diff_vs_dataset"] == 0.0,
         man["input_max_abs_diff_vs_dataset"]),
        ("group check ~0 (each <= 2/200)", max(gc.values()) <= 2, gc),
        ("LHR n_pairs == 148", cg["LHR"]["p_to_n"]["n_pairs"] == 148, cg["LHR"]["p_to_n"]["n_pairs"]),
        ("paired test shared images == 148", n_shared == 148, n_shared),
        ("groups 200/200", man["n"]["xn"] == man["n"]["xp"] == 200, man["n"]),
        ("per-image lift mean == eval_metrics lift (CycleGAN)",
         abs(np.mean(list(per["CycleGAN"].values())) - cg["LHR"]["p_to_n"]["lift"]) < 1e-6,
         (np.mean(list(per["CycleGAN"].values())), cg["LHR"]["p_to_n"]["lift"])),
        ("per-image lift mean == eval_metrics lift (committed)",
         abs(np.mean(list(per["committed"].values())) - cm["LHR"]["p_to_n"]["lift"]) < 1e-6,
         (np.mean(list(per["committed"].values())), cm["LHR"]["p_to_n"]["lift"])),
    ]
    print("CHECKS")
    for name, ok, val in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {val}")

    print("\nSIDE BY SIDE (same 400 test-fold images)")
    row = lambda name, f: print(f"  {name:34s} " + "  ".join(
        f"{f(r, d):>9s}" for d in ("n_to_p", "p_to_n") for r in (cm, cg)))
    print(f"  {'':34s} " + "  ".join(f"{h:>9s}" for h in ("n→p comm", "n→p CG", "p→n comm", "p→n CG")))
    f3 = lambda x: f"{x:.3f}"
    row("CV (τ=0.8)", lambda r, d: f3(r["CV"][d]))
    row("EditConc top-5% (uniform 0.05)", lambda r, d: f3(r["EditConcentration"][d]["concentration"]))
    row("fraction of pixels changed", lambda r, d: f3(r["EditConcentration"][d]["frac_changed"]))
    fk = {"n_to_p": "n_to_p_vs_real_positive", "p_to_n": "p_to_n_vs_real_negative"}
    row("FID vs real target class", lambda r, d: f"{r['FID'][fk[d]]:.2f}")
    row("IP MS-SSIM", lambda r, d: f3(r["IP"][d]["ms_ssim"]))
    row("IP HF-corr", lambda r, d: f3(r["IP"][d]["hf_corr"]))
    print(f"  CV mean: committed {cm['CV']['mean']:.3f}  CycleGAN {cg['CV']['mean']:.3f}")
    print("\n  LHR lift p→n (n→p undefined)")
    for name, r in (("committed", cm), ("CycleGAN", cg)):
        L = r["LHR"]["p_to_n"]; lo, hi = ci(list(per[name].values()))
        print(f"    {name:10s} {L['lift']:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  n={L['n_pairs']}  "
              f"(t={L['t']:.2f}, p={L['p']:.3g} hit vs null)")
    print("\n  paired LHR (committed − CycleGAN):")
    for line in paired.splitlines():
        if re.match(r"(paired|mean lift|Cohen|.* better on|Wilcoxon)", line.strip()):
            print("    " + line.strip())

    print("\nTRANSFER (rule: ratio only if denominator-shift 95% CI excludes 0)")
    for arm_dir, arm in rule["arms"].items():
        label = "CycleGAN" if arm_dir.endswith("/shards") else "committed"
        print(f"  {label}")
        for den, by_dir in arm["ratios"].items():
            for d, e in by_dir.items():
                lo, hi = e["denominator_ci95"]
                head = f"    /{den:15s} {d}: Δden {e['denominator_mean']:+.4f} [{lo:+.4f}, {hi:+.4f}] → "
                if e["defined"]:
                    print(head + "  ".join(f"{v.split('-')[-1] if 'mimic' not in v else 'mimic_ch'} "
                                           f"{x['ratio']:.3f} [{x['ci95'][0]:.3f}, {x['ci95'][1]:.3f}]"
                                           for v, x in e["per_validator"].items())
                          + f"  mean {e['mean_over_validators']:.3f}")
                else:
                    print(head + "undefined; raw validator shifts: " + "  ".join(
                        f"{v.replace('densenet121-res224-', '')} {s['mean']:+.4f} "
                        f"[{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}]"
                        for v, s in e["raw_validator_shifts"].items()))


if __name__ == "__main__":
    main(sys.argv[1].rstrip("/"))
