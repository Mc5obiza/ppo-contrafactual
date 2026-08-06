"""
heldout_validation.py
=====================
Answers ONE question, which nothing else in this pipeline can currently
answer: are the counterfactuals real disease edits, or adversarial
perturbations against f(.)?

WHY THIS EXISTS
---------------
pipeline.build_common() builds ONE classifier — xrv densenet121-res224-rsna —
and that same frozen object is used for BOTH:
  - the guidance gradient inside generator.generate_counterfactual (grad_cls),
  - the CV score in metrics.run_full_evaluation.
So CV measures "did gradient descent on f succeed against f". It is circular
by construction and is satisfied just as well by an adversarial direction as
by a semantic one. The eval of job 17648985 reported CV=0.975 while changing
73.6% / 77.2% of pixels at MAE 0.036 / 0.025 — small-amplitude, wide-support,
which is the fingerprint of an adversarial perturbation rather than a
localized opacity edit.

This script re-scores the SAME counterfactuals with classifiers that were
never in the guidance loop. If the score shift transfers, the edits carry
real pneumonia signal. If it collapses, they are adversarial and no amount
of reward tuning will fix that on its own — the action space and the
constraints have to change first.

VALIDATOR CHOICE
----------------
All candidates below are already cached under ~/.torchxrayvision/models_data/
by setup_env.sh's prefetch, so this runs offline on a compute node:
  densenet121-res224-nih      NIH ChestX-ray14 only     — fully independent
  densenet121-res224-chex     CheXpert only             — fully independent
  densenet121-res224-mimic_ch MIMIC-CXR only            — fully independent
  densenet121-res224-pc       PadChest only             — fully independent
  densenet121-res224-all      joint, INCLUDES RSNA/kaggle — NOT independent,
                              usable as a soft check but do not report it as
                              a held-out validator.
Default is nih + chex: two independent training corpora, two independent
label definitions, neither sharing data with the RSNA-trained guidance model.

WHAT IT REPORTS
---------------
Per classifier, per direction:
  mean f(x), mean f(x_cf), mean delta, and the paper's strict CV.
Plus three summary numbers that are the actual verdict:
  transfer ratio     delta_validator / delta_guidance. Threshold-free, so it
                     does not depend on the validator agreeing with the
                     guidance model's operating point. >0.5 = edits transfer;
                     <0.15 = adversarial.
  conditional CV     CV restricted to images the validator ALSO puts on the
                     expected side of 0.5 to begin with. The strict CV is
                     unfairly pessimistic on a validator otherwise, since an
                     image f_rsna scores 0.1 may be scored 0.6 by NIH — that
                     is validator disagreement, not a counterfactual failure.
  edit concentration fraction of total |x_cf - x| mass sitting in the top 5%
                     of pixels. Needs no model at all. A focal consolidation
                     should put >0.5 of its mass there; a diffuse adversarial
                     wash sits near 0.05 (= the uniform-mass baseline).

CACHING
-------
run_full_evaluation() never persisted the generated counterfactuals, so the
only way to re-score them was to regenerate — ~400 policy episodes. This
script writes checkpoints/heldout/cf_cache.pt on the first run, so adding a
validator later costs seconds instead of hours. Pass --regenerate to force a
fresh generation (do this after ANY change to config, the policies, or the
generator weights — the cache does not know it went stale).

USAGE
-----
    python -u heldout_validation.py                       # nih + chex
    python -u heldout_validation.py --validators nih,chex,mimic_ch
    python -u heldout_validation.py --n-eval 50           # quick check
    python -u heldout_validation.py --regenerate          # after retraining
or  sbatch run_heldout_validation.slurm
"""

import argparse
import json
import os

import numpy as np
import torch

import config
from pipeline import build_common, load_generator_weights, load_directional_policy
from src.inference import EVAL_PN_SEED_OFFSET, generate_batch_with_policy
from src.metrics import (
    counterfactual_validity,
    current_cfg_stamp as _current_cfg_stamp,
    edit_concentration as _edit_concentration,
    scan_groups,
    score_batch as _score,
)
from src.xrv_classifier import build_xrv_classifier, build_ensemble_classifier


# ROTATED 2026-08-01: chex moved INTO the guidance ensemble
# (config.GUIDANCE_ENSEMBLE_WEIGHTS), so it is no longer held out. These three
# must stay disjoint from that list — a "validator" that guided generation
# measures nothing. Head counts: mimic_ch 11, pc 15, nih 14.
# nih is retained for continuity with the pre-v2 run despite being the weakest
# of the three (NLP-mined Pneumonia label, ~1% prevalence).
DEFAULT_VALIDATORS = ["mimic_ch", "pc", "nih"]
CACHE_NAME = "cf_cache.pt"


def _normalize(name: str) -> str:
    """'chex' -> 'densenet121-res224-chex'; full names pass through."""
    name = name.strip()
    return name if name.startswith("densenet121") else f"densenet121-res224-{name}"


def _parse_spec(spec: str) -> list[str]:
    """Comma-separated weight spec -> list of full xrv weight names."""
    return [_normalize(p) for p in spec.split(",") if p.strip()]


def _spec_label(members: list[str]) -> str:
    """Report/JSON key for a scorer: a bare name, or 'ensemble(a+b)'."""
    short = [m.replace("densenet121-res224-", "") for m in members]
    return short[0] if len(short) == 1 else f"ensemble({'+'.join(short)})"


def _build_scorer(members: list[str], device: str):
    """
    Build a scorer for one or more weight sets.

    Single member -> XRVPneumoniaClassifier. Multiple -> the same
    EnsemblePneumoniaClassifier that guides generation, so the guidance row of
    this report reflects exactly what the counterfactuals were optimised
    against rather than one arbitrary member of it.
    """
    if len(members) == 1:
        return build_xrv_classifier(device=device, weights=members[0])
    return build_ensemble_classifier(device=device, weights_list=members)


# ─────────────────────────────────────────────────────────────────────────────
# Group scan + generation  (mirrors metrics.run_full_evaluation exactly)
# ─────────────────────────────────────────────────────────────────────────────

def _scan_groups(dataset, classifier, n_eval: int, device: str):
    """Thin alias — the scan itself now lives in src.metrics.scan_groups, so
    this script and run_full_evaluation cannot drift apart on which images
    land in Xn / Xp."""
    return scan_groups(dataset, classifier, n_eval, device, tag="heldout")


def _warn_if_cache_stale(cached_cfg: dict) -> None:
    """
    Loudly flag a cache generated under a different config. The whole point
    of the ablation is comparing arms, so silently scoring ARM 0's
    counterfactuals and labelling them ARM 1 is the single worst failure
    this script could have.
    """
    current = _current_cfg_stamp()
    # Keys the cached stamp does not carry at all mean it predates their being
    # added to the stamp — NOT that the value differs. Reporting those as drift
    # would flag every pre-existing cache as stale the moment a key is added
    # and push the user into a needless multi-hour regeneration. Compare only
    # the intersection, and list the rest as unknown.
    unknown = [k for k in current if k not in cached_cfg]
    drift = {k: (cached_cfg[k], v) for k, v in current.items()
             if k in cached_cfg and cached_cfg[k] != v}
    if unknown:
        print(f"[heldout] note: cache predates these stamp keys, not checked: "
              f"{', '.join(sorted(unknown))}")
    if drift:
        print("\n" + "!" * 78)
        print("[heldout] WARNING: the cache was generated under a DIFFERENT config:")
        for k, (was, now) in drift.items():
            print(f"    {k}: cached={was}  current={now}")
        print("[heldout] These numbers describe the CACHED arm, not the current one.")
        print("[heldout] Re-run with --regenerate to score the current config.")
        print("!" * 78 + "\n")


def _build_or_load_cache(args, cache_path: str):
    """
    Return (xn, xp, xc_np, xc_pn) as lists of (1,1,H,W) CPU tensors, either
    from the on-disk cache or by running the full generation path.

    The cache stores stacked tensors rather than lists — same content, far
    fewer objects to pickle.
    """
    if os.path.exists(cache_path) and not args.regenerate:
        print(f"[heldout] loading cached counterfactuals from '{cache_path}'")
        print(f"[heldout] (pass --regenerate if the policies or config have "
              f"changed since this cache was written)")
        blob = torch.load(cache_path, map_location="cpu")
        unstack = lambda t: [t[i : i + 1] for i in range(t.shape[0])]
        cached_cfg = blob.get("config", {})
        print(f"[heldout]   cached under config: "
              f"ANCHOR_TO_EPISODE_ORIGIN={cached_cfg.get('ANCHOR_TO_EPISODE_ORIGIN')}, "
              f"USE_GRADCAM_MASK={cached_cfg.get('USE_GRADCAM_MASK')}, "
              f"USE_SPARSITY_REWARD={cached_cfg.get('USE_SPARSITY_REWARD')}")
        _warn_if_cache_stale(cached_cfg)
        return (unstack(blob["xn"]), unstack(blob["xp"]),
                unstack(blob["xc_np"]), unstack(blob["xc_pn"]),
                blob.get("guidance_weights", "densenet121-res224-rsna"),
                cached_cfg)

    # ── full generation path ─────────────────────────────────────────────────
    if args.regenerate and os.path.exists(cache_path):
        print(f"[heldout] --regenerate: overwriting '{cache_path}'")
    ctx = build_common()

    print("\n" + "=" * 70)
    print("Restoring trained checkpoints")
    print("=" * 70)
    load_generator_weights(ctx.generator, ckpt_dir=config.CKPT_DIR)
    policy_n_to_p = load_directional_policy(
        target_c=1.0, label="n_to_p", ctx=ctx, ckpt_dir=config.CKPT_DIR
    )
    policy_p_to_n = load_directional_policy(
        target_c=0.0, label="p_to_n", ctx=ctx, ckpt_dir=config.CKPT_DIR
    )

    xn_list, xp_list = _scan_groups(
        ctx.dataset, ctx.classifier, args.n_eval, config.DEVICE
    )

    # index_offset values match eval_main.py's sharded generate stage (0 for
    # n→p, EVAL_PN_SEED_OFFSET for p→n), so a counterfactual cached here and
    # the same one written to an eval shard are byte-identical.
    print("[heldout] generating Xc (n→p) — this is the expensive part ...")
    xc_np = generate_batch_with_policy(
        xn_list, policy_n_to_p, deterministic=True,
        index_offset=0, label="heldout n_to_p", progress_every=10,
    )
    print("[heldout] generating Xc (p→n) ...")
    xc_pn = generate_batch_with_policy(
        xp_list, policy_p_to_n, deterministic=True,
        index_offset=EVAL_PN_SEED_OFFSET, label="heldout p_to_n", progress_every=10,
    )
    print(ctx.generator.cam_degeneracy_summary())

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    # Stamped so a stale cache is identifiable rather than silently mixing
    # arms of the ablation.
    cfg_stamp = _current_cfg_stamp()
    torch.save(
        {
            "xn": torch.cat(xn_list), "xp": torch.cat(xp_list),
            "xc_np": torch.cat(xc_np), "xc_pn": torch.cat(xc_pn),
            "guidance_weights": args.guidance_weights,
            "config": cfg_stamp,
        },
        cache_path,
    )
    print(f"[heldout] cached counterfactuals to '{cache_path}' — re-runs with "
          f"different validators are now free")

    # Free the heavy models before the scoring phase loads the validators.
    del ctx, policy_n_to_p, policy_p_to_n
    torch.cuda.empty_cache()

    return xn_list, xp_list, xc_np, xc_pn, args.guidance_weights, cfg_stamp


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────

# _score is src.metrics.score_batch, imported above — same batched scoring the
# metric suite uses, so a CV computed here and one in eval_metrics.json can
# never differ because of how the images were batched.


def _conditional_cv(f_x, f_xc, target: str, tau: float, boundary: float = 0.5):
    """
    CV restricted to the subset the validator itself places on the expected
    side of the boundary before any edit.

    The strict CV in metrics.counterfactual_validity requires the ORIGINAL to
    sit on the starting side (f_x <= 0.5 for n→p). The Xn / Xp groups were
    selected by the RSNA guidance model, so on an independent validator some
    originals start on the wrong side — that is validator disagreement about
    the real image, not a counterfactual failure, and it deflates strict CV
    for a reason unrelated to what we are measuring. Conditioning removes
    that confound. Returns (cv, n_eligible, n_total).
    """
    if target == "positive":
        eligible = f_x <= boundary
    else:
        eligible = f_x >= boundary

    n_elig = int(eligible.sum().item())
    if n_elig == 0:
        return float("nan"), 0, int(f_x.numel())

    cv = counterfactual_validity(
        f_x[eligible], f_xc[eligible], target=target, tau=tau, boundary=boundary
    )
    return cv, n_elig, int(f_x.numel())


# Edit concentration now lives in src/metrics.py (imported at the top of this
# module as `_edit_concentration`) so the main eval and this report compute it
# from one implementation. See metrics.edit_concentration for the rationale and
# the calibration figures.


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def _direction_block(f_x, f_xc, target: str, tau: float) -> dict:
    strict_cv = counterfactual_validity(f_x, f_xc, target=target, tau=tau)
    cond_cv, n_elig, n_tot = _conditional_cv(f_x, f_xc, target, tau)
    delta = (f_xc - f_x) if target == "positive" else (f_x - f_xc)
    return dict(
        mean_f_x=float(f_x.mean()),
        mean_f_xc=float(f_xc.mean()),
        mean_delta=float(delta.mean()),      # signed so that >0 = intended
        std_delta=float(delta.std()),
        strict_cv=strict_cv,
        conditional_cv=cond_cv,
        n_eligible=n_elig,
        n_total=n_tot,
    )


def _print_report(results: dict, guidance_name: str) -> None:
    print("\n" + "=" * 78)
    print("  HELD-OUT VALIDATION — do the counterfactuals transfer off f(.)?")
    print("=" * 78)

    hdr = f"{'classifier':<34}{'dir':<7}{'f(x)':>8}{'f(x_cf)':>10}{'Δ':>9}{'CV':>8}{'condCV':>9}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for name, blocks in results["classifiers"].items():
        tag = "GUIDANCE" if name == guidance_name else "held-out"
        for d in ("n_to_p", "p_to_n"):
            b = blocks[d]
            print(f"{name[:32]:<34}{d:<7}{b['mean_f_x']:>8.3f}{b['mean_f_xc']:>10.3f}"
                  f"{b['mean_delta']:>9.3f}{b['strict_cv']:>8.3f}"
                  f"{b['conditional_cv']:>9.3f}")
        n_elig_np = blocks["n_to_p"]["n_eligible"]
        n_elig_pn = blocks["p_to_n"]["n_eligible"]
        n_total = blocks["n_to_p"]["n_total"]
        print(f"{'':<34}({tag}; condCV computed over "
              f"{n_elig_np}/{n_total} and {n_elig_pn}/{n_total} images)")

    print("\n" + "-" * 78)
    print("  TRANSFER RATIO   Δ_validator / Δ_guidance   (threshold-free)")
    print("-" * 78)
    for name, blocks in results["classifiers"].items():
        if name == guidance_name:
            continue
        for d in ("n_to_p", "p_to_n"):
            ratio = results["transfer_ratio"][name][d]
            # NaN means the guidance delta was ~0, i.e. the counterfactual did
            # not move the guidance model either — the ratio is undefined, not
            # evidence of anything. Must not fall through to "ADVERSARIAL".
            verdict = (
                "undefined (guidance Δ≈0)" if not np.isfinite(ratio) else
                "edits transfer"           if ratio > 0.50 else
                "partial transfer"         if ratio > 0.15 else
                "ADVERSARIAL"
            )
            print(f"    {name[:32]:<34}{d:<8}{ratio:>7.3f}   {verdict}")

    print("\n" + "-" * 78)
    print("  ORIGINAL-IMAGE AGREEMENT  (control — is the validator usable?)")
    print("-" * 78)
    for name, rho in results["original_agreement"].items():
        if name == guidance_name:
            continue
        note = "" if rho > 0.4 else "   <- LOW: transfer ratio is confounded"
        print(f"    {name[:32]:<34}Spearman ρ on real images = {rho:>6.3f}{note}")

    print("\n" + "-" * 78)
    print("  EDIT CONCENTRATION  (model-free; top-5% of pixels' share of |Δ|)")
    print("-" * 78)
    for d in ("n_to_p", "p_to_n"):
        e = results["edit_concentration"][d]
        print(f"    {d:<10}concentration={e['concentration']:.3f}  "
              f"(uniform baseline {e['uniform_baseline']:.2f})   "
              f"MAE={e['mae']:.4f}   changed={100 * e['frac_changed']:.1f}%")
    print("    A focal consolidation should exceed ~0.5. Near the baseline means")
    print("    the edit is spread uniformly — diffuse, not localized.")
    print("=" * 78)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    ap.add_argument("--n-eval", type=int, default=200,
                    help="images per group (default 200, matches eval_main.py)")
    ap.add_argument("--validators", type=str, default=",".join(DEFAULT_VALIDATORS),
                    help="comma-separated xrv weight suffixes, e.g. 'mimic_ch,pc'. "
                         "Must be disjoint from --guidance-weights. 'all' is NOT "
                         "independent of anything (it includes RSNA).")
    ap.add_argument("--guidance-weights", type=str,
                    default=",".join(config.GUIDANCE_ENSEMBLE_WEIGHTS),
                    help="the classifier(s) that were in the guidance loop; "
                         "comma-separated means an ensemble. Defaults to "
                         "config.GUIDANCE_ENSEMBLE_WEIGHTS so this tracks the "
                         "pipeline automatically.")
    ap.add_argument("--tau", type=float, default=0.8, help="CV margin (paper: 0.8)")
    ap.add_argument("--regenerate", action="store_true",
                    help="ignore the cache and regenerate counterfactuals")
    ap.add_argument("--out-dir", type=str, default=os.path.join(config.CKPT_DIR, "heldout"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cache_path = os.path.join(args.out_dir, CACHE_NAME)

    xn, xp, xc_np, xc_pn, guidance_weights, cfg_stamp = _build_or_load_cache(
        args, cache_path
    )

    # ── score with the guidance model(s) + every validator ───────────────────
    guidance_members = _parse_spec(guidance_weights)
    guidance_label   = _spec_label(guidance_members)

    validator_members = _parse_spec(args.validators)
    # A validator that guided generation measures nothing, so drop overlaps
    # loudly rather than silently reporting a ratio of ~1.0 as if it were
    # evidence the edits are semantic.
    leaked = [v for v in validator_members if v in guidance_members]
    if leaked:
        print(f"[heldout] WARNING: dropping {leaked} from the validators — "
              f"they are part of the guidance ensemble and are therefore NOT "
              f"held out.")
    validator_members = [v for v in validator_members if v not in guidance_members]
    if not validator_members:
        raise SystemExit(
            "[heldout] no held-out validators left after removing guidance "
            "members. Pick validators disjoint from "
            f"{[m.replace('densenet121-res224-', '') for m in guidance_members]}."
        )

    # (label, members) pairs — guidance first so it heads the report table.
    scorers = [(guidance_label, guidance_members)] + [
        (_spec_label([v]), [v]) for v in validator_members
    ]

    results = dict(
        n_eval=len(xn),
        tau=args.tau,
        guidance_classifier=guidance_label,
        guidance_members=guidance_members,
        validators=[lbl for lbl, _ in scorers[1:]],
        config_at_generation=cfg_stamp,
        classifiers={},
        transfer_ratio={},
        original_agreement={},
    )

    raw_scores = {}
    for name, members in scorers:
        print(f"\n[heldout] scoring with '{name}' ...")
        clf = _build_scorer(members, config.DEVICE)

        f_xn = _score(xn, clf, config.DEVICE)
        f_xp = _score(xp, clf, config.DEVICE)
        f_cnp = _score(xc_np, clf, config.DEVICE)
        f_cpn = _score(xc_pn, clf, config.DEVICE)
        raw_scores[name] = (f_xn, f_xp)

        results["classifiers"][name] = dict(
            n_to_p=_direction_block(f_xn, f_cnp, "positive", args.tau),
            p_to_n=_direction_block(f_xp, f_cpn, "negative", args.tau),
        )

        del clf
        torch.cuda.empty_cache()

    # ── transfer ratios + the agreement control ──────────────────────────────
    from scipy import stats as scipy_stats

    g_blocks = results["classifiers"][guidance_label]
    g_xn, g_xp = raw_scores[guidance_label]
    for name, _members in scorers:
        if name == guidance_label:
            continue
        v_blocks = results["classifiers"][name]
        results["transfer_ratio"][name] = {
            d: float(v_blocks[d]["mean_delta"] / g_blocks[d]["mean_delta"])
            if abs(g_blocks[d]["mean_delta"]) > 1e-8 else float("nan")
            for d in ("n_to_p", "p_to_n")
        }
        v_xn, v_xp = raw_scores[name]
        rho, _p = scipy_stats.spearmanr(
            torch.cat([g_xn, g_xp]).numpy(), torch.cat([v_xn, v_xp]).numpy()
        )
        results["original_agreement"][name] = float(rho)

    # ── model-free edit concentration ────────────────────────────────────────
    results["edit_concentration"] = dict(
        n_to_p=_edit_concentration(xn, xc_np),
        p_to_n=_edit_concentration(xp, xc_pn),
    )

    _print_report(results, guidance_label)

    out_path = os.path.join(args.out_dir, "heldout_validation.json")
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n[heldout] results saved to '{out_path}'")


if __name__ == "__main__":
    main()
