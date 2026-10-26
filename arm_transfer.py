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

NON-CHEST DOMAINS
-----------------
The classifier zoo is dispatched off the active DomainProfile's build_guidance
hook, so the mammography domain's torchvision backbones load through
src/breast_classifier.py while the chest domains keep loading xrv DenseNets.
Two flags exist for that domain in particular:

  --guidance  an explicit denominator, for shards whose config stamp cannot
              describe them. The breast shards predate the
              USE_GUIDANCE_ENSEMBLE stamp AND record the chest constants in
              GUIDANCE_ENSEMBLE_WEIGHTS, so the stamp is worse than useless
              there. Verify any override with breast_shard_audit.py Gate A,
              which checks the named model reproduces the published CV.
  --json      write the numbers out. Historically this tool printed and
              persisted nothing, and every published ratio was transcribed by
              hand from a job log.

If <CKPT_DIR>/validator_gate.json exists, each validator's AUC on real images
is printed beside its ratio, and a validator that failed that floor is reported
as UNINTERPRETABLE rather than ADVERSARIAL — a classifier at chance does not
follow a genuine disease edit either, so its low ratio is not evidence.

    module load python/3.12.8 && source venv/bin/activate
    python arm_transfer.py checkpoints/eval_shards_none_cm0 [more dirs...]

    PPO_DOMAIN=cbis_ddsm_breast python arm_transfer.py \
        checkpoints/cbis_ddsm_breast/eval_shards \
        --guidance densenet121-breast-cbis \
        --validators resnet18-breast-cbis,resnet18-breast-heldout \
        --bootstrap --json checkpoints/cbis_ddsm_breast/heldout_transfer.json
"""

from __future__ import annotations

import glob
import os
import re
import sys

import numpy as np
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


def _guidance_members(stamp: dict, override: str | None = None) -> list[str]:
    """
    What actually guided this arm — taken from the shard's OWN stamp, never from
    the live domain or config.

    The stamp records the roster that was in force at generation time. Reading
    the domain's list instead looks equivalent and is not: an arm run with an
    overridden --ensemble-weights, or one whose roster was truncated in transit,
    guides on a roster the domain knows nothing about. That mistake produced a
    fake result here — a 3-member arm whose export was silently cut to [rsna]
    still had its denominator computed as the domain's 2-member ensemble, which
    inflated every ratio and made the arm look like the best in the sweep. Same
    bug class as heldout_validation's ensemble denominator; the stamp exists
    precisely so it cannot recur, so use it.
    """
    if override:
        # An explicit denominator, for shards whose stamp cannot describe them.
        # The mammography shards are the case this exists for: they were written
        # before the USE_GUIDANCE_ENSEMBLE stamp, and their
        # GUIDANCE_ENSEMBLE_WEIGHTS field records the CHEST constants
        # (densenet121-res224-rsna/chex) because the breast domain sets guidance
        # through its DomainProfile rather than through that config variable. The
        # pre-stamp fallback below happens to reach the right answer for them,
        # and relying on a coincidence is exactly the bug class the stamp was
        # introduced to end. breast_shard_audit.py Gate A is what verifies the
        # override is right: the named model must reproduce the published CV.
        return [w.strip() for w in override.split(",") if w.strip()]

    use_ens = stamp.get("USE_GUIDANCE_ENSEMBLE")
    members = list(stamp.get("GUIDANCE_ENSEMBLE_WEIGHTS", []))
    if use_ens is None:                         # pre-stamp shards
        print("  [warn] shards predate the USE_GUIDANCE_ENSEMBLE stamp; "
              "assuming the single guidance model")
        use_ens = False
    if use_ens:
        if not members:
            raise ValueError(
                "shards claim ensemble guidance but carry no "
                "GUIDANCE_ENSEMBLE_WEIGHTS — cannot determine the denominator."
            )
        return members
    try:
        import domains
        return [domains.active().guidance_weights]
    except Exception:
        return members[:1]


def _validators(override: str | None = None) -> list[str]:
    """
    The domain's held-out set by default; an explicit override when an arm moves
    one of them into guidance.

    Scaling the ensemble runs out of clean members fast: every remaining weight
    set shares TRAINING DATA with a validator (mimic_nb with mimic_ch, 'all'
    with all three). A member that saw a validator's data can lift transfer for
    a reason that is not "the edit carries real disease signal", so the honest
    move is to drop the overlapping validator and compare EVERY arm on the
    reduced set — which is what this override is for. Comparing one arm on
    {nih, pc} against another on {nih, pc, mimic_ch} would not be a comparison.
    """
    if override:
        # Short names are expanded to xrv weight sets, which is the convention
        # for the chest domains. A non-xrv domain names its models in its own
        # scheme (the breast domain's "-breast-" infix), so those pass through
        # untouched rather than being mangled into a densenet121-res224-* key
        # that exists in no zoo.
        return [
            w if (w.startswith("densenet121-") or "-breast-" in w)
            else f"densenet121-res224-{w}"
            for w in (t.strip() for t in override.split(",")) if w
        ]
    try:
        import domains
        return list(domains.active().heldout_validators)
    except Exception:
        return ["densenet121-res224-" + s for s in ("nih", "pc", "mimic_ch")]


def _validator_groups() -> dict[str, list[str]]:
    """
    validator name -> its checkpoint members, for cross-validated validators.

    train_breast_validator.py trains K fold models per architecture and the
    validator IS their mean-probability ensemble, so "resnet18-breast-heldout"
    names an object with five checkpoints behind it. Reading the mapping off
    the trainer's own report keeps the roster single-sourced: adding a fold or
    an architecture changes the report, and nothing here.
    """
    import glob as _glob
    import json as _json

    groups: dict[str, list[str]] = {}
    ckpt_dir = getattr(config, "BREAST_CKPT_DIR", None)
    if not ckpt_dir:
        return groups
    for path in _glob.glob(os.path.join(ckpt_dir, "validator_report_*.json")):
        try:
            rep = _json.load(open(path))
        except Exception as e:
            print(f"  [warn] could not read {path}: {e}")
            continue
        for r in rep.get("archs", {}).values():
            if r.get("members"):
                groups[r["weights"]] = list(r["members"])
    return groups


def _score(clf, X, bs: int = 32) -> torch.Tensor:
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            x = X[i : i + bs].to(config.DEVICE)
            x = x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x
            out.append(clf(x).squeeze(-1).float().cpu())
    return torch.cat(out)


def _per_image_deltas(clf, xn, xp, xc_np, xc_pn) -> tuple[np.ndarray, np.ndarray]:
    """Per-image movement TOWARD the target, per direction — what _deltas collapses to a mean."""
    d_np = (_score(clf, xc_np) - _score(clf, xn)).numpy()
    d_pn = (_score(clf, xp) - _score(clf, xc_pn)).numpy()
    return d_np, d_pn


def _deltas(clf, xn, xp, xc_np, xc_pn) -> tuple[float, float]:
    """Mean movement TOWARD the target, per direction."""
    d_np, d_pn = _per_image_deltas(clf, xn, xp, xc_np, xc_pn)
    return float(d_np.mean()), float(d_pn.mean())


def _bootstrap_ratio_ci(guidance_delta: np.ndarray, validator_delta: np.ndarray,
                         n_boot: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    """
    Percentile bootstrap CI on ratio-of-means = mean(validator_delta) / mean(guidance_delta).

    Resamples IMAGE INDICES jointly for numerator and denominator in every
    replicate, so the within-image pairing (both deltas come from the same
    generated counterfactual, just scored by a different classifier) is
    preserved. This is a ratio-of-means CI for ONE arm's shards — it does not
    assume any pairing ACROSS arms, which arm_transfer.py cannot rely on
    since the group-selection classifier (and therefore which images land in
    Xn/Xp) can differ between arms.
    """
    rng = np.random.default_rng(seed)
    n = len(guidance_delta)
    point = float(validator_delta.mean() / guidance_delta.mean())
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        den = guidance_delta[idx].mean()
        boot[i] = validator_delta[idx].mean() / den if den else np.nan
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return point, float(lo), float(hi)


def _verdict(r: float) -> str:
    return "ADVERSARIAL" if r < 0.15 else ("partial" if r < 0.5 else "TRANSFERS")


def _builders(pathology: str):
    """
    (single, ensemble) constructors for the ACTIVE domain's model zoo.

    The chest domains load torchxrayvision DenseNets; the mammography domain
    loads torchvision backbones this project trained, whose names exist in no
    zoo. Dispatching on the profile's build_guidance hook — the same flag
    DomainProfile.validate() uses to decide whether op_threshs is even a
    meaningful thing to check — keeps that difference in one place instead of
    spreading a domain name check through the scoring path.
    """
    try:
        import domains
        custom = domains.active().build_guidance is not None
    except Exception:
        custom = False

    if custom:
        from src.breast_classifier import (build_breast_classifier,
                                           build_breast_ensemble_classifier)
        return (lambda w: build_breast_classifier(weights=w, pathology=pathology),
                lambda ws: build_breast_ensemble_classifier(weights_list=list(ws),
                                                            pathology=pathology))
    from src.xrv_classifier import build_xrv_classifier, build_ensemble_classifier
    return (lambda w: build_xrv_classifier(weights=w, pathology=pathology),
            lambda ws: build_ensemble_classifier(weights_list=list(ws),
                                                 pathology=pathology))


def _expand(members: list[str], groups: dict[str, list[str]]) -> list[str]:
    """
    Resolve a validator name to the checkpoints it is made of.

    A cross-validated validator is one object with several checkpoints behind
    it (resnet18-breast-heldout = the mean over its five fold models). The
    caller names the validator; this turns that into the member list the
    ensemble constructor wants.
    """
    out = []
    for m in members:
        out.extend(groups.get(m, [m]))
    return out


def main(dirs: list[str], validators_override: str | None = None,
         bootstrap: bool = False, guidance_override: str | None = None,
         json_path: str | None = None) -> int:
    try:
        import domains
        pathology = domains.active().pathology
    except Exception:
        pathology = "Pneumonia"

    build_single, build_ens = _builders(pathology)
    groups = _validator_groups()
    cache: dict[tuple, object] = {}

    def clf_for(members: list[str]):
        key = tuple(members)
        if key not in cache:
            real = _expand(list(members), groups)
            cache[key] = build_ens(real) if len(real) > 1 else build_single(real[0])
        return cache[key]

    vals = _validators(validators_override)
    summary = {}
    record: dict = {"validators": vals, "arms": {}}

    # Gate D, if it has been run: a transfer ratio from a validator that cannot
    # separate real malignant from real benign on this domain says nothing about
    # the counterfactuals, so the verdict has to know about it.
    gate = {}
    gate_path = os.path.join(config.CKPT_DIR, "validator_gate.json")
    if os.path.exists(gate_path):
        import json as _json
        for r in _json.load(open(gate_path)).get("validators", []):
            gate[r["weights"]] = r
        print(f"[gate] read validator competence from {gate_path}")

    for d in dirs:
        name = os.path.basename(d.rstrip("/"))
        print(f"\n{'=' * 74}\n{name}\n{'=' * 74}")
        xn, xp, xc_np, xc_pn, stamp = _load_shards(d)
        gm = _guidance_members(stamp, guidance_override)
        short = lambda w: w.replace("densenet121-res224-", "").replace("-breast-", "/")
        print(f"  n={len(xn)} per group | guidance = {'+'.join(map(short, gm))} "
              f"| groups = {stamp.get('EVAL_GROUP_SOURCE', '?')} "
              f"| gradcam = {stamp.get('USE_GRADCAM_MASK', '?')}")

        if bootstrap:
            g_np_arr, g_pn_arr = _per_image_deltas(clf_for(gm), xn, xp, xc_np, xc_pn)
            g_np, g_pn = float(g_np_arr.mean()), float(g_pn_arr.mean())
        else:
            g_np, g_pn = _deltas(clf_for(gm), xn, xp, xc_np, xc_pn)
        print(f"  Δ_guidance  n→p={g_np:.4f}  p→n={g_pn:.4f}")
        print(f"  {'validator':12s} {'Δ n→p':>8s} {'Δ p→n':>8s} "
              f"{'ratio n→p':>10s} {'ratio p→n':>10s}   verdict")

        arm_rec = {"n_per_group": int(len(xn)), "guidance": gm,
                   "delta_guidance_n_to_p": None, "delta_guidance_p_to_n": None,
                   "group_source": stamp.get("EVAL_GROUP_SOURCE", "?"),
                   "validators": {}}
        ratios, headline = [], []
        for v in vals:
            if bootstrap:
                v_np_arr, v_pn_arr = _per_image_deltas(clf_for([v]), xn, xp, xc_np, xc_pn)
                d_np, d_pn = float(v_np_arr.mean()), float(v_pn_arr.mean())
            else:
                d_np, d_pn = _deltas(clf_for([v]), xn, xp, xc_np, xc_pn)
            r_np = d_np / g_np if g_np else float("nan")
            r_pn = d_pn / g_pn if g_pn else float("nan")
            ratios.append((r_np, r_pn))
            g = gate.get(v)
            usable = (g is None) or g.get("passes_floor", True)
            # A validator only belongs in the headline mean if its ratio can be
            # read at all: it must be able to tell the two classes apart on real
            # images (competence), and it must not have trained on the images it
            # is scoring (independence). Averaging a near-chance validator in
            # would drag the mean toward zero for a reason that has nothing to do
            # with the counterfactuals; averaging a contaminated one in would
            # drag it up for a reason that has nothing to do with them either.
            if g is None or (g.get("passes_floor") and not g.get("auc_is_contaminated")):
                headline.append((r_np, r_pn))
            verdict = (f"{_verdict(r_np)}/{_verdict(r_pn)}" if usable
                       else "UNINTERPRETABLE")
            print(f"  {short(v):22s} {d_np:8.4f} {d_pn:8.4f} "
                  f"{r_np:10.3f} {r_pn:10.3f}   {verdict}")
            entry = {"delta_n_to_p": d_np, "delta_p_to_n": d_pn,
                     "ratio_n_to_p": r_np, "ratio_p_to_n": r_pn,
                     "verdict": verdict}
            if g is not None:
                entry["auc_on_real_eval_originals"] = g["auc_on_cbis_eval_originals"]
                entry["auc_is_contaminated"] = g["auc_is_contaminated"]
                entry["passes_competence_floor"] = g["passes_floor"]
                mark = " (INFLATED — trained on these)" if g["auc_is_contaminated"] else ""
                print(f"    validator AUC on the 400 real originals: "
                      f"{g['auc_on_cbis_eval_originals']:.3f}{mark}")
                if not usable:
                    print(f"    near chance on this domain -> the ratio above "
                          f"measures the validator, not the edits")
            # Does this validator disagree about the EDIT, or about everything?
            # Rank agreement with the guidance model on the UNEDITED originals
            # separates the two. The same control heldout_validation.py reports.
            try:
                from scipy.stats import spearmanr
                orig = float(spearmanr(
                    np.concatenate([_score(clf_for(gm), xn).numpy(),
                                    _score(clf_for(gm), xp).numpy()]),
                    np.concatenate([_score(clf_for([v]), xn).numpy(),
                                    _score(clf_for([v]), xp).numpy()])).statistic)
                entry["original_agreement_spearman"] = orig
                print(f"    rank agreement with guidance on the 400 unedited "
                      f"originals: rho={orig:+.3f}")
            except Exception as e:
                print(f"    [warn] spearman control unavailable: {e}")

            if bootstrap:
                _, lo_np, hi_np = _bootstrap_ratio_ci(g_np_arr, v_np_arr)
                _, lo_pn, hi_pn = _bootstrap_ratio_ci(g_pn_arr, v_pn_arr)
                print(f"    95% CI  n→p [{lo_np:.3f}, {hi_np:.3f}]"
                      f"   p→n [{lo_pn:.3f}, {hi_pn:.3f}]  (10000 resamples)")
                entry["ci_n_to_p"] = [lo_np, hi_np]
                entry["ci_p_to_n"] = [lo_pn, hi_pn]
                entry["n_boot"] = 10000
            arm_rec["validators"][v] = entry

        use = headline or ratios
        m_np = sum(r for r, _ in use) / len(use)
        m_pn = sum(r for _, r in use) / len(use)
        summary[name] = (m_np, m_pn)
        label = ("MEAN" if len(use) == len(ratios)
                 else f"MEAN ({len(use)}/{len(ratios)} usable)")
        print(f"  {label:22s} {'':>17s} {m_np:10.3f} {m_pn:10.3f}")
        if len(use) != len(ratios):
            print(f"    averaged over the independent validators that cleared "
                  f"the competence floor; the rest are listed above but excluded")
        if not headline and ratios:
            print(f"    WARNING: no validator is both independent and competent "
                  f"here — this mean is not an independence check")
        arm_rec["delta_guidance_n_to_p"] = g_np
        arm_rec["delta_guidance_p_to_n"] = g_pn
        arm_rec["mean_ratio_n_to_p"] = m_np
        arm_rec["mean_ratio_p_to_n"] = m_pn
        arm_rec["mean_over_n_validators"] = len(use)
        arm_rec["mean_is_independence_check"] = bool(headline)
        record["arms"][name] = arm_rec

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

    if json_path:
        # arm_transfer.py historically wrote nothing, and every published
        # transfer number was hand-transcribed out of a job log. A table that
        # cites a file as its source needs that file to exist.
        import json as _json
        os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
        with open(json_path, "w") as f:
            _json.dump(record, f, indent=2)
        print(f"\nwrote {json_path}")
    return 0


if __name__ == "__main__":
    argv = sys.argv[1:]
    override = None
    if "--validators" in argv:
        i = argv.index("--validators")
        override = argv[i + 1]
        del argv[i : i + 2]
    guidance = None
    if "--guidance" in argv:
        i = argv.index("--guidance")
        guidance = argv[i + 1]
        del argv[i : i + 2]
    json_path = None
    if "--json" in argv:
        i = argv.index("--json")
        json_path = argv[i + 1]
        del argv[i : i + 2]
    bootstrap = "--bootstrap" in argv
    if bootstrap:
        argv.remove("--bootstrap")
    if not argv:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(argv, override, bootstrap, guidance, json_path))
