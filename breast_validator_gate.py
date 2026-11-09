"""
breast_validator_gate.py
========================
GATE D — can a validator read a CBIS-DDSM mammogram at all?

WHY THIS GATE DECIDES WHAT THE TRANSFER RATIO MEANS
----------------------------------------------------
A transfer ratio near zero has two completely different explanations, and the
ratio itself cannot tell them apart:

  1. the counterfactual is adversarial — it moved the guidance classifier
     without changing anything a second opinion would recognise as malignancy;
  2. the validator is blind on this data — it would score real malignant and
     real benign images the same way too, so of course it does not follow an
     edit between them.

The RSNA domain never had to separate these, because nih / pc / mimic_ch are
all competent pneumonia classifiers on chest radiographs by construction. The
mammography validators are not: one is trained on 432 images, another on a
different country's full-field-digital corpus and asked to read digitised film.
Either could be at chance here for reasons that say nothing about the edits.

So every validator is scored on the SAME 400 real eval originals — the exact
images whose counterfactuals it will adjudicate — against their radiologist
pathology labels (190 benign / 210 malignant, a near-even split). A validator
below MIN_ACCEPTABLE_AUC does not get its transfer ratio interpreted; it is
reported as UNINTERPRETABLE, never as ADVERSARIAL.

CONTAMINATION IS REPORTED, NOT CORRECTED
-----------------------------------------
The 400 originals are clean for the -breast-heldout and -breast-cmmd
validators by construction. They are NOT clean for resnet18-breast-cbis, which
trained on 336 of them. That AUC is therefore an inflated upper bound and is
labelled as one, because it is the honest characterisation of the validator the
project has been treating as held out — not a number to quietly drop.

    module load python/3.12.8 && source venv/bin/activate
    PPO_DOMAIN=cbis_ddsm_breast python breast_validator_gate.py
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

import config
import domains
from train_breast_classifier import MIN_ACCEPTABLE_AUC


def build_validator(weights: str, members: list[str] | None = None):
    """
    A single checkpoint, or the mean-probability ensemble of CV fold members.

    Fold ensembling is not a performance trick here: each fold model saw only
    k-2 folds of an already small pool, so any one of them is noisy. Averaging
    them is what makes the object being gated the same object that scores the
    counterfactuals.
    """
    from src.breast_classifier import (build_breast_classifier,
                                       build_breast_ensemble_classifier)
    if members and len(members) > 1:
        return build_breast_ensemble_classifier(weights_list=list(members))
    return build_breast_classifier(weights=(members or [weights])[0])


@torch.no_grad()
def score(clf, X: torch.Tensor, bs: int = 32) -> np.ndarray:
    out = []
    for i in range(0, len(X), bs):
        out.append(clf(X[i:i + bs].to(config.DEVICE)).squeeze(-1).float().cpu())
    return torch.cat(out).numpy()


def discover(reports: list[str]) -> list[dict]:
    """
    Assemble the validator roster: the domain's declared one, plus every
    trained CV ensemble a validator report names.
    """
    profile = domains.get("cbis_ddsm_breast")
    roster = [{"weights": w,
               "members": [w],
               "kind": "same corpus, same train fold as guidance",
               "independent": False}
              for w in profile.heldout_validators]

    for path in reports:
        if not os.path.exists(path):
            print(f"[gate] {path} not found — skipping")
            continue
        rep = json.load(open(path))
        corpus = rep.get("corpus", "?")
        for arch, r in rep["archs"].items():
            kind = ("external corpus (CMMD): no shared patients, scanner or "
                    "curation" if rep.get("tag") == "cmmd"
                    else "same corpus, patient-disjoint from guidance train "
                         "and from every eval image")
            roster.append({"weights": r["weights"], "members": r["members"],
                           "kind": kind, "independent": True,
                           "train_corpus": corpus,
                           "oof_auc_on_own_corpus": r["oof_auc"]})
    return roster


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index-json",
                    default=os.path.join(config.CKPT_DIR, "eval_group_index.json"))
    ap.add_argument("--reports", nargs="*", default=[
        os.path.join(config.BREAST_CKPT_DIR, "validator_report_heldout.json"),
        os.path.join(config.BREAST_CKPT_DIR, "validator_report_cmmd.json"),
    ])
    ap.add_argument("--shard-dir",
                    default=os.path.join(config.CKPT_DIR, "eval_shards"))
    ap.add_argument("--out",
                    default=os.path.join(config.CKPT_DIR, "validator_gate.json"))
    args = ap.parse_args()

    from sklearn.metrics import roc_auc_score
    from breast_shard_audit import load_shards

    if not os.path.exists(args.index_json):
        raise SystemExit(f"{args.index_json} not found — run breast_shard_audit.py")
    audit = json.load(open(args.index_json))
    if "xn_labels" not in audit:
        raise SystemExit(
            f"{args.index_json} predates the label fields; re-run "
            f"breast_shard_audit.py so the gate scores against ground truth "
            f"rather than against the group assignment that f(x) produced."
        )

    T, _ = load_shards(args.shard_dir)
    X = torch.cat([T["xn"], T["xp"]])
    y = np.array(audit["xn_labels"] + audit["xp_labels"])

    train_fold = set()
    contaminated = audit["gate_c"]["eval_images_in_guidance_train"]
    print(f"GATE D — validator competence on {len(y)} real CBIS-DDSM eval "
          f"originals")
    print(f"  labels: {int((y == 0).sum())} benign / {int((y == 1).sum())} malignant "
          f"(radiologist pathology, not the f(x) group assignment)")
    print(f"  floor:  AUC >= {MIN_ACCEPTABLE_AUC}\n")

    roster = discover(args.reports)
    results = []
    for v in roster:
        clf = build_validator(v["weights"], v.get("members"))
        s = score(clf, X)
        auc = float(roc_auc_score(y, s))
        # A validator that saw these images in training cannot have its AUC
        # here read as evidence of competence on unseen data.
        clean = v["independent"]
        passes = bool(auc >= MIN_ACCEPTABLE_AUC)
        flag = ("PASS" if passes else "FAIL — near chance on this domain")
        note = "" if clean else f"  [INFLATED: trained on {contaminated}/{len(y)} of these]"
        print(f"  {v['weights']:34s} AUC={auc:.4f}  {flag}{note}")
        print(f"      {v['kind']}")
        if "oof_auc_on_own_corpus" in v:
            print(f"      out-of-fold AUC on its own training corpus "
                  f"({v.get('train_corpus')}): {v['oof_auc_on_own_corpus']:.4f}")
        results.append({**v, "auc_on_cbis_eval_originals": auc,
                        "passes_floor": passes,
                        "auc_is_contaminated": not clean,
                        "n": int(len(y))})
        del clf
        torch.cuda.empty_cache()

    out = {"n_eval_originals": int(len(y)),
           "n_benign": int((y == 0).sum()), "n_malignant": int((y == 1).sum()),
           "min_acceptable_auc": MIN_ACCEPTABLE_AUC,
           "eval_images_in_guidance_train": contaminated,
           "validators": results}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")

    usable = [r for r in results if r["passes_floor"] and not r["auc_is_contaminated"]]
    print("\n" + "=" * 70)
    if usable:
        print(f"{len(usable)} independent validator(s) cleared the floor; their "
              f"transfer ratios are interpretable:")
        for r in usable:
            print(f"  - {r['weights']} (AUC {r['auc_on_cbis_eval_originals']:.3f})")
    else:
        print("NO independent validator cleared the floor.")
        print("Every transfer ratio from the failing ones must be reported as")
        print("UNINTERPRETABLE — a near-chance validator does not follow a real")
        print("malignancy edit either, so a low ratio is not evidence of an")
        print("adversarial one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
