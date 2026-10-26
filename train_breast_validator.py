"""
train_breast_validator.py
=========================
Trains the GENUINELY INDEPENDENT held-out validators for the mammography
domain, on the only CBIS-DDSM patients that qualify.

WHY THE EXISTING VALIDATOR IS NOT ONE
-------------------------------------
domains/cbis_ddsm_breast.py reserves resnet18-breast-cbis as the held-out
validator, and DomainProfile.validate() does enforce that it never appears in
the guidance fields. That check is about MODEL identity, and it passes. But
train_breast_classifier.py trains all three backbones on the SAME
patient_level_split(seed=0, val_frac=0.2) — 1168 train patients, 2290 images —
so the "held-out" validator shares every training image with the classifier it
is supposed to independently check.

breast_shard_audit.py measures what that costs: 336 of the 400 eval images
(84.0%), across 304 of 362 patients, sit inside that shared training fold. The
reserved validator was fit on five sixths of the images it is asked to
adjudicate. Its transfer ratio is an upper bound twice over — once because the
corpus is shared, again because the specific images are.

WHAT THIS TRAINS INSTEAD
------------------------
The 292 patients that no -breast-cbis backbone was ever fit on, MINUS every
patient represented in the eval groups: 234 patients / 432 images. A validator
trained there has seen neither the guidance model's training data nor the
images it will score.

WHAT IT STILL IS NOT, AND SAY SO WHEN REPORTING
------------------------------------------------
It is not disjoint from the PPO pool. The env's far-pool scan took the WHOLE
2857-image corpus (1603 images for n->p plus 1254 for p->n = 2857 exactly), so
no CBIS-DDSM patient is untouched by policy rollouts, and no subset of this
dataset can be. Making that literally true would mean retraining PPO on a
reduced pool, which would produce different counterfactuals than the ones on
disk. This validator is disjoint at the two levels that bear on the
measurement — guidance training data, and the scored images — and that is the
claim to make for it, not "fully independent".

FIVE FOLDS, NOT ONE SPLIT
-------------------------
432 images is small. A single 80/20 split would train on ~345 and estimate AUC
on ~87, which is too few to tell 0.62 from 0.72 — and that distinction is the
whole gate. K-fold patient-level CV instead: every image is trained on in K-1
folds and predicted exactly once out-of-fold, so the AUC is measured on all 432
while each fold model still never sees its own validation patients. The
validator itself is the mean-probability ensemble of the fold models, which is
also what makes it robust to any single fold landing badly.

THE FOLDS ARE NESTED, AND THAT IS NOT PEDANTRY
-----------------------------------------------
train_one saves the checkpoint with the best validation AUC and early-stops on
it. Hand it the outer fold and the "out-of-fold" AUC is no longer out of
anything: the epoch was chosen by looking at those labels, so the number is
selected-on-test and biased upward by however much 30 epochs of choice is worth
on ~87 images.

That bias is small in absolute terms and completely disqualifying here, because
the figure it inflates is a PASS/FAIL gate sitting right at 0.65 — and because
this whole exercise exists to stop a number being reported against data that
helped produce it. So each outer fold f gets an INNER selection fold
((f + 1) % k) carved out of its own training set: train on k-2 folds, early-stop
on the inner fold, predict the outer fold once, never having seen it. The cost
is real — training drops from ~345 images to ~259 — and it is the correct price.

HYPERPARAMETERS ARE INHERITED, NOT TUNED
-----------------------------------------
lr, weight decay, augmentation and the epoch budget come from
train_breast_classifier.py unchanged. They were not chosen against the
out-of-fold AUC reported here, and they must not be: tuning them against this
number would re-introduce, at the level of the experiment, exactly the
selection-on-test this file just went to some trouble to remove. If the pool is
too small to clear the floor, that is the finding.

    module load python/3.12.8 && source venv/bin/activate
    PPO_DOMAIN=cbis_ddsm_breast python train_breast_validator.py
or  sbatch run_breast_validator.slurm
"""

from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import config
import domains
from src.breast_classifier import _preprocess, build_backbone, checkpoint_path
from train_breast_classifier import MIN_ACCEPTABLE_AUC, set_seed, train_one


VALIDATOR_TAG = "heldout"

# Corpus tag per external domain. Kept as a map rather than derived from the
# registry key so the weight names stay short and stable even if a domain
# module is renamed — the names end up in checkpoints, JSON and a published
# table, and those should not move when a filename does.
EXTERNAL_TAGS = {"cmmd_breast": "cmmd"}


def fold_weights(arch: str, fold: int, tag: str = VALIDATOR_TAG) -> str:
    return f"{arch}-breast-{tag}-f{fold}"


def validator_weights(arch: str, tag: str = VALIDATOR_TAG) -> str:
    return f"{arch}-breast-{tag}"


def patient_folds(dataset, indices: list[int], k: int, seed: int) -> list[list[int]]:
    """
    Partition `indices` into k folds, splitting by PATIENT and never by image.

    Same reasoning as train_breast_classifier.patient_level_split: CBIS-DDSM
    puts up to six images of one breast in the dataset, so an image-level fold
    boundary leaves the same breast on both sides and the out-of-fold AUC
    becomes a memorisation score. Patients are dealt round-robin after a seeded
    shuffle, which keeps the folds close to equal in image count even though
    patients contribute 1-6 images each.
    """
    by_patient: dict[str, list[int]] = {}
    for i in indices:
        by_patient.setdefault(dataset.patient_id(i), []).append(i)

    patients = sorted(by_patient)
    random.Random(seed).shuffle(patients)

    folds: list[list[int]] = [[] for _ in range(k)]
    # Largest patients first, always into the currently smallest fold: plain
    # round-robin can leave one fold 30% larger when patient sizes vary 1-6.
    for p in sorted(patients, key=lambda q: -len(by_patient[q])):
        folds[min(range(k), key=lambda f: len(folds[f]))].extend(by_patient[p])
    return [sorted(f) for f in folds]


@torch.no_grad()
def predict(arch: str, ckpt: str, dataset, idx: list[int], device: str,
            batch_size: int = 32) -> np.ndarray:
    """P(malignant) for `idx`, from the SAVED checkpoint, in idx order."""
    model = build_backbone(arch, pretrained=False)
    state = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state["state_dict"] if "state_dict" in state else state)
    model = model.to(device).eval()

    out = []
    for x, _ in DataLoader(Subset(dataset, idx), batch_size=batch_size,
                           shuffle=False,
                           num_workers=config.DATALOADER_NUM_WORKERS):
        out.append(torch.sigmoid(model(_preprocess(x.to(device)))).flatten().cpu())
    return torch.cat(out).numpy()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archs", default=None,
                    help="comma-separated; default config.BREAST_VALIDATOR_ARCHS")
    ap.add_argument("--folds", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--index-json", default=None,
                    help="breast_shard_audit.py output; default "
                         "<CKPT_DIR>/eval_group_index.json")
    ap.add_argument("--out", default=None,
                    help="default <BREAST_CKPT_DIR>/validator_report.json")
    ap.add_argument("--external", default=None,
                    help="train on an EXTERNAL corpus domain instead of the "
                         "disjoint CBIS-DDSM pool, e.g. --external cmmd_breast. "
                         "No eval-group exclusion is needed or possible there: "
                         "the corpora share no patients at all.")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    from sklearn.metrics import roc_auc_score

    archs = ([a.strip() for a in args.archs.split(",") if a.strip()]
             if args.archs else list(config.BREAST_VALIDATOR_ARCHS))
    k = args.folds or int(config.BREAST_VALIDATOR_FOLDS)
    index_json = args.index_json or os.path.join(config.CKPT_DIR,
                                                 "eval_group_index.json")
    out_path = args.out or os.path.join(
        config.BREAST_CKPT_DIR,
        f"validator_report_{EXTERNAL_TAGS.get(args.external, VALIDATOR_TAG)}.json")

    if args.external:
        # External corpus: the whole thing is the pool. There is nothing to
        # exclude, because no CBIS-DDSM patient can appear in it — which is the
        # entire reason for training here rather than on a CBIS subset.
        tag = EXTERNAL_TAGS.get(args.external)
        if tag is None:
            raise SystemExit(
                f"no corpus tag registered for domain {args.external!r}. Add "
                f"one to EXTERNAL_TAGS in {__file__} so its checkpoints get a "
                f"stable, greppable name."
            )
        profile = domains.get(args.external)
        print(profile.summary())
        set_seed(args.seed)
        dataset = profile.build_dataset(config.IMG_SIZE)
        pool = list(range(len(dataset)))
        pool_pat = {dataset.patient_id(i) for i in pool}
        print(f"\n[validator] EXTERNAL corpus {args.external}: "
              f"{len(pool_pat)} patients / {len(pool)} images, all usable")
        print(f"[validator] independence is by construction here — different "
              f"corpus, no shared patients with CBIS-DDSM")
        print(f"[validator] the binding question is instead whether it can read "
              f"a CBIS-DDSM image at all; that is the AUC gate in "
              f"breast_validator_gate.py, not this script")
    else:
        tag = VALIDATOR_TAG
        profile = domains.active()
        if profile.build_guidance is None:
            raise SystemExit(f"{profile.name} is an xrv domain — nothing to train. "
                             f"Set PPO_DOMAIN=cbis_ddsm_breast.")
        print(profile.summary())

        # The disjoint pool is DERIVED, never hand-listed: it comes from the audit
        # that proved which dataset rows the shards actually contain. Recomputing it
        # here from a different source is how the two would drift apart.
        if not os.path.exists(index_json):
            raise SystemExit(
                f"{index_json} not found. Run breast_shard_audit.py first — it is "
                f"what establishes which patients the eval groups came from, and "
                f"without it 'disjoint' is an assertion rather than a fact."
            )
        audit = json.load(open(index_json))
        pool = audit["gate_c"]["disjoint_pool_indices"]
        print(f"\n[validator] disjoint pool from {index_json}: "
              f"{audit['gate_c']['disjoint_pool_patients']} patients / {len(pool)} images "
              f"({audit['gate_c']['disjoint_pool_malignant']} malignant / "
              f"{audit['gate_c']['disjoint_pool_benign']} benign)")
        print(f"[validator] excluded: {audit['gate_c']['guidance_train_patients']} guidance-train "
              f"patients and {audit['gate_c']['eval_patients']} eval-group patients")

        set_seed(args.seed)
        dataset = profile.build_dataset(config.IMG_SIZE)

        # Sanity: the pool must not intersect what it claims to exclude.
        eval_idx = set(audit["xn_idx"] + audit["xp_idx"])
        eval_pat = {dataset.patient_id(i) for i in eval_idx}
        pool_pat = {dataset.patient_id(i) for i in pool}
        assert not (set(pool) & eval_idx), "pool contains an eval image"
        assert not (pool_pat & eval_pat), "pool contains an eval patient"
        print(f"[validator] disjointness asserted: 0 shared images, 0 shared patients "
              f"with the eval groups")

    folds = patient_folds(dataset, pool, k, args.seed)
    print(f"[validator] {k} patient-level folds, sizes {[len(f) for f in folds]}")

    y_true = np.array([float(dataset.samples[i][-1]) for i in pool])
    pos = {i: n for n, i in enumerate(pool)}

    report = {"corpus": args.external or profile.name, "tag": tag,
              "pool_images": len(pool), "pool_patients": len(pool_pat),
              "folds": k, "seed": args.seed, "archs": {}}

    for arch in archs:
        print("\n" + "#" * 70)
        pool_desc = (f"external corpus {args.external}" if args.external
                     else "the disjoint CBIS-DDSM pool")
        print(f"# {arch} — {k}-fold patient-level CV on {pool_desc}")
        print("#" * 70)
        oof = np.full(len(pool), np.nan)
        fold_aucs, members = [], []

        for f in range(k):
            # Outer fold: predicted once, never trained on, never selected on.
            val_idx = folds[f]
            # Inner fold: early stopping and checkpoint selection happen here.
            inner = (f + 1) % k
            sel_idx = folds[inner]
            train_idx = [i for g in range(k) if g not in (f, inner)
                         for i in folds[g]]
            w = fold_weights(arch, f, tag)
            ckpt = checkpoint_path(w)

            if args.skip_existing and os.path.exists(ckpt):
                print(f"\n[fold {f}] {ckpt} exists — skipping training")
            else:
                print(f"\n[fold {f}] train {len(train_idx)} / select-on "
                      f"fold {inner} ({len(sel_idx)}) / predict {len(val_idx)}")
                train_one(arch=arch, dataset=dataset, train_idx=train_idx,
                          val_idx=sel_idx, epochs=args.epochs,
                          batch_size=args.batch_size, lr=args.lr,
                          device=config.DEVICE, patience=args.patience,
                          ckpt_path=ckpt)

            p = predict(arch, ckpt, dataset, val_idx, config.DEVICE)
            for n, i in enumerate(val_idx):
                oof[pos[i]] = p[n]
            yf = y_true[[pos[i] for i in val_idx]]
            a = float(roc_auc_score(yf, p)) if len(np.unique(yf)) > 1 else float("nan")
            fold_aucs.append(a)
            members.append(w)
            print(f"[fold {f}] out-of-fold AUC = {a:.4f}  "
                  f"(epoch chosen on fold {inner}, not on this one)")

        assert not np.isnan(oof).any(), "some pool image never got an OOF prediction"
        oof_auc = float(roc_auc_score(y_true, oof))
        print(f"\n[{arch}] POOLED OUT-OF-FOLD AUC = {oof_auc:.4f}   "
              f"(per-fold {['%.3f' % a for a in fold_aucs]})")
        print(f"[{arch}] floor {MIN_ACCEPTABLE_AUC}: "
              f"{'PASS' if oof_auc >= MIN_ACCEPTABLE_AUC else 'FAIL'}")

        report["archs"][arch] = {
            "weights": validator_weights(arch, tag),
            "members": members,
            "oof_auc": oof_auc,
            "fold_aucs": fold_aucs,
            "passes_floor": bool(oof_auc >= MIN_ACCEPTABLE_AUC),
            "min_acceptable_auc": MIN_ACCEPTABLE_AUC,
        }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {out_path}")

    print("\n" + "=" * 70)
    print("VALIDATOR SUMMARY — out-of-fold AUC on patients no fold trained on")
    print("=" * 70)
    for arch, r in report["archs"].items():
        print(f"  {validator_weights(arch, tag):34s} {r['oof_auc']:.4f}  "
              f"{'PASS' if r['passes_floor'] else 'FAIL (below floor)'}")
    print("\n  A validator below the floor cannot distinguish malignant from benign")
    print("  on real images, so a low transfer ratio from it says nothing about")
    print("  the counterfactuals. Report it as UNINTERPRETABLE, not ADVERSARIAL.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
