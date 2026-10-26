"""
domains/cmmd_breast.py
======================
CMMD (Chinese Mammography Database) — a mammography corpus that shares NOTHING
with CBIS-DDSM, used only to validate CBIS-DDSM counterfactuals.

WHY THIS DOMAIN EXISTS, AND WHAT IT IS NOT FOR
----------------------------------------------
Nothing is ever generated on CMMD. There is no PPO run, no generator, no eval
here. This module exists so that a classifier can be trained on a genuinely
different body of data and then pointed at the counterfactuals that
domains/cbis_ddsm_breast.py produced.

That distinction is the whole point of the file. The chest domain's held-out
validators (nih / pc / mimic_ch) are independent because different research
groups trained them on different hospitals' patients. The mammography domain
had no equivalent: its "held-out" resnet18-breast-cbis was trained by this
project, on the same patient split as the guidance model, and — per
breast_shard_audit.py — on 84% of the very images it would be asked to
adjudicate. Even train_breast_validator.py's patient-disjoint validator is
still CBIS-DDSM: same scanner era, same digitisation pipeline, same curation.

CMMD is the only axis of independence left. Different country (China vs the
USA), different decade, different acquisition (full-field digital vs digitised
film), different curators, and no patient overlap that is even possible.

THE DOMAIN SHIFT IS REAL AND IS THE POINT OF THE GATE
------------------------------------------------------
That independence is bought with a distribution shift, and the shift cuts both
ways: a classifier trained here may simply fail to read a digitised-film
mammogram at all. If it does, its transfer ratio on CBIS-DDSM counterfactuals
measures its own blindness rather than anything about the edits.

So this validator is gated on AUC over REAL CBIS-DDSM images before any
counterfactual number computed from it may be interpreted, and a failure is
reported as UNINTERPRETABLE rather than as ADVERSARIAL. See
train_breast_validator.py --external and the Gate D discussion there.

LABELS
------
description.csv ships one row per image with a per-breast Classification of
Benign or Malignant, which is the same malignant-vs-benign-lesion question
CBIS-DDSM's pathology column asks — not the cancer-vs-normal screening question
a screening cohort would pose. Every one of the 1775 patients is single-labelled
across all of their images, so a patient-level split is unambiguous.

Balance is worse than CBIS-DDSM's: 4094 malignant against 1108 benign (78.7%
positive, vs 44.7%). train_breast_validator.py's pos_weight handles it, and the
AUC gate is threshold-free, but it is worth knowing before reading any accuracy
figure from here.

PREPROCESSING CAVEAT
--------------------
This Kaggle mirror ships breast-cropped PNGs (description.csv Method_crop is
"Yolo" or "Contour") at 1280px height with variable width, whereas
CBISDDSMDataset feeds FULL mammograms including their black surround. Both end
up anisotropically squashed to img_size x img_size by the same Resize, and the
intensity normalisation below is byte-for-byte the one CBISDDSMDataset uses, so
the remaining difference is framing rather than pixel statistics. It is still a
difference, and it is part of what the AUC gate is measuring.

DATA
----
Kaggle mirror (public):  nguynththanhho/cmmd-mammography   (~1.5 GB, PNG)
Point config.CMMD_ROOT at the extracted directory. Expected layout:

    <root>/CMMD/description.csv
    <root>/CMMD/Cancer/<PatientID>_<n>-<m>.png     (the folder name is the
                                                    mirror's, not a label —
                                                    labels live in the CSV)
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

import config

from .base import DomainProfile


SUBDIR       = "CMMD"
DESCRIPTION  = "description.csv"

COL_PATIENT  = "ID"
COL_LABEL    = "Classification"
COL_PATH     = "Path"
COL_SIDE     = "LeftRight"
COL_ABNORM   = "Abnormality"

MALIGNANT_LABELS = ("MALIGNANT",)
BENIGN_LABELS    = ("BENIGN",)


def _root() -> str:
    root = getattr(config, "CMMD_ROOT", "") or ""
    if not root:
        raise ValueError(
            "config.CMMD_ROOT is unset. Download the mirror on the LOGIN NODE "
            "(compute nodes have no internet) with\n"
            "    python -c \"import kagglehub; "
            "print(kagglehub.dataset_download('nguynththanhho/cmmd-mammography'))\"\n"
            "and set CMMD_ROOT to the path it prints."
        )
    return root


def _load_image_as_pil(path: str) -> Image.Image:
    """
    Read one mammogram and return an 8-bit grayscale PIL image.

    Byte-identical to domains/cbis_ddsm_breast._load_image_as_pil. Copied rather
    than imported on purpose: it is the ONLY reason a classifier trained here can
    be pointed at CBIS-DDSM images at all, so it must not be able to drift for
    one domain and not the other, and a shared import across two dataset modules
    hides that dependency instead of stating it.
    """
    img = Image.open(path).convert("L")
    arr = np.asarray(img).astype(np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    arr = (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)
    return Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8), mode="L")


class CMMDDataset(Dataset):
    """
    CMMD mammograms, malignant vs benign.

    Yields (x, y) per the domains.base contract:
        x : (1, img_size, img_size) float32 in [0, 1]
        y : (1,) float32 — 1.0 MALIGNANT, 0.0 BENIGN

    Exposes patient_id(i) so splits.py and train_breast_validator.py can group
    by patient. One row per IMAGE, matching CBISDDSMDataset; unlike CBIS-DDSM
    there is no multi-lesion collapse to do, because the CSV is already
    one-row-per-image and every patient carries a single classification.
    """

    def __init__(self, root: str, img_size: int = 256):
        base = os.path.join(root, SUBDIR)
        if not os.path.isdir(base):
            base = root
        csv = os.path.join(base, DESCRIPTION)
        if not os.path.exists(csv):
            raise FileNotFoundError(
                f"{csv} not found. config.CMMD_ROOT={root!r} should be the "
                f"extracted kagglehub mirror — see this module's docstring for "
                f"the expected layout."
            )

        self.root      = base
        self.img_size  = img_size
        self.transform = T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])

        df = pd.read_csv(csv)
        df.columns = [c.strip() for c in df.columns]
        for col in (COL_PATIENT, COL_LABEL, COL_PATH):
            if col not in df.columns:
                raise ValueError(
                    f"expected column {col!r} in {csv}; found {sorted(df.columns)}. "
                    f"Fix the COL_* constants at the top of {__name__}."
                )

        lab = df[COL_LABEL].astype(str).str.strip().str.upper()
        unknown = sorted(set(lab) - set(MALIGNANT_LABELS) - set(BENIGN_LABELS))
        if unknown:
            raise ValueError(
                f"unexpected {COL_LABEL} values {unknown} in {csv}. Known: "
                f"{list(MALIGNANT_LABELS + BENIGN_LABELS)}. Extend the label "
                f"vocabularies at the top of {__name__} once you have decided "
                f"which side they belong on."
            )
        df["_label"] = lab.isin(MALIGNANT_LABELS).astype(float)
        df["_path"]  = df[COL_PATH].map(lambda p: os.path.join(base, str(p)))

        missing = int((~df["_path"].map(os.path.exists)).sum())
        if missing:
            print(f"[CMMD] {missing} rows in {DESCRIPTION} point at files that "
                  f"do not exist and were dropped")
            df = df[df["_path"].map(os.path.exists)].copy()

        # Sorted by path so dataset index order is a stable function of the
        # filenames rather than of pandas row order — the same determinism
        # contract CBISDDSMDataset's groupby(sort=True) provides, and the reason
        # a patient-level fold assignment is reproducible across processes.
        df = df.sort_values("_path", kind="stable").reset_index(drop=True)

        self.samples: list[tuple[str, float]] = list(
            zip(df["_path"].tolist(), df["_label"].astype(float).tolist())
        )
        self._meta = [
            {"patient_id": str(r[COL_PATIENT]).strip(),
             "side": str(r.get(COL_SIDE, "")).strip(),
             "abnormality": str(r.get(COL_ABNORM, "")).strip()}
            for _, r in df.iterrows()
        ]

        if not self.samples:
            raise RuntimeError(f"no CMMD rows resolved to files under {base}")

        n_pos = sum(1 for _, l in self.samples if l == 1.0)
        n_neg = len(self.samples) - n_pos
        if n_pos == 0 or n_neg == 0:
            raise RuntimeError(
                f"{base} yielded a SINGLE-CLASS dataset ({n_pos} malignant / "
                f"{n_neg} benign); a classifier cannot be trained on it."
            )

        n_pat = len({m["patient_id"] for m in self._meta})
        print(f"[CMMDDataset] {len(self.samples)} mammograms / {n_pat} patients "
              f"from {base}")
        print(f"  MALIGNANT: {n_pos}  |  BENIGN: {n_neg}  "
              f"({100.0 * n_pos / len(self.samples):.1f}% positive)")
        print(f"  NOTE: external corpus — used ONLY to validate CBIS-DDSM "
              f"counterfactuals, never to generate them.")

    def patient_id(self, idx: int) -> str:
        return self._meta[idx]["patient_id"]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        return self.transform(_load_image_as_pil(path)), \
            torch.tensor([label], dtype=torch.float32)


def build_dataset(img_size: int = 256) -> CMMDDataset:
    """Factory matching the domains.base contract."""
    return CMMDDataset(root=_root(), img_size=img_size)


VALIDATORS = ("resnet18-breast-cmmd",)


def _build_guidance():
    """
    Never used to guide anything — see the module docstring.

    Present because DomainProfile requires the field and because
    build_guidance being non-None is the flag that tells validate() this is not
    a torchxrayvision domain. Raising here rather than returning a model makes
    the "nothing is generated on CMMD" rule enforced instead of documented.
    """
    raise RuntimeError(
        "cmmd_breast is a validation-only domain: no counterfactuals are "
        "generated on it, so it has no guidance classifier. Train a validator "
        "with `train_breast_validator.py --external cmmd_breast` and score "
        "CBIS-DDSM shards with it."
    )


def _build_validators() -> dict:
    from src.breast_classifier import build_breast_classifier
    return {w: build_breast_classifier(weights=w) for w in VALIDATORS}


PROFILE = DomainProfile(
    name="cmmd_breast",
    build_dataset=build_dataset,
    pathology="Malignancy",
    # Labels only. There is no guidance model for this domain and there must
    # not be one; the names exist so validate()'s anti-circularity check has
    # something to compare the validators against.
    guidance_weights="none-cmmd",
    ensemble_weights=(),
    heldout_validators=VALIDATORS,
    build_guidance=_build_guidance,
    build_validators=_build_validators,
    fid_backbone="inception",
    clinical_metrics=(),
    notes=(
        "VALIDATION-ONLY external corpus. Nothing is generated here. A "
        "classifier trained on CMMD is the only mammography validator in this "
        "project that shares no patients, no scanner and no curation with the "
        "CBIS-DDSM counterfactuals it scores — but it must clear an AUC gate on "
        "real CBIS-DDSM images first, or its transfer ratio measures its own "
        "domain shift rather than the edits."
    ),
)
