"""
domains/cbis_ddsm_breast.py
===========================
Mammography domain: CBIS-DDSM, f(.) = P(MALIGNANT).

WHY THIS DOMAIN IS NOT LIKE THE OTHER TWO
-----------------------------------------
node21_nodule was a minimal-change port of rsna_pneumonia: same modality, same
xrv DenseNet zoo, same PSPNet lung segmenter. Only the images, the pathology
head and the clinical metrics moved.

Almost none of that survives here. Mammography is a different modality, and
torchxrayvision — which supplies f(.), the segmenter, the FID feature space and
the FOP detector for both chest domains — has no breast model of any kind.
`default_pathologies` has 18 entries and not one is a breast finding. So this
domain brings its own:

    f(.)          src/breast_classifier.py     (trained by train_breast_classifier.py)
    segmenter S   src/breast_segmentation.py   (BreastAnatomySegmenter, analytic)
    lesion seg    src/breast_segmentation.py   (trained by train_lesion_segmenter.py)
    FID space     the HELD-OUT breast backbone (src/metrics.py, "breast")
    metrics       LLR / MSI / BDC / MOP        (src/metrics.py)

They plug in through the build_* fields on DomainProfile; see the NON-CHEST
DOMAINS section of domains/base.py.

THE LABEL DOES NOT MEAN WHAT IT MEANT BEFORE — READ THIS
--------------------------------------------------------
CBIS-DDSM is a curated subset of lesion cases. Every image in it contains an
abnormality; there are NO normal mammograms in the dataset. So the binary label
cannot be disease-vs-healthy, and here it is

        y = 1.0  MALIGNANT
        y = 0.0  BENIGN  or  BENIGN_WITHOUT_CALLBACK

which makes the two PPO directions "benign -> malignant" and
"malignant -> benign", NOT "healthy -> diseased". Every CV number, every diff
plot and every sentence in the writeup has to say so. A reader who assumes the
chest framing will read "counterfactual flipped f(.) to negative" as "the model
removed the cancer" when it actually means "the model made the lesion look
benign", which is a different and much weaker claim.

Class balance is workable — roughly 1.4k malignant against 2.1k benign lesions
— which is what the env's far-pool scan and the metrics Xn/Xp split need. That
is a genuine advantage over the RSNA screening mammography competition set,
where positives are ~2%.

CONTAMINATION CAVEAT
--------------------
Unlike node21_nodule, there is none to declare from pretraining: f(.) is
trained by us, from ImageNet init, on a patient-level split of this dataset.
The weaker link here is different — the two guidance members and the held-out
validator all come from the SAME ~3.5k images and differ only in architecture,
so they are far more correlated than the chest ensemble's separate-corpus
members. The held-out transfer ratio heldout_validation.py reports is therefore
an optimistic bound. Say so rather than quoting it as if it matched the chest
setup.

DATA
----
Kaggle mirror (public dataset, no competition rules to accept):
    awsaf49/cbis-ddsm-breast-cancer-image-dataset      (~6 GB, JPEG)

setup_env.sh downloads it on the login node. Point config.CBIS_DDSM_ROOT at the
extracted directory. Expected layout:

    <root>/csv/dicom_info.csv
    <root>/csv/{mass,calc}_case_description_{train,test}_set.csv
    <root>/jpeg/<SeriesInstanceUID>/<n>.jpg

SCHEMA IS NOT VERIFIED HERE. The column names below are the documented ones,
but confirm them against your actual download on first run — every loader path
raises with the columns it ACTUALLY found, so a mirror-layout change is a
one-line fix to the constants directly below rather than a hunt through this
file. Same convention as domains/node21_nodule.py.
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


# ── CSV schema. Fix these if the download disagrees. ─────────────────────────
CSV_SUBDIR   = "csv"
DICOM_INFO   = "dicom_info.csv"
CASE_CSVS    = (
    "mass_case_description_train_set.csv",
    "mass_case_description_test_set.csv",
    "calc_case_description_train_set.csv",
    "calc_case_description_test_set.csv",
)

COL_PATIENT   = "patient_id"
COL_PATHOLOGY = "pathology"
COL_IMAGE     = "image file path"
COL_ROI       = "ROI mask file path"
COL_MARGINS   = "mass margins"
COL_SHAPE     = "mass shape"
COL_VIEW      = "image view"
COL_SIDE      = "left or right breast"

# dicom_info.csv rows are typed by this column; these are the two series we use.
COL_SERIES_DESC = "SeriesDescription"
SERIES_FULL     = "full mammogram images"
SERIES_ROI      = "ROI mask images"
COL_IMAGE_PATH  = "image_path"

# ── Label mapping ────────────────────────────────────────────────────────────
# BENIGN_WITHOUT_CALLBACK is a real clinical category, not a parsing artefact:
# a finding a radiologist judged benign confidently enough NOT to recall the
# patient. It is collapsed into the benign class here because this pipeline
# needs a BINARY label and it is unambiguously the non-malignant side.
#
# Named rather than inlined so the decision is greppable — if a later
# experiment wants three classes, or wants to drop these rows as ambiguous
# rather than merge them, this constant is the one place to change.
MALIGNANT_LABELS = ("MALIGNANT",)
BENIGN_LABELS    = ("BENIGN", "BENIGN_WITHOUT_CALLBACK")


def _csv_dir(root: str) -> str:
    """
    The csv/ subdirectory, tolerating a mirror that puts the CSVs at the root.

    Cheap to support and it removes the single most likely reason a fresh
    download fails on line one.
    """
    nested = os.path.join(root, CSV_SUBDIR)
    return nested if os.path.isdir(nested) else root


def _load_case_table(root: str) -> pd.DataFrame:
    """
    Concatenate the four case-description CSVs into one table.

    Mass and calcification cases are pooled deliberately. Both are lesions with
    a pathology label, both are what the classifier must call malignant or not,
    and splitting them would halve an already-small dataset. The `abnormality
    type` distinction is retained in the frame for anyone who wants to
    stratify later.
    """
    cdir  = _csv_dir(root)
    frames, missing = [], []
    for name in CASE_CSVS:
        path = os.path.join(cdir, name)
        if os.path.exists(path):
            frames.append(pd.read_csv(path))
        else:
            missing.append(name)

    if not frames:
        raise FileNotFoundError(
            f"none of the CBIS-DDSM case-description CSVs were found in "
            f"{cdir!r}. Looked for {list(CASE_CSVS)}. Set config.CBIS_DDSM_ROOT "
            f"to the extracted Kaggle mirror — see this module's docstring for "
            f"the expected layout."
        )
    if missing:
        # Not fatal: a partial mirror still trains, just on less data. But it
        # silently halves the dataset, so it must be visible in the log.
        print(f"[CBISDDSM] WARNING: missing case CSVs {missing} — continuing "
              f"with {len(frames)} of {len(CASE_CSVS)}")

    df = pd.concat(frames, ignore_index=True)
    df.columns = [c.strip() for c in df.columns]

    for col in (COL_PATIENT, COL_PATHOLOGY, COL_IMAGE):
        if col not in df.columns:
            raise ValueError(
                f"expected column {col!r} in the CBIS-DDSM case CSVs; found "
                f"{sorted(df.columns)}. Fix the COL_* constants at the top of "
                f"{__name__}."
            )
    return df


def _build_uid_index(root: str, series: str) -> dict[str, str]:
    """
    SeriesInstanceUID -> absolute path of one JPEG in that series.

    WHY THIS INDIRECTION EXISTS. The case CSVs' `image file path` values are
    the ORIGINAL TCIA DICOM paths, e.g.

        Mass-Training_P_00001_LEFT_CC/1.3.6.1.4.1.../1.3.6.1.4.1.../000000.dcm

    Nothing at that path exists in the Kaggle mirror, which stores JPEGs under
    jpeg/<SeriesInstanceUID>/. The only thing the two share is the UID, so
    dicom_info.csv is the join table. Matching on the UID substring is what
    makes the CSVs usable at all.
    """
    cdir = _csv_dir(root)
    path = os.path.join(cdir, DICOM_INFO)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. It is the join table between the case CSVs' "
            f"original TCIA paths and the mirror's jpeg/<UID>/ layout, so the "
            f"dataset cannot be assembled without it."
        )

    info = pd.read_csv(path, low_memory=False)
    info.columns = [c.strip() for c in info.columns]
    for col in (COL_SERIES_DESC, COL_IMAGE_PATH):
        if col not in info.columns:
            raise ValueError(
                f"expected column {col!r} in {path}; found {sorted(info.columns)}. "
                f"Fix the COL_* constants at the top of {__name__}."
            )

    sel = info[info[COL_SERIES_DESC].astype(str).str.strip() == series]
    if sel.empty:
        avail = sorted(info[COL_SERIES_DESC].dropna().astype(str).unique())
        raise ValueError(
            f"no rows in {path} with {COL_SERIES_DESC}=={series!r}. "
            f"Available: {avail}"
        )

    out: dict[str, str] = {}
    for rel in sel[COL_IMAGE_PATH].astype(str):
        # Mirror paths look like "CBIS-DDSM/jpeg/<UID>/<n>.jpg", sometimes with
        # a leading component that does not exist locally. Rebuild from the
        # "jpeg" marker so the row survives either way.
        parts = rel.replace("\\", "/").split("/")
        if "jpeg" not in parts:
            continue
        j = parts.index("jpeg")
        uid = parts[j + 1] if len(parts) > j + 1 else None
        if not uid:
            continue
        abs_path = os.path.join(root, *parts[j:])
        if uid not in out and os.path.exists(abs_path):
            out[uid] = abs_path
    return out


def _resolve(tcia_path: str, uid_index: dict[str, str]) -> str | None:
    """Original TCIA path -> local JPEG, by finding a UID component in common."""
    for part in str(tcia_path).replace("\\", "/").split("/"):
        hit = uid_index.get(part.strip())
        if hit is not None:
            return hit
    return None


def _load_image_as_pil(path: str) -> Image.Image:
    """
    Read one mammogram and return an 8-bit grayscale PIL image.

    Mirrors NODE21Dataset's normalisation (per-image min/max to [0,1], then
    8-bit) so every domain hands the generator pixel distributions produced the
    same way. That matters more than it looks: the LoRA fine-tuning learns the
    intensity statistics it is shown, so two domains normalising differently
    would not be comparable even at identical settings.
    """
    img = Image.open(path).convert("L")
    arr = np.asarray(img).astype(np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    arr = (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)
    return Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8), mode="L")


class CBISDDSMDataset(Dataset):
    """
    CBIS-DDSM full mammograms, malignant vs benign.

    Yields (x, y) per the domains.base contract:
        x : (1, img_size, img_size) float32 in [0, 1]
        y : (1,) float32 — 1.0 MALIGNANT, 0.0 BENIGN

    FULL MAMMOGRAMS, not the cropped lesion patches the mirror also ships. The
    crops would make the classification task easier and the edits far more
    visible, but they would also delete the thing this pipeline is built
    around: with no background and no anatomy, LOCK_BACKGROUND has nothing to
    lock, the breast mask is all-ones, and "did the edit stay inside the organ
    and land on the lesion" stops being answerable. The crops remain the
    obvious ablation, which is why _view is a constructor argument.

    Also exposes roi_mask_path(i), margin_label(i) and view_label(i). None are
    used in training or in the metric phase — see the note in src/metrics.py
    about why ground truth cannot be joined at metric time — but
    train_lesion_segmenter.py and check_lesion_localization.py index this
    dataset directly and need them.
    """

    def __init__(self, root: str, img_size: int = 256, view: str = "full"):
        if not os.path.isdir(root):
            raise FileNotFoundError(
                f"config.CBIS_DDSM_ROOT={root!r} is not a directory. Download "
                f"the dataset on the LOGIN NODE (compute nodes have no "
                f"internet) with `bash setup_env.sh`, then set the path it "
                f"prints."
            )

        self.root      = root
        self.img_size  = img_size
        self.view      = view
        self.transform = T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])

        df = _load_case_table(root)
        series = SERIES_FULL if view == "full" else "cropped images"
        uid_index = _build_uid_index(root, series)
        roi_index = _build_uid_index(root, SERIES_ROI)

        df["_path"] = df[COL_IMAGE].map(lambda p: _resolve(p, uid_index))
        df["_roi"]  = (
            df[COL_ROI].map(lambda p: _resolve(p, roi_index))
            if COL_ROI in df.columns else None
        )

        unresolved = int(df["_path"].isna().sum())
        df = df[df["_path"].notna()].copy()
        if unresolved:
            print(f"[CBISDDSM] {unresolved} case rows did not resolve to a JPEG "
                  f"and were dropped (mirror is missing those series)")

        # Label. Anything outside the two known vocabularies is a schema
        # surprise, so name it rather than silently coercing it to benign.
        path_norm = df[COL_PATHOLOGY].astype(str).str.strip().str.upper()
        unknown = sorted(set(path_norm) - set(MALIGNANT_LABELS) - set(BENIGN_LABELS))
        if unknown:
            raise ValueError(
                f"unexpected {COL_PATHOLOGY} values {unknown} in the CBIS-DDSM "
                f"CSVs. Known: {list(MALIGNANT_LABELS + BENIGN_LABELS)}. Extend "
                f"MALIGNANT_LABELS / BENIGN_LABELS at the top of {__name__} "
                f"once you have decided which side they belong on."
            )
        df["_label"] = path_norm.isin(MALIGNANT_LABELS).astype(float)

        # One row per IMAGE. A mammogram can carry several annotated lesions,
        # and the image is malignant if ANY of them is — the same collapse
        # ingestion.RSNACXRDataset does over RSNA's per-box rows.
        grouped = df.groupby("_path", sort=True)
        self.samples: list[tuple[str, float]] = []
        self._meta: list[dict] = []
        for path, g in grouped:
            self.samples.append((path, float(g["_label"].max())))
            self._meta.append({
                "patient_id": str(g[COL_PATIENT].iloc[0]),
                "roi":        next((r for r in g["_roi"] if isinstance(r, str)), None)
                              if "_roi" in g else None,
                "margins":    str(g[COL_MARGINS].iloc[0]) if COL_MARGINS in g else "",
                "shape":      str(g[COL_SHAPE].iloc[0]) if COL_SHAPE in g else "",
                "view":       str(g[COL_VIEW].iloc[0]) if COL_VIEW in g else "",
                "side":       str(g[COL_SIDE].iloc[0]) if COL_SIDE in g else "",
            })

        if not self.samples:
            raise RuntimeError(
                f"no CBIS-DDSM rows resolved to files under {root}. Check that "
                f"{DICOM_INFO} lists jpeg/<UID>/ paths that exist on disk."
            )

        # Both classes are structurally required: the env builds a far pool per
        # direction and metrics splits into Xn / Xp. A single-class dataset
        # produces an empty pool and fails deep inside a GPU job instead of
        # here. Copied verbatim in intent from NODE21Dataset.
        n_pos = sum(1 for _, l in self.samples if l == 1.0)
        n_neg = len(self.samples) - n_pos
        if n_pos == 0 or n_neg == 0:
            raise RuntimeError(
                f"{root} yielded a SINGLE-CLASS dataset ({n_pos} malignant / "
                f"{n_neg} benign). Both directions need a non-empty far pool, "
                f"so this cannot train. Check COL_PATHOLOGY and the label "
                f"vocabularies at the top of {__name__}."
            )

        print(f"[CBISDDSMDataset] {len(self.samples)} {view} mammograms from {root}")
        print(f"  MALIGNANT: {n_pos}  |  BENIGN: {n_neg}  "
              f"({100.0 * n_pos / len(self.samples):.1f}% positive)")
        print(f"  NOTE: no normal cases exist in CBIS-DDSM — this is "
              f"malignant-vs-benign, not disease-vs-healthy.")

    # ── side-channel accessors (not part of the domains.base contract) ───────

    def patient_id(self, idx: int) -> str:
        """Used for PATIENT-LEVEL splits — see train_breast_classifier.py."""
        return self._meta[idx]["patient_id"]

    def roi_mask_path(self, idx: int) -> str | None:
        """Ground-truth lesion mask, or None. Training/diagnostics only."""
        return self._meta[idx]["roi"]

    def margin_label(self, idx: int) -> str:
        """CBIS-DDSM `mass margins`, e.g. SPICULATED / CIRCUMSCRIBED.

        The ground truth the MSI metric is calibrated against before it is
        allowed to say anything about counterfactuals.
        """
        return self._meta[idx]["margins"]

    def view_label(self, idx: int) -> str:
        """CC or MLO. Only MLO views contain pectoral muscle."""
        return self._meta[idx]["view"]

    # ── contract ─────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = _load_image_as_pil(path)
        return self.transform(img), torch.tensor([label], dtype=torch.float32)


def build_dataset(img_size: int = 256) -> CBISDDSMDataset:
    """Factory matching the domains.base contract."""
    root = getattr(config, "CBIS_DDSM_ROOT", "") or ""
    if not root:
        raise ValueError(
            "config.CBIS_DDSM_ROOT is unset. Run `bash setup_env.sh` on the "
            "LOGIN NODE to download the Kaggle mirror, then set CBIS_DDSM_ROOT "
            "to the path it prints."
        )
    return CBISDDSMDataset(root=root, img_size=img_size)


# ─────────────────────────────────────────────────────────────────────────────
# Backend builders — the escape hatches from domains/base.py
# ─────────────────────────────────────────────────────────────────────────────
#
# Imports are function-local on purpose. This module is imported whenever the
# registry is enumerated (setup_env.sh, `domains.available()`), and importing
# torch model code at that point would make listing the domains cost several
# seconds and pull CUDA into processes that only wanted a name.

GUIDANCE = "densenet121-breast-cbis"
ENSEMBLE = ("densenet121-breast-cbis", "efficientnet_b0-breast-cbis")
VALIDATORS = ("resnet18-breast-cbis",)


def _build_guidance():
    """
    f(.) — the ensemble or the single model, per config.USE_GUIDANCE_ENSEMBLE.

    Mirrors the branch pipeline.build_common takes for the xrv domains, so the
    same config flag means the same thing in both.
    """
    from src.breast_classifier import (
        build_breast_classifier,
        build_breast_ensemble_classifier,
    )
    if config.USE_GUIDANCE_ENSEMBLE:
        return build_breast_ensemble_classifier(weights_list=list(ENSEMBLE))
    return build_breast_classifier(weights=GUIDANCE)


def _build_validators() -> dict:
    """
    {name: classifier} for heldout_validation.py.

    Disjoint from GUIDANCE/ENSEMBLE by construction, and DomainProfile.validate
    enforces it. See the CONTAMINATION CAVEAT above for why this validator is
    weaker than the chest domains' — same training set, different architecture.
    """
    from src.breast_classifier import build_breast_classifier
    return {w: build_breast_classifier(weights=w) for w in VALIDATORS}


def _build_masker():
    from src.breast_segmentation import build_breast_masker
    return build_breast_masker()


def _build_lesion_net():
    from src.breast_segmentation import build_lesion_segmenter
    return build_lesion_segmenter()


PROFILE = DomainProfile(
    name="cbis_ddsm_breast",
    build_dataset=build_dataset,
    # Advisory only — the non-xrv backend means validate() never resolves this
    # against torchxrayvision.default_pathologies. It is what gets printed and
    # stamped into eval output.
    pathology="Malignancy",
    guidance_weights=GUIDANCE,
    ensemble_weights=ENSEMBLE,
    heldout_validators=VALIDATORS,
    build_guidance=_build_guidance,
    build_validators=_build_validators,
    build_masker=_build_masker,
    build_lesion_net=_build_lesion_net,
    # Chest DenseNet features say nothing about a mammogram. "breast" measures
    # FID in the HELD-OUT backbone's feature space, so realism is not scored in
    # the same space the guidance models optimise. src/metrics.py also reports
    # Inception FID alongside for comparability with the literature.
    fid_backbone="breast",
    # CTR is cardiomegaly and SCP pleural effusion — neither exists on a
    # breast, so both are OFF rather than reported as noise. These four are the
    # mammography set; see src/metrics.py for what each measures and, for MSI,
    # the ground-truth calibration it must pass before being reported.
    clinical_metrics=("LLR", "MSI", "BDC", "MOP"),
    notes=(
        "CBIS-DDSM has NO normal cases: y=1 is MALIGNANT and y=0 is BENIGN, so "
        "the directions are benign->malignant and malignant->benign, not "
        "healthy<->diseased. Guidance members and the held-out validator share "
        "one ~3.5k-image training set and differ only by architecture, so the "
        "transfer ratio is an optimistic bound. State both when reporting CV."
    ),
)
