"""
domains/node21_nodule.py
========================
Lung-nodule domain: NODE21 chest radiographs, f(.) = "Nodule".

WHY THIS DOMAIN
---------------
Minimal-change port from rsna_pneumonia: same modality, same grayscale
preprocessing, same 224 xrv convention, so the PSPNet lung segmenter and the
whole background-lock / lung-mask machinery carry over untouched. Only the
images, the pathology head and the clinical metrics change.

WEIGHT SETS — READ BEFORE EDITING
---------------------------------
"Nodule" is DEAD (NaN op_thresh -> constant 0.5, zero gradient) in both rsna
and chex, the two models the pneumonia domain guides with. Measured from
torchxrayvision.models.model_urls:

    weights   Nodule    Mass      Lung Lesion
    -all      0.0240    0.0194    0.0534
    -nih      0.0542    0.0372    NaN
    -pc       0.0376    0.0057    NaN
    -chex     NaN       NaN       0.0993
    -rsna     NaN       NaN       NaN

So the guidance ensemble here is (all, pc) and the only remaining live-Nodule
validator is nih. domains.base.DomainProfile.validate enforces this — it will
raise rather than let a constant f(.) into a training run.

CONTAMINATION CAVEAT
--------------------
NODE21 draws images from ChestX-ray14, and densenet121-res224-all was trained
on NIH (which is ChestX-ray14). The guidance classifier has therefore very
likely seen these images, and -nih is inside -all's training mix, so the
held-out validator is weaker here than it was for the pneumonia domain. This
does not break the pipeline but it does weaken the anti-circularity argument —
state it explicitly in any writeup rather than reporting CV as if the split
were clean.

DATA
----
Download: https://node21.grand-challenge.org/Data/ (also Zenodo 5548363).
4,882 frontal CXRs — 1,134 with nodules (1,476 boxes), 3,748 nodule-free.

Set config.NODE21_ROOT to the extracted directory. Expected layout:

    <root>/images/*.mha        (or .png / .jpg / .dcm — dispatched on suffix)
    <root>/metadata.csv

SCHEMA IS NOT VERIFIED HERE. The column names below are the documented ones,
but confirm them against your actual download on first run — the loader raises
with the columns it actually found, so a mismatch is a one-line fix to the
constants directly below rather than a hunt through this file.
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


# ── CSV schema. Fix these two if the download disagrees. ──────────────────────
COL_IMAGE = "img_name"   # filename (with or without extension)
COL_LABEL = "label"      # 1 = nodule present, 0 = absent
# If the CSV carries only bounding boxes and no label column, the loader falls
# back to "positive iff this image has at least one box row", which is exactly
# how ingestion.RSNACXRDataset collapses the RSNA per-box rows.

IMAGE_SUBDIR = "images"
METADATA_CSV = "metadata.csv"


def _load_image_as_pil(path: str) -> Image.Image:
    """
    Read one radiograph and return an 8-bit grayscale PIL image.

    Mirrors ingestion.RSNACXRDataset._load_dicom_as_pil's normalisation
    (per-image min/max to [0,1], then 8-bit) so the two domains feed the
    generator pixel distributions produced the same way.
    """
    suffix = os.path.splitext(path)[1].lower()

    if suffix in (".mha", ".mhd", ".nii", ".gz"):
        try:
            import SimpleITK as sitk
        except ImportError as e:
            raise ImportError(
                "NODE21 ships MetaImage (.mha); reading it needs SimpleITK, "
                "which is not in this venv. Either `pip install SimpleITK` and "
                "add it to requirements.txt, or convert the release to PNG once "
                "and point config.NODE21_ROOT at that."
            ) from e
        arr = sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(np.float32)
        arr = arr[0] if arr.ndim == 3 else arr
    elif suffix == ".dcm":
        import pydicom
        dcm = pydicom.dcmread(path)
        arr = dcm.pixel_array.astype(np.float32)
        arr = arr * float(getattr(dcm, "RescaleSlope", 1)) + float(
            getattr(dcm, "RescaleIntercept", 0)
        )
    else:
        return Image.open(path).convert("L")

    lo, hi = float(arr.min()), float(arr.max())
    arr = (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)
    return Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8), mode="L")


class NODE21Dataset(Dataset):
    """
    NODE21 nodule / no-nodule chest radiographs.

    Yields (x, y) per the domains.base contract:
        x : (1, img_size, img_size) float32 in [0, 1]
        y : (1,) float32 — 1.0 nodule present, 0.0 absent
    """

    def __init__(self, root: str, img_size: int = 256):
        images_dir = os.path.join(root, IMAGE_SUBDIR)
        labels_csv = os.path.join(root, METADATA_CSV)
        for p in (images_dir, labels_csv):
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"{p} not found. Set config.NODE21_ROOT to the extracted "
                    f"NODE21 release; see this module's docstring for the "
                    f"expected layout."
                )

        self.images_dir = images_dir
        self.img_size   = img_size
        self.transform  = T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])

        df = pd.read_csv(labels_csv)
        if COL_IMAGE not in df.columns:
            raise ValueError(
                f"Expected an image-name column {COL_IMAGE!r} in {labels_csv}; "
                f"found {sorted(df.columns)}. Fix COL_IMAGE at the top of "
                f"{__name__}."
            )

        on_disk = {f: os.path.splitext(f)[0] for f in os.listdir(images_dir)}
        stem_to_file = {stem: f for f, stem in on_disk.items()}

        def _resolve(name) -> str | None:
            """CSV name -> filename on disk, tolerating a missing extension."""
            name = str(name)
            if name in on_disk:
                return name
            return stem_to_file.get(os.path.splitext(name)[0])

        self.samples: list[tuple[str, float]] = []

        if COL_LABEL in df.columns:
            # CSV is authoritative: it lists every image with an explicit label.
            for name, lbl in df.groupby(COL_IMAGE)[COL_LABEL].max().items():
                fname = _resolve(name)
                if fname is not None:
                    self.samples.append((fname, float(lbl)))
        else:
            # Bounding-box-only CSV. Rows exist ONLY for annotated images, so
            # the CSV cannot be the source of truth for which images exist —
            # iterating it would silently drop every negative. On NODE21 that
            # is 3748 of 4882 images, leaving an all-positive dataset that the
            # far-pool scan and the Xn/Xp split cannot work with, after the run
            # has already cost hours. Enumerate the DIRECTORY instead and treat
            # "has at least one box row" as the positive label.
            print(
                f"[NODE21Dataset] no {COL_LABEL!r} column in {labels_csv}; "
                f"deriving the binary label as 'has >=1 annotation row', "
                f"enumerating {len(on_disk)} files on disk so nodule-free "
                f"images are kept as negatives"
            )
            annotated = {
                f for f in (_resolve(n) for n in df[COL_IMAGE].unique())
                if f is not None
            }
            for fname in sorted(on_disk):
                self.samples.append((fname, 1.0 if fname in annotated else 0.0))

        if not self.samples:
            raise RuntimeError(
                f"No CSV rows matched files in {images_dir}. Check that "
                f"{COL_IMAGE!r} values correspond to filenames."
            )

        # Both classes are structurally required: the env builds a "far pool"
        # per direction and metrics splits into Xn / Xp. A single-class dataset
        # produces an empty pool and fails deep inside a GPU job instead of here.
        n_pos = sum(1 for _, l in self.samples if l == 1.0)
        n_neg = len(self.samples) - n_pos
        if n_pos == 0 or n_neg == 0:
            raise RuntimeError(
                f"{images_dir} yielded a SINGLE-CLASS dataset "
                f"({n_pos} positive / {n_neg} negative). Both directions need a "
                f"non-empty far pool, so this cannot train. Check COL_IMAGE / "
                f"COL_LABEL at the top of {__name__} against {labels_csv}."
            )

        print(f"[NODE21Dataset] {len(self.samples)} images loaded from {images_dir}")
        print(f"  Nodule: {n_pos}  |  No nodule: {n_neg}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        fname, label = self.samples[idx]
        img = _load_image_as_pil(os.path.join(self.images_dir, fname))
        return self.transform(img), torch.tensor([label], dtype=torch.float32)


def build_dataset(img_size: int = 256) -> NODE21Dataset:
    """Factory matching the domains.base contract."""
    root = getattr(config, "NODE21_ROOT", None)
    if not root:
        raise ValueError(
            "config.NODE21_ROOT is unset. NODE21 has no kagglehub-style "
            "auto-download (it is behind grand-challenge registration), so "
            "download it once and point NODE21_ROOT at the extracted directory."
        )
    return NODE21Dataset(root=root, img_size=img_size)


PROFILE = DomainProfile(
    name="node21_nodule",
    build_dataset=build_dataset,
    pathology="Nodule",
    guidance_weights="densenet121-res224-all",
    ensemble_weights=(
        "densenet121-res224-all",
        "densenet121-res224-pc",
    ),
    heldout_validators=("densenet121-res224-nih",),
    # CTR is cardiomegaly, SCP is pleural effusion — neither says anything
    # about a nodule, so both are OFF rather than reported as noise. The
    # replacement worth building is lesion extent inside NODE21's annotated
    # boxes, which is a direct measure the pneumonia domain never had.
    clinical_metrics=(),
    notes=(
        "Nodule is dead in rsna/chex — guidance must be (all, pc). NODE21 "
        "overlaps ChestX-ray14 and -all trained on NIH, so the held-out "
        "validator is weaker here; say so when reporting CV."
    ),
)
