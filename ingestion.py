"""
ingestion.py
============
Handles dataset download from Kaggle (RSNA Pneumonia Detection Challenge)
and exposes the RSNACXRDataset class for use by training pipelines.
"""

import math
import os
import numpy as np
import pandas as pd
import pydicom
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


# ─────────────────────────────────────────────
# Download
# ─────────────────────────────────────────────

_COMPETITION = "rsna-pneumonia-detection-challenge"


def _cached_dataset_root() -> str | None:
    """
    The kagglehub cache path for the competition, if it is already complete.

    kagglehub writes a sibling `<dir>.complete` marker once a download has
    finished, so the marker — not merely the directory existing — is what makes
    a cache trustworthy. A half-downloaded directory would otherwise look
    usable and fail thousands of DICOM reads later.
    """
    root = os.path.join(
        os.path.expanduser("~"), ".cache", "kagglehub", "competitions", _COMPETITION
    )
    images_dir, labels_csv = get_dataset_paths(root)
    if (
        os.path.exists(root + ".complete")
        and os.path.isdir(images_dir)
        and os.path.isfile(labels_csv)
    ):
        return root
    return None


def download_rsna_dataset() -> str:
    """
    Return the RSNA Pneumonia Detection dataset root, downloading it only if it
    is not already cached.

    PREFER THE CACHE. This used to call kagglehub unconditionally, which fails
    two ways that both look like a code bug:

      * on a COMPUTE NODE there is no internet, so a run that had everything it
        needed on disk died at ingestion;
      * on the LOGIN node it re-resolves the competition over the network and
        401s unless the Kaggle credentials are still valid AND the competition
        rules are still accepted — which is a live dependency on an external
        service for data that has not changed since 2018.

    The download path is kept for a genuinely cold machine; it just is no
    longer on the critical path of every run.
    """
    cached = _cached_dataset_root()
    if cached is not None:
        print(f"[ingestion] using cached dataset at: {cached}")
        return cached

    import kagglehub

    path = kagglehub.competition_download(_COMPETITION)
    print(f"[ingestion] dataset downloaded to: {path}")
    return path


def get_dataset_paths(root: str) -> tuple[str, str]:
    """
    Derive canonical image-dir and labels-CSV paths from the dataset root.

    Parameters
    ----------
    root : str
        Path returned by ``download_rsna_dataset()``.

    Returns
    -------
    (images_dir, labels_csv) : tuple[str, str]
    """
    images_dir = os.path.join(root, "stage_2_train_images")
    labels_csv = os.path.join(root, "stage_2_train_labels.csv")
    return images_dir, labels_csv


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class RSNACXRDataset(Dataset):
    """
    RSNA Pneumonia Detection Challenge dataset.

    Layout expected on disk
    -----------------------
    images_dir/
        <patientId>.dcm     (flat; ~26 000 files)
    labels_csv              stage_2_train_labels.csv
        columns: patientId, x, y, width, height, Target
                 Target 1 = Lung Opacity (pneumonia); 0 = no pneumonia
                 Multiple rows can exist per patientId (one per bounding box).
                 Collapsed to a single binary label: 1 if ANY row has Target==1.

    Pixel pipeline
    --------------
    DICOM pixel_array  →  float32 [0,1] via min/max normalisation  →
    resize to img_size  →  single-channel (L) tensor (C=1)
    """

    def __init__(self, images_dir: str, labels_csv: str, img_size: int = 256):
        self.images_dir = images_dir
        self.img_size   = img_size
        self.transform  = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),           # → [0,1] float32, shape (1, H, W)
        ])

        # ── build patient-level binary label ──────────────────────────────────
        df = pd.read_csv(labels_csv)

        required = {"patientId", "Target"}
        if not required.issubset(df.columns):
            raise ValueError(
                f"Expected columns {required} in {labels_csv}; "
                f"found {set(df.columns)}"
            )

        label_series = df.groupby("patientId")["Target"].max()

        dcm_ids = {
            os.path.splitext(f)[0]
            for f in os.listdir(images_dir)
            if f.endswith(".dcm")
        }
        valid_patients = label_series.index[label_series.index.isin(dcm_ids)]
        label_series   = label_series.loc[valid_patients]

        self.samples = [
            (pid, float(lbl))
            for pid, lbl in label_series.items()
        ]

        # ── retain the LESION BOXES, not just the collapsed label ──────────────
        # The groupby above reduces each patient to one bit and drops the
        # x/y/width/height columns. Those columns are 9,555 radiologist-drawn
        # opacity boxes — the only ground truth in this project about WHERE the
        # pathology is. Without them nothing can distinguish "the edit produced
        # disease signal" from "the edit produced a perturbation that happens to
        # flip the classifier", which is exactly the failure held-out validation
        # measured. Kept in NATIVE DICOM pixel coordinates; lesion_boxes()
        # rescales to the current img_size on request.
        #
        # Only Target==1 rows carry boxes; Target==0 rows have NaN coordinates.
        self._boxes: dict[str, list[tuple[float, float, float, float]]] = {}
        pos_rows = df[df["Target"] == 1]
        if {"x", "y", "width", "height"}.issubset(pos_rows.columns):
            for pid, grp in pos_rows.groupby("patientId"):
                boxes = [
                    (float(r.x), float(r.y), float(r.width), float(r.height))
                    for r in grp.itertuples()
                    if not (
                        pd.isna(r.x) or pd.isna(r.y)
                        or pd.isna(r.width) or pd.isna(r.height)
                    )
                ]
                if boxes:
                    self._boxes[pid] = boxes

        # patientId -> position in self.samples. Built LAZILY and invalidated
        # against the current length, because smoke_test.py truncates
        # `dataset.samples` in place after construction (deliberately — see its
        # _small_dataset). A map built in __init__ would silently keep the
        # pre-truncation positions and hand out indices into a list that no
        # longer has them.
        self._index_of_cache: dict[str, int] = {}
        self._index_of_len = -1

        # DICOM Rows/Columns per patient, filled lazily. RSNA is uniformly
        # 1024x1024, but reading it rather than assuming it means a box never
        # silently lands in the wrong place if that stops being true.
        self._native_size: dict[str, tuple[int, int]] = {}

        if not self.samples:
            raise RuntimeError(
                f"No matching .dcm files found.\n"
                f"  images_dir : {images_dir}\n"
                f"  labels_csv : {labels_csv}\n"
                f"Check that patientIds in the CSV match filenames in images_dir."
            )

        n_pos = sum(1 for _, lbl in self.samples if lbl == 1.0)
        print(f"[RSNACXRDataset] {len(self.samples)} patients loaded from {images_dir}")
        print(f"  Pneumonia (Target=1): {n_pos}  |  No pneumonia (Target=0): {len(self.samples) - n_pos}")

    # ──────────────────────────────────────────────────────────────────────────

    def _load_dicom_as_pil(self, patient_id: str) -> Image.Image:
        """Read a DICOM file and return a PIL grayscale image."""
        dcm_path = os.path.join(self.images_dir, f"{patient_id}.dcm")
        dcm      = pydicom.dcmread(dcm_path)
        arr      = dcm.pixel_array.astype(np.float32)

        slope     = float(getattr(dcm, "RescaleSlope",     1))
        intercept = float(getattr(dcm, "RescaleIntercept", 0))
        arr = arr * slope + intercept

        arr_min, arr_max = arr.min(), arr.max()
        if arr_max > arr_min:
            arr = (arr - arr_min) / (arr_max - arr_min)
        else:
            arr = np.zeros_like(arr)

        arr_uint8 = (arr * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr_uint8, mode="L")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        patient_id, label = self.samples[idx]
        img = self._load_dicom_as_pil(patient_id)
        x   = self.transform(img)                              # (1, H, W) float32
        return x, torch.tensor([label], dtype=torch.float32)

    # ── annotation access ─────────────────────────────────────────────────────
    # __getitem__ deliberately still returns (x, y) only: every consumer — the
    # far-pool scan, the env, the generator, the metrics group split — assumes
    # that 2-tuple, and widening it would be a breaking change across the whole
    # pipeline for the sake of one metric. Annotations are reached by INDEX
    # instead, which is why metrics.scan_groups now carries indices alongside
    # its tensors.

    def patient_id(self, idx: int) -> str:
        """patientId at a dataset index."""
        return self.samples[idx][0]

    def index_of(self, patient_id: str) -> int:
        """Dataset index of a patientId. Raises KeyError if absent."""
        if self._index_of_len != len(self.samples):
            self._index_of_cache = {pid: i for i, (pid, _) in enumerate(self.samples)}
            self._index_of_len = len(self.samples)
        return self._index_of_cache[patient_id]

    def native_size(self, patient_id: str) -> tuple[int, int]:
        """
        (rows, cols) of the source DICOM, cached.

        Read with stop_before_pixels=True — the header alone is a few KB, so
        this costs nothing next to decoding the image.
        """
        if patient_id not in self._native_size:
            dcm = pydicom.dcmread(
                os.path.join(self.images_dir, f"{patient_id}.dcm"),
                stop_before_pixels=True,
            )
            self._native_size[patient_id] = (int(dcm.Rows), int(dcm.Columns))
        return self._native_size[patient_id]

    def lesion_boxes(self, idx: int) -> list[tuple[float, float, float, float]]:
        """
        Radiologist-drawn opacity boxes for a dataset index, as
        (x, y, w, h) in CURRENT img_size pixel coordinates.

        Empty for a negative patient — the absence of a box is itself the label,
        so callers must treat [] as "no annotated lesion", not "missing data".

        The rescale is a plain linear factor because the image pipeline is a
        plain square Resize with no crop or pad (see self.transform), so a box
        maps by the same ratio on each axis independently. A crop or a
        letterbox would make this wrong, so it is asserted against the real
        DICOM dimensions rather than a hardcoded 1024.
        """
        pid = self.samples[idx][0]
        boxes = self._boxes.get(pid)
        if not boxes:
            return []

        rows, cols = self.native_size(pid)
        sx = self.img_size / float(cols)
        sy = self.img_size / float(rows)
        return [(x * sx, y * sy, w * sx, h * sy) for (x, y, w, h) in boxes]

    def lesion_mask(self, idx: int) -> "torch.Tensor":
        """
        Boxes for a dataset index rasterised to a (1, img_size, img_size) float
        mask, 1.0 inside any annotated lesion. All-zeros when there is no box.

        Overlapping boxes union rather than accumulate, so the mask stays a
        membership indicator and a doubly-covered pixel is not weighted twice.
        """
        m = torch.zeros(1, self.img_size, self.img_size, dtype=torch.float32)
        for (x, y, w, h) in self.lesion_boxes(idx):
            x0 = max(0, int(math.floor(x)))
            y0 = max(0, int(math.floor(y)))
            x1 = min(self.img_size, int(math.ceil(x + w)))
            y1 = min(self.img_size, int(math.ceil(y + h)))
            if x1 > x0 and y1 > y0:
                m[:, y0:y1, x0:x1] = 1.0
        return m


# ─────────────────────────────────────────────
# Convenience factory
# ─────────────────────────────────────────────

def build_dataset(img_size: int = 256) -> RSNACXRDataset:
    """
    Download (if needed) and return an RSNACXRDataset ready for use.

    Parameters
    ----------
    img_size : int
        Square resolution for resizing DICOM images.

    Returns
    -------
    RSNACXRDataset
    """
    root = download_rsna_dataset()
    images_dir, labels_csv = get_dataset_paths(root)
    return RSNACXRDataset(images_dir=images_dir, labels_csv=labels_csv, img_size=img_size)
