"""
ingestion.py
============
Handles dataset download from Kaggle (RSNA Pneumonia Detection Challenge)
and exposes the RSNACXRDataset class for use by training pipelines.
"""

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

def download_rsna_dataset() -> str:
    """
    Download the RSNA Pneumonia Detection dataset via kagglehub.
    Requires ~/.kaggle/kaggle.json to be present and valid.

    Returns
    -------
    str
        Path to the root directory of the downloaded dataset.
    """
    import kagglehub

    path = kagglehub.competition_download("rsna-pneumonia-detection-challenge")
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
