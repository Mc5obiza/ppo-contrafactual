"""
domains/rsna_pneumonia.py
=========================
The original domain: RSNA Pneumonia Detection Challenge, f(.) = "Pneumonia".

This is a THIN wrapper. ingestion.py keeps the download and the Dataset class
verbatim, so every number measured so far (checkpoints/eval_metrics.json,
checkpoints/heldout/heldout_validation.json) remains reproducible from this
profile with no behaviour change.

Weight sets: rsna and chex both expose a live "Pneumonia" head. nih/pc/mimic_ch
are reserved as held-out validators — see the disjointness check in
domains/base.DomainProfile.validate.
"""

from __future__ import annotations

from ingestion import build_dataset as _build_rsna
from .base import DomainProfile


PROFILE = DomainProfile(
    name="rsna_pneumonia",
    build_dataset=_build_rsna,
    pathology="Pneumonia",
    guidance_weights="densenet121-res224-rsna",
    ensemble_weights=(
        "densenet121-res224-rsna",
        "densenet121-res224-chex",
    ),
    heldout_validators=(
        "densenet121-res224-nih",
        "densenet121-res224-pc",
        "densenet121-res224-mimic_ch",
    ),
    clinical_metrics=("CTR", "SCP"),
    notes=(
        "Baseline domain. NOTE both clinical metrics are weak here: CTR is a "
        "cardiomegaly measure and SCP's sigmoid(-||yolo_feat||/50) proxy was "
        "measured degenerate (0.21% relative spread) in the last eval. Neither "
        "is pneumonia-specific."
    ),
)
