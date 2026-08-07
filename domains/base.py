"""
domains/base.py
===============
The DomainProfile contract.

WHY THIS EXISTS
---------------
Swapping datasets is not just swapping a Dataset class. Three things move
together and MUST stay consistent, and before this module they lived in three
unrelated files:

  1. the images + binary label            (ingestion.py)
  2. the xrv pathology head f(.) targets  ("Pneumonia", hardcoded in
                                           src/xrv_classifier.py)
  3. which clinical metrics mean anything (CTR/SCP, hardcoded in
                                           src/metrics.py)

Change (1) without (2) and you guide toward the wrong disease. Change (1)
without (3) and you report the cardiothoracic ratio for a lung-nodule run,
where it measures nothing. A DomainProfile binds all three so a new dataset is
one new file plus one config line, not an edit to three modules.

THE DATASET CONTRACT
--------------------
build_dataset(img_size) returns a torch Dataset whose __getitem__ yields

    (x, y)   x : (1, img_size, img_size) float32, values in [0, 1]
             y : (1,) float32, 1.0 = positive/diseased, 0.0 = negative

That is exactly what ingestion.RSNACXRDataset already returns, and every
downstream consumer (the far-pool scan, the env, the generator's _to_rgb, the
metrics group split) assumes it. Anything else belongs inside the domain
module, not leaking outward.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


# The op_threshs entry for a head can be NaN, in which case
# torchxrayvision.models.op_norm returns a CONSTANT 0.5 for that head with
# identically zero gradient — the head exists in model.pathologies and the
# forward pass succeeds, but f(.) is a constant. That is silent and fatal:
# guidance gradients vanish and Grad-CAM comes back spatially constant. This
# module refuses to build such a profile. See DomainProfile.validate.
_DEAD = "NaN op_thresh (op_norm returns a constant 0.5, zero gradient)"


@dataclass(frozen=True)
class DomainProfile:
    """
    Everything that changes when the dataset changes.

    Fields
    ------
    name                : registry key, also stamped into eval output
    build_dataset       : (img_size: int) -> torch.utils.data.Dataset,
                          per the dataset contract in this module's docstring
    pathology           : xrv head name f(.) targets, e.g. "Pneumonia".
                          Must exist in torchxrayvision default_pathologies.
    guidance_weights    : single-model f(.), used when
                          config.USE_GUIDANCE_ENSEMBLE is False
    ensemble_weights    : mean-probability ensemble f(.), used when it is True
    heldout_validators  : weight sets heldout_validation.py re-scores with.
                          MUST be disjoint from the guidance weights or CV
                          measures gradient descent against f succeeding
                          against f — the circularity the ensemble exists to
                          break.
    clinical_metrics    : which of the domain metrics in src/metrics.py apply.
                          "CTR" is cardiomegaly-specific, "SCP" is pleural
                          effusion-specific; neither means anything for a
                          nodule task. Empty tuple is legitimate.
    notes               : free text surfaced at build time
    """

    name: str
    build_dataset: Callable[[int], object]
    pathology: str
    guidance_weights: str
    ensemble_weights: tuple[str, ...]
    heldout_validators: tuple[str, ...] = ()
    clinical_metrics: tuple[str, ...] = ()
    notes: str = ""

    # ──────────────────────────────────────────────────────────────────────────

    def validate(self) -> None:
        """
        Fail loudly at construction on the two mistakes that are otherwise
        silent for a whole training run.

        Raises ValueError on either.
        """
        from torchxrayvision.datasets import default_pathologies
        from torchxrayvision.models import model_urls

        if self.pathology not in default_pathologies:
            raise ValueError(
                f"domain {self.name!r}: pathology {self.pathology!r} is not a "
                f"torchxrayvision head. Available: {default_pathologies}"
            )
        idx = default_pathologies.index(self.pathology)

        # 1. Every weight set this profile names must have a LIVE head.
        used = {self.guidance_weights, *self.ensemble_weights, *self.heldout_validators}
        dead = []
        for w in sorted(used):
            entry = model_urls.get(w)
            if entry is None:
                raise ValueError(
                    f"domain {self.name!r}: unknown weights {w!r}. "
                    f"Known: {sorted(k for k in model_urls if 'densenet121' in k)}"
                )
            ot = entry.get("op_threshs")
            if ot is not None and np.isnan(float(ot[idx])):
                dead.append(w)
        if dead:
            raise ValueError(
                f"domain {self.name!r}: head {self.pathology!r} is DEAD in "
                f"{dead} — {_DEAD}. f(.) would be a constant and every gradient "
                f"through it exactly zero. Pick weight sets whose op_threshs "
                f"entry for {self.pathology!r} is finite."
            )

        # 2. Anti-circularity: validators must never have guided generation.
        guidance = {self.guidance_weights, *self.ensemble_weights}
        overlap  = guidance & set(self.heldout_validators)
        if overlap:
            raise ValueError(
                f"domain {self.name!r}: {sorted(overlap)} appear as BOTH "
                f"guidance and held-out validators. heldout_validation.py "
                f"would then re-score counterfactuals with a model that "
                f"guided them, which is the circularity it exists to detect."
            )

    def summary(self) -> str:
        cm = ", ".join(self.clinical_metrics) if self.clinical_metrics else "none"
        return (
            f"[domain] {self.name}: f(.) targets {self.pathology!r} | "
            f"guidance={self.guidance_weights} | "
            f"ensemble={list(self.ensemble_weights)} | "
            f"validators={list(self.heldout_validators)} | "
            f"clinical metrics: {cm}"
        )
