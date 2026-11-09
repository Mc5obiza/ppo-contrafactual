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

NON-CHEST DOMAINS (the build_* escape hatches)
----------------------------------------------
The three things above were originally all torchxrayvision-shaped: f(.) was an
xrv DenseNet head, S was the xrv PSPNet lung segmenter, and the clinical
metrics were CTR/SCP. That holds for any chest radiograph and for nothing else.
Mammography (domains/cbis_ddsm_breast.py) has no xrv head, no lungs and no
heart, so a domain can override each backend independently:

    build_guidance()    -> f(.)                    replaces the xrv classifier
    build_validators()  -> {name: classifier}      replaces the xrv validators
    build_masker()      -> (seg_net, channel_idx)  replaces the PSPNet
    build_lesion_net()  -> lesion segmenter        NEW; no chest equivalent

Leave them None and the profile behaves exactly as before — that is what keeps
rsna_pneumonia and node21_nodule byte-identical. Set them and validate() stops
consulting torchxrayvision metadata that cannot describe your models, but it
KEEPS enforcing the anti-circularity rule, which is a property of the
experiment rather than of any particular model zoo.

A replacement masker must match the PSPNet call contract that
src/generator.py::_get_seg_mask assumes: it takes the x*2048-1024 scaled input,
returns (B, C, H, W) logits, and exposes .targets. It does NOT need to be
differentiable — every caller runs it under no_grad and uses the result as a
blend weight — but it does need to produce a NON-DEGENERATE mask, since
all-zero silently leaves the policy nothing it may edit and all-one silently
disables LOCK_BACKGROUND.
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
                          nodule task. "LLR"/"MSI"/"BDC"/"MOP" are the
                          mammography set. Empty tuple is legitimate.
    notes               : free text surfaced at build time

    Backend overrides (all default None = the torchxrayvision behaviour; see
    the NON-CHEST DOMAINS section of this module's docstring)
    ------------------------------------------------------------------------
    build_guidance      : () -> nn.Module, f(.). When set, the weight-set
                          fields above are labels only — validate() no longer
                          resolves them against torchxrayvision.model_urls.
    build_validators    : () -> dict[str, nn.Module] for heldout_validation.py,
                          keyed by the names in heldout_validators.
    build_masker        : () -> (seg_net, channel_idx) replacing the PSPNet.
    build_lesion_net    : () -> a lesion segmenter for the clinical metrics.
                          MUST NOT be part of the guidance loop: a segmenter
                          that helped steer generation makes the lesion metrics
                          circular in exactly the way CV already is.
    fid_backbone        : which feature space FID is measured in — "xrv",
                          "inception", or a domain-specific key handled in
                          src/metrics.py.
    """

    name: str
    build_dataset: Callable[[int], object]
    pathology: str
    guidance_weights: str
    ensemble_weights: tuple[str, ...]
    heldout_validators: tuple[str, ...] = ()
    clinical_metrics: tuple[str, ...] = ()
    notes: str = ""

    build_guidance:   Callable[[], object] | None = None
    build_validators: Callable[[], dict] | None = None
    build_masker:     Callable[[], tuple] | None = None
    build_lesion_net: Callable[[], object] | None = None
    fid_backbone: str = "xrv"

    # ──────────────────────────────────────────────────────────────────────────

    def validate(self) -> None:
        """
        Fail loudly at construction on the mistakes that are otherwise silent
        for a whole training run.

        Check 1 (live xrv head) is meaningful only for the torchxrayvision
        backend and is skipped when the profile brings its own classifier —
        op_threshs is an xrv implementation detail, not a property a ResNet
        trained on mammograms can have.

        Check 2 (anti-circularity) runs for EVERY backend. It is a property of
        the experiment, not of the model zoo: re-scoring counterfactuals with a
        model that guided them measures nothing, whichever library the model
        came from.

        Raises ValueError on either.
        """
        if self.build_guidance is None:
            self._validate_xrv_heads()

        # Anti-circularity: validators must never have guided generation.
        guidance = {self.guidance_weights, *self.ensemble_weights}
        overlap  = guidance & set(self.heldout_validators)
        if overlap:
            raise ValueError(
                f"domain {self.name!r}: {sorted(overlap)} appear as BOTH "
                f"guidance and held-out validators. heldout_validation.py "
                f"would then re-score counterfactuals with a model that "
                f"guided them, which is the circularity it exists to detect."
            )

        # A profile that overrides one half of the classifier backend but not
        # the other leaves heldout_validation.py trying to load xrv weight sets
        # by name for a domain whose models are not in the xrv zoo. That fails
        # far from here, inside a job, so refuse it now.
        if self.build_guidance is not None and self.heldout_validators and (
            self.build_validators is None
        ):
            raise ValueError(
                f"domain {self.name!r}: build_guidance is set (non-xrv f(.)) and "
                f"heldout_validators names {list(self.heldout_validators)}, but "
                f"build_validators is None. heldout_validation.py would try to "
                f"resolve those names against the torchxrayvision model zoo, "
                f"where they do not exist. Supply build_validators, or set "
                f"heldout_validators=() and accept that CV cannot be checked "
                f"for transfer."
            )

    def _validate_xrv_heads(self) -> None:
        """
        The torchxrayvision-only half of validate(): the named weight sets must
        exist and must expose a LIVE head for self.pathology.
        """
        from torchxrayvision.datasets import default_pathologies
        from torchxrayvision.models import model_urls

        if self.pathology not in default_pathologies:
            raise ValueError(
                f"domain {self.name!r}: pathology {self.pathology!r} is not a "
                f"torchxrayvision head. Available: {default_pathologies}. "
                f"If this domain is not chest radiography at all, give it a "
                f"build_guidance and this check is skipped."
            )
        idx = default_pathologies.index(self.pathology)

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

    def summary(self) -> str:
        cm = ", ".join(self.clinical_metrics) if self.clinical_metrics else "none"
        backend = "xrv" if self.build_guidance is None else "custom"
        extra = ""
        if self.build_masker is not None:
            extra += " | masker=custom"
        if self.build_lesion_net is not None:
            extra += " | lesion-seg=yes"
        return (
            f"[domain] {self.name}: f(.) targets {self.pathology!r} "
            f"({backend} backend) | "
            f"guidance={self.guidance_weights} | "
            f"ensemble={list(self.ensemble_weights)} | "
            f"validators={list(self.heldout_validators)} | "
            f"FID={self.fid_backbone}{extra} | "
            f"clinical metrics: {cm}"
        )
