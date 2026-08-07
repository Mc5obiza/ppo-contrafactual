"""
src/breast_classifier.py
========================
f(.) for the mammography domain — the torchxrayvision-free counterpart of
src/xrv_classifier.py.

WHY THIS EXISTS
---------------
Every classifier this project guided with until now came from the
torchxrayvision zoo, which is chest radiography and nothing else. There is no
pretrained breast/mammography head anywhere in it — `default_pathologies` has
18 entries and not one of them is a breast finding. So the CBIS-DDSM domain
cannot borrow f(.); it has to train it (train_breast_classifier.py) and wrap it
here.

This module is deliberately a CLASS-FOR-CLASS MIRROR of src/xrv_classifier.py:

    XRVPneumoniaClassifier      <->  BreastMalignancyClassifier
    EnsemblePneumoniaClassifier <->  BreastEnsembleClassifier
    build_xrv_classifier        <->  build_breast_classifier
    build_ensemble_classifier   <->  build_breast_ensemble_classifier

Same constructor shape, same (B,1|3,H,W) in [0,1] -> (B,1) probability
contract, same frozen-parameters-but-live-input-graph rule. Keeping the two
diffable is the point: pipeline.build_common swaps one for the other and
NOTHING downstream — generator guidance, env termination, the far-pool scan,
the Xn/Xp split, CV — is allowed to notice.

THE ONE THING THAT IS NOT LIKE THE XRV VERSION
----------------------------------------------
xrv models are single-channel and expect inputs scaled to [-1024, 1024]. These
are ImageNet-pretrained torchvision backbones: 3-channel, ImageNet mean/std.
_preprocess below is the single source of truth for that conversion and
train_breast_classifier.py imports it rather than reimplementing it — a
train/inference normalisation mismatch is silent and costs you the whole run.

WHAT THE LABEL MEANS
--------------------
CBIS-DDSM contains only abnormal cases; there are no normal mammograms in it.
So f(.) = P(MALIGNANT) among lesions, NOT P(cancer) among screening images, and
the two PPO directions are benign->malignant and malignant->benign. See
domains/cbis_ddsm_breast.py.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

import config


# ─────────────────────────────────────────────────────────────────────────────
# Weight naming
# ─────────────────────────────────────────────────────────────────────────────
#
# Profiles name classifiers as strings ("densenet121-breast-cbis") because
# DomainProfile.guidance_weights / ensemble_weights / heldout_validators are
# tuples of strings — that is the xrv-shaped interface every consumer already
# speaks, and heldout_validation.py keys its report by those names. The suffix
# is what makes them unambiguous alongside the xrv "densenet121-res224-*" keys
# in the same fields.

WEIGHT_SUFFIX = "-breast-cbis"

SUPPORTED_ARCHS = ("densenet121", "resnet18", "efficientnet_b0")


def arch_from_weights(weights: str) -> str:
    """
    "densenet121-breast-cbis" -> "densenet121".

    Raises ValueError naming the supported set, because the alternative is
    torchvision raising an opaque AttributeError from a getattr miles away.
    """
    if not weights.endswith(WEIGHT_SUFFIX):
        raise ValueError(
            f"{weights!r} is not a breast weight name — expected something "
            f"like 'densenet121{WEIGHT_SUFFIX}'. (xrv weight sets named "
            f"'densenet121-res224-*' belong to the chest domains and must be "
            f"built with src/xrv_classifier.py instead.)"
        )
    arch = weights[: -len(WEIGHT_SUFFIX)]
    if arch not in SUPPORTED_ARCHS:
        raise ValueError(
            f"unsupported backbone {arch!r} in weights {weights!r}. "
            f"Supported: {list(SUPPORTED_ARCHS)}"
        )
    return arch


def checkpoint_path(weights: str) -> str:
    """Where train_breast_classifier.py writes the weights named `weights`."""
    return os.path.join(config.BREAST_CKPT_DIR, f"{arch_from_weights(weights)}.pt")


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing — SHARED with train_breast_classifier.py, do not fork
# ─────────────────────────────────────────────────────────────────────────────

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)

INPUT_SIZE = 224


def _preprocess(x: torch.Tensor, size: int = INPUT_SIZE) -> torch.Tensor:
    """
    (B, 1|3, H, W) in [0, 1]  ->  (B, 3, size, size), ImageNet-normalised.

    Differentiable throughout: interpolate, repeat and the affine normalise all
    pass gradients, so d f(.) / d x survives for classifier guidance.

    A 3-channel input is averaged to 1 first and then re-expanded. That looks
    like a no-op but is not: it matches _to_rgb(...)'s convention everywhere
    else in this codebase, where a grayscale image is carried as 3 identical
    channels, and it guarantees the same answer whether the caller hands over
    (B,1,H,W) or (B,3,H,W).
    """
    if x.shape[1] == 3:
        x = x.mean(dim=1, keepdim=True)
    if x.shape[-1] != size or x.shape[-2] != size:
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    x = x.repeat(1, 3, 1, 1)

    mean = torch.tensor(_IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std  = torch.tensor(_IMAGENET_STD,  device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


def build_backbone(arch: str, pretrained: bool = False) -> nn.Module:
    """
    A torchvision backbone with a 1-logit head.

    Shared with train_breast_classifier.py so the architecture that produces a
    checkpoint and the architecture that loads it cannot drift — a mismatch
    here surfaces as a load_state_dict shape error at best and as silently
    mis-mapped weights at worst.

    `pretrained` is True only during training (ImageNet init); inference always
    loads our own checkpoint over the top, so downloading ImageNet weights
    there would be a pointless network call on an offline compute node.
    """
    if arch == "densenet121":
        m = tvm.densenet121(weights=tvm.DenseNet121_Weights.DEFAULT if pretrained else None)
        m.classifier = nn.Linear(m.classifier.in_features, 1)
    elif arch == "resnet18":
        m = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, 1)
    elif arch == "efficientnet_b0":
        m = tvm.efficientnet_b0(
            weights=tvm.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        )
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 1)
    else:
        raise ValueError(f"unsupported backbone {arch!r}. Supported: {list(SUPPORTED_ARCHS)}")
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Single classifier
# ─────────────────────────────────────────────────────────────────────────────

class BreastMalignancyClassifier(nn.Module):
    """
    One trained backbone exposed as a scalar P(malignant).

    Interface is IDENTICAL to XRVPneumoniaClassifier: input (B,3,H,W) or
    (B,1,H,W) in [0,1] -> output (B,1) probability in [0,1], parameters frozen,
    fully differentiable w.r.t. the input.

    Fully differentiable is load-bearing, not incidental:
    generate_counterfactual computes L_cls and grad_cls through this object, so
    it must NOT be wrapped in torch.no_grad(). The parameters are frozen below;
    the input graph must stay connected.
    """

    def __init__(
        self,
        device: str = config.DEVICE,
        weights: str = "densenet121-breast-cbis",
        pathology: str = "Malignancy",
    ):
        super().__init__()
        self.arch       = arch_from_weights(weights)
        self.weights    = weights
        self.device     = device
        self.input_size = INPUT_SIZE
        # Carried purely for interface parity with XRVPneumoniaClassifier —
        # a single-output head has no head index to select.
        self.pathology  = pathology

        ckpt = checkpoint_path(weights)
        if not os.path.exists(ckpt):
            raise FileNotFoundError(
                f"no trained breast classifier at '{ckpt}'. Unlike the chest "
                f"domains, this one cannot borrow pretrained weights — "
                f"torchxrayvision has no breast head. Train it first:\n"
                f"    sbatch run_breast_classifier.slurm\n"
                f"(or: python train_breast_classifier.py --arch {self.arch})"
            )

        model = build_backbone(self.arch, pretrained=False)
        state = torch.load(ckpt, map_location=device, weights_only=True)
        # Accept both a bare state_dict and the {"state_dict": ..., "val_auc": ...}
        # envelope train_breast_classifier.py writes, so a checkpoint stays
        # loadable whichever way it was saved.
        if isinstance(state, dict) and "state_dict" in state:
            self.val_auc = float(state.get("val_auc", float("nan")))
            state = state["state_dict"]
        else:
            self.val_auc = float("nan")
        model.load_state_dict(state, strict=True)

        self.model = model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 3, H, W) or (B, 1, H, W), values in [0, 1]
        Returns (B, 1) malignancy probability in [0, 1].
        """
        logits = self.model(_preprocess(x.float().to(self.device)))
        if logits.dim() == 1:
            logits = logits.unsqueeze(-1)
        return torch.sigmoid(logits)


def build_breast_classifier(
    device: str = config.DEVICE,
    weights: str = "densenet121-breast-cbis",
    pathology: str = "Malignancy",
) -> BreastMalignancyClassifier:
    """Convenience wrapper matching build_xrv_classifier's signature/logging style."""
    clf = BreastMalignancyClassifier(device=device, weights=weights, pathology=pathology)
    auc = "" if clf.val_auc != clf.val_auc else f", val AUC={clf.val_auc:.3f}"
    print(f"[models] loaded breast classifier ({weights}{auc}) as f(·)")
    return clf


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble
# ─────────────────────────────────────────────────────────────────────────────

class BreastEnsembleClassifier(nn.Module):
    """
    Mean-probability ensemble of BreastMalignancyClassifier members, exposing
    exactly the single-model interface.

    Mirrors EnsemblePneumoniaClassifier, and for the same reason: held-out
    validation on the RSNA domain showed counterfactuals optimised against a
    single classifier do not transfer to classifiers that never guided them
    (transfer ratio 0.13/0.14 for p->n). Requiring several independently
    trained models to agree makes "fool the classifier" a much narrower target
    than "make a semantic edit".

    That argument is if anything MORE urgent here. The xrv members were trained
    on separate corpora of hundreds of thousands of images; these members are
    trained by us on one ~3.5k-image dataset and differ only in architecture.
    They are correlated, so this ensemble is a weaker independence guarantee
    than the chest one — say so when reporting CV rather than implying parity.
    """

    def __init__(
        self,
        device: str = config.DEVICE,
        weights_list: list[str] | None = None,
        pathology: str = "Malignancy",
    ):
        super().__init__()
        weights_list = list(weights_list or [])
        if not weights_list:
            raise ValueError("weights_list must name at least one weight set")

        self.pathology    = pathology
        self.weights_list = weights_list
        self.device       = device
        self.members = nn.ModuleList([
            BreastMalignancyClassifier(device=device, weights=w, pathology=pathology)
            for w in weights_list
        ])
        self.input_size = self.members[0].input_size

        # Grad-CAM hook target, set EXPLICITLY. src/gradcam.find_target_layer
        # walks named_modules() and returns the first submodule named
        # "features" — on an ensemble that silently resolves to members.0's own
        # feature stack and passes one member's CAM off as the ensemble's,
        # without raising. Same trap documented in EnsemblePneumoniaClassifier.
        #
        # Members are ordered so that members[0] is a densenet121 wherever the
        # profile includes one, since that is the arch whose `.features`
        # attribute GRADCAM_TARGET_LAYER="features" actually resolves to.
        self.cam_member = self.members[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 3, H, W) or (B, 1, H, W) in [0, 1]
        Returns (B, 1) mean malignancy probability across members.

        Deliberately NOT wrapped in no_grad: generate_counterfactual needs
        d L_cls / d image to flow through every member.
        """
        return torch.stack([m(x) for m in self.members], dim=0).mean(dim=0)

    def per_member(self, x: torch.Tensor) -> torch.Tensor:
        """(B, M) per-member probabilities, members in weights_list order."""
        return torch.cat([m(x) for m in self.members], dim=-1)


def build_breast_ensemble_classifier(
    device: str = config.DEVICE,
    weights_list: list[str] | None = None,
    pathology: str = "Malignancy",
) -> BreastEnsembleClassifier:
    """Convenience wrapper matching build_ensemble_classifier's signature/logging style."""
    clf = BreastEnsembleClassifier(
        device=device, weights_list=weights_list, pathology=pathology
    )
    names = ", ".join(w.replace(WEIGHT_SUFFIX, "") for w in clf.weights_list)
    print(f"[models] loaded BREAST ENSEMBLE classifier as f(·): [{names}] "
          f"({len(clf.members)} members, mean probability)")
    print(f"[models]   NOTE: members share one training set and differ only by "
          f"architecture, so they are more correlated than the xrv ensemble's "
          f"separate-corpus members. Treat consensus here as a weaker "
          f"anti-adversarial guarantee.")
    return clf
