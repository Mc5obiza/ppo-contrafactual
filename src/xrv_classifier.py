"""
src/xrv_classifier.py
======================
Drop-in replacement for the custom ResNet-18 classifier (build_classifier
in models.py), using TorchXRayVision's pretrained multi-pathology
DenseNet-121 instead.

WHY: the custom binary ResNet-18 classifier may have learned a shortcut
correlation (e.g. global brightness/contrast/exposure) rather than
localized pneumonia-specific features — visible in generate_and_plot_diff
output as diffuse, whole-thorax changes instead of localized opacity
synthesis. A classifier trained jointly across MANY pathologies (forced
to discriminate pneumonia from effusion, cardiomegaly, edema, etc.) is
much less able to rely on a single global shortcut, since that shortcut
wouldn't help distinguish between conditions.

Two weight options:
  "densenet121-res224-rsna" -> trained on RSNA specifically (same dataset
                                 distribution you're already using — try
                                 this one first)
  "densenet121-res224-all"  -> joint NIH+CheXpert+PadChest+MIMIC+RSNA
                                 (broader, potentially more robust features)

This requires ZERO fine-tuning — it's a pretrained forward pass, same
"free lunch" pattern used for the FOP support-device detector fix.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchxrayvision as xrv

import config


class XRVPneumoniaClassifier(nn.Module):
    """
    Wraps TorchXRayVision's DenseNet-121 to expose a single scalar
    "Pneumonia" probability, matching the interface your existing
    ResNet-18 classifier provides (input: (B,3,H,W) or (B,1,H,W) in
    [0,1] -> output: (B,1) probability in [0,1]).

    Fully differentiable — gradients flow through exactly as they did
    with the ResNet-18, so no changes needed in generate_counterfactual's
    L_cls / grad_cls computation.
    """

    def __init__(self, device: str = config.DEVICE, weights: str = "densenet121-res224-rsna"):
        super().__init__()
        model = xrv.models.DenseNet(weights=weights)
        self.model      = model.to(device).eval()
        self.device     = device
        self.input_size = 224

        if "Pneumonia" not in model.pathologies:
            raise ValueError(
                f"'Pneumonia' not found in pathologies for weights='{weights}': "
                f"{model.pathologies}. Try 'densenet121-res224-all' instead."
            )
        self.pna_idx = model.pathologies.index("Pneumonia")

        # NOTE: frozen, same as the original classifier — this is used for
        # guidance gradients (d L_cls / d image), not trained itself.
        for p in self.model.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 3, H, W) or (B, 1, H, W), values in [0, 1]
            (matches the _to_rgb(...) convention used everywhere else in
            this codebase — RGB in, averaged to single-channel internally,
            since xrv models are single-channel).
        Returns (B, 1) pneumonia probability in [0, 1].
        """
        if x.shape[1] == 3:
            x = x.mean(dim=1, keepdim=True)

        x_xrv = x * 2048.0 - 1024.0
        if x_xrv.shape[-1] != self.input_size or x_xrv.shape[-2] != self.input_size:
            x_xrv = F.interpolate(
                x_xrv, size=(self.input_size, self.input_size),
                mode="bilinear", align_corners=False,
            )

        # NOT wrapped in torch.no_grad() — generate_counterfactual needs
        # gradients w.r.t. the input image to flow through this classifier
        # for classifier guidance (grad_cls). Parameters are frozen above,
        # but the input graph must stay connected.
        probs   = self.model(x_xrv.float().to(self.device))
        pna_prob = probs[:, self.pna_idx : self.pna_idx + 1]  # keep (B,1) shape

        return pna_prob


def build_xrv_classifier(
    device: str = config.DEVICE,
    weights: str = "densenet121-res224-rsna",
) -> XRVPneumoniaClassifier:
    """Convenience wrapper matching build_classifier()'s call signature/logging style."""
    clf = XRVPneumoniaClassifier(device=device, weights=weights)
    print(f"[models] loaded XRV pneumonia classifier ({weights}) as f(·) replacement")
    return clf


class EnsemblePneumoniaClassifier(nn.Module):
    """
    Mean-probability ensemble of several XRVPneumoniaClassifier members.

    Exposes EXACTLY the single-model interface — (B,1|3,H,W) in [0,1] ->
    (B,1) probability in [0,1], differentiable w.r.t. the input, all
    parameters frozen — so it is a drop-in replacement for
    XRVPneumoniaClassifier everywhere f(·) is consumed.

    WHY THIS EXISTS
    ---------------
    Held-out validation (checkpoints/heldout/heldout_validation.json, job
    17649721) showed that counterfactuals optimised against the single
    RSNA-trained model do not transfer to classifiers that never guided
    them: transfer ratio 0.21 / 0.59 for n→p but only 0.13 / 0.14 for p→n,
    with the original-image agreement control passing (Spearman ρ = 0.70).
    The p→n direction was fooling f(·) rather than removing disease.

    Adversarial directions rarely survive across independently trained
    models, so requiring several models to agree turns "fool the
    classifier" into a much narrower target than "make a semantic edit".
    This is the same argument this module's header already makes for
    preferring a multi-pathology model over a binary one — the head counts
    make the point concretely: densenet121-res224-rsna exposes only 2
    output heads (Pneumonia, Lung Opacity), while -chex exposes 11,
    -mimic_ch 11, -pc 15 and -nih 14. A 2-head single-corpus model has the
    least-constrained feature space and is therefore the easiest to fool.

    WHAT THIS CHANGES DOWNSTREAM
    ----------------------------
    pipeline.build_common() constructs ONE classifier object that flows to
    every consumer, so swapping this in changes all of them at once:
      - generator.classifier -> L_cls, the guidance gradient
      - env.f                -> d_t, i.e. episode termination AND reward
      - the env far-pool scan and metrics' Xn / Xp group split
      - CV in metrics.run_full_evaluation
    That breadth is intentional: termination now requires CONSENSUS, not
    just one model's agreement. It does mean CV, group sizes and far-pool
    sizes are NOT comparable to pre-ensemble runs — see the warning printed
    by build_ensemble_classifier.

    Members are held in an nn.ModuleList (not a plain list) so .to(),
    .eval() and state_dict() reach them, matching how the rest of the
    codebase treats submodules.
    """

    def __init__(self, device: str = config.DEVICE, weights_list: list[str] | None = None):
        super().__init__()
        weights_list = list(weights_list or config.GUIDANCE_ENSEMBLE_WEIGHTS)
        if not weights_list:
            raise ValueError("weights_list must name at least one weight set")

        self.members = nn.ModuleList(
            [XRVPneumoniaClassifier(device=device, weights=w) for w in weights_list]
        )
        self.weights_list = weights_list
        self.device = device
        # All members share the xrv 224 convention; kept for interface parity
        # with XRVPneumoniaClassifier.
        self.input_size = self.members[0].input_size

        # Grad-CAM hook target. Set EXPLICITLY rather than left to
        # gradcam.find_target_layer, which walks named_modules() and returns
        # the FIRST submodule named "features" — on an ensemble that silently
        # resolves to members.0 (i.e. one member's CAM masquerading as the
        # ensemble's) without raising. See generator.compute_gradcam_latent_mask.
        self.cam_member = self.members[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 3, H, W) or (B, 1, H, W) in [0, 1]
        Returns (B, 1) mean pneumonia probability across members.

        Deliberately NOT wrapped in no_grad: generate_counterfactual needs
        d L_cls / d image to flow through every member.
        """
        return torch.stack([m(x) for m in self.members], dim=0).mean(dim=0)

    def per_member(self, x: torch.Tensor) -> torch.Tensor:
        """
        (B, M) per-member probabilities, members in weights_list order.

        Not used by the default guidance path — generate_counterfactual takes
        BCE of the MEAN probability, which already routes most of the gradient
        to the lagging member (a saturated member's own gradient vanishes).
        Provided for diagnostics, and as the hook if mean-of-per-member-BCE
        turns out to be needed instead.
        """
        return torch.cat([m(x) for m in self.members], dim=-1)


def build_ensemble_classifier(
    device: str = config.DEVICE,
    weights_list: list[str] | None = None,
) -> EnsemblePneumoniaClassifier:
    """Convenience wrapper matching build_xrv_classifier's signature/logging style."""
    clf = EnsemblePneumoniaClassifier(device=device, weights_list=weights_list)
    names = ", ".join(w.replace("densenet121-res224-", "") for w in clf.weights_list)
    print(f"[models] loaded ENSEMBLE pneumonia classifier as f(·): [{names}] "
          f"({len(clf.members)} members, mean probability)")
    print(f"[models]   NOTE: f(·) is an ensemble — CV, far-pool sizes and the "
          f"Xn/Xp group split are NOT comparable to single-classifier runs.")
    return clf