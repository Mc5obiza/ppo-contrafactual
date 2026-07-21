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