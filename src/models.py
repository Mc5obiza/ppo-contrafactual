"""
src/models.py
=============
Pretrained model loaders:
  - ResNet-18 classifier (f)
  - TorchXRayVision PSPNet lung segmenter (S)
  - YOLOv8n backbone feature extractor (O)

All models are loaded in eval mode with frozen weights.
"""

import torch
import torch.nn as nn
import torchvision.models as tvm
from ultralytics import YOLO

import config


# ─────────────────────────────────────────────────────────────────────────────
# torch.load patch (must run before importing torchxrayvision)
# ─────────────────────────────────────────────────────────────────────────────

_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)


def apply_torch_load_patch():
    torch.load = _patched_torch_load


def revert_torch_load_patch():
    torch.load = _original_torch_load


# ─────────────────────────────────────────────────────────────────────────────
# ResNet-18 classifier
# ─────────────────────────────────────────────────────────────────────────────

def build_classifier(
    ckpt_path: str = config.CLASSIFIER_CKPT,
    device: str = config.DEVICE,
) -> nn.Module:
    """
    Instantiate a ResNet-18 binary classifier and load pretrained weights.

    The checkpoint may use either 'fc.weight' / 'fc.bias' keys (bare linear)
    or 'fc.0.weight' / 'fc.0.bias' (wrapped in Sequential) — both are handled.

    Returns
    -------
    nn.Module
        Frozen classifier in eval mode on ``device``.
    """
    model = tvm.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
    model.fc    = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())

    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    renamed = {
        k.replace("fc.weight", "fc.0.weight").replace("fc.bias", "fc.0.bias"): v
        for k, v in state_dict.items()
    }
    model.load_state_dict(renamed, strict=True)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    print(f"[models] loaded classifier from {ckpt_path}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# TorchXRayVision PSPNet segmenter
# ─────────────────────────────────────────────────────────────────────────────

def build_segmentation_net(device: str = config.DEVICE):
    """
    Load the TorchXRayVision PSPNet lung segmenter.

    Returns
    -------
    (seg_net, lung_channel_idx) : tuple[nn.Module, list[int]]
        Frozen segmenter + indices for Left/Right Lung output channels.
    """
    # Must patch torch.load before import
    apply_torch_load_patch()
    import torchxrayvision as xrv
    seg_net = xrv.baseline_models.chestx_det.PSPNet()
    revert_torch_load_patch()

    seg_net = seg_net.to(device).eval()
    for p in seg_net.parameters():
        p.requires_grad_(False)

    lung_channel_idx = [
        seg_net.targets.index("Left Lung"),
        seg_net.targets.index("Right Lung"),
    ]
    print(f"[models] loaded PSPNet | lung channels: {lung_channel_idx}")
    return seg_net, lung_channel_idx


# ─────────────────────────────────────────────────────────────────────────────
# YOLOv8n backbone feature extractor
# ─────────────────────────────────────────────────────────────────────────────

class YOLOv8FeatureExtractor(nn.Module):
    """
    Runs the first ``BACKBONE_CUTOFF`` layers of a YOLOv8n model and returns
    a pooled feature vector. Weights are frozen.
    """

    BACKBONE_CUTOFF = 9

    def __init__(self, weights: str = config.YOLOV8_WEIGHTS, device: str = config.DEVICE):
        super().__init__()
        self._yolo_wrapper = YOLO(weights)
        self.model = self._yolo_wrapper.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def _backbone_forward(self, x: torch.Tensor) -> torch.Tensor:
        y, out = [], x
        for i, layer in enumerate(self.model.model):
            if i > self.BACKBONE_CUTOFF:
                break
            if layer.f != -1:
                out = (
                    y[layer.f] if isinstance(layer.f, int)
                    else [out if j == -1 else y[j] for j in layer.f]
                )
            out = layer(out)
            y.append(out)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feat_map = self._backbone_forward(x)
            feat_vec = self.pool(feat_map).flatten(start_dim=1)
        return feat_vec


def build_object_detector(
    weights: str = config.YOLOV8_WEIGHTS,
    device: str = config.DEVICE,
) -> YOLOv8FeatureExtractor:
    """Convenience wrapper that prints a status line."""
    obj_det = YOLOv8FeatureExtractor(weights=weights, device=device)
    print(f"[models] loaded YOLOv8 backbone from {weights}")
    return obj_det
