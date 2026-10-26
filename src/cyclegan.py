"""
src/cyclegan.py
================
Standard Vanilla CycleGAN (Zhu et al., "Unpaired Image-to-Image Translation
using Cycle-Consistent Adversarial Networks", ICCV 2017) — the architecture
and losses exactly as the paper/reference implementation define them, with no
classifier guidance, no Grad-CAM masking, no background lock, and no PPO.

This exists to give the project's PPO-guided pipeline an unconditional,
unguided baseline measured under THIS project's own evaluation protocol (same DICOM preprocessing, same
persisted patient-level split, same Xn/Xp group scan, same CV/FID/IP metric
functions) rather than a literature-reported number — see
train_cyclegan_baseline.py and eval_cyclegan_baseline.py.

Single-channel (grayscale CXR) in/out, matching this project's dataset
contract of (1, H, W) tensors in [0, 1] (domains/base.py). Internally the
generator/discriminator operate in [-1, 1] (tanh output), which is the
convention the original architecture was tuned under; ImageTranslator wraps
that so every external caller only ever sees [0, 1].
"""

from __future__ import annotations

import random

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Generator — ResNet, 9 residual blocks (the paper's config for >=256x256)
# ─────────────────────────────────────────────────────────────────────────────

class ResnetBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
            nn.InstanceNorm2d(dim, affine=False, track_running_stats=False),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
            nn.InstanceNorm2d(dim, affine=False, track_running_stats=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResnetGenerator(nn.Module):
    """c7s1-64, d128, d256, R256 x n_blocks, u128, u64, c7s1-<output_nc>."""

    def __init__(self, input_nc: int = 1, output_nc: int = 1, ngf: int = 64,
                 n_blocks: int = 9):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=True),
            nn.InstanceNorm2d(ngf, affine=False, track_running_stats=False),
            nn.ReLU(True),
        ]
        # d128, d256 — two stride-2 downsampling convs
        n_down = 2
        for i in range(n_down):
            mult = 2 ** i
            layers += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2,
                          padding=1, bias=True),
                nn.InstanceNorm2d(ngf * mult * 2, affine=False, track_running_stats=False),
                nn.ReLU(True),
            ]
        mult = 2 ** n_down
        layers += [ResnetBlock(ngf * mult) for _ in range(n_blocks)]
        for i in range(n_down):
            mult = 2 ** (n_down - i)
            layers += [
                nn.ConvTranspose2d(ngf * mult, ngf * mult // 2, kernel_size=3,
                                    stride=2, padding=1, output_padding=1, bias=True),
                nn.InstanceNorm2d(ngf * mult // 2, affine=False, track_running_stats=False),
                nn.ReLU(True),
            ]
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ─────────────────────────────────────────────────────────────────────────────
# Discriminator — 70x70 PatchGAN
# ─────────────────────────────────────────────────────────────────────────────

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc: int = 1, ndf: int = 64, n_layers: int = 3):
        super().__init__()
        layers = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
        ]
        mult = 1
        for i in range(1, n_layers):
            prev_mult, mult = mult, min(2 ** i, 8)
            layers += [
                nn.Conv2d(ndf * prev_mult, ndf * mult, kernel_size=4, stride=2,
                          padding=1, bias=False),
                nn.InstanceNorm2d(ndf * mult, affine=False, track_running_stats=False),
                nn.LeakyReLU(0.2, True),
            ]
        prev_mult, mult = mult, min(2 ** n_layers, 8)
        layers += [
            nn.Conv2d(ndf * prev_mult, ndf * mult, kernel_size=4, stride=1,
                      padding=1, bias=False),
            nn.InstanceNorm2d(ndf * mult, affine=False, track_running_stats=False),
            nn.LeakyReLU(0.2, True),
        ]
        layers += [nn.Conv2d(ndf * mult, 1, kernel_size=4, stride=1, padding=1)]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ─────────────────────────────────────────────────────────────────────────────
# LSGAN loss (MSE against a constant target map) and the discriminator's
# image history buffer (Shrivastava et al. 2017, used by the original
# CycleGAN implementation to damp oscillation)
# ─────────────────────────────────────────────────────────────────────────────

class GANLoss(nn.Module):
    def __init__(self, target_real: float = 1.0, target_fake: float = 0.0):
        super().__init__()
        self.register_buffer("real_label", torch.tensor(target_real))
        self.register_buffer("fake_label", torch.tensor(target_fake))
        self.loss = nn.MSELoss()

    def __call__(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        target = (self.real_label if target_is_real else self.fake_label).expand_as(prediction)
        return self.loss(prediction, target)


class ImagePool:
    """Buffer of up to `pool_size` past generated images per direction."""

    def __init__(self, pool_size: int = 50):
        self.pool_size = pool_size
        self.images: list[torch.Tensor] = []

    def query(self, images: torch.Tensor) -> torch.Tensor:
        if self.pool_size == 0:
            return images
        out = []
        for image in images:
            image = image.unsqueeze(0)
            if len(self.images) < self.pool_size:
                self.images.append(image)
                out.append(image)
            elif random.random() > 0.5:
                idx = random.randrange(self.pool_size)
                out.append(self.images[idx].clone())
                self.images[idx] = image
            else:
                out.append(image)
        return torch.cat(out, dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# [0,1] <-> [-1,1] boundary — every external caller (training script, eval
# script) works in this project's [0,1] convention; the GAN internals stay in
# the [-1,1] range the tanh output / architecture were designed for.
# ─────────────────────────────────────────────────────────────────────────────

def to_gan_range(x01: torch.Tensor) -> torch.Tensor:
    return x01 * 2.0 - 1.0


def from_gan_range(x11: torch.Tensor) -> torch.Tensor:
    return ((x11 + 1.0) / 2.0).clamp(0.0, 1.0)


@torch.no_grad()
def translate(generator: nn.Module, x01: torch.Tensor) -> torch.Tensor:
    """Run one generator on a [0,1] batch, returning a [0,1] batch."""
    was_training = generator.training
    generator.eval()
    out = from_gan_range(generator(to_gan_range(x01)))
    generator.train(was_training)
    return out
