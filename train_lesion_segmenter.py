"""
train_lesion_segmenter.py
=========================
Trains the lesion U-Net that the LLR and MSI clinical metrics are computed
from. Run after train_breast_classifier.py, before any evaluation that reports
those two.

WHAT IT IS FOR
--------------
The chest domains measure clinical plausibility with CTR (cardiothoracic
ratio), which works because torchxrayvision's PSPNet already segments the heart
and lungs. Mammography has no equivalent pretrained segmenter, but CBIS-DDSM
ships something better: pixel-accurate ROI masks for every annotated lesion.
This trains a segmenter on those, and src/metrics.py then measures the lesion
in BOTH the original and the counterfactual — exactly the shape CTR has.

IT MUST NOT GUIDE GENERATION
----------------------------
This net is a measuring instrument. It is never passed to the generator or the
env (pipeline.build_common puts it on PipelineContext.lesion_net and nowhere
else), because a segmenter that had helped steer the edits would make LLR and
MSI circular in precisely the way CV already is — the failure
heldout_validation.py exists to expose. domains/base.DomainProfile states the
same rule for build_lesion_net.

WHY GROUND TRUTH IS USED HERE AND NOT IN THE METRIC PHASE
---------------------------------------------------------
The obvious design is to compare each counterfactual against its own
ground-truth ROI mask. That is not reachable: metrics.scan_groups returns
tensors with no dataset indices, and its determinism contract is relied on by
run_full_evaluation, heldout_validation.py and eval_main.py's sharded
generate/reduce split alike. Ground truth is therefore used where it does not
fight that contract — here, and in check_lesion_localization.py — and the
metric phase uses this trained net instead.

Usage
-----
    python train_lesion_segmenter.py
    python train_lesion_segmenter.py --epochs 40 --batch-size 8
or  sbatch run_lesion_segmenter.slurm
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import config
import domains
from src.breast_segmentation import LESION_CKPT_NAME, LesionSegmenter
from train_breast_classifier import patient_level_split, set_seed


# Below this, LLR and MSI are noise dressed as measurements. Not fatal here —
# the checkpoint is still written so the failure can be inspected — but loudly
# flagged, and build_lesion_segmenter repeats the warning at load time.
MIN_USEFUL_DICE = 0.40


class LesionMaskDataset(Dataset):
    """
    (image, mask) pairs for the images in `base` that HAVE a ground-truth ROI.

    Only a subset of the domain dataset survives: a case row without a
    resolvable ROI mask path contributes an image but no target. Those are
    dropped rather than treated as empty masks, which would train the net to
    predict "no lesion" on images that certainly contain one.
    """

    def __init__(self, base, img_size: int):
        self.base = base
        self.img_size = img_size
        self.resize = T.Resize(
            (img_size, img_size),
            interpolation=T.InterpolationMode.NEAREST,   # masks: never blend labels
        )

        self.index = [
            i for i in range(len(base))
            if getattr(base, "roi_mask_path", lambda _: None)(i)
        ]
        if not self.index:
            raise RuntimeError(
                "no ground-truth ROI masks resolved from the dataset. The "
                "lesion segmenter cannot be trained without them. Check that "
                "dicom_info.csv contains rows with "
                "SeriesDescription=='ROI mask images' and that those files "
                "exist under <root>/jpeg/."
            )
        print(f"[train-seg] {len(self.index)} of {len(base)} images have a "
              f"ground-truth ROI mask")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        idx = self.index[i]
        x, _ = self.base[idx]

        m = Image.open(self.base.roi_mask_path(idx)).convert("L")
        m = self.resize(m)
        # CBIS-DDSM ROI masks are nominally 0/255 but the JPEG encoding leaves
        # intermediate values along the boundary. Threshold high: a generous
        # threshold would inflate every lesion by a ring of compression
        # artefact, which biases both the area (LLR) and the perimeter (MSI).
        mask = (torch.from_numpy(np.asarray(m)).float() / 255.0 > 0.5).float()
        return x, mask.unsqueeze(0)

    def patient_id(self, i: int) -> str:
        """Delegate so patient_level_split works on THIS dataset's indexing."""
        return self.base.patient_id(self.index[i])


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6):
    """Soft Dice on probabilities. Paired with BCE — see train()."""
    p = torch.sigmoid(logits)
    num = 2.0 * (p * target).sum(dim=(1, 2, 3)) + eps
    den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return 1.0 - (num / den).mean()


@torch.no_grad()
def evaluate(model, loader, device: str, threshold: float = 0.5) -> float:
    """Mean per-image Dice at `threshold`."""
    model.eval()
    scores = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        p = (torch.sigmoid(model(x)) > threshold).float()
        inter = (p * y).sum(dim=(1, 2, 3))
        denom = p.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3))
        # An empty prediction on an empty target is a perfect score, not 0/0.
        d = torch.where(denom > 0, 2.0 * inter / denom, torch.ones_like(denom))
        scores.extend(d.cpu().tolist())
    return float(np.mean(scores)) if scores else float("nan")


def train(args) -> int:
    set_seed(args.seed)
    device = config.DEVICE if torch.cuda.is_available() else "cpu"
    print(f"[train-seg] device={device}  domain={config.DATASET_DOMAIN}")

    domain = domains.active()
    if domain.build_lesion_net is None:
        raise SystemExit(
            f"domain {domain.name!r} declares no lesion segmenter "
            f"(build_lesion_net is None), so there is nothing to train. This "
            f"script is for cbis_ddsm_breast."
        )

    base = domain.build_dataset(config.IMG_SIZE)
    ds = LesionMaskDataset(base, config.IMG_SIZE)

    # Patient-level, for the same reason as the classifier: CBIS-DDSM has
    # several views per breast, and a random split would put the same lesion in
    # both folds and report a memorised Dice.
    train_idx, val_idx = patient_level_split(ds, args.val_frac, args.seed)

    train_loader = DataLoader(
        torch.utils.data.Subset(ds, train_idx), batch_size=args.batch_size,
        shuffle=True, num_workers=config.DATALOADER_NUM_WORKERS, drop_last=True,
    )
    val_loader = DataLoader(
        torch.utils.data.Subset(ds, val_idx), batch_size=args.batch_size,
        shuffle=False, num_workers=config.DATALOADER_NUM_WORKERS,
    )

    model = LesionSegmenter(device=device, pretrained_encoder=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    bce = nn.BCEWithLogitsLoss()

    os.makedirs(config.BREAST_CKPT_DIR, exist_ok=True)
    ckpt_path = os.path.join(config.BREAST_CKPT_DIR, LESION_CKPT_NAME)

    best, since = -1.0, 0
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            # BCE alone collapses to all-background here: lesions occupy a few
            # percent of a mammogram, so "predict nothing" is already a very
            # low BCE. Dice is scale-invariant in exactly the way that fixes.
            loss = bce(logits, y) + dice_loss(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += float(loss)
        sched.step()

        dice = evaluate(model, val_loader, device)
        flag = ""
        if dice > best:
            best, since = dice, 0
            torch.save(
                {"state_dict": model.state_dict(), "val_dice": dice, "epoch": epoch},
                ckpt_path,
            )
            flag = "  <- best, saved"
        else:
            since += 1

        print(f"[lesion-unet] epoch {epoch + 1:3d}/{args.epochs}  "
              f"train_loss={running / max(len(train_loader), 1):.4f}  "
              f"val_Dice={dice:.4f}{flag}")

        if since >= args.patience:
            print(f"[lesion-unet] no improvement in {args.patience} epochs — "
                  f"early stop")
            break

    print(f"\n[lesion-unet] best val Dice {best:.4f} -> {ckpt_path}")
    if best < MIN_USEFUL_DICE:
        print(f"\n⚠ val Dice {best:.4f} < {MIN_USEFUL_DICE:.2f}.")
        print("  LLR and MSI are computed from this segmenter, so at this "
              "quality they measure the")
        print("  segmenter's failures rather than the counterfactuals. The "
              "checkpoint is saved so you")
        print("  can inspect it, but do not report those two metrics until "
              "this improves.")
        return 1

    print(f"\n✓ Next: python check_lesion_localization.py")
    print(f"  It calibrates MSI against the ground-truth `mass margins` labels "
          f"— MSI must separate")
    print(f"  SPICULATED from CIRCUMSCRIBED on REAL images before it can say "
          f"anything about counterfactuals.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train the CBIS-DDSM lesion segmenter.")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    return train(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
