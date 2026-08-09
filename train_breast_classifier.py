"""
train_breast_classifier.py
==========================
Trains f(.) for the mammography domain. Run this BEFORE anything else on
config.DATASET_DOMAIN = "cbis_ddsm_breast".

WHY THIS EXISTS AT ALL
----------------------
The chest domains never needed a training script: torchxrayvision ships
DenseNet-121s pretrained on NIH, CheXpert, PadChest, MIMIC and RSNA, and
src/xrv_classifier.py just loads one. That was the "free lunch" the whole
guidance design leaned on.

There is no such lunch for mammography. torchxrayvision's default_pathologies
has 18 entries and not one is a breast finding, so f(.) has to be trained here.

WHAT IT TRAINS AND WHY THREE OF THEM
------------------------------------
config.BREAST_BACKBONES, by default densenet121 + efficientnet_b0 + resnet18.
The first two become the guidance ensemble and the third is held out as the
validator heldout_validation.py re-scores with. That split is declared in
domains/cbis_ddsm_breast.py and enforced by DomainProfile.validate — it is what
keeps CV from being "gradient descent against f succeeded against f".

BE HONEST ABOUT HOW INDEPENDENT THEY ARE. The xrv ensemble's members were
trained by different groups on different hospitals' data. These three share one
~3.5k-image dataset and differ only in architecture. They will agree far more
than the chest members do, so consensus here is a weaker anti-adversarial
guarantee and the held-out transfer ratio is an optimistic bound. Report it
that way.

THE SPLIT IS PATIENT-LEVEL, AND THIS IS NOT OPTIONAL
----------------------------------------------------
CBIS-DDSM has up to four images per patient (LEFT/RIGHT x CC/MLO) plus one row
per lesion. A random image-level split therefore puts the SAME BREAST in train
and validation, and the reported AUC measures memorisation. Patients are
grouped here before any split is taken.

ONE BACKBONE PER JOB
--------------------
Training all three in a single process is what the 4h wall clock keeps running
out on: the third backbone starts three hours in and gets killed mid-run.
run_breast_classifier.slurm is therefore a job array — one array task per entry
of BREAST_BACKBONES, each with its own --arch and its own full time budget.
They share nothing but the dataset and the seed, and the split is a pure
function of (seed, val_frac), so the folds are identical across tasks.

The AUC floor that used to be checked at the end of the combined run cannot be
checked by any single array task — none of them sees the others' results. It
moved to `--summary`, which reads the saved checkpoints and gates on all of
them; the array's dependent cleanup task runs it.

Usage
-----
    python train_breast_classifier.py                 # all of BREAST_BACKBONES
    python train_breast_classifier.py --arch resnet18 # just one
    python train_breast_classifier.py --epochs 30
    python train_breast_classifier.py --summary       # gate on saved ckpts
or  sbatch run_breast_classifier.slurm                # array, one task per arch
"""

from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# config MUST be imported before anything that might pull in torchxrayvision —
# it registers its own top-level `config` package. See domains/__init__.py.
import config
import domains
from src.breast_classifier import _preprocess, build_backbone


# An f(.) near chance makes every downstream number meaningless: CV becomes a
# coin flip, the Xn/Xp split becomes arbitrary, and 20 hours of PPO optimise
# against noise. Fail here instead.
MIN_ACCEPTABLE_AUC = 0.65


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def patient_level_split(
    dataset, val_frac: float = 0.2, seed: int = 0
) -> tuple[list[int], list[int]]:
    """
    Split dataset indices by PATIENT, never by image.

    Returns (train_idx, val_idx).

    The grouping is the entire point — see this module's docstring. If the
    dataset cannot report a patient id we refuse rather than silently falling
    back to a random split, because the failure mode of that fallback is an
    inflated AUC that looks like success.
    """
    if not hasattr(dataset, "patient_id"):
        raise AttributeError(
            f"{type(dataset).__name__} has no patient_id(idx); a patient-level "
            f"split cannot be taken. A random split would leak the same breast "
            f"across folds and report a memorisation score as an AUC."
        )

    by_patient: dict[str, list[int]] = {}
    for i in range(len(dataset)):
        by_patient.setdefault(dataset.patient_id(i), []).append(i)

    patients = sorted(by_patient)
    rng = random.Random(seed)
    rng.shuffle(patients)

    n_val = max(1, int(len(patients) * val_frac))
    val_patients = set(patients[:n_val])

    train_idx, val_idx = [], []
    for p, idxs in by_patient.items():
        (val_idx if p in val_patients else train_idx).extend(idxs)

    print(f"[train-clf] {len(patients)} patients -> "
          f"{len(patients) - n_val} train / {n_val} val")
    print(f"[train-clf] {len(train_idx)} train images / {len(val_idx)} val images")
    return sorted(train_idx), sorted(val_idx)


def _augment(x: torch.Tensor) -> torch.Tensor:
    """
    Horizontal flip only, applied per-sample.

    Mammography is laterality-sensitive: a left and a right breast are mirror
    images, and both appear in the dataset, so a horizontal flip maps a real
    image to another real image. Rotation does NOT — mammograms are acquired in
    two standardised projections against the chest wall, and a rotated one is
    not a view any scanner produces. Adding rotation would teach the classifier
    to be invariant to something that never varies, at the cost of the
    orientation cues that actually carry signal.
    """
    flip = torch.rand(x.shape[0], device=x.device) < 0.5
    x = x.clone()
    x[flip] = torch.flip(x[flip], dims=[-1])
    return x


# ─────────────────────────────────────────────────────────────────────────────
# Train / evaluate
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device: str) -> tuple[float, float]:
    """(AUC, mean BCE loss) on `loader`."""
    from sklearn.metrics import roc_auc_score

    model.eval()
    probs, targets, losses = [], [], []
    crit = nn.BCEWithLogitsLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(_preprocess(x))
        losses.append(float(crit(logits, y)))
        probs.append(torch.sigmoid(logits).flatten().cpu())
        targets.append(y.flatten().cpu())

    p = torch.cat(probs).numpy()
    t = torch.cat(targets).numpy()
    # A validation fold that landed on one class cannot produce an AUC; say so
    # rather than letting sklearn raise from inside the training loop.
    auc = float(roc_auc_score(t, p)) if len(np.unique(t)) > 1 else float("nan")
    return auc, float(np.mean(losses))


def train_one(
    arch: str,
    dataset,
    train_idx: list[int],
    val_idx: list[int],
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    patience: int,
) -> float:
    """
    Train one backbone, save the best-AUC checkpoint, return that AUC.
    """
    print("\n" + "=" * 70)
    print(f"Training {arch}")
    print("=" * 70)

    train_loader = DataLoader(
        Subset(dataset, train_idx), batch_size=batch_size, shuffle=True,
        num_workers=config.DATALOADER_NUM_WORKERS, drop_last=True,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx), batch_size=batch_size, shuffle=False,
        num_workers=config.DATALOADER_NUM_WORKERS,
    )

    model = build_backbone(arch, pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    # Class imbalance: ~40% malignant. Mild, but pos_weight costs nothing and
    # keeps the operating point near 0.5, which matters because the env's
    # termination test and the metrics' Xn/Xp split are both hardcoded around
    # that threshold.
    n_pos = sum(1 for i in train_idx if float(dataset.samples[i][-1]) == 1.0)
    n_neg = len(train_idx) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    print(f"[train-clf] train balance: {n_pos} malignant / {n_neg} benign "
          f"(pos_weight={pos_weight.item():.3f})")
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    os.makedirs(config.BREAST_CKPT_DIR, exist_ok=True)
    ckpt_path = os.path.join(config.BREAST_CKPT_DIR, f"{arch}.pt")

    best_auc, best_epoch, since_improved = -1.0, -1, 0
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = crit(model(_preprocess(_augment(x))), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += float(loss)
        sched.step()

        auc, val_loss = evaluate(model, val_loader, device)
        flag = ""
        if auc > best_auc:
            best_auc, best_epoch, since_improved = auc, epoch, 0
            torch.save(
                {"state_dict": model.state_dict(), "val_auc": auc, "arch": arch,
                 "epoch": epoch},
                ckpt_path,
            )
            flag = "  <- best, saved"
        else:
            since_improved += 1

        print(f"[{arch}] epoch {epoch + 1:3d}/{epochs}  "
              f"train_loss={running / max(len(train_loader), 1):.4f}  "
              f"val_loss={val_loss:.4f}  val_AUC={auc:.4f}{flag}")

        if since_improved >= patience:
            print(f"[{arch}] no improvement in {patience} epochs — early stop")
            break

    print(f"[{arch}] best val AUC {best_auc:.4f} (epoch {best_epoch + 1}) "
          f"-> {ckpt_path}")
    return best_auc


# ─────────────────────────────────────────────────────────────────────────────
# Summary / gate
# ─────────────────────────────────────────────────────────────────────────────

def _saved_auc(arch: str) -> float:
    """Best val AUC recorded in `arch`'s checkpoint, or nan if there isn't one."""
    path = os.path.join(config.BREAST_CKPT_DIR, f"{arch}.pt")
    if not os.path.exists(path):
        return float("nan")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return float(ckpt.get("val_auc", float("nan")))


def report_and_gate(results: dict[str, float]) -> int:
    """
    Print the per-backbone table and apply the AUC floor. Returns an exit code.

    Split out of main() because with one backbone per array task no single
    training process can apply this gate — it only ever sees its own result.
    `--summary` calls it with the AUCs read back off the checkpoints.
    """
    print("\n" + "=" * 70)
    print("SUMMARY — best validation AUC per backbone")
    print("=" * 70)
    for a, auc in results.items():
        print(f"  {a:18s} {auc:.4f}")

    # The gate. A classifier at chance makes CV, the Xn/Xp split and every
    # clinical metric meaningless, and none of them will say so — they will
    # just produce numbers. Refuse here, where it costs an hour instead of a
    # 20-hour PPO run plus an evaluation.
    #
    # A missing checkpoint reads back as nan and fails the same test as a weak
    # one: an array task that died silently must not look like a pass.
    weak = {a: v for a, v in results.items()
            if not (v == v) or v < MIN_ACCEPTABLE_AUC}
    if weak:
        print(f"\n✗ FAILED: {list(weak)} scored below the "
              f"{MIN_ACCEPTABLE_AUC:.2f} AUC floor ({weak}).")
        print("  (nan = no checkpoint on disk — that array task never "
              "finished; check logs/ and resubmit just that --arch.)")
        print("  Do NOT start PPO training against these. f(.) at chance means "
              "CV measures nothing,")
        print("  the Xn/Xp split is arbitrary, and every clinical metric is "
              "computed over noise.")
        print("  Try: more epochs, a lower LR, or check the label mapping in "
              "domains/cbis_ddsm_breast.py.")
        return 1

    print(f"\n✓ All backbones cleared the {MIN_ACCEPTABLE_AUC:.2f} AUC floor.")
    print(f"  Next: python check_classifier_localization.py   "
          f"(is f(.) looking at lesions or at the pectoral edge?)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    ap.add_argument("--arch", type=str, default=None,
                    help="train just this backbone (default: all of "
                         "config.BREAST_BACKBONES)")
    ap.add_argument("--summary", action="store_true",
                    help="train nothing; read the saved checkpoints and apply "
                         "the AUC floor across all of BREAST_BACKBONES")
    ap.add_argument("--skip-existing", action="store_true",
                    help="leave a backbone alone if its checkpoint already "
                         "exists (makes resubmitting the array cheap)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if args.summary:
        return report_and_gate(
            {a: _saved_auc(a) for a in config.BREAST_BACKBONES}
        )

    set_seed(args.seed)
    device = config.DEVICE if torch.cuda.is_available() else "cpu"
    print(f"[train-clf] device={device}  domain={config.DATASET_DOMAIN}")

    domain = domains.active()
    if domain.build_guidance is None:
        raise SystemExit(
            f"config.DATASET_DOMAIN={config.DATASET_DOMAIN!r} uses the "
            f"torchxrayvision backend, whose classifiers are already "
            f"pretrained — there is nothing for this script to train. It is "
            f"for domains that bring their own f(.), i.e. cbis_ddsm_breast."
        )

    archs = [args.arch] if args.arch else list(config.BREAST_BACKBONES)
    if args.skip_existing:
        done = [a for a in archs
                if os.path.exists(os.path.join(config.BREAST_CKPT_DIR, f"{a}.pt"))]
        for a in done:
            print(f"[train-clf] {a}: checkpoint already exists "
                  f"(val_AUC={_saved_auc(a):.4f}) — skipping")
        archs = [a for a in archs if a not in done]
    if not archs:
        print("[train-clf] nothing left to train.")
        return 0

    # Built after the skip check so a no-op array task does not spend minutes
    # scanning the mirror just to exit.
    dataset = domain.build_dataset(config.IMG_SIZE)
    train_idx, val_idx = patient_level_split(dataset, args.val_frac, args.seed)

    results = {
        a: train_one(a, dataset, train_idx, val_idx, args.epochs,
                     args.batch_size, args.lr, device, args.patience)
        for a in archs
    }
    return report_and_gate(results)


if __name__ == "__main__":
    raise SystemExit(main())
