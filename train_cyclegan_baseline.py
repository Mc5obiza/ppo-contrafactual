"""
train_cyclegan_baseline.py
===========================
Trains the Vanilla CycleGAN baseline referenced in the baseline-comparison
table (report/report.tex, report/paper_cvpr.tex) as a MEASURED arm instead of
a literature-reported number.

WHAT THIS IS
------------
A standard, unconditional, unguided image-to-image CycleGAN (Zhu et al. 2017
— see src/cyclegan.py for the architecture) trained to map between the two
GROUND-TRUTH classes (normal / pneumonia) of the active domain. No classifier
guidance, no Grad-CAM masking, no background lock, no PPO — this is exactly
the "diffusion backbone + gradient guidance is what earns the gains, not
merely translating one class into the other" control the report's Section
4-style comparison is meant to isolate.

WHAT IT REUSES FROM THE REST OF THE PIPELINE (so the comparison is apples-to-
apples, not a separately-built dataset)
------------------------------------------------------------------------------
  * The DICOM preprocessing pipeline (ingestion.RSNACXRDataset / domain
    profile's build_dataset): min-max normalisation, resize to
    config.IMG_SIZE, single-channel [0,1] tensors. IDENTICAL to every other
    arm — this script builds the dataset exactly the way pipeline.py does.
  * The persisted patient-level split (splits.load_or_create). This script
    NEVER calls splits.create() itself — it loads the same
    checkpoints/split_<domain>.json every LoRA/PPO run already reads, and
    trains only on split.train. Training on split.test would leak the fold
    every other arm is evaluated on.
  * Domain membership for the two CycleGAN domains is the RADIOLOGIST LABEL
    (dataset.samples[i][1]), not the guidance classifier's f(x) score — an
    unconditional GAN needs a ground-truth partition to learn a mapping
    between; f(x) thresholding is an EVALUATION-time construct (Xn/Xp
    groups), not a training-time one, and is untouched here.

TRAINING BUDGET (see the module docstring's benchmarking note and the
run_manifest.json this script writes for the numbers actually used)
------------------------------------------------------------------------------
publicgpu caps a single SLURM job at 24h. A benchmark of this exact
architecture at 256x256 (see PR discussion / run_manifest) measured
~110 ms/image on a V100 at batch_size=4, and one epoch is defined as one full
pass over the LARGER of the two domain-partitioned training splits (matching
the reference CycleGAN implementation's convention: the shorter domain is
cycled with modulo indexing rather than truncated), which is ~16.5k images for
RSNA's normal/pneumonia split — so a 24h budget fits roughly 40 epochs at
batch_size=4, well short of the original paper's 200-epoch horaepoch2zebra-
scale budget but a genuine, standard-architecture, trained-to-a-documented-
budget run rather than a hidden or undisclosed shortcut. If this undercounts
real DataLoader/DICOM-decode overhead, CYCLEGAN_WALLCLOCK_BUDGET_S below stops
the run cleanly and it resumes from its last checkpoint — the reported
"epochs completed" in run_manifest.json is always the ground truth, not the
--epochs argument.

RESUMABILITY
------------
Exactly the PPOTrainer / run_ppo.slurm pattern: writes
<ckpt_dir>/cyclegan_latest.pt every epoch (full optimizer + scheduler state,
not just the generators), and if CYCLEGAN_WALLCLOCK_BUDGET_S elapses mid-run
it finishes the current epoch, checkpoints, and exits 0 with a "PAUSED"
message rather than being killed mid-write by SLURM's walltime. Re-running
this script with the same --ckpt-dir resumes automatically.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
from torch.utils.data import DataLoader, Subset

import config
import domains
import splits
import seeding
from src.cyclegan import (
    ResnetGenerator, NLayerDiscriminator, GANLoss, ImagePool, to_gan_range,
)


def _fmt_hms(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s"


# ─────────────────────────────────────────────────────────────────────────────
# Domain-partitioned (unpaired) data
# ─────────────────────────────────────────────────────────────────────────────

def _label_partition(dataset, indices) -> tuple[list[int], list[int]]:
    """
    Split `indices` (dataset positions, e.g. split.train) into domain A
    (ground-truth negative / normal) and domain B (positive / pneumonia).

    Reads dataset.samples directly rather than calling dataset[i] — the label
    is already in memory (ingestion.RSNACXRDataset.samples), so this avoids
    decoding 21k DICOM files just to partition by class.
    """
    samples = getattr(dataset, "samples", None)
    if samples is None:
        raise AttributeError(
            f"{type(dataset).__name__} exposes no .samples list of "
            f"(id, label) pairs — this baseline is implemented for the chest "
            f"radiograph domain contract (ingestion.RSNACXRDataset); a domain "
            f"without that attribute needs an explicit label lookup added here."
        )
    idx_a = [int(i) for i in indices if samples[i][1] < 0.5]
    idx_b = [int(i) for i in indices if samples[i][1] >= 0.5]
    return idx_a, idx_b


class _CycledSubset(torch.utils.data.Dataset):
    """Wraps a Subset so indexing past its own length wraps (modulo) — lets the
    shorter domain be iterated the same number of times as the longer one
    within one epoch, matching the reference CycleGAN unaligned-dataset
    convention instead of truncating every epoch to the smaller domain."""

    def __init__(self, subset: Subset, target_len: int):
        self.subset = subset
        self.target_len = target_len

    def __len__(self):
        return self.target_len

    def __getitem__(self, idx):
        return self.subset[idx % len(self.subset)]


def build_loaders(dataset, indices, batch_size: int, num_workers: int, seed: int):
    idx_a, idx_b = _label_partition(dataset, indices)
    n = max(len(idx_a), len(idx_b))
    print(f"[data] domain A (normal)   : {len(idx_a)} train images")
    print(f"[data] domain B (pneumonia): {len(idx_b)} train images")
    print(f"[data] one epoch = {n} images per domain (the larger domain; "
          f"the shorter one is cycled)")

    sub_a = _CycledSubset(Subset(dataset, idx_a), n)
    sub_b = _CycledSubset(Subset(dataset, idx_b), n)

    kwargs = seeding.loader_kwargs(seed)
    loader_a = DataLoader(sub_a, batch_size=batch_size, shuffle=True,
                           drop_last=True, num_workers=num_workers, **kwargs)
    loader_b = DataLoader(sub_b, batch_size=batch_size, shuffle=True,
                           drop_last=True, num_workers=num_workers, **kwargs)
    return loader_a, loader_b, n


# ─────────────────────────────────────────────────────────────────────────────
# Checkpointing
# ─────────────────────────────────────────────────────────────────────────────

def _ckpt_path(ckpt_dir: str) -> str:
    return os.path.join(ckpt_dir, "cyclegan_latest.pt")


def _save_ckpt(ckpt_dir, epoch, G_AB, G_BA, D_A, D_B, opt_G, opt_D, sched_G, sched_D,
               elapsed_s, hparams):
    os.makedirs(ckpt_dir, exist_ok=True)
    payload = {
        "epoch": epoch,
        "G_AtoB": G_AB.state_dict(), "G_BtoA": G_BA.state_dict(),
        "D_A": D_A.state_dict(), "D_B": D_B.state_dict(),
        "opt_G": opt_G.state_dict(), "opt_D": opt_D.state_dict(),
        "sched_G": sched_G.state_dict(), "sched_D": sched_D.state_dict(),
        "elapsed_s": elapsed_s,
        "hparams": hparams,
    }
    tmp = f"{_ckpt_path(ckpt_dir)}.{os.getpid()}.tmp"
    torch.save(payload, tmp)
    os.replace(tmp, _ckpt_path(ckpt_dir))
    # Plain-state-dict copies for the eval script — it only ever needs the two
    # generators, not optimizer/discriminator state.
    torch.save(G_AB.state_dict(), os.path.join(ckpt_dir, "G_AtoB.pt"))
    torch.save(G_BA.state_dict(), os.path.join(ckpt_dir, "G_BtoA.pt"))


def _load_ckpt(ckpt_dir, G_AB, G_BA, D_A, D_B, opt_G, opt_D, sched_G, sched_D):
    path = _ckpt_path(ckpt_dir)
    if not os.path.exists(path):
        return 0, 0.0
    ckpt = torch.load(path, map_location=config.DEVICE, weights_only=False)
    G_AB.load_state_dict(ckpt["G_AtoB"]); G_BA.load_state_dict(ckpt["G_BtoA"])
    D_A.load_state_dict(ckpt["D_A"]); D_B.load_state_dict(ckpt["D_B"])
    opt_G.load_state_dict(ckpt["opt_G"]); opt_D.load_state_dict(ckpt["opt_D"])
    sched_G.load_state_dict(ckpt["sched_G"]); sched_D.load_state_dict(ckpt["sched_D"])
    print(f"[resume] loaded {path} at epoch {ckpt['epoch']} "
          f"({_fmt_hms(ckpt['elapsed_s'])} already spent)")
    return ckpt["epoch"], ckpt["elapsed_s"]


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(
    ckpt_dir: str,
    n_epochs: int = 40,
    decay_start_epoch: int | None = None,
    batch_size: int = 4,
    lr: float = 2e-4,
    lambda_cycle: float = 10.0,
    lambda_identity: float = 0.5,
    pool_size: int = 50,
    ngf: int = 64,
    ndf: int = 64,
    n_blocks: int = 9,
    seed: int | None = None,
    wallclock_budget_s: float = float("inf"),
) -> dict:
    decay_start_epoch = decay_start_epoch if decay_start_epoch is not None else n_epochs // 2
    seed = seeding.seed_everything(seed, tag="cyclegan_baseline")

    domain = domains.active()
    print(domain.summary())
    dataset = domain.build_dataset(config.IMG_SIZE) if domain.name != "rsna_pneumonia" \
        else __import__("ingestion").build_dataset(img_size=config.IMG_SIZE)

    split = splits.load_or_create(dataset, domain=domain.name)
    loader_a, loader_b, epoch_len = build_loaders(
        dataset, split.train, batch_size, config.DATALOADER_NUM_WORKERS, seed,
    )

    device = config.DEVICE
    G_AB = ResnetGenerator(ngf=ngf, n_blocks=n_blocks).to(device)
    G_BA = ResnetGenerator(ngf=ngf, n_blocks=n_blocks).to(device)
    D_A = NLayerDiscriminator(ndf=ndf).to(device)
    D_B = NLayerDiscriminator(ndf=ndf).to(device)
    gan_loss = GANLoss().to(device)
    l1 = torch.nn.L1Loss()
    pool_a, pool_b = ImagePool(pool_size), ImagePool(pool_size)

    opt_G = torch.optim.Adam(list(G_AB.parameters()) + list(G_BA.parameters()),
                              lr=lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(list(D_A.parameters()) + list(D_B.parameters()),
                              lr=lr, betas=(0.5, 0.999))

    def _lr_lambda(epoch):
        if epoch < decay_start_epoch:
            return 1.0
        span = max(1, n_epochs - decay_start_epoch)
        return max(0.0, 1.0 - (epoch - decay_start_epoch) / span)

    sched_G = torch.optim.lr_scheduler.LambdaLR(opt_G, lr_lambda=_lr_lambda)
    sched_D = torch.optim.lr_scheduler.LambdaLR(opt_D, lr_lambda=_lr_lambda)

    hparams = dict(
        architecture="ResnetGenerator(9 resblocks, ngf=64, InstanceNorm) + "
                     "70x70 PatchGAN NLayerDiscriminator(ndf=64)",
        n_epochs=n_epochs, decay_start_epoch=decay_start_epoch,
        batch_size=batch_size, lr=lr, betas=[0.5, 0.999],
        lambda_cycle=lambda_cycle, lambda_identity=lambda_identity,
        pool_size=pool_size, ngf=ngf, ndf=ndf, n_blocks=n_blocks,
        img_size=config.IMG_SIZE, seed=seed, epoch_len=epoch_len,
        domain=domain.name, split_seed=split.seed, split_test_frac=split.test_frac,
        n_train_A=len(_label_partition(dataset, split.train)[0]),
        n_train_B=len(_label_partition(dataset, split.train)[1]),
    )

    start_epoch, elapsed_s = _load_ckpt(
        ckpt_dir, G_AB, G_BA, D_A, D_B, opt_G, opt_D, sched_G, sched_D,
    )
    if start_epoch >= n_epochs:
        print(f"[train] already at epoch {start_epoch} >= n_epochs={n_epochs}, nothing to do")
        return hparams | {"epochs_completed": start_epoch, "elapsed_s": elapsed_s, "paused": False}

    t_run_start = time.time()
    paused = False
    for epoch in range(start_epoch, n_epochs):
        t_epoch = time.time()
        running = dict(G=0.0, D=0.0, cycle=0.0, idt=0.0, n=0)
        for real_a, real_b in zip(loader_a, loader_b):
            real_a = to_gan_range(real_a[0].to(device))
            real_b = to_gan_range(real_b[0].to(device))

            # ---- generators ----
            for p in list(D_A.parameters()) + list(D_B.parameters()):
                p.requires_grad_(False)
            opt_G.zero_grad()
            fake_b = G_AB(real_a)
            rec_a = G_BA(fake_b)
            fake_a = G_BA(real_b)
            rec_b = G_AB(fake_a)
            idt_a = G_BA(real_a)   # G_BtoA(normal) should stay ~normal
            idt_b = G_AB(real_b)   # G_AtoB(pneumonia) should stay ~pneumonia

            loss_adv = gan_loss(D_B(fake_b), True) + gan_loss(D_A(fake_a), True)
            loss_cycle = lambda_cycle * (l1(rec_a, real_a) + l1(rec_b, real_b))
            loss_idt = (lambda_cycle * lambda_identity) * (l1(idt_a, real_a) + l1(idt_b, real_b))
            loss_G = loss_adv + loss_cycle + loss_idt
            loss_G.backward()
            opt_G.step()

            # ---- discriminators ----
            for p in list(D_A.parameters()) + list(D_B.parameters()):
                p.requires_grad_(True)
            opt_D.zero_grad()
            fake_a_pooled = pool_a.query(fake_a.detach())
            fake_b_pooled = pool_b.query(fake_b.detach())
            loss_D_A = 0.5 * (gan_loss(D_A(real_a), True) + gan_loss(D_A(fake_a_pooled), False))
            loss_D_B = 0.5 * (gan_loss(D_B(real_b), True) + gan_loss(D_B(fake_b_pooled), False))
            loss_D = loss_D_A + loss_D_B
            loss_D.backward()
            opt_D.step()

            running["G"] += float(loss_adv.item())
            running["D"] += float(loss_D.item())
            running["cycle"] += float(loss_cycle.item())
            running["idt"] += float(loss_idt.item())
            running["n"] += 1

        sched_G.step()
        sched_D.step()
        elapsed_s += time.time() - t_epoch
        n = max(1, running["n"])
        print(f"[epoch {epoch+1}/{n_epochs}] "
              f"adv={running['G']/n:.4f} D={running['D']/n:.4f} "
              f"cycle={running['cycle']/n:.4f} idt={running['idt']/n:.4f} "
              f"lr={sched_G.get_last_lr()[0]:.2e} "
              f"({_fmt_hms(time.time()-t_epoch)}, total {_fmt_hms(elapsed_s)})")

        _save_ckpt(ckpt_dir, epoch + 1, G_AB, G_BA, D_A, D_B, opt_G, opt_D,
                   sched_G, sched_D, elapsed_s, hparams)

        if elapsed_s >= wallclock_budget_s:
            print(f"[train] TRAINING PAUSED at epoch {epoch+1}/{n_epochs} "
                  f"— wallclock budget ({_fmt_hms(wallclock_budget_s)}) reached. "
                  f"Re-run this script with the same --ckpt-dir to resume.")
            paused = True
            break

    epochs_completed = epoch + 1
    record = hparams | {
        "epochs_completed": epochs_completed,
        "elapsed_s": elapsed_s,
        "paused": paused,
    }
    with open(os.path.join(ckpt_dir, "cyclegan_train_manifest.json"), "w") as f:
        json.dump(record, f, indent=2)
    print(f"\n[train] {'PAUSED' if paused else 'DONE'} after {epochs_completed} epochs, "
          f"{_fmt_hms(elapsed_s)} total. Manifest written to "
          f"{ckpt_dir}/cyclegan_train_manifest.json")
    return record


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-dir", default="checkpoints_cyclegan_baseline")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--decay-start-epoch", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--wallclock-budget-s", type=float,
                     default=float(os.environ.get("CYCLEGAN_WALLCLOCK_BUDGET_S", "inf")))
    return ap.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    train(
        ckpt_dir=args.ckpt_dir,
        n_epochs=args.epochs,
        decay_start_epoch=args.decay_start_epoch,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        wallclock_budget_s=args.wallclock_budget_s,
    )
