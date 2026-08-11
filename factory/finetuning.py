"""
src/finetuning.py
=================
LoRA denoising fine-tuning loop for the SD1.5 UNet + cond_proj.

Trains the generator on the RSNA dataset using a standard diffusion
denoising objective (L_diff = MSE between predicted and actual noise),
with AMP (fp16 autocast) and gradient clipping for stability.

CHANGE FROM ORIGINAL: added per-epoch checkpointing + resume support.
For long server runs (25+ epochs), losing everything to a walltime
cutoff or job preemption is expensive — this saves LoRA weights,
cond_proj, optimizer, and scaler state after every epoch, and can
resume from the latest checkpoint automatically if the job restarts.
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, WeightedRandomSampler
from peft import get_peft_model_state_dict, set_peft_model_state_dict

import config
import seeding


def _extract_labels(dataset) -> np.ndarray:
    """
    Pull the binary label for every sample in the dataset as cheaply as
    possible, without decoding full images if avoidable.

    Tries, in order:
      1. A `.labels` / `.targets` attribute (array-like, one entry per sample)
      2. A `.samples` / `.df` attribute with a recognizable label column
      3. Falls back to iterating `dataset[i]` and reading the second element
         (slow — only used if nothing else is available)

    Returns
    -------
    np.ndarray of shape (N,), values in {0, 1}
    """
    for attr in ("labels", "targets"):
        if hasattr(dataset, attr):
            arr = np.asarray(getattr(dataset, attr)).reshape(-1)
            print(f"[finetuning] using dataset.{attr} for label extraction")
            return arr.astype(int)

    # RSNACXRDataset (and similar) store (id, label) tuples in .samples —
    # label is always the last element, no image decode needed.
    if hasattr(dataset, "samples"):
        samples = dataset.samples
        if len(samples) > 0 and isinstance(samples[0], (tuple, list)) and len(samples[0]) >= 2:
            print("[finetuning] using dataset.samples for label extraction (no DICOM decode needed)")
            return np.array([s[-1] for s in samples]).astype(int)

    if hasattr(dataset, "df"):
        df = dataset.df
        for col in ("label", "target", "class", "Target"):
            if col in getattr(df, "columns", []):
                print(f"[finetuning] using dataset.df['{col}'] for label extraction")
                return df[col].to_numpy().astype(int)

    print(
        "[finetuning] WARNING: no labels/targets/df attribute found on dataset — "
        "falling back to full __getitem__ scan to build the label array. "
        "This is slow for large/lazy-loading datasets. Consider exposing "
        "a `.labels` array on your Dataset class to speed this up."
    )
    labels = np.empty(len(dataset), dtype=int)
    for i in range(len(dataset)):
        _x, c = dataset[i]
        labels[i] = int(c.item() if torch.is_tensor(c) else c)
    return labels


def _build_balanced_sampler(dataset, indices=None) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler that gives each class roughly equal
    expected representation per epoch, regardless of the dataset's true
    class balance.

    `indices` restricts sampling to a subset (the training fold). It is applied
    by ZEROING the weight of every excluded sample rather than by wrapping the
    dataset in a Subset: _extract_labels reads `dataset.samples` to avoid
    decoding 21k DICOMs, and a Subset does not carry that attribute, so wrapping
    would silently fall back to the slow per-image path. Zero-weight entries are
    never drawn by WeightedRandomSampler, so the effect is identical.
    """
    labels = _extract_labels(dataset)

    if indices is None:
        eligible = np.arange(len(labels))
    else:
        eligible = np.asarray(indices, dtype=np.int64)

    # Balance against the classes present in the SUBSET, not the whole dataset —
    # otherwise the weights encode the full set's balance and the fold is not
    # actually balanced.
    class_counts = np.bincount(labels[eligible], minlength=2)
    if (class_counts == 0).any():
        raise RuntimeError(
            f"[finetuning] one class has zero samples in the sampled subset: "
            f"counts={class_counts}. Cannot build a balanced sampler."
        )

    class_weights  = 1.0 / class_counts
    sample_weights = np.zeros(len(labels), dtype=np.float64)
    sample_weights[eligible] = class_weights[labels[eligible]]

    print(
        f"[finetuning] class counts: negative={class_counts[0]}, "
        f"positive={class_counts[1]} | balanced sampling enabled "
        f"(effective ratio ~1:1 per epoch) | drawing from {len(eligible)} "
        f"of {len(labels)} images"
    )

    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(eligible),
        replacement=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Checkpointing
# ─────────────────────────────────────────────────────────────────────────────

def _checkpoint_path(ckpt_dir: str) -> str:
    return os.path.join(ckpt_dir, "finetuning_latest.pt")


def _save_checkpoint(
    ckpt_dir: str,
    epoch: int,
    generator,
    optimizer,
    scaler,
) -> None:
    """
    Save everything needed to resume training after this epoch:
      - LoRA adapter weights (via peft's get_peft_model_state_dict —
        much smaller than the full UNet state_dict, since only LoRA
        params are trainable)
      - cond_proj weights
      - optimizer + scaler state (so resumed training has correct
        momentum/AMP scale, not a cold restart)
      - the epoch index itself, so resume knows where to continue

    Overwrites a single 'finetuning_latest.pt' each epoch rather than
    keeping one file per epoch — keeps disk usage flat across a long run.
    Change to an epoch-numbered filename below if you want to keep every
    epoch's checkpoint instead (uses more disk).
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    path = _checkpoint_path(ckpt_dir)

    checkpoint = {
        "epoch": epoch,
        "lora_state_dict": get_peft_model_state_dict(generator.unet),
        "cond_proj_state_dict": generator.cond_proj.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
    }
    # atomic-ish write: save to a temp file then rename, so a crash mid-save
    # doesn't corrupt the checkpoint you'd resume from.
    #
    # The PID in the temp name matters because ckpt_dir is config.CKPT_DIR —
    # SHARED across directions, deliberately, so a later run resumes as a no-op.
    # run_ppo.slurm runs both directions as a simultaneous array, and the two
    # tasks are seeded identically, so they reach each epoch boundary within
    # seconds of each other and would both be writing the same ".tmp" path.
    # torch.save from two processes into one file interleaves, and os.replace
    # would then promote the resulting garbage — breaking resume for both
    # directions while the run itself looked healthy (nothing re-reads this file
    # until a resubmission). Per-process temp files make the loser's rename a
    # harmless overwrite instead.
    #
    # The surviving file is EQUIVALENT, not bit-identical: measured on job
    # 17686489, the two tasks agreed at 100 of 101 logged steps and differed by
    # one ulp at the other, because STRICT_DETERMINISM defaults to False. Either
    # file is a valid state of the same training run, so resume is correct
    # whichever lands — but do not read this as bit-reproducibility.
    tmp_path = f"{path}.{os.getpid()}.tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, path)
    print(f"[finetuning] checkpoint saved: {path} (epoch {epoch})")


def _load_checkpoint(ckpt_dir: str, generator, optimizer, scaler) -> int:
    """
    Load the latest checkpoint if one exists, restoring LoRA weights,
    cond_proj, optimizer, and scaler state.

    Returns
    -------
    int : the epoch to resume FROM (i.e. training should start at
          `resume_epoch`, since `epoch` in the checkpoint was already
          completed). Returns 0 if no checkpoint was found (fresh start).
    """
    path = _checkpoint_path(ckpt_dir)
    if not os.path.exists(path):
        print(f"[finetuning] no checkpoint found at {path} — starting fresh")
        return 0

    print(f"[finetuning] found checkpoint at {path} — resuming")
    checkpoint = torch.load(path, map_location=config.DEVICE)

    try:
        set_peft_model_state_dict(generator.unet, checkpoint["lora_state_dict"])
    except (RuntimeError, ValueError, KeyError) as err:
        # LoRA A/B matrices are shaped (r, in) / (out, r), so a LORA_R or
        # LORA_TARGETS change makes a saved adapter unloadable. Resuming is
        # exactly where that bites, and the raw error does not say why.
        raise RuntimeError(
            f"[finetuning] cannot resume '{path}' into the adapter this config "
            f"builds (LORA_R={config.LORA_R}, LORA_TARGETS={config.LORA_TARGETS}).\n"
            f"The checkpoint was written under a different LoRA rank or target "
            f"set; those are baked into the adapter tensor shapes.\n"
            f"To start the new adapter from scratch, move the checkpoint aside:\n"
            f"    mv {path} {os.path.join(os.path.dirname(path), 'archive_v1_rank8', '')}\n"
            f"Original error:\n{err}"
        ) from err
    generator.cond_proj.load_state_dict(checkpoint["cond_proj_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])

    resume_epoch = checkpoint["epoch"] + 1
    print(f"[finetuning] resumed from epoch {checkpoint['epoch']} — "
          f"continuing at epoch {resume_epoch}")
    return resume_epoch


def run_lora_finetuning(
    generator,
    dataset,
    n_epochs: int    = config.N_DENOISE_EPOCHS,
    batch_size: int  = config.DENOISE_BATCH_SIZE,
    lr: float        = config.DENOISE_LR,
    log_every: int   = config.LOG_EVERY_N_STEPS,
    num_workers: int = config.DATALOADER_NUM_WORKERS,
    use_balanced_sampling: bool = True,
    ckpt_dir: str = config.CKPT_DIR,
    resume: bool = True,
    indices=None,
) -> None:
    """
    Fine-tune the generator's UNet LoRA adapters and cond_proj on the given dataset.

    Parameters
    ----------
    generator : PretrainedLatentDiffusionGenerator
    dataset : RSNACXRDataset
    n_epochs : int
    batch_size : int
    lr : float
    log_every : int
    num_workers : int
    use_balanced_sampling : bool
        If True (default), uses a WeightedRandomSampler so each epoch sees
        an approximately equal number of positive and negative examples.
    ckpt_dir : str
        Directory to save/load the per-epoch checkpoint
        ('finetuning_latest.pt' inside this dir).
    resume : bool
        If True (default), automatically resumes from the latest
        checkpoint in ckpt_dir if one exists — safe to leave on even for
        a fresh run (it just won't find anything to resume from). Set to
        False to always start from scratch and overwrite any existing
        checkpoint.
    """
    trainable_params = [
        p for p in
        list(generator.cond_proj.parameters()) + list(generator.unet.parameters())
        if p.requires_grad
    ]
    print(f"[finetuning] trainable params: {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    scaler    = torch.amp.GradScaler('cuda')

    start_epoch = 0
    if resume:
        start_epoch = _load_checkpoint(ckpt_dir, generator, optimizer, scaler)
        if start_epoch >= n_epochs:
            print(f"[finetuning] checkpoint already at epoch {start_epoch} >= "
                  f"n_epochs={n_epochs} — nothing to do")
            return

    loader_kwargs = dict(
        batch_size  = batch_size,
        num_workers = num_workers,
        drop_last   = False,
        # Workers seed themselves from entropy otherwise, so shuffling differs
        # per run even after seed_everything. See seeding.loader_kwargs.
        **seeding.loader_kwargs(),
    )

    # THE LEAK THIS CLOSES. This used to train on the whole dataset, including
    # the images the evaluation later reported on — so the generator had already
    # fitted the test images before FID and IP were measured against them.
    # `indices` is the training fold from splits.py; None means the caller
    # deliberately wants everything (config.USE_SPLIT=False).
    n_sampled = len(dataset) if indices is None else len(indices)
    if use_balanced_sampling:
        sampler = _build_balanced_sampler(dataset, indices=indices)
        train_loader = DataLoader(dataset, sampler=sampler, **loader_kwargs)
    elif indices is not None:
        train_loader = DataLoader(
            dataset, sampler=SubsetRandomSampler(list(map(int, indices))), **loader_kwargs
        )
    else:
        train_loader = DataLoader(dataset, shuffle=True, **loader_kwargs)

    print(
        f"[finetuning] {n_sampled} images"
        f"{'' if indices is None else f' (of {len(dataset)}; held-out fold excluded)'} | "
        f"{len(train_loader)} steps/epoch | batch_size={batch_size} | "
        f"balanced_sampling={use_balanced_sampling} | "
        f"epochs {start_epoch}->{n_epochs - 1}"
    )

    for epoch in range(start_epoch, n_epochs):
        running_loss, n_valid_steps = 0.0, 0
        running_loss_pos, n_valid_pos = 0.0, 0
        running_loss_neg, n_valid_neg = 0.0, 0
        print(f"\n[finetuning] epoch {epoch}/{n_epochs - 1} | {len(train_loader)} steps")

        for step, (x_batch, c_batch) in enumerate(train_loader):
            x_batch = x_batch.to(config.DEVICE)
            c_batch = c_batch.to(config.DEVICE)

            # Skip corrupted batches
            if not torch.isfinite(x_batch).all():
                print(f"  epoch {epoch} step {step:4d} | skipping non-finite batch")
                continue

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                losses = generator.compute_training_losses(x_batch, c_batch)
                loss   = losses["L_diff"]

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  epoch {epoch} step {step:4d} | skipping non-finite loss")
                continue

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss  += loss.item()
            n_valid_steps += 1

            # Track per-class loss so you can directly verify the fix is
            # working: pos and neg avg loss should converge over epochs
            # rather than one dominating.
            pos_frac = c_batch.view(-1).float().mean().item()
            if pos_frac > 0.5:
                running_loss_pos += loss.item()
                n_valid_pos += 1
            else:
                running_loss_neg += loss.item()
                n_valid_neg += 1

            if step % log_every == 0:
                print(
                    f"  epoch {epoch} step {step:4d}/{len(train_loader)} "
                    f"| L_diff={loss.item():.4f} | batch_pos_frac={pos_frac:.2f}"
                )

        avg = running_loss / max(n_valid_steps, 1)
        avg_pos = running_loss_pos / max(n_valid_pos, 1)
        avg_neg = running_loss_neg / max(n_valid_neg, 1)
        print(
            f"[finetuning] epoch {epoch} done | "
            f"avg L_diff={avg:.4f} | valid_steps={n_valid_steps}/{len(train_loader)} | "
            f"avg_L_diff_pos_batches={avg_pos:.4f} (n={n_valid_pos}) | "
            f"avg_L_diff_neg_batches={avg_neg:.4f} (n={n_valid_neg})"
        )

        # save after EVERY epoch — cheap relative to an epoch's training
        # time, and this is exactly what protects a long HPC run against
        # a walltime cutoff or preemption killing the job mid-run
        _save_checkpoint(ckpt_dir, epoch, generator, optimizer, scaler)