"""
src/finetuning.py
=================
LoRA denoising fine-tuning loop for the SD1.5 UNet + cond_proj.

Trains the generator on the RSNA dataset using a standard diffusion
denoising objective (L_diff = MSE between predicted and actual noise),
with AMP (fp16 autocast) and gradient clipping for stability.


"""

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

import config


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


def _build_balanced_sampler(dataset) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler that gives each class roughly equal
    expected representation per epoch, regardless of the dataset's true
    class balance.
    """
    labels = _extract_labels(dataset)
    class_counts = np.bincount(labels, minlength=2)
    if (class_counts == 0).any():
        raise RuntimeError(
            f"[finetuning] one class has zero samples: counts={class_counts}. "
            f"Cannot build a balanced sampler."
        )

    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]

    print(
        f"[finetuning] class counts: negative={class_counts[0]}, "
        f"positive={class_counts[1]} | balanced sampling enabled "
        f"(effective ratio ~1:1 per epoch)"
    )

    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(dataset),
        replacement=True,
    )


def run_lora_finetuning(
    generator,
    dataset,
    n_epochs: int    = config.N_DENOISE_EPOCHS,
    batch_size: int  = config.DENOISE_BATCH_SIZE,
    lr: float        = config.DENOISE_LR,
    log_every: int   = config.LOG_EVERY_N_STEPS,
    num_workers: int = 2,
    use_balanced_sampling: bool = True,
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
        This directly targets the n_to_p / p_to_n counterfactual validity
        asymmetry caused by RSNA's ~3:1 negative:positive class imbalance.
    """
    trainable_params = [
        p for p in
        list(generator.cond_proj.parameters()) + list(generator.unet.parameters())
        if p.requires_grad
    ]
    print(f"[finetuning] trainable params: {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    scaler    = torch.amp.GradScaler('cuda')

    loader_kwargs = dict(
        batch_size  = batch_size,
        num_workers = num_workers,
        drop_last   = False,
    )

    if use_balanced_sampling:
        sampler = _build_balanced_sampler(dataset)
        train_loader = DataLoader(dataset, sampler=sampler, **loader_kwargs)
    else:
        train_loader = DataLoader(dataset, shuffle=True, **loader_kwargs)

    print(
        f"[finetuning] {len(dataset)} images | "
        f"{len(train_loader)} steps/epoch | batch_size={batch_size} | "
        f"balanced_sampling={use_balanced_sampling}"
    )

    for epoch in range(n_epochs):
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