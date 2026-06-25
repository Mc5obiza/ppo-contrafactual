"""
src/finetuning.py
=================
LoRA denoising fine-tuning loop for the SD1.5 UNet + cond_proj.

Trains the generator on the RSNA dataset using a standard diffusion
denoising objective (L_diff = MSE between predicted and actual noise),
with AMP (fp16 autocast) and gradient clipping for stability.
"""

import torch
from torch.utils.data import DataLoader

import config


def run_lora_finetuning(
    generator,
    dataset,
    n_epochs: int    = config.N_DENOISE_EPOCHS,
    batch_size: int  = config.DENOISE_BATCH_SIZE,
    lr: float        = config.DENOISE_LR,
    log_every: int   = config.LOG_EVERY_N_STEPS,
    num_workers: int = 2,
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
    """
    trainable_params = [
        p for p in
        list(generator.cond_proj.parameters()) + list(generator.unet.parameters())
        if p.requires_grad
    ]
    print(f"[finetuning] trainable params: {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    scaler    = torch.cuda.amp.GradScaler()

    train_loader = DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        drop_last   = False,
    )
    print(
        f"[finetuning] {len(dataset)} images | "
        f"{len(train_loader)} steps/epoch | batch_size={batch_size}"
    )

    for epoch in range(n_epochs):
        running_loss, n_valid_steps = 0.0, 0
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
            if step % log_every == 0:
                print(
                    f"  epoch {epoch} step {step:4d}/{len(train_loader)} "
                    f"| L_diff={loss.item():.4f}"
                )

        avg = running_loss / max(n_valid_steps, 1)
        print(
            f"[finetuning] epoch {epoch} done | "
            f"avg L_diff={avg:.4f} | valid_steps={n_valid_steps}/{len(train_loader)}"
        )
