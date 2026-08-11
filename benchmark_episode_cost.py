"""
benchmark_episode_cost.py
=========================
Measure what one PPO episode and one LoRA epoch actually cost on the GPU this
job landed on, then size N_EPISODES / N_DENOISE_EPOCHS against the 24h cap.

WHY THIS EXISTS
---------------
config.py says so, in as many words, above N_EPISODES:

    ACTION REQUIRED before the real run: benchmark ~16 episodes first ...
    800 below is a conservative placeholder until you have that number —
    don't treat it as tuned.

That benchmark was never run, and the cost of not running it is on the record:
six jobs have been killed at their walltime, including two full PPO runs
(17644171, 17653310) that hit 24h and, in one case, left nothing recoverable.

The reason it matters MORE now than when that comment was written: the
2026-08-07 revert restored BUDGET_T=25 with FIXED_EPISODE_LENGTH=False and
EDIT_BUDGET_L1=inf, so an episode runs anywhere from 1 to 25 steps depending on
how fast the classifier flips. config.py's own note says predictable per-episode
cost "is the thing that makes a 24h walltime impossible to plan against" — and
that property was reverted away. So this measures the DISTRIBUTION, not just a
mean, and sizes against a high quantile.

WHAT IT DOES NOT DO
-------------------
It does not write any checkpoint, does not touch config, and does not train
anything it keeps. It runs one direction unless told otherwise, because the two
differ in cost (n→p has historically been the slower, harder direction) and the
whole point is to see that rather than average it away.

USAGE
-----
    python -u benchmark_episode_cost.py                      # 16 episodes, n_to_p
    python -u benchmark_episode_cost.py --episodes 24 --direction p_to_n
    python -u benchmark_episode_cost.py --skip-lora          # PPO cost only
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch

import config
import seeding
from pipeline import build_common
from factory.environment import MedicalCounterfactualEnv
from factory.ppo import PPOActorCritic, PPOTrainer


DIRECTIONS = {"n_to_p": 1.0, "p_to_n": 0.0}


def _fmt_hms(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def benchmark_ppo(ctx, direction: str, n_episodes: int) -> dict:
    """Time `n_episodes` rollouts one at a time, so per-episode spread is visible."""
    target_c = DIRECTIONS[direction]

    print("\n" + "=" * 70)
    print(f"PPO episode cost — {direction} (target_c={target_c})")
    print("=" * 70)

    t0 = time.time()
    env = MedicalCounterfactualEnv(
        dataset=ctx.dataset,
        classifier=ctx.classifier,
        target_c=target_c,
        generator=ctx.generator,
        seg_net=ctx.seg_net,
        obj_det=ctx.obj_det,
        budget_T=config.BUDGET_T,
        device=config.DEVICE,
        allowed_indices=ctx.train_indices(),
    )
    scan_s = time.time() - t0
    print(f"[bench] far-pool scan: {_fmt_hms(scan_s)} (one-off, per direction)")

    obs_dim = env.reset().shape[0]
    policy = PPOActorCritic(obs_dim=obs_dim, action_dim=config.ACTION_DIM).to(config.DEVICE)
    trainer = PPOTrainer(env, policy, lr=config.LR)

    per_episode, ep_lengths = [], []
    for i in range(n_episodes):
        t = time.time()
        traj = trainer.collect_rollout(n_episodes=1)
        dt = time.time() - t
        per_episode.append(dt)
        ep_lengths.append(len(traj))
        print(f"[bench] episode {i + 1}/{n_episodes}: {dt:6.1f}s  "
              f"({len(traj)} steps, {dt / max(len(traj), 1):5.1f}s/step)")

    # One gradient update, timed ALONE. Collecting the batch has to happen
    # first, but timing the two together double-counts rollout cost: the caller
    # already adds mean episode cost per episode and then amortises this figure
    # on top, so a combined measurement charges every episode roughly twice.
    # (The first run of this script did exactly that — it reported 85.4s "per
    # update" for n_to_p, which was ~8 episodes of rollout plus a gradient step
    # that turns out to be nearly free. The resulting N_EPISODES ceiling was
    # therefore about 2x too conservative, in the safe direction.)
    batch = trainer.collect_rollout(n_episodes=min(8, n_episodes))
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t = time.time()
    trainer.update(batch)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    update_s = time.time() - t

    per_episode.sort()
    return {
        "direction": direction,
        "scan_s": scan_s,
        "mean_s": statistics.mean(per_episode),
        "median_s": statistics.median(per_episode),
        # Size against a high quantile, not the mean: episode cost is
        # right-skewed under BUDGET_T=25 (most flip in one step, some run the
        # full budget), and a mean-sized run is a coin flip against the cap.
        "p90_s": per_episode[min(len(per_episode) - 1, int(0.9 * len(per_episode)))],
        "max_s": per_episode[-1],
        "mean_steps": statistics.mean(ep_lengths),
        "max_steps": max(ep_lengths),
        "update_s": update_s,
        "n": n_episodes,
    }


def benchmark_lora(ctx, n_batches: int) -> dict:
    """Time a few LoRA training batches and extrapolate one epoch."""
    from torch.utils.data import DataLoader, SubsetRandomSampler

    print("\n" + "=" * 70)
    print("LoRA epoch cost")
    print("=" * 70)

    train_idx = [int(i) for i in ctx.train_indices()]
    loader = DataLoader(
        ctx.dataset,
        batch_size=config.DENOISE_BATCH_SIZE,
        sampler=SubsetRandomSampler(train_idx),
        num_workers=config.DATALOADER_NUM_WORKERS,
        **seeding.loader_kwargs(),
    )
    steps_per_epoch = len(loader)

    params = [p for p in ctx.generator.unet.parameters() if p.requires_grad]
    params += list(ctx.generator.cond_proj.parameters())
    opt = torch.optim.AdamW(params, lr=config.DENOISE_LR)
    scaler = torch.amp.GradScaler("cuda")

    times = []
    for i, (x, y) in enumerate(loader):
        if i >= n_batches:
            break
        t = time.time()
        opt.zero_grad(set_to_none=True)
        # Mirrors run_lora_finetuning's inner step exactly (autocast fp16,
        # L_diff off generator.compute_training_losses, scaled backward) — a
        # benchmark of a different step would not be a benchmark of this loop.
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            losses = ctx.generator.compute_training_losses(
                x.to(config.DEVICE), y.to(config.DEVICE)
            )
            loss = losses["L_diff"]
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        torch.cuda.synchronize()
        dt = time.time() - t
        times.append(dt)
        print(f"[bench] lora batch {i + 1}/{n_batches}: {dt:6.2f}s")

    # Drop the first: it carries cuDNN autotune and allocator warm-up that no
    # later batch pays, and including it inflates a whole-epoch estimate by the
    # number of steps in the epoch.
    steady = times[1:] or times
    per_batch = statistics.mean(steady)
    return {
        "per_batch_s": per_batch,
        "steps_per_epoch": steps_per_epoch,
        "epoch_s": per_batch * steps_per_epoch,
        "n_train": len(train_idx),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=16,
                    help="episodes to time (default 16, per config.py's note)")
    ap.add_argument("--direction", choices=("n_to_p", "p_to_n", "both"),
                    default="n_to_p")
    ap.add_argument("--lora-batches", type=int, default=6)
    ap.add_argument("--skip-lora", action="store_true")
    ap.add_argument("--budget-hours", type=float, default=22.0,
                    help="usable wall clock (default 22h — the 24h cap minus "
                         "the margin run_ppo.slurm reserves)")
    args = ap.parse_args()

    seeding.seed_everything(tag="benchmark")
    ctx = build_common()

    lora = None if args.skip_lora else benchmark_lora(ctx, args.lora_batches)

    directions = list(DIRECTIONS) if args.direction == "both" else [args.direction]
    ppo = [benchmark_ppo(ctx, d, args.episodes) for d in directions]

    # ── report ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    print(f"config: BUDGET_T={config.BUDGET_T}  T_DIFF={config.T_DIFF}  "
          f"ENV_REFINEMENT_PASSES={config.ENV_REFINEMENT_PASSES}  "
          f"FIXED_EPISODE_LENGTH={config.FIXED_EPISODE_LENGTH}")

    budget_s = args.budget_hours * 3600
    lora_s = 0.0
    if lora:
        lora_s = lora["epoch_s"] * config.N_DENOISE_EPOCHS
        print(f"\nLoRA: {lora['per_batch_s']:.2f}s/batch x {lora['steps_per_epoch']} "
              f"steps = {_fmt_hms(lora['epoch_s'])}/epoch")
        print(f"      {config.N_DENOISE_EPOCHS} epochs = {_fmt_hms(lora_s)} "
              f"({100 * lora_s / budget_s:.0f}% of the {args.budget_hours:g}h budget)")

    for r in ppo:
        print(f"\n{r['direction']}:")
        print(f"  episode cost : mean {r['mean_s']:.1f}s | median {r['median_s']:.1f}s "
              f"| p90 {r['p90_s']:.1f}s | max {r['max_s']:.1f}s   (n={r['n']})")
        print(f"  episode len  : mean {r['mean_steps']:.2f} steps | max {r['max_steps']}")
        print(f"  update       : {r['update_s']:.1f}s per 8 episodes")
        print(f"  far-pool scan: {_fmt_hms(r['scan_s'])}")

        avail = budget_s - lora_s - r["scan_s"]
        if avail <= 0:
            print("  ** LoRA alone exceeds the budget — reduce N_DENOISE_EPOCHS first.")
            continue
        # Amortise the per-8-episode update cost onto each episode.
        per_ep_mean = r["mean_s"] + r["update_s"] / 8.0
        per_ep_p90  = r["p90_s"]  + r["update_s"] / 8.0
        print(f"  --> N_EPISODES <= {int(avail / per_ep_mean)} at mean cost, "
              f"{int(avail / per_ep_p90)} at p90 cost")
        print(f"      current N_EPISODES={config.N_EPISODES} would take "
              f"{_fmt_hms(config.N_EPISODES * per_ep_mean + lora_s + r['scan_s'])} "
              f"(mean) / "
              f"{_fmt_hms(config.N_EPISODES * per_ep_p90 + lora_s + r['scan_s'])} (p90)")
        verdict = (
            "FITS" if config.N_EPISODES * per_ep_p90 + lora_s + r["scan_s"] < budget_s
            else "DOES NOT FIT — lower N_EPISODES or BUDGET_T"
        )
        print(f"      verdict: {verdict}")

    print("\nSize against the p90 column, not the mean: with "
          "FIXED_EPISODE_LENGTH=False an episode ends on the first classifier "
          "flip, so cost is right-skewed and a mean-sized run is a coin flip.")

    # The numbers above describe THIS GPU. run_ppo.slurm's constraint set spans
    # roughly 3-5x in throughput (H200/H100 at one end, Quadro RTX 6000 at the
    # other), and SLURM gives you whichever is free — so a benchmark that landed
    # on an H200 sizes a run that will NOT finish on an RTX 6000. Say which card
    # produced these numbers and what to do about it, because the failure mode
    # is a 24h walltime kill a day later.
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    fast = any(k in gpu for k in ("H200", "H100", "A100"))
    print(f"\nThese numbers are for {gpu}. The constraint set in run_ppo.slurm "
          f"spans ~3-5x in throughput.")
    if fast:
        print("  This is one of the FAST cards. A run sized from these numbers "
              "may not finish\n  on an RTX 6000 or A40 from the same constraint "
              "set. Either pin the real run\n  with --constraint=\"gpuh200|gpuh100|gpua100\", "
              "or divide N_EPISODES by ~4.")
    else:
        print("  This is one of the SLOWER cards, so the sizing above is "
              "conservative for the\n  faster ones — safe to use as-is.")
    print("\nNothing was saved. config.py is unchanged.")


if __name__ == "__main__":
    main()
