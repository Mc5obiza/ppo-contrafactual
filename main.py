"""
main.py
=======
TRAINING pipeline (Steps 1-5):
    1. Ingest RSNA dataset
    2. Load pretrained models (classifier, segmenter, object detector)
    3. Build generator with LoRA
    4. LoRA fine-tuning (denoising)
    5. PPO training — TWO policies, one per direction (n→p, p→n)
       + per-direction guidance diagnostics and checkpoints

Evaluation is NOT run here. See eval_main.py (sbatch run_eval_ppo.slurm),
which reloads the checkpoints this script writes.

Usage
-----
    python main.py                        # both directions, sequentially
    python main.py --direction n_to_p     # one direction (what run_ppo.slurm does)

CHANGE LOG
----------
- SPLIT PER DIRECTION. Added --direction {n_to_p,p_to_n,both}; run_ppo.slurm
  is now a 2-task array, one direction per task.

  WHY: publicgpu caps --time at 24h (MaxTime=1-00:00:00 — it cannot be raised)
  and BOTH directions did not fit in one. Job 17653310 spent its full 24h on a
  Tesla P100 and reached ~30 of 100 updates of n_to_p alone. The directions
  share nothing but the Step 4 generator, which finetuning.py already resumes
  from disk as a no-op, so running them as separate jobs is a free 2x on the
  budget with no effect on the training itself.

  The other half of that fix is in factory/ppo.py: PPOTrainer.train() now
  checkpoints every few updates and honours a PPO_WALLCLOCK_BUDGET_S soft
  deadline, so a direction that still needs more than 24h pauses cleanly and
  resumes on resubmission instead of losing the run. _train_directional_policy
  returns None in that case and writes NO final artefacts — see its docstring.
- SPLIT FROM EVALUATION. Steps 6 (metric suite) and 6B (counterfactual
  diff plots) moved to eval_main.py; _pick_query_images moved with them,
  and Steps 1-3 moved to pipeline.build_common() so both entry points
  construct the same objects.

  WHY: PPO training is ~12-20h on the GPUs this job lands on (Quadro
  RTX 5000 / Tesla P100) and evaluation is a further ~2.5-5h. A single
  process that walltimes out in Step 6 discards the finished policies
  with it. Job 17644171 did exactly that — both policies were saved
  (22:21 and 11:05) and then the eval phase was killed at the 24h
  boundary, producing nothing recoverable. Splitting also means eval can
  be re-run (different n_eval, regenerated plots, metric fixes) without
  repeating a PPO run.

- Trains TWO separate PPOActorCritic policies instead of one: one for
  target_c=1.0 (n→p, normal→pneumonia) and one for target_c=0.0 (p→n,
  pneumonia→normal). MedicalCounterfactualEnv is built around a single
  fixed target_c per instance, so a shared conditional policy would
  require a larger refactor (randomized target_c per episode + dual
  far-index pools). Two independently trained policies matches the
  current architecture and lets each specialize on its direction — the
  two directions are known to behave very differently (CV asymmetry:
  0.76 n→p vs 0.94 p→n before the reward-symmetry fix in environment.py).
- Added run_guidance_diagnostics(): samples rollouts from a TRAINED
  policy and reports guidance_bias / pres_suppression_factor / n_passes
  statistics, to verify the environment.py reward/guidance_bias fix is
  actually changing policy behaviour (not just the reward math) before
  spending time on the full metric suite.
- Checkpoints are saved per-direction under ckpt_dir/n_to_p/ and
  ckpt_dir/p_to_n/, since there are two distinct ppo_policy.pt files.
  Note: generator.unet / cond_proj weights are identical in both
  subdirectories since LoRA fine-tuning (Step 4) is shared and only run
  once — this is intentional (one fine-tuned generator, two PPO policies
  steering it) but does mean the generator weights are duplicated on
  disk; consider saving them once at the top level if storage is tight.
"""

import argparse
import os
import json

import numpy as np
import torch

import config
import seeding
from pipeline import build_common
from factory.finetuning import run_lora_finetuning
from factory.environment import MedicalCounterfactualEnv
from factory.ppo import PPOActorCritic, PPOTrainer
from src.inference import DirectionalPolicy
from factory.evaluate import plot_reward_trace, save_checkpoints


# ─────────────────────────────────────────────────────────────────────────────
# Guidance-fix diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def run_guidance_diagnostics(
    env: MedicalCounterfactualEnv,
    policy: PPOActorCritic,
    n_episodes: int = 20,
) -> dict:
    """
    Run a short rollout with the trained policy, capturing the FULL info
    dict from env.step() (not just `tau`, unlike PPOTrainer.collect_rollout)
    so we can verify the reward/guidance_bias fix is actually behaving as
    intended before spending time on full evaluation.

    Specifically tracks, per step:
      - guidance_bias            (policy-chosen, action[7])
      - pres_suppression_factor  (post-fix: max(1/guidance_bias, 0.6))
      - n_passes                 (refinement passes this step, incl. extras)

    If pres_suppression_factor is landing near the 0.6 floor on most steps,
    the policy is still trying to suppress preservation guidance as hard as
    it's allowed to — worth another look at LAM_PRES_R / LAM_FID weights.
    If it's well above the floor on average, the reward fix is doing its job.

    Parameters
    ----------
    env        : MedicalCounterfactualEnv, already configured with the
                 target_c this training run used
    policy     : trained PPOActorCritic
    n_episodes : number of episodes to sample for diagnostics

    Returns
    -------
    dict of aggregated stats, also printed to stdout
    """
    guidance_biases   = []
    suppression_facts = []
    n_passes_list     = []
    rewards           = []

    policy.eval()
    with torch.no_grad():
        for _ in range(n_episodes):
            obs, done = env.reset(), False
            while not done:
                action, _log_prob, _value, _feas, _prog, _stop = (
                    policy.get_action_and_value(obs.unsqueeze(0), deterministic=False)
                )
                action = action.squeeze(0)
                obs, reward, done, info = env.step(action)

                guidance_biases.append(info["guidance_bias"])
                n_passes_list.append(info["n_passes"])
                rewards.append(reward)
                if "pres_suppression_factor" in info:
                    suppression_facts.append(info["pres_suppression_factor"])
    policy.train()

    stats = dict(
        target_c=float(env.c),
        n_episodes=n_episodes,
        n_steps=len(guidance_biases),
        guidance_bias_mean=float(np.mean(guidance_biases)) if guidance_biases else None,
        guidance_bias_std=float(np.std(guidance_biases)) if guidance_biases else None,
        n_passes_mean=float(np.mean(n_passes_list)) if n_passes_list else None,
        pres_suppression_mean=(
            float(np.mean(suppression_facts)) if suppression_facts else None
        ),
        pres_suppression_min=(
            float(np.min(suppression_facts)) if suppression_facts else None
        ),
        avg_reward=float(np.mean(rewards)) if rewards else None,
    )

    print("\n" + "─" * 70)
    print(f"  GUIDANCE DIAGNOSTICS  (target_c = {stats['target_c']})")
    print("─" * 70)
    print(f"  episodes sampled          : {stats['n_episodes']}  ({stats['n_steps']} steps)")
    print(f"  avg guidance_bias         : {stats['guidance_bias_mean']:.4f} "
          f"± {stats['guidance_bias_std']:.4f}")
    print(f"  avg n_passes              : {stats['n_passes_mean']:.2f}")
    if stats["pres_suppression_mean"] is not None:
        print(f"  avg pres_suppression      : {stats['pres_suppression_mean']:.4f}  "
              f"(floor = 0.60)")
        print(f"  min pres_suppression seen : {stats['pres_suppression_min']:.4f}")
        if stats["pres_suppression_mean"] < 0.70:
            print("  ⚠ WARNING: mean suppression is close to the floor — "
                  "policy is still leaning hard on guidance_bias to "
                  "suppress preservation. Consider re-checking LAM_PRES_R "
                  "/ LAM_FID relative to LAM_PROG/LAM_VAL.")
        else:
            print("  ✓ suppression is comfortably above the floor on average.")
    else:
        print("  pres_suppression_factor not found in info dict — "
              "environment.py may not have the guidance_bias fix applied.")
    print(f"  avg reward                : {stats['avg_reward']:.4f}")
    print("─" * 70)

    return stats


def _save_diagnostics(stats: dict, ckpt_dir: str, filename: str) -> None:
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, filename)
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[main] guidance diagnostics saved to '{path}'")


# ─────────────────────────────────────────────────────────────────────────────
# Per-direction PPO training
# ─────────────────────────────────────────────────────────────────────────────

def _train_directional_policy(
    target_c: float,
    label: str,
    dataset,
    classifier,
    generator,
    seg_net,
    obj_det,
    allowed_indices=None,
) -> DirectionalPolicy | None:
    """
    Build an env + policy for one direction, train it, run diagnostics,
    save a per-direction checkpoint, and return both env and policy
    bundled as a DirectionalPolicy.

    The checkpoint written here is what eval_main.py reloads — see
    pipeline.load_directional_policy(), which rebuilds this same env and
    restores ppo_policy.pt into a matching PPOActorCritic.

    Returns None if PPOTrainer.train() stopped on the wall-clock budget rather
    than finishing. In that case NOTHING downstream runs: no reward trace, no
    guidance diagnostics, and no save_checkpoints(). Those all describe a
    finished policy, and writing a half-trained ppo_policy.pt would be worse
    than writing nothing — eval_main.py cannot tell the two apart, so a paused
    run would silently be evaluated as a completed one. The resumable state
    lives in ppo_latest.pt; resubmit to continue.

    Parameters
    ----------
    target_c : 1.0 for n→p, 0.0 for p→n
    label    : human-readable name for logging/checkpoint dir ("n_to_p" / "p_to_n")
    """
    print("\n" + "=" * 70)
    print(f"STEP 5 — PPO training  [{label}, target_c={target_c}]")
    print("=" * 70)

    # Re-seed per direction, with a distinct offset per label.
    #
    # Two reasons this is not redundant with the seed in main(). First,
    # run_ppo.slurm submits the directions as SEPARATE array tasks, so without
    # an offset both tasks start from the identical seed and draw correlated
    # episode sequences. Second, with it, `--direction n_to_p` alone produces
    # the same policy as the n_to_p half of `--direction both` — the array and
    # single-process paths agree, which is the same property the eval sharding
    # already guarantees on its side.
    #
    # A literal index rather than hash(label): Python string hashing is salted
    # per process unless PYTHONHASHSEED is fixed, so hash() would hand out a
    # different offset on every run.
    seeding.seed_everything(
        offset=1 + sorted(DIRECTIONS).index(label), tag=label,
    )

    env = MedicalCounterfactualEnv(
        dataset=dataset,
        classifier=classifier,
        target_c=target_c,
        generator=generator,
        seg_net=seg_net,
        obj_det=obj_det,
        budget_T=config.BUDGET_T,
        device=config.DEVICE,
        # Rollouts draw only from the training fold — see splits.py.
        allowed_indices=allowed_indices,
    )

    obs     = env.reset()
    obs_dim = obs.shape[0]
    print(f"[{label}] obs_dim = {obs_dim}")

    policy = PPOActorCritic(obs_dim=obs_dim, action_dim=config.ACTION_DIM).to(config.DEVICE)
    print(f"[{label}] policy params = {sum(p.numel() for p in policy.parameters()):,}")

    trainer = PPOTrainer(env, policy, lr=config.LR)
    completed = trainer.train(
        n_episodes=config.N_EPISODES,
        log_every=max(1, config.N_EPISODES // 10),
        label=label,
        ckpt_dir=config.CKPT_DIR,
    )
    if not completed:
        print(f"\n[main] {label} PAUSED on the wall-clock budget — resumable "
              f"state is in '{os.path.join(config.CKPT_DIR, label, 'ppo_latest.pt')}'.")
        print(f"[main] skipping reward trace, guidance diagnostics and the final "
              f"checkpoint: this policy is not finished.")
        return None

    # per-direction reward trace + guidance diagnostics
    transitions = trainer.collect_rollout()
    plot_reward_trace(transitions)

    diag_stats = run_guidance_diagnostics(env, policy, n_episodes=20)
    _save_diagnostics(diag_stats, config.CKPT_DIR, filename=f"guidance_diagnostics_{label}.json")

    # per-direction checkpoint, so n→p and p→n policies don't overwrite
    # each other in the shared ckpt_dir
    ckpt_dir_direction = os.path.join(config.CKPT_DIR, label)
    save_checkpoints(policy, generator, ckpt_dir=ckpt_dir_direction)

    return DirectionalPolicy(env=env, policy=policy, target_c=target_c)


# ─────────────────────────────────────────────────────────────────────────────
# Training pipeline
# ─────────────────────────────────────────────────────────────────────────────

DIRECTIONS = {
    "n_to_p": 1.0,   # normal   -> pneumonia
    "p_to_n": 0.0,   # pneumonia -> normal
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="PPO counterfactual TRAINING pipeline (Steps 1-5). "
                    "Evaluation lives in eval_main.py."
    )
    parser.add_argument(
        "--direction",
        choices=[*DIRECTIONS, "both"],
        default="both",
        help="Which PPO policy to train. run_ppo.slurm submits one array task "
             "per direction so each gets its own 24h walltime; 'both' (the "
             "default) trains them sequentially in one process, which is what "
             "a bare 'python main.py' does and what previously did not fit.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    labels = list(DIRECTIONS) if args.direction == "both" else [args.direction]

    # Seed BEFORE anything constructs a model or touches the dataset — LoRA
    # adapter init, policy init and the far-pool scan all draw from these RNGs.
    # See seeding.py for the (real) limits of what this guarantees.
    seeding.seed_everything(tag="training")

    # ── 1-3. Dataset, pretrained models, generator ───────────────────────────
    ctx = build_common()

    # ── 4. LoRA fine-tuning (shared generator, both directions) ──────────────
    print("\n" + "=" * 70)
    print("STEP 4 — LoRA denoising fine-tuning")
    print("=" * 70)
    # Training fold only. Fine-tuning on the full dataset is what put the
    # evaluation images inside the generator's training set — see splits.py.
    run_lora_finetuning(ctx.generator, ctx.dataset, indices=ctx.train_indices())

    # ── 5. Train one policy per requested direction ───────────────────────────
    # MedicalCounterfactualEnv is built around a single fixed target_c per
    # instance, so a shared conditional policy would need a larger
    # refactor (randomized target_c per episode, dual far-index pools).
    # Two independently trained policies matches the current architecture
    # and lets each specialize on its (very differently-behaved)
    # direction — n→p is the harder direction per the CV asymmetry
    # (0.76 vs 0.94) and the reward/guidance_bias fix in environment.py.
    #
    # The directions are INDEPENDENT — they share only the Step 4 generator,
    # which is already trained and resumed above — so run_ppo.slurm submits
    # them as separate array tasks and each gets its own 24h walltime.
    finished, paused = [], []
    for label in labels:
        result = _train_directional_policy(
            target_c=DIRECTIONS[label], label=label,
            dataset=ctx.dataset, classifier=ctx.classifier, generator=ctx.generator,
            seg_net=ctx.seg_net, obj_det=ctx.obj_det,
            allowed_indices=ctx.train_indices(),
        )
        (finished if result is not None else paused).append(label)
        if result is None:
            # Out of wall clock. Any remaining direction would be killed
            # mid-rollout, so don't start one.
            break

    print("\n" + "=" * 70)
    print("✓ TRAINING COMPLETE" if not paused else "⏸ TRAINING PAUSED (wall-clock budget)")
    print("=" * 70)
    for label in finished:
        print(f"  {label}: done → {config.CKPT_DIR}/{label}/"
              f"  (ppo_policy.pt, unet_finetuned.pt, cond_proj.pt)")
    for label in paused:
        print(f"  {label}: PAUSED → {config.CKPT_DIR}/{label}/ppo_latest.pt "
              f"(resumable state only, no ppo_policy.pt yet)")
    for label in labels:
        if label not in finished and label not in paused:
            print(f"  {label}: NOT STARTED (ran out of wall clock beforehand)")

    if paused:
        print("\n  Resubmit to continue from the saved state — training picks up")
        print("  at the next update rather than restarting:")
        print(f"        sbatch run_ppo.slurm          # both directions")
        print(f"        sbatch --array=<task> run_ppo.slurm   # just the paused one")
    elif set(labels) >= set(DIRECTIONS):
        print("\n  Next: run the evaluation pipeline against these checkpoints —")
        print("        sbatch run_eval_ppo.slurm")
    else:
        missing = [d for d in DIRECTIONS if d not in finished]
        print(f"\n  Still to train before evaluation: {', '.join(missing)}")
        print(f"        python main.py --direction {missing[0]}")


if __name__ == "__main__":
    main()
