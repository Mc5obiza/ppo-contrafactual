"""
main.py
=======
End-to-end pipeline:
    1. Ingest RSNA dataset
    2. Load pretrained models (classifier, segmenter, object detector)
    3. Build generator with LoRA
    4. LoRA fine-tuning (denoising)
    5. PPO training — TWO policies, one per direction (n→p, p→n)
    6. Evaluation + visualisation, using the trained policies
    7. Save checkpoints

CHANGE LOG
----------
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
- Added _pick_query_images(): selects one confidently-negative
  (f(x)<0.2) and one confidently-positive (f(x)>0.8) query image for the
  qualitative diff plot, using the SAME thresholds as the Xn/Xp groups
  in metrics.run_full_evaluation, instead of the previous arbitrary
  dataset[0].
- Step 6 (evaluation) and Step 6B (diff plots) now pass the two trained
  policies through to run_full_evaluation / generate_and_plot_diff_bidirectional
  so reported CV/FID/FOP numbers and the two saved diff images reflect
  the actual PPO-guided pipeline, not the generator's hardcoded default
  schedule.
- Checkpoints are now saved per-direction under ckpt_dir/n_to_p/ and
  ckpt_dir/p_to_n/, since there are now two distinct ppo_policy.pt files.
  Note: generator.unet / cond_proj weights are identical in both
  subdirectories since LoRA fine-tuning (Step 4) is shared and only run
  once — this is intentional (one fine-tuned generator, two PPO policies
  steering it) but does mean the generator weights are duplicated on
  disk; consider saving them once at the top level if storage is tight.
"""

import os
import json

import numpy as np
import torch

import config
from ingestion import build_dataset
from src.models import  build_segmentation_net, build_object_detector
from src.generator import build_generator
from src.diff_viz import generate_and_plot_diff_bidirectional
from factory.finetuning import run_lora_finetuning
from factory.environment import MedicalCounterfactualEnv
from factory.ppo import PPOActorCritic, PPOTrainer
from src.inference import DirectionalPolicy
from src.metrics import run_full_evaluation
from factory.evaluate import plot_reward_trace, save_checkpoints
from src.xrv_classifier import build_xrv_classifier


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
# Query-image selection for the bidirectional diff plot
# ─────────────────────────────────────────────────────────────────────────────

def _pick_query_images(
    dataset,
    classifier,
    device: str,
    low: float = 0.2,
    high: float = 0.8,
    scan_batch_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Scan the dataset for the first confidently-negative image (f(x) < low)
    and first confidently-positive image (f(x) > high), matching the same
    grouping criteria used in metrics.run_full_evaluation, so the images
    used for the visual diff plot are representative of the Xn / Xp groups
    the quantitative metrics are actually computed over.

    Returns
    -------
    (x_neg, x_pos) : each (1, 1, H, W) on `device`
    """
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=scan_batch_size, shuffle=True, num_workers=2)
    to_rgb = lambda t: t.repeat(1, 3, 1, 1) if t.shape[1] == 1 else t

    x_neg = x_pos = None
    with torch.no_grad():
        for x_batch, _labels in loader:
            x_batch = x_batch.to(device)
            scores  = classifier(to_rgb(x_batch)).squeeze(-1)
            for i in range(len(x_batch)):
                s = scores[i].item()
                if x_neg is None and s < low:
                    x_neg = x_batch[i].unsqueeze(0).clone()
                if x_pos is None and s > high:
                    x_pos = x_batch[i].unsqueeze(0).clone()
            if x_neg is not None and x_pos is not None:
                break

    if x_neg is None or x_pos is None:
        raise RuntimeError(
            f"[main] could not find both a negative (f(x)<{low}) and "
            f"positive (f(x)>{high}) example for the diff plot — dataset "
            f"may be too small or classifier scores too concentrated. "
            f"found_neg={x_neg is not None}, found_pos={x_pos is not None}"
        )

    return x_neg, x_pos


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
) -> DirectionalPolicy:
    """
    Build an env + policy for one direction, train it, run diagnostics,
    save a per-direction checkpoint, and return both env and policy
    bundled as a DirectionalPolicy for use at evaluation time.

    Parameters
    ----------
    target_c : 1.0 for n→p, 0.0 for p→n
    label    : human-readable name for logging/checkpoint dir ("n_to_p" / "p_to_n")
    """
    print("\n" + "=" * 70)
    print(f"STEP 5 — PPO training  [{label}, target_c={target_c}]")
    print("=" * 70)

    env = MedicalCounterfactualEnv(
        dataset=dataset,
        classifier=classifier,
        target_c=target_c,
        generator=generator,
        seg_net=seg_net,
        obj_det=obj_det,
        budget_T=config.BUDGET_T,
        device=config.DEVICE,
    )

    obs     = env.reset()
    obs_dim = obs.shape[0]
    print(f"[{label}] obs_dim = {obs_dim}")

    policy = PPOActorCritic(obs_dim=obs_dim, action_dim=config.ACTION_DIM).to(config.DEVICE)
    print(f"[{label}] policy params = {sum(p.numel() for p in policy.parameters()):,}")

    trainer = PPOTrainer(env, policy, lr=config.LR)
    trainer.train(
        n_episodes=config.N_EPISODES,
        log_every=max(1, config.N_EPISODES // 10),
    )

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
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── 1. Dataset ────────────────────────────────────────────────────────────
    print("=" * 70)
    print("STEP 1 — Dataset ingestion")
    print("=" * 70)
    dataset = build_dataset(img_size=config.IMG_SIZE)

    # ── 2. Pretrained models ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 2 — Loading pretrained models")
    print("=" * 70)
    classifier                = build_xrv_classifier(weights="densenet121-res224-rsna")
    seg_net, lung_channel_idx = build_segmentation_net()
    obj_det                   = build_object_detector()

    # ── 3. Generator + LoRA ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 3 — Building generator with LoRA")
    print("=" * 70)
    generator = build_generator(
        classifier=classifier,
        seg_net=seg_net,
        lung_channel_idx=lung_channel_idx,
        obj_det=obj_det,
    )

    # ── 4. LoRA fine-tuning (shared generator, both directions) ──────────────
    print("\n" + "=" * 70)
    print("STEP 4 — LoRA denoising fine-tuning")
    print("=" * 70)
    run_lora_finetuning(generator, dataset)

    # ── 5. Train TWO policies, one per direction ──────────────────────────────
    # MedicalCounterfactualEnv is built around a single fixed target_c per
    # instance, so a shared conditional policy would need a larger
    # refactor (randomized target_c per episode, dual far-index pools).
    # Two independently trained policies matches the current architecture
    # and lets each specialize on its (very differently-behaved)
    # direction — n→p is the harder direction per the CV asymmetry
    # (0.76 vs 0.94) and the reward/guidance_bias fix in environment.py.
    policy_n_to_p = _train_directional_policy(
        target_c=1.0, label="n_to_p",
        dataset=dataset, classifier=classifier, generator=generator,
        seg_net=seg_net, obj_det=obj_det,
    )
    policy_p_to_n = _train_directional_policy(
        target_c=0.0, label="p_to_n",
        dataset=dataset, classifier=classifier, generator=generator,
        seg_net=seg_net, obj_det=obj_det,
    )

    # ── 6. Evaluation — now PPO-guided instead of generator-default ──────────
    print("\n" + "=" * 70)
    print("STEP 6 — Evaluation (PPO-guided generation)")
    print("=" * 70)
    eval_results = run_full_evaluation(
        dataset=dataset,
        classifier=classifier,
        generator=generator,
        seg_net=seg_net,
        lung_channel_idx=lung_channel_idx,
        obj_det=obj_det,
        n_eval=min(200, len(dataset)),
        ckpt_dir=config.CKPT_DIR,
        device=config.DEVICE,
        policy_n_to_p=policy_n_to_p,
        policy_p_to_n=policy_p_to_n,
        deterministic_policy=True,
    )
    print(
        f"Evaluation metrics: "
        f"CV={eval_results['CV']['mean']:.4f} "
        f"(n→p={eval_results['CV']['n_to_p']:.4f}, p→n={eval_results['CV']['p_to_n']:.4f}) | "
        f"FID_neg={eval_results['FID']['negative_group']:.4f} | "
        f"FID_pos={eval_results['FID']['positive_group']:.4f}"
    )
    _ip = eval_results.get('IP')
    if _ip:
        _g = lambda d, k: f"{d[k]:.4f}" if d and d.get(k) is not None else "N/A"
        print(
            f"Identity Preservation (IP): "
            f"n→p MS-SSIM={_g(_ip['n_to_p'], 'ms_ssim')}/HF={_g(_ip['n_to_p'], 'hf_corr')} | "
            f"p→n MS-SSIM={_g(_ip['p_to_n'], 'ms_ssim')}/HF={_g(_ip['p_to_n'], 'hf_corr')}"
        )

    print("\n" + "=" * 70)
    print("STEP 6B — Counterfactual diff plots (n→p and p→n, PPO-guided)")
    print("=" * 70)
    x_neg, x_pos = _pick_query_images(
        dataset, classifier, device=config.DEVICE, low=0.2, high=0.8
    )
    os.makedirs(config.CKPT_DIR, exist_ok=True)
    generate_and_plot_diff_bidirectional(
        x_neg=x_neg,
        x_pos=x_pos,
        generator=generator,
        classifier=classifier,
        save_dir=config.CKPT_DIR,
        directional_policy_n_to_p=policy_n_to_p,
        directional_policy_p_to_n=policy_p_to_n,
        deterministic_policy=True,
    )
    print(
        f"[main] saved: "
        f"{os.path.join(config.CKPT_DIR, 'counterfactual_diff_n_to_p.png')}, "
        f"{os.path.join(config.CKPT_DIR, 'counterfactual_diff_p_to_n.png')}"
    )

    print("\n✓ Pipeline complete.")


if __name__ == "__main__":
    main()