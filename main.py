"""
main.py
=======
End-to-end pipeline:
    1. Ingest RSNA dataset
    2. Load pretrained models (classifier, segmenter, object detector)
    3. Build generator with LoRA
    4. LoRA fine-tuning (denoising)
    5. PPO training
    6. Evaluation + visualisation
    7. Save checkpoints
"""

import os
import json

import torch

import config
from ingestion import build_dataset
from src.models import build_classifier, build_segmentation_net, build_object_detector
from src.generator import build_generator
from diff_viz import generate_and_plot_diff_bidirectional
from factory.finetuning import run_lora_finetuning
from factory.environment import MedicalCounterfactualEnv
from factory.ppo import PPOActorCritic, PPOTrainer
from src.metrics import run_full_evaluation
from factory.evaluate import plot_reward_trace, save_checkpoints


# ─────────────────────────────────────────────────────────────────────────────
# NEW: guidance-fix diagnostics
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
                    policy.get_action_and_value(obs.unsqueeze(0))
                )
                action = action.squeeze(0)
                obs, reward, done, info = env.step(action)

                guidance_biases.append(info["guidance_bias"])
                n_passes_list.append(info["n_passes"])
                rewards.append(reward)
                # pres_suppression_factor is only present after the fix to
                # environment.py's step(); fall back gracefully if an older
                # checkpoint/environment is being diagnosed.
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


def _save_diagnostics(stats: dict, ckpt_dir: str, filename: str = "guidance_diagnostics.json") -> None:
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, filename)
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[main] guidance diagnostics saved to '{path}'")


# ─────────────────────────────────────────────────────────────────────────────
# NEW: query-image selection for the bidirectional diff plot
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


def main():
    import numpy as np
    global np  # used inside run_guidance_diagnostics

    # ── 1. Dataset ────────────────────────────────────────────────────────────
    print("=" * 70)
    print("STEP 1 — Dataset ingestion")
    print("=" * 70)
    dataset = build_dataset(img_size=config.IMG_SIZE)

    # ── 2. Pretrained models ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 2 — Loading pretrained models")
    print("=" * 70)
    classifier               = build_classifier()
    seg_net, lung_channel_idx = build_segmentation_net()
    obj_det                  = build_object_detector()

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

    # ── 4. LoRA fine-tuning ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 4 — LoRA denoising fine-tuning")
    print("=" * 70)
    run_lora_finetuning(generator, dataset)

    # ── 5. PPO environment + policy ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 5 — PPO training")
    print("=" * 70)
    env = MedicalCounterfactualEnv(
        dataset=dataset,
        classifier=classifier,
        target_c=config.TARGET_C,
        generator=generator,
        seg_net=seg_net,
        obj_det=obj_det,
        budget_T=config.BUDGET_T,
        device=config.DEVICE,
    )

    obs     = env.reset()
    obs_dim = obs.shape[0]
    print(f"obs_dim = {obs_dim}")

    policy = PPOActorCritic(obs_dim=obs_dim, action_dim=config.ACTION_DIM).to(config.DEVICE)
    print(f"policy params = {sum(p.numel() for p in policy.parameters()):,}")

    trainer = PPOTrainer(env, policy, lr=config.LR)
    trainer.train(
        n_episodes=config.N_EPISODES,
        log_every=max(1, config.N_EPISODES // 10),
    )

    # ── 6. Evaluation ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 6 — Evaluation")
    print("=" * 70)

    transitions = trainer.collect_rollout()
    plot_reward_trace(transitions)

    # NEW: guidance-fix diagnostics — confirms the reward/guidance_bias fix
    # is actually changing policy behaviour (not just the reward math) for
    # THIS run's target_c, before spending time on the full metric suite.
    print("\n" + "=" * 70)
    print("STEP 6A — Guidance diagnostics (reward-fix sanity check)")
    print("=" * 70)
    diag_stats = run_guidance_diagnostics(env, policy, n_episodes=20)
    _save_diagnostics(diag_stats, config.CKPT_DIR)

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
    )
    # NEW: FOP was already computed by run_full_evaluation but never
    # surfaced in this summary line — added here since it's the metric
    # most directly relevant to verifying the foreign-object corruption
    # issue seen in the diff plots is actually fixed.
    print(
        f"Evaluation metrics: "
        f"CV={eval_results['CV']['mean']:.4f} "
        f"(n→p={eval_results['CV']['n_to_p']:.4f}, p→n={eval_results['CV']['p_to_n']:.4f}) | "
        f"FID_neg={eval_results['FID']['negative_group']:.4f} | "
        f"FID_pos={eval_results['FID']['positive_group']:.4f} | "
        f"FOP={eval_results['FOP']['mean']:.4f} "
        f"(n→p={eval_results['FOP']['n_to_p']:.4f}, p→n={eval_results['FOP']['p_to_n']:.4f})"
    )

    print("\n" + "=" * 70)
    print("STEP 6B — Counterfactual diff plots (n→p and p→n)")
    print("=" * 70)
    # CHANGED: previously plotted a single direction using dataset[0] with
    # config.TARGET_C. Now selects one confidently-negative and one
    # confidently-positive query image (same criteria as the Xn/Xp groups
    # in metrics.py) and produces TWO separate images, one per direction,
    # so both counterfactual directions can be visually sanity-checked
    # against the CV/FOP numbers above.
    x_neg, x_pos = _pick_query_images(
        dataset, classifier, device=config.DEVICE, low=0.2, high=0.8
    )
    os.makedirs(config.CKPT_DIR, exist_ok=True)
    diff_results = generate_and_plot_diff_bidirectional(
        x_neg=x_neg,
        x_pos=x_pos,
        generator=generator,
        classifier=classifier,
        M=5,
        strength=0.55,
        save_dir=config.CKPT_DIR,
    )
    print(
        f"[main] saved: "
        f"{os.path.join(config.CKPT_DIR, 'counterfactual_diff_n_to_p.png')}, "
        f"{os.path.join(config.CKPT_DIR, 'counterfactual_diff_p_to_n.png')}"
    )

    # ── 7. Save ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 7 — Saving checkpoints")
    print("=" * 70)
    save_checkpoints(policy, generator)

    print("\n✓ Pipeline complete.")


if __name__ == "__main__":
    main()