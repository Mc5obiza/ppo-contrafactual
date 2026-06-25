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

import torch

import config
from ingestion import build_dataset
from src.models import build_classifier, build_segmentation_net, build_object_detector
from src.generator import build_generator
from src.finetuning import run_lora_finetuning
from src.environment import MedicalCounterfactualEnv
from src.ppo import PPOActorCritic, PPOTrainer
from src.evaluate import (
    evaluate_counterfactual,
    plot_reward_trace,
    plot_trajectory,
    save_checkpoints,
)


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

    # ── 6. Evaluation + visualisation ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 6 — Evaluation")
    print("=" * 70)

    transitions = trainer.collect_rollout()
    plot_reward_trace(transitions)

    M_CHECKPOINTS = 5
    x_cf, trajectory = generator.generate_counterfactual(
        env.x_t, env.c, M=M_CHECKPOINTS
    )
    plot_trajectory(env.x_orig, trajectory, target_c=env.c)
    evaluate_counterfactual(env.x_orig, x_cf, classifier, target_c=config.TARGET_C)

    # ── 7. Save ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 7 — Saving checkpoints")
    print("=" * 70)
    save_checkpoints(policy, generator)

    print("\n✓ Pipeline complete.")


if __name__ == "__main__":
    main()
