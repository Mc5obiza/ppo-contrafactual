"""
src/evaluate.py
===============
Evaluation and visualisation utilities:
  - evaluate_counterfactual  — print classifier scores before/after
  - plot_reward_trace         — flat vs. shaped reward per step
  - plot_trajectory           — progressive counterfactual frames
  - save_checkpoints          — persist policy + UNet LoRA + cond_proj weights
"""

import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

import config
from factory.ppo import compute_gae


# ─────────────────────────────────────────────────────────────────────────────

def evaluate_counterfactual(
    x_query: torch.Tensor,
    x_cf: torch.Tensor,
    classifier,
    target_c: float,
    epsilon_stop: float = config.EPSILON_STOP,
) -> None:
    """
    Print classifier scores for the original and counterfactual images.

    Parameters
    ----------
    x_query : torch.Tensor  (1, 1, H, W)
    x_cf    : torch.Tensor  (1, 1, H, W)
    classifier : nn.Module
    target_c : float
    epsilon_stop : float
    """
    with torch.no_grad():
        x3_orig = x_query.repeat(1, 3, 1, 1) if x_query.shape[1] == 1 else x_query
        x3_cf   = x_cf.repeat(1, 3, 1, 1)   if x_cf.shape[1]    == 1 else x_cf
        p_orig  = classifier(x3_orig).item()
        p_cf    = classifier(x3_cf).item()

    gap   = abs(p_cf - target_c)
    valid = gap < epsilon_stop
    print(f"f(x_orig) = {p_orig:.4f}")
    print(f"f(x_cf)   = {p_cf:.4f}")
    print(f"target c  = {target_c}")
    print(f"|f(x_cf) - c| = {gap:.4f}  (valid if < {epsilon_stop})")
    print(f"valid counterfactual: {valid}")


# ─────────────────────────────────────────────────────────────────────────────

def plot_reward_trace(transitions: list) -> None:
    """
    Plot flat vs. GAE-shaped reward for each step of a collected rollout.

    Parameters
    ----------
    transitions : list[Transition]
    """
    shaped_rewards, advantages, _ = compute_gae(transitions)
    flat_rewards = [t.reward for t in transitions]

    print(f"{'step':>4} | {'flat R_t':>10} | {'shaped R~_t':>12} | {'advantage':>10} | {'tau_t':>6}")
    for i, t in enumerate(transitions):
        print(
            f"{i:4d} | {flat_rewards[i]:10.4f} | {shaped_rewards[i].item():12.4f} | "
            f"{advantages[i].item():10.4f} | {t.tau:6.3f}"
        )

    fig, ax = plt.subplots(figsize=(8, 4))
    steps = list(range(len(transitions)))
    ax.plot(steps, flat_rewards,           marker="o", label="flat $R_t$")
    ax.plot(steps, shaped_rewards.detach().cpu().numpy(), marker="s", label=r"shaped $\tilde{R}_t$")
    ax.set_xlabel("step")
    ax.set_ylabel("reward")
    ax.set_title("Flat vs. GAE-shaped reward per step")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────

def plot_trajectory(
    x_query: torch.Tensor,
    trajectory: list[torch.Tensor],
    target_c: float,
) -> None:
    """
    Visualise the original image alongside M denoising checkpoints.

    Parameters
    ----------
    x_query    : torch.Tensor  (1, 1, H, W) — original query image
    trajectory : list[Tensor]  — M intermediate counterfactual frames
    target_c   : float
    """
    n_cols = len(trajectory) + 1
    fig, axes = plt.subplots(1, n_cols, figsize=(3 * n_cols, 3))

    axes[0].imshow(x_query[0, 0].detach().cpu(), cmap="gray")
    axes[0].set_title("original x")
    axes[0].axis("off")

    for i, frame in enumerate(trajectory):
        axes[i + 1].imshow(frame[0, 0], cmap="gray")
        axes[i + 1].set_title(f"step {i + 1}/{len(trajectory)}")
        axes[i + 1].axis("off")

    plt.suptitle(f"Progressive counterfactual trajectory (target c = {target_c})")
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoints(policy, generator, ckpt_dir: str = config.CKPT_DIR) -> None:
    """
    Persist the PPO policy, fine-tuned UNet LoRA weights, and cond_proj.

    Parameters
    ----------
    policy    : PPOActorCritic
    generator : PretrainedLatentDiffusionGenerator
    ckpt_dir  : str
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(policy.state_dict(),                os.path.join(ckpt_dir, "ppo_policy.pt"))
    torch.save(generator.unet.state_dict(),        os.path.join(ckpt_dir, "unet_finetuned.pt"))
    torch.save(generator.cond_proj.state_dict(),   os.path.join(ckpt_dir, "cond_proj.pt"))
    print(f"[evaluate] saved checkpoints to '{ckpt_dir}/'")
    print(f"  → ppo_policy.pt")
    print(f"  → unet_finetuned.pt")
    print(f"  → cond_proj.pt")
