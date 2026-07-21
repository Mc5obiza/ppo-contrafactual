"""
src/inference.py
=================
Thin wrapper for policy-guided counterfactual generation, used at
evaluation time so reported metrics/plots reflect the trained PPO
policies rather than the generator's hardcoded default schedule.

Expects two independently trained (env, policy) pairs — one for
target_c=1.0 (n->p) and one for target_c=0.0 (p->n) — matching the
two-policy training setup in main.py.
"""

from dataclasses import dataclass

import torch

from factory.environment import MedicalCounterfactualEnv
from factory.ppo import PPOActorCritic


@dataclass
class DirectionalPolicy:
    """Bundles an env (fixed to one target_c) with its trained policy."""
    env: MedicalCounterfactualEnv
    policy: PPOActorCritic
    target_c: float


def generate_counterfactual_with_policy(
    x_orig: torch.Tensor,
    directional_policy: DirectionalPolicy,
    deterministic: bool = True,
    max_steps: int | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor], dict]:
    """
    Single entry point used by metrics.py and diff_viz.py — generates a
    counterfactual for x_orig by stepping the trained policy through its
    matching environment, instead of calling
    generator.generate_counterfactual with default arguments.

    Parameters
    ----------
    x_orig              : (1, 1, H, W) query image
    directional_policy   : DirectionalPolicy bundling the env + trained
                            policy for the direction x_orig should be
                            pushed toward (n→p or p→n)
    deterministic        : use the policy's mean action for reproducible
                            eval results (recommended: True)
    max_steps            : override the env's default budget_T if given

    Returns
    -------
    (x_cf, trajectory, info) — see MedicalCounterfactualEnv.generate_via_policy
    """
    return directional_policy.env.generate_via_policy(
        x_orig=x_orig,
        policy=directional_policy.policy,
        deterministic=deterministic,
        max_steps=max_steps,
    )


def generate_batch_with_policy(
    x_list: list[torch.Tensor],
    directional_policy: DirectionalPolicy,
    deterministic: bool = True,
) -> list[torch.Tensor]:
    """
    Batch version for use inside run_full_evaluation's _generate_batch —
    same signature shape as the old generator.generate_counterfactual
    loop, so it's a drop-in replacement.

    Parameters
    ----------
    x_list              : list of (1, 1, H, W) query images
    directional_policy   : DirectionalPolicy for this batch's direction
    deterministic        : use the policy's mean action (recommended: True)

    Returns
    -------
    list of (1, 1, H, W) counterfactual tensors, detached and on CPU
    """
    cfs = []
    for x in x_list:
        xc, _traj, _info = generate_counterfactual_with_policy(
            x, directional_policy, deterministic=deterministic
        )
        cfs.append(xc.detach().cpu())
    return cfs
