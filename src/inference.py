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

import time
from dataclasses import dataclass

import torch

from factory.environment import MedicalCounterfactualEnv
from factory.ppo import PPOActorCritic


# Seed-space offset for the p→n direction. The two directions run over
# different images, so without an offset both would draw the SAME diffusion
# noise for their i-th image and correlate the two halves of the metric suite
# for no reason. Large enough that no realistic n_eval can reach it. Every
# caller that generates both directions must use 0 / this, or their cached
# counterfactuals will not match each other.
EVAL_PN_SEED_OFFSET = 1_000_000


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
    index_offset: int = 0,
    base_seed: int = 0,
    label: str = "",
    progress_every: int = 1,
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
    index_offset         : GLOBAL index of x_list[0]. See the seeding note
                            below — this is what makes a sharded run produce
                            the same counterfactuals as a serial one.
    base_seed            : global seed offset. Changing it changes every
                            counterfactual, so it INVALIDATES any cached
                            shards; bump it only when you deliberately want a
                            different noise draw (e.g. a seed-variance study).
    label                : short tag for the progress lines ("n_to_p" etc.)
    progress_every       : print a progress line every N images (0 disables)

    Returns
    -------
    list of (1, 1, H, W) counterfactual tensors, detached and on CPU

    SEEDING / SHARD-INVARIANCE
    --------------------------
    generate_counterfactual draws init_noise and per-DDIM-step background-lock
    noise from the global RNG, so without reseeding, image k's counterfactual
    depends on how many images were generated before it in this process. That
    makes results depend on how the work is chunked, and makes reruns
    irreproducible.

    Seeding per image with (base_seed + index_offset + i) makes each
    counterfactual a function of its GLOBAL index alone. A shard covering
    images [50, 100) then produces byte-identical output to a serial run over
    [0, 400), which is what lets eval_main.py's generate/reduce split report
    the same numbers as a single-process run.

    Distinct directions must not share seeds — n→p and p→n operate on
    different images, but reusing the same noise across both would correlate
    the two halves of the metric suite for no reason. Callers pass disjoint
    index_offset ranges (see eval_main.py).
    """
    cfs = []
    t0 = time.time()
    n = len(x_list)
    tag = f"[{label}] " if label else ""
    for i, x in enumerate(x_list):
        torch.manual_seed(base_seed + index_offset + i)
        xc, _traj, _info = generate_counterfactual_with_policy(
            x, directional_policy, deterministic=deterministic
        )
        cfs.append(xc.detach().cpu())

        if progress_every and ((i + 1) % progress_every == 0 or i + 1 == n):
            elapsed = time.time() - t0
            rate    = elapsed / (i + 1)
            eta     = rate * (n - i - 1)
            print(
                f"{tag}[{i + 1}/{n}] global_idx={index_offset + i}  "
                f"{rate / 60:.2f} min/img  "
                f"elapsed {elapsed / 60:.1f} min  "
                f"eta {eta / 60:.1f} min",
                flush=True,
            )
    return cfs
