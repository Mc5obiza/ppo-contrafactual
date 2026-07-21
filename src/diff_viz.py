"""
src/diff_viz.py
===============
Visualise the pixel-level difference between an original image and its
counterfactual. Produces a 4-panel figure:
  [original]  |  [counterfactual]  |  [signed diff]  |  [absolute diff heatmap]

Also exposes compute_diff() for programmatic use (e.g. metrics, further analysis).

CHANGE LOG
----------
- generate_and_plot_diff(): accepts an optional `directional_policy`
  (src.inference.DirectionalPolicy). If provided, generation is driven
  by the trained PPO policy's actions (via env.generate_via_policy)
  instead of calling generator.generate_counterfactual directly with a
  fixed M/strength. This makes single-direction diff plots reflect the
  actual trained pipeline.
- Added generate_and_plot_diff_bidirectional(): runs BOTH directions
  (n→p and p→n) in one call, saving two SEPARATE image files and
  printing a side-by-side MAE/MSE/max-Δ/%-changed comparison table, so
  the asymmetry between directions is visible directly in the console
  output as well as in the two images.
- _print_stats() accepts an optional `label` to disambiguate console
  output when both directions are logged back to back.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Core diff computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_diff(
    x_orig: torch.Tensor,
    x_cf: torch.Tensor,
) -> dict[str, torch.Tensor | float]:
    """
    Compute pixel-level difference statistics between original and counterfactual.

    Parameters
    ----------
    x_orig : torch.Tensor  (1, 1, H, W)  — original image, values in [0, 1]
    x_cf   : torch.Tensor  (1, 1, H, W)  — counterfactual image, values in [0, 1]

    Returns
    -------
    dict with keys:
        diff_raw   : signed difference map  (1, 1, H, W)  in [-1, 1]
        diff_abs   : absolute difference    (1, 1, H, W)  in [ 0, 1]
        l1         : mean absolute error    (float)
        l2         : MSE                    (float)
        max_change : max absolute pixel change (float)
        pct_changed: % of pixels changed by more than 0.01 (float)
    """
    x_orig = x_orig.detach().cpu().float()
    x_cf   = x_cf.detach().cpu().float()

    diff_raw = x_cf - x_orig                          # signed, [-1, 1]
    diff_abs = diff_raw.abs()                          # absolute, [0, 1]

    l1          = diff_abs.mean().item()
    l2          = F.mse_loss(x_cf, x_orig).item()
    max_change  = diff_abs.max().item()
    pct_changed = (diff_abs > 0.01).float().mean().item() * 100.0

    return dict(
        diff_raw    = diff_raw,
        diff_abs    = diff_abs,
        l1          = l1,
        l2          = l2,
        max_change  = max_change,
        pct_changed = pct_changed,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_diff(
    x_orig: torch.Tensor,
    x_cf: torch.Tensor,
    target_c: float,
    p_orig: float | None = None,
    p_cf:   float | None = None,
    save_path: str | None = None,
) -> None:
    """
    Produce a 4-panel figure: original | counterfactual | signed diff | abs diff.

    The difference panels use a diverging colormap (blue = decreased intensity,
    red = increased intensity) so the radiologist can see which lung regions
    changed and in which direction.

    Parameters
    ----------
    x_orig    : torch.Tensor  (1, 1, H, W)
    x_cf      : torch.Tensor  (1, 1, H, W)
    target_c  : float  — target class value used during generation
    p_orig    : float | None  — optional classifier score on original
    p_cf      : float | None  — optional classifier score on counterfactual
    save_path : str | None    — if provided, save figure to this path
    """
    stats = compute_diff(x_orig, x_cf)

    orig_np = x_orig[0, 0].detach().cpu().numpy()
    cf_np   = x_cf[0, 0].detach().cpu().numpy()
    diff_np = stats["diff_raw"][0, 0].numpy()          # signed [-1, 1]
    abs_np  = stats["diff_abs"][0, 0].numpy()           # absolute [0, 1]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # ── panel 1: original ────────────────────────────────────────────────────
    axes[0].imshow(orig_np, cmap="gray", vmin=0, vmax=1)
    title_orig = "Original"
    if p_orig is not None:
        title_orig += f"\nf(x) = {p_orig:.3f}"
    axes[0].set_title(title_orig)
    axes[0].axis("off")

    # ── panel 2: counterfactual ───────────────────────────────────────────────
    axes[1].imshow(cf_np, cmap="gray", vmin=0, vmax=1)
    title_cf = f"Counterfactual  (c = {target_c})"
    if p_cf is not None:
        title_cf += f"\nf(x_cf) = {p_cf:.3f}"
    axes[1].set_title(title_cf)
    axes[1].axis("off")

    # ── panel 3: signed difference (diverging) ────────────────────────────────
    vmax_signed = max(abs(diff_np.min()), abs(diff_np.max()), 1e-6)
    im3 = axes[2].imshow(
        diff_np, cmap="RdBu_r",
        vmin=-vmax_signed, vmax=vmax_signed,
    )
    axes[2].set_title("Signed difference\n(red=↑, blue=↓)")
    axes[2].axis("off")
    fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    # ── panel 4: absolute difference heatmap ─────────────────────────────────
    im4 = axes[3].imshow(abs_np, cmap="hot", vmin=0, vmax=max(abs_np.max(), 1e-6))
    axes[3].set_title("Absolute difference\nheatmap")
    axes[3].axis("off")
    fig.colorbar(im4, ax=axes[3], fraction=0.046, pad=0.04)

    # ── summary stats ─────────────────────────────────────────────────────────
    fig.suptitle(
        f"MAE={stats['l1']:.4f}  |  MSE={stats['l2']:.4f}  |  "
        f"max Δ={stats['max_change']:.4f}  |  "
        f"pixels changed (>0.01): {stats['pct_changed']:.1f}%",
        fontsize=10,
        y=1.02,
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"[diff_viz] saved figure to '{save_path}'")

    plt.show()
    _print_stats(stats)


def _print_stats(stats: dict, label: str = "") -> None:
    header = "Counterfactual difference summary"
    if label:
        header += f" ({label})"
    print(f"\n── {header} " + "─" * max(0, 60 - len(header)))
    print(f"  MAE (L1)          : {stats['l1']:.6f}")
    print(f"  MSE (L2)          : {stats['l2']:.6f}")
    print(f"  Max pixel change  : {stats['max_change']:.6f}")
    print(f"  Pixels changed    : {stats['pct_changed']:.2f}%  (threshold > 0.01)")
    print("────────────────────────────────────────────────────────────────\n")


# ─────────────────────────────────────────────────────────────────────────────
# Convenience wrapper — calls generator + plots in one shot (single direction)
# ─────────────────────────────────────────────────────────────────────────────

def generate_and_plot_diff(
    x_orig: torch.Tensor,
    generator,
    classifier,
    target_c: float,
    M: int = 5,
    strength: float = 0.55,
    save_path: str | None = None,
    directional_policy=None,
    deterministic_policy: bool = True,
) -> tuple[torch.Tensor, list[torch.Tensor], dict]:
    """
    Generate a counterfactual from x_orig and immediately visualise the diff.

    Parameters
    ----------
    x_orig     : torch.Tensor  (1, 1, H, W)
    generator  : PretrainedLatentDiffusionGenerator
    classifier : nn.Module
    target_c   : float
    M          : int    — trajectory checkpoints (ignored if directional_policy given)
    strength   : float  — denoising strength (ignored if directional_policy given)
    save_path  : str | None
    directional_policy : src.inference.DirectionalPolicy | None
        If provided, generates via the trained PPO policy instead of
        generator.generate_counterfactual(..., M, strength) directly.
        M and strength are ignored in that case — the policy controls
        its own step count and per-step strength/guidance via its
        matching environment's step() logic.
    deterministic_policy : bool
        Use the policy's mean action (recommended for reproducible
        figures/metrics) rather than sampling. Only relevant when
        directional_policy is provided.

    Returns
    -------
    (x_cf, trajectory, diff_stats)
    """
    if directional_policy is not None:
        from src.inference import generate_counterfactual_with_policy
        x_cf, trajectory, _gen_info = generate_counterfactual_with_policy(
            x_orig, directional_policy, deterministic=deterministic_policy
        )
    else:
        x_cf, trajectory = generator.generate_counterfactual(
            x_orig, target_c, M=M, strength=strength
        )

    p_orig = p_cf = None
    try:
        with torch.no_grad():
            to_rgb = lambda t: t.repeat(1, 3, 1, 1) if t.shape[1] == 1 else t
            p_orig = classifier(to_rgb(x_orig)).item()
            p_cf   = classifier(to_rgb(x_cf)).item()
    except Exception:
        pass  # classifier unavailable — skip scores

    plot_diff(
        x_orig=x_orig,
        x_cf=x_cf,
        target_c=target_c,
        p_orig=p_orig,
        p_cf=p_cf,
        save_path=save_path,
    )

    stats = compute_diff(x_orig, x_cf)
    return x_cf, trajectory, stats


# ─────────────────────────────────────────────────────────────────────────────
# Bidirectional wrapper — n→p AND p→n in one call, two separate images
# ─────────────────────────────────────────────────────────────────────────────

def generate_and_plot_diff_bidirectional(
    x_neg: torch.Tensor,
    x_pos: torch.Tensor,
    generator,
    classifier,
    M: int = 5,
    strength: float = 0.55,
    save_dir: str | None = None,
    directional_policy_n_to_p=None,
    directional_policy_p_to_n=None,
    deterministic_policy: bool = True,
) -> dict[str, tuple[torch.Tensor, list[torch.Tensor], dict]]:
    """
    Generate and visualise counterfactuals in BOTH directions, producing
    two separate 4-panel figures (and two separate saved image files, if
    save_dir is given):

      n→p : x_neg (classifier predicts "normal", f(x) low)  pushed to c=1.0
      p→n : x_pos (classifier predicts "pneumonia", f(x) high) pushed to c=0.0

    This mirrors the Xn/Xp split already used throughout metrics.py
    (CV, FID, and FOP are all computed per-direction there) — the same
    two query images can be reused for a quick visual sanity-check
    alongside the quantitative numbers in eval_metrics.json.

    Parameters
    ----------
    x_neg      : (1,1,H,W) image classified as negative/normal
                 (typically f(x) < ~0.2, same threshold used in metrics.py)
    x_pos      : (1,1,H,W) image classified as positive/pneumonia
                 (typically f(x) > ~0.8)
    generator  : PretrainedLatentDiffusionGenerator
    classifier : nn.Module
    M          : trajectory checkpoints (ignored for directions using a policy)
    strength   : denoising strength (ignored for directions using a policy)
    save_dir   : directory to save both images into. Files are named
                 'counterfactual_diff_n_to_p.png' and
                 'counterfactual_diff_p_to_n.png'. If None, figures are
                 only shown, not saved.
    directional_policy_n_to_p, directional_policy_p_to_n :
                 src.inference.DirectionalPolicy | None
                 If provided, each direction uses its matching trained
                 policy for generation instead of the generator's
                 default schedule.
    deterministic_policy : bool
                 Passed through to generate_and_plot_diff for both calls.

    Returns
    -------
    dict with keys "n_to_p" and "p_to_n", each mapping to
    (x_cf, trajectory, diff_stats) as returned by generate_and_plot_diff.
    """
    save_path_np = save_path_pn = None
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path_np = os.path.join(save_dir, "counterfactual_diff_n_to_p.png")
        save_path_pn = os.path.join(save_dir, "counterfactual_diff_p_to_n.png")

    print("\n[diff_viz] generating n→p counterfactual (normal → pneumonia) ...")
    x_cf_np, traj_np, stats_np = generate_and_plot_diff(
        x_orig=x_neg,
        generator=generator,
        classifier=classifier,
        target_c=1.0,
        M=M,
        strength=strength,
        save_path=save_path_np,
        directional_policy=directional_policy_n_to_p,
        deterministic_policy=deterministic_policy,
    )
    _print_stats(stats_np, label="n→p")

    print("\n[diff_viz] generating p→n counterfactual (pneumonia → normal) ...")
    x_cf_pn, traj_pn, stats_pn = generate_and_plot_diff(
        x_orig=x_pos,
        generator=generator,
        classifier=classifier,
        target_c=0.0,
        M=M,
        strength=strength,
        save_path=save_path_pn,
        directional_policy=directional_policy_p_to_n,
        deterministic_policy=deterministic_policy,
    )
    _print_stats(stats_pn, label="p→n")

    # side-by-side numeric comparison, easy to eyeball whether the
    # n→p direction is still touching far more pixels than p→n
    print("\n── n→p vs p→n diff comparison ─────────────────────────────────")
    print(f"  {'metric':<14}{'n→p':>12}{'p→n':>12}")
    print(f"  {'MAE':<14}{stats_np['l1']:>12.4f}{stats_pn['l1']:>12.4f}")
    print(f"  {'MSE':<14}{stats_np['l2']:>12.4f}{stats_pn['l2']:>12.4f}")
    print(f"  {'max Δ':<14}{stats_np['max_change']:>12.4f}{stats_pn['max_change']:>12.4f}")
    print(f"  {'% changed':<14}{stats_np['pct_changed']:>11.1f}%{stats_pn['pct_changed']:>11.1f}%")
    print("────────────────────────────────────────────────────────────────\n")

    return {
        "n_to_p": (x_cf_np, traj_np, stats_np),
        "p_to_n": (x_cf_pn, traj_pn, stats_pn),
    }
