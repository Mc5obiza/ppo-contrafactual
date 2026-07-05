"""
src/diff_viz.py
===============
Visualise the pixel-level difference between an original image and its
counterfactual. Produces a 3-panel figure:
  [original]  |  [counterfactual]  |  [difference heatmap]

Also exposes compute_diff() for programmatic use (e.g. metrics, further analysis).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe: SLURM compute nodes have no $DISPLAY
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
    Produce a 3-panel figure: original | counterfactual | difference heatmap.

    The difference panel uses a diverging colormap (blue = decreased intensity,
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


def _print_stats(stats: dict) -> None:
    print("\n── Counterfactual difference summary ──────────────────────────")
    print(f"  MAE (L1)          : {stats['l1']:.6f}")
    print(f"  MSE (L2)          : {stats['l2']:.6f}")
    print(f"  Max pixel change  : {stats['max_change']:.6f}")
    print(f"  Pixels changed    : {stats['pct_changed']:.2f}%  (threshold > 0.01)")
    print("────────────────────────────────────────────────────────────────\n")


# ─────────────────────────────────────────────────────────────────────────────
# Convenience wrapper — calls generator + plots in one shot
# ─────────────────────────────────────────────────────────────────────────────

def generate_and_plot_diff(
    x_orig: torch.Tensor,
    generator,
    classifier,
    target_c: float,
    M: int = 5,
    strength: float = 0.55,
    save_path: str | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor], dict]:
    """
    Generate a counterfactual from x_orig and immediately visualise the diff.

    Parameters
    ----------
    x_orig     : torch.Tensor  (1, 1, H, W)
    generator  : PretrainedLatentDiffusionGenerator
    classifier : nn.Module
    target_c   : float
    M          : int    — trajectory checkpoints
    strength   : float  — denoising strength (0–1)
    save_path  : str | None

    Returns
    -------
    (x_cf, trajectory, diff_stats)
    """
    import torch.nn.functional as F

    x_cf, trajectory = generator.generate_counterfactual(
        x_orig, target_c, M=M, strength=strength
    )

    # Classifier scores (optional but informative)
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
