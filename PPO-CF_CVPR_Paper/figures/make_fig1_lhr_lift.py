"""Regenerate figures/fig1_lhr_lift.png (Fig. 1 of the paper).

In-lung LHR lift (p->n) with 95% t-intervals: the five trained arms, and four
reference runs on the default arm's 148 annotated images (fixed schedule at the
policy's mean action, no policy, iterative Latent Shift, CycleGAN). In-lung lift
restricts both the edit mass and the area null to the lung mask (Sec. 4.2), so a
lesion-blind edit that stays in the lung scores 1.0.

Sources (all cleared): ~/verification/out/J19_lhr_nulls/lhr_nulls.json (arms
J4, J5_none, J5c_fixed, J10_cyclegan, v1_single, ens3, sparsity, gradcam) and
~/verification/out/J19c_lhr_nulls_J15b/lhr_nulls.json (J15b_ls_iter); key
arms.<tag>.lift_lung.{mean, ci95, n}. MEMO_2026-11-05_J19_lhr_nulls.md.

    python figures/make_fig1_lhr_lift.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

V = os.path.expanduser("~/verification/out")
J19 = json.load(open(f"{V}/J19_lhr_nulls/lhr_nulls.json"))["arms"]
J19C = json.load(open(f"{V}/J19c_lhr_nulls_J15b/lhr_nulls.json"))["arms"]

ROWS = [  # label, source arms, tag, group
    ("2-model ensemble (default)", J19, "J4", "arm"),
    ("Single-classifier$^{*}$", J19, "v1_single", "arm"),
    ("3-model ensemble$^{*}$", J19, "ens3", "arm"),
    ("Sparsity reward", J19, "sparsity", "arm"),
    ("Grad-CAM constraint", J19, "gradcam", "arm"),
    ("Fixed schedule (policy mean)", J19, "J5c_fixed", "ref"),
    ("No policy", J19, "J5_none", "ref"),
    ("Latent Shift (iterative)", J19C, "J15b_ls_iter", "ref"),
    ("CycleGAN", J19, "J10_cyclegan", "ref"),
]

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
COL = {"arm": "#2a78d6", "ref": "#6f6e69"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
fig, ax = plt.subplots(figsize=(5.2, 2.55), dpi=300)
ys, gap = [], 0
for i, (_, _, _, g) in enumerate(ROWS):
    if i and g != ROWS[i - 1][3]:
        gap += 0.6
    ys.append(-(i + gap))
for y, (name, src, tag, g) in zip(ys, ROWS):
    s = src[tag]["lift_lung"]
    lo, hi = s["ci95"]
    ax.plot([lo, hi], [y, y], color=COL[g], linewidth=2.0, solid_capstyle="round", zorder=2)
    ax.plot(s["mean"], y, "o", markersize=6.5, color=COL[g], markeredgecolor="white",
            markeredgewidth=1.2, zorder=3)
    ax.text(1.385, y, f"{s['mean']:.3f}", va="center", ha="right", color=INK, fontsize=7.5,
            fontweight="bold" if "default" in name else "normal")
    ax.text(1.395, y, f"n={s['n']}", va="center", ha="left", color=MUTED, fontsize=6.5)
ax.axvline(1.0, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.0, zorder=1)
ax.text(1.0, ys[0] + 0.75, "chance (lesion-blind, in lung)", ha="left", va="center",
        color=INK2, fontsize=6.5, bbox=dict(facecolor="white", edgecolor="none", pad=0.8))
ax.tick_params(axis="both", colors=INK2, length=0, labelsize=7.5)
ax.set_yticks(ys)
ax.set_yticklabels([r[0] for r in ROWS], color=INK)
for t, r in zip(ax.get_yticklabels(), ROWS):
    if "default" in r[0]:
        t.set_fontweight("bold")
    if r[3] == "ref":
        t.set_color(INK2)
ax.set_xlim(0.97, 1.45)
ax.set_xticks([1.0, 1.1, 1.2, 1.3])
ax.set_ylim(ys[-1] - 0.6, ys[0] + 1.2)
ax.set_xlabel("In-lung Lesion Hit Rate lift, p→n (95% CI)", color=INK2, fontsize=7.5)
ax.xaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID)
fig.tight_layout(pad=0.3)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig1_lhr_lift.png")
fig.savefig(out, dpi=300, facecolor="white")
print("wrote", out)
