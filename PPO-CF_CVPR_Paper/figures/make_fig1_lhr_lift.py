"""Regenerate figures/fig1_lhr_lift.png (Fig. 1 of the paper).

LHR lift (p->n) of the five trained arms, values as in Table 4. The default arm
is its per-direction-adapter evaluation (J4, 2.595); the no-policy reference is
the matched no-policy run on the same images, guidance, adapters and seeds
(J5, 2.064). Sources: ~/verification/out/J4_perdir/eval_metrics.json,
~/verification/out/J5_none_perdir/eval_metrics.json, and each arm's
eval_metrics.json in the repository (checkpoints*/).

    python figures/make_fig1_lhr_lift.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS = [  # label, lift, annotated images
    ("Single-classifier", 2.293, 183),
    ("2-model ensemble (default)", 2.595, 148),
    ("3-model ensemble", 2.492, 27),
    ("Sparsity reward", 2.398, 148),
    ("Grad-CAM constraint", 1.636, 148),
]
CHANCE, NO_POLICY = 1.0, 2.064

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
BAR = "#2a78d6"  # single series: one hue for every bar

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
fig, ax = plt.subplots(figsize=(5.2, 2.3), dpi=300)
y = list(range(len(ARMS)))[::-1]
for yi, (name, lift, n) in zip(y, ARMS):
    ax.barh(yi, lift, height=0.62, color=BAR, edgecolor="white", linewidth=1.0, zorder=2)
    ax.text(lift + 0.04, yi, f"{lift:.3f}", va="center", ha="left", color=INK, fontsize=8.5,
            fontweight="bold" if "default" in name else "normal")
    ax.text(0.05, yi, f"n={n}", va="center", ha="left", color="white", fontsize=7)
for x, lab in ((CHANCE, "chance"), (NO_POLICY, "no policy")):
    ax.axvline(x, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.0, zorder=3)
    ax.text(x, len(ARMS) - 0.45, lab, ha="center", va="bottom", color=INK2, fontsize=7.5,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.0), zorder=4)
ax.set_yticks(y)
ax.set_yticklabels([a[0] for a in ARMS], color=INK)
for t in ax.get_yticklabels():
    if "default" in t.get_text():
        t.set_fontweight("bold")
ax.set_xlim(0, 3.0)
ax.set_ylim(-0.6, len(ARMS) - 0.1)
ax.set_xlabel("Lesion Hit Rate lift, p→n (1.0 = chance)", color=INK2, fontsize=8)
ax.xaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(axis="both", colors=INK2, length=0, labelsize=8)
fig.tight_layout(pad=0.3)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig1_lhr_lift.png")
fig.savefig(out, dpi=300, facecolor="white")
print("wrote", out)
