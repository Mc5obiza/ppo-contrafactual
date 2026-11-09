"""
Main-paper qualitative figure: three J4 (default policy) examples chosen by a stated rule,
not by eye. CPU, offline from saved shards.

  (a) n->p, typical : among CV-valid n->p images, the one at the median (lower middle) of the
                      mean held-out shift (mean over nih, pc, mimic_ch of f_v(x_c) - f_v(x)).
  (b) p->n, typical : among CV-valid, annotated p->n images, median in-lung lift (Eq. 5).
  (c) p->n, failure : among CV-valid, annotated p->n images, the lowest in-lung lift.

Inputs: J4 shards, out/J23_qual/J4_scores.npz (qual_scores.py, checked against transfer.json),
per-image in-lung lift from out/J19_lhr_nulls/lhr_nulls_per_image.npz (J4__idx, J4__lift_lung).
Validity is recomputed from the guidance scores (Eq. 4, tau = 0.8) and must reproduce J4's
189/200 and 134/200. Writes FIG.pdf and a JSON of every number printed on the figure.

Run from ~/verification/code/49e6050 with PYTHONPATH=$PWD:$HOME/verification/scripts:
    python fig_examples.py OUT_PREFIX
"""
from __future__ import annotations

import json
import os
import sys

import config  # noqa: F401  (before torchxrayvision)
import numpy as np
import torch

config.DEVICE = "cpu"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle  # noqa: F401

import arm_transfer as at
import ingestion
import offline_metrics as om
from src.models import apply_torch_load_patch, revert_torch_load_patch

V = os.path.expanduser("~/verification/out")
VALS = ("nih", "pc", "mimic_ch")


def main(prefix: str) -> None:
    sc = np.load(f"{V}/J23_qual/J4_scores.npz")
    xn, xp, xc_np, xc_pn, _ = at._load_shards(f"{V}/J4_perdir/shards")
    g = {k: sc[f"guidance__{k}"] for k in ("xn", "xc_np", "xp", "xc_pn")}
    valid_np = (g["xc_np"] - g["xn"] >= 0.8) & (g["xc_np"] > 0.5) & (g["xn"] <= 0.5)
    valid_pn = (g["xp"] - g["xc_pn"] >= 0.8) & (g["xc_pn"] < 0.5) & (g["xp"] >= 0.5)
    print(f"valid n->p {valid_np.sum()}  p->n {valid_pn.sum()}")
    assert valid_np.sum() == 189 and valid_pn.sum() == 134, "validity does not reproduce J4"

    held_np = np.mean([sc[f"{v}__xc_np"] - sc[f"{v}__xn"] for v in VALS], axis=0)

    z = np.load(f"{V}/J19_lhr_nulls/lhr_nulls_per_image.npz", allow_pickle=True)
    lift = {int(j): float(l) for j, l in zip(z["J4__idx"], z["J4__lift_lung"])}
    xp_idx = sc["xp_idx"].astype(int)
    xn_idx = sc["xn_idx"].astype(int)
    assert abs(np.mean(list(lift.values())) - 1.1649) < 1e-3

    # (a) median held-out shift among valid n->p
    cand = np.where(valid_np)[0]
    order = cand[np.argsort(held_np[cand], kind="stable")]
    a = int(order[(len(order) - 1) // 2])
    # (b), (c) among valid, annotated p->n
    cand = [i for i in np.where(valid_pn)[0] if int(xp_idx[i]) in lift]
    order = sorted(cand, key=lambda i: (lift[int(xp_idx[i])], i))
    b = int(order[(len(order) - 1) // 2])
    c = int(order[0])
    print(f"valid annotated p->n: {len(cand)}")

    apply_torch_load_patch()
    import torchxrayvision as xrv
    seg = xrv.baseline_models.chestx_det.PSPNet().eval()
    revert_torch_load_patch()
    lung_idx = [seg.targets.index("Left Lung"), seg.targets.index("Right Lung")]
    ds = ingestion.build_dataset(img_size=config.IMG_SIZE)

    def item(direction, i):
        if direction == "np":
            x, xc, gi = xn[i], xc_np[i], int(xn_idx[i])
            s0, s1 = g["xn"][i], g["xc_np"][i]
            v0 = {v: float(sc[f"{v}__xn"][i]) for v in VALS}
            v1 = {v: float(sc[f"{v}__xc_np"][i]) for v in VALS}
            M = None
        else:
            x, xc, gi = xp[i], xc_pn[i], int(xp_idx[i])
            s0, s1 = g["xp"][i], g["xc_pn"][i]
            v0 = {v: float(sc[f"{v}__xp"][i]) for v in VALS}
            v1 = {v: float(sc[f"{v}__xc_pn"][i]) for v in VALS}
            M = ds.lesion_mask(gi).to(torch.float32)[0].numpy()
        with torch.no_grad():
            L = (om.lung_prob(seg, lung_idx, x[None].float()).cpu()[0, 0] > 0.5).numpy()
        return dict(x=x[0].numpy(), xc=xc[0].numpy(), L=L, M=M, idx=gi, f0=float(s0), f1=float(s1),
                    v0=v0, v1=v1, lift=lift.get(gi) if direction == "pn" else None,
                    held=float(np.mean([v1[v] - v0[v] for v in VALS])))

    ex = [("(a) n$\\to$p, typical", item("np", a)),
          ("(b) p$\\to$n, typical", item("pn", b)),
          ("(c) p$\\to$n, worst lesion hit", item("pn", c))]

    plt.rcParams.update({"font.size": 7, "font.family": "DejaVu Sans"})
    W = 6.875                       # CVPR \textwidth in inches
    fig, axes = plt.subplots(1, 9, figsize=(W, 1.30))
    plt.subplots_adjust(left=0.004, right=0.996, top=0.795, bottom=0.215, wspace=0.04)
    vmax = max(float(np.abs(e["xc"] - e["x"]).max()) for _, e in ex)
    for k, (title, e) in enumerate(ex):
        A = axes[3 * k: 3 * k + 3]
        d = np.abs(e["xc"] - e["x"])
        A[0].imshow(e["x"], cmap="gray", vmin=0, vmax=1)
        A[1].imshow(e["xc"], cmap="gray", vmin=0, vmax=1)
        A[2].imshow(d, cmap="magma", vmin=0, vmax=np.percentile(d, 99.5))
        for ax in (A[0], A[2]):
            ax.contour(e["L"], levels=[0.5], colors="#5ac8fa", linewidths=0.5)
            if e["M"] is not None:
                ax.contour(e["M"], levels=[0.5], colors="#ffd60a", linewidths=0.8)
        for ax, lab in zip(A, ("source", "counterfactual", "$|x_c - x|$")):
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_title(lab, fontsize=6.5, pad=1.5)
        x0 = A[0].get_position().x0; x1 = A[2].get_position().x1
        fig.text((x0 + x1) / 2, 0.955, title, ha="center", va="center", fontsize=7.5, weight="bold")
        line1 = f"guidance {e['f0']:.2f}$\\to${e['f1']:.2f}"
        line2 = ("held-out " + "/".join(f"{e['v0'][v]:.2f}" for v in VALS) + " $\\to$ "
                 + "/".join(f"{e['v1'][v]:.2f}" for v in VALS))
        if e["lift"] is not None:
            line1 += f"   in-lung LHR {e['lift']:.2f}"
        fig.text((x0 + x1) / 2, 0.135, line1, ha="center", va="center", fontsize=6.5)
        fig.text((x0 + x1) / 2, 0.048, line2, ha="center", va="center", fontsize=6.0)
    fig.savefig(prefix + ".pdf", dpi=400)
    fig.savefig(prefix + ".png", dpi=200)
    rec = {t: {k: v for k, v in e.items() if k not in ("x", "xc", "L", "M")} for t, e in ex}
    rec["rule"] = __doc__.split("Inputs:")[0].strip()
    rec["positions"] = {"a_np": a, "b_pn": b, "c_pn": c}
    json.dump(rec, open(prefix + ".json", "w"), indent=1)
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
