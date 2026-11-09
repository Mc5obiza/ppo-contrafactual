"""
pick_best_counterfactual.py
============================
Scores every image in the committed-default arm's eval shards
(checkpoints/eval_shards_ens_default/) with the SAME 2-model ensemble that
guided them, ranks by decisive score movement + a visual-quality proxy
(low residual noise outside a clean flip), and saves the top candidates
per direction as Original-vs-Counterfactual panels (matching the existing
social_media/linkedin_before_after_*.png style) for manual review.

Run via srun (needs the xrv ensemble classifier + GPU/CPU):
    module load python/3.12.8 && source venv/bin/activate
    python pick_best_counterfactual.py
"""
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import config
from src.xrv_classifier import build_ensemble_classifier

SHARD_DIR = "checkpoints/eval_shards_ens_default"
OUT_DIR = "outputs/best_counterfactual_candidates"
TOP_K = 6
os.makedirs(OUT_DIR, exist_ok=True)


def load_shards(shard_dir):
    paths = sorted(glob.glob(os.path.join(shard_dir, "cf_shard_*.pt")))
    blobs = {}
    for p in paths:
        m = re.search(r"cf_shard_(\d+)\.pt$", p)
        blobs[int(m.group(1))] = torch.load(p, map_location="cpu", weights_only=False)
    order = sorted(blobs)
    cat = lambda k: torch.cat([blobs[i][k] for i in order])
    cat_idx = lambda k: [i for b in order for i in blobs[b][k]]
    return (cat("xn"), cat("xp"), cat("xc_np"), cat("xc_pn"),
            cat_idx("xn_idx"), cat_idx("xp_idx"))


def score(clf, X, device, bs=32):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            x = X[i:i + bs].to(device)
            x = x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x
            out.append(clf(x).squeeze(-1).float().cpu())
    return torch.cat(out)


def save_panel(orig, cf, f_orig, f_cf, direction, rank, extra_tag=""):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(orig[0].numpy(), cmap="gray")
    axes[0].set_title(f"Original\nf(x) = {f_orig:.3f}", fontsize=13)
    axes[0].axis("off")
    axes[1].imshow(cf[0].numpy(), cmap="gray")
    target = "pneumonia" if direction == "n_to_p" else "normal"
    axes[1].set_title(f"Counterfactual ({target})\nf(x_cf) = {f_cf:.3f}", fontsize=13)
    axes[1].axis("off")
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f"{direction}_rank{rank}{extra_tag}.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    device = config.DEVICE
    print("[pick] loading ensemble classifier...")
    clf = build_ensemble_classifier(device=device)
    clf.eval()

    print("[pick] loading shards...")
    xn, xp, xc_np, xc_pn, xn_idx, xp_idx = load_shards(SHARD_DIR)

    print("[pick] scoring n_to_p (xn -> xc_np, target=pneumonia)...")
    f_xn = score(clf, xn, device)
    f_xc_np = score(clf, xc_np, device)
    delta_np = f_xc_np - f_xn  # want: xn low, xc_np high -> large positive delta, xc_np close to 1

    print("[pick] scoring p_to_n (xp -> xc_pn, target=normal)...")
    f_xp = score(clf, xp, device)
    f_xc_pn = score(clf, xc_pn, device)
    delta_pn = f_xp - f_xc_pn  # want: xp high, xc_pn low -> large positive delta, xc_pn close to 0

    # residual L1 magnitude, as a diffuseness/artifact proxy (lower = cleaner-looking edit)
    res_np = (xc_np - xn).abs().flatten(1).mean(dim=1)
    res_pn = (xc_pn - xp).abs().flatten(1).mean(dim=1)

    # Rank by decisive flip (xc close to target extreme) first, then by delta, penalize large residual
    score_np = f_xc_np - 0.15 * res_np * 10  # residual is small (~0.01-0.05), scale for comparability
    score_pn = (1 - f_xc_pn) - 0.15 * res_pn * 10

    top_np = torch.argsort(score_np, descending=True)[:TOP_K]
    top_pn = torch.argsort(score_pn, descending=True)[:TOP_K]

    print("\n=== n_to_p candidates (target: pneumonia) ===")
    for rank, i in enumerate(top_np.tolist()):
        print(f"  rank {rank}: idx={xn_idx[i]} f(x)={f_xn[i]:.3f} -> f(x_cf)={f_xc_np[i]:.3f} "
              f"delta={delta_np[i]:.3f} residual={res_np[i]:.4f}")
        save_panel(xn[i], xc_np[i], f_xn[i].item(), f_xc_np[i].item(), "n_to_p", rank)

    print("\n=== p_to_n candidates (target: normal) ===")
    for rank, i in enumerate(top_pn.tolist()):
        print(f"  rank {rank}: idx={xp_idx[i]} f(x)={f_xp[i]:.3f} -> f(x_cf)={f_xc_pn[i]:.3f} "
              f"delta={delta_pn[i]:.3f} residual={res_pn[i]:.4f}")
        save_panel(xp[i], xc_pn[i], f_xp[i].item(), f_xc_pn[i].item(), "p_to_n", rank)

    print(f"\n[pick] wrote candidates to {OUT_DIR}/ — review and pick the best one per direction.")


if __name__ == "__main__":
    main()
