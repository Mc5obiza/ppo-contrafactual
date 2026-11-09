"""
Pneumonia AUROC of every guidance model and held-out validator on the RSNA test
fold, against RSNA's radiologist-derived labels (issue D8/E5).

Question it answers: is nih's low transfer ratio (0.044 n->p) a property of the
counterfactuals, or of a weak Pneumonia head? A validator that cannot rank real
RSNA test images is not evidence either way about the edits.

Run from ~/verification/code/49e6050 (J4 code: same split, preprocessing and
classifier wrapper as every reported number), see 29_validator_auroc.sbatch.

    python validator_auroc.py OUT.json

* Images: ctx.eval_indices() (the 20% patient-level test fold, 5,337 images),
  loaded once through ctx.dataset (256 px, min-max), scored by every model in
  the same batch. Labels: ctx.dataset.samples[i][1] (1 = at least one RSNA box).
* Models: XRVPneumoniaClassifier per weight set (the wrapper used for guidance
  and for held-out transfer), plus the rsna+chex mean ensemble (default guidance).
* AUROC: Mann-Whitney U with average ranks for ties; 95% CI = percentile
  bootstrap over images (2,000 resamples, seed 0, resampled jointly for all
  models so differences are paired).
* Also reported on the 400 J4 evaluation sources is NOT done here on purpose:
  those were selected by the ensemble's own score, so any AUROC there is biased.
"""
import json
import sys

import config  # project config first (torchxrayvision shadows `config`)
import numpy as np
import torch

from pipeline import build_common
from src.xrv_classifier import build_ensemble_classifier, build_xrv_classifier
from src.metrics import _to_rgb

WEIGHTS = {
    "rsna (guidance)": "densenet121-res224-rsna",
    "chex (guidance)": "densenet121-res224-chex",
    "mimic_nb (3-model guidance)": "densenet121-res224-mimic_nb",
    "nih (validator)": "densenet121-res224-nih",
    "pc (validator)": "densenet121-res224-pc",
    "mimic_ch (validator)": "densenet121-res224-mimic_ch",
}
ENSEMBLE = ["densenet121-res224-rsna", "densenet121-res224-chex"]


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ss = s[order]
    i = 0
    while i < len(ss):                       # average ranks over ties
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    n1 = int(y.sum()); n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def main(out_path: str) -> None:
    device = config.DEVICE
    ctx = build_common()
    idx = [int(i) for i in ctx.eval_indices()]
    y = np.array([int(ctx.dataset.samples[i][1]) for i in idx], dtype=np.int64)
    print(f"[auroc] test fold: {len(idx)} images, {int(y.sum())} positive ({y.mean():.3f})", flush=True)

    models = {name: build_xrv_classifier(device=device, weights=w) for name, w in WEIGHTS.items()}
    models["rsna+chex ensemble (default guidance)"] = build_ensemble_classifier(device=device, weights_list=ENSEMBLE)
    scores = {name: [] for name in models}
    bs = 64
    with torch.no_grad():
        for k in range(0, len(idx), bs):
            x = torch.stack([ctx.dataset[i][0] for i in idx[k:k + bs]]).to(device)
            xr = _to_rgb(x)
            for name, m in models.items():
                scores[name].append(m(xr).squeeze(-1).float().cpu().numpy())
            if (k // bs) % 20 == 0:
                print(f"[auroc] {k + x.shape[0]}/{len(idx)}", flush=True)
    scores = {n: np.concatenate(v) for n, v in scores.items()}

    rng = np.random.default_rng(0)
    boots = [rng.integers(0, len(y), len(y)) for _ in range(2000)]
    out = {"n": len(y), "n_pos": int(y.sum()), "prevalence": float(y.mean()), "models": {}}
    for name, s in scores.items():
        point = auroc(y, s)
        bs_vals = [auroc(y[b], s[b]) for b in boots if 0 < y[b].sum() < len(b)]
        lo, hi = np.percentile(bs_vals, [2.5, 97.5])
        out["models"][name] = {"auroc": point, "ci95": [float(lo), float(hi)],
                               "mean_score_pos": float(s[y == 1].mean()),
                               "mean_score_neg": float(s[y == 0].mean())}
        print(f"[auroc] {name:40s} AUROC {point:.3f} [{lo:.3f}, {hi:.3f}]  "
              f"mean f | pos {s[y == 1].mean():.3f}  neg {s[y == 0].mean():.3f}", flush=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[auroc] wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1])
