"""
transfer_calibration.py - J20: second review, items 2 and 10.

Calibrates the held-out transfer ratio rho_v (Supp. definition, guidance =
rsna+chex mean-probability ensemble, Pneumonia head) from both ends, and checks
the validators' label semantics.

A. Test fold (ctx.eval_indices(), 5,337 images, RSNA labels):
   * AUROC of every validator head: Pneumonia, Consolidation, Infiltration
     (nih, pc), Lung Opacity (mimic_ch). RSNA's positive class is "Lung Opacity".
   * Real-pair CEILING: rho_real_v = (mean f_v | pos - mean f_v | neg) /
     (mean f_g | pos - mean f_g | neg), bootstrap CI over images.
   * Operating points per head: Youden-J threshold, and prevalence-matched
     threshold (top-prevalence fraction called positive).
B. Saved counterfactual arms (J4 default, J5 none, J12 tuned fixed, J15 Latent
   Shift, J10 CycleGAN): rho_v per head and direction (Pneumonia rows must
   reproduce transfer.json: clearance check), and DECISION-FLIP rates at each
   head's operating points: among sources on the source side of t_v, the share
   whose counterfactual lands on the target side.
C. Adversarial FLOOR on J4's exact source images: perturbations against the
   guidance ensemble only, then scored by the validators exactly as in B.
   FGSM at eps in {1,2,4,8}/255 and PGD (L_inf, 10 steps, step eps/4, no random start) at eps in
   {2,4,8}/255, and PGD-L2 (10 steps, step budget/4) with each image's budget equal to that image's
   own J4 counterfactual ||x_c - x||_2 (norm-matched). Pixel space [0,1], 256^2.
   Loss: -log f_g(x) toward positive (n->p), -log(1 - f_g(x)) toward negative.

    python transfer_calibration.py OUT_DIR
Run from ~/verification/code/49e6050 with PPO_CKPT_DIR=views/committed (split file).
"""
import glob
import json
import os
import re
import sys
import time

import config  # project config first (torchxrayvision shadows `config`)
import numpy as np
import torch

import ingestion
import splits
from src.xrv_classifier import build_ensemble_classifier, build_xrv_classifier

DEV = "cuda" if torch.cuda.is_available() else "cpu"
config.DEVICE = DEV  # config hard-codes "cuda"; CPU runs need the override
V = os.path.expanduser("~/verification/out")
ARMS = {"J4": f"{V}/J4_perdir/shards", "J5_none": f"{V}/J5_none_perdir/shards",
        "J12_fixed_tuned": f"{V}/J12_grid_fixed_schedule/shards",
        "J15_latent_shift": f"{V}/J15_latent_shift/shards", "J10_cyclegan": f"{V}/J10_cyclegan/shards"}
HEADS = {"nih": ["Pneumonia", "Consolidation", "Infiltration"],
         "pc": ["Pneumonia", "Consolidation", "Infiltration"],
         "mimic_ch": ["Pneumonia", "Consolidation", "Lung Opacity"]}
GUIDANCE = ["densenet121-res224-rsna", "densenet121-res224-chex"]
LIMIT = int(os.environ.get("LIMIT", "0")) or None
NBOOT = int(os.environ.get("NBOOT", "2000"))
RNG = np.random.default_rng(0)


def load_models():
    g = build_ensemble_classifier(device=DEV, weights_list=GUIDANCE)
    vals = {f"{v}:{h}": build_xrv_classifier(device=DEV, weights=f"densenet121-res224-{v}", pathology=h)
            for v, hs in HEADS.items() for h in hs}
    for m in [g, *vals.values()]:
        m.eval()
    return g, vals


@torch.no_grad()
def score(models: dict, x: torch.Tensor, bs=50) -> dict:
    out = {k: [] for k in models}
    for i in range(0, len(x), bs):
        xb = x[i:i + bs].to(DEV)
        for k, m in models.items():
            out[k].append(m(xb).squeeze(-1).float().cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def auroc(y, s):
    from scipy.stats import rankdata
    r = rankdata(s); n1 = int(y.sum()); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ratio_ci(num_terms, den_terms):
    """rho = mean(num)/mean(den) with paired percentile bootstrap over images."""
    num_terms, den_terms = np.asarray(num_terms), np.asarray(den_terms)
    pt = float(num_terms.mean() / den_terms.mean()) if den_terms.mean() != 0 else float("nan")
    bs = []
    for _ in range(NBOOT):
        b = RNG.integers(0, len(num_terms), len(num_terms))
        d = den_terms[b].mean()
        if d != 0:
            bs.append(num_terms[b].mean() / d)
    return {"rho": pt, "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}


def load_arm(d):
    paths = sorted(glob.glob(os.path.join(d, "cf_shard_*.pt")),
                   key=lambda p: int(re.search(r"(\d+)\.pt$", p).group(1)))
    bl = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    cat = lambda k: torch.cat([b[k] for b in bl]).float()[:LIMIT]
    return {"n_to_p": (cat("xn"), cat("xc_np")), "p_to_n": (cat("xp"), cat("xc_pn"))}


def evaluate_pairs(g, vals, x, xc, s, ops, tag):
    """rho per head and decision flips for one (source, edited) set. s = +1 n->p, -1 p->n."""
    allm = {"guidance": g, **vals}
    f0, f1 = score(allm, x), score(allm, xc)
    den = s * (f1["guidance"] - f0["guidance"])
    res = {"n": int(len(x)), "guidance_delta": float(den.mean()),
           "guidance_cv_tau0.5": float(((f1["guidance"] >= 0.5) if s > 0 else (f1["guidance"] < 0.5)).mean()),
           "guidance_cv_tau0.8": float(((f1["guidance"] >= 0.8) if s > 0 else (f1["guidance"] <= 0.2)).mean()),
           "heads": {}}
    for k in allm:
        num = s * (f1[k] - f0[k])
        r = {"delta": float(num.mean()), **(ratio_ci(num, den) if k != "guidance" else {})}
        for op, t in ops.get(k, {}).items():
            src = (f0[k] < t) if s > 0 else (f0[k] >= t)
            hit = (f1[k] >= t) if s > 0 else (f1[k] < t)
            r[f"flip_{op}"] = {"t": float(t), "n_source_side": int(src.sum()),
                               "flip_rate": float((src & hit).sum() / max(src.sum(), 1)),
                               "target_side_rate_all": float(hit.mean())}
        res["heads"][k] = r
    print(f"[J20] {tag}: n={len(x)} gΔ={den.mean():.3f} " + " ".join(
        f"{k}={res['heads'][k].get('rho', float('nan')):.3f}" for k in vals), flush=True)
    return res, f0, f1


def attack(g, x, s, kind, eps, steps=10, l2_budget=None):
    """Untargeted-in-name, directional in fact: push f_g toward the target class."""
    x0 = x.to(DEV)
    adv = x0.clone()
    n_steps = 1 if kind == "fgsm" else steps
    for _ in range(n_steps):
        adv.requires_grad_(True)
        p = g(adv).squeeze(-1).clamp(1e-6, 1 - 1e-6)
        loss = (-torch.log(p) if s > 0 else -torch.log(1 - p)).sum()
        grad, = torch.autograd.grad(loss, adv)
        with torch.no_grad():
            if kind in ("fgsm", "pgd_linf"):
                step = eps if kind == "fgsm" else eps / 4
                adv = adv - step * grad.sign()
                adv = (x0 + (adv - x0).clamp(-eps, eps)).clamp(0, 1)
            else:  # pgd_l2, per-image budget
                b = l2_budget.to(DEV).view(-1, 1, 1, 1)
                gn = grad.flatten(1).norm(dim=1).view(-1, 1, 1, 1).clamp_min(1e-12)
                adv = adv - (b / 4) * grad / gn
                dlt = adv - x0
                dn = dlt.flatten(1).norm(dim=1).view(-1, 1, 1, 1).clamp_min(1e-12)
                adv = (x0 + dlt * torch.clamp(b / dn, max=1.0)).clamp(0, 1)
    return adv.detach().cpu()


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, "transfer_calibration.json")
    res = {"settings": {"guidance": GUIDANCE, "heads": HEADS, "nboot": NBOOT, "limit": LIMIT, "device": str(DEV)},
           "test_fold": {}, "arms": {}, "adversarial": {}}
    g, vals = load_models()
    allm = {"guidance": g, **vals}

    # ── A. test fold ──
    t0 = time.time()
    # Same dataset and persisted split as pipeline.build_common (J18), without
    # loading the generator: splits.load_or_create reads the fingerprinted file.
    dataset = ingestion.build_dataset(img_size=config.IMG_SIZE)
    split = splits.load_or_create(dataset, domain="rsna_pneumonia")
    idx = [int(i) for i in split.test][:LIMIT]
    y = np.array([int(dataset.samples[i][1]) for i in idx])
    sc = {k: [] for k in allm}
    for k0 in range(0, len(idx), 64):
        x = torch.stack([dataset[i][0] for i in idx[k0:k0 + 64]])
        for k, v in score(allm, x).items():
            sc[k].append(v)
    sc = {k: np.concatenate(v) for k, v in sc.items()}
    np.savez(os.path.join(out_dir, "test_fold_scores.npz"), idx=np.array(idx), y=y, **{k.replace(":", "__"): v for k, v in sc.items()})
    gpos, gneg = sc["guidance"][y == 1], sc["guidance"][y == 0]
    ops = {}
    for k, s_ in sc.items():
        # Youden J over observed thresholds (subsampled for speed)
        cand = np.quantile(s_, np.linspace(0.01, 0.99, 197))
        j = [((s_[y == 1] >= t).mean() - (s_[y == 0] >= t).mean()) for t in cand]
        t_y = float(cand[int(np.argmax(j))])
        t_prev = float(np.quantile(s_, 1 - y.mean()))
        ops[k] = {"youden": t_y, "prev_matched": t_prev}
        if k != "guidance":
            ops[k]["guidance_t0.5_equiv_rate"] = float(np.quantile(s_, 1 - (sc["guidance"] >= 0.5).mean()))
        entry = {"auroc": auroc(y, s_), "mean_pos": float(s_[y == 1].mean()), "mean_neg": float(s_[y == 0].mean()),
                 "thresholds": ops[k],
                 "sens_spec_youden": [float((s_[y == 1] >= t_y).mean()), float((s_[y == 0] < t_y).mean())]}
        if k != "guidance":
            # real-pair ceiling: population mean difference ratio, bootstrap over pos and neg separately
            pt = (s_[y == 1].mean() - s_[y == 0].mean()) / (gpos.mean() - gneg.mean())
            ip, ineg = np.where(y == 1)[0], np.where(y == 0)[0]
            bs = []
            for _ in range(NBOOT):
                bp, bn = RNG.choice(ip, len(ip)), RNG.choice(ineg, len(ineg))
                bs.append((s_[bp].mean() - s_[bn].mean()) / (sc["guidance"][bp].mean() - sc["guidance"][bn].mean()))
            entry["rho_real_ceiling"] = {"rho": float(pt), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
        res["test_fold"][k] = entry
        print(f"[J20] test fold {k:28s} AUROC {entry['auroc']:.3f}  ceiling "
              f"{entry.get('rho_real_ceiling', {}).get('rho', float('nan')):.3f}", flush=True)
    res["test_fold_meta"] = {"n": len(y), "n_pos": int(y.sum()), "seconds": time.time() - t0}
    json.dump(res, open(out_json, "w"), indent=1)

    # ── B. counterfactual arms ──
    j4 = None
    for tag, d in ARMS.items():
        if not glob.glob(os.path.join(d, "cf_shard_*.pt")):
            print(f"[J20] skip {tag}: no shards", flush=True); continue
        arm = load_arm(d)
        if tag == "J4":
            j4 = arm
        res["arms"][tag] = {}
        for direc, s in (("n_to_p", 1), ("p_to_n", -1)):
            x, xc = arm[direc]
            r, _, _ = evaluate_pairs(g, vals, x, xc, s, ops, f"{tag} {direc}")
            r["l2_mean"] = float((xc - x).flatten(1).norm(dim=1).mean())
            r["linf_mean"] = float((xc - x).flatten(1).abs().amax(dim=1).mean())
            res["arms"][tag][direc] = r
        json.dump(res, open(out_json, "w"), indent=1)

    # ── C. adversarial floor on J4's sources ──
    for direc, s in (("n_to_p", 1), ("p_to_n", -1)):
        x, xc = j4[direc]
        l2b = (xc - x).flatten(1).norm(dim=1)
        res["adversarial"][direc] = {}
        cfgs = [("fgsm", e) for e in (1, 2, 4, 8)] + [("pgd_linf", e) for e in (2, 4, 8)] + [("pgd_l2_matched", None)]
        for kind, e in cfgs:
            t0 = time.time()
            adv = torch.cat([attack(g, x[i:i + 25], s, "pgd_l2" if kind == "pgd_l2_matched" else kind,
                                    (e or 0) / 255.0, l2_budget=l2b[i:i + 25]) for i in range(0, len(x), 25)])
            name = f"{kind}_eps{e}/255" if e else kind
            r, _, _ = evaluate_pairs(g, vals, x, adv, s, ops, f"ADV {direc} {name}")
            r["l2_mean"] = float((adv - x).flatten(1).norm(dim=1).mean())
            r["linf_mean"] = float((adv - x).flatten(1).abs().amax(dim=1).mean())
            r["seconds"] = time.time() - t0
            res["adversarial"][direc][name] = r
            json.dump(res, open(out_json, "w"), indent=1)
    print(f"[J20] wrote {out_json}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
