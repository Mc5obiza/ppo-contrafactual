"""J12 paired comparisons (CPU): per-image guidance-ensemble and validator deltas via arm_transfer's own
loaders/builders; CV = s*delta_guidance >= 0.8 (crossing redundant on X_n/X_p); exact McNemar; paired
bootstrap (10,000, seed 0) of mean-over-validators transfer ratio, images resampled jointly."""
import json, sys
import numpy as np
from scipy.stats import binomtest
import config; config.DEVICE = "cpu"
import arm_transfer as at, domains

V = "/home2020/home/insa/arbaoui/verification/out"
ARMS = {"J12": f"{V}/J12_grid_fixed_schedule/shards", "J4": f"{V}/J4_perdir/shards",
        "J5c": f"{V}/J5c_fixed_schedule_corrected/shards", "J5": f"{V}/J5_none_perdir/shards"}
GUID = ["densenet121-res224-rsna", "densenet121-res224-chex"]
single, ens = at._builders(domains.active().pathology)
vals = at._validators(None)
G = ens(GUID); VAL = {v: single(v) for v in vals}
D = {}
for a, d in ARMS.items():
    xn, xp, xc_np, xc_pn, _ = at._load_shards(d)
    g = dict(zip(("n_to_p", "p_to_n"), at._per_image_deltas(G, xn, xp, xc_np, xc_pn)))
    vv = {v: dict(zip(("n_to_p", "p_to_n"), at._per_image_deltas(VAL[v], xn, xp, xc_np, xc_pn))) for v in vals}
    D[a] = {"g": {k: np.asarray(x) for k, x in g.items()}, "v": {v: {k: np.asarray(x) for k, x in vv[v].items()} for v in vals}}
    print(a, "loaded", len(xn), len(xp), flush=True)

def ratio_mean(a, idx, k):
    return np.mean([D[a]["v"][v][k][idx].sum() / D[a]["g"][k][idx].sum() for v in vals])

out = {}
rng = np.random.default_rng(0)
for k, s in (("n_to_p", 1.0), ("p_to_n", -1.0)):
    out[k] = {}
    valid = {a: (s * D[a]["g"][k] >= 0.8) for a in ARMS}
    for a in ARMS:
        out[k][a] = {"CV": float(valid[a].mean()), "transfer_mean": float(ratio_mean(a, np.arange(len(valid[a])), k))}
    n = len(valid["J12"])
    idx = rng.integers(0, n, (10000, n))
    for b in ("J4", "J5c", "J5"):
        x, y = valid["J12"], valid[b]
        n10, n01 = int((x & ~y).sum()), int((~x & y).sum())
        p = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
        dif = np.array([ratio_mean("J12", i, k) - ratio_mean(b, i, k) for i in idx])
        out[k][f"J12_vs_{b}"] = {"dCV": float(x.mean() - y.mean()), "J12_only": n10, "other_only": n01, "mcnemar_p": float(p),
                                 "d_transfer": float(ratio_mean("J12", np.arange(n), k) - ratio_mean(b, np.arange(n), k)),
                                 "d_transfer_ci95": [float(q) for q in np.percentile(dif, [2.5, 97.5])]}
for k in ("n_to_p", "p_to_n"):
    n = len(D["J12"]["g"][k]); idx = np.random.default_rng(0).integers(0, n, (10000, n))
    for a in ARMS:
        b = np.array([ratio_mean(a, i, k) for i in idx])
        out[k][a]["transfer_mean_ci95"] = [float(q) for q in np.percentile(b, [2.5, 97.5])]
np.savez(sys.argv[1].replace(".json", "_per_image.npz"), **{f"{a}_{w}_{k}": (D[a]["g"][k] if w == "g" else D[a]["v"][w][k])
         for a in ARMS for w in ["g"] + vals for k in ("n_to_p", "p_to_n")})
print(json.dumps(out, indent=1))
json.dump(out, open(sys.argv[1], "w"), indent=1)
