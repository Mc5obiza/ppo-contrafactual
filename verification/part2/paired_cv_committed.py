"""Per-image CV (tau=0.8) under the committed 2-ens guidance, for paired McNemar tests between arms on the same images.
Re-scores saved shards on CPU; first checks that the per-image fractions reproduce each eval_metrics.json CV."""
import os, sys, json, torch
os.environ.setdefault('PPO_DOMAIN', 'rsna_pneumonia')
sys.path.insert(0, os.path.expanduser('~/verification/code/49e6050'))
import config
from src.xrv_classifier import build_ensemble_classifier
from src.metrics import score_batch
from scipy.stats import binomtest
H = os.path.expanduser('~/verification/out'); R = os.environ['PPO_CF_REPO']
arms = {'committed': R + '/checkpoints_ens_default', 'J4': H + '/J4_perdir'}
clf = build_ensemble_classifier(device='cpu', weights_list=['densenet121-res224-rsna', 'densenet121-res224-chex'])
L = lambda p: torch.load(p, map_location='cpu', weights_only=False)
V = {}; res = {}
for a, d in arms.items():
    sh = [L(f'{d}/shards' if a!='committed' else f'{R}/checkpoints/eval_shards_ens_default/cf_shard_{s:03d}.pt') if a=='committed' else L(f'{d}/shards/cf_shard_{s:03d}.pt') for s in range(8)]
    cat = lambda k: [t.unsqueeze(0) for r in sh for t in r[k]]
    f = {k: score_batch(cat(k), clf, 'cpu') for k in ['xn', 'xp', 'xc_np', 'xc_pn']}
    v_np = ((f['xc_np'] - f['xn']) >= 0.8); v_pn = ((f['xp'] - f['xc_pn']) >= 0.8)
    E = json.load(open(d + '/eval_metrics.json'))['CV']
    V[a] = (v_np, v_pn)
    res[a] = dict(cpu_cv_np=v_np.float().mean().item(), cpu_cv_pn=v_pn.float().mean().item(), json_cv_np=E['n_to_p'], json_cv_pn=E['p_to_n'])
    print(a, res[a], flush=True)
for a, b in [('committed', 'J4')]:
    for i, dn in enumerate(['n_to_p', 'p_to_n']):
        x, y = V[a][i], V[b][i]; only_a = int((x & ~y).sum()); only_b = int((~x & y).sum())
        p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
        res[f'{a}_vs_{b}_{dn}'] = dict(diff=(x.float().mean() - y.float().mean()).item(), only_first_valid=only_a, only_second_valid=only_b, mcnemar_exact_p=p)
        print(f'{a} vs {b} {dn}: diff {res[f"{a}_vs_{b}_{dn}"]["diff"]:+.3f}  discordant {only_a}/{only_b}  McNemar exact p={p:.3g}')
json.dump(res, open(os.path.expanduser('~/verification/part2/paired_cv_committed_J4.json'), 'w'), indent=1)
