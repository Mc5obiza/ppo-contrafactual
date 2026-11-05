"""Item 3 (2026-11-01), pass 1 (code/HEAD, CPU): per-image scores for J4, J5, J5c, J9 on their saved shards.
Per image and direction: 2-ens guidance delta (CV = delta >= 0.8, as in eval), nih/pc/mimic_ch validator deltas
(arm_transfer._per_image_deltas), edit concentration (top-5% mass) and frac changed (>0.01) as in
src/metrics.edit_concentration. Saved to part5/arms_scores.npz-style JSON for pass 3."""
import os, sys, json, numpy as np, torch
os.environ['PPO_DOMAIN'] = 'rsna_pneumonia'
HEAD = os.path.expanduser('~/verification/code/HEAD'); sys.path.insert(0, HEAD); os.chdir(HEAD)
import config; config.DEVICE = 'cpu'
import arm_transfer as AT
H = os.path.expanduser('~/verification')
ARMS = {'J4': H + '/out/J4_perdir', 'J5': H + '/out/J5_none_perdir', 'J5c': H + '/out/J5c_fixed_schedule_corrected', 'J9': H + '/out/J9_seed1'}
bs, be = AT._builders('Pneumonia')
G = be(['densenet121-res224-rsna', 'densenet121-res224-chex'])
V = {v.split('-')[-1]: bs(v) for v in ['densenet121-res224-nih', 'densenet121-res224-pc', 'densenet121-res224-mimic_ch']}
def conc_frac(X, Xc):
    c, f = [], []
    for x, xc in zip(X, Xc):
        d = (xc.float() - x.float()).abs().flatten(); assert d.sum() > 0
        k = max(1, int(round(0.05 * d.numel()))); c.append((torch.topk(d, k).values.sum() / d.sum()).item()); f.append((d > 0.01).float().mean().item())
    return c, f
out = {}
for a, d in ARMS.items():
    xn, xp, xc_np, xc_pn, _ = AT._load_shards(d + '/shards')
    import glob; idx_n, idx_p = [], []
    for p in sorted(glob.glob(d + '/shards/cf_shard_*.pt')):
        s = torch.load(p, map_location='cpu', weights_only=False); idx_n += list(s['xn_idx']); idx_p += list(s['xp_idx'])
    r = dict(idx_n=[int(i) for i in idx_n], idx_p=[int(i) for i in idx_p])
    g_np, g_pn = AT._per_image_deltas(G, xn, xp, xc_np, xc_pn); r['g_np'], r['g_pn'] = g_np.tolist(), g_pn.tolist()
    for k, c in V.items():
        v_np, v_pn = AT._per_image_deltas(c, xn, xp, xc_np, xc_pn); r[f'v_{k}_np'], r[f'v_{k}_pn'] = v_np.tolist(), v_pn.tolist()
    r['conc_np'], r['frac_np'] = conc_frac(xn, xc_np); r['conc_pn'], r['frac_pn'] = conc_frac(xp, xc_pn)
    E = json.load(open(d + '/eval_metrics.json'))
    chk = dict(CV_np=(np.mean(np.array(r['g_np']) >= 0.8), E['CV']['n_to_p']), CV_pn=(np.mean(np.array(r['g_pn']) >= 0.8), E['CV']['p_to_n']),
               conc_np=(np.mean(r['conc_np']), E['EditConcentration']['n_to_p']['concentration']), frac_pn=(np.mean(r['frac_pn']), E['EditConcentration']['p_to_n']['frac_changed']))
    print(a, {k: f'{x:.5f} vs json {y:.5f}' for k, (x, y) in chk.items()}, flush=True)
    out[a] = r
json.dump(out, open(H + '/part5/arms_scores.json', 'w'))
