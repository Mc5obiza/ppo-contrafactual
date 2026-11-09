"""J8 paired CIs (decision 6). CPU only, first 50 images per direction (LHR: the 47 annotated p->n images).
Settings: d0 (dilation 0, feather 0.0), d2 (2, 1.0 = committed setting, rerun), d4 (4, 2.0); committed = committed shards 0+1.
Per image: CV indicator (2-ens guidance margin >= 0.8), LHR lift, edit concentration (top-5% mass), frac changed (>0.01),
guidance and validator deltas. Paired percentile bootstrap (10k, images resampled jointly across settings, per direction).
Transfer = ratio of means, resampled the same way; T = mean over nih/pc/mimic_ch.
Rerun noise = d2 - committed (same setting, different run). 'exceeds noise' = the 95% CI of the difference lies
entirely outside [-|noise|, +|noise|], noise = point estimate of d2 - committed for that metric."""
import os, sys, json, numpy as np, torch
os.environ['PPO_DOMAIN'] = 'rsna_pneumonia'
HEAD = os.path.expanduser('~/verification/code/HEAD'); sys.path.insert(0, HEAD); os.chdir(HEAD)
import config; config.DEVICE = 'cpu'
import arm_transfer as AT
from scipy.stats import binomtest
H = os.path.expanduser('~/verification')
dirs = {k: f'{H}/out/J8_mask_{k}/shards' for k in ['d0_f0.0', 'd2_f1.0', 'd4_f2.0']}
dirs['committed'] = f'{H}/part3/j8_noise/committed_first50'
P = json.load(open(H + '/part3/j8_noise/per_image_lhr.json'))
bs, be = AT._builders('Pneumonia')
G = be(['densenet121-res224-rsna', 'densenet121-res224-chex'])
V = {v.split('-')[-1]: bs(v) for v in ['densenet121-res224-nih', 'densenet121-res224-pc', 'densenet121-res224-mimic_ch']}
def conc_frac(X, Xc):
    c, f = [], []
    for x, xc in zip(X, Xc):
        d = (xc.float() - x.float()).abs().flatten(); assert d.sum() > 0
        k = max(1, int(round(0.05 * d.numel()))); c.append((torch.topk(d, k).values.sum() / d.sum()).item()); f.append((d > 0.01).float().mean().item())
    return np.array(c), np.array(f)
per = {}
for tag, d in dirs.items():
    xn, xp, xc_np, xc_pn, _ = AT._load_shards(d)
    s = torch.load(d + '/cf_shard_000.pt', map_location='cpu', weights_only=False)
    r = dict(idx_n=list(s['xn_idx']), idx_p=list(s['xp_idx']))
    r['g_np'], r['g_pn'] = AT._per_image_deltas(G, xn, xp, xc_np, xc_pn)
    for k, c in V.items(): r[f'v_{k}_np'], r[f'v_{k}_pn'] = AT._per_image_deltas(c, xn, xp, xc_np, xc_pn)
    r['conc_np'], r['frac_np'] = conc_frac(xn, xc_np); r['conc_pn'], r['frac_pn'] = conc_frac(xp, xc_pn)
    lk = sorted(P[d]); r['lhr_keys'] = lk; r['lhr'] = np.array([P[d][k]['lift'] if isinstance(P[d][k], dict) else P[d][k] for k in lk])
    per[tag] = r; print('scored', tag, flush=True)
ref = per['d2_f1.0']
for t in per: assert per[t]['idx_n'] == ref['idx_n'] and per[t]['idx_p'] == ref['idx_p'] and per[t]['lhr_keys'] == ref['lhr_keys']
n = len(ref['g_np']); nl = len(ref['lhr']); rng = np.random.default_rng(0); B = 10000
I = rng.integers(0, n, (B, n)); IL = rng.integers(0, nl, (B, nl))
def stat(r, m, idx=None):
    sel = (lambda a: a) if idx is None else (lambda a: a[idx])
    dn = m.split('_')[-1]
    if m.startswith('CV'): return (sel(r['g_' + dn]) >= 0.8).mean(-1)
    if m == 'LHR': return sel(r['lhr']).mean(-1)
    if m.startswith(('conc', 'frac')): return sel(r[m]).mean(-1)
    g = sel(r['g_' + dn]).mean(-1)
    if m.startswith('T_'): return np.mean([sel(r[f'v_{k}_{dn}']).mean(-1) / g for k in V], axis=0)
    return sel(r[f'v_{m.rsplit("_", 1)[0]}_{dn}']).mean(-1) / g    # e.g. nih_pn, mimic_ch_pn
mets = ['CV_np', 'CV_pn', 'LHR', 'conc_np', 'conc_pn', 'frac_np', 'frac_pn', 'T_np', 'T_pn'] + [f'{k}_{d}' for d in ['np', 'pn'] for k in V]
def diff(a, b, m):
    idx = IL if m == 'LHR' else I
    pt = stat(per[a], m) - stat(per[b], m); bt = stat(per[a], m, idx) - stat(per[b], m, idx)
    lo, hi = np.percentile(bt, [2.5, 97.5]); out = dict(diff=float(pt), lo=float(lo), hi=float(hi))
    if m.startswith('CV'):
        dn = m.split('_')[-1]; ya = per[a]['g_' + dn] >= 0.8; yb = per[b]['g_' + dn] >= 0.8
        n10, n01 = int((ya & ~yb).sum()), int((~ya & yb).sum()); out.update(only_a=n10, only_b=n01, mcnemar_p=binomtest(n10, n10 + n01).pvalue if n10 + n01 else 1.0)
    return out
res = {'n_images': n, 'n_lhr': nl, 'values': {t: {m: float(stat(per[t], m)) for m in mets} for t in per}, 'noise_d2_minus_committed': {}, 'cmp': {}}
for m in mets: res['noise_d2_minus_committed'][m] = diff('d2_f1.0', 'committed', m)
for s in ['d0_f0.0', 'd4_f2.0']:
    for b in ['d2_f1.0', 'committed']:
        for m in mets:
            x = diff(s, b, m); nz = abs(res['noise_d2_minus_committed'][m]['diff'])
            x['ci_excludes_0'] = bool(x['lo'] > 0 or x['hi'] < 0); x['exceeds_noise'] = bool(x['lo'] > nz or x['hi'] < -nz)
            res['cmp'][f'{s} - {b}|{m}'] = x
json.dump(res, open(H + '/part4/j8_paired.json', 'w'), indent=1)
f = lambda x: f"{x['diff']:+.4f} [{x['lo']:+.4f}, {x['hi']:+.4f}]" + (f" ({x['only_a']} vs {x['only_b']}, p={x['mcnemar_p']:.3f})" if 'only_a' in x else '')
print(f"\nn = {n} images per direction; LHR n = {nl}. Feathering (Gaussian sigma, latent cells) / dilation: d0 0.0/0, d2 1.0/2, d4 2.0/4, committed 1.0/2.")
print('Rule: "exceeds noise" = 95% paired CI entirely outside +-|d2 - committed|.\n')
print('| metric | d2 | committed | noise d2 - committed [CI] |'); print('|---|---|---|---|')
for m in mets: print(f"| {m} | {res['values']['d2_f1.0'][m]:.4f} | {res['values']['committed'][m]:.4f} | {f(res['noise_d2_minus_committed'][m])} |")
for s in ['d0_f0.0', 'd4_f2.0']:
    print(f'\n### {s} (setting value {res["values"][s]["LHR"]:.4f} LHR shown in rows)\n')
    print('| metric | value | − d2 rerun [95% CI] | exceeds noise? | − committed [95% CI] | exceeds noise? |'); print('|---|---|---|---|---|---|')
    for m in mets:
        a, b = res['cmp'][f'{s} - d2_f1.0|{m}'], res['cmp'][f'{s} - committed|{m}']
        yn = lambda x: 'yes' if x['exceeds_noise'] else ('CI excl. 0 only' if x['ci_excludes_0'] else 'no')
        print(f"| {m} | {res['values'][s][m]:.4f} | {f(a)} | {yn(a)} | {f(b)} | {yn(b)} |")
