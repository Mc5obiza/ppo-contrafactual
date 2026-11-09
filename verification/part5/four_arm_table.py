"""Item 3 (2026-11-01), pass 3: one table per direction for J4 (committed policy, per-direction adapters), J5 (no policy),
J5c (fixed schedule = policy mean action), J9 (second PPO seed; shared adapter; both directions resumed from checkpoints,
n->p at update 15, p->n at update 80; no RNG state saved). Same 400 test images and per-image seeds in all four.
Point estimates: eval_metrics.json / transfer.json (GPU), which the CPU per-image recomputation reproduces
(checked in arms_scores.log for CV/concentration/frac and below for transfer and LHR).
CIs: CV Wilson 95%; LHR lift t-interval over the 148 annotated p->n images; concentration / frac changed t-interval over
200 images; transfer per validator = the GPU 10k percentile bootstrap from transfer.json; transfer mean = 10k paired
percentile bootstrap of the mean of the three ratios-of-means (CPU per-image deltas, images resampled jointly).
Paired tests: LHR paired t (+ Wilcoxon), d = mean(diff)/sd(diff); CV McNemar exact, CI = paired bootstrap, d likewise.
Recovery = (J5c - J5) / (J4 - J5), paired bootstrap over images (10k, percentile)."""
import json, os, math, numpy as np
from scipy import stats
H = os.path.expanduser('~/verification'); P5 = H + '/part5'
S = json.load(open(P5 + '/arms_scores.json')); L = json.load(open(P5 + '/arms_lhr.json'))
D = {'J4': 'J4_perdir', 'J5': 'J5_none_perdir', 'J5c': 'J5c_fixed_schedule_corrected', 'J9': 'J9_seed1'}
lab = {'J4': 'committed policy (J4)', 'J5': 'no policy (J5)', 'J5c': 'fixed schedule, policy mean (J5c)', 'J9': 'J9 seed 1†'}
VAL = ['nih', 'pc', 'mimic_ch']; B = 10000; rng = np.random.default_rng(0)
E = {a: json.load(open(f'{H}/out/{d}/eval_metrics.json')) for a, d in D.items()}
T = {a: list(json.load(open(f'{H}/out/{d}/transfer.json'))['arms'].values())[0]['validators'] for a, d in D.items()}
lk = sorted(L[f'{H}/out/J4_perdir/shards'], key=int)
for a, d in D.items(): assert sorted(L[f'{H}/out/{d}/shards'], key=int) == lk
LH = {a: np.array([L[f'{H}/out/{d}/shards'][k] for k in lk]) for a, d in D.items()}
for a in D: assert S[a]['idx_n'] == S['J4']['idx_n'] and S[a]['idx_p'] == S['J4']['idx_p']
arr = lambda a, k: np.array(S[a][k])
n = len(S['J4']['g_np']); IB = rng.integers(0, n, (B, n)); ILB = rng.integers(0, len(lk), (B, len(lk)))
def wilson(p, n, z=1.96):
    c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n); return c - h, c + h
def tci(x):
    m = x.mean(); se = x.std(ddof=1) / math.sqrt(len(x)); q = stats.t.ppf(.975, len(x) - 1); return m, m - q * se, m + q * se
def tmean(a, dn, idx=None):
    s = (lambda v: v) if idx is None else (lambda v: v[idx])
    g = s(arr(a, 'g_' + dn)).mean(-1); return np.mean([s(arr(a, f'v_{k}_{dn}')).mean(-1) / g for k in VAL], axis=0)
res = {'table': {}, 'paired': {}, 'recovery': {}}
f3 = lambda m, lo, hi: f'{m:.3f} [{lo:.3f}, {hi:.3f}]'
for dn, DN in [('np', 'n_to_p'), ('pn', 'p_to_n')]:
    hdr = '| arm | CV (n=200) | ' + ('LHR lift (n=148) | ' if dn == 'pn' else '') + ' | '.join(f'transfer {k}' for k in VAL) + ' | transfer mean | edit concentration | fraction changed |'
    print(f'\n### {DN}\n\n{hdr}\n|' + '---|' * hdr.count('|')[:0].__len__() if False else f'\n### {DN}\n\n{hdr}')
    print('|' + '---|' * (hdr.count('|') - 1))
    for a in D:
        cv = E[a]['CV'][DN]; lo, hi = wilson(cv, n); row = dict(CV=[cv, lo, hi])
        cells = [f3(cv, lo, hi)]
        if dn == 'pn':
            m, l2, h2 = tci(LH[a]); assert abs(m - E[a]['LHR']['p_to_n']['lift']) < 5e-4, (a, m, E[a]['LHR']['p_to_n']['lift'])
            row['LHR'] = [E[a]['LHR']['p_to_n']['lift'], l2, h2, len(lk)]; cells.append(f3(E[a]['LHR']['p_to_n']['lift'], l2, h2))
        for k in VAL:
            V = T[a][f'densenet121-res224-{k}']; r, (cl, ch) = V[f'ratio_{DN}'], V[f'ci_{DN}']
            cpu = arr(a, f'v_{k}_{dn}').mean() / arr(a, 'g_' + dn).mean(); assert abs(cpu - r) < 2e-4, (a, k, dn, cpu, r)
            row[f'T_{k}'] = [r, cl, ch]; cells.append(f3(r, cl, ch))
        tm = np.mean([T[a][f'densenet121-res224-{k}'][f'ratio_{DN}'] for k in VAL]); bt = tmean(a, dn, IB); bl, bh = np.percentile(bt, [2.5, 97.5])
        row['T_mean'] = [tm, bl, bh]; cells.append(f3(tm, bl, bh))
        for key, ek in [('conc', 'concentration'), ('frac', 'frac_changed')]:
            m, l2, h2 = tci(arr(a, f'{key}_{dn}')); row[key] = [E[a]['EditConcentration'][DN][ek], l2, h2]; cells.append(f3(E[a]['EditConcentration'][DN][ek], l2, h2))
        res['table'][f'{a}_{DN}'] = row
        print(f'| {lab[a]} | ' + ' | '.join(cells) + ' |')
print('\n† J9: second PPO seed with a shared adapter (both directions use the same generator file); both directions were resumed from checkpoints (n→p at update 15, p→n at update 80 after a preemption) and no random-number state was saved.')
print('CIs: CV Wilson; LHR, concentration, fraction changed t-intervals; transfer per validator GPU bootstrap (10k); transfer mean paired bootstrap (10k, CPU per-image deltas).')
# paired comparisons
print('\n### Paired comparisons against the committed policy (J4)\n')
print('| comparison | metric | difference [95% CI] | d (paired) | p | detail |\n|---|---|---|---|---|---|')
for b in ['J5c', 'J9', 'J5']:
    x = LH[b] - LH['J4']; m, l2, h2 = tci(x); t, p = stats.ttest_rel(LH[b], LH['J4']); w = stats.wilcoxon(LH[b], LH['J4']).pvalue; dd = x.mean() / x.std(ddof=1)
    res['paired'][f'{b}-J4|LHR'] = dict(diff=m, lo=l2, hi=h2, d=dd, p_t=p, p_wilcoxon=w, n=len(x), better=int((x > 0).sum()))
    print(f'| {b} − J4 | LHR lift (p→n, n={len(x)}) | {m:+.3f} [{l2:+.3f}, {h2:+.3f}] | {dd:+.3f} | {p:.3g} (paired t); Wilcoxon {w:.3g} | {b} higher on {(x > 0).sum()}/{len(x)} |')
    for dn, DN in [('np', 'n→p'), ('pn', 'p→n')]:
        ya = (arr(b, 'g_' + dn) >= 0.8).astype(float); yb = (arr('J4', 'g_' + dn) >= 0.8).astype(float); x = ya - yb
        bt = x[IB].mean(-1); cl, ch = np.percentile(bt, [2.5, 97.5]); n10, n01 = int((x > 0).sum()), int((x < 0).sum())
        p = stats.binomtest(n10, n10 + n01).pvalue if n10 + n01 else 1.0; dd = x.mean() / x.std(ddof=1) if x.std() > 0 else 0.0
        res['paired'][f'{b}-J4|CV_{dn}'] = dict(diff=x.mean(), lo=cl, hi=ch, d=dd, p_mcnemar=p, only_b=n10, only_J4=n01, n=n)
        print(f'| {b} − J4 | CV {DN} (n={n}) | {x.mean():+.3f} [{cl:+.3f}, {ch:+.3f}] | {dd:+.3f} | {p:.3g} (McNemar exact) | valid only under {b}: {n10}, only under J4: {n01} |')
# recovery
print('\n### How much of the policy\'s gain over no policy the fixed schedule recovers\n')
print('| metric | policy gain J4 − J5 [95% CI] | fixed-schedule gain J5c − J5 [95% CI] | recovered (J5c − J5)/(J4 − J5) [95% CI] |\n|---|---|---|---|')
def rec(va, vb, vc, I):
    g4 = va[I].mean(-1) - vb[I].mean(-1); gc = vc[I].mean(-1) - vb[I].mean(-1); return g4, gc, gc / g4
for name, (va, vb, vc, I) in {'LHR lift (p→n, n=148)': (LH['J4'], LH['J5'], LH['J5c'], ILB),
                              'CV p→n (n=200)': tuple((arr(a, 'g_pn') >= 0.8).astype(float) for a in ['J4', 'J5', 'J5c']) + (IB,),
                              'CV n→p (n=200)': tuple((arr(a, 'g_np') >= 0.8).astype(float) for a in ['J4', 'J5', 'J5c']) + (IB,)}.items():
    p4, pc = va.mean() - vb.mean(), vc.mean() - vb.mean()
    g4, gc, r = rec(va, vb, vc, I); ok = np.isfinite(r)
    q = lambda v: np.percentile(v[ok], [2.5, 97.5])
    if abs(p4) < 0.02:
        print(f'| {name} | {p4:+.3f} [{q(g4)[0]:+.3f}, {q(g4)[1]:+.3f}] | {pc:+.3f} [{q(gc)[0]:+.3f}, {q(gc)[1]:+.3f}] | not defined: the policy gain is ~0 (ratio of near-zero numbers) |')
        res['recovery'][name] = dict(policy_gain=p4, fixed_gain=pc, ratio=None); continue
    rl, rh = q(r); res['recovery'][name] = dict(policy_gain=p4, fixed_gain=pc, ratio=pc / p4, lo=rl, hi=rh, frac_boot_finite=float(ok.mean()))
    print(f'| {name} | {p4:+.3f} [{q(g4)[0]:+.3f}, {q(g4)[1]:+.3f}] | {pc:+.3f} [{q(gc)[0]:+.3f}, {q(gc)[1]:+.3f}] | {100 * pc / p4:.0f}% [{100 * rl:.0f}%, {100 * rh:.0f}%] |')
json.dump(res, open(P5 + '/four_arm_table.json', 'w'), indent=1, default=float)
