"""Decision 4 (2026-11-01): (a) where the J6-mean schedule sits relative to the J12 grid, per control and direction;
(b) why REF beats the grid: tau. Grid raw a3 = 0 -> tau = sigmoid(0) = 0.5, and the env pays R_stop only if tau > 0.5
(factory/environment.py l.458). tau enters nothing else (l.513 -> reward and info only), so it changes no image and no
episode length. Episodes end at the first valid step (FIXED_EPISODE_LENGTH False), so the missing bonus is exactly
LAM_STOP * direction_scale * GAMMA**(n_steps-1) on valid episodes. Corrected return = logged return + that. CPU only."""
import json, glob, math, os
S = os.path.expanduser('~/verification/out/J12_grid/select')
sig = lambda x: 1 / (1 + math.exp(-x))
LAM_STOP, GAMMA = 2.0, 0.99
maps = [('strength', lambda a: 0.2 + 0.5 * sig(a[0])), ('cls_scale', lambda a: 0.3 + 0.9 * sig(a[1])),
        ('pres_scale', lambda a: 0.3 + 1.2 * sig(a[2])), ('tau', lambda a: sig(a[3])),
        ('n_passes', lambda a: 2 + (sig(a[4]) > .5) + (sig(a[5]) > .5)), ('strength_boost', lambda a: 0.85 + 0.30 * sig(a[6])),
        ('guidance_bias', lambda a: 0.70 + 0.60 * sig(a[7])), ('ramp_start_frac', lambda a: 0.1 + 0.4 * sig(a[8]))]
grid = dict(strength=[0.25, 0.45, 0.65], cls_scale=[0.50, 0.85, 1.15], guidance_bias=[0.85, 1.05, 1.27], n_passes=[2, 3])
fixed = dict(pres_scale=0.9, tau=0.5, strength_boost=1.0, ramp_start_frac=0.3)
rng = dict(strength=(0.2, 0.7), cls_scale=(0.3, 1.2), pres_scale=(0.3, 1.5), tau=(0, 1), n_passes=(2, 4), strength_boost=(0.85, 1.15), guidance_bias=(0.70, 1.30), ramp_start_frac=(0.1, 0.5))
sched = json.load(open(os.path.expanduser('~/verification/part3/J5c_schedule_dryrun.json')))
out = {}
for d in ['n_to_p', 'p_to_n']:
    a = sched[d]['mean_raw_action']; ds = 2.25 if d == 'n_to_p' else 1.0
    print(f'\n## {d}: J6-mean schedule vs grid\n\n| control | full range | J6 mean | grid values / fixed | nearest grid value | gap as % of range |\n|---|---|---|---|---|---|')
    for k, f in maps:
        v = f(a); g = grid.get(k, [fixed.get(k)]); near = min(g, key=lambda x: abs(x - v)); lo, hi = rng[k]
        print(f"| {k} | {lo}-{hi} | {v:.4f} | {'grid ' + str(g) if k in grid else 'fixed ' + str(g[0])} | {near} | {100 * abs(near - v) / (hi - lo):.1f}% |")
    res = {}
    for st in [1, 2]:
        for f in glob.glob(f'{S}/stage{st}_{d}_*.json'):
            r = json.load(open(f)); tau = sig(r['raw'][3])
            bonus = [(LAM_STOP * ds * GAMMA ** (x['n_steps'] - 1) if (x['valid'] and tau <= 0.5) else 0.0) for x in r['rows']]
            res.setdefault(r['name'], {})[st] = dict(logged=r['mean_return'], corrected=r['mean_return'] + sum(bonus) / len(bonus),
                                                    valid=r['valid_rate'], steps=r['mean_steps'], tau=tau, n=len(r['rows']), rows=r['rows'], bonus=bonus)
    s1 = sorted(((n, v[1]) for n, v in res.items() if n != 'REF_policy_mean'), key=lambda kv: -kv[1]['corrected'])
    top_logged = sorted((n for n in res if n != 'REF_policy_mean'), key=lambda n: -res[n][1]['logged'])[:6]
    top_corr = [n for n, _ in s1[:6]]
    ref = res['REF_policy_mean'][1]
    print(f"\nStage 1 (12 images): REF (tau {ref['tau']:.2f}, already paid) {ref['logged']:+.3f}. Top 8 grid by corrected return:\n")
    print('| config | logged | corrected | valid | steps | in stage 2 (logged top 6)? |\n|---|---|---|---|---|---|')
    for n, v in s1[:8]: print(f"| {n} | {v['logged']:+.3f} | {v['corrected']:+.3f} | {v['valid']:.2f} | {v['steps']:.2f} | {'yes' if n in top_logged else 'no'} |")
    print(f"\nTop 6 by corrected stage-1 return = top 6 by logged? {set(top_corr) == set(top_logged)}; overlap {len(set(top_corr) & set(top_logged))}/6")
    print(f"Grid configs whose corrected stage-1 return >= REF: {sum(v['corrected'] >= ref['logged'] for _, v in s1)}/54")
    s48 = {n: (sum(v[1]['rows'][i]['ret'] + v[1]['bonus'][i] for i in range(v[1]['n'])) + sum(v[2]['rows'][i]['ret'] + v[2]['bonus'][i] for i in range(v[2]['n']))) / (v[1]['n'] + v[2]['n'])
           for n, v in res.items() if 2 in v}
    print('Stage-2 configs, corrected mean return over 48 images:', {n: round(x, 3) for n, x in sorted(s48.items(), key=lambda kv: -kv[1])})
    out[d] = dict(stage1_corrected={n: v[1]['corrected'] for n, v in res.items()}, s48_corrected=s48, top6_logged=top_logged, top6_corrected=top_corr)
json.dump(out, open(os.path.expanduser('~/verification/part5/j12_coverage.json'), 'w'), indent=1)
