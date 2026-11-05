"""J6 distribution-level action statistics from the repaired step records (out/actions_repaired). CPU only."""
import json, glob, os, collections, numpy as np
H = os.path.expanduser('~/verification/out/actions_repaired')
keys = ['strength','cls_scale','pres_scale','tau','strength_boost','guidance_bias','ramp_start_frac','n_passes']
for d in ['n_to_p', 'p_to_n']:
    rows = [e for f in sorted(glob.glob(H + '/actions_shard_*.json')) for e in json.load(open(f))[d]]
    A = np.array([s['raw_action'] for e in rows for s in e['steps']]); ns = np.array([e['n_steps'] for e in rows])
    multi = [e for e in rows if e['n_steps'] > 1]
    print(f'\n== {d}: {len(rows)} images, {len(A)} steps')
    print(f'  episode length: mean {ns.mean():.2f}, median {np.median(ns):.0f}, max {ns.max()} (cap 25); counts {dict(sorted(collections.Counter(ns.tolist()).items()))}')
    print(f'  images with >1 step: {len(multi)}')
    print('  raw action: mean | sd over all steps | sd of first-step action across images | mean within-image sd | between-image share of variance')
    for j in range(A.shape[1]):
        s0 = np.array([e['steps'][0]['raw_action'][j] for e in rows])
        if len(multi) >= 10:
            w = np.mean([np.var([s['raw_action'][j] for s in e['steps']]) for e in multi])
            b = np.var([np.mean([s['raw_action'][j] for s in e['steps']]) for e in multi])
            ws, sh = f'{np.sqrt(w):.3f}', f'{b/(b+w):.2f}'
        else: ws, sh = 'n/a', 'n/a'
        print(f'    a{j}: {A[:,j].mean():+.3f} | {A[:,j].std():.3f} | {s0.std():.3f} | {ws} | {sh}')
    print('  mapped controls (all steps): mean, sd, [min, max];  first-step sd across images')
    for k in keys:
        c = np.array([s[k] for e in rows for s in e['steps']]); c0 = np.array([e['steps'][0][k] for e in rows])
        print(f'    {k:16s} {c.mean():.4f}  sd {c.std():.4f}  [{c.min():.4f}, {c.max():.4f}];  first-step sd {c0.std():.4f}')
    gb = np.array([s['guidance_bias'] for e in rows for s in e['steps']]); npas = np.array([s['n_passes'] for e in rows for s in e['steps']])
    print(f'  guidance_bias (range 0.70-1.30): >=1.295 on {np.mean(gb>=1.295)*100:.1f}% of steps, >=1.29 on {np.mean(gb>=1.29)*100:.1f}%; n_passes distribution {dict(sorted(collections.Counter(npas.tolist()).items()))}')
