"""Size of the nested-wrapper logging bug on J6 step-pooled statistics: logged (buggy) vs repaired, per dimension, both directions."""
import json, glob, os, numpy as np
H = os.path.expanduser('~/verification/out')
keys = ['strength','cls_scale','pres_scale','tau','strength_boost','guidance_bias','ramp_start_frac','n_passes']
def load(sub, d):
    return [e for f in sorted(glob.glob(f'{H}/{sub}/actions_shard_*.json')) for e in json.load(open(f))[d]]
for d in ['n_to_p', 'p_to_n']:
    B = load('actions', d); P = load('actions_repaired', d)
    Ab = np.array([s['raw_action'] for e in B for s in e['steps']]); Ap = np.array([s['raw_action'] for e in P for s in e['steps']])
    Sb = np.array([e['steps'][0]['raw_action'] for e in B]); Sp = np.array([e['steps'][0]['raw_action'] for e in P])
    print(f'\n== {d}: logged {len(Ab)} step records vs {len(Ap)} real steps; step-0 actions identical: {np.array_equal(Sb, Sp)}; n_steps identical: {[e["n_steps"] for e in B]==[e["n_steps"] for e in P]}')
    print('  dim  |Δmean| (logged-repaired)  Δsd   | as fraction of repaired sd')
    for j in range(Ab.shape[1]):
        dm = Ab[:, j].mean() - Ap[:, j].mean(); ds = Ab[:, j].std() - Ap[:, j].std()
        print(f'  a{j}   {dm:+.3f}                    {ds:+.3f}  | {abs(dm)/Ap[:,j].std():.2f}')
    for k in ['guidance_bias', 'n_passes']:
        cb = np.array([s[k] for e in B for s in e['steps']]); cp = np.array([s[k] for e in P for s in e['steps']])
        f = (lambda c: np.mean(c >= 1.295)) if k == 'guidance_bias' else (lambda c: np.mean(c == 3))
        print(f'  {k} {"at bound (>=1.295)" if k=="guidance_bias" else "==3"}: logged {f(cb):.3f} vs repaired {f(cp):.3f}')
    lb = [len(e['steps']) for e in B]; print(f'  episode length from len(steps): logged mean {np.mean(lb):.2f} (max {max(lb)}) vs n_steps mean {np.mean([e["n_steps"] for e in P]):.2f}')
