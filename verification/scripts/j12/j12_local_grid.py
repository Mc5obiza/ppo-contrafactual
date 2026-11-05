"""J12 LOCAL grid (proposed 2026-11-01; submitted 2026-11-03, user option a): constant schedules around the J6 mean action (policy mean,
part3/J5c_schedule_dryrun.json), per direction. Same images (48 training-fold images per group, same scan),
seeds, env, adapters, constant-policy mechanism, criterion and successive halving as j12_grid_select.py; only the
config list (and TOPK for p_to_n) changes. Results go to a separate select dir and are pooled with the coarse grid
by j12_select.py. The centre point of each local grid IS the J6 mean (i.e. the J5c constant); it is eligible.
NB: centring on the policy mean means this J12 variant is no longer RL-free.

Levels in control space, centre = J6 mean, steps ~10% of each control's range; a level set that would leave
the range is shifted inward (centre kept):
  n_to_p: strength {c, .225, .25} (c = 0.2001 sits on the lower bound), cls, pres, strength_boost, ramp  3 levels each
          -> 3^5 = 243 configs; guidance_bias, n_passes, tau at the J6 mean (bias 1.2991, 3 passes, tau 0.92)
  p_to_n: strength, cls, pres, strength_boost 3 levels each -> 81 configs; guidance_bias, ramp, n_passes, tau at the J6 mean
tau > 0.5 everywhere, so the stop bonus is paid as for the policy (no post-hoc correction needed)."""
import itertools, json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import j12_grid_select as G
sig = lambda x: 1 / (1 + math.exp(-x)); logit = lambda p: math.log(p / (1 - p))
MAPS = {0: (0.2, 0.5), 1: (0.3, 0.9), 2: (0.3, 1.2), 6: (0.85, 0.30), 8: (0.1, 0.4)}   # dim: (lo, scale)
NAMES = {0: 'strength', 1: 'cls_scale', 2: 'pres_scale', 6: 'strength_boost', 8: 'ramp_start_frac'}
STEP = {0: 0.05, 1: 0.09, 2: 0.12, 6: 0.03, 8: 0.04}
VARY = {'n_to_p': [0, 1, 2, 6, 8], 'p_to_n': [0, 1, 2, 6]}
TOPK = {'n_to_p': 6, 'p_to_n': 4}

def levels(dim, c):
    lo, sc = MAPS[dim]; eps = 1e-4
    s = STEP[dim]
    if dim == 0 and c - lo < 0.01: L = [c, 0.225, 0.25]                          # strength on its lower bound
    elif c + s > lo + sc: L = [c - 2 * s, c - s, c]                              # near the upper bound: shift inward
    elif c - s < lo: L = [c, c + s, c + 2 * s]                                   # near the lower bound: shift inward
    else: L = [c - s, c, c + s]
    assert all(lo < v < lo + sc for v in L), (dim, L)
    return L

def configs(direction):
    ref = json.load(open(os.path.expanduser('~/verification/part3/J5c_schedule_dryrun.json')))[direction]['mean_raw_action']
    centre = {d: MAPS[d][0] + MAPS[d][1] * sig(ref[d]) for d in VARY[direction]}
    out = []
    for combo in itertools.product(*[levels(d, centre[d]) for d in VARY[direction]]):
        raw = list(ref)
        for d, v in zip(VARY[direction], combo): raw[d] = logit((v - MAPS[d][0]) / MAPS[d][1])
        ctrl = {NAMES[d]: round(v, 4) for d, v in zip(VARY[direction], combo)}
        name = 'L' + '_'.join(f'{v:.4f}' for v in combo)
        out.append(dict(name=name, eligible=True, controls=ctrl, raw=raw,
                        is_centre=all(abs(v - centre[d]) < 1e-9 for d, v in zip(VARY[direction], combo))))
    assert sum(c['is_centre'] for c in out) == 1
    return out

if __name__ == '__main__':
    d = sys.argv[sys.argv.index('--direction') + 1]
    G.configs = configs; G.TOPK = TOPK[d]
    G.main()
