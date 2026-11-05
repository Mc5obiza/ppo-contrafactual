"""
J12 — RL-free, grid-tuned FIXED schedule. Selection on TRAINING-FOLD images only.

Same env, generator path, per-direction adapters and constant-policy mechanism as J5b/J5c
(fixed_schedule_eval.py); only the constant changes. Each config is a constant raw action
built from a grid in CONTROL space (inverse of the env's sigmoid maps, factory/environment.py
l.504-523). Controls not on the grid sit at the midpoint of their range (raw 0); nothing is
taken from the trained policy. The J5c constant (policy mean) is scored as a REFERENCE row
only and is never eligible for selection.

Selection criterion (what PPO maximised): mean discounted episode return, gamma = config.GAMMA,
of the env's own reward. Validity on the guidance ensemble, steps and time are logged too.

Successive halving:
  stage 1: all 54 grid configs (+ reference) on the first N1 = 12 training-fold images per group
  stage 2: the top K = 6 eligible stage-1 configs on the next N2 = 36 training-fold images
Images: scan_groups(..., allowed_indices=ctx.train_indices()), 2-ens guidance, classifier groups,
dataset index order; asserted disjoint from the test fold. Seeds: 2_000_000 (+1_000_000 for p->n)
+ position, disjoint from every eval seed.

    python j12_grid_select.py --direction n_to_p|p_to_n --stage 1|2 --task k --n-tasks K --out DIR
"""
import argparse, itertools, json, math, os, sys, time
import torch
import config, pipeline, per_direction
from src.metrics import scan_groups
from src.inference import generate_counterfactual_with_policy

N1, N2, TOPK = 12, 36, 6
SEED0 = 2_000_000
logit = lambda p: math.log(p / (1 - p))
GRID = dict(strength=[0.25, 0.45, 0.65],       # 0.2 + 0.5*sig(a0)
            cls_scale=[0.50, 0.85, 1.15],      # 0.3 + 0.9*sig(a1)
            guidance_bias=[0.85, 1.05, 1.27],  # 0.70 + 0.60*sig(a7)
            n_passes=[2, 3])                   # 2 + [sig(a4)>.5] + [sig(a5)>.5]

def raw_action(strength, cls_scale, guidance_bias, n_passes):
    a = [0.0] * 9                               # midpoints: pres 0.9, tau 0.5, boost 1.0, ramp 0.3
    a[0] = logit((strength - 0.2) / 0.5); a[1] = logit((cls_scale - 0.3) / 0.9)
    a[7] = logit((guidance_bias - 0.70) / 0.60)
    a[4] = 2.0 if n_passes >= 3 else -2.0; a[5] = 2.0 if n_passes >= 4 else -2.0
    return a

def configs(direction):
    out = [dict(name='g' + '_'.join(f'{v}' for v in c), eligible=True, controls=dict(zip(GRID, c)),
                raw=raw_action(*c)) for c in itertools.product(*GRID.values())]
    ref = json.load(open(os.path.expanduser('~/verification/part3/J5c_schedule_dryrun.json')))[direction]['mean_raw_action']
    out.append(dict(name='REF_policy_mean', eligible=False, controls=None, raw=ref))
    return out

class ConstantPolicy(torch.nn.Module):
    def __init__(self, action, dtype):
        super().__init__(); self.register_buffer('action', torch.tensor(action, dtype=dtype))
    def get_action_and_value(self, obs, deterministic=False):
        b = obs.shape[0]; z = torch.zeros(b, device=obs.device)
        return self.action.to(obs.device).unsqueeze(0).expand(b, -1).clone(), z, z, z, z, z

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--direction', required=True); ap.add_argument('--stage', type=int, required=True)
    ap.add_argument('--task', type=int, default=0); ap.add_argument('--n-tasks', type=int, default=1)
    ap.add_argument('--out', required=True); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ctx = pipeline.build_common()
    pipeline.load_generator_weights(ctx.generator, ckpt_dir=config.CKPT_DIR)
    if per_direction.enabled(): per_direction.use(ctx.generator, a.direction)
    target = 1.0 if a.direction == 'n_to_p' else 0.0
    dp = pipeline.load_directional_policy(target, a.direction, ctx, ckpt_dir=config.CKPT_DIR)
    train = ctx.train_indices(); test = set(int(i) for i in ctx.eval_indices())
    xn, xn_idx, xp, xp_idx = scan_groups(ctx.dataset, ctx.classifier, N1 + N2, config.DEVICE,
                                         tag='j12', allowed_indices=train)
    xs, idx = (xn, xn_idx) if a.direction == 'n_to_p' else (xp, xp_idx)
    assert len(xs) == N1 + N2 and not (set(idx) & test), 'selection images must be training-fold only'
    sel = range(0, N1) if a.stage == 1 else range(N1, N1 + N2)
    cfgs = configs(a.direction)
    if a.stage == 2:
        s1 = [json.load(open(os.path.join(a.out, f))) for f in sorted(os.listdir(a.out)) if f.startswith(f'stage1_{a.direction}_')]
        s1 = [r for r in s1 if r['eligible']]
        assert len(s1) == len([c for c in cfgs if c['eligible']]), 'stage 1 incomplete'
        keep = {r['name'] for r in sorted(s1, key=lambda r: -r['mean_return'])[:TOPK]}
        cfgs = [c for c in cfgs if c['name'] in keep]
    mine = cfgs[a.task::a.n_tasks]
    env = dp.env; rewards = []
    orig = env.step
    def step(action):
        o, r, d, i = orig(action); rewards.append(float(r)); return o, r, d, i
    env.step = step
    p0 = next(dp.policy.parameters())
    off = SEED0 + (1_000_000 if a.direction == 'p_to_n' else 0)
    for c in mine:
        path = os.path.join(a.out, f"stage{a.stage}_{a.direction}_{c['name']}.json")
        if os.path.exists(path): print('[j12] done already', path); continue
        dp.policy = ConstantPolicy(c['raw'], p0.dtype).to(p0.device)
        rows = []
        for i in sel:
            rewards.clear(); torch.manual_seed(off + i); t0 = time.time()
            _xc, _traj, info = generate_counterfactual_with_policy(xs[i].to(config.DEVICE), dp, deterministic=True)
            rows.append(dict(dataset_idx=int(idx[i]), ret=sum(r * config.GAMMA ** k for k, r in enumerate(rewards)),
                             ret_undiscounted=sum(rewards), valid=bool(info.get('valid', False)),
                             n_steps=int(info['n_steps_used']), seconds=time.time() - t0))
        res = dict(c, stage=a.stage, direction=a.direction, n_images=len(rows), rows=rows,
                   mean_return=sum(r['ret'] for r in rows) / len(rows), valid_rate=sum(r['valid'] for r in rows) / len(rows),
                   mean_steps=sum(r['n_steps'] for r in rows) / len(rows), gpu=torch.cuda.get_device_name(0))
        json.dump(res, open(path + '.tmp', 'w'), indent=1); os.replace(path + '.tmp', path)
        print(f"[j12] {a.direction} stage{a.stage} {c['name']}: return {res['mean_return']:.3f} valid {res['valid_rate']:.2f} "
              f"steps {res['mean_steps']:.1f} {sum(r['seconds'] for r in rows):.0f}s", flush=True)

if __name__ == '__main__':
    main()
