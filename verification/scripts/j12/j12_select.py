"""J12 selection (CPU). Per direction, among configs that have BOTH stage-1 and stage-2 rows in any of the given
select dirs (coarse grid, optionally the local grid), pick max mean discounted return over the same 48
training-fold images (12 stage-1 + 36 stage-2; image lists asserted identical across configs).

Return used = logged return + the stop bonus a config was denied only because its tau was <= 0.5:
the env pays R_stop = valid * [tau > 0.5] (factory/environment.py l.458), tau changes no image and no episode
length, and episodes end at the first valid step, so the bonus is exactly
LAM_STOP * direction_scale * GAMMA**(n_steps-1) on valid episodes (direction_scale 2.25 for n_to_p). The coarse
grid fixed tau = sigmoid(0) = 0.5, so without this every coarse config lost the bonus (part5/j12_coverage.py).
Configs with tau > 0.5 (local grid, REF) are unchanged. Ineligible rows (REF_policy_mean) are reported, never selected.

    python j12_select.py OUT.json SELECT_DIR [SELECT_DIR ...]
Writes the constant schedule in the fixed_schedule.json layout read by fixed_schedule_eval.py."""
import json, math, os, sys
sys.path.insert(0, os.path.expanduser('~/verification/code/49e6050'))
import config
out_path, dirs = sys.argv[1], sys.argv[2:]
sig = lambda x: 1 / (1 + math.exp(-x))
out = {}
for direction in ['n_to_p', 'p_to_n']:
    ds = 2.25 if direction == 'n_to_p' else 1.0
    cand, imgs = {}, None
    for d in dirs:
        load = lambda st: {r['name']: r for r in (json.load(open(os.path.join(d, f))) for f in os.listdir(d)
                                                   if f.startswith(f'stage{st}_{direction}_') and f.endswith('.json'))}
        s1, s2 = load(1), load(2)
        for n in s2:
            r1, r2 = s1[n], s2[n]
            assert r1['raw'] == r2['raw']
            rows = r1['rows'] + r2['rows']
            il = [x['dataset_idx'] for x in rows]
            imgs = imgs or il
            assert il == imgs and len(il) == 48, f'{direction} {n}: image list differs'
            tau = sig(r1['raw'][3])
            fix = lambda x: x['ret'] + (config.LAM_STOP * ds * config.GAMMA ** (x['n_steps'] - 1) if (x['valid'] and tau <= 0.5) else 0.0)
            key = f"{os.path.basename(os.path.dirname(os.path.abspath(d)))}:{n}"
            cand[key] = dict(name=n, src=d, raw=r1['raw'], controls=r1.get('controls'), eligible=r1['eligible'], tau=tau,
                             score=sum(map(fix, rows)) / 48, logged=sum(x['ret'] for x in rows) / 48,
                             valid=sum(x['valid'] for x in rows) / 48, steps=sum(x['n_steps'] for x in rows) / 48)
    elig = {k: v for k, v in cand.items() if v['eligible']}
    assert elig, f'{direction}: no eligible config with stage-2 rows'
    best = max(elig, key=lambda k: elig[k]['score'])
    rank = sorted(cand.items(), key=lambda kv: -kv[1]['score'])
    b = elig[best]
    out[direction] = dict(mean_raw_action=b['raw'], config=best, controls=b['controls'], tau=b['tau'],
                          train_fold_score_48=b['score'], train_fold_logged_48=b['logged'], valid_48=b['valid'], steps_48=b['steps'],
                          ranking=[(k, v['score'], v['eligible']) for k, v in rank], n_steps='n/a (J12 selection)', n_images=48,
                          selection_images=imgs, select_dirs=[os.path.abspath(d) for d in dirs])
    print(f"{direction}: selected {best}  score {b['score']:+.4f} (logged {b['logged']:+.4f}) valid {b['valid']:.3f} steps {b['steps']:.2f}")
    for k, v in rank[:8]: print(f"   {k:40s} {v['score']:+.4f} {'eligible' if v['eligible'] else 'REFERENCE (not eligible)'}")
tmp = f'{out_path}.{os.getpid()}.tmp'; json.dump(out, open(tmp, 'w'), indent=1); os.replace(tmp, out_path)
print('wrote', out_path)
