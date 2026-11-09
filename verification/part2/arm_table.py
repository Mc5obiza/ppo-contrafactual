"""Part 2 item 1: committed policy (J4) vs no-policy (J5) vs fixed schedule (J5b). Same 400 images/seeds."""
import json, os, statistics as st, math
from scipy import stats
import numpy as np
H = os.path.expanduser('~/verification'); R = os.path.expanduser('~/Jobs/Ayoub/ppo-contrafactual')
P = json.load(open(H + '/part2/per_image_lhr_arms.json'))
arms = {'committed (old, n->p adapter for p->n)': (R + '/checkpoints_ens_default', R + '/checkpoints/eval_shards_ens_default', None),
        'J4 policy, matched p->n adapter': (H + '/out/J4_perdir', H + '/out/J4_perdir/shards', H + '/out/J4_perdir/transfer.json'),
        'J5 no-policy': (H + '/out/J5_none_perdir', H + '/out/J5_none_perdir/shards', H + '/out/J5_none_perdir/transfer.json'),
        'J5b fixed schedule (mean action)': (H + '/out/J5b_fixed_schedule', H + '/out/J5b_fixed_schedule/shards', H + '/out/J5b_fixed_schedule/transfer.json')}
out = {}
for name, (d, sh, tr) in arms.items():
    E = json.load(open(d + '/eval_metrics.json')); x = np.array([v for v in P[sh].values()], float)
    def val(v): return v['lift'] if isinstance(v, dict) else v
    x = np.array([val(v) for v in P[sh].values()], float); n = len(x)
    m = x.mean(); se = x.std(ddof=1) / math.sqrt(n); tq = stats.t.ppf(0.975, n - 1)
    rng = np.random.default_rng(0); boot = np.sort([x[rng.integers(0, n, n)].mean() for _ in range(10000)])
    row = dict(CV_np=E['CV']['n_to_p'], CV_pn=E['CV']['p_to_n'], CV_mean=E['CV']['mean'],
               LHR_lift=E['LHR']['p_to_n']['lift'], LHR_lift_per_image_mean=m, LHR_n=n,
               LHR_CI_t=[m - tq * se, m + tq * se], LHR_CI_boot=[boot[249], boot[9749]],
               conc_np=E['EditConcentration']['n_to_p']['concentration'], conc_pn=E['EditConcentration']['p_to_n']['concentration'],
               frac_np=E['EditConcentration']['n_to_p']['frac_changed'], frac_pn=E['EditConcentration']['p_to_n']['frac_changed'],
               FID_np=E['FID']['n_to_p_vs_real_positive'], FID_pn=E['FID']['p_to_n_vs_real_negative'],
               MSSSIM_np=E['IP']['n_to_p']['ms_ssim'], MSSSIM_pn=E['IP']['p_to_n']['ms_ssim'],
               HF_np=E['IP']['n_to_p']['hf_corr'], HF_pn=E['IP']['p_to_n']['hf_corr'],
               LHR_t=E['LHR']['p_to_n']['t'], LHR_p=E['LHR']['p_to_n']['p'],
               CTR_dn=E['CTR']['mean_delta_n'], CTR_dp=E['CTR']['mean_delta_p'])
    if tr:
        A = list(json.load(open(tr))['arms'].values())[0]['validators']
        row['transfer'] = {v.split('-')[-1]: dict(np=V['ratio_n_to_p'], pn=V['ratio_p_to_n'], ci_np=V['ci_n_to_p'], ci_pn=V['ci_p_to_n']) for v, V in A.items()}
        row['transfer_mean_np'] = st.mean(V['ratio_n_to_p'] for V in A.values()); row['transfer_mean_pn'] = st.mean(V['ratio_p_to_n'] for V in A.values())
    out[name] = row
json.dump(out, open(H + '/part2/arm_table.json', 'w'), indent=1)
for k, r in out.items():
    print(f"\n## {k}")
    print(f"CV n->p {r['CV_np']:.3f}  p->n {r['CV_pn']:.3f}  mean {r['CV_mean']:.4f}")
    print(f"LHR lift {r['LHR_lift']:.3f} (per-image mean {r['LHR_lift_per_image_mean']:.4f}) n={r['LHR_n']}  t-CI [{r['LHR_CI_t'][0]:.3f}, {r['LHR_CI_t'][1]:.3f}]  boot-CI [{r['LHR_CI_boot'][0]:.3f}, {r['LHR_CI_boot'][1]:.3f}]  t={r['LHR_t']:.2f} p={r['LHR_p']:.2e}")
    print(f"conc n->p {r['conc_np']:.3f} p->n {r['conc_pn']:.3f} | frac n->p {r['frac_np']:.3f} p->n {r['frac_pn']:.3f}")
    print(f"FID n->p {r['FID_np']:.2f} p->n {r['FID_pn']:.2f} | MS-SSIM {r['MSSSIM_np']:.3f}/{r['MSSSIM_pn']:.3f} HF {r['HF_np']:.3f}/{r['HF_pn']:.3f} | CTR d {r['CTR_dn']:.4f}/{r['CTR_dp']:.4f}")
    if 'transfer' in r:
        for v, t in r['transfer'].items(): print(f"  {v:9s} n->p {t['np']:.3f}  p->n {t['pn']:.3f}")
        print(f"  MEAN      n->p {r['transfer_mean_np']:.3f}  p->n {r['transfer_mean_pn']:.3f}")
