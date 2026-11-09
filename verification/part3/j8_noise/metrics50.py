"""CPU metrics on 50-image dirs: CV (2-ens, tau .8), edit concentration + frac changed, LHR lift, transfer (arm_transfer.py, HEAD).
Same code for all five dirs; J8 rows are checked against their GPU eval_metrics.json / transfer.json."""
import os, sys, json, io, contextlib, runpy, statistics as st
os.environ['PPO_DOMAIN'] = 'rsna_pneumonia'
HEAD = os.path.expanduser('~/verification/code/HEAD'); sys.path.insert(0, HEAD); os.chdir(HEAD)
import config; config.DEVICE = 'cpu'
import torch
from src.xrv_classifier import build_ensemble_classifier
from src.metrics import score_batch, edit_concentration
H = os.path.expanduser('~/verification')
dirs = {k: f'{H}/out/J8_mask_{k}/shards' for k in ['d0_f0.0', 'd2_f1.0', 'd4_f2.0']}
dirs.update(committed_50=f'{H}/part3/j8_noise/committed_first50', J4_50=f'{H}/part3/j8_noise/J4_first50')
P = json.load(open(H + '/part3/j8_noise/per_image_lhr.json'))
clf = build_ensemble_classifier(device='cpu', weights_list=['densenet121-res224-rsna', 'densenet121-res224-chex'])
out = {}
for tag, d in dirs.items():
    s = torch.load(d + '/cf_shard_000.pt', map_location='cpu', weights_only=False)
    li = lambda k: [t.unsqueeze(0) for t in s[k]]
    f = {k: score_batch(li(k), clf, 'cpu') for k in ['xn', 'xp', 'xc_np', 'xc_pn']}
    ec_np = edit_concentration(li('xn'), li('xc_np')); ec_pn = edit_concentration(li('xp'), li('xc_pn'))
    lifts = [v['lift'] if isinstance(v, dict) else v for v in P[d].values()]
    sys.argv = ['arm_transfer.py', d, '--json', f'{H}/part3/j8_noise/transfer_{tag}.json']
    with contextlib.redirect_stdout(io.StringIO()):
        try: runpy.run_path(HEAD + '/arm_transfer.py', run_name='__main__')
        except SystemExit: pass
    V = list(json.load(open(f'{H}/part3/j8_noise/transfer_{tag}.json'))['arms'].values())[0]['validators']
    out[tag] = dict(CV_np=((f['xc_np'] - f['xn']) >= 0.8).float().mean().item(), CV_pn=((f['xp'] - f['xc_pn']) >= 0.8).float().mean().item(),
                    LHR=sum(lifts) / len(lifts), LHR_n=len(lifts),
                    conc_np=ec_np['concentration'], conc_pn=ec_pn['concentration'], frac_np=ec_np['frac_changed'], frac_pn=ec_pn['frac_changed'],
                    T_np=st.mean(v['ratio_n_to_p'] for v in V.values()), T_pn=st.mean(v['ratio_p_to_n'] for v in V.values()),
                    **{f'{k.split("-")[-1]}_{dn}': v[f'ratio_{dn}'] for k, v in V.items() for dn in ['n_to_p', 'p_to_n']})
    print(tag, {k: round(v, 4) for k, v in out[tag].items()}, flush=True)
json.dump(out, open(H + '/part3/j8_noise/metrics50.json', 'w'), indent=1)
# self-check vs J8 GPU outputs
for k in ['d0_f0.0', 'd2_f1.0', 'd4_f2.0']:
    E = json.load(open(f'{H}/out/J8_mask_{k}/eval_metrics.json')); T = list(json.load(open(f'{H}/out/J8_mask_{k}/transfer.json'))['arms'].values())[0]['validators']
    g = dict(CV_np=E['CV']['n_to_p'], CV_pn=E['CV']['p_to_n'], LHR=E['LHR']['p_to_n']['lift'], conc_np=E['EditConcentration']['n_to_p']['concentration'],
             conc_pn=E['EditConcentration']['p_to_n']['concentration'], frac_np=E['EditConcentration']['n_to_p']['frac_changed'], frac_pn=E['EditConcentration']['p_to_n']['frac_changed'],
             T_np=st.mean(v['ratio_n_to_p'] for v in T.values()), T_pn=st.mean(v['ratio_p_to_n'] for v in T.values()))
    print('self-check', k, 'max |CPU - GPU json| =', {m: f'{abs(out[k][m] - g[m]):.1e}' for m in g})
