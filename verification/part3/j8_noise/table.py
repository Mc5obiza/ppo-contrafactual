import json, os
M = json.load(open(os.path.expanduser('~/verification/part3/j8_noise/metrics50.json')))
ref = M['d2_f1.0']; mets = ['CV_np','CV_pn','LHR','conc_np','conc_pn','frac_np','frac_pn','T_np','T_pn','nih_p_to_n','pc_p_to_n','mimic_ch_p_to_n']
print(f"{'metric':16s} {'d2_f1.0':>8s} | noise: |d2-committed| |d2-J4| | {'d0_f0.0':>8s} {'Δ':>7s} flag            | {'d4_f2.0':>8s} {'Δ':>7s} flag")
for m in mets:
    nc = abs(ref[m]-M['committed_50'][m]); nj = abs(ref[m]-M['J4_50'][m]); row = f"{m:16s} {ref[m]:8.4f} | {nc:8.4f}            {nj:7.4f} |"
    for s in ['d0_f0.0','d4_f2.0']:
        dlt = M[s][m]-ref[m]; f = 'within noise' if abs(dlt) <= nc else ('beyond noise' if abs(dlt) > max(nc, nj) else 'beyond committed-noise only')
        row += f" {M[s][m]:8.4f} {dlt:+7.4f} {f:15s} |"
    print(row)
