"""Part 1 item 8: every value the reduce log prints == eval_metrics.json / transfer.json at printed precision."""
import json, re, sys
def num(s): return float(s)
def dp(s): return len(s.split('.')[1]) if '.' in s else 0
def close(printed, val, scale=1.0):
    return abs(round(val*scale, dp(printed)) - num(printed)) <= 0.51*10**(-dp(printed)) + 1e-12
def check(log, outdir):
    txt = open(log, errors='replace').read()
    m = list(re.finditer(re.escape(f"results saved to '{outdir}/eval_metrics.json'"), txt))
    assert m, 'no results-saved line for ' + outdir
    end = m[-1].start(); start = txt.rfind('► Counterfactual Validity', 0, end); blk = txt[start:end]
    E = json.load(open(outdir + '/eval_metrics.json')); bad = []; n = 0
    def cmp(name, printed, val, scale=1.0):
        nonlocal n; n += 1
        if not close(printed, val, scale): bad.append(f'{name}: log {printed} json {val}')
    cv = re.findall(r'(Xn → Xp|Xp → Xn|mean)\s*:\s*([\d.]+)\n', blk.split('► FID')[0])
    for k, v in cv: cmp('CV ' + k, v, E['CV'][{'Xn → Xp': 'n_to_p', 'Xp → Xn': 'p_to_n', 'mean': 'mean'}[k]])
    for lab, key in [('FID\\(Xn,  Xc_n→p\\)', 'negative_group'), ('FID\\(Xp,  Xc_p→n\\)', 'positive_group'),
                     ('FID\\(Xp,  Xc_n→p\\)', 'n_to_p_vs_real_positive'), ('FID\\(Xn,  Xc_p→n\\)', 'p_to_n_vs_real_negative')]:
        cmp('FID ' + key, re.search(lab + r' : ([\d.]+)', blk).group(1), E['FID'][key])
    for lab, d in [('Xn → Xp', 'n_to_p'), ('Xp → Xn', 'p_to_n')]:
        g = re.search(lab + r' : ([\d.]+)  \(uniform baseline [\d.]+\) \| MAE=([\d.]+) \| changed=([\d.]+)%', blk).groups()
        ec = E['EditConcentration'][d]
        cmp('conc ' + d, g[0], ec['concentration']); cmp('mae ' + d, g[1], ec['mae']); cmp('frac ' + d, g[2], ec['frac_changed'], 100)
        g = re.search(lab + r' : MS-SSIM=([\d.]+)  HF-corr=([\d.]+)', blk).groups()
        cmp('ms_ssim ' + d, g[0], E['IP'][d]['ms_ssim']); cmp('hf ' + d, g[1], E['IP'][d]['hf_corr'])
    L = E['LHR']['p_to_n']
    g = re.search(r'edit mass inside lesion : ([\d.]+) ± ([\d.]+)', blk).groups(); cmp('hit', g[0], L['hit_rate']); cmp('hit sd', g[1], L['std_hit_rate'])
    g = re.search(r'chance baseline \(area\)  : ([\d.]+) ± ([\d.]+)', blk).groups(); cmp('null', g[0], L['null_baseline']); cmp('null sd', g[1], L['std_null'])
    g = re.search(r'LIFT\s*: ([\d.]+)×  \[t=([\d.]+), p=([\de.+-]+), n=(\d+)\]', blk).groups()
    cmp('lift', g[0], L['lift']); cmp('lift t', g[1], L['t']); n += 1
    if int(g[3]) != L['n_pairs']: bad.append(f'n_pairs {g[3]} vs {L["n_pairs"]}')
    # one-line summary
    s = re.search(r'Evaluation metrics: CV=([\d.]+) \(n→p=([\d.]+), p→n=([\d.]+)\) \| FID_neg=([\d.]+) \| FID_pos=([\d.]+)', txt[end:]).groups()
    for v, j in zip(s, [E['CV']['mean'], E['CV']['n_to_p'], E['CV']['p_to_n'], E['FID']['negative_group'], E['FID']['positive_group']]): cmp('summary', v, j)
    # transfer block printed after it
    T = json.load(open(outdir + '/transfer.json'))['arms']; arm = list(T.values())[0]
    tb = txt[end:]; tb = tb[tb.find('validator       Δ n→p'):]; tb = tb[:tb.find('MEAN') + 80]
    for vname, V in arm['validators'].items():
        short = vname.split('-')[-1]
        g = re.search(r'\n  ' + short + r'\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)', tb).groups()
        for p, k in zip(g, ['delta_n_to_p', 'delta_p_to_n', 'ratio_n_to_p', 'ratio_p_to_n']): cmp(f'{short} {k}', p, V[k])
        ci = re.search(short + r'[^\n]*\n[^\n]*\n\s*95% CI  n→p \[([\d.-]+), ([\d.-]+)\]   p→n \[([\d.-]+), ([\d.-]+)\]', tb).groups()
        for p, v in zip(ci, V['ci_n_to_p'] + V['ci_p_to_n']): cmp(f'{short} CI', p, v)
    mean = re.search(r'MEAN\s+([\d.-]+)\s+([\d.-]+)', tb).groups()
    import statistics
    for p, k in zip(mean, ['ratio_n_to_p', 'ratio_p_to_n']): cmp('MEAN ' + k, p, statistics.mean(V[k] for V in arm['validators'].values()))
    return n, bad
if __name__ == '__main__':
    for log, out in zip(sys.argv[1::2], sys.argv[2::2]):
        n, bad = check(log, out)
        print(f"{'PASS' if not bad else 'FAIL'}  {out.split('/')[-1]:22s} {n} printed values compared  mismatches: {bad or 'none'}")
