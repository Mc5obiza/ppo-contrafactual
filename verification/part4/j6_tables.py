"""J6 (repaired step records, out/actions_repaired) — one table per direction for decision 2. CPU only.
Columns per action dimension (raw a_j and its mapped control, factory/environment.py l.504-524):
  mean, sd             over all env steps (pooled, population sd)
  between-share (multi) b/(b+w) over images with >1 step: b = var of per-image means, w = mean within-image var
                        (the definition behind the 0.67-0.78 figure; n/a if <10 such images)
  eta2 (all images)     between-image sum of squares / total sum of squares over all steps, all images
  sat lo / sat hi       share of steps with sigmoid(a_j) <= 0.01 / >= 0.99, i.e. control within 1% of its lower/upper bound
Episode-length distribution: exact counts of n_steps over the 200 images (cap T=25)."""
import json, glob, os, collections, numpy as np
H = os.path.expanduser('~/verification/out/actions_repaired')
dims = [('a0', 'strength', '0.2 + 0.5 s'), ('a1', 'cls_scale', '0.3 + 0.9 s'), ('a2', 'pres_scale', '0.3 + 1.2 s'),
        ('a3', 'tau (stop signal)', 's'), ('a4', 'extra pass 1', '[s > .5]'), ('a5', 'extra pass 2', '[s > .5]'),
        ('a6', 'strength_boost', '0.85 + 0.30 s'), ('a7', 'guidance_bias', '0.70 + 0.60 s'), ('a8', 'ramp_start_frac', '0.1 + 0.4 s')]
lo = [0.2, 0.3, 0.3, 0, 0, 0, 0.85, 0.70, 0.1]; sc = [0.5, 0.9, 1.2, 1, 1, 1, 0.30, 0.60, 0.4]
sig = lambda x: 1 / (1 + np.exp(-x))
out = {}
for d in ['n_to_p', 'p_to_n']:
    rows = [e for f in sorted(glob.glob(H + '/actions_shard_*.json')) for e in json.load(open(f))[d]]
    assert len(rows) == 200 and all(len(e['steps']) == e['n_steps'] for e in rows)
    ns = np.array([e['n_steps'] for e in rows]); multi = [e for e in rows if e['n_steps'] > 1]
    print(f'\n## {d}: {len(rows)} images, {ns.sum()} steps, {len(multi)} images with >1 step\n')
    print('| dim | control | mapping (s = sigmoid(a)) | raw mean | raw sd | control mean | control sd | between-share (multi-step images) | eta² (all images) | sat lo | sat hi |')
    print('|---|---|---|---|---|---|---|---|---|---|---|')
    out[d] = {'dims': {}}
    for j, (a, name, m) in enumerate(dims):
        per = [np.array([s['raw_action'][j] for s in e['steps']]) for e in rows]
        A = np.concatenate(per); S = sig(A)
        C = (S > .5).astype(float) if j in (4, 5) else lo[j] + sc[j] * S
        if j not in (4, 5): assert np.allclose(C, [s[name.split(' ')[0]] for e in rows for s in e['steps']], atol=1e-5), (d, name)
        def shares(v):
            pv = [v[k] for k in range(len(rows))]
            mi = [x for x, e in zip(pv, rows) if e['n_steps'] > 1]
            if len(mi) >= 10:
                w = np.mean([np.var(x) for x in mi]); b = np.var([x.mean() for x in mi]); bm = b / (b + w)
            else: bm = None
            allv = np.concatenate(pv); ssb = sum(len(x) * (x.mean() - allv.mean()) ** 2 for x in pv); sst = ((allv - allv.mean()) ** 2).sum()
            return bm, (ssb / sst if sst > 0 else None)
        split = lambda v: [v[sum(ns[:k]):sum(ns[:k + 1])] for k in range(len(rows))]
        bm_raw, eta_raw = shares(split(A))
        rec = dict(control=name, raw_mean=A.mean(), raw_sd=A.std(), ctrl_mean=C.mean(), ctrl_sd=C.std(),
                   between_share_multi=bm_raw, eta2_all=eta_raw, sat_lo=np.mean(S <= .01), sat_hi=np.mean(S >= .99))
        out[d]['dims'][a] = rec
        f = lambda x, p=3: 'n/a' if x is None else f'{x:.{p}f}'
        print(f'| {a} | {name} | {m} | {A.mean():+.3f} | {A.std():.3f} | {C.mean():.4f} | {C.std():.4f} | {f(bm_raw, 2)} | {f(eta_raw, 2)} | {np.mean(S <= .01)*100:.1f}% | {np.mean(S >= .99)*100:.1f}% |')
    npas = np.array([s['n_passes'] for e in rows for s in e['steps']])
    gb = np.array([s['guidance_bias'] for e in rows for s in e['steps']])
    cnt = dict(sorted(collections.Counter(ns.tolist()).items()))
    out[d].update(n_passes=dict(sorted(collections.Counter(npas.tolist()).items())), episode_length_counts=cnt,
                  ep_mean=ns.mean(), ep_median=float(np.median(ns)), gb_ge_1295=np.mean(gb >= 1.295))
    print(f'\nn_passes per step: {out[d]["n_passes"]} | guidance_bias >= 1.295 on {np.mean(gb>=1.295)*100:.1f}% of steps')
    print(f'\nEpisode length (n_steps per image; mean {ns.mean():.2f}, median {np.median(ns):.0f}, cap 25):\n')
    print('| steps | ' + ' | '.join(map(str, cnt)) + ' | total |'); print('|---|' + '---|' * (len(cnt) + 1))
    print('| images | ' + ' | '.join(str(v) for v in cnt.values()) + f' | {sum(cnt.values())} |')
json.dump(out, open(os.path.expanduser('~/verification/part4/j6_tables.json'), 'w'), indent=1, default=float)
