import torch,os,json
H=os.path.expanduser('~/verification/out')
C='checkpoints/eval_shards_ens_default'
runs={'J4':H+'/J4_perdir/shards','J5':H+'/J5_none_perdir/shards','J5b':H+'/J5b_fixed_schedule/shards'}
L=lambda p:torch.load(p,map_location='cpu',weights_only=False)
allc={'n':[],'p':[]}; res={k:{'n':[],'p':[]} for k in list(runs)+['J6']}; bad={k:[] for k in res}
keys=['shard','num_shards','n_eval','base_seed','n_slice','p_slice','xn_idx','xp_idx']
for s in range(8):
    c=L(f'{C}/cf_shard_{s:03d}.pt'); allc['n']+=c['xn_idx']; allc['p']+=c['xp_idx']
    for k,d in runs.items():
        r=L(f'{d}/cf_shard_{s:03d}.pt'); res[k]['n']+=r['xn_idx']; res[k]['p']+=r['xp_idx']
        for key in keys:
            if r.get(key)!=c.get(key): bad[k].append(f's{s} {key}')
        for key in ['xn','xp']:
            if not torch.equal(r[key],c[key]): bad[k].append(f's{s} {key} not byte-identical')
    # J6: own outputs
    r=L(f'{H}/actions/actions_shard_{s:03d}.cf.pt'); j=json.load(open(f'{H}/actions/actions_shard_{s:03d}.json'))
    for key in ['xn_idx','xp_idx']:
        if r[key]!=c[key]: bad['J6'].append(f's{s} cf.pt {key}')
    off=c['n_slice'][0] if False else None
    for dirn,ck,sl in [('n_to_p','xn_idx','n_slice'),('p_to_n','xp_idx','p_slice')]:
        di=[e['dataset_idx'] for e in j[dirn]]; gs=[e['global_seed_index'] for e in j[dirn]]
        if di!=c[ck]: bad['J6'].append(f's{s} {dirn} dataset_idx')
        if gs!=list(range(c[sl][0],c[sl][1])): bad['J6'].append(f's{s} {dirn} seed idx {gs[:3]} vs slice {c[sl]}')
    res['J6']['n']+=r['xn_idx']; res['J6']['p']+=r['xp_idx']
print('committed: n',len(allc['n']),'(unique',len(set(allc['n'])),') p',len(allc['p']),'(unique',len(set(allc['p'])),') base_seed',c['base_seed'],'n_eval',c['n_eval'])
for k in res:
    print(f"{k}: n {len(res[k]['n'])} p {len(res[k]['p'])} | n list == committed: {res[k]['n']==allc['n']} | p list == committed: {res[k]['p']==allc['p']} | field mismatches: {bad[k] or 'none'}")
