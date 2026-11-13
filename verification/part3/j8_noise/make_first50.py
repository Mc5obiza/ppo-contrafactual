"""Build 50-image single-shard dirs (= shards 0+1, the J8 images) for committed and J4, J8 layout."""
import torch, os, copy
H = os.path.expanduser('~/verification'); R = os.environ['PPO_CF_REPO']
L = lambda p: torch.load(p, map_location='cpu', weights_only=False)
ref = L(H + '/out/J8_mask_d2_f1.0/shards/cf_shard_000.pt')
for tag, d in [('committed_first50', R + '/checkpoints/eval_shards_ens_default'), ('J4_first50', H + '/out/J4_perdir/shards')]:
    a, b = L(d + '/cf_shard_000.pt'), L(d + '/cf_shard_001.pt')
    o = copy.copy(a)
    for k in ['xn', 'xp', 'xc_np', 'xc_pn']: o[k] = torch.cat([a[k], b[k]])
    for k in ['xn_idx', 'xp_idx']: o[k] = a[k] + b[k]
    o.update(shard=0, num_shards=1, n_eval=50, n_slice=[0, 50], p_slice=[0, 50])
    assert o['xn_idx'] == ref['xn_idx'] and o['xp_idx'] == ref['xp_idx'] and torch.equal(o['xn'], ref['xn']) and torch.equal(o['xp'], ref['xp'])
    os.makedirs(f'{H}/part3/j8_noise/{tag}', exist_ok=True); torch.save(o, f'{H}/part3/j8_noise/{tag}/cf_shard_000.pt'); print('wrote', tag)
