"""Decision 7: relative n_to_p vs p_to_n adapter difference per arm (full precision) and committed / arm ratio. CPU.
rel = ||W_np - W_pn||_F / ||W_np||_F over all LoRA tensors of unet_finetuned.pt (and separately over cond_proj.pt)."""
import torch, os, json
R = os.path.expanduser('~/Jobs/Ayoub/ppo-contrafactual')
arms = [('2-ens committed (default)', 'checkpoints'), ('single-classifier', 'checkpoints/archive_pre_ensemble_default_20260812'),
        ('3-model ensemble', 'checkpoints_ens3'), ('sparsity reward', 'checkpoints_sparsity'), ('Grad-CAM constraint', 'checkpoints_gradcam_fix')]
sd = lambda p: (lambda o: o.get('state_dict', o) if isinstance(o, dict) else o)(torch.load(p, map_location='cpu', weights_only=False))
def rel(a, b, keys):
    return sum(((a[k].double() - b[k].double()) ** 2).sum().item() for k in keys) ** .5 / sum((a[k].double() ** 2).sum().item() for k in keys) ** .5
res = {}
for name, d in arms:
    a, b = sd(f'{R}/{d}/n_to_p/unet_finetuned.pt'), sd(f'{R}/{d}/p_to_n/unet_finetuned.pt')
    ca, cb = sd(f'{R}/{d}/n_to_p/cond_proj.pt'), sd(f'{R}/{d}/p_to_n/cond_proj.pt')
    res[name] = dict(lora=rel(a, b, [k for k in a if 'lora' in k]), cond_proj=rel(ca, cb, list(ca)))
c = res['2-ens committed (default)']
print('| arm | rel. LoRA diff | committed / arm (LoRA) | rel. cond_proj diff | committed / arm (cond_proj) |'); print('|---|---|---|---|---|')
for n, r in res.items():
    q = lambda k: 'undefined (arm diff = 0)' if r[k] == 0 else f'{c[k] / r[k]:.1f}'
    print(f"| {n} | {r['lora']:.4e} | {q('lora')} | {r['cond_proj']:.4e} | {q('cond_proj')} |")
json.dump(res, open(os.path.expanduser('~/verification/part4/adapter_ratio.json'), 'w'), indent=1)
