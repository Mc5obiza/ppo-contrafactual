"""Per-arm n_to_p vs p_to_n generator adapter difference (CPU). LoRA tensors of unet_finetuned.pt and all of cond_proj.pt."""
import torch, os, hashlib
R = os.environ['PPO_CF_REPO']
arms = [('2-ens committed (default)', 'checkpoints'), ('single-classifier', 'checkpoints/archive_pre_ensemble_default_20260812'),
        ('3-model ensemble', 'checkpoints_ens3'), ('sparsity reward', 'checkpoints_sparsity'), ('Grad-CAM constraint', 'checkpoints_gradcam_fix')]
sd = lambda p: (lambda o: o.get('state_dict', o) if isinstance(o, dict) else o)(torch.load(p, map_location='cpu', weights_only=False))
print('| arm | dir | LoRA tensors differing | max abs Δ (LoRA) | rel. Frobenius ‖Δ‖/‖W_np‖ (LoRA) | non-LoRA max Δ | cond_proj max abs Δ | cond_proj rel. ‖Δ‖ | sha256 np / pn (first 8) |')
print('|---|---|---|---|---|---|---|---|---|')
for name, d in arms:
    a, b = sd(f'{R}/{d}/n_to_p/unet_finetuned.pt'), sd(f'{R}/{d}/p_to_n/unet_finetuned.pt')
    lk = [k for k in a if 'lora' in k]; nl = [k for k in a if 'lora' not in k]
    nd = sum(not torch.equal(a[k], b[k]) for k in lk); mx = max((a[k].float() - b[k].float()).abs().max().item() for k in lk)
    num = sum(((a[k].float() - b[k].float()) ** 2).sum().item() for k in lk) ** .5; den = sum((a[k].float() ** 2).sum().item() for k in lk) ** .5
    mxn = max([(a[k].float() - b[k].float()).abs().max().item() for k in nl if torch.is_tensor(a[k])] or [0.0])
    ca, cb = sd(f'{R}/{d}/n_to_p/cond_proj.pt'), sd(f'{R}/{d}/p_to_n/cond_proj.pt')
    cmx = max((ca[k].float() - cb[k].float()).abs().max().item() for k in ca)
    cr = sum(((ca[k].float() - cb[k].float()) ** 2).sum().item() for k in ca) ** .5 / sum((ca[k].float() ** 2).sum().item() for k in ca) ** .5
    h = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]
    print(f"| {name} | `{d}` | {nd}/{len(lk)} | {mx:.2e} | {num/den:.2e} | {mxn:.1e} | {cmx:.2e} | {cr:.2e} | {h(f'{R}/{d}/n_to_p/unet_finetuned.pt')} / {h(f'{R}/{d}/p_to_n/unet_finetuned.pt')} |", flush=True)
