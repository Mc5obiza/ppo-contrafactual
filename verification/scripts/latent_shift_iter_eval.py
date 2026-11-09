"""
Iterative Latent Shift baseline (J15b), scored under the J4 protocol. Issue D1.

Why: in the J15 smoke test (18537169, MEMO_2026-11-05_J15_smoke.md) single-gradient
Latent Shift crossed 0.5 on 15/16 images but its guidance shift plateaued at 0.3-0.65
along the one-gradient direction, so 0/16 met the CV margin tau = 0.8. A reviewer can call
that a weak baseline. This variant follows the gradient instead of a single straight line.

Everything is latent_shift_eval.py (J15) except the counterfactual search, which is
monkeypatched in. Same images, same AE (torchxrayvision ResNetAE 101-elastic), same
guidance/CV classifier (the rsna+chex ensemble), same validity criterion, same scoring,
same shard schema.

Search (hyperparameters fixed before any J15b result was seen; set from the J15 grid's range):
  z_0 = E(x);  z_{k+1} = z_k + s * ETA * ||z_0|| * g_k / ||g_k||,  g_k = d f(D(z_k)) / dz
  for k = 0 .. K_MAX-1, with ETA = 0.02 and K_MAX = 150, so the latent travels at most
  3.0 ||z_0||, inside J15's grid maximum of 4.0. Step 1 is J15's direction.
  Each image stops at the first step k >= 1 that meets eq. (cv) (s*Delta >= 0.8 and a 0.5
  crossing, scored by f), mirroring the policy's stop-at-validity rule. Images that never meet it
  keep the step with the largest signed shift and count as invalid. Nothing is tuned on LHR,
  transfer or any held-out quantity. Deterministic (no seeds).
  "lam" in the manifest/aux = travel = k * ETA (in units of ||z_0||); "step" = k.
"""
from __future__ import annotations

import json
import os
import sys

import torch

import latent_shift_eval as lse

ETA = 0.02
K_MAX = 150
TRAVEL = tuple(round(ETA * k, 4) for k in range(1, K_MAX + 1))


def latent_shift_iterative(ae, clf, x: torch.Tensor, s: int, device: str) -> dict:
    """Same signature and return keys as lse.latent_shift (+ 'step')."""
    size = tuple(x.shape[-2:])
    x = x.to(device)
    with torch.no_grad():
        f0 = clf(x).squeeze(-1)
        z0 = ae.encode(lse._to_ae(x))
    zn = lse._norm(z0)
    B = x.shape[0]
    z = z0.clone()
    xr = fr = None
    best_x = None
    best_step = torch.zeros(B, dtype=torch.long, device=device)
    best_gain = torch.full((B,), -float("inf"), device=device)
    found = torch.zeros(B, dtype=torch.bool, device=device)
    traj = []
    for k in range(K_MAX + 1):
        zk = z.detach().requires_grad_(True)
        xl = lse._from_ae(ae.decode(zk), size)
        fl = clf(xl).squeeze(-1)
        if k == 0:
            xr, fr = xl.detach().clone(), fl.detach().clone()
            best_x = xr.clone()
        else:
            with torch.no_grad():
                fd = fl.detach()
                gain = s * (fd - f0)
                valid = (gain >= lse.CV_MARGIN) & (s * (fd - 0.5) > 0)
                active = ~found
                upd = active & (valid | (gain > best_gain))
                best_x[upd] = xl.detach()[upd]
                best_step = torch.where(upd, torch.full_like(best_step, k), best_step)
                best_gain = torch.where(upd, gain, best_gain)
                found |= active & valid
            traj.append(fl.detach())
        if k == K_MAX or bool(found.all()):
            break
        # Images are independent and every module is in eval mode, so the gradient
        # of the batch sum is the per-image gradient.
        (g,) = torch.autograd.grad(fl.sum(), zk)
        with torch.no_grad():
            step = s * ETA * zn * g / lse._norm(g).clamp_min(1e-12)
            mask = (~found).view(-1, *([1] * (z.dim() - 1))).to(z.dtype)
            z = z + mask * step
    # pad the trajectory with NaN after an early stop so batches concatenate
    t = torch.stack(traj, 1)
    if t.shape[1] < K_MAX:
        t = torch.cat([t, torch.full((B, K_MAX - t.shape[1]), float("nan"), device=device)], 1)
    lam = torch.tensor(TRAVEL, dtype=torch.float32)[(best_step.cpu() - 1).clamp_min(0)]
    return {"xc": best_x.cpu(), "xr": xr.cpu(), "lam": lam,
            "step": best_step.cpu(), "found": found.cpu(), "f0": f0.cpu(), "f_rec": fr.cpu(),
            "traj": t.cpu()}


def main() -> None:
    lse.latent_shift = latent_shift_iterative   # looked up at call time by lse._run_group
    lse.LAMBDA_GRID = TRAVEL                     # manifest lam_hist / stamp: travel schedule
    lse.main()

    out = sys.argv[sys.argv.index("--out") + 1]
    shard_path = os.path.join(out, "shards", "cf_shard_000.pt")
    shard = torch.load(shard_path, map_location="cpu", weights_only=False)
    shard["config"].update(GENERATOR="latent_shift_iterative", LS_ITER_ETA=ETA, LS_ITER_K_MAX=K_MAX)
    torch.save(shard, shard_path)
    mpath = os.path.join(out, "latent_shift_eval_manifest.json")
    with open(mpath) as f:
        m = json.load(f)
    m["variant"] = {"name": "latent_shift_iterative", "eta": ETA, "k_max": K_MAX,
                    "max_travel": ETA * K_MAX,
                    "note": "lambda_grid / lam_hist hold travel k*ETA in units of ||z_0||"}
    with open(mpath, "w") as f:
        json.dump(m, f, indent=2)
    print(f"[latent-shift-iter] restamped {shard_path} and {mpath}")


if __name__ == "__main__":
    main()
