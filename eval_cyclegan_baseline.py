"""
eval_cyclegan_baseline.py
==========================
Evaluates the trained Vanilla CycleGAN baseline (train_cyclegan_baseline.py)
under EXACTLY the protocol every other arm in this project is scored under:

  * Xn/Xp groups from metrics.scan_groups (f(x) < 0.2 / f(x) > 0.8, using the
    SAME guidance classifier every other arm uses), restricted to the held-out
    test fold of the SAME persisted split (splits.py) — never the train fold
    the CycleGAN itself was fit on, and never a freshly-drawn one.
  * n_eval <= 200 per group.
  * CV, FID, IP (MS-SSIM/HF-corr), CTR, SCP computed by the exact same
    src/metrics.py functions every other arm calls — via
    run_full_evaluation's PRECOMPUTED path (xn_list/xp_list/xc_np/xc_pn
    supplied directly), so there is no second implementation of any metric to
    drift out of sync with the guided pipeline's numbers.

Generation itself is the one thing that's arm-specific: xc_np = G_AtoB(Xn),
xc_pn = G_BtoA(Xp) — a single unconditional generator forward pass per image,
no classifier guidance, no Grad-CAM mask, no PPO rollout. That is the entire
point of this baseline arm.

Usage
-----
    module load python/3.12.8 && module load cuda/cuda-12.1 && source venv/bin/activate
    python -u eval_cyclegan_baseline.py --ckpt-dir checkpoints_cyclegan_baseline
"""

from __future__ import annotations

import argparse
import os

import torch

import config
from pipeline import build_common
from src.cyclegan import ResnetGenerator, translate
from src.metrics import scan_groups, run_full_evaluation


def _load_generators(ckpt_dir: str, device: str) -> tuple[torch.nn.Module, torch.nn.Module]:
    g_ab_path = os.path.join(ckpt_dir, "G_AtoB.pt")
    g_ba_path = os.path.join(ckpt_dir, "G_BtoA.pt")
    if not (os.path.exists(g_ab_path) and os.path.exists(g_ba_path)):
        raise FileNotFoundError(
            f"No trained CycleGAN generators at '{g_ab_path}' / '{g_ba_path}'. "
            f"Run train_cyclegan_baseline.py (or sbatch run_cyclegan_baseline.slurm) first."
        )
    g_ab = ResnetGenerator().to(device)
    g_ba = ResnetGenerator().to(device)
    g_ab.load_state_dict(torch.load(g_ab_path, map_location=device))
    g_ba.load_state_dict(torch.load(g_ba_path, map_location=device))
    g_ab.eval(); g_ba.eval()
    print(f"[eval] loaded G_AtoB <- {g_ab_path}")
    print(f"[eval] loaded G_BtoA <- {g_ba_path}")
    return g_ab, g_ba


def _generate_batched(generator: torch.nn.Module, x_list: list[torch.Tensor],
                       device: str, batch_size: int = 16) -> list[torch.Tensor]:
    out = []
    for i in range(0, len(x_list), batch_size):
        batch = torch.cat(x_list[i:i + batch_size]).to(device)
        xc = translate(generator, batch).cpu()
        out.extend(xc[j:j + 1] for j in range(xc.shape[0]))
    return out


def main(ckpt_dir: str, n_eval: int = 200) -> dict:
    device = config.DEVICE

    print("=" * 70)
    print("Vanilla CycleGAN baseline — evaluation under this project's protocol")
    print("=" * 70)
    ctx = build_common()   # same dataset/classifier/seg_net/obj_det/split every arm uses

    g_ab, g_ba = _load_generators(ckpt_dir, device)

    print("\n[eval] scanning Xn/Xp on the HELD-OUT test fold ...")
    xn_list, xn_idx, xp_list, xp_idx = scan_groups(
        ctx.dataset, ctx.classifier, n_eval=n_eval, device=device,
        tag="cyclegan-baseline", allowed_indices=ctx.eval_indices(),
    )

    print(f"\n[eval] generating counterfactuals: "
          f"{len(xn_list)} via G_AtoB (n->p), {len(xp_list)} via G_BtoA (p->n) ...")
    xc_np = _generate_batched(g_ab, xn_list, device)
    xc_pn = _generate_batched(g_ba, xp_list, device)

    print("\n[eval] scoring with run_full_evaluation (precomputed path — same "
          "CV/FID/IP/CTR/SCP functions every other arm uses) ...")
    results = run_full_evaluation(
        dataset=ctx.dataset,
        classifier=ctx.classifier,
        generator=None,
        seg_net=ctx.seg_net,
        lung_channel_idx=ctx.lung_channel_idx,
        obj_det=ctx.obj_det,
        n_eval=n_eval,
        ckpt_dir=ckpt_dir,
        device=device,
        clinical_metrics=ctx.domain.clinical_metrics,
        xn_list=xn_list, xp_list=xp_list, xc_np=xc_np, xc_pn=xc_pn,
        xn_idx=xn_idx, xp_idx=xp_idx,
        allowed_indices=ctx.eval_indices(),
    )
    print(f"\n[eval] wrote {ckpt_dir}/eval_metrics.json")
    return results


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-dir", default="checkpoints_cyclegan_baseline")
    ap.add_argument("--n-eval", type=int, default=200)
    return ap.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    main(args.ckpt_dir, args.n_eval)
