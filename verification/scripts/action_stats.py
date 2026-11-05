"""
Per-image, per-step action statistics of the COMMITTED policies on the exact
evaluation images of eval 17695832, plus per-image runtime and peak GPU memory.

Inference only. Re-runs the committed evaluation's generation path for one
shard (same code: 49e6050, same generator view, same per-image seed
base_seed + global_index, mean action) while recording, at every env.step:
the raw 9-d action (= policy mean) and the mapped controls the env reports.

It also writes max|x_cf - x_cf_saved| per image. If that is not ~0 the run is
not reproducing the committed counterfactuals (hardware nondeterminism or a
different generator), and the action statistics describe a near-replica, not
the committed outputs. Say so if you quote them.

With ADAPTER_NP_VIEW / ADAPTER_PN_VIEW set (per_direction.py), each direction
generates with its own adapter, and the p->n diff against the committed shard
is then EXPECTED to be non-zero (the committed eval used the n->p adapter for
p->n); compare p->n against the J4 shards instead, using the counterfactuals
saved to <out.json minus .json>.cf.pt.

Run from ~/verification/code/49e6050 with PPO_CKPT_DIR=~/verification/views/committed.
    python action_stats.py <committed shard .pt> <out.json>
"""
import json
import sys
import time

import torch

import config
import per_direction
from pipeline import build_common, load_generator_weights, load_directional_policy
from src.inference import generate_counterfactual_with_policy

CONTROL_KEYS = ("strength", "cls_scale", "pres_scale", "tau", "strength_boost",
                "guidance_bias", "ramp_start_frac", "n_passes", "valid")


def record_steps(dp, sink: list):
    """Make dp.env.step append (raw action, mapped controls) to `sink`.

    Installs the wrapper ONCE per env and only redirects it to a new sink on
    later calls. (The first version re-wrapped env.step for every image, so
    the wrappers nested and every earlier image's list also collected the
    later images' steps; n_steps was right, the saved step lists were not.
    Fixed 2026-10-29; out/actions_repaired/ holds the truncated J6 records.)
    """
    env = dp.env
    if getattr(env, "_rec_sink", None) is not None:
        env._rec_sink = sink
        return
    env._rec_sink = sink
    orig_step = env.step

    def step(action):
        obs, reward, done, info = orig_step(action)
        env._rec_sink.append({
            "raw_action": [float(a) for a in action.detach().flatten().cpu()],
            **{k: (float(info[k]) if k != "valid" else bool(info[k])) for k in CONTROL_KEYS},
            "reward": float(reward),
        })
        return obs, reward, done, info

    env.step = step


def run_direction(dp, xs, xcs_saved, idxs, offset, label, ctx, cf_sink):
    if per_direction.enabled():
        per_direction.use(ctx.generator, label)
    rows, cfs = [], []
    for i in range(xs.shape[0]):
        steps = []
        record_steps(dp, steps)
        torch.manual_seed(0 + offset + i)          # eval_main: base_seed + global index
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        t0 = time.time()
        xc, _traj, info = generate_counterfactual_with_policy(
            xs[i:i + 1].to(config.DEVICE), dp, deterministic=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        cfs.append(xc.detach().cpu())
        rows.append({
            "dataset_idx": int(idxs[i]), "global_seed_index": offset + i,
            "steps": steps, "n_steps": len(steps),
            "seconds": time.time() - t0,
            "peak_gpu_mem_mib": (torch.cuda.max_memory_allocated() / 2**20
                                 if torch.cuda.is_available() else None),
            "max_abs_diff_vs_saved_cf": float((xc.detach().cpu().float()
                                               - xcs_saved[i:i + 1].float()).abs().max()),
        })
        print(f"[{label}] {i + 1}/{xs.shape[0]} idx={idxs[i]} steps={len(steps)} "
              f"{rows[-1]['seconds']:.1f}s diff={rows[-1]['max_abs_diff_vs_saved_cf']:.2e}",
              flush=True)
    cf_sink[label] = torch.cat(cfs) if cfs else torch.empty(0)
    return rows


def main(shard_path: str, out_path: str) -> None:
    shard = torch.load(shard_path, map_location="cpu", weights_only=False)
    ctx = build_common()
    load_generator_weights(ctx.generator, ckpt_dir=config.CKPT_DIR)
    dp_np = load_directional_policy(1.0, "n_to_p", ctx, ckpt_dir=config.CKPT_DIR)
    dp_pn = load_directional_policy(0.0, "p_to_n", ctx, ckpt_dir=config.CKPT_DIR)
    n0, p0 = shard["n_slice"][0], shard["p_slice"][0]
    cf = {}
    out = {
        "shard": shard["shard"], "stamp_of_saved_shard": shard.get("config"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "per_direction_adapters": per_direction.enabled(),
        "n_to_p": run_direction(dp_np, shard["xn"], shard["xc_np"], shard["xn_idx"], n0, "n_to_p", ctx, cf),
        "p_to_n": run_direction(dp_pn, shard["xp"], shard["xc_pn"], shard["xp_idx"], 1_000_000 + p0, "p_to_n", ctx, cf),
    }
    torch.save({"xc_np": cf["n_to_p"], "xc_pn": cf["p_to_n"],
                "xn_idx": list(shard["xn_idx"]), "xp_idx": list(shard["xp_idx"])},
               out_path[:-len(".json")] + ".cf.pt")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
