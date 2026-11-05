"""
J5b — fixed-schedule arm: run eval_main.py (code 49e6050) with each committed
policy replaced by a CONSTANT action, namely the committed policy's mean raw
action over every env step J6 recorded on the 400 evaluation images.

Same images and per-image seeds as J5 and the committed eval, and the same
per-direction adapters as J4/J5 when ADAPTER_PN_VIEW is set (per_direction.py); the only difference to the trained arm is that the action no longer
depends on the image or the step. The env maps the constant raw action to
controls exactly as it maps the policy's, and episodes still end on validity
or at t = T.

The mean is taken over all recorded steps pooled (an image with two steps
contributes two actions), separately per direction. It is the mean of the RAW
9-d action; the mapped controls are map(mean raw), not mean(mapped).

The schedule is written to <shard_dir>/fixed_schedule.json; a shard dir that
holds a different schedule is refused (same guard as mask_ablation_eval.py).

    python fixed_schedule_eval.py <J6 actions dir> <eval_main args...>
"""
import glob
import json
import os
import sys

import torch

import pipeline

N_SHARDS = 8


def mean_actions(actions_dir: str) -> dict:
    paths = sorted(glob.glob(os.path.join(actions_dir, "actions_shard_*.json")))
    if len(paths) != N_SHARDS:
        raise SystemExit(f"expected {N_SHARDS} J6 outputs in {actions_dir}, found {len(paths)}")
    out = {}
    for label in ("n_to_p", "p_to_n"):
        acts, n_img = [], 0
        for p in paths:
            for row in json.load(open(p))[label]:
                n_img += 1
                # [:n_steps]: J6 run 18497965 saved cumulative step lists
                # (nested-wrapper bug in action_stats.py); the first n_steps
                # entries are the image's own steps, exactly.
                acts += [s["raw_action"] for s in row["steps"][:row["n_steps"]]]
        a = torch.tensor(acts, dtype=torch.float64)
        out[label] = {"mean_raw_action": a.mean(0).tolist(),
                      "std_raw_action": a.std(0).tolist(),
                      "n_steps": len(acts), "n_images": n_img}
    out["source"] = [os.path.abspath(p) for p in paths]
    return out


class ConstantPolicy(torch.nn.Module):
    """Stands in for PPOActorCritic inside env.generate_via_policy."""

    def __init__(self, action, dtype):
        super().__init__()
        self.register_buffer("action", torch.tensor(action, dtype=dtype))

    def get_action_and_value(self, obs, deterministic=False):
        b = obs.shape[0]
        a = self.action.to(obs.device).unsqueeze(0).expand(b, -1).clone()
        z = torch.zeros(b, device=obs.device)
        return a, z, z, z, z, z


actions_dir, args = sys.argv[1], sys.argv[2:]
# J12: a schedule JSON (j12_select.py output, same layout as fixed_schedule.json) instead of a J6 dir
sched = json.load(open(actions_dir)) if actions_dir.endswith(".json") else mean_actions(actions_dir)

shard_dir = args[args.index("--shard-dir") + 1]
os.makedirs(shard_dir, exist_ok=True)
stamp = os.path.join(shard_dir, "fixed_schedule.json")
want = {k: sched[k]["mean_raw_action"] for k in ("n_to_p", "p_to_n")}
if os.path.exists(stamp):
    have = {k: json.load(open(stamp))[k]["mean_raw_action"] for k in ("n_to_p", "p_to_n")}
    if have != want:
        raise SystemExit(f"{shard_dir} holds a different fixed schedule")
else:
    tmp = f"{stamp}.{os.getpid()}.tmp"
    json.dump(sched, open(tmp, "w"), indent=1)
    os.replace(tmp, stamp)

_orig_load = pipeline.load_directional_policy


def _load_fixed(target_c, label, ctx, ckpt_dir=None, **kw):
    # Load the real policy first so a wrong view still fails loudly.
    dp = _orig_load(target_c, label, ctx, **({"ckpt_dir": ckpt_dir} if ckpt_dir else {}), **kw)
    p0 = next(dp.policy.parameters())
    dp.policy = ConstantPolicy(sched[label]["mean_raw_action"], p0.dtype).to(p0.device)
    print(f"[fixed-schedule] {label}: constant raw action "
          f"{[round(x, 4) for x in sched[label]['mean_raw_action']]} "
          f"(mean of {sched[label]['n_steps']} steps on {sched[label]['n_images']} images)",
          flush=True)
    return dp


pipeline.load_directional_policy = _load_fixed   # before eval_main binds it
import eval_main  # noqa: E402
import per_direction  # noqa: E402

if per_direction.enabled():
    per_direction.install_eval_main(eval_main)
eval_main.main(args)
