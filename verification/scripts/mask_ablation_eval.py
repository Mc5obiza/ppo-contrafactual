"""
Run eval_main.py with the background-lock mask geometry overridden.

The shard stamp (metrics.current_cfg_stamp) does NOT record BG_MASK_DILATION or
BG_MASK_FEATHER, so this wrapper writes them to <out_dir>/mask_setting.json and
refuses to reuse a shard dir that holds a different setting.

    python mask_ablation_eval.py <dilation> <feather> <eval_main args...>
"""
import json
import os
import sys

import config

dilation, feather = int(sys.argv[1]), float(sys.argv[2])
config.BG_MASK_DILATION = dilation
config.BG_MASK_FEATHER = feather

args = sys.argv[3:]
shard_dir = args[args.index("--shard-dir") + 1]
os.makedirs(shard_dir, exist_ok=True)
stamp = os.path.join(shard_dir, "mask_setting.json")
want = {"BG_MASK_DILATION": dilation, "BG_MASK_FEATHER": feather}
if os.path.exists(stamp):
    have = json.load(open(stamp))
    if have != want:
        raise SystemExit(f"{shard_dir} holds mask setting {have}, not {want}")
else:
    json.dump(want, open(stamp, "w"))

print(f"[mask-ablation] BG_MASK_DILATION={dilation} BG_MASK_FEATHER={feather}", flush=True)
import eval_main  # noqa: E402
import per_direction  # noqa: E402

if per_direction.enabled():
    per_direction.install_eval_main(eval_main)
eval_main.main(args)
