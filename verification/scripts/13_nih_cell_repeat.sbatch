#!/bin/bash
# J13 — conditional waiver of check 4: recompute the single-classifier transfer cell
# (nih, n->p) with the exact J2 invocation (code/HEAD arm_transfer.py --bootstrap --json,
# same saved shards checkpoints/eval_shards). Submit 2x on gpul40s (J2 ran on L40S,
# hpc-n967) and 1x on another GPU type. RUN=<tag> names the output JSON.
#SBATCH --job-name=verif-nih-cell
#SBATCH --partition=publicgpu
#SBATCH --account=insa
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
R="${PPO_CF_REPO:?set PPO_CF_REPO to the ppo-contrafactual checkout}"
OUT="$HOME/verification/out/J13_nih_cell"; mkdir -p "$OUT"
cd "$HOME/verification/code/HEAD"
module load python/3.12.8; module load cuda/cuda-12.1
source venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 YOLO_OFFLINE=1 PPO_DOMAIN=rsna_pneumonia
echo "host $(hostname)"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -u arm_transfer.py "$R/checkpoints/eval_shards" --bootstrap --json "$OUT/${RUN}.json"
python - "$OUT/${RUN}.json" <<'PY'
import json, sys
a = list(json.load(open(sys.argv[1]))['arms'].values())[0]; v = a['validators']['densenet121-res224-nih']
print(f"[J13] nih ratio_n_to_p={v['ratio_n_to_p']!r} delta_n_to_p={v['delta_n_to_p']!r} delta_guidance_n_to_p={a['delta_guidance_n_to_p']!r} ci={v['ci_n_to_p']}")
PY
