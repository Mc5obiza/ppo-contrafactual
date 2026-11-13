#!/bin/bash
# J12 final GENERATE (8 shards): the training-fold-selected constant schedule on the TEST fold, the same 400 images
# and per-image seeds as J4/J5/J5c (n_eval 200 per direction, base seed 0), each direction with its OWN adapter
# (adapters_committed.env: n->p checkpoints/n_to_p 658cdd79..., p->n checkpoints/p_to_n 6ae44230...).
# Step 1 (CPU, seconds): j12_select.py pools the select dirs in SELECT_DIRS (default: coarse grid only; with the local
# grid: "out/J12_grid/select out/J12_local/select"), scores every config with stage-1+2 rows on the same 48
# training-fold images (stop-bonus-corrected discounted return) and writes $OUT/J12_schedule.json. Every array task
# runs it; the output is deterministic and written atomically, and fixed_schedule_eval.py refuses a shard dir
# stamped with a different schedule.
#   G=$(sbatch --parsable [--export=ALL,SELECT_DIRS="..."] 18_J12_final_generate.sbatch)
#   sbatch --dependency=afterok:$G 19_J12_final_reduce.sbatch
# Submitted 2026-11-03 with SELECT_DIRS = coarse re-scored on A100 + local (A100); user options a + "A100 + re-score coarse".
#SBATCH --job-name=verif-J12-final-gen
#SBATCH --array=0-7
#SBATCH --partition=publicgpu
#SBATCH --account=insa
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpuh200|gpuh100|gpua100|gpul40s"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
set -euo pipefail
TAG=J12_grid_fixed_schedule
OUT="$HOME/verification/out/$TAG"; mkdir -p "$OUT/shards"
V="$HOME/verification"
SELECT_DIRS="${SELECT_DIRS:-$V/out/J12_grid/select}"
cd "$HOME/verification/code/49e6050"
module load python/3.12.8; module load cuda/cuda-12.1
source venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 YOLO_OFFLINE=1 PYTHONPATH="$PWD"
export PPO_CKPT_DIR="$HOME/verification/views/committed"
source "$HOME/verification/scripts/adapters_committed.env"   # n->p committed, p->n pn_lora adapter
echo "=== $TAG shard ${SLURM_ARRAY_TASK_ID}: code 49e6050 view=committed on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv
python -u preflight_gpu.py
python -u "$V/scripts/j12/j12_select.py" "$OUT/J12_schedule.json" $SELECT_DIRS
python -u "$HOME/verification/scripts/fixed_schedule_eval.py" "$OUT/J12_schedule.json" \
    --stage generate --shard "${SLURM_ARRAY_TASK_ID}" --num-shards 8 --n-eval 200 \
    --arm trained --shard-dir "$OUT/shards" --tag "$TAG" --base-seed 0
echo "=== finished at $(date)"
