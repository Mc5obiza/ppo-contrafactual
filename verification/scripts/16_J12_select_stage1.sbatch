#!/bin/bash
# J12 stage 1 — 55 constant schedules (54 grid + policy-mean reference) x 12 TRAINING-fold images per direction.
# Tasks 0-5 = n_to_p chunks, 6-11 = p_to_n chunks. All on one GPU type (L40S) so the ranking has no cross-hardware term.
# Submitted 2026-10-31 (pilot = task 6, 18511258). Pilot: sbatch --array=6 (one p->n chunk, 9-10 configs) to calibrate cost.
#SBATCH --job-name=verif-J12-s1
#SBATCH --array=0-11
#SBATCH --constraint=gpul40s
#SBATCH --time=12:00:00
#SBATCH --partition=publicgpu
#SBATCH --account=insa
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
set -euo pipefail
OUT="$HOME/verification/out/J12_grid"; mkdir -p "$OUT/select"
cd "$HOME/verification/code/49e6050"
module load python/3.12.8; module load cuda/cuda-12.1
source venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 YOLO_OFFLINE=1 PYTHONPATH="$PWD:$HOME/verification/scripts"
export PPO_CKPT_DIR="$HOME/verification/views/committed"
source "$HOME/verification/scripts/adapters_committed.env"   # n->p committed adapter, p->n pn_lora adapter
echo "host $(hostname) at $(date)"; nvidia-smi --query-gpu=name,memory.total --format=csv
D=$([ $SLURM_ARRAY_TASK_ID -lt 6 ] && echo n_to_p || echo p_to_n)
python -u "$HOME/verification/scripts/j12/j12_grid_select.py" --direction $D --stage 1 \
    --task $((SLURM_ARRAY_TASK_ID % 6)) --n-tasks 6 --out "$OUT/select"
