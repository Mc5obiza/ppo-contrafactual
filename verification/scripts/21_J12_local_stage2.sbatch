#!/bin/bash
# J12 LOCAL grid stage 2: top 6 (n_to_p) / top 4 (p_to_n) local stage-1 configs x the same 36 further TRAINING-fold images. Tasks 0-2 n_to_p, 3-5 p_to_n.
# Submit with --dependency=afterok:<local stage 1>. Submitted 2026-11-03 (user option a).
#SBATCH --job-name=verif-J12L-s2
#SBATCH --array=0-5
#SBATCH --constraint=gpul40s
#SBATCH --time=16:00:00
#SBATCH --partition=publicgpu
#SBATCH --account=insa
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home2020/home/insa/arbaoui/verification/logs/%x-%A_%a.out
#SBATCH --error=/home2020/home/insa/arbaoui/verification/logs/%x-%A_%a.err
set -euo pipefail
OUT="$HOME/verification/out/J12_local"; mkdir -p "$OUT/select"
cd "$HOME/verification/code/49e6050"
module load python/3.12.8; module load cuda/cuda-12.1
source venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 YOLO_OFFLINE=1 PYTHONPATH="$PWD:$HOME/verification/scripts"
export PPO_CKPT_DIR="$HOME/verification/views/committed"
source "$HOME/verification/scripts/adapters_committed.env"   # n->p committed adapter, p->n pn_lora adapter
echo "host $(hostname) at $(date)"; nvidia-smi --query-gpu=name,memory.total --format=csv
D=$([ $SLURM_ARRAY_TASK_ID -lt 3 ] && echo n_to_p || echo p_to_n)
python -u "$HOME/verification/scripts/j12/j12_local_grid.py" --direction $D --stage 2 \
    --task $((SLURM_ARRAY_TASK_ID % 3)) --n-tasks 3 --out "$OUT/select"
