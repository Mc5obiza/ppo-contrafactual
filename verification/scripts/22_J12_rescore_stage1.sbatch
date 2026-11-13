#!/bin/bash
# J12 coarse-grid RE-SCORE on A100, stage 1 (j12/j12_rescore_coarse.py): the 6 coarse stage-2 configs per direction, same 48 training-fold images. Submitted 2026-11-03.
#SBATCH --job-name=verif-J12R-s1
#SBATCH --array=0-3
#SBATCH --constraint=gpua100
#SBATCH --time=12:00:00
#SBATCH --partition=publicgpu
#SBATCH --account=insa
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
set -euo pipefail
OUT="$HOME/verification/out/J12_coarse_a100"; mkdir -p "$OUT/select"
cd "$HOME/verification/code/49e6050"
module load python/3.12.8; module load cuda/cuda-12.1
source venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 YOLO_OFFLINE=1 PYTHONPATH="$PWD:$HOME/verification/scripts"
export PPO_CKPT_DIR="$HOME/verification/views/committed"
source "$HOME/verification/scripts/adapters_committed.env"   # n->p committed adapter, p->n pn_lora adapter
echo "host $(hostname) at $(date)"; nvidia-smi --query-gpu=name,memory.total --format=csv
# task 0 = n_to_p (all 6 configs, cheap), tasks 1-3 = p_to_n (2 configs each)
if [ $SLURM_ARRAY_TASK_ID = 0 ]; then D=n_to_p; T=0; NT=1; else D=p_to_n; T=$((SLURM_ARRAY_TASK_ID - 1)); NT=3; fi
python -u "$HOME/verification/scripts/j12/j12_rescore_coarse.py" --direction $D --stage 1 \
    --task $T --n-tasks $NT --out "$OUT/select"
