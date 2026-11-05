#!/bin/bash
# J18 — Pneumonia AUROC of guidance models and held-out validators on the RSNA
# test fold (issue D8/E5). Inference only. Writes out/J18_validator_auroc/auroc.json.
#   sbatch 29_validator_auroc.sbatch
#SBATCH --job-name=verif-J18-auroc
#SBATCH --partition=publicgpu
#SBATCH --account=insa
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpuh200|gpuh100|gpua100|gpul40s"
#SBATCH --exclude=hpc-n983
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/home2020/home/insa/arbaoui/verification/logs/%x-%j.out
#SBATCH --error=/home2020/home/insa/arbaoui/verification/logs/%x-%j.err
set -euo pipefail
OUT="$HOME/verification/out/J18_validator_auroc"; mkdir -p "$OUT"
module load python/3.12.8; module load cuda/cuda-12.1
cd "$HOME/verification/code/49e6050"
source venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 YOLO_OFFLINE=1
export PPO_CKPT_DIR="$HOME/verification/views/committed"   # only for the split file
export PYTHONPATH="$PWD:$HOME/verification/scripts"
echo "=== J18 on $(hostname) at $(date)"; nvidia-smi --query-gpu=name --format=csv
python -u "$HOME/verification/scripts/validator_auroc.py" "$OUT/auroc.json"
echo "=== finished at $(date)"
