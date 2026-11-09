#!/usr/bin/env bash
#SBATCH --job-name=breast-validator
#SBATCH --partition=publicgpu
#SBATCH --account=insa
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpuh200|gpuh100|gpua100|gpul40s|gpua40|gpurtx6000"
#SBATCH --exclude=hpc-n983
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Trains the genuinely independent held-out validators for the mammography
# domain (train_breast_validator.py).
#
# WHY THIS IS NOT run_breast_classifier.slurm. That script trains all three
# -breast-cbis backbones on ONE patient split, which is exactly why the
# "reserved" resnet18 validator is not independent: it shares every training
# image with the classifier it is supposed to check, and 84% of the eval images
# with the counterfactuals it is supposed to adjudicate. This job trains on the
# complement — the 234 patients that neither the guidance models nor the eval
# groups touch.
#
# NOT AN ARRAY. train_breast_classifier.slurm is an array because each backbone
# took ~25 min on 2290 images and three of them overran a 4h wall clock. Here
# the whole pool is 432 images, so all folds of all architectures finish inside
# one job, and keeping them in one process is what lets the out-of-fold AUC be
# pooled across folds in a single pass.
#
# PREREQUISITE: breast_shard_audit.py must have written
# <CKPT_DIR>/eval_group_index.json. That file is what defines the disjoint pool;
# this job refuses to guess it.
#
#   sbatch run_breast_validator.slurm
#   sbatch run_breast_validator.slurm --epochs 50 --folds 5
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}"
mkdir -p logs

module load python/3.12.8
module load cuda/13.0

if [ ! -d venv ]; then
    echo "venv/ not found — run 'bash setup_env.sh' on the login node first." >&2
    exit 1
fi
source venv/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export YOLO_OFFLINE=1

# The whole job is mammography-specific; the domain is not optional here.
export PPO_DOMAIN=cbis_ddsm_breast
export PPO_NUM_WORKERS="${PPO_NUM_WORKERS:-8}"

echo "=== Job ${SLURM_JOB_ID} on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
python -u train_breast_validator.py "$@"
echo "=== finished at $(date) ==="
