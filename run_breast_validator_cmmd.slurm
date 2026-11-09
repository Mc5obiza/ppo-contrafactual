#!/usr/bin/env bash
#SBATCH --job-name=breast-validator-cmmd
#SBATCH --partition=publicgpu
#SBATCH --account=insa
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpuh200|gpuh100|gpua100|gpul40s|gpua40|gpurtx6000"
#SBATCH --exclude=hpc-n983
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Trains the EXTERNAL-corpus held-out validator: same script, CMMD instead of
# the disjoint CBIS-DDSM pool.
#
# 5202 images against the disjoint pool's 432, so this one gets a longer wall
# clock than run_breast_validator.slurm — but it is also the only mammography
# validator in the project whose independence does not rest on a subset
# argument. Different country, different scanner generation, different
# curators, zero possible patient overlap.
#
# It still has to prove it can read a CBIS-DDSM image before its transfer ratio
# means anything: breast_validator_gate.py, run after this.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}"
mkdir -p logs

module load python/3.12.8
module load cuda/13.0
source venv/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export YOLO_OFFLINE=1
export PPO_NUM_WORKERS="${PPO_NUM_WORKERS:-8}"

echo "=== Job ${SLURM_JOB_ID} on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
python -u train_breast_validator.py --external cmmd_breast "$@"
echo "=== finished at $(date) ==="
