#!/usr/bin/env bash
#SBATCH --job-name=breast-lesion-seg
#SBATCH --partition=publicgpu
#SBATCH --account=insa
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Trains the lesion U-Net the LLR and MSI clinical metrics are computed from
# (train_lesion_segmenter.py).
#
# Needed for EVALUATION, not for training: run_ppo.slurm never touches this net
# — it is deliberately kept out of the guidance loop so the metrics derived
# from it stay independent of the edits they measure. So this can run
# concurrently with PPO training, and must finish before run_eval_reduce.slurm.
#
# 6h covers 40 epochs of a ResNet-18 U-Net over the ~3.5k images that carry a
# ground-truth ROI mask, on the slowest GPU in this partition.
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

echo "=== Job started on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
python -u train_lesion_segmenter.py "$@"
echo "=== Job finished at $(date) ==="
