#!/usr/bin/env bash
#SBATCH --job-name=breast-classifier
#SBATCH --partition=publicgpu
#SBATCH --account=insa
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Trains f(.) for the cbis_ddsm_breast domain (train_breast_classifier.py).
#
# THIS IS A PREREQUISITE, NOT AN OPTIONAL STEP. Unlike the chest domains, whose
# classifiers come pretrained from torchxrayvision, mammography has no
# pretrained head anywhere in that zoo — so run_ppo.slurm cannot start until
# this has produced checkpoint/breast/*.pt.
#
# Order:
#   1. bash setup_env.sh                  (login node — venv + weights + data)
#   2. sbatch run_breast_classifier.slurm (this)
#   3. sbatch run_lesion_segmenter.slurm  (LLR/MSI need it)
#   4. sbatch run_ppo.slurm
#
# 4h is generous for 3 backbones x 20 epochs over ~3.5k 256px images; it is
# sized for the slowest GPU this partition hands out (Tesla P100), not the
# fastest.
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

# Force offline mode: fail loudly instead of hanging on a network call that can
# never succeed from a compute node. The ImageNet backbone weights this script
# initialises from are prefetched into the torch hub cache by setup_env.sh.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export YOLO_OFFLINE=1

echo "=== Job started on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
python -u train_breast_classifier.py "$@"
echo "=== Job finished at $(date) ==="
