#!/usr/bin/env bash
#SBATCH --job-name=ppo-counterfactual
#SBATCH --partition=publicgpu
#SBATCH --account=insa
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=melkiayoub724@gmail.com
#
# Runs the full PPO counterfactual pipeline (main.py) on a compute node.
# Compute nodes have NO internet access, so this script only ever loads the
# venv and pretrained weights that setup_env.sh already downloaded on the
# login node. Run setup_env.sh first — this job will fail fast otherwise.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}"
mkdir -p logs

module load python/3.12.8
module load cuda/13.0
source venv/bin/activate

# Force offline mode: fail loudly instead of silently hanging on a network
# call that can never succeed from a compute node.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export YOLO_OFFLINE=1

if [ ! -d venv ]; then
    echo "venv/ not found — run 'bash setup_env.sh' on the login node first." >&2
    exit 1
fi

echo "=== Job started on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

python main.py

echo "=== Job finished at $(date) ==="
