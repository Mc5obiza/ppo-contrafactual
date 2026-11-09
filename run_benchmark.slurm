#!/usr/bin/env bash
#SBATCH --job-name=ppo-benchmark
#SBATCH --partition=publicgpu
#SBATCH --account=insa
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpuh200|gpuh100|gpua100|gpul40s|gpua40|gpurtx6000"
#SBATCH --exclude=hpc-n983
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Measures per-episode and per-LoRA-epoch cost on the GPU this job lands on,
# then sizes N_EPISODES against the 24h partition cap. See
# benchmark_episode_cost.py for why this is not optional.
#
# RUN THIS BEFORE run_ppo.slurm. config.py's own comment above N_EPISODES says
# "ACTION REQUIRED before the real run: benchmark ~16 episodes first"; it was
# never run, and six jobs have since been killed at their walltime.
#
# Writes NOTHING except its log — no checkpoints, no config edits. Safe to run
# at any time, including while other jobs are queued.
#
# 2h walltime is generous for 16 episodes plus 6 LoRA batches. If it times out,
# that is itself the finding: per-episode cost is far higher than any current
# estimate and N_EPISODES=800 is not survivable.
#
# Usage:
#   sbatch run_benchmark.slurm                          # 16 episodes, n_to_p
#   sbatch run_benchmark.slurm --direction both         # both directions
#   sbatch run_benchmark.slurm --episodes 32 --skip-lora

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}"
mkdir -p logs

module load python/3.12.8
# Matches torch's cu121 build — see the note in run_ppo.slurm.
module load cuda/cuda-12.1

if [ ! -d venv ]; then
    echo "venv/ not found — run 'bash setup_env.sh' on the login node first." >&2
    exit 1
fi
source venv/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export YOLO_OFFLINE=1

bash record_run_manifest.sh

echo "=== Benchmark started on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

python -u preflight_gpu.py

status=0
python -u benchmark_episode_cost.py "$@" || status=$?

echo "=== Benchmark finished at $(date) with exit status ${status} ==="
exit ${status}
