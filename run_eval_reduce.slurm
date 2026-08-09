#!/usr/bin/env bash
#SBATCH --job-name=ppo-eval-reduce
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
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=melkiayoub724@gmail.com
#
# STAGE 2 of 2 — compute metrics from the shards run_eval_cf.slurm produced.
# Writes checkpoints/eval_metrics.json and the two counterfactual diff plots.
#
#   sbatch run_eval_cf.slurm                                  # note the job id
#   sbatch --dependency=afterok:<jobid> run_eval_reduce.slurm
#
# The dependency matters: this job fails fast (non-zero, naming the missing
# shard ids) if the array has not fully completed, because metrics over a
# partial shard set are computed on a short Xn/Xp and FID's upward bias grows
# sharply as N falls — a partial reduce would print a plausible wrong number
# rather than an error.
#
# WALLTIME. 4h, which is mostly slack. The expensive part — 400 policy episodes
# — already happened in the array; what is left is FID/CV/IP/CTR/SCP over
# cached tensors plus two single-image diff plots, i.e. minutes. The only real
# costs are loading ~2GB of model weights and the CTR segmentation pass, both
# now batched.
#
# It still needs a GPU (and the same node constraints as the array — see
# run_eval_cf.slurm's header for why hpc-n983 is excluded by name) because the
# metric backbones, the segmenter and the diff plots all run on device.
#
# Re-running this is cheap and safe: the shards are read-only inputs, so a
# metric fix or a changed threshold is a resubmit of this job alone, with no
# regeneration. But do NOT edit the generation knobs in config.py first — the
# shards carry a config stamp and this job refuses to report counterfactuals
# from one ablation arm under another arm's config.

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

# Must match run_eval_cf.slurm's NUM_SHARDS. Passing it here is a cross-check,
# not the source of truth: the shards themselves record the count they were
# written with, and eval_main.py errors if the two disagree.
NUM_SHARDS=8

# Must be the same directory eval_main.py defaults --shard-dir to, which is
# config.CKPT_DIR/eval_shards — per-domain, so hardcoding "checkpoints" here
# would report "the generate stage has not run" for every non-rsna domain.
SHARD_DIR=$(python -c 'import os, config; print(os.path.join(config.CKPT_DIR, "eval_shards"))')
if [ ! -d "${SHARD_DIR}" ]; then
    echo "${SHARD_DIR} does not exist — the generate stage has not run." >&2
    echo "    sbatch run_eval_cf.slurm" >&2
    exit 1
fi

echo "=== Eval reduce started on $(hostname) at $(date) ==="
echo "shards present: $(ls -1 ${SHARD_DIR}/cf_shard_*.pt 2>/dev/null | wc -l) / ${NUM_SHARDS}"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

python -u preflight_gpu.py

python -u eval_main.py --stage reduce --num-shards "${NUM_SHARDS}"

echo "=== Eval reduce finished at $(date) ==="
