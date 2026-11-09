#!/usr/bin/env bash
#SBATCH --job-name=cyclegan-baseline
#SBATCH --partition=publicgpu
#SBATCH --account=insa
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpuv100|gpuh200|gpuh100|gpua100|gpul40s"
#SBATCH --exclude=hpc-n983
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Trains the Vanilla CycleGAN baseline (train_cyclegan_baseline.py) — see that
# script's docstring for what this is and why it exists (a measured,
# same-protocol replacement for the literature-reported CycleGAN row in
# report/report.tex's / report/paper_cvpr.tex's baseline-comparison table).
#
# BUDGET: benchmarked at ~110ms/image (V100, batch_size=4, 256x256, 9-resblock
# ResnetGenerator + 70x70 PatchGAN) -> ~40 epochs (one epoch = one full pass
# over the larger of the two label-partitioned training domains, ~16.5k images
# for RSNA normal/pneumonia) fits a 24h budget with margin. The 22h soft
# wallclock budget below leaves room for job startup + the final checkpoint
# write, exactly like run_ppo.slurm's PPO_WALLCLOCK_BUDGET_S.
#
# RESUMING: if walltime is hit anyway, cyclegan_latest.pt has everything
# needed to continue — just resubmit this script unchanged.
#
# GPU constraint widened to include V100 (unlike run_ppo.slurm, which needs
# fp16 tensor cores for the diffusion path): this is fp32 InstanceNorm-based
# convnets, cheap enough that the V100 nodes are worth including rather than
# waiting for a scarcer modern card. See run_ppo.slurm's own header for the
# hpc-n983 (Blackwell/sm_120) exclusion rationale — same torch build, same
# problem.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}"
mkdir -p logs

module load python/3.12.8
module load cuda/cuda-12.1
source venv/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export YOLO_OFFLINE=1

CKPT_DIR="checkpoints_cyclegan_baseline"
export CYCLEGAN_WALLCLOCK_BUDGET_S=$((22 * 3600))
# Overlap DICOM decode with GPU compute — the benchmark that sized --epochs 40
# below (see this script's header) measured pure GPU cost with synthetic
# tensors, so worker throughput matters here in a way it didn't there.
export PPO_NUM_WORKERS=6

if [ ! -d venv ]; then
    echo "venv/ not found — run 'bash setup_env.sh' on the login node first." >&2
    exit 1
fi

echo "=== Job ${SLURM_JOB_ID:-?} started on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
echo "=== wall-clock budget: $((CYCLEGAN_WALLCLOCK_BUDGET_S / 3600))h of 24h ==="

bash record_run_manifest.sh "${CKPT_DIR}"

python -u preflight_gpu.py

python -u train_cyclegan_baseline.py \
    --ckpt-dir "${CKPT_DIR}" \
    --epochs 40 \
    --batch-size 4

echo "=== Job finished at $(date) ==="
echo "=== If this reported PAUSED: sbatch run_cyclegan_baseline.slurm (resumes automatically)"
echo "=== Once training is DONE: python -u eval_cyclegan_baseline.py --ckpt-dir ${CKPT_DIR} ==="
