#!/usr/bin/env bash
#SBATCH --job-name=ppo-counterfactual
#SBATCH --partition=publicgpu
#SBATCH --account=insa
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpuh200|gpuh100|gpua100|gpul40s|gpua40|gpurtx6000"
#SBATCH --exclude=hpc-n983
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=melkiayoub724@gmail.com
#
# Runs the TRAINING half of the PPO counterfactual pipeline (main.py):
# Steps 1-5 (ingestion -> pretrained models -> generator -> LoRA fine-tuning
# -> PPO training -> checkpoints).
#
# Evaluation is NOT run here, and main.py contains no evaluation code — the
# metric suite and diff plots live in eval_main.py. After BOTH array tasks
# succeed, submit the two eval stages, which reload
# checkpoints/{n_to_p,p_to_n}/ppo_policy.pt:
#   sbatch run_eval_cf.slurm                                  # note the job id
#   sbatch --dependency=afterok:<jobid> run_eval_reduce.slurm
# (Evaluation is sharded across an array because generation is ~29h and the
# partition caps jobs at 24h — see run_eval_cf.slurm's header. The old
# single-job run_eval_ppo.slurm is superseded and cannot finish at n_eval=200.)
#
# ─────────────────────────────────────────────────────────────────────────────
# ONE ARRAY TASK PER DIRECTION  (--array=0-1)
# ─────────────────────────────────────────────────────────────────────────────
# Task 0 trains n_to_p, task 1 trains p_to_n. They are independent: the only
# thing they share is the Step 4 LoRA generator, which finetuning.py resumes
# from checkpoints/finetuning_latest.pt as a no-op once it exists.
#
# WHY: publicgpu has MaxTime=1-00:00:00 and it CANNOT be raised, so 24h is a
# hard ceiling per job — but it is a ceiling per JOB, not per pipeline. Job
# 17653310 ran both directions in one job and spent its entire 24h reaching
# ~30 of 100 updates of n_to_p alone. Splitting gives each direction its own
# full 24h at zero cost to the training itself.
#
# To rerun just one direction:  sbatch --array=0 run_ppo.slurm
#
# ─────────────────────────────────────────────────────────────────────────────
# GPU CONSTRAINT — why this is not optional
# ─────────────────────────────────────────────────────────────────────────────
# Job 17653310 landed on hpc-n744, a Tesla P100-PCIE-16GB. The P100 (sm_60) has
# NO fp16 tensor cores, so config.DTYPE=torch.float16 buys nothing there, and
# PPO measured ~4.6 min PER EPISODE — a full direction is ~74h at that rate.
#
# In this cluster's feature vocabulary the P100 nodes carry `gpudp` but NOT
# `gputc`, so the constraint above excludes them (and the RTX 5000) by naming
# the modern tensor-core cards explicitly. Verify a constraint change with:
#   sinfo -p publicgpu -N -h -O "NodeList:12,Features:90"
#
# The OR list is a HARD requirement, not a preference: this cluster runs SLURM
# 22.05.10 and `--prefer` (soft constraints) was only added in 23.02. Node
# weighting still means the faster cards get picked first when free.
#
# Widen it (append |gpuv100) if the queue wait becomes worse than the speed
# gain. Do NOT add gpup100 back.
#
# --exclude=hpc-n983 — TOO NEW, NOT TOO OLD
# ------------------------------------------
# hpc-n983 is an "NVIDIA RTX PRO 6000 Blackwell Server Edition" (sm_120), but
# it advertises the SAME `gpurtx6000` feature as the Turing Quadro RTX 6000
# nodes hpc-n[871-882] (sm_75) that this job actually wants. The feature string
# cannot tell them apart and SLURM constraints have no NOT operator, so the
# only way to keep the Turing nodes while dropping the Blackwell one is to
# exclude it by name.
#
# WHY IT MATTERS: venv torch is 2.5.1+cu121, whose kernels are compiled for
#   ['sm_50','sm_60','sm_70','sm_75','sm_80','sm_86','sm_90']
# Blackwell is sm_120, so every CUDA op dies with "no kernel image is available
# for execution on the device". Job 17662452_0 lost 5 minutes to exactly this,
# failing in peft's get_peft_model() during Step 3.
#
# Check the current list yourself with:
#   python -c "import torch; print(torch.cuda.get_arch_list())"
# Removing this exclusion requires a torch built for sm_120 (CUDA 12.8+), which
# means rebuilding the venv — not just deleting the line.
#
# ─────────────────────────────────────────────────────────────────────────────
# RESUMING — a walltime kill is no longer fatal
# ─────────────────────────────────────────────────────────────────────────────
# PPOTrainer.train() writes checkpoints/<direction>/ppo_latest.pt every
# config.PPO_CKPT_EVERY_N_UPDATES updates, and stops cleanly before starting an
# update that would run past PPO_WALLCLOCK_BUDGET_S below. If a task reports
# "TRAINING PAUSED", just resubmit it — it continues from the saved update
# rather than restarting. Nothing needs to be deleted between attempts.
#
# The budget is set BELOW --time on purpose: the margin covers job startup
# (dataset scan + model loading), the final checkpoint write, and the guidance
# diagnostics that run after training. Raise --time and this together, or
# neither.
#
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

# Soft deadline for PPOTrainer.train(), ~2h under the 24h --time limit.
export PPO_WALLCLOCK_BUDGET_S=$((22 * 3600))

if [ ! -d venv ]; then
    echo "venv/ not found — run 'bash setup_env.sh' on the login node first." >&2
    exit 1
fi

# Map array task -> direction. Defaults to task 0 so the script still works if
# it is ever run without --array (e.g. bash run_ppo.slurm for a quick check).
case "${SLURM_ARRAY_TASK_ID:-0}" in
    0) DIRECTION=n_to_p ;;
    1) DIRECTION=p_to_n ;;
    *) echo "Unexpected SLURM_ARRAY_TASK_ID='${SLURM_ARRAY_TASK_ID}' — expected 0 or 1." >&2
       exit 1 ;;
esac

echo "=== Job ${SLURM_JOB_ID:-?} (array task ${SLURM_ARRAY_TASK_ID:-0}, direction ${DIRECTION})"
echo "=== started on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
echo "=== wall-clock budget: $((PPO_WALLCLOCK_BUDGET_S / 3600))h of 24h ==="

# Does this torch actually have kernels for the GPU we landed on? See
# preflight_gpu.py — costs one torch import (~30s) and replaces an opaque CUDA
# failure that otherwise surfaces ~5 minutes in. Shared with run_eval_ppo.slurm.
python -u preflight_gpu.py

# Unbuffered so the log is readable while the job is still running — PPO
# training only prints every N updates, and if it stalls you want to see
# exactly which phase it stopped in.
python -u main.py --direction "${DIRECTION}"

echo "=== Job finished at $(date) ==="
echo "=== If this task reported PAUSED: sbatch --array=${SLURM_ARRAY_TASK_ID:-0} run_ppo.slurm"
echo "=== Once BOTH directions are done: sbatch run_eval_cf.slurm, then"
echo "===   sbatch --dependency=afterok:<that job id> run_eval_reduce.slurm ==="
