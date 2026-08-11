#!/usr/bin/env bash
#SBATCH --job-name=ppo-smoke-test
#SBATCH --partition=publicgpu
#SBATCH --account=insa
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpuh200|gpuh100|gpua100|gpul40s|gpua40|gpurtx6000"
#SBATCH --exclude=hpc-n983
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=melkiayoub724@gmail.com
#
# Smoke test of the TRAINING pipeline at tiny scale (smoke_test.py):
# ingestion -> LoRA fine-tuning -> PPO both directions -> checkpoints.
#
# Evaluation is NOT covered here — that is run_smoke_test_eval.slurm, which
# reloads these checkpoints in a COLD process. Splitting them mirrors the real
# jobs (run_ppo.slurm then run_eval_ppo.slurm) and is a strictly better test of
# the reload path: run in-process here, eval would reuse the generator, envs and
# CUDA context training just built, hiding the very failures that matter.
#
# WHAT THIS IS FOR: catching crashes, shape errors, OOM, missing prefetched
# weights and mis-wired flags BEFORE committing 24h of GPU time to run_ppo.slurm.
# Budgets are slashed (T_DIFF 50->10, BUDGET_T 25->3, N_EPISODES 800->24,
# N_DENOISE_EPOCHS 15->1), so nothing it prints is a meaningful metric.
# A PASS means "it runs", not "it works well".
#
# Checkpoints land in checkpoints/smoke_<arm>/ and can NEVER overwrite the real
# run's policies or LoRA weights.
#
# Recommended order:
#   sbatch run_smoke_test.slurm         # this file — training
#   sbatch run_smoke_test_eval.slurm    # eval half, reads checkpoints/smoke_all
#
# Walltime is 3h — generous for a run that should take well under one. If it
# hits the limit, something is hanging (most likely dataset ingestion or an
# offline weight fetch retrying), which is itself the finding.
#
# Compute nodes have NO internet access, so this only ever uses the venv and
# pretrained weights setup_env.sh already downloaded on the login node.
# Run setup_env.sh first — this job will fail fast otherwise.
#
# Usage:
#   sbatch run_smoke_test.slurm                    # arm=all (most coverage)
#   SMOKE_ARM=baseline sbatch run_smoke_test.slurm  # pre-change pipeline
#   SMOKE_ARM=mask     sbatch run_smoke_test.slurm  # Grad-CAM mask + anchoring
#
#   Arms:
#     baseline — every new flag off; verifies nothing regressed
#     mask     — Grad-CAM action-space mask + episode-origin anchoring
#     all      — mask + anchoring + sparsity reward + annealing + CV safeguard
#
# Recommended before the real run: `baseline` then `all`. The first proves the
# refactor did not break the existing pipeline; the second proves the new code
# paths execute on real data.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}"
mkdir -p logs

module load python/3.12.8
# torch is built for cu121 (torch.version.cuda == "12.1") and the driver is
# CUDA 12.6. cuda/13.0 (the module system default) prepends its lib64 to
# LD_LIBRARY_PATH ahead of torch's own RUNPATH and only fails to break things
# because CUDA 13 sonames are .so.13 while torch links .so.12. Load the
# matching toolkit instead of relying on that.
module load cuda/cuda-12.1

if [ ! -d venv ]; then
    echo "venv/ not found — run 'bash setup_env.sh' on the login node first." >&2
    exit 1
fi
source venv/bin/activate

# Force offline mode: fail loudly instead of silently hanging on a network
# call that can never succeed from a compute node.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export YOLO_OFFLINE=1

# sbatch does not forward the submitting shell's environment by default for
# these, so re-export with defaults. Override at submit time, e.g.
#   SMOKE_ARM=baseline sbatch run_smoke_test.slurm
# SMOKE_N_EVAL is not set here — this job no longer runs evaluation. It is an
# input to run_smoke_test_eval.slurm instead.
export SMOKE_ARM="${SMOKE_ARM:-all}"
# RAISED 96 -> 400 when the train/test split landed. The subset is now split
# 80/20 like the real run, so a 96-image subset leaves a ~19-image eval fold —
# and after the f(x)<0.2 / f(x)>0.8 group thresholds, too few images to fill
# SMOKE_N_EVAL. The smoke test would then fail for a reason that is not a bug.
# 400 keeps ~40 per class in the eval fold. The cost is negligible: PPO episode
# count is fixed by N_EPISODES, and one LoRA epoch over 320 images is seconds.
export SMOKE_SUBSET_N="${SMOKE_SUBSET_N:-400}"

echo "=== Smoke test (training) started on $(hostname) at $(date) ==="
echo "    SMOKE_ARM=${SMOKE_ARM}  SUBSET_N=${SMOKE_SUBSET_N}"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

# Same hardware check as run_ppo.slurm — see preflight_gpu.py. A smoke test
# that dies on an unsupported GPU reports a hardware problem as a code
# problem, which is the opposite of what it exists to do.
python -u preflight_gpu.py

# Unbuffered so the log is useful while the job is still running (and if it
# hangs, you can see exactly which stage it stopped in).
# `set -euo pipefail` is active, so a plain `python ...` followed by
# `status=$?` is dead code — a non-zero exit kills the script before the
# assignment runs, and the log ends with no "finished" line, which reads as a
# hang rather than as the failure it is. Capture the status explicitly.
status=0
python -u smoke_test.py || status=$?

echo "=== Smoke test finished at $(date) with exit status ${status} ==="
exit ${status}
