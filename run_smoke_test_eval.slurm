#!/usr/bin/env bash
#SBATCH --job-name=ppo-smoke-eval
#SBATCH --partition=publicgpu
#SBATCH --account=insa
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpuh200|gpuh100|gpua100|gpul40s|gpua40|gpurtx6000"
#SBATCH --exclude=hpc-n983
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=melkiayoub724@gmail.com
#
# Smoke test for the EVALUATION half only (smoke_test_eval.py): reloads trained
# policies in a COLD process and runs eval_main.main() end to end at tiny scale.
#
# WHY SEPARATE FROM run_smoke_test.slurm: that one runs training and evaluation
# back to back in one process, so eval reloads checkpoints while the generator,
# envs and CUDA context from training are still live. The failure modes that
# matter for run_eval_ppo.slurm — missing files, state_dict key mismatches,
# obs_dim drift between the saved policy and the env eval rebuilds — only show
# up in a cold process. This is that cold process.
#
# Budgets are slashed (T_DIFF 50->10, BUDGET_T 25->3, n_eval 200->12), so the
# CV/FID/IP numbers it prints are meaningless by construction. A PASS means
# "the checkpoints load and the eval path runs", not "the model is good".
#
# Usage:
#   sbatch run_smoke_test_eval.slurm
#       Reads checkpoints/smoke_all/ — i.e. what `SMOKE_ARM=all sbatch
#       run_smoke_test.slurm` wrote. Verifies the plumbing.
#
#   SMOKE_EVAL_SRC=checkpoints sbatch run_smoke_test_eval.slurm
#       Reads the REAL checkpoints/{n_to_p,p_to_n}/. Run this after
#       run_ppo.slurm finishes and BEFORE run_eval_ppo.slurm — it proves the
#       real policies load in ~minutes instead of discovering a reload bug
#       2 hours into a 5-hour eval job.
#
# SAFETY: everything written (eval_metrics.json, the two diff PNGs) goes to
# checkpoints/smoke_eval/, never to the directory being read. So the second
# form above cannot overwrite real evaluation results. smoke_test_eval.py
# asserts this by snapshotting the source directory before and after.
#
# Compute nodes have NO internet access, so this only ever uses the venv and
# pretrained weights setup_env.sh already downloaded on the login node.

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

# sbatch does not forward the submitting shell's environment by default, so
# re-export with defaults. Override at submit time, e.g.
#   SMOKE_EVAL_SRC=checkpoints sbatch run_smoke_test_eval.slurm
export SMOKE_EVAL_SRC="${SMOKE_EVAL_SRC:-checkpoints/smoke_all}"
# Must match run_smoke_test.slurm's subset — this job reloads the checkpoints
# that one wrote, and a different subset means a different split, which
# splits.load() will (correctly) refuse as a dataset-fingerprint mismatch.
# See that file for why 96 became 400.
export SMOKE_SUBSET_N="${SMOKE_SUBSET_N:-400}"
export SMOKE_N_EVAL="${SMOKE_N_EVAL:-12}"

echo "=== Eval smoke test started on $(hostname) at $(date) ==="
echo "    SMOKE_EVAL_SRC=${SMOKE_EVAL_SRC}  SUBSET_N=${SMOKE_SUBSET_N}  N_EVAL=${SMOKE_N_EVAL}"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

# Same hardware check as run_ppo.slurm — see preflight_gpu.py. A smoke test
# that dies on an unsupported GPU reports a hardware problem as a code
# problem, which is the opposite of what it exists to do.
python -u preflight_gpu.py

# Unbuffered so the log is useful while the job is still running (and if it
# hangs, you can see exactly which stage it stopped in).
# See the note in run_smoke_test.slurm: `set -euo pipefail` makes a bare
# `status=$?` after a failing command unreachable.
status=0
python -u smoke_test_eval.py || status=$?

echo "=== Eval smoke test finished at $(date) with exit status ${status} ==="
exit ${status}
