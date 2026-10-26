#!/usr/bin/env bash
#SBATCH --job-name=breast-classifier
#SBATCH --partition=publicgpu
#SBATCH --account=insa
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# Trains f(.) for the cbis_ddsm_breast domain (train_breast_classifier.py).
#
# THIS IS A PREREQUISITE, NOT AN OPTIONAL STEP. Unlike the chest domains, whose
# classifiers come pretrained from torchxrayvision, mammography has no
# pretrained head anywhere in that zoo — so run_ppo.slurm cannot start until
# this has produced checkpoint/breast/*.pt.
#
# ONE BACKBONE PER ARRAY TASK
# ---------------------------
# This used to train all of BREAST_BACKBONES in one 4h job, and that is exactly
# what kept timing out: the third backbone started ~3h in and got killed
# mid-epoch (job 17672062, resnet18). Each array task now trains exactly one
# backbone and gets the full 4h to itself. The tasks are independent — they
# share only the dataset and the seed, and the patient split is a pure function
# of (seed, val_frac), so every task validates on the same fold.
#
# The array index selects the arch from config.BREAST_BACKBONES, so the list
# stays single-sourced; --array above must cover it and the script checks that.
#
# Order:
#   1. bash setup_env.sh                  (login node — venv + weights + data)
#   2. sbatch run_breast_classifier.slurm (this)
#   3. sbatch run_lesion_segmenter.slurm  (LLR/MSI need it)
#   4. sbatch run_ppo.slurm
#
# Useful variants:
#   sbatch --array=2 run_breast_classifier.slurm      # just resnet18
#   sbatch run_breast_classifier.slurm --epochs 30    # extra args pass through
#   SKIP_EXISTING=1 sbatch run_breast_classifier.slurm  # resume: only missing
#
# After the array finishes, apply the AUC floor across all three:
#   python train_breast_classifier.py --summary
# or let the array do it for you (submits the gate as a dependent job):
#   jid=$(sbatch --parsable run_breast_classifier.slurm)
#   sbatch --dependency=afterany:$jid --wrap \
#     "source venv/bin/activate && python train_breast_classifier.py --summary"
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

# Which backbone is this task? Read the list from config so adding a fourth
# backbone needs one edit, not two. Out of range is a hard error rather than a
# silent no-op: a task that trains nothing must not exit 0 and let the gate
# downstream think the checkpoint is coming.
IDX="${SLURM_ARRAY_TASK_ID:-0}"
ARCH=$(python - "$IDX" <<'PY'
import sys
import config
backbones = list(config.BREAST_BACKBONES)
i = int(sys.argv[1])
if i >= len(backbones):
    sys.exit(
        f"array index {i} is past the end of BREAST_BACKBONES "
        f"({len(backbones)} entries: {backbones}) — fix #SBATCH --array"
    )
print(backbones[i])
PY
)

EXTRA=()
if [ "${SKIP_EXISTING:-0}" != "0" ]; then
    EXTRA+=(--skip-existing)
fi

echo "=== Job ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID} task ${IDX} (${ARCH}) on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
python -u train_breast_classifier.py --arch "$ARCH" "${EXTRA[@]}" "$@"
echo "=== Task ${IDX} (${ARCH}) finished at $(date) ==="
