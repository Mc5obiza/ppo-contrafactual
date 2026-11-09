#!/usr/bin/env bash
#SBATCH --job-name=breast-classifier-gate
#SBATCH --partition=public
#SBATCH --account=insa
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Applies the MIN_ACCEPTABLE_AUC floor across every backbone in
# config.BREAST_BACKBONES by reading the val_auc each one recorded in its
# checkpoint. Exits non-zero if any is weak or missing.
#
# WHY THIS IS A SEPARATE JOB
# --------------------------
# run_breast_classifier.slurm is a job array, one backbone per task. No task
# sees the others' results, so none of them can apply a gate that is about all
# three. This job runs after they finish and does it in one place.
#
# No GPU: it only torch.loads checkpoints onto the CPU and compares floats.
#
# Usage:
#   sbatch run_breast_classifier_gate.slurm
# or chained behind the training array so it fires automatically:
#   jid=$(sbatch --parsable run_breast_classifier.slurm)
#   sbatch --dependency=afterany:$jid run_breast_classifier_gate.slurm
#
# afterany, not afterok: if one backbone fails the gate should still run and
# name it. That is the whole point of the report.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}"
mkdir -p logs

module load python/3.12.8

if [ ! -d venv ]; then
    echo "venv/ not found — run 'bash setup_env.sh' on the login node first." >&2
    exit 1
fi
source venv/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export YOLO_OFFLINE=1

echo "=== Gate started on $(hostname) at $(date) ==="
python -u train_breast_classifier.py --summary
echo "=== Gate finished at $(date) ==="
