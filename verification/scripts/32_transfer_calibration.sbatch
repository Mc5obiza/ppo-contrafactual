#!/bin/bash
# J20 — second review, items 2 and 10: transfer-ratio calibration.
# A: test-fold AUROC, real-pair rho ceiling and per-head operating points for
#    validator heads Pneumonia / Consolidation / Infiltration / Lung Opacity.
# B: rho and decision-flip rates at those operating points on J4, J5, J12, J15, J10.
# C: FGSM / PGD-Linf / norm-matched PGD-L2 against the guidance ensemble on J4's
#    sources (adversarial floor). CPU (GPU queue is busy with J16/J17).
#SBATCH --job-name=verif-J20-calib
#SBATCH --partition=public
#SBATCH --account=insa
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
OUT="$HOME/verification/out/${TAG:-J20_transfer_calibration}"; mkdir -p "$OUT"
cd "$HOME/verification/code/49e6050"
module load python/3.12.8
source venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 YOLO_OFFLINE=1 CUDA_VISIBLE_DEVICES=
export PPO_CKPT_DIR="$HOME/verification/views/committed"   # only for the split file
export PYTHONPATH="$PWD:$HOME/verification/scripts" OMP_NUM_THREADS=32 MKL_NUM_THREADS=32
echo "=== J20 on $(hostname) at $(date)"
python -u "$HOME/verification/scripts/transfer_calibration.py" "$OUT"
sha256sum "$HOME/verification/scripts/transfer_calibration.py" > "$OUT/script.sha256"
echo "=== finished at $(date)"
