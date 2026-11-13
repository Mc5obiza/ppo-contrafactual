#!/bin/bash
# J3 — Grad-CAM degeneracy + CAM-lesion lift diagnostic, default layer AND the
# denseblock3 remedy. Inference only (classifier forward/backward for CAMs; no
# generator, no policy).
#
# Paper claims (sec 5.5): chex 66/263 (25.1%) vs rsna 1/263 at the default
# layer; CAM-lesion lift 2.052 (n=127 clean) vs 1.347 (n=23 partially
# degenerate); remedy 0%/0.4% degeneracy, baseline CAM-lesion lift 1.196.
# On disk only the remedy run's degeneracy_log.json survives (chex 0/263,
# rsna 1/263); the default-layer counts and all four lifts have no artifact.
#
# CAVEAT: check_classifier_localization.py samples ENSEMBLE_N_IMAGES=150 images;
# confirm in the log that it reports the same 263 member-image observations. If
# the sampling is not seeded identically, expect different counts, and treat
# the result as a replication, not a recovery.
#SBATCH --job-name=verif-gradcam-diag
#SBATCH --partition=publicgpu
#SBATCH --account=insa
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpuh100|gpua100|gpul40s|gpuh200"
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
OUT="$HOME/verification/out/gradcam"; mkdir -p "$OUT"
cd "$HOME/verification/code/HEAD"     # writes outputs/gradcam_check/ inside the snapshot
module load python/3.12.8; module load cuda/cuda-12.1
source venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 YOLO_OFFLINE=1 PPO_DOMAIN=rsna_pneumonia

echo "=== default target layer"
python -u check_classifier_localization.py 2>&1 | tee "$OUT/default_layer.txt"
cp outputs/gradcam_check/degeneracy_log.json "$OUT/degeneracy_default.json"

echo "=== features.denseblock3 (remedy / trained Grad-CAM arm)"
GRADCAM_TARGET_LAYER_OVERRIDE=features.denseblock3 \
    python -u check_classifier_localization.py 2>&1 | tee "$OUT/denseblock3.txt"
cp outputs/gradcam_check/degeneracy_log.json "$OUT/degeneracy_denseblock3.json"
