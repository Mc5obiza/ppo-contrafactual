#!/usr/bin/env bash
# setup_env.sh
# ============
# Run ONCE on a node with internet access (the login node), NOT on a compute
# node. Compute nodes on this cluster have no internet access, so every
# dependency and every pretrained-model file must be downloaded here first;
# the SLURM job (run_ppo.slurm) then runs fully offline against what this
# script fetched.
#
# Usage:
#   bash setup_env.sh
#
# Re-running is safe: pip install and the *.from_pretrained() prefetch calls
# are all idempotent (skip work that's already done).

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# sbatch's --output/--error paths (see run_ppo.slurm) must exist BEFORE
# submission — sbatch opens them itself and fails the job instantly
# (0-second runtime, no log file anywhere) if the directory is missing.
mkdir -p logs

echo "=== 1/5  Loading Python module ==="
module load python/3.12.8

echo "=== 2/5  Creating virtualenv (./venv) ==="
if [ ! -d venv ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install --upgrade pip setuptools wheel

echo "=== 3/5  Installing dependencies ==="
# torch/torchvision are installed separately, pinned to CUDA 12.1 wheels
# (matches the driver on this cluster's V100 nodes; also works on newer GPUs).
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

echo "=== 4/5  Prefetching pretrained model weights (needs internet) ==="
python - <<'PYEOF'
import torch
import config

print("[prefetch] YOLOv8n weights ...")
from ultralytics import YOLO
YOLO(config.YOLOV8_WEIGHTS)

print("[prefetch] TorchXRayVision PSPNet segmenter ...")
from src.models import build_segmentation_net
build_segmentation_net(device="cpu")

print("[prefetch] TorchXRayVision DenseNet-121 (RSNA) — used by src/metrics.py FID) ...")
import torchxrayvision as xrv
xrv.models.DenseNet(weights="densenet121-res224-rsna")

print("[prefetch] TorchXRayVision DenseNet-121 (all) — used by src/metrics.py FOP) ...")
xrv.models.DenseNet(weights="densenet121-res224-all")

print("[prefetch] Stable Diffusion v1.5 (VAE + UNet + scheduler) ...")
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
AutoencoderKL.from_pretrained(config.DIFFUSION_MODEL_ID, subfolder="vae", torch_dtype=torch.float16)
UNet2DConditionModel.from_pretrained(config.DIFFUSION_MODEL_ID, subfolder="unet", torch_dtype=torch.float16)
DDIMScheduler.from_pretrained(config.DIFFUSION_MODEL_ID, subfolder="scheduler")

print("[prefetch] Inception-v3 (used by src/metrics.py FID computation) ...")
from torchvision.models import inception_v3, Inception_V3_Weights
inception_v3(weights=Inception_V3_Weights.DEFAULT)

print("[prefetch] done — verifying full offline model build ...")
PYEOF

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python - <<'PYEOF'
import torch
import config
config.DEVICE = "cpu"
from src.models import build_classifier, build_segmentation_net, build_object_detector
from src.generator import build_generator
from torchvision.models import inception_v3, Inception_V3_Weights
inception_v3(weights=Inception_V3_Weights.DEFAULT)

classifier = build_classifier(device="cpu")
seg_net, lung_idx = build_segmentation_net(device="cpu")
obj_det = build_object_detector(device="cpu")
build_generator(
    classifier=classifier, seg_net=seg_net, lung_channel_idx=lung_idx,
    obj_det=obj_det, device="cpu", dtype=torch.float32,
)
print("[prefetch] OFFLINE MODEL BUILD OK")
PYEOF

echo "=== 5/5  RSNA dataset (Kaggle) ==="
if [ -n "${KAGGLE_API_TOKEN:-}" ] || [ -f "$HOME/.kaggle/kaggle.json" ] || { [ -n "${KAGGLE_USERNAME:-}" ] && [ -n "${KAGGLE_KEY:-}" ]; }; then
    echo "[dataset] Kaggle credentials found, downloading RSNA dataset ..."
    python - <<'PYEOF' || true
from ingestion import download_rsna_dataset
try:
    download_rsna_dataset()
except Exception as e:
    print(f"[dataset] download failed: {e}")
    print(
        "[dataset] If this is a 403/permission error, you must accept the "
        "competition rules (while logged in as the token's account) at:\n"
        "  https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/rules\n"
        "Then re-run: bash setup_env.sh"
    )
PYEOF
else
    cat <<'MSG'
[dataset] SKIPPED — no Kaggle credentials found.

The RSNA Pneumonia Detection dataset is downloaded via kagglehub, which
needs a Kaggle API token. Kaggle's "Create New Token" button now generates a
Bearer-style token (looks like "KGAT_xxxxxxxx...") shown as text instead of
downloading kaggle.json — this is NOT the same as the old username+key pair,
so set it up as an env var:

    export KAGGLE_API_TOKEN="KGAT_your_token_here"

(older-style username+key credentials also still work, either via
~/.kaggle/kaggle.json = {"username":"...","key":"..."} or via
KAGGLE_USERNAME / KAGGLE_KEY env vars — use whichever your account shows.)

Get the token at https://www.kaggle.com/settings -> "Create New Token".
Then accept the competition rules at:
  https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/rules
(you must be logged in as the same account the token belongs to)

Re-run: bash setup_env.sh
  (or just: python -c "from ingestion import download_rsna_dataset; download_rsna_dataset()")

This step needs internet, so it must be done on the login node — the
compute node running run_ppo.slurm will not be able to download it.
MSG
fi

echo ""
echo "=== Setup complete ==="
echo "Environment ready at: $(pwd)/venv"
echo "Submit the training job with: sbatch run_ppo.slurm"
