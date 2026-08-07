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

echo "=== 1/6  Loading Python module ==="
module load python/3.12.8

echo "=== 2/6  Creating virtualenv (./venv) ==="
if [ ! -d venv ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install --upgrade pip setuptools wheel

echo "=== 3/6  Installing dependencies ==="
# torch/torchvision are installed separately, pinned to CUDA 12.1 wheels
# (matches the driver on this cluster's V100 nodes; also works on newer GPUs).
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

echo "=== 4/6  Prefetching pretrained model weights (needs internet) ==="
python - <<'PYEOF'
import torch
import config

print("[prefetch] YOLOv8n weights ...")
from ultralytics import YOLO
YOLO(config.YOLOV8_WEIGHTS)

print("[prefetch] TorchXRayVision PSPNet segmenter ...")
from src.models import build_segmentation_net
build_segmentation_net(device="cpu")

print("[prefetch] TorchXRayVision DenseNet-121 weights ...")
import torchxrayvision as xrv
import domains

# Every weight set ANY registered domain names, not just the active one —
# compute nodes have no internet, so a domain switch must not require another
# login-node round trip. Adding a domain to domains/_REGISTRY automatically
# pulls its weights here. The two literals are needed regardless of domain:
# rsna backs FID's feature extractor, all backs the FOP detector.
needed = {"densenet121-res224-rsna", "densenet121-res224-all"}
for _name in domains.available():
    _d = domains.get(_name)
    needed |= {_d.guidance_weights, *_d.ensemble_weights, *_d.heldout_validators}

for _w in sorted(needed):
    print(f"[prefetch]   -> {_w}")
    xrv.models.DenseNet(weights=_w)

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

# Build f(.) for EVERY registered domain, offline. This is what actually fails
# on a compute node if a weight set was missed above, and it simultaneously
# exercises the dead-head guard in XRVPneumoniaClassifier — a NaN op_thresh
# raises here, on the login node, instead of silently training against a
# constant f(.) hours into a job.
import domains
from src.xrv_classifier import build_xrv_classifier
for _name in domains.available():
    _d = domains.get(_name)
    for _w in {_d.guidance_weights, *_d.ensemble_weights, *_d.heldout_validators}:
        build_xrv_classifier(device="cpu", weights=_w, pathology=_d.pathology)
    print(f"[prefetch]   domain {_name}: f(.) head {_d.pathology!r} OK on all weight sets")

# Optional per-domain readers. Missing ones are reported, not fatal — the RSNA
# domain does not need SimpleITK, so a missing .mha reader must not block setup.
try:
    import SimpleITK  # noqa: F401
    print("[prefetch] SimpleITK OK (.mha reader for the node21_nodule domain)")
except ImportError:
    print("[prefetch] WARNING: SimpleITK missing — the node21_nodule domain "
          "cannot read NODE21's .mha files. `pip install SimpleITK` before "
          "switching config.DATASET_DOMAIN to it.")

print("[prefetch] OFFLINE MODEL BUILD OK")
PYEOF

echo "=== 5/6  RSNA dataset (Kaggle) ==="
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
echo "=== 6/6  Other dataset domains ==="
python - <<'PYEOF'
import config, domains, os
active = config.DATASET_DOMAIN
print(f"[domains] registered: {domains.available()}")
print(f"[domains] active (config.DATASET_DOMAIN): {active}")
if "node21_nodule" in domains.available():
    root = getattr(config, "NODE21_ROOT", "") or ""
    if root and os.path.isdir(root):
        print(f"[domains] node21_nodule: NODE21_ROOT ok -> {root}")
    else:
        print(
            "[domains] node21_nodule: NODE21_ROOT is unset or missing.\n"
            "          NODE21 is behind grand-challenge registration, so there is\n"
            "          no auto-download. Fetch it ON THE LOGIN NODE (compute nodes\n"
            "          have no internet) from\n"
            "            https://node21.grand-challenge.org/Data/\n"
            "          extract it, then set config.NODE21_ROOT to that directory\n"
            "          and config.DATASET_DOMAIN = 'node21_nodule'."
        )
PYEOF

echo ""
echo "=== Setup complete ==="
echo "Environment ready at: $(pwd)/venv"
echo "Submit the training job with: sbatch run_ppo.slurm"
