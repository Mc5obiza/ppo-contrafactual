#!/bin/bash
# J7 — runtime and peak memory on ONE fixed GPU type, for a quotable
# "per-image runtime on <GPU>" sentence. The committed eval ran on a mix of
# H100 NVL, A100-80GB and Quadro RTX 6000, so its logs give a range, not a
# number. Runs action_stats.py on shard 0 only (25 + 25 images), pinned to H100.
# Reports seconds/image and torch peak allocated memory per image, per direction.
#SBATCH --job-name=verif-runtime
#SBATCH --partition=publicgpu
#SBATCH --account=insa
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpuh100"
# Not --exclusive (that would reserve a whole 4-GPU node). Other jobs may share
# the node's CPUs/PCIe, so report the median seconds/image, not the minimum.
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home2020/home/insa/arbaoui/verification/logs/%x-%j.out
#SBATCH --error=/home2020/home/insa/arbaoui/verification/logs/%x-%j.err
set -euo pipefail
R="$HOME/Jobs/Ayoub/ppo-contrafactual"
OUT="$HOME/verification/out/runtime"; mkdir -p "$OUT"
cd "$HOME/verification/code/49e6050"
module load python/3.12.8; module load cuda/cuda-12.1
source venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 YOLO_OFFLINE=1 PYTHONPATH="$PWD"
export PPO_CKPT_DIR="$HOME/verification/views/committed"
nvidia-smi --query-gpu=name,memory.total --format=csv | tee "$OUT/gpu.txt"
python -u "$HOME/verification/scripts/action_stats.py" \
    "$R/checkpoints/eval_shards_ens_default/cf_shard_000.pt" "$OUT/runtime_shard_000_h100.json"
