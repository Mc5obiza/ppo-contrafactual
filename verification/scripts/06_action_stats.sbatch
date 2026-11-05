#!/bin/bash
# J6 — per-image/per-step actions of the COMMITTED policies on all 400 eval
# images, with a reproduction check against the saved counterfactuals.
# Inference only; same cost as one committed eval (~25-45 min per shard).
# Settles: adaptivity (sec 1, Table 1 caption, limitations "Adaptivity is not
# measured"), whether n->p guidance bias is pinned at its 1.30 ceiling at eval
# time as it is in the training diagnostics (1.2980 +/- 0.0019).
#SBATCH --job-name=verif-actions
#SBATCH --array=0-7
#SBATCH --partition=publicgpu
#SBATCH --account=insa
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpuh200|gpuh100|gpua100|gpul40s"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home2020/home/insa/arbaoui/verification/logs/%x-%A_%a.out
#SBATCH --error=/home2020/home/insa/arbaoui/verification/logs/%x-%A_%a.err
set -euo pipefail
R="$HOME/Jobs/Ayoub/ppo-contrafactual"
OUT="$HOME/verification/out/actions"; mkdir -p "$OUT"
cd "$HOME/verification/code/49e6050"
module load python/3.12.8; module load cuda/cuda-12.1
source venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 YOLO_OFFLINE=1 PYTHONPATH="$PWD"
export PPO_CKPT_DIR="$HOME/verification/views/committed"
source "$HOME/verification/scripts/adapters_committed.env"   # n->p committed, p->n pn_lora adapter
K=$(printf '%03d' "${SLURM_ARRAY_TASK_ID}")
nvidia-smi --query-gpu=name,memory.total --format=csv
python -u "$HOME/verification/scripts/action_stats.py" \
    "$R/checkpoints/eval_shards_ens_default/cf_shard_${K}.pt" "$OUT/actions_shard_${K}.json"
