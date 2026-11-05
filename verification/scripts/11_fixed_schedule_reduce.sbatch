#!/bin/bash
# J5b — fixed-schedule arm, REDUCE stage + transfer. Through the same wrapper,
# so the diff plots also use the constant action, not the trained policies.
#SBATCH --job-name=verif-fixed-reduce
#SBATCH --partition=publicgpu
#SBATCH --account=insa
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpuh200|gpuh100|gpua100|gpul40s"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home2020/home/insa/arbaoui/verification/logs/%x-%j.out
#SBATCH --error=/home2020/home/insa/arbaoui/verification/logs/%x-%j.err
set -euo pipefail
TAG=J5b_fixed_schedule
OUT="$HOME/verification/out/$TAG"
cd "$HOME/verification/code/49e6050"
module load python/3.12.8; module load cuda/cuda-12.1
source venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 YOLO_OFFLINE=1 PYTHONPATH="$PWD"
export PPO_CKPT_DIR="$HOME/verification/views/committed"
source "$HOME/verification/scripts/adapters_committed.env"   # n->p committed, p->n pn_lora adapter
python -u "$HOME/verification/scripts/fixed_schedule_eval.py" "$HOME/verification/out/actions" \
    --stage reduce --num-shards 8 --n-eval 200 \
    --arm trained --shard-dir "$OUT/shards" --out-dir "$OUT" --tag "$TAG" --base-seed 0
cd "$HOME/verification/code/HEAD"
PPO_DOMAIN=rsna_pneumonia python -u arm_transfer.py "$OUT/shards" --bootstrap --json "$OUT/transfer.json"
