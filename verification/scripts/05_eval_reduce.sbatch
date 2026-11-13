#!/bin/bash
# REDUCE stage for a 04_eval_generate.sbatch run. Same --export variables.
#   sbatch --dependency=afterok:<gen jobid> --export=ALL,VIEW=pn_lora,ARM=trained,TAG=J4_pn_lora 05_eval_reduce.sbatch
#SBATCH --job-name=verif-eval-reduce
#SBATCH --partition=publicgpu
#SBATCH --account=insa
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpuh200|gpuh100|gpua100|gpul40s"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
: "${VIEW:?set VIEW}" "${ARM:?set ARM}" "${TAG:?set TAG}"
OUT="$HOME/verification/out/$TAG"
cd "$HOME/verification/code/49e6050"
module load python/3.12.8; module load cuda/cuda-12.1
source venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 YOLO_OFFLINE=1
export PPO_CKPT_DIR="$HOME/verification/views/$VIEW"
export PYTHONPATH="$PWD"
ENTRY=eval_main.py
if [ -n "${PN_VIEW:-}" ]; then   # per-direction adapters (scripts/per_direction.py)
    export ADAPTER_NP_VIEW="$HOME/verification/views/$VIEW"
    export ADAPTER_PN_VIEW="$HOME/verification/views/$PN_VIEW"
    ENTRY="$HOME/verification/scripts/eval_per_direction.py"
fi
python -u "$ENTRY" --stage reduce --num-shards 8 --n-eval "${NEVAL:-200}" \
    --arm "$ARM" --shard-dir "$OUT/shards" --out-dir "$OUT" --tag "$TAG" --base-seed 0 ${EXTRA:-}
# Transfer for the new arm (HEAD tool, see 02); denominator read from the new shards' stamp.
cd "$HOME/verification/code/HEAD"
PPO_DOMAIN=rsna_pneumonia python -u arm_transfer.py "$OUT/shards" --bootstrap --json "$OUT/transfer.json"
