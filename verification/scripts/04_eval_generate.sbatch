#!/bin/bash
# Generic 8-shard GENERATE stage at the committed code (49e6050), writing only
# under ~/verification/out/<TAG>/. Follow with 05_eval_reduce.sbatch.
#
# Required --export variables:
#   VIEW   committed | pn_lora      (which generator; see 00_make_snapshot.sh)
#   ARM    trained | none | random
#   TAG    output name, e.g. J4_pn_lora, J5_none_ens
# Optional: NEVAL (default 200), EXTRA (extra eval_main flags)
#   PN_VIEW  per-direction adapters: n->p generates with VIEW's n_to_p/ adapter,
#            p->n with PN_VIEW's p_to_n/ adapter (scripts/per_direction.py). The
#            log prints one "[adapter] <direction> <- <file> sha256=…" line per
#            direction. ADAPTER_EXPECT_{NP,PN}_SHA (exported) make a wrong file fatal.
#   J4 / J5 as run on 2026-10-28:
#     sbatch --export=ALL,VIEW=committed,PN_VIEW=pn_lora,ARM=trained,TAG=J4_perdir,ADAPTER_EXPECT_NP_SHA=…,ADAPTER_EXPECT_PN_SHA=… 04_eval_generate.sbatch
#
# J4 (p->n with the generator the p->n policy was TRAINED with):
#   sbatch --export=ALL,VIEW=pn_lora,ARM=trained,TAG=J4_pn_lora 04_eval_generate.sbatch
# J5 (no-policy arm under the committed protocol: 2-ens guidance, classifier
#     groups, committed generator, same 400 images and seeds as Table 2):
#   sbatch --export=ALL,VIEW=committed,ARM=none,TAG=J5_none_ens 04_eval_generate.sbatch
#SBATCH --job-name=verif-eval-gen
#SBATCH --array=0-7
#SBATCH --partition=publicgpu
#SBATCH --account=insa
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpuh200|gpuh100|gpua100|gpul40s"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
set -euo pipefail
: "${VIEW:?set VIEW}" "${ARM:?set ARM}" "${TAG:?set TAG}"
OUT="$HOME/verification/out/$TAG"; mkdir -p "$OUT/shards"
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
echo "=== $TAG shard ${SLURM_ARRAY_TASK_ID}: code 49e6050 view=$VIEW pn_view=${PN_VIEW:-same} arm=$ARM on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv
python -u preflight_gpu.py
python -u "$ENTRY" --stage generate \
    --shard "${SLURM_ARRAY_TASK_ID}" --num-shards 8 --n-eval "${NEVAL:-200}" \
    --arm "$ARM" --shard-dir "$OUT/shards" --tag "$TAG" --base-seed 0 ${EXTRA:-}
echo "=== finished at $(date)"
