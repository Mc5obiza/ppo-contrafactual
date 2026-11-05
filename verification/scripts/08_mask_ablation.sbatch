#!/bin/bash
# J8 — background-lock mask dilation/feathering ablation, committed policies,
# first 50 images per direction (n_eval=50), 3 settings (one array task each):
#   0: dilation 0, feather 0.0  (lung mask only, no band)
#   1: dilation 2, feather 1.0  (committed setting; must match the committed
#      shards on these 50 images, i.e. a built-in reproduction check)
#   2: dilation 4, feather 2.0  (wider band)
# Inference only. Settles nothing currently claimed; supports the sec 3.2
# statement that the lock is soft, and tells you whether LHR / edit
# concentration / transfer are sensitive to the band width.
#
# NOTE: src/generator.py at 49e6050 binds config.BG_MASK_DILATION/FEATHER as
# import-time default arguments. mask_ablation_eval.py patches config BEFORE
# eval_main (and therefore the generator) is imported, so the override takes
# effect. Setting 1 reproducing the committed shards on these 50 images is the
# proof; if setting 0 or 2 come out byte-identical to setting 1, the override
# did not take and the results must be discarded.
#SBATCH --job-name=verif-mask-ablation
#SBATCH --array=0-2
#SBATCH --partition=publicgpu
#SBATCH --account=insa
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpuh200|gpuh100|gpua100|gpul40s"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home2020/home/insa/arbaoui/verification/logs/%x-%A_%a.out
#SBATCH --error=/home2020/home/insa/arbaoui/verification/logs/%x-%A_%a.err
set -euo pipefail
SETTINGS=("0 0.0" "2 1.0" "4 2.0")
read -r DIL FEA <<< "${SETTINGS[$SLURM_ARRAY_TASK_ID]}"
TAG="J8_mask_d${DIL}_f${FEA}"
OUT="$HOME/verification/out/$TAG"; mkdir -p "$OUT/shards"
cd "$HOME/verification/code/49e6050"
module load python/3.12.8; module load cuda/cuda-12.1
source venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 YOLO_OFFLINE=1 PYTHONPATH="$PWD"
export PPO_CKPT_DIR="$HOME/verification/views/committed"
source "$HOME/verification/scripts/adapters_committed.env"   # n->p committed, p->n pn_lora adapter
W="$HOME/verification/scripts/mask_ablation_eval.py"
nvidia-smi --query-gpu=name,memory.total --format=csv
python -u "$W" "$DIL" "$FEA" --stage generate --shard 0 --num-shards 1 --n-eval 50 \
    --arm trained --shard-dir "$OUT/shards" --tag "$TAG" --base-seed 0
python -u "$W" "$DIL" "$FEA" --stage reduce --num-shards 1 --n-eval 50 \
    --arm trained --shard-dir "$OUT/shards" --out-dir "$OUT" --tag "$TAG" --base-seed 0
cd "$HOME/verification/code/HEAD"
PPO_DOMAIN=rsna_pneumonia python -u arm_transfer.py "$OUT/shards" --bootstrap --json "$OUT/transfer.json"
