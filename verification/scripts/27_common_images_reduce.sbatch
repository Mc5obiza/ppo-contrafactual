#!/bin/bash
# J16 — REDUCE stage for 26_common_images_generate.sbatch. Same --export variables.
# Then: transfer (HEAD tool; denominator from the shards' stamp: rsna for the
# single-classifier arm), paired LHR vs J4 on all shared annotated images, and a
# per-image LHR dump (J4, this arm, and the original single-classifier shards).
#   sbatch --dependency=afterok:<gen> --export=ALL,VIEW=single_np,PN_VIEW=single_pn,TAG=J16_single_common,... 27_common_images_reduce.sbatch
#SBATCH --job-name=verif-J16-common-reduce
#SBATCH --partition=publicgpu
#SBATCH --account=insa
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpuh200|gpuh100|gpua100|gpul40s"
#SBATCH --exclude=hpc-n983
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/home2020/home/insa/arbaoui/verification/logs/%x-%j.out
#SBATCH --error=/home2020/home/insa/arbaoui/verification/logs/%x-%j.err
set -euo pipefail
: "${VIEW:?set VIEW}" "${PN_VIEW:?set PN_VIEW}" "${TAG:?set TAG}"
V="$HOME/verification"; R="$HOME/Jobs/Ayoub/ppo-contrafactual"
CODE="${CODE:-$V/code/49e6050_single}"
OUT="$V/out/$TAG"
J4="$V/out/J4_perdir/shards"
export COMMON_IMAGES_FROM="${SRC:-$J4}"
cd "$CODE"
module load python/3.12.8; module load cuda/cuda-12.1
source venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 YOLO_OFFLINE=1
export PPO_CKPT_DIR="${POLICY_DIR:-$V/views/$VIEW}"   # policies; adapters come from the views below
export ADAPTER_NP_VIEW="$V/views/$VIEW" ADAPTER_PN_VIEW="$V/views/$PN_VIEW"
export ADAPTER_EXPECT_NP_SHA ADAPTER_EXPECT_PN_SHA
export PYTHONPATH="$PWD:$V/scripts"
echo "=== $TAG reduce on $(hostname) at $(date)"; nvidia-smi --query-gpu=name --format=csv
python -u "$V/scripts/eval_common_images.py" --stage reduce --num-shards 8 --n-eval 200 \
    --arm "${ARM:-trained}" --shard-dir "$OUT/shards" --out-dir "$OUT" --tag "$TAG" --base-seed 0
# paired LHR vs the default (J4) on the SAME images; per-image dump incl. the original single-arm shards
python -u paired_arm_test.py "$J4" "$OUT/shards" 2>&1 | tee "$OUT/paired_lhr_vs_J4.txt"
python -u "$V/scripts/per_image_lhr.py" "$OUT/per_image_lhr.json" "$OUT/shards" "$J4" "$R/checkpoints/eval_shards"
# transfer: denominator read from the shards' own stamp (USE_GUIDANCE_ENSEMBLE=False -> rsna)
cd "$V/code/HEAD"
unset PPO_CKPT_DIR
export PYTHONPATH="$PWD"
PPO_DOMAIN=rsna_pneumonia python -u arm_transfer.py "$OUT/shards" --bootstrap --json "$OUT/transfer.json"
echo "=== finished at $(date)"
