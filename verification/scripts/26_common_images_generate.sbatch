#!/bin/bash
# J16 — GENERATE stage on a FIXED image set (eval_common_images.py): J4's 400
# test-fold images and per-image seeds, instead of a rescan with the arm's own
# guidance classifier. Issue D3. Follow with 27_common_images_reduce.sbatch.
#
# Required --export variables:
#   VIEW / PN_VIEW   per-direction views (n->p adapter from VIEW, p->n from PN_VIEW)
#   TAG              output name
#   ADAPTER_EXPECT_NP_SHA / ADAPTER_EXPECT_PN_SHA   make a wrong adapter fatal
# Optional: POLICY_DIR (dir holding n_to_p/ and p_to_n/ ppo_policy.pt; default views/$VIEW),
#           CODE (default code/49e6050_single = 49e6050 with USE_GUIDANCE_ENSEMBLE=False),
#           ARM (default trained), SRC (default out/J4_perdir/shards)
#
# Single-classifier policy (archive_pre_ensemble_default_20260812), own adapters:
#   sbatch --export=ALL,VIEW=single_np,PN_VIEW=single_pn,TAG=J16_single_common,\
#     ADAPTER_EXPECT_NP_SHA=db36b707817b319244ed5945fc796e09907ae88476f1143149f1e7a16e3ffe69,\
#     ADAPTER_EXPECT_PN_SHA=01d6c7e3bcb28317ce3d30c49a0e523a473e6896dc67959169627fa0a71a9853 \
#     26_common_images_generate.sbatch
#SBATCH --job-name=verif-J16-common-gen
#SBATCH --array=0-7
#SBATCH --partition=publicgpu
#SBATCH --account=insa
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpuh200|gpuh100|gpua100|gpul40s"
#SBATCH --exclude=hpc-n983
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home2020/home/insa/arbaoui/verification/logs/%x-%A_%a.out
#SBATCH --error=/home2020/home/insa/arbaoui/verification/logs/%x-%A_%a.err
set -euo pipefail
: "${VIEW:?set VIEW}" "${PN_VIEW:?set PN_VIEW}" "${TAG:?set TAG}"
: "${ADAPTER_EXPECT_NP_SHA:?}" "${ADAPTER_EXPECT_PN_SHA:?}"
V="$HOME/verification"
CODE="${CODE:-$V/code/49e6050_single}"
OUT="$V/out/$TAG"; mkdir -p "$OUT/shards"
export COMMON_IMAGES_FROM="${SRC:-$V/out/J4_perdir/shards}"
cd "$CODE"
module load python/3.12.8; module load cuda/cuda-12.1
source venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 YOLO_OFFLINE=1
export PPO_CKPT_DIR="${POLICY_DIR:-$V/views/$VIEW}"   # policies; adapters come from the views below
export ADAPTER_NP_VIEW="$V/views/$VIEW" ADAPTER_PN_VIEW="$V/views/$PN_VIEW"
export ADAPTER_EXPECT_NP_SHA ADAPTER_EXPECT_PN_SHA
export PYTHONPATH="$PWD:$V/scripts"
echo "=== $TAG shard ${SLURM_ARRAY_TASK_ID}: code $CODE view=$VIEW pn_view=$PN_VIEW groups=$COMMON_IMAGES_FROM on $(hostname) at $(date)"
grep -nE '^(TRAIN_SEED|USE_GUIDANCE_ENSEMBLE) ' config.py
nvidia-smi --query-gpu=name,memory.total --format=csv
python -u preflight_gpu.py
python -u "$V/scripts/eval_common_images.py" --stage generate \
    --shard "${SLURM_ARRAY_TASK_ID}" --num-shards 8 --n-eval 200 \
    --arm "${ARM:-trained}" --shard-dir "$OUT/shards" --tag "$TAG" --base-seed 0
echo "=== finished at $(date)"
