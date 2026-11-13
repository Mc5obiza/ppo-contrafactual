#!/bin/bash
# J2 — held-out transfer ratios + 10k percentile bootstrap CIs for EVERY arm,
# persisted as JSON. Inference only: the three validator classifiers score the
# saved shards (<= 800 images per arm). No generation.
#
# Uses code/HEAD because --bootstrap/--json exist only there (not in any commit
# a run was stamped with). The point estimates for 2-ens and single must
# reproduce logs/arm_transfer_ens_default_vs_old.out exactly; if they do not,
# the HEAD tool has drifted and its CIs must not be used.
#
# Paper claims this settles: sec 5.3 CIs [0.048,0.114] [0.007,0.084]
# [0.076,0.126] [0.155,0.216]; Table 4 transfer for 3-ens (0.350/0.276*),
# sparsity (0.311/0.212), Grad-CAM (0.161/0.103); "pc 0.462 -> 0.623,
# non-overlapping CIs"; sparsity "non-overlapping CIs on two of three
# validators"; Grad-CAM "falls across all three validators".
#SBATCH --job-name=verif-transfer-boot
#SBATCH --partition=publicgpu
#SBATCH --account=insa
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpuh100|gpua100|gpul40s|gpuh200"
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
R="${PPO_CF_REPO:?set PPO_CF_REPO to the ppo-contrafactual checkout}"
OUT="$HOME/verification/out/${OUTNAME:-transfer}"; mkdir -p "$OUT"
cd "$HOME/verification/code/HEAD"
module load python/3.12.8; module load cuda/cuda-12.1
source venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 YOLO_OFFLINE=1 PPO_DOMAIN=rsna_pneumonia
nvidia-smi --query-gpu=name,memory.total --format=csv

run () {  # $1 tag, $2 shard dir, rest: extra flags
    local tag="$1" dir="$2"; shift 2
    echo "=== $tag ($dir)"
    python -u arm_transfer.py "$dir" --bootstrap --json "$OUT/$tag.json" "$@"
}
run "${ENS2_TAG:-ens2_committed}" "${ENS2:-$R/checkpoints/eval_shards_ens_default}"
[ "${ONLY_ENS2:-0}" = 1 ] && exit 0   # J4 re-run: only the 2-ens entry changes
run single         "$R/checkpoints/eval_shards"
run sparsity       "$R/checkpoints_sparsity/eval_shards"
run gradcam_fix    "$R/checkpoints_gradcam_fix/eval_shards"
run ens3_all3      "$R/checkpoints_ens3/eval_shards"
run ens3_nih_pc    "$R/checkpoints_ens3/eval_shards" \
    --validators densenet121-res224-nih,densenet121-res224-pc
run none_rsna      "$R/checkpoints/eval_shards_none"
run random_rsna    "$R/checkpoints/eval_shards_random"
