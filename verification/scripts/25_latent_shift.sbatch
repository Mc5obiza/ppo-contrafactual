#!/bin/bash
# J15 — Latent Shift baseline (Gifsplanation, Cohen et al. MIDL 2021; torchxrayvision
# ResNetAE 101-elastic) under the J4 protocol: J4's 400 test-fold images
# (Xn/Xp from out/J4_perdir/shards; same indices as E_orig), rsna+chex ensemble as
# the guidance AND CV classifier, J4 metric code (49e6050), J2 transfer tool (HEAD).
# Issue D1. Writes only under ~/verification/out/$TAG/. Inference only.
#
#   sbatch --export=ALL,TAG=J15_latent_shift_smoke,LIMIT=8 25_latent_shift.sbatch   # smoke
#   sbatch 25_latent_shift.sbatch                                                     # full
#
# Outputs
#   eval_metrics.json                CV, FID, IP, EditConcentration(+frac_changed), LHR p->n, CTR, SCP
#   latent_shift_eval_manifest.json  AE sha256, lambda grid, search stats, idx sha, input + group checks
#   latent_shift_aux.pt              D(E(x)), chosen lambda, score trajectories (supplement)
#   shards/cf_shard_000.pt           committed schema (num_shards=1)
#   paired_lhr_vs_J4.txt             paired per-image LHR, Latent Shift vs default (J4)
#   per_image_lhr.json               Latent Shift and J4
#   transfer_den_ens2.json           PRIMARY: denominator = rsna+chex ensemble shift
#   transfer_den_rsna.json           sensitivity: denominator = rsna alone
#   transfer_rule.json               shift CIs + ratio-or-undefined, both denominators, both arms
#SBATCH --job-name=verif-J15-latent-shift
#SBATCH --partition=publicgpu
#SBATCH --account=insa
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpuh200|gpuh100|gpua100|gpul40s"
#SBATCH --exclude=hpc-n983
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
TAG="${TAG:-J15_latent_shift}"
LIMIT="${LIMIT:-0}"
OUT="$HOME/verification/out/$TAG"; mkdir -p "$OUT"
J4="$HOME/verification/out/J4_perdir/shards"
ENS2="densenet121-res224-rsna,densenet121-res224-chex"
AE_W="$HOME/.torchxrayvision/models_data/nihpcrsnamimic_ch-resnet101-2-ae-test2-elastic-e250.pt"
echo "877ded430383a2bbce756737b316d76de42defed3079eebf2cce305f7be34df9  $AE_W" | sha256sum -c -

module load python/3.12.8; module load cuda/cuda-12.1
cd "$HOME/verification/code/49e6050"
source venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 YOLO_OFFLINE=1
export PPO_CKPT_DIR="$HOME/verification/views/committed"   # only for the split file
export PYTHONPATH="$PWD:$HOME/verification/scripts"
echo "=== $TAG (LIMIT=$LIMIT) on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv
python -u preflight_gpu.py

# 1. generate + score (J4 metric code)
python -u "$HOME/verification/scripts/latent_shift_eval.py" \
    --committed-shards "$J4" --out "$OUT" --limit "$LIMIT"

# 2. paired LHR on the shared annotated p->n images (CPU arithmetic)
python -u paired_arm_test.py "$J4" "$OUT/shards" 2>&1 | tee "$OUT/paired_lhr_vs_J4.txt"
python -u "$HOME/verification/scripts/per_image_lhr.py" "$OUT/per_image_lhr.json" "$OUT/shards" "$J4"

# 3. held-out transfer. --guidance is REQUIRED (the shard stamp names no guidance model).
cd "$HOME/verification/code/HEAD"
unset PPO_CKPT_DIR
export PYTHONPATH="$PWD"
export PPO_DOMAIN=rsna_pneumonia
python -u arm_transfer.py "$OUT/shards" --bootstrap --guidance "$ENS2" \
    --json "$OUT/transfer_den_ens2.json"
python -u arm_transfer.py "$OUT/shards" --bootstrap --guidance densenet121-res224-rsna \
    --json "$OUT/transfer_den_rsna.json"
# 4. definedness rule, both denominators, Latent Shift and J4 (J4 must reproduce Table 3).
python -u "$HOME/verification/scripts/transfer_rule.py" "$OUT/transfer_rule.json" "$OUT/shards" "$J4"
echo "=== finished at $(date)"
