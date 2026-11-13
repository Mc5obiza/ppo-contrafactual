#!/bin/bash
# J10 — Vanilla CycleGAN baseline (checkpoints_cyclegan_baseline, 40 epochs,
# seed 0, train fold) under the committed protocol: the committed run's 400
# test-fold images (Xn/Xp from checkpoints/eval_shards_ens_default), committed
# metric code (49e6050), committed transfer tool (HEAD, reproduced Table 3 in J2).
# Writes only under ~/verification/out/$TAG/. Inference only, ~20-30 min.
#
#   sbatch 12_cyclegan_eval.sbatch
#
# Outputs
#   eval_metrics.json            CV, FID, IP, EditConcentration(+frac_changed), LHR p->n, CTR, SCP
#   cyclegan_eval_manifest.json  generator/source sha256, idx sha, input + group checks
#   shards/cf_shard_000.pt       committed schema (num_shards=1)
#   transfer_den_ens2.json       PRIMARY: denominator = rsna+chex ensemble shift
#                                (the classifier that selected Xn/Xp and scores CV)
#   transfer_den_rsna.json       sensitivity: denominator = rsna alone
#   paired_lhr_vs_committed.txt  paired per-image LHR, CycleGAN vs committed arm
#   per_image_lhr.json           CycleGAN and committed arm
#   transfer_rule.json           shift CIs + ratio-or-undefined, both denominators, both arms
#SBATCH --job-name=verif-cyclegan
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
R="${PPO_CF_REPO:?set PPO_CF_REPO to the ppo-contrafactual checkout}"
TAG="${TAG:-J10_cyclegan}"
OUT="$HOME/verification/out/$TAG"; mkdir -p "$OUT"
COMMITTED="$R/checkpoints/eval_shards_ens_default"
CG_CKPT="$R/checkpoints_cyclegan_baseline"
CG_SRC="$HOME/verification/code/HEAD/src/cyclegan.py"   # byte-identical to $R/src/cyclegan.py (checked below)
ENS2="densenet121-res224-rsna,densenet121-res224-chex"

cmp "$CG_SRC" "$R/src/cyclegan.py"
module load python/3.12.8; module load cuda/cuda-12.1
cd "$HOME/verification/code/49e6050"
source venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 YOLO_OFFLINE=1
export PPO_CKPT_DIR="$HOME/verification/views/committed"   # only for the split file
export PYTHONPATH="$PWD"
echo "=== $TAG on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv
python -u preflight_gpu.py

# 1. generate + score (committed metric code)
python -u "$HOME/verification/scripts/cyclegan_eval.py" \
    --committed-shards "$COMMITTED" --cyclegan-ckpt "$CG_CKPT" \
    --cyclegan-src "$CG_SRC" --out "$OUT"

# 2. paired LHR on the shared annotated p->n images (CPU arithmetic)
python -u paired_arm_test.py "$COMMITTED" "$OUT/shards" 2>&1 | tee "$OUT/paired_lhr_vs_committed.txt"
python -u "$HOME/verification/scripts/per_image_lhr.py" "$OUT/per_image_lhr.json" "$OUT/shards" "$COMMITTED"

# 3. held-out transfer. --guidance is REQUIRED: the shard stamp names no guidance
#    model (a CycleGAN has none), and without it arm_transfer falls back to rsna.
cd "$HOME/verification/code/HEAD"
unset PPO_CKPT_DIR            # same environment as 02_transfer_bootstrap (J2, reproduced Table 3)
# PYTHONPATH still points at 49e6050 from step 1; transfer_rule.py lives in
# scripts/, so its `import arm_transfer` would load the 49e6050 copy (no _builders).
export PYTHONPATH="$PWD"
export PPO_DOMAIN=rsna_pneumonia
python -u arm_transfer.py "$OUT/shards" --bootstrap --guidance "$ENS2" \
    --json "$OUT/transfer_den_ens2.json"
python -u arm_transfer.py "$OUT/shards" --bootstrap --guidance densenet121-res224-rsna \
    --json "$OUT/transfer_den_rsna.json"
# 4. definedness rule (denominator-shift CI must exclude 0), both denominators,
#    CycleGAN and committed shards (the committed run must reproduce Table 3).
python -u "$HOME/verification/scripts/transfer_rule.py" "$OUT/transfer_rule.json" "$OUT/shards" "$COMMITTED"
echo "=== finished at $(date)"
