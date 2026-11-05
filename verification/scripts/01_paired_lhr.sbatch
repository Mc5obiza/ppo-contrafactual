#!/bin/bash
# J1 — per-image LHR + every paired LHR test the paper quotes. CPU ONLY.
# Recomputes from saved shards; no generation, no model forward pass.
# Prereq: bash ~/verification/scripts/00_make_snapshot.sh
#
# Paper claims this settles (sec/5_experiments.tex):
#   single vs 2-ens        d=0.482, p=7.1e-5, n=76
#   2-ens vs sparsity      +0.275 [+0.188,+0.363], t=6.229, p=4.7e-9, d=0.512, n=148
#   2-ens vs Grad-CAM      +1.037 [+0.900,+1.178], t=14.660, p=1.5e-30, d=1.205, 141/148
#   3-ens vs 2-ens         n=12, CI [-0.298,+0.482]
#   policy-free sweep      d=0.221/0.223/0.151 and 95% CIs (deltas already verified)
#   trained vs random      (sanity: must reproduce logs/paired_arm_test_trained_random.out)
#SBATCH --job-name=verif-paired-lhr
#SBATCH --partition=public
#SBATCH --account=insa
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home2020/home/insa/arbaoui/verification/logs/%x-%j.out
#SBATCH --error=/home2020/home/insa/arbaoui/verification/logs/%x-%j.err
set -euo pipefail
R="$HOME/Jobs/Ayoub/ppo-contrafactual"
OUT="$HOME/verification/out/${OUTNAME:-paired_lhr}"; mkdir -p "$OUT"
cd "$HOME/verification/code/49e6050"
module load python/3.12.8
source venv/bin/activate
export PYTHONPATH="$PWD"

S1="$R/checkpoints/eval_shards"                 # single-classifier (17686489)
S2="${S2:-$R/checkpoints/eval_shards_ens_default}"     # committed 2-ens (17694482); J4 re-run: S2=~/verification/out/J4_perdir/shards
S3="$R/checkpoints_ens3/eval_shards"
SP="$R/checkpoints_sparsity/eval_shards"
GC="$R/checkpoints_gradcam_fix/eval_shards"
NO="$R/checkpoints/eval_shards_none"
RA="$R/checkpoints/eval_shards_random"
C="$R/checkpoints/eval_shards_none"

python -u "$HOME/verification/scripts/per_image_lhr.py" "$OUT/per_image_lhr.json" \
    "$S1" "$S2" "$S3" "$SP" "$GC" "$NO" "$RA" \
    "${C}_cm0" "${C}_cm1" "${C}_cm0s1" "${C}_cm1s1" "${C}_cm0s2" "${C}_cm1s2" \
    "${C}_cm2" "${C}_cm3" "${C}_cm4"

pair () { echo "=== $1  vs  $2"; python -u paired_arm_test.py "$1" "$2"; }
{
  pair "$S2" "$S1"
  pair "$S2" "$SP"
  pair "$S2" "$GC"
  pair "$S3" "$S2"
  pair "${C}_cm1"   "${C}_cm0"
  pair "${C}_cm1s1" "${C}_cm0s1"
  pair "${C}_cm1s2" "${C}_cm0s2"
  pair "$S1" "$RA"
  pair "$S1" "$NO"
  pair "$S2" "$NO"
} 2>&1 | tee "$OUT/paired_tests.txt"
