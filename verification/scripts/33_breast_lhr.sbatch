#!/bin/bash
# J22 — second review item 6 (exploratory): LHR in BOTH directions on the CBIS-DDSM
# counterfactual shards from the test-cancer branch, with in-breast, VAE-floor and
# mismatched-ROI nulls. CPU, no regeneration. Runs from the repository root (breast domain code).
#SBATCH --job-name=verif-J22-breast-lhr
#SBATCH --partition=public
#SBATCH --account=insa
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home2020/home/insa/arbaoui/verification/logs/%x-%j.out
#SBATCH --error=/home2020/home/insa/arbaoui/verification/logs/%x-%j.err
set -euo pipefail
OUT="$HOME/verification/out/J22_breast_lhr"; mkdir -p "$OUT"
cd "$HOME/Jobs/Ayoub/ppo-contrafactual"
module load python/3.12.8
source venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
export PYTHONPATH="$PWD:$HOME/verification/scripts"
echo "=== J22 on $(hostname) at $(date); repo $(git rev-parse --short HEAD)"
python -u "$HOME/verification/scripts/breast_lhr.py" "$OUT"
sha256sum "$HOME/verification/scripts/breast_lhr.py" "$HOME/verification/scripts/breast_segmentation_testcancer.py" > "$OUT/script.sha256"
echo "=== finished at $(date)"
