#!/bin/bash
# J14 — E4/E8 offline metrics from saved shards (no regeneration):
# in-lock-mask share of |x_c - x|, fraction changed vs VAE round trip, KID and
# bootstrap FID CIs. Arms: J4 (default, per-direction adapters), J5 (no policy),
# J5c (fixed schedule at policy mean). Inference only; runs on CPU (public partition) because the GPU queue is full with J12.
#SBATCH --job-name=verif-offline-metrics
#SBATCH --partition=public
#SBATCH --account=insa
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home2020/home/insa/arbaoui/verification/logs/%x-%j.out
#SBATCH --error=/home2020/home/insa/arbaoui/verification/logs/%x-%j.err
set -euo pipefail
OUT="$HOME/verification/out/J14_offline_metrics"; mkdir -p "$OUT"
cd "$HOME/verification/code/49e6050"
module load python/3.12.8
source venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 YOLO_OFFLINE=1 PYTHONPATH="$PWD"
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16; nproc
V="$HOME/verification/out"
python -u "$HOME/verification/scripts/offline_metrics.py" "$OUT/offline_metrics.json" \
    "$V/J4_perdir/shards:J4" "$V/J5_none_perdir/shards:J5" "$V/J5c_fixed_schedule_corrected/shards:J5c"
