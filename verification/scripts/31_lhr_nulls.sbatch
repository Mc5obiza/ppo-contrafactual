#!/bin/bash
# J19 — second review, item 1: LHR under lung-restricted, editable-region and
# mismatched-box nulls; LHR of |D(E(x)) - x| and of |x_c - D(E(x))|; uniform
# in-lung / in-mask wash references. Offline from saved shards, CPU only.
#   ARMS="dir:tag dir:tag ..." (default: every arm with shards except J16)
#SBATCH --job-name=verif-J19-lhrnull
#SBATCH --partition=public
#SBATCH --account=insa
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
TAG="${TAG:-J19_lhr_nulls}"
OUT="$HOME/verification/out/$TAG"; mkdir -p "$OUT"
cd "$HOME/verification/code/49e6050"
module load python/3.12.8
source venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 YOLO_OFFLINE=1 PYTHONPATH="$PWD:$HOME/verification/scripts"
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=
V="$HOME/verification/out"; C="${PPO_CF_REPO:?set PPO_CF_REPO to the ppo-contrafactual checkout}"
DEFAULT_ARMS="$V/J4_perdir/shards:J4 $V/J5_none_perdir/shards:J5_none $V/J5c_fixed_schedule_corrected/shards:J5c_fixed
 $V/J12_grid_fixed_schedule/shards:J12_fixed_tuned $V/J9_seed1/shards:J9_seed1 $V/J15_latent_shift/shards:J15_latent_shift
 $V/J10_cyclegan/shards:J10_cyclegan $C/checkpoints/eval_shards_ens_default:evalA_ens_default
 $C/checkpoints/eval_shards:v1_single $C/checkpoints/eval_shards_random:v1_random $C/checkpoints_ens3/eval_shards:ens3
 $C/checkpoints_sparsity/eval_shards:sparsity $C/checkpoints_gradcam_fix/eval_shards:gradcam"
python -u "$HOME/verification/scripts/lhr_nulls.py" "$OUT" ${ARMS:-$DEFAULT_ARMS}
sha256sum "$HOME/verification/scripts/lhr_nulls.py" > "$OUT/script.sha256"
