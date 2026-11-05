#!/bin/bash
# J17 — PPO-only seed replicates (issue D3), both directions as array tasks.
# Each direction trains against its OWN already-trained adapter (train_ppo_only.py),
# so only the PPO seed differs from the arm's original run and the J9 shared-adapter
# caveat does not apply. Hard cutoff for the paper: 2026-11-09 12:00.
#
#   sbatch --export=ALL,SEED=2,GUIDANCE=ens    28_seed_replicate.sbatch
#   sbatch --export=ALL,SEED=1,GUIDANCE=single 28_seed_replicate.sbatch
#   sbatch --export=ALL,SEED=2,GUIDANCE=single 28_seed_replicate.sbatch
# then evaluate on J4's images with 26/27 and POLICY_DIR=out/seedrep_<G>_seed<S>.
# If a task prints PAUSED (wall-clock budget), resubmit the same task: main.py
# resumes from ppo_latest.pt (no RNG state is saved; record it).
#SBATCH --job-name=verif-J17-seedrep
#SBATCH --array=0-1
#SBATCH --partition=publicgpu
#SBATCH --account=insa
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpuh200|gpuh100|gpul40s"
#SBATCH --exclude=hpc-n983
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/home2020/home/insa/arbaoui/verification/logs/%x-%A_%a.out
#SBATCH --error=/home2020/home/insa/arbaoui/verification/logs/%x-%A_%a.err
set -euo pipefail
: "${SEED:?set SEED}" "${GUIDANCE:?set GUIDANCE (ens|single)}"
R="$HOME/Jobs/Ayoub/ppo-contrafactual"
V="$HOME/verification"
NAME="seedrep_${GUIDANCE}_seed${SEED}"
CODE="$V/code/49e6050_${NAME}"
RUN="$V/out/$NAME"; mkdir -p "$RUN"
if [ "$GUIDANCE" = ens ]; then
    NPV=committed; PNV=pn_lora
    NPS=658cdd7956a858c16a6b13c136ec01e4a61668b738826c74b51cdbf5ddc86f99
    PNS=6ae442301a803162fd66d020521c36d8aa800138acb209e27f20ec59766c9651
elif [ "$GUIDANCE" = single ]; then
    NPV=single_np; PNV=single_pn
    NPS=db36b707817b319244ed5945fc796e09907ae88476f1143149f1e7a16e3ffe69
    PNS=01d6c7e3bcb28317ce3d30c49a0e523a473e6896dc67959169627fa0a71a9853
else
    echo "GUIDANCE must be ens|single" >&2; exit 2
fi
if [ ! -d "$CODE" ]; then   # task 0 makes the code copy; task 1 waits for it
    if [ "${SLURM_ARRAY_TASK_ID}" = 0 ]; then
        cp -r "$V/code/49e6050" "$CODE.tmp"
        sed -i "s/^TRAIN_SEED = 0/TRAIN_SEED = ${SEED}/" "$CODE.tmp/config.py"
        [ "$GUIDANCE" = single ] && sed -i 's/^USE_GUIDANCE_ENSEMBLE  = True/USE_GUIDANCE_ENSEMBLE  = False/' "$CODE.tmp/config.py"
        mv "$CODE.tmp" "$CODE"
    else
        while [ ! -d "$CODE" ]; do sleep 20; done
    fi
fi
[ -f "$RUN/split_rsna_pneumonia.json" ] || cp "$R/checkpoints/split_rsna_pneumonia.json" "$RUN/"
cd "$CODE"
grep -nE '^(TRAIN_SEED|USE_GUIDANCE_ENSEMBLE) ' config.py
module load python/3.12.8; module load cuda/cuda-12.1
source venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 YOLO_OFFLINE=1
export PPO_CKPT_DIR="$RUN"
export PPO_WALLCLOCK_BUDGET_S=$((11 * 3600 + 1800))   # soft budget 11.5 h of the 12 h limit
export ADAPTER_NP_VIEW="$V/views/$NPV" ADAPTER_PN_VIEW="$V/views/$PNV"
export ADAPTER_EXPECT_NP_SHA="$NPS" ADAPTER_EXPECT_PN_SHA="$PNS"
export PYTHONPATH="$PWD:$V/scripts"
DIRECTION=$([ "${SLURM_ARRAY_TASK_ID}" = 0 ] && echo n_to_p || echo p_to_n)
export SEEDREP_DIRECTION="$DIRECTION"
echo "=== $NAME $DIRECTION on $(hostname) at $(date)"; nvidia-smi --query-gpu=name,memory.total --format=csv
python -u preflight_gpu.py
python -u "$V/scripts/train_ppo_only.py" --direction "$DIRECTION"
echo "=== finished at $(date)"
