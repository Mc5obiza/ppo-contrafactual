#!/bin/bash
# J9 / J10 — RETRAIN (LoRA + PPO, both directions) at the committed code 49e6050.
#   J9  extra seeds of the committed 2-ens config:   SEED=1 GUIDANCE=ens  (and SEED=2)
#   J10 single-classifier retrain, hardware-matched: SEED=0 GUIDANCE=single
#       (NB: run 17686489 is ALREADY config-matched to the committed run; J10
#        only adds a same-hardware / same-generator replicate. Low priority.)
#
#   sbatch --export=ALL,SEED=1,GUIDANCE=ens 09_retrain.sbatch
#   then, after BOTH array tasks finish:
#   sbatch --export=ALL,VIEW=<see below>,ARM=trained,TAG=J9_seed1 04_eval_generate.sbatch
#   (make a view for the new run: the run dir itself works as PPO_CKPT_DIR,
#    e.g. export VIEW=../out/retrain_ens_seed1 — 04 resolves views/$VIEW.)
#
# WHY BOTH TASKS ARE PINNED TO ONE GPU TYPE: each array task fine-tunes its own
# LoRA from scratch. In the committed run they ran on A100 and H100 and the
# adapters diverged from epoch 1 (logs 17694482_0/_1), so the p->n policy was
# trained against one generator and evaluated with the other. On identical
# hardware (17686489, 17700222: both H200) the final LoRA losses agree. After
# the run, CHECK: paste the per-epoch "avg L_diff" lines of the two logs; they
# must match. If they do not, evaluate p->n with its own generator (view like
# pn_lora in 00_make_snapshot.sh).
#
# Cost: LoRA 5-6 h + PPO 3-6 h per direction (committed: 8h40m A100, 11h00m
# H100). If a task prints PAUSED at the 22 h budget, resubmit the same task;
# main.py resumes from ppo_latest.pt.
#SBATCH --job-name=verif-retrain
#SBATCH --array=0-1
#SBATCH --partition=publicgpu
#SBATCH --account=insa
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpuh100&gpu80g"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
set -euo pipefail
: "${SEED:?set SEED}" "${GUIDANCE:?set GUIDANCE (ens|single)}"
R="${PPO_CF_REPO:?set PPO_CF_REPO to the ppo-contrafactual checkout}"
V="$HOME/verification"
NAME="retrain_${GUIDANCE}_seed${SEED}"
CODE="$V/code/49e6050_${NAME}"
RUN="$V/out/$NAME"; mkdir -p "$RUN"
if [ ! -d "$CODE" ]; then   # race-safe enough: task 1 waits for task 0's copy
    if [ "${SLURM_ARRAY_TASK_ID}" = 0 ]; then
        cp -r "$V/code/49e6050" "$CODE.tmp"
        sed -i "s/^TRAIN_SEED = 0/TRAIN_SEED = ${SEED}/" "$CODE.tmp/config.py"
        [ "$GUIDANCE" = single ] && sed -i 's/^USE_GUIDANCE_ENSEMBLE  = True/USE_GUIDANCE_ENSEMBLE  = False/' "$CODE.tmp/config.py"
        mv "$CODE.tmp" "$CODE"
    else
        while [ ! -d "$CODE" ]; do sleep 20; done
    fi
fi
# Same persisted split as every reported run (copied, not linked, so nothing
# can write through to the repo).
[ -f "$RUN/split_rsna_pneumonia.json" ] || cp "$R/checkpoints/split_rsna_pneumonia.json" "$RUN/"
cd "$CODE"
grep -nE '^(TRAIN_SEED|USE_GUIDANCE_ENSEMBLE) ' config.py
module load python/3.12.8; module load cuda/cuda-12.1
source venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 YOLO_OFFLINE=1
export PPO_CKPT_DIR="$RUN"
DIRECTION=$([ "${SLURM_ARRAY_TASK_ID}" = 0 ] && echo n_to_p || echo p_to_n)
echo "=== $NAME $DIRECTION on $(hostname) at $(date)"; nvidia-smi --query-gpu=name,memory.total --format=csv
python -u preflight_gpu.py
python -u main.py --direction "$DIRECTION"
echo "=== finished at $(date)"
