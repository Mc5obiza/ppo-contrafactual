#!/usr/bin/env bash
#SBATCH --job-name=ppo-eval-cf
#SBATCH --array=0-7
#SBATCH --partition=publicgpu
#SBATCH --account=insa
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpuh200|gpuh100|gpua100|gpul40s|gpua40|gpurtx6000"
#SBATCH --exclude=hpc-n983
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=melkiayoub724@gmail.com
#
# STAGE 1 of 2 — generate counterfactuals. Each array task produces one shard
# of the Xn/Xp counterfactual set and writes checkpoints/eval_shards/
# cf_shard_NNN.pt. When all 8 have finished, run run_eval_reduce.slurm to turn
# them into eval_metrics.json and the diff plots.
#
#   sbatch run_eval_cf.slurm
#   sbatch --dependency=afterok:<jobid> run_eval_reduce.slurm
#
# WHY AN ARRAY AND NOT ONE JOB. The previous single-process eval
# (run_eval_ppo.slurm, job 17665348) was cancelled at its 12h walltime having
# produced nothing. It managed ~4.4 min/image and the suite needs 400
# counterfactuals (Xn=200 + Xp=200), so generation alone is ~29h. `sinfo -p
# publicgpu` reports TIMELIMIT 1-00:00:00 — 24h is a HARD partition cap, so no
# --time value could ever have made that fit. Generation is independent per
# image, so it is sharded instead.
#
# WALLTIME. 8 shards x 25+25 images = 50 images each. At the measured 4.4
# min/image that is ~3.7h, so 8h is ~2x headroom for a slow or contended node.
# Nothing here makes a single image cheaper — the guidance block's two backward
# passes were examined and are NOT safely fusable in fp16 (see the long comment
# in generator.generate_counterfactual) — so this walltime rests on the
# per-image cost that job 17665348 actually measured, not on a hoped-for
# speedup. If a task does overrun it costs one shard, not the run, and can be
# resubmitted alone:
#     sbatch --array=<missing shard ids> run_eval_cf.slurm
#
# If shards come back slower than ~6h, raise --array to 0-15 and NUM_SHARDS to
# 16 rather than raising --time; more, smaller tasks schedule sooner on a busy
# partition and lose less work when one dies.
#
# PREREQUISITE: run_ppo.slurm must have completed and written
#   <config.CKPT_DIR>/n_to_p/ppo_policy.pt
#   <config.CKPT_DIR>/p_to_n/ppo_policy.pt
# The check below fails fast with the exact command to finish a paused one.
#
# GPU CONSTRAINT — must match run_ppo.slurm, and for the same two reasons.
# See that file's header for the full rationale; in short:
#   * P100 nodes have no fp16 tensor cores and are ~3x slower here.
#   * hpc-n983 is an RTX PRO 6000 *Blackwell* (sm_120) that advertises the same
#     `gpurtx6000` feature as the Turing Quadro RTX 6000 nodes. venv torch is
#     2.5.1+cu121, whose kernels stop at sm_90, so every CUDA op there dies
#     with "no kernel image is available for execution on the device" — this
#     killed training job 17662452_0 five minutes in. SLURM constraints have no
#     NOT operator and the feature tag cannot distinguish the two cards, so the
#     node has to be excluded by name.
# preflight_gpu.py below is the backstop if a new node ever joins a tag.
#
# SHARDS MUST AGREE. Every task independently rescans the dataset for Xn/Xp
# (deterministic — see metrics.scan_groups) and seeds each image's diffusion
# noise from its GLOBAL index, so the concatenated shards are byte-identical to
# what one process would have produced. Each shard also records a config stamp;
# the reduce stage refuses to pool shards whose stamps disagree. Do NOT edit
# config.py between submitting this array and the reduce job.
#
# Compute nodes have NO internet access, so this script only ever loads the
# venv and pretrained weights that setup_env.sh already downloaded on the
# login node. Run setup_env.sh first — this job will fail fast otherwise.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}"
mkdir -p logs

module load python/3.12.8
module load cuda/13.0

if [ ! -d venv ]; then
    echo "venv/ not found — run 'bash setup_env.sh' on the login node first." >&2
    exit 1
fi
source venv/bin/activate

# Force offline mode: fail loudly instead of silently hanging on a network
# call that can never succeed from a compute node.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export YOLO_OFFLINE=1

# Keep this in sync with --array above; it is passed to eval_main.py so the
# shards record how many of them there are meant to be, which is what lets the
# reduce stage detect a missing one instead of quietly averaging over fewer.
NUM_SHARDS=8

# Fail before loading ~2GB of model weights if training never ran.
#
# ASK CONFIG FOR THE DIRECTORY, never hardcode "checkpoints". CKPT_DIR is
# per-domain ("checkpoints" only for rsna_pneumonia, "checkpoints/<domain>"
# for everything else, and PPO_CKPT_DIR overrides both) — see the long comment
# in config.py. This guard used to hardcode checkpoints/<direction>/ and so
# aborted job 17678229 with "training never ran for this direction" while the
# cbis_ddsm_breast policies sat in checkpoints/cbis_ddsm_breast/<direction>/.
CKPT_DIR=$(python -c 'import config; print(config.CKPT_DIR)')
echo "checkpoint dir: ${CKPT_DIR}"
#
# ppo_policy.pt is written ONLY when a direction finishes all N_EPISODES. A
# direction that stopped on the wall-clock budget leaves ppo_latest.pt (the
# resumable state) and no ppo_policy.pt — deliberately, because the two are
# indistinguishable once loaded and eval would silently report a half-trained
# policy as a finished one. So distinguish "never started" from "paused" here
# and name the exact command that finishes it.
for i in 0 1; do
    direction=$([ "$i" = 0 ] && echo n_to_p || echo p_to_n)
    if [ ! -f "${CKPT_DIR}/${direction}/ppo_policy.pt" ]; then
        if [ -f "${CKPT_DIR}/${direction}/ppo_latest.pt" ]; then
            echo "${direction} is PAUSED, not finished: ${CKPT_DIR}/${direction}/ppo_latest.pt" >&2
            echo "exists but ppo_policy.pt does not. Resume it with:" >&2
            echo "    sbatch --array=${i} run_ppo.slurm" >&2
        else
            echo "${CKPT_DIR}/${direction}/ppo_policy.pt not found and no ppo_latest.pt" >&2
            echo "either — training never ran for this direction. Start it with:" >&2
            echo "    sbatch --array=${i} run_ppo.slurm" >&2
        fi
        exit 1
    fi
done

echo "=== Eval shard ${SLURM_ARRAY_TASK_ID}/${NUM_SHARDS} started on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

# Same hardware check as run_ppo.slurm — see preflight_gpu.py.
python -u preflight_gpu.py

# Unbuffered so the log is useful while the job is still running — generation
# prints a per-image progress line with a running min/img and ETA, which is the
# only way to tell a slow node from a hung one.
python -u eval_main.py \
    --stage generate \
    --shard "${SLURM_ARRAY_TASK_ID}" \
    --num-shards "${NUM_SHARDS}"

echo "=== Eval shard ${SLURM_ARRAY_TASK_ID} finished at $(date) ==="
