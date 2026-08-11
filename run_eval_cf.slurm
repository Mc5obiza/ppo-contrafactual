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
#   checkpoints/n_to_p/ppo_policy.pt
#   checkpoints/p_to_n/ppo_policy.pt
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
# torch is built for cu121 (torch.version.cuda == "12.1") and the driver is
# CUDA 12.6. cuda/13.0 (the module system default) prepends its lib64 to
# LD_LIBRARY_PATH ahead of torch's own RUNPATH and only fails to break things
# because CUDA 13 sonames are .so.13 while torch links .so.12. Load the
# matching toolkit instead of relying on that.
module load cuda/cuda-12.1

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

# Checkpoints live under a PER-DOMAIN directory, not under checkpoints/.
# config.CKPT_DIR resolves to "checkpoints" for rsna_pneumonia and to
# "checkpoints/<domain>" for every other domain (config.py, CKPT_DIR). Ask
# config rather than hardcoding, or this guard reports "training never ran"
# for a domain whose policies are sitting right there under another name.
CKPT_DIR=$(python -c 'import config; print(config.CKPT_DIR)')
echo "[slurm] domain checkpoint dir: ${CKPT_DIR}"

# ── control arm ───────────────────────────────────────────────────────────────
# EVAL_ARM selects which system generates the counterfactuals:
#   trained (default) — the learned policies
#   none              — no policy; gradient guidance on the generator's fixed
#                       default schedule. THE control: it says whether PPO
#                       contributes anything at all.
#   random            — an untrained policy at the same action bounds.
#
# Each arm gets its own shard directory. eval_main.py refuses to run a non-
# trained arm into the default one, because the reduce stage pools by glob and
# the config stamp cannot tell arms apart (--arm is a CLI flag, not config), so
# sharing a directory would report a blend of two systems as one result.
#
#   EVAL_ARM=none sbatch run_eval_cf.slurm
#   sbatch --dependency=afterok:<jobid> --export=ALL,EVAL_ARM=none run_eval_reduce.slurm
EVAL_ARM="${EVAL_ARM:-trained}"
if [ "${EVAL_ARM}" = "trained" ]; then
    SHARD_DIR="${CKPT_DIR}/eval_shards"
else
    SHARD_DIR="${CKPT_DIR}/eval_shards_${EVAL_ARM}"
fi

# ── countermeasure arm (Phase 3a) ─────────────────────────────────────────────
# EVAL_CM selects an anti-adversarial countermeasure, and EVAL_TAG names the
# directory it writes to. Both act on the GENERATION path, so they are run with
# EVAL_ARM=none: Phase 2 showed the PPO policy contributes nothing measurable,
# and the no-policy arm is ~11x cheaper (0.02 vs 0.23 min/img).
#
#   EVAL_ARM=none EVAL_CM=ensemble EVAL_TAG=cm1 sbatch run_eval_cf.slurm
#   EVAL_ARM=none EVAL_CM=gradcam  EVAL_TAG=cm2 sbatch run_eval_cf.slurm
#   EVAL_ARM=none EVAL_CM=both     EVAL_TAG=cm3 sbatch run_eval_cf.slurm
#
# The flags go on the command line rather than into config.py because the header
# above forbids editing config.py between this array and its reduce job.
EVAL_CM="${EVAL_CM:-}"
EVAL_TAG="${EVAL_TAG:-}"
# Ground-truth Xn/Xp groups. Required for the countermeasure arms: ARM 1 changes
# f(.), and f-thresholded groups would hand each arm a DIFFERENT image set,
# quietly destroying the paired comparison the whole sweep depends on.
EVAL_GROUP_SOURCE="${EVAL_GROUP_SOURCE:-}"
CM_FLAGS=""
case "${EVAL_CM}" in
    "")        ;;
    ensemble)  CM_FLAGS="--guidance-ensemble" ;;
    gradcam)   CM_FLAGS="--gradcam-mask" ;;
    both)      CM_FLAGS="--guidance-ensemble --gradcam-mask" ;;
    *) echo "Unknown EVAL_CM='${EVAL_CM}' — expected ensemble|gradcam|both." >&2
       exit 1 ;;
esac
if [ -n "${EVAL_CM}" ] && [ -z "${EVAL_TAG}" ]; then
    echo "EVAL_CM=${EVAL_CM} requires EVAL_TAG (e.g. EVAL_TAG=cm1) so this arm" >&2
    echo "does not write into the baseline's shard directory." >&2
    exit 1
fi
if [ -n "${EVAL_TAG}" ]; then
    SHARD_DIR="${SHARD_DIR}_${EVAL_TAG}"
fi
echo "[slurm] arm=${EVAL_ARM}  cm=${EVAL_CM:-none}  tag=${EVAL_TAG:-none}  shard_dir=${SHARD_DIR}"

# Record commit + dirty state + the config values that drive behaviour. This is
# the enforceable version of the "Do NOT edit config.py between submitting this
# array and the reduce job" note above: it cannot stop an edit, but it makes one
# visible afterwards instead of silently mixing two configs into one result.
bash record_run_manifest.sh "${CKPT_DIR}"

# Fail before loading ~2GB of model weights if training never ran.
#
# ppo_policy.pt is written ONLY when a direction finishes all N_EPISODES. A
# direction that stopped on the wall-clock budget leaves ppo_latest.pt (the
# resumable state) and no ppo_policy.pt — deliberately, because the two are
# indistinguishable once loaded and eval would silently report a half-trained
# policy as a finished one. So distinguish "never started" from "paused" here
# and name the exact command that finishes it.
# Only the trained arm needs policies on disk. The `none` arm has no policy by
# definition and `random` builds one at init, so requiring a checkpoint there
# would block the two arms that exist precisely to be run without one.
for i in 0 1; do
    [ "${EVAL_ARM}" = "trained" ] || break
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
    --num-shards "${NUM_SHARDS}" \
    --arm "${EVAL_ARM}" \
    --shard-dir "${SHARD_DIR}" \
    ${EVAL_TAG:+--tag "${EVAL_TAG}"} \
    ${EVAL_GROUP_SOURCE:+--group-source "${EVAL_GROUP_SOURCE}"} \
    ${CM_FLAGS}

echo "=== Eval shard ${SLURM_ARRAY_TASK_ID} finished at $(date) ==="
