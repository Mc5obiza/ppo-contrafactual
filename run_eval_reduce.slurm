#!/usr/bin/env bash
#SBATCH --job-name=ppo-eval-reduce
#SBATCH --partition=publicgpu
#SBATCH --account=insa
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpuh200|gpuh100|gpua100|gpul40s|gpua40|gpurtx6000"
#SBATCH --exclude=hpc-n983
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=melkiayoub724@gmail.com
#
# STAGE 2 of 2 — compute metrics from the shards run_eval_cf.slurm produced.
# Writes checkpoints/eval_metrics.json and the two counterfactual diff plots.
#
#   sbatch run_eval_cf.slurm                                  # note the job id
#   sbatch --dependency=afterok:<jobid> run_eval_reduce.slurm
#
# The dependency matters: this job fails fast (non-zero, naming the missing
# shard ids) if the array has not fully completed, because metrics over a
# partial shard set are computed on a short Xn/Xp and FID's upward bias grows
# sharply as N falls — a partial reduce would print a plausible wrong number
# rather than an error.
#
# WALLTIME. 4h, which is mostly slack. The expensive part — 400 policy episodes
# — already happened in the array; what is left is FID/CV/IP/CTR/SCP over
# cached tensors plus two single-image diff plots, i.e. minutes. The only real
# costs are loading ~2GB of model weights and the CTR segmentation pass, both
# now batched.
#
# It still needs a GPU (and the same node constraints as the array — see
# run_eval_cf.slurm's header for why hpc-n983 is excluded by name) because the
# metric backbones, the segmenter and the diff plots all run on device.
#
# Re-running this is cheap and safe: the shards are read-only inputs, so a
# metric fix or a changed threshold is a resubmit of this job alone, with no
# regeneration. But do NOT edit the generation knobs in config.py first — the
# shards carry a config stamp and this job refuses to report counterfactuals
# from one ablation arm under another arm's config.

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

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export YOLO_OFFLINE=1

# Must match run_eval_cf.slurm's NUM_SHARDS. Passing it here is a cross-check,
# not the source of truth: the shards themselves record the count they were
# written with, and eval_main.py errors if the two disagree.
NUM_SHARDS=8

# Per-domain checkpoint dir — see the note in run_eval_cf.slurm. The generate
# stage writes its shards under config.CKPT_DIR, so this must resolve the same
# way or the reduce job reports "generate has not run" against a full dir.
CKPT_DIR=$(python -c 'import config; print(config.CKPT_DIR)')

# Provenance for the metrics this job is about to write — see
# record_run_manifest.sh.
bash record_run_manifest.sh "${CKPT_DIR}"

# Must resolve the SAME arm and directory as the generate array that produced
# these shards — see run_eval_cf.slurm's EVAL_ARM block. Pass it through with
#   sbatch --dependency=afterok:<jobid> --export=ALL,EVAL_ARM=none run_eval_reduce.slurm
EVAL_ARM="${EVAL_ARM:-trained}"
if [ "${EVAL_ARM}" = "trained" ]; then
    SHARD_DIR="${CKPT_DIR}/eval_shards"
    OUT_DIR="${CKPT_DIR}"
else
    SHARD_DIR="${CKPT_DIR}/eval_shards_${EVAL_ARM}"
    OUT_DIR="${CKPT_DIR}/eval_${EVAL_ARM}"
fi

# Countermeasure arm — must resolve IDENTICALLY to run_eval_cf.slurm, including
# the flags. --guidance-ensemble/--gradcam-mask are not cosmetic here: the reduce
# stage rebuilds f(.) and the generator to compute metrics and draw the diff
# plots, so an arm reduced without its flags would score ensemble-guided
# counterfactuals with a single-classifier f(.).
#
#   sbatch --dependency=afterok:<jobid> \
#          --export=ALL,EVAL_ARM=none,EVAL_CM=ensemble,EVAL_TAG=cm1 \
#          run_eval_reduce.slurm
EVAL_CM="${EVAL_CM:-}"
EVAL_TAG="${EVAL_TAG:-}"
# Must match the generate array exactly — it is part of the config stamp, so a
# mismatch is caught by _load_shards rather than silently reported.
EVAL_GROUP_SOURCE="${EVAL_GROUP_SOURCE:-}"
# Seed replicate. Each value needs its own EVAL_TAG — eval_main refuses to pool
# shards drawn under different seeds.
EVAL_BASE_SEED="${EVAL_BASE_SEED:-}"
# Ensemble roster for EVAL_CM=ensemble/both. Adding a member that shares training
# data with a held-out validator inflates transfer for the wrong reason — drop
# the overlapping validator when comparing (arm_transfer.py --validators).
EVAL_ENSEMBLE_WEIGHTS="${EVAL_ENSEMBLE_WEIGHTS:-}"
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
    echo "EVAL_CM=${EVAL_CM} requires EVAL_TAG — see run_eval_cf.slurm." >&2
    exit 1
fi
if [ -n "${EVAL_TAG}" ]; then
    SHARD_DIR="${SHARD_DIR}_${EVAL_TAG}"
    OUT_DIR="${OUT_DIR}_${EVAL_TAG}"
fi
echo "[slurm] arm=${EVAL_ARM}  cm=${EVAL_CM:-none}  tag=${EVAL_TAG:-none}"
echo "[slurm] shard_dir=${SHARD_DIR}  out_dir=${OUT_DIR}"
if [ ! -d "${SHARD_DIR}" ]; then
    echo "${SHARD_DIR} does not exist — the generate stage has not run." >&2
    echo "    sbatch run_eval_cf.slurm" >&2
    exit 1
fi

echo "=== Eval reduce started on $(hostname) at $(date) ==="
echo "shards present: $(ls -1 ${SHARD_DIR}/cf_shard_*.pt 2>/dev/null | wc -l) / ${NUM_SHARDS}"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

python -u preflight_gpu.py

# --shard-dir, --arm and --out-dir must ALL be passed. This script computed
# SHARD_DIR from EVAL_ARM and then did not forward it, so eval_main.py fell back
# to its defaults (--arm trained, the trained arm's shard dir) no matter what
# EVAL_ARM said. A control-arm reduce would therefore have loaded the TRAINED
# arm's shards, drawn its diff plots with the TRAINED policies, and written the
# result out as the control's — making the three arms agree by construction and
# turning "the PPO policy contributes nothing" into an artefact of this line
# rather than a finding. The parser's own arm/shard-dir guard could not catch it
# precisely because --arm was never passed, so it saw a consistent trained run.
python -u eval_main.py \
    --stage reduce \
    --num-shards "${NUM_SHARDS}" \
    --arm "${EVAL_ARM}" \
    --shard-dir "${SHARD_DIR}" \
    --out-dir "${OUT_DIR}" \
    ${EVAL_TAG:+--tag "${EVAL_TAG}"} \
    ${EVAL_GROUP_SOURCE:+--group-source "${EVAL_GROUP_SOURCE}"} \
    ${EVAL_BASE_SEED:+--base-seed "${EVAL_BASE_SEED}"} \
    ${EVAL_ENSEMBLE_WEIGHTS:+--ensemble-weights "${EVAL_ENSEMBLE_WEIGHTS}"} \
    ${CM_FLAGS}

echo "=== Eval reduce finished at $(date) ==="
