#!/usr/bin/env bash
#
# record_run_manifest.sh — snapshot the exact state a job ran under.
#
# WHY THIS EXISTS
# ---------------
# Reproducing a result meant reconstructing three things that were never
# recorded together: which commit was checked out, whether the tree was dirty,
# and what config.py said at submit time. All three moved constantly —
# config.py was edited three times on a single day — and run_eval_cf.slurm's
# only defence was a shouted comment ("Do NOT edit config.py between submitting
# this array and the reduce job"), which is a request, not a control.
#
# The concrete failure this prevents: checkpoints/eval_metrics.json exists with
# no way to tell which code produced it, and the RSNA policy checkpoints that
# generated the project's headline numbers are gone entirely. A manifest is
# cheap insurance against the next instance of that.
#
# Writes TWO copies deliberately:
#   logs/<jobname>-<jobid>.manifest   travels with the log, survives a wiped
#                                     checkpoint dir
#   <ckpt_dir>/run_manifest.txt       travels with the weights, survives a
#                                     wiped logs/ (which is gitignored)
#
# Usage — execute early in a slurm script, after `source venv/bin/activate` so
# `python -c 'import config'` resolves:
#
#     bash record_run_manifest.sh "${CKPT_DIR:-}"              # shared dir
#     bash record_run_manifest.sh "${CKPT_DIR}/${DIRECTION}" --per-task
#
# Pass --per-task when every array task writes to a DIFFERENT directory (as
# run_ppo.slurm does — one dir per edit direction). Without it, only task 0
# writes the checkpoint-dir copy, which is right for an array whose tasks share
# one output dir (run_eval_cf.slurm) and wrong for one whose tasks do not.
#
# Never fails the job: provenance is worth having but not worth killing a
# 20-hour run over, so every step is best-effort.

set -uo pipefail   # deliberately NOT -e; see above

ckpt_dir="${1:-}"
per_task=0
[ "${2:-}" = "--per-task" ] && per_task=1
job_name="${SLURM_JOB_NAME:-local}"
job_id="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-$$}}"
task_id="${SLURM_ARRAY_TASK_ID:-}"
tag="${job_name}-${job_id}${task_id:+_${task_id}}"

manifest="$(
    echo "run manifest"
    echo "================================================================"
    echo "written        : $(date -Is)"
    echo "job            : ${tag}"
    echo "node           : $(hostname)"
    echo "slurm partition: ${SLURM_JOB_PARTITION:-n/a}"
    echo
    echo "-- git ---------------------------------------------------------"
    echo "commit         : $(git rev-parse HEAD 2>/dev/null || echo '<not a git repo>')"
    echo "branch         : $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
    # A dirty tree means the commit above does NOT describe what ran. Name the
    # files rather than just flagging it, so the diff can be reconstructed.
    dirty="$(git status --porcelain 2>/dev/null)"
    if [ -n "${dirty}" ]; then
        echo "tree           : DIRTY — the commit above does not describe this run"
        echo "${dirty}" | sed 's/^/                 /'
    else
        echo "tree           : clean"
    fi
    echo
    echo "-- environment -------------------------------------------------"
    echo "python         : $(python -V 2>&1)"
    echo "torch          : $(python -c 'import torch; print(torch.__version__, "cuda", torch.version.cuda)' 2>/dev/null || echo '?')"
    echo "gpu            : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || echo 'n/a')"
    echo
    echo "-- config.py ---------------------------------------------------"
    # Hash the file AND dump the values that actually change behaviour, so a
    # mismatch is both detectable (hash) and diagnosable (values).
    echo "sha256         : $(sha256sum config.py 2>/dev/null | cut -d' ' -f1)"
    python - <<'PYEOF' 2>/dev/null || echo "  <could not import config>"
import config
keys = [
    "DATASET_DOMAIN", "CKPT_DIR", "DEVICE", "DTYPE", "IMG_SIZE",
    # Which images the run saw, and which RNG stream it drew. These are the
    # fields that decide whether a result is reproducible AT ALL, so they lead:
    # a run whose split or seed is unknown cannot be repeated even with the
    # exact commit.
    "USE_SPLIT", "SPLIT_SEED", "TEST_FRAC", "EVAL_GROUP_SOURCE",
    "TRAIN_SEED", "STRICT_DETERMINISM",
    "USE_GUIDANCE_ENSEMBLE", "USE_GRADCAM_MASK", "USE_SPARSITY_REWARD",
    "USE_SPARSITY_ANNEALING", "FIXED_EPISODE_LENGTH", "EDIT_BUDGET_L1",
    "BUDGET_T", "N_EPISODES", "N_DENOISE_EPOCHS", "T_DIFF",
    "POLICY_TRUNK", "LORA_R", "ALPHA_CLS", "ALPHA_PRES",
]
for k in keys:
    print(f"  {k:<24}= {getattr(config, k, '<unset>')}")
PYEOF
)"

mkdir -p logs
printf '%s\n' "${manifest}" > "logs/${tag}.manifest"
echo "[manifest] wrote logs/${tag}.manifest"

# Only ONE task of an array writes the checkpoint-dir copy, unless --per-task
# says each task has its own directory. Tasks of an array share a config and a
# commit, so the copies would be identical — but eight concurrent writers to one
# path can interleave into a corrupt file, and a corrupt manifest is worse than
# none. The per-task log copies above are unaffected; they have distinct names.
if [ -n "${ckpt_dir}" ] && \
   { [ "${per_task}" = "1" ] || [ -z "${task_id}" ] || [ "${task_id}" = "0" ]; }; then
    mkdir -p "${ckpt_dir}"
    printf '%s\n' "${manifest}" > "${ckpt_dir}/run_manifest.txt"
    echo "[manifest] wrote ${ckpt_dir}/run_manifest.txt"
fi

exit 0
