#!/usr/bin/env bash
#SBATCH --job-name=ppo-counterfactual-eval
#SBATCH --partition=publicgpu
#SBATCH --account=insa
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpuh200|gpuh100|gpua100|gpul40s|gpua40|gpurtx6000"
#SBATCH --exclude=hpc-n983
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=melkiayoub724@gmail.com
#
# ┌──────────────────────────────────────────────────────────────────────────┐
# │ SUPERSEDED — do not submit this at n_eval=200. It CANNOT finish.         │
# │                                                                          │
# │ Use instead:                                                             │
# │     sbatch run_eval_cf.slurm                       # 8-task array        │
# │     sbatch --dependency=afterok:<id> run_eval_reduce.slurm               │
# │                                                                          │
# │ This script runs eval_main.py's single-process --stage all path. Job     │
# │ 17665348 proved that path cannot fit: it managed ~4.4 min/image and the  │
# │ suite needs 400 counterfactuals, so generation alone is ~29h. `sinfo -p  │
# │ publicgpu` reports TIMELIMIT 1-00:00:00, a HARD 24h cap, so raising      │
# │ --time below is not a fix — that is why generation is now sharded        │
# │ across a SLURM array and the metrics run as a separate cheap job.        │
# │                                                                          │
# │ Kept only for small local smoke tests, where it is still the simplest    │
# │ way to exercise the whole pipeline end to end in one process:            │
# │     python -u eval_main.py --stage all --n-eval 8                        │
# │ The everything-below-here rationale (GPU constraints, offline mode, the  │
# │ ppo_policy.pt precondition) still applies and is mirrored in the two new │
# │ scripts.                                                                 │
# └──────────────────────────────────────────────────────────────────────────┘
#
# Runs the EVALUATION pipeline (eval_main.py) against the checkpoints that
# run_ppo.slurm produced. Steps 6 and 6B only: the full Singla et al. (2022)
# metric suite (CV/FID/FOP/IP/CTR/SCP) and the two counterfactual diff plots.
#
# PREREQUISITE: run_ppo.slurm must have completed and written
#   <config.CKPT_DIR>/n_to_p/ppo_policy.pt
#   <config.CKPT_DIR>/p_to_n/ppo_policy.pt
# eval_main.py fails fast with a clear message if they are missing.
#
# WHY THIS IS A SEPARATE JOB FROM TRAINING: PPO training measured ~20h on a
# Tesla P100 and ~12.5h on a Quadro RTX 5000, with evaluation a further
# ~2.5-5h on top. Job 17644171 ran both in one process, finished training,
# and was killed at the 24h boundary partway through evaluation — the
# policies survived on disk but no metrics were produced. Splitting the two
# means a slow eval can never cost a PPO run, and eval can be re-run freely
# (different n_eval, regenerated plots, metric fixes) without retraining.
#
# GPU CONSTRAINT — must match run_ppo.slurm, and for the same two reasons.
# See that file's header for the full rationale; in short:
#   * P100 nodes have no fp16 tensor cores and are ~3x slower here. Evaluation
#     was >4h and UNFINISHED on one.
#   * hpc-n983 is an RTX PRO 6000 *Blackwell* (sm_120) that advertises the same
#     `gpurtx6000` feature as the Turing Quadro RTX 6000 nodes. venv torch is
#     2.5.1+cu121, whose kernels stop at sm_90, so every CUDA op there dies
#     with "no kernel image is available for execution on the device" — this
#     killed training job 17662452_0 five minutes in. SLURM constraints have no
#     NOT operator and the feature tag cannot distinguish the two cards, so the
#     node has to be excluded by name.
# preflight_gpu.py below is the backstop if a new node ever joins a tag.
#
# WALLTIME: 12h — and this is exactly what proved insufficient. The estimate
# below ("~2.5h on an RTX 5000") predated config.FIXED_EPISODE_LENGTH and was
# never re-measured; the note that follows it correctly flagged that the old
# timings had become a lower bound, but the walltime was left at 12h anyway.
# Job 17665348 then measured the real figure: ~4.4 min/image, ~29h for the 400
# counterfactuals, i.e. over the partition's 24h cap. See the banner at the top
# of this file for the replacement.
#
# THE COST THAT INVALIDATED THE OLD ESTIMATE: config.FIXED_EPISODE_LENGTH makes
# generate_via_policy run the full BUDGET_T steps instead of stopping as soon
# as the counterfactual is judged valid, so a policy that used to converge
# early now always pays for all 8 steps. That is deliberate — eval and training
# step the environment identically — but it means evaluation cannot get
# cheaper than 8 steps x n_eval. Nothing else in metrics.py or diff_viz.py
# reads episode length.
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

# Fail before loading ~2GB of model weights if training never ran.
#
# ASK CONFIG FOR THE DIRECTORY, never hardcode "checkpoints" — CKPT_DIR is
# per-domain and PPO_CKPT_DIR overrides it. See run_eval_cf.slurm's copy of
# this guard for the failure a hardcoded path caused.
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

echo "=== Eval job started on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

# Same hardware check as run_ppo.slurm — see preflight_gpu.py.
python -u preflight_gpu.py

# Unbuffered so the log is useful while the job is still running — evaluation
# is a long single phase and you want to see which metric it is on.
python -u eval_main.py

echo "=== Eval job finished at $(date) ==="
