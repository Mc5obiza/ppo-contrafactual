#!/usr/bin/env bash
#SBATCH --job-name=ppo-counterfactual-heldout
#SBATCH --partition=publicgpu
#SBATCH --account=insa
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=melkiayoub724@gmail.com
#
# Runs heldout_validation.py: re-scores the PPO counterfactuals with
# classifiers that were NEVER in the guidance loop, to separate real disease
# edits from adversarial perturbations against f(.).
#
# WHY THIS IS NOT PART OF eval_main.py: eval_main.py's CV is computed with the
# SAME classifier that produced the guidance gradient (see pipeline.py's
# build_common -> densenet121-res224-rsna, used both for grad_cls in
# generator.generate_counterfactual and for CV in metrics.run_full_evaluation).
# That number is circular. This job is the independent check, kept separate so
# it can be re-run against new validators without touching the main eval.
#
# WALLTIME: 8h. The cost is dominated by generating ~400 policy episodes, the
# same work eval_main.py's Step 6 does (~2.5h on an RTX 5000, >4h on a P100).
# AFTER the first run this is minutes, not hours — the counterfactuals are
# cached to checkpoints/heldout/cf_cache.pt and re-scoring with an extra
# validator only reloads them.
#
# PREREQUISITES
#   - run_ppo.slurm has written checkpoints/{n_to_p,p_to_n}/ppo_policy.pt
#   - setup_env.sh has run on the LOGIN node. The validator weights
#     (densenet121-res224-nih, -chex) must already be cached under
#     ~/.torchxrayvision/models_data/ — compute nodes have no internet.
#     Verify with:  ls ~/.torchxrayvision/models_data/
#     Expected:     nih-densenet121-*.pt  and  chex-densenet121-*.pt

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

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export YOLO_OFFLINE=1

for direction in n_to_p p_to_n; do
    if [ ! -f "checkpoints/${direction}/ppo_policy.pt" ]; then
        echo "checkpoints/${direction}/ppo_policy.pt not found — run 'sbatch run_ppo.slurm' first." >&2
        exit 1
    fi
done

# Fail before ~3h of generation rather than at the scoring step if a validator
# was never prefetched on the login node.
#
# chex moved INTO the guidance ensemble (config.GUIDANCE_ENSEMBLE_WEIGHTS), so
# the held-out set is now mimic_ch + pc + nih. rsna/chex are checked too since
# the guidance ensemble itself must load for the pipeline to build at all.
#
# NOTE the first field is the CACHE FILE prefix, which is not always the weight
# suffix: xrv stores densenet121-res224-rsna as "kaggle-densenet121-*.pt"
# (RSNA was a Kaggle competition). Checking for "rsna-densenet121-*" fails even
# on a correctly prefetched machine.
for pair in "kaggle:rsna" "chex:chex" "mimic_ch:mimic_ch" "pc:pc" "nih:nih"; do
    file_prefix="${pair%%:*}"
    weight_suffix="${pair##*:}"
    if ! ls "${HOME}/.torchxrayvision/models_data/${file_prefix}-densenet121-"*.pt >/dev/null 2>&1; then
        echo "torchxrayvision '${weight_suffix}' DenseNet weights not cached under" >&2
        echo "  ${HOME}/.torchxrayvision/models_data/ (expected ${file_prefix}-densenet121-*.pt)" >&2
        echo "Compute nodes have no internet. On the LOGIN node run:" >&2
        echo "  source venv/bin/activate && python -c \"import torchxrayvision as xrv; xrv.models.DenseNet(weights='densenet121-res224-${weight_suffix}')\"" >&2
        exit 1
    fi
done

echo "=== Held-out validation started on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

# Pass extra args straight through, e.g.:
#   sbatch run_heldout_validation.slurm --regenerate
#   sbatch run_heldout_validation.slurm --validators nih,chex,mimic_ch
python -u heldout_validation.py "$@"

echo "=== Held-out validation finished at $(date) ==="
