#!/bin/bash
# Item 3 pass 2: per-image LHR (committed code's paired_arm_test._per_image_lift via scripts/per_image_lhr.py), CPU.
set -euo pipefail
cd "$HOME/Jobs/Ayoub/ppo-contrafactual" && module load python/3.12.8 >/dev/null 2>&1 && source venv/bin/activate
cd "$HOME/verification/code/49e6050"; export PPO_DOMAIN=rsna_pneumonia PYTHONPATH="$PWD"
V=$HOME/verification/out
python -u "$HOME/verification/scripts/per_image_lhr.py" "$HOME/verification/part5/arms_lhr.json" \
  $V/J4_perdir/shards $V/J5_none_perdir/shards $V/J5c_fixed_schedule_corrected/shards $V/J9_seed1/shards
