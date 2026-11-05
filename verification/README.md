# Verification workspace (copy)

Copy of `~/verification/`, the workspace used to re-check every number in the PPO-CF paper:
memos, sanity checks, job list, analysis scripts and their outputs, sbatch files, logs,
and the review PDFs. `~/verification/` stays the live copy; re-sync with

    rsync -a --no-links --exclude='/code/' --exclude='/views/' --exclude='*.bundle' \
      --exclude='*.pt' --exclude='*.pyc' --exclude='__pycache__/' --exclude='*.png' \
      --include='/out/' --include='/out/**/' --include='/out/**/*.json' --include='/out/**/*.md' \
      --include='/out/**/*.txt' --include='/out/**/*.log' --include='/out/**/*.csv' \
      --include='/out/**/*.manifest' --exclude='/out/**' --prune-empty-dirs \
      ~/verification/ verification/

Left out (about 6.8 GB): generated shards and checkpoints (`*.pt`), the frozen code
snapshots in `code/` (they are git commits 49e6050 and HEAD of this repo), symlinked
checkpoint views in `views/`, and `evidence-2026-10-28.bundle` (that is the
`evidence-2026-10-28` branch). From `out/` only the text results are kept
(eval metrics, transfer, schedules, logs).
