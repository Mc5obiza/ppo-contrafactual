"""
PPO-only seed replicate (issue D3): main.py with Step 4 (LoRA fine-tuning)
replaced by loading the direction's OWN already-trained adapter, sha-checked.

    SEEDREP_DIRECTION=n_to_p|p_to_n ADAPTER_NP_VIEW=… ADAPTER_PN_VIEW=… \
    ADAPTER_EXPECT_NP_SHA=… ADAPTER_EXPECT_PN_SHA=… \
        python train_ppo_only.py --direction <same as SEEDREP_DIRECTION>

Why: a full retrain (09_retrain) re-fine-tunes LoRA (~5.5 h) and, as in J9, the
two array tasks share one finetuning_latest.pt, so both directions can end on
ONE adapter. Here each direction trains its policy against exactly the adapter
it is evaluated with (the J4 protocol), so the replicate varies only the PPO
seed (TRAIN_SEED in the code copy; main.py re-seeds per direction with its
offset, so the PPO seed does not depend on the skipped LoRA step).

per_direction.use() reloads the generator through pipeline.load_generator_weights
and raises unless the generator now holds exactly the file's LoRA and cond_proj
weights and, if ADAPTER_EXPECT_*_SHA is set, the file's sha256 matches.
"""
import os

import config  # project config first (torchxrayvision shadows `config`)
import per_direction
import main as M

LABEL = os.environ["SEEDREP_DIRECTION"]
if LABEL not in ("n_to_p", "p_to_n"):
    raise ValueError(f"SEEDREP_DIRECTION={LABEL!r}")


def _load_own_adapter(generator, dataset, indices=None, **kwargs):
    print(f"[seedrep] Step 4 skipped: loading the trained {LABEL} adapter instead of fine-tuning",
          flush=True)
    per_direction.use(generator, LABEL)


M.run_lora_finetuning = _load_own_adapter
print(f"[seedrep] TRAIN_SEED={config.TRAIN_SEED} USE_GUIDANCE_ENSEMBLE={config.USE_GUIDANCE_ENSEMBLE} "
      f"direction={LABEL} PPO_CKPT_DIR={os.environ.get('PPO_CKPT_DIR')} "
      f"PPO_WALLCLOCK_BUDGET_S={os.environ.get('PPO_WALLCLOCK_BUDGET_S')}", flush=True)
M.main()
