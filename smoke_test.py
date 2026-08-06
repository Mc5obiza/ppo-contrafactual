"""
smoke_test.py
=============
Smoke test of the TRAINING pipeline at tiny scale: ingestion → LoRA
fine-tuning → PPO (both directions) → checkpoints.

PURPOSE: prove every training code path RUNS and wires together on real data
and a real GPU. It does NOT produce meaningful numbers — budgets are cut far
below the real run (T_DIFF, BUDGET_T, N_EPISODES all slashed). Use it to catch
crashes, shape errors, OOM, missing prefetched weights, and mis-wired flags
before spending 24h of GPU time.

Its main job right now is exercising the Grad-CAM action-space constraint and
the sparsity/annealing/CV-safeguard code, which have only been unit-tested
against stubs — never against the real classifier, generator, and dataset.

Arms (env var SMOKE_ARM, default "all"):
    baseline  — every new flag off; reproduces the pre-change pipeline
    mask      — Grad-CAM mask + episode-origin anchoring
    all       — mask + anchoring + sparsity reward + annealing + CV safeguard

Usage:
    SMOKE_ARM=all python smoke_test.py
    sbatch run_smoke_test.slurm

IMPORTANT — WHY CONFIG IS PATCHED BEFORE THE PIPELINE IMPORTS:
Several functions bind config values as DEFAULT ARGUMENTS, which are evaluated
once at import time (e.g. run_lora_finetuning's `n_epochs=config.N_DENOISE_EPOCHS`,
compute_gradcam_latent_mask's floor/gamma, MedicalCounterfactualEnv.__init__'s
budget_T, pipeline.load_generator_weights' `ckpt_dir=config.CKPT_DIR`).
Assigning to config AFTER importing those modules would silently have no
effect on them. Runtime-read flags (USE_GRADCAM_MASK, USE_SPARSITY_REWARD,
...) are read inside function bodies and would be fine either way — but doing
everything before the imports is the only rule that is uniformly correct, so
that is what this script does.

SCOPE — TRAINING ONLY:
This script covers main.py (Steps 1-5) and stops at the checkpoints it writes.
It deliberately does NOT run evaluation: that is smoke_test_eval.py /
run_smoke_test_eval.slurm, which reloads those checkpoints in a COLD process
and is a strictly better test of the eval path than running it here would be
(in-process, eval would reuse the generator, envs and CUDA context training
just built — hiding exactly the reload failures that matter). Keeping the two
separate also mirrors the real jobs: run_ppo.slurm then run_eval_ppo.slurm.

    sbatch run_smoke_test.slurm        # this file  — training
    sbatch run_smoke_test_eval.slurm   # eval half  — cold checkpoint reload
"""

import os
import sys
import time
import traceback

# ─────────────────────────────────────────────────────────────────────────────
# 0. Arm selection
# ─────────────────────────────────────────────────────────────────────────────

ARM = os.environ.get("SMOKE_ARM", "all").strip().lower()
if ARM not in ("baseline", "mask", "all"):
    print(f"[smoke] unknown SMOKE_ARM={ARM!r}; expected baseline | mask | all", file=sys.stderr)
    sys.exit(2)

SUBSET_N = int(os.environ.get("SMOKE_SUBSET_N", 96))   # balanced dataset subset
# NOTE: SMOKE_N_EVAL is deliberately NOT read here — this script no longer runs
# evaluation. It belongs to smoke_test_eval.py, which honours it.

print("=" * 78)
print(f"  SMOKE TEST (training) — arm={ARM}, subset={SUBSET_N} images")
print("=" * 78)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Config overrides — MUST happen before the pipeline imports (see docstring)
# ─────────────────────────────────────────────────────────────────────────────

import config

# Checkpoints go to a dedicated directory so a smoke run can NEVER clobber the
# real run's policies / LoRA weights. run_lora_finetuning(resume=True) also
# looks here, so it starts fresh instead of resuming a real checkpoint.
config.CKPT_DIR       = os.path.join("checkpoints", f"smoke_{ARM}")
config.CKPT_POLICY    = os.path.join(config.CKPT_DIR, "ppo_policy.pt")
config.CKPT_UNET      = os.path.join(config.CKPT_DIR, "unet_finetuned.pt")
config.CKPT_COND_PROJ = os.path.join(config.CKPT_DIR, "cond_proj.pt")
os.makedirs(config.CKPT_DIR, exist_ok=True)

# ── budgets: cut hard. Image/latent GEOMETRY is deliberately left alone —
# changing IMG_SIZE/LATENT_SIZE would diverge from the real run and could mask
# exactly the shape bugs this test exists to find.
config.T_DIFF                = 10   # real: 50 — ~5 UNet steps per generate call
config.BUDGET_T              = 3    # real: 25
config.ENV_REFINEMENT_PASSES = 1    # real: 2
config.N_EPISODES            = 24   # real: 800 -> 3 updates at 8 episodes/update
config.N_DENOISE_EPOCHS      = 1    # real: 15
config.DENOISE_BATCH_SIZE    = 2    # real: 8  — keep VRAM headroom on the V100
config.LOG_EVERY_N_STEPS     = 5

# ── safeguard: sized so BOTH probe branches (baseline + drop-check) actually
# fire within 3 updates. At the real scale these are 24 / 10 / 12.
#
# KNOWN LIMITATION of the tiny scale: 24 episodes gives 3 updates, so the ramp
# window resolves to [1, 2) and lambda_sparsity takes the values
# [0.0, 0.0, LAM_SPARSITY_TARGET] — it jumps straight from zero to target and
# never passes through a PARTIAL ramp value. So this test covers "the term is
# off", "the term is at full weight", and "both probe branches execute", but
# NOT the interpolation itself. That is covered by unit tests against a stub
# (linear and cosine shapes, monotonicity, the latched freeze) rather than
# here — raising N_EPISODES enough to widen the ramp would cost far more GPU
# time than the coverage is worth.
#
# Also note ENV_EVAL_HOLDOUT_N=4 makes CV resolution 25 points per image, so
# the safeguard may well freeze the ramp on pure noise during this run. That is
# expected and is reported as a NOTE, not a failure — it confirms the automatic
# freeze fires. Do NOT read anything into the CV values themselves.
config.ENV_EVAL_HOLDOUT_N       = 4
config.CV_CHECK_EVERY_N_UPDATES = 1
config.CV_CHECK_MAX_STEPS       = 2

# ── the flags under test ──────────────────────────────────────────────────────
if ARM == "baseline":
    config.USE_GRADCAM_MASK         = False
    config.ANCHOR_TO_EPISODE_ORIGIN = False
    config.USE_SPARSITY_REWARD      = False
elif ARM == "mask":
    config.USE_GRADCAM_MASK         = True
    config.ANCHOR_TO_EPISODE_ORIGIN = True
    config.USE_SPARSITY_REWARD      = False
else:  # all
    config.USE_GRADCAM_MASK         = True
    config.ANCHOR_TO_EPISODE_ORIGIN = True
    config.USE_SPARSITY_REWARD      = True
    config.USE_SPARSITY_ANNEALING   = True

print(f"[smoke] USE_GRADCAM_MASK         = {config.USE_GRADCAM_MASK}")
print(f"[smoke] ANCHOR_TO_EPISODE_ORIGIN = {config.ANCHOR_TO_EPISODE_ORIGIN}")
print(f"[smoke] USE_SPARSITY_REWARD      = {config.USE_SPARSITY_REWARD}")
print(f"[smoke] USE_SPARSITY_ANNEALING   = {config.USE_SPARSITY_ANNEALING}")
print(f"[smoke] CKPT_DIR                 = {config.CKPT_DIR}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# 2. Now safe to import the pipeline
# ─────────────────────────────────────────────────────────────────────────────

import torch
import pipeline as shared          # Steps 1-3, shared with the eval entry point
import main as train_main          # Steps 1-5 (training entry point)
import factory.ppo as ppo_mod

# ─────────────────────────────────────────────────────────────────────────────
# 3. Shrink the expensive bits by patching the pipeline namespaces
# ─────────────────────────────────────────────────────────────────────────────
# NOTE on patch targets: build_dataset is resolved from pipeline's globals at
# call time (build_common calls it), so it is patched on the shared module, not
# on main. Names main.py owns directly are patched on main.

_real_build_dataset = shared.build_dataset


def _small_dataset(*a, **kw):
    """
    Real RSNA dataset, truncated in place to a CLASS-BALANCED subset.

    Truncating `dataset.samples` rather than wrapping in torch Subset is
    deliberate: __len__, __getitem__ and finetuning's fast label-extraction
    path all derive from `.samples`, so this keeps every consumer consistent
    and avoids Subset hiding `.samples` (which would force
    _extract_labels into a full DICOM-decoding scan).

    Balance matters: _build_balanced_sampler raises if either class is empty,
    and both PPO directions need a non-empty "far" pool.
    """
    ds = _real_build_dataset(*a, **kw)
    half = max(1, SUBSET_N // 2)
    pos = [i for i, s in enumerate(ds.samples) if float(s[-1]) == 1.0][:half]
    neg = [i for i, s in enumerate(ds.samples) if float(s[-1]) == 0.0][:half]
    keep = sorted(pos + neg)
    ds.samples = [ds.samples[i] for i in keep]
    print(f"[smoke] dataset truncated to {len(ds.samples)} images "
          f"({len(pos)} pneumonia / {len(neg)} normal)")
    return ds


_real_diag = train_main.run_guidance_diagnostics


def _small_diag(env, policy, n_episodes=20):
    return _real_diag(env, policy, n_episodes=2)


shared.build_dataset                = _small_dataset
train_main.run_guidance_diagnostics = _small_diag

# Capture the trainers and directional policies so the post-run assertions can
# inspect what actually happened (main() returns neither).
TRAINERS = []
DIRECTIONAL = []

_real_trainer_init = ppo_mod.PPOTrainer.__init__


def _recording_init(self, env, policy, lr=config.LR):
    _real_trainer_init(self, env, policy, lr=lr)
    TRAINERS.append(self)


ppo_mod.PPOTrainer.__init__ = _recording_init

_real_train_dir = train_main._train_directional_policy


def _recording_train_dir(*a, **kw):
    dp = _real_train_dir(*a, **kw)
    DIRECTIONAL.append(dp)
    return dp


train_main._train_directional_policy = _recording_train_dir

# ─────────────────────────────────────────────────────────────────────────────
# 4. Run the training entry point
# ─────────────────────────────────────────────────────────────────────────────
# Training only — evaluation is smoke_test_eval.py (see module docstring).

t0 = time.time()
failures = []
try:
    train_main.main()
except Exception:
    traceback.print_exc()
    failures.append("train_main.main() raised — see traceback above")
elapsed = time.time() - t0

# ─────────────────────────────────────────────────────────────────────────────
# 5. Post-run checks on the NEW code paths
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 78)
print("  SMOKE TEST CHECKS")
print("=" * 78)


def check(cond, label, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"  — {detail}" if detail else ""))
    if not cond:
        failures.append(label)


check(len(DIRECTIONAL) == 2, "both directional policies trained",
      f"got {len(DIRECTIONAL)}")

for dp in DIRECTIONAL:
    tag = "n_to_p" if float(dp.target_c) == 1.0 else "p_to_n"
    env = dp.env

    check(len(env._far_indices_eval) > 0, f"[{tag}] CV holdout split non-empty",
          f"{len(env._far_indices_train)} train / {len(env._far_indices_eval)} held out")
    check(
        not (set(env._far_indices_train) & set(env._far_indices_eval)),
        f"[{tag}] holdout is disjoint from the rollout pool",
    )

    if config.USE_GRADCAM_MASK:
        tot, bad = env._cam_episodes_total, env._cam_episodes_degenerate
        check(tot > 0, f"[{tag}] Grad-CAM ran", f"{tot} episodes")
        check(bad == 0, f"[{tag}] no degenerate Grad-CAMs",
              f"{bad}/{tot} degenerate"
              + (" — those episodes ran lung-only, arm is NOT clean" if bad else ""))
        check(env._cam_mask_mean is not None, f"[{tag}] mask stats recorded",
              f"mean={env._cam_mask_mean}, frac_open={env._cam_mask_frac_open}")
    else:
        check(env._cam_latent_mask is None, f"[{tag}] no CAM mask (baseline arm)")

if config.USE_SPARSITY_REWARD:
    for i, tr in enumerate(TRAINERS):
        lam_end = tr._lambda_for_update(3, 3)
        if tr.lambda_frozen is not None:
            # The freeze latching is a PASS condition, not a failure: at this
            # scale ENV_EVAL_HOLDOUT_N=4 gives 25 CV points per image, so the
            # safeguard trips on noise in a large fraction of runs (see the
            # config section above, which documents exactly this). Once latched,
            # _lambda_for_update returns lambda_frozen — so asserting
            # lam_end > 0.0 here would fail the test on its own documented
            # behaviour. What we want covered is that ONE of the two happened:
            # the ramp climbed, or the safeguard caught it.
            print(f"  NOTE  [trainer {i}] ramp froze at lambda={tr.lambda_frozen:.4f} — "
                  f"expected at this tiny scale (CV is noise here); confirms the "
                  f"safeguard fires automatically. The non-zero-lambda check "
                  f"below applies only to an unfrozen ramp and is skipped.")
        else:
            check(lam_end > 0.0, f"[trainer {i}] lambda_sparsity reached a non-zero value",
                  f"{lam_end:.4f} at end of ramp")
        check(len(tr.ramp_history) > 0, f"[trainer {i}] CV probe fired during the ramp",
              f"{len(tr.ramp_history)} probe(s): "
              + ", ".join(f"u{h['update']} cv={h['cv']:.3f} lam={h['lambda_sparsity']:.3f}"
                          for h in tr.ramp_history))
        check(tr.cv_pre_ramp is not None, f"[trainer {i}] pre-ramp CV baseline captured",
              f"{tr.cv_pre_ramp}")

# Artefacts training is supposed to leave behind. These are exactly what
# run_eval_ppo.slurm reloads, so a miss here breaks the eval half — which is
# why all three files are asserted, not just the policy.
for label in ("n_to_p", "p_to_n"):
    for rel in ("ppo_policy.pt", "unet_finetuned.pt", "cond_proj.pt"):
        p = os.path.join(config.CKPT_DIR, label, rel)
        check(os.path.exists(p), f"checkpoint written: {label}/{rel}")

print("\n" + "-" * 78)
summary = f"  arm={ARM}   elapsed={elapsed / 60:.1f} min"
if torch.cuda.is_available():
    # Peak VRAM is the number that tells you whether the REAL run's larger
    # budgets (DENOISE_BATCH_SIZE 2->8, T_DIFF 10->50) have headroom on the
    # 32GB V100. Watch this even when the test passes.
    summary += f"   peak GPU={torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB"
print(summary)
print("-" * 78)

if failures:
    print(f"\n✗ SMOKE TEST FAILED — {len(failures)} problem(s):")
    for f in failures:
        print(f"    - {f}")
    sys.exit(1)

print("\n✓ SMOKE TEST PASSED — every TRAINING stage ran end to end.")
print("  Budgets are slashed, so nothing here says the model is any good; this")
print("  only proves the training pipeline wires together and the flags are live.")
print("\n  Next: sbatch run_smoke_test_eval.slurm  — reloads these checkpoints in")
print("        a cold process and covers the evaluation half.")
