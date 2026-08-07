"""
smoke_test_eval.py
==================
Smoke test for the EVALUATION half of the pipeline (eval_main.py) — the
cold-process checkpoint reload that run_eval_ppo.slurm actually performs.

WHY THIS EXISTS SEPARATELY FROM smoke_test.py
---------------------------------------------
smoke_test.py runs training and evaluation back to back in ONE process, so
when eval_main.main() reloads the checkpoints, the generator, envs and CUDA
context it needs were all just built by the training half. That hides the
failure modes this test targets, every one of which only appears in a cold
process:

  - checkpoint files missing, or written to a different path than eval reads
  - state_dict key mismatch loading unet_finetuned.pt into a freshly
    PEFT-wrapped UNet
  - obs_dim disagreement between the saved policy and the env eval rebuilds
  - anything eval implicitly relied on training having left in memory

This runs eval_main.main() and nothing else, exactly as the real eval job
does.

USE 1 — validate the plumbing after a training smoke run (default):
    SMOKE_ARM=all sbatch run_smoke_test.slurm      # writes checkpoints/smoke_all/
    sbatch run_smoke_test_eval.slurm               # reads it back cold

USE 2 — validate the REAL policies before committing to the ~2.5-5h eval job:
    SMOKE_EVAL_SRC=checkpoints sbatch run_smoke_test_eval.slurm

Use 2 works because obs_dim is
    z_orig(4*32*32) + z_t(4*32*32) + step + d_t + c + seg_mask(256*256) = 73731
which is a function of IMG_SIZE and LATENT_SIZE only — NOT of T_DIFF,
BUDGET_T or ENV_REFINEMENT_PASSES. So this test can slash the generation
budgets for speed and still load a real full-budget policy. That is also why
the geometry config below is left strictly alone.

SAFETY — READ DIRECTORY IS NOT THE WRITE DIRECTORY
--------------------------------------------------
eval_main.main() reads policies from and writes eval_metrics.json + the two
diff PNGs to the SAME config.CKPT_DIR. Pointed at the real checkpoints/ that
would overwrite genuine eval results with meaningless n_eval=12 numbers. So
this script splits them: policies are read from SMOKE_EVAL_SRC, while
config.CKPT_DIR (everything written) is a throwaway smoke directory. A
post-run check asserts the source directory was not modified.

As with smoke_test.py: a PASS means "the eval path runs and the checkpoints
load", NOT "the numbers are good". Every metric here is computed over a
handful of images at a slashed diffusion budget and is meaningless.
"""

import os
import sys
import time
import traceback

# ─────────────────────────────────────────────────────────────────────────────
# 0. Where to read checkpoints from
# ─────────────────────────────────────────────────────────────────────────────

SMOKE_EVAL_SRC = os.environ.get("SMOKE_EVAL_SRC", os.path.join("checkpoints", "smoke_all"))
SUBSET_N       = int(os.environ.get("SMOKE_SUBSET_N", 96))
SMOKE_N_EVAL   = int(os.environ.get("SMOKE_N_EVAL", 12))

print("=" * 78)
print(f"  EVAL SMOKE TEST — src={SMOKE_EVAL_SRC}, subset={SUBSET_N}, n_eval={SMOKE_N_EVAL}")
print("=" * 78)

# Fail before importing ~2GB of model code if there is nothing to evaluate.
_missing = [
    os.path.join(SMOKE_EVAL_SRC, d, "ppo_policy.pt")
    for d in ("n_to_p", "p_to_n")
    if not os.path.exists(os.path.join(SMOKE_EVAL_SRC, d, "ppo_policy.pt"))
]
if _missing:
    print(f"[smoke-eval] missing trained policies: {_missing}", file=sys.stderr)
    print(f"[smoke-eval] run a training pipeline that writes to '{SMOKE_EVAL_SRC}' first:",
          file=sys.stderr)
    print(f"[smoke-eval]   SMOKE_ARM=all sbatch run_smoke_test.slurm   (-> checkpoints/smoke_all)",
          file=sys.stderr)
    print(f"[smoke-eval]   sbatch run_ppo.slurm                        (-> checkpoints)",
          file=sys.stderr)
    sys.exit(2)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Config overrides — MUST happen before the pipeline imports
# ─────────────────────────────────────────────────────────────────────────────
# Same rule as smoke_test.py: several functions bind config values as default
# arguments evaluated at import time, so assigning to config afterwards would
# silently not apply.

import config

# WRITE target — deliberately NOT SMOKE_EVAL_SRC. Everything eval_main.main()
# produces (eval_metrics.json, both diff PNGs) lands here, so pointing this
# test at the real checkpoints/ can never overwrite real results.
config.CKPT_DIR       = os.path.join("checkpoints", "smoke_eval")
config.CKPT_POLICY    = os.path.join(config.CKPT_DIR, "ppo_policy.pt")
config.CKPT_UNET      = os.path.join(config.CKPT_DIR, "unet_finetuned.pt")
config.CKPT_COND_PROJ = os.path.join(config.CKPT_DIR, "cond_proj.pt")
os.makedirs(config.CKPT_DIR, exist_ok=True)

# ── budgets: cut hard. Image/latent GEOMETRY is deliberately left alone — it
# determines obs_dim, so changing it would break loading real policies (and
# would mask exactly the shape bugs this test exists to find).
config.T_DIFF                = 10   # real: 50
config.BUDGET_T              = 3    # real: 25
config.ENV_REFINEMENT_PASSES = 1    # real: 2

# The policy being loaded was trained under whatever flags its run used. These
# affect GENERATION behaviour, not the policy's shape, so eval can run with
# them off — but a mask-arm policy evaluated with the mask off is not a
# like-for-like reproduction of its training conditions. That is fine for a
# smoke test (we are checking the path executes) and is NOT how the real eval
# job should be configured: there, config must match the training run.
print(f"[smoke-eval] USE_GRADCAM_MASK         = {config.USE_GRADCAM_MASK}")
print(f"[smoke-eval] ANCHOR_TO_EPISODE_ORIGIN = {config.ANCHOR_TO_EPISODE_ORIGIN}")
print(f"[smoke-eval] read  from : {SMOKE_EVAL_SRC}")
print(f"[smoke-eval] write to   : {config.CKPT_DIR}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# 2. Snapshot the source directory so we can prove we did not write to it
# ─────────────────────────────────────────────────────────────────────────────

def _snapshot(root: str) -> dict:
    """
    (path -> (size, mtime)) for every file under root, EXCLUDING this test's
    own write directory.

    The exclusion matters in the documented "point at the real checkpoints"
    mode: SMOKE_EVAL_SRC=checkpoints makes the write target
    checkpoints/smoke_eval a CHILD of the source tree, so the test's own
    eval_metrics.json and diff PNGs land inside the directory it is asserting
    is untouched — reporting a failure whose entire evidence is files this
    process was supposed to create. With the default
    SMOKE_EVAL_SRC=checkpoints/smoke_all the two are siblings and this never
    fires, which is why it went unnoticed.
    """
    write_dir = os.path.abspath(config.CKPT_DIR)
    out = {}
    for dirpath, dirnames, filenames in os.walk(root):
        if os.path.abspath(dirpath) == write_dir:
            dirnames[:] = []          # do not descend into the write target
            continue
        for fn in filenames:
            p = os.path.join(dirpath, fn)
            try:
                st = os.stat(p)
                out[p] = (st.st_size, st.st_mtime)
            except OSError:
                pass
    return out


SRC_BEFORE = _snapshot(SMOKE_EVAL_SRC)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Now safe to import the pipeline
# ─────────────────────────────────────────────────────────────────────────────

import torch
import pipeline as shared
import eval_main

# ─────────────────────────────────────────────────────────────────────────────
# 4. Patch: shrink the dataset, redirect reads to SMOKE_EVAL_SRC, capture state
# ─────────────────────────────────────────────────────────────────────────────
# NOTE on patch targets: eval_main.py does `from pipeline import ...`, which
# binds those functions into eval_main's own namespace at import time. Patching
# pipeline.load_directional_policy here would therefore NOT affect the call
# eval_main makes — it must be patched on eval_main. build_dataset is different:
# build_common() resolves it from pipeline's globals at call time, so it is
# patched on the shared module.

_real_build_dataset = shared.build_dataset


def _small_dataset(*a, **kw):
    """
    Real RSNA dataset truncated in place to a CLASS-BALANCED subset, matching
    smoke_test.py. Truncating `.samples` rather than wrapping in a Subset keeps
    __len__/__getitem__ and the fast label path consistent.

    Balance matters here for two reasons: both directions need a non-empty
    "far" pool for their env, and run_full_evaluation needs to find images on
    both sides of its f(x)<0.2 / f(x)>0.8 thresholds.
    """
    ds = _real_build_dataset(*a, **kw)
    half = max(1, SUBSET_N // 2)
    pos = [i for i, s in enumerate(ds.samples) if float(s[-1]) == 1.0][:half]
    neg = [i for i, s in enumerate(ds.samples) if float(s[-1]) == 0.0][:half]
    keep = sorted(pos + neg)
    ds.samples = [ds.samples[i] for i in keep]
    print(f"[smoke-eval] dataset truncated to {len(ds.samples)} images "
          f"({len(pos)} pneumonia / {len(neg)} normal)")
    return ds


_real_load_gen = eval_main.load_generator_weights
_real_load_pol = eval_main.load_directional_policy
_real_eval     = eval_main.run_full_evaluation

LOADED_FROM = []
DIRECTIONAL = []


def _src_load_gen(generator, ckpt_dir=None):
    """Force the READ path to SMOKE_EVAL_SRC regardless of config.CKPT_DIR."""
    path = _real_load_gen(generator, ckpt_dir=SMOKE_EVAL_SRC)
    LOADED_FROM.append(path)
    return path


def _src_load_pol(target_c, label, ctx, ckpt_dir=None):
    dp = _real_load_pol(target_c=target_c, label=label, ctx=ctx, ckpt_dir=SMOKE_EVAL_SRC)
    DIRECTIONAL.append(dp)
    return dp


def _small_eval(**kw):
    """
    Belt-and-braces cap on n_eval. main() is now invoked with an explicit
    --n-eval below, so this normally has nothing to do; it stays because every
    eval image costs a full policy episode and an accidental n_eval=200 here
    would turn a smoke test into a 29h job. CV/FID/IP over this few images at
    T_DIFF=10 are meaningless — path coverage only.
    """
    kw["n_eval"] = min(SMOKE_N_EVAL, kw.get("n_eval", SMOKE_N_EVAL))
    print(f"[smoke-eval] run_full_evaluation forced to n_eval={kw['n_eval']} "
          f"(metrics will be meaningless — path coverage only)")
    return _real_eval(**kw)


shared.build_dataset             = _small_dataset
eval_main.load_generator_weights = _src_load_gen
eval_main.load_directional_policy = _src_load_pol
eval_main.run_full_evaluation    = _small_eval

# ─────────────────────────────────────────────────────────────────────────────
# 5. Run the evaluation entry point, cold
# ─────────────────────────────────────────────────────────────────────────────

t0 = time.time()
failures = []
try:
    # --stage all is the single-process path: scan, generate and score in one
    # go. That is what this smoke test is for. The real eval job now shards
    # generation across a SLURM array (--stage generate) and scores separately
    # (--stage reduce); those stages share every code path exercised here
    # except the shard write/read, which run_eval_cf.slurm covers.
    eval_main.main(["--stage", "all", "--n-eval", str(SMOKE_N_EVAL)])
except Exception:
    traceback.print_exc()
    failures.append("eval_main.main() raised — see traceback above")
elapsed = time.time() - t0

# ─────────────────────────────────────────────────────────────────────────────
# 6. Checks
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 78)
print("  EVAL SMOKE TEST CHECKS")
print("=" * 78)


def check(cond, label, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"  — {detail}" if detail else ""))
    if not cond:
        failures.append(label)


# ── the reload itself ─────────────────────────────────────────────────────────
check(len(LOADED_FROM) == 1, "fine-tuned generator weights loaded",
      LOADED_FROM[0] if LOADED_FROM else "load_generator_weights never ran")

check(len(DIRECTIONAL) == 2, "both directional policies reloaded",
      f"got {len(DIRECTIONAL)}")

targets = sorted(float(dp.target_c) for dp in DIRECTIONAL)
check(targets == [0.0, 1.0], "both directions present (target_c 0.0 and 1.0)",
      f"got {targets}")

for dp in DIRECTIONAL:
    tag = "n_to_p" if float(dp.target_c) == 1.0 else "p_to_n"
    # The policy must match the env eval rebuilt — this is the assertion that
    # would have caught an obs_dim / ACTION_DIM drift between train and eval.
    obs_dim = dp.env.get_observation().shape[0]
    trunk_in = dp.policy.shared[0].in_features
    check(trunk_in == obs_dim, f"[{tag}] policy obs_dim matches the rebuilt env",
          f"policy={trunk_in}, env={obs_dim}")
    check(dp.policy.actor.out_features == config.ACTION_DIM,
          f"[{tag}] policy action_dim matches config",
          f"policy={dp.policy.actor.out_features}, config={config.ACTION_DIM}")
    check(len(dp.env._far_indices_train) > 0, f"[{tag}] far pool non-empty",
          f"{len(dp.env._far_indices_train)} train / {len(dp.env._far_indices_eval)} held out")

# ── artefacts, in the WRITE directory ────────────────────────────────────────
for rel in ("eval_metrics.json", "counterfactual_diff_n_to_p.png",
            "counterfactual_diff_p_to_n.png"):
    p = os.path.join(config.CKPT_DIR, rel)
    check(os.path.exists(p), f"artefact written: {rel}", p)

# ── metric sanity: present and in range, NOT that the values are good ────────
metrics_path = os.path.join(config.CKPT_DIR, "eval_metrics.json")
if os.path.exists(metrics_path):
    import json
    with open(metrics_path) as f:
        m = json.load(f)
    cv = m.get("CV", {})
    vals = {k: cv.get(k) for k in ("mean", "n_to_p", "p_to_n")}
    in_range = all(v is not None and 0.0 <= v <= 1.0 for v in vals.values())
    check(in_range, "CV values present and within [0, 1]", str(vals))
    fid = m.get("FID", {})
    fid_ok = all(
        fid.get(k) is not None and fid[k] == fid[k] and fid[k] >= 0.0
        for k in ("negative_group", "positive_group")
    )
    check(fid_ok, "FID values present and non-negative (finite)",
          f"neg={fid.get('negative_group')}, pos={fid.get('positive_group')}")

# ── safety: the source checkpoints must be untouched ─────────────────────────
src_after = _snapshot(SMOKE_EVAL_SRC)
changed = [p for p in set(SRC_BEFORE) | set(src_after)
           if SRC_BEFORE.get(p) != src_after.get(p)]
check(not changed, "source checkpoint directory left untouched",
      f"{SMOKE_EVAL_SRC} unchanged" if not changed else f"MODIFIED: {changed[:5]}")

print("\n" + "-" * 78)
summary = f"  src={SMOKE_EVAL_SRC}   elapsed={elapsed / 60:.1f} min"
if torch.cuda.is_available():
    summary += f"   peak GPU={torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB"
print(summary)
print("-" * 78)

if failures:
    print(f"\n✗ EVAL SMOKE TEST FAILED — {len(failures)} problem(s):")
    for f in failures:
        print(f"    - {f}")
    sys.exit(1)

print("\n✓ EVAL SMOKE TEST PASSED — checkpoints reload cold and the full eval")
print("  path runs end to end. Metrics from this run are NOT meaningful")
print("  (n_eval and T_DIFF slashed); this only proves the wiring.")
