"""
config.py
=========
Central configuration: all hyperparameters, paths, and constants.
Import this module everywhere instead of scattering magic numbers.
"""

import os

import torch

# ─────────────────────────────────────────────
# Hardware
# ─────────────────────────────────────────────
DEVICE = "cuda"
DTYPE  = torch.float16

# Worker processes for every dataset-scanning DataLoader (the env far-pool
# scan, metrics.scan_groups, eval_main._pick_query_images).
#
# WHY THIS IS A KNOB. torch DataLoader workers allocate POSIX semaphores in
# /dev/shm. SLURM cgroups cap /dev/shm per node, and on a small-cap node the
# worker spawn dies with a bare "OSError: [Errno 28] No space left on device"
# out of multiprocessing.synchronize.SemLock — which reads like a full disk
# and has nothing to do with disk. Job 17669486 hit exactly this on hpc-n868
# while the identical code passed on the V100 nodes.
#
# Override per submission without touching code:
#     PPO_NUM_WORKERS=0 sbatch run_smoke_test_eval.slurm
# 0 loads in-process: slower DICOM decode, no shared memory needed at all.
DATALOADER_NUM_WORKERS = int(os.environ.get("PPO_NUM_WORKERS", "2"))

# ─────────────────────────────────────────────
# Model paths / IDs
# ─────────────────────────────────────────────
CLASSIFIER_CKPT    = "checkpoint/classifier.pt"
YOLOV8_WEIGHTS     = "yolov8n.pt"
DIFFUSION_MODEL_ID = "runwayml/stable-diffusion-v1-5"

# ─────────────────────────────────────────────
# Guidance classifier f(·)
# ─────────────────────────────────────────────
# Members of the ENSEMBLE that guides generation, terminates episodes, and
# defines the Xn/Xp split (src/xrv_classifier.py, wired in pipeline.build_common).
#
# WHY AN ENSEMBLE: held-out validation (job 17649721) showed counterfactuals
# optimised against the single RSNA model do not transfer to classifiers that
# never guided them — transfer ratio 0.13/0.14 for p→n, with the agreement
# control passing. That direction was fooling f(·), not removing disease.
# Adversarial directions rarely survive across independently trained models,
# so requiring consensus makes the adversarial shortcut a much narrower target.
#
# MEMBERSHIP RULE: anything listed here is NO LONGER a valid held-out
# validator. Keep heldout_validation.py's DEFAULT_VALIDATORS disjoint from
# this list. Currently held out: mimic_ch, pc, nih.
#   rsna -> in-domain match for the RSNA dataset, but only 2 output heads
#   chex -> 11 heads; multi-task training resists the global-brightness
#           shortcut this module's docstring warns about
# Cost is ~1 extra classifier fwd+bwd per guided step per additional member.
# ─────────────────────────────────────────────
# Dataset domain
# ─────────────────────────────────────────────
# Selects the images, the pathology f(.) targets, the guidance / validator
# models and which clinical metrics apply — all four move together. See
# domains/base.py. Registered: "rsna_pneumonia", "node21_nodule",
# "cbis_ddsm_breast".
#
# Adding a dataset is a new module in domains/ plus a line in domains._REGISTRY.
# It is NOT an edit to ingestion.py, src/xrv_classifier.py or src/metrics.py.
DATASET_DOMAIN = "cbis_ddsm_breast"

# Root of an extracted NODE21 release (grand-challenge registration required,
# so there is no auto-download). Only read by domains/node21_nodule.py.
NODE21_ROOT = ""

# ─────────────────────────────────────────────
# Breast / mammography domain (cbis_ddsm_breast)
# ─────────────────────────────────────────────
# Root of the extracted CBIS-DDSM Kaggle mirror
# (awsaf49/cbis-ddsm-breast-cancer-image-dataset). setup_env.sh downloads it on
# the login node and prints the path to put here. Expected layout:
#     <root>/csv/{dicom_info,mass_case_description_train_set,...}.csv
#     <root>/jpeg/<SeriesUID>/<n>.jpg
CBIS_DDSM_ROOT = ""

# Where train_breast_classifier.py and train_lesion_segmenter.py write, and
# where src/breast_classifier.py and src/breast_segmentation.py read from.
# Separate from CKPT_DIR because these are INPUTS to a run (frozen f(.) and the
# metric segmenter), not outputs of one — they are trained once and reused
# across every PPO run and ablation arm.
BREAST_CKPT_DIR = "checkpoint/breast"

# Backbones train_breast_classifier.py trains. The first two become the
# guidance ensemble and the third the held-out validator; see
# domains/cbis_ddsm_breast.py, which is the authority on that split. densenet121
# is deliberately first: GRADCAM_TARGET_LAYER="features" resolves cleanly on it,
# and it is the ensemble's cam_member.
BREAST_BACKBONES = ["densenet121", "efficientnet_b0", "resnet18"]

# Which BreastAnatomySegmenter channel bounds the generator's edits when
# LOCK_BACKGROUND is on.
#   "breast" -> the whole organ. Direct analogue of the chest domains' lung
#               field, and the default.
#   "dense"  -> fibroglandular tissue only. Strictly tighter; lesions live
#               there, so it should cost little coverage while removing a lot
#               of editable fat. An ablation, not an assumed improvement.
BREAST_EDIT_REGION = "breast"

# Subtract the pectoral-muscle triangle from the editable region on MLO views.
# The detector is geometric and approximate (CBIS-DDSM ships no pectoral ground
# truth), but it is only ever used to EXCLUDE area — a loose fit costs the
# policy some editable breast, it can never license an edit outside the organ.
MASK_EXCLUDE_PECTORAL = True

# The weight sets below are the rsna_pneumonia defaults, kept for backwards
# compatibility. The ACTIVE values come from the domain profile — see
# pipeline.build_common — so switching DATASET_DOMAIN switches them too.
GUIDANCE_ENSEMBLE_WEIGHTS = [
    "densenet121-res224-rsna",
    "densenet121-res224-chex",
]

# REVERTED 2026-08-07 to False, i.e. back to v1's single-model f(·).
#
# True  -> f(·) is the mean-probability ensemble over GUIDANCE_ENSEMBLE_WEIGHTS.
# False -> f(·) is the single GUIDANCE_SINGLE_WEIGHTS model, which is what the
#          archived checkpoints/archive_v1_rank8/ policies were trained against
#          and what produced the CV=0.975 numbers in checkpoints/eval_metrics.json.
#
# READ THIS BEFORE QUOTING CV. The ensemble exists because held-out validation
# (checkpoints/heldout/heldout_validation.json, job 17649721) showed the
# single-rsna counterfactuals do NOT transfer: strict CV 0.0 on nih and 0.0-0.08
# on chex, transfer ratio 0.13-0.20, with the original-image agreement control
# passing. So with this False, CV measures gradient descent against rsna
# succeeding against rsna, and is NOT evidence the counterfactuals are
# explanatory. Run heldout_validation.py alongside any CV reported from here.
USE_GUIDANCE_ENSEMBLE  = False
GUIDANCE_SINGLE_WEIGHTS = "densenet121-res224-rsna"

# Default root for CXRDataset (src/models.py) — an ImageFolder-style
# {NORMAL,PNEUMONIA}/ layout, unused by the current main.py pipeline.
CXR_DATA_DIR = "data/cxr"

# ─────────────────────────────────────────────
# Image / latent geometry
# ─────────────────────────────────────────────
IMG_SIZE        = 256
LATENT_CHANNELS = 4
LATENT_SIZE     = 32
SEG_INPUT_SIZE  = 512

# ─────────────────────────────────────────────
# Diffusion / generation
# ─────────────────────────────────────────────
T_DIFF     = 50
ALPHA_CLS  = 4.0
ALPHA_PRES = 0.05
LAM_CLS    = 1.0
LAM_PRES_D = 0.5
LAM_CYC    = 0.3

# Background lock (RePaint / Blended-Latent-Diffusion masked blending).
# When True, generate_counterfactual pins everything outside the (dilated,
# feathered) lung mask to the original image at every denoising step, so
# edits stay inside the lung field regardless of `strength`. Set False to
# reproduce the old unconstrained SDEdit behaviour for A/B comparison.
LOCK_BACKGROUND  = True
BG_MASK_DILATION = 2      # latent cells (~2 * IMG_SIZE/LATENT_SIZE px dilation)
BG_MASK_FEATHER  = 1.0    # Gaussian sigma (latent cells) for seam feathering

# ─────────────────────────────────────────────
# Grad-CAM action-space constraint
# ─────────────────────────────────────────────
# HARD structural constraint on WHERE the policy may edit, not a reward term.
# The (dilated, feathered) lung mask is multiplied by a Grad-CAM-derived
# per-location edit budget, and the product gates BOTH the guidance gradient
# and the background-lock composite in generate_counterfactual. Because the
# final denoising step composites against the clean anchor latent
#     z_final = m * z_t + (1 - m) * z_anchor
# the output latent deviation per location is bounded by m — m=0 pins exactly,
# m=1 is fully free, monotone in between. This restricts the action space
# rather than penalising it after the fact.
#
# Ablation toggle. False -> edit_mask is exactly the old dilated+feathered lung
# mask and generate_counterfactual is bit-identical to previous behaviour.
#
# ENABLED 2026-08-01 as ARM 2. Requires the ensemble-aware Grad-CAM path in
# gradcam.ensemble_grad_cam — without it, resolve_target_layer(ensemble) walks
# named_modules() and silently returns members.0's `features`, producing an
# rsna-only CAM labelled as the ensemble's. Verified behaviour, not a guess.
# Watch the degenerate-CAM rate PPOTrainer.train reports at the end of a run.
USE_GRADCAM_MASK = False   # REVERTED 2026-08-07 (was ARM 2)

# Grad-CAM hook target inside the FROZEN classifier. Spatial resolution of the
# xrv DenseNet-121 at its 224 input (cell size quoted for IMG_SIZE=256):
#   "features"             ->  7x7  (~37px cells) — proven default, see
#                              check_classifier_localization.py
#   "features.denseblock3" -> 14x14 (~18px cells) — finer localisation, opt-in
#   "features.denseblock2" -> 28x28 (~9px cells)  — finest, noisiest
# NOTE: the 7x7 default can concentrate edits into a REGION, not onto a lesion
# boundary. If the mask looks too coarse in the overlay diagnostics, step down
# to denseblock3 before touching the floor/gamma knobs.
GRADCAM_TARGET_LAYER = "features"

# m = floor + (1 - floor) * cam^gamma, then Gaussian-blurred at latent res.
#   floor : minimum edit budget retained inside the lung field. 0.0 hard-pins
#           low-attention lung regions (aggressive; the setting that actually
#           makes the constraint bite) at higher CV risk. Raise toward 0.3 if
#           CV degrades.
#   gamma : <1 broadens the mask (permissive), >1 sharpens it (more focal).
#   blur  : kills the blockiness of the upsampled 7x7 CAM.
#   pct   : percentile taken as the CAM's "1.0". Preferred over min-max, which
#           lets a single outlier cell define the scale and flattens the rest.
GRADCAM_MASK_FLOOR = 0.15
GRADCAM_MASK_GAMMA = 0.75
GRADCAM_MASK_BLUR  = 0.8
GRADCAM_NORM_PCT   = 0.99

# Anchor the background lock and the preservation targets (L1 map, object-
# detector features) to the EPISODE's original image instead of the previous
# refinement pass's output.
#
# WHY THIS EXISTS: environment.step() chains generate_counterfactual over
# n_passes, feeding each pass the previous pass's output, and self.x_t carries
# across all BUDGET_T steps. With this False, z0_orig_4d / S_x / O_x are all
# derived from the pass input, so the lock only ever bounds drift relative to
# ONE call ago — up to ~100 chained encode/renoise/denoise/decode round trips
# per episode accumulate diffuse change that no mask constrains. The Grad-CAM
# mask above is substantially weaker without this.
#
# Deliberately a SEPARATE flag from USE_GRADCAM_MASK so the two effects can be
# attributed independently in the ablation matrix.
#
# ENABLED 2026-08-01 as ARM 1 of the locality ablation, on its own. Evidence
# from the eval of job 17648985 (checkpoints/counterfactual_diff_{n_to_p,p_to_n}
# .png): 73.6% / 77.2% of pixels changed at MAE 0.036 / 0.025, with visible
# change OUTSIDE the lung field (background rim, soft tissue, bottom table band,
# foreign-object markers) despite LOCK_BACKGROUND=True — exactly the accumulated
# per-pass drift this flag exists to bound.
#
# ABLATION ORDER — flip these ONE AT A TIME, re-running eval between arms, so
# the effects stay attributable:
#   ARM 1 (now)  ANCHOR_TO_EPISODE_ORIGIN = True    <- this flag
#   ARM 2        + USE_GRADCAM_MASK        = True   (see line ~69)
#   ARM 3        + USE_SPARSITY_REWARD     = True   (see line ~147; note that
#                  USE_SPARSITY_ANNEALING is already True but INERT until this
#                  is set — PPOTrainer logs that fact at startup)
# Expect CV to DROP as these come on. That is the intended direction: CV 0.975
# with 75% of pixels changed is a worse result than CV 0.7 with 5% changed and
# an independent classifier agreeing (see heldout_validation.py).
ANCHOR_TO_EPISODE_ORIGIN = True

# ─────────────────────────────────────────────
# PPO environment
# ─────────────────────────────────────────────
# REDUCED 25 -> 8 alongside the joint stop criterion below. With EDIT_BUDGET_L1
# in play, episodes that previously terminated at step 1 now run toward the
# budget, and 25 steps x ~2.5 refinement passes is a 10-25x cost increase over
# the measured v1 rate (~1s per generate_counterfactual call). 8 keeps a
# genuine multi-step MDP at ~3-5x v1 cost.
# If avg_ep_len pins at exactly this value in the PPO log, the fix is to LOOSEN
# EDIT_BUDGET_L1 — raising BUDGET_T back just buys more diffuse editing.
BUDGET_T            = 25   # REVERTED 2026-08-07 from 8 back to the v1 value

# Episodes run EXACTLY BUDGET_T steps and never terminate early on validity.
#
# WHY: step() used to end an episode the moment `is_valid` went True, i.e. the
# episode length was decided by the environment's judgement rather than by a
# fixed turn count. Two consequences: per-episode cost was unpredictable (the
# thing that makes a 24h walltime impossible to plan against), and the flip
# bonus was a terminal payout the policy could collect and quit on.
#
# NOTE: this is NOT a speed optimisation. Job 17653310 logged avg_ep_len=8.00
# on every update, i.e. `is_valid` never fired and episodes already ran the
# full budget. The change buys predictability, not wall clock.
#
# REWARD SEMANTICS CHANGE: R_hit / R_stop are per-step terms (see
# environment._compute_reward), so with no early exit they now accrue on EVERY
# step where the counterfactual is valid instead of once at termination. The
# objective becomes "reach validity and HOLD it for the rest of the budget",
# which removes the hit-and-quit incentive and matches the EDIT_BUDGET_L1
# rationale below. It also means avg_return is NOT numerically comparable to
# runs from before this flag — compare `frac_valid` in the per-update log line
# instead, not avg_return, when reading against job 17653310 and earlier.
#
# Set False to restore `done = is_valid or step >= T` exactly.
FIXED_EPISODE_LENGTH = False   # REVERTED 2026-08-07: restores v1's
                               # `done = is_valid or step >= T`

EPSILON_STOP        = 0.08
ETA_OBJ             = 0.5
FAR_THRESHOLD       = 0.5
ENV_SCAN_BATCH_SIZE = 64

# Mean pixel-space L1 vs x_orig that a counterfactual must stay UNDER to count
# as valid, in addition to |f(x_cf) - c| < EPSILON_STOP (see
# MedicalCounterfactualEnv._compute_reward).
#
# WHY: with validity defined by classifier score alone, a cheap diffuse wash
# satisfied it in a single step — v1 measured avg_ep_len=1.00 for n→p, which
# made BUDGET_T, GAE and the value head vestigial and left PPO no temporal
# credit assignment to learn from. Requiring the flip to ALSO be small is what
# turns this back into an MDP: the policy has to refine rather than blast.
#
# CALIBRATION: v1 counterfactuals measured MAE 0.042 (n→p) / 0.029 (p→n) over
# 200 images each, so 0.02 is deliberately below what the old policy produced —
# reachable, but not by the v1 strategy. Loosen toward 0.03 if episodes never
# terminate early (see the BUDGET_T note above).
EDIT_BUDGET_L1      = float("inf")   # REVERTED 2026-08-07 from 0.02. v1 had no
                                     # L1 gate on validity at all; inf makes
                                     # budget_ok always True in
                                     # environment._compute_reward, so validity
                                     # keys off |f(x_cf) - c| alone as it did.

# ─────────────────────────────────────────────
# Reward weights
# ─────────────────────────────────────────────
# REBALANCED. The v1 decomposition (reconstructed from job 17648782's logged
# avg_return=33.931 at avg_ep_len=1.00, direction_scale=2.25):
#     R_hit x6 = 6.00 | R_stop x8 = 8.00   -> 14.00 of 15.08  (93%)
#     R_prog x1.5                          ->  1.40           ( 9%)
#     R_pres x0.5                          -> -0.21           (-1.4%)
#     R_fid  x0.5 (on MSE 0.0033)          -> -0.0016         (-0.01%)
# i.e. the flip bonus WAS the reward and preservation was numerically absent,
# which is the direct mechanism behind 66-75% of pixels changing. Target here
# is roughly 40 / 40 / 20 flip / preservation / progress.
#
# These values are a STARTING POINT derived from the above, not tuned. Read the
# actual split off the per-term decomposition PPOTrainer now logs each update
# and correct from there — don't burn a full run guessing.
LAM_PROG   = 1.5
LAM_VAL    = 2.0
# LAM_FID 0.5 -> 2.0 because R_fid switched from MSE to L1 in environment.py
# (L2 prefers many small changes over few large ones — it actively rewarded the
# diffuse failure mode). L1 is ~25x larger numerically at these magnitudes, so
# 2.0 is a modest real increase on top of the rescaling, not a 4x one.
LAM_FID    = 2.0
LAM_PRES_R = 3.0
LAM_BUD    = 0.1
LAM_STOP   = 2.0
LAM_HIT    = 2.0
LAM_CLOSE2 = 2.5

# Number of generator refinements applied per PPO environment step.
ENV_REFINEMENT_PASSES = 2

# ─────────────────────────────────────────────
# Sparsity reward + annealing schedule
# ─────────────────────────────────────────────
# Soft reward term penalising total edit magnitude, ramped in on a delay so the
# policy first learns a reliable flip under the Grad-CAM-constrained action
# space and only then gets pressured toward minimal edits within that space.
#
# Ablation toggles, independent:
#   USE_SPARSITY_REWARD    — include the term at all
#   USE_SPARSITY_ANNEALING — ramp it in vs. apply LAM_SPARSITY_TARGET from
#                            update 0 (the comparison that demonstrates the
#                            annealing itself matters). Inert unless
#                            USE_SPARSITY_REWARD is True; PPOTrainer warns
#                            rather than failing silently if set alone.
# ENABLED 2026-08-01. Nothing in v1 penalised diffuse editing with a
# sparsity-inducing norm: R_fid was an L2 contributing one part in ten thousand
# of the return. Measured edit concentration was 0.28-0.30 against a 0.05
# uniform baseline (top-5% of pixels' share of |Δ|), i.e. mostly diffuse.
# PPOTrainer used to log "the annealing schedule is inert" at startup; that
# line should stop firing now.
USE_SPARSITY_REWARD    = False   # REVERTED 2026-08-07 (was ARM 3)
USE_SPARSITY_ANNEALING = True

# Target weight. Same scale as LAM_FID, which is worth noting: R_fid is already
# an L2 (MSE) penalty on total edit magnitude at LAM_FID=0.5, so a magnitude
# penalty is PARTLY in the reward already. The new term is not redundant —
# L1 is sparsity-inducing while L2 is not (L2 prefers many small changes over
# few large ones, i.e. exactly the diffuse behaviour being targeted) — but the
# two do overlap, so treat LAM_FID as a confound when reading the ablation.
# RAISED 0.5 -> 2.0 in the same rebalance that cut LAM_HIT/LAM_STOP. At the old
# value, against a flip bonus of 14, this term could never bind.
LAM_SPARSITY_TARGET = 2.0

# Ramp window, as FRACTIONS of total PPO updates rather than absolute update
# indices. Total updates = ceil(N_EPISODES / episodes_per_update), and
# N_EPISODES is explicitly an unbenchmarked placeholder (see its comment
# below) — absolute indices would silently land in the wrong place once it is
# re-tuned from the real V100 timing. Resolved and logged at train() entry.
#   [0, START)   -> lambda_sparsity = 0            (validity only)
#   [START, END) -> ramp 0 -> LAM_SPARSITY_TARGET
#   [END, 1]     -> LAM_SPARSITY_TARGET
SPARSITY_RAMP_START_FRAC = 0.2
SPARSITY_RAMP_END_FRAC   = 0.6
SPARSITY_RAMP_SHAPE      = "linear"   # "linear" | "cosine"

# Penalise only edit mass OUTSIDE the attended region — a (1-cam)-weighted L1
# instead of a plain total-magnitude L1. Closer to the actual goal (concentrate
# edits where the classifier attends) but NOT what was specified, so off by
# default. Requires USE_GRADCAM_MASK; silently equivalent to plain L1 without
# a cached CAM.
#
# ENABLED 2026-08-01 as ARM 3, together with USE_GRADCAM_MASK above (this is
# inert without a cached CAM — it silently degrades to a plain unweighted L1).
# Rationale from the measurements: edit concentration was 0.282/0.297 against a
# 0.05 uniform baseline, i.e. there IS a concentrated core but most of the edit
# mass sits in a diffuse tail. Penalising only the mass OUTSIDE the attended
# region targets that tail directly, where a plain L1 also penalises the core
# you want to keep.
SPARSITY_OUTSIDE_CAM_ONLY = True

# ─────────────────────────────────────────────
# CV monitoring safeguard during the sparsity ramp
# ─────────────────────────────────────────────
# Far-pool images withheld from rollout sampling and used ONLY for the CV
# probe. Must be genuinely held out: _sample_query_image draws from the
# training split only, so enabling this shrinks the rollout pool by this many
# images and changes the sampling stream (runs are not bit-comparable to
# earlier ones).
#
# SIZING MATTERS: CV is a fraction over this many images, so its resolution is
# 1/N. At N=8 a single image flipping moves CV by 12.5 points, far above any
# sensible tolerance — the ramp freeze would latch on noise in essentially
# every run. N=24 gives 4.2 points per image, so an 8-point tolerance sits
# safely above the noise floor at two images' worth.
ENV_EVAL_HOLDOUT_N = 24

# How often to probe CV, and the drop from the pre-ramp baseline that latches
# the freeze. COST: each probe runs ENV_EVAL_HOLDOUT_N full policy episodes
# against the 8 episodes a training update consumes — one probe is ~3 updates
# of rollout compute. Probes are therefore restricted to the ramp window (plus
# a short lead-in for the baseline) and use a shortened budget.
CV_CHECK_EVERY_N_UPDATES = 10
# LOWERED 12 -> 4. This is applied as min(CV_CHECK_MAX_STEPS, env.T) in
# PPOTrainer._measure_holdout_cv, so at BUDGET_T=8 the old value of 12 resolved
# to 8 and the probe ran at FULL length — the "~halves per-episode probe cost"
# this comment used to claim never happened. It must be strictly below BUDGET_T
# to do anything; 4 is half of 8. Keep the two coupled if BUDGET_T changes.
CV_CHECK_MAX_STEPS       = 4      # must be < BUDGET_T or this knob is inert
CV_DROP_TOLERANCE        = 0.08   # 8 points ≈ two images at N=24

# ─────────────────────────────────────────────
# GAE
# ─────────────────────────────────────────────
GAMMA      = 0.99
GAE_LAMBDA = 0.95
DELTA_MIX  = 0.2

# ─────────────────────────────────────────────
# PPO training
# ─────────────────────────────────────────────
PPO_CLIP     = 0.2
C_VALUE      = 0.5
C_ENTROPY    = 0.01
LAM_FEAS_AUX = 0.1
LAM_PROG_AUX = 0.1
LAM_STOP_AUX = 0.1
LR           = 3e-4

# How often PPOTrainer.train() writes a resumable checkpoint
# (ckpt_dir/<label>/ppo_latest.pt: policy + optimizer + update counters + the
# sparsity-ramp/CV-safeguard state). Mirrors the per-epoch resume that
# factory/finetuning.py already does for LoRA.
#
# WHY THIS EXISTS: publicgpu caps --time at 24h and a full direction does not
# reliably fit. Job 17653310 was killed at the walltime having completed ~30 of
# 100 updates of n_to_p and left NOTHING recoverable — there was no PPO
# checkpoint until save_checkpoints() at the very end of the direction. With
# this, a direction that needs 30h is two submissions rather than a loss.
#
# 5 updates = 40 episodes, which bounds what a kill can cost to well under an
# hour on any GPU in run_ppo.slurm's constraint set. Saving is ~1MB (the conv
# trunk policy is only ~198k params), so the cadence is not I/O-limited — lower
# it freely if you want a tighter bound.
PPO_CKPT_EVERY_N_UPDATES = 5

# Hinge penalty on pre-squash actor outputs beyond ACT_SAT_THRESHOLD, added to
# L_total in PPOTrainer.update().
#
# WHY: every action dim is squashed through a sigmoid in environment.step(). In
# v1 the actor's mean output drifted to ~±5.5, where sigmoid is flat: guidance
# diagnostics measured guidance_bias = 1.2976 ± 0.0054 (n→p) and 0.7005 ±
# 0.0012 (p→n) — the two ENDS of its (0.70, 1.30) range — across 20 episodes
# sampled WITH exploration noise at std≈0.61. The squashing function absorbed
# all of it, so the policy was effectively deterministic and state-independent,
# a bang-bang solution to the flip-dominated reward above.
#
# Beyond |logit| = 3 sigmoid is already ≈0.95, so the responsive region is left
# completely free and only the dead tail is penalised. Verify via
# guidance_bias_std in guidance_diagnostics_*.json: > 0.05 means the policy is
# actually varying with the input.
ACT_SAT_THRESHOLD = 3.0
LAM_ACT_SAT       = 1e-2

# ─────────────────────────────────────────────
# Policy trunk architecture
# ─────────────────────────────────────────────
# "conv" — small conv encoder over the latent/mask grids  (~200k params)
# "mlp"  — the original Linear(obs_dim, 512) -> Linear(512, 256)  (~37.9M)
#
# WHY THIS CHANGED: the MLP trunk was 37,750,784 of the policy's 37,885,462
# parameters — 99.6% — fit on roughly 1,000 environment steps. It consumed the
# observation as one flat 73,731-vector, of which the 256x256 segmentation mask
# alone is 65,536 entries (88.9%), with no spatial structure at all. That is
# the direct reason v1's policy learned a constant, state-independent action:
# it had nowhere near the data to fit that layer, so it fit a bias.
#
# The conv trunk reshapes the SAME flat observation internally — get_observation()
# is untouched, so environment.py, pipeline.py and inference.py are unaffected
# and load_directional_policy still reads obs_dim off a live env.reset().
#
# Keep "mlp" available: the archived rank-8 policies under
# checkpoints/archive_v1_rank8/ have that trunk and will not load into a conv
# policy.
POLICY_TRUNK = "mlp"   # REVERTED 2026-08-07: the archived v1 policies carry the
                       # Linear(73731, 512) trunk. "conv" cannot load them.

# ─────────────────────────────────────────────
# Experiment
# ─────────────────────────────────────────────
TARGET_C   = 1.0

# REVISED for actual hardware: single Tesla V100-32GB (not the H200s
# visible in shared-node nvidia-smi output — those weren't this job's
# allocation). V100 is ~3-5x slower than H200 for this kind of workload,
# and PPO episodes are the more expensive phase (each step runs guided
# DDIM sampling through the full UNet, ENV_REFINEMENT_PASSES times).
# 2000 was sized for H200 throughput and is very likely NOT safe within
# a 24h wall-clock limit on a single V100.
#
# ACTION REQUIRED before the real run: benchmark ~16 episodes first
# (episodes_per_update=8 -> 2 updates), read seconds/episode off the
# "Update N | episodes=..." log lines, then set:
#   N_EPISODES = int((ppo_budget_hours * 3600) / seconds_per_episode)
# 800 below is a conservative placeholder until you have that number —
# don't treat it as tuned.
N_EPISODES = 800

# Bumped 8 -> 9: new action dim controls guidance_ramp_start_frac in
# generate_counterfactual (see environment.py step() and generator.py).
ACTION_DIM = 9

# ─────────────────────────────────────────────
# LoRA fine-tuning
# ─────────────────────────────────────────────

# REVISED for V100-32GB: 32 was sized assuming H200 (143GB) headroom and
# will very plausibly OOM on 32GB once VAE + UNet + LoRA adapters +
# classifier + seg_net + obj_det are all resident simultaneously. Start
# here; if a smoke-test epoch runs cleanly with room to spare (check
# nvidia-smi during the run), try 12 or 16 on the next attempt. Don't
# jump straight back to 32 on this GPU.
DENOISE_BATCH_SIZE = 8

# REVISED: same story as N_EPISODES above — 30 was an H200-scale guess.
# ACTION REQUIRED: benchmark 1 epoch first, read seconds/epoch off the
# "[finetuning] epoch 0 done | ..." log line, then set:
#   N_DENOISE_EPOCHS = int((lora_budget_hours * 3600) / seconds_per_epoch)
# 15 below is a conservative placeholder pending that number. Given PPO
# is the pricier phase per hour on this GPU, this revision leans the
# 24h split further toward PPO than the original 30/2000 split did —
# roughly 8h LoRA / 14h PPO / ~2h buffer (checkpointing, eval, startup)
# as a starting allocation, adjust once you have real per-unit timings.
#
# RAISED 15 -> 30 alongside the LoRA capacity change below. The archived rank-8
# checkpoint had reached epoch 30, so retraining the LARGER adapter for only 15
# would compare capacity against training time and make a null result
# uninterpretable. 30 matches what the current generator actually received.
N_DENOISE_EPOCHS   = 30

DENOISE_LR         = 1e-4
LOG_EVERY_N_STEPS  = 20

# RAISED 8 -> 16 with a widened target set (Phase 3, 2026-08-01).
#
# HYPOTHESIS: p→n is the direction held-out validation found adversarial
# (transfer ratio 0.13/0.14 vs 0.21/0.59 for n→p). That asymmetry is physically
# sensible — adding diffuse opacity to a clear lung is easy, while REMOVING an
# established consolidation means synthesising plausible parenchyma, vessels
# and texture underneath. That is generative capacity, not reward shaping, and
# r=8 on attention projections alone is a small adapter for it.
#
# TARGET SET, verified against the actual SD1.5 UNet architecture:
#   to_q/to_k/to_v/to_out.0   32 each, ~102k params/module   (the original set)
#   ff.net.0.proj             16,      822k/module           GEGLU input proj
#   ff.net.2                  16,    1,640k/module           FF output, largest
#   proj_in / proj_out        16 each, 103k/module           Conv2d, peft OK
# Expected trainable ~6M, up from 1,594,368. build_generator already calls
# print_trainable_parameters() — if it still reports 1.59M, the strings below
# did not match anything.
#
# DELIBERATELY EXCLUDED: ResNet conv1/conv2 (22 each, 922k/module). That is
# where local texture lives and is the escalation if r=16 proves insufficient,
# but it roughly quadruples trainable params and the overfitting risk on 26,684
# images.
#
# TRAP: peft matches target_modules by DOTTED SUFFIX. "ff.net.2" must be spelled
# in full — a bare "2" would match unrelated modules. Same reason "to_out.0" was
# already fully qualified rather than "0".
#
# BREAKING: a rank change makes every existing generator artefact unloadable.
# finetuning.py resumes via set_peft_model_state_dict and pipeline.py loads
# unet_finetuned.pt — both fail on shape mismatch. The rank-8 artefacts are
# archived under checkpoints/archive_v1_rank8/.
LORA_R       = 8
LORA_ALPHA   = 16
LORA_DROPOUT = 0.05
LORA_TARGETS = ["to_q", "to_k", "to_v", "to_out.0"]

# REVERTED 2026-08-07 from rank 16 / widened targets back to the rank-8 v1
# settings, so checkpoints/archive_v1_rank8/ loads. The rank-16 artefacts are
# archived under checkpoints/archive_v2_rank16/ — restoring them means putting
# these four lines back to:
#     LORA_R       = 16
#     LORA_TARGETS = ["to_q", "to_k", "to_v", "to_out.0",
#                     "ff.net.0.proj", "ff.net.2", "proj_in", "proj_out"]

# ─────────────────────────────────────────────
# Checkpoint output
# ─────────────────────────────────────────────
# PER-DOMAIN, and this is a correctness requirement rather than tidiness.
# main.py resumes LoRA fine-tuning from CKPT_DIR/finetuning_latest.pt and PPO
# from CKPT_DIR/<label>/ppo_latest.pt whenever those files exist. With one
# shared directory, training a second dataset would (a) silently resume from
# the FIRST dataset's weights and (b) overwrite its policies — losing them,
# since the generator artefacts are gitignored.
#
# rsna_pneumonia keeps the bare "checkpoints" path so every existing artefact
# and every run_*.slurm script keeps working unchanged. Any other domain gets
# its own subdirectory. Override explicitly with PPO_CKPT_DIR when needed
# (e.g. to run an ablation arm without touching the baseline).
#
# Resolved from the DATASET_DOMAIN string rather than by importing `domains`,
# because domains/ imports config and that would be circular.
CKPT_DIR = os.environ.get("PPO_CKPT_DIR") or (
    "checkpoints"
    if DATASET_DOMAIN == "rsna_pneumonia"
    else os.path.join("checkpoints", DATASET_DOMAIN)
)
CKPT_POLICY       = os.path.join(CKPT_DIR, "ppo_policy.pt")
CKPT_UNET         = os.path.join(CKPT_DIR, "unet_finetuned.pt")
CKPT_COND_PROJ    = os.path.join(CKPT_DIR, "cond_proj.pt")