"""
config.py
=========
Central configuration: all hyperparameters, paths, and constants.
Import this module everywhere instead of scattering magic numbers.
"""

import torch

# ─────────────────────────────────────────────
# Hardware
# ─────────────────────────────────────────────
DEVICE = "cuda"
DTYPE  = torch.float16

# ─────────────────────────────────────────────
# Model paths / IDs
# ─────────────────────────────────────────────
CLASSIFIER_CKPT    = "checkpoint/classifier.pt"
YOLOV8_WEIGHTS     = "yolov8n.pt"
DIFFUSION_MODEL_ID = "runwayml/stable-diffusion-v1-5"

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
# PPO environment
# ─────────────────────────────────────────────
BUDGET_T            = 25
EPSILON_STOP        = 0.08
ETA_OBJ             = 0.5
FAR_THRESHOLD       = 0.5
ENV_SCAN_BATCH_SIZE = 64

LAM_PROG   = 1.5
LAM_VAL    = 2.0
LAM_FID    = 0.5
LAM_PRES_R = 0.5
LAM_BUD    = 0.1
LAM_STOP   = 8.0
LAM_HIT    = 6.0
LAM_CLOSE2 = 2.5

# Number of generator refinements applied per PPO environment step.
ENV_REFINEMENT_PASSES = 2

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
N_DENOISE_EPOCHS   = 15

DENOISE_LR         = 1e-4
LOG_EVERY_N_STEPS  = 20

LORA_R       = 8
LORA_ALPHA   = 16
LORA_DROPOUT = 0.05
LORA_TARGETS = ["to_q", "to_k", "to_v", "to_out.0"]

# ─────────────────────────────────────────────
# Checkpoint output
# ─────────────────────────────────────────────
CKPT_DIR          = "checkpoints"
CKPT_POLICY       = "checkpoints/ppo_policy.pt"
CKPT_UNET         = "checkpoints/unet_finetuned.pt"
CKPT_COND_PROJ    = "checkpoints/cond_proj.pt"