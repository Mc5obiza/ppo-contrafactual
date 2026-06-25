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
T_DIFF     = 25
ALPHA_CLS  = 2.0
ALPHA_PRES = 0.1
LAM_CLS    = 1.0
LAM_PRES_D = 0.5
LAM_CYC    = 0.3

# ─────────────────────────────────────────────
# PPO environment
# ─────────────────────────────────────────────
BUDGET_T            = 25
EPSILON_STOP        = 0.05
ETA_OBJ             = 0.5
FAR_THRESHOLD       = 0.5
ENV_SCAN_BATCH_SIZE = 64

LAM_PROG   = 1.0
LAM_VAL    = 2.0
LAM_FID    = 0.5
LAM_PRES_R = 0.5
LAM_BUD    = 0.1
LAM_STOP   = 5.0

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
N_EPISODES = 50
ACTION_DIM = 8

# ─────────────────────────────────────────────
# LoRA fine-tuning
# ─────────────────────────────────────────────
N_DENOISE_EPOCHS   = 5
DENOISE_BATCH_SIZE = 4
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
