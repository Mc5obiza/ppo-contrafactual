# Technical Interview Guide
## RL-Guided Latent Diffusion for Visual Counterfactual Explanations in Chest Radiography

**Repository analysed:** `ppo-contrafactual/`
**Candidate:** AI research intern (3-month supervised internship)
**Interviewer role:** Principal AI Research Engineer (CV / RL / GenAI)
**Purpose:** locate the *boundary* of the candidate's understanding — not to audit code quality, not to prove authorship.

---

## Table of Contents

1. [Repository Analysis](#1-repository-analysis)
   - [1.1 Architecture overview](#11-architecture-overview)
   - [1.2 Module inventory](#12-module-inventory)
   - [1.3 Dependency graph](#13-dependency-graph)
   - [1.4 Training pipeline](#14-training-pipeline)
   - [1.5 Inference / evaluation pipeline](#15-inference--evaluation-pipeline)
   - [1.6 RL environment](#16-rl-environment)
   - [1.7 Reward implementation](#17-reward-implementation)
   - [1.8 Diffusion implementation](#18-diffusion-implementation)
   - [1.9 Evaluation pipeline & metrics](#19-evaluation-pipeline--metrics)
   - [1.10 Configuration system](#110-configuration-system)
   - [1.11 Dataset loading](#111-dataset-loading)
   - [1.12 External libraries](#112-external-libraries)
   - [1.13 Most complex files / core research logic](#113-most-complex-files--core-research-logic)
   - [1.14 Bottlenecks and bug-prone regions](#114-bottlenecks-and-bug-prone-regions)
   - [1.15 Ground-truth numbers on disk](#115-ground-truth-numbers-on-disk-use-these-to-check-answers)
2. [Section 1 — High-Level Understanding](#section-1--high-level-understanding)
3. [Section 2 — Original Research Problem (GAN → Diffusion)](#section-2--original-research-problem-gan--diffusion)
4. [Section 3 — Diffusion](#section-3--diffusion)
5. [Section 4 — PPO](#section-4--ppo)
6. [Section 5 — Reward Function](#section-5--reward-function)
7. [Section 6 — Code Ownership](#section-6--code-ownership)
8. [Section 7 — Debugging](#section-7--debugging)
9. [Section 8 — Hyperparameters](#section-8--hyperparameters)
10. [Section 9 — Experiments](#section-9--experiments)
11. [Section 10 — Research Thinking](#section-10--research-thinking)
12. [Repository Walkthrough](#repository-walkthrough)
13. [Ownership Questions](#ownership-questions)
14. [Deep Technical Questions](#deep-technical-questions)
15. [Beyond The Project](#beyond-the-project)
16. [Interview Evaluation Rubric](#interview-evaluation-rubric)

**Total estimated time:** 150–210 min for the full guide. A 90-minute interview should use Sections 1, 3, 4, 5, 6, plus 6–8 items from Ownership and 4–6 from Deep Technical.

---

# 1. Repository Analysis

## 1.1 Architecture overview

The system generates **medical counterfactuals**: given a chest X-ray, produce a minimally-edited version that flips a pneumonia classifier's decision, while preserving patient identity and anatomy. Three layers stack:

```
                 ┌─────────────────────────────────────────────────┐
                 │  PPO policy  (πθ, ~200k params, conv trunk)     │
                 │  emits 9 continuous knobs per step              │
                 └───────────────────────┬─────────────────────────┘
                                         │ strength, cls_scale, pres_scale,
                                         │ tau, 2× extra-pass bits,
                                         │ strength_boost, guidance_bias,
                                         │ ramp_start_frac
                 ┌───────────────────────▼─────────────────────────┐
                 │  Generator: SD1.5 VAE + UNet + LoRA(r=16)       │
                 │  guided DDIM (50 steps) + background lock       │
                 │  + Grad-CAM edit-budget mask                    │
                 └───────────────────────┬─────────────────────────┘
                                         │ x_cf
                 ┌───────────────────────▼─────────────────────────┐
                 │  Frozen critics: f(·) 2-member DenseNet ensemble│
                 │  S(·) PSPNet lung seg,  O(·) YOLOv8n backbone   │
                 └─────────────────────────────────────────────────┘
```

The key architectural decision: **PPO does not control the diffusion latents directly.** It controls the *hyperparameters of the sampling procedure* — an 9-dimensional continuous action that reparameterises `generate_counterfactual()` at every environment step. The diffusion model itself is frozen apart from LoRA adapters trained separately in a prior phase.

Two independent policies are trained: `n_to_p` (target_c = 1.0) and `p_to_n` (target_c = 0.0), because `MedicalCounterfactualEnv` is constructed around one fixed `target_c`.

## 1.2 Module inventory

| Path | LoC | Role |
|---|---:|---|
| `config.py` | 516 | Single source of hyperparameters + extensive change-log commentary |
| `main.py` | 379 | Training entry point (Steps 1–5), `--direction {n_to_p,p_to_n,both}` |
| `eval_main.py` | 511 | Evaluation entry (Steps 6/6B), `--stage {generate,reduce,all}`, sharding |
| `pipeline.py` | 268 | `build_common()`, `load_generator_weights()`, `load_directional_policy()` |
| `ingestion.py` | 176 | Kaggle RSNA download, `RSNACXRDataset` (DICOM → [0,1] 256²) |
| `heldout_validation.py` | 510 | Adversarial-vs-semantic diagnostic on off-guidance classifiers |
| `check_classifier_localization.py` | ~140 | Grad-CAM sanity check on the classifier |
| `smoke_test.py` / `smoke_test_eval.py` | 328 / 328 | Cheap end-to-end crash tests for train / eval paths |
| `preflight_gpu.py` | ~80 | GPU capability check before a SLURM job commits |
| `factory/environment.py` | 674 | `MedicalCounterfactualEnv` — Gym-style env, reward, `generate_via_policy` |
| `factory/ppo.py` | 994 | `Transition`, `PPOActorCritic`, `compute_gae`, `PPOTrainer` |
| `factory/finetuning.py` | 333 | LoRA denoising fine-tune loop, balanced sampler, per-epoch resume |
| `factory/evaluate.py` | 141 | Reward-trace / trajectory plots, `save_checkpoints` |
| `src/generator.py` | 691 | `PretrainedLatentDiffusionGenerator` — the core research file |
| `src/metrics.py` | 1336 | CV / FID / FOP / IP / edit-concentration / CTR / SCP + `run_full_evaluation` |
| `src/gradcam.py` | 318 | Shared Grad-CAM, `DegenerateCAMError`, `ensemble_grad_cam` |
| `src/xrv_classifier.py` | 206 | `XRVPneumoniaClassifier`, `EnsemblePneumoniaClassifier` |
| `src/models.py` | 280 | PSPNet segmenter, YOLOv8 feature extractor, legacy ResNet-18 loader |
| `src/inference.py` | 147 | `DirectionalPolicy`, per-image seeding for shard-invariance |
| `src/diff_viz.py` | 360 | Pixel-diff 4-panel + bidirectional comparison figures |
| `*.slurm` | — | 7 batch scripts; `run_ppo.slurm` is a 2-task array, `run_eval_cf.slurm` an 8-task array |

## 1.3 Dependency graph

```
config.py ──────────────────────────── (imported by everything)

ingestion.py ─┐
              ├──► pipeline.build_common() ──► PipelineContext
src/models.py ┤                                 │
src/xrv_classifier.py ┤                         │
src/generator.py ─────┘                         │
        ▲                                       │
        └── src/gradcam.py                      │
                                                │
             ┌──────────────────────────────────┴────────────────────────┐
             ▼                                                            ▼
        main.py                                                     eval_main.py
             │                                                            │
   factory/finetuning.py (Step 4)                        pipeline.load_directional_policy
             │                                                            │
   factory/environment.py ◄──────────────────────────────────────────────┤
             │                                                            │
   factory/ppo.py (PPOTrainer)                                 src/inference.py
             │                                                            │
   factory/evaluate.py (save_checkpoints)                       src/metrics.py
                                                                src/diff_viz.py

heldout_validation.py ──► pipeline.* + src/metrics.{scan_groups, score_batch,
                                       counterfactual_validity, edit_concentration,
                                       current_cfg_stamp}
```

Notable coupling facts to probe:
- `factory/ppo.py` imports `src.metrics.counterfactual_validity` **inside** `_measure_holdout_cv()` (deliberate late import to avoid a cycle).
- `factory/environment.py` never imports the trainer; `lambda_sparsity` is *pushed in* one-way by `PPOTrainer` before each rollout.
- `src/metrics.py` imports `src.inference` inside `run_full_evaluation` for the same reason.

## 1.4 Training pipeline (`main.py`)

1. **Step 1** `ingestion.build_dataset()` — kagglehub download, `RSNACXRDataset` (~26 684 patients).
2. **Step 2** `build_ensemble_classifier()` (rsna + chex DenseNet-121), `build_segmentation_net()` (xrv PSPNet), `build_object_detector()` (YOLOv8n, first 10 layers).
3. **Step 3** `build_generator()` — SD1.5 VAE + UNet, PEFT LoRA r=16 on `to_q/to_k/to_v/to_out.0/ff.net.0.proj/ff.net.2/proj_in/proj_out`, trainable params upcast to fp32.
4. **Step 4** `run_lora_finetuning()` — 30 epochs, ε-prediction MSE, AMP fp16 + GradScaler, `WeightedRandomSampler` for class balance, per-epoch atomic checkpoint.
5. **Step 5** per direction: build env → `env.reset()` to read `obs_dim` → `PPOActorCritic` → `PPOTrainer.train(n_episodes=800, episodes_per_update=8)` → reward trace → `run_guidance_diagnostics(20 episodes)` → `save_checkpoints()` into `checkpoints/<label>/`.

Wall-clock survival machinery is a first-class citizen: `PPO_WALLCLOCK_BUDGET_S` (22 h of a 24 h SLURM cap), resumable `ppo_latest.pt` every 5 updates, and `train()` returning `False` so `main.py` refuses to write final artefacts for a half-trained policy.

## 1.5 Inference / evaluation pipeline (`eval_main.py`)

- `--stage generate --shard K --num-shards 8`: every shard independently re-runs `scan_groups`, slices `[lo, hi)`, generates counterfactuals with per-image seeding, writes `cf_shard_KKK.pt` including a `current_cfg_stamp()`.
- `--stage reduce`: loads all shards, hard-fails on missing shards / `num_shards` disagreement / `n_eval` disagreement / config drift between shards / config drift vs current `config.py`, then calls `run_full_evaluation(..., xn_list=…, xc_np=…)` on the precomputed path.
- `--stage all`: single-process; per the module docstring, ~4.4 min/image × 400 images ≈ 29 h, which does not fit the 24 h partition cap — smoke tests only.

## 1.6 RL environment (`factory/environment.py`)

- **Far pool**: `_build_far_index_pool()` scans the whole dataset once and keeps indices with `|f(x) − c| > 0.5`. First 24 indices are sliced off as `_far_indices_eval` for the CV safeguard probe; the rest are `_far_indices_train`.
- **Observation** (`get_observation`): `concat[z_orig(4096), z_t(4096), step/T(1), d_t(1), c(1), seg_mask(65 536)] = 73 731` floats.
- **Action** (9 dims, all sigmoid-squashed in `step()`):
  | idx | name | mapping | range |
  |---|---|---|---|
  | 0 | `strength` | `0.2 + 0.5σ` | (0.2, 0.7) |
  | 1 | `cls_scale` | `0.3 + 0.9σ` | (0.3, 1.2) |
  | 2 | `pres_scale` | `0.3 + 1.2σ` | (0.3, 1.5) |
  | 3 | `tau` | `σ` | (0, 1) — gates `R_stop` and `DELTA_MIX` |
  | 4,5 | extra passes | `σ > 0.5` | {0,1} each |
  | 6 | `strength_boost` | `0.85 + 0.30σ` | (0.85, 1.15) |
  | 7 | `guidance_bias` | `0.70 + 0.60σ` | (0.70, 1.30) |
  | 8 | `ramp_start_frac` | `0.1 + 0.4σ` | (0.1, 0.5) |
- **Step**: chains `generate_counterfactual` `n_passes = 2 + extra1 + extra2` times, each pass escalating strength (`×1.15^idx`-ish) and cls_scale (`×1 + 0.35·idx`) and decaying pres_scale.
- **Termination**: `FIXED_EPISODE_LENGTH=True` → always exactly `BUDGET_T=8` steps.
- **Per-episode caches**: `_S_x_orig` (seg mask), `_cam_latent_mask` (Grad-CAM), `_sparsity_weight_map` = `1 − cam` at image resolution.

## 1.7 Reward implementation (`_compute_reward`)

```
R_prog   = d_t − d_next                       λ_prog   = 1.5
R_val    = −d_next                            λ_val    = 2.0
R_close2 = −d_next²                           λ_close2 = 2.5
R_fid    = −mean|x_next − x_orig|             λ_fid    = 2.0     (L1, was MSE)
R_pres   = −(seg_term + 0.5·KL(O_x‖O_next))   λ_pres_R = 3.0
R_bud    = −1.0  (constant)                   λ_bud    = 0.1
R_hit    = 1[valid]                           λ_hit    = 2.0
R_stop   = 1[valid ∧ tau>0.5]                 λ_stop   = 2.0
R_sparsity = −(w·|Δ|).sum()/w.sum(), w = 1−cam  λ = annealed 0 → 2.0
total = direction_scale · Σ λ_i R_i           direction_scale = 2.25 if c==1 else 1
valid = (d_next < 0.08) ∧ (mean|Δ| < 0.02)
```

`seg_term = (S_x ⊙ |x_next − x_orig|).sum() / S_x.sum()` — mask-weighted L1 inside the lungs. `KL` is taken over `softmax` of the pooled YOLOv8 feature vector.

## 1.8 Diffusion implementation (`src/generator.py`)

- **Encode**: grayscale → `repeat(1,3,1,1)` → `vae.encode(2x−1).latent_dist.sample() × scaling_factor` → flat 4096.
- **Decode**: reshape → `vae.decode(z/scale)` → clamp → luminance-weighted gray (0.299/0.587/0.114).
- **Conditioning**: `cond_proj` = `Linear(65→256) → SiLU → Linear(256→768)`. Input = `[c_scalar, adaptive_avg_pool1d(z_flat → 64)]`. Output is a **single cross-attention token**. No CLIP, no text encoder.
- **Sampling** (`generate_counterfactual`): SDEdit-style — encode, add noise at `t_start = T(1−strength)`, then DDIM march. Per step:
  1. `eps_hat = unet(z_t, t, cond)`, `step_out = scheduler.step(...)`, take `z0_hat = step_out.pred_original_sample`, `z_t = step_out.prev_sample`.
  2. `guidance_weight = clip((i − ramp_start_idx)/ramp_span, 0, 1)`.
  3. If weight > 0: decode `z0_hat` → classifier BCE `L_cls`; preservation `L_pres = seg_term + 0.5·MSE(O(x̂), O(x_anchor))`; **two separate `autograd.grad` calls** (deliberately unfused, see the fp16 note).
  4. `z_t += −w·(4.0·1.15·cls_scale)·∇L_cls − w·(0.05·pres_scale)·∇L_pres`, masked by `edit_mask`.
  5. **Background lock**: `z_t = edit_mask·z_t + (1−edit_mask)·add_noise(z_anchor, t_next)`; on the final step `z_bg = z_anchor` clean.
- `edit_mask = feather(dilate(lung_latent_mask)) ⊙ cam_scale`, `cam = floor + (1−floor)·CAM^γ` with floor 0.15, γ 0.75, blur 0.8, 99th-percentile normalisation.

## 1.9 Evaluation pipeline & metrics (`src/metrics.py`)

| Metric | Implementation | Status on disk |
|---|---|---|
| **CV** | strict: moved ≥ τ=0.8 **and** crossed 0.5 | 0.995 / 0.955 |
| **FID** | XRV DenseNet-121 (`-rsna`) 1024-d features, `sqrtm` + eps regularisation; also cross-group FID vs the *target* class | 103.8 / 59.7 (same-group) |
| **FOP** | needs a "Support Devices" head that no xrv weight set exposes | always `null` |
| **IP** | model-free replacement: MS-SSIM + high-pass Pearson correlation | 0.85/0.94, 0.80/0.80 |
| **Edit concentration** | top-5 % of pixels' share of Σ\|Δ\|; uniform baseline 0.05 | 0.282 / 0.297 |
| **CTR** | PSPNet heart/lung column spans, paired t-test | Δ 0.007 / 0.025 |
| **SCP** | `sigmoid(−‖YOLO feat‖/50)` on lower third — has a built-in degeneracy detector | flagged degenerate (std 0.0008) |

`heldout_validation.py` adds transfer ratio (Δ_validator / Δ_guidance), conditional CV, and a Spearman agreement control.

## 1.10 Configuration system

Flat module of module-level constants, imported as `config.*` everywhere. No CLI overrides, no YAML, no experiment IDs. Ablations are boolean flags (`USE_GRADCAM_MASK`, `ANCHOR_TO_EPISODE_ORIGIN`, `USE_SPARSITY_REWARD`, `USE_SPARSITY_ANNEALING`, `SPARSITY_OUTSIDE_CAM_ONLY`, `LOCK_BACKGROUND`, `FIXED_EPISODE_LENGTH`). `metrics.current_cfg_stamp()` snapshots the 13 knobs that change pixels, and the reduce stage refuses to pool shards whose stamps differ.

## 1.11 Dataset loading

`RSNACXRDataset`: `pandas.groupby("patientId")["Target"].max()` collapses multi-box rows to a patient-level binary label, intersected with the `.dcm` files present. `__getitem__` reads DICOM, applies RescaleSlope/Intercept, **per-image min–max normalisation to [0,1]**, → uint8 → PIL "L" → `Resize(256) → ToTensor()`. Order is sorted-by-patientId (groupby index), which is what makes `scan_groups` deterministic across SLURM array tasks.

**There is no train/val/test split anywhere in the repo.**

## 1.12 External libraries

`torch`, `torchvision`, `diffusers` (AutoencoderKL, UNet2DConditionModel, DDIMScheduler), `peft` (LoRA), `transformers`/`accelerate` (transitively), `torchxrayvision` (DenseNet-121 classifiers + PSPNet segmenter), `ultralytics` (YOLOv8n), `pydicom` + `pylibjpeg`, `scipy` (sqrtm, ttest_rel), `kagglehub`, `matplotlib`, `pandas`, `numpy`.

## 1.13 Most complex files / core research logic

**Core research logic (ask about all of these):**
1. `src/generator.py :: generate_counterfactual` — lines 375–608. Guided DDIM + background lock + CAM masking. *The* file.
2. `factory/environment.py :: step` + `_compute_reward` — action decoding and the reward.
3. `factory/ppo.py :: compute_gae` + `PPOTrainer.update` — the learning signal.
4. `src/gradcam.py :: grad_cam` / `ensemble_grad_cam` — the action-space constraint's source.
5. `src/metrics.py :: run_full_evaluation` + `counterfactual_validity` — what "success" means.

**Ideal for live walkthrough (dense, self-contained, ~60 lines each):**
- `generate_counterfactual`'s per-timestep loop (lines 522–604).
- `_compute_reward` (lines 353–467).
- `compute_gae` (lines 283–335).
- `compute_gradcam_latent_mask` (lines 229–335).
- `PPOActorCritic._build_conv_trunk` / `_conv_features`.

**Highest cyclomatic/conceptual complexity:** `src/metrics.py` (1336 lines, 7 metrics + 2 feature extractors + sharding contract), `factory/ppo.py` (994 lines, trainer + annealing + CV safeguard + checkpoint/resume + wall-clock guard).

## 1.14 Bottlenecks and bug-prone regions

**Compute bottlenecks**
- The guidance block: **two full VAE-decoder backward passes** per guided DDIM step, on 60–80 % of steps, × `n_passes` (2–4) × `BUDGET_T` (8) per episode. Roughly 1 250–4 500 UNet evaluations per episode.
- `_build_far_index_pool()` — a full-dataset classifier scan on every env construction (twice in training, twice more at eval).
- `_measure_holdout_cv()` — 24 full policy episodes ≈ 3 updates of rollout compute.
- Eval generation: ~4.4 min/image → 29 h for 400 images, hence the 8-way sharding.

**Where bugs are most likely (and where several already lived):**
| Region | Failure mode |
|---|---|
| `compute_gae` ↔ `update` | `shaped_rewards` was computed and discarded; `DELTA_MIX` was dead for the whole first phase |
| `direction_scale` | Applied to only *some* reward terms → preservation implicitly de-weighted on the harder direction |
| Action squashing | Actor logits drifted to ±5.5 → sigmoid saturated → policy effectively constant (guidance_bias std 0.005 under sampling noise 0.61) |
| `foreign_object_preservation` | Returned a degenerate 1.0 for three versions running (threshold-space mismatch, then wrong detector, then no head at all) |
| FID | Missing ImageNet normalisation; singular covariance in `sqrtm` |
| Grad-CAM | `find_target_layer` on an ensemble silently returns `members.0.features` and logs success |
| Grad-CAM degeneracy | A constant CAM becomes a *uniform mask at floor* — the constraint is silently disabled, run still "succeeds" |
| Anchoring | Without `x_anchor`, chained passes only bound drift vs the previous call — ~100 round trips of unconstrained accumulation |
| Shard determinism | Global-RNG noise draws make results depend on chunking; fixed by per-global-index seeding |
| fp16 guidance | Fusing the two backwards changes the trajectory by 0.6–4.7 % (vs fp16 eps 0.1 %) because `ALPHA_PRES=0.05` loses headroom |

## 1.15 Ground-truth numbers on disk (use these to check answers)

**`checkpoints/eval_metrics.json`** — CV 0.995 (n→p) / 0.955 (p→n) / 0.975 mean; FID same-group 103.83 / 59.73; FOP `null`; IP MS-SSIM 0.849 / 0.796, HF-corr 0.938 / 0.798; SCP std ≈ 0.0008 on mean 0.479.

**`checkpoints/heldout/heldout_validation.json`** — transfer ratio: nih 0.205 (n→p) / 0.132 (p→n); chex 0.590 / 0.143. Spearman agreement control 0.697 / 0.707 (passes). Edit concentration 0.282 / 0.297 vs 0.05 uniform baseline; MAE 0.042 / 0.029; 66.3 % / 62.3 % of pixels changed. Generated under `BUDGET_T=25, USE_GRADCAM_MASK=False, USE_SPARSITY_REWARD=False`.

**`checkpoints/guidance_diagnostics_*.json`** — n→p: guidance_bias 1.0090 ± 0.0869, n_passes 2.84, avg_reward −4.895. p→n: 0.9502 ± 0.0865, n_passes 3.03, avg_reward −1.367.

**Training log (`logs/ppo-counterfactual-*.out`, update 80)** — `avg_ep_len=8.00, avg_return=−12.720, L_ppo=0.0411, L_value=1.3598, std_mean=0.6096, lam_sparse=2.0`; reward split `flip=0 % pres=7 %`, `val=−0.887 (52 %)`, `close2=−0.536 (32 %)`; `edit_L1=0.0144 (budget 0.02)`, `score_ok=2 % budget_ok=86 % valid=0 %`; `|logit|max=0.93`. Far pool: 11 393 train / 24 held out.

**Report vs repo discrepancies** (probe these in Section 6 and Ownership — ask *which is right*, do not accuse):
- `report.tex` Table 2 reports FID **5.67 / 3.20**; `eval_metrics.json` says **103.83 / 59.73**.
- `report.tex` hyperparameter table lists `BUDGET_T=25`, `λ_stop=8.0`, `λ_hit=6.0`, `λ_fid=0.5`, `λ_pres=0.5`, LoRA rank 8 — current `config.py` has 8, 2.0, 2.0, 2.0, 3.0, rank 16.
- `report.tex` clinical table shows CTR = 0.479 ± 0.001 and SCP = 0.424 ± 0.055; the JSON has those **the other way round** (CTR mean_xn 0.4237, SCP mean_xn 0.4794).
- CV mean appears as 0.965 in Table 2 and 0.970 in the baseline table; the JSON says 0.975.
- The report says training ran on "a single Tesla V100-32GB"; SLURM logs show **Quadro RTX 6000**.
- The baseline table reports Vanilla CycleGAN and GANterfactual numbers "evaluated under the identical protocol" — **no baseline implementation exists anywhere in the repository**.

---

# Section 1 — High-Level Understanding

**Difficulty: Easy → Medium · Estimated time: 15 min**
*Goal: establish whether the candidate holds the system as one object or as a pile of scripts.*

---

### Q1.1 — Walk me from a single input chest X-ray to a finished counterfactual. Name every model the pixels pass through and say which ones have gradients.
**Difficulty:** Easy

**Why this question matters**
This is the cheapest possible check on whether the candidate has an end-to-end mental model. Someone who only wired modules together will describe files; someone who understands will describe *tensors*.

**Expected good answer**
DICOM → min-max [0,1] 256² grayscale → repeat to 3 channels → VAE encode (÷ scaling factor) to 4×32×32 latent → SDEdit init: add noise at timestep `T(1−strength)` → DDIM loop of `50·strength` steps. Inside the loop, from `ramp_start_frac` onward, each step decodes the current `pred_original_sample` back to pixels, runs the frozen classifier ensemble and YOLO feature extractor, takes ∂L/∂z0_hat for classifier and preservation losses separately, and adds a masked linear combination to `z_t`. Then the background outside the edit mask is overwritten with the anchor latent re-noised. Final decode → grayscale via luminance weights. That whole procedure is one *pass*; the env chains 2–4 of them per step, and 8 steps per episode. Gradients: only through VAE decoder + classifier + YOLO, and only w.r.t. `z0_hat` — never into any weights. LoRA weights had gradients in a *separate earlier phase*; the PPO policy has gradients w.r.t. its own parameters only.

**Red flags**
- Says "PPO generates the image" or "PPO denoises the latents."
- Cannot say where `strength` enters (it selects the starting timestep, not a blend weight).
- Thinks the classifier is being trained.
- Forgets that the policy sees `z_t` via a *stochastic* VAE encode.

**Follow-up questions**
- Where exactly does the policy's action enter that chain? Name three of the nine dimensions and their effect.
- Which tensors in that path are fp16 and which are fp32, and why does it matter?
- If I removed the VAE and worked in pixel space, which parts break?

---

### Q1.2 — Explain the project to me in two minutes as if I were a radiologist, then in two minutes as if I were a PhD reviewer at NeurIPS.
**Difficulty:** Easy

**Why this question matters**
Tests audience modelling and separates memorised abstract from actual comprehension. Most candidates can do one of the two.

**Expected good answer**
*Radiologist version:* "Our classifier says this X-ray shows pneumonia. This system produces a second image of the same patient that the classifier calls normal, changing as little as possible. The difference between the two images is where the model thinks the disease is — a visual explanation you can check against your own reading."
*Reviewer version:* Frames it as constrained optimisation over the sampling trajectory of a frozen latent diffusion model, with a learned policy replacing a hand-tuned guidance schedule, evaluated under Singla et al.'s CV/FID/FOP protocol plus off-guidance transfer validation. Should mention the central tension: CV is measured against the same classifier that provided the guidance gradient, so a high CV is not by itself evidence of a semantic edit.

**Red flags**
- Both versions are identical.
- Radiologist version uses "latent", "PPO", "reward shaping."
- Reviewer version cannot state the paper's contribution over Singla et al.

**Follow-up questions**
- What would a radiologist actually *do* with this output in a clinical workflow?
- What is the single strongest objection a reviewer would raise?

---

### Q1.3 — Of everything in `ppo-contrafactual/`, what did you personally design, what did you implement from a specification, and what came from your supervisor or from a paper?
**Difficulty:** Easy (but diagnostic)

**Why this question matters**
Sets the honesty baseline for the whole interview. A candidate who claims everything will contradict themselves within 20 minutes; a candidate who draws a clean line earns credibility for the claims they *do* make.

**Expected good answer**
A concrete, differentiated map. E.g.: metrics are a faithful reimplementation of Singla et al. Eq. 10–12; the FOP→IP substitution was their own call after discovering no xrv weight set exposes a Support Devices head; the SLURM sharding and resumable checkpointing were their own engineering; the Grad-CAM action-space constraint came from a supervisor suggestion but the `floor + (1−floor)·CAM^γ` formulation and the degeneracy handling were theirs. Names specific things they got *wrong* first.

**Red flags**
- "I did everything."
- Cannot name a single decision that came from someone else.
- Cannot name a single thing they'd do differently.

**Follow-up questions**
- Which single file would you say is *most* yours? Open it.
- Which file do you understand *least* well?

---

### Q1.4 — Why are there two policies instead of one policy conditioned on the target class?
**Difficulty:** Medium

**Why this question matters**
This is a real architectural decision documented in `main.py`, and it has both an engineering justification and a research cost. Tests whether the candidate can hold both.

**Expected good answer**
`MedicalCounterfactualEnv.__init__` takes a single `target_c` and pre-computes a far-index pool against it; a conditional policy would need randomised `target_c` per episode and dual far-pools, i.e. a real refactor. Two policies also let each specialise on a direction that behaves very differently. The cost: no representation sharing, double the training compute, and the harder direction (n→p) can't borrow from the easier one. The observation *already* contains `c` as a feature (`c_tensor.flatten()` in `get_observation`), so the conditional version is architecturally within reach — the blocker is the environment, not the network.

**Red flags**
- Doesn't know `c` is already in the observation vector.
- Claims the two directions are symmetric.
- Presents it purely as a design win with no cost.

**Follow-up questions**
- The observation already contains `c`. So what *exactly* stops you from training one policy today?
- Which of the two directions would benefit more from sharing, and why?

---

# Section 2 — Original Research Problem (GAN → Diffusion)

**Difficulty: Medium · Estimated time: 15 min**

---

### Q2.1 — Singla et al. used a conditional GAN. Your system uses latent diffusion. What specifically does diffusion buy you here — not in general, *here*, for this task?
**Difficulty:** Medium

**Why this question matters**
Separates "diffusion is SOTA" from an argument grounded in the counterfactual objective.

**Expected good answer**
The key property is that diffusion gives an *iterative, interruptible* trajectory with a well-defined intermediate state (`pred_original_sample`) that you can differentiate a classifier through at every step. A GAN generator is a single forward map: you get one shot at guidance, in latent space, with no natural notion of "how far along the edit is." That iterativity is precisely what makes an RL formulation coherent — there is a sequential decision process to control. Second: SDEdit gives a principled *strength* knob (how far back up the noise schedule to go) that trades edit magnitude against fidelity, which is exactly the counterfactual's core trade-off. Third: a pretrained SD1.5 prior + LoRA needs far less data than training a CXR GAN from scratch on 26 k images, and avoids mode collapse / training instability.

**Red flags**
- "Diffusion produces better images" with no mechanism.
- Cannot connect diffusion's iterativity to the RL formulation.
- Doesn't mention SDEdit or `strength`.

**Follow-up questions**
- CycleGAN doesn't need paired data either. What does diffusion give you that CycleGAN doesn't?
- Name a concrete way the GAN approach is *better* than yours.

---

### Q2.2 — What did you give up by moving to diffusion? Give me at least three costs.
**Difficulty:** Medium

**Why this question matters**
Trade-off literacy. A candidate who cannot name costs has not compared the two, only adopted one.

**Expected good answer**
(1) **Inference cost** — ~4.4 min per image versus a GAN's single forward pass; this is what forced 8-way SLURM sharding and made the whole eval a 29 h job. (2) **Stochasticity** — the VAE encode uses `latent_dist.sample()` and the background lock injects fresh noise every step, so counterfactuals are not reproducible without per-image seeding (`generate_batch_with_policy` seeds from the *global* index for exactly this reason). (3) **Domain mismatch** — SD1.5 is trained on 512² RGB natural images with CLIP text conditioning; here it runs at 256²→32² latents on grayscale CXRs with a randomly-initialised 1-token conditioner. (4) No explicit adversarial realism term, so nothing directly enforces on-manifold outputs — which is visible in the FID numbers.

**Red flags**
- Only names inference cost.
- Doesn't realise generation is stochastic.
- Doesn't know SD1.5 was trained at 512².

**Follow-up questions**
- 32×32 latents versus the 64×64 SD1.5 was trained on — what does that do to the UNet's attention maps?
- Which of these costs would you attack first with one more month?

---

### Q2.3 — Your baseline table in the report gives CycleGAN CV = 0.424/0.346 and GANterfactual CV = 0.998/0.872. I can't find any GAN code in the repository. Where did those numbers come from?
**Difficulty:** Medium (integrity probe)

**Why this question matters**
The single most important honesty question in the interview. There is no CycleGAN or GANterfactual implementation anywhere in `ppo-contrafactual/`, yet the report states both baselines were "evaluated under the same Xn/Xp splitting, n_eval ≤ 200-per-group protocol… with FID computed against the same XRV-DenseNet feature extractor." That claim cannot be true of code that doesn't exist. How the candidate handles this predicts everything about how they'd behave on a real research team.

**Expected good answer**
A straight answer. Acceptable versions: "Those are transcribed from the Singla et al. paper, and the protocol sentence overclaims — I should have marked them as literature values with a different protocol." Or "I ran them in a separate repo/notebook that isn't here, here's where." Or "Those are placeholder/estimated numbers I put in to structure the table and never replaced — that's a mistake." Any of these is fine. The good answer also *volunteers* what the correct fix is: label them as reported-from-paper, note the protocol difference, or drop the table.

**Red flags**
- Insists the baselines were run under the identical protocol without being able to point at code or logs.
- Vague deflection ("my supervisor gave me those").
- Doesn't recognise why "identical protocol" is a strong claim.
- Becomes defensive rather than curious.

**Follow-up questions**
- If you *did* have to run a fair GAN baseline, what would you use and what would take longest?
- Your FID uses an RSNA-trained DenseNet — the same corpus as your guidance model's first member. Can a number from that feature space be compared to a published Inception-FID at all?

---

# Section 3 — Diffusion

**Difficulty: Medium → Expert · Estimated time: 25–30 min**

---

### Q3.1 — Open `generate_counterfactual`. Explain what `strength` does mechanically. Why does `strength=0.7` produce a bigger edit than `strength=0.3`?
**Difficulty:** Medium

**Why this question matters**
`strength` is action dim 0 and the primary lever the policy controls. If the candidate can't explain it precisely, they don't understand SDEdit.

**Expected good answer**
```python
n_keep    = max(1, int(len(full_timesteps) * strength))
start_idx = len(full_timesteps) - n_keep
timesteps = full_timesteps[start_idx:]
z_t = self.scheduler.add_noise(z0_orig_4d, init_noise, full_timesteps[start_idx:start_idx+1])
```
`strength` is the *fraction of the 50-step DDIM schedule that is actually run*. Higher strength → start further up the noise schedule → more of the original latent's information is destroyed by the added noise → the denoiser has more freedom to resynthesise, so the output drifts further from the input. It is not a blending weight and not a guidance scale. It also controls compute: `strength=0.7` is 35 UNet evaluations, `0.2` is 10.

**Red flags**
- Calls it a "blend factor" or "guidance strength."
- Doesn't connect it to `add_noise` at a specific timestep.
- Doesn't notice it also determines the number of steps (and thus runtime).

**Follow-up questions**
- The env caps `pass_strength` at `min(0.9, …)`. What happens at strength = 1.0?
- Config bounds this to (0.2, 0.7). Why not let the policy use the full (0,1)?

---

### Q3.2 — The guidance gradient is taken w.r.t. `z0_hat` (`pred_original_sample`) and then added to `z_t` (`prev_sample`). Textbook classifier guidance perturbs the score at `z_t`. Justify that choice, or tell me it's wrong.
**Difficulty:** Expert

**Why this question matters**
This is the most conceptually interesting deviation in the whole codebase, and it is *not* commented as a deliberate choice. It separates candidates who read the guidance literature from those who assembled a loop that ran.

**Expected good answer**
The strong answer recognises the deviation and reasons about it. Classical classifier guidance (Dhariwal & Nichol) modifies the predicted noise: `ε̂ ← ε̂ − s·√(1−ᾱ_t)·∇_{z_t} log p(y|z_t)`, requiring a classifier trained on *noisy* inputs. Here the classifier is a clean-image DenseNet, so evaluating it at `z_t` would be nonsense — hence the decode of `z0_hat`, which is the "universal guidance"/reconstruction-guidance trick: evaluate the clean-data classifier on the model's current estimate of x₀. The deviation is that having computed ∇ w.r.t. `z0_hat`, the code adds it directly to `z_t` rather than mapping it back through the DDIM update (the correct chain would scale by ∂z_{t−1}/∂z0_hat, which for DDIM is a known ᾱ-dependent coefficient). Practically this means the *effective* guidance scale drifts with the timestep in an uncontrolled way — partially absorbed by the linear `guidance_weight` ramp and the learned `cls_scale`. A candidate who says "it's an approximation, the timestep-dependent factor is being folded into a learned scale" is answering at the right level.

**Red flags**
- Doesn't know classifier guidance requires a noise-aware classifier in the original formulation.
- Sees no difference between guiding at `z_t` and at `z0_hat`.
- Claims the current form is exactly Dhariwal & Nichol.

**Follow-up questions**
- What is the correct Jacobian factor for a DDIM step, and what would happen if you inserted it?
- Would classifier-*free* guidance work here? What would you condition on?
- Why is the guidance applied *after* `scheduler.step` rather than by modifying `eps_hat` before it?

---

### Q3.3 — Explain the background lock. Then explain why `eval_metrics.json` still shows 66 % of pixels changed.
**Difficulty:** Hard

**Why this question matters**
Ties an implementation detail to a measured number that contradicts the naive reading of it. This is the question that finds people who never reconciled their code with their results.

**Expected good answer**
The lock is RePaint / Blended-Latent-Diffusion: at every DDIM step, everything outside the (dilated, feathered) edit mask is overwritten with the *anchor* latent forward-diffused to the next timestep; on the last step the clean anchor latent is used. So out-of-mask latents follow the identity trajectory by construction. Why 66 % of pixels still move, for at least four compounding reasons:
1. `edit_mask` is **soft**, not binary. It's a sigmoid lung probability, dilated and Gaussian-feathered, multiplied by a CAM with `floor=0.15`. So *every* location gets at least ~15 % of the free trajectory blended in — the lock never fully pins anything outside a hard-zero region.
2. Even where the mask is exactly 0, the output is the **VAE reconstruction** of the anchor, not the anchor pixels. VAE round-trip error is a floor on `frac_changed`, and `diff_viz` counts anything > 0.01 as "changed."
3. The measurement predates the constraints: the heldout JSON stamps `USE_GRADCAM_MASK=False, BUDGET_T=25`.
4. Latent-space masking is not pixel-space masking — a 32×32 latent cell is an 8×8 pixel block, and the VAE decoder's receptive field spreads in-mask changes across the seam.

**Red flags**
- Claims the lock guarantees zero out-of-lung change (the docstring says "collapses to the VAE round-trip floor by construction" — a good candidate quotes that *and* explains the soft-mask caveat).
- Doesn't know `frac_changed` uses a 0.01 threshold.
- Doesn't know which config produced the 66 % number.

**Follow-up questions**
- Measure it: what's the VAE round-trip `frac_changed` on an untouched image? Did you ever run that control?
- If you set `GRADCAM_MASK_FLOOR = 0.0`, what breaks and what improves?
- Why blend in *latent* space rather than compositing in pixel space at the end?

---

### Q3.4 — Your conditioning is `Linear(65→256) → SiLU → Linear(256→768)`, producing one cross-attention token, and during LoRA training the input includes a pooled version of the **clean** latent `z0`. What's the problem?
**Difficulty:** Expert

**Why this question matters**
There is a plausible information-leak in `compute_training_losses`, and the 1-token conditioning is architecturally unusual. Tests whether the candidate can audit their own training objective.

**Expected good answer**
In `compute_training_losses`, `psi_x = z0.detach()` and the conditioning is built from `adaptive_avg_pool1d(psi_x → 64)`. So the model is asked to predict the noise added to `z0` while being handed a 64-dimensional summary of `z0` itself. The pooling is aggressive (4096→64, a 64× reduction, so it's roughly per-8×8-latent-block channel means), but it is still a shortcut: at high `t`, where `z_t` carries little signal, the conditioning is the *only* informative input, and the model can partially solve the task by learning `noise ≈ (z_t − √ᾱ·decode_cond)/√(1−ᾱ)`. That would make `L_diff` look good while learning less about CXR structure than it appears. A strong answer proposes the diagnostic: ablate the conditioning to `c` alone, or shuffle `psi_x` within the batch, and see if `L_diff` moves.
Separately: one token means the cross-attention layers have a single key/value to attend to at every spatial position — the attention softmax over one element is identically 1, so cross-attention degenerates into a per-position learned bias. That is a *very* weak conditioning mechanism, and it means the LoRA on `to_k`/`to_v` is doing much less than it appears.

**Red flags**
- Doesn't see the leak.
- Doesn't realise softmax over a single token is constant.
- Thinks `cond_proj` is a text encoder replacement in any meaningful sense.

**Follow-up questions**
- With one token, what are `to_q`/`to_k`/`to_v` LoRA adapters actually learning?
- How would you condition instead? Sketch it.
- At inference `psi_x` comes from the *input* image, so the leak is "legitimate" — does that make training/inference consistent or inconsistent?

---

### Q3.5 — Why LoRA rank 16 with alpha 16, on that specific target set? Config says it was raised from rank 8, and the archived rank-8 policies are kept. Justify the change.
**Difficulty:** Hard

**Why this question matters**
There's a documented hypothesis behind the rank change (`config.py` lines 471–502). Tests whether the candidate can restate it as a *scientific* argument and knows the confound it introduces.

**Expected good answer**
The hypothesis: p→n was the direction held-out validation found adversarial (transfer 0.13/0.14 vs 0.21/0.59 for n→p), and that asymmetry is physically sensible — adding diffuse opacity to a clear lung is easy; *removing* an established consolidation requires synthesising plausible parenchyma, vessels and texture underneath. That's generative capacity, not reward shaping. Rank 8 on attention projections alone (1.59 M params) is a small adapter for it, so the target set was widened to include the feed-forward blocks (`ff.net.0.proj`, `ff.net.2`) and `proj_in`/`proj_out`, reaching ~6 M trainable. `alpha=r` means scaling factor 1.0. ResNet `conv1/conv2` were deliberately excluded (would ~4× params, overfitting risk on 26 684 images).
**The confound**: `N_DENOISE_EPOCHS` was simultaneously raised 15→30 to match what the rank-8 checkpoint received — so capacity and training time are no longer independently varied against the *original* 15-epoch plan. A candidate who names this without prompting is thinking like a researcher. Also: `peft` matches `target_modules` by dotted suffix, so `"ff.net.2"` must be fully spelled out; a bare `"2"` would match unrelated modules.

**Red flags**
- "Higher rank is better."
- Doesn't know `alpha/r` is the scaling factor.
- Cannot state what breaks: a rank change makes every prior generator artefact unloadable (adapter A/B are shaped `(r, in)`/`(out, r)`) — `pipeline.load_generator_weights` raises a custom error for exactly this.
- Doesn't know how to verify the target strings matched (`print_trainable_parameters()` — if it still reports 1 594 368, nothing matched).

**Follow-up questions**
- Did you ever run rank-8 and rank-16 with the *same* epoch count? If not, what does the comparison actually tell you?
- Why `alpha = r` and not `2r`?
- Why exclude the ResNet convs when texture synthesis is exactly what you claim p→n needs?

---

### Q3.6 — DDIM with 50 steps and η = 0. Why DDIM and not DDPM here? Does your pipeline actually get DDIM's determinism?
**Difficulty:** Hard

**Why this question matters**
The determinism claim is falsified by the code itself, in two places. Good trap.

**Expected good answer**
DDIM for two reasons: fewer steps for equivalent quality (50 vs ~1000), which matters when a single episode runs 1 250–4 500 UNet evals; and a deterministic η=0 trajectory, which makes the counterfactual a function of the initial noise alone. **But the pipeline does not get determinism**, because (a) `encoder()` uses `posterior.latent_dist.sample()` rather than `.mode()`, so the starting latent is a fresh draw from the VAE posterior on every call — including every call to `get_observation()`, which means the *observation itself* is stochastic; and (b) the background lock calls `torch.randn_like(z_anchor_4d)` at every step to re-noise the anchor. This is why `generate_batch_with_policy` has to seed `torch.manual_seed(base_seed + index_offset + i)` per image to make sharded and serial runs byte-identical.

**Red flags**
- Claims the pipeline is deterministic.
- Doesn't notice `latent_dist.sample()`.
- Doesn't connect the seeding logic in `inference.py` to a determinism problem.

**Follow-up questions**
- The observation contains `z_orig` and `z_t`, both from `sample()`. What does that noise do to PPO's value function?
- Would `.mode()` be strictly better? What would you lose?
- Is the background-lock noise even needed on the final step? Look at the `i < n_steps - 1` branch.

---

# Section 4 — PPO

**Difficulty: Medium → Expert · Estimated time: 25 min**

---

### Q4.1 — Define the MDP precisely. State, action, transition, reward, horizon. Then tell me whether it's actually an MDP.
**Difficulty:** Medium

**Why this question matters**
Forces precision. The "is it actually an MDP" half exposes whether the candidate has thought about partial observability and stochasticity.

**Expected good answer**
- **State**: `[z_orig(4096), z_t(4096), step/T, d_t, c, seg_mask(65 536)]` = 73 731-d.
- **Action**: 9 continuous, sampled from `Normal(actor_logits, exp(log_std))`, then sigmoid-squashed to bounded ranges inside `env.step`.
- **Transition**: 2–4 chained calls to `generate_counterfactual`, each a full guided DDIM run — stochastic (VAE sampling + lock noise).
- **Reward**: the 9-term weighted sum, computed on `x_next` vs `x_orig`.
- **Horizon**: fixed 8 steps (`FIXED_EPISODE_LENGTH=True`), never terminates early.
- **Not strictly Markov**: the observation gives `z_t` (a fresh VAE sample, so a noisy view of `x_t`), but `x_t` itself is what carries forward. The policy cannot see how many refinement passes were used previously, the accumulated `edit_l1`, or the CAM mask. `d_t` is a 1-d summary of the classifier state. The step counter makes the fixed horizon observable, which is what saves it.

**Red flags**
- Says the state is "the image."
- Doesn't know the obs dimension or can't decompose it.
- Doesn't realise the action space is squashed *outside* the network.
- Doesn't notice `edit_l1` — half of the validity condition — is absent from the observation.

**Follow-up questions**
- `edit_l1 < 0.02` is half of `is_valid`, and the policy can't observe it. What does that do to learnability?
- Is a fixed horizon a good idea here? What did the previous variable-horizon version do?
- The seg mask is 88.9 % of the observation and gets average-pooled to 32×32 in the trunk anyway. Why is it in the observation at full resolution?

---

### Q4.2 — Why PPO and not DQN? Why not SAC?
**Difficulty:** Medium

**Why this question matters**
Standard, but the *right* answer here is specific to this environment's data regime, not generic algorithm trivia.

**Expected good answer**
DQN is out immediately: 9-dimensional continuous actions. Discretising even coarsely explodes combinatorially, and the action semantics (guidance scales) are genuinely continuous.
SAC vs PPO is the interesting comparison. SAC is off-policy and far more sample-efficient — which *should* be the deciding factor here, because environment steps are brutally expensive (~1 s per `generate_counterfactual` call, 2–4 per step, 8 steps per episode; the whole run is only 800 episodes ≈ 6 400 steps). That is an extremely small dataset for on-policy learning. PPO's advantages: simpler, more robust to hyperparameters, no replay buffer to store 73 731-d observations in (a 100 k-transition buffer would be ~30 GB in fp32), and no target-network/entropy-temperature tuning. A strong candidate concedes that SAC is arguably the better fit and names the replay-buffer memory cost as the honest reason PPO won.
PPO's fit to *this* problem specifically: bounded action space handled naturally, the clip objective keeps the guidance schedule from lurching between updates, and the shared-trunk actor-critic supports the auxiliary heads.

**Red flags**
- "PPO is more stable" and nothing else.
- Doesn't know PPO is on-policy or what that costs at 6 400 total steps.
- Claims SAC can't do this.

**Follow-up questions**
- 800 episodes × 8 steps = 6 400 transitions, 4 epochs per update, batch = 64 transitions. Is that enough to fit even a 200 k-param policy?
- What would you actually store in a SAC replay buffer to avoid the 73 731-d problem?
- Would a contextual bandit be sufficient? The previous version measured `avg_ep_len = 1.00`.

---

### Q4.3 — Read `compute_gae`, then read the `L_value` line in `update()`. What is the value head being regressed onto?
**Difficulty:** Expert

**Why this question matters**
This is the highest-signal question in the PPO section. There is a change-log entry in `factory/ppo.py` presenting this as a *fix*, and it is at minimum debatable — arguably a regression. A candidate who owns the RL side will engage with it; a candidate who doesn't will read the comment aloud.

**Expected good answer**
```python
shaped_r[t] = tr.reward if tr.tau > 0.5 else tr.reward + config.DELTA_MIX * adv[t]
...
L_value = F.mse_loss(values, shaped_rewards)
```
The value head is being regressed onto a **single-step reward** (optionally blended with that step's advantage), not onto a return. Standard PPO regresses `V(s_t)` onto `returns = adv + value` (the GAE bootstrap target) or onto discounted Monte-Carlo returns. Regressing onto `r_t` teaches the critic to predict the *immediate* reward, so with γ = 0.99 and an 8-step horizon the critic systematically under-predicts by roughly the discounted tail. That biases every advantage: `δ_t = r_t + γV(s_{t+1}) − V(s_t)` uses a `V` that isn't a value function at all. The change-log's rationale ("the standard place a shaped reward signal is meant to land") is wrong — shaped rewards belong in the *reward*, i.e. upstream of GAE, not as the critic's regression target.
Supporting evidence in the logs: `L_value = 1.3598` at update 80 while `avg_return = −12.720` — the critic is fitting something with a much smaller scale than the return.
Note also that `shaped_r` uses `adv[t]` *before* the normalisation on line 330, so it mixes an unnormalised advantage into the target.
A strong candidate proposes the experiment: run both targets for 100 updates and compare `frac_valid`, since `returns` is still computed and available.

**Red flags**
- Reads the comment as a justification without evaluating it.
- Doesn't know what the standard value target is.
- Doesn't notice `returns` is computed and then unused.
- Doesn't spot that `DELTA_MIX` was dead code for the entire first phase of the project.

**Follow-up questions**
- The change-log says `DELTA_MIX` "had ZERO effect on training" before this. How long was that true, and what does that mean for any result from that period?
- Look at `td_targets` — `next_vals[i] = transitions[min(i+1, len-1)].value`. What happens at an episode boundary inside a flat batch of 8 episodes?
- `compute_gae` is called with `last_value=0.0`. When is that correct?

---

### Q4.4 — Every action dimension is squashed through a sigmoid inside `env.step`, but `log_prob` is computed on the pre-squash Gaussian. Walk me through the consequences.
**Difficulty:** Expert

**Why this question matters**
This is a real, subtle issue that the codebase *partially* addresses with a hinge penalty (`LAM_ACT_SAT`) without naming the underlying cause. Excellent discriminator.

**Expected good answer**
The environment applies `σ(a)` but the policy's distribution — and therefore the entropy bonus, the log-probs and the PPO ratio — all live in the unsquashed space. Two consequences:
1. **The ratio is still correct.** Because the squash is a deterministic function applied identically at collection and at update, `π(a|s)` is consistent, so the importance ratio is valid. (Unlike SAC, PPO doesn't need the tanh Jacobian correction for its objective — only for a correct *entropy* value.)
2. **The entropy bonus is measured in the wrong space.** `C_ENTROPY = 0.01 · H[N(μ, σ)]` rewards spreading the *pre-squash* Gaussian, which the policy can satisfy for free by pushing μ into the sigmoid's flat tail — where large pre-squash variance maps to near-zero variance in *effective* action space. That is precisely the failure documented in `config.py`: v1's actor mean drifted to ~±5.5, and `guidance_bias` measured 1.2976 ± 0.0054 and 0.7005 ± 0.0012 — the two **ends** of its (0.70, 1.30) range — under sampling noise σ ≈ 0.61. The policy became deterministic and state-independent: a bang-bang solution.
The mitigation (`L_act_sat = relu(|logits| − 3.0)²`, weight 1e-2) penalises the dead tail while leaving |logit| < 3 (where σ ≈ 0.95) untouched. It treats the symptom. The principled fixes: apply the tanh/sigmoid Jacobian correction to log_prob and entropy (SAC-style), or use a Beta distribution on the bounded support.
The current logs show it works: `|logit|max = 0.93` at update 80, and `guidance_bias_std` is now 0.0869 (vs the 0.005 disaster, well above the 0.05 threshold the config names as the health check).

**Red flags**
- Doesn't see any issue.
- Thinks the PPO ratio is broken (it isn't).
- Cannot connect the hinge penalty to saturation.
- Doesn't know what a bang-bang policy is.

**Follow-up questions**
- Write the Jacobian correction for a sigmoid squash. Why is it `−Σ log(σ(a)(1−σ(a)))`?
- Why a squared hinge and not L1?
- Action dims 4 and 5 are thresholded to bits (`σ(a) > 0.5`). What does the gradient of the entropy bonus mean for a dimension whose effect is a step function?

---

### Q4.5 — The policy has four heads: critic, feasibility, progress, stop. What is each auxiliary head learning, and is any of them useless?
**Difficulty:** Hard

**Why this question matters**
One of them is self-referential. Finds candidates who added heads because a spec listed them versus candidates who reasoned about the supervision signal.

**Expected good answer**
- `critic` → `L_value` (discussed above).
- `feasibility_head` → BCE against `(returns > 0)`. Learns "will this state produce positive return" — but `returns` depends on the critic's own values, so the label is partly self-generated. Weak but not vacuous.
- `progress_head` → MSE against `td_targets`. Roughly a second value estimate on a different target. Redundant with the critic.
- `stop_head` → BCE against `(t.tau > 0.5)`, where `tau` **is the policy's own action dim 3**. This head is trained to predict what the policy itself just chose. It carries no environmental information whatsoever — it's a distillation of one action dimension into a separate head. With `LAM_STOP_AUX = 0.1` it mostly acts as a mild representation regulariser on the shared trunk.
Additionally: none of the three auxiliary outputs is ever *consumed*. `get_action_and_value` returns `_feas, _prog, _stop` and every call site discards them (`action, _log_prob, _value, _feas, _prog, _stop = …`). They only affect training through the shared trunk's gradients.

**Red flags**
- Doesn't notice the stop head's label is the policy's own action.
- Doesn't notice the outputs are never used.
- Claims each head is essential.

**Follow-up questions**
- If you deleted all three, what do you predict changes? How would you test that in one run?
- `tau` also gates `R_stop` and the `DELTA_MIX` blend. Under `FIXED_EPISODE_LENGTH=True`, what is `tau` *for*?

---

# Section 5 — Reward Function

**Difficulty: Medium → Expert · Estimated time: 25 min**
*All questions here are taken directly from `factory/environment.py::_compute_reward` and `config.py` lines 218–308.*

---

### Q5.1 — Read me the reward decomposition from update 80 of the actual training log. Explain what it says about your reward design.
**Difficulty:** Hard

**Why this question matters**
The candidate built the logging *specifically* to verify the rebalance landed. If they can't read their own diagnostic, they never used it.

**Reference (from `logs/`):**
```
reward split [flip=0% pres=7%] val=-0.887(52%) close2=-0.536(32%) bud=-0.100(6%)
             pres=-0.064(4%) prog=+0.051(3%) fid=-0.029(2%) sparsity=-0.025(1%)
             hit=+0.000(0%) stop=+0.000(0%)
edit_L1=0.0144 (budget 0.02) | steps score_ok=2% budget_ok=86% valid=0%
```

**Expected good answer**
The rebalance overshot in the opposite direction. The target stated in `config.py` was ~40/40/20 flip/preservation/progress, correcting v1's 93 % flip dominance. What actually happened: flip is **0 %** because `valid` never fires — `hit` and `stop` are identically zero, since validity requires *both* `d_next < 0.08` (only 2 % of steps) and `edit_l1 < 0.02` (86 % of steps). The reward is now 84 % dominated by `val` + `close2`, which are both monotone functions of the same quantity `d_next` — so the policy is effectively being trained on a single scalar objective with two redundant terms, plus a constant −0.1 per step from `R_bud`. Preservation is 7 %, still far from the 40 % target.
The diagnosis: the joint validity criterion was calibrated (`EDIT_BUDGET_L1 = 0.02`, deliberately below the v1-measured 0.042/0.029) to be "reachable but not by the v1 strategy" — and it turns out to be unreachable *at all* by the current policy. The budget half is nearly free (86 %) and the score half is the binding constraint (2 %). So the sparse bonus provides zero gradient and the whole flip incentive has vanished. Fix: either loosen `EPSILON_STOP`, or replace the binary `R_hit` with a shaped version, or accept that `val`/`close2` are carrying the objective and delete the dead terms.

**Red flags**
- Can't explain why `hit` and `stop` are 0.
- Doesn't notice `val` and `close2` are the same signal twice.
- Reads the "flip=0 %" as a success because "flip used to be 93 %."
- Doesn't know which half of `is_valid` is binding, despite the log printing both.

**Follow-up questions**
- `score_ok=2 %`, `budget_ok=86 %`. Which knob do you turn, and by how much?
- `R_val = −d_next` and `R_close2 = −d_next²` with λ 2.0 and 2.5. What does the quadratic buy you that the linear doesn't?
- `avg_return = −12.72` and `avg_reward` in `guidance_diagnostics_n_to_p.json` is −4.895 for the *same* policy. Why do they differ?

---

### Q5.2 — `R_bud = −1.0`, always, every step, with λ_bud = 0.1. What does that term actually do to the policy gradient?
**Difficulty:** Hard

**Why this question matters**
A term that is constant across all states and actions. Tests understanding of what advantage estimation removes.

**Expected good answer**
Under a *fixed* horizon, a constant per-step reward contributes an identical amount to every trajectory's return, so after GAE and the per-update advantage normalisation (`adv = (adv − adv.mean())/(adv.std() + 1e-8)`) it contributes essentially nothing to the policy gradient. It shifts `V(s)` by a constant, which the critic absorbs. It is 6 % of the absolute reward mass in the log, so it *dilutes* the reported percentages of every other term without affecting learning.
The term is a vestige of the variable-horizon design: when episodes could end early, `−1.0` per step was a genuine time penalty creating pressure to finish quickly. `FIXED_EPISODE_LENGTH = True` made it inert. A good candidate says: it should be deleted, or `FIXED_EPISODE_LENGTH` should be `False`, but keeping both is incoherent.

**Red flags**
- Claims it "encourages efficiency" under a fixed horizon.
- Doesn't connect it to advantage normalisation.
- Doesn't realise it distorts the reported reward-split percentages.

**Follow-up questions**
- The `direction_scale = 2.25` multiplier is also uniform. The change-log admits it "washes out almost entirely." Why did you keep it?
- Which reward terms in the current design *do* survive advantage normalisation, and why?

---

### Q5.3 — `R_fid` was changed from MSE to L1 and λ_fid went 0.5 → 2.0. Explain both halves of that change. Is the term now redundant with `R_sparsity`?
**Difficulty:** Hard

**Why this question matters**
The rationale is written in the config, so a candidate can parrot it. The redundancy question is not answered there.

**Expected good answer**
**Why L1**: L2 is minimised by spreading a fixed total change over many pixels (many small errors beat few large ones), which is *exactly* the diffuse whole-thorax editing being eliminated. L1 is the sparsity-inducing norm — its subgradient is constant in magnitude, so it doesn't discount small deviations. This is standard Lasso-vs-Ridge reasoning applied to an image-space penalty.
**Why 2.0**: mostly rescaling, not re-weighting. MSE at these magnitudes is ~0.0033 while L1 is ~0.04 — roughly 12×. So 0.5 → 2.0 is a 4× nominal increase against a 12× *decrease* in the quantity's scale; net, the term's contribution went **down**. The config comment claims "L1 is ~25× larger numerically" and calls 2.0 "a modest real increase" — worth pressing on the arithmetic.
**Redundancy**: `R_fid = −mean|Δ|` and `R_sparsity` with `SPARSITY_OUTSIDE_CAM_ONLY=True` is `−(w·|Δ|).sum()/w.sum()` where `w = 1 − cam`. These are the same functional form differing only by a spatial weighting. Since `cam ∈ [0.15, 1]`, `w ∈ [0, 0.85]`. If the CAM is near-uniform (which the degeneracy machinery exists precisely because it sometimes is), `R_sparsity` collapses to a scaled copy of `R_fid` and the two are a single term with weight `λ_fid + λ_sparsity·k`. The config acknowledges partial overlap and says "treat LAM_FID as a confound when reading the ablation" — a strong candidate goes further and says the ablation of `USE_SPARSITY_REWARD` is *uninterpretable* without also varying `LAM_FID`.
In the log both are tiny: `fid = −0.029 (2 %)`, `sparsity = −0.025 (1 %)`. Combined, 3 %.

**Red flags**
- Recites "L1 is sparsity-inducing" with no mechanism.
- Doesn't do the scale arithmetic.
- Doesn't see the redundancy.

**Follow-up questions**
- Sparsity in *pixel* space penalises spatial extent. Would latent-space L1 be better? What does the code say about why it isn't done? (`_compute_reward` comment: an extra VAE encode per step.)
- With `w = 1 − cam` and floor 0.15, the maximum weight is 0.85. Does the `/w.sum()` normalisation fix that?

---

### Q5.4 — `R_pres` combines a segmentation-weighted L1 with `η·KL(softmax(O(x))‖softmax(O(x_cf)))` where `O` is a pooled YOLOv8 backbone feature vector. Defend the KL term.
**Difficulty:** Expert

**Why this question matters**
This is the least defensible term in the reward and it has a λ of 3.0 — nominally the largest weight in the table. Tests whether the candidate inherited it or reasoned about it.

**Expected good answer**
A candidate who understands will not defend it wholesale. Problems:
1. `O(x)` is a **pooled feature vector** from the first 10 layers of a COCO-pretrained YOLOv8n — global average pooled to a flat vector. Applying `softmax` to that vector treats arbitrary real-valued channel activations as a categorical distribution over channels. There is no probabilistic semantics; KL over it is a distance measure with no principled interpretation, dominated by whichever channel happens to have the largest activation.
2. **Domain mismatch**: COCO features on chest X-rays. Nothing in that backbone was trained on support devices, tubes, or anatomy.
3. **Inconsistency**: the *same* preservation concept is implemented differently in the generator's guidance (`F.mse_loss(O_hat, O_x)`) and in the environment's reward (KL of softmaxes). The policy's guidance and the policy's reward are optimising different functions of the same features.
4. `λ_pres_R = 3.0` is the largest λ, but in the log `pres = −0.064 (4 %)` — the *magnitude* of the raw term is tiny, so the weight is misleading. Weight ≠ influence.
A good answer proposes: replace with a cosine distance on the features (scale-free, no fake probability), or better, drop YOLO entirely and use the MS-SSIM / high-frequency-correlation pair already implemented in `metrics.identity_preservation` — which the candidate *already* wrote as the model-free FOP replacement and which is domain-appropriate.

**Red flags**
- Defends the softmax-KL as principled.
- Doesn't notice the guidance/reward inconsistency.
- Confuses the weight λ = 3.0 with the term's actual 4 % contribution.
- Doesn't know YOLOv8n here is COCO-pretrained and never fine-tuned.

**Follow-up questions**
- `metrics.identity_preservation` computes MS-SSIM and HF-correlation and needs no extra model. Why isn't that the preservation reward?
- `ETA_OBJ = 0.5` sits *inside* `R_pres` and is multiplied again by `λ_pres_R = 3.0`. Why two nested weights?
- The `clamp(1e-8, 1.0)` before the log — what would happen without it, and what does the need for it tell you about the softmax's output?

---

### Q5.5 — Explain the sparsity annealing schedule and the CV-freeze safeguard. Why is the freeze *latched*?
**Difficulty:** Hard

**Why this question matters**
This is a genuinely thoughtful piece of machinery (`_lambda_for_update`, `_apply_cv_safeguard`) with a non-obvious design decision. Tests whether the candidate can reconstruct the reasoning.

**Expected good answer**
The schedule: `λ_sparsity = 0` for the first 20 % of updates, ramps linearly to 2.0 over updates [20 %, 60 %], then constant. Rationale: let the policy learn a reliable flip first under the Grad-CAM-constrained action space, then apply minimality pressure within that space. Bounds are *fractions* of total updates rather than absolute indices because `N_EPISODES = 800` is an explicitly unbenchmarked placeholder.
The safeguard: every 10 updates during the ramp window, run 24 held-out episodes at a shortened budget (`CV_CHECK_MAX_STEPS = 4 < BUDGET_T = 8`), compute CV with the *same* criterion as `metrics.run_full_evaluation` (τ=0.8, boundary 0.5), and if it has dropped more than 0.08 below the pre-ramp baseline, freeze λ.
**Why latched**: it freezes at `lambda_last_ok` — the last λ that *passed* — not the current one, because the current value is what caused the drop. And it never un-freezes on recovery, because an oscillating λ would make the run scientifically uninterpretable: you couldn't attribute a final result to any particular sparsity weight. The state is persisted in `ppo_latest.pt` alongside the weights, because resuming with `cv_pre_ramp=None` would re-baseline the probe against a mid-ramp policy and silently release the latch.
**Sizing**: `ENV_EVAL_HOLDOUT_N = 24` gives CV resolution 1/24 ≈ 4.2 points per image; the 8-point tolerance sits at two images' worth, above the noise floor. At N=8 a single image would move CV by 12.5 points and the freeze would latch on noise.

**Red flags**
- Doesn't know why the freeze holds `lambda_last_ok` rather than the current λ.
- Doesn't know the probe images are a deterministic slice `_far_indices[:24]`.
- Doesn't know the schedule keys off `update_count`, not the `n_epochs` argument of `update()` (PPO's inner passes) nor `N_DENOISE_EPOCHS`.

**Follow-up questions**
- Is the CV probe actually held out from **evaluation**? Trace `_far_indices[:24]` against what `scan_groups` returns. *(This is the leak — see Deep Technical Q13.)*
- One probe costs 24 episodes vs the 8 an update consumes. Was ~3 updates of compute per probe worth it?
- The freeze can only latch during the ramp window. What if CV degrades *after* the ramp completes?

---

# Section 6 — Code Ownership

**Difficulty: Medium → Hard · Estimated time: 25 min**
*Open the files. Screen-share. These are all real functions.*

---

### Q6.1 — Open `factory/ppo.py` and find `PPOActorCritic._build_conv_trunk`. Explain why it exists and what the `expected` check on line 189 is protecting against.
**Difficulty:** Medium

**Why this question matters**
The conv trunk replaced an MLP for a specific, quantified reason. It's the clearest example of a parameter-budget argument in the repo.

**Expected good answer**
The MLP trunk was `Linear(73731, 512) → Linear(512, 256)` = 37 750 784 of the policy's 37 885 462 parameters — 99.6 % — being fit on roughly 1 000 environment steps. It consumed the observation as one flat vector with no spatial structure, of which the 256×256 seg mask alone was 88.9 % of the entries. That is the direct mechanism for v1's constant, state-independent action: it had nowhere near the data to fit that layer, so it fit a bias.
The conv trunk reshapes the *same* flat observation internally: `z_orig(4,32,32)`, `z_t(4,32,32)`, and the mask average-pooled from 256² to 32², stacked to 9 channels → three convs → global average pool → concat the 3 scalars → 2 linear layers. ~200 k params. Critically, `get_observation()` is untouched, so `environment.py`, `pipeline.py` and `inference.py` are unaffected and `load_directional_policy` still reads `obs_dim` off a live `env.reset()`.
The `expected` check recomputes the layout from `config.LATENT_CHANNELS/LATENT_SIZE/IMG_SIZE` and raises at *construction* if `obs_dim` disagrees — so a geometry change fails loudly instead of silently reinterpreting the observation as garbage (e.g. slicing `z_t` out of the middle of the mask).

**Red flags**
- Doesn't know the parameter counts.
- Thinks the conv trunk changed the observation.
- Can't say why average pooling rather than max/threshold for the mask (keeps soft lung *probability* values).

**Follow-up questions**
- `POLICY_TRUNK = "mlp"` is kept. Why? *(archived rank-8 policies won't load into a conv policy)*
- The 3 scalars are concatenated *after* the conv, not broadcast as extra channels. Why?
- Would you have kept the mask in the observation at all, given the conv immediately pools it to 32×32?

---

### Q6.2 — Open `src/gradcam.py`. Explain `DegenerateCAMError` — not what it is, but why it needed to be its own exception class.
**Difficulty:** Hard

**Why this question matters**
The docstring makes a subtle scientific-validity argument. Whether the candidate reconstructs it is very informative.

**Expected good answer**
A constant CAM is not a loud failure. Fed through `compute_gradcam_latent_mask`'s `floor + (1 − floor)·cam`, an all-zero CAM becomes a **uniform mask at `floor`** — every location gets an identical edit budget. So the locality constraint is silently disabled while the run completes normally and the ablation arm still *looks* valid. That is the most damaging way this can fail: it produces a clean-looking "+gradcam_mask" result that carries no localisation signal at all.
The realistic cause: the classifier returns a probability; once it rounds to exactly 1.0 or 0.0 in fp32 (logit magnitude beyond ~17), `d(prob)/d(activations)` is identically zero, so `relu(Σ w_c A_c)` has zero span.
The docstring's further point — and this is the discriminator — is that back-propagating `logit(p)` instead of `p` does **not** fix it: the `p(1−p)` factor is a spatial constant that cancels under CAM normalisation anyway, and once `p` has rounded, `logit(clamp(p))` differentiates a constant. Recovering the gradient would require hooking the classifier's final `Linear` for the pre-sigmoid logit.
The counting machinery follows: `_cam_episodes_degenerate` in the env, `_cam_partial_degenerate` / `_cam_full_degenerate` in the generator, and `PPOTrainer.train` printing the rate at the end — because a run with many degenerate episodes must not be reported as a clean ablation arm.

**Red flags**
- "It's just error handling."
- Thinks returning a uniform mask at floor would be acceptable.
- Says logit-space back-prop would fix it.

**Follow-up questions**
- Why does the degeneracy check happen *before* normalisation in `grad_cam`?
- `ensemble_grad_cam` skips degenerate members rather than letting them contribute a zero map. Why does that distinction matter for the average?
- The logs show one run with 0/257 degenerate and another with 2/97 (2.1 %). What threshold would make you throw a run out?

---

### Q6.3 — In `src/generator.py` lines 574–575 there are two separate `torch.autograd.grad` calls with a long comment about not fusing them. Explain the argument, and tell me whether you'd trust it.
**Difficulty:** Expert

**Why this question matters**
The comment makes a specific, measured numerical claim. Tests numerical-precision literacy and whether the candidate can evaluate a claim rather than accept it.

**Expected good answer**
The argument: `action` is a fixed linear combination `−w_cls·∇L_cls − w_pres·∇L_pres` whose coefficients are Python floats known *before* the backward, so in exact arithmetic differentiating `w_cls·L_cls + w_pres·L_pres` once gives the identical vector and saves one full VAE-decoder backward (~1.5× on the whole eval). But this runs in fp16 — `z0_hat_g` is fp16 all the way from the encoder, so the decoder backward is fp16 too. Folding the weights into the loss scales the whole backward chain *before* the fp16 intermediates rather than after, and with `ALPHA_PRES = 0.05` the preservation branch loses several bits of headroom against fp16's subnormal floor. The comment reports 0.6–4.7 % max relative deviation in `action` across settings versus an fp16 eps of 0.1 % — a real trajectory change, not rounding.
Whether to trust it: the reasoning is sound in principle, and 0.05 × a small gradient is genuinely close to fp16 subnormals (~6e-8). But this is the single most expensive operation in the pipeline (two decoder backwards on 60–80 % of DDIM steps × 2–4 passes × 8 steps per episode), so the right move is not to accept the comment permanently — it's to do exactly what the comment says: move the guidance block to fp32, or use a gradient-scaling factor, and re-validate CV/FID. The 1.5× speedup on a 29-hour eval is 10 hours.

**Red flags**
- Doesn't know why fp16 makes the two forms differ.
- Doesn't know what a subnormal is.
- Doesn't identify this as the pipeline's dominant cost.

**Follow-up questions**
- What is fp16's smallest normal, and where does `0.05 × ∇L_pres` sit relative to it?
- `retain_graph=True` on the first call — what is being retained, and how much memory?
- If you moved just this block to fp32, what's the VRAM cost on a 23 GB RTX 6000?

---

### Q6.4 — `src/metrics.py::foreign_object_preservation` has a three-version history in its docstring and currently always returns `None`. Walk me through v1 → v2 → v3 and say what you'd do next.
**Difficulty:** Medium

**Why this question matters**
A well-documented debugging arc the candidate should own completely. If they can't narrate it, the docstring isn't theirs.

**Expected good answer**
- **v1**: thresholded the raw YOLO backbone feature-vector L2 norm against `fo_threshold = 0.5`. Feature norms are routinely tens-to-hundreds, so the condition was true for essentially every image → degenerate 1.0/1.0/1.0 always. Classic threshold-space bug: comparing an unbounded norm against a probability threshold.
- **v2**: used the full YOLO detection head instead of the backbone, fixing the threshold space — but `yolov8n.pt` is COCO-pretrained on everyday objects and has never seen a pacemaker or chest tube.
- **v3**: replaced YOLO with a TorchXRayVision DenseNet-121 requesting the CheXpert "Support Devices" head. Discovered no shipped xrv weight set actually exposes that head — the label exists only in xrv's dataset label-loading code. So the class sets `self.available = False`, `foreign_object_preservation` returns `None`, and `_print_results` deliberately doesn't print it.
- **What replaced it**: `identity_preservation` — MS-SSIM (global anatomical identity) plus Pearson correlation of high-pass-filtered images (thin high-contrast structures: tubes, lines, leads). Model-free, no downloads.
- **Next**: either fine-tune a small detector on annotated support devices (Singla et al. trained a Fast R-CNN on ~300 manually annotated images per object, 80 % accuracy, never released), or accept IP and stop reporting FOP entirely rather than carrying a `null` field.

**Red flags**
- Doesn't know it currently returns `None`.
- Can't explain the v1 threshold-space bug.
- Doesn't know what replaced it.

**Follow-up questions**
- `results` still carries `FOP: {n_to_p: null, …}`. Why keep a null field?
- MS-SSIM 0.849 (n→p) vs 0.796 (p→n). Which direction preserves more, and does that match the diffusion argument you gave earlier?

---

### Q6.5 — In `factory/environment.py::step`, `pres_suppression_floor = 0.6` is applied as `clamp(1/guidance_bias, min=0.6)`. Does that floor ever bind?
**Difficulty:** Medium

**Why this question matters**
It doesn't, and the code says so. Tests whether the candidate can do the arithmetic and whether they're comfortable saying "this line does nothing."

**Expected good answer**
No. `guidance_bias = 0.70 + 0.60·σ(a₇) ∈ (0.70, 1.30)`, so `1/guidance_bias ∈ (0.769, 1.428)` — always above 0.6. The per-pass decay `max(1 − 0.15·pass_idx, 0.4)` only reaches 0.55 at `pass_idx = 3`, also above its 0.4 floor. Both are dead code today, left as a safety margin against future action-range changes; the real constraint is the action transform itself. Verified empirically: `guidance_diagnostics_*.json` shows `pres_suppression_min` of 0.858 (p→n) and 0.830 (n→p) across 160 steps each — nowhere near 0.6.
The honest framing: if you want a floor that actually matters, tighten it above the natural minimum (e.g. `min=1.0` to prevent `guidance_bias` from reducing `pres_scale` at all) or widen the sigmoid range and re-add a binding floor.
Note the interaction with `main.py::run_guidance_diagnostics`, which prints a warning if `pres_suppression_mean < 0.70` — a threshold that also cannot be crossed given the action bounds.

**Red flags**
- Says yes without computing.
- Doesn't check the diagnostics JSON that exists precisely to answer this.
- Defends it as an active constraint.

**Follow-up questions**
- `run_guidance_diagnostics` warns below 0.70, which is unreachable. Is that warning ever useful?
- If the floor is inert, what *is* stopping the policy from suppressing preservation?

---

### Q6.6 — `src/inference.py` seeds `torch.manual_seed(base_seed + index_offset + i)` before every image, and defines `EVAL_PN_SEED_OFFSET = 1_000_000`. Explain both.
**Difficulty:** Medium

**Why this question matters**
A correct, non-obvious reproducibility contract that spans three files. Either the candidate designed it or they can't explain why the offset is a million.

**Expected good answer**
`generate_counterfactual` draws `init_noise` and per-step background-lock noise from the **global** RNG. Without reseeding, image *k*'s counterfactual depends on how many images were generated before it in that process — so results depend on how the work is chunked, and a sharded eval would not match a serial one. Seeding from the **global** index makes each counterfactual a function of its position in the full list alone, so shard `[50, 100)` produces byte-identical output to a serial run over `[0, 400)`. That is what makes `eval_main.py`'s generate/reduce split report the same numbers as `--stage all`.
`EVAL_PN_SEED_OFFSET` puts the two directions in disjoint seed space. Without it, n→p image *i* and p→n image *i* would draw the same diffusion noise — different images, correlated noise, which correlates the two halves of the metric suite for no reason. A million is simply larger than any realistic `n_eval`, so the ranges can never collide. Every caller that generates both directions must use `0` and this constant; `stage_generate` uses `index_offset=n_lo` and `EVAL_PN_SEED_OFFSET + p_lo` correspondingly.
The verification recipe is in the `eval_main.py` docstring: `--stage all --n-eval 8` versus two `--num-shards 2` shards + reduce.

**Red flags**
- "For reproducibility" with no mechanism.
- Doesn't know what the offset prevents.
- Doesn't know it depends on `scan_groups` being order-deterministic (the other half of the contract).

**Follow-up questions**
- What makes `scan_groups` deterministic? What would break it? *(shuffle=False + sorted groupby on patientId)*
- Did you actually run the 8-image verification? What did it show?

---

# Section 7 — Debugging

**Difficulty: Medium → Hard · Estimated time: 20 min**
*These should produce stories with timestamps, job IDs and numbers. Vagueness is the signal.*

---

### Q7.1 — What is the single worst bug you shipped and later caught? Tell me how you found it, not just what it was.
**Difficulty:** Medium

**Why this question matters**
The *detection method* is what distinguishes someone who debugs from someone who was told.

**Expected good answer (candidate-specific; strong candidates pick one of these and can date it)**
Strong candidates:
- **`DELTA_MIX` was dead code.** `compute_gae` returned `shaped_rewards`, `update()` unpacked it and never referenced it — every loss used `advantages`, `returns` or a separately-computed `td_targets`. So a documented, configured piece of reward shaping had *zero* effect on training for the entire first phase. Detection: reading the unpacking against the usage, or noticing `plot_reward_trace`'s two curves were visually coincident.
- **`direction_scale` partial application.** Applied only to the "push" terms (`prog, val, close2, hit, stop`) and not `fid`/`pres`, making preservation 2.25× *less* important precisely on the direction that already struggled. Detection: the qualitative diff plots — 81 % of pixels changed with foreign-object corruption on n→p specifically.
- **Sigmoid saturation.** `guidance_bias` measured 1.2976 ± 0.0054 and 0.7005 ± 0.0012 — the two ends of its range — across 20 episodes sampled *with* σ ≈ 0.61 exploration noise. Detection: `run_guidance_diagnostics` was written specifically to catch this, and the std was the tell.
Any of these, told with the detection method and the fix, is a good answer.

**Red flags**
- A generic story ("shape mismatch", "CUDA OOM").
- Can't say how it was found.
- Names a bug but not its blast radius (which results are invalidated).

**Follow-up questions**
- Which previously-reported results does that bug invalidate?
- What would have caught it a week earlier?

---

### Q7.2 — v1 measured `avg_ep_len = 1.00` on the n→p direction. Explain what that means and everything you changed because of it.
**Difficulty:** Hard

**Why this question matters**
This is the deepest *conceptual* failure in the project's history — the MDP had collapsed to a bandit — and the fix touched four different places. It's the best test of whether the candidate reasons about RL structure.

**Expected good answer**
Episodes were terminating after one step, so `BUDGET_T`, GAE, the value head and temporal credit assignment were all vestigial — PPO was solving a contextual bandit. Two independent causes:
1. **Score-only stop criterion**: validity was `|f(x_cf) − c| < ε` alone. A cheap diffuse wash satisfied it in a single step.
2. **Action range too wide**: `cls_scale` was `0.5 + 2.5·σ` (max 3.0), so one step could brute-force the classifier flip.
Four changes:
- Added the **joint** validity condition: score *and* `edit_l1 < EDIT_BUDGET_L1 = 0.02`. Calibrated deliberately below v1's measured 0.042/0.029 — reachable, but not by the v1 strategy.
- **Narrowed** `cls_scale` to `0.3 + 0.9·σ` (max 1.2), so reaching the target requires several steps.
- **Reduced** `BUDGET_T` 25 → 8, because with the joint criterion episodes now actually run to the budget and 25 steps × 2.5 passes was a 10–25× cost increase.
- **Fixed the horizon** (`FIXED_EPISODE_LENGTH=True`): episodes run exactly 8 steps and never terminate on validity, which (a) makes per-episode cost plannable against a hard 24 h walltime, and (b) turns `R_hit`/`R_stop` from a terminal payout the policy could collect and quit on into per-step terms — the objective becomes "reach validity and *hold* it."
Crucially, the candidate should note the honest caveat in the code: fixing the horizon was **not** a speed optimisation. Job 17653310 already logged `avg_ep_len=8.00` on every update — `is_valid` never fired and episodes already ran the full budget. It buys predictability, not wall clock. And `avg_return` is no longer comparable to pre-flag runs; compare `frac_valid` instead.

**Red flags**
- Doesn't know why `avg_ep_len=1` is a problem.
- Names one cause instead of two.
- Claims the fixed horizon made things faster.
- Doesn't know `avg_return` became incomparable.

**Follow-up questions**
- With `frac_valid = 0 %` in the current logs, has the fix over-corrected?
- If `avg_ep_len` pins at exactly `BUDGET_T`, the config says the fix is to *loosen* `EDIT_BUDGET_L1`. Why not raise `BUDGET_T`?

---

### Q7.3 — Job 17644171 saved both policies and then produced nothing. Job 17653310 spent 24 hours and left nothing recoverable. Job 17665348 was cancelled at 12 h. What did each one teach you, and what did you build in response?
**Difficulty:** Medium

**Why this question matters**
Three distinct infrastructure failures, three distinct fixes, all documented. This is real research-engineering experience and it should be told fluently.

**Expected good answer**
- **17644171**: train+eval in one process. Both policies were written (22:21 and 11:05) and then eval was killed at the 24 h boundary — the 20 h of PPO was recoverable in principle but no entry point could reload it. → Split into `main.py` / `eval_main.py` with `pipeline.build_common()` shared, so both construct identical objects, and eval reloads checkpoints instead of depending on a live process.
- **17653310**: SIGKILLed at the walltime ~30 of 100 updates into `n_to_p` with no PPO checkpoint yet — `save_checkpoints()` only ran at the very end of a direction. → `PPOTrainer._save_state` every 5 updates via `.tmp` + `os.replace`, carrying optimizer *and* the sparsity/CV schedule state; plus `PPO_WALLCLOCK_BUDGET_S` (22 h of 24), which stops *before* beginning an update that would overrun, using the **slowest** update seen rather than the mean (CV-probe updates cost ~3× a plain one). Plus `--direction` splitting `run_ppo.slurm` into a 2-task array so each direction gets its own 24 h.
- **17665348**: eval alone was cancelled at 12 h — ~4.4 min/image × 400 images ≈ 29 h. `sinfo -p publicgpu` caps TIMELIMIT at 24 h, so *no* walltime value can make single-process eval fit. This is a hard partition limit, not a tuning problem. → 8-way `--stage generate` array + a cheap `--stage reduce`, with per-global-index seeding for shard invariance and a `current_cfg_stamp()` consistency check that hard-fails on missing shards or config drift.

**Red flags**
- Cannot distinguish the three.
- Attributes the eval failure to "it was slow" rather than a partition cap.
- Doesn't know why the wall-clock estimate uses max rather than mean.

**Follow-up questions**
- Why does `_PROCESS_START_S` get taken at module import rather than at `train()` entry?
- `train()` returns `bool` and `main.py` writes *nothing* on `False`. Why is writing a half-trained `ppo_policy.pt` worse than writing nothing?

---

### Q7.4 — Give me an experiment that failed and that you did *not* fix.
**Difficulty:** Medium

**Why this question matters**
Everyone has a fixed-bug story. Unfixed failures are where honesty and prioritisation show.

**Expected good answer**
Candidates should have several available:
- **SCP is degenerate.** `sigmoid(−‖YOLO feat‖/50)` on the lower third saturates: std 0.0008 on mean 0.479 across all four groups, i.e. it's a constant and its paired t-tests are testing numerical noise. The code *detects* this (`scp_degenerate` at < 2 % relative spread) and refuses to print the t-tests — the right response to an unfixable metric, but SCP is still in the results dict.
- **FOP is uncomputable offline** — no shipped weight set has the head.
- **p→n never stopped being adversarial** by the transfer test: 0.132 (nih) / 0.143 (chex), both below the 0.15 "ADVERSARIAL" threshold the report itself defines. The ensemble was introduced *because* of this, but there's no post-ensemble held-out validation on disk.
- **The ablation matrix was never completed.** `config.py` lays out ARM 1 / ARM 2 / ARM 3 explicitly with an instruction to flip them one at a time and re-run eval between arms; all three are currently `True` simultaneously, and the only `eval_metrics.json` on disk was produced with all three `False`.

**Red flags**
- "Nothing failed."
- Names only trivial failures.
- Doesn't know the ablation arms were never separated.

**Follow-up questions**
- Which failure costs you the most in a paper review?
- If you had two more GPU-weeks, in what order do you attack these?

---

# Section 8 — Hyperparameters

**Difficulty: Medium → Hard · Estimated time: 15 min**
*Every value below is real and currently in `config.py`. Do not ask about values that aren't.*

---

### Q8.1 — `N_EPISODES = 800`, `episodes_per_update = 8`, `n_epochs = 4`, no minibatching. Compute the actual size of one PPO update and tell me whether that's sane.
**Difficulty:** Hard

**Why this question matters**
Forces arithmetic on the training regime. The numbers are small enough to be alarming, and the candidate should be alarmed.

**Expected good answer**
One update = 8 episodes × 8 steps = **64 transitions**, used as a single full batch for 4 gradient epochs. Total training = 800 episodes = 6 400 transitions = 100 updates. For a ~200 k-parameter conv policy that is very little data, and 4 full-batch epochs on 64 samples with no minibatch shuffling means each update is close to 4 identical gradient steps. Standard PPO implementations use 2 048–4 096 steps per update split into minibatches of 64. Here the *whole* update is one minibatch.
Consequences: advantage normalisation over 64 samples is noisy; the PPO clip does very little because the policy barely moves within an update; and the value function is fit on 64 points. `config.py` is explicit that 800 is "a conservative placeholder — don't treat it as tuned," with an ACTION REQUIRED note to benchmark 16 episodes and set `N_EPISODES = ppo_budget_hours·3600 / seconds_per_episode`. There is no evidence in the logs that benchmark was ever done and the value revised.
Also worth catching: `log_every = max(1, N_EPISODES // 10) = 80`, so the trainer logs at update 1 and update 80 only — which is exactly why the log file contains a single reward-split line.

**Red flags**
- Doesn't compute 64.
- Thinks this is a normal PPO batch size.
- Doesn't know the benchmark note exists, or claims it was done without evidence.

**Follow-up questions**
- At ~1 s per `generate_counterfactual` call and ~2.9 passes per step, how long is one episode? One update? 100 updates?
- If you could only increase one of `N_EPISODES` or `episodes_per_update`, which?

---

### Q8.2 — `T_DIFF = 50`, and `strength ∈ (0.2, 0.7)`. How many UNet evaluations does one PPO *episode* cost?
**Difficulty:** Medium

**Why this question matters**
Cost literacy. This number governs every design decision downstream.

**Expected good answer**
Per pass: `50 × strength` steps → 10 to 35. Per env step: × `n_passes` (2–4, measured mean 2.84 for n→p, 3.03 for p→n) → ~28 to 140. Per episode: × `BUDGET_T = 8` → roughly **250 to 1 100 UNet evaluations**, and on the 60–80 % of steps past the guidance ramp, each one *also* carries a VAE decode plus two decoder backwards plus a DenseNet ensemble forward+backward plus a YOLO forward. The code's own estimate is ~1 250–4 500 UNet evals per episode (counting the guidance forward passes). That's why one Grad-CAM forward+backward per *episode* is described as negligible, and why 400 eval images take 29 hours.
Why 50 and not 20 or 1000: DDIM's whole point is that 50 steps approximates the 1000-step DDPM trajectory well; going lower degrades sample quality, going higher is linear cost with diminishing returns.

**Red flags**
- Can't do the multiplication.
- Doesn't know `strength` changes the step count.
- Doesn't connect this to why eval had to be sharded.

**Follow-up questions**
- The guidance ramp starts at `0.1–0.5` of the trajectory. What fraction of DDIM steps carry the expensive backward?
- Would 20 DDIM steps be acceptable? How would you decide?

---

### Q8.3 — `GAMMA = 0.99` with an 8-step horizon. Comment.
**Difficulty:** Medium

**Why this question matters**
γ = 0.99 is copy-pasted from continuous-control defaults with 1000-step episodes. Over 8 steps it's essentially 1.0.

**Expected good answer**
Over 8 steps, `0.99⁷ ≈ 0.932` — the discount barely does anything. The effective horizon `1/(1−γ) = 100` is 12× the actual episode length, so γ = 0.99 is functionally undiscounted. That's arguably *fine* for a fixed-horizon episodic task where you genuinely care about the terminal state equally — but it means γ is not a tuned parameter, it's a default. A candidate who says "it's effectively 1.0 and that's appropriate here, but I should have written 1.0 and said so" is answering well. One who says "0.99 balances short and long-term reward" over 8 steps is reciting.
Note the interaction: with `FIXED_EPISODE_LENGTH`, `R_hit`/`R_stop` accrue on every step validity is held, so a near-undiscounted return directly counts "how many of the 8 steps were valid" — which is the intended semantics.

**Red flags**
- Generic discount-factor explanation.
- Doesn't compute `0.99⁷`.

**Follow-up questions**
- `GAE_LAMBDA = 0.95` over 8 steps — same question.
- `compute_gae` is called with `last_value = 0.0`. With a fixed horizon, when is that the right bootstrap?

---

### Q8.4 — Justify `EDIT_BUDGET_L1 = 0.02` and `EPSILON_STOP = 0.08` *together*, using the numbers from the training log.
**Difficulty:** Hard

**Why this question matters**
These two jointly define `is_valid`, and the log shows the joint criterion is currently unreachable. This is the highest-value hyperparameter question in the set.

**Expected good answer**
`EDIT_BUDGET_L1 = 0.02` was calibrated against measured v1 counterfactuals (MAE 0.042 n→p, 0.029 p→n over 200 images each), deliberately set below what the old policy produced — reachable but not by the v1 strategy. `EPSILON_STOP = 0.08` requires the ensemble's mean probability to land within 0.08 of the target.
The log says: `budget_ok = 86 %`, `score_ok = 2 %`, `valid = 0 %`, `edit_L1 = 0.0144` (comfortably under 0.02). So the budget half is nearly free and the *score* half is binding — the opposite of the intended design, where the budget was supposed to be the hard part. `frac_valid = 0 %` means `R_hit` and `R_stop` contribute exactly nothing, which is confirmed by `hit=+0.000(0%) stop=+0.000(0%)` in the reward split.
The correct action is to loosen `EPSILON_STOP` (or shape the flip reward continuously) rather than touching the budget. Note the tension with metrics: CV uses τ = 0.8 and a boundary crossing at 0.5 — a much weaker condition than `|f − c| < 0.08`. So the *environment* demands far more than the *metric* rewards, which is why CV can be 0.975 while `frac_valid` is 0.

**Red flags**
- Justifies each in isolation.
- Doesn't check the log.
- Doesn't notice `is_valid` and CV use different criteria.

**Follow-up questions**
- The env requires `|f − c| < 0.08`; CV requires a 0.8 move plus a boundary cross. Should they be the same criterion?
- With an *ensemble* mean probability, is 0.08 harder or easier than with a single model?

---

### Q8.5 — `ALPHA_CLS = 4.0` but the code multiplies by an extra `1.15`, and `ALPHA_PRES = 0.05` — an 80× ratio before the policy's per-step scaling. Where did those come from?
**Difficulty:** Medium

**Why this question matters**
Two magic numbers, one of them hidden inside a source line rather than in config. Tests whether the candidate remembers tuning them or inherited them.

**Expected good answer**
Honest answer expected: these are hand-tuned constants, and the `1.15` is an undocumented in-line multiplier at `generator.py:579` that should be folded into `ALPHA_CLS` or made a config value. The 80× ratio isn't arbitrary in the sense that the two gradients have very different natural scales — `∇L_cls` is the gradient of a BCE on a probability, `∇L_pres` is the gradient of a mask-normalised L1 plus a feature MSE, and the latter is numerically much larger, so `ALPHA_PRES` has to be small to keep them comparable. But the *specific* values are not derived.
This connects directly to Q6.3: `ALPHA_PRES = 0.05` is precisely what makes the preservation branch lose fp16 headroom and why the two backwards can't be fused.
The policy then modulates these with `cls_scale ∈ (0.3, 1.2)` and `pres_scale ∈ (0.3, 1.5)`, plus `guidance_bias` and per-pass escalation — so the effective ratio the policy can reach spans roughly 20× to 320×.

**Red flags**
- Claims a principled derivation with no evidence.
- Doesn't know about the hardcoded `1.15`.
- Doesn't connect `ALPHA_PRES` to the fp16 discussion.

**Follow-up questions**
- Did you ever log the two gradient norms side by side? What would that tell you?
- Should the ratio be a policy action instead of two fixed constants? *(It partly is — `guidance_bias`.)*

---

# Section 9 — Experiments

**Difficulty: Hard → Expert · Estimated time: 20 min**

---

### Q9.1 — CV is 0.975. `heldout_validation.py` exists because you didn't believe it. Explain the circularity and what the transfer ratios told you.
**Difficulty:** Hard

**Why this question matters**
The central scientific problem of the project, and the candidate wrote a whole diagnostic script about it. Non-negotiable.

**Expected good answer**
`pipeline.build_common()` constructs **one** classifier object that flows to the guidance gradient in `generate_counterfactual`, to `d_t` (episode reward and validity) in the env, to the far-pool scan, to the Xn/Xp split in `scan_groups`, **and** to CV in `run_full_evaluation`. So CV measures "did gradient descent on *f* succeed against *f*." It is circular by construction and is satisfied just as well by an adversarial perturbation as by a semantic edit.
The corroborating fingerprint: CV 0.975 while changing 66 % / 62 % of pixels at MAE 0.042 / 0.029 — small-amplitude, wide-support, which is exactly what an adversarial perturbation looks like and not what a focal consolidation looks like.
`heldout_validation.py` re-scores the *same* counterfactuals with classifiers that never touched the guidance loop. Results: transfer ratio nih 0.205 (n→p) / 0.132 (p→n), chex 0.590 / 0.143, against the script's own thresholds (> 0.5 transfers, < 0.15 adversarial). So **p→n is adversarial in both validators**, and n→p is partial-to-good. The agreement control passes (Spearman ρ 0.697 / 0.707 on real images), so the validators are usable and the low ratios are not validator disagreement.
Edit concentration corroborates model-free: 0.282 / 0.297 against a 0.05 uniform baseline, where calibration says a compact 12×12 patch scores 1.000 and a diffuse 75 %-of-pixels edit scores 0.067. So there *is* a concentrated core but most of the edit mass is in a diffuse tail.
The response was structural, not cosmetic: move `chex` into the guidance ensemble (`GUIDANCE_ENSEMBLE_WEIGHTS`), on the argument that adversarial directions rarely survive across independently trained models, and rotate the validator set to `mimic_ch, pc, nih` to keep them disjoint.

**Red flags**
- Reports CV 0.975 as the headline result without qualification.
- Doesn't know the transfer ratios or which direction failed.
- Doesn't know the agreement control's purpose.
- Doesn't know that moving `chex` into guidance disqualified it as a validator.

**Follow-up questions**
- The heldout JSON stamps `USE_GRADCAM_MASK=False, BUDGET_T=25`. So does it describe the *current* system?
- Is there any post-ensemble held-out validation on disk? *(No.)*
- The ensemble makes CV, group sizes and far-pool sizes incomparable to pre-ensemble runs — `build_ensemble_classifier` prints exactly that warning. Which of your reported numbers are now stale?

---

### Q9.2 — FID uses `_XRVFeatureExtractor` with `densenet121-res224-rsna` weights — the same corpus as a guidance ensemble member. Is your FID measuring realism?
**Difficulty:** Expert

**Why this question matters**
A second, subtler circularity that the codebase does *not* comment on. Almost nobody catches this unprompted.

**Expected good answer**
Partially, no. FID is computed in the 1024-d penultimate feature space of an RSNA-trained DenseNet-121 — architecturally and corpus-wise the same family as `members[0]` of the guidance ensemble. A perturbation optimised to move that model's *output* will also move its *features*, so the same adversarial direction that inflates CV can inflate FID. This is consistent with the observed pattern the code itself flags: v1 measured MAE 0.036 with FID 103.8 — a tiny pixel change producing a huge feature-space shift, which the comment correctly identifies as the signature of an off-manifold edit.
The mitigation already present is good: `run_full_evaluation` computes **cross-group** FID as well — `FID(Xp, Xc_n→p)` and `FID(Xn, Xc_p→n)` — the counterfactual against its *target* class. Same-group FID is ambiguous (high could mean a real class change *or* an artifact); cross-group is the actual realism test where low is unambiguously good, and high-on-both means off-manifold. Those cross-group values are computed in the current code but are **absent from the `eval_metrics.json` on disk**, because that JSON predates the change.
The clean fix: report FID in a feature space independent of guidance — the `inception` backbone is already implemented as an option, or use `-mimic_ch`/`-pc`.

**Red flags**
- Doesn't see the circularity.
- Can't explain what cross-group FID adds.
- Doesn't notice the on-disk JSON lacks the cross-group fields.

**Follow-up questions**
- Which xrv weight set would you use for an independent FID, and why not `-all`?
- FID with N ≈ 200 against 1024 feature dims — the docstring warns about this. Quantify the bias direction.

---

### Q9.3 — Which config produced `checkpoints/eval_metrics.json`, and can you reproduce it today?
**Difficulty:** Hard

**Why this question matters**
Provenance. The candidate built `current_cfg_stamp()` specifically to prevent this problem, and the artefact on disk predates the mechanism.

**Expected good answer**
The heldout cache stamps `ANCHOR_TO_EPISODE_ORIGIN=True, USE_GRADCAM_MASK=False, USE_SPARSITY_REWARD=False, BUDGET_T=25, ENV_REFINEMENT_PASSES=2`, and its numbers (CV, edit concentration) line up with `eval_metrics.json`. Current `config.py` has `BUDGET_T=8`, all three ablation flags `True`, `LORA_R=16` (was 8, with the rank-8 artefacts archived under `checkpoints/archive_v1_rank8/`), a two-member guidance *ensemble* (was a single rsna model), `FIXED_EPISODE_LENGTH=True`, and a rebalanced reward.
So **no**, it cannot be reproduced today without reverting config.py wholesale — and the current `--stage reduce` would refuse to even pool those counterfactuals, because `_load_shards` compares the cached stamp against `current_cfg_stamp()` and raises on drift. That check is the candidate's own work and it is doing exactly its job.
The honest conclusion: `eval_metrics.json` describes **v1**, not the system as configured. Any report table citing it alongside current hyperparameters is mixing two systems.

**Red flags**
- Claims the numbers are current.
- Doesn't know `current_cfg_stamp()` exists or what it covers.
- Doesn't recognise the report/JSON mismatch as a provenance problem.

**Follow-up questions**
- `current_cfg_stamp()` deliberately excludes reward-shaping knobs. Justify that exclusion.
- If you re-ran eval today with the current config, which metric do you predict moves most, and in which direction? *(Config predicts CV should DROP — "CV 0.975 with 75 % of pixels changed is a worse result than CV 0.7 with 5 % changed.")*

---

### Q9.4 — `config.py` specifies a three-arm ablation to be flipped one at a time with eval between arms. All three are on. What can you actually attribute?
**Difficulty:** Expert

**Why this question matters**
Directly tests experimental discipline against an experimental protocol the candidate wrote down and then didn't follow.

**Expected good answer**
Nothing, at the level of individual mechanisms. The protocol is explicit:
```
ARM 1  ANCHOR_TO_EPISODE_ORIGIN = True
ARM 2  + USE_GRADCAM_MASK       = True
ARM 3  + USE_SPARSITY_REWARD    = True
```
"flip these ONE AT A TIME, re-running eval between arms, so the effects stay attributable." All three are currently `True` and the only eval on disk is from all three `False`. So the *only* available comparison is all-off vs all-on — a three-factor change measured as one. Worse, the confounds are known and documented: `LAM_FID` overlaps `R_sparsity` (both magnitude penalties, differing only in norm and spatial weighting), `SPARSITY_OUTSIDE_CAM_ONLY` is *inert without* `USE_GRADCAM_MASK`, so ARMs 2 and 3 are not even independent, and `BUDGET_T` changed 25→8 simultaneously.
A strong candidate states the minimum honest claim ("the combination of the three constraints changes X") and prices the proper experiment: 4 arms × ~29 h generation, or run the ablation at reduced `n_eval` with the sharded pipeline that already exists.

**Red flags**
- Claims per-mechanism attribution.
- Doesn't know ARM 3 depends on ARM 2.
- Doesn't know the cost of the correct experiment.

**Follow-up questions**
- Design the cheapest experiment that separates the anchor effect from the CAM-mask effect. What's the minimum `n_eval`?
- If you could only run ONE ablation arm, which and why?

---

# Section 10 — Research Thinking

**Difficulty: Hard → Expert · Estimated time: 15 min**

---

### Q10.1 — Restart the project today, same 3 months, same GPUs, same supervisor. What do you build differently on day one?
**Difficulty:** Hard

**Why this question matters**
Separates "I would tune better" from structural insight.

**Expected good answer**
Should lead with the *evaluation* problem, not the model. Strong versions:
- **Hold out a real test split on day one.** There is currently none: PPO samples from the far pool of the full RSNA set, and `scan_groups` scans that same set. Everything reported is in-sample.
- **Make the guidance classifier and the evaluation classifier disjoint from the start**, rather than discovering circularity in month three and patching with an ensemble.
- **Establish the "adversarial or semantic?" test before building the generator** — edit concentration and off-guidance transfer are cheap and would have caught the diffuse failure mode in week two instead of month three.
- **Build the sharded eval first.** Three separate jobs died to walltime before the infrastructure existed.
- Possibly: skip PPO initially. Establish that a *hand-tuned* guidance schedule can produce localised, transferring counterfactuals; only then ask whether a learned schedule beats it. There is no such baseline in the repo, so the value added by PPO over a fixed schedule is currently unmeasured.

**Red flags**
- Only hyperparameter changes.
- Doesn't mention the missing train/test split.
- Doesn't question whether PPO was necessary.

**Follow-up questions**
- Is there a fixed-schedule baseline anywhere? How would you build one in a day? *(`generate_counterfactual` already has defaults: M=5, strength=0.55, scales 1.0 — `run_full_evaluation` falls back to exactly that when policies are `None`.)*
- If the fixed schedule matched PPO, would the project have failed?

---

### Q10.2 — Unlimited compute, 6 months. What's the actual research question you'd chase?
**Difficulty:** Expert

**Why this question matters**
Tests whether the candidate sees past the implementation to a question worth answering.

**Expected good answer**
Good answers name a *falsifiable* question, not a bigger model. Examples:
- "Can a counterfactual generator be constrained so that validity against a held-out classifier is guaranteed rather than measured?" — i.e. make transfer a training objective, e.g. by requiring ensemble consensus with adversarially-trained members, or by certifying the edit's support.
- "Does a learned sampling schedule beat a tuned fixed one, and on which metric?" — the null result that the current setup can't rule out.
- "What is the minimal sufficient edit?" — formalise counterfactual minimality as an L0 problem and measure the Pareto front of validity vs support size, rather than trading them off with hand-set λ.
- Reader study: do radiologists agree the highlighted region is where they'd look? That's the only ground truth that matters for an *explanation* method, and no proxy metric substitutes.

**Red flags**
- "Scale up, use SDXL, train longer."
- No falsifiable claim.
- Doesn't mention any form of human/clinical validation.

**Follow-up questions**
- How would you get radiologist annotations without a hospital partnership?
- What's the smallest experiment that would kill your favourite hypothesis?

---

### Q10.3 — Someone tells you the whole approach is unsound because a classifier's counterfactual explains the *classifier*, not the *disease*. Respond.
**Difficulty:** Expert

**Why this question matters**
The deepest conceptual objection to counterfactual XAI. Tests philosophical clarity under pressure.

**Expected good answer**
The objection is **correct**, and the right response is to accept it and narrow the claim. A counterfactual explains the model's decision boundary; that is the stated goal of the method. It becomes clinically meaningful only under an extra assumption — that the classifier has learned disease-relevant features — and that assumption is *testable*, which is exactly what `check_classifier_localization.py` (Grad-CAM: does the classifier look at lung pathology or at global brightness?) and `heldout_validation.py` (does the edit transfer to independently-trained models?) are for. Both exist because the candidate's own supervisor or their own reading raised this.
The strongest version adds: the counterfactual is *most* useful precisely when it reveals the model is wrong — a counterfactual that flips the classifier by brightening the whole thorax is a valuable finding about the classifier, not a failed explanation. The failure mode isn't producing such an edit; it's *reporting* it as a disease explanation. And note this is precisely why the p→n transfer ratio of 0.13 matters more than the CV of 0.955.

**Red flags**
- Defensive dismissal.
- Claims the counterfactual explains the disease.
- Doesn't reference the diagnostics they built for exactly this.

**Follow-up questions**
- Your classifier is a frozen pretrained DenseNet you never trained. Does that make the objection stronger or weaker?
- What would convince a sceptical radiologist?

---

# Repository Walkthrough

**Format:** open the file, screen-share, read the code together. Budget 5–8 min per item; pick 5–7.
**Difficulty:** as marked.

---

### W1. `src/generator.py` → `generate_counterfactual` (lines 375–608)
**Why it exists:** the core research contribution — guided DDIM with spatial locking.
**Ask:** "Walk me line by line from `n_keep` to the final decode. Every time a tensor changes shape or dtype, say so."
**Expected:** SDEdit init → per-step UNet → `pred_original_sample` extraction → ramp → decode-and-guide → mask → lock. Should identify that `z0_hat` and `z_t` come from the *same* `step_out` and that guidance uses the former to correct the latter.
**Common wrong answers:** "it denoises the image"; confusing `prev_sample` with `pred_original_sample`; thinking `trajectory` affects the output (it's only for plots).
**Difficulty:** Hard

### W2. `src/generator.py` → `compute_gradcam_latent_mask` (lines 229–335)
**Why it exists:** converts classifier attention into a hard action-space constraint.
**Ask:** "Why is percentile normalisation done *after* upsampling to 32×32, and why 0.99?"
**Expected:** the raw CAM is 7×7 at the DenseNet's 224 input; taking a percentile over 49 cells is meaningless, and min–max lets a single outlier cell define the scale and flatten everything else. Upsample first → percentile over 1024 cells. Then `cam^γ` (γ<1 broadens), blur to kill the blockiness of an upsampled 7×7, then `floor + (1−floor)·cam`.
**Common wrong answers:** "normalisation is standard"; not knowing the native CAM resolution; thinking the CAM is computed per-step.
**Difficulty:** Hard

### W3. `src/generator.py` → the background-lock block (lines 592–600)
**Why it exists:** the only hard spatial guarantee in the pipeline.
**Ask:** "Why re-noise the anchor to `t_next` rather than `t`? And what happens on the last iteration?"
**Expected:** `z_t` after `scheduler.step` is already at `t_next`, so the background must be brought to the same noise level or the composite mixes two noise scales. On the final step the *clean* anchor is used, pinning out-of-mask output to the anchor's VAE reconstruction.
**Common wrong answers:** doesn't know `prev_sample` is at `t_next`; thinks the final step also adds noise.
**Difficulty:** Expert

### W4. `factory/environment.py` → `step` (lines 477–580)
**Why it exists:** the action-to-generation interface.
**Ask:** "Decode all nine dimensions. Then explain the per-pass escalation formulas."
**Expected:** the table in §1.6, plus: strength grows `×(1 + 0.15·i)` capped at 0.9, cls_scale grows `×(1 + 0.35·i)`, pres_scale decays `×max(1 − 0.15·i, 0.4)`. So later passes push harder on the classifier and protect less — a hardcoded schedule *on top of* the learned one.
**Common wrong answers:** doesn't notice the escalation is hardcoded rather than learned; can't say why `n_passes` is action-controlled at all.
**Difficulty:** Medium

### W5. `factory/environment.py` → `_compute_reward` (lines 353–467)
**Why it exists:** the entire learning signal.
**Ask:** "Which of these terms could you delete with no effect on the policy gradient?"
**Expected:** `R_bud` (constant), and arguably one of `R_val`/`R_close2` (same signal). Should also note `R_hit`/`R_stop` are currently identically zero. Should explain the `weighted` dict exists so the split can be audited without re-deriving λs.
**Common wrong answers:** "all of them matter"; doesn't know `R_bud` is a constant.
**Difficulty:** Hard

### W6. `factory/environment.py` → `generate_via_policy` (lines 586–675)
**Why it exists:** the missing link between training and evaluation — before it, eval called `generator.generate_counterfactual` with *defaults*, so reported metrics never reflected the trained policy.
**Ask:** "The docstring warns against `max_steps`. Why?"
**Expected:** `get_observation()`'s step-fraction is `current_step / self.T`, computed against `self.T` not `max_steps`. A differing `max_steps` feeds the policy step fractions it never saw in training. Prefer temporarily setting `self.T` — which is exactly what `_measure_holdout_cv` does, in a `try/finally`.
**Common wrong answers:** doesn't know the observation contains a step fraction; doesn't know eval used generator defaults before this existed.
**Difficulty:** Hard

### W7. `factory/ppo.py` → `compute_gae` (lines 283–335)
**Why it exists:** advantage estimation over a flat, multi-episode transition list.
**Ask:** "Trace `next_val` through the reversed loop. What happens at `tr.done`?"
**Expected:** `done` resets `gae = delta_t` and sets `next_val = 0.0` for the boundary, then `next_val = tr.value` at the end of each iteration supplies `V(s_{t+1})` to the earlier step. Should notice `shaped_r[t]` uses `adv[t]` computed *before* the normalisation on line 330.
**Common wrong answers:** thinks the flat list breaks episode boundaries (it doesn't — `done` handles it); doesn't notice the pre-normalisation use of `adv`.
**Difficulty:** Expert

### W8. `factory/ppo.py` → `PPOTrainer.update` (lines 718–804)
**Why it exists:** the PPO objective plus four auxiliary losses plus the saturation hinge.
**Ask:** "Point at every place a config constant enters, and tell me which one you'd tune first."
**Expected:** `PPO_CLIP`, `C_VALUE`, `C_ENTROPY`, `LAM_FEAS_AUX`, `LAM_PROG_AUX`, `LAM_STOP_AUX`, `LAM_ACT_SAT`, `ACT_SAT_THRESHOLD`, plus the hardcoded `clip_grad_norm_(…, 0.5)` and `n_epochs=4`. Should note the grad-norm clip is *not* in config.
**Common wrong answers:** misses the hardcoded 0.5; can't justify a tuning order.
**Difficulty:** Hard

### W9. `factory/ppo.py` → `_measure_holdout_cv` (lines 528–573)
**Why it exists:** an automatic safeguard, not just a log line.
**Ask:** "Why import `counterfactual_validity` inside the function?"
**Expected:** avoids a circular import (`metrics` imports `inference` which imports `ppo`). Should also explain the `try/finally` restoring both `env.T` and `env.lambda_sparsity`, and that the CV criterion deliberately matches `run_full_evaluation` (τ=0.8, boundary 0.5) rather than defining a second notion of validity.
**Common wrong answers:** "style"; doesn't know the import cycle.
**Difficulty:** Medium

### W10. `src/gradcam.py` → `grad_cam` (lines 244–318)
**Why it exists:** shared between the diagnostic script and the PPO constraint, deliberately.
**Ask:** "Why `register_forward_hook` + `retain_grad()` instead of `register_full_backward_hook`?"
**Expected:** torchxrayvision's `DenseNet.forward` runs `F.relu(features, inplace=True)` on the exact tensor a backward hook would wrap, which PyTorch forbids. `retain_grad()` involves no custom-autograd-Function wrapping, so the downstream in-place op is handled by ordinary autograd versioning. Also: `find_target_layer` prefers a submodule literally named `features` because hooking a single internal DenseNet conv gives only ~32 of ~1024 channels (DenseNet concatenates), producing a near-uniform map.
**Common wrong answers:** "backward hooks are deprecated"; doesn't know why the full `features` block matters for DenseNet specifically.
**Difficulty:** Expert

### W11. `src/gradcam.py` → `ensemble_grad_cam` (lines 173–241)
**Why it exists:** passing an ensemble to `find_target_layer` **silently returns `members.0.model.features` and logs success** — a one-member CAM presented as the ensemble's, and member 0 is `rsna`, the 2-head model most prone to the shortcut the mask exists to constrain.
**Ask:** "Why average *after* per-map normalisation?"
**Expected:** so a member with larger raw activation magnitudes cannot dominate. Degenerate members are *skipped* rather than contributing a zero map that would drag the mean toward uniform; only if every member is degenerate does it raise.
**Common wrong answers:** doesn't see the silent-failure mode; would average raw maps.
**Difficulty:** Hard

### W12. `src/metrics.py` → `counterfactual_validity` (lines 178–233)
**Why it exists:** the headline metric.
**Ask:** "Why require both `moved ≥ τ` **and** `crossed`? Isn't crossing implied?"
**Expected:** no. A score moving 0.05 → 0.55 crosses but moves only 0.5 < τ. A score moving 0.05 → 0.90 does both. Conversely, 0.05 → 0.95 on an image the model already scored above 0.5 wouldn't be "crossing." The conjunction rules out both large-but-inconsequential moves and small boundary nudges. Should also note the `finite_mask` filtering and the NaN-on-empty contract.
**Common wrong answers:** thinks one implies the other; doesn't know τ = 0.8 comes from the paper.
**Difficulty:** Medium

### W13. `src/metrics.py` → `edit_concentration` (lines 713–760)
**Why it exists:** a model-free adversarial-vs-semantic discriminator that needs no downloads.
**Ask:** "Why is the uniform baseline exactly `top_frac`?"
**Expected:** if `|Δ|` is constant over all pixels, the top 5 % of pixels hold exactly 5 % of the total mass. So 0.05 is the noise floor by construction and any value above it measures concentration. Calibration in the docstring: a diffuse 75 %-of-pixels edit at MAE 0.037 scores 0.067; a compact 12×12 patch scores 1.000. Measured: 0.282 / 0.297 — a concentrated core with a heavy diffuse tail.
**Common wrong answers:** doesn't derive the baseline; can't interpret 0.28.
**Difficulty:** Medium

### W14. `eval_main.py` → `_load_shards` (lines 294–387)
**Why it exists:** every check in it is fatal on purpose.
**Ask:** "Why is a missing shard an *error* rather than a warning?"
**Expected:** a missing shard just makes Xn/Xp shorter, and FID's bias grows sharply as N falls — so a partial reduce emits a plausible-looking but wrong number rather than failing. Same for config drift: mixing two ablation arms produces a metric describing neither. It even tells you the exact resubmit command with the missing array indices.
**Common wrong answers:** "defensive programming"; doesn't connect it to FID's small-N bias.
**Difficulty:** Medium

### W15. `factory/finetuning.py` → `_extract_labels` + `_build_balanced_sampler` (lines 26–102)
**Why it exists:** RSNA is imbalanced and DICOM decode is expensive.
**Ask:** "Why the three-tier fallback, and what's the cost of the last tier?"
**Expected:** tries `.labels`/`.targets`, then `.samples` (RSNA stores `(pid, label)` tuples — no DICOM decode), then `.df`, then a full `__getitem__` scan which decodes all ~26 684 DICOMs just to read a float. The sampler uses `1/class_counts` weights with `replacement=True` for ~1:1 per epoch. The per-class loss tracking (`avg_L_diff_pos_batches` / `neg_batches`) exists to verify the balance is working rather than assuming it.
**Common wrong answers:** doesn't know why `.samples` is the fast path.
**Difficulty:** Medium

---

# Ownership Questions

*Almost impossible to answer without having worked on THIS repository. Use 8–12; ask follow-ups on any answer that lacks a number, a filename, or a job ID.*

**Difficulty:** Medium → Hard · **Estimated time:** 20 min for a subset

1. Which of the 18 commits are yours, and what does `be5b609 "many changes"` actually contain? Why is the history so coarse?
2. `checkpoints/archive_v1_rank8/README.md` exists. Who wrote it, what does it say, and why did you archive rather than delete?
3. `config.py` is 516 lines of which roughly 60 % is commentary with job IDs and measured numbers. Was that your convention or your supervisor's? What does it cost you?
4. Job 17648782 logged `avg_return=33.931` at `avg_ep_len=1.00`. Reconstruct the reward decomposition from that single number. *(The config does exactly this: hit×6=6.00, stop×8=8.00 → 14.00 of 15.08 = 93 %.)*
5. Which file took the longest to get right, in hours? What was the last thing you fixed in it?
6. `smoke_test.py` and `smoke_test_eval.py` are near-identical in size. Why do both exist? *(cold-process checkpoint reload, which train+eval-in-one-process would hide)*
7. What is `preflight_gpu.py` checking, and which failure prompted you to write it?
8. `run_ppo.slurm` has `--constraint="gpuh200|gpuh100|gpua100|gpul40s|gpua40|gpurtx6000"` and `--exclude=hpc-n983`. Why the explicit allowlist rather than excluding bad nodes? *(SLURM 22.05.10 has no NOT operator and `--prefer` arrived in 23.02)*
9. `PPO_WALLCLOCK_BUDGET_S = 22*3600` against a 24 h limit. Why 22 and not 23?
10. Why is `_PROCESS_START_S` captured at *module import* rather than at `train()` entry? What breaks otherwise? *(`--direction both` would hand the second direction a fresh full budget)*
11. Which config value have you changed the most times? What was its trajectory?
12. `yolov8n.pt` is checked into the repo (6.5 MB). Deliberate?
13. What is in `outputs/gradcam_check/`? What did comparing `gradcam_current_*` against `gradcam_xrv_*` tell you, and what did you do about it?
14. `checkpoints/smoke_all/` and `checkpoints/smoke_eval/` both contain `eval_metrics.json`. Are they real results?
15. Which parameter's behaviour surprised you most when you finally measured it?
16. `CV_CHECK_MAX_STEPS` was lowered 12 → 4. What did you discover was wrong with 12? *(applied as `min(CV_CHECK_MAX_STEPS, env.T)`, so at `BUDGET_T=8` the old value resolved to 8 — the claimed cost saving never happened)*
17. `ENV_EVAL_HOLDOUT_N = 24` and not 8 or 32. Derive 24 from `CV_DROP_TOLERANCE`.
18. `LAM_STOP` and `LAM_HIT` went 8.0/6.0 → 2.0/2.0 in one edit. What measurement drove that, and did it work?
19. The report's hyperparameter table lists `BUDGET_T=25`, `λ_stop=8.0`, LoRA rank 8. Config says 8, 2.0, 16. Which describes the system you're presenting?
20. The report says training was on "a single Tesla V100-32GB"; SLURM logs show a Quadro RTX 6000, and `config.py` has a revision note reasoning about V100 specifically. Which hardware did the reported results actually run on?
21. Report Table 2 gives FID 5.67 / 3.20; `eval_metrics.json` gives 103.83 / 59.73. Where does the discrepancy come from?
22. In the report's clinical table, CTR shows 0.479 ± 0.001 and SCP shows 0.424 ± 0.055 — the JSON has those the other way round. Walk me through how that table was produced.
23. CV mean appears as 0.965 in one report table and 0.970 in another; the JSON says 0.975. Which is right?
24. `checkpoints/reward_trace.png` — the report says "this rollout terminates in two steps." Under `FIXED_EPISODE_LENGTH=True` that's impossible. When was that figure generated?
25. `guidance_diagnostics_n_to_p.json` reports `avg_reward = −4.895` and p→n reports `−1.367`. Why is one direction 3.6× worse, and does that match the transfer ratios?
26. `pipeline.load_generator_weights` prefers the 1.7 GB `unet_finetuned.pt` over the 22 MB PEFT checkpoint, and the docstring admits they're equivalent today. Why keep the expensive path?
27. `main.py`'s `run_guidance_diagnostics` samples with `deterministic=False` while eval uses `deterministic=True`. Deliberate?
28. Which piece of this codebase would you be most nervous handing to a new intern, and why?
29. If your supervisor asked you to cut the repo to the 500 lines that matter, what survives?
30. Name one thing in this repository you disagree with but implemented anyway.

---

# Deep Technical Questions

*Senior-ML-engineer / PhD level. Use 5–8. Several have no clean answer — the reasoning is the assessment.*

**Difficulty:** Hard → Expert · **Estimated time:** 25 min for a subset

1. `encoder()` returns `latent_dist.sample()`, so `get_observation()` returns a different vector for the same `x_t` on every call. Quantify what that does to the value function's fit and to PPO's importance ratio. Would `.mode()` be strictly better?
2. Derive the correct Jacobian factor for mapping ∇ w.r.t. `pred_original_sample` into a perturbation of `prev_sample` under DDIM with η=0. What does its ᾱ-dependence imply for a *constant* `ALPHA_CLS`?
3. `guidance_weight` ramps linearly 0→1 across the trajectory, meaning guidance is *weakest* at high noise. Guidance literature generally argues the opposite. Defend or attack the schedule.
4. The `edit_mask` gates the guidance gradient *and* the lock composite, and `GRADCAM_MASK_FLOOR = 0.15`. Prove or disprove: "the output latent deviation per location is bounded by the mask value there."
5. `L_value = MSE(values, shaped_rewards)` regresses the critic onto a one-step reward. Derive the resulting bias in the advantage estimate as a function of γ and horizon.
6. PPO's entropy bonus is computed on the pre-squash Gaussian. Write the corrected entropy for a sigmoid-squashed policy and explain why the uncorrected version *actively rewards* saturation.
7. With `episodes_per_update=8` and 8 steps, advantages are normalised over 64 samples. Estimate the variance of that normalisation and its effect on the effective learning rate.
8. LoRA `alpha = r = 16` gives scaling 1.0. Under what conditions does `alpha/r > 1` help, and would this fine-tuning regime (26 684 images, 30 epochs, balanced sampling) be one of them?
9. SD1.5's UNet was trained on 64×64 latents. You run 32×32. Analyse the effect on the self-attention layers' effective receptive field and on the positional statistics the model learned.
10. `_build_condition` pools a 4096-d latent to 64 dims and emits a single cross-attention token. Show that softmax attention over one token is a constant, and derive what the cross-attention layer degenerates to.
11. During LoRA training, the conditioning contains a pooled version of the clean `z0` whose noise the model must predict. Bound how much of `L_diff` could be explained by that leak at high `t`.
12. `R_pres` uses `KL(softmax(O_x) ‖ softmax(O_next))` on a pooled YOLO feature vector. Explain why this is not a divergence between distributions in any meaningful sense, and propose a replacement with matching scale.
13. `_far_indices_eval = _far_indices[:24]` is a deterministic prefix in dataset order; `scan_groups` returns the first `n_eval` images passing a threshold, also in dataset order. Prove or disprove: the CV-probe "hold-out" images are inside the evaluation set. What does that do to the safeguard's meaning?
14. There is no train/val/test split anywhere. PPO trains on the far pool of the full dataset; `scan_groups` scans the same dataset. Characterise exactly what is being measured, and design the minimal fix that doesn't invalidate the existing checkpoints.
15. FID uses an RSNA-trained DenseNet whose sibling guides generation. Formalise the circularity and propose a feature space that breaks it without introducing a domain gap.
16. `frechet_inception_distance` adds `eps·I` before `sqrtm`. With N=200 and 1024 feature dims, the covariance has rank ≤ 199. Analyse the bias `eps=1e-6` introduces versus the instability it prevents.
17. Edit concentration of 0.282 against a 0.05 baseline: derive what mixture of "focal core + diffuse tail" produces exactly that number, and design a two-parameter statistic that separates the two components.
18. Transfer ratio nih 0.205 vs chex 0.590 on the *same* n→p counterfactuals. Give two competing explanations and an experiment that distinguishes them. *(Hint: 2 heads vs 11, NLP-mined labels at ~1 % prevalence, and corpus overlap.)*
19. Moving `chex` into the guidance ensemble because it was the best-transferring validator is a form of selection on the test set. Argue both sides.
20. The ensemble takes BCE of the **mean** probability. Compare against mean-of-per-member-BCE for the guidance gradient. Which routes more gradient to a saturated member, and why does the code claim the mean already does? *(`per_member` exists as the hook if it doesn't.)*
21. A constant CAM produces a uniform mask at `floor`. Design a *statistical* test that distinguishes "the classifier genuinely attends uniformly" from "the gradient vanished," using only the CAM.
22. `direction_scale = 2.25` is uniform and washes out under advantage normalisation. Construct a reward modification that *does* give the harder direction more signal without changing the optimal policy.
23. `R_fid` (L1) and `R_sparsity` ((1−cam)-weighted L1) are the same functional form. Give the exact condition under which they are linearly dependent, and what that implies for the ablation.
24. `_ms_ssim` adapts the number of scales to image size and renormalises the weights. Analyse whether a renormalised 4-scale MS-SSIM is comparable to the standard 5-scale value.
25. `_hf_correlation` maps Pearson [−1,1] to [0,1] via `(ρ+1)/2`, so uncorrelated images score 0.5. Is that a sensible floor for a preservation metric? What does 0.798 (p→n) actually mean?
26. `compute_ctr_batch` measures cardiac and thoracic diameter as the horizontal span of thresholded masks — max minus min column index. Analyse the sensitivity to a single false-positive pixel and propose a robust alternative.
27. `compute_scp` is `sigmoid(−‖f‖/50)` on the lower third of the image. Given the measured std of 0.0008 on mean 0.479, invert the function to find the feature-norm range and explain the saturation.
28. Two backwards vs one fused backward differ by 0.6–4.7 % in fp16 with `ALPHA_PRES=0.05`. Derive where in the decoder backward the precision is lost and propose a gradient-scaling factor that makes fusion safe.
29. `torch.manual_seed(base_seed + index_offset + i)` per image guarantees shard invariance. Identify what *else* it silently controls (dropout in LoRA? cuDNN algorithm selection?) and whether determinism actually holds on GPU.
30. Formalise "minimal counterfactual" as a constrained optimisation problem, then show which of the nine reward terms is a Lagrangian relaxation of which constraint — and which terms have no such interpretation.

---

# Beyond The Project

*Stretch questions. Use 3–5. Assess design reasoning, not correctness.*

**Difficulty:** Hard → Expert · **Estimated time:** 15 min

### B1. Replace PPO with something else. Pick an algorithm and tell me what has to change in `environment.py`.
**Difficulty:** Hard
**Listen for:** SAC needs a replay buffer — 73 731-d observations make that ~30 GB for 100 k transitions, so the observation must be compressed first (store `x_t` and re-encode, or store the conv trunk's 256-d feature). CMA-ES or a simple evolutionary search over a *fixed* schedule is a genuinely serious alternative given only 9 parameters and expensive evaluations. Bayesian optimisation over the 9-d action if a per-image constant schedule suffices. **Best answer:** first establish that a state-dependent policy beats a constant one at all — the guidance diagnostics (`guidance_bias_std ≈ 0.087`) suggest the learned policy varies only modestly.

### B2. Remove segmentation entirely. What breaks and what do you replace it with?
**Difficulty:** Hard
**Listen for:** `S(·)` appears in four places: `edit_mask` (via `latent_mask`), `R_pres`'s `seg_term`, the observation's 65 536-d mask channel, and CTR. Removing it kills the anatomical prior on *where* edits may occur, leaving only the Grad-CAM mask — which is exactly the thing the code says leaks onto mediastinum/diaphragm/border, and which the lung mask exists to intersect down. Replacements: a learned mask predicted by the policy (turns a hard constraint into a soft one — bad), an unsupervised saliency prior, or SAM with a lung-field prompt. CTR would need a different heart localiser.

### B3. Train end-to-end: backprop the counterfactual objective into the LoRA weights instead of using RL.
**Difficulty:** Expert
**Listen for:** requires differentiating through the whole DDIM trajectory — memory is the blocker (50 steps × UNet activations), addressable with gradient checkpointing or the adjoint/DEQ trick. Real objections: (a) the discrete actions (`n_passes`, extra-pass bits) aren't differentiable; (b) it collapses to a *fixed* schedule, losing the per-image adaptivity that motivates RL at all; (c) the classifier gradient is already used at inference — end-to-end training would bake it in and risk exactly the adversarial solution the held-out validation caught. **Best answer notes** this is essentially "train the LoRA with a classifier loss," which is the failure mode, and that RL's value here is arguably as a *regulariser* on how much the classifier can be exploited.

### B4. Get inference from 4.4 min/image to under 10 s. Where does the time go and what do you cut?
**Difficulty:** Hard
**Listen for:** budget = 8 steps × ~2.9 passes × ~25 DDIM steps × (UNet + VAE decode + 2 decoder backwards + ensemble + YOLO). Ordered attack: (1) fuse the two guidance backwards in fp32 — ~1.5×; (2) drop `BUDGET_T` to 1–2 at inference (the policy is nearly state-independent anyway); (3) distil to a few-step sampler (LCM/consistency); (4) guide on a *downsampled* decode or skip the decode using a latent-space classifier; (5) cache the anchor's `S_x`/`O_x`/CAM across passes — already done; (6) batch across images. Should identify the decoder backwards as the dominant term, per the code's own comment.

### B5. Scale from pneumonia to 14 CheXpert pathologies.
**Difficulty:** Expert
**Listen for:** 14 pathologies × 2 directions = 28 policies under the current architecture — obviously untenable, forcing the conditional-policy refactor already named as a limitation. `c` becomes a 14-d target vector; `cond_proj` already takes a scalar `c` and would extend naturally; the far-pool scan becomes per-pathology. Harder problems: co-occurrence (pneumonia and effusion appear together, so "flip pneumonia only" is a *constrained* edit that must hold other heads fixed) and the Grad-CAM mask becoming pathology-specific. Should note the reward needs a term penalising unintended flips on the other 13 heads.

### B6. Replace SD1.5 with something better suited.
**Difficulty:** Medium
**Listen for:** SDXL (bigger, 128×128 latents, better at 512+ but heavier); a medical-domain diffusion model (RoentGen, trained on MIMIC-CXR with text) — much better prior, and it has real text conditioning so `cond_proj` could be replaced by prompts; a latent VAE trained on CXRs (SD's VAE was trained on natural images and its round-trip error *is* the pipeline's noise floor). Should note that changing the VAE alone might be the highest-leverage single swap, since VAE round-trip error sets the floor on `frac_changed`.

### B7. Adapt to 3D CT.
**Difficulty:** Expert
**Listen for:** the honest answer is that almost nothing transfers. No 2D SD prior; options are slice-wise diffusion with a consistency constraint, a 3D latent diffusion model (memory-brutal), or a 2.5D approach. The observation would be enormous — the conv trunk becomes 3D. Segmentation is easier (TotalSegmentator). The *counterfactual* framing gets more interesting: a 3D nodule edit must be volumetrically coherent, and CT has real HU units, so "minimal edit" becomes physically meaningful rather than a [0,1] intensity heuristic. Should flag that per-image min–max normalisation (as in `ingestion.py`) is *wrong* for CT and must be replaced with HU windowing.

---

# Interview Evaluation Rubric

Score each category 1–5. Total /40.

| Category | Score (/5) | Notes |
|---|---|---|
| **Project Ownership** — can name what they built vs. inherited; recalls specific decisions, numbers, job IDs | | |
| **Diffusion Understanding** — SDEdit/strength, guidance mechanics, DDIM vs DDPM, LoRA, VAE, masking | | |
| **Reinforcement Learning** — MDP formulation, PPO objective, GAE, value targets, action-space design | | |
| **Software Engineering** — module boundaries, reproducibility contracts, checkpointing, failure handling | | |
| **Debugging Skills** — narrates real bugs with detection method and blast radius; not just fixes | | |
| **Research Thinking** — recognises circularity, designs ablations, states falsifiable claims, prices experiments | | |
| **Communication** — audience-adaptive, concise, says "I don't know" cleanly, engages with pushback | | |
| **TOTAL** | **/40** | |

### Scoring guidance per category

| Score | Meaning |
|---|---|
| 5 | Anticipates the follow-up; corrects the interviewer where the interviewer is wrong; cites specific numbers unprompted |
| 4 | Answers correctly and completely; needs one prompt for the deeper layer |
| 3 | Correct at the level of the code; cannot generalise beyond it |
| 2 | Recites comments/docstrings without evaluating them |
| 1 | Cannot answer, or answers something else |

### Interpretation

- **35–40 — Outstanding undergraduate researcher.** Owns the research reasoning, not just the code. Caught the circularity, can price experiments, and disagrees with parts of their own repo for stated reasons. Hire; give them a research question, not a spec.
- **28–34 — Strong AI internship candidate.** Solid grasp of both the diffusion and RL halves, real debugging stories, honest about limits. Will need guidance on experimental design but will execute well and ask the right questions.
- **20–27 — Good implementation but limited research depth.** Can explain what the code does but not why the alternatives were rejected; recites documented rationales without evaluating them. Fine for an engineering-heavy role; would need close supervision on research direction.
- **Below 20 — Needs stronger technical understanding.** Cannot explain core mechanisms (what `strength` does, what the value head regresses onto, why CV is circular) without reading the comments aloud. Reassess against a narrower, better-scoped project.

### Independent honesty flag (not scored — record separately)

Track how the candidate handles the **report-vs-repository discrepancies** (§1.15, Q2.3, Ownership 19–24): the missing GAN baseline implementation, FID 5.67 vs 103.83, the transposed CTR/SCP rows, the stale hyperparameter table, the V100/RTX-6000 mismatch.

- **Green** — acknowledges directly, explains the provenance, proposes the correction unprompted.
- **Amber** — acknowledges after being shown the evidence; doesn't volunteer the fix.
- **Red** — insists the report is accurate against contradicting artefacts in their own repository, or deflects.

A Red here should override a high total score. A Green combined with a mid score is a far better hire than the reverse.

---

*Interviewer note: this candidate's repository contains an unusually strong "negative-result infrastructure" layer — `heldout_validation.py`, `edit_concentration`, the SCP degeneracy detector, `DegenerateCAMError`, and the shard config-stamp check all exist to catch the project lying to itself. That is rare and worth probing hard, because it is either genuine scientific instinct or it was handed to them. Sections 5, 7 and 9 are designed to tell those two apart.*
