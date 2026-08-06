# ppo-contrafactual — Project Structure

A research pipeline that generates **medical counterfactual chest X-rays** (e.g., turning a pneumonia-negative image into pneumonia-positive and vice versa) using a Stable-Diffusion generator guided by a PPO-trained policy, and evaluates whether the edits are real disease signal or adversarial noise.

## Top-level entry points

| File | Purpose |
|---|---|
| `main.py` | **Training** pipeline (Steps 1-5): ingest RSNA data → load pretrained models → LoRA-finetune the generator → train two PPO policies (n→p and p→n directions) → save checkpoints. Split off from evaluation because training alone takes ~12-20h and a walltime kill mid-eval used to discard finished policies. |
| `eval_main.py` | **Evaluation** pipeline (Steps 6-6B): reloads checkpoints from `main.py`, runs the full metric suite (CV/FID/FOP/CTR/SCP) and generates counterfactual diff plots. Independent process so it can be re-run without repeating PPO training. Runs in **two stages** — `--stage generate --shard K --num-shards N` produces one shard of counterfactuals (SLURM array), `--stage reduce` turns all shards into `eval_metrics.json`. Generating 400 counterfactuals is ~29h and the partition caps jobs at 24h, so the single-process `--stage all` is for smoke tests only. |
| `pipeline.py` | Shared construction code (`build_common()`) used by both `main.py` and `eval_main.py` so they build identical objects (dataset, models, generator). |
| `config.py` | Central config: hyperparameters, device/dtype, model paths, constants. Imported everywhere instead of hardcoding values. |
| `ingestion.py` | Downloads the RSNA Pneumonia dataset via Kaggle and defines `RSNACXRDataset`. |
| `heldout_validation.py` | Diagnostic: re-scores generated counterfactuals with classifiers that were **never** part of the guidance loop, to check if edits are genuine disease signal vs. adversarial perturbations against the guidance classifier. |
| `check_classifier_localization.py` | Grad-CAM diagnostic to see whether the classifier looks at real lung pathology or a "shortcut" (brightness/contrast) when predicting pneumonia. |
| `smoke_test.py` | Cheap end-to-end smoke test of the **training** path (tiny budgets) to catch crashes/shape errors before spending real GPU time. |
| `smoke_test_eval.py` | Smoke test of the **evaluation** path specifically — tests cold-process checkpoint reload, which training+eval-in-one-process would hide. |

## `factory/` — core ML components

| File | Purpose |
|---|---|
| `factory/environment.py` | Gym-style RL environment: samples "far" images, calls the generator each step, computes shaped reward, decides episode termination. |
| `factory/ppo.py` | PPO implementation: `PPOActorCritic` network, `compute_gae` (advantage estimation), `PPOTrainer` (rollout + update loop, sparsity-reward annealing). |
| `factory/finetuning.py` | LoRA denoising fine-tuning loop for the SD1.5 UNet, with per-epoch checkpointing/resume. |
| `factory/evaluate.py` | Evaluation/visualization helpers: score printing, reward-trace plots, trajectory plots, checkpoint saving. |

## `src/` — models & generation utilities

| File | Purpose |
|---|---|
| `src/models.py` | Loads all frozen pretrained models: ResNet-18 classifier, TorchXRayVision segmenter, YOLOv8 feature extractor, diffusion generator, eval dataset. |
| `src/xrv_classifier.py` | Alternate classifier: TorchXRayVision's pretrained multi-pathology DenseNet-121, used instead of the custom ResNet-18 to avoid shortcut-learning issues. |
| `src/generator.py` | The Stable Diffusion 1.5 + LoRA generator: encode/decode latents, gradient-guided DDIM denoising to produce counterfactuals. |
| `src/gradcam.py` | Shared Grad-CAM implementation, used both to constrain the PPO action space and for the standalone diagnostic script. |
| `src/inference.py` | `DirectionalPolicy` wrapper bundling an env + trained PPO policy per direction, for evaluation-time generation. |
| `src/metrics.py` | Implements the Singla et al. (2022) counterfactual metrics: CV, FID, FOP, CTR, SCP. |
| `src/diff_viz.py` | Visualizes pixel-level diffs between original and counterfactual images (4-panel + bidirectional comparison). |

## Other directories/files

- **`checkpoint/` / `checkpoints/`** — saved model weights (classifier checkpoint, PPO policies, LoRA weights).
- **`outputs/gradcam_check/`** — Grad-CAM diagnostic PNGs.
- **`logs/`** — SLURM job stdout/stderr logs.
- **`report/report.tex`** — LaTeX writeup of the project/results.
- **`*.slurm`** — SLURM batch scripts to submit each corresponding Python entry point to the cluster (`run_ppo.slurm`, `run_eval_cf.slurm`, `run_eval_reduce.slurm`, `run_smoke_test.slurm`, `run_smoke_test_eval.slurm`, `run_classifier.slurm`, `run_heldout_validation.slurm`). Evaluation is two jobs: `sbatch run_eval_cf.slurm` (8-task array, generates counterfactual shards) then `sbatch --dependency=afterok:<jobid> run_eval_reduce.slurm` (computes the metrics). `run_eval_ppo.slurm` is the superseded single-job version, kept only for small local smoke tests.
- **`setup_env.sh`** — environment setup script.
- **`requirements.txt` / `requirements.lock.txt`** — Python dependencies (loose vs. pinned).
- **`yolov8n.pt`** — pretrained YOLOv8 nano weights used as an object-detector feature extractor.
- **`venv/`**, **`__pycache__/`**, **`.git/`**, **`.gitignore`**, **`.claude/`** — environment/tooling, not project logic.

## Overall flow

`ingestion.py` → `pipeline.build_common()` (shared by `main.py`/`eval_main.py`) → `src/models.py` loads frozen classifier/segmenter/detector → `src/generator.py` + `factory/finetuning.py` LoRA-tunes the diffusion model → `factory/environment.py` + `factory/ppo.py` train two directional PPO policies → `eval_main.py` reloads checkpoints and runs `src/metrics.py` + `src/diff_viz.py`, with `heldout_validation.py` and `check_classifier_localization.py` as sanity/diagnostic tools on the side.
