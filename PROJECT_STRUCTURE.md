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
| `domains/` | Dataset-domain registry. A `DomainProfile` (`domains/base.py`) binds the images, what `f(·)` targets, the guidance/validator models, the segmenters, the FID feature space and which clinical metrics apply — so adding a dataset is one new module plus one registry line, and a job selects it with `PPO_DOMAIN=<name>` (default `rsna_pneumonia`). Registered: `rsna_pneumonia`, `node21_nodule`, `cbis_ddsm_breast`, and the validation-only `cmmd_breast`. The first two use the torchxrayvision backend; the breast domain overrides every backend through the profile's `build_*` hooks, because torchxrayvision is chest-only. |
| `heldout_validation.py` | Diagnostic: re-scores generated counterfactuals with classifiers that were **never** part of the guidance loop, to check if edits are genuine disease signal vs. adversarial perturbations against the guidance classifier. |
| `splits.py` | The patient-level train/test split, computed once and persisted. LoRA fine-tuning, the PPO far-pool and the metrics group scan all read it, so the images the evaluation reports on are ones the generator and policy never saw. Before this existed there was no separation at all. |
| `seeding.py` | Seeds every RNG the training path draws from, and documents what that does *not* guarantee (a resumed run does not replay an uninterrupted one). Training was previously unseeded, so two identical submissions produced two different policies. |
| `benchmark_episode_cost.py` | Measures per-episode and per-LoRA-epoch cost on the actual GPU, then sizes `N_EPISODES` against the 24h partition cap. **Run before `run_ppo.slurm`** — `config.N_EPISODES` is an unbenchmarked placeholder and six jobs have been killed at their walltime. |
| `record_run_manifest.sh` | Writes commit, dirty-file list and the behaviour-driving config values next to both the log and the checkpoints, so a result can be traced back to the code that produced it. |
| `compare_arms.py` | Assembles the three-arm control table (trained / none / random) from the per-arm `eval_metrics.json` files. Refuses any metrics file older than the policies it claims to evaluate, so a pending or failed reduce cannot leave an earlier run's numbers sitting in the table. Note the arms are not all compute-matched: `trained` vs `random` isolates learning, `random` vs `none` isolates having a multi-step schedule. |
| `test_invariants.py` | CPU-only checks on the five properties whose silent failure produces a *plausible but wrong* number rather than a crash: split disjointness, `scan_groups` tensor↔index alignment, LHR calibration, metric serialisation, and the shard split/merge (a lossless partition, with partial or mixed-arm shard sets refused rather than reduced). Runs in ~1 min on the login node. Complements `smoke_test.py`, which catches the crash-shaped failures instead. |
| `check_classifier_localization.py` | Grad-CAM diagnostic to see whether the classifier looks at real lung pathology or a "shortcut" (brightness/contrast) when predicting pneumonia. |
| `smoke_test.py` | Cheap end-to-end smoke test of the **training** path (tiny budgets) to catch crashes/shape errors before spending real GPU time. |
| `smoke_test_eval.py` | Smoke test of the **evaluation** path specifically — tests cold-process checkpoint reload, which training+eval-in-one-process would hide. |
| `train_breast_classifier.py` | Trains `f(·)` for the mammography domain (3 backbones: 2 guidance + 1 held-out validator). **Required before `run_ppo.slurm` on that domain** — torchxrayvision has no breast head to borrow. Patient-level splits; refuses to pass below 0.65 val AUC. |
| `train_lesion_segmenter.py` | Trains the lesion U-Net the `LLR`/`MSI` clinical metrics are computed from, on CBIS-DDSM's shipped ROI masks. Deliberately kept **out** of the guidance loop so those metrics stay independent of the edits they measure. |
| `check_lesion_localization.py` | Gate for the mammography metric stack: breast-mask sanity, lesion Dice vs ground truth, **MSI calibration against the `mass margins` labels**, and Grad-CAM∩lesion overlap. MSI must separate SPICULATED from CIRCUMSCRIBED on *real* images before it may be reported at all. |

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
| `src/breast_classifier.py` | `f(·)` for mammography — the torchxrayvision-free counterpart of `xrv_classifier.py`, class-for-class mirrored so `pipeline.build_common` can swap one for the other with nothing downstream noticing. |
| `src/breast_segmentation.py` | The PSPNet replacement, which plays **two** roles the chest segmenter conflated: `BreastAnatomySegmenter` (analytic organ/pectoral/dense-tissue masks → `LOCK_BACKGROUND`) and `LesionSegmenter` (trained U-Net → the clinical metrics). |
| `src/generator.py` | The Stable Diffusion 1.5 + LoRA generator: encode/decode latents, gradient-guided DDIM denoising to produce counterfactuals. |
| `src/gradcam.py` | Shared Grad-CAM implementation, used both to constrain the PPO action space and for the standalone diagnostic script. |
| `src/inference.py` | `DirectionalPolicy` wrapper bundling an env + trained PPO policy per direction, for evaluation-time generation. |
| `src/metrics.py` | Implements the Singla et al. (2022) counterfactual metrics: CV, FID, FOP, CTR, SCP. |
| `src/diff_viz.py` | Visualizes pixel-level diffs between original and counterfactual images (4-panel + bidirectional comparison). |

## Other directories/files

- **`PPO-CF_CVPR_Paper/`** — LaTeX source of the CVPR 2027 paper and supplement (the working copy).
- **`CVPR2027_submission/`** — the built, anonymous upload: paper and supplement PDFs, the OpenReview abstract, and an Overleaf-ready copy of the sources.
- **`verification/`** — copy of the audit workspace behind every number in the paper: memos, job list, wrapper scripts and their text results (see its README).
- **`internship-report/`** — LaTeX source of the internship report (local only, not tracked: it names the author).
- **`checkpoints*/`** — result files of the older training/eval runs (`eval_metrics.json`, diff plots, run manifests, guidance diagnostics). The weights next to them are not tracked.
- **`checkpoint/`** — local classifier weights (`config.CLASSIFIER_CKPT`); not tracked.
- **`outputs/`** — diagnostic images written by `check_classifier_localization.py` and `pick_best_counterfactual.py`; not tracked.
- **`logs/`** — run manifests (`*.manifest`); the SLURM stdout/stderr logs themselves are not tracked.
- **Recommended order for a fresh run** — `sbatch run_benchmark.slurm` (size `N_EPISODES` against the 24h cap) → `python test_invariants.py` + `sbatch run_smoke_test.slurm` → `sbatch run_ppo.slurm` → `sbatch run_eval_cf.slurm` then `run_eval_reduce.slurm` → `sbatch run_heldout_validation.slurm`. The control arms run the same eval pair with `EVAL_ARM=none` / `EVAL_ARM=random`.
- **`*.slurm`** — SLURM batch scripts to submit each corresponding Python entry point to the cluster (`run_ppo.slurm`, `run_eval_cf.slurm`, `run_eval_reduce.slurm`, `run_smoke_test.slurm`, `run_smoke_test_eval.slurm`, `run_classifier.slurm`, `run_heldout_validation.slurm`). Evaluation is two jobs: `sbatch run_eval_cf.slurm` (8-task array, generates counterfactual shards) then `sbatch --dependency=afterok:<jobid> run_eval_reduce.slurm` (computes the metrics). `run_eval_ppo.slurm` is the superseded single-job version, kept only for small local smoke tests.
- **`setup_env.sh`** — environment setup script.
- **`requirements.txt` / `requirements.lock.txt`** — Python dependencies (loose vs. pinned).
- **`yolov8n.pt`** — pretrained YOLOv8 nano weights used as an object-detector feature extractor (downloaded locally; not tracked).
- **`venv/`**, **`__pycache__/`**, **`.git/`**, **`.gitignore`**, **`.claude/`** — environment/tooling, not project logic.

## Overall flow

`ingestion.py` → `pipeline.build_common()` (shared by `main.py`/`eval_main.py`) → `src/models.py` loads frozen classifier/segmenter/detector → `src/generator.py` + `factory/finetuning.py` LoRA-tunes the diffusion model → `factory/environment.py` + `factory/ppo.py` train two directional PPO policies → `eval_main.py` reloads checkpoints and runs `src/metrics.py` + `src/diff_viz.py`, with `heldout_validation.py` and `check_classifier_localization.py` as sanity/diagnostic tools on the side.
