# Steps 1-2 results (2026-10-28, read-only; nothing submitted, no git write)

## Step 1: code state of the 3-ens, sparsity, Grad-CAM runs

Targets (run manifests, both array tasks agree):
- 3-ens 17700222: config.py sha256 61b26e0bf4f3693f...
- sparsity 17701295: cda22868dc4dc0b9...
- Grad-CAM 17702339 / 17704444: b257353e96d0681e...

Result: **no git object reproduces any of the three.**
- Stash 907735d (2026-08-23 23:49, "WIP on evaluation-integrity: 35f0ea7") is a
  dropped stash (`git stash list` is empty; the object is unreachable). Its
  config.py = ebe6a72b...; its index parent a565c2b = 35f0ea7's config (f687e57a...);
  no untracked parent. Its config.py comments quote the arms' *results*
  ("CV collapsed to 0.017", "d=0.512", "REVERTED (gap-4 retrain complete)") and
  every flag is back off. It is the post-hoc reverted state, written after
  all three runs, not any arm's state.
- Exhaustive: all 696 blobs in the object store (reachable, unreachable,
  dangling; 48 commits, 1 pack + loose) were sha256-hashed. 0 matches.
  15 distinct config.py versions exist across all 48 commits; none matches.
- Minimal reconstruction (35f0ea7 config.py with only the logged flag flipped,
  with/without trailing comment, with GRADCAM_TARGET_LAYER=features.denseblock3,
  with mimic_nb added): 0 matches. The exact edits are not recoverable.
- What the manifests do pin: base 35f0ea7; dirty files (3-ens: config.py,
  domains/rsna_pneumonia.py, arm_transfer.py, check_classifier_localization.py,
  ...; sparsity and Grad-CAM: config.py, factory/ppo.py, ...); and the config
  values listed in the manifest. Across the three manifests the logged config
  values differ only in CKPT_DIR, USE_GRADCAM_MASK and USE_SPARSITY_REWARD
  (3-ens membership is in the shard stamp and log, not in the manifest list).
- Note: 907735d's dirty set has factory/ppo.py (as in sparsity/Grad-CAM) but
  not domains/rsna_pneumonia.py (as in 3-ens).

Proposed git commands: `git_evidence_PROPOSED.sh` (refuses to run without
`--approved`).

## Step 2: generator per direction

sha256 of every unet_finetuned.pt / cond_proj.pt: `generator_sha256.txt`.
Tensor-by-tensor comparison (CPU, mmap): `generator_tensor_diff.json`
(script `scripts/gen_tensor_diff.py`).

| arm | unet bytes np vs pn | differing LoRA tensors (of 256) | max abs diff LoRA | base UNet | cond_proj |
|---|---|---|---|---|---|
| committed `checkpoints/` (A100 / H100) | MISMATCH | 256 | 0.118 | identical | differs (0.075) |
| single `archive_pre_ensemble_default_20260812/` (both H200, same node) | MISMATCH | 225 | 0.0172 | identical | differs (0.0019) |
| 3-ens `checkpoints_ens3/` (both H200) | MISMATCH | 225 | 0.0028 | identical | differs (0.00058) |
| sparsity `checkpoints_sparsity/` (both L40S, LoRA resumed from shared epoch-1 ckpt) | MISMATCH | 225 | 0.00050 | identical | differs (0.00031) |
| Grad-CAM `checkpoints_gradcam_fix/` | **identical** | 0 | 0 | identical | identical |
| (non-arm) cbis_ddsm_breast | MISMATCH | | | | |
| (non-arm) smoke_all, smoke_all_pre_ensemble, smoke_baseline, smoke_mask | identical | | | | |

Corrections to VERIFICATION.md:
- C2 / §1 item 2 says same-GPU-type runs (17686489, ens3) share a generator
  because their final losses agree. **They do not:** the weights differ (smaller
  than in the committed arm, but non-zero). So every arm except Grad-CAM
  evaluates p→n with a generator its p→n policy was not trained against.
- §1 item 2 says the Grad-CAM arm has the same mismatch (H100 vs L40S). **Its two
  files are byte-identical.** Reason: the resumed n→p task 17704444 found the
  shared `checkpoints_gradcam_fix/finetuning_latest.pt` at epoch 29 ("nothing to
  do", log line 47) and loaded it. Since its saved unet equals p→n's, that
  file held p→n's adapter. So the Grad-CAM n→p policy trained updates 1-49
  against its own LoRA (no longer on disk) and updates 50-100 against p→n's.
- JOBS_NEEDED J9 premise ("on identical hardware the final LoRA losses agree",
  so pinning both tasks to one GPU type keeps the LoRAs equivalent) is not
  supported: same node, same GPU type (17686489, hpc-n978 H200) still gave
  different adapters.

## Grad-CAM arm: which adapter the n→p policy was trained with, per phase

Both tasks of 17702339 started at 19:47:35-36 (2026-08-15) with
CKPT_DIR=checkpoints_gradcam_fix and fine-tuned their own LoRA from scratch,
each saving every epoch to the SAME file `checkpoints_gradcam_fix/finetuning_latest.pt`.

| phase | job / log | adapter in the generator | evidence (log line) |
|---|---|---|---|
| LoRA, n→p task | 17702339_0 (hpc-n982, H100) | own adapter, trained in-process | l.45 `no checkpoint found … starting fresh`; l.4187 `epoch 29 done | avg L_diff=0.1229`; l.4188 `checkpoint saved: …finetuning_latest.pt (epoch 29)` |
| LoRA, p→n task | 17702339_1 (hpc-n968, L40S) | own adapter, trained in-process | l.45 `starting fresh`; l.4187 `epoch 29 done | avg L_diff=0.1245`; l.4188 `checkpoint saved: … (epoch 29)` |
| **n→p PPO updates 1-49** | 17702339_0 | **n→p's own adapter** (in memory; no reload between l.4188 and PPO start) | l.4200 `[ppo] no resumable state … starting fresh`; l.4215 `state saved … (update 49, 392 episodes)`; l.4216 `wall-clock budget reached at update 49/100 (21.43h of 22.00h …)`; l.4218 `n_to_p PAUSED`; l.4224 `no ppo_policy.pt yet` |
| p→n PPO updates 1-100 | 17702339_1 | p→n's own adapter | l.4200 starting fresh; l.4229 `update 100`; l.4256 `p_to_n: done → … (ppo_policy.pt, unet_finetuned.pt, cond_proj.pt)`; finished 08-16 15:43:53 |
| **n→p PPO updates 50-100** | 17704444_0 (hpc-n982, H100; started 08-16 18:56:14) | **adapter loaded from the shared `finetuning_latest.pt` = p→n's adapter** | l.45 `found checkpoint at checkpoints_gradcam_fix/finetuning_latest.pt — resuming`; l.46 `resumed from epoch 29`; l.47 `checkpoint already at epoch 30 >= n_epochs=30 — nothing to do`; l.59-60 `resumed from update 49 (392 episodes done)`; l.76 `update 100`; l.126 `n_to_p: done` |

Why the shared file held p→n's adapter when 17704444 read it:
1. Last writer by time: n→p's PPO started after 3.65 h of startup (l.4201 of
   _0: `3.65h already spent on startup`) ≈ 23:26; p→n's after 3.82 h (l.4201 of
   _1) ≈ 23:37. The file's mtime is 2026-08-15 23:35:28, i.e. p→n's epoch-29
   save; nothing wrote it afterwards (mtime < 17704444 start, 08-16 18:56).
2. By content: `checkpoints_gradcam_fix/n_to_p/unet_finetuned.pt` (saved by
   17704444 from the loaded adapter) is tensor-for-tensor identical to
   `p_to_n/unet_finetuned.pt` (saved by 17702339_1 from its in-process adapter).

So the Grad-CAM n→p policy trained updates 1-49 against n→p's own adapter
(which no longer exists on disk: its epoch-29 save was overwritten ~9 min later)
and updates 50-100 against p→n's adapter. Its evaluation (17705866) used
p→n's adapter for both directions. The two adapters differed (final losses
0.1229 vs 0.1245; different GPUs, H100 vs L40S).
