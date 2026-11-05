# Jobs needed to close the remaining gaps (nothing has been submitted)

All scripts are in `~/verification/scripts/`. Run `bash 00_make_snapshot.sh` once
on the login node first. It builds code snapshots of commit 49e6050 (the committed
run's code) and HEAD with `git archive`, plus symlinked checkpoint "views". Every job
reads the original weights/shards and writes **only** under `~/verification/out/` and
`~/verification/logs/`; none writes into the repo.

Cost basis (from logs): the committed 8-shard eval took 26-46 min per shard on mixed
GPUs, about 4.7 GPU-h in total. The committed training took 8h40m (n→p, A100) +
11h00m (p→n, H100), of which 5.7 h / 5.1 h was LoRA. The partition is `publicgpu`
(24 h limit, account `insa`), with 7 of 41 nodes idle when I looked. CPU work goes to
`public`.

Sorted by value per GPU-hour. GPU-h are totals across array tasks.

| # | what is missing | paper claims that depend on it | cheapest way | GPU-h | partition / GPU | script |
|---|---|---|---|---|---|---|
| **J1** | per-image LHR for every arm; all paired LHR statistics | §5.4 single-vs-ens d=0.482, p=7.1e-5 (n=76); 3-ens n=12 CI; seed-sweep d=0.221/0.223/0.151 and CIs; §5.5 sparsity +0.275, CI, t=6.229, p, d=0.512; Grad-CAM +1.037, CI, t=14.660, p, d=1.205, 141/148; Table 4 last column; sanity re-run of the random-action test | re-score saved shards (arithmetic on \|x_c − x\| and boxes; no model) | **0** (CPU, < 1 h) | `public`, 4 CPUs | `01_paired_lhr.sbatch` (+ `per_image_lhr.py`) |
| **J2** | transfer ratios + 10k bootstrap CIs as JSON, all arms | §5.3 four CIs and the "do not overlap" sentence; Table 4 transfer for 3-ens (0.350/0.276*), sparsity (0.311/0.212), Grad-CAM (0.161/0.103); "pc 0.462→0.623, non-overlapping"; sparsity "non-overlapping on two of three validators"; Grad-CAM "falls on all three"; R6 interval level | validator inference on saved shards (≤ 800 images × 3 models per arm) | ~0.5 | `publicgpu`, any of H200/H100/A100/L40S | `02_transfer_bootstrap.sbatch` |
| **J3** | Grad-CAM degeneracy and CAM-lesion lift at the default layer and at denseblock3 | §5.5: 66/263 vs 1/263, 25.1%; 2.052 (n=127) vs 1.347 (n=23); remedy 1.196 | classifier forward/backward only | ~0.5-1 | `publicgpu`, any | `03_gradcam_diag.sbatch` |
| **J4** | committed p→n results with the generator the p→n policy was **trained** with (C2) | Table 2 p→n column (CV 0.715, LHR 2.673, conc., FID, IP, CTR), Table 3 row 2 p→n, Table 4 row 2, abstract's 0.186 and 2.67× | inference only: regenerate with `views/pn_lora`, reduce, transfer | ~4.5 | `publicgpu`, any | `04_eval_generate.sbatch` + `05_eval_reduce.sbatch` with `VIEW=pn_lora ARM=trained TAG=J4_pn_lora` |
| **J5** | no-policy (fixed default schedule) arm **under the committed protocol**: 2-ens guidance, classifier groups, committed generator, same 400 images and seeds | §5.2 "no-policy arm already scores 2.046"; Fig. 1 baseline line; whether the policy adds anything over a fixed schedule (contribution 1) | inference only | ~3-4.5 | `publicgpu`, any | `04`+`05` with `VIEW=committed ARM=none TAG=J5_none_ens` |
| **J6** | eval-time per-image, per-step actions of the committed policies (+ reproduction check vs the saved counterfactuals) | intro / Table 1 caption "image- and step-dependent controls"; limitation "Adaptivity is not measured"; C10 | inference only (re-runs the committed generation path with step logging) | ~4.5 | `publicgpu`, any | `06_action_stats.sbatch` (+ `action_stats.py`) |
| **J7** | per-image runtime and peak GPU memory on one fixed GPU | none stated today; needed if you add a cost sentence (the logs only give a range over three GPU types: n→p 4-7 s/img, p→n 29-76 s/img) | inference on shard 0 (50 images), pinned to H100 | ~0.7 | `publicgpu`, `gpuh100` | `07_runtime_memory.sbatch` |
| **J8** | mask dilation/feathering ablation, 50 images/direction, 3 settings (setting 1 reproduces the committed one) | §3.2 lock description (soft lock, band width); reviewer-facing sensitivity of LHR / concentration / transfer to the band | inference only | ~3.5 | `publicgpu`, any | `08_mask_ablation.sbatch` (+ `mask_ablation_eval.py`) |

Suggested submission (after `00_make_snapshot.sh`):

```bash
cd ~/verification/scripts
sbatch 01_paired_lhr.sbatch
sbatch 02_transfer_bootstrap.sbatch
sbatch 03_gradcam_diag.sbatch
J4=$(sbatch --parsable --export=ALL,VIEW=pn_lora,ARM=trained,TAG=J4_pn_lora 04_eval_generate.sbatch)
sbatch --dependency=afterok:$J4 --export=ALL,VIEW=pn_lora,ARM=trained,TAG=J4_pn_lora 05_eval_reduce.sbatch
J5=$(sbatch --parsable --export=ALL,VIEW=committed,ARM=none,TAG=J5_none_ens 04_eval_generate.sbatch)
sbatch --dependency=afterok:$J5 --export=ALL,VIEW=committed,ARM=none,TAG=J5_none_ens 05_eval_reduce.sbatch
sbatch 06_action_stats.sbatch
sbatch 07_runtime_memory.sbatch
sbatch 08_mask_ablation.sbatch
```

Checks to run on the outputs before quoting anything:

- J2: the `ens2_committed` and `single` point estimates must equal
  `logs/arm_transfer_ens_default_vs_old.out` (0.046/0.186, 0.462/0.308, 0.337/0.384 and
  0.080/0.100, 0.414/0.224, 0.197/0.112). If they do not, the HEAD tool has drifted
  from the one that made Table 3, and its CIs must not be used.
- J1: `S1 vs RA` must reproduce `logs/paired_arm_test_trained_random.out` exactly.
- J6/J7: `max_abs_diff_vs_saved_cf` should be ~0. If not, hardware nondeterminism
  means the actions describe a near-replica of the committed outputs; say so.
- J8: setting `d2_f1.0` must reproduce the committed shards on its 50 images; if
  settings 0 or 2 come out identical to it, the override did not take.
- J3: confirm the log reports 263 member-image observations. If the sampling differs,
  the result is a replication, not a recovery of the paper's numbers.

## Retrains (separate: expensive, and queue time dominates)

| # | what | claims | cost | realistic wall time | partition / GPU | script |
|---|---|---|---|---|---|---|
| **J9** | 2 extra seeds of the committed 2-ens config (TRAIN_SEED 1 and 2), each followed by J4-style eval | run-to-run error bars on CV 0.838, LHR 2.673, transfer; whether single-vs-ens and both negative results survive seed variance (every arm is one run today) | per seed: ~20 GPU-h train (LoRA ~5 h + PPO ~4-6 h per direction) + ~4.7 GPU-h eval → **~50 GPU-h for 2 seeds** | 1 day if both pairs start immediately; plan **2-4 days** including queue waits, a possible PAUSED resubmission at the 22 h budget, and the eval | `publicgpu`, **both tasks on `gpuh100`**, so the two directions' LoRAs stay equivalent (see C2) | `09_retrain.sbatch` with `SEED=1 GUIDANCE=ens`, `SEED=2 GUIDANCE=ens`; then `04`/`05` with `VIEW=../out/retrain_ens_seed1` etc. |
| **J10** | hardware-matched single-classifier retrain | §5.4 single-vs-ens. **Low priority**: run 17686489 is already config-matched (C1); this only removes the hardware and generator differences | ~20 GPU-h + 4.7 eval | 1.5-3 days | as J9 | `09_retrain.sbatch` with `SEED=0 GUIDANCE=single` |

After any retrain, paste the per-epoch `avg L_diff` lines of the two array logs side by
side. They must match; if not, evaluate p→n with its own generator (as in J4).

## Not obtainable by any job here

- **Convergence curves** for the reported policies: the code logs only updates 1 and
  80, and `ppo_latest.pt` stores no history. J9 would add two more points per
  direction unless the logging interval in `factory/ppo.py` is changed for that run
  (a code change, so the run would no longer be at commit 49e6050 exactly).
- **"~70% and 24%" overlaps**: these figures are not in the current .tex and could not
  be reproduced (computed overlaps: Xn 82.5%, Xp 39%, annotated 76/183 = 41.5%,
  76/148 = 51.4%). I need the source sentence to check them.
- **Code state of the 3-ens, sparsity and Grad-CAM runs**: all three ran from dirty
  trees on 35f0ea7 with modified `config.py` (and `factory/ppo.py` for sparsity). The
  stash 907735d (2026-08-23) may contain those edits but is not stamped to any run.
  Only the manifests' config hashes (`61b26e0b…`, `cda22868…`, `b257353e…`) and the
  logged flags identify them.
