# Sanity checks on job outputs (no number here is cleared for use unless marked)

## J4 (18497956) shard 0 — 2026-10-28 22:13, hpc-n977, NVIDIA H200, 20 min

Adapter lines (`logs/verif-eval-gen-18497956_0.out`):
- l.72  `[adapter] n_to_p <- …/checkpoints/n_to_p/unet_finetuned.pt sha256=658cdd7956a858c1… | cond_proj <- …/checkpoints/n_to_p/cond_proj.pt | generator == file: unet LoRA 256/256, cond_proj 4/4 | expected sha256 MATCH`
- l.101 `[adapter] p_to_n <- …/checkpoints/p_to_n/unet_finetuned.pt sha256=6ae442301a803162… | cond_proj <- …/checkpoints/p_to_n/cond_proj.pt | generator == file: unet LoRA 256/256, cond_proj 4/4 | expected sha256 MATCH`

vs committed `checkpoints/eval_shards_ens_default/cf_shard_000.pt` (17695832_0, H100 NVL):

| check | result |
|---|---|
| adapter per direction as specified | PASS |
| shard/n_eval/base_seed/slices/xn_idx/xp_idx equal; inputs xn, xp byte-identical; config stamp identical | PASS |
| n→p (same adapter, same code) reproduces committed n→p | FAIL (not bitwise): 0/25 images max\|Δ\|<1e-3; mean\|Δ\| 1.46e-3; median per-image max 0.068; max 0.241. Only difference: GPU type (H200 vs H100 NVL) |
| p→n differs from committed (adapter changed) | expected: mean\|Δ\| 1.23e-2; median per-image max 0.298; max 0.819 |

Consequence: J4 is a near-replica on different hardware, not a bitwise re-run;
the J4-vs-committed p→n difference includes a hardware component, of which
the n→p difference is the in-run estimate.

## J2 (18497959) on existing shards — 2026-10-28 22:38, hpc-n967, NVIDIA L40S, 2 min

Check (JOBS_NEEDED): ens2_committed and single point estimates must equal
`logs/arm_transfer_ens_default_vs_old.out` (Table 3 source, 2026-08-14 20:07).
**Result: FAIL (strict), last-printed-digit differences only:**

| | Table 3 log | J2 |
|---|---|---|
| ens2 pc Δn→p | 0.4151 | 0.4152 |
| ens2 mimic_ch Δp→n | 0.2999 | 0.2998 |
| ens2 MEAN ratio p→n | 0.293 | 0.292 |
| single Δ_guidance p→n | 0.8677 | 0.8676 |
| single nih ratio n→p | 0.080 | 0.081 |
| single mimic_ch Δn→p / Δp→n | 0.1801 / 0.0969 | 0.1802 / 0.0968 |
| single MEAN ratio p→n | 0.146 | 0.145 |
All other ens2/single cells (Δ_guidance ens2, nih, pc ratios, n=200, roster) identical.

Cause analysis (not proven):
- `_score` byte-identical in 35f0ea7 and HEAD; `src/xrv_classifier.py` unchanged
  35f0ea7..HEAD; config/domain diffs touch only breast-domain and env plumbing
  (PPO_DOMAIN=rsna_pneumonia set in the job); HEAD's --bootstrap path computes
  the same mean of per-image deltas (float32 summation-order effects are ~1e-7).
- Δ_guidance itself moved (plain classifier inference on the same saved shards),
  so the differences arise in classifier forward passes: most likely GPU
  numerics (TF32 convolutions / cuDNN algorithm choice; L40S here, original GPU
  not recorded).
- Decisive test (needs approval, new job): run 35f0ea7:arm_transfer.py and
  HEAD:arm_transfer.py on the same node; identical outputs => no tool drift.
Status: J2 CIs NOT cleared.

## J5 (18497963) shards 0-2 — L40S, ~3.5 min/shard
Adapter lines per direction as specified (n_to_p 658cdd79… MATCH, p_to_n
6ae44230… MATCH, 256/256, 4/4). Runtime consistent with the original
no-policy eval 17689480 (2 min 46 s per shard). `ARM = none` confirmed.

## J4 (18497956) shards 0-6 logs — adapter lines correct in all (both directions MATCH).

## J1 (18497958) on existing shards — 2026-10-28 23:03, public (CPU), 3 min
Check (JOBS_NEEDED): `S1 vs RA` must reproduce `logs/paired_arm_test_trained_random.out` exactly.
**PASS**: 183 images; 2.2934 (sd 0.7795) vs 2.2039 (sd 0.7717); +0.0895 [+0.0211, +0.1596];
t=2.551 p=0.01157; W=6667.0 p=0.01468; d=+0.189; 105/183 — all identical.
Cross-checks vs paper/VERIFICATION values (recomputation, nothing written to the paper):
- S2 vs S1 (n=76): +0.2851 [+0.1550, +0.4184], t=4.206, p=7.128e-05, d=+0.482 — matches d=0.482, p=7.1e-5, n=76
- S2 vs sparsity (n=148): +0.2750 [+0.1884, +0.3630], t=6.229, p=4.675e-09, d=+0.512 — matches
- S2 vs Grad-CAM (n=148): +1.0370 [+0.9003, +1.1776], t=14.660, p=1.474e-30, d=+1.205, 141/148 — matches
- 3-ens vs S2 (n=12): -0.1008 [-0.4821, +0.2978], t=-0.484, p=0.6376, d=-0.140 (paper value not on file)
- sweep cm1 vs cm0, seeds 0/1/2: d=+0.221/+0.223/+0.151; Δ +0.0694/+0.0693/+0.0575 — match
- arm means equal the eval_metrics JSON lifts (2.6727, 2.2934, 2.3977, 1.6356)
CIs: not checked against the paper (grep of the current .tex found none of them).
Note: `S1 vs NO` and `S2 vs NO` use the rsna-only no-policy shards (single-classifier
groups); they are not the committed-protocol baseline (that is J5).

## J3 (18497962) — 2026-10-28 23:04, L40S, 2 min
Check (JOBS_NEEDED): log reports 263 member-image observations. **PASS** (263).
Default layer: chex 66/263 (25.1%), rsna 1/263 (0.4%); CAM-lesion lift clean 2.052 (n=127),
partially-degenerate 1.347 (n=23). denseblock3: chex 0/263, rsna 1/263 (0.4%); lift 1.196 (n=149).
All match the paper's §5.5 values.

## J4 all 8 shards, J5 all 8 shards, J8 task 1 — 2026-10-28 23:25
- Adapter lines: 16/16 shards show n_to_p <- checkpoints/n_to_p (658cdd79… MATCH) and
  p_to_n <- checkpoints/p_to_n (6ae44230… MATCH), LoRA 256/256, cond_proj 4/4. PASS.
- J4 and J5: slices, xn_idx/xp_idx, base_seed, n_eval and inputs identical to the
  committed shards in all 8; J5 config stamp identical to committed. PASS.
- **Hardware reproducibility: FAIL.** J4 n→p (same adapter/code/images/seeds) vs committed n→p:

| shard | committed GPU | J4 GPU | n→p mean\|Δ\| (same adapter) | p→n mean\|Δ\| (adapter changed) |
|---|---|---|---|---|
| 0 | H100 NVL | H200 | 1.46e-3 | 1.23e-2 |
| 1 | A100-80GB | H200 | 1.88e-3 | 1.43e-2 |
| 2 | A100-80GB | L40S | 1.44e-3 | 1.44e-2 |
| 3 | Quadro RTX 6000 | L40S | 2.29e-2 | 2.28e-2 |
| 4 | Quadro RTX 6000 | L40S | 2.31e-2 | 2.89e-2 |
| 5 | Quadro RTX 6000 | L40S | 2.33e-2 | 2.50e-2 |
| 6 | Quadro RTX 6000 | H200 | 2.27e-2 | 2.23e-2 |
| 7 | Quadro RTX 6000 | L40S | 2.23e-2 | 2.61e-2 |

  On the RTX 6000 shards the GPU change alone moves n→p as much as the adapter
  change moves p→n. J4-vs-committed (and J5/J5b/J8-vs-committed) differences
  therefore contain a hardware component as large as the effect under test.
- J8 d2_f1.0 (50 images = committed shards 0-1, H100/A100; J8 task on H200):
  n→p vs committed mean|Δ| 1.55e-3 (same regime as shards 0-2); p→n vs J4 9.3e-3;
  mask_setting.json = {2, 1.0}; first-50 indices = committed first-50. Override
  check for settings 0 and 2 pending.

## J1 re-run on J4 shards (18497960) — mechanics PASS
S2 = out/J4_perdir/shards; paired counts unchanged (76/148/148/12/200×3/183/183/76);
S1 vs RA identical to J1. **J4-derived numbers NOT cleared** (hardware confound above).

## J4 reduce (18497957), J5 reduce (18497964), J2 on J4 (18497961) — 2026-10-28 23:39, H200
Mechanics (PASS):
- Both reduces: diff plots generated with n_to_p <- checkpoints/n_to_p (658cdd79… MATCH) and
  p_to_n <- checkpoints/p_to_n (6ae44230… MATCH), 256/256, 4/4; eval_metrics.json,
  transfer.json, plots written; no errors.
- LHR p→n n_pairs = 148, n_without_annotation = 52 in committed, J4 and J5 (same images).
- J1-on-J4 per-image lifts: n=148, mean equals J4 eval_metrics LHR lift (|Δ|<1e-9).
- J2-on-J4 (`transfer_J4/ens2_J4_perdir.json`) equals the reduce's `J4_perdir/transfer.json`
  on all 16 ratio/delta fields (max |Δ| = 0.0).
Status: values NOT cleared. J4-vs-committed and J5-vs-committed differences contain the
GPU component documented above; J5-vs-J4 is within new runs but on mixed GPUs
(J4 shards: 3×H200 + 5×L40S; J5: 4×H200 + 4×L40S, different shard assignment).

## J6 (18497965) all 8 shards — 2026-10-29 00:00
- Adapter lines 8/8 shards: both directions MATCH (256/256, 4/4). PASS.
- Indices identical to J4 and committed. PASS.
- JOBS_NEEDED check "max_abs_diff_vs_saved_cf ~0": **FAIL** for n→p (in-job max 0.23-0.77;
  RTX6000 shards largest). p→n in-job diff is vs committed (other adapter): not applicable.
- **Same-GPU-type determinism: FAIL.** J6 vs J4 = identical protocol (per-direction adapters,
  trained policy, same images/seeds/code). Shards 2-4 ran on L40S in BOTH jobs, yet
  0/25 images are bitwise equal: n→p mean|Δ| 1.6-2.5e-3, p→n 0.7-1.0e-2.
  All 8 shards: n→p 1.3-2.5e-3, p→n 0.5-1.3e-2 (J6 vs J4); J6 n→p vs committed on
  RTX6000 shards 2.3e-2. The pipeline is not deterministic even on one GPU type
  (STRICT_DETERMINISM=False at 49e6050). The J4-vs-committed p→n pixel difference on
  shards 0-2 (1.2-1.4e-2) is within this rerun noise.
- Consequence for J6's purpose: the actions describe a near-replica run, not the committed
  outputs (JOBS_NEEDED rule: "say so if you quote them").
- **Metric-level rerun noise floor** (J4 vs J6 through the committed paired_arm_test.py; J6 cfs
  written into J4 shard layout at out/J6_as_shards/): p→n LHR lift difference +0.0156,
  95% CI [-0.0453, +0.0984], t=0.418, p=0.676, d=+0.034, n=148; J4 better on 78/148.
  Per-image sd of the paired difference ≈ 0.46. => differences between arms in LHR lift
  below ~0.1 are within rerun noise of this pipeline.

## J8 (18497969) all 3 tasks — 2026-10-28/29
- Adapter lines MATCH ×4 per task (generate + reduce plots); mask_setting.json per task
  = {0,0.0}, {2,1.0}, {4,2.0}; reduce + transfer written; no errors. PASS.
- d2_f1.0 reproduces committed n→p at near-replica level (mean|Δ| 1.55e-3, = rerun noise);
  p→n vs J4 mean|Δ| 9.3e-3 (= rerun noise). PASS at near-replica level, not bitwise.
- Override took effect: 0/50 images identical between settings; vs d2_f1.0:
  d0_f0.0 n→p 1.32e-2, p→n 2.07e-2; d4_f2.0 n→p 7.5e-3, p→n 1.75e-2.
  n→p differences are 3-8× rerun noise; p→n ~2× rerun noise. Tasks ran on different GPUs
  (L40S / H200 / L40S), which adds a cross-hardware component.
- With 50 images per direction and the noise floor above, J8 metric differences are
  NOT cleared.

## BUG in action_stats.py (affects J6 records, J5b schedule, J7 step lists) — found 2026-10-29 00:45
record_steps() re-wrapped env.step for every image; wrappers nested, so each image's saved
`steps` list = its own steps + all later images' steps in the shard. Verified exactly in all
16 shard-directions (list length = cumulative tail of n_steps; tail of row i == row i+1).
- `n_steps`, timings, peak memory, generated images: unaffected (env stepped once per real step).
- J6 step records: recoverable exactly by truncating to n_steps -> out/actions_repaired/.
  True totals: n→p 201 steps / 200 images; p→n 1093 steps / 200 images.
- **J5b (18497966/18497967): schedule wrong** (step-weighted toward later images, duplicates).
  n→p: max |used − correct| = 0.020 (action sd 0.33-1.19) — negligible.
  p→n: max |used − correct| = 0.193 (dim 7, sd 2.0) — J5b p→n is NOT the specified arm.
  Correct means (repaired):
    n→p [-9.114, -0.671, -2.006, 2.446, 3.041, -6.357, 4.074, 6.55, -1.09]
    p→n [1.705, 1.7, 0.432, 0.486, -1.607, -0.26, -1.142, 2.968, -0.608]
- Fixed: action_stats.py installs the wrapper once (unit-tested: lengths [1,3,2]);
  fixed_schedule_eval.py truncates to n_steps. A J5b rerun needs approval.

## J5b (18497966 gen, 18497967 reduce) — mechanics
8/8 shards: adapters MATCH both directions, `[fixed-schedule]` constant-action line for both
directions, shard written, no errors; reduce: adapters MATCH, plots, transfer written.
GPUs mixed (H100 PCIe, A100×2, L40S×4, H100 NVL). **Values NOT cleared** (wrong p→n schedule,
rerun noise floor).

## J7 (18497968) — H100 NVL (same GPU model as committed shard 0), 21 min
- Check "max_abs_diff_vs_saved_cf ~0": **FAIL** — n→p max 0.34, p→n max 0.79 (J7 uses the
  committed single-adapter protocol, so both directions should have reproduced).
  Same GPU model as the committed shard, still not reproduced: nondeterminism, not only hardware.
- Nested-wrapper overhead negligible: sec/step median first vs second half of shard
  n→p 3.82 vs 3.78, p→n 6.52 vs 6.76.
- Node note: hpc-n982 is tagged gpu40g but reports H100 NVL 95 GB; hpc-n932 reports H100 PCIe.

## J9 training 18497970 — early check (2026-10-29 ~05:51, 21 min in)

- Both tasks started 2026-10-29 05:29:41 on hpc-n934, 4x NVIDIA H100 NVL (sm_90) each; constraint gpuh100&gpu80g honoured.
- Log lines 1-2 (config.py grep, both tasks): `109:TRAIN_SEED = 1`, `167:USE_GUIDANCE_ENSEMBLE  = True`; runtime `[seed] training: seed=1 | cudnn.deterministic=True`; f(.) = ENSEMBLE [rsna, chex]. PASS.
- LoRA: both tasks `no checkpoint found at …/out/retrain_ens_seed1/finetuning_latest.pt — starting fresh`. PASS. Lines 1-47 of both logs differ only in the direction header.
- Epoch 0: both tasks `avg L_diff=0.1331`, pos 0.1367 (n=986), neg 0.1310 (n=1683) — identical to 4 digits; step-level losses already differ in the 4th digit (step 240: 0.1454 vs 0.1453), so the two adapters are not bitwise identical.
- Rate ~11 min/epoch -> LoRA 30 epochs ~5.5 h, then PPO.
- RISK (same mechanism as the Grad-CAM arm): both tasks save every epoch to the SAME `out/retrain_ens_seed1/finetuning_latest.pt`. Harmless if neither task pauses; if one PAUSES and is resumed it would load the other task's adapter. Do not resubmit a paused task without asking.

## J9 training 18497970 — preempted and requeued (checked 2026-10-29 18:56)

sacct -D (publicgpu preemption; Requeue=1, Restarts=4 for task 0):
- task 1 (p_to_n): ran 05:29:41-11:05:26 (5h36), PREEMPTED. Finished LoRA (epoch 29 saved, log l.4185-4186, avg L_diff 0.1216), started PPO from scratch (l.4196), logged update 1, preempted before any PPO save (no p_to_n/ directory).
- task 0 (n_to_p): ran 05:29:41-08:32:44 (3h03), 09:06:34-09:12:48, 10:02:22-11:04:55, 11:07:28-11:07:58, each ended by PREEMPTION. Both tasks are PENDING again (ReqNodeNotAvail: the gpu80g H100 nodes are allocated to other jobs; no reservation).
- On disk: finetuning_latest.pt (epoch 29, mtime 10:44:24; holds lora + cond_proj + optimiser only, no direction label); n_to_p/ppo_latest.pt (update_count 15, 120 episodes, mtime 10:59:40; holds the policy, NOT the LoRA).

Consequences:
1. Logs: the sbatch writes the .out in truncate mode, so every restart wiped the previous run's log. Task 0's logs from runs 1-4 are LOST, apart from the lines quoted in the 05:51 check above. Task 1's first-run log was saved (logs/J9_preserved/*.saved-185619, 4200 lines) before its restart could wipe it. From now on scripts/j9_log_keeper.sh (pid in logs/J9_preserved/keeper.pid) copies each running task's log every 60 s to logs/J9_preserved/<log>.restart<N>.
2. The n_to_p LoRA is a mix: epochs 0 to ~15 trained by task 0; at the 09:06 and 10:02 restarts task 0 resumed from the shared file, which task 1 kept writing, so it continued from task 1's adapter state. Whether task 0 or task 1 wrote the final epoch-29 file (both finished near 10:44) cannot be told without task 0's lost log.
3. n_to_p PPO updates 1-15 trained against task 0's in-memory adapter. On the next restart, BOTH tasks will load the same finetuning_latest.pt (epoch 29 -> "nothing to do"), so n_to_p updates 16-100 and p_to_n updates 1-100 train against the same adapter F, and both final unet_finetuned.pt files will be F (the same situation as the Grad-CAM arm). ppo_latest.pt stores no RNG state, so the resumed n_to_p run does not continue the uninterrupted random stream.
4. Size of the effect: unlike Grad-CAM (H100 vs L40S), both J9 tasks run the same LoRA objective with the same seed on the same node type; their epoch-0 losses agreed to 4 digits. Any difference between the two adapters comes from GPU nondeterminism only. Not measured.
- 2026-10-29: user decision: let 18497970 resume as is (option 1). J9 is recorded as a seed replicate in which both directions share adapter F, with the n_to_p updates 1-15 caveat above.

## J9 (seed 1) — training 18497970, eval 18497971, reduce 18497972 — checked 2026-10-31 ~22:35
Status: mechanics checked; **no J9 number is cleared** (user: J9 not cleared).
The pending job had already finished before decision 5 (2026-10-31) arrived. Nothing was changed on it (no scontrol, no resubmit).

GPU type per task and run (sacct -D; nvidia-smi lines in logs / J9_preserved copies):
| task | run | dates | node | GPU | log |
|---|---|---|---|---|---|
| 0 n_to_p | 1-4 | 10-29 05:29-11:07 (4 runs, all PREEMPTED) | hpc-n934 | H100 NVL (node is gpuh100/gpu80g only) | lost (truncated), except the 05:51 lines above |
| 0 n_to_p | 5 | 10-31 09:03-12:52 COMPLETED | hpc-n934 | H100 NVL x4 visible, sm_90 | verif-retrain-18497970_0.out (= J9_preserved/...restart4 + tail) |
| 1 p_to_n | 1 | 10-29 05:29-11:05 PREEMPTED | hpc-n934 | H100 NVL | J9_preserved/*.saved-185619 |
| 1 p_to_n | 2 | 10-30 13:14-13:28 PREEMPTED | hpc-n934 | H100 NVL | restart1 |
| 1 p_to_n | 3 | 10-31 09:14-13:17 PREEMPTED | hpc-n934 | H100 NVL | restart2 |
| 1 p_to_n | 4 | 10-31 13:22-13:23 PREEMPTED | hpc-n934 | H100 NVL | restart3 (11 lines, died in startup) |
| 1 p_to_n | 5 | 10-31 14:06-16:12 COMPLETED | hpc-n934 | H100 NVL | verif-retrain-18497970_1.out |
| eval gen | shards 0, 7 | 10-31 16:12-16:46 | n934, n982 | H100 NVL | verif-eval-gen-18497971_{0,7}.out |
| eval gen | shards 1-6 | 10-31 16:14-16:36 | n977/n978/n979/n981 | H200 | verif-eval-gen-18497971_{1..6}.out |
| reduce | — | 10-31 16:46-16:51 | hpc-n978 | H200 (node feature; no nvidia-smi line in log) | verif-eval-reduce-18497972.out |
All training ran on one GPU type (H100 NVL). Eval generation was mixed (2 shards H100 NVL, 6 shards H200), as 04_eval_generate allows.

Checks:
1. Config, both tasks, every logged run: `TRAIN_SEED = 1`, `USE_GUIDANCE_ENSEMBLE = True`, `[seed] training: seed=1 | cudnn.deterministic=True`, f(.) = ENSEMBLE [rsna, chex], validators nih/pc/mimic_ch, split 21347/5337 loaded. PASS.
2. LoRA resume: every run after the first logs `found checkpoint at …/retrain_ens_seed1/finetuning_latest.pt — resuming` / `resumed from epoch 29` / `nothing to do`. finetuning_latest.pt mtime 2026-10-29 10:44:24, unchanged since the 10-29 check. PASS (as predicted: adapter F).
3. Adapters: n_to_p/unet_finetuned.pt and p_to_n/unet_finetuned.pt are byte-identical (sha256 59bb77b9…), and so are the two cond_proj.pt (3e038d53…). Both directions share adapter F. Same situation as the Grad-CAM arm; consistent with the 10-29 decision (shared-adapter replicate).
4. PPO resume: n_to_p `resumed from update 15 (120 episodes done)`, then updates 20…100, `n_to_p: done`. p_to_n run 3 `no resumable state … starting fresh`, updates 5…80 saved, preempted; run 5 `resumed from update 80 (640 episodes done)`, updates 85…100, `p_to_n: done`. **New caveat**: p_to_n also has a resume point (update 80), and ppo_latest.pt stores no RNG state, so neither direction is one uninterrupted random stream. Resume logic is file-based (paths and update counters), no node-type field. All runs were on the same node type, so node type does not enter in any case.
5. Policies: J9 ppo_policy.pt n_to_p 475b95be…, p_to_n 335f074f… — distinct from each other and from committed (f35aca5f…, e0d70deb…). Eval loads n_to_p policy from views/J9_seed1_np/n_to_p and p_to_n policy from views/J9_seed1_np/p_to_n → out/retrain_ens_seed1/p_to_n/ppo_policy.pt. PASS.
6. Eval adapter lines (all 8 shards): n_to_p and p_to_n both load sha256 59bb77b9… (F), `generator == file: unet LoRA 256/256, cond_proj 4/4`. No ADAPTER_EXPECT sha was passed, so there is no "expected sha256 MATCH" token; the hash printed equals the file hash. PASS.
7. Images/seeds: all 8 shards vs J4_perdir: shard, num_shards, n_eval, n_slice, p_slice, xn_idx, xp_idx, base_seed equal; xn, xp byte-identical; config stamp identical. Split file byte-identical to the committed split. PASS.
8. Reduce JSON vs reduce log (part1/check_reduce.py): 55 printed values compared, mismatches: none. PASS.
9. L_diff side-by-side (JOBS_NEEDED): not possible — task 0's LoRA epochs are in the lost logs; moot anyway because both final files are F.
No J9 metric was read for this check beyond what check_reduce compares mechanically.
