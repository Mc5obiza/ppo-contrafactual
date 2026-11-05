# Follow-up on your seven decisions — 2026-10-30

Nothing was written to the paper or the repo. New files are under `~/verification/part3/` and
`~/verification/scripts/` (13-19, `j12/`). Jobs submitted: only the three J13 recomputes (item 1).
Nothing else was submitted.

## 1. Check 4 — single-classifier nih n→p transfer cell, recomputed

Same shards (`checkpoints/eval_shards`), same code (`code/HEAD/arm_transfer.py --bootstrap --json`,
the exact J2 invocation), script `scripts/13_nih_cell_repeat.sbatch`, outputs `out/J13_nih_cell/`.

| run | job | GPU | ratio_n_to_p (full precision) | prints as | Δ_nih n→p | Δ_guidance n→p | CI |
|---|---|---|---|---|---|---|---|
| Table 3 source log (2026-08-14) | 17699844 (interactive `bash`) | not recorded; node hpc-n979 is H200-only, but the allocation lists no GPU (AllocTRES `cpu=8,mem=32G`) | — | 0.080 | 0.0737 | 0.9152 | — |
| J2 (2026-10-28) | 18497959 | L40S (hpc-n967) | 0.0805007606 | 0.081 | 0.0736782 | 0.9152480 | [0.048, 0.114] |
| J13 l40s_1 | 18507884 | L40S (hpc-n968) | 0.0805007606 | 0.081 | 0.0736782 | 0.9152480 | [0.0477, 0.1139] |
| J13 l40s_2 | 18507885 | L40S (hpc-n968) | 0.0805007606 | 0.081 | 0.0736782 | 0.9152480 | [0.0477, 0.1139] |
| J13 other type | 18507886 | A100-PCIE-40GB (hpc-n889) | 0.0804751045 | 0.080 | 0.0736536 | 0.9152350 | [0.0476, 0.1138] |
| extra: CPU (not requested; no GPU) | login node | CPU, 24 threads | 0.0804752 | 0.080 | 0.0736616 | 0.9153 (printed) | [0.048, 0.114] |

(The third job was submitted on gpuh100. Its start estimate was 2026-11-07, so I moved it to
`gpuh200|gpua100` with scontrol and it ran on an A100. Logged in JOBS_SUBMITTED.txt.)

**Spread across the three requested runs: 2.6×10⁻⁵** (max 0.0805008 − min 0.0804751).
- The two L40S runs are bitwise identical to each other and to J2, including the bootstrap CI endpoints.
  The tool is deterministic on a fixed GPU type.
- A100 differs from L40S by 2.6×10⁻⁵. CPU agrees with A100 to 1×10⁻⁷ on this cell.
- At 3 decimals the CI is [0.048, 0.114] on L40S, A100 and CPU. The point estimate prints 0.081 on
  L40S and 0.080 on A100/CPU.
- Over all 12 single-arm cells (4 per validator × 3), CPU vs L40S |diff| ≤ 4.4×10⁻⁵
  (`part3/j13_cpu/single_cpu.json`, `single_cpu.log`). No platform reproduces every printed digit of
  the Table 3 log. CPU flips nih ratio p→n to 0.101 (log 0.100) and Δ_guidance n→p to 0.9153 (log 0.9152).

Against the rule:
- The cell is **not identical on every run**: it changes with GPU type. So the "unusable" branch does not fire.
- The spread is **not about 0.001** either. It is ~40× smaller.
- Your planned sentence, "point estimates were recomputed and agree with the logged values to within
  0.001", is true: the largest gap from any printed Table 3 value is one unit in the last digit, and the
  actual gaps are ≤ 5×10⁻⁵ past a rounding boundary. It overstates the disagreement, though. More accurate:
  "agree to within 10⁻⁴; differences come from GPU type and only change the last printed digit."
- An H200 run (the Table 3 node's type) was not done.

Context for the spread. J2's value, 0.0805008, sits 8×10⁻⁷ above the rounding boundary 0.0805.
All eight last-digit mismatches J2 showed against the Table 3 log are within 5×10⁻⁵ of a rounding
boundary in J2's full-precision output:
ens2 pc Δn→p 0.4151791, mimic_ch Δp→n 0.2997585, MEAN p→n 0.2924831; single Δ_guidance p→n 0.8676322,
nih ratio n→p 0.0805008, mimic_ch Δn→p 0.1801628 and Δp→n 0.0967958, MEAN p→n 0.1454852.
Separately, the same HEAD transfer code on CPU vs GPU (J8 shards, item 3) differs by 2-5×10⁻⁵ in the mean ratio.

## 2. J6 — the action-logging bug, then the distribution statistics

**What was logged wrongly.** `action_stats.py` recorded steps by wrapping `env.step`. The version that
ran in J6 (18497965) installed a new wrapper for every image, and each wrapper called the
previous one. The real `env.step` still ran once per real step. As the call unwound, though, every
wrapper appended the step to its own image's list. So image *k*'s step was appended to the lists of
images 0..*k* in the same shard (and direction). Image *i*'s saved `steps` list is therefore its own
steps followed by every later image's steps in that shard. `n_steps` was computed as `len(steps)`
when image *i* finished, before any later image ran, so it is correct.

**Verified on all 400 records (`part3/j6_bug_effect.py`).** For every image:
- `len(steps) == Σ_{j≥i} n_steps_j`
- `steps[n_i:] ==` the next image's list
- the repaired record equals `steps[:n_i]`
- `n_steps`, seconds, peak memory, max_abs_diff and indices are unchanged

Logged vs real step records: n→p 2621 vs 201, p→n 14517 vs 1093.

**What it does and does not affect.**
- Unaffected:
  - each image's *first* step (its list always starts with its own steps)
  - episode length taken from `n_steps`
  - runtime and peak memory
  - the generated images (the env stepped once per real step)
  - the action values themselves (every wrapper saw the same action and the same returned info)
- Affected: anything pooled over the logged step lists, i.e. per-dimension means, sds and quantiles,
  "fraction of steps at a bound", the n_passes distribution, and episode length if taken from
  `len(steps)`. Pooled this way, an image at position *p* in a 25-image shard is counted *p*+1 times,
  and later images dominate.
- Size of the distortion (logged minus repaired):

| dim | n→p Δmean | n→p Δsd | n→p \|Δmean\|/sd | p→n Δmean | p→n Δsd | p→n \|Δmean\|/sd |
|---|---|---|---|---|---|---|
| a0 strength | −0.014 | +0.004 | 0.01 | +0.106 | +0.019 | 0.06 |
| a1 cls_scale | +0.006 | +0.007 | 0.02 | +0.059 | −0.023 | 0.07 |
| a2 pres_scale | −0.001 | −0.001 | 0.00 | +0.072 | +0.032 | 0.13 |
| a3 tau | +0.016 | +0.002 | 0.02 | +0.001 | −0.002 | 0.00 |
| a4 extra pass 1 | +0.014 | −0.001 | 0.01 | −0.109 | +0.049 | 0.06 |
| a5 extra pass 2 | −0.012 | +0.010 | 0.01 | +0.042 | +0.010 | 0.06 |
| a6 strength_boost | +0.016 | +0.002 | 0.02 | −0.035 | +0.008 | 0.05 |
| a7 guidance_bias | +0.020 | −0.002 | 0.02 | +0.193 | −0.005 | 0.10 |
| a8 ramp_start | −0.008 | +0.007 | 0.01 | −0.009 | +0.032 | 0.01 |

  Fraction of steps with guidance bias ≥ 1.295: n→p 0.969 logged vs 0.965 real; p→n 0.225 vs 0.212.
  Episode length from `len(steps)`: n→p mean 13.1 vs 1.00; p→n 72.6 vs 5.46.

**Can corrected values be computed without a GPU?** Yes, exactly, by truncating each list to
`n_steps`. That is what `out/actions_repaired/` holds (verified above). No J6 rerun is needed. The fixed
`action_stats.py` installs the wrapper once and has been unit-tested.

**Statistics (waived for distribution-level use only; J6 is a near-replica, check 5).**
Source: repaired records, `part3/j6_distribution_stats.txt`.

| | n→p (200 images, 201 steps) | p→n (200 images, 1093 steps) |
|---|---|---|
| episode length | 1 step for 199 images, 2 for 1; mean 1.00 | median 1, mean 5.46; 129 at 1 step, 31 at the 25-step cap |
| guidance bias (range 0.70-1.30) | mean 1.2985, range 1.2885-1.3000; ≥1.295 on 96.5% of steps | mean 1.230, range 1.002-1.300; ≥1.295 on 21.2% |
| n_passes | 3 on every step | 2: 60%, 3: 35%, 4: 5% |
| strength | 0.2001 (0.2000-0.2030) | 0.578 (0.385-0.700) |
| cls_scale | 0.608, sd 0.063 (0.40-0.77) | 1.037, sd 0.082 (0.73-1.20) |
| pres_scale | 0.464, sd 0.100 (0.33-0.79) | 1.021, sd 0.144 (0.56-1.37) |
| tau | 0.900, sd 0.080 (0.48-0.99) | 0.597, sd 0.165 (0.29-0.94) |
| strength_boost | 1.144, sd 0.005 | 0.930, sd 0.035 |
| ramp_start_frac | 0.211, sd 0.060 (0.12-0.37) | 0.271, sd 0.094 (0.10-0.43) |

Raw-action per-dimension mean / sd over steps / sd of first-step action across images:
- n→p: a0 −9.11/1.18/1.17, a1 −0.67/0.33/0.33, a2 −2.01/0.68/0.68, a3 2.45/0.80/0.80, a4 3.04/1.07/1.08,
  a5 −6.36/0.94/0.94, a6 4.07/0.69/0.69, a7 6.55/1.19/1.19, a8 −1.09/0.79/0.79.
- p→n: a0 1.71/1.69/1.25, a1 1.70/0.86/1.18, a2 0.43/0.54/0.66, a3 0.49/0.83/0.47, a4 −1.61/1.68/1.80,
  a5 −0.26/0.72/1.23, a6 −1.14/0.77/0.81, a7 2.97/2.00/1.37, a8 −0.61/1.57/1.19.

**Does any dimension vary with the image?**
- n→p: the raw actions vary across images (first-step sd 0.33-1.19), but four controls are saturated
  by their sigmoid maps and are effectively constant: strength (sd 0.0004), strength_boost (0.005),
  guidance bias (0.0016), n_passes (0). cls_scale, pres_scale, tau and ramp_start do vary with the image
  (first-step sd 0.06-0.10). Within-image variation cannot be estimated, because only one image has a
  second step. (My earlier not-cleared file printed a "between-image share" for n→p from that single
  image. That value was meaningless; it is corrected here.)
- p→n: the 71 multi-step images give a between-image share of variance of 0.67-0.78 per raw
  dimension. Both image and step matter, and image matters more.
- Paired with J5b (n→p validity identical on all 200 images under the constant action), image-dependent n→p controls
  exist but make no measurable difference to n→p validity.

## 3. J8 mask ablation (cross-setting comparison only) with rerun noise as reference

50 images per direction (committed shards 0-1). All five rows were computed with one CPU script,
`part3/j8_noise/metrics50.py`. For the J8 rows it reproduces J8's GPU eval_metrics exactly on CV, LHR
and concentration, and within ≤5×10⁻⁵ on transfer. Reference rows were built from the same 50 images
(`part3/j8_noise/make_first50.py`: inputs, indices and seeds asserted identical).

Noise = |d2_f1.0 − committed| on the same 50 images, as you specified. Caveat: for **p→n**, committed
used the other adapter, so that column is adapter + rerun. I therefore also show |d2_f1.0 − J4|
(same adapter and protocol, the pure rerun). "within noise" means |Δ| ≤ the committed-based noise. No
cell changes its flag between the two noise columns.

| metric | d2_f1.0 | noise vs committed | noise vs J4 | d0_f0.0 (Δ) | flag | d4_f2.0 (Δ) | flag |
|---|---|---|---|---|---|---|---|
| CV n→p | 0.960 | 0.000 | 0.000 | 0.940 (−0.020) | beyond noise (1 image of 50) | 0.960 (0.000) | within noise |
| CV p→n | 0.740 | 0.060 | 0.040 | 0.600 (−0.140) | beyond noise | 0.780 (+0.040) | within noise |
| LHR lift (n=47) | 2.489 | 0.034 | 0.051 | 2.502 (+0.013) | within noise | 2.359 (−0.130) | beyond noise |
| conc. n→p | 0.248 | 0.0001 | 0.0006 | 0.277 (+0.028) | beyond noise | 0.242 (−0.007) | beyond noise |
| conc. p→n | 0.317 | 0.005 | 0.001 | 0.352 (+0.035) | beyond noise | 0.300 (−0.018) | beyond noise |
| frac. changed n→p | 0.664 | 0.0005 | 0.0008 | 0.603 (−0.060) | beyond noise | 0.681 (+0.018) | beyond noise |
| frac. changed p→n | 0.629 | 0.001 | 0.001 | 0.564 (−0.065) | beyond noise | 0.677 (+0.048) | beyond noise |
| transfer mean n→p | 0.262 | 0.005 | 0.004 | 0.247 (−0.016) | beyond noise | 0.266 (+0.004) | within noise |
| transfer mean p→n | 0.251 | 0.038 | 0.023 | 0.226 (−0.025) | within noise | 0.296 (+0.046) | beyond noise |
| nih p→n | 0.124 | 0.060 | 0.052 | 0.087 (−0.038) | within noise | 0.197 (+0.073) | beyond noise |
| pc p→n | 0.320 | 0.004 | 0.005 | 0.292 (−0.029) | beyond noise | 0.354 (+0.033) | beyond noise |
| mimic_ch p→n | 0.307 | 0.049 | 0.012 | 0.298 (−0.009) | within noise | 0.338 (+0.031) | within noise |

Reading:
- Without dilation or feathering (d0), edits are more concentrated and touch fewer pixels, but p→n validity drops 0.14.
- With a wider band (d4), edits are more diffuse, LHR lift drops 0.13, and p→n transfer rises.
- Limits: the noise reference is a single rerun pair, n=50 (47 annotated), and the three settings ran on
  different GPUs (L40S/H200/L40S). Treat "beyond noise" as "larger than one observed rerun difference",
  not as a significance test.

## 4. J5b — correct route, n→p check, cost (NOT submitted)

- **Cheapest correct route:** existing J6 outputs. The exact truncation above gives the correct per-direction mean,
  so no corrected J6 pass is needed. The fixed `fixed_schedule_eval.py` already truncates. A CPU dry run of
  its `mean_actions` (`part3/J5c_schedule_dryrun.json`) gives n→p over 201 steps and p→n over 1093 steps:
  - n→p `[-9.1141, -0.6706, -2.0065, 2.4456, 3.0410, -6.3569, 4.0737, 6.5497, -1.0896]`
  - p→n `[1.7052, 1.7000, 0.4318, 0.4857, -1.6072, -0.2604, -1.1416, 2.9683, -0.6078]`
- **n→p affected too, negligibly.** Raw constant off by ≤0.020. Mapped controls differ by ≤1.3×10⁻³
  (cls_scale 0.6058→0.6045, tau 0.9214→0.9202, ramp 0.2001→0.2007; the rest <2×10⁻⁴). The pass count is unchanged (3).
  p→n mapped-control differences: pres_scale 1.048→1.028, strength 0.630→0.623, cls_scale 1.068→1.061,
  guidance bias 1.276→1.271. I recommend rerunning both directions: the generate stage runs both
  anyway, and n→p costs about 15 GPU-min.
- **Scripts:** `scripts/14_J5c_fixed_schedule_generate.sbatch`, `15_J5c_fixed_schedule_reduce.sbatch`
  (tag `J5c_fixed_schedule_corrected`, a new directory: the stamp guard would refuse the old J5b
  directory). Same images, seeds, adapters and code as J5b.
- **Cost:** J5b took 3.29 GPU-h for generation (8 shards, 15-37 min each, mixed GPUs) plus 0.07 for the reduce.
  Estimate **~3.5 GPU-h** (range 3-4.5; p→n episode lengths change slightly with the constant).
- To submit:
  `J=$(sbatch --parsable 14_J5c_fixed_schedule_generate.sbatch); sbatch --dependency=afterok:$J 15_J5c_fixed_schedule_reduce.sbatch`
- **May I submit J5c?**

## 5. J12 — grid-tuned fixed schedule (scripts ready, NOT submitted)

Design (`scripts/j12/j12_grid_select.py`, `j12_select.py`, sbatch 16-19):
- **Same machinery as J5b/J5c.** The env, generator path, per-direction adapters and constant-policy
  injection are reused; only the constant changes.
- **Grid in control space**, mapped to raw actions by inverting the env's sigmoids (round-trip tested,
  0 mismatches):
  - strength {0.25, 0.45, 0.65}
  - cls_scale {0.50, 0.85, 1.15}
  - guidance bias {0.85, 1.05, 1.27}
  - passes {2, 3}
  - That is 54 configs per direction. All other controls sit at the midpoint of their range; **nothing
    comes from the trained policy.**
  - The policy-mean constant (J5c) is scored as a reference row and is never eligible for selection.
- **Selection images: training fold only.** `scan_groups(..., allowed_indices=ctx.train_indices())`, 2-ens
  guidance, classifier groups. The script asserts no overlap with the test fold. Seeds 2,000,000+ are
  disjoint from every eval seed.
- **Criterion:** mean discounted episode return of the env's own reward (γ=0.99), i.e. what PPO maximised.
  Validity, steps and time are logged as well. *Decision for you:* reward is the fair criterion. Selecting on
  training-fold LHR would give the baseline ground-truth boxes the policy never saw.
- **Successive halving:**
  - stage 1: 54 configs + the reference, on 12 images per direction (12 array tasks)
  - stage 2: top 6 per direction, on 36 further images
  - `j12_select.py` (CPU) picks the best of these on all 48 images
  - final: the chosen constant on the same 400 test images and seeds (sbatch 18/19, through the patched
    `fixed_schedule_eval.py`, which now also accepts a schedule JSON; J5c behaviour is unchanged)
- **All selection runs on one GPU type (L40S)**, so the ranking has no cross-hardware component.

GPU-hour estimate (per-step cost from J7: n→p 3.8 s, p→n 7.0 s; episode length is the main unknown):

| stage | optimistic (n→p 3 / p→n 3 steps) | central (8 / 12 in stage 1; 3 / 8 in stage 2) | worst (all 25 steps) |
|---|---|---|---|
| stage 1 (55 × 12 × 2) | 6 h | 21 h | 49 h |
| stage 2 (6 × 36 × 2) | 2 h | 4 h | 16 h |
| model load + group scan (18 tasks) | 1.5 h | 1.5 h | 1.5 h |
| final test eval + reduce | 3 h | 3.5 h | 8 h |
| **total** | **~13 GPU-h** | **~30 GPU-h** | **~75 GPU-h** |

I suggest a pilot first: `sbatch --array=6 16_J12_select_stage1.sbatch` (one p→n chunk, 10 configs,
~1.5-6 GPU-h). Its outputs are reused, since the script skips configs already done. **May I submit
the pilot, and do you accept the reward criterion?**

## 6. J9

- **State:** still PENDING (ReqNodeNotAvail; all four H100 nodes allocated, no reservations). Task 0's
  current start estimate is **2026-11-07 10:29**. Task 1 has no estimate. The eval jobs wait on them.
- **Estimates disagree.** A `sbatch --test-only` probe with J9's exact resources gives **2026-11-01 13:22** for
  `gpuh100&gpu80g` (hpc-n934) and **2026-11-01 12:27** for plain `gpuh100` (hpc-n932). Plain gpuh100 is
  about an hour sooner by that probe, not much sooner. These estimates move with the queue.
- **Memory on the "gpu40g" H100 nodes.** The two nodes tagged gpu40g are hpc-n932 and hpc-n982. nvidia-smi
  in our own logs reports **H100 PCIe 81,559 MiB** (n932, J5b shard 0) and **H100 NVL 95,830 MiB** (n982,
  J7). So the tag does not match the hardware, and on those GPUs memory is not a risk. Peak *training*
  memory has never been logged, but the sparsity arm trained both directions completely on L40S (46 GB),
  and no log has an out-of-memory error. The real risk of plain gpuh100 is hardware, not memory: the
  remaining PPO updates would run on H100 PCIe vs NVL across (or within) the two directions. That
  reintroduces the cross-SKU difference the gpu80g pin was meant to avoid. The LoRA is already the shared
  adapter F, so the difference would now be in PPO numerics only.
- I'd keep the pin. I have not changed J9.

## 7. Adapter weight differences per arm, and the 0.016 source

n_to_p vs p_to_n generator files of each arm (`part3/adapter_diff_table.py`, CPU, full tensors; base
UNet weights identical in every arm):

| arm | dir | training GPUs (n→p / p→n) | LoRA tensors differing | max \|Δ\| LoRA | rel. ‖Δ‖/‖W‖ LoRA | cond_proj max \|Δ\| | cond_proj rel. ‖Δ‖ | sha256 n→p / p→n |
|---|---|---|---|---|---|---|---|---|
| 2-ens committed (default) | `checkpoints` | A100 / H100 | 256/256 | 1.18e-1 | **3.85e-1** | 7.46e-2 | 2.51e-1 | 658cdd79 / 6ae44230 |
| single-classifier | `checkpoints/archive_pre_ensemble_default_20260812` | H200 / H200 | 225/256 | 1.72e-2 | 1.18e-2 | 1.93e-3 | 6.82e-3 | db36b707 / 01d6c7e3 |
| 3-model ensemble | `checkpoints_ens3` | H200 / H200 | 225/256 | 2.76e-3 | 3.94e-3 | 5.77e-4 | 2.02e-3 | ef0703e4 / 6fea64fb |
| sparsity reward | `checkpoints_sparsity` | L40S / L40S | 225/256 | 5.05e-4 | 1.49e-3 | 3.14e-4 | 1.02e-3 | 83cacddd / a2188f63 |
| Grad-CAM constraint | `checkpoints_gradcam_fix` | H100 / L40S | 0/256 | 0 | 0 | 0 | 0 | dc015f6a / dc015f6a (identical; n→p used p→n's adapter from update 50) |
| J9 seed 1 (in progress) | `out/retrain_ens_seed1` | H100 NVL / H100 NVL | — | — | — | — | — | one shared `finetuning_latest.pt` (adapter F); no per-direction files yet |

Correction to my last memo: I said the other arms' mismatch is "7× to 240× smaller" from max |Δ|. By
relative norm the committed arm's mismatch (38.5% of the adapter's norm) is **33× (single) to 260×
(sparsity)** larger. The conclusion is unchanged.

**Source of "0.016":** J4 (18497956) vs J6 (18497965), p→n LHR lift, paired on the 148 annotated
images with the committed `paired_arm_test.py`. Output: `out/J6_as_shards/paired_J4_vs_J6.txt`
(+0.0156, 95% CI [−0.0453, +0.0984], t=0.418, p=0.676, d=0.034, J4 better on 78/148).
`out/J6_as_shards/` = J6's own counterfactuals in J4's shard layout (re-verified today: 8/8 shards, J6
outputs byte-equal, inputs and indices equal to J4).

What "identical protocol" does and does not cover:
- Same: code 49e6050, per-direction adapters, trained policies, images, seeds, and the same
  `generate_counterfactual_with_policy(..., deterministic=True)` call after `torch.manual_seed`.
- Different:
  - The entry point: J4 = `eval_main --stage generate`; J6 = `action_stats.py`, which adds the step
    recorder and per-image CUDA syncs.
  - **GPUs per shard**: only shards 2-4 ran on the same type (L40S) in both jobs. Shards 0, 1, 6:
    H200 (J4) vs L40S (J6). Shard 5: L40S vs H200. Shard 7: L40S vs H100 NVL.

So 0.016 is one observed rerun difference that **includes a cross-hardware component on 5 of 8 shards**. It
is not a same-hardware figure, and a single pair is not a noise distribution. A better summary of rerun
noise for a paired LHR difference here is the CI half-width (±0.07).
I wrote "same-protocol" in the last memo; please read it with this caveat.

---
Open questions for you: (a) submit J5c (~3.5 GPU-h)? (b) submit the J12 pilot (~1.5-6 GPU-h), and accept
the reward criterion? (c) item 1's waiver decision, once you have the numbers above.
