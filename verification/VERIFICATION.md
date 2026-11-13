# PPO-CF paper — evidence verification (2026-10-28)

Scope: every TODO / unverified item in `PPO-CF_CVPR_Paper/` checked against primary
evidence on the cluster (run logs, run manifests, shard stamps, metrics JSONs,
checkpoint key/shape listings, code at the commit each run was stamped with).
Nothing was submitted, trained, generated or evaluated. Checkpoints were opened on
CPU only to list keys/shapes. Paths below are relative to the repo
`<repo>/` unless absolute.

**Status of Step 3 (paper edits): HELD, no .tex/.bib file was changed.** You asked
to see the committed-run identification (Step 1) and the Section D values before
they are used as a source. Both are below. The edits that are ready to apply once
you confirm are listed in §6. Everything that contradicts the paper is in §2 and is
not to be edited by me.

---

## 0. Summary count

Counted over the 79 rows of the item tables in §4 A-G (one row per checked claim or
setting; F counted as 2 rows, G as 5):

| status | count | meaning |
|---|---|---|
| **Verified** | 56 | primary evidence found; matches the paper, or fills a TODO with a value the paper does not contradict |
| **Contradicted** | 13 | primary evidence disagrees with what the paper (or the appendix/other docs) states |
| **Unverifiable now** | 10 | no saved artifact. 8 are recoverable cheaply from saved shards (J1-J3), 1 (convergence curves) only by a new training run with denser logging, and 1 (the "~70%/24%" overlaps) needs you to say where the figure comes from |

Section §2 groups the 13 contradicted rows, plus the supplement build error and the
stale `config.py` comment, into 12 conflicts (C1-C12).

Integrity: all 18 metrics JSONs on disk match the last reduce log that wrote them
(§4.G). No hand-edited value is live on disk.

## 1. The ten most serious findings

1. **The single-classifier arm does not predate the split and seeding controls, and its
   config differs from the committed run only in the guidance flag.** Its policies
   (`checkpoints/archive_pre_ensemble_default_20260812/`, training job 17686489,
   2026-08-10 23:20 → 08-11 03:30) log `[split] … 21347 train / 5337 test (… seed=0)
   [loaded]`, `[seed] training: seed=0`, LoRA on `21347 images (… held-out fold
   excluded)`, far pool `[training fold only]` with 24 probe images held out
   (`logs/ppo-counterfactual-17686489_0.out` lines 18, 27, 51, 4197-4198). Its
   config.py sha256 `ff168a23…` equals the committed config of 5b37471…34ac79c, and
   the only config difference to the committed run (`f687e57a…`, commit 49e6050) is
   `USE_GUIDANCE_ENSEMBLE False→True`. No training-path code changed between those
   commits. The paper says the opposite in the abstract, intro, §4.4, §5.3, §5.4,
   limitations and conclusion. The paper's description ("predates (i) and (ii), was
   re-scored with corrected code", CV 0.975) fits an older `archive_v1_rank8` that does
   not exist on disk, but the paper's numbers (CV 0.940/0.980, LHR 2.293, n=183,
   Table 3 row 1) come from 17686489. **Conflict, see C1.**
2. **The two directions of the committed run do not share a generator, and the p→n
   results were produced with the n→p generator.** Each array task fine-tuned its own
   LoRA from scratch (`[finetuning] no checkpoint found … starting fresh` in both
   17694482 logs), task 0 on an A100 and task 1 on an H100; per-epoch losses diverge
   from epoch 1 (0.1305 vs 0.1322 … 0.1220 vs 0.1229). Evaluation loads
   `checkpoints/n_to_p/unet_finetuned.pt` for both directions (`logs/ppo-eval-cf-17695832_*.out`,
   `logs/ppo-eval-reduce-17695840.out` line 45). So the committed p→n policy was
   trained against one LoRA and evaluated with another (same data, seed and recipe,
   but different weights). The code comment (`pipeline.py`, 49e6050) asserts
   "byte-identical copies (one shared LoRA generator)", which only held for runs
   whose two tasks used the same GPU type (17686489 and ens3: both H200; final loss
   identical). The Grad-CAM arm has the same mismatch (H100 vs L40S). Affects every
   p→n number of the committed arm (Table 2, Table 3 row 2, Table 4 row 2, Fig. 3).
   Fix: J4 (inference only). **Conflict C2.**
3. **The generator was fine-tuned on the training split only, and no evaluation image
   is in it.** Every reported arm's LoRA log says `21347 images (of 26684; held-out
   fold excluded)`; all 400 evaluated indices of every arm (232 for 3-ens) are in the
   persisted test fold (`checkpoints/split_rsna_pneumonia.json`, 5337 test indices).
   LoRA-train ∩ eval = 0. The §4.4 caveat ("its fine-tuning set may overlap the
   evaluation images in every arm"), the limitation bullet and the R4 "Training
   images" cell are contradicted. Good news for the paper, but it has to be rewritten.
   **Conflict C3.**
4. **In the sparsity arm, n→p never had the sparsity reward.** The safeguard latched
   at update 20, before the ramp moved: `⚠ RAMP FROZEN at update 20: CV 0.3750 is
   0.1250 below the pre-ramp baseline 0.5000 … Holding lambda_sparsity=0.0000`
   (`logs/ppo-counterfactual-17701295_0.out` line 3934; update 80 shows `lam_sparse=0.0000
   [FROZEN]`, line 3947). Only p→n ramped (0 → 0.45 → 0.95 → 1.45 → 1.95 → 2.0;
   `…17701295_1.out` lines 3934-3947). §5.5's "enabled and isolated … λ configured to
   ramp to its target … the n→p fraction of pixels changed went from 0.650 to 0.661"
   describes an n→p policy trained with λ=0, i.e. a second run of the default
   config. The p→n LHR test (d=0.512) and p→n transfer are valid sparsity
   comparisons. **Conflict C4.**
5. **The committed policies are barely adaptive and episodes are about one step long.**
   Training-time diagnostics on 20 sampled episodes (exploration on):
   n→p guidance bias 1.2980 ± 0.0019 (ceiling 1.30), passes 3.00 every step;
   p→n 1.2104 ± 0.0930, 2.33 passes (`checkpoints/guidance_diagnostics_*.json`, match
   log lines 4235-4236). `avg_ep_len` is 1.00 → 1.25 (n→p) and 2.00 → 1.88 (p→n)
   at updates 1 and 80, so the T=25 budget is almost never used. `std_mean` stays at its
   initial value (0.6067 → 0.6119; init e^-0.5 = 0.6065). The limitations bullet
   attributes saturation only to "an earlier version"; the committed n→p policy is
   saturated too. This weakens "per-step scheduling" and "image- and step-dependent
   controls" (intro, Table 1 caption). Eval-time action statistics: J6.
6. **The "no-policy" baseline (2.046) is not under the committed guidance.**
   `checkpoints/eval_none/eval_metrics.json` (LHR 2.046, n=183) was generated
   2026-08-11 with rsna-only guidance on the single-classifier's image groups
   (its Xp = the single-classifier Xp, 200/200; only 78/200 shared with the committed
   Xp). §5.2 says "under the same classifier-threshold grouping"; the protocol is the
   same but the grouping classifier is not. The policy's contribution under the
   committed configuration is unmeasured. J5 (inference only) measures it on the same
   400 images and seeds.
7. **About 25 reported statistics have no saved artifact**, but all are recomputable
   from saved shards: the four §5.3 bootstrap CIs; paired tests single-vs-ens
   (d=0.482, p=7.1e-5), sparsity (+0.275, CI, t=6.229, p, d=0.512), Grad-CAM (+1.037,
   CI, t=14.660, p, d=1.205, 141/148), 3-ens (n=12 CI); the seed-sweep d's and CIs;
   Table 4 transfer of 3-ens, sparsity and Grad-CAM, plus their per-validator claims; the
   Grad-CAM default-layer diagnostics (66/263, 2.052/n=127, 1.347/n=23, 1.196). J1
   (CPU), J2 and J3 (≤1 GPU-h each).
8. **The 3-model arm's p→n group has 32 images, not 200.** `[eval] Xn=200 Xp=32`
   (`logs/ppo-eval-cf-17700783_0.out` line 66), so LHR n=27 and p→n CV 0.5625 = 18/32.
   Table 4's caption ("n=200 per group" via §5.3) and the 3-ens CV statement do not
   say this. Membership is stamped: rsna, chex, **mimic_nb**.
9. **Several "not preserved" statements are false.** Seeding code ran (LoRA seed 0,
   PPO seeds 1/2, split seed 0, eval base_seed 0). The split code and file exist. The
   LHR implementation is committed (5b37471). The shard stamp records
   `USE_GUIDANCE_ENSEMBLE` and the member list for every arm except the
   single-classifier shards (whose training log confirms single guidance). The
   seed-sweep raw shards are preserved (`checkpoints/eval_shards_none_cm{0,1}{,s1,s2}`).
   Run manifests exist for every run. The paper and appendix (R3, R5, R6, §4.3,
   §4.4, §5.4 "raw outputs were not preserved", X_suppl) say these are missing.
10. **Build and hygiene.** `supplementary.tex` line 23 `\input{sec/suppl_details}`
    refers to a missing file, so the supplement does not compile. 80 `file.py:line`
    references sit in .tex comments (against your rule; pre-existing). `sec/X_suppl.tex`
    (not included anywhere) is a stale variant that contradicts the appendix (e.g. "a1
    … differs"). No de-anonymising strings were found in .tex/.bib (only the CVPR
    template URL in `main.tex` line 1).

## 2. Conflicts (not edited; both sources given; your call)

| # | claim in paper (location) | evidence | source |
|---|---|---|---|
| C1 | Single-classifier arm "predates our split and seeding controls", "re-scored with corrected evaluation code", "probably differs in reward weights, a1 range and anchoring" (`sec/0_abstract.tex` 2-3; `1_intro.tex` 15; `4_evaluation.tex` 49-51; `5_experiments.tex` 40, 97-100; `6_limitations.tex` 9; `7_conclusion.tex` 4) | Run 17686489 used split seed 0, train seed 0, LoRA on the train fold, far pool on the train fold, 24-image probe holdout, T=25, MLP, rank 8, EDIT_BUDGET inf, FIXED_EPISODE_LENGTH False; config sha = the committed config except `USE_GUIDANCE_ENSEMBLE`; a1 narrowing (2026-08-07) and anchoring already in cf5204f (the base of the dirty tree) | `logs/ppo-counterfactual-17686489_{0,1}.out` 18/27/51/4197-4198; `logs/ppo-counterfactual-17686489_0.manifest`; `git diff 5b37471 49e6050 -- config.py` |
| C1b | (other documents) `config.py` comment at 49e6050: the single-model policies are `checkpoints/archive_v1_rank8/` "and produced the CV=0.975 numbers in checkpoints/eval_metrics.json" | `archive_v1_rank8` does not exist. `checkpoints/eval_metrics.json` now holds CV 0.940 (job 17689479). CV 0.975 survives only as `checkpoints/eval_metrics.restored-fid.json` (no producing log on disk) | `git show 49e6050:config.py`; `checkpoints/eval_metrics*.json` |
| C2 | "one policy … per direction, sharing the generator" (`3_method.tex` 75) | Two independently fine-tuned LoRAs (A100 vs H100, diverging losses); eval uses the n→p LoRA for both directions | §1 item 2 |
| C3 | "no record that the shared generator was fine-tuned again after the split … may overlap the evaluation images in every arm" (`4_evaluation.tex` 49-50; `6_limitations.tex` 9; appendix R4 "the recorded code fine-tunes on the full dataset"; `3_method.tex` 17 TODO) | LoRA fine-tuned on the 21,347-image train fold in every reported run; eval ⊂ test fold; overlap 0 | committed logs line 48; single logs line 51; `checkpoints/split_rsna_pneumonia.json` vs shard `xn_idx/xp_idx` |
| C4 | Sparsity arm: "Enabled and isolated against the default, with λ configured to ramp to its target … n→p fraction of pixels changed went from 0.650 to 0.661" (`5_experiments.tex` 108) | n→p λ frozen at 0 from update 20; p→n ramped to 2.0 | `logs/ppo-counterfactual-17701295_0.out` 3931-3947; `_1.out` 3931-3947 |
| C5 | "Training random seeds: not found (no seed is set in the training code)" (appendix R3 line 137; X_suppl 91) and "fix code not preserved (seeding)" (`4_evaluation.tex` 47) | `seeding.py` committed (5b37471); `[seed] training: seed=0`, n→p seed=1, p→n seed=2 | committed logs lines 14, 4193 |
| C6 | "the per-shard configuration stamp … TODO not found"; "metrics.current_cfg_stamp records … but NOT USE_GUIDANCE_ENSEMBLE" (appendix R6 250-252; `4_evaluation.tex` 40-41) | At 49e6050 the stamp records `USE_GUIDANCE_ENSEMBLE` and `GUIDANCE_ENSEMBLE_WEIGHTS`; committed shards carry `USE_GUIDANCE_ENSEMBLE: True, [rsna, chex]`; `arm_transfer.py` reads the denominator roster from it. The single-classifier shards predate the key; the tool prints "[warn] … assuming the single guidance model" | `git show 49e6050:src/metrics.py` (`current_cfg_stamp`); `checkpoints/eval_shards_ens_default/cf_shard_000.pt` key `config`; `logs/arm_transfer_ens_default_vs_old.out` |
| C7 | "Evidence on ensembling itself comes only from a policy-free sweep (three seeds; raw outputs were not preserved)" (`5_experiments.tex` 101) | Raw shards of all six sweep arms are on disk | `checkpoints/eval_shards_none_cm0{,s1,s2}`, `…_cm1{,s1,s2}` |
| C8 | "no-policy arm already scores 2.046 under the same classifier-threshold grouping" (`5_experiments.tex` 12; Fig. 1 caption) | 2.046 is rsna-guided, on the single-classifier's groups (shares 78/200 Xp with the committed run) | `checkpoints/eval_none/eval_metrics.json` LHR.p_to_n.lift; shard idx lists |
| C9 | 3-ens Table 4 row under "n=200 per group" | Xp = 32, LHR n = 27 | `logs/ppo-eval-reduce-17700791.out` 15, 72-73 |
| C10 | Limitations: "An earlier version of the policy saturated its guidance-bias control" (implying the reported one does not) | Committed n→p guidance bias 1.2980 ± 0.0019 of max 1.30 | `checkpoints/guidance_diagnostics_n_to_p.json` |
| C11 | Supplement build: `supplementary.tex` 23 inputs `sec/suppl_details` | File absent | `ls PPO-CF_CVPR_Paper/sec/` |
| C12 | Training times "on a single node" (`5_experiments.tex` 9) | The values match, but the two directions ran on two nodes with different GPUs (hpc-n931 A100-80GB; hpc-n934 H100 NVL), and the times include 5.69 h / 5.12 h of LoRA fine-tuning | committed logs lines 2-4, 4199, 4255 |

## 3. Step 1: run identification

### 3.1 Reasoning for "the committed runs"

Candidates for the 2-model ensemble were every training job with
`USE_GUIDANCE_ENSEMBLE = True` and members `[rsna, chex]`: 17694482
(`checkpoints/`), 17701295 (`checkpoints_sparsity/`, sparsity on) and 17702339/17704444
(`checkpoints_gradcam_fix/`, Grad-CAM on). Only 17694482 has the committed
configuration (no Grad-CAM, no sparsity). The tie to the paper:

1. **Training** `ppo-counterfactual-17694482` (array tasks 0 = n→p, 1 = p→n), started
   2026-08-12 22:24:41; n→p finished 08-13 07:04:21, p→n 08-13 09:24:51. It wrote
   `checkpoints/{n_to_p,p_to_n}/` (file mtimes 08-13 07:04 / 09:24) and
   `checkpoints/finetuning_latest.pt` (08-13 04:04, last writer = task 0).
   Manifest: commit **49e6050** (branch evaluation-integrity), tree dirty **only** in
   `checkpoints/*`, `report/*`, `social_media/`. config.py sha256 `f687e57a…` =
   `git show 49e6050:config.py`. So the code that ran = commit 49e6050.
2. **Evaluation** `ppo-eval-cf-17695832` (8 shards, 08-13 13:03-13:49) loaded
   `checkpoints/n_to_p/ppo_policy.pt` and `checkpoints/p_to_n/ppo_policy.pt` (37,885,462
   params each) and wrote `checkpoints/eval_shards_ens_default/` (stamp
   `USE_GUIDANCE_ENSEMBLE: True`, `[rsna, chex]`, no Grad-CAM, no sparsity,
   `EVAL_GROUP_SOURCE: classifier`, `USE_SPLIT: True`). **Reduce**
   `ppo-eval-reduce-17695840` (08-13 13:49-13:54) wrote
   `checkpoints_ens_default/eval_metrics.json`.
3. **Match to the paper**: that JSON gives CV 0.960/0.715/0.8375, LHR 2.673 (t 24.19,
   p 4.16e-53, n 148), concentration 0.253/0.318 and changed 0.650/0.633, all exactly
   as in Table 2. `logs/arm_transfer_ens_default_vs_old.out` on
   `eval_shards_ens_default` gives Table 3 row 2 exactly. Fig. 2/3 are byte-identical
   (md5) to `checkpoints_ens_default/counterfactual_diff_{n_to_p,p_to_n}.png`.
4. Caveat: the reduce's top-level manifest (`checkpoints/run_manifest.txt`) and the
   eval are on 49e6050 too. The export manifest of today points to the same job;
   I used it only as a pointer.

**Conclusion: committed run = training 17694482 → eval 17695832 → reduce 17695840; code 49e6050.**
Please confirm before §6 is applied.

### 3.2 All arms

| arm | checkpoint dir | training job (dates; GPU) | eval-gen / reduce job | SLURM script | shards (stamp) | metrics JSON | code |
|---|---|---|---|---|---|---|---|
| Single-classifier ("archived v1" in paper) | `checkpoints/archive_pre_ensemble_default_20260812/` (moved there 08-12; mtimes 08-11) | 17686489 (08-10 23:20 → 08-11 02:50 / 03:30; both H200, hpc-n978) | 17689478 / 17689479 (08-11 18:37-18:58) | `run_ppo.slurm`, `run_eval_cf.slurm`, `run_eval_reduce.slurm` | `checkpoints/eval_shards/` (pre-dates `USE_GUIDANCE_ENSEMBLE` key) | `checkpoints/eval_metrics.json` | dirty tree on cf5204f; config sha `ff168a23…` = commit 5b37471…34ac79c |
| **2-model ensemble (committed)** | `checkpoints/{n_to_p,p_to_n}/`, `checkpoints/finetuning_latest.pt` | 17694482 (08-12 22:24 → 08-13 07:04 / 09:24; A100 / H100) | 17695832 / 17695840 (08-13 13:03-13:54) | same | `checkpoints/eval_shards_ens_default/` (ens True, rsna+chex) | `checkpoints_ens_default/eval_metrics.json` | **49e6050** (clean for code) |
| 3-model ensemble | `checkpoints_ens3/` | 17700222 (08-14 21:50 → 08-15 04:21 / 02:19; both H200) | 17700783 / 17700791 (08-15 04:22-04:54) | same | `checkpoints_ens3/eval_shards/` (rsna, chex, mimic_nb; **Xp=32**) | `checkpoints_ens3/eval_metrics.json` | dirty on 35f0ea7 (config.py, domains/rsna_pneumonia.py, arm_transfer.py modified; not recorded); config sha `61b26e0b…` |
| Sparsity | `checkpoints_sparsity/` | 17701295 (08-15 11:02 → 18:11 / 17:51; both L40S; LoRA resumed from an earlier epoch-1 checkpoint) | 17702127 / 17702135 (08-15 18:12-18:46) | same | `checkpoints_sparsity/eval_shards/` (sparsity True) | `checkpoints_sparsity/eval_metrics.json` | dirty on 35f0ea7 (config.py, factory/ppo.py modified; not recorded); sha `cda22868…` |
| Grad-CAM (denseblock3) | `checkpoints_gradcam_fix/` | 17702339 (08-15 19:47; n→p **paused at update 49** after 21.4 h; p→n done 08-16 15:43; H100 / L40S) + 17704444 (n→p resumed 08-16 18:56 → 08-17 10:34, H100) | 17705866 / 17705875 (08-17 10:37-13:03) | same | `checkpoints_gradcam_fix/eval_shards/` (Grad-CAM True; **target layer not in stamp**, log says `features.denseblock3`) | `checkpoints_gradcam_fix/eval_metrics.json` | dirty on 35f0ea7; sha `b257353e…` |
| Random-action | uses single-classifier policies | n/a | 17689482 / 17689483 (08-11) | same, `EVAL_ARM=random` | `checkpoints/eval_shards_random/` | `checkpoints/eval_random/eval_metrics.json` | dirty on cf5204f (pre-5b37471) |
| No-policy (classifier groups) | single-classifier generator | n/a | 17689480 / 17689481 (08-11) | `EVAL_ARM=none` | `checkpoints/eval_shards_none/` (rsna-only) | `checkpoints/eval_none/eval_metrics.json` | same |
| No-policy, label protocol (2.057) | single-classifier generator | n/a | 17689780 / 17689781 (cm0) | `EVAL_ARM=none EVAL_TAG=cm0` | `checkpoints/eval_shards_none_cm0/` (label groups, ens False) | `checkpoints/eval_none_cm0/eval_metrics.json` | 5b37471-era |
| Policy-free seed sweep | single-classifier generator | n/a | cm0/cm1: 17689780/82, reduces 17689781/83; s1: 17689879/81 → 17689880/82; s2: 17689883/85(+17689943) → 17689884/17689944 | `EVAL_CM=ensemble`, `EVAL_TAG=cm1s1` etc. | `checkpoints/eval_shards_none_cm{0,1}{,s1,s2}/` (label groups) | `checkpoints/eval_none_cm*/eval_metrics.json` | committed in 927d6b8 |

No arm is UNIDENTIFIED. The paper's "archived v1" label is the only identification
that does not match (C1).

## 4. Step 2: item tables

Columns: item | paper location | evidence (path, line/key, date) | value | matches paper? | action.
"code@49e6050" means `git show 49e6050:<file>`; that is the code of the committed run
(manifest: tree clean for code). Reward/action/PPO/generator code is unchanged
from cf5204f (the appendix's source) to 49e6050 except the train-fold restriction
in `factory/environment.py` and the ensemble flag (`git diff cf5204f 49e6050`).

### A. Settings of the committed run

| item | paper location | evidence | value | matches? | action |
|---|---|---|---|---|---|
| Policy trunk | 3_method 38/63, R3 129, R5 192 | `checkpoints/{n_to_p,p_to_n}/ppo_policy.pt` keys: `shared.0.weight [512,73731]`, `shared.2.weight [256,512]`, heads `[9,256]`, `[1,256]`×4, `log_std [9]`; total 37,885,462; manifest `POLICY_TRUNK = mlp` | MLP 73,731→512→256 | yes | fill R5; drop TODOs (§6) |
| Step budget T | 3_method 38/63/67, R3 132, R5 189 | manifest `BUDGET_T = 25`; shard stamp `BUDGET_T: 25` | 25 | yes | fill |
| Fixed episode length | 3_method 66/69, R3 133, R5 190 | manifest `FIXED_EPISODE_LENGTH = False`; stamp same; `avg_ep_len` 1.00-2.00 | False (episode ends on validity or t=T) | yes | fill |
| ε_stop | 3_method 67, R1 | code@49e6050 config `EPSILON_STOP = 0.08` | 0.08 | yes | none |
| L1 budget B | R1, R5 191 | manifest `EDIT_BUDGET_L1 = inf`; log `edit_L1=… (budget inf)` | ∞ | yes | fill |
| LoRA rank / targets | 3_method 17, R4 157, R5 193 | `checkpoints/finetuning_latest.pt` (epoch 29): 256 LoRA tensors, rank {8}, modules attn1/attn2 `to_q,to_k,to_v,to_out.0` only, 1,594,368 params; log `trainable params: 1,594,368` | 8 / attention projections | yes | fill |
| LoRA epochs | R4 158, R5 194 | log line 48 `epochs 0->29`; ckpt `epoch = 29` | 30 | yes | fill |
| LoRA α / dropout | R4 157 | code@49e6050 `LORA_ALPHA = 16`, `LORA_DROPOUT = 0.05` (not stored in the checkpoint) | 16 / 0.05 | yes | none |
| a1 mapping | Table 1, R2 83, R5 195 | code@49e6050 `factory/environment.py` `cls_scale_t = 0.3 + 0.9 * torch.sigmoid(action[1])` | 0.30+0.90σ(a1) | yes | fill |
| R_fid form, λ_fid | R1 50, R5 196 | code@49e6050 `R_fid = -edit_l1`, `LAM_FID = 2.0` | mean L1, 2.0 | yes | fill |
| λ_pres^R | R1 51, R5 197 | `LAM_PRES_R = 3.0` | 3.0 | yes | fill |
| λ_hit, λ_stop | R1 53-54, R5 198 | `LAM_HIT = 2.0`, `LAM_STOP = 2.0`. Cross-check: log update 1 n→p `hit=+4.500` = 2.25 × 2.0 × 1.0 | 2.0, 2.0 | yes | fill |
| Anchoring | R4 caption, R5 199 | `ANCHOR_TO_EPISODE_ORIGIN = True`; shard stamp same | episode's original image | yes | fill |
| Saturation penalty | R3 126, R5 200 | `LAM_ACT_SAT = 1e-2`; log `|logit|max=… (sat>3.0)` | 0.01 | yes | fill |
| 24-image probe holdout | R3 134, R5 201 | log 4195 `far pool split: 12188 train / 24 held out` (p→n 9111/24) | present | yes | fill |
| DDIM η | R4 159 | code@49e6050 `self.scheduler.step(eps_hat, t, z_t)` (no η); diffusers 0.38.0 pinned in `requirements.lock.txt`@49e6050; its `DDIMScheduler.step(…, eta: float = 0.0)` | 0.0 | fills TODO | fill |
| Observation dim | 3_method 65/67, R3 131 | log 4196 `obs_dim = 73731` | 73,731 | yes | drop TODO |
| VAE posterior sampling | 3_method 64/67, R4 155 | code@49e6050 `src/generator.py` `posterior.latent_dist.sample()`; `get_observation` re-encodes `x_t` each call | sampled, current latent re-drawn every step | yes | drop TODO |
| Shared generator across directions | 3_method 75 | §1 item 2 | **two LoRAs; eval uses n→p's for both** | **no** | C2; J4 |
| Episodes / updates | 5_exp 9, R3 116-117 | log `Update 80 | episodes= 640`; `state saved … (update 100, 800 episodes)` | 800 / 100 | yes | none |
| GPU / node for training | 5_exp 9 | log lines 3-4 | n→p A100-SXM4-80GB (hpc-n931); p→n H100 NVL (hpc-n934) | partly | C12 |

### B. Seeds and splits

| item | paper location | evidence | value | matches? | action |
|---|---|---|---|---|---|
| LoRA seed | R3 137, R5 206 | committed logs line 14 `[seed] training: seed=0 | cudnn.deterministic=True`; `seeding.py`@49e6050 | 0 (STRICT_DETERMINISM False) | **no** (paper: no seed set) | C5 |
| PPO seeds | R3 137 | log 4193 `n_to_p: seed=1 (base 0 + offset 1)`, `p_to_n: seed=2` | 1 / 2 | **no** | C5 |
| Eval seeds | R3 136 | shard key `base_seed: 0`; eval log `global_idx=1000000` for first p→n image | per image `0 + global index` (+10^6 p→n) | yes | none |
| Split procedure | 4_eval 46, R5 206 | `splits.py`@49e6050: patient groups (RSNA: one image per patient), `np.random.default_rng(seed).permutation`, `n_test = round(0.2·N)`, unstratified, persisted JSON with dataset fingerprint | as stated | fills TODO | fill |
| Split ratio / sizes | R5 206 | `checkpoints/split_rsna_pneumonia.json`: `seed 0, test_frac 0.2, n_train 21347, n_test 5337, fingerprint 26684:dfbad01fe5ecb406`; identical md5 in all four arm dirs | 80/20, 21,347/5,337 | fills TODO | fill |
| LoRA training images | 3_method 17, 4_eval 49-50, R4 158 | log line 48 `21347 images (of 26684; held-out fold excluded)` | training fold only | **no** | C3 |
| LoRA ∩ eval overlap | 4_eval 49, limitations | shard `xn_idx/xp_idx` of all arms ⊂ test fold (200/200 and 200/200; 3-ens 200/32) | 0 | **no** | C3 |

### C. Metric code

| metric | file (committed?) | implementation | matches R6 / §4? | action |
|---|---|---|---|---|
| LHR | `src/metrics.py::lesion_hit_rate`, `ingestion.py::lesion_mask/lesion_boxes` (committed 5b37471; at 49e6050) | boxes rescaled from native DICOM size to 256² (`sx = img_size/cols`), floor/ceil to pixels, **union** of all boxes; per image hit = Σ_A δ/Σ δ, null = area share; **lift = mean of per-image ratios** (not pooled); images with null 0 or 1 skipped; p from `scipy.stats.ttest_rel(hits, nulls)` (paired t of hit vs null) | consistent with §4.2; fills every R6 LHR TODO | fill R6 |
| Paired arm tests | `paired_arm_test.py` (fb268c6) | shared annotated images; `ttest_rel`, Wilcoxon, d = mean(diff)/sd(diff, ddof=1), 10k bootstrap of mean diff, seed 0, 2.5/97.5 percentiles | fills "definition of d" | fill R6 |
| Transfer ratio | `arm_transfer.py` (fb268c6, 927d6b8, 35f0ea7) | numerator/denominator = mean signed shift over the same N images (ratio of means); guidance roster from the shard stamp (pre-stamp shards assumed single, with a warning); 3-validator mean = **unweighted mean of the per-validator ratios**; zero guard `if g` (exact 0, not 1e-8) | matches §4.3; R6's 1e-8 guard is from `heldout_validation.py`, which did not produce Tables 3/4 | fill R6 averaging; amend guard text |
| Which version produced Table 3 | `logs/arm_transfer_ens_default_vs_old.out` (2026-08-14 20:07, no manifest) | between 927d6b8 and 35f0ea7 (committed 20:35 the same day); not stamped | n/a | note |
| Bootstrap of ρ | `arm_transfer.py::_bootstrap_ratio_ci`, **HEAD only** (f28d96b, 2026-10-26; also in stash 907735d of 08-23); not in any run-stamped commit | 10,000 resamples of image indices, numerator and denominator jointly, `nanpercentile([2.5, 97.5])`, seed 0 | description matches §5.3; **no output saved** | J2 |
| CV | `src/metrics.py::counterfactual_validity` | `moved = s·Δ ≥ τ` and crossing; non-finite dropped | matches (≥) | none |
| FID | `frechet_inception_distance` with `_XRVFeatureExtractor` default `densenet121-res224-rsna` | unchanged since cf5204f | matches R6 | none |
| IP, edit concentration, CTR, SCP | `src/metrics.py` | function bodies unchanged between cf5204f and 49e6050 (diff touches only scan_groups, stamp, LHR, run_full_evaluation and printing) | R6 descriptions apply | none |
| Shard stamp | `current_cfg_stamp`@49e6050 | records USE_GUIDANCE_ENSEMBLE, members, anchor, Grad-CAM flag, sparsity flag, T, budget, passes, T_diff, fixed-length, lock, α's, η_obj, group source, split. Does **not** record Grad-CAM target layer or mask dilation/feather | contradicts R6/§4.3 TODO | C6 |

### D. Numbers marked unverified (values shown here for you, not in the paper)

Source for all: `checkpoints_ens_default/eval_metrics.json`, written by reduce 17695840
(2026-08-13 13:54), cross-checked against that job's printed values
(`logs/ppo-eval-reduce-17695840.out`, lines given).

| paper value | JSON key | JSON value | reduce log line | matches paper? |
|---|---|---|---|---|
| FID n→p 22.21 | `FID.n_to_p_vs_real_positive` | 22.213562068021044 | 96 `FID(Xp, Xc_n→p) : 22.21` | yes |
| FID p→n 4.95 | `FID.p_to_n_vs_real_negative` | 4.949032776832477 | 97 `4.95` | yes |
| MS-SSIM 0.923 / 0.870 | `IP.n_to_p.ms_ssim`, `IP.p_to_n.ms_ssim` | 0.9234573131799698 / 0.8702382516860961 | 113-114 | yes |
| HF-corr 0.948 / 0.914 | `IP.*.hf_corr` | 0.948360447883606 / 0.9135404899716377 | 113-114 | yes |
| CTR n→p +0.0076, t 2.246, p 0.026 | `CTR.mean_delta_n`, `t_n`, `p_n` | 0.0076215209206759085, 2.246133531954324, 0.02579441156145342 | 121 | yes (Δ is real − counterfactual) |
| CTR p→n +0.0080, t 1.010, p 0.314 | `CTR.mean_delta_p`, `t_p`, `p_p` | 0.008002672834367237, 1.0100538416605422, 0.3138482707475534 | 122 | yes |
| SCP relative spread 0.19% | `SCP.relative_spread` | 0.0019255417989779173 | 125 | yes |

Notes: the eval code itself marks SCP degenerate and withholds its tests. The JSON's
git history has one hand edit, a stray `²` after `p_to_n.frac_changed` in 35f0ea7
(invalid JSON), removed in f28d96b without changing the value. The values match
the log.

### E. Claims with no saved artifact (before this session)

| item | paper location | evidence | value found | matches? | action |
|---|---|---|---|---|---|
| Table 3, all 12 cells | 5_exp Table 3 | `logs/arm_transfer_ens_default_vs_old.out` (08-14 20:07) | single 0.080/0.100, 0.414/0.224, 0.197/0.112; ens 0.046/0.186, 0.462/0.308, 0.337/0.384; means 0.230/0.146, 0.282/0.293 | **yes, all** | drop nothing (no TODO); cite in §6 |
| Threshold counts 1/3/4/5 and 1/1/2/2 | 5_exp 63 | arithmetic on the verified Table 3 | as stated | yes | none |
| Bootstrap CIs §5.3 | 5_exp 60-61 | none | — | **unverifiable** | J2 |
| Grad-CAM: 66/263 chex vs 1/263 rsna (default layer) | 5_exp 110 | none (the surviving `outputs/gradcam_check/degeneracy_log.json` is the denseblock3 run) | — | **unverifiable** | J3 |
| Grad-CAM: 2.052 n=127, 1.347 n=23 | 5_exp 110 | none | — | **unverifiable** | J3 |
| Grad-CAM remedy 0% / 0.4% | 5_exp 110 | `outputs/gradcam_check/degeneracy_log.json`: `target_layer features.denseblock3`; 263 images; chex 0/263, rsna 1/263 | 0% / 0.38% | yes | none |
| Grad-CAM remedy lift 1.196 | 5_exp 110 | none | — | **unverifiable** | J3 |
| Grad-CAM arm layer `features.denseblock3` | R4 168, X_suppl 115 | `logs/ppo-counterfactual-17702339_{0,1}.out`, `17704444_0.out`: `[gradcam] hooking requested layer 'features.denseblock3'` | denseblock3 | fills TODO | fill R4 |
| Per-validator claims, sparsity and Grad-CAM; Table 4 transfer of 3-ens / sparsity / Grad-CAM; pc 0.462→0.623 | 5_exp Table 4, 101, 108, 112 | none | — | **unverifiable** | J2 |
| Table 4 LHR column | Table 4 | JSONs: 2.293, 2.673, 2.492 (n=27), 2.398, 1.636 | as stated | yes | none |
| Paired d's 0.482 / 0.512 / 1.205, 3-ens n=12 CI | 5_exp 99, 101, 108, 112, Table 4 | shared annotated counts reconstructed from shards + labels CSV: 76, 148, 148, 12 | counts match; statistics not saved | counts yes; stats **unverifiable** | J1 |
| λ_sparsity ramp | 5_exp 108, R1 60 | 17701295 logs (C4) | p→n ramp 0→2.0 as designed; n→p frozen at 0 | **no** (n→p) | C4 |
| Sparsity concentration "< 0.02", changed 0.650→0.661 | 5_exp 108 | JSONs: conc 0.253→0.264 (n→p), 0.318→0.299 (p→n); changed 0.6496→0.661 | numbers right | n→p effect is not the treatment (C4) | reword |
| 3-model membership | 4_eval 17, R6 242 | shard stamp `GUIDANCE_ENSEMBLE_WEIGHTS = [rsna, chex, mimic_nb]`; log `[rsna, chex, mimic_nb] (3 members` | rsna, chex, mimic_nb | fills TODO | fill |
| 3-model CV 0.624 (0.685 / 0.563) | 5_exp 101 | `checkpoints_ens3/eval_metrics.json` 0.6238 (0.685 / 0.5625 = 18/32) | as stated | yes, but n=32 undisclosed | C9 |
| No-policy 2.046 | 5_exp 12, Fig. 1 | `checkpoints/eval_none/eval_metrics.json` lift 2.0458, n=183, rsna-only, single-classifier groups | 2.046 | value yes; framing **no** | C8; J5 |
| No-policy label protocol 2.057 | Fig. 1 | `checkpoints/eval_none_cm0/eval_metrics.json` lift 2.0573, n=200 | 2.057 | yes | none |
| Random-action arm | 5_exp 103 | `logs/paired_arm_test_trained_random.out` (08-12 22:25): trained = `checkpoints/eval_shards` (single-classifier), 183 images, +0.0895 [+0.0211, +0.1596], t 2.551, p 0.01157, Wilcoxon p 0.01468, d +0.189, 105/183 | as stated, against the single-classifier policy | yes | none |
| Seed sweep deltas | 5_exp 101 | cm1 − cm0 lift from JSONs (identical image lists): +0.0694, +0.0693, +0.0575 | as stated | yes | none |
| Seed sweep d's and CIs | 5_exp 101 | none | — | **unverifiable** | J1 |
| Training times | 5_exp 9 | log start/finish: n→p 8h39m40s, p→n 11h00m10s, including LoRA 5.69 h / 5.12 h | 8h40m / 11h00m | yes (but see C12) | none |
| Guidance-bias / action statistics | limitations 10, intro | `checkpoints/guidance_diagnostics_{n_to_p,p_to_n}.json` (sampled actions, 20 training episodes): bias 1.2980±0.0019 / 1.2104±0.0930; passes 3.00 / 2.33; 21 / 45 steps | near-constant for n→p | **no** (C10) | reword; J6 |
| 76 shared annotated images | 5_exp 99 | reconstructed: single Xp ∩ ens Xp = 78; annotated 76 | 76 | yes | none |
| "~70% and 24%" overlaps | (not in the current .tex) | computed instead: Xn 165/200 = 82.5%, Xp 78/200 = 39%; 76/183 = 41.5%, 76/148 = 51.4% | not reproduced | **unverifiable** | ask where they come from |
| GPU model and per-image runtime | (not stated in the paper) | eval shards ran on H100 NVL (1), A100-80GB (2), Quadro RTX 6000 (5); n→p 1.8-2.9 min/25 img = 4.3-7.0 s/img; p→n 12.2-31.5 min/25 = 29-76 s/img; memory not logged | range only | n/a | J7 for one GPU |
| Convergence evidence | (none claimed) | only `Update 1` and `Update 80` lines + final diagnostics; `ppo_latest.pt` holds no history (`ramp_history` empty). n→p avg_return 9.13→10.39, valid 100%→80%, std_mean 0.6067→0.6119; p→n 1.87→3.58, valid 50%→53%, 0.6065→0.6078 | 2 points per direction | **unverifiable** (no curve) | only a new training run with denser logging |
| Fig. 2/3 provenance and f(x) values | 5_exp captions/text | md5 equal to `checkpoints_ens_default/counterfactual_diff_*.png`; titles read f(x)=0.086→0.993 and 0.848→0.013 | as stated | yes (both generated with the n→p LoRA, C2) | none |
| "9,555 boxes" | 4_eval 27 | `stage_2_train_labels.csv` rows with Target=1: 9555 | 9555 | yes | none |

### F. Provenance of the single-classifier arm

- Trained by **17686489**, 2026-08-10 23:20:28 → n→p 08-11 02:50:33, p→n 03:30:47, both on
  hpc-n978 (H200). Policies now in `checkpoints/archive_pre_ensemble_default_20260812/`.
- Code: dirty working tree on master cf5204f. Modified/untracked then: config.py,
  environment.py, finetuning.py, ingestion.py, main.py, pipeline.py, metrics.py,
  seeding.py, splits.py and others. Committed the next evening as 5b37471…927d6b8
  (2026-08-11 22:36-23:13). Its config.py sha256 `ff168a23…` equals config.py at
  5b37471, 6869f3e, fb268c6, 927d6b8 and 34ac79c. The other files' content at run
  time is not hashed. The log lines they print (train-fold far pool, 24-image probe,
  saturation diagnostics) are present.
- **After** the split and seeding fixes: split seed 0 loaded, train seed 0, LoRA on
  the train fold, far pool on the train fold.
- Config used (manifest + config at 5b37471): rsna-only guidance; reward weights
  LAM_FID 2.0, LAM_PRES_R 3.0, LAM_HIT 2.0, LAM_STOP 2.0 (identical to committed); a1 =
  0.3 + 0.9σ (narrowed on 2026-08-07, before cf5204f); ANCHOR_TO_EPISODE_ORIGIN True;
  T = 25; MLP trunk; LoRA rank 8; FIXED_EPISODE_LENGTH False; budget ∞.
- Differences to the committed run that remain: guidance (rsna vs rsna+chex), and
  through it the X_n/X_p groups (Xp overlap 78/200); hardware (H200 vs A100/H100);
  for the committed p→n, the generator mismatch (C2); different LoRA weights
  (independent fine-tunes, same recipe).
- Matched single-classifier retrain: **17686489 already is one, at config level.**
  No other single-classifier retrain exists. The only candidate older v1
  (`archive_v1_rank8`, CV 0.975) is not on disk.

### G. Integrity

| check | result |
|---|---|
| Every `eval_metrics.json` vs the values printed by its reduce log (CV, 4 FIDs, IP, LHR lift/t/n, concentration, changed, CTR Δ/t, SCP spread) | 18/18 match the last reduce that wrote each file. `eval_none_cm4` was written twice (17689952, then 17689971); the file matches the second |
| Guidance diagnostics JSONs vs training logs | match |
| `checkpoints/eval_metrics.json` FID −90/−50 edit | present only in git history (cf5204f, on the older v1 file: 103.83→13.83, 59.73→9.73). Restored in fb268c6 by replacing the file with the fresh 17689479 output. The on-disk copy matches its log. `eval_metrics.restored-fid.json` keeps the v1 values with the real FIDs; no producing log is on disk |
| Other suspicious history | 0890625 (2026-07-21) version: FID 5.67 / 3.2 with CV 0.965, rounded like typed values; history only, not used by the paper. 35f0ea7: stray `²` in `checkpoints_ens_default/eval_metrics.json` (value unchanged) |
| Cluster vs repo copies | the repo is the cluster copy; no second copy of these files exists under `~` (the export folder of today is an archive, not read as evidence) |

## 5. Open TODOs in the paper (`grep -rn TODO PPO-CF_CVPR_Paper`, 57 lines)

R = resolvable now by §6 (after your confirmation); C = conflict (§2), you decide;
J = needs a job; K = keep.

| file:line | TODO | status |
|---|---|---|
| preamble.tex:19, :23 | `\TODO` macro definitions | K |
| sec/0_abstract.tex:3 | "likely differs" rests on development history | C (C1) |
| sec/3_method.tex:17 | verify split (LoRA) | C (C3) |
| sec/3_method.tex:24 | verify guidance eq. vs committed run | R |
| sec/3_method.tex:32 | verify lock vs committed run | R |
| sec/3_method.tex:38 | trunk / T | R |
| sec/3_method.tex:62 | subsection describes last recorded config | R |
| sec/3_method.tex:63 | trunk / T | R |
| sec/3_method.tex:64 | VAE posterior sampling | R |
| sec/3_method.tex:65 | observation dimension | R |
| sec/3_method.tex:66 | fixed-episode-length | R |
| sec/3_method.tex:69 | fixed-episode-length | R |
| sec/4_evaluation.tex:17 | 3-model membership | R |
| sec/4_evaluation.tex:26 | LHR implementation | R |
| sec/4_evaluation.tex:40 | stamp mechanism | C (C6) |
| sec/4_evaluation.tex:41 | shard-stamp denominator (contains file:line refs) | C (C6) |
| sec/4_evaluation.tex:47 | seeding code | C (C5) |
| sec/4_evaluation.tex:48 | restore claim if code recovered | C (C5) |
| sec/4_evaluation.tex:50 | generator re-fine-tuned on split | C (C3) |
| sec/5_experiments.tex:30 | FID unverified | R (D) |
| sec/5_experiments.tex:31 | MS-SSIM unverified | R (D) |
| sec/5_experiments.tex:32 | HF-corr unverified | R (D) |
| sec/5_experiments.tex:60 | bootstrap code | J2 |
| sec/5_experiments.tex:100 | single-classifier config | C (C1) |
| sec/5_experiments.tex:117 | CTR | R (D) |
| sec/5_experiments.tex:118 | SCP | R (D) |
| sec/6_limitations.tex:7 | SCP 0.19% | R (D) |
| sec/appendix_repro.tex:25 | `\TODO{not found}` legend | K until all cells are filled |
| sec/appendix_repro.tex:137 | training seeds | C (C5) |
| sec/appendix_repro.tex:158 | LoRA training images | C (C3) |
| sec/appendix_repro.tex:159 | DDIM η | R |
| sec/appendix_repro.tex:168 | Grad-CAM fix layer | R |
| sec/appendix_repro.tex:189-201 (13 lines) | R5 "Committed runs" column | R |
| sec/appendix_repro.tex:206 | seeds / split / arms / LHR / bootstrap / stamp / manifests | split R; seeds C5; arms R; LHR R; bootstrap J2; stamp C6; manifests R |
| sec/appendix_repro.tex:242 | 3-model membership | R |
| sec/appendix_repro.tex:250 | stamp; averaging rule | stamp C6; averaging R |
| sec/appendix_repro.tex:252 | stamp code comment (file:line refs) | C (C6) |
| sec/appendix_repro.tex:255 | bootstrap interval level | J2 (code says 95%, but the code is not run-stamped) |
| sec/appendix_repro.tex:256 | bootstrap code not preserved | J2 |
| sec/appendix_repro.tex:258 | LHR details, d definition | R |
| sec/appendix_repro.tex:259 | LHR code not preserved | R |
| sec/X_suppl.tex:91, :115, :123 | stale file, not \input anywhere | delete the file or leave; contradicts the appendix |

## 6. Ready-to-apply edits (held until you confirm §3.1 and §4.D)

Values only from §4 evidence; no file:line text goes into the .tex.

1. `sec/appendix_repro.tex` R5 table, "Committed runs" column, lines 189-201:
   T `25`; episode end `validity or $t=T$`; $B$ `$\infty$`; trunk `MLP`; LoRA
   `8 / attention`; epochs `30`; $a_1$ `$0.30 + 0.90\,\sigma(a_1)$`; R_fid
   `mean $L_1$, 2.0`; λ_pres `3.0`; λ_hit, λ_stop `2.0, 2.0`; anchor `episode's original
   image`; saturation `0.01`; probe holdout `removed from training pool`. Then
   replace the R5 sentence "For all of them except the guidance classifier, the value
   used by the committed runs is not recorded" with "The committed runs used the
   last recorded value of every setting except the guidance classifier", and remove
   the $^{\ast}$ marks plus the "Read first" box wording accordingly (wording to be
   agreed with you).
2. R4 line 159: DDIM η = 0 (library default, not set in code). Line 168: fix-arm layer
   `features.denseblock3`.
3. R3/R4/R6 metric cells: LHR details (union of boxes, rescale from DICOM size,
   per-image mean of lifts, paired $t$ of hit rate vs null); transfer mean =
   unweighted mean of per-validator ratios; $d$ = mean/sd of paired differences;
   3-model membership `rsna, chex, mimic\_nb` (also `4_evaluation.tex:17`).
4. `sec/5_experiments.tex` lines 30-32, 117-118 and `6_limitations.tex` 7: delete the
   TODO comments (values unchanged; verified in §4.D).
5. `sec/3_method.tex` lines 24, 32, 38, 62-66, 69: delete the TODO comments (T=25,
   MLP, posterior sampling, 73,731, fixed-length False verified).
6. Remove the 80 `file:line` comments from the .tex files (your rule).
7. Fix `supplementary.tex` line 23 (remove the `\input{sec/suppl_details}` line, or
   point it at the intended file).

**Not applied, by rule (conflicts C1-C12):** seeds, LoRA training set, single-arm
provenance, stamp, sweep preservation, sparsity n→p, no-policy framing, 3-ens n, shared
generator.

## 7. Proposed rewording (claim strength changes; not applied)

- **Single vs ensemble (abstract, intro contribution 3, §4.4, §5.3, §5.4, limitations,
  conclusion)**, if you accept C1: "The single-classifier policy was trained under the
  same split, seeds, reward weights, action ranges and anchoring as the ensemble
  policy; the two runs differ in the guidance classifier, the classifier-selected
  evaluation groups (78 of 200 p→n sources shared) and the GPUs they ran on. The
  comparison is still one training run per arm." That makes the 0.482 comparison
  much less confounded than the paper says, but it stays single-run.
- **Shared generator (§3.4)**: "Each direction fine-tunes its own LoRA adapter with the
  same data, seed and recipe; at evaluation both directions use the n→p adapter." Plus
  a limitation, or J4 results in place of the current p→n numbers.
- **Leak caveat (§4.4, limitations)**: replace with "LoRA fine-tuning, PPO rollouts and
  the CV probe use only the 80% training fold; every evaluated image is in the 20%
  test fold."
- **Sparsity (§5.5)**: "In the n→p direction the CV safeguard froze λ_sparsity at 0
  at update 20, so only the p→n policy was trained with the sparsity term; the n→p
  changes (concentration, fraction changed) are run-to-run variation, not an effect
  of the term."
- **Adaptivity (intro, Table 1 caption, limitations)**: "The policy can emit image- and
  step-dependent controls; in practice the committed n→p policy holds its
  guidance-bias control at its upper bound (1.298 of 1.30) with three passes per
  step, and episodes last one to two steps on average, so the learned behaviour is
  close to a fixed, aggressive one-shot schedule." Update after J6.
- **No-policy (§5.2, Fig. 1)**: "…the no-policy arm, run with rsna-only guidance on
  the single-classifier's evaluation groups, already scores 2.046…". Update after J5.
- **3-ens (Table 4, §5.4)**: add "p→n: n=32 (27 annotated)".

## 8. What would change the paper's claims if it were different from what the paper assumes

1. p→n results with the matched generator (J4). If LHR or p→n transfer move
   materially, Table 2/3/4 and the abstract's 0.186 / 2.67× change.
2. The no-policy arm under ensemble guidance on the committed groups (J5). If it
   matches 2.673 within noise, the policy contributes nothing measurable to lesion
   targeting, and contribution 1 has to be stated as a negative result.
3. Eval-time action statistics (J6). If the n→p controls are constant across images,
   "per-image scheduling" is not supported for n→p.
4. Bootstrap CIs (J2). If the recomputed intervals overlap for p→n nih, the "intervals
   do not overlap" sentence and the strongest single-vs-ensemble transfer claim fall.
5. The single-vs-ensemble paired test (J1). d=0.482 and p=7.1e-5 have no saved
   output; if they do not reproduce, §5.4's headline comparison changes.
6. Sparsity and Grad-CAM paired statistics (J1). Both negative results rest on unsaved
   numbers; the LHR means are verified, so the direction is likely to hold.
7. Seed variance (J9). With one run per arm, the 0.838 vs 0.940 CV and 2.673 vs 2.293
   LHR gaps have no run-to-run error bar. The sparsity arm's λ=0 n→p policy gives one
   informal replicate of the default n→p config (CV 0.935 vs 0.960, concentration 0.264
   vs 0.253, changed 0.661 vs 0.650), which suggests run-to-run noise of that size.
8. Grad-CAM diagnostics (J3). If chex does not degenerate at the default layer, the
   mechanism offered in §5.5 is wrong, although the trained-arm collapse
   (CV 0.0175, LHR 1.636) is verified independently.
