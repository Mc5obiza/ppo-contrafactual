# MEMO 2026-11-05 — J19: LHR under lung-aware nulls (second review, item 1)

Job 18537359 (scripts/31_lhr_nulls.sbatch → scripts/lhr_nulls.py), CPU, offline from saved shards.
Output: out/J19_lhr_nulls/{lhr_nulls.json, lhr_nulls_per_image.npz}. J19b (18537360) adds J16.

## Clearance
- `lift_img` (paper's metric, recomputed here independently of metrics.lesion_hit_rate) reproduces every
  reported value exactly: J4 2.595, J5 2.064, v1 single 2.293, ens3 2.492, sparsity 2.398, Grad-CAM 1.636.
- Source images identical across arms sharing an index (x_maxdiff_vs_cache = 0 for all 13 arms).
- Lung mask S: XRV PSPNet as in offline_metrics.py (= generator definition); L = S > 0.5. m = generator edit mask.
- Smoke-tested on 6 images on the login node before submission.

## Definitions (per annotated p→n image, A = RSNA box mask, d = |x_c − x|)
- lift_img  = (Σ_A d / Σ d) / (|A|/P)                         — paper metric
- lift_lung = (Σ_{A∩L} d / Σ_L d) / (|A∩L| / |L|)               — edit and null both restricted to lung
- lift_m    = same with the soft editable-region mask m as weight
- *_vae     = same with d = |x_c − D(E(x))|;  floor_* = with d = |D(E(x)) − x|
- uni_L_img = lift_img of a uniform wash over L (what "edit stays in lung, ignores lesion" scores)
- shuf_*    = lift with every other annotated image's box (mean over j ≠ i): generic lesion-prone-zone component

## Results (J4 default, n = 148, mean [95% t-CI])
| metric | J4 | J5 no policy | J5c fixed | J12 tuned fixed | J15 Latent Shift | J10 CycleGAN |
|---|---|---|---|---|---|---|
| lift_img (paper) | 2.595 | 2.064 | 2.525 | 2.408 | 1.174 | 1.483 |
| uniform wash in lung | 3.206 | 3.206 | 3.206 | 3.206 | 3.206 | 3.206 |
| uniform wash in m | 1.903 | 1.903 | 1.903 | 1.903 | 1.903 | 1.903 |
| **lift_lung** | **1.165 [1.123, 1.207]** | 1.088 | 1.162 | 1.177 | 1.055 | 1.160 |
| lift_m | 1.660 | 1.447 | 1.634 | 1.612 | 1.118 | 1.339 |
| lift_lung, d vs D(E(x)) | 1.163 | 1.082 | 1.161 | 1.179 | 1.050 | 1.068 |
| VAE floor lift_img / lift_lung | 1.058 / 0.982 | same | same | same | same | same |
| mismatched-box lift_lung | 1.089 | 1.043 | 1.071 | 1.080 | 1.022 | 1.096 |
| in-lung share of Σd | 0.661 | 0.556 | 0.643 | 0.613 | 0.301 | 0.359 |

Reading:
1. **The lift survives every null, but it is small.** Within the lung, the default puts 16.5% more edit mass on the box
   than a lung-uniform edit (p = 1.6e-12). The VAE round trip alone is not lesion-directed (floor lung lift 0.98), and
   measuring the edit against D(E(x)) leaves lift_lung unchanged (1.163), so it is not a VAE artefact.
2. **The whole-image 2.6× is mostly "the edit stays in the lung".** A uniform in-lung wash scores 3.21 under the paper's
   null, higher than every method. The paper's calibration sentence ("uniform wash 1.00") uses a whole-image wash and must
   be supplemented with the in-lung wash.
3. **About half of the in-lung lift is generic.** Scoring each edit against other patients' boxes gives 1.089: edits favour
   lesion-prone lung zones in general. The image-specific part is lift_lung − shuf_lung = +0.076 [+0.037, +0.117], p = 3e-4.

## Paired comparisons: which claims survive (lift_lung, paired t / Wilcoxon)
| comparison | paper (lift_img) | lift_lung diff [boot 95%] | p | verdict |
|---|---|---|---|---|
| J4 vs J5 no policy | +0.532 | +0.077 [+0.037, +0.120] | 5e-4 | **survives**, ~7× smaller |
| J4 vs J5c fixed | +0.071 (p=0.12) | +0.003 | 0.86 | unchanged (no difference) |
| J4 vs J12 tuned fixed | +0.187 (sig.) | −0.012 [−0.047, +0.023] | 0.51 | **vanishes**: J12's LHR "loss" was locality |
| J4 vs J9 second seed | +0.117 (p=0.04) | −0.025 | 0.23 | vanishes |
| J4 vs J15 Latent Shift | +1.421 | +0.110 [+0.055, +0.166] | 1.5e-4 | survives, much smaller |
| J4 vs J10 CycleGAN | +1.113 | +0.005 | 0.84 | **vanishes**: CycleGAN equals policy in lung |
| Eval A vs v1 single (76 shared) | +0.285 | +0.074 [+0.022, +0.125] | 0.006 | survives |
| J4 vs J16 single, same 148 images (J19b) | +0.300 | +0.082 [+0.042, +0.124] | 2e-4 | **survives**; J16 lung lift 1.083 = no policy (1.088) |
| Eval A vs sparsity | +0.275 | +0.026 | 0.085 | **vanishes** (Wilcoxon 0.20) |
| Eval A vs Grad-CAM | +1.037 | +0.053 [+0.005, +0.102] | 0.035 | weak (Wilcoxon 6e-4) |
| Eval A vs ens3 (n=12) | +0.101 | −0.027 | 0.70 | unresolved, as before |
| v1 trained vs random actions | +0.090 (p=0.012) | +0.003 | 0.85 | **vanishes** |

## Consequences for the paper (proposal, not yet applied)
- Report lung-restricted lift as the primary LHR, with whole-image lift and the uniform in-lung reference (3.21) as context.
  Report the mismatched-box null as the "generic zone" component.
- Abstract / Sec 5.1 "2.60× above chance" → "1.17× within the lung (p < 1e-11), of which about half is lesion-specific".
- Table 5 / Fig 1 arm ranking: only ensemble > single survives; sparsity "worse" and trained > random do not.
  The "Grad-CAM much worse" result shrinks to a weak in-lung effect; its whole-image loss is mostly mass leaving the lung.
- Sec 5.2 / Supp E5: J12's significant LHR loss vs the policy is a locality effect; within the lung, a tuned fixed schedule
  matches the policy. This strengthens the "fixed schedule suffices" reading (review item 7).
- The CycleGAN baseline matches the policy in lung lift; Latent Shift is lower.

## Addendum: J20 smoke (login node, first 40 images per group; NOT for the paper)
- Killed by its own 40-min timeout after FGSM eps 4/255, not a crash. Parts A (test fold) and B (arms) completed.
- Test-fold real-pair ceilings on 40 images are noisy (full J18 means give nih 0.46, pc 0.58, mimic_ch 0.42).
- **Early warning for item 2:** FGSM against the guidance ensemble alone already gives rho ≈ 0.30 (nih), 0.43 (pc),
  0.24 (mimic_ch) at eps 4/255 for n→p, the same range as J4's counterfactuals (0.03 / 0.44 / 0.30 on these 40).
  If the full run confirms this, rho_v alone does not separate adversarial from class-relevant edits, and the
  "< 0.15 = adversarial regime" reading in Sec 4 / 5 must go.
- J10 CycleGAN has a tiny guidance delta (0.11 n→p, 0.05 p→n), so its rho is unstable (> 1); report its deltas, not rho.
- J20 edited while pending: PGD 10 steps (see JOBS_SUBMITTED).

## Addendum 2026-11-06: J19c (18542421), iterative Latent Shift (J15b)
- lift_img 1.451, **lift_lung 1.106 [1.077, 1.135]** (p = 5e-11), lift_m 1.257, mismatched-box lung 1.043, in-lung share 0.367.
- Paired vs J4 on 148 images (from lhr_nulls_per_image.npz with lhr_nulls.paired): lift_lung +0.059 [+0.020, +0.099], p = 0.004
  (Wilcoxon 0.002); lift_img +1.144 (p = 4e-37); lift_m +0.403 (p = 1e-22). Policy beats iterative Latent Shift in lung, by a small margin.
  (computed separately because
  J15b was not in the script's PAIRS list).

## Clearance addendum 2026-11-06: independent recomputation agrees
scripts/inlung_lhr.py (written separately on 2026-11-05 21:45 for the J15 baseline work; out/inlung_lhr/inlung_all.json)
gives in-lung lift J4 1.1649, J5 1.0881, J10 1.1601, J15 1.0548, J15b 1.1060 and paired differences vs J4 +0.0768,
+0.0048, +0.1101, +0.0589: identical to lhr_nulls.py to 4 decimals. J19 is cleared.

## Paper integration 2026-11-06 (commit on branch mammo-heldout-transfer)
In-lung LHR is the primary LHR (Eq. 5); whole-image lift is context. Main text Sec 4.2/4.3/5, Fig 1 (dot plot, in-lung),
Table 3 (transfer with ceiling + attack floor), Supp E3 (baselines), E5 (in-lung column), E7 (nulls), E8 (calibration).
