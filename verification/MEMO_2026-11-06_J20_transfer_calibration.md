# MEMO 2026-11-06 — J20: calibrating the transfer ratio (second review, items 2 and 10)

Job 18537367 (scripts/32 → scripts/transfer_calibration.py), CPU, 1 h 15. Output: out/J20_transfer_calibration/
{transfer_calibration.json, test_fold_scores.npz}. PGD reduced to 10 steps while pending (JOBS_SUBMITTED).

## Clearance
- J4 Pneumonia rho reproduces out/J4_perdir/transfer.json: n→p 0.04 / 0.47 / 0.33, p→n 0.17 / 0.31 / 0.36 (nih/pc/mimic_ch).
- Real-pair ceilings reproduce J18's means: 0.465 / 0.583 / 0.423.
- Guidance = rsna+chex mean ensemble throughout (also the denominator for J15/J10, whose own guidance differs).

## A. Validators on the 5,337-image test fold (item 10: label semantics)
| head | AUROC | real-pair ceiling rho [95%] |
|---|---|---|
| guidance ensemble | 0.872 | — |
| nih Pneumonia / Consolidation / Infiltration | 0.766 / 0.802 / 0.740 | 0.47 / 0.34 / 0.26 |
| pc Pneumonia / Consolidation / Infiltration | 0.783 / 0.798 / 0.804 | 0.58 / 0.61 / 0.77 |
| mimic_ch Pneumonia / Consolidation / **Lung Opacity** | 0.728 / 0.720 / **0.770** | 0.42 / 0.45 / **0.63** |
Lung Opacity (RSNA's actual positive class) is mimic_ch's best head; Consolidation is nih's best.

## B. J4 default: rho per head and decision flips at the Youden operating point
| | nih P/C/I | pc P/C/I | mimic_ch P/C/LO |
|---|---|---|---|
| rho n→p | 0.04 / 0.16 / 0.10 | 0.47 / 0.49 / 0.51 | 0.33 / 0.37 / 0.38 |
| rho p→n | 0.17 / 0.09 / 0.09 | 0.31 / 0.25 / 0.38 | 0.36 / 0.35 / 0.41 |
| flip n→p (Youden) | 0.22 / 0.27 / 0.25 | 0.91 / 0.91 / 0.82 | 0.69 / 0.76 / 0.57 |
| flip p→n (Youden) | 0.48 / 0.49 / 0.34 | 0.91 / 0.80 / 0.88 | 0.80 / 0.73 / 0.73 |
Guidance flip at its own Youden point: 1.00 / 0.97. Prevalence-matched thresholds are high (top 22%), so they make n→p
flips hard and p→n flips trivial; Youden is the operating point to report.
Reading: the semantically matched heads give the same picture as Pneumonia (Lung Opacity +0.05). Item 10 is resolved without
changing conclusions. **Decision flips are much higher than rho suggests**: pc and mimic_ch flip 70–90% of decisions.
"28% transfer" is a ratio of mean probability shifts; most validators still cross their own operating point.

## C. Adversarial floor (attacks on the guidance ensemble only, J4's 200 + 200 sources)
| attack | guidance Δ / CV@0.8 | L2 | rho nih/pc/mimic (Pneumonia) | Youden flips nih/pc/mimic |
|---|---|---|---|---|
| **n→p** J4 counterfactual | 0.90 / 1.00 | 12.4 | 0.04 / 0.47 / 0.33 | 0.22 / 0.91 / 0.69 |
| PGD-Linf 4/255 | 0.91 / 1.00 | 3.4 | 0.22 / 0.31 / 0.23 | 0.46 / 0.34 / 0.37 |
| PGD-Linf 8/255 | 0.91 / 1.00 | 6.1 | 0.27 / 0.42 / 0.37 | 0.58 / 0.79 / 0.92 |
| PGD-L2, norm-matched to each CF | 0.91 / 1.00 | 9.1 | 0.18 / 0.45 / 0.49 | 0.36 / 0.97 / 0.98 |
| **p→n** J4 counterfactual | 0.76 / 0.84 | 17.6 | 0.17 / 0.31 / 0.36 | 0.48 / 0.91 / 0.80 |
| PGD-Linf 4/255 | 0.82 / 0.96 | 3.2 | 0.02 / 0.02 / 0.07 | 0.07 / 0.07 / 0.16 |
| PGD-Linf 8/255 | 0.84 / 0.99 | 5.7 | 0.04 / 0.04 / 0.18 | 0.17 / 0.16 / 0.47 |
| PGD-L2, norm-matched | 0.84 / 0.99 | 13.4 | 0.07 / 0.46 / 0.02 | 0.35 / 0.99 / 0.23 |

Reading:
1. **p→n: the counterfactuals sit clearly between floor and ceiling.** An L∞ attack with the same guidance CV transfers
   at 0.02–0.07; the counterfactuals transfer at 0.17–0.36, about 60% of the real-pair ceiling (0.42–0.58).
2. **n→p: rho does NOT separate the counterfactuals from an attack.** Attacks on the guidance ensemble alone transfer at
   0.2–0.5 (DenseNet-121s trained on overlapping CXR data share adversarial directions), and the norm-matched L2 attack
   equals or beats the counterfactual on pc and mimic_ch in both rho and flips. Only nih separates (CF 0.04 < attack 0.18–0.27,
   in the wrong direction for the CF).
3. Consequence: the "< 0.15 = adversarial regime" threshold has no support and must be dropped. rho is a graded
   measure to report against both anchors (floor = matched-CV attack, ceiling = real pairs). The norm-matched L2 attack
   has Linf 0.37–0.55, so it is a large, possibly semi-semantic perturbation; report it with that caveat.
4. The n→p/p→n asymmetry is itself a finding: adding evidence of opacity is easy to fake transferably, removing it is not.

## Other arms (same machinery; Pneumonia rho n→p | p→n)
- J5 no policy: 0.10/0.47/0.34 | 0.11/0.25/0.15.  J12 tuned fixed: 0.04/0.47/0.33 | 0.18/0.20/0.25.
- J15 Latent Shift (single-shot): guidance CV@0.8 = 0.02 / 0.67, so rho is not comparable.
- J10 CycleGAN: guidance Δ 0.12 / 0.06; rho > 1 is a small-denominator artefact. Report deltas, not rho.
- J15b iterative Latent Shift (its own reduce, transfer_den_ens2.json): CV 1.00 / 0.995, rho nih 0.04 / 0.19,
  pc 0.34 / 0.24. **A second, independent method with high CV and low transfer** (review item 3).
