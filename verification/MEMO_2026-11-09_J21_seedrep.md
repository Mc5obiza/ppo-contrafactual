# MEMO 2026-11-09 — J21 PPO seed replicates (second review, item 4): CLEARED

Runs: J17 trained PPO-only seed replicates (scripts/28, train_ppo_only.py; LoRA step skipped, each
direction loads its own already-trained adapter). J21 evaluated them on J4's 400 images and seeds
(scripts/26+27). In-lung LHR and nulls: J19d (scripts/lhr_nulls.py, run on the login node after
scancel of 18556456; log logs/J19d_lhr_nulls_local.log; out/J19d_lhr_nulls_seedrep).

| run | guidance | PPO seeds (n->p / p->n) | adapters (n->p / p->n) |
|---|---|---|---|
| J4 (default, reference) | rsna+chex | 1 / 2 | 658cdd79 / 6ae44230 |
| J21 ens seed2 | rsna+chex | 3 / 4 (TRAIN_SEED=2) | 658cdd79 / 6ae44230 |
| J9 seed1 (existing "second seed") | rsna+chex | TRAIN_SEED=1 | shared adapter, resumed (n->p u15, p->n u80) |
| J16 single (original single policy) | rsna | 1 / 2 | db36b707 / 01d6c7e3 |
| J21 single seed1 | rsna | 2 / 3 (TRAIN_SEED=1) | db36b707 / 01d6c7e3 |
| J21 single seed2 | rsna | 3 / 4 (TRAIN_SEED=2) | db36b707 / 01d6c7e3 |

## Clearance checks
1. Inputs = J4: xn, xp pixel-identical (max diff 0); xn_idx, xp_idx, n_slice, p_slice, base_seed equal, all 8 shards. PASS (all four J16/J21 runs).
2. Adapter sha256 MATCH lines in every gen task log (8/8 per run); shas as in the table. PASS.
3. Policies: each run loaded its own out/seedrep_*/<dir>/ppo_policy.pt (log lines); six distinct sha256. No counterfactual image is identical across runs (0/200 in every pair). PASS.
4. Config stamp: USE_GUIDANCE_ENSEMBLE True (ens) / False (single). PASS.
5. Reduce vs CPU recomputation: whole-image LHR from paired_lhr_vs_J4.txt (2.4002, 2.2204, 2.3556) = J19d lift_img (2.400, 2.220, 2.356). J4/J5/J16/J9 J19d values reproduce J19/J19b exactly. PASS.
6. Training: all six tasks reached update 100 / 800 episodes. **Caveat:** the three p->n tasks were PREEMPTED at 2026-11-06 00:54 and resumed from ppo_latest.pt (ens s2 at update 70, single s1 at 75, single s2 at 20) without saved RNG state. n->p trained uninterrupted.
7. Single-classifier n->p CV is 0.855 (171/200) in all three seeds: coincidence of counts, images differ (mean |diff| 0.029-0.039 between seeds).

## Results (J4's 400 images; out/J19d_lhr_nulls_seedrep/seed_table.txt)
| run | CV n->p | CV p->n | rho mean n->p | rho mean p->n | nih n->p | in-lung LHR [t 95%] | whole |
|---|---|---|---|---|---|---|---|
| J4 ens | 0.945 | 0.670 | 0.281 | 0.276 | 0.044 | 1.165 [1.123, 1.207] | 2.595 |
| J21 ens s2 | 0.950 | 0.765 | 0.269 | 0.229 | 0.024 | 1.140 [1.106, 1.175] | 2.400 |
| J9 ens (caveats) | 0.915 | 0.495 | 0.332 | 0.208 | 0.204 | 1.190 [1.151, 1.229] | 2.479 |
| J16 single s0 | 0.855 | 0.840 | 0.245 | 0.141 | 0.117 | 1.083 [1.044, 1.123] | 2.296 |
| J21 single s1 | 0.855 | 0.800 | 0.299 | 0.116 | 0.140 | 1.080 [1.046, 1.114] | 2.220 |
| J21 single s2 | 0.855 | 0.850 | 0.232 | 0.141 | 0.039 | 1.107 [1.069, 1.144] | 2.356 |
| J5 no policy | 0.940 | 0.225 | 0.307 | 0.169 | 0.101 | 1.088 [1.054, 1.122] | 2.064 |

Paired in-lung LHR (148 images, bootstrap 10k, one rng seed 0 stream):
- J4 - ens s2 +0.025 [-0.007, +0.057] p=0.14; J4 - J9 -0.025 [-0.066, +0.016] p=0.23.
- Every ens run minus every single run: 8/9 pairs p<0.01; ens s2 - single s2 +0.034 [-0.003, +0.072], p=0.076 (Wilcoxon 0.011).
- Mean of (J4, ens s2) minus mean of 3 single runs, per image: +0.063 [+0.037, +0.092], p=1.9e-5. Including J9: +0.075 [+0.050, +0.102].
- vs no policy: J4 +0.077, ens s2 +0.052 [+0.014, +0.091] p=0.008, J9 +0.102; single s0 -0.005 (p=0.80), s1 -0.008 (0.66), s2 +0.019 (0.31).
- Own minus other boxes: ens s2 +0.072 [+0.038, +0.105]; single runs +0.057 to +0.082.

## Reading
Survives seed variation: ensemble > single classifier on in-lung LHR (every run above every run: 1.140-1.190 vs 1.080-1.107) and on p->n transfer (0.208-0.276 vs 0.116-0.141); ensemble policy > no policy on in-lung LHR and p->n CV in every run; single-classifier policy = no policy on in-lung LHR in every seed.
Does NOT survive: "ensemble has higher n->p transfer" (0.269-0.332 vs 0.232-0.299 overlap; single s1 0.299 > J4 0.281).
Seed-sensitive: p->n CV of the ensemble (0.495-0.765); every ensemble run is below every single run on p->n CV (0.495-0.765 vs 0.800-0.850), measured with each run's own guidance classifier.
High-CV / low-transfer n->p pattern holds for all seeds (CV 0.915-0.950, rho 0.27-0.33).

## J23 — rule-selected qualitative figure (main paper Fig. 1)
scripts/qual_scores.py (per-image guidance + validator scores on J4; self-check vs J4 transfer.json: 8/8 within 3e-4, PASS)
and scripts/fig_examples.py (selection rule in its docstring; asserts J4 validity 189/134 and in-lung LHR 1.1649 from J19 per-image).
Picks: (a) n->p xn position 108, dataset idx 2029: guidance 0.040->0.986, nih 0.009->0.016, pc 0.381->0.735, mimic_ch 0.071->0.518.
(b) p->n position 190, idx 12032: lift 1.160 (median of 100 valid annotated). (c) p->n position 56, idx 2535: lift 0.511 (lowest),
nih 0.210->0.472 (wrong way). Outputs out/J23_qual/fig_examples.{pdf,png,json}; copied to PPO-CF_CVPR_Paper/figures/fig2_examples.pdf.
