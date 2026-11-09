# MEMO 2026-11-06 — J22: LHR on CBIS-DDSM mammography counterfactuals (second review, item 6; exploratory)

Job 18542590 (scripts/33_breast_lhr.sbatch → scripts/breast_lhr.py), CPU, 12 min. Output: out/J22_breast_lhr/
{breast_lhr.json, breast_lhr_per_image.npz}. The user asked to "go on with the best decision that respects the deadline":
option 1 (exploratory supplement section, offline, no GPU).

## Inputs and checks
- Shards: checkpoints/cbis_ddsm_breast/eval_shards (branch test-cancer, Aug 9). Indices: eval_group_index.json
  (breast_shard_audit.py gate B). Every source image was re-checked pixel-identical to dataset[idx]: 0 mismatches.
- ROI = union of all ROI masks of the image (the dataset _meta keeps only the first), Resize 256, > 0.5. The mirror's
  JPEG masks carry ~16 grey levels of compression noise but 0% mid-range pixels; masks with >1% in 21..234 count as
  not binary (0 found). Dropped: n→p 9 size mismatches; p→n 3 size mismatches + 1 without ROI.
- Organ mask B: channel 0 of BreastAnatomySegmenter (copied verbatim from test-cancer into
  scripts/breast_segmentation_testcancer.py), > 0.5.
- Smoke test on 12 images per direction on the login node before submission.

## Results (mean [95% t-CI]; n→p n=191, p→n n=196)
| | benign→malignant | malignant→benign |
|---|---|---|
| in-breast lift | 1.547 [1.420, 1.674] | 1.892 [1.747, 2.038] |
| in-breast, vs D(E(x)) | 1.539 | 1.857 |
| in-breast, other ROIs | 1.134 | 1.133 |
| VAE floor in breast | 1.249 [1.196, 1.302] | 1.276 [1.236, 1.315] |
| whole image | 4.053 | 4.373 |
| uniform wash over B (whole image) | 3.010 | 2.734 |
| own − other ROIs | +0.413 [+0.294, +0.547] | +0.760 [+0.615, +0.913] |
| edit − VAE floor | +0.298 [+0.177, +0.425] | +0.617 [+0.483, +0.756] |

## Caveats (all stated in Supp. E9)
- The guidance classifier saw 336/400 eval images in training (audit gate C). CV 1.00/1.00 is meaningless.
- Validators fail the 0.65 AUROC floor (0.62-0.64, CBIS-disjoint and CMMD), so no transfer is reported.
- The run predates the split/seeding controls. There is no no-policy baseline, so lift cannot be attributed to the policy.
- The VAE floor is lesion-correlated in mammograms (unlike RSNA).

Paper: Supp. E9 + one sentence in Limitations (commit after 9d75399).
