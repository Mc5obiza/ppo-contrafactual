| arm | dir | LoRA tensors differing | max abs Δ (LoRA) | rel. Frobenius ‖Δ‖/‖W_np‖ (LoRA) | non-LoRA max Δ | cond_proj max abs Δ | cond_proj rel. ‖Δ‖ | sha256 np / pn (first 8) |
|---|---|---|---|---|---|---|---|---|
| 2-ens committed (default) | `checkpoints` | 256/256 | 1.18e-01 | 3.85e-01 | 0.0e+00 | 7.46e-02 | 2.51e-01 | 658cdd79 / 6ae44230 |
| single-classifier | `checkpoints/archive_pre_ensemble_default_20260812` | 225/256 | 1.72e-02 | 1.18e-02 | 0.0e+00 | 1.93e-03 | 6.82e-03 | db36b707 / 01d6c7e3 |
| 3-model ensemble | `checkpoints_ens3` | 225/256 | 2.76e-03 | 3.94e-03 | 0.0e+00 | 5.77e-04 | 2.02e-03 | ef0703e4 / 6fea64fb |
| sparsity reward | `checkpoints_sparsity` | 225/256 | 5.05e-04 | 1.49e-03 | 0.0e+00 | 3.14e-04 | 1.02e-03 | 83cacddd / a2188f63 |
| Grad-CAM constraint | `checkpoints_gradcam_fix` | 0/256 | 0.00e+00 | 0.00e+00 | 0.0e+00 | 0.00e+00 | 0.00e+00 | dc015f6a / dc015f6a |
