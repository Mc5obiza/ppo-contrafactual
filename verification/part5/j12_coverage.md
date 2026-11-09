
## n_to_p: J6-mean schedule vs grid

| control | full range | J6 mean | grid values / fixed | nearest grid value | gap as % of range |
|---|---|---|---|---|---|
| strength | 0.2-0.7 | 0.2001 | grid [0.25, 0.45, 0.65] | 0.25 | 10.0% |
| cls_scale | 0.3-1.2 | 0.6045 | grid [0.5, 0.85, 1.15] | 0.5 | 11.6% |
| pres_scale | 0.3-1.5 | 0.4422 | fixed 0.9 | 0.9 | 38.1% |
| tau | 0-1 | 0.9202 | fixed 0.5 | 0.5 | 42.0% |
| n_passes | 2-4 | 3.0000 | grid [2, 3] | 3 | 0.0% |
| strength_boost | 0.85-1.15 | 1.1450 | fixed 1.0 | 1.0 | 48.3% |
| guidance_bias | 0.7-1.3 | 1.2991 | grid [0.85, 1.05, 1.27] | 1.27 | 4.9% |
| ramp_start_frac | 0.1-0.5 | 0.2007 | fixed 0.3 | 0.3 | 24.8% |

Stage 1 (12 images): REF (tau 0.92, already paid) +11.256. Top 8 grid by corrected return:

| config | logged | corrected | valid | steps | in stage 2 (logged top 6)? |
|---|---|---|---|---|---|
| g0.25_1.15_0.85_2 | +6.810 | +11.310 | 1.00 | 1.00 | yes |
| g0.25_0.85_1.27_2 | +6.805 | +11.305 | 1.00 | 1.00 | yes |
| g0.25_1.15_1.05_2 | +6.802 | +11.302 | 1.00 | 1.00 | yes |
| g0.25_1.15_1.27_2 | +6.801 | +11.301 | 1.00 | 1.00 | yes |
| g0.25_0.85_1.05_2 | +6.799 | +11.299 | 1.00 | 1.00 | yes |
| g0.25_0.85_0.85_2 | +6.776 | +11.276 | 1.00 | 1.00 | yes |
| g0.25_0.5_1.27_2 | +6.758 | +11.258 | 1.00 | 1.00 | no |
| g0.25_0.85_1.05_3 | +6.754 | +11.254 | 1.00 | 1.00 | no |

Top 6 by corrected stage-1 return = top 6 by logged? True; overlap 6/6
Grid configs whose corrected stage-1 return >= REF: 7/54
Stage-2 configs, corrected mean return over 48 images: {'g0.25_1.15_0.85_2': 11.218, 'g0.25_1.15_1.27_2': 11.203, 'g0.25_0.85_1.27_2': 11.187, 'g0.25_0.85_1.05_2': 11.161, 'g0.25_0.85_0.85_2': 11.158, 'g0.25_1.15_1.05_2': 10.879}

## p_to_n: J6-mean schedule vs grid

| control | full range | J6 mean | grid values / fixed | nearest grid value | gap as % of range |
|---|---|---|---|---|---|
| strength | 0.2-0.7 | 0.6231 | grid [0.25, 0.45, 0.65] | 0.65 | 5.4% |
| cls_scale | 0.3-1.2 | 1.0610 | grid [0.5, 0.85, 1.15] | 1.15 | 9.9% |
| pres_scale | 0.3-1.5 | 1.0276 | fixed 0.9 | 0.9 | 10.6% |
| tau | 0-1 | 0.6191 | fixed 0.5 | 0.5 | 11.9% |
| n_passes | 2-4 | 2.0000 | grid [2, 3] | 2 | 0.0% |
| strength_boost | 0.85-1.15 | 0.9226 | fixed 1.0 | 1.0 | 25.8% |
| guidance_bias | 0.7-1.3 | 1.2707 | grid [0.85, 1.05, 1.27] | 1.27 | 0.1% |
| ramp_start_frac | 0.1-0.5 | 0.2410 | fixed 0.3 | 0.3 | 14.7% |

Stage 1 (12 images): REF (tau 0.62, already paid) -0.201. Top 8 grid by corrected return:

| config | logged | corrected | valid | steps | in stage 2 (logged top 6)? |
|---|---|---|---|---|---|
| g0.25_1.15_0.85_3 | +1.296 | +3.280 | 1.00 | 1.83 | yes |
| g0.25_1.15_1.27_3 | +1.069 | +3.053 | 1.00 | 1.83 | yes |
| g0.25_0.5_1.27_3 | -1.470 | +0.364 | 0.92 | 3.00 | yes |
| g0.25_0.85_1.27_3 | -1.523 | +0.310 | 0.92 | 3.00 | yes |
| g0.25_0.85_1.05_2 | -1.599 | +0.231 | 0.92 | 3.17 | yes |
| g0.25_0.85_0.85_3 | -1.662 | +0.170 | 0.92 | 3.08 | yes |
| g0.25_0.85_1.27_2 | -1.809 | +0.019 | 0.92 | 3.25 | no |
| g0.25_1.15_1.05_2 | -1.820 | +0.010 | 0.92 | 3.17 | no |

Top 6 by corrected stage-1 return = top 6 by logged? True; overlap 6/6
Grid configs whose corrected stage-1 return >= REF: 11/54
Stage-2 configs, corrected mean return over 48 images: {'g0.25_1.15_1.27_3': -2.622, 'g0.25_0.85_1.27_3': -3.963, 'g0.25_1.15_0.85_3': -4.067, 'g0.25_0.85_0.85_3': -4.365, 'g0.25_0.5_1.27_3': -4.987, 'g0.25_0.85_1.05_2': -6.811}
