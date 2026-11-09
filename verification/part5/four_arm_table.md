
### n_to_p

| arm | CV (n=200) | transfer nih | transfer pc | transfer mimic_ch | transfer mean | edit concentration | fraction changed |
|---|---|---|---|---|---|---|---|
| committed policy (J4) | 0.945 [0.904, 0.969] | 0.044 [0.005, 0.082] | 0.465 [0.436, 0.494] | 0.333 [0.302, 0.364] | 0.281 [0.259, 0.303] | 0.254 [0.251, 0.258] | 0.647 [0.639, 0.655] |
| no policy (J5) | 0.940 [0.898, 0.965] | 0.101 [0.059, 0.142] | 0.474 [0.444, 0.503] | 0.345 [0.314, 0.376] | 0.307 [0.284, 0.329] | 0.268 [0.264, 0.272] | 0.641 [0.634, 0.649] |
| fixed schedule, policy mean (J5c) | 0.945 [0.904, 0.969] | 0.044 [0.004, 0.083] | 0.469 [0.439, 0.499] | 0.334 [0.302, 0.365] | 0.282 [0.260, 0.304] | 0.254 [0.251, 0.258] | 0.647 [0.638, 0.655] |
| J9 seed 1† | 0.915 [0.868, 0.946] | 0.204 [0.161, 0.246] | 0.458 [0.428, 0.488] | 0.334 [0.304, 0.365] | 0.332 [0.310, 0.355] | 0.279 [0.274, 0.283] | 0.675 [0.667, 0.684] |

### p_to_n

| arm | CV (n=200) | LHR lift (n=148) | transfer nih | transfer pc | transfer mimic_ch | transfer mean | edit concentration | fraction changed |
|---|---|---|---|---|---|---|---|---|
| committed policy (J4) | 0.670 [0.602, 0.731] | 2.595 [2.450, 2.741] | 0.165 [0.138, 0.193] | 0.306 [0.278, 0.335] | 0.357 [0.317, 0.397] | 0.276 [0.254, 0.298] | 0.314 [0.309, 0.320] | 0.634 [0.624, 0.644] |
| no policy (J5) | 0.225 [0.173, 0.288] | 2.064 [1.961, 2.166] | 0.113 [0.083, 0.143] | 0.245 [0.218, 0.275] | 0.149 [0.099, 0.197] | 0.169 [0.146, 0.192] | 0.278 [0.273, 0.284] | 0.599 [0.590, 0.608] |
| fixed schedule, policy mean (J5c) | 0.660 [0.592, 0.722] | 2.525 [2.385, 2.664] | 0.151 [0.124, 0.179] | 0.261 [0.236, 0.288] | 0.319 [0.278, 0.357] | 0.244 [0.224, 0.264] | 0.312 [0.306, 0.317] | 0.631 [0.621, 0.641] |
| J9 seed 1† | 0.495 [0.426, 0.564] | 2.479 [2.350, 2.608] | 0.154 [0.126, 0.182] | 0.218 [0.192, 0.246] | 0.252 [0.208, 0.295] | 0.208 [0.187, 0.230] | 0.298 [0.292, 0.304] | 0.620 [0.610, 0.629] |

† J9: second PPO seed with a shared adapter (both directions use the same generator file); both directions were resumed from checkpoints (n→p at update 15, p→n at update 80 after a preemption) and no random-number state was saved.
CIs: CV Wilson; LHR, concentration, fraction changed t-intervals; transfer per validator GPU bootstrap (10k); transfer mean paired bootstrap (10k, CPU per-image deltas).

### Paired comparisons against the committed policy (J4)

| comparison | metric | difference [95% CI] | d (paired) | p | detail |
|---|---|---|---|---|---|
| J5c − J4 | LHR lift (p→n, n=148) | -0.071 [-0.161, +0.019] | -0.128 | 0.122 (paired t); Wilcoxon 0.225 | J5c higher on 64/148 |
| J5c − J4 | CV n→p (n=200) | +0.000 [+0.000, +0.000] | +0.000 | 1 (McNemar exact) | valid only under J5c: 0, only under J4: 0 |
| J5c − J4 | CV p→n (n=200) | -0.010 [-0.065, +0.045] | -0.026 | 0.856 (McNemar exact) | valid only under J5c: 14, only under J4: 16 |
| J9 − J4 | LHR lift (p→n, n=148) | -0.117 [-0.227, -0.006] | -0.172 | 0.0385 (paired t); Wilcoxon 0.00445 | J9 higher on 56/148 |
| J9 − J4 | CV n→p (n=200) | -0.030 [-0.055, -0.010] | -0.175 | 0.0312 (McNemar exact) | valid only under J9: 0, only under J4: 6 |
| J9 − J4 | CV p→n (n=200) | -0.175 [-0.240, -0.110] | -0.377 | 3.62e-07 (McNemar exact) | valid only under J9: 7, only under J4: 42 |
| J5 − J4 | LHR lift (p→n, n=148) | -0.532 [-0.645, -0.419] | -0.764 | 1.85e-16 (paired t); Wilcoxon 3.28e-19 | J5 higher on 20/148 |
| J5 − J4 | CV n→p (n=200) | -0.005 [-0.030, +0.020] | -0.027 | 1 (McNemar exact) | valid only under J5: 3, only under J4: 4 |
| J5 − J4 | CV p→n (n=200) | -0.445 [-0.515, -0.375] | -0.876 | 7.43e-26 (McNemar exact) | valid only under J5: 1, only under J4: 90 |

### How much of the policy's gain over no policy the fixed schedule recovers

| metric | policy gain J4 − J5 [95% CI] | fixed-schedule gain J5c − J5 [95% CI] | recovered (J5c − J5)/(J4 − J5) [95% CI] |
|---|---|---|---|
| LHR lift (p→n, n=148) | +0.532 [+0.425, +0.647] | +0.461 [+0.366, +0.563] | 87% [71%, 102%] |
| CV p→n (n=200) | +0.445 [+0.375, +0.515] | +0.435 [+0.370, +0.505] | 98% [86%, 111%] |
| CV n→p (n=200) | +0.005 [-0.020, +0.030] | +0.005 [-0.020, +0.030] | not defined: the policy gain is ~0 (ratio of near-zero numbers) |
