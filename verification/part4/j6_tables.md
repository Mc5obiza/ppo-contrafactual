
## n_to_p: 200 images, 201 steps, 1 images with >1 step

| dim | control | mapping (s = sigmoid(a)) | raw mean | raw sd | control mean | control sd | between-share (multi-step images) | eta² (all images) | sat lo | sat hi |
|---|---|---|---|---|---|---|---|---|---|---|
| a0 | strength | 0.2 + 0.5 s | -9.114 | 1.176 | 0.2001 | 0.0004 | n/a | 1.00 | 100.0% | 0.0% |
| a1 | cls_scale | 0.3 + 0.9 s | -0.671 | 0.333 | 0.6082 | 0.0628 | n/a | 1.00 | 0.0% | 0.0% |
| a2 | pres_scale | 0.3 + 1.2 s | -2.006 | 0.681 | 0.4641 | 0.1003 | n/a | 1.00 | 0.0% | 0.0% |
| a3 | tau (stop signal) | s | +2.446 | 0.801 | 0.8999 | 0.0795 | n/a | 1.00 | 0.0% | 0.0% |
| a4 | extra pass 1 | [s > .5] | +3.041 | 1.073 | 1.0000 | 0.0000 | n/a | 1.00 | 0.0% | 10.9% |
| a5 | extra pass 2 | [s > .5] | -6.357 | 0.941 | 0.0000 | 0.0000 | n/a | 1.00 | 95.0% | 0.0% |
| a6 | strength_boost | 0.85 + 0.30 s | +4.074 | 0.692 | 1.1438 | 0.0046 | n/a | 1.00 | 0.0% | 20.4% |
| a7 | guidance_bias | 0.70 + 0.60 s | +6.550 | 1.191 | 1.2985 | 0.0016 | n/a | 1.00 | 0.0% | 97.5% |
| a8 | ramp_start_frac | 0.1 + 0.4 s | -1.090 | 0.788 | 0.2105 | 0.0602 | n/a | 1.00 | 0.0% | 0.0% |

n_passes per step: {3.0: 201} | guidance_bias >= 1.295 on 96.5% of steps

Episode length (n_steps per image; mean 1.00, median 1, cap 25):

| steps | 1 | 2 | total |
|---|---|---|---|
| images | 199 | 1 | 200 |

## p_to_n: 200 images, 1093 steps, 71 images with >1 step

| dim | control | mapping (s = sigmoid(a)) | raw mean | raw sd | control mean | control sd | between-share (multi-step images) | eta² (all images) | sat lo | sat hi |
|---|---|---|---|---|---|---|---|---|---|---|
| a0 | strength | 0.2 + 0.5 s | +1.705 | 1.685 | 0.5784 | 0.0874 | 0.71 | 0.77 | 0.0% | 9.3% |
| a1 | cls_scale | 0.3 + 0.9 s | +1.700 | 0.855 | 1.0372 | 0.0818 | 0.70 | 0.73 | 0.0% | 1.2% |
| a2 | pres_scale | 0.3 + 1.2 s | +0.432 | 0.536 | 1.0206 | 0.1436 | 0.78 | 0.72 | 0.0% | 0.0% |
| a3 | tau (stop signal) | s | +0.486 | 0.826 | 0.5970 | 0.1651 | 0.68 | 0.75 | 0.0% | 0.0% |
| a4 | extra pass 1 | [s > .5] | -1.607 | 1.680 | 0.1125 | 0.3160 | 0.72 | 0.77 | 8.8% | 0.0% |
| a5 | extra pass 2 | [s > .5] | -0.260 | 0.718 | 0.3385 | 0.4732 | 0.75 | 0.74 | 0.0% | 0.0% |
| a6 | strength_boost | 0.85 + 0.30 s | -1.142 | 0.766 | 0.9300 | 0.0349 | 0.75 | 0.81 | 0.0% | 0.0% |
| a7 | guidance_bias | 0.70 + 0.60 s | +2.968 | 1.995 | 1.2304 | 0.0631 | 0.67 | 0.76 | 0.0% | 22.1% |
| a8 | ramp_start_frac | 0.1 + 0.4 s | -0.608 | 1.572 | 0.2713 | 0.0938 | 0.72 | 0.82 | 2.4% | 0.0% |

n_passes per step: {2.0: 655, 3.0: 383, 4.0: 55} | guidance_bias >= 1.295 on 21.2% of steps

Episode length (n_steps per image; mean 5.46, median 1, cap 25):

| steps | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 11 | 12 | 13 | 17 | 22 | 25 | total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| images | 129 | 15 | 10 | 3 | 3 | 2 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 31 | 200 |
