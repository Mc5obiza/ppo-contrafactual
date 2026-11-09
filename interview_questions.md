# Interview Questions — PPO-Guided Latent Diffusion Counterfactuals for Chest Radiography

Grounded in the actual codebase and internship report (SDU / Maersk Mc-Kinney Moller Institute, 15 June – 15 August 2026, supervised by Prof. Sadok Ben Yahia).

---

## 1. Motivation and Problem Framing

1. Why generate counterfactual chest X-rays with a diffusion model guided by RL, instead of just using Grad-CAM or another saliency method on the classifier directly?
2. What does "counterfactual validity" mean in this project, and why is it a fundamentally different (weaker) claim than "this system explains disease"?
3. Why RSNA pneumonia specifically, and what does the binary framing (normal vs. pneumonia) buy you versus a multi-label setup?
4. The report's central thesis is "the most dangerous metric in a machine learning project is the one the system was optimised against." Unpack that in terms of what your PPO reward function actually optimises versus what CV measures.
5. Why was WP6 (validation/diagnostics) not part of the original plan, and what changed your priorities mid-internship?

## 2. RL / PPO Formulation

6. Walk through your 9-dimensional action space in `environment.py`'s `step()` — what does each of `strength`, `cls_scale`, `pres_scale`, `tau`, the two extra-pass booleans, `strength_boost`, `guidance_bias`, and `ramp_start_frac` actually control in the denoising loop?
7. Why is the action ceiling on `strength_t` capped at 1.2 (`0.2 + 0.5*sigmoid`) rather than the old 3.0 ceiling, and what failure mode did that fix?
8. Describe the observation vector built by `get_observation()` — what's in it, and why did you add a conv trunk as an alternative to the MLP trunk?
9. What is `direction_scale` in `_compute_reward`, why does it exist, and why does your own code comment say the fix to it is "close to a no-op" under GAE's advantage normalization?
10. Explain the `shaped_rewards`/`DELTA_MIX` bug in `compute_gae`/`update()` — what was computed, what was actually used, and how did you find it was dead code?
11. Why does `is_valid` require BOTH `score_ok` (classifier crossed threshold) AND `budget_ok` (edit_l1 under budget) rather than the score condition alone? What happened when it was score-only?
12. What is `FIXED_EPISODE_LENGTH` and why did you move away from early termination on validity? What was `avg_ep_len` measuring before and after?
13. Explain the actor-saturation collapse: what did `guidance_bias` measure at convergence in v1, and how does the `ACT_SAT_THRESHOLD`/`LAM_ACT_SAT` hinge loss address it?
14. Your `PPOActorCritic` has auxiliary feasibility, progress, and stop heads (`feasibility_head`, `progress_head`, `stop_head`) that aren't used in `get_action_and_value`'s sampling. What are they actually for, and where do they enter the loss?
15. Why train two entirely separate policies (n→p and p→n) instead of one target-conditioned policy? What's the cost of that design choice?
16. Walk through the reward decomposition: `R_prog`, `R_val`, `R_close2`, `R_fid`, `R_pres`, `R_bud`, `R_hit`, `R_stop`, `R_sparsity` — what does each term reward or penalize, and what were their weights before vs. after the v1→v2 rebalance?
17. Why is `R_fid` computed as L1 mean pixel difference rather than the original L2/MSE, and what magnitude change did that force in `LAM_FID`?
18. What is the CV safeguard / sparsity ramp-freeze mechanism in `PPOTrainer`, and why is the freeze "latched" rather than allowed to un-freeze if CV recovers?
19. Explain your PPO hyperparameters (`GAMMA=0.99`, `GAE_LAMBDA=0.95`, `PPO_CLIP=0.2`, `LR=3e-4`, `C_ENTROPY=0.01`) and whether any were tuned versus left at defaults.
20. Why does the reward-term "share" diagnostic in `_format_reward_split` normalize by the sum of absolute values rather than signed total reward?

## 3. Diffusion Guidance Mechanics

21. Explain classifier guidance in your pipeline: how is `grad_cls` computed, and why do you take TWO separate backward passes (`grad_cls`, `grad_pres`) instead of fusing them into one weighted loss?
22. Your code comment says fusing the two backward passes is "not the same vector in fp16." Explain why, quantitatively, and what precision decision you made instead.
23. What is the "background lock" (RePaint / Blended Latent Diffusion style), and how does it differ from masked guidance? Why do you need both?
24. What does `guidance_ramp_start_frac` do, and why did you replace a hard on/off switch at the trajectory midpoint with a smooth ramp?
25. Explain the anchoring bug: what happens across ~100 chained encode/renoise/denoise/decode passes per episode if `x_anchor` isn't fixed to the episode's original image, and how does `ANCHOR_TO_EPISODE_ORIGIN` fix it?
26. Why is `edit_mask` built by multiplying the lung segmentation mask by the Grad-CAM mask rather than replacing one with the other?
27. What is your LoRA configuration (rank, alpha, dropout, target modules), and why target only `to_q/to_k/to_v/to_out.0`? What fraction of the 866M-parameter UNet does this actually train?
28. Why upcast the LoRA-trainable parameters to fp32 while leaving the frozen base UNet/VAE in fp16?
29. What does `cond_proj` do, and how does target-class conditioning actually reach the UNet's cross-attention given SD1.5 wasn't designed for scalar/latent conditioning?

## 4. Grad-CAM Action-Space Constraint

30. What's the difference between using Grad-CAM as a post-hoc penalty on diffuse edits versus as a hard action-space constraint, and why did you choose the latter?
31. Why is the Grad-CAM computed once per episode rather than per PPO step or per refinement pass?
32. Explain `DegenerateCAMError` — what causes a Grad-CAM to go spatially constant, and why is falling back to "no constraint" safer than returning a uniform mask at `floor`?
33. Why does `find_target_layer` specifically look for a submodule named "features" rather than hooking the last Conv2d, and what failure mode did that avoid on DenseNet?
34. Why does `grad_cam()` use a forward hook + `retain_grad()` instead of `register_full_backward_hook`? What PyTorch constraint forced that choice?
35. In `ensemble_grad_cam`, why is passing the ensemble object directly to `find_target_layer` dangerous, and what would it silently do?
36. Your final results show the Grad-CAM constraint "confirmed to hurt" lesion targeting with the largest effect size in the project (Cohen's d=1.205, CV collapsing from 0.838 to 0.017). Walk through why a constraint intended to help ended up being the worst intervention you tried.

## 5. Classifier Design and Ensemble

37. Why replace the original ResNet-18 classifier with TorchXRayVision's DenseNet-121? What specifically pointed you at shortcut learning as the diagnosis?
38. What is the "dead head" problem you guard against in `XRVPneumoniaClassifier.__init__` (`op_threshs` NaN), and why does it matter for gradient-based guidance?
39. Explain the rationale for `EnsemblePneumoniaClassifier`: why does a 2-head single-corpus model get called "the easiest to fool," and how does head-count across weight sets (`-rsna` 2, `-chex`/`-mimic_ch` 11, `-pc` 15, `-nih` 14) support that argument?
40. The ensemble uses mean-probability across members with a single BCE loss, not mean-of-per-member-BCE. Why does that "route most of the gradient to the lagging member," and when might that matter?
41. Your 3-model ensemble experiment showed mixed results — improved held-out transfer on `pc` but unresolved on `nih` at that sample size. What would it take to resolve `nih`?

## 6. Metrics and What They Actually Measure

42. Walk through the exact definition of Counterfactual Validity (`counterfactual_validity` in `metrics.py`) — why does it require both "moved by τ=0.8" AND "crossed the boundary," rather than either alone?
43. Why did you move FID's feature extractor from ImageNet Inception-v3 to TorchXRayVision DenseNet-121 features, and what normalization bug did you find and fix in the Inception path?
44. What numerical instability does adding `eps * I` to the FID covariance matrices before `sqrtm` fix, and why does it occur at your sample sizes?
45. What happened to the original FOP (Foreign-Object Preservation) metric across its three implementation attempts (raw YOLO L2 norm, full YOLO detection head, XRV Support-Devices head), and why is it ultimately reported as "not computable"?
46. What is Identity Preservation (MS-SSIM + HF-corr), and why was it introduced as a substitute for FOP rather than just leaving FOP blank?
47. Explain "edit concentration" — why does a uniform edit score exactly `top_frac` (0.05), and what did your v1 counterfactuals actually score (0.282 n→p / 0.297 p→n) versus what a semantic edit "should" score?
48. What is Lesion Hit Rate (LHR), and why is it "the metric that matters" compared to CV, FID, CTR, or SCP? Why is LHR only defined for the p→n direction and not n→p?
49. Your SCP proxy is described as a "weak instrument." What is it actually measuring, and why do its statistically significant deltas (p=0.0042) not mean anything clinically?
50. Why do you compute CTR and SCP with a paired t-test on the within-image delta rather than comparing group means independently?

## 7. Comparison to Prior Work

51. Your evaluation follows Singla et al. (2022) exactly — CV, FID, FOP, CTR, SCP definitions. What specifically did you inherit versus modify from that protocol?
52. How does your pipeline compare architecturally to GANterfactual (Mertes et al. 2022) — conditional GAN vs. your gradient-guided latent diffusion + PPO? What does PPO buy you that a fixed guidance schedule wouldn't?
53. Your final comparison shows the 2-model ensemble default beating both an unguided Vanilla CycleGAN (CV 0.385, FID 55/45) and GANterfactual (FID 4-6x worse than your default) under the identical protocol. What does the CycleGAN comparison specifically isolate — what is it missing that your pipeline has?
54. GANterfactual reaches marginally higher CV in the easier n→p direction (0.998 vs 0.980) but trails you in p→n (0.872 vs 0.950). What does that direction-asymmetry suggest about the two approaches' relative strengths?
55. Why wasn't a baseline comparison performed until late in the project, and how do you defend the earlier chapters' "no baseline" limitation given it's later resolved?

## 8. Validation Integrity — the Central Finding

56. Explain the held-out validation result precisely: guidance CV was 0.995/0.885, but `-chex` held-out CV was 0.080/0.000 and `-nih` was 0.000/0.000. Walk through why this is damning rather than just "the model is imperfect."
57. What is the "agreement control," and why does ~70% agreement on unedited images make the held-out result conclusive rather than just "these validators disagree with the guidance model in general"?
58. What is "transfer ratio," and how do the numbers (0.205/0.590 n→p vs. 0.132/0.143 p→n) support the claim that "p→n was fooling f(·), not removing disease"?
59. You found and fixed a "denominator bug" in held-out validation that inflated every previously published transfer ratio (NIH n→p: 0.117 → corrected 0.085). Explain what the bug was and why it wasn't caught immediately.
60. Explain the shard-routing bug that would have made every control arm's output "silently fall back to the trained arm's shards." How did you catch it, and what would the false result have looked like if you hadn't?
61. Describe the "comma-truncated ensemble roster" bug — how did SLURM env-var passing silently reduce a 3-member ensemble to 1 member, and what statistical coincidence (CV matching to 4 decimal places) tipped you off?
62. Walk through the hand-altered FID values you found in your own recorded results (103.83 → the file showed 13.83, an exact −90.0 offset preserving all mantissa digits). How did you conclude it was a manual edit rather than a computation bug, and what does this teach about trusting your own historical artifacts?
63. What is `test_invariants.py` actually checking, and why does a CPU-only, ~1-minute test suite matter for a project whose real compute runs on SLURM?
64. Why did you build a persisted, patient-level train/test split (`splits.py`) partway through the project, and what was wrong with every headline number computed before that existed?

## 9. Failure Modes and Limitations

65. If a reviewer says "your CV of 0.975 is basically meaningless, you just showed your generator can fool its own guidance classifier" — is that a fair characterization of your full body of results, or only of the pre-ensemble numbers?
66. The `pres_suppression_floor` (0.6) in `environment.step()` never actually binds given `guidance_bias`'s real range (0.70, 1.30). Why leave dead code / an inert safety margin in a shipped config rather than removing it?
67. Both directional policies saturate `guidance_bias` at opposite extremes of its range with near-zero variance. What does that tell you about whether the *policy* or the *action bound* is the binding constraint, and why does that matter for interpreting the guidance-diagnostics numbers as "the policy learned a strategy"?
68. FID sample sizes are N≈200 against a 1024-d feature space. Why is that a real problem and not just "more data would be nice" — what specifically breaks about the sqrtm computation and the confidence you can put on the number?
69. You describe the compute budget (N_EPISODES=800, N_DENOISE_EPOCHS=30) as "conservative placeholders sized for the available walltime, not values established by ablation." What would change about your conclusions if you doubled the budget and the metrics moved?
70. The sparsity reward was designed to concentrate edits but instead measurably hurt lesion targeting (d=0.512, p=4.7e-9) without achieving its stated goal. What's your best hypothesis for why a mechanism that should mechanically reduce diffuse editing made localization worse?

## 10. Infrastructure / Engineering

71. Walk through your SLURM setup: partition, GPU constraint string, and why you `--exclude` node `hpc-n983` specifically.
72. What does `preflight_gpu.py` check, and what class of bug does it prevent that a bare CUDA error message wouldn't make obvious?
73. Explain the resumable-checkpoint design in `PPOTrainer.train()` — why is `cv_pre_ramp`/`lambda_frozen`/`ramp_history` saved alongside the policy weights, and what would silently go wrong on resume if they weren't?
74. Why does `PPO_WALLCLOCK_BUDGET_S` use the SLOWEST update seen rather than the mean to estimate whether another update fits in the remaining budget?
75. Describe the concurrent-write race between two directional training processes and how you fixed it. Why didn't atomic rename alone protect you?
76. What made job 17653310's walltime kill (SIGKILLed ~30/100 updates into training with no checkpoint saved) a turning point for your infrastructure design?

## 11. Ethical / Clinical Implications

77. This system generates synthetic pathology on real patient anatomy. What safeguards, if any, exist against a generated "counterfactual" being mistaken for or used as an actual diagnostic image?
78. Given the held-out validation finding — that a system reporting near-perfect self-validated performance was substantially adversarial rather than clinically meaningful — what does that imply about trusting *any* single-model-validated claim in clinical ML more broadly?
79. If this pipeline were deployed as a clinician-facing explanation tool today (not as a research artifact), what would you refuse to ship, and why?

## 12. Devil's-Advocate / Pressure Questions

80. "You spent half the internship building infrastructure to catch your own bugs instead of improving the model. Wasn't that wasted effort?" — defend the time allocation.
81. "Your headline metric (CV) is circular by construction — you're using the same classifier family (or a superset) for guidance and evaluation even in the ensemble-guided setup. Isn't `-chex` held-out validation itself compromised once `-chex` joins the guidance ensemble?"
82. "PPO adds enormous complexity — 9-dim action space, GAE, auxiliary heads, sparsity annealing — for what your own report shows is close to a no-op reward-scaling effect under advantage normalization. Why not a simpler fixed guidance schedule with a hyperparameter sweep?"
83. "You tried three countermeasures (ensemble, Grad-CAM, sparsity) and two made things worse. Doesn't that suggest the underlying generation mechanism is fundamentally broken rather than fixable with better guidance signals?"
84. "Cohen's d=0.15–0.22 for the one countermeasure that helped (ensemble guidance) is a small effect. Is 'better than the alternatives, all of which are worse or unproven' really a strong enough result to call this default 'validated'?"
