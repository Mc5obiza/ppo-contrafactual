# Answer Key
## Companion to `technical_interview_questions.md`

**Repository:** `ppo-contrafactual/`
**Purpose:** model answers, with every factual claim traced to a file, a line, a log, or an arithmetic derivation you can redo in front of the candidate.

### How to use this document

Each answer has up to four parts:

- **Model answer** — what a top candidate says.
- **Ground truth** — the artefact that settles it (file:line, JSON field, log line, or a derivation). Use this to adjudicate, not your memory.
- **Grading notes** — how to score partial answers, and what a weaker-but-acceptable version looks like.
- **If they push back** — where the candidate may be right and the question paper wrong.

Questions marked 🔒 **can only be answered by the candidate** (personal history, hours spent, what surprised them). For those, this key supplies the *evidence base* the interviewer should check the answer against — not an answer.

---

## Corrections to the question paper

Four claims in `technical_interview_questions.md` are wrong or overstated. I verified them against the source while writing this key. **Do not press the candidate on these — they're right and the paper isn't.**

1. **Q4.3 follow-up, "`td_targets` crosses episode boundaries."** It does not. The expression is
   `t.reward + GAMMA * (0.0 if t.done else next_vals[i])` — `t.done` zeroes the bootstrap at every episode boundary, and under `FIXED_EPISODE_LENGTH` `done` is `True` on the last step of each episode. The `min(i+1, len-1)` clamp only affects the final transition, which is itself `done` and therefore zeroed. **No bug.** If a candidate defends the code here, credit them.

2. **Deep Q24, "renormalised 4-scale MS-SSIM."** At `IMG_SIZE=256` with `window_size=11`, `max_levels = floor(log2(256/11)) + 1 = floor(4.54) + 1 = 5`, so `levels = min(5, 5) = 5`. The full 5-scale weight vector is used and already sums to 1.0001, so renormalisation is a no-op. The adaptive branch only bites below 176 px. Ask the question as "when *would* the renormalisation bite, and would that number be comparable?"

3. **Deep Q27 / §1.9, "the SCP sigmoid is saturated."** This is the *codebase's* diagnosis (`metrics.py:1188-1194`) and it is wrong. Inverting the measured mean: `sigmoid(x) = 0.4794 → x = −0.0826 → ‖feat‖ = 4.13`. At 0.479 the sigmoid is at its **steepest** point (derivative 0.2496), not saturated. Working the std back: `σ_x = 0.000797 / 0.2496 = 0.00319 → σ_‖feat‖ = 0.160`. So YOLO feature norms on the lower third are **4.13 ± 0.16** — a 3.9 % relative spread — and the `/50` divisor then compresses that to ±0.0032 in logit space. The metric is degenerate because **the input barely varies and the divisor crushes what variation exists**, not because of saturation. A candidate who says this has out-reasoned their own code comment; that is a 5.

4. **Q5.1 / §1.15, the reward-split log line.** The paper is agnostic about which direction it came from. It is **`p_to_n`**, provable two ways without reading the filename: `bud = −0.100` exactly equals `λ_bud × 1.0 × 1.0`, so `direction_scale = 1.0`, so `c ≠ 1.0`; and `fid = −0.029 = 2.0 × 0.0144 = λ_fid × edit_L1` with no 2.25 factor. Confirmed by `logs/ppo-counterfactual-17662452_1.out:3` (`array task 1, direction p_to_n`). This matters — several answers below depend on it.

---

## Three findings that are not in the question paper

Discovered while verifying this key. All three are strong senior-level probes.

**F1 — The CV safeguard has never run.** The ramp window resolves to updates `[20, 60)` (logged verbatim: `sparsity reward ON: target=2.0, ramp linear over updates [20, 60); 100 total updates`). `_cv_probe_due` fires only for `10 ≤ update_idx ≤ 60` at multiples of 10. Both retained training logs are from *resumed* processes that re-entered at update 68 (`p_to_n`) and update 88 (`n_to_p`) — past the window. No log in `logs/` contains `CV baseline`, `CV probe at update`, or `RAMP FROZEN`. So roughly 200 lines of safeguard machinery — `_measure_holdout_cv`, `_apply_cv_safeguard`, `_cv_probe_due`, the latch semantics, the persisted `cv_pre_ramp`/`lambda_frozen`/`ramp_history` state — have no evidence of ever having executed, and `ramp_history` is presumably empty in both `ppo_latest.pt`. Ask: *"Show me a CV probe firing."*

**F2 — The far-pool scan is not reproducible across nodes.** Across the 8 eval shards, the same deterministic scan over the same dataset with the same classifier produced **three different far-pool sizes**:
```
4 shards:  15240 train (n→p) / 11396 (p→n)
3 shards:  15243        / 11393
1 shard:   15239        / 11397
```
A spread of 4 images. `run_eval_cf.slurm` allows `gpuh200|gpuh100|gpua100|gpul40s|gpua40|gpurtx6000`, so shards land on different silicon and ~4 images sit within float noise of the `|f(x) − c| > 0.5` boundary. The `scan_groups` "DETERMINISM CONTRACT" docstring guarantees *dataset order*, which does hold — but it says nothing about *threshold decisions*, and `current_cfg_stamp()` compares config, not the derived Xn/Xp lists. So the shard-consistency check would not catch this. The 0.2/0.8 thresholds used by `scan_groups` are further into the tails than 0.5, so Xn/Xp are probably stable — but that is an assumption nothing in the repo tests.

**F3 — The Grad-CAM degeneracy rate is process-local, not run-local.** `[ppo] Grad-CAM mask: 257/257 episodes constrained` appears at the end of an 800-episode run. 257 = the 256 episodes of updates 69–100 plus the one trailing `collect_rollout()`. `_cam_episodes_total` lives on the env, which is rebuilt on every resubmission, so the reported rate covers only the *last* process. The `p_to_n` run resumed from update 68 (544 episodes already done) — so the degeneracy rate for 68 % of that run is unrecorded. Since the whole point of the counter is to certify a clean `+gradcam_mask` ablation arm, this undermines it. Same for `_cam_partial_degenerate` on the generator.

---

# Section 1 — High-Level Understanding

## A1.1 — Input image → counterfactual, and where gradients live

**Model answer**

Pixel path:
```
DICOM (variable size, 12–16 bit)
  → RescaleSlope/Intercept, per-image min–max → [0,1] float32   ingestion.py:132-143
  → uint8 → PIL "L" → Resize(256) → ToTensor()                  ingestion.py:84-87
  → repeat(1,3,1,1)                                             generator.py:121
  → vae.encode(2x − 1).latent_dist.sample() × 0.18215           generator.py:130-135
  → z ∈ (1,4,32,32), flattened to 4096
  → add_noise at t = timesteps[start_idx], start_idx = 50(1−strength)
  → DDIM loop over 50·strength steps                            generator.py:522-604
  → vae.decode(z / 0.18215) → clamp(−1,1) → (x+1)/2
  → 0.299R + 0.587G + 0.114B  →  (1,1,256,256) float32          generator.py:122-124, 145
```
Inside each DDIM step, past the guidance ramp: decode `z0_hat` → classifier ensemble (2× DenseNet-121, each resizing to 224 and rescaling to `x·2048 − 1024`) → BCE; and YOLOv8n backbone (first 10 layers, pooled) → MSE; two `autograd.grad` calls w.r.t. `z0_hat`; masked linear combination added to `z_t`; then the background composite.

**Gradients.** Only w.r.t. `z0_hat`, and only through the VAE *decoder*, the classifier, and the YOLO backbone. Every one of those has `requires_grad_(False)` on its parameters (`generator.py:76-77, 84-85`; `xrv_classifier.py:64-65`; `models.py:127-128`). No weight anywhere is updated during generation. LoRA weights had gradients in Step 4 only. The PPO policy has gradients w.r.t. its own ~198 k parameters, computed from rewards — never from the image.

That whole procedure is **one pass**. `env.step` chains 2–4 passes (`ENV_REFINEMENT_PASSES=2` plus two policy-controlled bits), and an episode is 8 steps.

**Ground truth** — `generator.py:375-608`; `environment.py:499` (`n_passes = 2 + extra_pass_1 + extra_pass_2`); `config.BUDGET_T = 8`.

**Grading notes**
- 5: gets the double `_to_rgb`/`_to_gray` asymmetry (encode replicates one channel to three, decode collapses three to one with *luminance* weights — not the inverse operation, so the round trip is lossy even before quantisation).
- 4: correct path, misses the asymmetry.
- 3: describes files rather than tensors.
- 2 or below: says PPO denoises, or thinks the classifier is trained.

---

## A1.2 — Two audiences

**Model answer**

*Radiologist (no jargon, ~45 s):* "The model called this film positive for pneumonia. This tool produces a second version of the same film — same patient, same anatomy — that the model calls negative, changing as little as it can. Subtracting the two shows you exactly which region the model was relying on. If that region is a real opacity, the model is reading the film the way you would. If it's the shoulder or the image border, it isn't, and you've learned something more useful than a heat map."

*Reviewer (~90 s):* Frame as constrained optimisation over the sampling trajectory of a frozen latent diffusion model. A 9-d continuous policy reparameterises a guided-DDIM sampler per step; the diffusion prior is SD1.5 with LoRA-adapted attention and feed-forward blocks; spatial locality is enforced structurally (a lung-segmentation mask intersected with a Grad-CAM edit budget, applied as RePaint-style latent blending) rather than penalised post hoc. Evaluation follows Singla et al. 2022 (CV/FID/FOP), extended with a model-free identity-preservation pair and an off-guidance transfer test. **Then volunteer the weakness before the reviewer finds it:** CV is computed against the same classifier that supplied the guidance gradient, so it is circular; the off-guidance transfer ratio is the number that actually matters, and on p→n it is 0.13–0.14, i.e. adversarial.

**Grading notes** A candidate who volunteers the circularity unprompted in the reviewer version has essentially answered Q9.1 already — credit both.

---

## A1.3 🔒 — What did you personally build?

**Evidence base for adjudication** (the answer itself must come from the candidate):

- **Traceable to Singla et al. 2022**: CV (Eq. 11), FID (Eq. 10), FOP (Eq. 12), the Xn/Xp protocol at 0.2/0.8, τ=0.8, the CTR and SCP clinical proxies. `metrics.py:1-31` says so explicitly.
- **Candidate-original by necessity** (no paper prescribes these): the FOP→IP substitution (`metrics.py:563-587` — the paper's Fast R-CNN was trained on private data and never released); `edit_concentration` and its synthetic calibration; the SCP degeneracy detector; `DegenerateCAMError` and its whole argument; the shard `current_cfg_stamp()` contract; `PPO_WALLCLOCK_BUDGET_S`; `generate_via_policy` (without which eval never reflected the trained policy at all).
- **Ambiguous / likely supervisor-directed**: the two-directional-policy structure, the 9-d action space, the reward term list, the Grad-CAM constraint concept.
- **Inherited/unowned**: `src/models.py:166-210` `build_generator` is dead code that imports from a non-existent `ppo_counterfactual` module and calls a `load_lora_weights` method that does not exist on the class, with a docstring saying *"ADJUST if your actual method name/signature differs — this is a best guess."* Nothing imports it. That function is a strong tell for provenance — ask about it directly.

**Red flag** Claims authorship of everything, *and* cannot explain `models.py:166`.

---

## A1.4 — Two policies vs one conditional policy

**Model answer**

`MedicalCounterfactualEnv.__init__` takes a scalar `target_c` and immediately builds a far-index pool against it (`environment.py:153`), which is a full-dataset classifier scan. A conditional policy needs randomised `target_c` per episode, which means both pools resident and a sampling scheme across them — a real refactor of the env's initialisation, not a network change.

**The observation already carries `c`** (`environment.py:345`: `self.c_tensor.flatten()` is the third of the three scalars), and the conv trunk feeds all three scalars into `head` (`ppo.py:223, 232`). So the *policy* is already conditional-ready; the *environment* is the blocker. That asymmetry is the precise answer.

Costs of the split: no representation sharing (the hard direction can't borrow from the easy one — named as a limitation in `report.tex`), 2× training compute, and two sets of hyperparameters to keep in sync. Benefit beyond expedience: the directions genuinely behave differently, so specialisation is not obviously wrong.

**Ground truth** — the far pools are near-complementary: 15 267 images with `f < 0.5` (n→p) and 11 417 with `f > 0.5` (p→n), summing to exactly 26 684. So a conditional policy would see the whole dataset rather than 57 %/43 % of it.

**Grading notes** 5 requires spotting that `c` is already in the observation. 3 for the engineering argument alone.

---

# Section 2 — Original Research Problem

## A2.1 — What diffusion buys, specifically here

**Model answer**

Three arguments, in decreasing strength:

1. **Iterativity is what makes the RL formulation coherent.** A conditional GAN is a single forward map `x → G(x, c)`; there is one place to inject guidance and no notion of partial progress. A diffusion sampler is a 50-step trajectory with a well-defined intermediate estimate of the clean image (`pred_original_sample`) at every step, which you can decode and differentiate a *clean-data* classifier through. That gives 50 intervention points per pass, and a genuine sequential decision problem for a policy to control. Without it there is no MDP — which is exactly what the `avg_ep_len = 1.00` collapse demonstrated when the action space let one step do everything.

2. **SDEdit gives a principled edit-magnitude knob.** `strength` selects how far back up the noise schedule to jump, which is a monotone, interpretable control over how much of the original is destroyed. That is *the* counterfactual trade-off — validity against fidelity — exposed as a single scalar the policy can learn to set per image. A GAN has no analogue; you'd tune a latent-space step size with no such semantics.

3. **Data efficiency.** SD1.5 + LoRA (1.6 M → 6 M trainable) on 26 684 images, versus training a CXR GAN from scratch. Plus no mode collapse and no adversarial training instability.

**Grading notes** Argument 1 is the one that matters; a candidate who only gives 3 has given a generic answer. Argument 2 is a good sign — it shows they know *why* `strength` is action dim 0.

---

## A2.2 — What was given up

**Model answer**

1. **Inference cost, catastrophically.** ~4.4 min/image measured (`eval_main.py:27-30`). 400 images ≈ 29 h against a hard 24 h partition cap — *no* walltime value makes single-process eval fit. This forced 8-way sharding, a generate/reduce split, per-image seeding, and a shard-consistency protocol: roughly 250 lines of infrastructure that exist purely because generation is slow.
2. **Stochasticity.** `encoder()` uses `latent_dist.sample()` (`generator.py:134`), and the background lock draws `randn_like` every step (`generator.py:596`). Counterfactuals are not reproducible without explicit per-image seeding (`inference.py:130`).
3. **Domain mismatch.** SD1.5: 512² RGB natural images, CLIP text conditioning, 64×64 latents. Here: 256² grayscale CXR, a randomly-initialised 65→768 MLP producing one token, 32×32 latents.
4. **No adversarial realism term.** Nothing pushes outputs onto the CXR manifold. A GAN discriminator does exactly that. This shows up as FID 103.8 at MAE 0.036 — a tiny pixel change producing a huge feature-space displacement, which is the signature of an off-manifold edit (`metrics.py:1107-1113`).

**Grading notes** 4 is the sharpest and the least likely to be volunteered.

---

## A2.3 — The baseline table

**The situation, stated factually.** `report.tex:508-521` presents:

| Method | CV n→p | CV p→n | FID | FID |
|---|---|---|---|---|
| Vanilla CycleGAN | 0.424 | 0.346 | 55.0 | 45.0 |
| GANterfactual | 0.998 | 0.872 | 25.0 | 20.0 |

and `report.tex:692-701` claims both were "evaluated under the same Xn/Xp splitting, n_eval ≤ 200-per-group protocol … with CV computed at the same τ = 0.8 threshold and FID computed against the same XRV-DenseNet feature extractor."

`grep -rn "cyclegan\|CycleGAN\|GANterfactual" --include=*.py .` returns **two hits, both comments** in `metrics.py:357-358` about the paper's FID normalisation convention. There is no generator, no training script, no checkpoint, no log, and no `eval_metrics.json` variant containing baseline rows. The round numbers (55.0, 45.0, 25.0, 20.0) are also not what an empirical FID looks like.

**Acceptable answers**, in descending order:
- "Those are transcribed from Singla et al. and the protocol paragraph overclaims. It should say 'as reported in the original paper, under their protocol, not directly comparable.' I'll fix it." — **fully acceptable**; this is a citation-framing error, not fabrication.
- "I ran them in a separate repo — here it is." — acceptable if producible.
- "They're placeholders I wrote to structure the table and never replaced. That's a mistake and it needs to come out." — acceptable and honest.

**Unacceptable:** maintaining that they were run under the identical protocol while unable to point to any artefact.

**Follow-up worth asking regardless:** even if the numbers came from the paper, Singla et al. report FID against **Inception-v3**, whereas this repo's default is a 1024-d XRV DenseNet (`metrics.py:934` `fid_backbone="xrv"`). Those two numbers are not on the same scale and cannot share a table column. A candidate who spots that has recovered most of the credit.

**Grading note** Score this on *how they handle being shown contradicting evidence from their own repo*, not on whether the error exists. Record it on the honesty flag in the rubric.

---

# Section 3 — Diffusion

## A3.1 — What `strength` does

**Model answer**

```python
full_timesteps = self.scheduler.timesteps            # 50 entries, descending
n_keep    = max(1, int(len(full_timesteps) * strength))
start_idx = len(full_timesteps) - n_keep
timesteps = full_timesteps[start_idx:]
init_noise = torch.randn_like(z0_orig_4d)
z_t = self.scheduler.add_noise(z0_orig_4d, init_noise,
                               full_timesteps[start_idx:start_idx+1])
```
`strength` is **the fraction of the 50-step DDIM schedule actually executed**. Because `timesteps` is descending, taking the tail `[start_idx:]` keeps the *low-noise* end, and the trajectory starts at `full_timesteps[start_idx]`. Higher strength ⇒ larger starting timestep ⇒ larger `ᾱ`-weighted noise injected by `add_noise` ⇒ more of `z0`'s information destroyed ⇒ the denoiser has more freedom, so the output drifts further from the input.

It is **not** a blend weight and **not** a guidance scale. It also sets the compute: `strength = 0.7` → 35 UNet evaluations, `0.2` → 10. So dim 0 controls edit magnitude and cost simultaneously — the policy's cheapest lever and its most expensive one are the same knob.

**Follow-ups**
- *At strength = 1.0*: `start_idx = 0`, so you start at pure noise and the SDEdit initialisation carries no information about the input. The output would be an unconditional sample, and the *only* thing tying it to the patient would be the background lock and the single conditioning token. The env caps `pass_strength` at `min(0.9, …)` (`environment.py:518-521`) precisely to keep the init informative.
- *Why bound to (0.2, 0.7)*: below ~0.2 there are too few steps for guidance to act at all (and `ramp_start_frac` up to 0.5 would leave ≤ 5 guided steps); above ~0.7 identity preservation collapses. The bound is a prior on the useful region, which is legitimate action-space design.

---

## A3.2 — Guidance taken w.r.t. `z0_hat`, applied to `z_t`

**Model answer**

Two separate things are going on, and a strong candidate separates them.

**(a) Guiding on the decoded `z0_hat` is correct and deliberate.** Classical classifier guidance (Dhariwal & Nichol) perturbs the score, `ε̂ ← ε̂ − s√(1−ᾱ_t) ∇_{z_t} log p(y|z_t)`, and requires a classifier trained on *noisy* inputs at every timestep. Here `f` is a clean-image DenseNet; evaluating it at `z_t` would be meaningless. So the code uses the reconstruction-guidance / "universal guidance" trick: form the model's current estimate of the clean sample, decode it, evaluate the clean-data classifier there. That is standard and right.

**(b) Adding the gradient directly to `prev_sample` is an approximation.** Having computed `g = ∇_{z0_hat} L`, the chain rule to `z_{t−1}` requires `∂z_{t−1}/∂z0_hat`. For DDIM with η = 0,

```
z_{t-1} = √ᾱ_{t-1} · ẑ0 + √(1 − ᾱ_{t-1}) · ε̂
```
and if you hold `ε̂` fixed, `∂z_{t−1}/∂ẑ0 = √ᾱ_{t−1}·I`. So the correctly-scaled update is `√ᾱ_{t−1}·g`, not `g`. The code omits `√ᾱ_{t−1}`, which ranges from ≈0.05 at high noise to ≈0.99 at low noise on SD1.5's schedule — a **~20× swing** in effective guidance strength across a trajectory, monotonically increasing as denoising proceeds.

**Why it doesn't visibly explode.** The linear `guidance_weight` ramp (0 at `ramp_start_idx`, 1 at the end) is *also* monotonically increasing, so it compounds rather than cancels the omitted factor — guidance ends up very strongly back-loaded. And `cls_scale ∈ (0.3, 1.2)` is learned, so PPO can absorb the *average* magnitude but not its timestep dependence.

**Verdict:** defensible as an approximation folded into a learned scale, but it should be documented as one. Inserting `√ᾱ_{t−1}` and re-tuning `ALPHA_CLS` is a one-line experiment worth running.

**Grading notes**
- 5: separates (a) from (b), names `√ᾱ`, notes the ramp compounds rather than corrects.
- 4: knows the clean-classifier problem and that a Jacobian factor is missing.
- 3: knows guidance needs a noise-aware classifier and that decoding `z0_hat` sidesteps it.
- 2: sees no distinction between guiding at `z_t` and at `z0_hat`.

**On classifier-free guidance:** it would require training the LoRA with class-conditional dropout and running two UNet forwards per step. Given the conditioning is a single token derived largely from the image itself (§A3.4), CFG on `c` alone would be extremely weak — the model barely uses `c`. Worth saying.

---

## A3.3 — The lock, versus 66 % of pixels changed

**Model answer**

The mechanism (`generator.py:592-600`):
```python
if i < n_steps - 1:
    z_bg = scheduler.add_noise(z_anchor_4d, randn_like(...), timesteps[i+1])
else:
    z_bg = z_anchor_4d                       # clean, on the final step
z_t = edit_mask * z_t + (1.0 - edit_mask) * z_bg
```
Outside the mask, the latent is *replaced* by the anchor forward-diffused to the same noise level the trajectory is at — so it follows the identity trajectory and cannot accumulate change. Re-noising to `t_next` rather than `t` is required because `z_t` at this point is already `prev_sample`, i.e. at `t_next`; compositing two different noise levels would produce a visible statistical seam.

**Why 66 % of pixels still move — four compounding reasons, in order of size:**

1. **The mask is soft and floored.** `edit_mask = feather(dilate(sigmoid(lung_logits))) × cam`, and `cam = 0.15 + 0.85·CAM^0.75` (`config.GRADCAM_MASK_FLOOR = 0.15`). So the *minimum* mask value anywhere is 0.15, never 0. Every location retains ≥ 15 % of the free trajectory at every step. Compounded over ~25 steps × ~3 passes × 8 env steps, that is not a small leak. **The lock cannot pin anything, anywhere, by construction, as currently configured.**
2. **Even at mask = 0 the output is the anchor's VAE *reconstruction*, not the anchor pixels.** The final decode passes everything through `vae.decode`, so out-of-mask regions carry SD1.5's VAE round-trip error. `diff_viz.compute_diff` counts `|Δ| > 0.01` as "changed" (`diff_viz.py:71`), and 1/255 ≈ 0.004, so VAE error alone can exceed the threshold on textured regions.
3. **The measurement predates the constraint.** `checkpoints/heldout/heldout_validation.json` stamps `USE_GRADCAM_MASK: false, BUDGET_T: 25`. The 66.3 %/62.3 % figures describe a 25-step, no-CAM configuration.
4. **Latent masking ≠ pixel masking.** One latent cell is 8×8 pixels, and the VAE decoder's receptive field spreads in-mask changes across the boundary regardless of how sharp the latent mask is.

**The control nobody ran:** decode(encode(x)) on an untouched image and measure `frac_changed`. That single number is the floor on everything reported. It takes 30 seconds to compute and would tell you how much of the 66 % is attributable to the *method* at all. Its absence is the honest gap here.

**Grading notes** 5 requires reason 1 (the floor) — it is the one that makes the docstring's "collapses to the VAE round-trip floor by construction" false as configured. 4 for reasons 2+3. Anyone who claims the lock guarantees zero out-of-lung change without qualification: 2.

---

## A3.4 — Single-token conditioning and the training leak

**Model answer**

**The architecture.** `cond_in_dim = 1 + 4·32·32//64 = 65`; `cond_proj = Linear(65,256) → SiLU → Linear(256,768)` ≈ 214 k params; output `.unsqueeze(1)` → shape `(B, 1, 768)`. One token.

**Consequence:** cross-attention computes `softmax(QKᵀ/√d)V` over a sequence of length 1. The softmax of a single logit is identically 1, for every query position, at every layer, always. So cross-attention degenerates to `V` broadcast to all positions — i.e. a **per-layer learned bias vector** that depends on the image only through the 65-d conditioning. `to_q` has no effect on the output whatsoever (its result is annihilated by the constant softmax), and `to_k` likewise. Of the four originally-targeted LoRA modules, **two are provably inert in the cross-attention blocks.** They still matter in self-attention blocks, which share the same module names — so the LoRA isn't wasted, but the config comment's per-module accounting is misleading.

**The training leak.** In `compute_training_losses`:
```python
z0 = self.encoder(x)
psi_x = z0.detach()
...
z_t_4d = self.scheduler.add_noise(z0_4d, noise, timesteps)
cond = self._build_condition(c_scalar, psi_x)
eps_pred = self.unet(z_t_4d, timesteps, encoder_hidden_states=cond)
L_diff = F.mse_loss(eps_pred, noise)
```
The model must predict the noise added to `z0` while being handed a 64-d summary of `z0`. Since `z_t = √ᾱ z0 + √(1−ᾱ) ε`, we get `ε = (z_t − √ᾱ z0)/√(1−ᾱ)` — so *any* information about `z0` is directly information about `ε`. At high `t` where `√ᾱ → 0` the leak is weak; at low `t` where `√(1−ᾱ)` is small, knowing `z0` is nearly equivalent to knowing `ε`, and the leak is severe. The pooling is a 64× reduction (channel-mean over 8×8-latent blocks), so it can't reconstruct `z0` — but it can supply the low-frequency component, which is exactly what dominates at low `t`.

**Is it a bug?** Not straightforwardly, because at *inference* `psi_x` also comes from the input image (`generator.py:463`), so train and test are consistent. The problem is scientific, not functional: `L_diff` is not measuring what you think, and a low `L_diff` does not certify that the LoRA learned CXR structure. The per-class loss tracking in `finetuning.py:307-313` therefore also can't be read the way the comment claims.

**The diagnostic:** shuffle `psi_x` within the batch (`psi_x = psi_x[torch.randperm(B)]`) and re-measure `L_diff`. If it barely moves, the conditioning was carrying nothing and the leak is harmless. If it jumps, the model was leaning on it. Ten lines, one epoch.

**Grading notes**
- 5: identifies the single-token softmax degeneracy *and* the leak's `t`-dependence *and* proposes the shuffle test.
- 4: identifies one of the two clearly.
- 3: notices the conditioning is unusually weak.
- 2: describes `cond_proj` as "the text encoder replacement" with no further analysis.

---

## A3.5 — LoRA rank 16, widened targets

**Model answer**

**The hypothesis** (`config.py:471-481`): p→n is the direction held-out validation found adversarial (transfer 0.132/0.143 vs 0.205/0.590 for n→p). That asymmetry is physically sensible — adding diffuse opacity to a clear lung is a low-frequency additive edit; *removing* an established consolidation requires synthesising plausible parenchyma, vessels and texture underneath, i.e. genuine generative capacity. Reward shaping cannot supply capacity. Hence more adapter.

**The change:** r 8 → 16, targets widened from `{to_q, to_k, to_v, to_out.0}` (≈102 k params/module, 1 594 368 total) to add `ff.net.0.proj` (GEGLU input, ~822 k/module), `ff.net.2` (FF output, ~1 640 k/module, the largest), and `proj_in`/`proj_out` (Conv2d, ~103 k/module). Expected ~6 M trainable. `alpha = r = 16` ⇒ scaling `α/r = 1.0`.

**Deliberately excluded:** ResNet `conv1`/`conv2` (22 modules each, ~922 k/module) — "where local texture lives," and the named escalation if r=16 is insufficient. Excluded because it roughly quadruples trainable params against 26 684 images.

**The `peft` trap:** `target_modules` matches by **dotted suffix**, so `"ff.net.2"` must be fully spelled — a bare `"2"` would match unrelated modules. Verification: `build_generator` calls `print_trainable_parameters()` (`generator.py:677`); if it still reports 1 594 368, no new string matched.

**The confound the candidate should raise unprompted:** `N_DENOISE_EPOCHS` went 15 → 30 *in the same edit*, so capacity and training duration were varied together. The stated reason is defensible (the archived rank-8 checkpoint had reached epoch 30, so comparing r=16@15ep against r=8@30ep would confound capacity with training time in the *other* direction) — but the consequence is that there is no clean rank ablation available either way.

**What breaks:** adapter matrices are shaped `(r, in)` and `(out, r)`, so a rank or target change makes every prior generator artefact unloadable. Both `pipeline.load_generator_weights` (`pipeline.py:153-170`) and `finetuning._load_checkpoint` (`finetuning.py:174-186`) catch the raw shape error and re-raise with an explanation naming `LORA_R`/`LORA_TARGETS` — good engineering, and worth crediting.

**Grading notes** 5 requires the confound *and* the suffix-matching trap. On "why `alpha = r`": with `α/r = 1` the adapter's contribution isn't rescaled when rank changes, so r=8 and r=16 runs are comparable at equal `α/r` — which is the right default when you are deliberately varying rank.

---

## A3.6 — DDIM, and whether determinism actually holds

**Model answer**

**Why DDIM:** (i) 50 steps instead of ~1000, which matters when one episode is hundreds of UNet evaluations; (ii) with η = 0 the reverse process is a deterministic ODE, so the output is a function of the initial noise alone — which is what makes seeded reproducibility possible in principle.

**Why the pipeline doesn't get it.** Two independent noise sources:
1. `encoder()` returns `posterior.latent_dist.sample()` (`generator.py:134`) — a fresh draw from the VAE posterior on **every call**. And `get_observation()` calls `encoder(self.x_t)` every step (`environment.py:340`). So the policy's observation is stochastic even for a fixed `x_t`: `z_t` is `μ + σ·ε` with a new `ε` each time.
2. The background lock draws `torch.randn_like(z_anchor_4d)` at every non-final step (`generator.py:596`).

Hence `generate_batch_with_policy` must call `torch.manual_seed(base_seed + index_offset + i)` per image (`inference.py:130`) to make sharded and serial runs agree.

**What the observation noise costs.** SD1.5's VAE posterior σ is small relative to μ, so this is a low-amplitude perturbation — but it is injected into 4096 of the 73 731 observation dimensions on every step, and it means `V(s)` is being fit to a target that isn't a deterministic function of the state. It also means `deterministic=True` in `get_action_and_value` does **not** make evaluation deterministic: the mean *action* is deterministic given the observation, but the observation isn't.

**Should it be `.mode()`?** For `get_observation`, almost certainly yes — there is no reason to inject sampling noise into an RL state. For the generation path, `.sample()` vs `.mode()` is a real choice (mode is the posterior mean, slightly blurrier), but SDEdit adds far more noise immediately afterwards, so the posterior draw is negligible there. The clean fix: `.mode()` in `get_observation`, leave `generate_counterfactual` alone. That is a two-line change with a testable prediction (lower `L_value`).

**Is the final-step lock noise needed?** No — the `i < n_steps - 1` branch already handles it: on the last step `z_bg = z_anchor_4d` clean. Correct as written.

**Grading notes** 5: catches `latent_dist.sample()` *and* connects it to the observation, not just to generation. This is the single most-missed detail in the diffusion section.

---

# Section 4 — PPO

## A4.1 — The MDP, and whether it is one

**Model answer**

| Component | Value |
|---|---|
| State | `[z_orig(4096), z_t(4096), step/T, d_t, c, mask(65 536)]` = **73 731** |
| Action | 9-d continuous, `a ~ N(μ_θ(s), exp(log_std))`, then sigmoid-squashed **inside the env** |
| Transition | 2–4 chained guided-DDIM passes; stochastic (VAE posterior + lock noise) |
| Reward | 9 weighted terms, computed on `(x_next, x_orig)` |
| Horizon | fixed 8, `done = current_step >= T` only |
| γ | 0.99 |

**Is it Markov?** Not quite, in three specific ways:

1. **`edit_l1` is not in the observation.** It is half of `is_valid` (`edit_l1 < 0.02`) and it is a function of `(x_next, x_orig)` — the policy would have to infer it from `z_orig` and `z_t`, which is possible in principle (both are in the state) but requires the conv trunk to learn a latent-space proxy for pixel-space L1 across a VAE decoder. This is the sharpest observability gap in the design: **the agent is being rewarded on a quantity it cannot cheaply observe.**
2. **The Grad-CAM mask is invisible.** It is fixed per episode and gates every action's effect, so from the policy's perspective the transition kernel changes between episodes for unobservable reasons.
3. **`z_t` is a noisy sample, not a sufficient statistic** for `x_t` (§A3.6).

What *saves* it is the step counter: `step/T` makes the fixed horizon observable, so a stationary policy is well-defined. Without it, the same observation at step 1 and step 7 would demand different actions.

**On the mask being 88.9 % of the observation:** it is `_get_seg_mask(self.x_t)` — recomputed from the *current* image every step, so it is not constant. But the conv trunk immediately does `F.adaptive_avg_pool2d(mask, (32,32))` (`ppo.py:229`), discarding 98.4 % of it. Passing a 32×32 mask through `get_observation` would cut the observation from 73 731 to 9 219 with zero information loss to the current trunk — and it is not done because `get_observation` was deliberately frozen so the MLP↔conv A/B stayed valid and `load_directional_policy` kept reading `obs_dim` off a live `reset()`. Defensible, but it costs a full PSPNet forward at 512² per step for data that is then thrown away.

**Grading notes** 5 requires the `edit_l1` observability gap. It is the most consequential structural issue in the RL formulation and it is not commented anywhere in the repo.

---

## A4.2 — PPO vs DQN vs SAC

**Model answer**

**DQN:** ruled out by the action space — 9 continuous dimensions. Discretising at even 5 levels per dim is 5⁹ ≈ 2 M actions, and the semantics (guidance scales) are genuinely continuous. Not a close call.

**SAC vs PPO — the interesting one.** SAC is off-policy and would reuse every transition many times. That should dominate here: environment steps cost ~3 s each (2.9 passes × ~1 s), and the entire run is 800 episodes × 8 = **6 400 transitions**. On-policy PPO throws each one away after 4 epochs. Sample efficiency is the single most valuable property in this environment, and SAC has it.

**What actually argues for PPO:**
- **Replay-buffer memory.** Observations are 73 731 floats = 295 KB in fp32. A 100 k-transition buffer is **~30 GB** — impossible on a 23 GB RTX 6000 alongside VAE + UNet + 2 DenseNets + PSPNet + YOLO. You would have to store `x_t` (256 KB, no better) or the trunk's 256-d feature (1 KB, but then the encoder can't be updated). That is a real engineering blocker, not a preference.
- Bounded actions handled naturally by squash + clip.
- No target networks, no entropy temperature to tune, fewer failure modes when you get ~5 runs total.
- The clip objective limits per-update policy movement, which matters when the "environment" is a diffusion sampler whose behaviour changes qualitatively with the guidance schedule.

**Honest verdict:** SAC with a compressed observation is probably the better algorithm for this problem, and the memory argument is the honest reason it wasn't used. A candidate who says that scores higher than one who defends PPO on the merits.

**On the batch arithmetic** — see A8.1: 64 transitions per update is far below any standard PPO configuration.

**Was a bandit sufficient?** v1 measured `avg_ep_len = 1.00`, meaning it *was* a contextual bandit in practice. The whole redesign (joint validity, narrowed `cls_scale`, fixed horizon) exists to restore a genuine multi-step problem. Whether that succeeded is testable: if the optimal policy is state-independent, a bandit over the 9-d action would match it. **No such baseline exists** — and `guidance_bias_std ≈ 0.087` on a range of 0.60 means the learned policy varies by ~14 % of its range, which is *some* state dependence but not a lot.

---

## A4.3 — What the value head regresses onto

**Model answer**

```python
shaped_r[t] = tr.reward if tr.tau > 0.5 else tr.reward + config.DELTA_MIX * adv[t]
...
L_value = F.mse_loss(values, shaped_rewards)
```

**The critic is regressed onto a single-step reward, not a return.** Standard PPO uses either the GAE bootstrap target `returns = advantages + values` (computed on `ppo.py:324` and then **never used**) or discounted Monte-Carlo returns. Regressing `V(s_t)` onto `r_t` produces a network that predicts immediate reward.

**Quantify the damage.** With per-step reward ≈ −1.59 and an 8-step horizon at γ = 0.99, the true `V(s_0) ≈ −12.5`, while this critic learns `V ≈ −1.59`. Then
```
δ_t = r_t + γV(s_{t+1}) − V(s_t) ≈ r_t + γ·r_{t+1} − r_t = γ·r_{t+1} ≈ −1.57
```
So every TD residual is dominated by a term that no longer measures "was this step better than expected." Crucially the bias is **roughly constant across steps**, so per-update advantage normalisation removes most of its *mean* — the policy gradient is not catastrophically wrong. What is destroyed is GAE's actual purpose: variance reduction via a learned baseline. A baseline that predicts `r_t` instead of the return-to-go subtracts almost nothing useful, so the advantages are close to raw discounted rewards with a constant offset.

**Corroborating evidence:** `L_value = 1.3598` at update 80 against `avg_return = −12.720`. A critic fitting returns would show an MSE on the order of the return's variance; 1.36 is consistent with fitting a quantity of magnitude ~1.6.

**Two further observations a strong candidate makes:**
- `shaped_r[t]` uses `adv[t]` computed **before** the normalisation on line 330, so it mixes an *unnormalised* advantage into the target — the two are on different scales.
- The change-log's rationale ("the standard place a shaped reward signal is meant to land") is wrong. Potential-based shaping belongs in the **reward**, upstream of GAE, where Ng et al.'s policy-invariance theorem applies. Injecting it into the critic's regression target has no such guarantee.

**The experiment:** `returns` is still computed. Swapping one line and running 100 updates on one direction gives a direct comparison on `frac_valid`. Cheap.

**On `DELTA_MIX` having been dead:** `compute_gae` returned three tensors, `update()` unpacked all three, and `shaped_rewards` was referenced nowhere. So a configured, documented, per-step shaping mechanism had **zero** effect on training for the entire first phase. Every result from that period is a result about a system without it — which is fine, provided nothing was ever attributed to it.

**Grading notes**
- 5: identifies the wrong target, quantifies the bias, notes advantage normalisation absorbs most of it, notes the pre-normalisation `adv` mixing, and rejects the change-log's reasoning.
- 4: identifies the wrong target and knows the standard alternative.
- 3: notices `returns` is unused.
- 2: reads the comment aloud as justification.

**If they push back** — a candidate may argue that with a *fixed* horizon and `done` correctly zeroing the bootstrap, the returns are well-defined and the critic choice is merely suboptimal rather than wrong. That is fair; "wrong" overstates it. Credit the nuance.

---

## A4.4 — Sigmoid squashing outside the distribution

**Model answer**

**The setup:** the policy emits `logits`, samples `a ~ N(logits, σ)`, and `log_prob` is computed on that Gaussian (`ppo.py:271-276`). The env then applies `σ(·)` and an affine map (`environment.py:479-498`). The squash is *outside* the policy.

**(1) The PPO ratio is still valid.** The squash is a fixed deterministic function applied identically at collection and at update, so `π_θ(a|s)` is a consistent density over the pre-squash variable and `exp(logp_new − logp_old)` is a correct importance ratio for the pre-squash action. Unlike SAC, PPO's objective does not need the change-of-variables correction. **A candidate who says the ratio is broken is wrong** — credit anyone who defends it.

**(2) The entropy bonus is measured in the wrong space, and this is the real problem.** `−C_ENTROPY · H[N(μ,σ)] = −0.01 · Σ(½log(2πeσ²))` depends on `σ` only — it is completely insensitive to `μ`. But the *effective* action entropy after squashing depends critically on `μ`: at `|μ| > 3` the sigmoid is flat, so a large pre-squash σ maps to near-zero variance in the action the environment actually sees. The optimiser can therefore satisfy the entropy bonus while the policy becomes deterministic in effect. The correct entropy for the squashed variable is
```
H[σ(a)] = H[a] + E[ log |dσ/da| ] = H[a] + E[ log(σ(a)(1−σ(a))) ]
```
and the second term → −∞ as `|μ|` grows, which is exactly the pressure the current objective lacks.

**(3) This is not hypothetical.** v1 measured `guidance_bias = 1.2976 ± 0.0054` (n→p) and `0.7005 ± 0.0012` (p→n) — the two **endpoints** of its (0.70, 1.30) range — across 20 episodes sampled *with* exploration noise `σ ≈ 0.61`. The squash absorbed all of it. The policy was deterministic, state-independent, and bang-bang.

**(4) The mitigation and its status.** `L_act_sat = mean(relu(|logits| − 3.0)²)` at weight 1e-2 (`ppo.py:779-781`). Beyond `|logit| = 3`, `σ ≈ 0.953`, so the responsive region is untouched and only the dead tail is penalised. It works: `|logit|max = 0.93` at update 80 (well inside the free region, so the hinge is currently *inactive*), and `guidance_bias_std` is now **0.0869** (n→p) and **0.0865** (p→n) — 17× the failed value and comfortably above the 0.05 health threshold `config.py:389` names.

But it treats the symptom. The principled fixes: the Jacobian correction above, or a Beta/truncated-Normal distribution on the bounded support, either of which makes the entropy bonus mean what it says.

**(5) Dims 4 and 5 are worse.** `extra_pass_1 = int(σ(a₄) > 0.5)` is a step function of the action. The reward is piecewise constant in `a₄`, so the policy gradient w.r.t. `a₄` is informative only through the *sampling* distribution straddling the threshold — effectively a REINFORCE estimator on a Bernoulli, with all the variance that implies, while the entropy bonus treats `a₄` as continuous. A Bernoulli head would be the correct construction.

**Grading notes** 5 requires (2) *and* the link to the measured v1 numbers. (5) is a bonus that very few will reach.

---

## A4.5 — The four heads

**Model answer**

| Head | Target | Verdict |
|---|---|---|
| `critic` | `shaped_rewards` | See A4.3 — mis-targeted |
| `feasibility_head` | `BCE(·, (returns > 0).float())` | Weak: `returns = adv + value`, so the label is partly generated by the critic it shares a trunk with. Semi-circular but carries some environment signal via `adv`. |
| `progress_head` | `MSE(·, td_targets)` | Essentially a second value head on a different target. Redundant with `critic`. |
| `stop_head` | `BCE(·, (tau > 0.5).float())` | **Self-referential.** `tau` *is* action dim 3. The head is trained to predict what the policy itself just sampled. It contains zero information about the environment. |

**The `stop_head` in detail.** Its label is a deterministic threshold of the policy's own output, so the best it can do is distil `a₃`'s sign into a separate linear head off the shared trunk. With `LAM_STOP_AUX = 0.1` it acts as a mild auxiliary regulariser — it forces the trunk to encode enough to reconstruct one action dimension — but calling it a "stop head" implies it predicts termination, which under `FIXED_EPISODE_LENGTH = True` never happens.

**None of the three is ever consumed.** Every call site discards them:
```python
action, _log_prob, _value, _feas, _prog, _stop = policy.get_action_and_value(...)
```
(`environment.py:663-665`, `ppo.py:650`, `main.py:136-138`). They influence training only through gradients into the shared trunk.

**What `tau` is actually for**, under a fixed horizon:
1. gates `R_stop` (`environment.py:433`) — but `R_stop` requires `is_valid`, which is 0 % of steps, so it contributes nothing;
2. gates the `DELTA_MIX` blend in `compute_gae`;
3. supplies the `stop_head` label.
So one of the nine action dimensions currently has **no effect on the generated image at all**, and its only live pathway is into a value target that is itself mis-specified. Removing `tau` (8-d action) is a clean simplification with a testable prediction: nothing changes.

**Grading notes** 5: catches the self-reference *and* that the outputs are unused *and* that `tau` is inert. 3: notices the outputs are unused.

---

# Section 5 — Reward Function

## A5.1 — Reading the actual reward split

**The line** (`logs/ppo-counterfactual-17662452_1.out:58-60`, direction **p_to_n**, `direction_scale = 1.0`):
```
Update 80 | episodes=640 | avg_ep_len=8.00 | avg_return=-12.720 | L_ppo=0.0411
          | L_value=1.3598 | std_mean=0.6096 | lam_sparse=2.0000
reward split [flip=0% pres=7%] val=-0.887(52%) close2=-0.536(32%) bud=-0.100(6%)
             pres=-0.064(4%) prog=+0.051(3%) fid=-0.029(2%) sparsity=-0.025(1%)
             hit=+0.000(0%) stop=+0.000(0%)
edit_L1=0.0144 (budget 0.02) | steps score_ok=2% budget_ok=86% valid=0% | |logit|max=0.93
```

**Model answer**

**First, verify it is a true decomposition.** The terms sum to
`−0.887 − 0.536 − 0.100 − 0.064 + 0.051 − 0.029 − 0.025 = −1.590`,
and `−1.590 × 8 steps = −12.72 = avg_return`. Exact. The logging is correct and the numbers can be trusted.

**Now invert each term to recover the raw quantities** (`direction_scale = 1.0`, so `weighted = λ · R`):

| term | weighted | λ | raw | interpretation |
|---|---|---|---|---|
| `bud` | −0.100 | 0.1 | −1.000 | constant, by definition |
| `fid` | −0.029 | 2.0 | −0.0145 | = `edit_L1` ✓ (logged 0.0144) |
| `val` | −0.887 | 2.0 | −0.4435 | **mean `d_next` = 0.444** |
| `close2` | −0.536 | 2.5 | −0.2144 | **mean `d_next²` = 0.214** |
| `pres` | −0.064 | 3.0 | −0.0213 | seg L1 + 0.5·KL |
| `prog` | +0.051 | 1.5 | +0.0340 | mean per-step reduction in `d` |
| `sparsity` | −0.025 | 2.0 | −0.0125 | (1−cam)-weighted L1 |

**Three conclusions that fall straight out:**

1. **The flip incentive is gone.** `hit` and `stop` are identically zero because `is_valid` requires `score_ok ∧ budget_ok` and `score_ok` holds on 2 % of steps. The rebalance targeted 40 % flip / 40 % preservation / 20 % progress (`config.py:229`); it achieved **0 / 7 / 3**, with 84 % of the mass in `val + close2`. It did not miss the target — it inverted it.

2. **Which half of validity binds.** `budget_ok = 86 %`, `score_ok = 2 %`. `edit_L1 = 0.0144` against a budget of 0.02 — the policy is *comfortably* inside the edit budget and nowhere near the classifier target. Since `d_next = 0.444` with `c = 0`, `f(x_cf) ≈ 0.44` on average when it needs to be < 0.08. **The correct fix is to loosen `EPSILON_STOP`, not to touch `EDIT_BUDGET_L1`.** The config's own guidance ("Loosen toward 0.03 if episodes never terminate early") points at the wrong knob for this failure.

3. **`val` and `close2` are the same signal.** Both are monotone functions of `d_next`, contributing 84 % between them. And from `E[d] = 0.4435` and `E[d²] = 0.2144`: `Var(d) = 0.2144 − 0.1967 = 0.0177`, `σ_d = 0.133`. So `d_next` is 0.44 ± 0.13 — tightly clustered, well away from 0.08, and the quadratic term is buying almost nothing over the linear one at that operating point (the quadratic only differentiates strongly near `d → 0`).

**Why `avg_reward` differs across artefacts:** `guidance_diagnostics_p_to_n.json` reports `−1.367` per step against the log's `−1.590`. Different checkpoint (diagnostics run after update 100, log at update 80) and `deterministic=False` sampling in the diagnostics vs the same in rollout — plus `lambda_sparsity` is whatever the env last had. Both are per-step; neither is per-episode.

**Grading notes**
- 5: does the inversion (or enough of it to identify `d_next ≈ 0.44`), identifies the binding constraint as the *score*, and prescribes `EPSILON_STOP`.
- 4: explains why `hit`/`stop` are zero and that the rebalance overshot.
- 3: reads the percentages.
- 2: presents `flip = 0 %` as success because v1 was 93 %.

---

## A5.2 — `R_bud = −1.0`

**Model answer**

`R_bud` is a constant, independent of state and action. Under a fixed 8-step horizon:

- Every trajectory accrues exactly `8 × λ_bud × R_bud = −0.8` (× `direction_scale`).
- In GAE, a constant `c` added to every reward shifts every `δ_t` by `c(1 + γ(…))`, producing an advantage offset that is **the same for every transition in the batch**. Advantage normalisation (`ppo.py:330`) then removes it exactly.
- The critic absorbs the corresponding shift in `V`.

So its contribution to the policy gradient is **zero**, up to the tiny non-uniformity at episode boundaries where `done` truncates the bootstrap.

**What it does do:** it is 6 % of the absolute reward mass, so it inflates the denominator in `_format_reward_split` and *deflates* every other term's reported share by ~6 %. A diagnostic that exists to check whether the rebalance landed is being measurably skewed by a term that has no effect on learning.

**Why it exists:** it is a vestige of the variable-horizon design. When `done = is_valid or step >= T`, a per-step penalty of −1 created genuine pressure to reach validity in fewer steps, because shorter episodes accrued less of it. `FIXED_EPISODE_LENGTH = True` removed the only mechanism through which it could act. Keeping both is incoherent: either delete `R_bud`, or restore early termination.

**Related:** `direction_scale = 2.25` is also uniform and the change-log already concedes it "washes out almost entirely" under advantage normalisation. So the reward has **two** decorative scalars. See Deep Q22 for what a non-vacuous version would look like.

**Grading notes** 5 also notes the diagnostic-skew effect. 2 if they claim it "encourages efficiency."

---

## A5.3 — L1 vs MSE, and redundancy with sparsity

**Model answer**

**Why L1.** For a fixed total change budget, L2 is minimised by spreading it: `Σx²` subject to `Σx = k` is minimised at `x_i = k/n`. So MSE actively *prefers* the diffuse, whole-thorax edit the redesign exists to eliminate. L1's subgradient has constant magnitude regardless of the deviation's size, so it does not discount small changes and it is the sparsity-inducing norm (the Lasso argument, applied per pixel).

**Why 2.0 — and the arithmetic doesn't support the config's story.** `config.py:236-239` claims "L1 is ~25× larger numerically at these magnitudes, so 2.0 is a modest real increase." Check it: `config.py:226` gives MSE ≈ 0.0033 and the measured L1 is 0.0144–0.042. That is a factor of **4.4 to 12.7**, not 25. Taking the log's 0.0144: old contribution `0.5 × 0.0033 = 0.00165`; new contribution `2.0 × 0.0144 = 0.0288`. So the term grew ~17× in weighted magnitude — a substantial re-weighting, not a modest one. It is still only 2 % of the reward mass, so it doesn't matter much in practice, but the config's characterisation is wrong and a candidate who checks it is thinking properly.

**Redundancy with `R_sparsity`.** With `SPARSITY_OUTSIDE_CAM_ONLY = True`:
```
R_fid      = −mean(|Δ|)
R_sparsity = −Σ(w ⊙ |Δ|) / Σw ,   w = 1 − cam ,  cam ∈ [0.15, 1] ⇒ w ∈ [0, 0.85]
```
Both are weighted means of `|Δ|` differing only in the weighting. They are **exactly linearly dependent** iff `w` is spatially constant — which is precisely the degenerate-CAM case that `DegenerateCAMError` exists to detect. So the two terms are independent exactly to the extent the CAM carries localisation signal, and identical exactly when it doesn't.

**Measure the actual overlap from the log:** `R_fid = −0.0145` (plain mean) and `R_sparsity = −0.0125` (w-weighted mean). Ratio 0.862. If edits were uniformly distributed the w-weighted mean would equal the plain mean and the ratio would be 1.0. At 0.862, edit mass is only *slightly* concentrated toward the attended region — a 14 % tilt. So the CAM constraint is doing something, but not much, at this checkpoint.

Combined they are 3 % of the reward mass, against `λ_fid + λ_sparsity = 4.0` — the largest nominal weight pair in the table producing the smallest contribution. That is the core problem with reading this reward: **λ is not influence.**

**On latent-space sparsity:** would need an extra VAE encode per step (`environment.py:390-392` says so). But note the term is computed inside `torch.no_grad()` on an image that has *already* been produced — and `get_observation()` is about to encode `x_t` anyway on the very next line of the step loop. The encode is arguably already free.

**Grading notes** 5: checks the 25× claim and computes the 0.862 concentration ratio. 4: the L1/L2 mechanism plus the redundancy condition.

---

## A5.4 — The YOLO KL term

**Model answer**

A strong candidate does **not** defend it. Four problems:

1. **`softmax` of a feature vector is not a distribution.** `O(x)` is `AdaptiveAvgPool2d(1)` over the first 10 YOLOv8n layers, flattened — arbitrary post-activation reals with no normalisation semantics. Applying softmax across the *channel* axis and calling the result a probability mass function is a category error. The resulting "KL" is dominated by whichever channel has the largest activation (softmax is exponential), so it is closer to a soft-max-channel comparison than a divergence. It is also asymmetric for no principled reason.

2. **Domain mismatch.** COCO-pretrained, never fine-tuned (`models.py:122-129`), applied to chest X-rays. Nothing in that backbone has seen a pacemaker, a chest tube, or lung parenchyma. `metrics.py:518-526` says exactly this about the *same* network when it was used for FOP — and the FOP path was abandoned for that reason while the reward path kept it.

3. **Inconsistency between guidance and reward.** The generator's preservation loss uses `F.mse_loss(O_hat, O_x.detach())` (`generator.py:543`); the environment's reward uses KL-of-softmaxes (`environment.py:414-417`). Same concept, same features, two different functions. The policy's *guidance* optimises one and its *reward* scores the other.

4. **`λ_pres_R = 3.0` is the largest weight and contributes 4 %.** Raw `R_pres = −0.0213`, of which the seg-L1 and the KL are not separately logged — so we cannot even tell which half is doing the work. That is a logging gap worth naming: `_compute_reward` should return `seg_term` and `kl_obj` separately.

**What to replace it with.** `metrics.identity_preservation` already computes MS-SSIM and high-pass correlation — model-free, domain-appropriate, no extra forward pass, and *already written by this candidate* for exactly the "is patient-specific content preserved?" question. Using it as the preservation reward would remove YOLO from the training loop entirely (it is otherwise only used by the degenerate SCP metric), saving a forward pass per step and per guided DDIM step.

**Minimum viable fix if YOLO must stay:** cosine distance, `1 − cos(O_x, O_next)`. Scale-free, symmetric, bounded, no fake probabilities.

**On `clamp(1e-8, 1.0)`:** without it, `log(p/q)` produces `inf`/`NaN` when any softmax output underflows to 0 — which for a softmax over unnormalised features with a large dynamic range is likely, not exceptional. **The need for that clamp is itself evidence the softmax is inappropriate**: a well-formed categorical distribution over a feature vector would not routinely underflow.

**On the nested weights:** `ETA_OBJ = 0.5` inside `R_pres`, then `λ_pres_R = 3.0` outside, so the KL's effective weight is 1.5 and the seg-L1's is 3.0. Two knobs for one ratio, which makes tuning ambiguous. It exists because `ETA_OBJ` is *shared* with the generator's guidance loss (`generator.py:544`), so it isn't purely redundant — but the coupling is undocumented and means changing the reward balance silently changes the guidance.

**Grading notes** 5: the softmax critique *and* the guidance/reward inconsistency *and* the λ≠influence point. 2: defends the KL as principled.

---

## A5.5 — Annealing and the CV freeze

**Model answer**

**Schedule** (`_lambda_for_update`): `λ = 0` for updates `[0, 20)`; linear ramp `0 → 2.0` over `[20, 60)`; `2.0` thereafter. Logged verbatim: `sparsity reward ON: target=2.0, ramp linear over updates [20, 60); 100 total updates`.

Rationale: let the policy find a reliable flip inside the Grad-CAM-constrained action space first, then apply minimality pressure within that space. Applying full sparsity from update 0 would push toward "change nothing," which trivially satisfies the sparsity term and never discovers a flip.

Bounds are **fractions** because `N_EPISODES = 800` is an explicit placeholder; absolute indices would silently land in the wrong place after re-tuning. Resolved once at `train()` entry and logged.

**Safeguard** (`_apply_cv_safeguard`): every 10 updates within `[start − lead_in, end]`, run all 24 held-out far-pool images through `generate_via_policy` deterministically at a shortened budget, compute CV with **the same criterion as `run_full_evaluation`** (τ = 0.8, boundary 0.5 — deliberately not a second definition of validity), and compare to the pre-ramp baseline.

**Why latched:** it freezes at `lambda_last_ok` — the last λ that *passed* a probe — because the current λ is the one that caused the drop. And it never releases on recovery, because an oscillating λ would make the run uninterpretable: you could not attribute a final metric to any particular sparsity weight. The state (`cv_pre_ramp`, `lambda_frozen`, `lambda_last_ok`, `ramp_history`) is persisted in `ppo_latest.pt` because restoring weights alone would re-baseline the probe against a mid-ramp policy and silently release the latch.

**Sizing:** CV over N images has resolution 1/N. At N = 24 that is 4.2 points/image, so an 8-point tolerance is two images' worth — above the noise floor. At N = 8 one image would move CV by 12.5 points and the freeze would latch on noise in essentially every run. This is a real and correct derivation.

**Now the three problems the candidate should raise:**

1. **It has never run.** No log contains `CV baseline`, `CV probe at update`, or `RAMP FROZEN`. Both retained training logs resume at updates 68 and 88 — past the `[10, 60]` probe window. Roughly 200 lines of machinery, unexercised. (See Finding F1.)
2. **Off-by-one between the ramp and the probe.** `lam = self._lambda_for_update(update_count, ...)` is called *before* `update_count += 1`; `_apply_cv_safeguard(update_count, ...)` is called *after*. So the probe labelled "update 10" evaluates a policy trained with the λ for index 9. Cosmetic, but it means `ramp_history` mislabels its own λ.
3. **The holdout is not held out from evaluation.** See Deep Q13 — `_far_indices[:24]` and `scan_groups`'s first 200 are both dataset-order prefixes over overlapping score ranges.

**On the probe budget:** `env.T` is temporarily set to `min(4, 8) = 4`. The claim in `generate_via_policy`'s docstring is that this avoids feeding the policy unseen step-fractions. Check it: with `T = 4` the fractions are `{0.25, 0.5, 0.75, 1.0}`, all of which are in the training set `{k/8}`. So they are in-distribution — but the *correspondence* shifts: at probe step 1 the policy sees 0.25, where in training step 1 meant 0.125. The policy is told it is a quarter of the way through when it has taken one action. That is a subtler form of the same problem, and the docstring's framing understates it.

**Grading notes** 5 requires the latch rationale *and* at least one of the three problems. Finding F1 unprompted is exceptional.

---

# Section 6 — Code Ownership

## A6.1 — `_build_conv_trunk` and the `expected` check

**Model answer**

**Why it exists.** The MLP trunk was `Linear(73731, 512) → ReLU → Linear(512, 256) → ReLU`. First layer alone: `73 731 × 512 + 512 = 37 750 784` parameters, out of a total 37 885 462 — **99.6 %** — fit on roughly 1 000 environment steps at the time. It consumed the observation as an unstructured vector in which the 256² mask is 65 536 of 73 731 entries (88.9 %). No spatial structure, no weight sharing, no inductive bias. That is the direct mechanism for v1's constant, state-independent action: with 37 M parameters and ~10³ samples, the only thing the layer could fit was a bias.

**What the conv trunk does.** Reshapes the *same* flat observation internally: `z_orig → (4,32,32)`, `z_t → (4,32,32)`, `mask → (1,256,256) → adaptive_avg_pool2d → (1,32,32)`, stack to 9 channels → `Conv(9→32,s1) → Conv(32→64,s2) → Conv(64→128,s2) → AdaptiveAvgPool2d(1) → Flatten` → concat the 3 scalars → `Linear(131→256) → Linear(256→256)`.

Parameter count: `2 624 + 18 496 + 73 856 + 33 792 + 65 792` (trunk) `+ 2 313` (actor) `+ 257` (critic) `+ 3×257` (aux) `+ 9` (log_std) ≈ **197 910** — matching the config's "~198k". A 191× reduction.

**Average pooling, not max or threshold**, because the mask holds soft lung *probabilities* from `sigmoid(lung_logits).amax(...)`; thresholding would discard the boundary uncertainty that the dilate/feather machinery elsewhere treats as meaningful.

**Scalars concatenated after the conv, not broadcast as channels**, because they carry episode progress state (`step/T`, `d_t`, `c`) which is not spatial. Broadcasting them as 3 constant channels would let the first conv mix them into every spatial location and then global-average-pool most of that signal away — diluting exactly the three numbers that tell the policy where it is in the episode.

**What `expected` protects against.** `expected = 2·4096 + 3 + 65 536 = 73 731`, recomputed from `config.LATENT_CHANNELS/LATENT_SIZE/IMG_SIZE`. If someone changes `IMG_SIZE` to 512, `obs_dim` becomes 266 243 and the slicing `obs[:, :Z]`, `obs[:, Z:2Z]`, `obs[:, 2Z+3:]` would silently reinterpret the observation — `z_t` would be sliced out of the middle of the mask, producing a network that trains on garbage without erroring. The check turns that into a construction-time `ValueError` naming the three config values to look at.

**Why `"mlp"` is kept:** the archived rank-8 policies under `checkpoints/archive_v1_rank8/` have that trunk and will not load into a conv policy — `_load_state` catches exactly this and re-raises with the `POLICY_TRUNK`/`ACTION_DIM` explanation (`ppo.py:447-463`).

**Would I keep the mask in the observation?** No — it is pooled to 32×32 immediately, so `get_observation` could emit a 32×32 mask and cut the observation from 73 731 to 9 219 with **zero** information loss to the current trunk. It isn't done because `get_observation` was deliberately frozen so the MLP↔conv comparison stays valid and `load_directional_policy` keeps deriving `obs_dim` from a live `reset()`. Defensible, but it costs a PSPNet forward at 512² every step for data that is then discarded.

---

## A6.2 — Why `DegenerateCAMError` needs its own class

**Model answer**

**The failure is silent, and that is the whole point.** A spatially constant CAM fed through `floor + (1 − floor)·cam` produces a **uniform mask at `floor` = 0.15**. Every location gets an identical edit budget. The locality constraint is therefore disabled — but nothing errors, the episode completes, the metrics compute, and the run is reported as a `+gradcam_mask` ablation arm. That is the single most damaging way this can fail: it manufactures a clean-looking result for a mechanism that was not active.

**The cause.** The classifier outputs a probability. Once it rounds to exactly 1.0 (or 0.0) in fp32 — logit magnitude beyond ~17 — `d(prob)/d(activations)` is identically zero, so `weights = grad.mean(dim=(1,2))` is all zeros and `relu(Σ w_c A_c)` has zero span.

**The subtle part, and the discriminator.** Back-propagating `logit(p)` instead of `p` does **not** fix it, for two independent reasons:
1. `d logit(p)/d p = 1/(p(1−p))` is a **spatial constant**, so it multiplies every CAM location equally and cancels under the `cam / cam.max()` normalisation. It cannot change *where* the CAM is high.
2. Once `p` has rounded to 1.0, the information is already destroyed — `logit(clamp(p, ε, 1−ε))` differentiates a constant and yields exactly zero.

Recovering the gradient requires the **pre-sigmoid logit**, i.e. hooking the classifier's final `Linear`. The docstring says this was verified, not assumed.

**Why detect before normalising.** `grad_cam` computes `cam = cam − cam.min()`, then checks `cam.max() <= 0` *before* dividing. If you normalised first you would divide by zero (or by an epsilon) and produce either NaN or an arbitrary map — destroying the evidence that the CAM was degenerate.

**Why skip rather than average degenerate members.** In `ensemble_grad_cam`, a degenerate member's map is all-zeros. Including it in `torch.stack(cams).mean(0)` would halve every value in a 2-member ensemble, dragging the mean toward uniform — i.e. producing exactly the silent-disable failure the exception exists to prevent, but *partially*, which is harder to detect. Skipping preserves the surviving member's localisation at full contrast; only if **every** member is degenerate does it raise.

**Grading note on the observed rates.** One run logs `257/257 constrained, 0 degenerate`; another `95/97, 2 degenerate (2.1%)`. Separately, `generator.py:294` notes partial degeneracy fired on **87/87 episodes** in eval job 17665348 — i.e. one of the two members was degenerate on every single episode, making that eval's "ensemble CAM" effectively single-member. The right threshold for discarding a run is not really a percentage: any non-zero *full* degeneracy rate contaminates the arm, and a *partial* rate near 100 % means the arm should be relabelled. See Finding F3 for why the reported rate covers only the last process.

---

## A6.3 — Two backwards, and whether to trust the comment

**Model answer**

**The claim.** `action = −w·A·∇L_cls − w·B·∇L_pres` where `A = 4.0·1.15·cls_scale` and `B = 0.05·pres_scale` are Python floats known before the backward. In exact arithmetic, `∇(A·L_cls + B·L_pres) = A∇L_cls + B∇L_pres`, so one fused backward gives the identical vector and saves a full VAE-decoder backward — roughly 1.5× on the whole evaluation.

**Why it isn't identical in fp16.** `z0_hat_g` is fp16 (the latent is fp16 from the encoder onward), so the decoder backward runs in fp16. Folding the coefficients into the loss scales the *entire* backward chain before the fp16 intermediates rather than after. With `B = 0.05·pres_scale` and `pres_scale ∈ (0.3, 1.5)`, the effective pre-scale can be as low as 0.015. fp16's smallest normal is 6.10e-5 and its smallest subnormal 5.96e-8; multiplying a gradient chain by 0.015 before it propagates costs ~6 bits of headroom and pushes small intermediate activations into the subnormal range, where relative precision collapses. The comment's measurement — 0.6–4.7 % max relative deviation in `action` against an fp16 eps of ~0.1 % (2⁻¹⁰ = 9.8e-4) — is consistent with that mechanism.

**Do I trust it?** The mechanism is sound and the measurement is specific. But this is the pipeline's dominant cost — two decoder backwards on ~60–80 % of DDIM steps, × 2.9 passes, × 8 steps, × 400 eval images. 1.5× on a 29-hour generation is **~10 hours per eval run**. So accepting the comment permanently is the wrong response; the right one is to do what the comment itself prescribes: move the guidance block to fp32 (or apply a gradient-scaling factor to the preservation branch, e.g. compute `∇(A·L_cls + 1000·B·L_pres)` and divide by 1000 after) and re-validate CV/FID.

**A cheaper fix nobody tried:** raise `ALPHA_PRES` and lower `pres_scale`'s range correspondingly. `ALPHA_PRES = 0.05` and `pres_scale ∈ (0.3, 1.5)` gives an effective range of (0.015, 0.075); the identical range is reachable with `ALPHA_PRES = 1.0` and `pres_scale ∈ (0.015, 0.075)`. The *loss* would then be scaled by ~1.0 and the fp16 headroom problem largely evaporates, at the cost of a differently-shaped action range. Whether that changes the trajectory is exactly the kind of thing the existing measurement harness could test.

**`retain_graph=True`** keeps the autograd graph of the decoder forward + classifier forward + YOLO forward alive between the two `grad` calls. Cost is the stored activations of the SD1.5 VAE decoder at 256² — on the order of a few hundred MB in fp16, which is significant but not the binding constraint on a 23 GB card.

**Grading notes** 5: knows what fp16 subnormals are, identifies this as the dominant cost, and proposes either the fp32 move or a rescaling. 2: cannot say why the two forms differ numerically.

---

## A6.4 — The FOP three-version history

**Model answer**

- **v1** — thresholded `‖YOLO backbone feature‖₂` against `fo_threshold = 0.5`. Feature norms are routinely 10–100 (we can now say: **≈4.13** on the lower third, from the SCP inversion), so `norm > 0.5` was true for essentially every image and FOP was a degenerate **1.0** for both directions and the mean. A threshold-space error: an unbounded L2 norm compared against a probability threshold.
- **v2** — switched to the full YOLO *detection head*, which does output calibrated confidences, fixing the threshold space. But `yolov8n.pt` is COCO-pretrained on 80 everyday-object classes and has never seen a pacemaker, chest tube, central line or ECG lead. Right units, wrong detector.
- **v3 (current)** — replaced YOLO with a TorchXRayVision DenseNet-121 requesting the CheXpert `"Support Devices"` head, which *is* one of CheXpert's 14 original labels. Discovered no shipped xrv weight set actually exposes it — the label exists only in xrv's dataset label-loading code, not in any released classifier's `model.pathologies`. So `_XRVSupportDeviceDetector.available` is `False`, `foreign_object_preservation` returns `None` early, and `_print_results` deliberately omits it (`metrics.py:1272-1274`).

**What replaced it.** `identity_preservation` (`metrics.py:674-706`): MS-SSIM for global anatomical identity across scales, plus Pearson correlation of high-pass-filtered images as a direct proxy for whether thin, high-contrast structures (tubes, lines, leads) survive. Model-free, no downloads, runs anywhere the rest of the eval runs. This is the substitution the candidate should own — the original paper's FOP needed a Fast R-CNN trained on ~300 manually annotated images per object class, private data, 80 % accuracy, never released, so a faithful FOP is simply not computable offline.

**Why keep the `null` field.** Schema stability: `eval_metrics.json` is consumed by the report and by any downstream comparison, and dropping a key silently changes the shape. Defensible. The counter-argument — a `null` in a results file invites someone to read it as "FOP = 0" — is also fair, and the honest resolution is a `"FOP": {"available": false, "reason": "..."}` object.

**On the IP asymmetry.** MS-SSIM 0.849 (n→p) vs 0.796 (p→n); HF-corr 0.938 vs 0.798. So **n→p preserves more on both sub-metrics**, while simultaneously having the *larger* MAE (0.042 vs 0.029). That combination is physically coherent: n→p adds low-frequency diffuse opacity — high L1, high-frequency detail untouched, so HF-corr stays at 0.938. p→n removes/smooths existing consolidation — lower L1, but the smoothing destroys fine texture, so HF-corr drops to 0.798. That is exactly the inpainting-vs-addition asymmetry, read off the metrics.

**Note this contradicts the report.** `report.tex:630-633` states p→n is easier partly because of "higher IP (HF-corr 0.793 vs …)" — but 0.793 is *lower* than the n→p value quoted in the same sentence (0.899), and both differ from the JSON (0.798 / 0.938). The claim is self-contradictory against its own numbers. Worth raising.

---

## A6.5 — Does the suppression floor bind?

**Model answer**

**No, provably.** `guidance_bias = 0.70 + 0.60·σ(a₇)`, and `σ ∈ (0,1)`, so `guidance_bias ∈ (0.70, 1.30)` strictly. Therefore `1/guidance_bias ∈ (0.769, 1.428)`, whose infimum 0.769 is above the floor of 0.6. `torch.clamp(1/guidance_bias, min=0.6)` is the identity on the entire reachable domain.

Same for the per-pass decay: `max(1 − 0.15·pass_idx, 0.4)` with `pass_idx ∈ {0,1,2,3}` gives `{1.0, 0.85, 0.70, 0.55}` — the 0.4 floor is unreachable.

**Empirically confirmed** by the artefact written for exactly this purpose: `pres_suppression_min` = 0.858 (p→n) and 0.830 (n→p) across 160 steps each. Both far above 0.6, and both consistent with the analytic bound.

**The warning is also unreachable.** `run_guidance_diagnostics` prints a warning if `pres_suppression_mean < 0.70` (`main.py:176-180`). Since the *minimum possible value* is 0.769, that branch is dead code. So the diagnostic has a health check that cannot fire — which means "no warning" carries no information, and someone reading the output would wrongly conclude the policy is behaving well.

**What actually constrains preservation suppression** is the action transform itself: `guidance_bias`'s range was chosen to be narrow. If you want a binding floor, either tighten it above the natural minimum (`min=1.0` would prevent `guidance_bias` from reducing `pres_scale` at all) or widen the sigmoid range and re-add a floor that bites. Keeping an inert clamp plus an unreachable warning is worse than having neither, because both read as active safeguards.

**Credit for:** doing the arithmetic rather than guessing; checking the JSON that exists to answer it; and being willing to say "this line does nothing."

---

## A6.6 — Per-image seeding and the direction offset

**Model answer**

**The problem.** `generate_counterfactual` draws `init_noise = torch.randn_like(z0_orig_4d)` and, on every non-final DDIM step, `noise_bg = torch.randn_like(z_anchor_4d)` — both from the **global** RNG. Without reseeding, the noise consumed for image *k* depends on how many random numbers were drawn before it in that process, i.e. on how many images preceded it. So results depend on how the work is chunked, and reruns are irreproducible.

**The fix.** `torch.manual_seed(base_seed + index_offset + i)` before each image makes its counterfactual a function of its **global** index alone. A shard covering `[50, 100)` then produces byte-identical output to a serial run over `[0, 400)` — which is precisely what makes `--stage generate` + `--stage reduce` report the same numbers as `--stage all`.

**The offset.** n→p and p→n operate on *different* images (Xn vs Xp), but with both starting at `index_offset = 0` image *i* of each direction would draw identical noise. That correlates the two halves of the metric suite for no reason — any noise-driven artefact would appear in both, and the FID/CV estimates for the two directions would no longer be independent. `EVAL_PN_SEED_OFFSET = 1_000_000` is simply larger than any realistic `n_eval`, so the ranges cannot collide. `stage_generate` uses `index_offset = n_lo` and `EVAL_PN_SEED_OFFSET + p_lo` (`eval_main.py:257-265`), and `run_full_evaluation`'s in-process path uses `0` and `EVAL_PN_SEED_OFFSET` (`metrics.py:1073-1075`) — they must match or a cached shard won't reproduce.

**The other half of the contract.** Seeding by global index only helps if every process derives the *same* global index for the same image. That requires `scan_groups` to be order-deterministic, which rests on two things: `shuffle=False` in the DataLoader, and `RSNACXRDataset.samples` being built from a `pandas.groupby("patientId")` whose index is sorted by key — so dataset order is a function of patient ID, not of `os.listdir` order.

**The gap a strong candidate finds.** Order-determinism is guaranteed; **threshold-determinism is not**. Across the 8 eval shards the far-pool size came out as 15 240 / 15 243 / 15 239 (n→p) and 11 396 / 11 393 / 11 397 (p→n) — three different answers to the same deterministic scan, because shards land on different GPU models (`--constraint="gpuh200|gpuh100|gpua100|gpul40s|gpua40|gpurtx6000"`) and ~4 images sit within float noise of the `|f(x) − c| > 0.5` boundary. `scan_groups` uses 0.2/0.8, further into the tails, so Xn/Xp are *probably* stable — but nothing tests that, and `current_cfg_stamp()` compares config, not the derived lists. Storing a hash of the Xn/Xp patient IDs in each shard would close it in five lines.

**Was the verification run?** The recipe is documented (`--stage all --n-eval 8` vs two `--num-shards 2` shards + reduce). `checkpoints/smoke_eval/` and `checkpoints/smoke_all/` both exist and both contain an `eval_metrics.json`, which is consistent with someone having done it — ask the candidate to state what the two files showed.

---

# Section 7 — Debugging

## A7.1 🔒 — The worst bug, and how it was found

**Evidence base.** Three candidates, all documented, ranked by severity:

**(a) `DELTA_MIX` was dead code.** `compute_gae` returned `(shaped_rewards, advantages, returns)`; `update()` unpacked all three; `shaped_rewards` appeared nowhere else in the function. So a configured, documented, per-step shaping mechanism had **zero** effect on training for the whole first phase.
*Detection:* reading the unpacking against the usage. Or noticing that `plot_reward_trace` draws flat vs shaped reward and the two curves are visually coincident — which `report.tex:603-606` observes and then **misattributes** to advantage normalisation. The report explains an artefact of dead code as a property of PPO. That misattribution is itself worth probing.
*Blast radius:* nothing was ever attributed to `DELTA_MIX`, so no result is invalidated — but the value of `DELTA_MIX = 0.2` in the hyperparameter table describes a knob that did nothing.

**(b) `direction_scale` partial application.** Applied to `{prog, val, close2, hit, stop}` and not `{fid, pres}`, so preservation was effectively 2.25× less important **specifically on the direction that already struggled most**.
*Detection:* the qualitative diff plots — 81 % of pixels changed with foreign-object corruption on n→p specifically. A direction-specific artefact points at a direction-specific term.
*Blast radius:* every n→p result before the fix.
*Nuance the candidate should add:* the fix is close to a no-op for learning, because a uniform scalar washes out under advantage normalisation. What it removed was the harmful **relative** re-weighting, not a missing signal. The code's change-log says this; the report repeats it correctly (`report.tex:648-657`) — one of the places the report is good.

**(c) Sigmoid saturation.** `guidance_bias` measured at the two endpoints of its range with std 0.005 under sampling noise 0.61.
*Detection:* `run_guidance_diagnostics` was written specifically to catch this, and the **std** was the tell — the mean alone looks like a plausible learned value.
*Blast radius:* v1's policy was effectively a constant, so v1 is not evidence that a *learned* schedule beats a fixed one.

**Grading:** the story must include the detection method. "I found it by reading the code" is acceptable if they can say *what they were looking for*.

---

## A7.2 — `avg_ep_len = 1.00`

**Model answer**

**What it means.** Episodes terminated after one step, so `BUDGET_T`, GAE's multi-step recursion, the value head's bootstrap and all temporal credit assignment were vestigial. PPO was solving a **contextual bandit** dressed as an MDP. Every piece of RL machinery beyond the policy-gradient step was decoration.

**Two independent causes:**
1. **Score-only validity.** `done = is_valid or t >= T` with `is_valid = |f(x_cf) − c| < ε`. A single diffuse wash satisfied it immediately.
2. **Action range too wide.** `cls_scale = 0.5 + 2.5·σ` (max 3.0) meant one step could brute-force the flip.

**Four changes:**
| Change | From → To | Purpose |
|---|---|---|
| Joint validity | score-only → `score_ok ∧ (edit_l1 < 0.02)` | flip must also be *cheap*, forcing refinement |
| `cls_scale` | `0.5+2.5σ` → `0.3+0.9σ` | cap per-step classifier push |
| `BUDGET_T` | 25 → 8 | 25 × 2.5 passes was a 10–25× cost increase once episodes actually ran |
| `FIXED_EPISODE_LENGTH` | absent → `True` | predictable cost; `R_hit`/`R_stop` become per-step |

**Calibration of `EDIT_BUDGET_L1 = 0.02`:** v1 counterfactuals measured MAE 0.042 (n→p) / 0.029 (p→n) over 200 images each, so 0.02 is deliberately below what the old policy produced — reachable, but not by the v1 strategy.

**The honest caveat.** Fixing the horizon was **not** a speed optimisation. Job 17653310 logged `avg_ep_len = 8.00` on every update, i.e. `is_valid` never fired and episodes already ran the full budget. It buys predictability against a hard walltime, not wall clock. It also means `avg_return` is no longer numerically comparable to pre-flag runs (the same trajectory now collects `R_hit`/`R_stop` on more steps), so the comparable quantity is `frac_valid`.

**Has it over-corrected?** Yes. `frac_valid = 0 %`, `score_ok = 2 %`, `budget_ok = 86 %`. The joint criterion went from trivially satisfiable to unsatisfiable, and the binding half is the *score*, not the budget. So the corrective for the corrective is to loosen `EPSILON_STOP` — see A5.1.

**Why not raise `BUDGET_T` instead?** Because `avg_ep_len` pinning at `BUDGET_T` means episodes are ending on the *budget*, not on success; adding steps buys more of the same unsuccessful editing at linear cost. The config says this. But note that reasoning assumed the *edit budget* was binding; with the *score* binding, more steps would in fact let `d_next` fall further — so the config's rule of thumb is right for the wrong failure and should be re-derived.

---

## A7.3 — The three walltime failures

**Model answer**

| Job | Failure | Root cause | Fix built |
|---|---|---|---|
| **17644171** | Both policies saved (22:21, 11:05), eval killed at 24 h, no metrics | train + eval in one process | `main.py` / `eval_main.py` split with `pipeline.build_common()` shared, so eval reloads checkpoints instead of depending on a live process; eval becomes re-runnable |
| **17653310** | SIGKILL at walltime ~30/100 updates into n_to_p, **nothing** recoverable | `save_checkpoints()` only ran at the end of a direction | `PPOTrainer._save_state` every 5 updates (`.tmp` + `os.replace`), carrying optimizer *and* sparsity/CV state; `PPO_WALLCLOCK_BUDGET_S` soft deadline; `train()` returns `bool`; `--direction` splits `run_ppo.slurm` into a 2-task array |
| **17665348** | Eval alone cancelled at 12 h, no metrics | ~4.4 min/image × 400 = ~29 h vs a **24 h partition cap** | 8-way `--stage generate` array + cheap `--stage reduce`; per-global-index seeding; `current_cfg_stamp()` consistency protocol |

**The distinction that matters for 17665348:** this is a *hard partition limit*, not a tuning problem. `sinfo -p publicgpu` caps TIMELIMIT at 24 h and it cannot be raised, so **no** `--time` value makes a single-process eval fit. Recognising that the constraint is structural rather than parametric is what motivates sharding rather than optimisation.

**Why `_PROCESS_START_S` is captured at module import.** Two reasons, both in the comment: (i) Steps 1–4 (dataset scan, model loading, LoRA resume) run before any `train()` call and are charged against the same SLURM `--time`; (ii) `python main.py --direction both` calls `train()` twice — timing from `train()` entry would hand the second direction a fresh full budget, so a 22 h budget would permit ~44 h and the job would still be SIGKILLed. `ppo.py` is imported by `main.py` before any of that work starts, so import time is a good proxy for process start.

**Why the estimate uses `max`, not `mean`.** `slowest_update_s = max(slowest_update_s, …)`. An update carrying a CV probe costs ~3 extra updates of rollout compute, so a mean systematically under-predicts exactly the updates most at risk of overrunning. Using the max makes the guard conservative in the right direction.

**Why writing nothing beats writing a half-trained policy.** `eval_main.py` cannot distinguish a paused `ppo_policy.pt` from a completed one — same filename, same shape, loads fine. A partial artefact would be silently evaluated and reported as a finished result. The resumable state lives in a *differently named* file (`ppo_latest.pt`) carrying optimizer and schedule state that `ppo_policy.pt` deliberately does not, so the two can never be confused.

**Evidence the machinery worked:** `logs/ppo-counterfactual-17662452_1.out:50-51` — `found resumable state at checkpoints/p_to_n/ppo_latest.pt — resuming` / `resumed from update 68 (544 episodes done)`, then running to `update 100, 800 episodes` and printing `✓ TRAINING COMPLETE`. Both directions completed. The resume path is not theoretical.

---

## A7.4 — An unfixed failure

**Model answer** — four are available, and the candidate should pick more than one:

1. **SCP is degenerate and stayed that way.** Relative spread < 2 % across all four groups (std 0.0008 on mean 0.479). The code *detects* it (`scp_degenerate`) and refuses to print the t-tests, which is the correct response to an unfixable proxy — but the values are still in `eval_metrics.json`, and `report.tex:456-458` reports SCP deltas with p-values as evidence "that the edit is not clinically inert." Reporting a t-test on a constant as a finding is the failure; the detector exists and the report ignores it.
2. **FOP is uncomputable offline.** Not fixable without training a detector on annotated data.
3. **p→n never stopped being adversarial by the transfer test.** 0.132 / 0.143, both below the 0.15 threshold the script itself defines as ADVERSARIAL. The ensemble was introduced *because* of this — but **no post-ensemble held-out validation exists on disk**. The fix was applied and never verified.
4. **The ablation matrix was never separated.** All three arms are `True` simultaneously; the only `eval_metrics.json` has all three `False`. See A9.4.

Add, from the log analysis: **the CV safeguard has never executed** (Finding F1).

**Grading:** "nothing failed" is a 1. Naming only (2) is a 2 — it is the failure least attributable to the candidate.

---

# Section 8 — Hyperparameters

## A8.1 — The size of one PPO update

**Model answer**

```
episodes_per_update = 8   (main.py's trainer.train default)
BUDGET_T            = 8   (fixed, avg_ep_len = 8.00 confirmed in logs)
⇒ transitions per update = 64
n_epochs            = 4   (update()'s default, no minibatching)
N_EPISODES          = 800 ⇒ 100 updates ⇒ 6 400 transitions total
```

**Why 64 is alarming.** Reference PPO configurations use 2 048–4 096 steps per update, split into minibatches of 64–256, for 4–10 epochs. Here the *entire update* is one 64-sample batch, so:
- The 4 epochs are 4 gradient steps on identical data with no shuffling — each pass sees exactly the same batch in the same order. There is no minibatch stochasticity at all.
- Advantage normalisation over 64 samples has substantial estimator variance; the effective advantage scale therefore fluctuates update to update, which is equivalent to a noisy learning rate.
- The PPO clip barely engages: with 4 full-batch steps at lr 3e-4 on a 198 k-param network the policy moves very little within an update, so `ratio` stays near 1 and `min(surr1, surr2)` almost always selects `surr1`. Corroborated by `L_ppo = 0.0411` — very small.
- The value function is fit on 64 points per update.

**Total data.** 6 400 transitions for a 198 k-parameter policy. That is a small-data regime, and it is exactly the argument that motivated the conv trunk (A6.1) — but even 198 k parameters on 6 400 samples is 31 parameters per sample.

**The config is explicit that this is untuned.** `config.py:426-434` carries an ACTION REQUIRED note: benchmark ~16 episodes, read seconds/episode off the log, then set `N_EPISODES = ppo_budget_hours·3600 / seconds_per_episode`. "800 below is a conservative placeholder until you have that number — don't treat it as tuned." Nothing in the logs indicates that benchmark was run and the value revised; `N_EPISODES` is still 800.

**The logging consequence.** `log_every = max(1, 800 // 10) = 80`, and `train()` prints when `update_count == 1 or update_count % 80 == 0`. Over 100 updates that is **two** log lines. That is why the entire training history available for inspection is a single reward-split line — the diagnostic built specifically to verify the rebalance is emitted twice per run.

**Which to increase?** `episodes_per_update`, not `N_EPISODES`. Going 8 → 32 gives 256 transitions per update at the same total cost (25 updates instead of 100), which trades update *count* for gradient *quality*. Given `L_ppo = 0.041` (the clip is not binding) and the noise in a 64-sample advantage normalisation, gradient quality is the scarcer resource here. Reduce `log_every` to 5 regardless — it is free.

---

## A8.2 — UNet evaluations per episode

**Model answer**

```
per pass:    n_steps = int(50 · pass_strength)
             pass_strength = strength_t · strength_boost · (1 + 0.15·pass_idx), capped 0.9
             strength_t ∈ (0.2, 0.7), strength_boost ∈ (0.85, 1.15)
             ⇒ n_steps ∈ [8, 45]
per env step: × n_passes ∈ {2,3,4}, measured mean 2.84 (n→p) / 3.03 (p→n)
per episode:  × BUDGET_T = 8
```
Taking a mid-range `strength ≈ 0.45` and `n_passes ≈ 2.9`: `50 × 0.45 × 2.9 × 8 ≈ 520` UNet forwards per episode, with a full range of roughly **130 to 1 400**.

**On top of that**, every guided step (those past `ramp_start_idx`, i.e. 50–90 % of steps depending on `ramp_start_frac ∈ (0.1, 0.5)`) additionally runs: one VAE decode, two DenseNet-121 forwards (the ensemble), one YOLO forward, and **two** backward passes through decoder + classifier + YOLO. So the dominant cost is not the UNet at all.

**Reconciling with the code's "~1 250–4 500 UNet evaluations"** (`environment.py:82`, `generator.py:257`): that figure is ~2.5–3× my UNet-only count, which is what you get if you count *all* network evaluations (UNet + decode + 2 classifiers + YOLO ≈ 5 model passes per guided step). The comment's phrasing is loose; the order of magnitude is right. A candidate who notices the discrepancy and reconciles it is doing exactly the right thing.

**Fraction carrying the backward:** `guidance_weight > 0` from index `ramp_start_idx = int(n_steps · ramp_start_frac)` onward, with `ramp_start_frac ∈ (0.1, 0.5)`. So 50–90 % of steps, mean ≈ 70 % at the midpoint of the range. Measured `ramp_start_frac` is not in the diagnostics JSON — it is in the info dict but `run_guidance_diagnostics` doesn't aggregate it, which is a small logging gap worth mentioning.

**Why `T_DIFF = 50`.** DDIM's contribution is that ~50 steps approximate the 1000-step DDPM trajectory closely; below ~20 quality degrades visibly, above ~100 you pay linearly for diminishing returns. 50 is the standard operating point and is not a tuned value here.

**Would 20 work?** Testable cheaply and worth doing: fix a seed, generate the same 20 images at `T_DIFF ∈ {20, 35, 50}`, compare CV and IP. If 20 holds, it is a 2.5× speedup on the dominant cost — larger than the backward-fusion win. **This experiment does not appear to have been run**, which is notable given how much engineering went into working around the runtime.

---

## A8.3 — γ = 0.99 over 8 steps

**Model answer**

`0.99⁷ = 0.9321`, so the reward at the final step is discounted to 93 % of the first. The effective horizon `1/(1−γ) = 100` is 12.5× the actual episode length. **γ = 0.99 here is functionally 1.0.**

Is that wrong? No — for a fixed-horizon episodic task where all 8 steps matter roughly equally, undiscounted is the *right* choice. The issue is that 0.99 is a continuous-control default (tuned for 1000-step MuJoCo episodes), not a decision. The honest statement is: "it's effectively undiscounted, which is what I want; I should have written γ = 1.0 and said why."

**The interaction that makes near-undiscounted correct here:** under `FIXED_EPISODE_LENGTH`, `R_hit`/`R_stop` accrue on *every* step validity is held. The objective is therefore "reach validity and hold it," and an undiscounted return literally counts how many of the 8 steps were valid. Discounting would say an early flip is worth more than a late one — which is the hit-and-quit incentive the fixed horizon exists to remove. So γ ≈ 1 is coherent with the design.

**`GAE_LAMBDA = 0.95`:** `γλ = 0.9405`, `(γλ)⁷ = 0.652`. So GAE does mix across the whole episode with mild down-weighting, which is a genuine choice rather than a degenerate one. λ = 0.95 trades a little bias for variance reduction; over 8 steps the difference between λ = 0.95 and λ = 1.0 (pure Monte-Carlo advantage) is small. Given the critic is mis-targeted (A4.3), λ → 1.0 would actually be *safer* here, because it reduces reliance on the bad baseline.

**`last_value = 0.0`:** correct whenever the final transition is terminal, which under a fixed horizon it always is — and `compute_gae`'s `if tr.done` branch handles it independently anyway (`delta_t = tr.reward − tr.value`, no bootstrap). So the default is never actually consulted for the last transition. **Dead-code note:** inside that branch, `next_val = 0.0` is immediately overwritten by `next_val = tr.value` two lines later. The reset that matters is `gae = delta_t`; the `next_val = 0.0` assignment does nothing.

---

## A8.4 — `EDIT_BUDGET_L1 = 0.02` and `EPSILON_STOP = 0.08` together

**Model answer**

They jointly define `is_valid = (d_next < 0.08) ∧ (edit_l1 < 0.02)`, and the log settles which half binds:

```
edit_L1 = 0.0144 (budget 0.02) | score_ok = 2%  budget_ok = 86%  valid = 0%
```

`edit_l1 = 0.0144` is 72 % of the budget — comfortably inside. From the reward inversion (A5.1), `d_next = 0.444 ± 0.133` against a target of `< 0.08`, i.e. **3.3 standard deviations away**. The score condition is not marginally failing; it is failing structurally.

**So the design intent inverted.** `EDIT_BUDGET_L1 = 0.02` was chosen to be the *hard* half — calibrated below the v1-measured 0.042/0.029 so the old strategy couldn't satisfy it. It is now the easy half, because the policy learned to edit sparingly and gave up on flipping. `EPSILON_STOP = 0.08` — which was never re-derived when the ensemble replaced the single classifier — became the binding constraint.

**Why the ensemble makes 0.08 harder.** `f(x) = mean(f_rsna(x), f_chex(x))`. For the mean to reach 0.92 (n→p), both members must be pushed high; a single saturated member caps the mean at ~0.5 + 0.5·f_other. The guidance takes BCE of the *mean*, which does route gradient toward the lagging member (a saturated member's own gradient vanishes) — but consensus is strictly harder to achieve than agreement with one model. That is the *intent* (it narrows the adversarial target), but `EPSILON_STOP` was not loosened to compensate.

**The inconsistency with CV.** The environment demands `|f − c| < 0.08`. CV demands a shift `≥ τ = 0.8` **and** a boundary crossing at 0.5. Starting from `mean f(x) = 0.083` (n→p, from the heldout JSON), CV needs `f_cf ≥ 0.883`; the env needs `f_cf > 0.92`. **The env is strictly stricter than the metric it is supposed to optimise.** So the system can (and does) score CV = 0.975 while `frac_valid = 0 %` — the two are measuring different things, and only one of them is in the reward.

**The fix.** Make the reward's validity criterion match the reported metric: `is_valid = (moved ≥ τ) ∧ (crossed boundary) ∧ (edit_l1 < budget)`. That aligns the objective with the evaluation, removes an unexplained second threshold, and restores a non-zero flip bonus immediately. Failing that, loosen `EPSILON_STOP` to ~0.3 and re-measure.

---

## A8.5 — `ALPHA_CLS = 4.0`, the hidden `1.15`, and `ALPHA_PRES = 0.05`

**Model answer**

**The hidden multiplier.** `generator.py:579`:
```python
- guidance_weight * (config.ALPHA_CLS * 1.15 * cls_guidance_scale) * grad_cls
```
The `1.15` is a literal in the source, not in config, and is undocumented. Effective `ALPHA_CLS = 4.6`. It should be folded into the config value or promoted to a named constant — as written, someone tuning `ALPHA_CLS` is tuning a number that is silently 15 % larger than they think, and `current_cfg_stamp()` records `ALPHA_CLS: 4.0`, so a cached shard's provenance stamp is wrong about the actual guidance strength.

**The ratio.** `4.6 / 0.05 = 92×` before the policy's per-step modulation. With `cls_scale ∈ (0.3, 1.2)` and `pres_scale ∈ (0.3, 1.5)`, the reachable effective ratio spans `(4.6·0.3)/(0.05·1.5) = 18×` to `(4.6·1.2)/(0.05·0.3) = 368×`. Then `guidance_bias` multiplies the classifier side and divides the preservation side, and per-pass escalation moves both — so the policy has substantial control over the balance.

**Is 92× arbitrary?** Not entirely. The two gradients have genuinely different natural scales: `∇L_cls` is the gradient of a BCE on a probability (bounded, and small once the classifier is confident), while `∇L_pres` is the gradient of a mask-normalised L1 plus a feature-space MSE — both larger in magnitude. So `ALPHA_PRES` must be small for the two to be comparable in the summed update. But the *specific* values are hand-tuned, not derived, and no gradient-norm logging exists to justify them.

**The diagnostic nobody added.** Log `‖grad_cls‖` and `‖grad_pres‖` side by side for 100 guided steps. That single measurement would (i) justify or refute the 92×, (ii) tell you whether `ALPHA_PRES` needs to be that small at all, and (iii) directly settle the fp16 question in A6.3 — because if `∇L_pres` is genuinely 92× larger, then scaling it by 0.05 lands it in the *same* range as the classifier branch and the subnormal argument weakens considerably.

**The connection to A6.3.** `ALPHA_PRES = 0.05` is precisely what makes the preservation branch lose fp16 headroom and blocks fusing the two backwards. So a magic number chosen for gradient balance is imposing a 1.5× cost on the entire pipeline. Rescaling (`ALPHA_PRES = 1.0`, `pres_scale ∈ (0.015, 0.075)`) preserves the ratio and removes the constraint — see A6.3.

**Should the ratio be an action?** It partly already is: `guidance_bias` (dim 7) multiplies one side and divides the other, which *is* a learned ratio control. Its measured range in practice — 1.009 ± 0.087 (n→p), 0.950 ± 0.087 (p→n) — means the policy centres it near 1.0 and varies it by ±9 %. So the policy is not fighting the hand-set ratio; it is accepting it. That is either evidence the ratio is roughly right, or evidence the reward gives it no incentive to explore. Distinguishing those two would need a run with `ALPHA_PRES` deliberately mis-set.

---

# Section 9 — Experiments

## A9.1 — The circularity, and what the transfer test showed

**Model answer**

**The circularity.** `pipeline.build_common()` constructs **one** classifier object (`pipeline.py:92`) which flows to:
- `generator.classifier` → `L_cls`, the guidance gradient (`generator.py:538`)
- `env.f` → `d_t`, hence reward and `is_valid` (`environment.py:280, 545`)
- the far-pool scan (`environment.py:229`)
- `scan_groups`'s Xn/Xp split (`metrics.py:100`)
- CV in `run_full_evaluation` (`metrics.py:1085-1092`)

So CV measures *"did gradient descent on f succeed against f."* It is circular by construction and an adversarial perturbation satisfies it exactly as well as a semantic edit. `xrv_classifier.py:139-144` states this breadth is intentional, and `build_ensemble_classifier` prints a warning that CV, group sizes and far-pool sizes are not comparable to single-classifier runs.

**The corroborating fingerprint.** CV 0.975 while changing 66.3 % / 62.3 % of pixels at MAE 0.042 / 0.029 — small amplitude, wide support. That is the signature of an adversarial perturbation, not a focal consolidation.

**What the transfer test showed** (`checkpoints/heldout/heldout_validation.json`, guidance = rsna alone):

| validator | n→p | p→n | verdict (script's own thresholds: >0.5 transfers, <0.15 adversarial) |
|---|---|---|---|
| nih | 0.205 | 0.132 | partial / **ADVERSARIAL** |
| chex | 0.590 | 0.143 | transfers / **ADVERSARIAL** |

**p→n is adversarial under both independent validators.** n→p is partial-to-good depending on the validator.

**The control passes.** Spearman ρ on *real* images: 0.697 (nih), 0.707 (chex), both well above the 0.4 threshold. So the validators agree with the guidance model about real chest X-rays — the low transfer ratios are not validator disagreement, they are a property of the counterfactuals.

**Why the transfer ratio and not CV.** Strict CV on a validator is unfairly pessimistic: it requires the *original* to sit on the starting side of 0.5 according to the validator, and the Xn/Xp groups were selected by the guidance model. Hence `_conditional_cv` restricting to eligible images, and hence the threshold-free ratio `Δ_validator / Δ_guidance` as the headline. Both are reported; the ratio is the verdict.

**Edit concentration corroborates model-free.** 0.282 / 0.297 against a 0.05 uniform baseline. Calibration: a diffuse 75 %-of-pixels edit at MAE 0.037 scores 0.067; a compact 12×12 patch scores 1.000. So there *is* a concentrated core, but most of the mass sits in a diffuse tail.

**The response.** Structural, not cosmetic: move `chex` into `GUIDANCE_ENSEMBLE_WEIGHTS` (adversarial directions rarely survive across independently trained models, so requiring consensus narrows the target), and rotate validators to `mimic_ch, pc, nih` to keep them disjoint. `config.py:36-38` states the membership rule explicitly.

**The gap.** There is **no post-ensemble held-out validation on disk**. The fix was applied and never verified, and `heldout_validation.json` stamps `USE_GRADCAM_MASK: false, BUDGET_T: 25` — it describes v1. So the strongest scientific claim currently supportable is "v1's p→n was adversarial," not "the ensemble fixed it."

**Grading:** anyone who reports CV 0.975 as the headline without this qualification caps at 2 on Research Thinking.

---

## A9.2 — FID in the guidance model's feature space

**Model answer**

**The circularity.** `_XRVFeatureExtractor` defaults to `densenet121-res224-rsna` — the same weight set as `members[0]` of the guidance ensemble, and the default `fid_backbone` is `"xrv"`. FID is therefore computed in the 1024-d penultimate feature space of a model whose *output* the generator was explicitly optimised to move. Since features and output are causally connected, a perturbation that flips the classifier also displaces the features. FID inherits the circularity.

**The observed signature.** v1: MAE 0.036 with FID 103.8. A pixel-space change of 0.036 producing a feature-space Fréchet distance of ~104 is enormous — that ratio is the quantitative statement of "off-manifold."

**What the code does right.** `run_full_evaluation` computes **cross-group** FID as well: `FID(Xp, Xc_n→p)` and `FID(Xn, Xc_p→n)` — the counterfactual against its *target* class. The reasoning (`metrics.py:1101-1113`) is exactly right:
- same-group FID high ⇒ ambiguous (real class change, or artefact);
- cross-group FID low ⇒ unambiguously good (the CF is indistinguishable from real images of the target class);
- **high on both ⇒ off-manifold**, which is the diagnosis.

**The gap.** Those four cross-group values are computed by the current code but are **absent from `checkpoints/eval_metrics.json`**, which contains only `negative_group` and `positive_group`. That JSON predates the change. So the disambiguating measurement exists in code and not in results — and `report.tex` quotes FID numbers (5.67/3.20) that appear nowhere in any artefact in the repo.

**The clean fix.** Report FID in a feature space with no relationship to guidance. The `"inception"` backbone is already implemented (`metrics.py:397`), and among xrv weights `-mimic_ch` or `-pc` are independent. **Not** `-all`: `metrics.py:451` and `heldout_validation.py:35-37` both note it includes RSNA, so it is independent of nothing.

**FID at N ≈ 200 against 1024 dims.** The covariance estimate has rank ≤ 199 in a 1024-d space, so `Σ` is singular and `sqrtm(Σ₁Σ₂)` is numerically unstable — hence the `eps·I` regularisation. The bias direction is known and **upward**: with too few samples, `tr(Σ₁ + Σ₂ − 2(Σ₁Σ₂)^½)` is systematically overestimated because the cross term is underestimated when the two sample covariances are noisy and nearly orthogonal in their null spaces. `run_full_evaluation` warns below 200 (`metrics.py:1079-1082`) and the docstring recommends N ≫ feature dim. At N = 200, FID values are not comparable to published numbers computed at N = 10 000+, in either backbone.

---

## A9.3 — Provenance of `eval_metrics.json`

**Model answer**

**It describes v1.** `checkpoints/heldout/cf_cache.pt`'s stamp — `ANCHOR_TO_EPISODE_ORIGIN: true, USE_GRADCAM_MASK: false, USE_SPARSITY_REWARD: false, BUDGET_T: 25, ENV_REFINEMENT_PASSES: 2` — and its metrics (CV, concentration, MAE) line up with `eval_metrics.json`.

**What has changed since:**

| Knob | Then | Now |
|---|---|---|
| `BUDGET_T` | 25 | 8 |
| `USE_GRADCAM_MASK` | False | True |
| `USE_SPARSITY_REWARD` | False | True |
| `FIXED_EPISODE_LENGTH` | absent/False | True |
| `LORA_R` | 8 | 16 (+ widened targets) |
| guidance `f(·)` | single rsna | 2-member ensemble |
| reward λs | hit 6.0, stop 8.0, fid 0.5, pres 0.5 | 2.0, 2.0, 2.0, 3.0 |
| `EDIT_BUDGET_L1` | absent | 0.02 |

**Can it be reproduced today?** No. Beyond reverting `config.py` wholesale, the rank-8 LoRA artefacts are archived and would not load into the current adapter (`pipeline.py:153-170` raises on exactly this), and the ensemble changes the Xn/Xp split so even the *input* images differ.

**And the current pipeline would refuse to produce it.** `_load_shards` compares each shard's stamp against `current_cfg_stamp()` and raises with a per-key drift report if they differ. That guard is the candidate's own work, and it is doing precisely its job — the artefact on disk is exactly the kind of thing it was built to prevent from being silently re-reported.

**On the stamp's exclusions.** `current_cfg_stamp()` deliberately omits reward-shaping knobs, because at eval time the policies are loaded from checkpoints and reward is never computed — so `LAM_*` cannot change the pixels. That is correct *for a shard*. It is **not** sufficient as full provenance: the reward weights determined the policy, and the stamp cannot distinguish two policies trained under different rewards. A `policy_sha256` field would close that, and would have made the v1/v2 confusion impossible.

**What would move most on a re-run?** The config predicts **CV should drop**, deliberately: "Expect CV to DROP as these come on. That is the intended direction: CV 0.975 with 75 % of pixels changed is a worse result than CV 0.7 with 5 % changed and an independent classifier agreeing." A candidate who cites that — and frames a CV decrease as the *success* criterion — has understood the project's actual thesis. Secondary predictions: `frac_changed` and MAE down; edit concentration up; cross-group FID is the number to watch.

---

## A9.4 — Three ablation arms, all on

**Model answer**

**What the protocol says** (`config.py:150-158`):
```
ARM 1 (now)  ANCHOR_TO_EPISODE_ORIGIN = True
ARM 2        + USE_GRADCAM_MASK        = True
ARM 3        + USE_SPARSITY_REWARD     = True
"flip these ONE AT A TIME, re-running eval between arms, so the effects stay attributable"
```

**What exists:** all three `True` in config; the only `eval_metrics.json` has all three `False`. So the sole available comparison is **all-off vs all-on** — a three-factor change measured as one contrast. Per-mechanism attribution: **none**.

**And the arms are not even independent:**
- `SPARSITY_OUTSIDE_CAM_ONLY = True` requires a cached CAM; without `USE_GRADCAM_MASK` it silently degrades to a plain unweighted L1 (`config.py:296-299`). So ARM 3's *form* depends on ARM 2.
- `R_fid` (L1, λ=2.0) and `R_sparsity` (weighted L1, λ→2.0) are the same functional form — the config concedes "treat `LAM_FID` as a confound." From A5.3, the measured weighting ratio is 0.862, i.e. only a 14 % tilt, so the two are nearly collinear in practice.
- `BUDGET_T` 25 → 8, `LORA_R` 8 → 16, the reward rebalance, and the ensemble all changed alongside. That is **seven** simultaneous factors, not three.

**The minimum honest claim:** "the v2 configuration as a whole differs from v1 as follows" — a system-level comparison with no mechanism attribution.

**Cost of doing it properly:** 4 arms × 400 images × 4.4 min ≈ 117 h of generation, or ~15 h wall-clock across 8 shards per arm. Two GPU-days for the full matrix. That is affordable and was not spent.

**The cheapest experiment that separates anchor from CAM mask.** Both change *pixels*, so they need generation — but not metrics on 400 images. `edit_concentration` and `frac_changed` are model-free, per-image, and low-variance (they average over 65 536 pixels per image). At n_eval = 30 per group, one arm is ~4.4 h — four arms fit in a single overnight 8-shard job. Report concentration + MAE + frac_changed only; skip FID entirely, since FID is the one metric that genuinely needs N ≥ 200. That is the experiment: **four arms, n_eval = 30, model-free metrics only.**

**If only one arm:** ARM 1 (`ANCHOR_TO_EPISODE_ORIGIN`). It is the only one whose mechanism is unambiguous (up to ~100 chained encode/renoise/decode round trips per episode with no reference to the original), it requires no CAM and so has no degeneracy confound, and the config's own evidence attributes the observed 73–77 % pixel change primarily to it.

---

# Section 10 — Research Thinking

## A10.1 — Restarting

**Model answer** — should lead with *evaluation*, not the model.

1. **A held-out test split on day one.** There is none. `_build_far_index_pool` scans the whole dataset and PPO samples from it; `scan_groups` scans the same dataset for Xn/Xp. Every reported number is in-sample. This is a 20-line fix (split `dataset.samples` by patient ID before anything else) and it is the single largest threat to the results' validity.
2. **Guidance and evaluation classifiers disjoint from the start**, rather than discovering circularity in month three and patching with an ensemble whose effect was never measured.
3. **The adversarial-vs-semantic test before the generator.** `edit_concentration` needs no model and no downloads; running it on week-two outputs would have surfaced the diffuse failure mode two months earlier. Same for a VAE round-trip control.
4. **Sharded eval infrastructure first.** Three jobs died to walltime before it existed.
5. **A fixed-schedule baseline.** `run_full_evaluation` already falls back to `generator.generate_counterfactual(x, c, M=1)` with defaults (`strength=0.55`, both scales 1.0, `ramp_start_frac=0.25`) when `policy_*` is `None` — so the baseline is **one flag away** and costs one eval run. Without it, the project cannot state what PPO contributed. That is the most important missing experiment in the repository.

**Grading:** 5 requires the missing train/test split *or* the missing fixed-schedule baseline. Both is exceptional. Hyperparameter-only answers cap at 2.

---

## A10.2 — Unlimited compute

**Model answer** — must name a *falsifiable* question. Strong versions:

- **"Can transfer be made an objective rather than a diagnostic?"** Train against an ensemble while *validating* against held-out models each update, and require consensus on a member that is resampled per episode. The falsifiable claim: transfer ratio > 0.5 on genuinely held-out validators, in both directions. This is the natural continuation of what the repo already found.
- **"Does a learned schedule beat a tuned fixed one?"** The null hypothesis the project cannot currently reject. Unlimited compute makes this trivial: grid the 9-d action space coarsely, take the best fixed setting, compare. If they match, PPO contributed nothing and the paper is about guided diffusion, not RL.
- **"What is the minimal sufficient edit?"** Formalise as `min ‖x_cf − x‖₀ s.t. f(x_cf) crosses`, and trace the Pareto front instead of collapsing it with hand-set λ. Every reward-weight argument in this project is an attempt to pick one point on a front nobody plotted.
- **Reader study.** For an *explanation* method this is the only ground truth: do radiologists agree the highlighted region is where they would look? No proxy substitutes, and no amount of CV/FID/IP does either.

**Red flag:** "scale up, use SDXL, train longer" — none of which is a question.

---

## A10.3 — "It explains the classifier, not the disease"

**Model answer**

**The objection is correct.** Accept it and narrow the claim. A counterfactual characterises the model's decision boundary; that is what the method does and what it should be advertised as. It becomes clinically meaningful only under an additional assumption — that the classifier has learned disease-relevant features — and the right response is that this assumption is **testable**, which is exactly why two diagnostics exist:
- `check_classifier_localization.py` — does the classifier attend to lung pathology, or to a global brightness/exposure shortcut? Its docstring names the failure mode explicitly, and it runs both classifier variants automatically so the comparison is not a manual edit.
- `heldout_validation.py` — does the edit carry signal that independently-trained models recognise?

**The stronger version.** The method is *most* valuable when it reveals the model is wrong. A counterfactual that flips the classifier by brightening the whole thorax is a genuine finding — about the classifier. The failure mode is not producing such an edit; it is **reporting it as a disease explanation**. That is precisely why the p→n transfer ratio of 0.132/0.143 matters more than the CV of 0.955, and why the honest headline for this project is "the p→n direction was fooling f(·), and here is how we detected it" rather than "CV = 0.975."

**On the frozen pretrained classifier.** It cuts both ways. Weaker objection: the candidate didn't train it, so the counterfactual is at least explaining an artefact that exists independently of this project — an off-the-shelf model someone might actually deploy. Stronger objection: because it was never trained or validated here, there is *no* evidence it is a good pneumonia detector on this data, beyond a Grad-CAM eyeball check. Notably `xrv_classifier.py` chose the multi-pathology DenseNet over a custom binary ResNet-18 on precisely this reasoning — a model forced to discriminate pneumonia from effusion, cardiomegaly and edema is much less able to lean on a single global shortcut, since that shortcut wouldn't separate the conditions. That is a good argument and the candidate should make it.

**What would convince a radiologist:** side-by-side originals and counterfactuals with the difference map, blinded, and a forced choice — "is this edit anatomically plausible, and is it where you would look?" Fifty image pairs and three readers is a weekend, and it is worth more than any additional metric.

---

# Repository Walkthrough — Answer Key

**W1. `generate_counterfactual` (generator.py:375-608)**
Beat sheet: `anchor_is_init` resolution → encode `x` → build cond → encode `x_anchor` separately if different → `n_keep/start_idx/timesteps` → `add_noise` → precompute `S_x`, `latent_mask`, `edit_mask` (dilate → feather → × cam) and `O_x` **once** outside the loop → per step: UNet, `scheduler.step`, take `pred_original_sample` **and** `prev_sample` from the same `step_out`, ramp weight, guided correction, mask, background composite, optional trajectory checkpoint → final decode.
Key discriminator: `z0_hat` and `z_t` come from the *same* `step_out` — `pred_original_sample` is the clean-image estimate at `t`, `prev_sample` is the next latent. `trajectory` is decode-only and never feeds back.
Wrong answers: "it denoises the image"; confusing the two `step_out` fields; thinking `M` affects the output (it only picks how many frames to save).

**W2. `compute_gradcam_latent_mask` (generator.py:229-335)**
Percentile *after* upsampling because the native CAM is 7×7 = 49 cells at the DenseNet's 224 input; a 0.99 quantile over 49 values is the max, which is exactly min–max normalisation and lets one outlier cell define the scale and flatten everything else. After interpolation to 32×32 = 1024 cells, the 0.99 quantile is ≈ the 10th largest — a robust scale. Then `cam^γ` with γ = 0.75 < 1 *broadens* (raises mid values toward 1), blur σ = 0.8 kills the blockiness of an upsampled 7×7, then `floor + (1−floor)·cam` maps to [0.15, 1].
Wrong answers: "normalisation is standard"; not knowing the 7×7; thinking γ > 1 broadens; believing the mask is recomputed per step (it is cached per episode — one DenseNet fwd+bwd against hundreds of UNet evals).

**W3. Background lock (generator.py:592-600)**
`z_t` after `scheduler.step` is `prev_sample`, i.e. already at `t_next` — so the background must be brought to `t_next`, not `t`, or you composite two different noise levels and create a statistical seam the next UNet call will try to "fix." On the final iteration `z_bg = z_anchor_4d` clean, pinning out-of-mask output to the anchor's VAE reconstruction.
Wrong answers: not knowing `prev_sample` is at `t_next`; thinking the last step also adds noise; claiming the lock gives an exact pixel guarantee (see A3.3 — the 0.15 floor).

**W4. `environment.step` (environment.py:477-580)**
All nine dims per the table in §1.6. The per-pass escalation is **hardcoded, not learned**: `strength × (1 + 0.15·i)` capped at 0.9, `cls_scale × (1 + 0.35·i)`, `pres_scale × max(1 − 0.15·i, 0.4)`. So later passes push harder on the classifier and protect less — a fixed schedule layered on top of the learned one, which partially confounds what the policy is actually controlling.
`n_passes` is action-controlled because compute-per-step is itself a decision: an easy image should not pay for four passes. But note dims 4 and 5 are thresholded bits, giving `n_passes ∈ {2,3,4}` with the two bits interchangeable — a two-bit encoding of a three-valued quantity, so the mapping is not injective.
Wrong answers: thinking the escalation is learned; not noticing the bits are redundant.

**W5. `_compute_reward` (environment.py:353-467)**
Deletable with no effect on the policy gradient: `R_bud` (constant, removed by advantage normalisation), and one of `R_val`/`R_close2` (both monotone in `d_next`, 84 % of mass between them). `R_hit`/`R_stop` are currently identically zero, so also inert — though not by design.
The `weighted` dict exists so `PPOTrainer._format_reward_split` can print a *true* decomposition without re-deriving the λs anywhere else; the verification is that the terms sum to the mean per-step reward, which they do exactly (−1.590 × 8 = −12.72).
Wrong answers: "all of them matter"; not knowing `R_bud` is constant.

**W6. `generate_via_policy` (environment.py:586-675)**
Before it existed, `run_full_evaluation` and `diff_viz` called `generator.generate_counterfactual` with **default** arguments — so every reported CV/FID/FOP number and every diff plot described the fine-tuned generator on a hardcoded schedule, not the trained PPO policy. That is the single most important thing this function fixed.
On `max_steps`: `get_observation`'s step fraction is `current_step / self.T`, computed against `self.T` not `max_steps`, so a differing `max_steps` feeds the policy fractions it never saw (never reaching ~1.0, or exceeding it). Prefer temporarily setting `self.T` — which `_measure_holdout_cv` does, in a `try/finally`. Also note `'converged'` reports the validity of the **final step**, not "terminated early"; under a fixed horizon `n_steps_used == budget` always, so reading it as early termination would make every episode look non-convergent.
Wrong answers: not knowing the observation has a step fraction; not knowing eval used generator defaults before.

**W7. `compute_gae` (ppo.py:283-335)**
Reversed loop: at `t`, `next_val` holds `V(s_{t+1})` from the previous (later) iteration; `next_val = tr.value` at the end supplies it to the earlier step. On `done`: `delta_t = tr.reward − tr.value` (no bootstrap past a terminal state) and `gae = delta_t` (no accumulation across the boundary). Both correct.
Two things to catch: (i) `next_val = 0.0` inside the `done` branch is **dead** — overwritten two lines later; (ii) `shaped_r[t]` uses `adv[t]` computed *before* the normalisation on line 330, so it blends an unnormalised advantage into a target that is later compared against normalised quantities.
Wrong answers: claiming the flat multi-episode list breaks boundaries (it doesn't); missing the pre-normalisation `adv`.

**W8. `PPOTrainer.update` (ppo.py:718-804)**
Config constants: `PPO_CLIP`, `C_VALUE`, `C_ENTROPY`, `LAM_FEAS_AUX`, `LAM_PROG_AUX`, `LAM_STOP_AUX`, `LAM_ACT_SAT`, `ACT_SAT_THRESHOLD`, `GAMMA` (via `td_targets`), `DEVICE`. **Not** in config: `clip_grad_norm_(..., 0.5)` and `n_epochs = 4` — both hardcoded, both things you would tune.
Tuning order: (1) the `L_value` target (A4.3) — it is a correctness issue, not a tuning knob; (2) `episodes_per_update` 8 → 32 (A8.1); (3) `C_ENTROPY`, given the saturation history; (4) the aux weights, which are all 0.1 and none of which has a measured effect.
Wrong answers: missing the hardcoded 0.5; no justified ordering.

**W9. `_measure_holdout_cv` (ppo.py:528-573)**
The import is inside the function to break a cycle: `ppo` ← `environment` ← ... and `metrics` → `inference` → `ppo`. A module-level `from src.metrics import ...` in `ppo.py` would close the loop. The `try/finally` restores both `env.T` and `env.lambda_sparsity` — the latter because the sparsity weight must not perturb a probe whose whole purpose is to detect the sparsity ramp's effect. And the CV criterion deliberately matches `run_full_evaluation` (τ = 0.8, boundary 0.5) rather than inventing a second notion of validity — otherwise the safeguard would be defending a number nobody reports.
Add: it has never actually run (Finding F1).
Wrong answers: "style"; not knowing the cycle.

**W10. `grad_cam` (gradcam.py:244-318)**
`register_full_backward_hook` is unusable because torchxrayvision's `DenseNet.forward` runs `F.relu(features, inplace=True)` on exactly the tensor a backward hook would wrap, which PyTorch forbids. `retain_grad()` involves no custom-autograd-Function wrapping, so the downstream in-place op is handled by ordinary autograd version counting.
`find_target_layer` prefers a submodule named `features` because DenseNet **concatenates** every dense layer's output onto a growing stack — hooking one internal conv captures ~32 of the ~1024 channels that actually reach the classifier, producing a near-uniform map regardless of prediction confidence. Both are recorded as hard-won fixes carried over from the diagnostic script, deliberately shared so the diagnostic remains evidence about the mask PPO is actually constrained by.
Note `act = act_tensor[0]` reflects the value *after* any downstream in-place op (the ReLU) — standard for Grad-CAM, and stated as such.
Wrong answers: "backward hooks are deprecated"; not knowing why the full `features` block matters for DenseNet specifically.

**W11. `ensemble_grad_cam` (gradcam.py:173-241)**
Passing the ensemble to `find_target_layer` does not fail — it walks `named_modules()`, returns the first submodule named `features` (which is `members.0.model.features`), and **prints its usual success message**. The result is a one-member CAM presented as the ensemble's, and member 0 is `rsna`: the 2-head single-corpus model most prone to the very shortcut the mask exists to constrain. Silent, plausible, and wrong — the worst combination.
Averaging happens *after* each map's own [0,1] normalisation so a member with larger raw activations cannot dominate. Degenerate members are skipped rather than contributing a zero map that would drag the mean toward uniform; only all-degenerate raises. There is also a shape-guard interpolation in case members ever differ in spatial size, so a mismatch fails with a clear message instead of deep inside `torch.stack`.
Wrong answers: not seeing the silent-failure mode; averaging raw maps.

**W12. `counterfactual_validity` (metrics.py:178-233)**
The conjunction is not redundant, in either direction:
- 0.05 → 0.55 **crosses** but moves 0.50 < τ = 0.8 → rejected, correctly (a boundary nudge is not a counterfactual).
- 0.05 → 0.90 does both → accepted.
- For `target="positive"`, `crossed = (f_x ≤ 0.5) & (f_xc > 0.5)`, so an image the guidance model already scored above 0.5 is rejected regardless of how far it moved — the *starting side* is part of the definition. That is what makes strict CV pessimistic on an independent validator, and why `heldout_validation._conditional_cv` exists.
Also worth noting: `finite_mask` filtering, and NaN (not 0) on an empty batch — so an empty group is reported as unmeasured rather than as failure.
Wrong answers: thinking one condition implies the other.

**W13. `edit_concentration` (metrics.py:713-760)**
If `|Δ|` is spatially constant, the top 5 % of pixels hold exactly 5 % of the mass — so `top_frac` *is* the null by construction, requiring no model, no calibration set, and no assumptions. Calibration in the docstring: diffuse 75 %-of-pixels edit at MAE 0.037 → 0.067; compact 12×12 patch → 1.000. Measured 0.282/0.297 sits between: a real concentrated core with most of the mass in a diffuse tail.
The rationale is the sharp part: a semantic edit and an adversarial perturbation are separated by their **L0/L∞ profile**, not their magnitude. A focal consolidation is large-amplitude over small support; an adversarial perturbation is small-amplitude over wide support. Two edits with identical L1 can sit at opposite ends of this statistic.
Wrong answers: not deriving the baseline; unable to interpret 0.28.

**W14. `_load_shards` (eval_main.py:294-387)**
Every check is fatal because the failure mode is a *plausible wrong number*, not a crash. A missing shard silently shortens Xn/Xp, and FID's bias grows sharply as N falls — so a partial reduce emits a believable FID that is wrong in a known direction. A config mismatch pools two ablation arms into a metric describing neither. The error messages carry the fix, including the exact `sbatch --array=` line listing the missing indices.
The one gap: it compares `config`, not the derived Xn/Xp lists — and those are demonstrably not identical across nodes (Finding F2). A `sha256` of the concatenated patient IDs would close it.
Wrong answers: "defensive programming" without the FID connection.

**W15. `_extract_labels` / `_build_balanced_sampler` (finetuning.py:26-102)**
Three tiers: `.labels`/`.targets` → `.samples` (RSNA stores `(patientId, label)` tuples, so `s[-1]` needs no image decode) → `.df` → full `__getitem__` scan. The last tier decodes all ~26 684 DICOMs, applies rescale/normalise/resize/ToTensor, and throws the image away to read one float — minutes to tens of minutes, and it warns loudly with the fix ("expose a `.labels` array").
The sampler uses `1/class_counts` weights with `replacement=True` and `num_samples=len(dataset)`, giving ≈1:1 expected class representation per epoch. The per-class loss tracking (`avg_L_diff_pos_batches` / `neg_batches`) exists to verify the balance is working rather than assuming it — though note it classifies a *batch* by `pos_frac > 0.5`, so it is a batch-level not sample-level attribution, and with balanced sampling most batches sit near 0.5 where the split is close to arbitrary. And per A3.4, `L_diff` may be partly explained by the conditioning leak, so those curves cannot be read as cleanly as the comment suggests.
Wrong answers: not knowing why `.samples` is the fast path.

---

# Ownership Questions — Evidence Base

Most of these are 🔒: only the candidate can answer. What follows is the repository evidence to check the answer *against*, plus the answer where the repo settles it.

**1. The 18 commits.** `git log --oneline` gives 18, of which several are duplicated messages (`4608229` and `23953d1` share "feat: wire PPO actions…"; `a422476`/`58b0236` both "added evaluation"), suggesting rebase/merge churn. `be5b609 "many changes"` and `0890625 "added images"` are the two most recent and cover the largest span. 54 files tracked. History is coarse enough that it cannot be used to attribute work — say so, and ask the candidate to reconstruct the order of major changes from `config.py`'s dated comments instead (several are stamped `2026-08-01`).

**2. `checkpoints/archive_v1_rank8/README.md`** 🔒 — the directory contains `finetuning_latest.pt`, `cond_proj.pt`, `ppo_policy.pt`, `unet_finetuned.pt` plus `n_to_p/` and `p_to_n/` subdirectories. Archiving rather than deleting is correct and defensible: a rank change makes the artefacts unloadable but not worthless, and `pipeline.py:166-168` explicitly points users at this README for the rank-8 settings. Check the candidate knows the README is *referenced from an error message*.

**3. The commentary convention** 🔒. `config.py` is 516 lines of which roughly 60 % is prose carrying job IDs (17648782, 17648985, 17649721, 17653310), measured numbers, and explicit ACTION REQUIRED notes. Costs: it goes stale (the file still describes `LAM_FID 0.5 → 2.0` reasoning with a "~25×" figure that is wrong — see A5.3), and it encodes experimental findings in a place with no version history tied to results. Benefit: it is the only place in the repo where *why* is recorded at all. Ask what they would use instead — an experiment log, a results database keyed by config hash.

**4. Reconstructing job 17648782.** Answerable from the repo. `avg_return = 33.931` at `avg_ep_len = 1.00` and `direction_scale = 2.25` ⇒ per-step reward `33.931 / 2.25 = 15.08`. With v1 weights (`λ_hit = 6`, `λ_stop = 8`) and a valid, tau-positive step: `R_hit·6 = 6.00`, `R_stop·8 = 8.00`, total 14.00 of 15.08 = **93 %**. The rest: `λ_prog·1.5 → 1.40` (9 %), `λ_pres·0.5 → −0.21` (−1.4 %), `λ_fid·0.5` on MSE 0.0033 → `−0.0016` (−0.01 %). That last line is the finding: the fidelity term was **one part in ten thousand** of the return, which is the mechanism behind 66–75 % of pixels changing.

**5. Longest file** 🔒 — `src/metrics.py` (1336 lines) and `factory/ppo.py` (994) are the largest; `src/generator.py` (691) holds the most conceptual difficulty. `metrics.py` also has the longest documented bug history (FOP v1→v2→v3, FID normalisation, FID `sqrtm`, SCP degeneracy, batching rewrites). File mtimes: `generator.py` and `metrics.py` are the two most recently touched (Aug 6–7), consistent with them being where late work concentrated.

**6. Two smoke tests.** `smoke_test.py` covers the training path; `smoke_test_eval.py` exists because the evaluation path's distinguishing risk is **cold-process checkpoint reload** — `load_generator_weights` + `load_directional_policy` restoring into freshly constructed objects. A combined train+eval process never exercises that, because the objects are already in memory with the right shapes. That is exactly the failure class that `LORA_R` and `POLICY_TRUNK` changes produce, and both loaders have custom error handling for it.

**7. `preflight_gpu.py`.** Fully answerable from the file. Job **17662452_0** was scheduled onto `hpc-n983`, an "NVIDIA RTX PRO 6000 Blackwell Server Edition" (sm_120), and died ~5 minutes in **inside `peft.get_peft_model()`** with `CUDA error: no kernel image is available for execution on the device`. The venv is torch 2.5.1+cu121, whose kernels stop at sm_90. The traceback points into peft/diffusers and reads like a software bug; it is really "wrong node," and five minutes of dataset scanning and model loading happen before the first CUDA op trips it. The check exits 0 and prints the GPU + arch list on success, so every job log records which hardware it actually ran on — which is how the V100-vs-RTX-6000 question (Q20) is settled.

**8. The SLURM constraint allowlist.** Two reasons, both in the script: SLURM 22.05.10 has **no NOT operator** in `--constraint` and `--prefer` (soft constraints) only arrived in 23.02, so you cannot say "not this node." Worse, feature strings **do not encode compute capability** — `hpc-n983` advertises the same `gpurtx6000` tag as the Turing Quadro RTX 6000 nodes it must be distinguished from. Hence both the explicit `--exclude=hpc-n983` *and* `preflight_gpu.py` as a backstop that does not depend on the tags being accurate. This is a genuinely good piece of cluster engineering.

**9. Why 22 h of 24** 🔒 — the guard stops *before beginning* an update that would overrun, using the slowest update seen. The margin must cover: one full update (which can be a CV-probe update at ~3× cost), the `_save_state` write, and process teardown. With updates of order minutes, 2 h is generous — arguably too generous, since it discards ~8 % of every job's allocation. A candidate who says "1 h would have been enough and I was being conservative after losing 24 h once" is answering well.

**10. `_PROCESS_START_S` at import.** Answered in A7.3: Steps 1–4 run before any `train()` call and are charged against the same `--time`, and `--direction both` calls `train()` twice — timing from `train()` entry would give the second direction a fresh full budget, permitting ~44 h against a 24 h wall.

**11. Most-changed value** 🔒 — traceable candidates: `BUDGET_T` (25 → 8), `N_EPISODES` (2000 → 800, per the "REVISED for actual hardware" comment), `N_DENOISE_EPOCHS` (30 → 15 → 30), `DENOISE_BATCH_SIZE` (32 → 8), `LORA_R` (8 → 16), `LAM_FID` (0.5 → 2.0), `LAM_SPARSITY_TARGET` (0.5 → 2.0), `LAM_HIT` (6 → 2), `LAM_STOP` (8 → 2), `CV_CHECK_MAX_STEPS` (12 → 4). Note several were revised for hardware that then wasn't used (the V100 sizing comments vs the RTX 6000 that actually ran).

**12. `yolov8n.pt`.** Settled: it is **gitignored** (`.gitignore` line 8) and **not tracked** (`git ls-files | grep -i yolo` returns nothing). It is present in the working tree because `ultralytics.YOLO(weights)` downloads it on first use. So the answer is "no, not committed" — and a candidate who says it is committed hasn't checked. Related and more interesting: `checkpoint/classifier.pt` **is** tracked despite `checkpoint/` being in `.gitignore`, because gitignore does not untrack already-committed files. That is the legacy ResNet-18 classifier, whose loader `build_classifier` is no longer called by any pipeline path (the ensemble replaced it) — so a committed binary artefact for a dead code path.

**13. `outputs/gradcam_check/`.** Twelve PNGs: `gradcam_current_{0..5}.png` and `gradcam_xrv_{0..5}.png`. `check_classifier_localization.py` sets `CLASSIFIER_MODES = ["current", "xrv"]` and runs both automatically on 6 positive-labeled images. So `current` = the custom ResNet-18, `xrv` = the TorchXRayVision DenseNet. The comparison is the evidence base for `xrv_classifier.py`'s opening argument — that the binary ResNet-18 may have learned a global brightness/contrast shortcut, visible as diffuse whole-thorax heat versus localised lung-region heat. **Ask the candidate to open two of these side by side and say what they show.** If the ResNet CAMs are diffuse and the xrv ones focal, that is the direct justification for the classifier swap; if they look similar, the swap was made on other grounds. This is the most checkable "did you look at your own outputs" question available.

**14. `checkpoints/smoke_all/` and `smoke_eval/`.** Not real results. `smoke_test.py` / `smoke_test_eval.py` run with tiny budgets to catch crashes and shape errors before spending GPU time; the `eval_metrics.json` files they leave are structurally valid and numerically meaningless (tiny `n_eval`, so FID especially is garbage). The risk is that a filename identical to the real artefact invites confusion — `smoke_all/eval_metrics.json` and `checkpoints/eval_metrics.json` differ only by directory. Ask whether any reported number ever came from a smoke directory.

**15. Most surprising parameter** 🔒 — the strongest available answer is `guidance_bias`: a policy sampling at σ ≈ 0.61 producing an action with std 0.005, pinned at the extreme of its range. The surprise is that the *distribution* looked healthy while the *behaviour* was constant, and only the diagnostic's std column revealed it. Second-strongest: `LAM_FID` contributing 0.01 % of the return while nominally weighted 0.5.

**16. `CV_CHECK_MAX_STEPS` 12 → 4.** Fully answerable. It is applied as `min(CV_CHECK_MAX_STEPS, env.T)` (`ppo.py:556`). At `BUDGET_T = 8`, the old value of 12 resolved to 8 — so the probe ran at **full** episode length and the comment's claimed "~halves per-episode probe cost" never happened. It must be strictly below `BUDGET_T` to do anything. The two are coupled and must move together.

**17. Deriving `ENV_EVAL_HOLDOUT_N = 24`.** CV is a fraction over N images, so its resolution is 1/N and a single image flipping moves it by 100/N points. `CV_DROP_TOLERANCE = 0.08` = 8 points. For the tolerance to sit safely above the noise floor you want one image to be well under 8 points: at N = 24, 4.2 points/image, so the tolerance is exactly two images' worth. At N = 8, one image is 12.5 points — above the tolerance, so the freeze would latch on a single image's noise in essentially every run. At N = 32 you'd get 3.1 points/image (2.6 images per tolerance) at 33 % more probe cost. 24 is the smallest value that clears the noise floor with margin.

**18. `LAM_STOP`/`LAM_HIT` 8.0/6.0 → 2.0/2.0.** Driven by the job-17648782 decomposition (item 4): flip terms were 93 % of the return. Target was ~40 % flip / 40 % preservation / 20 % progress. **It did not work** — measured 0 % / 7 % / 3 %, because the simultaneous introduction of the joint validity criterion made `is_valid` unreachable, so the flip terms went to exactly zero rather than to 40 %. Two changes were made at once and they interacted: the weight cut assumed the bonus would still fire. This is the cleanest example in the project of a confounded fix.

**19–24. Report-vs-repository.** See §1.15 of the question paper and A2.3. Summary of what the artefacts say:

| Report claim | Repository artefact |
|---|---|
| FID 5.67 / 3.20 | `eval_metrics.json`: 103.83 / 59.73 |
| `BUDGET_T = 25`, `λ_stop = 8.0`, `λ_hit = 6.0`, `λ_fid = 0.5`, `λ_pres = 0.5`, LoRA r=8 | `config.py`: 8, 2.0, 2.0, 2.0, 3.0, r=16 |
| CTR = 0.479 ± 0.001; SCP = 0.424 ± 0.055 | JSON: CTR `mean_xn` = 0.4237 ± 0.0552; SCP `mean_xn` = 0.4794 ± 0.0008 — **transposed** |
| CV mean 0.965 (Tab. 2) / 0.970 (baseline tab.) | JSON: 0.975 |
| "single Tesla V100-32GB" | `logs/`: Quadro RTX 6000, 23040 MiB |
| CycleGAN / GANterfactual "under the identical protocol" | no GAN code, no logs, no checkpoints |
| p→n has "higher IP (HF-corr 0.793 …)" | JSON: n→p 0.938 > p→n 0.798; the sentence contradicts its own quoted numbers |
| "this rollout terminates in two steps" (Fig. reward trace) | `FIXED_EPISODE_LENGTH = True` ⇒ impossible; figure predates the flag |

The transposition (row 3) is the most diagnostic: it is a copy-paste error, not a fabrication, and it is *provable* because the ± values travel with the wrong means. A candidate who recognises it instantly as a transposition is demonstrating they know their own numbers; one who defends the values is not.

**25. Why n→p `avg_reward` is 3.6× worse.** `guidance_diagnostics_n_to_p.json` −4.895 vs `p_to_n` −1.367. First divide out `direction_scale = 2.25`: −4.895 / 2.25 = **−2.176**, so the underlying per-step reward is 1.6× worse, not 3.6×. Does it match the transfer ratios? **No — it runs the other way.** n→p transfers *better* (0.205/0.590) than p→n (0.132/0.143), yet n→p earns less reward. That is a genuinely interesting inversion: the direction the reward function rates worse is the direction that produces more semantically real edits. It is consistent with the reward being dominated by `d_next` (classifier proximity), which an adversarial edit optimises more easily than a semantic one. **If a candidate spots this, it is a 5** — it is arguably the strongest single piece of evidence in the repo that the reward function is measuring the wrong thing.

**26. Preferring the 1.7 GB artefact.** `unet_finetuned.pt` is what `save_checkpoints()` writes and what `main.py` declares as "the trained generator"; `finetuning_latest.pt` (22 MB, PEFT format) is the per-epoch resume file. They are equivalent **today** because `PPOTrainer` optimises only `policy.parameters()`, so the generator is unchanged between the end of Step 4 and the end of Step 5. Preferring the post-training artefact means that stays true *by construction* if anyone later makes PPO touch the generator, instead of silently loading stale adapters. Cost: ~30 s and 1.7 GB of I/O per eval process — × 8 shards. Defensible; a `sha256` comparison would give the same guarantee for free.

**27. `deterministic=False` in diagnostics vs `True` in eval.** Deliberate and correct. The diagnostic's purpose is to measure whether the policy's actions **vary** — `guidance_bias_std` is the headline number, and the threshold for health is `> 0.05`. Sampling deterministically would return the mean action every time and the std would be identically zero, making the diagnostic vacuous. Evaluation uses the mean action because reported metrics must be reproducible. Two different questions, two correct settings. A candidate who cannot explain this is likely not the author of `run_guidance_diagnostics`.

**28–30.** 🔒 Judgement questions. Useful probes for the answers: `src/models.py:166-210` (`build_generator` — dead code importing a non-existent module, with a docstring admitting it is "a best guess"); `src/models.py:217-273` (`CXRDataset` — a class whose docstring says "If reflex7/cxr-data-set actually ships as RSNA-style CSV + DICOM … this class needs to be rewritten to parse that format — say so and it'll be adjusted", which is unmistakably a conversational artefact left in source); and `config.CXR_DATA_DIR`, documented as "unused by the current main.py pipeline." A candidate who nominates these as the things they'd hand over least willingly, or cut first, is being honest about the repo's seams.

---

# Deep Technical Questions — Answers

**D1. Stochastic observations.** `encoder()` returns `μ + σ⊙ε` from the VAE posterior on every call, and `get_observation()` calls it every step. Effects: (i) `V(s)` is regressed onto targets from a state that isn't a deterministic function of the environment, adding irreducible variance to the critic — a floor on `L_value` no amount of training removes; (ii) the PPO ratio is *unaffected*, because `logp_old` and `logp_new` are both conditioned on the **same stored** observation `t.obs`, so the noise is frozen at collection time and the importance weight is consistent; (iii) `deterministic=True` at eval does not give deterministic *behaviour*, only a deterministic map from a stochastic observation. SD1.5's posterior σ is small relative to μ, so amplitude is low — but it is injected into 4096 of 73 731 dims every step. `.mode()` in `get_observation` is strictly better (no reason to add noise to an RL state) and costs nothing; leave `generate_counterfactual` on `.sample()`, where SDEdit immediately adds far more noise anyway.

**D2. The DDIM Jacobian.** For η = 0, `z_{t−1} = √ᾱ_{t−1}·ẑ₀ + √(1−ᾱ_{t−1})·ε̂`. Holding `ε̂` fixed, `∂z_{t−1}/∂ẑ₀ = √ᾱ_{t−1}·I`, so a gradient computed w.r.t. `ẑ₀` must be scaled by `√ᾱ_{t−1}` before being added to `z_{t−1}`. On SD1.5's schedule `√ᾱ` runs from ≈0.05 at t≈1000 to ≈0.999 at t≈0 — a ~20× monotone increase. Omitting it means a constant `ALPHA_CLS` delivers *much* weaker effective guidance early and full strength late. Because `guidance_weight` also ramps up, the two compound: guidance is extremely back-loaded, roughly as `(i/n)·√ᾱ_i`. Consequence for tuning: `ALPHA_CLS` is effectively fitting the *integral* of a curve whose shape is unmanaged, so any value found by hand is specific to the `strength` distribution the policy happens to use.

**D3. The ramp direction.** The code ramps guidance **up** as denoising proceeds (weakest at high noise). Standard practice in guided diffusion is the opposite intuition — early steps determine global structure, so that is when guidance has leverage; late steps only refine texture, so late guidance produces high-frequency, adversarial-looking artefacts. **Attack:** back-loaded guidance is a mechanism for exactly the failure this project measured (small-amplitude, wide-support, high-frequency edits with low transfer). **Defend:** at high noise, `ẑ₀` is a poor estimate of the clean image, so decoding it and evaluating a clean-image classifier gives a gradient computed on a blurry, off-distribution reconstruction — the guidance signal is genuinely unreliable there, and the ramp is a crude confidence weighting. Also, the docstring says the ramp replaced a *hard on/off switch at the midpoint* which "could kick a step into a noisy collapse," so the ramp fixed a real instability. **Resolution:** the two arguments point at different things — the ramp should probably be a *band* (off very early, strong in the middle, tapering late), and `ramp_start_frac` being a learned action means the policy could discover that if the action also controlled a ramp *end*. It currently cannot; guidance always reaches full strength at the final step.

**D4. Is the deviation bound true?** The claim (`config.py:83-90`) is that because the final step composites `z_final = m·z_t + (1−m)·z_anchor`, the per-location output latent deviation is bounded by `m`. **Disproof as configured:** `m ≥ floor = 0.15` everywhere, so the bound is `‖z_final − z_anchor‖ ≤ m·‖z_t − z_anchor‖` with `m ≥ 0.15` — it bounds deviation *proportionally*, but never to zero, and `‖z_t − z_anchor‖` is itself unbounded because `z_t` is a free trajectory. **Partial proof:** the bound is genuine and tight in the per-location, per-step sense — at the final composite, a location with `m = 0` is pinned exactly to `z_anchor`. So the statement is true as written for `m = 0` and monotone in between; it is the *reachability* of `m = 0` that `GRADCAM_MASK_FLOOR = 0.15` removes. **Also**: it bounds deviation in *latent* space, and the VAE decoder is not 1-Lipschitz, so it says nothing quantitative about pixels.

**D5. Bias from regressing V onto one-step reward.** True value `V^π(s_t) = E[Σ_{k≥0} γ^k r_{t+k}]`. The fitted critic converges to `V̂(s_t) ≈ E[r_t]`. The bias is the discounted tail `E[Σ_{k≥1} γ^k r_{t+k}]`, which for near-constant per-step reward `r̄` and remaining horizon `H−t` is `r̄·γ(1−γ^{H−t−1})/(1−γ)`. At `r̄ = −1.59`, `γ = 0.99`, `H = 8`: at `t = 0` the bias is ≈ `−1.59 × 6.79 = −10.8`; at `t = 7` it is 0. So the bias is **not constant** — it decays linearly toward the horizon. Advantage normalisation removes the *mean* but not this within-episode gradient, so the resulting advantages carry a spurious systematic trend that makes early steps look worse than late ones relative to truth. That is a real, directional distortion: it biases the policy toward whatever behaviour it exhibits late in an episode.

**D6. Squashed-policy entropy.** For `u = σ(a)` with `a ~ N(μ, σ²)`, the change of variables gives
`H[u] = H[a] − E_a[ log |du/da| ] = H[a] − E_a[ log(σ(a)(1−σ(a))) ]`.
Since `σ(a)(1−σ(a)) ≤ 0.25`, `−log(·) ≥ log 4 > 0`, and as `|μ| → ∞` the term `→ +∞`, meaning `H[u] → −∞` (the density concentrates at an endpoint). The implemented bonus uses `H[a] = Σ ½log(2πeσ_i²)`, which depends only on `σ` — **completely insensitive to μ**. So the optimiser can maximise the bonus while driving `μ` into the flat tail, collapsing effective action entropy to ~0. The uncorrected bonus does not merely fail to prevent saturation; combined with any reward gradient pushing `μ` outward, it actively subsidises it by removing the only counter-pressure. (Note the sign convention in the code: `L_total = … − C_ENTROPY·entropy`, so maximising entropy. Correct.)

**D7. Advantage-normalisation variance at n = 64.** `adv = (adv − μ̂)/(σ̂ + 1e-8)` with `μ̂, σ̂` estimated from 64 samples. For roughly Gaussian advantages, `Var(σ̂)/σ² ≈ 1/(2n) = 0.0078`, so `σ̂` has ~8.8 % relative standard deviation. Since the policy-gradient magnitude scales as `1/σ̂`, the **effective learning rate fluctuates by ~9 % (1 s.d.) update to update**, purely from estimation noise. Correlated advantages within an episode (8 consecutive steps of the same trajectory) reduce the effective sample size well below 64 — with 8 episodes, the effective n for the *mean* is closer to 8 than 64 if within-episode correlation is high, pushing the relative error toward 25 %. That is not fatal but it is a meaningful noise source, and it argues directly for larger `episodes_per_update` (A8.1).

**D8. `alpha/r`.** The adapter contributes `(α/r)·B·A·x`. With `α = r`, scaling is 1.0 and the adapter's contribution magnitude is rank-independent — which is exactly what you want when *comparing* ranks, because r=8 and r=16 runs then differ only in expressiveness, not in effective step size. `α/r > 1` amplifies the adapter, which helps when the LoRA must make a *large* distribution shift and the base model actively resists it (strong domain transfer, few epochs). Is this regime one of them? Arguments for: SD1.5 natural-RGB → grayscale CXR is a large domain shift. Arguments against: 30 epochs over 26 684 images at lr 1e-4 is a substantial optimisation budget, so the adapter has time to grow its own weights and doesn't need the multiplier; and `α/r > 1` effectively raises the learning rate on the adapter path, which interacts with AdamW's per-parameter scaling in ways that make it a poor substitute for just tuning `DENOISE_LR`. Verdict: `α = r` is the right default here; if under-fitting is suspected, raise `DENOISE_LR` or the rank, not `α`.

**D9. 32×32 latents on a 64×64-trained UNet.** Three effects. (i) **Self-attention receptive field**: SD1.5's UNet applies self-attention at several resolutions; at 32² input the deepest attention blocks operate on 4×4 or 8×8 grids instead of 8×8/16×16, so the model's learned notion of "global context" now spans a physically larger fraction of the image. Objects the model learned to compose at a given token count are now composed at half the linear resolution. (ii) **Convolutional scale statistics**: the ResNet blocks learned filters tuned to the spatial frequencies present at 64² latents; at 32² the same anatomical structure occupies half as many latent cells, so it presents at frequencies the model has less training density for. (iii) **No positional embedding issue** — SD1.5's UNet uses convolutions and unpositioned self-attention spatially (only the timestep gets an embedding), so unlike a ViT there is no interpolation problem. Net: the model is operating off-distribution but degrades gracefully, which is why it works at all. The clean fix is `IMG_SIZE = 512` (→ 64² latents), which quadruples UNet cost — and given the pipeline is already at 4.4 min/image, that is why it wasn't done.

**D10. Single-token cross-attention.** With sequence length 1, `A = softmax(QK^T/√d) ∈ ℝ^{HW×1}` has every row summing to 1 over a single entry ⇒ `A ≡ 1`. Therefore `out = A·V = V` broadcast to all `HW` positions, and `Q` (hence `to_q` and the query projection's LoRA) has **no effect on the output**. `K` likewise only enters through the softmax that was annihilated. So cross-attention degenerates to `x ← x + W_out·(W_v·c)` — a per-layer learned bias vector determined entirely by the 768-d conditioning token. The model can still be conditioned (the bias differs per image via `psi_x`), but it has no *spatial* conditioning mechanism at all: the same vector is added everywhere. That is a strong limitation for a task whose whole point is a spatially localised edit — and it is why all the localisation has to come from the guidance gradient and the masks instead.

**D11. Bounding the conditioning leak.** `z_t = √ᾱ_t z₀ + √(1−ᾱ_t) ε`, so `ε = (z_t − √ᾱ_t z₀)/√(1−ᾱ_t)`. Given `z_t` (an input) and a pooled `P z₀` (in the conditioning), the model can reconstruct the component of `ε` lying in the row space of `P`, with error scaling as `√ᾱ_t/√(1−ᾱ_t)` times the reconstruction error of `z₀` from `P z₀`. `P` is a 64× average pool over 4096 dims, so it preserves the block-mean subspace — dimension 64 of 4096, i.e. **1.6 %** of the coordinates, but the *low-frequency* ones, which carry disproportionate variance in natural latents. The leak's strength scales as `√(ᾱ_t/(1−ᾱ_t))`, which is ≪1 for most of the schedule and blows up only as `t → 0`. Since `t_idx ~ Uniform(0, 50)` in training, only a small fraction of samples sit in the high-leak regime. **Conclusion: the leak is real but bounded and probably minor** — which is a more useful answer than either "it's fine" or "it's fatal." The `randperm` shuffle test settles it empirically in one epoch.

**D12. Why the YOLO KL is not a divergence.** `softmax` requires a categorical support; the channel axis of a pooled CNN feature vector is not one — channels are not mutually exclusive outcomes, the values are not log-probabilities, and the map is not invariant to the arbitrary affine reparameterisation that any preceding BatchNorm could apply. The resulting quantity is a temperature-1 soft-argmax comparison dominated by the largest activation, and it is asymmetric with no reason to prefer that direction. Replacement with matching scale: `d_cos = 1 − ⟨O_x, O_next⟩/(‖O_x‖‖O_next‖)` — bounded in [0, 2], symmetric, scale-invariant, and no underflow (hence no `clamp` needed). To match the current term's numerical magnitude, calibrate `ETA_OBJ` so `E[d_cos]` on a representative batch equals the current `E[kl_obj]` — a five-minute measurement.

**D13. Is the CV-probe holdout inside the eval set?** Trace it:
- `_far_indices` = all `i` with `|f(x_i) − c| > 0.5`, in **dataset index order**; `_far_indices_eval = _far_indices[:24]`.
- `scan_groups` walks the dataset with `shuffle=False`, collecting `f(x) < 0.2` into `Xn` and `f(x) > 0.8` into `Xp`, breaking once both reach `n_eval`. So `Xn` = the **first 200** images with `f < 0.2`, in dataset index order.
- For the n→p env (`c = 1.0`), the far condition is `f < 0.5`, which **strictly contains** `f < 0.2`.
Therefore the first 24 far-pool indices are the first 24 images with `f < 0.5`, and every one of those with `f < 0.2` also appears in `Xn` — and, being among the earliest such indices, appears in the *first* positions of `Xn`, well inside the first 200. Given the ensemble puts 15 267 / 26 684 images below 0.5 and Xn needs only 200, the leading indices are dense in both sets. **Conclusion: overlap is near-certain, and the "held-out" probe images are almost entirely inside the evaluation set.** Consequence: the safeguard is not measuring generalisation — it is measuring training-set CV on images the policy also trains near, so it would under-detect exactly the overfitting it exists to catch. Verification is five lines:
```python
env = MedicalCounterfactualEnv(..., target_c=1.0)
xn, xp = scan_groups(dataset, classifier, 200)
# rebuild indices for xn, intersect with env._far_indices_eval
```
Cheapest fix: `_far_indices_eval = _far_indices[-24:]` (take the tail, far from `scan_groups`'s prefix) — one character, and it removes the overlap almost entirely.

**D14. No train/test split.** What is measured: **in-sample counterfactual quality**. PPO's rollouts draw from `_far_indices_train`, which is the whole far pool minus 24; `scan_groups` scans the same `dataset` object; so every Xn/Xp image was available to PPO during training and, at 800 episodes over an 11–15 k pool, roughly 5 % of them were actually visited. That last number is the saving grace — the sampling is sparse enough that severe memorisation is implausible, and the policy has only 198 k parameters. But "probably not overfit" is not a measurement, and the LoRA generator *did* see all 26 684 images for 30 epochs.
**Minimal fix that preserves existing checkpoints:** split by patient ID at the `RSNACXRDataset` level with a deterministic hash (`int(sha1(pid), 16) % 10 < 8` → train), expose `dataset.test_indices`, and have `scan_groups` take an optional index whitelist. PPO's env keeps using the train split (so existing policies remain validly trained), and evaluation moves to the held-out 20 %. The existing checkpoints stay usable because nothing about the policy changes — only which images it is evaluated on. Cost: one re-run of the eval stage (~15 h across 8 shards). This is the single highest-value 30-line change in the repository.

**D15. Formalising the FID circularity.** Let `g = ∇_x log p_f(y|x)` be the guidance direction and `φ` the FID feature map. Generation follows `g`; FID measures `‖μ(φ(X_cf)) − μ(φ(X_real))‖² + tr(...)`. Circularity arises because `f = h ∘ φ_f` where `φ_f` is the same DenseNet trunk that `φ` uses, so `g = ∇_x (h ∘ φ_f)(x) = J_{φ_f}(x)^T ∇h`, i.e. **the guidance direction lies in the row space of the FID feature Jacobian by construction**. Moving `f`'s output necessarily moves `φ_f`'s features, so FID cannot be an independent measure of realism. Feature space that breaks it without a domain gap: `densenet121-res224-mimic_ch` or `-pc` — chest X-ray trained (no domain gap), different corpora, disjoint from `GUIDANCE_ENSEMBLE_WEIGHTS` and from `heldout_validation`'s current validator set only if you rotate that set again. Explicitly **not** `-all` (contains RSNA) and **not** `-chex` (now in the guidance ensemble).

**D16. `eps·I` before `sqrtm`.** With N = 200 samples in 1024 dims, `Σ̂` has rank ≤ 199, so `Σ̂₁Σ̂₂` has ≥ 825 zero eigenvalues. `scipy.linalg.sqrtm` on a matrix with a large null space and floating-point perturbation returns eigenvalues with small imaginary parts and unstable signs, producing spurious contributions to `tr(2(Σ₁Σ₂)^½)`. Adding `1e-6·I` shifts every eigenvalue away from zero, making the square root well-defined. The **bias** introduced: each of the 1024 eigenvalues is raised by ε, so `tr(Σ₁+Σ₂)` increases by `2·1024·1e-6 = 2.05e-3`, and `tr(2(Σ₁Σ₂)^½)` increases by roughly `2·1024·1e-6 = 2.05e-3` as well — these largely cancel, leaving a net bias of order 1e-6 to 1e-3. Against measured FID values of 60–104, that is **negligible**. The instability it prevents is not: without it, `sqrtm` can return imaginary components of order 1 (the code warns if `max_imag > 1e-3`). So the trade is overwhelmingly favourable. The *real* small-N problem is not ε — it is that `Σ̂` is a poor estimate, biasing FID **upward** by an amount that scales as `d/N` ≈ 5 here. That bias is not addressed by ε and is why N ≥ 1000 is recommended.

**D17. Decomposing concentration 0.282.** Model `|Δ|` as a mixture: fraction `p` of pixels at amplitude `A` (focal core), the rest at amplitude `a` (diffuse tail), `A ≫ a`. Total mass `M = pA + (1−p)a` per pixel. If `p ≤ 0.05`, the top-5 % set contains the entire core plus `(0.05−p)` of the tail, so
`C = [pA + (0.05−p)a] / [pA + (1−p)a]`.
Solving `C = 0.282` with the measured `frac_changed = 0.66` and MAE = 0.042: taking `p = 0.02`, `C = 0.282` requires `A/a ≈ 19`. So the data are consistent with **~2 % of pixels carrying a ~19× amplitude core, with ~64 % of pixels carrying a diffuse wash** — a real lesion-scale edit sitting on top of a global perturbation. A two-parameter statistic that separates them: report `(p*, A/a)` obtained by fitting the Lorenz curve of `|Δ|` (cumulative mass vs cumulative pixel fraction) with a two-segment model, or more simply report **both** `C_{0.01}` and `C_{0.20}` — a pure focal edit has `C_{0.01} ≈ C_{0.20} ≈ 1`, a pure diffuse edit has `C_{0.01} = 0.01, C_{0.20} = 0.20`, and the mixture separates them. Two extra lines in `edit_concentration`.

**D18. nih 0.205 vs chex 0.590 on the same counterfactuals.** Two competing explanations:
- **(H1) Label quality.** NIH ChestX-ray14's Pneumonia label is NLP-mined from radiology reports with ~1 % prevalence and known high noise; CheXpert's is also NLP-derived but from a curated labeller with far higher prevalence. A noisier label ⇒ a noisier decision function ⇒ smaller and less reliable `Δ` regardless of what the edit contains. `heldout_validation.py:103-104` flags nih as "the weakest of the three" for exactly this reason.
- **(H2) Feature-space overlap.** chex has 11 output heads vs nih's 14, but more importantly a different training distribution; if chex's pneumonia features happen to be more correlated with rsna's, the edit transfers better for reasons of representational similarity rather than semantic content.
**Distinguishing experiment:** measure each validator's *baseline agreement* with the guidance model on **real** images at the level of `Δ` sensitivity, not just rank correlation. Concretely: take real Xn and real Xp images, compute `Δ_real = f_val(x_p) − f_val(x_n)` for matched pairs, and normalise each validator's transfer ratio by its own real-image `Δ`. If nih's low ratio disappears after normalisation, it was H1 (the validator is just less responsive to pneumonia generally). If it persists, it is H2. The existing Spearman control (0.697 vs 0.707 — nearly identical) already argues *against* a gross usability difference, which tilts toward H2 — but Spearman is rank-based and insensitive to the *magnitude* compression H1 predicts, so it does not settle it.

**D19. Selection on the test set.** **Against:** chex was chosen for the guidance ensemble at least partly because it was the better-transferring validator (0.590 vs 0.205 on n→p). Using a held-out evaluator's performance to select a training component is textbook test-set leakage: the remaining validators (mimic_ch, pc, nih) were not chosen this way, but the *ensemble* has now been fitted to a signal derived from held-out data, and any subsequent improvement in transfer to chex-like models is partly guaranteed rather than discovered. **For:** the stated rationale is not "chex scored well" but a structural argument — chex has 11 heads vs rsna's 2, and multi-task training resists the global-brightness shortcut; that argument is independent of the measured ratio and was made in `xrv_classifier.py` before the ensemble existed. Also, the goal is not to maximise a benchmark but to make the adversarial shortcut geometrically harder, and adding *any* independently-trained model does that. **Resolution:** the decision is defensible on the structural argument alone, but the transfer ratio for chex must never again be reported as evidence — and `config.py:36-38` explicitly enforces exactly that ("anything listed here is NO LONGER a valid held-out validator"). The candidate got this right; the residual risk is that the *choice* of which model to promote was informed by the number.

**D20. BCE-of-mean vs mean-of-BCE.** Let `p̄ = (p₁+p₂)/2`. Guidance uses `L = BCE(p̄, c)`, so `∂L/∂x = ∂L/∂p̄ · ½(∂p₁/∂x + ∂p₂/∂x)` — the two members' input gradients are averaged with **equal** weight, scaled by a single shared factor `∂BCE/∂p̄ = (p̄−c)/(p̄(1−p̄))`. If member 1 is saturated (`p₁ ≈ 1`, `∂p₁/∂x ≈ 0`), its contribution to the sum vanishes *automatically*, so the gradient is carried by member 2. That is the sense in which "the mean already routes gradient to the lagging member" — it is true, but passively: the saturated member drops out rather than the lagging one being up-weighted.
Mean-of-BCE, `L' = ½(BCE(p₁,c) + BCE(p₂,c))`, gives `∂L'/∂x = ½Σ_i (p_i−c)/(p_i(1−p_i)) · ∂p_i/∂x`. Here the per-member factor `1/(p_i(1−p_i))` **explodes** as `p_i → 0` or 1, actively amplifying whichever member is furthest from the target. That is stronger up-weighting of the lagging member — and also numerically dangerous, since it amplifies a member whose input gradient is simultaneously vanishing (0 × ∞). Verdict: BCE-of-mean is the safer default and the code's claim is correct; `per_member` is the right escape hatch and correctly documented as unused.

**D21. Testing "uniform attention" vs "vanished gradient."** They are distinguishable because the *pre-ReLU* map differs. A genuinely uniform attention gives `Σ_c w_c A_c ≈ const > 0` with nonzero variance in the individual `w_c A_c` terms; a vanished gradient gives `w_c = 0` for all `c`, so **every** term is exactly zero. Test: before the ReLU and before normalisation, compute `s = Σ_c |w_c| · ‖A_c‖`. If `s < τ_machine` (say 1e-12 relative to the activation scale), the gradient vanished. If `s` is substantial but `Var_{spatial}(Σ_c w_c A_c)/Mean² < δ`, attention is genuinely flat. Two extra lines in `grad_cam`, and it would let `DegenerateCAMError` carry which of the two happened — which matters, because a vanished gradient means "resample or hook the logit" while genuinely-flat attention means "this image has no localisable evidence," and only the first is a bug.

**D22. A non-vacuous direction boost.** Any state-and-action-independent scalar on the whole reward is removed by advantage normalisation. To give the harder direction more signal without changing its optimal policy, use **potential-based shaping** (Ng et al. 1999): `F(s, s') = γΦ(s') − Φ(s)` for any `Φ`, which provably leaves the optimal policy unchanged while altering the learning dynamics. Concretely: `Φ(s) = −k·d_t` with `k` larger for n→p. This is dense, state-dependent, survives normalisation, and is guaranteed policy-invariant. (Note `R_prog = d_t − d_next` is *already* exactly this shape with `γ = 1` — so the mechanism is present and merely under-weighted at `λ_prog = 1.5`.) The correct "direction boost" is therefore **`λ_prog` per direction**, not a global multiplier. Alternatives that also work: per-direction `C_ENTROPY` (more exploration where the problem is harder) or per-direction `N_EPISODES` — both change learning without touching the objective, which is the config's own suggestion.

**D23. When `R_fid` and `R_sparsity` are linearly dependent.** `R_fid = −(1/N)Σ|Δ_i|`, `R_sparsity = −(Σ w_i|Δ_i|)/(Σ w_i)`. These are proportional for all `Δ` iff `w` is constant, i.e. `w_i = w̄ ∀i` ⟺ the CAM is spatially constant ⟺ precisely the `DegenerateCAMError` condition. For non-constant `w` they are distinct functionals, but their *correlation over the realised distribution of `Δ`* can still be near 1 if edits are roughly uniform. Measured: `R_sparsity/R_fid = 0.0125/0.0145 = 0.862`, versus 1.0 under exact proportionality — a 14 % departure. **Implication for the ablation:** with `λ_fid = 2.0` already applying an L1 magnitude penalty, turning `USE_SPARSITY_REWARD` on adds a term that is 86 % collinear with one already present. The ablation therefore measures "L1 penalty at weight 2.0 vs ~3.7 with a 14 % spatial tilt," not "sparsity vs no sparsity." To make it interpretable you must either zero `LAM_FID` in the sparsity-on arm or report the arms at matched total magnitude-penalty weight.

**D24. MS-SSIM scale count.** At 256² with `window_size = 11`: `max_levels = floor(log2(256/11)) + 1 = floor(4.54) + 1 = 5`, and `levels = min(5, 5) = 5`. So **the full standard 5-scale MS-SSIM is used and the renormalisation is a no-op** (weights already sum to 1.0001). The adaptive branch only engages below 176 px. When it *does* engage — say 3 levels — renormalising `[0.0448, 0.2856, 0.3001]` to sum 1 produces a measure that weights coarse structure far more heavily than standard MS-SSIM, so values are **not** comparable to published MS-SSIM or across image sizes. The guard is correct (it prevents pooling below the window size) but the resulting number should be labelled with its level count. Here it doesn't matter; the code is fine.

**D25. `(ρ+1)/2` as a preservation floor.** Uncorrelated high-pass images score 0.5, and *anti*-correlated ones score 0. Is 0.5 a sensible floor? Two problems: (i) the metric is not zero at "no preservation" — it is 0.5, so the dynamic range of the useful region is [0.5, 1.0] and a reported 0.798 sits only **60 %** of the way up that range, not 80 %; (ii) it is possible to score *below* the no-information floor by inverting edges, which is meaningless as "preservation." Interpreting 0.798 (p→n): rescaled to the useful range, `(0.798 − 0.5)/0.5 = 0.596` — so p→n retains roughly 60 % of the original high-frequency structure by this measure, versus n→p's `(0.938 − 0.5)/0.5 = 0.876`. Stated that way the direction gap is much starker than the raw numbers suggest. Better: report raw `ρ` (in [−1,1]) and let 0 mean "no relationship," or clamp negatives to 0 and report `ρ⁺`.

**D26. CTR column-span sensitivity.** `heart_cols = np.where(heart_mask.any(axis=0))[0]`, `cardiac_diam = heart_cols[-1] − heart_cols[0]`. A **single** false-positive pixel at column 0 or column W−1 sets the diameter to nearly the full image width. `any(axis=0)` means one pixel in a column is enough. The estimator is therefore the **range** of a thresholded mask — maximally sensitive to outliers, with breakdown point 1/n. Given `sigmoid(seg_out) > 0.5` on a PSPNet applied to VAE-reconstructed counterfactual images (off-distribution for the segmenter), stray activations are likely. This plausibly explains `std_xc_pn = 0.131` versus `std_xp = 0.101` — the counterfactuals have *higher* CTR variance than the reals, which is what a range statistic on a noisier mask produces.
Robust alternative: take the 2nd–98th percentile of columns weighted by per-column pixel count, or `cardiac_diam = max over rows of (row-wise span)` restricted to rows with ≥ k mask pixels, or simply take the largest connected component before measuring. Any of these has a nonzero breakdown point. One line with `scipy.ndimage.label`.

**D27. Inverting SCP.** `sigmoid(x) = 0.4794 ⇒ x = ln(0.4794/0.5206) = −0.0826 ⇒ ‖feat‖ = 50 × 0.0826 = 4.13`. For the std: `dσ/dx = σ(1−σ) = 0.4794 × 0.5206 = 0.2496`, so `σ_x = 0.000797/0.2496 = 0.00319` and `σ_‖feat‖ = 50 × 0.00319 = 0.160`. **Feature norms are 4.13 ± 0.16 — a 3.9 % relative spread.**
So the metric is *not* saturated: at 0.479 the sigmoid is within 4 % of its steepest point (max derivative 0.25 at 0.5). The degeneracy has two other causes: (i) the pooled YOLO feature norm on the lower third of a chest X-ray genuinely barely varies across images — COCO features have no reason to respond to costophrenic blunting; (ii) the `/50` divisor compresses whatever variation exists by 50×, mapping a 0.32-wide norm range onto a 0.0064-wide logit range. **The codebase's own diagnosis ("the sigmoid proxy is saturated") is wrong**, and a candidate who derives this has out-reasoned their own comment. A divisor of ~0.5 instead of 50 would put the norms across the sigmoid's full useful range — though it would still be measuring a COCO feature norm, so it would be a *sensitive* meaningless metric rather than an insensitive one.

**D28. Where fp16 precision is lost, and the fix.** The backward chain is `L_pres → x̂ → decoder → z0_hat`. Folding `B = ALPHA_PRES·pres_scale ≈ 0.015–0.075` into the loss multiplies the seed gradient `∂L/∂x̂` by `B` *before* it propagates through ~15 decoder layers, each of which involves products of activations and weights held in fp16. The smallest fp16 normal is 6.10e-5; intermediate gradient magnitudes in a VAE decoder backward at 256² are commonly 1e-4 to 1e-6 already, so a further ×0.015 pushes a substantial fraction into the subnormal range where relative precision degrades from ~1e-3 to arbitrarily bad, and eventually to zero at 5.96e-8. Unfused, the same chain propagates at unit scale and is multiplied by `B` only at the end, in fp32 (`action` arithmetic happens after `.to(self.dtype)` but on already-computed gradients).
**Fix:** loss scaling, exactly as AMP does. Compute `∇(A·L_cls + S·B·L_pres)` with `S = 2¹⁰ = 1024`, then form `action = −w·(grad_combined_cls_part) − w·(grad_pres_part/S)` — except the two are entangled in a single backward, so the practical version is to scale **both** losses by `S` and divide the single resulting gradient by `S`: `grad = autograd.grad(S·(A·L_cls + B·L_pres), z0_hat)[0] / S`. That keeps the whole chain in fp16's normal range, gives one backward instead of two (~1.5× on the pipeline), and is exactly the trick `torch.amp.GradScaler` already applies in `finetuning.py`. Validate by comparing `action` against the two-backward version on 100 real steps: max relative deviation should drop below the 0.1 % fp16 epsilon.

**D29. What else `manual_seed` controls, and whether determinism holds.** `torch.manual_seed` reseeds the global CPU generator **and all CUDA generators**. Beyond `init_noise` and the background-lock noise it therefore also controls: LoRA dropout (`LORA_DROPOUT = 0.05`) — but the UNet is in `eval()` and generation runs under `no_grad`, so PEFT's dropout is inactive; and any other `randn`/`rand` consumed downstream in the same process. It does **not** control cuDNN algorithm selection (governed by `torch.backends.cudnn.benchmark`/`deterministic`, neither of which is set anywhere in the repo), nor non-deterministic reduction orders in cuBLAS/cuDNN kernels, nor atomics in scatter-type ops.
So **bitwise determinism does not hold on GPU**, and the repository contains direct evidence: the same deterministic far-pool scan produced 15 240 / 15 243 / 15 239 across eight shards on heterogeneous hardware (Finding F2). Seeding makes results reproducible **on identical hardware with an identical software stack**, which is enough for the shard-invariance claim only if all shards land on the same GPU model — which `--constraint="gpuh200|gpuh100|..."` explicitly does not guarantee. To actually get it: pin a single constraint, set `torch.use_deterministic_algorithms(True)`, `cudnn.deterministic=True`, `cudnn.benchmark=False`, and `CUBLAS_WORKSPACE_CONFIG=:4096:8`. That would cost throughput, which is likely why it wasn't done — but the claim in the docstring should then be softened from "byte-identical" to "identical up to floating-point non-determinism."

**D30. Minimal counterfactual as constrained optimisation, and which terms are Lagrangian.**
```
min_{x'}  ‖x' − x‖₀            subject to   f(x') crosses the boundary toward c
                                            x' ∈ manifold of real CXRs
                                            x' agrees with x outside the lung field
                                            x' preserves patient identity
```
Lagrangian relaxation maps as follows:

| Constraint | Relaxed term | λ |
|---|---|---|
| `f(x')` reaches `c` | `R_val = −d`, `R_close2 = −d²` | 2.0, 2.5 |
| ...as a hard feasibility indicator | `R_hit`, `R_stop` | 2.0, 2.0 |
| `‖x'−x‖₀` small (relaxed to L1) | `R_fid`, `R_sparsity` | 2.0, →2.0 |
| identity preserved | `R_pres` (seg L1 + obj KL) | 3.0 |
| edits inside the lung field | **not a reward term** — enforced structurally by `edit_mask` and the background lock | — |
| on-manifold | **no term at all** | — |

Terms with **no** constraint interpretation: `R_prog = d_t − d_next` is potential-based shaping (it does not encode a constraint; it is a policy-invariant reshaping of the same objective) and `R_bud = −1` is a constant (it encodes a time constraint that a fixed horizon has already made vacuous).
**Two observations fall out.** First, the locality constraint is the only one handled *structurally* rather than by relaxation — which is the project's actual methodological contribution and the right way to do it, since a hard constraint should not be traded against the objective. Second, **nothing at all encodes the on-manifold constraint**, which is exactly why FID can be 103.8 while every relaxed term is satisfied. That gap is the single clearest statement of what the reward is missing.

---

# Beyond The Project — Answers

**B1. Replacing PPO.** Ranked:
- **CMA-ES / random search over a fixed 9-d schedule.** Only 9 bounded parameters and an expensive, noisy objective — this is the textbook regime for derivative-free optimisation, and it needs no value function, no advantage estimation and no credit assignment. It also directly tests whether state-dependence is needed at all. **This should have been the baseline.**
- **SAC**, with the observation compressed before it hits the replay buffer (store `x_t` uint8 at 256² = 65 KB, or the frozen trunk's 256-d feature at 1 KB). Off-policy reuse is worth far more than PPO's simplicity at 6 400 total transitions.
- **Bayesian optimisation** per-image if per-image adaptation is the goal — 9-d is well within GP-BO's comfortable range, and each evaluation is minutes, which is exactly BO's target regime.
Changes needed in `environment.py`: for CMA-ES/BO, essentially none — `generate_via_policy` already accepts an arbitrary action source if you replace the `policy.get_action_and_value` call. For SAC, add an observation-compression path and make `get_observation` return the 32×32 mask (A6.1) to make buffering feasible.

**B2. Removing segmentation.** `S(·)` appears in four places: `edit_mask` (via `latent_mask`), `R_pres`'s `seg_term`, the 65 536-d observation channel, and CTR. Removing it leaves only the Grad-CAM mask as a spatial prior — and `generator.py:507-510` states exactly why that is insufficient: the CAM is "a coarse saliency estimate that leaks onto mediastinum/diaphragm/border," and the lung mask exists to intersect it down. So you would lose the *anatomical* guarantee and keep only a *saliency* one, from a 7×7 map. Replacements: SAM with a lung-field box prompt (zero-shot, no training); an unsupervised prior (thresholded intensity + morphology, crude but real); or letting the policy predict the mask — which converts a hard constraint into a soft one the policy can learn to disable, i.e. the wrong direction. CTR would need a separate heart localiser. Verdict: removing it is a regression unless replaced.

**B3. End-to-end training.** Requires differentiating through the full DDIM trajectory. Memory is the first blocker (50 steps × UNet activations at 32² latents), solvable with gradient checkpointing at maybe 30 % extra compute, or with the adjoint method. The **real** objections are three:
1. `n_passes` and the two extra-pass bits are discrete and non-differentiable, so part of the action space is unreachable without a relaxation.
2. It collapses to a **fixed** schedule — the whole per-image adaptivity that motivates RL disappears, unless you keep a state-dependent network, at which point you've just replaced the policy gradient with a pathwise gradient.
3. Most importantly: end-to-end training of the classifier loss into the LoRA is **exactly the recipe for the adversarial solution** that held-out validation caught. Gradient descent on `f` through the generator's own weights is a far more efficient adversarial attack than gradient descent on the sampling schedule. RL's real contribution here may be as an *implicit regulariser*: because the policy can only modulate a low-dimensional sampling schedule, it cannot express arbitrary perturbations. That is worth stating explicitly — it reframes the architecture's restrictiveness as the point rather than a limitation.

**B4. 4.4 min → 10 s (a 26× speedup).** Budget: `8 steps × 2.9 passes × ~25 DDIM steps × (1 UNet + 1 decode + 2 DenseNets + 1 YOLO + 2 backwards through decoder+classifier+YOLO)`. Attack in order of leverage:
1. **`BUDGET_T` 8 → 1 at inference.** The policy is nearly state-independent (`guidance_bias_std` ≈ 0.087 on a 0.60 range) and `frac_valid = 0 %` at step 8 anyway, so the marginal value of steps 2–8 is unmeasured. **8×** if it holds; test first.
2. **Fuse the two backwards with loss scaling** (D28). **1.5×**.
3. **Few-step distillation** (LCM / consistency) — 50 → 4 DDIM steps. **Up to 6×**, at the cost of a distillation run.
4. **Guide on a downsampled decode** (128² instead of 256²) or use a latent-space classifier so the decode disappears from the guidance path entirely. **~2×** on the dominant term.
5. **Drop YOLO from the preservation loss** (replace with MS-SSIM per A5.4) — removes one forward and one backward per guided step.
6. **Batch across images** — currently one image at a time; the UNet at 32² latents is nowhere near saturating a modern GPU.
1+2+3 alone is ~70×, so 10 s is reachable. The already-cached anchors (`S_x`, `O_x`, CAM computed once per episode/pass) are correct and should stay.

**B5. Scaling to 14 pathologies.** 14 × 2 = 28 policies under the current architecture — untenable, forcing the conditional-policy refactor already named as a limitation. `c` becomes a 14-d vector (or a (pathology, direction) pair); `cond_proj` takes `cond_in_dim = 14 + 64` with no structural change; the far-pool scan becomes per-pathology and cannot be precomputed for all of them at once (memory), so it moves to a lazy per-episode filter. The hard problems are **not** the plumbing:
- **Co-occurrence.** Pneumonia and effusion appear together. "Flip pneumonia only" is a *constrained* edit that must hold the other 13 heads fixed, which needs a new reward term penalising unintended flips — and that term is what makes the problem genuinely harder than 14 copies of the binary case.
- **The CAM becomes pathology-specific**, which is fine (it is already taken w.r.t. the pneumonia output), but the lung mask does not — cardiomegaly edits need the cardiac silhouette, which the current lung-only mask explicitly excludes. The segmentation prior would have to become pathology-conditional.
- **Evaluation cost scales by 14×** against a pipeline that already needs 8-way sharding.

**B6. Replacing SD1.5.** Highest leverage first:
- **Replace the VAE.** SD1.5's VAE was trained on natural RGB; its round-trip error on grayscale CXR is the floor on `frac_changed` and contributes to the 66 % figure (A3.3). A VAE fine-tuned on CXR — or even just fine-tuning the existing decoder on RSNA — would lower that floor directly, and it is much cheaper than retraining a diffusion model.
- **RoentGen** or a similar MIMIC-CXR-trained latent diffusion model: a genuinely in-domain prior, and it has real text conditioning, which would replace the single-token `cond_proj` with prompts and restore actual spatial cross-attention (D10).
- **SDXL**: bigger and better at ≥512², but 128² latents at 256² input would be worse, not better, and it is ~3× the compute. Not the right move here.
Ranking the VAE first is the answer that shows the candidate knows where their error floor comes from.

**B7. 3D CT.** Almost nothing transfers, and saying so is the correct answer.
- **No 2D SD prior applies.** Options: slice-wise diffusion with an inter-slice consistency loss (cheap, produces flicker); a true 3D latent diffusion model (memory-brutal); 2.5D (three orthogonal 2D views fused).
- **The observation explodes.** A 3D latent is `4 × 32 × 32 × 32`; the conv trunk becomes 3D and the mask channel becomes unmanageable at full resolution — the pooling shortcut in A6.1 becomes mandatory, not optional.
- **Segmentation gets easier** (TotalSegmentator gives 100+ structures off the shelf).
- **`ingestion.py`'s per-image min–max normalisation is outright wrong for CT** and must be replaced by HU windowing (e.g. lung window −600/1500). Per-image min–max destroys the absolute density scale that makes CT quantitative — and this is the single most concrete thing that must change, so a candidate who names it is showing real domain awareness.
- **The counterfactual framing improves.** With HU units, "minimal edit" becomes physically meaningful (a change in density measured in Hounsfield units, not in [0,1] intensity), and a nodule edit must be volumetrically coherent — a much stronger constraint than 2D, and a better test of whether the generator has learned anatomy.

---

# Using this key

**Do not read answers aloud.** Every model answer above is longer and more complete than any candidate will produce under interview conditions. The purpose is so you can recognise a correct partial answer and know which follow-up opens the next layer.

**Weight the sections.** For a 3-month undergraduate internship, Sections 1, 5, 6 and 7 carry the most signal about what the candidate actually did. Sections 3, 4 and the Deep Technical set measure ceiling, not ownership — a candidate can be a strong hire while scoring 3s there.

**The five highest-signal questions**, if you only have 45 minutes:
1. **Q5.1** (read your own reward split) — separates ownership from assembly more reliably than anything else here.
2. **Q9.1** (the circularity) — the project's central scientific idea; not knowing it disqualifies.
3. **Q6.2** (`DegenerateCAMError`) — tests whether the candidate reasons about silent failure.
4. **Q7.2** (`avg_ep_len = 1.00`) — the deepest conceptual debugging arc in the project.
5. **Q2.3** (the baseline table) — the honesty flag.

**One caution.** This repository is unusually strong on negative-result infrastructure: `heldout_validation.py`, `edit_concentration`, the SCP degeneracy detector, `DegenerateCAMError`, the shard config-stamp check, and the `is_valid`/CV criterion alignment all exist to catch the project lying to itself. That instinct is rare and valuable. But the same repository leaves the CV safeguard unexercised, the ablation matrix unseparated, the fixed-schedule baseline unrun, and a report whose numbers do not match its own artefacts. **The gap between the quality of the instrumentation and the discipline of its use is the thing to understand about this candidate** — whether it reflects time pressure, supervision, or something about how the code came to exist. Sections 5, 7 and 9 are built to tell those apart.


