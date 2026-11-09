# LinkedIn post — draft (corrected numbers)

Status: DRAFT, not posted. Replaces the earlier version that quoted
"99.5% → 8%" — those numbers don't appear anywhere in the logged results.
The real numbers below are sourced directly from this repo; see the
"Source" line under each claim so they stay checkable.

---

I spent the last stretch of this project trying to fool my own experiment
— on purpose.

The setup: a PPO policy edits chest X-rays to flip a pneumonia
classifier's diagnosis (positive → negative and back), guided by that
same classifier's gradient. Early on the numbers looked great: 97% mean
Counterfactual Validity, beating a GAN-based baseline from the literature
(GANterfactual, 94.7%) while cutting FID realism-gap by 4-6x.
[Source: report/report.tex Table "Comparison with baseline counterfactual
generation methods" — Mean CV 0.970 vs 0.947, FID 5.67/3.20 vs 25.0/20.0]

Except CV is scored by the same classifier that guided the edit. That's
circular — it can't tell a genuine disease-relevant edit from a targeted
perturbation that only fools that one model.

So I ran the edits through three classifiers that never saw the guidance
process at all — trained on different hospitals' data, no shared
training signal. If the edits encode real pneumonia features, an
independent classifier should agree with the shift. If they're an
adversarial shortcut, it won't.

The result was a real split, not a clean win:
- Against the most independent classifier (NIH), only 12-14% of the
  guidance model's score movement carried over — the rest looks like an
  adversarial exploit specific to the guidance classifier.
- Against a second classifier (PC), 38-57% transferred — partial to good.
- Against a third (MIMIC), 21-27% transferred — partial.
[Source: logs/ppo-counterfactual-heldout-17689740.out, "TRANSFER RATIO"
table]

A model-free check on the edit maps backed this up: the changed pixels
are spread diffusely across the lung field (concentration 0.25-0.29)
rather than concentrated on a lesion the way a real opacity would be
(baseline for "no localization at all" is 0.05).
[Source: same log, "EDIT CONCENTRATION" section]

The fix that actually worked, across three re-run seeds: guide the
policy against an ensemble of classifiers instead of one. Harder for the
policy to find one perturbation that fools several independently-trained
models at once. Ensemble guidance was the only tested countermeasure
that raised lesion-targeting lift (measured against radiologist-drawn
bounding boxes, not any classifier) — a spatial Grad-CAM constraint I
expected to help actually made it worse.
[Source: report/report.tex, "Results: control arms, countermeasures, and
an ensemble-size sweep" — cm0 2.057 lift → cm1 (ensemble) 2.127, cm2
(Grad-CAM) 1.534, replicated across 3 seeds]

I've now made ensemble guidance the default for this pipeline going
forward. [Source: config.py, USE_GUIDANCE_ENSEMBLE]

The headline metric on its own would have told a nicer story. Checking
it against something that had no reason to agree told a truer one.

---
