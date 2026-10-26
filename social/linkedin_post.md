# LinkedIn Post

My reinforcement learning agent found a way to cheat its own reward function, and catching that turned into the real core of my internship.

I've spent the past months at SDU Sønderborg working with Professor Sadok Ben Yahia on using AI to explain AI diagnoses, by editing chest X-rays.

Here's the idea in plain terms. Take a chest X-ray a deep learning model reads as pneumonia. What's the smallest, most realistic change that would make it call the scan normal instead? Generating that "counterfactual" image shows exactly what the model is keying on, turning a black-box diagnosis into something a radiologist can actually inspect.

Three things stood out. My PPO agent was flipping the classifier's decision by repainting 70% of the image instead of the lesion. The reward turned out to be 93% flip bonus, almost nothing for staying faithful to the scan. Rebalancing it and switching the fidelity penalty from L2 to L1 fixed the problem, since L2 quietly rewards spreading small changes everywhere. I also caught a scaling bug where an asymmetry fix was silently weakening preservation on the direction that already needed it most.

The one I'm proudest of: my validity metric was circular. I was scoring counterfactuals with the same classifier that generated them, measuring how well an optimizer optimized itself rather than whether the images showed real disease. I built an independent held-out evaluation and rebuilt guidance around a two-model ensemble to close the gap.

The pipeline: a PPO agent steers a LoRA fine-tuned Stable Diffusion model on RSNA pneumonia X-rays, nudged by a classifier's gradient while a segmentation mask keeps edits anatomically honest.

Results: FID dropped from 25 to 5.67 against my strongest GAN baseline, with a 0.98 validity rate on the harder direction.

Grateful to Professor Ben Yahia for the trust and pushback, and to SDU for the room to do this properly.

Question for this crowd: for medical AI, does a visual explanation like this build more trust than a number or heatmap would?

#AI #MachineLearning #ExplainableAI #MedicalImaging #DeepLearning

---

## Image attachment order

1. `social/counterfactual_diff_n_to_p_ensemble.png` (heatmap showing where the edit lands, normal to pneumonia)
2. `social/counterfactual_diff_p_to_n_ensemble.png` (heatmap showing where the edit lands, pneumonia to normal)
3. `social/chart_baseline_comparison.png`
