# Research Internship Report — LaTeX source

Melki Ayoub — ISSAT Sousse, 1st Year Engineering Cycle
*PPO-Guided Latent Diffusion for Visual Counterfactual Explanations in Chest
Radiography (RSNA dataset)*

## Build

```bash
pdflatex main && pdflatex main && pdflatex main   # 3 passes: ToC/LoF/LoT + refs
```

Builds clean on TeX Live 2018 (no `titlesec`, `emptypage`, `hyphenat` or
`seqsplit` required — the chapter heading style is implemented directly against
the `report` class).

## Layout

```
main.tex                 document skeleton + bibliography
preamble.tex             packages, geometry, heading style, macros
frontmatter/
  coverpage.tex          cover page (logo slots — see below)
  acknowledgments.tex
  acronyms.tex
chapters/
  introduction.tex       General Introduction
  ch1_context.tex        Internship Context (SDU / Maersk Mc-Kinney Moller)
  ch2_environment.tex    Research & Development Environment
  ch3_tasks.tex          Tasks Performed
  ch4_results.tex        Results, Discussion, Perspectives
  conclusion.tex         General Conclusion
backmatter/
  glossary.tex
figures/                 copied from the repo (see provenance below)
logos/                   EMPTY — drop logo files here (see below)
```

## Logos

Both logos are in place on the cover:

| File | Slot | Notes |
|---|---|---|
| `logos/ISSATSo_logo.jpg` | ISSAT Sousse, cover header | source has a wide white margin, so it is trimmed (`trim=55 300 90 320,clip`) before scaling to 2.75 cm |
| `logos/SDU.png` | Host institution | 1.9 cm. The image already contains the "University of Southern Denmark" wordmark, so the text line that would repeat it is suppressed when the logo is present |

Each is wrapped in `\IfFileExists`, so the cover still builds with a text
fallback if a logo file is ever moved or renamed. If you swap either image,
re-check the trim values and the cover's vertical fit — the page is tuned to
sit just inside one page.

## Page layout

The document is `oneside`, so chapters start on the next available page and
there are **no blank filler pages** (36 pages, verified zero blanks). If you
need it double-sided for printing, change the class option in `main.tex` to
`twoside,openright` — that reintroduces blanks by design, so that every
chapter opens on a right-hand page.

## Figure provenance

All figures are copied verbatim from the repository — none were regenerated
or edited.

| `figures/` | Source in `ppo-contrafactual/` |
|---|---|
| `counterfactual_diff_n_to_p.png` | `checkpoints/` |
| `counterfactual_diff_p_to_n.png` | `checkpoints/` |
| `reward_trace.png` | `checkpoints/` |
| `gradcam_current_0.png` | `outputs/gradcam_check/` |
| `gradcam_xrv_0.png` | `outputs/gradcam_check/` |

## Numeric provenance

Every figure quoted in the report traces to one of:

- `checkpoints/eval_metrics.json` (working-tree version) — CV, FID, IP, CTR, SCP
- `checkpoints/heldout/heldout_validation.json` — held-out CV, transfer ratios,
  agreement control, edit concentration
- `checkpoints/guidance_diagnostics_{n_to_p,p_to_n}.json` — guidance bias,
  preservation suppression, average reward
- `logs/ppo-counterfactual-17653310.out` — dataset counts, LoRA parameter
  counts, observation/policy dimensions, far-pool sizes, GPU model
- `config.py` — hyperparameters and the recorded design rationale
- `run_*.slurm` — measured wall-clock runtimes, job IDs

## Deviations from the drafting brief — read this

Four points where the report deliberately departs from what was requested,
because the repository evidence did not support it:

1. **Metrics differ from `report/report.tex`.** That file reports CV mean
   0.965 and FID 5.67 / 3.20. The *committed* `eval_metrics.json` contains
   FID values of exactly `5.67` and `3.20` — two decimal places, while every
   other float in the same file carries full double precision. The
   working-tree version contains full-precision computed values of
   `103.83` / `59.73`. This report uses the working-tree values throughout.

2. **No baseline comparison.** `report/report.tex` contains two tables
   comparing against GANterfactual and Vanilla CycleGAN. No baseline
   implementation exists anywhere in the repository, and those tables'
   values are round numbers with no serialised source. They are omitted, and
   the absence of a baseline is stated as a limitation
   (Ch. 4, "Strengths and Limitations").

3. **GPU hardware.** The brief specified Tesla V100-32GB. The only GPU
   actually recorded in the logs is **Tesla P100-PCIE-16GB**, and the SLURM
   scripts record measured timings on a P100 and a Quadro RTX 5000. "V100"
   appears only once in `config.py`, as a forward-looking planning note. The
   report states the P100 and RTX 5000.

4. **No `trap`-on-EXIT or `sacct` usage.** The brief mentioned these. The
   batch scripts use `#SBATCH --mail-type=END,FAIL`, `set -euo pipefail` and
   explicit precondition checks. The report describes what is actually there.

## Dates

Internship period is rendered as **15/06/2026 – 15/08/2026**, academic year
**2025–2026**, inferred from the partial dates supplied ("15 Juin", "15
august") and cross-checked against the git history (first commit 2026-06-25,
most recent run 2026-08-03). Correct in `frontmatter/coverpage.tex` if wrong.

## Thin sections

One section is thinner than the others by necessity: **Ch. 4 §"Status of the
Countermeasures"**. The ensemble-guidance / Grad-CAM-mask / sparsity-reward
configuration was still training at the close of the internship (job
17653310), and no post-change held-out validation exists in the repository.
The report states this explicitly rather than implying the countermeasures
worked. If that run has since completed, this is the section to extend.
