# PPO-CF — CVPR 2026 paper

Self-contained folder: LaTeX source, compiled PDF, bibliography, official
CVPR 2026 template files, and copies of every figure used in the paper.

## Contents
- `paper.tex` — the paper source
- `paper.pdf` — compiled output (7 pages)
- `paper.bib` — references
- `figures/` — all 4 figures used in the paper (local copies)
- `cvpr.sty`, `ieeenat_fullname.bst`, `preamble.tex` — official CVPR 2026
  author-kit files
- `silence.sty`, `cleveref.sty` — minimal local stand-ins for two packages
  not installed on the machine this was built on (no network access to
  fetch them). They only affect warning suppression and the
  `\cref`/`\Cref` convenience macros — the paper's own cross-references
  are written out manually (`Table~\ref{...}`, `Section~\ref{...}`), so
  nothing in the actual paper content depends on them. If you rebuild on
  a machine with a full, current TeX Live install, you can safely delete
  both files and let the real packages be used instead.

## To rebuild
```
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```
