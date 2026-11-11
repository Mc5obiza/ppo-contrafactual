# CVPR 2027 submission: final files (anonymous, double-blind)

Built 2026-11-11 from the `PPO-CF_CVPR_Paper/` sources (review mode, "Anonymous CVPR submission").

| File | Upload to OpenReview field | Deadline |
|---|---|---|
| `PPO-CF_CVPR2027_paper.pdf` | Paper Submission | Nov 16, 2026 |
| `PPO-CF_CVPR2027_supplementary.pdf` | Supplementary Material | Nov 23, 2026 |
| `openreview_abstract.txt` | Abstract (paste; matches the PDF) | Nov 16, 2026 |

Paper: 10 pages = 8 pages of body + LLM Usage Statement + references (the limit is 8 pages excluding references).
Supplement: 13 pages.

`cvpr.sty` is byte-identical to the official CVPR 2027 kit (cvpr-org/author-kit, tag CVPR2027-v1).
That kit has no paper ID: the header shows "CVPR" only.

Checks run on these exact PDFs:
- no author names, affiliations, e-mails, local paths or code links (the only URL is the RSNA dataset citation);
- empty PDF Title/Author metadata;
- no undefined references or citations, and no `??`;
- no overfull boxes, even with `\hfuzz=\vfuzz=0pt`;
- no back-reference page numbers in the bibliography (`pagebackref` removed);
- all fonts embedded.

Build note: the local TeX Live 2018 cannot evaluate `\textwidth-12pt` (line 434 of the official cvpr.sty)
and would print "-12pt" in the header. These PDFs were built with that line changed to
`\dimexpr\textwidth-12pt\relax` in a build-only copy. Overleaf's current TeX Live needs no change.

## LaTeX source (Overleaf)

`latex_source/` (also zipped as `PPO-CF_overleaf_source.zip`) is the anonymous LaTeX for these PDFs.
In Overleaf: New Project > Upload Project > choose the zip. The compiler is pdfLaTeX, and the main document is `main.tex`.
To build the supplement, set the main document to `supplementary.tex` (Menu > Main document).
