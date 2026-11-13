#!/bin/bash
# F6 — build the CVPR 2027 upload from a CLEAN copy of the paper sources and run
# the anonymity and layout checks. Writes only under ~/verification/review/submission_<sha>/.
#
#   bash ~/verification/scripts/export_submission.sh
#
# Copies ONLY what the build needs (an allow-list, not a deny-list): main.tex,
# supplementary.tex, preamble.tex, cvpr.sty, ieeenat_fullname.bst, main.bib, sec/*.tex
# and the figure files the sources actually \includegraphics. silence.sty and
# cleveref.sty are copied from the paper folder (TeX Live 2018 here lacks them).
# Never copied: .git, refs/, the issues PDF, ABSTRACT_B_DRAFT.tex, unused figures,
# figures/make_fig1_lhr_lift.py.
#
# Checks (non-zero exit on failure):
#   * 0 undefined references/citations in both builds
#   * main: Conclusion on page <= 8, LLM Usage Statement after it and before References
#   * pdfinfo Author/Title/Subject/Keywords empty on both PDFs
#   * no identifying string in either PDF's text or in the copied sources
set -euo pipefail
P="$HOME/Jobs/Ayoub/ppo-contrafactual/PPO-CF_CVPR_Paper"
T="$P"
SHA=$(git -C "$P" rev-parse --short HEAD)$(git -C "$P" diff --quiet -- . || echo "-dirty")
OUT="$HOME/verification/review/submission_$SHA"
rm -rf "$OUT"; mkdir -p "$OUT/src/sec" "$OUT/src/figures"
cd "$P"
cp main.tex supplementary.tex preamble.tex cvpr.sty ieeenat_fullname.bst main.bib "$OUT/src/"
cp sec/*.tex "$OUT/src/sec/"
for f in $(grep -h -o -E '\\includegraphics(\[[^]]*\])?\{[^}]+\}' sec/*.tex | sed -E 's/.*\{([^}]+)\}/\1/' | sort -u); do
    cp "$f" "$OUT/src/$f"
done
cp "$T/silence.sty" "$T/cleveref.sty" "$OUT/src/"
cd "$OUT/src"
build() {
    pdflatex -interaction=nonstopmode "$1.tex" >/dev/null || true
    bibtex "$1" >/dev/null || true
    pdflatex -interaction=nonstopmode "$1.tex" >/dev/null || true
    pdflatex -interaction=nonstopmode "$1.tex" >/dev/null || true
    [ -f "$1.pdf" ] || { echo "FAIL: $1.pdf not built"; exit 1; }
    cp "$1.pdf" "$OUT/"
}
build main; build supplementary
fail=0
for d in main supplementary; do
    n=$(grep -c -E "Warning: (Reference|Citation).*undefined" "$d.log" || true)
    echo "[$d] undefined refs/cites: $n"; [ "$n" = 0 ] || fail=1
    pdfinfo "$OUT/$d.pdf" | grep -E "^(Pages|Title|Author|Subject|Keywords):" | sed "s/^/[$d] /"
    for k in Title Author Subject Keywords; do
        v=$(pdfinfo "$OUT/$d.pdf" | sed -n "s/^$k: *//p")
        [ -z "$v" ] || { echo "FAIL: $d $k='$v'"; fail=1; }
    done
    pdftotext -layout "$OUT/$d.pdf" "$OUT/$d.txt"
done
python3 - "$OUT/main.txt" <<'EOF' || fail=1
import sys
pages = open(sys.argv[1]).read().split('\f')
def first(s):
    return next((i for i, p in enumerate(pages, 1) if s in p), None)
c, l, r = first('Conclusion'), first('LLM Usage Statement'), first('References')
print(f"[main] Conclusion on page {c}, LLM statement on page {l}, References on page {r}, {len(pages)-1} pages")
ok = c is not None and c <= 8 and l is not None and r is not None and l <= r
if not ok:
    print("FAIL: page budget / section order"); sys.exit(1)
EOF
PAT='Arbaoui|Ayoub|Melki|Ben Yahia|Slimane|Sadok|INSA|ESI-SBA|esi-sba|Strasbourg|Sidi Bel|hpc-n|hpc-glogin|/home2020|arbaoui@|verification/|github\.com/[A-Za-z0-9_-]+/ppo'
for f in "$OUT/main.txt" "$OUT/supplementary.txt"; do
    if grep -n -i -E "$PAT" "$f"; then echo "FAIL: identifying string in $(basename "$f")"; fail=1; fi
done
if grep -rn -i -E "$PAT" "$OUT/src" --include=*.tex --include=*.bib; then echo "FAIL: identifying string in sources"; fail=1; fi
echo "[export] $OUT  (main.pdf, supplementary.pdf)"
[ "$fail" = 0 ] && echo "[export] ALL CHECKS PASSED" || { echo "[export] CHECKS FAILED"; exit 1; }
