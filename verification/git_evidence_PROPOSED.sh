#!/bin/bash
# APPROVED 2026-10-28 (identity passed per command). Creates git refs in $PPO_CF_REPO/.git
# and a bundle in ~/verification/. Refuses to run without --approved.
#
# What it writes into the repo's .git (nothing else in the repo is touched):
#   - blobs/trees/commits for evidence files (git hash-object -w / commit-tree)
#   - refs/heads/evidence-2026-10-28   (orphan branch: evidence files only)
#   - annotated tags under refs/tags/evidence/
# It does NOT touch HEAD, the default index, the working tree, or any branch
# other than the new one. It uses a private index under ~/verification/git/.
#
# Why an orphan, evidence-only branch instead of "one commit per arm state":
# no git object reproduces the config.py of the 3-ens, sparsity or Grad-CAM
# runs (61b26e0b / cda22868 / b257353e; all 696 blobs incl. unreachable ones
# were hashed on 2026-10-28), and the other dirty files of those runs and of
# the single-classifier run were never hashed. A commit labelled "arm state"
# would assert code that cannot be shown to have run. So each arm gets one
# commit holding its run evidence, and a *-code tag that points at the exact
# code (committed arm: 49e6050) or, for the others, at the BASE of the dirty
# tree, with the tag message saying it is only the base.
set -euo pipefail
[ "${1:-}" = "--approved" ] || { echo "not approved; nothing done"; exit 1; }

REPO="${PPO_CF_REPO:?set PPO_CF_REPO to the ppo-contrafactual checkout}"
V="$HOME/verification"
BR=refs/heads/evidence-2026-10-28
cd "$REPO"
git rev-parse -q --verify "$BR" >/dev/null && { echo "$BR exists; stop"; exit 1; }
git tag -l 'evidence/*' | grep -q . && { echo "evidence/* tags exist; stop"; exit 1; }
mkdir -p "$V/git"
# Identity passed per command with -c (approved 2026-10-28); git config is
# not modified. Every object-creating command below goes through g().
g () { git -c user.name="Anonymous" -c user.email="anonymous@example.com" "$@"; }
export GIT_INDEX_FILE="$V/git/evidence.index"
rm -f "$GIT_INDEX_FILE"

TRAILER=$'\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>'

# arm | evidence jobs | ckpt dir (rel) | shard dir | metrics json | code ref | code-tag message
ARMS=(
"committed|17694482 17695832 17695840|checkpoints|checkpoints/eval_shards_ens_default|checkpoints_ens_default/eval_metrics.json|49e6050|EXACT code of training 17694482 / eval 17695832 / reduce 17695840 (manifest: tree dirty only in checkpoints/, report/, social_media/; config.py sha256 f687e57a... = this commit)"
"single|17686489 17689478 17689479|checkpoints/archive_pre_ensemble_default_20260812|checkpoints/eval_shards|checkpoints/eval_metrics.json|cf5204f|CODE STATE NOT RECOVERABLE: the exact code of this run cannot be reconstructed from any git object; this tag points only at the base commit cf5204f of the dirty working tree, not at the code that ran. Run(s): 17686489. Its config.py sha256 ff168a23... equals config.py at 5b37471 (and 6869f3e, fb268c6, 927d6b8, 34ac79c); the other modified files were not hashed at run time"
"ens3|17700222 17700783 17700791|checkpoints_ens3|checkpoints_ens3/eval_shards|checkpoints_ens3/eval_metrics.json|35f0ea7|CODE STATE NOT RECOVERABLE: the exact code of this run cannot be reconstructed from any git object; this tag points only at the base commit 35f0ea7 of the dirty working tree, not at the code that ran. Run(s): 17700222. Run config.py sha256 61b26e0b... is reproduced by no git object (all blobs incl. unreachable hashed 2026-10-28); domains/rsna_pneumonia.py and arm_transfer.py were also modified"
"sparsity|17701295 17702127 17702135|checkpoints_sparsity|checkpoints_sparsity/eval_shards|checkpoints_sparsity/eval_metrics.json|35f0ea7|CODE STATE NOT RECOVERABLE: the exact code of this run cannot be reconstructed from any git object; this tag points only at the base commit 35f0ea7 of the dirty working tree, not at the code that ran. Run(s): 17701295. Run config.py sha256 cda22868... is reproduced by no git object; factory/ppo.py was also modified"
"gradcam|17702339 17704444 17705866 17705875|checkpoints_gradcam_fix|checkpoints_gradcam_fix/eval_shards|checkpoints_gradcam_fix/eval_metrics.json|35f0ea7|CODE STATE NOT RECOVERABLE: the exact code of this run cannot be reconstructed from any git object; this tag points only at the base commit 35f0ea7 of the dirty working tree, not at the code that ran. Run(s): 17702339/17704444. Run config.py sha256 b257353e... is reproduced by no git object; factory/ppo.py was also modified"
)

add () {  # $1 file on disk, $2 path in tree
    local b; b=$(git hash-object -w -- "$1")
    git update-index --add --cacheinfo "100644,$b,$2"
}

parent=""
for row in "${ARMS[@]}"; do
    IFS='|' read -r arm jobs ck shards metrics code msg <<< "$row"
    E="$V/git/$arm"; rm -rf "$E"; mkdir -p "$E"
    {
        echo "arm: $arm"; echo "jobs: $jobs"; echo "checkpoint dir: $ck"
        echo "code tag: evidence/$arm-code -> $(git rev-parse "$code")"; echo "  $msg"
        echo; echo "sha256 of artefacts (computed $(date -Iseconds)):"
        for d in n_to_p p_to_n; do for f in ppo_policy.pt unet_finetuned.pt cond_proj.pt; do
            [ -e "$ck/$d/$f" ] && sha256sum "$ck/$d/$f"; done; done
        sha256sum "$metrics" "$shards"/cf_shard_*.pt 2>/dev/null
        echo; echo "generator tensor diff n_to_p vs p_to_n (~/verification/generator_tensor_diff.json):"
        /usr/bin/python3 -c "import json,sys; d=json.load(open('$V/generator_tensor_diff.json')); \
k=[x for x in d if x.endswith('/$ck')]; print(json.dumps(d[k[0]], indent=1) if k else 'n/a')"
    } > "$E/EVIDENCE.txt"
    add "$E/EVIDENCE.txt" "evidence/$arm/EVIDENCE.txt"
    for j in $jobs; do
        for f in $(ls logs/*-"$j"[_.]* 2>/dev/null | sort -u); do add "$f" "evidence/$arm/$f"; done
    done
    tree=$(git write-tree)
    c=$(g commit-tree "$tree" ${parent:+-p "$parent"} \
        -m "evidence: $arm arm (jobs $jobs)" -m "$msg$TRAILER")
    g tag -a "evidence/$arm" -m "run evidence for the $arm arm (jobs $jobs)" "$c"
    g tag -a "evidence/$arm-code" -m "$msg" "$(git rev-parse "$code^{commit}")"
    parent=$c
done
git update-ref "$BR" "$parent" ""          # "" = must not exist yet

# Keep unreachable history alive so --all bundles it.
g tag -a evidence/stash-907735d -m "dropped stash of 2026-08-23 (WIP on evaluation-integrity: 35f0ea7). Post-hoc reverted state: its config.py (ebe6a72b...) matches no run" 907735d
g tag -a evidence/unreachable-2e33f34 -m "unreachable chain ab1d7a7 -> 64aa2cf -> 5b12295 -> 2e33f34 (2026-07-08..12)" 2e33f34

unset GIT_INDEX_FILE
# Free space: the bundle holds every object reachable from any ref; bound it by
# the whole object store (loose + packs, excluding tmp_pack garbage) x 2.
need=$(( ( $(git count-objects -v | awk '/^size:/{s+=$2} /^size-pack:/{s+=$2} END{print s}') * 2 ) / 1024 / 1024 + 1 ))
avail=$(df -B1G --output=avail "$V" | tail -1 | tr -d ' ')
echo "bundle: need <= ${need} GiB, available ${avail} GiB on $(df --output=target "$V" | tail -1)"
[ "$avail" -gt "$need" ] || { echo "not enough free space; refs created, bundle NOT made"; exit 1; }
git bundle create "$V/evidence-2026-10-28.bundle" --all
git bundle verify "$V/evidence-2026-10-28.bundle"
sha256sum "$V/evidence-2026-10-28.bundle" | tee "$V/evidence-2026-10-28.bundle.sha256"
git for-each-ref refs/tags/evidence refs/heads/evidence-2026-10-28
