#!/bin/bash
# Run ONCE on the login node before any other script here. No GPU, ~2 min.
#
# Builds code snapshots with `git archive` (read-only on the repo; it does not
# touch the repo's .git or working tree) and checkpoint "views" made of
# symlinks, so every job below reads the original weights but writes only under
# ~/verification/. Nothing here is submitted.
#
#   code/49e6050   the exact code of the committed run (its manifest: tree dirty
#                  only in checkpoints/ and report/, config.py sha256 f687e57a...)
#   code/HEAD      current branch tip; only for tools that exist only there
#                  (arm_transfer.py --bootstrap, Grad-CAM layer override)
#   views/committed   checkpoints/{n_to_p,p_to_n} as used by eval 17695832
#                     (generator = n_to_p/unet_finetuned.pt for BOTH directions)
#   views/pn_lora     same policies, but generator = p_to_n/unet_finetuned.pt
#                     (n_to_p/ deliberately has no unet so the loader falls
#                     through to p_to_n/ — see pipeline.load_generator_weights)
set -euo pipefail
REPO="${PPO_CF_REPO:?set PPO_CF_REPO to the ppo-contrafactual checkout}"
V="$HOME/verification"
mkdir -p "$V/code" "$V/views" "$V/out" "$V/logs"

for c in 49e6050 HEAD; do
    d="$V/code/$c"
    if [ ! -d "$d" ]; then
        mkdir -p "$d"
        git -C "$REPO" archive "$c" | tar -x -C "$d"
    fi
    [ -e "$d/venv" ]       || ln -s "$REPO/venv" "$d/venv"
    [ -e "$d/yolov8n.pt" ] || ln -s "$REPO/yolov8n.pt" "$d/yolov8n.pt"
    mkdir -p "$d/logs"
done
# check_classifier_localization.py writes to outputs/gradcam_check/ relative to
# its cwd; make sure that is the snapshot's own copy, not the repo's.
rm -f "$V/code/HEAD/outputs/gradcam_check/degeneracy_log.json"

mk_view () {   # $1 view name, $2 dir supplying unet+cond_proj, $3 1 = also give n_to_p a unet
    local view="$V/views/$1"; mkdir -p "$view/n_to_p" "$view/p_to_n"
    ln -sfn "$REPO/checkpoints/split_rsna_pneumonia.json" "$view/split_rsna_pneumonia.json"
    for dir in n_to_p p_to_n; do
        ln -sfn "$REPO/checkpoints/$dir/ppo_policy.pt" "$view/$dir/ppo_policy.pt"
    done
    ln -sfn "$REPO/checkpoints/$2/unet_finetuned.pt" "$view/p_to_n/unet_finetuned.pt"
    ln -sfn "$REPO/checkpoints/$2/cond_proj.pt"      "$view/p_to_n/cond_proj.pt"
    if [ "$3" = 1 ]; then
        ln -sfn "$REPO/checkpoints/$2/unet_finetuned.pt" "$view/n_to_p/unet_finetuned.pt"
        ln -sfn "$REPO/checkpoints/$2/cond_proj.pt"      "$view/n_to_p/cond_proj.pt"
    fi
}
mk_view committed n_to_p 1
mk_view pn_lora   p_to_n 0

echo "snapshots: $(ls "$V/code")   views: $(ls "$V/views")"
