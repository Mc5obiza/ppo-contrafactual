"""
eval_main.py on a FIXED image set: the test-fold images of another evaluation's
shards, in the same order, instead of rescanning with this code's own guidance
classifier. Issue D3: evaluate the single-classifier policy on exactly J4's 400
images, so the single-vs-ensemble comparison is paired on all of them.

    COMMON_IMAGES_FROM=<shard dir> [ADAPTER_NP_VIEW=… ADAPTER_PN_VIEW=…] \
        python eval_common_images.py --stage generate|reduce …   (eval_main.py CLI)

* scan_groups (imported by name into eval_main, eval_main.py:79) is replaced by a
  loader returning the source shards' Xn/Xp tensors and indices, after checking
  the split keys of the source stamp against the live config, every index
  against the allowed (test) fold, and every tensor against dataset[idx]
  (cyclegan_eval's checks).
* The ORDER is the source's, so _generate_shard's index_offset gives every image
  the same per-image seed as in the source evaluation.
* config.EVAL_GROUP_SOURCE is set to COMMON_GROUP_TAG (default "classifier:ens2"),
  a label only (src.metrics changes behaviour only for "label"), so the shard
  stamp records where the groups came from. Run reduce through this wrapper too,
  or eval_main._load_shards refuses the stamp.
* Group membership under THIS code's guidance classifier is printed: under
  rsna-only guidance an ens2-selected Xn image with f_rsna >= 0.5 (or Xp image
  with f_rsna <= 0.5) can never count as valid.
* With ADAPTER_PN_VIEW set, each direction generates with its own adapter
  (per_direction.py), as in J4.
"""
import os
import sys

import config  # project config first (torchxrayvision shadows `config`)

SRC = os.environ["COMMON_IMAGES_FROM"]
config.EVAL_GROUP_SOURCE = os.environ.get("COMMON_GROUP_TAG", "classifier:ens2")

import cyclegan_eval as cge   # _load_committed
import eval_main
import per_direction
from src.metrics import current_cfg_stamp, score_batch

SPLIT_KEYS = ("USE_SPLIT", "SPLIT_SEED", "TEST_FRAC")


def fixed_scan_groups(dataset, classifier, n_eval, device=config.DEVICE, batch_size=32,
                      tag="metrics", allowed_indices=None):
    xn, xp, xn_idx, xp_idx, stamp, src_n_eval = cge._load_committed(SRC)
    if n_eval != src_n_eval:
        raise ValueError(f"--n-eval {n_eval} != source n_eval {src_n_eval}")
    live = current_cfg_stamp()
    drift = {k: (stamp.get(k), live.get(k)) for k in SPLIT_KEYS if stamp.get(k) != live.get(k)}
    if drift:
        raise ValueError(f"split config differs from the source shards: {drift}")
    if allowed_indices is not None:
        allowed = set(int(i) for i in allowed_indices)
        outside = [i for i in xn_idx + xp_idx if i not in allowed]
        if outside:
            raise ValueError(f"{len(outside)} source indices are outside the allowed fold")
    md = max(float((dataset[i][0].reshape(x.shape) - x).abs().max())
             for x, i in zip(list(xn) + list(xp), xn_idx + xp_idx))
    if md > 1e-5:
        raise ValueError(f"source tensors differ from dataset[idx] (max |diff| {md:.3g})")
    xn_list = [xn[i:i + 1] for i in range(xn.shape[0])]
    xp_list = [xp[i:i + 1] for i in range(xp.shape[0])]
    f_n, f_p = score_batch(xn_list, classifier, device), score_batch(xp_list, classifier, device)
    print(f"[{tag}] FIXED groups from {SRC}: Xn={len(xn_list)} Xp={len(xp_list)} "
          f"(max |src - dataset| {md:.3g}); under this guidance classifier: "
          f"Xn f>=0.2: {int((f_n >= 0.2).sum())}, f>=0.5: {int((f_n >= 0.5).sum())}; "
          f"Xp f<=0.8: {int((f_p <= 0.8).sum())}, f<=0.5: {int((f_p <= 0.5).sum())}", flush=True)
    return xn_list, list(xn_idx), xp_list, list(xp_idx)


eval_main.scan_groups = fixed_scan_groups
if per_direction.enabled():
    per_direction.install_eval_main(eval_main)
print(f"[common-images] groups <- {SRC}; EVAL_GROUP_SOURCE stamp = {config.EVAL_GROUP_SOURCE!r}; "
      f"USE_GUIDANCE_ENSEMBLE = {config.USE_GUIDANCE_ENSEMBLE}", flush=True)
eval_main.main(sys.argv[1:])
