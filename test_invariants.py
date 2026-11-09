"""
test_invariants.py
==================
CPU-only checks on the properties that make the evaluation trustworthy.

WHY THESE AND NOT OTHERS
------------------------
This is not a general test suite and does not try to be. It pins the five
invariants whose silent failure would produce a plausible-looking WRONG number
rather than a crash — the failure mode that already cost this project a
headline result:

  1. the train/test split is disjoint and patient-level
  2. scan_groups' tensors and indices stay aligned
  3. lesion_hit_rate is calibrated (uniform edit == chance, on-lesion >> chance)
  4. degenerate and uncomputed metrics never reach eval_metrics.json
  5. the shard split/merge is a lossless partition, and a partial or
     mixed-arm shard set is refused rather than reduced

A bug in any of these is invisible in a log. A bug elsewhere (a shape error, a
missing checkpoint) announces itself, and smoke_test.py already covers that
class.

Runs in ~1 minute on the login node, no GPU:

    module load python/3.12.8 && source venv/bin/activate
    python -u test_invariants.py
"""

from __future__ import annotations

import sys

import numpy as np
import torch

import config
import ingestion
import splits
import src.metrics as M


FAILURES: list[str] = []


def check(cond: bool, label: str, detail: str = "") -> None:
    if cond:
        print(f"  PASS  {label}")
    else:
        FAILURES.append(label)
        print(f"  FAIL  {label}" + (f"  ({detail})" if detail else ""))


def _small_dataset(n_per_class: int = 60):
    """Real RSNA data truncated in place, the way smoke_test.py does it."""
    ds = ingestion.build_dataset(img_size=config.IMG_SIZE)
    pos = [i for i, s in enumerate(ds.samples) if float(s[-1]) == 1.0][:n_per_class]
    neg = [i for i, s in enumerate(ds.samples) if float(s[-1]) == 0.0][:n_per_class]
    ds.samples = [ds.samples[i] for i in sorted(pos + neg)]
    return ds


# ─────────────────────────────────────────────────────────────────────────────

def test_split(ds) -> None:
    print("\n[1] train/test split")
    s = splits.create(ds, "test_domain", seed=0, test_frac=0.3)

    check(len(np.intersect1d(s.train, s.test)) == 0, "folds are disjoint")
    check(len(s.train) + len(s.test) == len(ds), "folds cover the dataset")

    tr = {ds.patient_id(int(i)) for i in s.train}
    te = {ds.patient_id(int(i)) for i in s.test}
    check(not (tr & te), "no patient appears in both folds",
          f"{len(tr & te)} shared")

    again = splits.create(ds, "test_domain", seed=0, test_frac=0.3)
    check(np.array_equal(s.test, again.test), "same seed reproduces the split")

    other = splits.create(ds, "test_domain", seed=1, test_frac=0.3)
    check(not np.array_equal(s.test, other.test), "a different seed changes it")


def test_index_alignment(ds) -> None:
    """
    The subtlest failure in the whole pipeline: if scan_groups' indices drift
    from its tensors, every annotation metric scores each edit against the
    WRONG patient's lesion and still returns a perfectly plausible number.
    """
    print("\n[2] scan_groups index alignment")

    labels = {ds.patient_id(i): float(ds.samples[i][1]) for i in range(len(ds))}
    order = [ds.patient_id(i) for i in range(len(ds))]

    class GroundTruthClassifier:
        """Reports the true label, so Xp must be exactly the real positives."""
        def __init__(self):
            self.k = 0
        def __call__(self, x):
            out = torch.tensor([[labels[order[self.k + j]]] for j in range(x.shape[0])])
            self.k += x.shape[0]
            return out

    s = splits.create(ds, "test_domain", seed=0, test_frac=0.5)
    xn, xn_i, xp, xp_i = M.scan_groups(
        ds, GroundTruthClassifier(), n_eval=10, device="cpu", batch_size=8,
        allowed_indices=s.test,
    )

    check(len(xn) > 0 and len(xp) > 0, "both groups are non-empty",
          f"Xn={len(xn)} Xp={len(xp)}")
    check(set(xn_i) | set(xp_i) <= {int(i) for i in s.test},
          "every returned index is in the eval fold")
    check(all(ds.samples[i][1] == 1.0 for i in xp_i), "Xp indices are truly positive")
    check(all(ds.samples[i][1] == 0.0 for i in xn_i), "Xn indices are truly negative")
    check(all(torch.equal(xp[k], ds[xp_i[k]][0].unsqueeze(0)) for k in range(len(xp))),
          "tensor[k] is the image at index[k]")


def test_lhr_calibration(ds) -> None:
    print("\n[3] lesion_hit_rate calibration")
    H = config.IMG_SIZE
    idx = [i for i in range(len(ds)) if ds.lesion_boxes(i)][:40]
    check(len(idx) > 10, "annotated images available", f"{len(idx)} found")
    if len(idx) <= 10:
        return

    base = [torch.zeros(1, 1, H, H) for _ in idx]

    # A spatially uniform edit must score EXACTLY chance. This is the property
    # that makes "lift > 1" mean something.
    uniform = [torch.full((1, 1, H, H), 0.05) for _ in idx]
    r = M.lesion_hit_rate(base, uniform, idx, ds)
    check(abs(r["lift"] - 1.0) < 0.02, "uniform edit scores lift 1.0",
          f"lift={r['lift']:.3f}")

    # An edit confined to the annotation must score far above chance.
    onles = [ds.lesion_mask(i).unsqueeze(0) * 0.4 for i in idx]
    r = M.lesion_hit_rate(base, onles, idx, ds)
    check(r["lift"] > 3.0 and r["p"] < 1e-6, "on-lesion edit scores well above chance",
          f"lift={r['lift']:.2f} p={r['p']:.2g}")
    check(abs(r["hit_rate"] - 1.0) < 1e-6, "on-lesion hit rate is 1.0",
          f"{r['hit_rate']:.4f}")

    # An edit that avoids the annotation entirely must score zero.
    off = []
    for i in idx:
        c = torch.zeros(1, 1, H, H)
        c[:, :, :, :] = (1.0 - ds.lesion_mask(i).unsqueeze(0)) * 0.4
        off.append(c)
    r = M.lesion_hit_rate(base, off, idx, ds)
    check(r["hit_rate"] < 1e-6, "off-lesion edit scores 0", f"{r['hit_rate']:.4f}")

    # Degenerate inputs must be reported, not silently scored.
    r = M.lesion_hit_rate(base, base, idx, ds)
    check("_omitted" in r, "identity edit is omitted, not scored")

    class NoAnnotations:
        pass
    r = M.lesion_hit_rate(base, onles, idx, NoAnnotations())
    check("_omitted" in r, "dataset without annotations is omitted")
    check("_omitted" in M.lesion_hit_rate(base, onles, None, ds),
          "missing indices are omitted")


def test_serialisation() -> None:
    print("\n[4] results serialisation")
    results = {
        "CV": {"n_to_p": 0.99, "p_to_n": 0.95, "mean": 0.97},
        "FOP": {"n_to_p": None, "p_to_n": None, "mean": None},
        "SCP": {
            "mean_xn": 0.479, "std_xn": 0.0008,
            "t_n": -2.89, "p_n": 0.0042,
            "mean_delta_n": -0.00014, "std_delta_n": 0.0007,
            "degenerate": True, "relative_spread": 0.0023,
        },
    }
    out = M._strip_unreportable(results)

    check("_omitted" in out["FOP"], "uncomputed FOP is omitted, not serialised as null")
    check("p_n" not in out["SCP"] and "t_n" not in out["SCP"],
          "degenerate metric loses its t-tests")
    check("mean_xn" in out["SCP"], "degenerate metric keeps its group means")
    check(out["CV"] == results["CV"], "a healthy metric passes through untouched")


def test_shard_roundtrip() -> None:
    """
    The sharded eval path is the ONLY way the full n_eval=200 run can execute
    (generation exceeds the 24h partition cap single-process), yet every smoke
    test so far has exercised --stage all. These checks cover the two halves
    that a smoke test cannot reach cheaply: that the slice function partitions
    exactly, and that reduce refuses every shard set it must not pool.

    A missed check here does not crash — it silently shortens Xn/Xp, and FID's
    bias grows sharply as N falls. That is the same shape of failure as the
    hand-edited FID this project is recovering from.
    """
    print("\n[5] shard split/merge")

    import glob as _glob
    import os
    import tempfile

    import eval_main as E

    # ── the slice is a partition, for every shape the array can take ─────────
    bad = []
    for n in range(0, 33):
        for k in range(1, 9):
            covered = []
            for s in range(k):
                lo, hi = E._shard_slice(n, s, k)
                if lo > hi:
                    bad.append(f"n={n} k={k} s={s}: lo>hi")
                covered.extend(range(lo, hi))
            if covered != list(range(n)):
                bad.append(f"n={n} k={k}: covered {covered} != range({n})")
    check(not bad, "shard slices tile [0,n) with no gap or overlap",
          f"{len(bad)} bad cases, first: {bad[0] if bad else ''}")

    check(E._shard_slice(200, 7, 8) == (175, 200), "final shard reaches n",
          str(E._shard_slice(200, 7, 8)))

    # ── reduce merges shards back into the original order ────────────────────
    stamp = M.current_cfg_stamp()

    def _write(d, s, k, n_eval=4, cfg=None, idx=True, total=8):
        """One synthetic shard: images are their own global index, so the
        merged order is checkable by value."""
        lo, hi = E._shard_slice(total, s, k)
        blob = {
            "shard": s, "num_shards": k, "n_eval": n_eval,
            "config": stamp if cfg is None else cfg,
            "xn": torch.arange(lo, hi, dtype=torch.float32).reshape(-1, 1, 1, 1),
            "xp": torch.arange(lo, hi, dtype=torch.float32).reshape(-1, 1, 1, 1) + 100,
            "xc_np": torch.arange(lo, hi, dtype=torch.float32).reshape(-1, 1, 1, 1) + 200,
            "xc_pn": torch.arange(lo, hi, dtype=torch.float32).reshape(-1, 1, 1, 1) + 300,
        }
        if idx:
            blob["xn_idx"] = list(range(lo, hi))
            blob["xp_idx"] = [i + 1000 for i in range(lo, hi)]
        torch.save(blob, E._shard_path(d, s))

    def _fails(d, num_shards=None) -> str:
        try:
            E._load_shards(d, num_shards)
            return ""
        except Exception as exc:                     # noqa: BLE001 — asserting refusal
            return f"{type(exc).__name__}: {exc}"

    with tempfile.TemporaryDirectory() as d:
        for s in range(4):
            _write(d, s, 4)
        xn, xp, xc_np, xc_pn, n_eval, xn_idx, xp_idx = E._load_shards(d, 4)
        check(len(xn) == 8 and len(xp) == 8 and len(xc_np) == 8 and len(xc_pn) == 8,
              "merged length equals the whole set",
              f"xn={len(xn)} xp={len(xp)} np={len(xc_np)} pn={len(xc_pn)}")
        flat = torch.cat(xn).reshape(-1)
        check(torch.equal(flat, torch.arange(8, dtype=torch.float32)),
              "merge preserves global order (not shard-completion order)",
              str(flat.tolist()))
        check(torch.equal(torch.cat(xc_np).reshape(-1),
                          torch.arange(8, dtype=torch.float32) + 200),
              "counterfactuals stay aligned with their sources")
        check(xn_idx == list(range(8)) and xp_idx == [i + 1000 for i in range(8)],
              "indices merge in the same order as their tensors",
              f"{xn_idx}, {xp_idx}")

    # Each of these would otherwise produce a number describing nothing.
    with tempfile.TemporaryDirectory() as d:
        for s in (0, 1, 3):
            _write(d, s, 4)
        err = _fails(d, 4)
        check("missing shard(s) [2]" in err, "a missing shard is fatal, not short", err)

    with tempfile.TemporaryDirectory() as d:
        # Both declare num_shards=2 so the earlier num_shards check passes and
        # the config comparison is the one actually under test.
        _write(d, 0, 2)
        drifted = dict(stamp)
        drifted["USE_GRADCAM_MASK"] = not drifted.get("USE_GRADCAM_MASK", False)
        _write(d, 1, 2, cfg=drifted)
        check("different config" in _fails(d), "shards from two arms are never pooled",
              _fails(d)[:80])

    with tempfile.TemporaryDirectory() as d:
        _write(d, 0, 2, n_eval=4)
        _write(d, 1, 2, n_eval=9)
        check("disagree on n_eval" in _fails(d), "shards must agree on n_eval",
              _fails(d)[:80])

    with tempfile.TemporaryDirectory() as d:
        for s in range(2):
            _write(d, s, 2)
        check("--num-shards 8" in _fails(d, 8),
              "--num-shards is cross-checked against what is on disk", _fails(d, 8)[:80])

    with tempfile.TemporaryDirectory() as d:
        check("no shards matching" in _fails(d), "an empty shard dir is fatal")

    # Shards written before index threading must degrade, not crash: the LHR
    # block is dropped, every other metric still computes.
    with tempfile.TemporaryDirectory() as d:
        for s in range(2):
            _write(d, s, 2, idx=False)
        old = E._load_shards(d, 2)
        check(old[-2] is None and old[-1] is None,
              "pre-index shards load with xn_idx/xp_idx None, not an exception",
              f"{old[-2]}, {old[-1]}")


def main() -> int:
    print("=" * 70)
    print("INVARIANT CHECKS  (CPU only)")
    print("=" * 70)
    ds = _small_dataset()
    print(f"dataset: {len(ds)} images, "
          f"{sum(1 for i in range(len(ds)) if ds.lesion_boxes(i))} annotated")

    test_split(ds)
    test_index_alignment(ds)
    test_lhr_calibration(ds)
    test_serialisation()
    test_shard_roundtrip()

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"FAILED — {len(FAILURES)} invariant(s) broken:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("ALL INVARIANTS HOLD")
    return 0


if __name__ == "__main__":
    sys.exit(main())
