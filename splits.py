"""
splits.py
=========
One deterministic, persisted, patient-level train/test split, shared by every
stage that touches the dataset.

WHY THIS EXISTS
---------------
There was no train/test separation anywhere in this pipeline. `build_common()`
constructs a single dataset object and hands the same one to LoRA fine-tuning,
to the PPO far-pool, and to the metrics group scan — so the 400 images the
evaluation reports on had been seen by the generator during fine-tuning and by
the policy during rollouts. Every headline number was measured on training data.

A split only fixes that if all three stages agree on it, which means it cannot
be recomputed per stage from an RNG: `eval_main.py --stage generate` runs as 8
independent SLURM array tasks in 8 separate processes, and `--stage reduce`
runs in a ninth. So the assignment is computed once, written to disk, and read
back by everyone. That file is the contract.

PATIENT-LEVEL, NOT IMAGE-LEVEL
------------------------------
Splitting by image puts two views of the same patient on both sides, and the
generator learns that patient's anatomy from the train copy before being
evaluated on the test copy. For RSNA this happens to be moot — `samples` is
already one entry per patientId (ingestion.py groups by it) — but it is not
moot for CBIS-DDSM, where one patient contributes several views. Grouping is
done whenever the dataset exposes `patient_id(idx)`, so a domain that needs it
gets it without a second implementation.

USAGE
-----
    import splits
    s = splits.load_or_create(dataset, domain="rsna_pneumonia")
    train_idx, test_idx = s.train, s.test

Both are sorted numpy arrays of dataset indices.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass

import numpy as np

import config


# Bumped when the split ALGORITHM changes in a way that would silently produce
# a different assignment from the same seed. An existing file whose version
# does not match is refused rather than reused, because a split that quietly
# changed underneath a half-finished experiment is worse than a crash.
SPLIT_FORMAT_VERSION = 1


@dataclass(frozen=True)
class Split:
    """Dataset indices for each fold, plus the provenance that produced them."""
    train: np.ndarray
    test: np.ndarray
    seed: int
    test_frac: float
    domain: str
    fingerprint: str

    def summary(self) -> str:
        n = len(self.train) + len(self.test)
        return (
            f"[split] {self.domain}: {len(self.train)} train / {len(self.test)} test "
            f"({len(self.test) / max(n, 1):.1%} held out, seed={self.seed})"
        )


def _dataset_fingerprint(dataset) -> str:
    """
    Cheap identity check for the dataset a split was computed against.

    Length alone is not enough: two datasets can have the same size and a
    different order, and the split stores positional indices. Hashing the
    first/last few group keys catches a reordering without walking 26k entries.
    """
    n = len(dataset)
    keys = [str(_group_key(dataset, i)) for i in (0, 1, 2, n - 3, n - 2, n - 1) if 0 <= i < n]
    h = hashlib.sha256(("|".join(keys) + f"|{n}").encode()).hexdigest()[:16]
    return f"{n}:{h}"


def _group_key(dataset, idx: int):
    """
    The unit a split must not straddle: patient when the dataset knows about
    patients, otherwise the index itself (each sample its own group).
    """
    fn = getattr(dataset, "patient_id", None)
    return fn(idx) if callable(fn) else idx


def _split_path(domain: str) -> str:
    # Alongside the checkpoints for the domain, so a split travels with the
    # weights it describes rather than living in a global directory where two
    # domains would fight over one filename.
    return os.path.join(config.CKPT_DIR, f"split_{domain}.json")


def create(dataset, domain: str, seed: int, test_frac: float) -> Split:
    """
    Compute a fresh patient-level split. Deterministic in (seed, dataset order).
    """
    if not 0.0 < test_frac < 1.0:
        raise ValueError(f"test_frac must be in (0,1), got {test_frac}")

    n = len(dataset)
    keys = [_group_key(dataset, i) for i in range(n)]

    # sorted() so the group ordering is a function of the keys themselves, not
    # of dict insertion order — otherwise the same seed gives a different split
    # on a dataset whose iteration order changed.
    groups = sorted({str(k) for k in keys})
    rng = np.random.default_rng(seed)
    shuffled = np.array(groups, dtype=object)[rng.permutation(len(groups))]

    n_test = max(1, int(round(test_frac * len(groups))))
    test_groups = set(shuffled[:n_test].tolist())

    test_idx  = np.array([i for i, k in enumerate(keys) if str(k) in test_groups], dtype=np.int64)
    train_idx = np.array([i for i, k in enumerate(keys) if str(k) not in test_groups], dtype=np.int64)

    return Split(
        train=np.sort(train_idx),
        test=np.sort(test_idx),
        seed=seed,
        test_frac=test_frac,
        domain=domain,
        fingerprint=_dataset_fingerprint(dataset),
    )


def save(split: Split, path: str | None = None) -> str:
    path = path or _split_path(split.domain)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    payload = {
        "version": SPLIT_FORMAT_VERSION,
        "domain": split.domain,
        "seed": split.seed,
        "test_frac": split.test_frac,
        "fingerprint": split.fingerprint,
        "n_train": int(len(split.train)),
        "n_test": int(len(split.test)),
        # Only the test indices are stored. Train is everything else, so the two
        # cannot drift apart into an overlap — the one corruption that would
        # reintroduce exactly the leakage this module exists to prevent.
        "test_indices": split.test.tolist(),
    }

    # Atomic: a split truncated by a walltime kill mid-write would be read back
    # as a valid-looking shorter test set.
    #
    # The temp name carries the PID because run_ppo.slurm is an ARRAY: both
    # directions run at once, in the same CKPT_DIR, and reach this line within
    # seconds of each other on a cold checkpoint dir. A shared ".tmp" would have
    # the two processes writing the same path concurrently, and os.replace would
    # then atomically promote the interleaved bytes. Per-process temp files make
    # the loser's rename a harmless no-op instead — both write identical content
    # (same seed, same fingerprint), so whichever lands is correct.
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, path)
    return path


def load(dataset, domain: str, path: str | None = None) -> Split | None:
    """
    Read a persisted split, or None if absent.

    Raises if the file exists but does not describe THIS dataset — a stale
    split silently applied to a changed dataset is a leak that no downstream
    check would catch.
    """
    path = path or _split_path(domain)
    if not os.path.exists(path):
        return None

    with open(path) as f:
        payload = json.load(f)

    if payload.get("version") != SPLIT_FORMAT_VERSION:
        raise RuntimeError(
            f"{path} was written by split format v{payload.get('version')}, this is "
            f"v{SPLIT_FORMAT_VERSION}. Delete it and regenerate, then RE-RUN every "
            f"stage that consumed it — mixing two splits reintroduces the leakage."
        )

    fp = _dataset_fingerprint(dataset)
    if payload.get("fingerprint") != fp:
        raise RuntimeError(
            f"{path} was computed against a different dataset "
            f"(stored {payload.get('fingerprint')!r}, current {fp!r}).\n"
            f"The stored indices are positional, so applying them to this dataset "
            f"would put unknown images in the test fold. Either restore the "
            f"original dataset or delete the split and retrain from scratch."
        )

    test = np.array(payload["test_indices"], dtype=np.int64)
    mask = np.ones(len(dataset), dtype=bool)
    mask[test] = False
    train = np.nonzero(mask)[0].astype(np.int64)

    return Split(
        train=train,
        test=np.sort(test),
        seed=int(payload["seed"]),
        test_frac=float(payload["test_frac"]),
        domain=domain,
        fingerprint=fp,
    )


def load_or_create(
    dataset,
    domain: str | None = None,
    seed: int | None = None,
    test_frac: float | None = None,
    path: str | None = None,
) -> Split:
    """
    The entry point every stage should call. Reads the split if one exists,
    otherwise computes and persists it.

    Defaults come from config so all three stages agree without passing
    anything around.
    """
    domain    = domain    if domain    is not None else config.DATASET_DOMAIN
    seed      = seed      if seed      is not None else config.SPLIT_SEED
    test_frac = test_frac if test_frac is not None else config.TEST_FRAC

    existing = load(dataset, domain, path)
    if existing is not None:
        if existing.seed != seed or abs(existing.test_frac - test_frac) > 1e-9:
            # Do NOT silently regenerate: another stage may already have trained
            # against the stored split, and replacing it now would evaluate on
            # images the generator has seen.
            raise RuntimeError(
                f"persisted split for {domain} has seed={existing.seed}, "
                f"test_frac={existing.test_frac}, but config asks for seed={seed}, "
                f"test_frac={test_frac}. Refusing to regenerate — anything already "
                f"trained used the stored split. To change it, delete "
                f"{path or _split_path(domain)} and re-run ALL stages."
            )
        print(existing.summary() + "  [loaded]")
        return existing

    fresh = create(dataset, domain, seed, test_frac)
    saved = save(fresh, path)
    print(fresh.summary() + f"  [created -> {saved}]")
    return fresh
