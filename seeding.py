"""
seeding.py
==========
One place that makes a run reproducible, and one place that says how far that
reproducibility actually goes.

WHY THIS EXISTS
---------------
The training path had no seeding of any kind. The only `torch.manual_seed` in
the whole tree was in the eval loop (src/inference.py), which made the SHARDED
evaluation bit-reproducible while the training that produced the policies was
not: two identical submissions of run_ppo.slurm produced two different
policies, and nothing recorded which one a checkpoint came from.

That is survivable for a single headline number. It is fatal for the control
arms, where trained / random / no-policy are meant to differ by TREATMENT — if
each arm also carries its own uncontrolled seed, a difference between them
cannot be attributed to the treatment at all.

WHAT IS AND IS NOT GUARANTEED
-----------------------------
Seeded here:
  * Python `random`, NumPy, torch CPU and all CUDA devices
  * cuDNN convolution algorithm selection (deterministic=True,
    benchmark=False) — benchmark mode otherwise picks an algorithm based on
    observed timings, so the same code can take a different numerical path on a
    busy node than on an idle one
  * DataLoader worker RNGs, via `worker_init_fn` / `generator` (see
    `loader_kwargs`) — each worker otherwise seeds itself from entropy

NOT guaranteed, deliberately:
  * `torch.use_deterministic_algorithms(True)` is OFF by default. Several ops
    in the diffusion path have no deterministic CUDA implementation and it
    raises rather than falling back, so switching it on trades a reproducible
    run for a crashing one. Enable via config.STRICT_DETERMINISM when
    investigating a specific nondeterminism, not as a standing setting.
  * Resuming does NOT replay the original sequence. A resumed PPO run re-seeds
    at the same base and then fast-forwards through a different number of
    draws, so run A (24h, resumed once) and run B (24h, uninterrupted) diverge
    even at the same seed. `offset` exists to make that explicit rather than
    accidental — pass the resumed update count.
  * The environment itself samples from several RNG streams per step (VAE
    posterior, DDIM init noise, per-step background-lock noise), so identical
    seeds reproduce a run only against identical config AND identical hardware.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

import config


def seed_everything(seed: int | None = None, offset: int = 0, tag: str = "") -> int:
    """
    Seed every RNG this project draws from. Returns the seed actually applied,
    so callers can record it next to whatever they write.

    Parameters
    ----------
    seed   : base seed; defaults to config.TRAIN_SEED
    offset : added to the base. Use it to give two things that must not share a
             stream (e.g. the two edit directions) different sequences from one
             configured seed, instead of silently sharing one.
    tag    : label for the log line, e.g. "n_to_p"
    """
    base = config.TRAIN_SEED if seed is None else seed
    s = int(base) + int(offset)

    random.seed(s)
    np.random.seed(s % (2**32))          # numpy legacy API rejects >= 2**32
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)        # no-op without CUDA

    # PYTHONHASHSEED only takes effect at interpreter start, so setting it here
    # cannot affect this process. Set anyway: subprocesses inherit it, and its
    # absence is a real source of set/dict-ordering drift.
    os.environ.setdefault("PYTHONHASHSEED", str(s))

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if getattr(config, "STRICT_DETERMINISM", False):
        # Documented to raise on ops with no deterministic kernel. That is the
        # point when it is on — but say so, because the traceback it produces
        # otherwise reads as an unrelated bug in the diffusion path.
        torch.use_deterministic_algorithms(True, warn_only=False)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        print("[seed] STRICT_DETERMINISM on — ops without a deterministic "
              "CUDA kernel will RAISE rather than fall back.")

    print(
        f"[seed] {tag or 'run'}: seed={s}"
        + (f" (base {base} + offset {offset})" if offset else "")
        + " | cudnn.deterministic=True, benchmark=False"
    )
    return s


def loader_kwargs(seed: int | None = None) -> dict:
    """
    DataLoader kwargs that make worker RNGs reproducible.

    A DataLoader with num_workers>0 forks workers that seed themselves from
    entropy unless told otherwise, so shuffling and any dataset-side randomness
    differ per run even after seed_everything. Merge into an existing kwargs
    dict:

        DataLoader(ds, **{**my_kwargs, **seeding.loader_kwargs()})
    """
    base = config.TRAIN_SEED if seed is None else seed

    g = torch.Generator()
    g.manual_seed(int(base))

    def _worker_init(worker_id: int) -> None:
        ws = (int(base) + worker_id) % (2**32)
        np.random.seed(ws)
        random.seed(ws)

    return {"generator": g, "worker_init_fn": _worker_init}
