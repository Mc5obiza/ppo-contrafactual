"""
compare_arms.py
===============
Assemble the three-arm control table from the per-arm eval_metrics.json files.

WHY THIS EXISTS
---------------
Phase 2's whole claim — "the PPO policy contributes something" — lives or dies
on one table, and until now nothing in the project produced one. Each reduce job
writes its own eval_metrics.json (checkpoints/ for trained,
checkpoints/eval_<arm>/ otherwise); this reads them side by side.

HOW TO READ IT
--------------
The arms are NOT all compute-matched, so the comparisons are not interchangeable:

  * `none` calls generator.generate_counterfactual(..., M=1) — ONE guidance step,
    the generator's default fixed schedule.
  * `random` and `trained` both step env.generate_via_policy up to BUDGET_T
    generator calls per image. They differ only in whether the policy weights
    were trained.

So:
  trained vs random  -> does LEARNING help, at matched compute?    <- the claim
  random  vs none    -> does having a multi-step schedule help at all?

A trained arm that beats `none` but ties `random` has not shown that PPO learned
anything; it has shown that more gradient steps help.

CV is not the gate. CV measures "did gradient descent on f succeed against f" and
is circular by construction (see heldout_validation.py). The columns that carry
evidence are LHR (edit mass on the radiologist box, vs its area null) and the
held-out transfer numbers from heldout_validation.py.

    module load python/3.12.8 && source venv/bin/activate
    python compare_arms.py
"""

from __future__ import annotations

import json
import os

import config

ARMS = ("trained", "none", "random")


def _path(arm: str) -> str:
    return os.path.join(
        config.CKPT_DIR if arm == "trained" else os.path.join(config.CKPT_DIR, f"eval_{arm}"),
        "eval_metrics.json",
    )


def _policy_mtime() -> float:
    """Newest ppo_policy.pt across directions — the moment the current policies
    came into existence."""
    times = [
        os.path.getmtime(p)
        for p in (
            os.path.join(config.CKPT_DIR, d, "ppo_policy.pt")
            for d in ("n_to_p", "p_to_n")
        )
        if os.path.exists(p)
    ]
    return max(times) if times else 0.0


def _load() -> dict[str, dict]:
    """
    Load each arm's metrics, REFUSING any file that predates the policies.

    The trained arm writes to CKPT_DIR/eval_metrics.json — the same path every
    previous run of this project used. A pending or failed reduce therefore
    leaves a perfectly readable file from an OLDER run sitting exactly where a
    fresh one belongs, and the table would print it beside the new control arms
    as though the three had been produced together. That is the comparison
    silently describing two different systems, which is the failure this whole
    apparatus exists to prevent.

    An evaluation cannot legitimately predate the policies it evaluates, so
    mtime is a sufficient and cheap test.
    """
    cutoff = _policy_mtime()
    out = {}
    for arm in ARMS:
        p = _path(arm)
        if not os.path.exists(p):
            print(f"[compare] {arm:8s} not ready — no {p}")
            continue
        age = os.path.getmtime(p)
        if cutoff and age < cutoff:
            print(
                f"[compare] {arm:8s} SKIPPED — {p} is STALE: written "
                f"{_stamp(age)}, before the current policies ({_stamp(cutoff)}). "
                f"It is a leftover from an earlier run, not this arm's result."
            )
            continue
        with open(p) as fh:
            out[arm] = json.load(fh)
    return out


def _stamp(t: float) -> str:
    import datetime
    return datetime.datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M")


def _get(d: dict, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def _fmt(v, spec=".4f") -> str:
    if v is None:
        return "—"
    if isinstance(v, (int, float)):
        return format(v, spec)
    return str(v)


ROWS = [
    ("CV  n→p",              lambda d: _get(d, "CV", "n_to_p"),                    ".4f"),
    ("CV  p→n",              lambda d: _get(d, "CV", "p_to_n"),                    ".4f"),
    ("CV  mean",             lambda d: _get(d, "CV", "mean"),                      ".4f"),
    ("LHR hit rate  p→n",    lambda d: _get(d, "LHR", "p_to_n", "hit_rate"),       ".4f"),
    ("LHR null (box area)",  lambda d: _get(d, "LHR", "p_to_n", "null_baseline"),  ".4f"),
    ("LHR LIFT  p→n",        lambda d: _get(d, "LHR", "p_to_n", "lift"),           ".3f"),
    ("LHR p-value",          lambda d: _get(d, "LHR", "p_to_n", "p"),              ".2e"),
    ("LHR n pairs",          lambda d: _get(d, "LHR", "p_to_n", "n_pairs"),        "d"),
    ("EditConc n→p",         lambda d: _get(d, "EditConcentration", "n_to_p", "top_k_mass"), ".4f"),
    ("EditConc p→n",         lambda d: _get(d, "EditConcentration", "p_to_n", "top_k_mass"), ".4f"),
    ("IP MS-SSIM n→p",       lambda d: _get(d, "IP", "n_to_p", "ms_ssim"),         ".4f"),
    ("IP MS-SSIM p→n",       lambda d: _get(d, "IP", "p_to_n", "ms_ssim"),         ".4f"),
    ("FID negative group",   lambda d: _get(d, "FID", "negative_group"),           ".2f"),
    ("FID positive group",   lambda d: _get(d, "FID", "positive_group"),           ".2f"),
]


def main() -> int:
    data = _load()
    if not data:
        print("[compare] no arms available yet.")
        return 1

    present = [a for a in ARMS if a in data]
    w = 22
    print("\n" + "=" * (w + 14 * len(present)))
    print("THREE-ARM CONTROL TABLE".center(w + 14 * len(present)))
    print("=" * (w + 14 * len(present)))
    header = "metric".ljust(w) + "".join(a.rjust(14) for a in present)
    print(header)
    print("-" * len(header))
    for label, fn, spec in ROWS:
        cells = []
        for a in present:
            try:
                cells.append(_fmt(fn(data[a]), spec))
            except Exception:
                cells.append("—")
        if all(c == "—" for c in cells):
            continue
        print(label.ljust(w) + "".join(c.rjust(14) for c in cells))
    print("-" * len(header))

    # The comparisons that actually carry the claim.
    if {"trained", "random"} <= set(present):
        t = _get(data["trained"], "LHR", "p_to_n", "lift")
        r = _get(data["random"], "LHR", "p_to_n", "lift")
        tc = _get(data["trained"], "CV", "mean")
        rc = _get(data["random"], "CV", "mean")
        print("\ntrained vs random  (matched compute — isolates LEARNING)")
        print(f"    LHR lift : {_fmt(t, '.3f')} vs {_fmt(r, '.3f')}")
        print(f"    CV  mean : {_fmt(tc)} vs {_fmt(rc)}   [circular — not the gate]")
    if {"random", "none"} <= set(present):
        r = _get(data["random"], "LHR", "p_to_n", "lift")
        n = _get(data["none"], "LHR", "p_to_n", "lift")
        print("\nrandom vs none  (isolates HAVING a multi-step schedule)")
        print(f"    LHR lift : {_fmt(r, '.3f')} vs {_fmt(n, '.3f')}")

    missing = [a for a in ARMS if a not in data]
    if missing:
        print(f"\n[compare] still missing: {', '.join(missing)}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
