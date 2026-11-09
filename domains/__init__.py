"""
domains/
========
Dataset-domain registry.

Adding a dataset is one new module in this package plus one line in config —
NOT an edit to ingestion.py, src/xrv_classifier.py and src/metrics.py. See
domains/base.py for what a DomainProfile binds together and why.

    import domains
    d = domains.active()          # profile named by config.DATASET_DOMAIN
    ds = d.build_dataset(config.IMG_SIZE)

Modules are imported lazily, so a domain whose optional dependency is missing
(e.g. SimpleITK for NODE21's .mha files) cannot break an unrelated run.
"""

from __future__ import annotations

import importlib

import config

from .base import DomainProfile


# IMPORT-ORDER LANDMINE. torchxrayvision ships its own TOP-LEVEL `config`
# package (site-packages/torchxrayvision/baseline_models/jfhealthcare/config)
# and registers it as sys.modules["config"]. So
#
#     import torchxrayvision      # poisons sys.modules["config"]
#     import config               # -> torchxrayvision's, NOT config.py
#
# and every `config.X` lookup afterwards raises AttributeError from a module
# nobody meant to import. Every existing entry point happens to `import config`
# before touching torchxrayvision, so this has never fired — but it is luck,
# not design, and the failure is baffling when it does. Convert it to a message
# that names the fix.
if not hasattr(config, "DATASET_DOMAIN"):
    raise ImportError(
        f"the imported `config` is not this project's config.py — got "
        f"{getattr(config, '__file__', '<unknown>')}. torchxrayvision registers "
        f"its own top-level `config` package, so importing torchxrayvision "
        f"BEFORE the project's `import config` shadows it. Fix: import config "
        f"first, before torchxrayvision (and before this package)."
    )


# registry key -> module in this package exposing a module-level PROFILE
_REGISTRY: dict[str, str] = {
    "rsna_pneumonia":   ".rsna_pneumonia",
    "node21_nodule":    ".node21_nodule",
    "cbis_ddsm_breast": ".cbis_ddsm_breast",
    "cmmd_breast":      ".cmmd_breast",
}

_cache: dict[str, DomainProfile] = {}


def available() -> list[str]:
    """Registered domain names."""
    return sorted(_REGISTRY)


def get(name: str) -> DomainProfile:
    """
    Resolve a domain by registry key, validating it before returning.

    validate() is called here rather than at dataclass construction so an
    import of this package never reaches out to torchxrayvision metadata for
    domains nobody asked for.
    """
    if name in _cache:
        return _cache[name]
    if name not in _REGISTRY:
        raise ValueError(
            f"unknown domain {name!r}. Registered: {available()}. "
            f"Add one by dropping a module in domains/ that exposes a "
            f"module-level PROFILE, then listing it in domains._REGISTRY."
        )

    module = importlib.import_module(_REGISTRY[name], package=__name__)
    profile = getattr(module, "PROFILE", None)
    if not isinstance(profile, DomainProfile):
        raise TypeError(
            f"domains/{name}.py must expose a module-level PROFILE of type "
            f"DomainProfile; got {type(profile).__name__}"
        )

    profile.validate()
    _cache[name] = profile
    return profile


def active() -> DomainProfile:
    """The domain named by config.DATASET_DOMAIN."""
    return get(config.DATASET_DOMAIN)


__all__ = ["DomainProfile", "available", "get", "active"]
