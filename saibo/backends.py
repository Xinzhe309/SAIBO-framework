"""Unified accessors for SAIBO method backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Backend:
    """Metadata and loader for a SAIBO backend."""

    name: str
    description: str
    load: Callable[[], object]


def _load_labo_optimizer():
    from .methods.labo.koh.optimizer import KOHOptimizer

    return KOHOptimizer


def _load_lgbo_proposer():
    from .methods.lgbo.lgbo_core import propose_lgbo_batch

    return propose_lgbo_batch


_BACKENDS = {
    "labo": Backend(
        name="labo",
        description="LLM-as-low-fidelity Bayesian optimization with KOH fusion.",
        load=_load_labo_optimizer,
    ),
    "lgbo": Backend(
        name="lgbo",
        description="LLM preference-guided Bayesian optimization with region lifting.",
        load=_load_lgbo_proposer,
    ),
}


def available_methods() -> tuple[str, ...]:
    """Return the method names available in this SAIBO build."""
    return tuple(_BACKENDS)


def get_backend(name: str) -> Backend:
    """Return backend metadata for `name`."""
    key = name.strip().lower()
    if key not in _BACKENDS:
        allowed = ", ".join(available_methods())
        raise ValueError(f"Unknown SAIBO backend {name!r}. Choose one of: {allowed}.")
    return _BACKENDS[key]
