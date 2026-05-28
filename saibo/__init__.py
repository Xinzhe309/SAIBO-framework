"""SAIBO public framework package.

SAIBO exposes LABO and LGBO as optional Bayesian-optimization backends while
keeping each method's core implementation separate.
"""

from __future__ import annotations

from .backends import available_methods, get_backend
from .runners import run_dry, run_wet

__all__ = ["available_methods", "get_backend", "run_dry", "run_wet"]

__version__ = "1.0.0"
