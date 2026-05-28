"""Example user-defined dry evaluator for SAIBO.

The dry evaluator can be a class with evaluate(point) or a plain function. The
point is a dictionary keyed by the names declared in the task profile.
"""

from __future__ import annotations

import numpy as np


class ExampleBlackBox:
    """Small maximization objective used by the public dry-run example."""

    feature_names = ["x1", "x2"]
    feature_types = ["float", "float"]
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)

    def evaluate(self, point: dict[str, float]) -> float:
        x1 = float(point["x1"])
        x2 = float(point["x2"])
        return float(1.2 - (x1 - 0.76) ** 2 - 0.6 * (x2 - 0.32) ** 2)


def evaluate(point: dict[str, float]) -> float:
    """Function-form evaluator; usable as examples/custom_evaluator.py:evaluate."""
    return ExampleBlackBox().evaluate(point)
