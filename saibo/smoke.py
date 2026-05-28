"""Small runnable SAIBO smoke experiments.

These checks exercise the copied LABO and LGBO method paths on synthetic tasks.
They are intentionally tiny so they can run in CI or on a laptop.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch


class SyntheticBlackBox:
    """Small deterministic high-fidelity objective for LABO checks."""

    def __init__(self) -> None:
        self.bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)

    def evaluate(self, x: dict[str, float]) -> float:
        a = float(x["x1"])
        b = float(x["x2"])
        return float(1.0 - (a - 0.72) ** 2 - 0.5 * (b - 0.35) ** 2)


class DeterministicLowFidelityClient:
    """LLM-shaped client that returns JSON predictions without network calls."""

    def generate(self, prompt: str, **_: object) -> str:
        payload = self._extract_candidate_payload(prompt)

        predictions = []
        for item in payload.get("data_points", []):
            features = item.get("features", [])
            if not features or "target" in item:
                continue
            values = [float(value) for value in features]
            a = values[0]
            b = values[1] if len(values) > 1 else 0.5
            value = 0.85 - (a - 0.65) ** 2 - 0.4 * (b - 0.4) ** 2
            if len(values) > 2:
                value -= 0.05 * sum((extra - 0.5) ** 2 for extra in values[2:])
            predictions.append({"features": values, "target": round(float(value), 6)})
        return json.dumps({"data_points": predictions})

    @staticmethod
    def _extract_candidate_payload(prompt: str) -> dict[str, Any]:
        decoder = json.JSONDecoder()
        candidates = []
        for index, char in enumerate(prompt):
            if char != "{":
                continue
            try:
                payload, end = decoder.raw_decode(prompt[index:])
            except json.JSONDecodeError:
                continue
            _ = end
            if isinstance(payload, dict) and isinstance(payload.get("data_points"), list):
                candidates.append(payload)

        for payload in reversed(candidates):
            data_points = payload.get("data_points", [])
            if any(isinstance(item, dict) and "features" in item and "target" not in item for item in data_points):
                return payload
        return {"data_points": []}


def _labo_client(online: bool):
    if not online:
        return DeterministicLowFidelityClient()

    from .methods.labo.API.llm_clients import InternS1Client

    return InternS1Client(timeout=120.0)


def run_labo_smoke(output_dir: str | Path, *, online: bool = False) -> dict[str, Any]:
    """Run a compact LABO path and return a JSON-friendly summary."""
    from .methods.labo.koh.optimizer import KOHOptimizer

    out = Path(output_dir) / ("labo_online" if online else "labo_offline")
    out.mkdir(parents=True, exist_ok=True)

    llm_config = SimpleNamespace(
        temperature=0.0,
        top_p=1.0,
        max_tokens=512,
        alpha=1.0,
        beta=0.0,
        value_range=[-1.0, 1.5],
    )
    koh_config = SimpleNamespace(
        n_candidates=24 if online else 40,
        mismatch_threshold=0.75,
        force_hf_after_n_lf=None,
        gp_training_iter=2,
        max_loops=1 if online else 2,
        acquisition_type="ucb",
        acquisition_beta=1.0,
        always_update_lf_loops=1,
        random_seed=7,
    )

    blackbox = SyntheticBlackBox()
    optimizer = KOHOptimizer(
        task_name="synthetic",
        task_data_dir=str(out),
        feature_names=["x1", "x2"],
        feature_types=["float", "float"],
        bounds=blackbox.bounds,
        target_name="objective",
        llm_client=_labo_client(online),
        hf_blackbox=blackbox,
        llm_config=llm_config,
        koh_config=koh_config,
        file_prefix="smoke",
    )
    optimizer.run(
        max_iterations=1,
        n_initial_points=3,
        q=1,
        fixed_initial_points=[
            {"x1": 0.10, "x2": 0.10},
            {"x1": 0.50, "x2": 0.50},
            {"x1": 0.90, "x2": 0.30},
        ],
    )
    return {
        "method": "labo",
        "online": bool(online),
        "status": "ok",
        "best_y": optimizer._best_y(),
        "iterations": optimizer.iteration_log,
        "output_dir": str(out),
    }


def _ackley_history(space, toy_results):
    points = [
        {"x1": -3.0, "x2": 3.0},
        {"x1": 0.5, "x2": -0.5},
        {"x1": 2.0, "x2": 2.0},
    ]
    vectors = [[point["x1"], point["x2"]] for point in points]
    values = toy_results("ackley", vectors)
    history = []
    for point, value in zip(points, values):
        history.append(f"x1={point['x1']:.6g}, x2={point['x2']:.6g} -> f={float(value['f']):.6g}")
    X = torch.tensor([space.normalize_point(point) for point in points], dtype=torch.double)
    Y = torch.tensor([[float(value["f"])] for value in values], dtype=torch.double)
    return X, Y, history


def _call_lgbo_online(system_prompt: str, user_prompt: str) -> str:
    if not os.getenv("API_KEY") and os.getenv("INTERN_S1_API_KEY"):
        os.environ["API_KEY"] = os.environ["INTERN_S1_API_KEY"]

    from .methods.lgbo.llm_client import call_chat

    return call_chat(
        system_prompt,
        user_prompt,
        temperature=0.2,
        max_tokens=1024,
        timeout=180,
    )


def run_lgbo_smoke(output_dir: str | Path, *, online: bool = False) -> dict[str, Any]:
    """Run a compact LGBO preference-to-acquisition path."""
    from .methods.lgbo.fun.toy_fun import get_bo_bounds, toy_results
    from .methods.lgbo.lgbo_core import ContinuousSpace, propose_lgbo_batch, serialize_plan
    from .methods.lgbo.prompt import (
        DEFAULT_SYSTEM_PROMPT,
        build_toy_user_prompt,
        parse_assistant_response,
    )

    out = Path(output_dir) / ("lgbo_online" if online else "lgbo_offline")
    out.mkdir(parents=True, exist_ok=True)

    space = ContinuousSpace.from_parameters(
        [
            {"name": f"x{i + 1}", "bounds": list(bounds)}
            for i, bounds in enumerate(get_bo_bounds("ackley", 2))
        ]
    )
    X_hist, Y_hist, history = _ackley_history(space, toy_results)
    user_prompt = build_toy_user_prompt(
        func_name="ackley",
        d=2,
        bounds=space.bounds,
        history=history,
        batch_q=1,
    )

    if online:
        assistant_text = _call_lgbo_online(DEFAULT_SYSTEM_PROMPT, user_prompt)
    else:
        assistant_text = "Final Answer:\n[point, [0.0, 0.0], 0.7]"

    parsed = parse_assistant_response(assistant_text)
    if not parsed.get("mode") and online:
        assistant_text = _call_lgbo_online(
            DEFAULT_SYSTEM_PROMPT,
            user_prompt
            + "\n\nOutput only one line: [point, [x1, x2], confidence] or [region, [[lb1, lb2], [ub1, ub2]], confidence].",
        )
        parsed = parse_assistant_response(assistant_text)
    if not parsed.get("mode"):
        raise RuntimeError("Could not parse LGBO LLM preference.")

    points, plan, z_new = propose_lgbo_batch(
        X_norm=X_hist,
        y=Y_hist,
        parsed_preference=parsed,
        space=space,
        goal="min",
        batch_q=1,
        policy="tilt",
        grid_size=64,
        guidance_scale=3.0,
        cand_size=256,
        num_paths_batch=32,
        seed=17,
    )
    payload = {
        "method": "lgbo",
        "online": bool(online),
        "status": "ok",
        "parsed_mode": parsed.get("mode"),
        "confidence": parsed.get("confidence"),
        "plan": serialize_plan(plan),
        "normalized_points": z_new.tolist(),
        "points": points,
        "output_dir": str(out),
    }
    (out / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def run_all_smokes(output_dir: str | Path, *, online: bool = False) -> list[dict[str, Any]]:
    """Run both method smoke checks."""
    return [
        run_labo_smoke(output_dir, online=online),
        run_lgbo_smoke(output_dir, online=online),
    ]
