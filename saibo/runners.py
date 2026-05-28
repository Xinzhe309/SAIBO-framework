"""SAIBO dry and wet interfaces.

Dry runners own an evaluator and close the optimization loop automatically.
Wet runners consume observed history and return the next planned batch without
calling a high-fidelity evaluator.
"""

from __future__ import annotations

import json
import importlib
import importlib.util
import inspect
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from .smoke import (
    DeterministicLowFidelityClient,
    SyntheticBlackBox,
    _call_lgbo_online,
    _labo_client,
)


class NoHighFidelityEvaluator:
    """Guard object for wet planners."""

    def evaluate(self, x: Mapping[str, float]) -> float:
        _ = x
        raise RuntimeError("Wet mode does not call high-fidelity evaluation.")


class DryEvaluator:
    """Adapter for user-provided dry evaluators."""

    def __init__(
        self,
        target: Any,
        *,
        feature_names: Sequence[str],
        feature_types: Sequence[str],
        bounds: np.ndarray,
    ) -> None:
        self.target = target
        self.feature_names = list(feature_names)
        self.feature_types = list(feature_types)
        self.bounds = np.asarray(bounds, dtype=float)

    def evaluate(self, point: Mapping[str, Any]) -> float:
        if hasattr(self.target, "evaluate"):
            result = self.target.evaluate(dict(point))
        elif callable(self.target):
            result = self.target(dict(point))
        else:
            raise TypeError("Dry evaluator must be callable or expose evaluate(point).")
        return _coerce_objective_value(result)


class SignedEvaluator:
    """Expose a maximization score while preserving raw objective semantics."""

    def __init__(self, evaluator: DryEvaluator, sign: float) -> None:
        self.evaluator = evaluator
        self.sign = float(sign)
        self.bounds = evaluator.bounds

    def evaluate(self, point: Mapping[str, Any]) -> float:
        return self.sign * self.evaluator.evaluate(point)


def load_task_json(path: str | Path) -> dict[str, Any]:
    """Load a SAIBO wet-task JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _coerce_objective_value(result: Any) -> float:
    if isinstance(result, Mapping):
        for key in ("y", "f", "objective", "value", "target", "score"):
            if key in result:
                return float(result[key])
        raise ValueError("Evaluator returned a mapping without y/f/objective/value/target/score.")
    return float(result)


def _load_object_from_spec(spec: str) -> Any:
    if ":" not in spec:
        raise ValueError("Evaluator spec must look like 'path/to/file.py:object' or 'module.name:object'.")
    module_ref, object_ref = spec.rsplit(":", 1)
    if not object_ref.strip():
        raise ValueError("Evaluator spec is missing the object name after ':'.")

    module_path = Path(module_ref)
    if module_path.exists() or module_ref.endswith(".py"):
        module_path = module_path.resolve()
        if not module_path.exists():
            raise FileNotFoundError(f"Evaluator file not found: {module_path}")
        parent = str(module_path.parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)
        module_name = f"_saibo_user_eval_{abs(hash(str(module_path)))}"
        spec_obj = importlib.util.spec_from_file_location(module_name, module_path)
        if spec_obj is None or spec_obj.loader is None:
            raise ImportError(f"Could not import evaluator file: {module_path}")
        module = importlib.util.module_from_spec(spec_obj)
        spec_obj.loader.exec_module(module)
    else:
        module = importlib.import_module(module_ref)

    obj: Any = module
    for attr in object_ref.split("."):
        obj = getattr(obj, attr)
    if inspect.isclass(obj):
        return obj()
    if hasattr(obj, "create_evaluator") and callable(obj.create_evaluator):
        return obj.create_evaluator()
    return obj


def _bounds_from_any(bounds: Any, feature_names: Sequence[str] | None = None) -> np.ndarray:
    if isinstance(bounds, Mapping):
        if feature_names is None:
            feature_names = list(bounds.keys())
        return np.asarray([[float(bounds[name][0]), float(bounds[name][1])] for name in feature_names], dtype=float)
    array = np.asarray(bounds, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("bounds must have shape [d, 2] or be a mapping name -> [lo, hi]")
    return array


def _feature_spec_from_evaluator(target: Any) -> tuple[list[str], list[str], np.ndarray]:
    parameters = getattr(target, "parameters", None)
    if parameters:
        return _feature_spec(parameters)

    raw_names = getattr(target, "feature_names", None)
    raw_bounds = getattr(target, "bounds", None)
    if raw_bounds is None:
        raise ValueError("Custom dry evaluator needs bounds, or task-json parameters must be provided.")

    if raw_names is None and isinstance(raw_bounds, Mapping):
        raw_names = list(raw_bounds.keys())
    if raw_names is None:
        bounds_array = _bounds_from_any(raw_bounds)
        raw_names = [f"x{i + 1}" for i in range(bounds_array.shape[0])]
    else:
        bounds_array = _bounds_from_any(raw_bounds, raw_names)

    names = [str(name) for name in raw_names]
    raw_types = getattr(target, "feature_types", ["float"] * len(names))
    types = ["int" if str(dtype).lower() in {"int", "integer"} else "float" for dtype in raw_types]
    return names, types, bounds_array


def _load_dry_evaluator(
    evaluator_spec: str,
    task_data: Mapping[str, Any],
) -> tuple[DryEvaluator, list[str], list[str], np.ndarray]:
    target = _load_object_from_spec(evaluator_spec)
    if task_data.get("parameters"):
        feature_names, feature_types, bounds = _feature_spec(_parameters_from_data(task_data))
    else:
        feature_names, feature_types, bounds = _feature_spec_from_evaluator(target)
    evaluator = DryEvaluator(
        target,
        feature_names=feature_names,
        feature_types=feature_types,
        bounds=bounds,
    )
    return evaluator, feature_names, feature_types, bounds


def _write_summary(output_dir: str | Path, filename: str, payload: Mapping[str, Any]) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / filename
    path.write_text(json.dumps(_json_ready(payload), ensure_ascii=True, indent=2), encoding="utf-8")
    return path


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _goal_sign(goal: str) -> float:
    name = str(goal or "max").strip().lower()
    if name == "max":
        return 1.0
    if name == "min":
        return -1.0
    raise ValueError("goal must be 'max' or 'min'")


def _points_to_plain(points: Sequence[Mapping[str, Any]]) -> list[dict[str, float]]:
    return [{str(k): float(v) for k, v in point.items()} for point in points]


def _as_text_lines(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [line.strip() for line in value.splitlines() if line.strip()]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _append_section(lines: list[str], title: str, value: Any) -> None:
    section_lines = _as_text_lines(value)
    if not section_lines:
        return
    if lines:
        lines.append("")
    lines.append(f"[{title}]")
    for item in section_lines:
        lines.append(f"- {item}")


def _parameter_profile_lines(data: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    for param in data.get("parameters", []) or []:
        if not isinstance(param, Mapping):
            continue
        name = str(param.get("name", "")).strip()
        if not name:
            continue
        bounds = param.get("bounds")
        bounds_text = ""
        if isinstance(bounds, Sequence) and len(bounds) == 2:
            bounds_text = f" in [{bounds[0]}, {bounds[1]}]"
        dtype = str(param.get("type", "continuous")).strip()
        desc = str(param.get("description", "")).strip()
        suffix = f"; {desc}" if desc else ""
        lines.append(f"{name} ({dtype}){bounds_text}{suffix}")
    return lines


def _task_profile_text(data: Mapping[str, Any]) -> str:
    """Build user-facing task context from a dry/wet task JSON."""
    lines: list[str] = []
    task_name = str(data.get("task_name", "")).strip()
    if task_name:
        lines.extend(["[Task]", f"- {task_name}"])
    _append_section(lines, "Task Introduction", data.get("background"))
    _append_section(lines, "Objective", data.get("objective"))
    _append_section(lines, "Parameters", _parameter_profile_lines(data))
    _append_section(lines, "Core Experience", data.get("core_experience"))
    _append_section(lines, "Expert Rules", data.get("expert_rules"))
    _append_section(lines, "Known Failure Modes", data.get("failure_modes"))
    _append_section(lines, "Constraints", data.get("constraints"))
    _append_section(lines, "Measurement Notes", data.get("measurement_notes"))
    _append_section(lines, "Prompt Guidance", data.get("prompt_guidance"))
    return "\n".join(lines).strip()


def _task_profile_addendum(data: Mapping[str, Any], backend: str) -> str:
    generic = _as_text_lines(data.get("prompt_addendum"))
    specific = _as_text_lines(data.get(f"{backend}_prompt_addendum"))
    lines = generic + specific
    if not lines:
        return ""
    out = ["[Backend-Specific Guidance]"]
    out.extend(f"- {line}" for line in lines)
    return "\n".join(out)


def _insert_before_section(prompt: str, section_name: str, insert_text: str) -> str:
    insert = insert_text.strip()
    if not insert:
        return prompt
    marker = f"\n[{section_name}]"
    if marker in prompt:
        return prompt.replace(marker, f"\n\n{insert}\n{marker}", 1)
    return f"{prompt.rstrip()}\n\n{insert}"


def _build_labo_system_prompt(data: Mapping[str, Any]) -> str:
    if data.get("labo_system_prompt"):
        return str(data["labo_system_prompt"]).strip()
    return (
        "You are a scientific low-fidelity evaluator for Bayesian optimization.\n"
        "Use the task introduction, core experience, expert rules, constraints, "
        "and measurement notes as soft prior knowledge when estimating objective values.\n"
        "Return only valid JSON. Do not include markdown, explanations, comments, or hidden reasoning.\n"
        "Always preserve the feature order supplied in each input point."
    )


def _build_labo_user_prompt(data: Mapping[str, Any]) -> str:
    profile = _task_profile_text(data)
    addendum = _task_profile_addendum(data, "labo")
    context_blocks = [block for block in (profile, addendum) if block]
    context = "\n\n".join(context_blocks)
    prefix = f"{context}\n\n" if context else ""
    return prefix + """[Prediction Task]
Predict low-fidelity objective values for the candidate points.

Historical measurements:
{history_json}

Candidate points:
{points_json}

Return exactly one JSON object with this schema:
{
  "data_points": [
    {"features": [0.0, 0.0], "target": 0.0}
  ]
}

Requirements:
- Include one entry for each candidate point.
- Preserve the candidate order.
- Keep feature arrays in the same order as the input.
- The target must be a numeric scalar.
- Use the task profile as prior knowledge, but do not mention it in the output.
- Output JSON only."""


def _apply_labo_task_profile(optimizer: Any, data: Mapping[str, Any]) -> None:
    if not data:
        return
    user_prompt = _build_labo_user_prompt(data)
    optimizer.user_prompt = user_prompt
    optimizer.predictor.user_prompt = user_prompt
    optimizer.generator.system_prompt = _build_labo_system_prompt(data)
    value_range = data.get("value_range")
    if isinstance(value_range, Sequence) and len(value_range) == 2:
        optimizer.generator.value_range = [float(value_range[0]), float(value_range[1])]


def _load_optional_task_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    return load_task_json(path)


def _parameters_from_data(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    parameters = list(data.get("parameters", []))
    if not parameters:
        raise ValueError("data-json must include a non-empty 'parameters' list")
    return parameters


def _feature_spec(parameters: Sequence[Mapping[str, Any]]) -> tuple[list[str], list[str], np.ndarray]:
    names: list[str] = []
    types: list[str] = []
    bounds: list[list[float]] = []
    for param in parameters:
        name = str(param.get("name", "")).strip()
        if not name:
            raise ValueError("every parameter needs a name")
        raw_bounds = param.get("bounds")
        if not isinstance(raw_bounds, Sequence) or len(raw_bounds) != 2:
            raise ValueError(f"parameter {name!r} requires bounds=[lo, hi]")
        lo, hi = float(raw_bounds[0]), float(raw_bounds[1])
        if lo >= hi:
            raise ValueError(f"parameter {name!r} has invalid bounds")
        raw_type = str(param.get("type", "continuous")).strip().lower()
        dtype = "int" if raw_type in {"int", "integer"} else "float"
        names.append(name)
        types.append(dtype)
        bounds.append([lo, hi])
    return names, types, np.asarray(bounds, dtype=float)


def _coerce_point(
    raw_point: Mapping[str, Any],
    feature_names: Sequence[str],
    feature_types: Sequence[str],
    bounds: np.ndarray,
) -> dict[str, float]:
    point: dict[str, float] = {}
    for index, name in enumerate(feature_names):
        value = float(raw_point[name])
        low, high = bounds[index]
        value = min(max(value, float(low)), float(high))
        if feature_types[index] == "int":
            value = float(int(round(value)))
        point[name] = value
    return point


def _extract_initial_point(raw: Any) -> Mapping[str, Any] | None:
    if not isinstance(raw, Mapping):
        return None
    if isinstance(raw.get("x"), Mapping):
        return raw["x"]
    if isinstance(raw.get("point"), Mapping):
        return raw["point"]
    return raw


def _lhs_initial_points(
    feature_names: Sequence[str],
    feature_types: Sequence[str],
    bounds: np.ndarray,
    n_points: int,
    seed: int,
) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    n = max(1, int(n_points))
    d = len(feature_names)
    samples = np.zeros((n, d), dtype=float)
    for dim, (low, high) in enumerate(bounds):
        width = (float(high) - float(low)) / n
        order = rng.permutation(n)
        samples[:, dim] = float(low) + order * width + rng.uniform(0.0, width, size=n)
    points = []
    for row in samples:
        raw = {name: row[index] for index, name in enumerate(feature_names)}
        points.append(_coerce_point(raw, feature_names, feature_types, bounds))
    return points


def _initial_points_from_data(
    data: Mapping[str, Any],
    feature_names: Sequence[str],
    feature_types: Sequence[str],
    bounds: np.ndarray,
    *,
    n_points: int,
    seed: int,
) -> list[dict[str, float]]:
    raw_points = list(data.get("initial_points", []) or [])
    if not raw_points:
        raw_points = list(data.get("observations", data.get("history", [])) or [])

    points = []
    for raw in raw_points:
        point = _extract_initial_point(raw)
        if point is None:
            continue
        if all(name in point for name in feature_names):
            points.append(_coerce_point(point, feature_names, feature_types, bounds))
        if len(points) >= n_points:
            break

    if len(points) >= n_points:
        return points[:n_points]
    generated = _lhs_initial_points(
        feature_names,
        feature_types,
        bounds,
        n_points=n_points - len(points),
        seed=seed + len(points),
    )
    return points + generated


def _observations_from_points(
    points: Sequence[Mapping[str, float]],
    evaluator: DryEvaluator,
    *,
    y_key: str = "y",
) -> list[dict[str, Any]]:
    observations = []
    for point in points:
        plain_point = {str(k): float(v) for k, v in point.items()}
        observations.append({"x": plain_point, y_key: evaluator.evaluate(plain_point)})
    return observations


def _observations_from_data(data: Mapping[str, Any], y_key: str) -> list[dict[str, Any]]:
    observations = list(data.get("observations", data.get("history", [])))
    if len(observations) < 2:
        raise ValueError("wet mode needs at least two observations")
    valid = []
    for obs in observations:
        if not isinstance(obs, Mapping):
            continue
        x = obs.get("x", obs.get("point"))
        if not isinstance(x, Mapping):
            continue
        if y_key not in obs and "y" not in obs:
            continue
        valid.append({"x": dict(x), "y": float(obs.get(y_key, obs.get("y")))})
    if len(valid) < 2:
        raise ValueError("wet mode needs at least two observations with x and y")
    return valid


def _labo_configs(
    *,
    n_candidates: int,
    max_loops: int,
    seed: int,
    objective_transform: float,
) -> tuple[SimpleNamespace, SimpleNamespace]:
    llm_config = SimpleNamespace(
        temperature=0.0,
        top_p=1.0,
        max_tokens=512,
        alpha=1.0,
        beta=0.0,
        value_range=None,
    )
    koh_config = SimpleNamespace(
        n_candidates=int(n_candidates),
        mismatch_threshold=0.75,
        force_hf_after_n_lf=None,
        gp_training_iter=2,
        max_loops=int(max_loops),
        acquisition_type="ucb",
        acquisition_beta=1.0,
        always_update_lf_loops=1,
        random_seed=int(seed),
    )
    llm_config.objective_transform = float(objective_transform)
    return llm_config, koh_config


def run_labo_dry(
    output_dir: str | Path,
    *,
    online: bool = False,
    rounds: int = 1,
    batch_q: int = 1,
    seed: int = 7,
    task_json: str | Path | None = None,
    evaluator_spec: str | None = None,
) -> dict[str, Any]:
    """Run LABO in dry mode with an internal or user-provided evaluator."""
    from .methods.labo.koh.optimizer import KOHOptimizer

    task_profile = _load_optional_task_json(task_json)
    out = Path(output_dir) / "labo_dry"
    out.mkdir(parents=True, exist_ok=True)

    if evaluator_spec:
        evaluator, feature_names, feature_types, bounds = _load_dry_evaluator(evaluator_spec, task_profile)
        goal = str(task_profile.get("goal", "max"))
        sign = _goal_sign(goal)
        blackbox = SignedEvaluator(evaluator, sign)
        objective_transform = sign
        task_name = str(task_profile.get("task_name", "custom_dry_task"))
        initial_points = _initial_points_from_data(
            task_profile,
            feature_names,
            feature_types,
            bounds,
            n_points=max(3, int(task_profile.get("n_initial_points", 3))),
            seed=seed,
        )
        n_candidates = int(task_profile.get("n_candidates", 64 if online else 96))
    else:
        goal = "max"
        objective_transform = 1.0
        blackbox = SyntheticBlackBox()
        feature_names = ["x1", "x2"]
        feature_types = ["float", "float"]
        bounds = blackbox.bounds
        task_name = "synthetic"
        initial_points = [
            {"x1": 0.10, "x2": 0.10},
            {"x1": 0.50, "x2": 0.50},
            {"x1": 0.90, "x2": 0.30},
        ]
        n_candidates = 24 if online else 40

    llm_config, koh_config = _labo_configs(
        n_candidates=n_candidates,
        max_loops=max(1, int(rounds) * 2),
        seed=seed,
        objective_transform=objective_transform,
    )
    optimizer = KOHOptimizer(
        task_name=task_name,
        task_data_dir=str(out),
        feature_names=feature_names,
        feature_types=feature_types,
        bounds=bounds,
        target_name="saibo_score" if evaluator_spec else "objective",
        llm_client=_labo_client(online),
        hf_blackbox=blackbox,
        llm_config=llm_config,
        koh_config=koh_config,
        file_prefix="dry",
        objective_transform=objective_transform,
    )
    _apply_labo_task_profile(optimizer, task_profile)
    optimizer.run(
        max_iterations=max(1, int(rounds)),
        n_initial_points=len(initial_points),
        q=max(1, int(batch_q)),
        fixed_initial_points=initial_points,
    )
    best_score = optimizer._best_y()
    payload = {
        "interface": "dry",
        "method": "labo",
        "online": bool(online),
        "status": "ok",
        "goal": goal,
        "best_score": best_score,
        "best_objective": objective_transform * best_score,
        "iterations": optimizer.iteration_log,
        "task_profile": str(task_json) if task_json else None,
        "evaluator": evaluator_spec,
        "output_dir": str(out),
    }
    payload["summary_path"] = str(_write_summary(out, "summary.json", payload))
    return _json_ready(payload)


def _labo_wet_optimizer(
    data: Mapping[str, Any],
    output_dir: str | Path,
    *,
    online: bool,
    batch_q: int,
    seed: int,
):
    from .methods.labo.koh.optimizer import KOHOptimizer

    parameters = _parameters_from_data(data)
    feature_names, feature_types, bounds = _feature_spec(parameters)
    y_key = str(data.get("y_key", data.get("target_name", "y")))
    goal = str(data.get("goal", "max"))
    sign = _goal_sign(goal)
    observations = _observations_from_data(data, y_key)
    if len(observations) < 3:
        raise ValueError("LABO wet mode needs at least three observations for KOH fitting")

    llm_config, koh_config = _labo_configs(
        n_candidates=int(data.get("n_candidates", 128)),
        max_loops=1,
        seed=seed,
        objective_transform=sign,
    )
    out = Path(output_dir) / "labo_wet"
    out.mkdir(parents=True, exist_ok=True)
    optimizer = KOHOptimizer(
        task_name=str(data.get("task_name", "wet_task")),
        task_data_dir=str(out),
        feature_names=feature_names,
        feature_types=feature_types,
        bounds=bounds,
        target_name="saibo_score",
        llm_client=_labo_client(online) if online else DeterministicLowFidelityClient(),
        hf_blackbox=NoHighFidelityEvaluator(),
        llm_config=llm_config,
        koh_config=koh_config,
        file_prefix="wet",
        objective_transform=sign,
    )
    _apply_labo_task_profile(optimizer, data)
    for index, obs in enumerate(observations, start=1):
        x = {name: float(obs["x"][name]) for name in feature_names}
        optimizer.data_manager.add_hf_experiment(x, sign * float(obs["y"]), iteration=index)
    optimizer.data_manager.save_all()
    return optimizer, observations, goal, y_key, max(1, int(batch_q or data.get("batch_q", 1)))


def run_labo_wet(
    data_json: str | Path,
    output_dir: str | Path,
    *,
    online: bool = False,
    batch_q: int = 1,
    seed: int = 11,
) -> dict[str, Any]:
    """Plan one LABO wet step without calling high-fidelity evaluation."""
    from .methods.labo.koh.utils import numpy_to_dict_list

    data = load_task_json(data_json)
    optimizer, observations, goal, y_key, q = _labo_wet_optimizer(
        data,
        output_dir,
        online=online,
        batch_q=batch_q,
        seed=seed,
    )
    optimizer._train_models(force_recompute_lf=True, loop_count=1)
    candidates = optimizer._sample_candidates(
        n_samples=int(data.get("n_candidates", 128)),
        loop_count=1,
    )
    mu_h, sigma2_h, _mu_delta, sigma2_delta = optimizer._koh_posterior_predict(candidates)
    selected_indices, scores = optimizer._select_q_points(candidates, mu_h, sigma2_h, q=q)
    selected = [candidates[index] for index in selected_indices]
    do_hf, ratio, ratios = optimizer.mismatch_decision.decide(selected_indices, sigma2_delta, sigma2_h)
    points = _points_to_plain(numpy_to_dict_list(np.asarray(selected), optimizer.feature_names))
    recommended_fidelity = "high_fidelity" if do_hf else "low_fidelity"

    lf_predictions = []
    if not do_hf:
        means, variances = optimizer.predictor.predict_batch(points, optimizer.data_manager.get_history_data())
        lf_predictions = [
            {"mean": float(mean), "variance": float(variance)}
            for mean, variance in zip(means, variances)
        ]

    payload = {
        "interface": "wet",
        "method": "labo",
        "online": bool(online),
        "status": "ok",
        "source": str(data_json),
        "goal": goal,
        "y_key": y_key,
        "n_observations": len(observations),
        "recommended_fidelity": recommended_fidelity,
        "mismatch_ratio": float(ratio),
        "selected_ratios": [float(value) for value in ratios],
        "points": points,
        "acquisition_scores": [float(scores[index]) for index in selected_indices],
        "lf_predictions": lf_predictions,
        "output_dir": str(Path(output_dir) / "labo_wet"),
    }
    payload["summary_path"] = str(_write_summary(payload["output_dir"], "summary.json", payload))
    return _json_ready(payload)


def _lgbo_space_and_history(data: Mapping[str, Any]):
    from .methods.lgbo.lgbo_core import ContinuousSpace, tensor_from_observations

    parameters = _parameters_from_data(data)
    space = ContinuousSpace.from_parameters(parameters)
    y_key = str(data.get("y_key", "y"))
    observations = list(data.get("observations", data.get("history", [])))
    X_hist, Y_hist = tensor_from_observations(observations, space, y_key=y_key)
    return space, observations, X_hist, Y_hist, y_key


def _lgbo_user_prompt(data: Mapping[str, Any], observations: Sequence[Mapping[str, Any]], batch_q: int, y_key: str) -> str:
    from .methods.lgbo.prompt import build_user_prompt

    prompt = build_user_prompt(
        background=str(data.get("background", "Optimize a bounded scientific objective.")),
        parameters=_parameters_from_data(data),
        objective=str(data.get("objective", "Optimize the target response.")),
        constraints=data.get("constraints", ""),
        history=observations,
        batch_q=batch_q,
        y_key=y_key,
        extra_request=data.get("extra_request"),
    )
    context_blocks = [
        _task_profile_text(
            {
                "core_experience": data.get("core_experience"),
                "expert_rules": data.get("expert_rules"),
                "failure_modes": data.get("failure_modes"),
                "measurement_notes": data.get("measurement_notes"),
                "prompt_guidance": data.get("prompt_guidance"),
            }
        ),
        _task_profile_addendum(data, "lgbo"),
    ]
    context = "\n\n".join(block for block in context_blocks if block)
    return _insert_before_section(prompt, "History", context)


def _lgbo_preference(data: Mapping[str, Any], user_prompt: str, *, online: bool) -> tuple[str, dict[str, Any]]:
    from .methods.lgbo.prompt import DEFAULT_SYSTEM_PROMPT, parse_assistant_response

    system_prompt = str(data.get("lgbo_system_prompt", data.get("system_prompt", DEFAULT_SYSTEM_PROMPT)))
    if online:
        assistant_text = _call_lgbo_online(system_prompt, user_prompt)
    else:
        fallback = data.get("offline_preference")
        if fallback is None:
            parameters = _parameters_from_data(data)
            center = []
            for param in parameters:
                lo, hi = param["bounds"]
                center.append((float(lo) + float(hi)) * 0.5)
            fallback = f"[point, {center}, 0.7]"
        assistant_text = f"Final Answer:\n{fallback}"
    parsed = parse_assistant_response(assistant_text)
    if not parsed.get("mode") and online:
        assistant_text = _call_lgbo_online(
            system_prompt,
            user_prompt
            + "\n\nOutput only one final answer line in the required bracketed format.",
        )
        parsed = parse_assistant_response(assistant_text)
    if not parsed.get("mode"):
        raise RuntimeError("Could not parse LGBO LLM preference.")
    return assistant_text, parsed


def run_lgbo_wet(
    data_json: str | Path,
    output_dir: str | Path,
    *,
    online: bool = False,
    batch_q: int = 1,
    seed: int = 11,
) -> dict[str, Any]:
    """Plan one LGBO wet step without evaluating the objective."""
    from .methods.lgbo.lgbo_core import propose_lgbo_batch, serialize_plan

    data = load_task_json(data_json)
    space, observations, X_hist, Y_hist, y_key = _lgbo_space_and_history(data)
    q = max(1, int(batch_q or data.get("batch_q", 1)))
    user_prompt = _lgbo_user_prompt(data, observations, q, y_key)
    assistant_text, parsed = _lgbo_preference(data, user_prompt, online=online)
    points, plan, z_new = propose_lgbo_batch(
        X_norm=X_hist,
        y=Y_hist,
        parsed_preference=parsed,
        space=space,
        goal=str(data.get("goal", "max")),
        batch_q=q,
        policy=str(data.get("policy", "tilt")),
        grid_size=int(data.get("grid_size", 64)),
        guidance_scale=float(data.get("guidance_scale", 3.0)),
        cand_size=int(data.get("cand_size", 256)),
        num_paths_batch=int(data.get("num_paths_batch", 32)),
        seed=seed,
    )
    payload = {
        "interface": "wet",
        "method": "lgbo",
        "online": bool(online),
        "status": "ok",
        "source": str(data_json),
        "goal": str(data.get("goal", "max")),
        "y_key": y_key,
        "n_observations": len(observations),
        "parsed_preference": parsed,
        "assistant_text": assistant_text,
        "lgbo_plan": serialize_plan(plan),
        "normalized_points": z_new.tolist(),
        "points": points,
        "output_dir": str(Path(output_dir) / "lgbo_wet"),
    }
    payload["summary_path"] = str(_write_summary(payload["output_dir"], "summary.json", payload))
    return _json_ready(payload)


def run_lgbo_dry(
    output_dir: str | Path,
    *,
    online: bool = False,
    rounds: int = 1,
    batch_q: int = 1,
    seed: int = 7,
    task_json: str | Path | None = None,
    evaluator_spec: str | None = None,
) -> dict[str, Any]:
    """Run LGBO in dry mode on an internal or user-provided evaluator."""
    from .methods.lgbo.fun.toy_fun import get_bo_bounds, toy_results
    from .methods.lgbo.lgbo_core import (
        ContinuousSpace,
        propose_lgbo_batch,
        serialize_plan,
        tensor_from_observations,
    )
    from .methods.lgbo.prompt import build_toy_user_prompt
    from .smoke import _ackley_history

    task_profile = _load_optional_task_json(task_json)
    out = Path(output_dir) / "lgbo_dry"
    out.mkdir(parents=True, exist_ok=True)

    if evaluator_spec:
        evaluator, feature_names, feature_types, bounds = _load_dry_evaluator(evaluator_spec, task_profile)
        if any(dtype == "int" for dtype in feature_types):
            raise ValueError("LGBO dry currently supports continuous custom evaluator parameters only.")
        parameters = task_profile.get("parameters") or [
            {"name": name, "type": "continuous", "bounds": [float(lo), float(hi)]}
            for name, (lo, hi) in zip(feature_names, bounds)
        ]
        run_data = {**task_profile, "parameters": parameters}
        y_key = str(run_data.get("y_key", "y"))
        goal = str(run_data.get("goal", "max"))
        space = ContinuousSpace.from_parameters(parameters)
        initial_points = _initial_points_from_data(
            run_data,
            feature_names,
            feature_types,
            bounds,
            n_points=max(2, int(run_data.get("n_initial_points", 3))),
            seed=seed,
        )
        observations = _observations_from_points(initial_points, evaluator, y_key=y_key)
        X_hist, Y_hist = tensor_from_observations(observations, space, y_key=y_key)
        function_name = str(run_data.get("task_name", "custom_dry_task"))
        best_value = float(Y_hist.max().item()) if goal == "max" else float(Y_hist.min().item())
    else:
        parameters = [
            {"name": f"x{i + 1}", "bounds": list(bounds)}
            for i, bounds in enumerate(get_bo_bounds("ackley", 2))
        ]
        run_data = {
            **task_profile,
            "parameters": task_profile.get("parameters", parameters),
            "goal": "min",
            "y_key": "f",
        }
        y_key = "f"
        goal = "min"
        function_name = "ackley"
        space = ContinuousSpace.from_parameters(parameters)
        X_hist, Y_hist, history = _ackley_history(space, toy_results)
        observations = []
        for item in history:
            left, value = item.split(" -> f=")
            point = {}
            for pair in left.split(", "):
                name, raw_value = pair.split("=")
                point[name] = float(raw_value)
            observations.append({"x": point, y_key: float(value)})
        best_value = float(Y_hist.min().item())

    rows: list[dict[str, Any]] = []
    round_payloads: list[dict[str, Any]] = []

    for round_id in range(1, max(1, int(rounds)) + 1):
        if evaluator_spec:
            user_prompt = _lgbo_user_prompt(run_data, observations, max(1, int(batch_q)), y_key)
        else:
            history = [
                ", ".join(f"{name}={obs['x'][name]:.6g}" for name in space.names)
                + f" -> f={float(obs[y_key]):.6g}"
                for obs in observations
            ]
            user_prompt = build_toy_user_prompt(
                func_name="ackley",
                d=2,
                bounds=space.bounds,
                history=history,
                batch_q=max(1, int(batch_q)),
            )
        context_blocks = [
            _task_profile_text(
                {
                    "background": task_profile.get("background"),
                    "objective": task_profile.get("objective"),
                    "core_experience": task_profile.get("core_experience"),
                    "expert_rules": task_profile.get("expert_rules"),
                    "failure_modes": task_profile.get("failure_modes"),
                    "measurement_notes": task_profile.get("measurement_notes"),
                    "prompt_guidance": task_profile.get("prompt_guidance"),
                }
            ),
            _task_profile_addendum(task_profile, "lgbo"),
        ]
        user_prompt = _insert_before_section(
            user_prompt,
            "History",
            "\n\n".join(block for block in context_blocks if block),
        )
        assistant_text, parsed = _lgbo_preference(
            run_data,
            user_prompt,
            online=online,
        )
        points, plan, z_new = propose_lgbo_batch(
            X_norm=X_hist,
            y=Y_hist,
            parsed_preference=parsed,
            space=space,
            goal=goal,
            batch_q=max(1, int(batch_q)),
            policy="tilt",
            grid_size=64,
            guidance_scale=3.0,
            cand_size=256,
            num_paths_batch=32,
            seed=seed + round_id,
        )
        if evaluator_spec:
            values = [evaluator.evaluate(point) for point in points]
        else:
            vectors = [[point[name] for name in space.names] for point in points]
            values = [float(result["f"]) for result in toy_results("ackley", vectors)]
        y_new = torch.tensor([[float(value)] for value in values], dtype=torch.double)
        X_new = torch.tensor([space.normalize_point(point) for point in points], dtype=torch.double)
        X_hist = torch.cat([X_hist, X_new], dim=0)
        Y_hist = torch.cat([Y_hist, y_new], dim=0)
        for idx, (point, value) in enumerate(zip(points, values), start=1):
            value = float(value)
            if goal == "max":
                best_value = max(best_value, value)
            else:
                best_value = min(best_value, value)
            observations.insert(0, {"x": point, y_key: value})
            rows.append({"round": round_id, "index": idx, **point, y_key: value, f"best_{y_key}": best_value})
        round_payloads.append(
            {
                "round": round_id,
                "parsed_preference": parsed,
                "lgbo_plan": serialize_plan(plan),
                "normalized_points": z_new.tolist(),
                "points": points,
            }
        )

    payload = {
        "interface": "dry",
        "method": "lgbo",
        "online": bool(online),
        "status": "ok",
        "function": function_name,
        "goal": goal,
        f"best_{y_key}": best_value,
        "rounds": round_payloads,
        "points": rows,
        "task_profile": str(task_json) if task_json else None,
        "evaluator": evaluator_spec,
        "output_dir": str(out),
    }
    payload["summary_path"] = str(_write_summary(out, "summary.json", payload))
    return _json_ready(payload)


def run_dry(
    method: str,
    output_dir: str | Path,
    *,
    online: bool = False,
    rounds: int = 1,
    batch_q: int = 1,
    seed: int = 7,
    task_json: str | Path | None = None,
    evaluator_spec: str | None = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Run the SAIBO dry interface for one or all methods."""
    key = method.strip().lower()
    if key == "all":
        return [
            run_labo_dry(
                output_dir,
                online=online,
                rounds=rounds,
                batch_q=batch_q,
                seed=seed,
                task_json=task_json,
                evaluator_spec=evaluator_spec,
            ),
            run_lgbo_dry(
                output_dir,
                online=online,
                rounds=rounds,
                batch_q=batch_q,
                seed=seed,
                task_json=task_json,
                evaluator_spec=evaluator_spec,
            ),
        ]
    if key == "labo":
        return run_labo_dry(
            output_dir,
            online=online,
            rounds=rounds,
            batch_q=batch_q,
            seed=seed,
            task_json=task_json,
            evaluator_spec=evaluator_spec,
        )
    if key == "lgbo":
        return run_lgbo_dry(
            output_dir,
            online=online,
            rounds=rounds,
            batch_q=batch_q,
            seed=seed,
            task_json=task_json,
            evaluator_spec=evaluator_spec,
        )
    raise ValueError("method must be 'labo', 'lgbo', or 'all'")


def run_wet(
    method: str,
    data_json: str | Path,
    output_dir: str | Path,
    *,
    online: bool = False,
    batch_q: int = 1,
    seed: int = 11,
) -> dict[str, Any]:
    """Run the SAIBO wet interface for one method."""
    key = method.strip().lower()
    if key == "labo":
        return run_labo_wet(data_json, output_dir, online=online, batch_q=batch_q, seed=seed)
    if key == "lgbo":
        return run_lgbo_wet(data_json, output_dir, online=online, batch_q=batch_q, seed=seed)
    raise ValueError("wet mode requires method 'labo' or 'lgbo'")
