# SAIBO Task Profile

SAIBO uses a shared task profile JSON to pass task introduction and core
experience into LABO and LGBO prompts.

Dry mode accepts an optional profile:

```bash
saibo dry --method all --task-json examples/task_profile.example.json
saibo dry --method labo --task-json examples/task_profile.example.json --evaluator examples/custom_evaluator.py:ExampleBlackBox
```

Wet mode uses the same fields inside the required data file:

```bash
saibo wet --method labo --data-json examples/wet_input.example.json
saibo wet --method lgbo --data-json examples/wet_input.example.json
```

## Fields

```json
{
  "task_name": "short_task_name",
  "background": "What is being optimized and why.",
  "objective": "What target should be maximized or minimized.",
  "goal": "max",
  "parameters": [
    {
      "name": "x1",
      "type": "continuous",
      "bounds": [0.0, 1.0],
      "description": "Meaning of this variable."
    }
  ],
  "initial_points": [
    {"x1": 0.1}
  ],
  "core_experience": [
    "Expert prior, literature rule, mechanism hint, or practical observation."
  ],
  "expert_rules": [
    "Rules the LLM should treat as soft guidance unless stated otherwise."
  ],
  "failure_modes": [
    "Known risky, unstable, or misleading regions."
  ],
  "constraints": [
    "Hard or soft constraints for candidate reasoning."
  ],
  "measurement_notes": [
    "Noise, scale, assay, simulator, or measurement interpretation notes."
  ],
  "prompt_guidance": [
    "General guidance shared by LABO and LGBO prompts."
  ],
  "labo_prompt_addendum": [
    "Extra guidance only for LABO low-fidelity value prediction."
  ],
  "lgbo_prompt_addendum": [
    "Extra guidance only for LGBO point or region preference."
  ]
}
```

## Backend Use

LABO injects the profile into its low-fidelity value-prediction prompt. The LLM
must still return only JSON predictions.

LGBO injects the profile into its point/region preference prompt. The LLM must
still return a parseable point or region plus confidence.

## Dry Evaluator

Custom dry runs can provide an evaluator:

```bash
saibo dry --method lgbo --task-json examples/task_profile.example.json --evaluator examples/custom_evaluator.py:evaluate
```

The evaluator can be a class instance target:

```python
class MyBlackBox:
    feature_names = ["x1", "x2"]
    feature_types = ["float", "float"]
    bounds = [[0.0, 1.0], [0.0, 1.0]]

    def evaluate(self, point):
        return float(...)
```

or a function:

```python
def evaluate(point):
    return float(...)
```

When `task-json` includes `parameters`, SAIBO uses those parameter names and
bounds. Otherwise, it reads `feature_names`, `feature_types`, and `bounds` from
the evaluator object.
