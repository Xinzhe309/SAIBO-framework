<p align="center">
  <img src="SAIBO.png" alt="SAIBO logo" width="220">
</p>

<h3 align="center">Scientific Artificial Intelligence Bayesian Optimization</h3>




<p align="center">
  <img alt="Status" src="https://img.shields.io/badge/Status-Early%20Research%20Release-facc15">
  <img alt="Methods" src="https://img.shields.io/badge/Base%20Methods-LABO%20%7C%20LGBO-6f42c1">
  <img alt="BO" src="https://img.shields.io/badge/Core-Bayesian%20Optimization-f97316">
</p>

---

## What is SAIBO?

**SAIBO** stands for **Scientific Artificial Intelligence Bayesian Optimization**.

SAIBO is a research framework for bringing scientific reasoning agents into Bayesian optimization loops. It is designed for scientific discovery problems where experiments are expensive, data are scarce, search spaces are large, and useful prior knowledge already exists in papers, expert rules, physical mechanisms, historical observations, and cross-domain analogies.

In scientific discovery, generating candidates is only the first step. The harder question is:

> Given limited experimental budget, what should we try next?

SAIBO treats this as an agent-native black-box optimization problem. Large language models and scientific agents provide structured reasoning signals, while Bayesian optimization remains responsible for uncertainty-aware search and experimental decision-making.

In plain terms: **SAIBO lets agents reason like scientific collaborators, and lets Bayesian optimization decide how to spend real experimental budget.**

---

## Why SAIBO?

Classical Bayesian optimization is powerful, but early-stage scientific optimization is often far from classical:

| Scientific optimization challenge | SAIBO design direction |
|---|---|
| Initial data are extremely limited. | Use LLM and agent reasoning to inject scientific prior knowledge. |
| Experiments are expensive and slow. | Reserve high-fidelity evaluations for candidates that are worth real cost. |
| Search spaces are high-dimensional, mixed, and structured. | Represent candidate spaces through task-aware continuous, discrete, and semantic transformations. |
| Existing literature and expert knowledge are underused. | Use agents as carriers of scientific reasoning, retrieval, reflection, and belief updates. |
| Direct LLM proposals can be brittle. | Keep GP surrogates and acquisition functions in control of final optimization decisions. |

SAIBO is built around a simple belief:

**scientific reasoning should not replace optimization; it should become a first-class signal inside the optimization loop.**

---

## Current Public Base Methods

This repository is being organized around two base methods that are being prepared for public release.

### LABO: LLM-Accelerated Bayesian Optimization

**LABO** uses an LLM as a low-fidelity oracle inside Bayesian optimization.

It supports broad low-cost exploration with LLM predictions, then selectively triggers high-fidelity experiments when model uncertainty suggests that real evidence is needed. The method is built around a multi-fidelity surrogate that fuses LLM-fidelity observations with true experimental observations.

Core idea:

```text
LLM prediction = cheap low-fidelity signal
Real experiment = expensive high-fidelity signal
BO decides when each signal is enough
```

### LGBO: LLM-Guided Bayesian Optimization

**LGBO** uses LLM preferences to guide Bayesian optimization.

Instead of asking the LLM to directly solve the optimization problem, LGBO asks it to identify promising points or regions. These preferences are converted into stable surrogate-model updates, allowing the optimizer to lean toward scientifically plausible regions while preserving uncertainty-aware exploration.

Core idea:

```text
LLM suggests where the search should lean
BO decides which experiment to run next
```

Together, LABO and LGBO form the first public SAIBO foundation:

- LABO focuses on **LLM-as-low-fidelity evaluation**.
- LGBO focuses on **LLM-as-preference guidance**.
- SAIBO unifies both under an agent-native scientific optimization framework.

---

## Quick Start

Install from a source checkout:

```bash
pip install -r requirements.txt
pip install -e .
```

Run the built-in smoke checks:

```bash
saibo smoke --method all
```

Run dry optimization with the built-in example benchmarks:

```bash
saibo dry --method labo --rounds 1
saibo dry --method lgbo --rounds 1
saibo dry --method all --rounds 1
```

Run wet planning from observed history:

```bash
saibo wet --method labo --data-json examples/wet_input.example.json
saibo wet --method lgbo --data-json examples/wet_input.example.json
```

Dry mode owns an evaluator and closes the loop automatically. Wet mode reads existing observations and returns the next planned batch without calling a high-fidelity evaluator.

---

## Task Profiles

SAIBO uses a shared task profile JSON to inject task introduction and core experience into LABO and LGBO prompts.

```bash
saibo dry --method all --task-json examples/task_profile.example.json
```

The same fields are used in wet mode through `--data-json`:

```bash
saibo wet --method labo --data-json examples/wet_input.example.json
saibo wet --method lgbo --data-json examples/wet_input.example.json
```

Important fields:

```json
{
  "background": "What is being optimized and why.",
  "objective": "What target should be maximized or minimized.",
  "goal": "max",
  "parameters": [
    {"name": "x1", "type": "continuous", "bounds": [0.0, 1.0]}
  ],
  "core_experience": [
    "Expert prior, mechanism hint, or practical observation."
  ],
  "expert_rules": [
    "Soft or hard rules for candidate reasoning."
  ],
  "constraints": [
    "Feasibility or safety constraints."
  ],
  "measurement_notes": [
    "Noise, assay, simulator, or interpretation notes."
  ]
}
```

LABO injects this profile into its low-fidelity value-prediction prompt and still requires JSON numeric predictions. LGBO injects it into its point/region preference prompt and still requires a parseable point or region plus confidence.

More details are in `PROMPT_PROFILE.md`.

---

## Custom Dry Evaluators

For user-defined dry experiments, provide an evaluator with `--evaluator`:

```bash
saibo dry --method labo \
  --task-json examples/task_profile.example.json \
  --evaluator examples/custom_evaluator.py:ExampleBlackBox

saibo dry --method lgbo \
  --task-json examples/task_profile.example.json \
  --evaluator examples/custom_evaluator.py:evaluate
```

The evaluator can be a class:

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

If `task-json` includes `parameters`, SAIBO uses those names and bounds. Otherwise, it reads `feature_names`, `feature_types`, and `bounds` from the evaluator object.

---

## Online LLM Calls

Offline mode is the default for smoke and example checks. To call Intern S1, set an API key and pass `--online`:

```bash
export API_KEY=your_key_here
export INTERN_S1_API_KEY=your_key_here

saibo dry --method lgbo --online --task-json examples/task_profile.example.json
saibo wet --method labo --online --data-json examples/wet_input.example.json
```

Do not commit real API keys. `.env.example` documents the supported environment variables.

---

## Roadmap

The following components are part of the broader SAIBO direction but are not included in the initial public release.

| Module | Status | Direction |
|---|---|---|
| Graph-based Bayesian Optimization | Planned | Semantic-feature Bayesian optimization for chemistry and reaction discovery. |
| Latent-space BO | Planned | Mixed continuous, discrete, and categorical search through task-aware latent representations. |
| Multi-objective BO | Planned | Modular acquisition strategies such as qEHVI, scalarization BO, and surrogate-assisted evolutionary search. |
| Multi-scale agents | Planned | Micro, meso, and macro scientific agents coordinated through shared uncertainty and evidence states. |
| Agent-ready knowledge base | Planned | Literature, multimodal data, memory, and retrieval services for scientific reasoning agents. |
| Decision-model post-training | Planned | Training smaller decision controllers for gating, risk control, and experiment escalation. |

These modules are listed to clarify the research direction. They are not yet released as stable code.

---

## Framework View

SAIBO connects scientific reasoning and black-box optimization:

```text
Scientific knowledge
  papers, mechanisms, expert rules, historical data
        |
        v
Scientific reasoning agent
  retrieval, hypothesis generation, region judgment, reflection
        |
        v
Optimization signal
  low-fidelity predictions, preferences, constraints, uncertainty hints
        |
        v
Bayesian optimization loop
  surrogate model, acquisition, gating, candidate selection
        |
        v
High-fidelity experiment or simulator
  observation, feedback, belief update
```

The goal is not to make the LLM the optimizer. The goal is to make scientific reasoning usable, auditable, and useful inside optimization.

---

## Release Notice

SAIBO is currently an early research release.

The initial public version focuses on the two base methods, **LABO** and **LGBO**. Other modules described in the framework are ongoing work and may change significantly before release.

This repository is intended to serve as the public home for the SAIBO project while the codebase, examples, documentation, and experiments are gradually organized.

---

## License

SAIBO is released under the MIT License. See [LICENSE](LICENSE) for details.

---

## Citation

If you use SAIBO, please cite the relevant base methods.

LABO:

```bibtex
@article{chen2026labo,
  title={LABO: LLM-Accelerated Bayesian Optimization through Broad Exploration and Selective Experimentation},
  author={Chen, Zhuo and Yuan, Xinzhe and Zhang, Jianshu and Dong, Jinzong and Zhou, Ruichen and Niu, Yingchun and Zhou, Tianhang and Liu, Yu Yang Fredrik and Li, Yuqiang and Ye, Nanyang and others},
  journal={arXiv preprint arXiv:2605.22054},
  year={2026}
}
```

LGBO:

```bibtex
@inproceedings{yuanunleashing,
  title={Unleashing LLMs in Bayesian Optimization: Preference-Guided Framework for Scientific Discovery},
  author={Yuan, Xinzhe and Chen, Zhuo and Zhang, Jianshu and Xiong, Huan and Ye, Nanyang and Li, Yuqiang and Gu, Qinying},
  booktitle={The Fourteenth International Conference on Learning Representations}
}
```
