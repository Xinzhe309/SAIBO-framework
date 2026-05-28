<p align="center">
  <img src="SAIBO.png" alt="SAIBO logo" width="220">
</p>

<h3 align="center">科学智能贝叶斯优化</h3>

<p align="center">
  <b>Scientific Artificial Intelligence Bayesian Optimization</b>
</p>

<p align="center">
  <b>面向科学发现的 Agent 原生贝叶斯优化框架</b>
</p>

<p align="center">
  <img alt="Status" src="https://img.shields.io/badge/Status-Early%20Research%20Release-facc15">
  <img alt="Methods" src="https://img.shields.io/badge/Base%20Methods-LABO%20%7C%20LGBO-6f42c1">
  <img alt="BO" src="https://img.shields.io/badge/Core-Bayesian%20Optimization-f97316">
</p>

---

## SAIBO 是什么？

**SAIBO** 是 **Scientific Artificial Intelligence Bayesian Optimization** 的缩写，中文名为 **科学智能贝叶斯优化**。

SAIBO 是一个面向科学发现的研究框架，目标是把大模型、科学智能体和贝叶斯优化统一到同一个实验决策闭环中。它关注的是一类常见的科学优化问题：实验昂贵、数据稀缺、搜索空间大，同时又存在可利用的先验知识，例如论文、专家经验、物理机制、历史观测和跨领域类比。

在科学发现中，生成候选只是第一步。真正困难的问题通常是：

> 在有限实验预算下，下一次实验到底应该做什么？

SAIBO 将这一问题视为 Agent 原生的黑盒优化问题：大语言模型和科学智能体提供结构化的科学推理信号，贝叶斯优化负责不确定性建模、采集和实验决策。

简单来说：**SAIBO 让 Agent 像科学合作者一样推理，让贝叶斯优化决定如何花费真实实验预算。**

---

## 为什么需要 SAIBO？

经典贝叶斯优化已经是科学发现中的重要工具，但真实科学优化问题往往比标准 BO 设置更复杂：

| 科学优化挑战 | SAIBO 的设计方向 |
|---|---|
| 初始数据极少。 | 利用 LLM 和 Agent 推理注入科学先验知识。 |
| 实验昂贵且周期长。 | 只在真正值得的时候触发高保真实验。 |
| 搜索空间高维、混合且结构化。 | 通过任务相关的连续、离散和语义表示来组织搜索。 |
| 文献和专家知识难以进入优化模型。 | 将 Agent 作为科学推理、检索和经验注入的载体。 |
| 直接让 LLM 给答案容易不稳定。 | 保留 GP 代理模型和 acquisition function 对最终实验选择的控制权。 |

SAIBO 的核心观点是：

**科学推理不应该替代优化，而应该成为优化循环中的一类一等信号。**

---

## 当前公开基座方法

当前 SAIBO 围绕两个基座方法进行整理和发布：**LABO** 与 **LGBO**。

### LABO：LLM-Accelerated Bayesian Optimization

**LABO** 将 LLM 作为贝叶斯优化中的低保真 oracle。

它使用 LLM 预测进行低成本、广覆盖探索，再在代理模型认为需要真实证据的位置触发高保真实验。该方法通过多保真代理模型融合 LLM 低保真信号和真实实验高保真观测。

核心思想：

```text
LLM prediction = 廉价低保真信号
Real experiment = 昂贵高保真信号
BO decides when each signal is enough
```

### LGBO：LLM-Guided Bayesian Optimization

**LGBO** 将 LLM 的偏好判断引入贝叶斯优化。

LGBO 不要求 LLM 直接解决优化问题，而是让 LLM 判断哪些点或区域更值得探索。随后，这些点偏好或区域偏好会被转化为稳定的代理模型更新，使优化器在保持不确定性探索能力的同时，向更符合科学先验的区域倾斜。

核心思想：

```text
LLM suggests where the search should lean
BO decides which experiment to run next
```

LABO 与 LGBO 构成 SAIBO 第一阶段的公开基础：

- LABO 关注 **LLM-as-low-fidelity evaluation**。
- LGBO 关注 **LLM-as-preference guidance**。
- SAIBO 将二者统一到 Agent 原生科学优化框架下。

---

## 快速开始

从源码目录安装：

```bash
pip install -r requirements.txt
pip install -e .
```

运行内置 smoke test：

```bash
saibo smoke --method all
```

运行内置干实验示例：

```bash
saibo dry --method labo --rounds 1
saibo dry --method lgbo --rounds 1
saibo dry --method all --rounds 1
```

从已有观测记录进行湿实验规划：

```bash
saibo wet --method labo --data-json examples/wet_input.example.json
saibo wet --method lgbo --data-json examples/wet_input.example.json
```

`dry` 模式拥有 evaluator，会自动评估目标函数并闭环更新。`wet` 模式只读取已有观测，输出下一批推荐点，不会调用真实高保真实验。

---

## 任务 Profile

SAIBO 使用共享的任务 profile JSON，把任务介绍和核心经验注入 LABO 与 LGBO 的 prompt。

```bash
saibo dry --method all --task-json examples/task_profile.example.json
```

湿实验模式也使用同样字段，只是通过 `--data-json` 输入，同时包含已有观测：

```bash
saibo wet --method labo --data-json examples/wet_input.example.json
saibo wet --method lgbo --data-json examples/wet_input.example.json
```

常用字段：

```json
{
  "background": "正在优化什么，以及为什么优化。",
  "objective": "需要最大化或最小化的目标。",
  "goal": "max",
  "parameters": [
    {"name": "x1", "type": "continuous", "bounds": [0.0, 1.0]}
  ],
  "core_experience": [
    "专家经验、文献规律、机制判断或实践观察。"
  ],
  "expert_rules": [
    "用于候选判断的软规则或硬规则。"
  ],
  "constraints": [
    "可行性、安全性或实验约束。"
  ],
  "measurement_notes": [
    "噪声、测量方式、模拟器或结果解释说明。"
  ]
}
```

LABO 会把这些信息注入低保真数值预测 prompt，但仍然要求 LLM 只输出 JSON 数值预测。LGBO 会把这些信息注入点/区域偏好 prompt，但仍然要求 LLM 输出可解析的 point 或 region 以及 confidence。

更多字段说明见 `PROMPT_PROFILE.md`。

---

## 自定义干实验 Evaluator

用户可以通过 `--evaluator` 接入自己的干实验函数模型：

```bash
saibo dry --method labo \
  --task-json examples/task_profile.example.json \
  --evaluator examples/custom_evaluator.py:ExampleBlackBox

saibo dry --method lgbo \
  --task-json examples/task_profile.example.json \
  --evaluator examples/custom_evaluator.py:evaluate
```

evaluator 可以是一个类：

```python
class MyBlackBox:
    feature_names = ["x1", "x2"]
    feature_types = ["float", "float"]
    bounds = [[0.0, 1.0], [0.0, 1.0]]

    def evaluate(self, point):
        return float(...)
```

也可以是一个函数：

```python
def evaluate(point):
    return float(...)
```

如果 `task-json` 中包含 `parameters`，SAIBO 会优先使用其中的变量名和边界；否则会从 evaluator 对象读取 `feature_names`、`feature_types` 和 `bounds`。

---

## 在线 LLM 调用

默认 smoke 和示例检查都可以离线运行。如需调用 Intern S1，设置环境变量并加上 `--online`：

```bash
export API_KEY=your_key_here
export INTERN_S1_API_KEY=your_key_here

saibo dry --method lgbo --online --task-json examples/task_profile.example.json
saibo wet --method labo --online --data-json examples/wet_input.example.json
```

不要提交真实 API key。`.env.example` 中列出了支持的环境变量。

---

## 后续路线图

以下模块属于 SAIBO 更完整的研究方向，但不包含在当前第一阶段公开发布内容中。

| 模块 | 状态 | 方向 |
|---|---|---|
| TagBO | 计划中 | 面向化学反应和材料发现的 tag-aware / semantic-feature BO。 |
| Latent-space BO | 计划中 | 在任务相关隐空间中统一连续、离散和分类变量。 |
| Multi-objective BO | 计划中 | 接入 qEHVI、scalarization BO 和代理模型辅助搜索。 |
| Multi-scale agents | 计划中 | 组织 micro、meso、macro Agent 的跨尺度证据协同。 |
| Agent-ready knowledge base | 计划中 | 为科学 Agent 提供文献、多模态数据、记忆和检索服务。 |
| Decision-model post-training | 计划中 | 训练小型控制器用于门控、风险控制和实验升级决策。 |

这些内容目前作为研究路线展示，尚未作为稳定代码发布。

---

## 框架视角

SAIBO 连接 scientific reasoning 与 black-box optimization：

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

SAIBO 的目标不是让 LLM 直接替代优化器，而是让科学推理以可使用、可审计、可迭代的方式进入优化循环。

---

## 发布说明

SAIBO 目前处于早期研究发布阶段。

第一阶段公开版本优先整理 **LABO** 与 **LGBO** 两个基座方法。框架中提到的其他模块仍在开发和验证中，后续接口、实现和实验设置都可能继续调整。

当前仓库用于组织 SAIBO 项目的公开主页、代码、示例和文档。

---

## 许可

SAIBO 采用 MIT License 发布，详见 [LICENSE](LICENSE)。

---

## 引用

如果你在研究中使用 SAIBO，请根据使用的基座方法引用对应论文。

LABO：

```bibtex
@article{chen2026labo,
  title={LABO: LLM-Accelerated Bayesian Optimization through Broad Exploration and Selective Experimentation},
  author={Chen, Zhuo and Yuan, Xinzhe and Zhang, Jianshu and Dong, Jinzong and Zhou, Ruichen and Niu, Yingchun and Zhou, Tianhang and Liu, Yu Yang Fredrik and Li, Yuqiang and Ye, Nanyang and others},
  journal={arXiv preprint arXiv:2605.22054},
  year={2026}
}
```

LGBO：

```bibtex
@inproceedings{yuanunleashing,
  title={Unleashing LLMs in Bayesian Optimization: Preference-Guided Framework for Scientific Discovery},
  author={Yuan, Xinzhe and Chen, Zhuo and Zhang, Jianshu and Xiong, Huan and Ye, Nanyang and Li, Yuqiang and Gu, Qinying},
  booktitle={The Fourteenth International Conference on Learning Representations}
}
```
