# AGENTS.md — Ticket Router

> 本文档面向 AI coding agent. 如果你刚接触到这个项目, 请先阅读本文件以了解项目背景、技术栈、代码组织方式以及开发规范.

---

## 1. 项目概述 (Project Overview)

该项目目标是在多语言客服工单数据集上, 针对同一个用户任务(客服工单路由与初步回复)构建三种不同设计范式的 AI 系统, 并从 **公平性 (fairness)、问责性 (accountability)、透明性 (transparency)、可解释性 (explainability)、稳健性 (robustness)** 五个维度进行批判性比较分析.

三种系统范式如下:

1. **Rule-Based System**: 基于显式关键词规则、正则和 if-then-else 决策树, 将用户请求映射到 queue / priority / tags, 并使用固定模板生成 preliminary answer.
2. **Supervised System**: 基于历史数据训练监督学习模型. 包含传统 ML 基线 (TF-IDF + Logistic Regression / XGBoost) 和预训练语言模型 (rembert / XLM-RoBERTa 多任务微调).
3. **Goal-Based / Agentic AI System**: 使用大语言模型(LLM, 如 Qwen3 系列、GPT-4o), 通过高层 system prompt 定义角色与目标, 直接生成 queue / priority / tags / answer 的 JSON 输出. 包含 Base、+RAG、+Tool 等配置消融.

项目的核心交付物不仅是可运行的代码, 更是一份负责任的 AI 分析报告, 比较三种设计范式如何中介(mediate)用户意图与系统输出.

---

## 2. 技术栈 (Technology Stack)

| 层级         | 工具/库                                                               |
| ------------ | --------------------------------------------------------------------- |
| 语言与运行时 | **Python 3.14** (`.python-version` 中锁定)                            |
| 包管理器     | **uv** (`pyproject.toml` + `uv.lock`)                                 |
| 构建后端     | `uv_build>=0.11.6`                                                    |
| 数据处理     | `pandas`, `numpy`                                                     |
| 可视化       | `matplotlib`, `seaborn`                                               |
| 传统 ML      | `scikit-learn` (TF-IDF, Logistic Regression, SVM), `xgboost`          |
| 深度学习     | `transformers` (HuggingFace), `torch`, 可选 `peft` / `unsloth` (LoRA) |
| 向量检索     | `faiss-cpu` 或 `chromadb`                                             |
| LLM API      | SiliconFlow API (Qwen3 系列), 可选 OpenAI API                         |
| 本地推理     | `ollama` 或 `vllm` (视需求)                                           |
| 异步请求     | `asyncio`, `aiohttp`                                                  |
| 评估指标     | `scikit-learn` metrics, 可选 `sacrebleu`, `rouge-score`               |
| 测试框架     | `pytest`                                                              |

**注意**: 当前 `pyproject.toml` 的 `dependencies` 为空, 但 `.venv` 中已安装部分依赖(如 pandas、numpy、seaborn). 添加新依赖时请使用 `uv add <package>` 并确保 `uv.lock` 被更新.

---

## 3. 项目结构 (Project Structure)

```
.
├── pyproject.toml              # uv 项目配置, requires-python >=3.14
├── uv.lock                     # uv 锁文件
├── .python-version             # 3.14
├── README.md                   # 项目标题 (目前仅占位)
├── AGENTS.md                   # 本文件
│
├── docs/
│   ├── REQUIREMENTS.md         # 课程项目要求(中文)
│   ├── Project_Description_Responsible_ALgorithm2026Spring.pdf
│   └── superpowers/
│       ├── specs/2026-04-15-customer-service-design.md   # 系统设计文档
│       └── plans/2026-04-15-customer-service.md          # 详细实施计划(含TDD步骤)
│
├── dataset/
│   └── multilingual-customer-support-tickets/
│       ├── dataset-tickets-multi-lang3-4k.csv            # 核心 4k 多语言数据集(5种语言)
│       ├── dataset-tickets-multi-lang-4-20k.csv          # 20k 英德增强数据
│       ├── aa_dataset-tickets-multi-lang-5-2-50-version.csv  # 28k 英德数据
│       ├── dataset-tickets-german_normalized.csv         # 德语标准化小数据
│       ├── dataset-tickets-german_normalized_50_5_2.csv  # 德语标准化大数据
│       └── README.md                                     # 数据集字段说明
│
├── src/ticket_router/          # 主 Python 包(目前仅 __init__.py, 大量模块待实现)
│   ├── __init__.py
│   ├── config.py               # (计划中) 常量、路径、queue/priority 定义
│   ├── data/
│   │   ├── loader.py           # (计划中) CSV 加载、去重、训练测试划分
│   │   └── templates.py        # (计划中) 模板池构建与选择
│   ├── logging_utils.py        # (计划中) JSONL 结构化日志 + token 费用估算
│   ├── test_set.py             # (计划中) 统一测试集与困难案例集构建
│   ├── rule_based/             # (计划中) 基于规则的系统
│   ├── supervised/             # (计划中) 监督学习系统
│   ├── goal_based/             # (计划中) 目标导向/Agentic AI 系统
│   └── evaluation/             # (计划中) 评估指标、LLM-as-judge、报表
│
├── scripts/
│   ├── eda.py                  # 已存在:  exploratory data analysis 脚本
│   ├── 01_build_test_set.py    # (计划中)
│   ├── 02_run_rule_based.py    # (计划中)
│   ├── 03_train_supervised.py  # (计划中)
│   ├── 04_run_goal_based.py    # (计划中)
│   └── 05_evaluate_all.py      # (计划中)
│
├── tests/                      # (计划中) pytest 测试目录
└── outputs/                    # 运行输出目录, 已被 .gitignore 排除
```

**当前状态**: 项目处于早期阶段. `scripts/eda.py` 已可运行并生成 EDA 报告与可视化. 核心源码模块(`src/ticket_router/config.py` 及以下大部分子模块)和测试目录尚未创建, 但有一份非常详细的实施计划可参考(`docs/superpowers/plans/2026-04-15-customer-service.md`).

---

## 4. 常用命令 (Build & Test Commands)

### 4.1 环境同步

```bash
# 同步虚拟环境(安装 uv.lock 中锁定的依赖)
uv sync

# 添加新依赖
uv add <package-name>
```

### 4.2 运行脚本

```bash
# 运行 EDA 脚本(生成 outputs/eda/ 下的报告与图表)
python scripts/eda.py

# 后续脚本(待实现)
python scripts/01_build_test_set.py
python scripts/02_run_rule_based.py
python scripts/03_train_supervised.py
python scripts/04_run_goal_based.py
python scripts/05_evaluate_all.py
```

### 4.3 测试

```bash
# 运行全部测试
pytest tests/ -v

# 运行单个测试文件
pytest tests/test_config.py -v
```

### 4.4 代码检查(可选)

项目中尚未配置 `ruff` 或 `mypy`, 但鼓励使用.

---

## 5. 代码组织与模块划分

项目的架构遵循"共享基础设施 + 三种范式隔离"的原则:

### 5.1 共享层 (`src/ticket_router/` 根目录及 `data/`)

- **`config.py`**: 定义所有常量(10 个 queue, 3 个 priority, 5 种语言), 以及数据集路径、输出路径.
- **`data/loader.py`**: 加载 CSV, 处理 `20k`/`28k` 数据集与 `4k` 之间的去重, 防止数据泄露.
- **`data/templates.py`**: 从 `4k` 数据集的 `answer` 字段提炼模板池, 供 Rule-Based 和 Supervised 系统共用.
- **`logging_utils.py`**: 提供 `JSONLLogger` 类, 所有系统输出均以 JSONL 格式记录, 保证可复现性.
- **`test_set.py`**: 使用 `StratifiedShuffleSplit` 从 `4k` 数据中分层抽样 1,200 条统一测试集, 并构建 100 条困难案例集.

### 5.2 Rule-Based 层 (`rule_based/`)

- 输入解析器 (`parser.py`): 按语言提取关键词.
- 规则引擎 (`engine.py`): 基于 JSON 规则文件匹配 queue / priority / tags.
- 回复生成器 (`responder.py`): 根据 (queue, priority, tags) 选择模板并填充占位符.
- 规则文件 (`rules/en.json`, `de.json`, `es.json`, `fr.json`, `pt.json`): 多语言关键词规则.

### 5.3 Supervised 层 (`supervised/`)

- 特征工程 (`features.py`): TF-IDF 管道.
- 传统 ML (`train_traditional.py`): LR、XGBoost 训练与保存(`joblib`).
- 预训练模型 (`train_mbert.py`): rembert / XLM-RoBERTa 的多任务微调.
- 可选生成实验 (`train_t5.py`): `mt0-base` 的 seq2seq 微调(分析语言不平等议题).
- 统一推理入口 (`inference.py`).

### 5.4 Goal-Based 层 (`goal_based/`)

- API 客户端 (`client.py`): SiliconFlow / OpenAI 异步客户端, 带速率限制和重试.
- 提示词 (`prompt.py`): Base / +RAG / +Tool 的 system prompt.
- RAG 检索器 (`rag.py`): 基于 FAISS 的向量检索.
- 工具定义 (`tool.py`): `classify_keywords()` 模拟 agentic 行为.
- 批量运行器 (`runner.py`): 控制 temperature、多次运行、JSON 解析失败处理.

### 5.5 评估层 (`evaluation/`)

- 客观指标 (`metrics.py`): Accuracy、Macro-F1、混淆矩阵、一致性指标.
- LLM-as-Judge (`llm_judge.py`): 使用 GPT-4o / Qwen3-32B 对输出进行独立评分.
- 报告生成 (`report.py`): 对比表格与可视化.

---

## 6. 开发规范与约定 (Development Conventions)

### 6.1 语言与文档

- **项目文档以中文为主**: 课程要求、设计文档、实施计划均为中文.
- **代码注释**: 使用中文, 标点符号使用英文 (例如 `// 这是注释, 不要用中文标点`). 保持简洁, 说明 *why* 而非 *what*.
- **Git commit message**: 使用 conventional commits 风格, 例如 `feat:`, `docs:`, `refactor:`, `test:`, `fix:`.
- **异常处理**: 不要滥用 try-except, 仅在预期可能失败的外部操作 (文件读写、网络请求等) 使用, 并保留原始异常信息以便 debug. 业务逻辑错误应使用显式检查 + 明确报错, 而非捕获所有异常.

### 6.2 测试驱动开发 (TDD)

实施计划强烈建议采用 TDD 流程:

1. 先写失败的测试 (`pytest` 应 FAIL).
2. 编写最小实现使测试通过.
3. 运行脚本生成产物并验证.
4. `git add` + `git commit`.

### 6.3 数据隔离与去重 (Critical)

这是项目中的**硬性约束**, 违反将直接影响实验有效性:

- **统一测试集必须且只能**来源于 `dataset-tickets-multi-lang3-4k.csv`.
- `4k` 与 `20k` / `28k` **无重叠**, 可直接隔离使用.
- `20k` 与 `28k` 之间存在 **8,306 条重复样本** (约 40% 重叠). 若合并使用, 必须通过 `subject` + `body` 去重.
- **德语标准化数据集**的 queue 分类体系与主数据集完全不同, **不得纳入分类准确率统计**, 仅可用于定性鲁棒性观察.

### 6.4 输出规范

- 所有批量运行结果必须输出到 `outputs/` 下的 JSONL 文件.
- Goal-Based 系统同一请求至少运行 3 次, 记录输出方差.
- 对 JSON 解析失败的 LLM 输出, 先用正则提取, 仍失败则标记为 `"format_error"`.

### 6.5 模型路径与缓存

- 使用 HuggingFace `transformers` 时, 注意模型下载缓存路径. 若环境磁盘空间有限, 避免同时加载过多大模型.
- SiliconFlow API 的免费模型(Qwen3-0.6B/4B/9B)有速率限制(RPM 1,000, TPM 80,000), 批量请求必须实现异步并发控制.

---

## 7. 测试策略 (Testing Strategy)

### 7.1 单元测试

- 所有 `src/ticket_router/` 下的模块应对应有 `tests/test_*.py`.
- 优先测试数据加载器(去重逻辑)、模板选择器、规则引擎、JSONL 日志器等基础设施.

### 7.2 系统级测试

- **统一测试集**: 1,200 条, 按 language + queue + priority 分层抽样, 三种系统共用同一份测试集.
- **困难案例集**: 100 条, 覆盖歧义请求、拼写错误、小 queue、高情绪强度等边缘场景.

### 7.3 各系统测试量

| 系统                    | 测试量          | 说明                         |
| ----------------------- | --------------- | ---------------------------- |
| Rule-Based              | 1,200 条        | 全语言                       |
| Supervised (传统 ML)    | 1,200 条        | 英语为主, 部分模型可测全语言 |
| Supervised (mBERT)      | 1,200 条        | 全语言                       |
| Goal-Based (0.6B/4B/9B) | 1,200 条 × 3 次 | 免费模型跑满, 控制非确定性   |
| Goal-Based (14B)        | 400 条 × 3 次   | 分层抽样                     |
| Goal-Based (32B)        | 200 条 × 3 次   | 困难案例优先                 |
| Goal-Based (100B+)      | 100 条 × 3 次   | 困难案例 + 随机抽样          |

### 7.4 评估方式

- **客观指标**: Accuracy, Macro-F1, Weighted-F1, 各 queue Recall, 混淆矩阵, 语言间 Macro-F1 方差.
- **一致性指标**: Goal-Based 同一请求 3 次运行的 queue 一致率、tag Jaccard 相似度.
- **LLM-as-Judge**: 使用固定强模型(GPT-4o 或 Qwen3-32B)评分, 维度包括 queue_correctness、priority_correctness、tag_relevance、answer_helpfulness、answer_faithfulness.
- **人工复核**: 对 LLM-as-Judge 争议案例进行人工抽样复核, 并在最终报告中披露局限性.

---

## 8. 安全与负责任 AI 考量 (Security & Responsible AI)

### 8.1 数据安全

- `dataset/` 目录中的 CSV 文件包含模拟客服数据, 虽不涉真实 PII, 但仍建议避免将原始数据完整上传到公共仓库. (当前 `.gitignore` 已排除 `dataset/` 和 `outputs/`).
- 若后续使用真实 API key(SiliconFlow / OpenAI), 必须存储在 `.env` 文件中, 并确保 `.env` 已被 `.gitignore` 排除.

### 8.2 负责任 AI 分析框架

项目最终报告必须围绕以下五个支柱展开. 任何代码改动若影响这些维度, 应在文档或 commit message 中说明:

1. **Transparency & Explainability**: Rule-Based 的规则路径必须完全可追溯; Supervised 需提供特征重要性或 SHAP; Goal-Based 需记录 prompt 与原始输出.
2. **Fairness & Bias**: 特别关注小 queue (`Human Resources`, `General Inquiry`) 和小语种 (ES, FR, PT) 的系统性表现差距.
3. **Robustness & Reliability**: 困难案例集准确率、拼写错误容忍度、Goal-Based 多次运行一致性.
4. **Accountability**: 出错时责任归属分析(规则设计者 / 数据标注者 / prompt 工程师 / 模型提供商).
5. **Alignment with User Intent**: LLM-as-Judge 打分 + 典型失败案例分析.

### 8.3 AI 工具披露

根据课程要求, **每当你在书面文档中使用 AI 工具时, 必须准确披露使用了哪个 AI 工具以及用于什么目的**.

---

## 9. 快速上手 (Quick Start for Agents)

1. **阅读设计文档**: 先看 `docs/superpowers/specs/2026-04-15-customer-service-design.md` 了解系统架构.
2. **查看实施计划**: 再看 `docs/superpowers/plans/2026-04-15-customer-service.md` 获取逐步实施步骤(含测试代码片段).
3. **同步环境**: 运行 `uv sync`.
4. **运行 EDA**: `python scripts/eda.py` 以熟悉数据集特征.
5. **开始实现**: 按照实施计划的 Task 1, 2, 3... 逐步推进, 遵循"先写测试 -> 再写实现 -> 运行通过 -> commit"的流程.

---

## 10. 参考链接

- 课程要求: `docs/REQUIREMENTS.md`
- 系统设计文档: `docs/superpowers/specs/2026-04-15-customer-service-design.md`
- 详细实施计划: `docs/superpowers/plans/2026-04-15-customer-service.md`
- 数据集说明: `dataset/multilingual-customer-support-tickets/README.md`
