<!-- AGENTS.md -->

# AGENTS.md — Ticket Router

> 本文档面向 AI coding agent. 如果你刚接触到这个项目, 请先阅读本文件以了解项目背景、技术栈、代码组织方式以及开发规范.

---

## 1. 项目概述 (Project Overview)

该项目目标是在多语言客服工单数据集上, 针对同一个用户任务(客服工单路由与初步回复)构建三种不同设计范式的 AI 系统, 并从 **公平性 (fairness)、问责性 (accountability)、透明性 (transparency)、可解释性 (explainability)、稳健性 (robustness)** 五个维度进行批判性比较分析.

三种系统范式如下:

1. **Rule-Based System**: 基于显式关键词规则、正则和 if-then-else 决策树, 将用户请求映射到 queue / priority / tags, 并使用固定模板生成 preliminary answer.
2. **Supervised System**: 基于历史数据训练监督学习模型. 包含传统 ML 基线 (TF-IDF + Logistic Regression / XGBoost) 和预训练语言模型 (RemBERT 多任务微调).
3. **Goal-Based / Agentic AI System**: 使用大语言模型(LLM, 如 Qwen3 系列、DeepSeek-V3 等), 通过高层 system prompt 定义角色与目标, 直接生成 queue / priority / tags / answer 的 JSON 输出. 包含本地 vLLM 推理和 SiliconFlow batch API 两种运行模式.

项目的核心交付物不仅是可运行的代码, 更是一份负责任的 AI 分析报告, 比较三种设计范式如何中介(mediate)用户意图与系统输出.

---

## 2. 技术栈 (Technology Stack)

| 层级          | 工具/库                                                                       |
| ------------- | ----------------------------------------------------------------------------- |
| 语言与运行时  | **Python 3.13** (`.python-version` 锁定, `requires-python = ">=3.13, <3.14"`) |
| 包管理器      | **uv** (monorepo, 各子包独立 `pyproject.toml` + `uv.lock`)                    |
| 构建后端      | `uv_build>=0.11.7`                                                            |
| 数据处理      | `pandas>=3.0.2`, `numpy`, `pandera[pandas]` (schema 校验)                     |
| 可视化        | `matplotlib`, `seaborn`                                                       |
| 传统 ML       | `scikit-learn>=1.8.0` (TF-IDF, Logistic Regression), `xgboost>=3.2.0`         |
| 深度学习      | `torch>=2.10.0`, `transformers[torch]>=5.5.4`, `datasets>=4.8.4`              |
| 量化压缩      | `llmcompressor>=0.10.0.1` (W8A8 AWQ, 用于本地 Qwen3 量化)                     |
| 本地 LLM 推理 | `vllm>=0.19.0` (结构化输出, GPU 推理)                                         |
| 公平性分析    | `aif360>=0.6.1`, `fairlearn>=0.13.0`                                          |
| LLM API       | SiliconFlow batch API (DeepSeek-V3/R1, QwQ-32B 等)                            |
| 评估指标      | `scikit-learn` metrics (Accuracy, Macro-F1, Confusion Matrix)                 |
| 类型检查      | `pyright` (配置见 `pyrightconfig.json`)                                       |
| 任务运行      | `just` (配置见 `justfile`)                                                    |

**注意**: 根目录 `pyproject.toml` 的 `dependencies` 为空, 各子包在 `packages/*/pyproject.toml` 中声明自己的依赖. 添加新依赖时请在对应子包目录下使用 `uv add <package>`.

---

## 3. 项目结构 (Project Structure)

```
.
├── pyproject.toml              # 根项目配置 (workspace 占位, requires-python >=3.13, <3.14)
├── uv.lock                     # 根 uv 锁文件
├── .python-version             # 3.13
├── justfile                    # 常用任务定义
├── pyrightconfig.json          # 类型检查配置
├── README.md                   # 项目标题
├── AGENTS.md                   # 本文件
├── CLAUDE.md                   # Claude 专用上下文
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
├── packages/                   # monorepo: 各子包独立管理依赖
│   ├── ticket_router.base/     # 共享基础层: 类型、配置、数据加载、评估协议
│   │   ├── pyproject.toml
│   │   └── src/ticket_router.base/
│   │       ├── config.py       # 路径、常量、数据集位置
│   │       ├── types.py        # Queue/Priority/Language Enum, Record/Prediction dataclass, Pandera schema
│   │       ├── predictor.py    # Predictor & Trainer Protocol 定义
│   │       ├── utils.py        # JSONLLogger, write_pred, combine_texts, to_records
│   │       ├── data/
│   │       │   ├── loader.py   # 加载 4k / test_set / train_set
│   │       │   └── utils.py    # build_train_test_set, build_difficult_cases
│   │       └── eval/
│   │           └── metrics.py  # 分类指标、一致性指标(Jaccard)
│   │
│   ├── ticket_router.supervised/   # 监督学习系统
│   │   ├── pyproject.toml
│   │   ├── scripts/
│   │   │   ├── 01_build_test_set.py          # 构建统一测试集(1200条)+困难案例集(100条)
│   │   │   ├── 03_run_supervised_traditional.py  # 训练 LR + XGBoost, 输出预测
│   │   │   └── 04_run_mbert.py               # RemBERT 训练/推理 (smoke/train/infer 模式)
│   │   └── src/ticket_router.supervised/
│   │       ├── features.py     # TF-IDF + AdaptiveSVD + StandardScaler 管道
│   │       ├── utils.py        # SKModel 包装器, save_model, create_datasets
│   │       └── models/
│   │           ├── lr.py       # LogisticRegression (OvR for tags)
│   │           ├── xgb.py      # XGBoostClassifier
│   │           └── mbert.py    # RemBERT 微调 (queue + priority 分离训练)
│   │
│   ├── ticket_router_agent/    # Goal-Based / LLM 系统
│   │   ├── pyproject.toml
│   │   ├── scripts/
│   │   │   ├── quantize_qwen.py    # Qwen3 W8A8 量化 (calibration -> AWQ)
│   │   │   ├── gen_batch.py        # 生成 SiliconFlow batch API 请求 JSONL
│   │   │   └── run_batch.py        # 本地 vLLM 推理 (支持量化模型)
│   │   └── src/ticket_router_agent/
│   │       ├── config.py       # 模型选择、MAX_TOKEN_LENGTH、SAVE_DIR
│   │       ├── types.py        # TicketOutput Pydantic model, JSON schema
│   │       ├── prompt.py       # system prompt + few-shot examples 构建
│   │       ├── infer.py        # vLLMPredictor (结构化输出 via Pydantic)
│   │       └── utils.py        # 模型路径/名称工具函数
│   │
│   └── ticket_router_rule/     # Rule-Based 系统 (目前为空, 待实现)
│
├── scripts/
│   └── eda.py                  # Exploratory data analysis
│
├── models/                     # 本地量化模型缓存 (已被 .gitignore 排除)
│   ├── qwen3-0.6B-awq/
│   ├── qwen3-1.7B-awq/
│   └── qwen3-4B-awq/
│
└── outputs/                    # 运行输出目录 (已被 .gitignore 排除)
    ├── test_set.jsonl          # 1200 条统一测试集
    ├── train_set.jsonl         # ~2800 条训练集
    ├── difficult_cases.jsonl   # 100 条困难案例集
    ├── supervised/
    │   ├── lr_predictions.jsonl
    │   ├── xgb_predictions.jsonl
    │   ├── mbert_predictions.jsonl
    │   └── models/             # 训练好的模型 (.joblib, RemBERT checkpoint)
    └── goal_based/
        ├── batch_file/         # SiliconFlow batch 请求文件
        ├── batch_result/       # SiliconFlow batch 结果
        └── *_predictions.jsonl # 本地 vLLM 预测结果
```

**当前状态**: 项目处于中期阶段.

- 基础架构(`ticket_router.base`)已完成: 统一类型系统、数据加载器、训练/测试划分、评估指标、JSONL 日志.
- 数据划分已完成: `test_set.jsonl`(1200条)、`train_set.jsonl`、`difficult_cases.jsonl`(100条)已生成.
- Supervised 系统初版已完成: LR、XGBoost 已实现并产出预测; RemBERT 训练/推理已实现.
- Goal-Based 系统初版已完成: 本地 vLLM 推理(Qwen3 量化模型)和 SiliconFlow batch API 请求生成已实现.
- Rule-Based 系统尚未实现.
- 统一的评估报告脚本和 LLM-as-Judge 尚未完成.
- 测试目录 `tests/` 尚未创建.

---

## 4. 常用命令 (Build & Test Commands)

### 4.1 环境同步

```bash
# 同步根虚拟环境(基本无依赖)
uv sync

# 同步子包虚拟环境(推荐在工作目录下执行)
cd packages/ticket_router.base && uv sync
cd packages/ticket_router.supervised && uv sync
cd packages/ticket_router_agent && uv sync

# 添加新依赖到指定子包
uv add --project packages/ticket_router.supervised <package-name>
```

### 4.2 运行脚本 (via just)

```bash
# 构建测试集
just prepare-data

# 训练传统 ML (LR + XGBoost) 并输出预测
just run-ml

# RemBERT 训练/推理 (smoke test / full training / inference)
just run-mbert --smoke      # 200条 smoke test
just run-mbert --train      # 完整训练
just run-mbert --infer      # 仅推理

# Qwen3 量化
just quan-qwen --quantize 0.6B
just quan-qwen --quantize all

# 本地 vLLM 推理
just run-vllm Qwen/Qwen3-0.6B --sample-num 1200
just run-vllm models/qwen3-0.6B-awq --sample-num 1200

# 生成 SiliconFlow batch 请求
just gen-batch
```

### 4.3 运行脚本 (直接调用)

```bash
# EDA
python scripts/eda.py

# 构建测试集
uv run --project packages/ticket_router.supervised packages/ticket_router.supervised/scripts/01_build_test_set.py

# 传统 ML
uv run --project packages/ticket_router.supervised packages/ticket_router.supervised/scripts/03_run_supervised_traditional.py

# RemBERT
uv run --project packages/ticket_router.supervised packages/ticket_router.supervised/scripts/04_run_mbert.py

# vLLM 推理
uv run --project packages/ticket_router_agent packages/ticket_router_agent/scripts/run_batch.py models/qwen3-0.6B-awq --sample-num 1200
```

### 4.4 类型检查

```bash
# pyright 会读取 pyrightconfig.json 中的配置
pyright
```

---

## 5. 代码组织与模块划分

项目采用 **monorepo** 架构, 各子包通过 `ticket_router.base` 共享基础设施, 保持三种范式之间的隔离.

### 5.1 共享层 (`ticket_router.base`)

- **`types.py`**: 核心领域类型. `Queue` (10个), `Priority` (3个), `Language` (5个) 均为 `StrEnum`. `Record` / `Prediction` / `GroundRecord` 为 frozen dataclass. 包含 Pandera schema 用于 DataFrame 运行时校验.
- **`config.py`**: 数据集路径、输出路径、随机种子、样本数量常量. 使用 `pathlib.Path` 并在导入时断言数据集存在.
- **`predictor.py`**: 定义 `Predictor` 和 `Trainer` Protocol, 所有下游模型必须实现此接口.
- **`data/loader.py`**: 加载 4k 原始数据、test_set、train_set. 返回经 Pandera 校验的 DataFrame.
- **`data/utils.py`**: `build_train_test_set` 使用 `StratifiedShuffleSplit` 按 queue+prioriy+language 分层抽样. `build_difficult_cases` 使用启发式规则(小 queue + 高 priority + 长 body)筛选困难案例.
- **`eval/metrics.py`**: `compute_classification_metrics` (Accuracy, Macro-F1, Weighted-F1, per-class Recall, Confusion Matrix). `compute_consistency` (queue agreement, tag Jaccard, answer similarity).
- **`utils.py`**: `JSONLLogger` 上下文管理器、`write_pred` 统一预测保存格式、`combine_texts` 拼接 subject+body.

### 5.2 Supervised 层 (`ticket_router.supervised`)

- **`features.py`**: `build_tfidf_pipeline` 返回 `TfidfVectorizer(ngram_range=(1,2)) -> AdaptiveSVD -> StandardScaler` 管道. `AdaptiveSVD` 会自动将 `n_components` 裁剪到不超过实际特征数.
- **`utils.py`**: `SKModel` 包装器统一 sklearn/xgboost 模型的 `predict()` 接口, 支持 `LabelEncoder` 和 `MultiLabelBinarizer` 反编码. `create_datasets` 将 Record 转换为 HuggingFace `Dataset`.
- **`models/lr.py`**: `LRTrainer` / `LRPredictor`. 使用 `class_weight="balanced"`, 标签训练 queue、priority、tags(多标签 OvR). 当前 tags 预测已训练但未在 `predict()` 中启用.
- **`models/xgb.py`**: `XGBTrainer` / `XGBPredictor`. `multi:softprob` 目标, 支持概率输出.
- **`models/mbert.py`**: `MBERTTrainer` / `MBERTPredictor`. 基于 `google/rembert`, 分别训练 queue 和 priority 两个分类头. 使用 HuggingFace `Trainer` API, 支持 fp16 和 early stopping.

### 5.3 Goal-Based 层 (`ticket_router_agent`)

- **`prompt.py`**: `build_system_prompt` 构建 system prompt, 包含 queue/priority 枚举和 few-shot 示例(覆盖全部 10 个 queue). `build_prompt` 组合 system + user message.
- **`types.py`**: `TicketOutput` Pydantic model, 定义 `queue`, `priority`, `tags`, `preliminary_answer` 字段和约束. `TICKET_SCHEMA` 用于 vLLM structured output.
- **`infer.py`**: `vLLMPredictor`. 基于 `vllm.LLM`, 使用 `StructuredOutputsParams(json=TICKET_SCHEMA)` 强制 JSON 输出. JSON 解析失败时回退到默认 queue/priority 并标记 `ErrorFlag.JSON_ERR`.
- **`scripts/quantize_qwen.py`**: 使用 `llmcompressor` 对 Qwen3 进行 W8A8 AWQ 量化. 从训练集中分层采样 calibration 数据.
- **`scripts/gen_batch.py`**: 生成 SiliconFlow batch API 请求 JSONL, 支持 DeepSeek-V3/R1, QwQ-32B, DeepSeek-V3.1-Terminus.
- **`scripts/run_batch.py`**: 本地 vLLM 推理入口, 支持原始 HuggingFace 模型和量化后模型.

### 5.4 Rule-Based 层 (`ticket_router_rule`)

- **当前为空, 待实现.** 可参考旧包 `packages/ticket_router/rule_based/` 中的原型实现(关键词匹配 + 模板选择).

---

## 6. 开发规范与约定 (Development Conventions)

### 6.1 语言与文档

- **项目文档以中文为主**: 课程要求、设计文档均为中文.
- **代码注释**: 使用中文, 标点符号使用英文 (例如 `# zhe shi zhu shi, buyao yong zhongwen biaodian`). 保持简洁, 说明 _why_ 而非 _what_.
- **Git commit message**: 使用 conventional commits 风格, 例如 `feat:`, `docs:`, `refactor:`, `test:`, `fix:`.
- **异常处理**: 不要滥用 try-except, 仅在预期可能失败的外部操作 (文件读写、网络请求等) 使用, 并保留原始异常信息以便 debug. 业务逻辑错误应使用显式检查 + 明确报错, 而非捕获所有异常.

### 6.2 Git 操作规范

- **不要自行执行 `git commit`/`git push`/`git rebase` 等任何 git 写操作**, 除非用户**显式说明**要求提交.
- 可以执行 `git status`、`git diff`、`git log` 等只读操作来帮用户查看状态, 但涉及工作区变更的写操作必须等待用户明确指令.

### 6.2 接口契约

- 所有模型必须实现 `Predictor` Protocol: `predict(records) -> PredictionBatch`.
- 训练器可选实现 `Trainer` Protocol: `train(records, val_records) -> Predictor`.
- `Prediction` dataclass 包含 `request_id`, `queue`, `priority`, `tag_1`, `tag_2`, `answer`, `queue_confidence`, `priority_confidence`, `raw_output`, `error`.
- `ErrorFlag` 使用 `IntFlag`, 支持组合标记. `SUCCESS=0`, `JSON_ERR`, 以及预留的正则解析错误位.

### 6.3 数据隔离与去重 (Critical)

这是项目中的**硬性约束**, 违反将直接影响实验有效性:

- **统一测试集必须且只能**来源于 `dataset-tickets-multi-lang3-4k.csv`.
- `4k` 与 `20k` / `28k` **无重叠**, 可直接隔离使用.
- `20k` 与 `28k` 之间存在 **8,306 条重复样本** (约 40% 重叠). 若合并使用, 必须通过 `subject` + `body` 去重.
- **德语标准化数据集**的 queue 分类体系与主数据集完全不同, **不得纳入分类准确率统计**, 仅可用于定性鲁棒性观察.

### 6.4 输出规范

- 所有批量运行结果必须输出到 `outputs/` 下的 JSONL 文件.
- 预测保存统一使用 `ticket_router.base.utils.write_pred`, 格式为 `PredSave` (含 `request_id`, `language`, `predicted`, `ground_truth`).
- Goal-Based 系统同一请求多次运行时, 需记录输出方差(当前 vLLM 脚本每次独立运行, 一致性分析需后处理).
- 对 JSON 解析失败的 LLM 输出, vLLM structured output 已大幅降低失败率; 若仍失败则标记为 `ErrorFlag.JSON_ERR`.

### 6.5 模型路径与缓存

- 使用 HuggingFace `transformers` 时, 注意模型下载缓存路径. 若环境磁盘空间有限, 避免同时加载过多大模型.
- 量化模型保存在 `./models/qwen3-{size}-awq/`, 已被 `.gitignore` 排除.
- RemBERT 微调 checkpoint 保存在 `outputs/supervised/models/mbert/`, 已被 `.gitignore` 排除.

---

## 7. 测试策略 (Testing Strategy)

### 7.1 单元测试 (待建立)

- 当前项目**尚未创建 `tests/` 目录**.
- 计划为 `ticket_router.base` 的以下模块优先补充测试:
  - `data/utils.py`: 分层抽样的 strata 分布、困难案例的 heuristic 筛选逻辑.
  - `utils.py`: `JSONLLogger` 写入/读取、`combine_texts` 边界条件.
  - `eval/metrics.py`: 各指标在极端情况(全对、全错、单类)下的正确性.

### 7.2 系统级测试

- **统一测试集**: 1,200 条, 按 language + queue + priority 分层抽样, 三种系统共用同一份测试集. **已生成.**
- **困难案例集**: 100 条, 覆盖小 queue、高 priority、长 body. **已生成.**

### 7.3 各系统测试量 (当前实际运行状态)

| 系统                    | 测试量   | 说明                  | 状态   |
| ----------------------- | -------- | --------------------- | ------ |
| Rule-Based              | 1,200 条 | 全语言                | 待实现 |
| Supervised (LR/XGB)     | 1,200 条 | 全语言                | 已完成 |
| Supervised (RemBERT)    | 1,200 条 | 全语言                | 已完成 |
| Goal-Based (Qwen3 0.6B) | 1,200 条 | 本地 vLLM, few-shot   | 已完成 |
| Goal-Based (DeepSeek等) | 1,200 条 | SiliconFlow batch API | 待提交 |

### 7.4 评估方式 (待完善)

- **客观指标**: Accuracy, Macro-F1, Weighted-F1, 各 queue Recall, 混淆矩阵, 语言间 Macro-F1 方差.
- **一致性指标**: Goal-Based 多次运行的 queue 一致率、tag Jaccard 相似度.
- **LLM-as-Judge**: 使用固定强模型对输出进行独立评分. 脚本尚未实现.
- **公平性分析**: 使用 `aif360` / `fairlearn` 分析小 queue 和小语种的系统性偏差. 脚本尚未实现.

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
2. **查看实施计划**: 再看 `docs/superpowers/plans/2026-04-15-customer-service.md` 获取逐步实施步骤.
3. **同步环境**: 进入需要工作的子包目录, 运行 `uv sync`.
4. **运行 EDA**: `python scripts/eda.py` 以熟悉数据集特征.
5. **构建测试集**: `just prepare-data` (若尚未生成).
6. **查看已有输出**: `outputs/supervised/` 和 `outputs/goal_based/` 中已有预测结果, 可作为基准.
7. **开始工作**:
   - 若实现 Rule-Based: 在 `packages/ticket_router_rule/` 下创建模块.
   - 若实现评估/报告: 在根目录 `scripts/` 或 `packages/ticket_router.base/eval/` 下扩展.
   - 若改进 Supervised/Agent: 在对应包下修改, 保持 Protocol 兼容.

---

## 10. 参考链接

- 课程要求: `docs/REQUIREMENTS.md`
- 系统设计文档: `docs/superpowers/specs/2026-04-15-customer-service-design.md`
- 详细实施计划: `docs/superpowers/plans/2026-04-15-customer-service.md`
- 数据集说明: `dataset/multilingual-customer-support-tickets/README.md`
