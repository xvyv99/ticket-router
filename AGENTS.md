# AGENTS.md — Ticket Router

> This document is for AI coding agents. If you are new to this project, read this first for project background, technology stack, code organization, and development conventions.

---

## 1. Project Overview

This project builds three AI system design paradigms for the same task — multilingual customer support ticket routing and preliminary reply — and critically compares them across five dimensions: **fairness, accountability, transparency, explainability, and robustness**.

The three paradigms:

1. **Rule-Based System**: explicit keyword rules, regex, and if-then-else decision trees mapping requests to queue / priority / tags, with fixed templates for preliminary answers.
2. **Supervised System**: supervised ML models trained on historical data. Includes traditional baselines (TF-IDF + Logistic Regression / XGBoost) and a pretrained language model (RemBERT multi-task fine-tuning).
3. **Goal-Based / Agentic AI System**: uses LLMs (Qwen3 series, DeepSeek-V3, etc.) with a high-level system prompt defining roles and goals, directly generating queue / priority / tags / answer as structured JSON. Supports both local vLLM inference and SiliconFlow batch API.

The core deliverable is not just runnable code but a responsible-AI analysis report comparing how the three paradigms mediate user intent and system output.

---

## 2. Technology Stack

| Layer | Tools / Libraries |
|---|---|
| Language & Runtime | **Python 3.13** (locked in `.python-version`, `requires-python = ">=3.13, <3.14"`) |
| Package Manager | **uv** (monorepo, independent `pyproject.toml` + `uv.lock` per sub-package) |
| Build Backend | `uv_build>=0.11.7` |
| Data Processing | `pandas>=3.0.2`, `numpy`, `pandera[pandas]` (schema validation) |
| Visualization | `matplotlib`, `seaborn` |
| Traditional ML | `scikit-learn>=1.8.0` (TF-IDF, Logistic Regression), `xgboost>=3.2.0` |
| Deep Learning | `torch>=2.10.0`, `transformers[torch]>=5.5.4`, `datasets>=4.8.4` |
| Quantization | `llmcompressor>=0.10.0.1` (W8A8 AWQ for local Qwen3) |
| Local LLM Inference | `vllm>=0.19.0` (structured output, GPU inference) |
| Fairness Analysis | `aif360>=0.6.1`, `fairlearn>=0.13.0` |
| LLM API | SiliconFlow batch API (DeepSeek-V3/R1, QwQ-32B, etc.) |
| Evaluation | `scikit-learn` metrics (Accuracy, Macro-F1, Confusion Matrix) |
| Type Checking | `pyright` (config in `pyrightconfig.json`) |
| Task Runner | `just` (config in `justfile`) |

**Note**: the root `pyproject.toml` has empty `dependencies`. Each sub-package under `packages/*/pyproject.toml` declares its own dependencies. Use `uv add --project packages/<name> <package>` to add new dependencies.

---

## 3. Project Structure

```
.
├── pyproject.toml              # Root workspace placeholder (requires-python >=3.13, <3.14)
├── uv.lock                     # Root uv lockfile
├── .python-version             # 3.13
├── justfile                    # Common task definitions
├── AGENTS.md                   # This file
├── CLAUDE.md                   # Claude-specific context
│
├── docs/
│   ├── REQUIREMENTS.md         # Course project requirements
│   ├── Project_Description_Responsible_ALgorithm2026Spring.pdf
│   └── superpowers/
│       ├── specs/2026-04-15-customer-service-design.md   # System design doc
│       └── plans/2026-04-15-customer-service.md          # Detailed implementation plan
│
├── dataset/
│   └── multilingual-customer-support-tickets/
│       ├── dataset-tickets-multi-lang3-4k.csv            # Core 4k multilingual dataset (5 languages)
│       ├── dataset-tickets-multi-lang-4-20k.csv          # 20k English-German augmentation
│       ├── aa_dataset-tickets-multi-lang-5-2-50-version.csv  # 28k English-German data
│       ├── dataset-tickets-german_normalized.csv         # Small German-normalized data
│       ├── dataset-tickets-german_normalized_50_5_2.csv  # Large German-normalized data
│       └── README.md                                     # Dataset field descriptions
│
├── packages/                   # Monorepo: independent sub-packages
│   ├── ticket_router.base/     # Shared foundation: types, config, data loading, eval protocols
│   │   ├── pyproject.toml
│   │   └── src/ticket_router.base/
│   │       ├── config.py       # Paths, constants, dataset locations
│   │       ├── types.py        # Queue/Priority/Language enums, Record/Prediction dataclasses, Pandera schemas
│   │       ├── predictor.py    # Predictor & Trainer Protocol definitions
│   │       ├── utils.py        # JSONLLogger, write_pred, combine_texts, to_records
│   │       ├── data/
│   │       │   ├── loader.py   # Load 4k / test_set / train_set
│   │       │   └── utils.py    # build_train_test_set, build_difficult_cases
│   │       └── eval/
│   │           └── metrics.py  # Classification metrics, consistency metrics (Jaccard)
│   │
│   ├── ticket_router.supervised/   # Supervised learning system
│   │   ├── pyproject.toml
│   │   ├── scripts/
│   │   │   ├── 01_build_test_set.py              # Unified test set (1200) + difficult cases (100)
│   │   │   ├── 03_run_supervised_traditional.py  # Train LR + XGBoost, output predictions
│   │   │   └── 04_run_mbert.py                   # RemBERT train/inference (smoke/train/infer modes)
│   │   └── src/ticket_router.supervised/
│   │       ├── features.py     # TF-IDF + AdaptiveSVD + StandardScaler pipeline
│   │       ├── utils.py        # SKModel wrapper, save_model, create_datasets
│   │       └── models/
│   │           ├── lr.py       # LogisticRegression (OvR for tags)
│   │           ├── xgb.py      # XGBoostClassifier
│   │           └── mbert.py    # RemBERT fine-tuning (queue + priority heads)
│   │
│   ├── ticket_router_agent/    # Goal-Based / LLM system
│   │   ├── pyproject.toml
│   │   ├── scripts/
│   │   │   ├── quantize_qwen.py    # Qwen3 W8A8 quantization (calibration -> AWQ)
│   │   │   ├── gen_batch.py        # Generate SiliconFlow batch API request JSONL
│   │   │   └── run_batch.py        # Local vLLM inference (supports quantized models)
│   │   └── src/ticket_router_agent/
│   │       ├── config.py       # Model selection, MAX_TOKEN_LENGTH, SAVE_DIR
│   │       ├── types.py        # TicketOutput Pydantic model, JSON schema
│   │       ├── prompt.py       # System prompt + few-shot examples builder
│   │       ├── infer.py        # vLLMPredictor (structured output via Pydantic)
│   │       └── utils.py        # Model path/name utilities
│   │
│   └── ticket_router_rule/     # Rule-Based system (currently empty, to be implemented)
│
├── scripts/
│   └── eda.py                  # Exploratory data analysis
│
├── models/                     # Local quantized model cache (.gitignored)
│   ├── qwen3-0.6B-awq/
│   ├── qwen3-1.7B-awq/
│   └── qwen3-4B-awq/
│
└── outputs/                    # Run outputs (.gitignored)
    ├── test_set.jsonl          # 1200-sample unified test set
    ├── train_set.jsonl         # ~2800-sample training set
    ├── difficult_cases.jsonl   # 100-sample difficult case set
    ├── supervised/
    │   ├── lr_predictions.jsonl
    │   ├── xgb_predictions.jsonl
    │   └── mbert_predictions.jsonl
    └── goal_based/
        ├── batch_file/         # SiliconFlow batch request files
        ├── batch_result/       # SiliconFlow batch results
        └── *_predictions.jsonl # Local vLLM predictions
```

**Current phase**: mid-stage.

- Foundation (`ticket_router.base`) complete: unified type system, data loaders, train/test split, evaluation metrics, JSONL logging.
- Data partitioning complete: `test_set.jsonl` (1200), `train_set.jsonl`, `difficult_cases.jsonl` (100) generated.
- Supervised system v1 complete: LR/XGBoost implemented with predictions; RemBERT training/inference done.
- Goal-Based system v1 complete: local vLLM inference (Qwen3 quantized) and SiliconFlow batch API request generation done.
- Rule-Based system not yet implemented.
- Unified evaluation report script and LLM-as-Judge not yet done.
- `tests/` directory not yet created.

---

## 4. Commands

### 4.1 Environment Sync

```bash
# Sync root virtualenv (essentially empty)
uv sync

# Sync sub-package virtualenvs (run from their directories)
cd packages/ticket_router.base && uv sync
cd packages/ticket_router.supervised && uv sync
cd packages/ticket_router_agent && uv sync

# Add a new dependency to a specific sub-package
uv add --project packages/ticket_router.supervised <package-name>
```

### 4.2 Run via just (recommended)

```bash
# Build test set
just prepare-data

# Train traditional ML (LR + XGBoost) and output predictions
just run-ml

# RemBERT (smoke test / full training / inference)
just run-mbert --smoke      # 200-sample smoke test
just run-mbert --train      # Full training
just run-mbert --infer      # Inference only

# Qwen3 quantization
just quan-qwen --quantize 0.6B
just quan-qwen --quantize all

# Local vLLM inference
just run-vllm Qwen/Qwen3-0.6B --sample-num 1200
just run-vllm models/qwen3-0.6B-awq --sample-num 1200

# Generate SiliconFlow batch requests
just gen-batch
```

### 4.3 Run via uv directly

```bash
# EDA
python scripts/eda.py

# Build test set
uv run --project packages/ticket_router.supervised packages/ticket_router.supervised/scripts/01_build_test_set.py

# Traditional ML
uv run --project packages/ticket_router.supervised packages/ticket_router.supervised/scripts/03_run_supervised_traditional.py

# RemBERT
uv run --project packages/ticket_router.supervised packages/ticket_router.supervised/scripts/04_run_mbert.py

# vLLM inference
uv run --project packages/ticket_router_agent packages/ticket_router_agent/scripts/run_batch.py models/qwen3-0.6B-awq --sample-num 1200
```

### 4.4 Type Checking

```bash
pyright
```

---

## 5. Code Organization & Module Responsibilities

The project uses a **monorepo** architecture. Sub-packages share infrastructure through `ticket_router.base` while remaining isolated.

### 5.1 Shared Layer (`ticket_router.base`)

| Module | Responsibility |
|---|---|
| `types.py` | Core domain types. `Queue` (10), `Priority` (3), `Language` (5) are `StrEnum`. `Record` / `Prediction` / `GroundRecord` are frozen dataclasses. Includes Pandera schemas for DataFrame validation. |
| `config.py` | Dataset paths, output paths, random seed, sample size constants. Uses `pathlib.Path` with existence assertions at import time. |
| `predictor.py` | `Predictor` and `Trainer` Protocol definitions — all downstream models must implement these. |
| `data/loader.py` | Load 4k raw data, test_set, train_set. Returns Pandera-validated DataFrames. |
| `data/utils.py` | `build_train_test_set` uses `StratifiedShuffleSplit` stratified by queue+priority+language. `build_difficult_cases` uses heuristic rules (small queue + high priority + long body). |
| `eval/metrics.py` | `compute_classification_metrics` (Accuracy, Macro-F1, Weighted-F1, per-class Recall, Confusion Matrix). `compute_consistency` (queue agreement, tag Jaccard, answer similarity). |
| `utils.py` | `JSONLLogger` context manager, `write_pred` unified prediction save, `combine_texts` for subject+body concatenation. |

### 5.2 Supervised Layer (`ticket_router.supervised`)

| Module | Responsibility |
|---|---|
| `features.py` | `build_tfidf_pipeline` returns `TfidfVectorizer(ngram_range=(1,2)) -> AdaptiveSVD -> StandardScaler`. `AdaptiveSVD` auto-clips `n_components` to ≤ actual feature count. |
| `utils.py` | `SKModel` wrapper unifies sklearn/xgboost model `predict()` interface, supports `LabelEncoder` and `MultiLabelBinarizer` decoding. `create_datasets` converts Records to HuggingFace `Dataset`. |
| `models/lr.py` | `LRTrainer` / `LRPredictor`. Uses `class_weight="balanced"`, trains queue, priority, tags (multi-label OvR). Tags model is trained but not yet enabled in `predict()`. |
| `models/xgb.py` | `XGBTrainer` / `XGBPredictor`. `multi:softprob` objective, supports probability output. |
| `models/mbert.py` | `MBERTTrainer` / `MBERTPredictor`. Based on `google/rembert`, trains separate queue and priority classification heads. Uses HuggingFace `Trainer` API with fp16 and early stopping. |

### 5.3 Goal-Based Layer (`ticket_router_agent`)

| Module | Responsibility |
|---|---|
| `prompt.py` | `build_system_prompt` constructs system prompt with queue/priority enums and few-shot examples (covering all 10 queues). `build_prompt` combines system + user message. |
| `types.py` | `TicketOutput` Pydantic model with `queue`, `priority`, `tags`, `preliminary_answer` fields and constraints. `TICKET_SCHEMA` for vLLM structured output. |
| `infer.py` | `vLLMPredictor` using `vllm.LLM` with `StructuredOutputsParams(json=TICKET_SCHEMA)` for forced JSON output. Falls back to default queue/priority on parse failure with `ErrorFlag.JSON_ERR`. |
| `scripts/quantize_qwen.py` | W8A8 AWQ quantization for Qwen3 via `llmcompressor`. Stratified calibration data sampling from training set. |
| `scripts/gen_batch.py` | Generates SiliconFlow batch API request JSONL for DeepSeek-V3/R1, QwQ-32B, DeepSeek-V3.1-Terminus. |
| `scripts/run_batch.py` | Local vLLM inference entry point, supports both raw HuggingFace and quantized models. |

### 5.4 Rule-Based Layer (`ticket_router_rule`)

- **Currently empty, to be implemented.** Refer to the old prototype in `packages/ticket_router/rule_based/` (keyword matching + template selection).

---

## 6. Development Conventions

### 6.1 Python Style Guide

This section is the **definitive style reference** for all Python code in this project. All modules under `packages/ticket_router_*/` and `scripts/` must follow these rules.

#### 6.1.1 Import Ordering

Strict three-group ordering, one blank line between groups:

```python
# 1. Standard library
from typing import List, Dict, Tuple, Any, Protocol
from pathlib import Path
from logging import getLogger
import json

# 2. Third-party
import pandas as pd
from sklearn.linear_model import LogisticRegression
from vllm import LLM

# 3. Internal — absolute for cross-package, relative for same-package
from ticket_router.base.types import Record, PredictionBatch
from ticket_router.base.config import SEED

from .types import TicketOutput
from .prompt import build_prompt
```

- Never use `from typing import *` or wildcard third-party imports.
- Prefer relative imports within the same package, absolute imports across packages.

#### 6.1.2 Type Annotations

- **All function parameters and return values must be annotated.**
- Use Python 3.10+ union syntax: `str | None`, `List[Record] | RecordDF`.
- Never `Optional[str]` or `Union[str, None]` — always `str | None`.
- Import from `typing`: `List`, `Dict`, `Tuple`, `Any`, `Protocol`, `Annotated`.
- Pandera DataFrame generics: `DataFrame[ITCustomerSupportSchema]`.
- pyright ignore comments: `# pyright: ignore[reportAttributeAccessIssue]`.

```python
# Correct
def predict(self, records: List[Record] | RecordDF) -> PredictionBatch: ...
def load_data(path: Path | None = None) -> DataFrame[Schema]: ...

# Wrong
def predict(self, records): ...
from typing import Optional
def load_data(path: Optional[Path] = None) -> pd.DataFrame: ...
```

#### 6.1.3 Naming Conventions

| Category | Convention | Examples |
|---|---|---|
| Classes | PascalCase | `LRPredictor`, `JSONLLogger`, `ErrorFlag` |
| Functions / methods | snake_case | `compute_metrics`, `build_prompt` |
| Constants / config | UPPER_SNAKE_CASE | `SEED`, `OUTPUT_DIR`, `QUEUE2ID` |
| Module-private functions | `_` prefix | `_jaccard`, `_load_rules` |
| Class-private attributes | `_` prefix | `_model_queue`, `_queue_path` |
| Module files | snake_case | `predictor.py`, `data_loader.py` |

#### 6.1.4 Classes & Data Structures

```python
# Immutable dataclass
from dataclasses import dataclass

@dataclass(frozen=True)
class Prediction(GroundRecord):
    request_id: str
    queue_confidence: float | None
    priority_confidence: float | None
    raw_output: str | None
    error: ErrorFlag

# String enum
from enum import StrEnum, IntFlag, auto

class Queue(StrEnum):
    TECHNICAL_SUPPORT = "Technical Support"
    # ...

class ErrorFlag(IntFlag):
    SUCCESS = 0
    JSON_ERR = auto()

# Protocol interface
from typing import Protocol

class Predictor(Protocol):
    supports_tags: bool
    supports_preliminary_answer: bool

    def predict(self, records: List[Record] | RecordDF) -> PredictionBatch:
        raise NotImplementedError

# Class attribute declarations (type-annotate private attrs)
class LRPredictor(Predictor):
    supports_tags = False
    supports_preliminary_answer = False
    _model_queue: SKModel
    _model_priority: SKModel
```

#### 6.1.5 Comments & Docstrings

- **Module-level docstrings**: brief, describe the module's purpose.
  ```python
  """Logistic Regression training for queue, priority, and tags."""
  ```
- **Code comments in Chinese**, but **punctuation in English half-width** (`, .` not `，` `。`).
- Explain **why**, not what. Avoid redundant comments.
- TODO markers: `# TODO: specific description`.
- Function-level docstrings only when the logic is non-trivial; omit for simple functions.

#### 6.1.6 Error Handling

```python
# assert for preconditions
assert DATASET_4K_PATH.exists(), f"4k dataset not found at {DATASET_4K_PATH}"
assert len(valid) > test_num, (
    f"Not enough valid samples ({len(valid)}) for test_num={test_num}."
)

# raise for business logic errors
if val_records is None:
    raise ValueError("MBERTTrainer requires val_records for early stopping")

# try-except only for external operations (IO, network, JSON parse)
try:
    data = json.loads(raw)
except json.JSONDecodeError:
    logger.warning("JSON parse failed")
    # fallback
```

**Never**:
- Bare `except:`
- `except Exception:`
- try-except as a substitute for if-checks in business logic

#### 6.1.7 Formatting

- 4-space indentation, no tabs.
- No hard 79-char limit, but keep lines reasonable (<= 100 recommended).
- Module-level constants grouped at the top of the file.
- Two blank lines between top-level functions and between classes; one blank line between class methods.
- Two blank lines between classes and top-level functions.

#### 6.1.8 Ruff (Mandatory)

**After modifying or creating any Python file, you must run:**

```bash
ruff format <modified_files>
ruff check <modified_files>
```

Fix all ruff errors before considering work complete. If a ruff rule conflicts with existing project style, follow the existing code style and configure a ruff rule override in `pyproject.toml`.

### 6.2 Monorepo Architecture Rules

Each sub-package follows a consistent module layout:

| Module | Responsibility |
|---|---|
| `types.py` | All enums, dataclasses, Pandera schemas, type aliases |
| `predictor.py` | Predictor / Trainer Protocol definitions |
| `config.py` | Path constants, dataset locations, global config, log format |
| `utils.py` | Utility functions, JSONLLogger, data conversion helpers |
| `data/loader.py` | Data loading, returns Pandera-validated DataFrames |
| `data/utils.py` | Data splitting, stratified sampling, difficult case selection |
| `eval/metrics.py` | Evaluation metric computation |

New package/module checklist:
- [ ] Types in `types.py`
- [ ] Protocols in `predictor.py` (if needed)
- [ ] Config in `config.py`
- [ ] Utilities in `utils.py`
- [ ] Implements `Predictor` Protocol (if applicable)

### 6.3 Config & Constants

```python
from pathlib import Path

PROJECT_ROOT = Path.cwd()
DATA_DIR = PROJECT_ROOT / "dataset" / "multilingual-customer-support-tickets"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Assert dataset existence at import time
assert DATASET_4K_PATH.exists(), f"4k dataset not found at {DATASET_4K_PATH}"

# Logging setup
from logging import getLogger, basicConfig

LOGGING_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logger = getLogger(__name__)

if __name__ == "__main__":
    basicConfig(level="INFO", format=LOGGING_FORMAT)
    main()

# Constants
SEED = 42
TEST_SAMPLE_NUM = 1200
DIFFICULT_CASE_NUM = 100
MBERT_INFER_BATCH_SIZE = 64
```

### 6.4 Interface Contract

- All models must implement the `Predictor` Protocol: `predict(records) -> PredictionBatch`.
- Trainers may optionally implement the `Trainer` Protocol: `train(records, val_records) -> Predictor`.
- `Prediction` dataclass contains: `request_id`, `queue`, `priority`, `tag_1`, `tag_2`, `answer`, `queue_confidence`, `priority_confidence`, `raw_output`, `error`.
- `ErrorFlag` uses `IntFlag`, supporting combined flags. `SUCCESS=0`, `JSON_ERR`, plus reserved bits for regex parse errors.

### 6.5 Data Isolation (Critical)

These are **hard constraints** — violations directly invalidate experimental results:

- The **unified test set must come only** from `dataset-tickets-multi-lang3-4k.csv`.
- `4k` and `20k` / `28k` have **no overlap** — they can be isolated directly.
- `20k` and `28k` share **8,306 duplicates** (~40% overlap). If merging, must deduplicate by `subject` + `body`.
- The **German-normalized datasets** use a completely different queue taxonomy — **do not include them in classification accuracy statistics**. They may only be used for qualitative robustness observations.

### 6.6 Output Convention

- All batch run results must go to `outputs/` as JSONL.
- Always use `ticket_router.base.utils.write_pred` for saving predictions — format is `PredSave` (with `request_id`, `language`, `predicted`, `ground_truth`).
- When running Goal-Based inference multiple times on the same request, record output variance (current vLLM scripts run independently; consistency analysis requires post-processing).
- For LLM outputs that fail JSON parsing: vLLM structured output has significantly reduced failures; if still failing, mark as `ErrorFlag.JSON_ERR`.

### 6.7 Model Paths & Caching

- Be aware of HuggingFace model download cache paths when using `transformers`. Avoid loading too many large models simultaneously if disk space is limited.
- Quantized models are saved to `./models/qwen3-{size}-awq/` (`.gitignored`).
- RemBERT fine-tuning checkpoints are saved to `outputs/supervised/models/mbert/` (`.gitignored`).

### 6.8 Git Rules for Agents

- **Do not autonomously execute `git commit`, `git push`, `git rebase`, or any other git write operations** unless the user **explicitly** requests it.
- Read-only commands (`git status`, `git diff`, `git log`) are always fine.
- **Do not modify** the old package `packages/ticket_router/` — it is an early prototype superseded by the monorepo.

### 6.9 Language Policy

- Project documentation (AGENTS.md, CLAUDE.md, README.md): **English**.
- Code comments: **Chinese**, with **English half-width punctuation**.
- Git commit messages: **Conventional Commits** style (`feat:`, `docs:`, `refactor:`, `test:`, `fix:`).

---

## 7. Testing Strategy

### 7.1 Unit Tests (to be established)

- The `tests/` directory has **not yet been created**.
- Planned priority testing targets in `ticket_router.base`:
  - `data/utils.py`: strata distribution in stratified splits, heuristic filtering logic for difficult cases.
  - `utils.py`: `JSONLLogger` write/read, `combine_texts` edge cases.
  - `eval/metrics.py`: correctness of each metric under extreme conditions (all-correct, all-wrong, single-class).

### 7.2 System-Level Tests

- **Unified test set**: 1,200 samples, stratified by language + queue + priority. Shared across all three systems. **Already generated.**
- **Difficult case set**: 100 samples, covering small queues, high priority, long bodies. **Already generated.**

### 7.3 Per-System Test Coverage (current state)

| System | Coverage | Notes | Status |
|---|---|---|---|
| Rule-Based | 1,200 samples | All languages | To do |
| Supervised (LR/XGB) | 1,200 samples | All languages | Done |
| Supervised (RemBERT) | 1,200 samples | All languages | Done |
| Goal-Based (Qwen3 0.6B) | 1,200 samples | Local vLLM, few-shot | Done |
| Goal-Based (DeepSeek et al.) | 1,200 samples | SiliconFlow batch API | Pending |

### 7.4 Evaluation Methods (to be completed)

- **Objective metrics**: Accuracy, Macro-F1, Weighted-F1, per-queue Recall, confusion matrix, inter-language Macro-F1 variance.
- **Consistency metrics**: queue agreement rate across multiple Goal-Based runs, tag Jaccard similarity.
- **LLM-as-Judge**: independent scoring of outputs using a fixed strong model. Script not yet implemented.
- **Fairness analysis**: systematic bias analysis for small queues (`Human Resources`, `General Inquiry`) and minority languages (ES, FR, PT) using `aif360` / `fairlearn`. Script not yet implemented.

---

## 8. Security & Responsible AI

### 8.1 Data Security

- CSV files in `dataset/` contain simulated customer support data. Although they do not contain real PII, avoid uploading raw data in full to public repositories. (`dataset/` and `outputs/` are currently `.gitignored`.)
- If real API keys (SiliconFlow / OpenAI) are used later, store them in `.env` files and ensure `.env` is `.gitignored`.

### 8.2 Responsible AI Analysis Framework

The final project report must center on five pillars. Any code change affecting these dimensions should be documented in code or commit messages:

1. **Transparency & Explainability**: Rule-Based decision paths must be fully traceable; Supervised must provide feature importance or SHAP; Goal-Based must log prompts and raw outputs.
2. **Fairness & Bias**: pay special attention to systematic performance gaps for small queues (`Human Resources`, `General Inquiry`) and minority languages (ES, FR, PT).
3. **Robustness & Reliability**: difficult-case accuracy, spelling error tolerance, multi-run consistency for Goal-Based.
4. **Accountability**: attribution analysis when errors occur (rule designer / data annotator / prompt engineer / model provider).
5. **Alignment with User Intent**: LLM-as-Judge scoring + qualitative failure case analysis.

### 8.3 AI Tool Disclosure

Per course requirements, **whenever you use an AI tool in written documentation, you must accurately disclose which AI tool was used and for what purpose**.

---

## 9. Quick Start for Agents

1. **Read the design doc**: `docs/superpowers/specs/2026-04-15-customer-service-design.md` for system architecture.
2. **Read the implementation plan**: `docs/superpowers/plans/2026-04-15-customer-service.md` for step-by-step instructions.
3. **Sync the environment**: `cd` into the relevant sub-package directory and run `uv sync`.
4. **Run EDA**: `python scripts/eda.py` to familiarize with dataset characteristics.
5. **Build test set**: `just prepare-data` (if not already generated).
6. **Check existing outputs**: `outputs/supervised/` and `outputs/goal_based/` contain baseline predictions.
7. **Start working**:
   - For Rule-Based: create modules under `packages/ticket_router_rule/`.
   - For evaluation/reporting: extend under root `scripts/` or `packages/ticket_router.base/eval/`.
   - For improvements to Supervised/Agent: modify within the respective package, maintaining Protocol compatibility.

---

## 10. References

- Course requirements: `docs/REQUIREMENTS.md`
- System design doc: `docs/superpowers/specs/2026-04-15-customer-service-design.md`
- Implementation plan: `docs/superpowers/plans/2026-04-15-customer-service.md`
- Dataset README: `dataset/multilingual-customer-support-tickets/README.md`

## 11. Recurring Code Patterns

### 11.1 Adding a New Predictor

```python
from ticket_router.base.predictor import Predictor
from ticket_router.base.types import Prediction, PredictionBatch, Queue, Priority, ErrorFlag

class MyPredictor(Predictor):
    supports_tags: bool = False
    supports_preliminary_answer: bool = False

    def predict(self, records: List[Record] | RecordDF) -> PredictionBatch:
        # Must return PredictionBatch(predictions=[...], parse_err_count=0, parse_json_err_count=0)
        ...
```

### 11.2 match-case Enum Dispatch

```python
def task2labels(task: Task) -> Dict[int, str]:
    match task:
        case Task.QUEUE:
            return ID2QUEUE
        case Task.PRIORITY:
            return ID2PRIORITY
        case _:
            raise ValueError(f"Unsupported task: {task}")
```

### 11.3 JSONL Logging

```python
from dataclasses import asdict
from ticket_router.base.utils import JSONLLogger

with JSONLLogger(save_path) as logger:
    for pred in predictions:
        logger.write(asdict(pred))
```

### 11.4 Stratified String Concatenation

```python
df["_strat"] = (
    df["queue"].astype(str)
    + "|"
    + df["priority"].astype(str)
    + "|"
    + df["language"].astype(str)
)
```

### 11.5 Batch Prediction Save

```python
from ticket_router.base.utils import write_pred

write_pred(batch.predictions, df_test, OUTPUT_DIR / "predictions.jsonl")
```
