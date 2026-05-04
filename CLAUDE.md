# CLAUDE.md — Ticket Router

> Cheat sheet for Claude. Project context, architecture decisions, current status, coding conventions, and quick-reference tables to get up to speed in a single conversation.

---

## 1. What This Project Is

A multilingual customer support ticket routing system. Three AI design paradigms are being built and compared across five dimensions: **fairness, accountability, transparency, explainability, and robustness**.

Core task: given a customer email (subject + body), output **queue** (10 classes), **priority** (3 classes), **tags**, and a **preliminary answer**.

---

## 2. Architecture

### 2.1 Monorepo + Protocol-Driven

The project uses a **monorepo** with sub-packages under `packages/`, each having its own `pyproject.toml` + `uv.lock`:

| Package | Responsibility | Key Dependencies |
|---|---|---|
| `ticket_router.base` | Shared types, data loading, evaluation metrics, Predictor/Trainer Protocol | pandas, pandera, scikit-learn, aif360, fairlearn |
| `ticket_router.supervised` | LR, XGBoost, RemBERT fine-tuning | torch, transformers, xgboost, datasets, joblib |
| `ticket_router_agent` | Local vLLM + SiliconFlow batch API | vllm, pydantic, llmcompressor |
| `ticket_router_rule` | **Currently empty**, to be implemented | — |

**Key Protocol** (`ticket_router.base.predictor`):
- `Predictor.predict(records) -> PredictionBatch` — unified entry point for all models.
- `Trainer.train(records, val_records) -> Predictor` — optional trainer interface.

**Key Types** (`ticket_router.base.types`):
- `Queue` / `Priority` / `Language`: `StrEnum`.
- `Record`: input, with `request_id`, `subject`, `body`, `language` + ground truth fields.
- `Prediction`: output, with `queue`, `priority`, `tag_1`, `tag_2`, `answer`, `queue_confidence`, `priority_confidence`, `raw_output`, `error`.
- `ErrorFlag`: `IntFlag`, `SUCCESS=0`, supports combined error flags.

Unified save: `ticket_router.base.utils.write_pred(predictions, records, path)` outputs standard JSONL.

### 2.2 Data Flow

```
dataset-tickets-multi-lang3-4k.csv
    -> 01_build_test_set.py
        -> outputs/test_set.jsonl          (1200 samples, stratified)
        -> outputs/train_set.jsonl         (~2800 samples)
        -> outputs/difficult_cases.jsonl   (100 samples)

train_set.jsonl -> LR/XGB/RemBERT training  -> outputs/supervised/*_predictions.jsonl
test_set.jsonl  -> vLLM/Qwen3 inference     -> outputs/goal_based/*_predictions.jsonl
```

**Hard constraint**: the test set must come **only** from the 4k dataset. The 20k and 28k datasets are for training augmentation only, but they share 8,306 duplicates — must deduplicate by `subject` + `body` if merging.

---

## 3. Current Status

### Done
- [x] `ticket_router.base`: type system, Pandera schemas, data loading, stratified splitting, difficult-case selection, JSONL logging, classification/consistency metrics.
- [x] `ticket_router.supervised`: LR + XGBoost implemented and run on full test set; RemBERT training/inference implemented.
- [x] `ticket_router_agent`: local vLLM inference (Qwen3 0.6B/1.7B/4B, AWQ quantized) + SiliconFlow batch API request generation.
- [x] Qwen3 W8A8 quantization scripts (`llmcompressor`).
- [x] Unified test/train/difficult-case sets generated.

### To Do
- [ ] **Rule-Based system** (`packages/ticket_router_rule/`): completely empty. Refer to old prototype in `packages/ticket_router/rule_based/` (keyword matching + template selection).
- [ ] **Tags prediction**: LR tag model trained but not enabled in `LRPredictor.predict()`; XGBoost and RemBERT not trained for tags.
- [ ] **Preliminary answer**: supervised systems currently only predict queue + priority.
- [ ] **Unified evaluation script**: reads JSONL outputs from each system, computes comparison metrics, generates reports. Planned for root `scripts/` or `packages/ticket_router.base/eval/`.
- [ ] **LLM-as-Judge**: score answer quality using a strong model.
- [ ] **Fairness analysis**: analyze language/queue-level bias with `aif360` / `fairlearn`.
- [ ] **Test directory**: `tests/` not yet created, no pytest coverage.
- [ ] **Multi-run consistency**: variance analysis for Goal-Based on the same request repeated 3 times.

---

## 4. Python Style Quick Reference

### 4.1 Import Order (3 groups, blank line between)

```python
# 1. Standard library
from typing import List, Dict, Tuple, Any, Protocol
from pathlib import Path
from logging import getLogger

# 2. Third-party
import pandas as pd
from sklearn.metrics import accuracy_score

# 3. Internal — absolute for cross-package, relative for same-package
from ticket_router.base.types import Record, PredictionBatch
from .prompt import build_prompt
```

Never use `from typing import *` or wildcard third-party imports.

### 4.2 Type Annotations

- All function parameters and return values must be annotated.
- Use Python 3.10+ union syntax: `str | None`, `List[Record] | RecordDF`.
- Never use `Optional[str]` / `Union[str, None]` — always `str | None`.
- Import from `typing`: `List`, `Dict`, `Tuple`, `Any`, `Protocol`, `Annotated`.
- Pandera DataFrame generics: `DataFrame[Schema]`.
- pyright ignore format: `# pyright: ignore[reportAttributeAccessIssue]`.

### 4.3 Naming

| Category | Convention | Examples |
|---|---|---|
| Classes | PascalCase | `LRPredictor`, `JSONLLogger`, `ErrorFlag` |
| Functions/methods | snake_case | `compute_metrics`, `build_prompt` |
| Constants/config | UPPER_SNAKE_CASE | `SEED`, `OUTPUT_DIR`, `QUEUE2ID` |
| Module-private functions | `_` prefix | `_jaccard`, `_load_rules` |
| Class-private attrs | `_` prefix | `_model_queue`, `_queue_path` |
| Module files | snake_case | `predictor.py`, `data_loader.py` |

### 4.4 Classes & Data Structures

```python
# Immutable dataclass
from dataclasses import dataclass

@dataclass(frozen=True)
class Prediction(GroundRecord):
    request_id: str
    queue_confidence: float | None
    error: ErrorFlag

# Enum
from enum import StrEnum, IntFlag, auto

class Queue(StrEnum):
    TECHNICAL_SUPPORT = "Technical Support"

class ErrorFlag(IntFlag):
    SUCCESS = 0
    JSON_ERR = auto()

# Protocol
class Predictor(Protocol):
    supports_tags: bool
    supports_preliminary_answer: bool

    def predict(self, records: List[Record] | RecordDF) -> PredictionBatch: ...

# Class attribute declarations
class LRPredictor(Predictor):
    supports_tags = False
    supports_preliminary_answer = False
    _model_queue: SKModel
    _model_priority: SKModel
```

### 4.5 Comments & Docstrings

- Module-level docstrings: brief, describe the module's purpose.
  ```python
  """Logistic Regression training for queue, priority, and tags."""
  ```
- Code comments in **Chinese**, but punctuation in **English half-width** (`, .` not `，` `。`).
- Explain **why**, not what.
- TODO markers: `# TODO: specific description`.
- Function-level docstrings only when the logic is non-trivial.

### 4.6 Error Handling

```python
# assert for preconditions
assert DATASET_4K_PATH.exists(), f"4k dataset not found at {DATASET_4K_PATH}"

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

**Never**: bare `except:`, `except Exception:`, or try-except as a substitute for if-checks in business logic.

### 4.7 Formatting

- 4-space indentation, no tabs.
- No hard 79-char limit, but keep lines reasonable (<= 100 recommended).
- Module-level constants grouped at the top of the file.
- Two blank lines between top-level functions/classes, one blank line between class methods.
- **After modifying any Python file, run**:
  ```bash
  ruff format <modified_files>
  ruff check <modified_files>
  ```
  Fix all ruff errors before considering work complete.

---

## 5. Claude Work Rules

1. **No autonomous git writes**: never run `git commit`, `git push`, `git rebase`, `git reset`, `git merge`, etc. unless the user **explicitly** requests it. Read-only git commands (`status`, `diff`, `log`) are fine.

2. **Do not modify `packages/ticket_router/`**: the old package is an early prototype superseded by the monorepo structure. The user has explicitly excluded it.

3. **Maintain Protocol compatibility**: every new Predictor must implement `predict()` returning exactly `PredictionBatch`. This is the prerequisite for unified evaluation across all three systems.

4. **Data isolation red line**: evaluation and training scripts may only use `train_set.jsonl` for training. Never use `test_set.jsonl` or `difficult_cases.jsonl` for training, hyperparameter tuning, or few-shot examples.

5. **Output format**: batch results must be saved as JSONL under `outputs/` using `write_pred()` for unified formatting.

6. **When implementing evaluation/reporting**: read existing JSONL outputs (`outputs/supervised/*.jsonl`, `outputs/goal_based/*.jsonl`). Do not re-run model inference.

---

## 6. Key File Reference

| Purpose | Path |
|---|---|
| Core types & enums | `packages/ticket_router.base/src/ticket_router.base/types.py` |
| Dataset path config | `packages/ticket_router.base/src/ticket_router.base/config.py` |
| Predictor Protocol | `packages/ticket_router.base/src/ticket_router.base/predictor.py` |
| Data loader | `packages/ticket_router.base/src/ticket_router.base/data/loader.py` |
| Stratified split / difficult cases | `packages/ticket_router.base/src/ticket_router.base/data/utils.py` |
| Eval metrics | `packages/ticket_router.base/src/ticket_router.base/eval/metrics.py` |
| Unified prediction save | `packages/ticket_router.base/src/ticket_router.base/utils.py` |
| LR/XGBoost models | `packages/ticket_router.supervised/src/ticket_router.supervised/models/` |
| RemBERT fine-tuning | `packages/ticket_router.supervised/src/ticket_router.supervised/models/mbert.py` |
| vLLM inference | `packages/ticket_router_agent/src/ticket_router_agent/infer.py` |
| Prompt builder | `packages/ticket_router_agent/src/ticket_router_agent/prompt.py` |
| Just tasks | `justfile` |
| Type check config | `pyrightconfig.json` |
