---
name: ticket-router-style
description: >
  Enforce the Python coding style of the Ticket Router monorepo project.
  Use when writing, editing, or reviewing any Python code under
  packages/ticket_router_*/ or scripts/ in this project.
  Covers import ordering, type annotations, naming conventions,
  error-handling patterns, monorepo structure, and mandatory ruff checks.
---

# Ticket Router Python Style Guide

> 本 skill 约束所有在 Ticket Router 项目中编写的 Python 代码, 确保与现有 33+ 个模块的风格完全一致.

---

## 1. 导入组织 (Import Ordering)

严格三层分组, 组间空一行:

```python
# 1. 标准库
from typing import List, Dict, Tuple, Any, Protocol
from pathlib import Path
from logging import getLogger
import json

# 2. 第三方库
import pandas as pd
from sklearn.linear_model import LogisticRegression
from vllm import LLM

# 3. 项目内部导入
# 跨包: 绝对导入
from ticket_router_base.types import Record, PredictionBatch
from ticket_router_base.config import SEED

# 同包: 相对导入
from .types import TicketOutput
from .prompt import build_prompt
```

- 禁止使用 `from typing import *`
- 禁止使用通配符导入第三方库
- 同包内优先相对导入, 跨包必须绝对导入

---

## 2. 类型注解 (Type Annotations)

- 所有函数参数和返回值必须加类型注解
- 使用 Python 3.10+ 联合语法: `str | None`, `List[Record] | RecordDF`
- 禁止 `Optional[str]` / `Union[str, None]`, 统一用 `str | None`
- 从 `typing` 导入: `List`, `Dict`, `Tuple`, `Any`, `Protocol`, `Annotated`
- Pandera DataFrame 泛型: `DataFrame[ITCusomterSupportSchema]`
- pyright 忽略注释格式: `# pyright: ignore[reportAttributeAccessIssue]`

---

## 3. 命名规范 (Naming Conventions)

| 类别 | 规范 | 示例 |
|------|------|------|
| 类名 | PascalCase | `LRPredictor`, `JSONLLogger`, `ErrorFlag` |
| 函数/方法 | snake_case | `compute_metrics`, `build_prompt` |
| 常量/配置 | UPPER_SNAKE_CASE | `SEED`, `OUTPUT_DIR`, `QUEUE2ID` |
| 模块私有函数 | `_` 前缀 | `_jaccard`, `_load_rules` |
| 类私有属性 | `_` 前缀 | `_model_queue`, `_queue_path` |
| 模块文件 | snake_case | `predictor.py`, `data_loader.py` |

---

## 4. 类与数据结构设计

### 4.1 不可变数据类

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Prediction(GroundRecord):
    request_id: str
    queue_confidence: float | None
    priority_confidence: float | None
    raw_output: str | None
    error: ErrorFlag
```

### 4.2 枚举

```python
from enum import StrEnum, IntFlag, auto

class Queue(StrEnum):
    TECHNICAL_SUPPORT = "Technical Support"
    # ...

class ErrorFlag(IntFlag):
    SUCCESS = 0
    JSON_ERR = auto()
```

### 4.3 Protocol 接口

```python
from typing import Protocol

class Predictor(Protocol):
    supports_tags: bool
    supports_preliminary_answer: bool

    def predict(self, records: List[Record] | RecordDF) -> PredictionBatch:
        raise NotImplementedError
```

### 4.4 类属性声明

```python
class LRPredictor(Predictor):
    supports_tags = False          # 类属性
    supports_preliminary_answer = False

    _model_queue: SKModel          # 类型注解声明私有属性
    _model_priority: SKModel
```

---

## 5. 注释与文档 (Comments & Docstrings)

- **模块级 docstring**: 简短描述, 说明模块用途
  ```python
  """Logistic Regression training for queue, priority, and tags."""
  ```
- **注释使用中文**, 但**标点符号使用英文半角** (`, .` 而非 `，` `。`)
- **TODO 标记**: `# TODO: 具体待办描述`
- 注释说明 **why** 而非 **what**, 避免冗余注释
- 函数级 docstring 仅在逻辑复杂时使用, 简单函数可省略

---

## 6. 错误处理 (Error Handling)

### 6.1 assert 用于前置条件

```python
assert DATASET_4K_PATH.exists(), f"4k dataset not found at {DATASET_4K_PATH}"
assert len(valid) > test_num, (
    f"Not enough valid samples ({len(valid)}) for test_num={test_num}."
)
```

### 6.2 raise 用于业务逻辑错误

```python
if val_records is None:
    raise ValueError("MBERTTrainer requires val_records for early stopping")
```

### 6.3 try-except 仅用于外部操作

```python
try:
    out = parse_llm_output(raw)
except json.JSONDecodeError:
    json_err_count += 1
    # fallback logic
```

### 6.4 禁止事项

- 禁止裸 `except:`
- 禁止 `except Exception:`
- 禁止在业务逻辑中滥用 try-except 替代 if-check

---

## 7. 代码格式 (Formatting)

- **4 空格缩进**, 禁止 tab
- 无严格 79 字符限制, 但行不过长 (建议 <= 100)
- 模块级常量集中在文件顶部
- 函数间空 **两行**, 类方法间空 **一行**
- 类与顶层函数间空 **两行**
- 使用 ruff 作为格式化工具 (见第 11 节)

---

## 8. Monorepo 架构规范

项目采用 `packages/` monorepo, 各包通过 `ticket_router_base` 共享基础设施.

### 8.1 包内模块职责

| 模块 | 职责 |
|------|------|
| `types.py` | 集中定义所有 Enum, dataclass, Pandera Schema, 类型别名 |
| `predictor.py` | 定义 Predictor / Trainer Protocol |
| `config.py` | 路径常量, 数据集位置, 全局配置, 日志格式 |
| `utils.py` | 工具函数, JSONLLogger, 数据转换辅助函数 |
| `data/loader.py` | 数据加载, 返回经 Pandera 校验的 DataFrame |
| `data/utils.py` | 数据划分, 分层抽样, 困难案例筛选 |
| `eval/metrics.py` | 评估指标计算 |

### 8.2 新包/新模块 checklist

- [ ] 类型定义放在 `types.py`
- [ ] Protocol 放在 `predictor.py` (如需要)
- [ ] 配置放在 `config.py`
- [ ] 工具函数放在 `utils.py`
- [ ] 实现 Predictor Protocol (如需要)

---

## 9. 配置与常量管理

### 9.1 路径管理

```python
from pathlib import Path

PROJECT_ROOT = Path.cwd()
DATA_DIR = PROJECT_ROOT / "dataset" / "multilingual-customer-support-tickets"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
```

### 9.2 数据集存在性断言

```python
assert DATASET_4K_PATH.exists(), f"4k dataset not found at {DATASET_4K_PATH}"
```

### 9.3 日志配置

```python
from logging import getLogger, basicConfig

LOGGING_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# 模块顶部
logger = getLogger(__name__)

# main 函数底部
if __name__ == "__main__":
    basicConfig(level="INFO", format=LOGGING_FORMAT)
    main()
```

### 9.4 常量

```python
SEED = 42
TEST_SAMPLE_NUM = 1200
DIFFICULT_CASE_NUM = 100
MBERT_INFER_BATCH_SIZE = 64
```

---

## 10. 正反面示例

### 10.1 导入组织 (正确 vs 错误)

```python
# 正确
from typing import List, Dict
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score

from ticket_router_base.types import Record
from ticket_router_base.config import SEED

# 错误: 导入未分组, 混合相对/绝对
import pandas as pd
from .types import Record
from ticket_router_base.config import SEED
from typing import List
import json
```

### 10.2 类型注解 (正确 vs 错误)

```python
# 正确
def predict(self, records: List[Record] | RecordDF) -> PredictionBatch:
    ...

def load_data(path: Path | None = None) -> DataFrame[Schema]:
    ...

# 错误
def predict(self, records):
    ...

from typing import Optional
def load_data(path: Optional[Path] = None) -> pd.DataFrame:
    ...
```

### 10.3 错误处理 (正确 vs 错误)

```python
# 正确
if val_records is None:
    raise ValueError("MBERTTrainer requires val_records")

try:
    data = json.loads(raw)
except json.JSONDecodeError:
    logger.warning("JSON parse failed")

# 错误
try:
    if val_records is None:
        raise Exception("bad")
except:
    pass
```

---

## 11. 强制代码检查 (ruff)

**每次修改或新增 Python 文件后, 必须执行:**

```bash
# 1. 格式化
ruff format <modified_files>

# 2. 静态检查
ruff check <modified_files>
```

- 修复所有 ruff 报错后再视为完成
- 若 ruff 规则与项目现有风格冲突, 以现有代码风格为准, 在 pyproject.toml 中配置 ruff 规则覆盖
- 当前项目尚未配置 ruff, 首次使用时需添加 `ruff` 到开发依赖

---

## 12. 特殊代码模式速查

### 12.1 match-case 枚举分发

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

### 12.2 JSONL 日志写入

```python
from dataclasses import asdict
from ticket_router_base.utils import JSONLLogger

with JSONLLogger(save_path) as logger:
    for pred in predictions:
        logger.write(asdict(pred))
```

### 12.3 分层字符串拼接

```python
df["_strat"] = (
    df["queue"].astype(str)
    + "|"
    + df["priority"].astype(str)
    + "|"
    + df["language"].astype(str)
)
```

### 12.4 预测批量保存

```python
from ticket_router_base.utils import write_pred

write_pred(batch.predictions, df_test, OUTPUT_DIR / "predictions.jsonl")
```
