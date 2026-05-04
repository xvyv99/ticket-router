<!-- CLAUDE.md -->

# CLAUDE.md — Ticket Router 工作上下文

> 本文档为 Claude 专用. 提供项目核心上下文、当前状态、架构决策和编码速查, 帮助 Claude 在单次对话中快速进入状态.

---

## 1. 项目是什么

多语言客服工单路由系统, 需构建 **Rule-Based / Supervised / Goal-Based** 三种范式, 并从公平性、问责性、透明性、可解释性、稳健性五个维度进行比较分析.

核心任务: 输入一封客服邮件(subject + body), 输出 queue(10类) / priority(3类) / tags / preliminary_answer.

---

## 2. 架构核心决策

### 2.1 Monorepo + 协议驱动

项目使用 **monorepo**, 子包位于 `packages/`, 各包独立 `pyproject.toml` + `uv.lock`:

| 包名                       | 职责                                                     | 依赖重点                                         |
| -------------------------- | -------------------------------------------------------- | ------------------------------------------------ |
| `ticket_router_base`       | 共享类型、数据加载、评估指标、Predictor/Trainer Protocol | pandas, pandera, scikit-learn, aif360, fairlearn |
| `ticket_router.supervised` | LR, XGBoost, RemBERT 微调                                | torch, transformers, xgboost, datasets, joblib   |
| `ticket_router_agent`      | 本地 vLLM + SiliconFlow batch API                        | vllm, pydantic, llmcompressor                    |
| `ticket_router_rule`       | **当前为空**, 待实现                                     | -                                                |

**关键协议** (`ticket_router_base.predictor`):

- `Predictor.predict(records) -> PredictionBatch`: 所有模型统一入口.
- `Trainer.train(records, val_records) -> Predictor`: 训练器可选实现.

**关键类型** (`ticket_router_base.types`):

- `Queue` / `Priority` / `Language`: `StrEnum`.
- `Record`: 输入, 含 `request_id`, `subject`, `body`, `language` + ground truth.
- `Prediction`: 输出, 含 `queue`, `priority`, `tag_1`, `tag_2`, `answer`, `queue_confidence`, `priority_confidence`, `raw_output`, `error`.
- `ErrorFlag`: `IntFlag`, `SUCCESS=0`, 支持组合错误标记.

统一保存: `ticket_router_base.utils.write_pred(predictions, records, path)` 输出标准 JSONL.

### 2.2 数据流

```
dataset-tickets-multi-lang3-4k.csv
    -> 01_build_test_set.py
        -> outputs/test_set.jsonl      (1200条, 分层抽样)
        -> outputs/train_set.jsonl     (~2800条)
        -> outputs/difficult_cases.jsonl (100条)

train_set.jsonl -> LR/XGB/RemBERT 训练 -> outputs/supervised/*_predictions.jsonl
test_set.jsonl  -> vLLM/Qwen3 推理    -> outputs/goal_based/*_predictions.jsonl
```

**硬性约束**: 测试集**只能**来自 4k 数据. 20k/28k 用于训练增强, 但二者有 8306 条重复, 合并必须去重.

---

## 3. 当前状态速览

### 已完成

- [x] `ticket_router_base`: 类型系统、Pandera schema、数据加载、分层抽样、困难案例筛选、JSONL 日志、分类指标、一致性指标.
- [x] `ticket_router.supervised`: LR + XGBoost 已实现并跑通全量测试集; RemBERT 训练/推理已实现.
- [x] `ticket_router_agent`: vLLM 本地推理(Qwen3 0.6B/1.7B/4B, 支持 AWQ 量化) + SiliconFlow batch API 请求生成.
- [x] Qwen3 W8A8 量化脚本 (`llmcompressor`).
- [x] 统一测试集、训练集、困难案例集已生成.

### 待实现 / 待完善

- [ ] **Rule-Based 系统** (`packages/ticket_router_rule/`): 完全空白. 可参考旧包 `packages/ticket_router/rule_based/` 中的原型(关键词匹配 + 模板选择).
- [ ] **Tags 预测启用**: LR 已训练 tag 模型但未在 `LRPredictor.predict()` 中启用; XGB 未训练 tag; RemBERT 未训练 tag.
- [ ] **Preliminary answer 生成**: Supervised 系统目前只预测 queue + priority, 不生成 answer.
- [ ] **统一评估脚本**: 读取各系统 JSONL 输出, 计算对比指标, 生成报表. 计划放在根目录 `scripts/` 或 `packages/ticket_router_base/eval/`.
- [ ] **LLM-as-Judge**: 使用强模型对 answer 质量评分.
- [ ] **公平性分析**: 使用 `aif360` / `fairlearn` 分析语言/queue 层面的偏差.
- [ ] **测试目录**: `tests/` 尚未创建, 当前无 pytest 覆盖.
- [ ] **多次运行一致性**: Goal-Based 同一请求 3 次运行的方差分析脚本.

---

## 4. 编码速查

### 4.1 添加新模型

若新增模型(如 Rule-Based 或新 LLM), 遵循以下模式:

```python
from ticket_router_base.predictor import Predictor
from ticket_router_base.types import Prediction, PredictionBatch, Queue, Priority, ErrorFlag

class MyPredictor(Predictor):
    supports_tags: bool = False          # 是否支持 tag 预测
    supports_preliminary_answer: bool = False  # 是否生成 answer

    def predict(self, records) -> PredictionBatch:
        # records: List[Record] | RecordDF
        # 必须返回 PredictionBatch(predictions=[...], parse_err_count=0, parse_json_err_count=0)
        ...
```

### 4.2 运行脚本

```bash
# 使用 just (推荐)
just prepare-data       # 构建测试集
just run-ml             # LR + XGBoost
just run-mbert --infer  # RemBERT 推理
just run-vllm <model>   # 本地 vLLM

# 或直接用 uv run --project <包路径> <脚本路径>
uv run --project packages/ticket_router.supervised packages/ticket_router.supervised/scripts/03_run_supervised_traditional.py
```

### 4.3 环境管理

```bash
# 根项目无实质依赖, 不要在这里 uv add
cd packages/ticket_router_base && uv sync
cd packages/ticket_router.supervised && uv sync
cd packages/ticket_router_agent && uv sync

# 添加依赖到指定包
uv add --project packages/ticket_router_agent <package>
```

### 4.4 类型检查

项目配置了 `pyrightconfig.json`, 包含多包 `extraPaths`. 运行:

```bash
pyright
```

---

## 5. Claude 工作注意事项

1. **不要自行执行任何 git 写操作**: 包括但不限于 `git commit`、`git push`、`git rebase`、`git reset`、`git merge` 等. **除非用户显式说明要求提交**, 否则只能执行 `git status`、`git diff`、`git log` 等只读操作.

2. **不要修改旧包 `packages/ticket_router/`**: 该包为早期原型代码, 已被新的 monorepo 结构取代. 用户明确要求"不包括 packages/ticket_router".

3. **保持 Protocol 兼容**: 任何新 Predictor 必须实现 `predict()` 方法, 返回类型严格为 `PredictionBatch`. 这是三种系统能够被统一评估的前提.

4. **数据隔离红线**: 任何评估、训练脚本只能使用 `train_set.jsonl` 作为训练数据. 禁止将 `test_set.jsonl` 或 `difficult_cases.jsonl` 用于训练.

5. **中文注释, 英文标点**: 代码注释使用中文, 但标点符号使用英文半角.

6. **异常处理**: 仅在 IO、网络请求处使用 try-except. 业务逻辑错误用显式检查 + raise ValueError.

7. **输出规范**: 批量运行结果必须保存到 `outputs/` 下 JSONL, 使用 `write_pred()` 统一格式.

8. **当实现评估/报告时**: 应读取各系统已有的 JSONL 输出(`outputs/supervised/*.jsonl`, `outputs/goal_based/*.jsonl`), 不要重新运行模型推理.

---

## 6. 关键文件速查表

| 目的               | 路径                                                                             |
| ------------------ | -------------------------------------------------------------------------------- |
| 核心类型/Enum      | `packages/ticket_router_base/src/ticket_router_base/types.py`                    |
| 数据集路径配置     | `packages/ticket_router_base/src/ticket_router_base/config.py`                   |
| Predictor Protocol | `packages/ticket_router_base/src/ticket_router_base/predictor.py`                |
| 数据加载器         | `packages/ticket_router_base/src/ticket_router_base/data/loader.py`              |
| 分层抽样/困难案例  | `packages/ticket_router_base/src/ticket_router_base/data/utils.py`               |
| 评估指标           | `packages/ticket_router_base/src/ticket_router_base/eval/metrics.py`             |
| 统一预测保存       | `packages/ticket_router_base/src/ticket_router_base/utils.py`                    |
| LR/XGB 模型        | `packages/ticket_router.supervised/src/ticket_router.supervised/models/`         |
| RemBERT 微调       | `packages/ticket_router.supervised/src/ticket_router.supervised/models/mbert.py` |
| vLLM 推理          | `packages/ticket_router_agent/src/ticket_router_agent/infer.py`                  |
| Prompt 构建        | `packages/ticket_router_agent/src/ticket_router_agent/prompt.py`                 |
| just 任务          | `justfile`                                                                       |
| 类型检查配置       | `pyrightconfig.json`                                                             |
