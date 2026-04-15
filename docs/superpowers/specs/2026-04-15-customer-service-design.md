# Customer Service 智能客服助手设计文档

## 1. 项目背景与任务定义

本项目旨在探索 AI 客服系统的多样化构建路径与负责任 AI (Responsible AI) 实践. 我们选择 **Customer Service** 作为核心应用领域, 基于真实数据集 `multilingual-customer-support-tickets`, 系统性构建三种不同设计范式(Rule-Based, Supervised, Goal-Based)的 AI 客服系统, 并从公平性、透明性、鲁棒性、可问责性以及用户意图对齐等维度展开深度比较分析.

### 1.1 核心任务

系统接收一条客户请求(由 `subject` + `body` 组成), 输出以下四项结果:

1. **queue**: 请求应分配到的部门, 共 10 个类别
   - `Technical Support`, `Product Support`, `Customer Service`, `IT Support`, `Billing and Payments`, `Returns and Exchanges`, `Sales and Pre-Sales`, `Service Outages and Maintenance`, `General Inquiry`, `Human Resources`
2. **priority**: 请求紧急程度, 输出 `high`, `medium`, `low`
3. **tags**: 1-3 个最相关的标签, 用于客服后台分类和精细化模板选择
4. **preliminary_answer**: 标准客服回复, 用于安抚客户、说明处理流程或提供初步解决方案

### 1.2 三种设计范式

- **Rule-Based System**: 通过关键词匹配、正则表达式和 if-then-else 规则判定 queue / priority / tags, 使用模板生成回复.
- **Supervised System**: 基于数据集训练监督学习模型预测 queue / priority / tags, 传统 ML 使用模板回复, NLP 生成模型可做生成实验.
- **Goal-Based System**: 使用大语言模型(LLM), 通过高层 system prompt 定义角色与目标, 自主理解请求并直接生成全部输出.

## 2. 数据集说明

目录 `dataset/multilingual-customer-support-tickets/` 下包含多个相关数据文件, 各有侧重:

### 2.1 核心多语言数据集
- **文件**: `dataset-tickets-multi-lang3-4k.csv`
- **规模**: 4,000 条工单记录
- **语言分布**: 英语(en, 1,391 条), 德语(de, 848 条), 西班牙语(es, 812 条), 法语(fr, 476 条), 葡萄牙语(pt, 473 条)
- **Queue 分布**: `Technical Support`(1,317), `Product Support`(690), `Customer Service`(627), `IT Support`(445), `Billing and Payments`(338), `Returns and Exchanges`(197), `Service Outages and Maintenance`(141), `Sales and Pre-Sales`(137), `General Inquiry`(55), `Human Resources`(53)
- **关键字段**: `subject`, `body`, `answer`, `queue`, `priority`, `language`, `type`, `business_type`, `tag_1` ~ `tag_9`
- **用途**: **统一测试集来源**以及三种系统的基准训练/验证数据. 它是唯一覆盖 5 种语言的数据集, 是跨语言公平性分析的核心.

### 2.2 大规模英德双语数据集
- **文件**: `dataset-tickets-multi-lang-4-20k.csv` (20,000 条, EN/DE)
- **文件**: `aa_dataset-tickets-multi-lang-5-2-50-version.csv` (28,587 条, EN/DE, 多 version)
- **用途**: 用于 **Supervised 系统的英德双语增强训练**. 通过对比"仅使用 4k 全语言小数据"与"使用 20k+/28k 英德大数据"的模型表现, 可以分析"为提升主流语言性能而扩大数据规模"是否本身会加剧对小语种的覆盖缺失, 这本身是一个重要的 Fairness & Bias 讨论点.

### 2.3 德语标准化数据集
- **文件**: `dataset-tickets-german_normalized.csv` (2,125 条)
- **文件**: `dataset-tickets-german_normalized_50_5_2.csv` (13,178 条)
- **特点**: 德语文本经过标准化处理, 字段精简(无 tag 字段)
- **用途**: 用于 **鲁棒性测试**. 可以测试 Rule-Based 在规范化文本上的表现是否优于真实口语化输入, 从而暴露规则系统对非标准语言风格(拼写错误、口语化、缩写)的脆弱性.

数据集本身存在显著的类别不平衡和小样本 queue, 这为 Fairness & Bias 分析提供了天然素材.

## 3. 统一输入输出格式

### 3.1 输入格式(三种系统一致)

```json
{
  "request_id": "R-0001",
  "subject": "Return Request for Canon PIXMA MG3620",
  "body": "Dear Customer Support, I am writing to request a return...",
  "language": "en"
}
```

### 3.2 输出格式

```json
{
  "queue": "Returns and Exchanges",
  "priority": "low",
  "tags": ["Technical Support", "Hardware Failure"],
  "preliminary_answer": "Dear Customer, Thank you for contacting us..."
}
```

## 4. 系统架构详解

### 4.1 Rule-Based System

#### 组件
- **输入解析器**: 提取关键词(产品名、情绪词、动作词), 按 5 种语言分别匹配
- **规则引擎**: 10 个 Queue 的 if-then-else 规则树, Priority 判定规则, Tag 规则匹配
- **回复生成器**: 根据 (Queue, Priority, Tags) 组合选择预定义模板, 填充客户名/产品名等占位符
- **兜底策略**: 未命中任何规则时, 返回通用安抚模板 + 转人工提示

#### 多语言规则策略
- 高频 Queue(`Technical Support`, `Customer Service`, `Billing and Payments`, `Returns and Exchanges`, `Sales and Pre-Sales`): 为 EN/DE/ES/FR/PT 五种语言生成完整规则
- 低频 Queue(`Human Resources`, `General Inquiry`, `Service Outages and Maintenance`, `IT Support`, `Product Support`): 用 LLM 辅助生成多语言规则, 允许部分语言共享通用关键词 + 英语兜底规则
- 规则生成可借助 LLM, 但规则执行完全确定, 不调用任何外部模型

#### 回复策略
严格使用模板回复. 模板池由数据集 `answer` 字段总结提炼而来, 按 Queue + Priority + 关键 Tag 组合索引.

### 4.2 Supervised System

#### 分类任务
训练模型预测 `queue`, `priority`, `tags`.

#### 模型组合
1. **传统 ML 基线**:
   - `TF-IDF + Logistic Regression`
   - `TF-IDF + XGBoost`
   - 可选 `TF-IDF + Linear SVM`
   - 特征工程统一, 可做简单的多语言预处理(小写、去停用词)

2. **预训练语言模型**:
   - `mBERT` 或 `XLM-RoBERTa` 做多任务 fine-tune
   - 同时输出 queue / priority / tags, 利用预训练知识处理 5 种语言

3. **生成实验(可选)**:
   - `T5-small` 或 `BART-base` 在数据集上做 seq2seq fine-tune, 输入为 `subject` + `body`, 目标为 `answer`
   - 可借助 `20k` 或 `28k` 英德数据集扩大训练规模, 生成质量预期比仅使用 4k 更稳定. 该实验若执行, 其核心分析点不再是"小数据量不可靠", 而是"数据驱动的生成能力天然偏向数据丰富的语言(EN/DE), 对 ES/FR/PT 等小语种形成服务空白"——这本身是一个直接的 Fairness & Bias 议题.

#### 回复策略
分类模型统一使用模板回复(与 Rule-Based 共用模板池). 仅当 T5/BART 生成实验被纳入时, 才对比生成回复与模板回复的差异.

#### 训练与测试
- **统一测试集**: 必须且仅来自 `dataset-tickets-multi-lang3-4k.csv` 的独立划分部分, 与 `20k` / `28k` 数据集严格隔离.
- **训练集**:
  - 主训练数据: `dataset-tickets-multi-lang3-4k.csv` 的剩余部分(确保训练数据不少于 1,000 条).
  - 增强训练数据(可选): `20k` / `28k` 数据集. **使用前必须做去重**, 即通过 `subject` + `body` 的文本匹配, 移除任何与 `4k` 测试集或验证集重复的样本, 以避免数据泄露.
  - 若无法确认 `20k` / `28k` 与 `4k` 的关系, 保守策略是仅使用 `4k` 自身划分出的训练数据.
- 对 Queue 的类别不平衡, 可尝试 class weight 或采样策略, 但需在报告中披露并分析其对公平性的影响.

### 4.3 Goal-Based System

#### 模型 Scaling 策略
按模型规模从小到大构建 scaling 曲线, 测试量随规模递减以控制成本:

| 模型 | 规模 | 来源 | 测试量 | 说明 |
|---|---|---|---|---|
| Qwen3-0.6B | 0.6B | SiliconFlow | 1,200 条 × 3 次 | 观察极小模型的失败模式 |
| Qwen3-4B | 4B | SiliconFlow(免费) | 1,200 条 × 3 次 | 免费, RPM 1,000 |
| Qwen3-9B | 9B | SiliconFlow(免费) | 1,200 条 × 3 次 | 免费, 作为中等规模代表 |
| Qwen3-14B | 14B | SiliconFlow(付费) | 400 条 × 3 次 | 抽样, 覆盖各 queue/language |
| Qwen3-32B | 32B | SiliconFlow(付费) | 200 条 × 3 次 | 困难案例 + 边缘 queue |
| GPT-4o / Qwen3-235B-A22B | 100B+ | API | 100 条 × 3 次 | 能力天花板, 作为 gold standard |

#### 配置消融实验
每个模型可跑以下配置(视模型和进度选择):
1. **原始配置(Base)**: 纯 system prompt + user request, 要求输出 JSON
2. **+RAG**: 从数据集构建向量库(FAISS/Chroma), 检索 Top-3 相似历史工单作为 in-context example
3. **+Tool**: LLM 可调用一个内部函数 `classify_keywords()` 辅助判断 queue/priority, 模拟 Agentic 行为
4. **+Fine-tune(可选)**: 在英语子集上用 LoRA 微调本地 8B 模型(如 Llama 3.1 8B 或 Qwen3-8B). 视 Week 2 进度决定是否执行.

#### 非确定性控制
- 同一请求至少运行 3 次, 记录输出方差
- temperature 固定为 0.7(或按模型推荐), 其余 generation 参数保持一致
- 对 JSON 解析失败的输出, 先用正则提取, 再失败则标记为"格式错误"

## 5. 数据流

### 5.1 Rule-Based
```
输入 → 语言检测 → 关键词匹配(queue) → 规则树(priority) → tag 规则匹配 → 模板选择(queue+priority+tags) → 输出
```

### 5.2 Supervised
```
输入 → 特征提取/Tokenizer → 分类模型(queue, priority, tags) → 模板选择(queue+priority+tags) → 输出
```

### 5.3 Goal-Based
```
输入 → System Prompt + (可选 RAG/Tool) → LLM 直接生成 JSON → 解析验证 → 输出
```

## 6. 测试策略

### 6.1 统一测试集构建
从原始 4,000 条数据中**分层抽样**生成 **1,200 条统一测试集**, 确保:
- 每种语言比例与原始数据一致
- 每个 queue 都有代表性样本(小 queue 不过采样, 不人为扩充)
- priority 分布自然
- 三种系统使用完全相同的测试集

### 6.2 困难案例集
额外人工挑选 **100 条困难案例**, 用于所有模型的深度错误分析:
- 歧义请求(同时涉及多个 queue, 如 Billing + Returns)
- 非标准语言风格(口语化、拼写错误、语法混乱)
- 跨文化表达差异
- 小 queue 样本(`Human Resources`, `General Inquiry`)
- 高情绪强度或隐含紧急度的请求

### 6.3 各系统测试量

| 系统 | 测试量 | 说明 |
|---|---|---|
| Rule-Based | 1,200 条 | 全语言 |
| Supervised(传统 ML) | 1,200 条 | 英语为主, 部分模型可测全语言 |
| Supervised(mBERT) | 1,200 条 | 全语言 |
| Goal-Based(0.6B/4B/9B) | 1,200 条 × 3 次 | 免费模型跑满 |
| Goal-Based(14B) | 400 条 × 3 次 | 分层抽样 |
| Goal-Based(32B) | 200 条 × 3 次 | 困难案例优先 |
| Goal-Based(100B+) | 100 条 × 3 次 | 困难案例 + 随机抽样 |

### 6.4 并发与异步
针对 Goal-Based 的大规模请求, 必须实现异步并发请求(如 `asyncio` + `aiohttp`), 严格控制速率不超过 API 限制(RPM 1,000, TPM 80,000 for 免费模型).

## 7. 错误处理机制

### 7.1 Rule-Based
- **规则未命中**: 返回通用模板 + "已为您转接人工客服, 请稍候"
- **语言不在支持列表**: 默认降级到英语规则, 记录失败日志
- **多个规则冲突**: 按预设优先级取最高规则, 记录冲突事件

### 7.2 Supervised
- **低置信度**: queue/priority 最高概率低于 0.4 时, 标记"建议人工复核"
- **tag 标注冲突**: 多标签预测不一致时, 按置信度排序取 Top-3
- **生成模型异常**: T5/BART 输出无意义文本时, 回退到模板回复

### 7.3 Goal-Based
- **JSON 解析失败**: 先用正则提取关键字段, 仍失败则标记"格式错误"
- **幻觉检测**: 若 LLM 生成的 queue/tag 不在预定义列表中, 强制映射到最近合法值或标记错误
- **多次运行不一致**: 3 次输出差异大时, 计算 queue/tag/answer 的重合率, 标记"高波动性"

## 8. 评估体系

### 8.1 客观指标
- **分类指标**: Accuracy, Macro-F1, Weighted-F1, 各 queue 的 Recall, Confusion Matrix
- **语言公平性**: 各语言 Macro-F1 的方差和标准差
- **小 queue 公平性**: `Human Resources` 和 `General Inquiry` 的 Recall 与 Precision
- **一致性指标**: Goal-Based 同一请求 3 次运行的 queue 一致率、tag Jaccard 相似度、answer 语义相似度
- **生成指标(可选)**: BLEU, ROUGE-L 对比 `answer` 字段

### 8.2 LLM-as-Judge
固定使用一个强模型(GPT-4o 或 Qwen3-32B)作为评委, 避免用被测模型自评.

**评分维度**:
1. `queue_correctness` (0/1)
2. `priority_correctness` (0/1)
3. `tag_relevance` (1-5 分): tag 标注是否与请求相关
4. `answer_helpfulness` (1-5 分): 回复是否有帮助、语气是否得体
5. `answer_faithfulness` (0/1): 回复中是否包含未在请求中出现的虚假信息(幻觉检测)

**评估方式**:
- **Absolute Scoring**: 对最终报告中的典型案例和困难案例, 给出独立分数
- **Pairwise Comparison**: 对 Goal-Based 的不同配置(如 Base vs +RAG), 同一请求下比较输出优劣

**局限性声明**: LLM-as-judge 可能存在语言偏见(对英语更宽容)、对长回复的偏好、以及无法真正理解客户情绪. 因此 judge 分数会与人工抽样复核交叉验证, 并在报告中明确讨论.

## 9. Responsible AI 分析框架

报告的核心对比围绕以下 5 个支柱展开:

| 支柱 | 核心问题 | 评估方式 |
|---|---|---|
| **Transparency & Explainability** | 用户和开发者能否理解为什么得到这个回复? | Rule: 规则路径完全可追溯; Supervised: SHAP/feature importance; Goal: 黑箱, 只能看到 prompt 和输出 |
| **Fairness & Bias** | 系统是否对某些语言/queue/请求类型有系统性偏见? | 各语言 Macro-F1 对比; 小 queue Recall 差距; LLM-as-judge 跨语言评分差异 |
| **Robustness & Reliability** | 面对歧义、拼写错误、边缘输入时表现如何? | 困难案例集准确率; Goal-Based 3 次运行一致性; LLM-as-judge 对困难案例的评分分布 |
| **Accountability** | 出错时责任归属于谁? | 定性分析: 规则设计者 / 数据标注者 / prompt 工程师 / 模型提供商 / 部署公司 |
| **Alignment with User Intent** | 输出在多大程度上满足客户真实需求? | LLM-as-judge 打分 + 抽样人工复核; 典型失败案例分析 |

## 10. 技术栈

- **Python 3.13+**
- **数据处理**: pandas, numpy, scikit-learn
- **传统 ML**: scikit-learn (TF-IDF, Logistic Regression, SVM), xgboost
- **深度学习**: transformers (HuggingFace), torch, 可选 peft/unsloth 做 LoRA
- **向量库/RAG**: faiss-cpu 或 chromadb
- **LLM API**: SiliconFlow API (Qwen3 系列), 可选 OpenAI API
- **本地推理**: ollama 或 vllm
- **异步请求**: asyncio, aiohttp
- **评估**: scikit-learn metrics, 可选 sacrebleu, rouge-score
- **日志**: JSONL 结构化日志 + token 计数器 + 费用估算模块

## 11. 时间线与风险缓冲

### 11.1 建议时间线(Deadline: 4 月底)

| 周次 | 任务 |
|---|---|
| **Week 1** | 数据清洗; 统一测试集与困难案例集构建; Rule-Based 多语言规则实现; Rule-Based 全量测试 |
| **Week 2** | Supervised 传统 ML 基线; mBERT fine-tune; 可选 T5/BART 生成实验 |
| **Week 3** | Goal-Based Base/RAG/Tool 实现; 0.6B/4B/9B/14B 批量测试; 32B/100B+ 抽样测试 |
| **Week 4** | LLM-as-judge 评估; 日志整理; Phase 3 报告撰写; 交付物整合 |

### 11.2 风险与缓冲

| 风险 | 缓冲方案 |
|---|---|
| Rule-Based 多语言规则工作量过大 | 高频 Queue 优先完整覆盖, 低频 Queue 用 LLM 辅助生成 + 允许简化 |
| T5/BART 生成实验进度不足 | 标记为 optional; 若执行则聚焦语言不平等分析; 若放弃则不影响核心对比 |
| Goal-Based 批量测试 API 费用/速率受限 | 严格执行"模型越大、测试量越少"; 9B 及以下免费模型跑满; 付费模型仅抽样 |
| Fine-tune 本地模型时间不够 | 标记为 optional, 视 Week 2 进度决定 |
| LLM-as-judge 评分不稳定 | 同一 case 评 3 次取平均; 争议案例引入人工复核 |

## 12. 设计总结

**最终推荐方案**:

- **Rule-Based**: 10 Queue × 5 语言, LLM 辅助生成规则, 高频规则完整, 低频简化, 严格模板回复.
- **Supervised**: TF-IDF+LR / TF-IDF+XGBoost / mBERT 多任务分类, 统一模板回复, T5-small 生成为 optional 子实验.
- **Goal-Based**: Qwen3-0.6B→4B→9B→14B→32B→100B+ 的 scaling 实验, 核心配置为 Base / +RAG / +Tool, fine-tune 为 optional; 测试量随模型规模递减.
- **评估**: 统一测试集 + 困难案例集 + 客观指标 + LLM-as-judge + 抽样人工复核, 从 Transparency, Fairness, Robustness, Accountability, Alignment 五个维度做深度对比分析.
