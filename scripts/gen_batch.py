from typing import List, Dict
from logging import getLogger, basicConfig
from dataclasses import dataclass

from ticket_router_base.data import BaseDataset, load_test_set
from ticket_router_base.data.descriptor import TaskDescriptor
from ticket_router_base.types import Record
from ticket_router_base.config import LOGGING_FORMAT
from ticket_router_base.utils import JSONLLogger

from ticket_router_agent.config import SAVE_DIR
from ticket_router_agent.prompt import build_conversation

logger = getLogger(__name__)


@dataclass(frozen=True)
class ModelConfig:
    name: str
    model_id: str
    thinking_budget: int | None
    max_tokens: int
    temperature: float


BATCH_SAVE_DIR = SAVE_DIR / "batch_file"
BATCH_SAVE_DIR.mkdir(parents=True, exist_ok=True)

MODELS_CFG = [
    ModelConfig(
        name="DeepSeek-V3",
        model_id="deepseek-ai/DeepSeek-V3",
        thinking_budget=None,
        max_tokens=256,
        temperature=0.7,
    ),
    ModelConfig(
        name="DeepSeek-R1",
        model_id="deepseek-ai/DeepSeek-R1",
        thinking_budget=1024,
        max_tokens=256,
        temperature=0.7,
    ),
    ModelConfig(
        name="QwQ-32B",
        model_id="Qwen/QwQ-32B",
        thinking_budget=1024,
        max_tokens=256,
        temperature=0.7,
    ),
    ModelConfig(
        name="V3.1-Terminus",
        model_id="deepseek-ai/DeepSeek-V3.1-Terminus",
        thinking_budget=None,
        max_tokens=256,
        temperature=0.7,
    ),
]


def build_messages(subject: str, body: str, language: str, task_descriptor: TaskDescriptor, dataset: BaseDataset):
    from ticket_router_base.types import Record

    # Map string language to Language enum if needed
    lang_enum = None
    for k, v in dataset.str2lang.items():
        if k == language or v.value == language:
            lang_enum = v
            break

    rec = Record(
        request_id="tmp",
        title=subject,
        body=body,
        language=lang_enum,
        labels={},
        discrete_features={},
        generation_target=None,
        sensitive_attributes={"language": language},
    )
    return build_conversation(rec, dataset)


def build_request_body(model_cfg: ModelConfig, messages: list) -> dict:
    body = {
        "model": model_cfg.model_id,
        "messages": messages,
        "temperature": model_cfg.temperature,
        "max_tokens": model_cfg.max_tokens,
    }
    if model_cfg.thinking_budget is not None:
        body["thinking_budget"] = model_cfg.thinking_budget
    return body


def generate_batch_jsonl(
    model_cfg: ModelConfig, records: List[Record], task_descriptor: TaskDescriptor, dataset: BaseDataset
) -> List[Dict]:
    requests = []
    for rec in records:
        messages = build_messages(
            rec.title or "",
            rec.body,
            rec.language.value if rec.language is not None else "",
            task_descriptor,
            dataset,
        )
        body = build_request_body(model_cfg, messages)
        request = {
            "custom_id": f"req-{rec.request_id}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }
        requests.append(request)
    return requests


def main():
    from ticket_router_base.data.datasets import MultilingualCustomerSupportDataset

    dataset = MultilingualCustomerSupportDataset()
    test_records = load_test_set()

    for cfg in MODELS_CFG:
        file_path = BATCH_SAVE_DIR / f"batch_{cfg.name}_fewshot.jsonl"
        requests = generate_batch_jsonl(cfg, test_records, dataset.task_descriptor, dataset)
        with JSONLLogger(file_path) as jsonl:
            for req in requests:
                jsonl.write(req)

        size_kb = file_path.stat().st_size // 1024
        logger.info(f"{file_path.name}: {len(requests)} lines, {size_kb} KB")

    logger.info(f"Done. Saved batch JSONL files to: {BATCH_SAVE_DIR}")


if __name__ == "__main__":
    basicConfig(level="INFO", format=LOGGING_FORMAT)
    main()
