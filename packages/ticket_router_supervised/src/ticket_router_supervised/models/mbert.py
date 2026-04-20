"""mBERT fine-tuning training and inference."""

from pathlib import Path
from typing import List, Tuple, Dict
from logging import getLogger

import torch
from tqdm import tqdm
from transformers import (
    RemBertTokenizer,
    RemBertForSequenceClassification,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset

from ticket_router_base.config import OUTPUT_DIR, SEED
from ticket_router_base.types import (
    ID2QUEUE,
    ID2PRIORITY,
    QUEUE2ID,
    PRIORITY2ID,
    Record,
    Prediction,
    PredictionBatch,
    ErrorFlag,
    Priority,
    Queue,
    RecordDF,
)
from ticket_router_base.predictor import Predictor, Trainer as TrainerProtocol
from ticket_router_base.utils import to_records, combine_texts

from ticket_router_supervised.utils import create_datasets
from ticket_router_supervised.config import TORCH_DEVICE

logger = getLogger(__name__)

MODEL_DIR = OUTPUT_DIR / "supervised" / "models" / "mbert"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MBERT_INFER_BATCH_SIZE = 64

QUEUE_MODEL_PATH = MODEL_DIR / "queue_best"
PRIORITY_MODEL_PATH = MODEL_DIR / "priority_best"

DEFAULT_TRAIN_ARGS = TrainingArguments(
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    weight_decay=0.01,
    max_grad_norm=1.0,
    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    seed=SEED,
    logging_steps=50,
    report_to=["none"],
)


def tokenize_func(examples, tokenizer, max_length: int = 256):
    return tokenizer(
        examples["text"], padding="max_length", truncation=True, max_length=max_length
    )


def train_mbert(
    train_ds: Dataset,
    val_ds: Dataset,
    id2label: Dict[int, str],
    label2id: Dict[str, int],
    target_col: str,
    save_path: Path,
    model_name: str = "google/rembert",
    epochs: int = 3,
):
    tokenizer = RemBertTokenizer.from_pretrained(model_name)
    model = RemBertForSequenceClassification.from_pretrained(
        model_name, num_labels=len(label2id)
    )
    model.config.id2label = id2label
    model.config.label2id = label2id

    train_tok = train_ds.map(lambda x: tokenize_func(x, tokenizer), batched=True)
    val_tok = val_ds.map(lambda x: tokenize_func(x, tokenizer), batched=True)
    train_tok = train_tok.rename_column(target_col, "labels")
    val_tok = val_tok.rename_column(target_col, "labels")
    train_tok.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    val_tok.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    args = DEFAULT_TRAIN_ARGS
    args.output_dir = str(MODEL_DIR / target_col)
    args.num_train_epochs = epochs
    trainer = Trainer(
        model=model, args=args, train_dataset=train_tok, eval_dataset=val_tok
    )
    trainer.train()
    trainer.save_model(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    return trainer


def predict_mbert(
    model_path: Path,
    records: List[Record] | RecordDF,
    id2label: Dict[int, str],
    batch_size: int = MBERT_INFER_BATCH_SIZE,
) -> List[Tuple[float, float]]:
    """Run inference using a fine-tuned mBERT model."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    model.to(TORCH_DEVICE)

    texts = combine_texts(records)

    results = []
    with torch.inference_mode():
        for i in tqdm(range(0, len(texts), batch_size), desc="Batched inference"):
            batch_texts = texts[i : i + batch_size]
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=256,
            )
            inputs = {k: v.to(TORCH_DEVICE) for k, v in inputs.items()}
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

            for j in range(len(batch_texts)):
                pred_id = int(torch.argmax(probs[j]))
                confidence = float(probs[j][pred_id])
                pred_label = id2label.get(pred_id, str(pred_id))

                results.append((pred_label, confidence))
    return results


class MBERTPredictor(Predictor):
    supports_tags = False
    supports_preliminary_answer = False

    def __init__(self, queue_model_path: Path | str, priority_model_path: Path | str):
        self._queue_path = Path(queue_model_path)
        self._priority_path = Path(priority_model_path)

    def predict(self, records: List[Record] | RecordDF) -> PredictionBatch:
        records = to_records(records)

        q_results = predict_mbert(self._queue_path, records, id2label=ID2QUEUE)
        p_results = predict_mbert(self._priority_path, records, id2label=ID2PRIORITY)

        predictions = []
        for i, rec in enumerate(records):
            pred = Prediction(
                request_id=rec.request_id,
                queue=Queue(q_results[i][0]),
                priority=Priority(p_results[i][0]),
                tag_1=None,
                tag_2=None,
                answer=None,
                queue_confidence=q_results[i][1],
                priority_confidence=p_results[i][1],
                raw_output=None,
                error=ErrorFlag.SUCCESS,
            )

            predictions.append(pred)
        return PredictionBatch(
            predictions=predictions, parse_err_count=0, parse_json_err_count=0
        )


class MBERTTrainer(TrainerProtocol):
    def train(
        self,
        records: List[Record] | RecordDF,
        val_records: List[Record] | RecordDF | None = None,
    ) -> MBERTPredictor:
        if val_records is None:
            raise ValueError("MBERTTrainer requires val_records for early stopping")
        train_ds = create_datasets(records)
        val_ds = create_datasets(val_records)

        logger.info("Starting remBERT training for queue...")
        train_mbert(
            train_ds,
            val_ds,
            id2label=ID2QUEUE,
            label2id=QUEUE2ID,
            target_col="queue",
            save_path=QUEUE_MODEL_PATH,
        )
        logger.info("Queue model training complete!")

        logger.info("Starting remBERT training for priority...")
        train_mbert(
            train_ds,
            val_ds,
            id2label=ID2PRIORITY,
            label2id=PRIORITY2ID,
            target_col="priority",
            save_path=PRIORITY_MODEL_PATH,
        )
        logger.info("Priority model training complete!")

        return MBERTPredictor(
            queue_model_path=MODEL_DIR / "queue_best",
            priority_model_path=MODEL_DIR / "priority_best",
        )
