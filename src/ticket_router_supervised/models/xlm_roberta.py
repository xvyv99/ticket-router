"""XLM-RoBERTa-base fine-tuning training and inference for arbitrary classification tasks."""

from logging import getLogger
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer as HFTrainer,
    TrainingArguments,
)
from tqdm import tqdm

from ticket_router_base.config import MODEL_DIR, SEED
from ticket_router_base.data import BaseDataset
from ticket_router_base.predictor import (
    Predictor,
    Trainer as TrainerProtocol,
    register_model,
)
from ticket_router_base.types import (
    ErrorFlag,
    Prediction,
    Record,
)
from ticket_router_base.utils import combine_texts

from ticket_router_supervised.config import TORCH_DEVICE, SAVE_DIR
from ticket_router_supervised.utils import create_datasets

logger = getLogger(__name__)

MODEL_NAME = "FacebookAI/xlm-roberta-base"

MODEL_DIR = MODEL_DIR / "xlm-roberta"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

XLMROBERTA_INFER_BATCH_SIZE = 256

DEFAULT_TRAIN_ARGS = TrainingArguments(
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    weight_decay=0.01,
    max_grad_norm=1.0,
    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    seed=SEED,
    logging_steps=30,
    report_to=["none"],
)


def tokenize_func(examples, tokenizer, max_length: int = 256):
    return tokenizer(
        examples["text"], padding="max_length", truncation=True, max_length=max_length
    )


def train_xlm_roberta(
    train_ds: Dataset,
    val_ds: Dataset,
    id2label: Dict[int, str],
    label2id: Dict[str, int],
    target_col: str,
    save_path: Path,
    epochs: int = 3,
) -> HFTrainer:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(label2id)
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
    trainer = HFTrainer(
        model=model, args=args, train_dataset=train_tok, eval_dataset=val_tok
    )
    trainer.train()
    trainer.save_model(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    return trainer


def predict_xlm_roberta(
    model_path: Path,
    records: List[Record],
    id2label: Dict[int, str],
    batch_size: int = XLMROBERTA_INFER_BATCH_SIZE,
) -> List[Tuple[str, float]]:
    """Run inference using a fine-tuned XLM-RoBERTa model."""
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


@register_model
class XLMRoBERTaPredictor(Predictor):
    name = "xlm-roberta"
    DEFAULT_SAVE_DIR = SAVE_DIR

    _model_paths: Dict[str, Path]
    dataset: BaseDataset

    def __init__(self, model_paths: Dict[str, Path], dataset: BaseDataset):
        self._model_paths = model_paths
        self.dataset = dataset

    def predict(self, records: List[Record], run_id: int = 0) -> List[Prediction]:
        task_results: Dict[str, List[Tuple[str, float]]] = {}
        for task_name, model_path in self._model_paths.items():
            id2label = self.dataset.get_id2label(task_name)
            task_results[task_name] = predict_xlm_roberta(model_path, records, id2label)

        predictions = []
        for i, rec in enumerate(records):
            labels: Dict[str, str] = {}
            confidences: Dict[str, float] = {}
            for task_name in self.dataset.task_names:
                result = task_results[task_name]
                labels[task_name] = result[i][0]
                confidences[task_name] = result[i][1]

            pred = Prediction(
                request_id=rec.request_id,
                sensitive_attributes=rec.sensitive_attributes,
                labels=labels,
                discrete_features=rec.discrete_features,
                generation_target=None,
                confidences=confidences,
                raw_output=None,
                error=ErrorFlag.SUCCESS,
            )
            predictions.append(pred)

        return predictions


class XLMRoBERTaTrainer(TrainerProtocol):
    dataset: BaseDataset

    def __init__(self, dataset: BaseDataset):
        self.dataset = dataset

    def train(
        self,
        records: List[Record],
        val_records: List[Record] | None = None,
        epochs: int = 3,
    ) -> XLMRoBERTaPredictor:
        if val_records is None:
            raise ValueError(
                "XLMRoBERTaTrainer requires val_records for early stopping"
            )
        train_ds = create_datasets(records, self.dataset)
        val_ds = create_datasets(val_records, self.dataset)

        model_paths: Dict[str, Path] = {}
        for task in self.dataset.all_tasks:
            logger.info(f"Starting XLM-RoBERTa training for {task.name}...")
            save_path = MODEL_DIR / f"{task.name}_best"
            train_xlm_roberta(
                train_ds,
                val_ds,
                id2label=self.dataset.get_id2label(task.name),
                label2id=self.dataset.get_label2id(task.name),
                target_col=task.name,
                save_path=save_path,
                epochs=epochs,
            )
            logger.info(f"{task.name} model training complete!")
            model_paths[task.name] = save_path

        return XLMRoBERTaPredictor(model_paths=model_paths, dataset=self.dataset)
