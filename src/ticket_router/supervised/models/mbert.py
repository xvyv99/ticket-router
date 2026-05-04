"""mBERT fine-tuning training and inference for arbitrary classification tasks."""

from logging import getLogger

from transformers import TrainingArguments

from ticket_router_base.config import SEED
from ticket_router_base.predictor import register_model

from .hf import HFPredictor, HFTrainer
from ticket_router.supervised.config import SAVE_DIR

logger = getLogger(__name__)

DEFAULT_TRAIN_ARGS = TrainingArguments(
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    lr_scheduler_type="cosine_with_restarts",
    warmup_steps=21,
    weight_decay=0.01,
    max_grad_norm=1.0,
    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    seed=SEED,
    logging_steps=30,
    report_to=["none"],
)


@register_model
class MBERTPredictor(HFPredictor):
    name = "mbert"
    DEFAULT_SAVE_DIR = SAVE_DIR
    INFER_BATCH_SIZE = 64


class MBERTTrainer(HFTrainer):
    predictor_cls = MBERTPredictor
    model_name = "google/rembert"
    DEFAULT_TRAIN_ARGS = DEFAULT_TRAIN_ARGS
