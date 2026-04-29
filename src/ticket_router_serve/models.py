"""Model pool — lazy loading and unified access to all predictors."""

from logging import getLogger

from openai import OpenAI

from ticket_router_base.config import MODEL_DIR as BASE_MODEL_DIR
from ticket_router_base.data import get_dataset
from ticket_router_base.predictor import Predictor

from ticket_router_supervised.models import LRPredictor, XGBPredictor
from ticket_router_supervised.models.mbert import MBERTPredictor
from ticket_router_supervised.models.xlm_roberta import XLMRoBERTaPredictor
from ticket_router_rule.predictor import RuleBasedPredictor

from ticket_router_serve.config import DASHSCOPE_API_KEY

logger = getLogger(__name__)

DATASET = get_dataset("multilingual-customer-support")()

SUPPORTED_MODELS = {"lr", "xgb", "rule-based", "rembert", "xlm-roberta"}

SUPERVISED_MODEL_DIR = BASE_MODEL_DIR / "supervised"
RULE_BASED_MODEL_DIR = BASE_MODEL_DIR / "rule_based"


def _load_lr_models() -> dict:
    """Load trained LR models from disk."""
    import joblib

    models: dict = {}
    for task in DATASET.all_tasks:
        path = SUPERVISED_MODEL_DIR / f"lr_{task.name}.joblib"
        if path.exists():
            models[task.name] = joblib.load(path)
    return models


def _load_xgb_models() -> dict:
    """Load trained XGBoost models from disk."""
    import joblib

    models: dict = {}
    for task in DATASET.all_tasks:
        path = SUPERVISED_MODEL_DIR / f"xgb_{task.name}.joblib"
        if path.exists():
            models[task.name] = joblib.load(path)
    return models


def _load_rule_models() -> tuple[dict, dict]:
    """Load trained rule-based models from disk, returning (models, feature_modes)."""
    import pickle

    models: dict = {}
    feature_modes: dict = {}
    for task in DATASET.all_tasks:
        path = RULE_BASED_MODEL_DIR / "models" / f"{task.name}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                model = pickle.load(f)
            models[task.name] = model
            # Infer feature_mode from the loaded model's attribute if available
            feature_modes[task.name] = getattr(model, "_feature_mode", "text")
    return models, feature_modes


class ModelPool:
    """Global model pool with lazy loading."""

    _instance: "ModelPool | None" = None

    def __init__(self) -> None:
        self._predictors: dict[str, Predictor] = {}
        self._dashscope_client: OpenAI | None = None
        self._initialized: bool = False

    @classmethod
    def get_instance(cls) -> "ModelPool":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def initialize(self) -> None:
        """Pre-load lightweight models at startup."""
        if self._initialized:
            return
        logger.info("Initializing model pool...")

        # Load LR
        try:
            from ticket_router_supervised.cfg import SupervisedCfg

            lr_models = _load_lr_models()
            if lr_models:
                self._predictors["lr"] = LRPredictor(
                    dataset=DATASET, models=lr_models, cfg=SupervisedCfg()
                )
                logger.info("LR predictor loaded")
        except Exception as e:
            logger.warning(f"Failed to load LR predictor: {e}")

        # Load XGBoost
        try:
            from ticket_router_supervised.cfg import SupervisedCfg

            xgb_models = _load_xgb_models()
            if xgb_models:
                self._predictors["xgb"] = XGBPredictor(
                    dataset=DATASET, models=xgb_models, cfg=SupervisedCfg()
                )
                logger.info("XGB predictor loaded")
        except Exception as e:
            logger.warning(f"Failed to load XGB predictor: {e}")

        # Load rule-based
        try:
            from ticket_router_rule.cfg import RuleBasedCfg

            rule_models, feature_modes = _load_rule_models()
            if rule_models:
                self._predictors["rule-based"] = RuleBasedPredictor(
                    dataset=DATASET,
                    models=rule_models,
                    feature_modes=feature_modes,
                    cfg=RuleBasedCfg(),
                )
                logger.info("Rule-based predictor loaded")
        except Exception as e:
            logger.warning(f"Failed to load rule-based predictor: {e}")

        self._initialized = True
        logger.info("Model pool initialized")

    def get_predictor(self, model_name: str) -> Predictor:
        """Get or lazily load a predictor by name."""
        if model_name not in SUPPORTED_MODELS:
            raise ValueError(f"Unknown model: {model_name}")

        if model_name in self._predictors:
            return self._predictors[model_name]

        # Lazy load heavy models
        if model_name == "rembert":
            predictor = MBERTPredictor.load_model(DATASET)
            self._predictors["rembert"] = predictor
            logger.info("RemBERT predictor loaded (lazy)")
            return predictor

        if model_name == "xlm-roberta":
            predictor = XLMRoBERTaPredictor.load_model(DATASET)
            self._predictors["xlm-roberta"] = predictor
            logger.info("XLM-RoBERTa predictor loaded (lazy)")
            return predictor

        raise ValueError(f"Model not available: {model_name}")

    @property
    def dashscope_client(self) -> OpenAI:
        """Lazy initialization of DashScope API client."""
        if self._dashscope_client is None:
            if not DASHSCOPE_API_KEY:
                raise RuntimeError("DASHSCOPE_API_KEY not configured")
            self._dashscope_client = OpenAI(
                api_key=DASHSCOPE_API_KEY,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
            logger.info("DashScope client initialized")
        return self._dashscope_client

    def call_qwen3(self, messages: list[dict[str, str]]) -> str:
        """Call DashScope API with qwen3.6-plus model."""
        try:
            response = self.dashscope_client.chat.completions.create(
                model="qwen3.6-plus",
                messages=messages,
                timeout=30.0,
            )
            content = response.choices[0].message.content
            return content if content is not None else ""
        except Exception as e:
            logger.error(f"Qwen3 API call failed: {e}")
            raise


# Convenience function
def get_pool() -> ModelPool:
    return ModelPool.get_instance()
