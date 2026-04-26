def normalize_model_id(model_id: str) -> str:
    return model_id.replace("/", "-").lower()
