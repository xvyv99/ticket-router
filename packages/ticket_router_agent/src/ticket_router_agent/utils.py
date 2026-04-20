from pathlib import Path


def normalize_model_id(model_id: str) -> str:
    return model_id.replace("/", "_")


def is_model_path(model_name_or_path: str) -> bool:
    maybe_path = Path(model_name_or_path)

    return maybe_path.exists() and maybe_path.is_dir()


def save_prefix_from_model_choice(model_choice: str) -> str:
    if is_model_path(model_choice):
        return normalize_model_id(model_choice)
    else:
        return model_choice
