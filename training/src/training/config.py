

from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Configuration not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")

    required_sections = {
        "experiment",
        "data",
        "model",
        "training",
        "checkpoint",
    }

    missing = required_sections - set(config)

    if missing:
        raise ValueError(
            f"Missing configuration sections: {sorted(missing)}"
        )

    return config
