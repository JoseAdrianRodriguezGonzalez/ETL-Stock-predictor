

from pathlib import Path
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv


PROJECT_ROOT = Path(
    os.getenv("PROJECT_ROOT")
    or os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
        )
    )
)

load_dotenv(PROJECT_ROOT / ".env")


def load_config(config_path: str):
    path = Path(config_path)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.is_file():
        raise FileNotFoundError(f"Configuration not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)
