import argparse
import os

from training.config import PROJECT_ROOT,load_config
from training.main import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="S&P 500 forecasting experiments"
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment YAML configuration",
    )

    args = parser.parse_args()

    project_root = PROJECT_ROOT

    config_path = args.config

    if not os.path.isabs(config_path):
        config_path = os.path.join(project_root, config_path)

    config = load_config(config_path)

    run_experiment(
        config=config,
        project_root=project_root,
    )


if __name__ == "__main__":
    main()
