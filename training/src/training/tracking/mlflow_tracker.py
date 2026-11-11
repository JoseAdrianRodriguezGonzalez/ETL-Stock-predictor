import json
import math
import os
from contextlib import contextmanager
from typing import Iterator

import mlflow
import torch

class MLflowTracker:
    def __init__(
        self,
        enabled: bool = True,
        experiment_name: str = "sp500_forecasting",
        tracking_uri: str | None = None,
    ) -> None:
        self.enabled = enabled

        if not self.enabled:
            return

        uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI")

        if not uri:
            raise ValueError("MLFLOW_TRACKING_URI is not configured")

        mlflow.set_tracking_uri(uri)
        mlflow.set_registry_uri(uri)
        mlflow.set_experiment(experiment_name)

    @contextmanager
    def start_run(self, run_name: str) -> Iterator:
        if not self.enabled:
            yield None
            return

        with mlflow.start_run(run_name=run_name) as run:
            yield run

    @staticmethod
    def _flatten_config(
        config: dict,
        prefix: str = "",
    ) -> dict[str, str]:
        params = {}

        for key, value in config.items():
            name = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                params.update(
                    MLflowTracker._flatten_config(value, name)
                )
            elif isinstance(value, (list, tuple)):
                params[name] = json.dumps(value)
            elif value is not None:
                params[name] = str(value)

        return params

    def log_config(self, config: dict) -> None:
        if not self.enabled:
            return

        params = self._flatten_config(config)

        # MLflow limits parameter values to 500 characters.
        params = {
            key: value[:500]
            for key, value in params.items()
        }

        mlflow.log_params(params)

    def log_tags(self, tags: dict[str, str]) -> None:
        if not self.enabled:
            return

        mlflow.set_tags(tags)

    def log_epoch_metrics(
        self,
        epoch: int,
        metrics: dict[str, float],
    ) -> None:
        if not self.enabled:
            return

        self.log_metrics(metrics, step=epoch)

    def log_metrics(
        self,
        metrics: dict[str, float | int | None],
        step: int | None = None,
    ) -> None:
        if not self.enabled:
            return

        valid_metrics = {}

        for name, value in metrics.items():
            if value is None:
                continue

            numeric_value = float(value)

            if math.isfinite(numeric_value):
                valid_metrics[name] = numeric_value

        if valid_metrics:
            mlflow.log_metrics(
                valid_metrics,
                step=step,
            )
    def log_artifact(
        self,
        filepath: str,
        artifact_path: str | None = None,
    ) -> None:
        if not self.enabled:
            return

        if not os.path.isfile(filepath):
            raise FileNotFoundError(
                f"Artifact not found: {filepath}"
            )

        mlflow.log_artifact(
            local_path=filepath,
            artifact_path=artifact_path,
        )


    def log_artifacts(
        self,
        directory: str,
        artifact_path: str | None = None,
    ) -> None:
        if not self.enabled:
            return

        if not os.path.isdir(directory):
            raise NotADirectoryError(
                f"Artifact directory not found: {directory}"
            )

        mlflow.log_artifacts(
            local_dir=directory,
            artifact_path=artifact_path,
        )


    def log_dict(
        self,
        dictionary: dict,
        artifact_file: str,
    ) -> None:
        if not self.enabled:
            return

        mlflow.log_dict(
            dictionary,
            artifact_file,
        )


    def log_pytorch_model(
        self,
        model: torch.nn.Module,
        model_name: str,
        registered_model_name: str | None = None,
        signature=None,
        input_example=None,
    ):
        if not self.enabled:
            return None

        import mlflow.pytorch

        return mlflow.pytorch.log_model(
            pytorch_model=model,
            name=model_name,
            registered_model_name=registered_model_name,
            signature=signature,
            input_example=input_example,
        )
