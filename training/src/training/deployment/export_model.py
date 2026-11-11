

import json
import os
import shutil

import mlflow
import mlflow.pytorch
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

from mlflow import MlflowClient

from training.config import load_config
from training.tracking.mlflow_tracker import MLflowTracker


FEATURE_COLUMNS = [
    "daily_return",
    "overnight_gap",
    "intraday_return",
    "high_low_range",
    "log_volume",
    "vwap_deviation",
]


class TritonModelWrapper(nn.Module):
    """
    Adapts the original forecasting model to Triton's
    [batch, 1] output contract.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        predictions = self.model(x)

        if predictions.ndim == 1:
            predictions = predictions.unsqueeze(-1)

        return predictions


def export_champion(
    config: dict,
    project_root: str,
) -> dict:

    tracking_config = config.get("tracking", {})
    deployment_config = config.get("deployment", {})

    if not tracking_config.get("enabled", True):
        raise ValueError(
            "MLflow tracking must be enabled to export champion."
        )

    registered_model_name = tracking_config.get(
        "registered_model_name",
        "sp500_forecasting",
    )

    alias = deployment_config.get("alias", "champion")

    # --------------------------------------------------
    # MLflow initialization
    # --------------------------------------------------

    MLflowTracker(
        enabled=True,
        experiment_name=tracking_config.get(
            "experiment_name",
            "sp500_forecasting",
        ),
    )

    client = MlflowClient()

    # --------------------------------------------------
    # Retrieve champion
    # --------------------------------------------------

    version = client.get_model_version_by_alias(
        name=registered_model_name,
        alias=alias,
    )

    if not version.run_id:
        raise ValueError(
            "The registered model version has no source run ID."
        )

    model_version = str(version.version)
    run_id = version.run_id

    print("\nSelected model:")
    print(f"  Name: {registered_model_name}")
    print(f"  Version: {model_version}")
    print(f"  Alias: {alias}")
    print(f"  Run ID: {run_id}")

    # Load the exact registered version, not a moving alias.
    model_uri = (
        f"models:/{registered_model_name}/{model_version}"
    )

    model = mlflow.pytorch.load_model(model_uri)

    model = model.to("cpu").eval()

    # --------------------------------------------------
    # Resolve deployment paths
    # --------------------------------------------------

    model_repository = deployment_config.get(
        "model_repository",
        "deployment/model_repository",
    )

    artifact_directory = deployment_config.get(
        "artifact_directory",
        "deployment/assets",
    )

    scaler_artifact_path = deployment_config.get(
        "scaler_artifact_path",
        "outputs/preprocessing/scaler.joblib",
    )

    if not os.path.isabs(model_repository):
        model_repository = os.path.join(
            project_root,
            model_repository,
        )

    if not os.path.isabs(artifact_directory):
        artifact_directory = os.path.join(
            project_root,
            artifact_directory,
        )

    os.makedirs(artifact_directory, exist_ok=True)

    # --------------------------------------------------
    # Download scaler from the same training run
    # --------------------------------------------------

    scaler_path = client.download_artifacts(
        run_id=run_id,
        path=scaler_artifact_path,
        dst_path=artifact_directory,
    )

    scaler_destination = os.path.join(
        artifact_directory,
        "scaler.joblib",
    )

    if os.path.abspath(scaler_path) != os.path.abspath(
        scaler_destination
    ):
        shutil.copy2(
            scaler_path,
            scaler_destination,
        )

    print(f"Scaler downloaded: {scaler_destination}")

    # --------------------------------------------------
    # Model input dimensions
    # --------------------------------------------------

    base_config_path = config.get("search", {}).get(
        "base_config"
    )

    if base_config_path is None:
        raise ValueError(
            "The search.base_config setting is required "
            "to recover model input dimensions."
        )

    if not os.path.isabs(base_config_path):
        base_config_path = os.path.join(
            project_root,
            base_config_path,
        )

    base_config = load_config(base_config_path)

    sequence_length = base_config["data"]["sequence_length"]
    input_size = base_config["model"]["params"]["input_size"]

    model_name = deployment_config.get(
        "model_name",
        "sp500_forecasting",
    )

    model_version_directory = os.path.join(
        model_repository,
        model_name,
        "1",
    )

    os.makedirs(
        model_version_directory,
        exist_ok=True,
    )

    onnx_path = os.path.join(
        model_version_directory,
        "model.onnx",
    )

    # --------------------------------------------------
    # ONNX export
    # --------------------------------------------------

    wrapped_model = TritonModelWrapper(model).eval()

    dummy_input = torch.randn(
        2,
        sequence_length,
        input_size,
        dtype=torch.float32,
    )

    opset_version = deployment_config.get(
        "opset_version",
        17,
    )

    with torch.inference_mode():
        torch_output = wrapped_model(dummy_input)

    torch.onnx.export(
        wrapped_model,
        (dummy_input,),
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        opset_version=opset_version,
        dynamo=False,
    )

    # --------------------------------------------------
    # ONNX validation
    # --------------------------------------------------

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    session = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"],
    )

    onnx_output = session.run(
        ["output"],
        {"input": dummy_input.numpy()},
    )[0]

    np.testing.assert_allclose(
        torch_output.numpy(),
        onnx_output,
        rtol=1e-3,
        atol=1e-5,
    )
    # --------------------------------------------------
    # Dynamic batch validation
    # --------------------------------------------------

    print("\nValidating dynamic batch sizes...")

    for batch_size in [1, 2, 8, 32]:

        test_input = torch.randn(
            batch_size,
            sequence_length,
            input_size,
            dtype=torch.float32,
        )

        with torch.inference_mode():
            expected = wrapped_model(test_input).numpy()

        actual = session.run(
            ["output"],
            {"input": test_input.numpy()},
        )[0]

        np.testing.assert_allclose(
            expected,
            actual,
            rtol=1e-3,
            atol=1e-5,
        )

        print(
            f"  Batch {batch_size}: OK "
            f"(output shape: {actual.shape})"
        )
    print("\nONNX validation successful.")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {onnx_output.shape}")
    print(f"  ONNX file: {onnx_path}")

    # --------------------------------------------------
    # Deployment metadata
    # --------------------------------------------------

    metadata = {
        "registered_model_name": registered_model_name,
        "model_version": model_version,
        "alias": alias,
        "run_id": run_id,
        "architecture": base_config["model"]["name"],
        "sequence_length": sequence_length,
        "input_size": input_size,
        "feature_columns": FEATURE_COLUMNS,
        "scaler_path": "assets/scaler.joblib",
        "onnx_path": onnx_path,
    }

    metadata_path = os.path.join(
        artifact_directory,
        "model_metadata.json",
    )

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    return metadata

