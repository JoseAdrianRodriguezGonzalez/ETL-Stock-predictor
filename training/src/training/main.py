import os
import random
import pandas as pd 
from training.tracking.mlflow_tracker import MLflowTracker
import numpy as np
import torch
import torch.nn as nn
import uuid
from training.datasets.pytorch import (
    MarketTensorDataset,
    create_dataloader,
    get_device,
)
from training.model.factory import get_model
from training.pipelines.data_pipeline import MarketDataPipeline
from training.training.trainer import fit
from training.training.early_stopping import EarlyStopping
from training.evaluation.evaluator import evaluate_model
from training.evaluation.metrics import prediction_diagnostics
from training.evaluation.baselines import persistence_baseline
from training.evaluation.plots import (
    plot_training_history,
    plot_predictions,
    plot_residuals,
    plot_prediction_scatter,
)
from mlflow.models import infer_signature
def _run_experiment(config: dict, project_root: str,tracker:MLflowTracker,run_id:str) -> dict:
    seed = config["experiment"]["seed"]
    run_dir = os.path.join(
        project_root,
        "artifacts",
        "runs",
        run_id,
    )

    checkpoint_path = os.path.join(
        run_dir,
        "checkpoints",
        "best_model.pth",
    )

    scaler_path = os.path.join(
        run_dir,
        "preprocessing",
        "scaler.joblib",
    )

    evaluation_dir = os.path.join(
        run_dir,
        "evaluation",
    )

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    os.makedirs(evaluation_dir, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]

    dataset_path = os.path.join(
        project_root,
        data_config["path"],
    )

    pipeline = MarketDataPipeline(
        sequence_length=data_config["sequence_length"],
    )

    result = pipeline.run(
        dataset_path=dataset_path,
        expected_symbol=data_config["symbol"],
    )
    result.preprocessor.save(scaler_path)
    train_dataset = MarketTensorDataset(
        result.train.X,
        result.train.y,
    )

    validation_dataset = MarketTensorDataset(
        result.validation.X,
        result.validation.y,
    )

    test_dataset = MarketTensorDataset(
        result.test.X,
        result.test.y,
    )

    batch_size = data_config["batch_size"]

    train_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
    )

    validation_loader = create_dataloader(
        validation_dataset,
        batch_size=batch_size,
    )

    test_loader = create_dataloader(
        test_dataset,
        batch_size=batch_size,
    )

    requested_device = training_config["device"]

    if requested_device == "auto":
        device = get_device()
    else:
        device = torch.device(requested_device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    print(f"Experiment: {config['experiment']['name']}")
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    model = get_model(
        model_config["name"],
        **model_config["params"],
    ).to(device)

    loss_name = training_config["loss"].lower()

    if loss_name == "mse":
        criterion = nn.MSELoss()
    elif loss_name == "smooth_l1":
        criterion = nn.SmoothL1Loss()
    else:
        raise ValueError(f"Unknown loss: {loss_name}")

    optimizer_config = training_config["optimizer"]

    if optimizer_config["name"].lower() == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            **optimizer_config["params"],
        )
    else:
        raise ValueError(
            f"Unknown optimizer: {optimizer_config['name']}"
        )

    scheduler_config = training_config["scheduler"]

    scheduler = None

    if scheduler_config["name"] == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            **scheduler_config["params"],
        )

    stopping_config = training_config["early_stopping"]

    early_stopping = EarlyStopping(
        **stopping_config,
    )


    tracker.log_tags(
        {
            "model_architecture": model_config["name"],
            "dataset_symbol": data_config["symbol"],
            "device": str(device),
            "seed": str(seed),
        }
    )
    history = fit(
        model=model,
        train_loader=train_loader,
        val_loader=validation_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=training_config["epochs"],
        early_stopping=early_stopping,
        scheduler=scheduler,
        checkpoint_path=checkpoint_path,
        config=config,
        on_epoch_end=tracker.log_epoch_metrics
    )
    history_path = os.path.join(
        run_dir,
        "training_history.json",
    )

    with open(history_path, "w", encoding="utf-8") as file:
        import json

        json.dump(history, file, indent=4)
    print(f"Best checkpoint: {checkpoint_path}")

# Evaluation
    model_results = evaluate_model(
        model=model,
        dataloader=test_loader,
        device=device,
    )

    baseline_results = persistence_baseline(
        model_results["y_true"],
    )

    predictions_df = pd.DataFrame({
        "target_date": result.test.target_dates,
        "y_true": model_results["y_true"],
        "y_pred": model_results["y_pred"],
    })

    predictions_df["residual"] = (
        predictions_df["y_true"] - predictions_df["y_pred"]
    )

    predictions_path=os.path.join(evaluation_dir,"predictions.csv")
    predictions_df.to_csv(predictions_path, index=False)


    print("\nTest Results")

    for metric, value in model_results["metrics"].items():
        if value is None:
            print(f"{metric}: N/A")
        else:
            print(f"{metric}: {value:.6f}")

    print("\nPersistence Baseline")

    for metric, value in baseline_results["metrics"].items():
        if value is None:
            print(f"{metric}: N/A")
        else:
            print(f"{metric}: {value:.6f}")
# Evaluation artifacts
    plot_dir = evaluation_dir
    diagnostics = prediction_diagnostics(
        model_results["y_true"],
        model_results["y_pred"],
    )
    metrics_report = {
        "model": model_results["metrics"],
        "baseline": baseline_results["metrics"],
        "diagnostics": diagnostics,
    }

    metrics_path = os.path.join(
        evaluation_dir,
        "metrics.json",
    )

    with open(metrics_path, "w", encoding="utf-8") as file:
        import json

        json.dump(
            metrics_report,
            file,
            indent=4,
            ensure_ascii=False,
            default=lambda value: (
                value.item()
                if isinstance(value, np.generic)
                else value.tolist()
                if isinstance(value, np.ndarray)
                else str(value)
            ),
        )


    tracker.log_metrics(
        {
            f"test_{name}": value
            for name, value in model_results["metrics"].items()
        }
    )

    tracker.log_metrics(
        {
            f"baseline_{name}": value
            for name, value in baseline_results["metrics"].items()
        }
    )

    tracker.log_metrics(
        {
            f"diagnostics_{name}": value
            for name, value in diagnostics.items()
        }
    )

    print("\nPrediction Diagnostics")

    for metric, value in diagnostics.items():
        print(f"{metric}: {value:.6f}")
    os.makedirs(plot_dir, exist_ok=True)

    target_dates = result.test.target_dates

    plot_training_history(
        history=history,
        output_path=os.path.join(plot_dir, "training_history.png"),
    )

    plot_predictions(
        y_true=model_results["y_true"],
        y_pred=model_results["y_pred"],
        target_dates=target_dates,
        output_path=os.path.join(plot_dir, "predictions.png"),
    )

    plot_residuals(
        y_true=model_results["y_true"],
        y_pred=model_results["y_pred"],
        output_path=os.path.join(plot_dir, "residuals.png"),
    )

    plot_prediction_scatter(
        y_true=model_results["y_true"],
        y_pred=model_results["y_pred"],
        output_path=os.path.join(plot_dir, "prediction_scatter.png"),
    )


    # Registrar todos los artefactos locales de esta ejecución.
    tracker.log_artifacts(
        directory=run_dir,
        artifact_path="outputs",
    )

    # --------------------------------------------------
    # MLflow Model Logging
    # --------------------------------------------------

    import copy

    tracking_config = config.get("tracking", {})

    registered_model_name = None

    if tracking_config.get("register_model", False):
        registered_model_name = tracking_config.get(
            "registered_model_name",
            "sp500_forecasting",
        )

    # Crear una copia independiente en CPU.
    # El modelo original permanece en el dispositivo de entrenamiento.
    model_for_logging = copy.deepcopy(model)
    model_for_logging = model_for_logging.to("cpu").eval()

    # Obtener un ejemplo de entrada en CPU.
    sample_inputs, _ = next(iter(train_loader))

    sample_inputs = (
        sample_inputs[:1]
        .detach()
        .cpu()
    )

    # Generar salida para construir la firma del modelo.
    with torch.inference_mode():
        sample_outputs = model_for_logging(sample_inputs)

    input_example = sample_inputs.numpy()
    output_example = sample_outputs.detach().cpu().numpy()

    signature = infer_signature(
        input_example,
        output_example,
    )

    # Registrar el modelo en MLflow.
    model_info = tracker.log_pytorch_model(
        model=model_for_logging,
        model_name="forecasting_model",
        registered_model_name=registered_model_name,
        signature=signature,
        input_example=input_example,
        serialization_format="pickle",
    )

    if model_info is not None:
        print(f"MLflow model logged: {model_info.model_uri}")

    print(f"Evaluation plots saved to: {plot_dir}")

    return {
        "history": history,
        "model": model,
        "test_loader": test_loader,
        "device": device,
        "checkpoint_path": checkpoint_path,
        "evaluation": model_results,
        "baseline": baseline_results,
    }
def run_experiment(config: dict, project_root: str) -> dict:
    tracking_config = config.get("tracking", {})

    tracker = MLflowTracker(
        enabled=tracking_config.get("enabled", True),
        experiment_name=tracking_config.get(
            "experiment_name",
            "sp500_forecasting",
        ),
    )

    with tracker.start_run(
        run_name=config["experiment"]["name"]
    ) as run:

        run_id = (
            run.info.run_id
            if run is not None
            else uuid.uuid4().hex
        )

        tracker.log_config(config)

        result = _run_experiment(
            config=config,
            project_root=project_root,
            tracker=tracker,
            run_id=run_id,
        )

        result["run_id"] = run_id

        return result
