import os
import random

import numpy as np
import torch
import torch.nn as nn

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
def run_experiment(config: dict, project_root: str) -> dict:
    seed = config["experiment"]["seed"]

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

    checkpoint_config = config["checkpoint"]

    checkpoint_path = os.path.join(
        project_root,
        checkpoint_config["directory"],
        checkpoint_config["filename"],
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
    )

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
    plot_dir = os.path.join(
        project_root,
        "artifacts",
        "evaluation",
        model_config["name"].lower(),
    )
    diagnostics = prediction_diagnostics(
        model_results["y_true"],
        model_results["y_pred"],
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
