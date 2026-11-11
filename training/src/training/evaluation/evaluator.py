import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.evaluation.metrics import regression_metrics


def predict(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:

    model.eval()
    model.to(device)

    predictions = []
    targets_list = []

    with torch.inference_mode():
        for inputs, targets in dataloader:
            inputs = inputs.to(device, non_blocking=True)

            outputs = model(inputs)

            predictions.append(outputs.cpu().numpy())
            targets_list.append(targets.numpy())

    if not predictions:
        raise ValueError("Cannot evaluate an empty dataloader")

    y_pred = np.concatenate(predictions)
    y_true = np.concatenate(targets_list)

    return y_true, y_pred


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:

    y_true, y_pred = predict(
        model=model,
        dataloader=dataloader,
        device=device,
    )

    metrics = regression_metrics(y_true, y_pred)

    return {
        "metrics": metrics,
        "y_true": y_true,
        "y_pred": y_pred,
    }
