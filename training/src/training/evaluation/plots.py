

import os

import matplotlib.pyplot as plt
import numpy as np


def _save_figure(fig: plt.Figure, output_path: str) -> None:
    output_dir = os.path.dirname(output_path)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_training_history(
    history: dict,
    output_path: str,
) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(epochs, history["train_loss"], label="Train Loss")
    ax.plot(epochs, history["val_loss"], label="Validation Loss")

    ax.set_title("Training and Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    _save_figure(fig, output_path)


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_dates: np.ndarray,
    output_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(target_dates, y_true, label="Actual", alpha=0.8)
    ax.plot(target_dates, y_pred, label="Predicted", alpha=0.8)

    ax.set_title("Actual vs Predicted Returns")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Return")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.autofmt_xdate()
    _save_figure(fig, output_path)


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str,
) -> None:
    residuals = y_true - y_pred

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(residuals, bins=30)
    ax.axvline(0, linestyle="--", label="Zero Error")

    ax.set_title("Residual Distribution")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(alpha=0.3)

    _save_figure(fig, output_path)


def plot_prediction_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(y_true, y_pred, alpha=0.6)

    min_value = min(y_true.min(), y_pred.min())
    max_value = max(y_true.max(), y_pred.max())

    ax.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--",
        label="Ideal Prediction",
    )

    ax.set_title("Predicted vs Actual")
    ax.set_xlabel("Actual Returns")
    ax.set_ylabel("Predicted Returns")
    ax.legend()
    ax.grid(alpha=0.3)

    _save_figure(fig, output_path)

