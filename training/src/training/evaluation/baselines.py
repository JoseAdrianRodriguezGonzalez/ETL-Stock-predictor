

import numpy as np

from training.evaluation.metrics import regression_metrics


def persistence_baseline(y_true: np.ndarray) -> dict:

    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.zeros_like(y_true)

    return {
        "metrics": regression_metrics(y_true, y_pred),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def mean_baseline(
    y_train: np.ndarray,
    y_true: np.ndarray,
) -> dict:

    y_train = np.asarray(y_train, dtype=np.float64).reshape(-1)
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)

    if len(y_train) == 0:
        raise ValueError("Training targets cannot be empty")

    mean_return = np.mean(y_train)
    y_pred = np.full_like(y_true, mean_return)

    return {
        "metrics": regression_metrics(y_true, y_pred),
        "y_true": y_true,
        "y_pred": y_pred,
    }

