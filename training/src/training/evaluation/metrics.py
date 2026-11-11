

import numpy as np


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float | None]:

    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    if len(y_true) == 0:
        raise ValueError("Cannot evaluate empty predictions")

    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("Targets and predictions must be finite")

    errors = y_true - y_pred

    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(errors))

    # A neutral prediction has no directional information.
    if np.all(y_pred == 0):
        directional_accuracy = None
    else:
        valid = y_true != 0
        if not np.any(valid):
            directional_accuracy = None
        else:
            directional_accuracy = np.mean(
                np.sign(y_true[valid]) == np.sign(y_pred[valid])
            )

    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "directional_accuracy": (
            float(directional_accuracy)
            if directional_accuracy is not None
            else None
        ),
    }


def prediction_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have equal shapes")

    if len(y_true) < 2:
        raise ValueError("At least two samples are required")

    std_true = np.std(y_true)
    std_pred = np.std(y_pred)

    correlation = np.corrcoef(y_true, y_pred)[0, 1]

    slope = np.polyfit(y_true, y_pred, 1)[0]

    return {
        "mean_actual": float(np.mean(y_true)),
        "mean_predicted": float(np.mean(y_pred)),
        "std_actual": float(std_true),
        "std_predicted": float(std_pred),
        "std_ratio": float(std_pred / std_true)
        if std_true > 0 else float("nan"),
        "pearson_correlation": float(correlation),
        "regression_slope": float(slope),
    }
