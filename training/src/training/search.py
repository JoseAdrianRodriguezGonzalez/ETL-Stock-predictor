

import copy
import itertools
import os
import random

import pandas as pd

from training.config import load_config
from training.main import run_experiment

from training.tracking.mlflow_tracker import MLflowTracker
MODEL_PARAMETER_NAMES = (
    "hidden_size",
    "num_layers",
    "dropout",
    "nonlinearity",
)

SUPPORTED_METRICS = {
    "val_mse",
    "val_rmse",
    "val_mae",
}


def generate_search_space(
    model_name: str,
    search: dict,
) -> list[dict]:
    """
    Generate valid hyperparameter combinations for a model.
    """

    space = copy.deepcopy(search["space"])

    model_specific = search.get("model_specific", {}).get(
        model_name, {}
    )

    space.update(model_specific)

    keys = list(space.keys())
    values = [space[key] for key in keys]

    combinations = []

    for combination in itertools.product(*values):
        params = dict(zip(keys, combination))

        # Dropout is only effective when num_layers > 1
        if params["num_layers"] == 1 and params["dropout"] != 0.0:
            continue

        combinations.append(params)

    return combinations


def generate_trials(
    model_name: str,
    search: dict,
) -> list[dict]:
    """
    Randomly sample hyperparameter combinations without replacement.
    """

    combinations = generate_search_space(
        model_name=model_name,
        search=search,
    )

    n_trials = search["n_trials_per_model"]

    if n_trials > len(combinations):
        raise ValueError(
            f"Requested {n_trials} trials for {model_name}, "
            f"but only {len(combinations)} valid combinations exist."
        )

    rng = random.Random(search["seed"])

    return rng.sample(combinations, n_trials)


def build_trial_config(
    base_config: dict,
    search_config: dict,
    model_name: str,
    trial_params: dict,
    trial_number: int,
) -> dict:
    """
    Build an independent experiment configuration for one trial.
    """

    config = copy.deepcopy(base_config)

    config["experiment"]["name"] = (
        f"{model_name}_trial_{trial_number:03d}"
    )

    config["experiment"]["seed"] = search_config["search"]["seed"]

    config["model"]["name"] = model_name

    model_params = config["model"].setdefault("params", {})

    for name in MODEL_PARAMETER_NAMES:
        if name in trial_params:
            model_params[name] = trial_params[name]

    config["data"]["batch_size"] = trial_params["batch_size"]

    optimizer_params = config["training"]["optimizer"]["params"]

    optimizer_params["lr"] = trial_params["learning_rate"]
    optimizer_params["weight_decay"] = trial_params["weight_decay"]

    # Hyperparameter selection must use validation, never test.
    config["evaluation"] = {
        "split": "validation",
    }

    # Use the search-level MLflow configuration.
    config["tracking"] = copy.deepcopy(
        search_config.get("tracking", {})
    )

    # Model registration happens only after selecting the winner.
    config["tracking"]["register_model"] = False

    return config


def run_random_search(
    search_config: dict,
    project_root: str,
) -> pd.DataFrame:
    """
    Execute random search across the configured model architectures.
    """

    search = search_config["search"]

    if search.get("strategy", "random") != "random":
        raise ValueError("Only random search is currently supported.")

    metric = search["metric"]
    direction = search["direction"]

    if metric not in SUPPORTED_METRICS:
        raise ValueError(
            f"Unsupported search metric: {metric}. "
            f"Supported metrics: {sorted(SUPPORTED_METRICS)}"
        )

    if direction not in ("minimize", "maximize"):
        raise ValueError(
            "Search direction must be 'minimize' or 'maximize'."
        )

    base_config_path = search["base_config"]

    if not os.path.isabs(base_config_path):
        base_config_path = os.path.join(
            project_root,
            base_config_path,
        )

    base_config = load_config(base_config_path)

    results = []

    for model_name in search["models"]:

        trials = generate_trials(
            model_name=model_name,
            search=search,
        )

        print(f"\n{'=' * 60}")
        print(f"Random Search: {model_name.upper()}")
        print(f"Trials: {len(trials)}")
        print(f"{'=' * 60}")

        for trial_number, trial_params in enumerate(
            trials,
            start=1,
        ):
            print(
                f"\nTrial {trial_number}/{len(trials)} "
                f"for {model_name.upper()}"
            )
            print(f"Parameters: {trial_params}")

            trial_config = build_trial_config(
                base_config=base_config,
                search_config=search_config,
                model_name=model_name,
                trial_params=trial_params,
                trial_number=trial_number,
            )

            result = run_experiment(
                config=trial_config,
                project_root=project_root,
            )
            persistence_metrics = result["baseline"]["metrics"]
            mean_metrics = result["mean_baseline"]["metrics"]
            metrics = result["evaluation"]["metrics"]

            record = {
                "model": model_name,
                "trial": trial_number,
                "run_id": result["run_id"],

                "val_mse": metrics["mse"],
                "val_rmse": metrics["rmse"],
                "val_mae": metrics["mae"],
                "val_directional_accuracy": metrics["directional_accuracy"],

                "val_persistence_mse": persistence_metrics["mse"],
                "val_persistence_rmse": persistence_metrics["rmse"],

                "val_mean_baseline_mse": mean_metrics["mse"],
                "val_mean_baseline_rmse": mean_metrics["rmse"],

                "checkpoint_path": result["checkpoint_path"],
                **trial_params,
            }
            results.append(record)

            print(f"Validation MSE: {metrics['mse']:.8f}")

    results_df = pd.DataFrame(results)

    results_df.sort_values(
        by=metric,
        ascending=(direction == "minimize"),
        inplace=True,
    )

    results_df.reset_index(drop=True, inplace=True)
# --------------------------------------------------
# Best Model Selection
# --------------------------------------------------

    best_trial = results_df.iloc[0]

    best_baseline_mse = min(
        best_trial["val_persistence_mse"],
        best_trial["val_mean_baseline_mse"],
    )

    eligible = best_trial["val_mse"] < best_baseline_mse

    print("\nModel Selection")

    print(f"Best model: {best_trial['model']}")
    print(f"Best trial: {best_trial['trial']}")
    print(f"Model validation MSE: {best_trial['val_mse']:.8f}")
    print(f"Best baseline MSE: {best_baseline_mse:.8f}")

    if eligible:
        print("Status: ELIGIBLE FOR PROMOTION")
    else:
        print("Status: NOT ELIGIBLE FOR PROMOTION")
        print("The model does not outperform the baselines.")



    # --------------------------------------------------
    # MLflow Model Selection and Promotion
    # --------------------------------------------------

    tracking_config = search_config.get("tracking", {})

    if not tracking_config.get("enabled", True):
        print("MLflow tracking disabled. Skipping promotion.")

    elif not eligible:
        print(
            "Model is not eligible. "
            "Champion will remain unchanged."
        )

    else:
        tracker = MLflowTracker(
            enabled=True,
            experiment_name=tracking_config.get(
                "experiment_name",
                "sp500_forecasting",
            ),
        )

        registered_model_name = tracking_config.get(
            "registered_model_name",
            "sp500_forecasting",
        )

        # Retrieve current champion.
        current_champion = tracker.get_champion(
            registered_model_name=registered_model_name,
            alias="champion",
        )

        candidate_mse = float(best_trial["val_mse"])

        current_champion_mse = None

        if current_champion is not None:
            champion_tags = current_champion.tags or {}

            stored_mse = champion_tags.get("validation_mse")

            if stored_mse is None:
                raise ValueError(
                    "Current champion does not have validation_mse metadata. "
                    "Cannot safely compare models."
                )

            current_champion_mse = float(stored_mse)

        # Promote only if there is no champion or the candidate is better.
        should_promote = (
            current_champion is None
            or candidate_mse < current_champion_mse
        )

        if should_promote:

            print("\nRegistering selected model...")

            registered_version = tracker.register_model_from_run(
                run_id=best_trial["run_id"],
                registered_model_name=registered_model_name,
                model_name="forecasting_model",
                metrics={
                    "validation_mse": candidate_mse,
                    "validation_rmse": float(best_trial["val_rmse"]),
                    "validation_mae": float(best_trial["val_mae"]),
                },
                tags={
                    "architecture": str(best_trial["model"]),
                    "source_run_id": str(best_trial["run_id"]),
                    "selection_metric": metric,
                    "selection_strategy": "random_search",
                },
            )

            tracker.promote_model(
                registered_model_name=registered_model_name,
                version=registered_version.version,
                alias="champion",
            )

        else:
            print(
                "\nCandidate did not outperform the current champion."
            )
            print(f"Candidate MSE: {candidate_mse:.8f}")
            print(f"Champion MSE:  {current_champion_mse:.8f}")
            print("Champion remains unchanged.")

    reports_dir = os.path.join(
        project_root,
        "artifacts",
        "reports",
    )

    os.makedirs(reports_dir, exist_ok=True)

    report_path = os.path.join(
        reports_dir,
        "random_search_results.csv",
    )

    results_df.to_csv(
        report_path,
        index=False,
    )

    print(f"\n{'=' * 60}")
    print("RANDOM SEARCH COMPLETED")
    print(f"{'=' * 60}")
    print(f"Metric: {metric}")
    print(f"Direction: {direction}")
    print(f"Results: {report_path}")
    print("\nBest configuration:")
    print(results_df.iloc[0].to_string())

    return results_df

