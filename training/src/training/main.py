import os

from training.pipelines.data_pipeline import MarketDataPipeline
import torch
from training.datasets.pytorch import (
    MarketTensorDataset,
    create_dataloader,
     get_device,
    move_batch_to_device,
)


def main() -> None:
    project_root = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
        )
    )

    dataset_path = os.path.join(
        project_root,
        "data",
        "processed",
        "sp500",
        "historical.parquet",
    )

    pipeline = MarketDataPipeline(
        sequence_length=30,
    )

    result = pipeline.run(
        dataset_path=dataset_path,
        expected_symbol="^GSPC",
    )

    for name, dataset in [
        ("Train", result.train),
        ("Validation", result.validation),
        ("Test", result.test),
    ]:
        print(f"\n{name}")
        print(f"X shape: {dataset.X.shape}")
        print(f"y shape: {dataset.y.shape}")
        print(f"Samples: {dataset.n_samples}")

        print(
            f"Target dates: "
            f"{dataset.target_dates[0]} -> "
            f"{dataset.target_dates[-1]}"
        )
   # Convert NumPy windows to PyTorch datasets
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

# Create DataLoaders
    train_loader = create_dataloader(
        train_dataset,
        batch_size=32,
    )

    validation_loader = create_dataloader(
        validation_dataset,
        batch_size=32,
    )

    test_loader = create_dataloader(
        test_dataset,
        batch_size=32,
    )

# Inspect one batch
    device = get_device()

    print("\nDEVICE")
    print(f"Selected device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    X_batch, y_batch = next(iter(train_loader))

    X_batch, y_batch = move_batch_to_device(
        X_batch,
        y_batch,
        device,
    )

    print("\nGPU BATCH INSPECTION")
    print(f"X shape: {X_batch.shape}")
    print(f"y shape: {y_batch.shape}")
    print(f"X device: {X_batch.device}")
    print(f"y device: {y_batch.device}")
if __name__ == "__main__":
    main()

