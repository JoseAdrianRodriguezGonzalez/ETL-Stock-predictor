import os
import torch.nn as nn
from training.pipelines.data_pipeline import MarketDataPipeline
import torch
from training.datasets.pytorch import (
    MarketTensorDataset,
    create_dataloader,
     get_device,
    move_batch_to_device,
)
from training.model.get_model import get_model
from training.training.trainer import fit
from training.training.early_stopping import EarlyStopping
def main(name_model) -> None:
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
    model=get_model(name_model)
    model=model.to(device)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    criterion= nn.MSELoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
    )
 
    early_stopping = EarlyStopping(
        patience=7,
        min_delta=1e-6,
    )
    history = fit(
        model=model,
        train_loader=train_loader,
        val_loader=validation_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=3,
        early_stopping=early_stopping,
        scheduler=scheduler,
    )
if __name__ == "__main__":
    main(name_model="rnn")
    main(name_model="lstm")

