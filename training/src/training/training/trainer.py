import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.training.early_stopping import EarlyStopping


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()

    total_loss = 0.0
    total_samples = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        predictions = model(inputs)

        if predictions.shape != targets.shape:
            raise ValueError(
                f"Prediction shape {predictions.shape} "
                f"does not match target shape {targets.shape}"
            )

        loss = criterion(predictions, targets)

        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Training DataLoader is empty")

    return total_loss / total_samples


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.inference_mode():
        for inputs, targets in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            predictions = model(inputs)

            if predictions.shape != targets.shape:
                raise ValueError(
                    f"Prediction shape {predictions.shape} "
                    f"does not match target shape {targets.shape}"
                )

            loss = criterion(predictions, targets)

            batch_size = targets.size(0)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Validation DataLoader is empty")

    return total_loss / total_samples


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    early_stopping: EarlyStopping | None = None,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None = None,
) -> dict[str, list[float]]:
    if num_epochs <= 0:
        raise ValueError("num_epochs must be positive")

    history = {
        "train_loss": [],
        "val_loss": [],
        "learning_rate": [],
    }

    model.to(device)

    for epoch in range(num_epochs):
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["learning_rate"].append(current_lr)

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"  Train loss: {train_loss:.8f}")
        print(f"  Val loss:   {val_loss:.8f}")
        print(f"  LR:         {current_lr:.2e}")

        if scheduler is not None:
            scheduler.step(val_loss)

        if early_stopping is not None:
            early_stopping(model, val_loss)

            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

    if early_stopping is not None:
        early_stopping.restore_model(model)

    return history
