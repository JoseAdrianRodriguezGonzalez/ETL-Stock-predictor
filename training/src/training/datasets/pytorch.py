import numpy as np
import torch
from torch.utils.data import Dataset


class MarketTensorDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> None:
        if X.ndim != 3:
            raise ValueError(
                f"X must have 3 dimensions, received {X.ndim}"
            )

        if y.ndim != 1:
            raise ValueError(
                f"y must have 1 dimension, received {y.ndim}"
            )

        if len(X) == 0:
            raise ValueError("Dataset cannot be empty")

        if len(X) != len(y):
            raise ValueError(
                "X and y must have the same number of samples"
            )

        if not np.isfinite(X).all():
            raise ValueError("X contains non-finite values")

        if not np.isfinite(y).all():
            raise ValueError("y contains non-finite values")

        self.X = torch.as_tensor(
            np.ascontiguousarray(X),
            dtype=torch.float32,
        )

        self.y = torch.as_tensor(
            np.ascontiguousarray(y),
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[index], self.y[index]


from torch.utils.data import DataLoader


def create_dataloader(
    dataset: MarketTensorDataset,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def move_batch_to_device(
    X: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        X.to(device, non_blocking=True),
        y.to(device, non_blocking=True),
    )
