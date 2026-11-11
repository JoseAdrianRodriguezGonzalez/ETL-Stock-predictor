

import copy

import torch
import torch.nn as nn


class EarlyStopping:
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 1e-6,
    ) -> None:
        if patience <= 0:
            raise ValueError("patience must be positive")

        if min_delta < 0:
            raise ValueError("min_delta cannot be negative")

        self.patience = patience
        self.min_delta = min_delta

        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False
        self.improved= False
        self.best_model_state = None

    def __call__(
        self,
        model: nn.Module,
        val_loss: float,
    ) -> None:
        self.improved= False
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.improved=True
            self.best_model_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        else:
            self.counter += 1

            if self.counter >= self.patience:
                self.early_stop = True

    def restore_model(self, model: nn.Module) -> None:
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
