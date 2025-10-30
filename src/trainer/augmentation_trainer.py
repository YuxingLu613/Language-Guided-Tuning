from __future__ import annotations

"""Minimal placeholder AugmentationTrainer.
This project previously referenced an AugmentationTrainer that
was removed. Some modules (e.g. augmentation_noadvisor_trainer.py)
still inherit from it. To avoid breaking imports we provide a thin
wrapper around the base `Trainer` that does nothing extra. If full
augmentation capabilities are required later, extend this class.
"""

from typing import Any, Dict, Optional
from torch.utils.data import DataLoader
from .trainer import Trainer


class AugmentationTrainer(Trainer):
    """A no-op trainer that simply inherits all behaviour from `Trainer`."""

    # Overriding only to maintain interface; call parent methods directly.
    def __init__(self, model, config_path):  # type: ignore[override]
        super().__init__(model, config_path)
        self.augmentation_enabled = False

    # For compatibility: if subclasses expect these helpers, provide stubs.
    def _ensure_dataset_wrapped(self, train_loader: DataLoader):
        pass

    def _ensure_exclusive_logdir(self):
        pass

    # Maintain signature but defer to base implementation.
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):  # type: ignore[override]
        return super().train(train_loader, val_loader) 