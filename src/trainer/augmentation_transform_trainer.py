from __future__ import annotations

from typing import Dict, Any, Optional, cast, Sized
from pathlib import Path
from torch.utils.data import DataLoader
import torch
from torchvision import transforms

from .trainer import Trainer
from ..llm.augmentation_chooser import LLMAugmentationChooser
from ..utils.augment_ops import get_transform
from ..utils.augmented_dataset import AugmentedDataset


class AugTransformTrainer(Trainer):
    """Trainer that lets an LLM choose classic augmentation transforms each epoch."""

    def __init__(self, model, config_path):  # type: ignore[override]
        super().__init__(model, config_path)

        aug_cfg: Dict[str, Any] = self.config.get("augmentation", {})
        self.augmentation_enabled: bool = aug_cfg.get("enabled", False)
        self.chooser = LLMAugmentationChooser(self.config["llm"])
        self.current_aug_name: str = "none"
        # judger / prompt optimizer already from base Trainer

        self._logdir_isolated = False

    def _ensure_exclusive_logdir(self):
        """Create a timestamped subdirectory under log_dir for this run."""
        if self._logdir_isolated:
            return
        from datetime import datetime
        from ..utils.metrics_logger import MetricsLogger
        base_dir = Path(self.config["logging"]["log_dir"])
        ts = datetime.now().strftime("_aug_%Y%m%d_%H%M%S")
        new_dir = base_dir / ts
        new_dir.mkdir(parents=True, exist_ok=True)
        self.config["logging"]["log_dir"] = str(new_dir)
        self.metrics_logger = MetricsLogger(new_dir, self.config)
        self._logdir_isolated = True

    # ------------------------------------------------------------
    def _wrap_dataset(self, train_loader: DataLoader):
        if not isinstance(train_loader.dataset, AugmentedDataset):
            train_loader.dataset = AugmentedDataset(train_loader.dataset)  # type: ignore[attr-defined]
        self.base_ds = train_loader.dataset.base_dataset  # type: ignore[attr-defined]
        # save original transform sequence
        self.orig_tf: transforms.Compose = self.base_ds.transform  # type: ignore[attr-defined]

    def _apply_transform(self, aug_name: str):
        if aug_name == self.current_aug_name:
            return  # nothing to change
        extra_tf = get_transform(aug_name)
        # Compose: extra_tf then original components
        new_transform = transforms.Compose([extra_tf] + self.orig_tf.transforms)
        self.base_ds.transform = new_transform  # type: ignore[attr-defined]
        self.current_aug_name = aug_name

    # ------------------------------------------------------------
    def _evaluate_augmentation(self, train_loader: DataLoader, val_loader: Optional[DataLoader]):
        """Run baseline vs candidate augmentation selection using Judger loop."""
        # Save original states
        orig_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        orig_opt_state = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in self.optimizer.state_dict().items()}
        baseline_aug = self.current_aug_name

        # Step 1: baseline metrics (train 1 eval epoch)
        # Baseline metrics – only use validation set for fair comparison
        self.train_epoch(train_loader, is_evaluation=True)  # just to keep training steps comparable
        if val_loader is not None:
            baseline_metrics = self.validate(val_loader)  # Dict[str, float] (val only)
        else:
            # If no validation loader, fall back to train metrics (not recommended)
            baseline_metrics = self.train_epoch(train_loader, is_evaluation=True)

        attempt = 0
        guidance = None
        while attempt < self.judger.max_attempts:
            # build context for chooser
            ctx = {
                "current_augmentation": self.current_aug_name,
                "epoch": None,
                "dataset_size": len(cast("Sized", train_loader.dataset)),
                "metrics": baseline_metrics,
            }
            choice, should_use = self.chooser.choose(baseline_metrics, ctx, guidance)
            raw_choice = choice

            self._log_agent_output("chooser", {"attempt": attempt, "choice": choice})

            if not should_use or choice == self.current_aug_name:
                break  # no new augmentation to test

            # Apply candidate transform
            prev_aug = self.current_aug_name
            self._apply_transform(choice)

            # Train one epoch with candidate
            self.train_epoch(train_loader, is_evaluation=True)
            if val_loader is not None:
                cand_metrics = self.validate(val_loader)
            else:
                cand_metrics = self.train_epoch(train_loader, is_evaluation=True)

            # Judge
            use_opt, reason, suggestion = self.judger.judge_optimization(  # type: ignore[arg-type]
                baseline_metrics,  # type: ignore[arg-type]
                cand_metrics,      # type: ignore[arg-type]
                {}, {}, None
            )

            self._log_agent_output("judger", {"attempt": attempt, "use_opt": use_opt, "suggestion": suggestion})

            if use_opt:
                # keep candidate as current baseline
                baseline_metrics = cand_metrics
                break

            # Not adopted → rollback transform
            self._apply_transform(prev_aug)

            # reset model weights and optimizer
            self.model.load_state_dict(orig_state)
            self.optimizer.load_state_dict(orig_opt_state)

            if suggestion:
                # refine guidance
                guidance = self.prompt_optimizer.refine_suggestion(raw_choice, suggestion)
                self._log_agent_output("prompt_optimizer", {"attempt": attempt, "guidance": guidance})
            else:
                break

            attempt += 1

        # done
        self.judger.reset_attempts()
        return baseline_metrics

    # ------------------------------------------------------------
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):  # type: ignore[override]
        if self.augmentation_enabled:
            self._wrap_dataset(train_loader)
            self._ensure_exclusive_logdir()

        best_val = float("inf")
        patience = 0
        for epoch in range(self.config["training"]["epochs"]):
            # training
            train_metrics = self.train_epoch(train_loader, epoch)
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                metrics = {"train": train_metrics, "val": val_metrics}
                if val_metrics["loss"] < best_val:
                    best_val = val_metrics["loss"]
                    patience = 0
                else:
                    patience += 1
                if patience >= self.config["training"].get("early_stopping_patience", 5):
                    break
            else:
                metrics = {"train": train_metrics}

            # log metrics
            self.metrics_logger.update(epoch, metrics, self.current_params)

            if self.augmentation_enabled:
                # run judger-based evaluation
                self._evaluate_augmentation(train_loader, val_loader) 

        # --------------- 训练结束后保存模型 -----------------
        from pathlib import Path
        save_path = Path(self.metrics_logger.log_dir) / "model.pt"
        torch.save(self.model.state_dict(), save_path)
        print("模型已保存到", save_path) 