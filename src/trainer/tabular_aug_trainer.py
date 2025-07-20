from __future__ import annotations

"""Trainer for tabular regression tasks (e.g. Housing) that integrates the
LLMAugmentationChooser to dynamically select feature-space augmentation
transforms each epoch.

It reuses the optimisation / logging pipeline of `HousingTrainer` but adds:
1. Automatic wrapping of the `HousingDataset` so its `.transform` can be
   updated on-the-fly.
2. A chooser loop at the **end** of every epoch: based on the latest metrics
   decide whether to change augmentation before the next epoch.

The logic deliberately avoids LLM Judger interaction – we directly apply the
choice when `should_use` is True and the augmentation differs from the current
one, matching the user's requirement that this part be independent of Judger.
"""

from typing import Dict, Any, Optional, Sized

from torch.utils.data import DataLoader

from .housing_trainer import HousingTrainer
from ..llm.augmentation_chooser import LLMAugmentationChooser
from ..utils.tabular_aug_ops import get_transform as get_tabular_transform


class TabularAugTrainer(HousingTrainer):
    """Extension of HousingTrainer with LLM-driven augmentation selection."""

    # ----------------------------------------------------------
    def __init__(self, model, config_path):  # type: ignore[override]
        super().__init__(model, config_path)

        aug_cfg: Dict[str, Any] = self.config.get("augmentation", {})
        self.augmentation_enabled: bool = aug_cfg.get("enabled", False)
        self.current_aug_name: str = "none"
        # 限定回归任务可选择的增强集合
        _TABULAR_AUGS = ["gaussian_noise", "feature_dropout", "none"]
        self.chooser = LLMAugmentationChooser(self.config["llm"], allowed_augs=_TABULAR_AUGS)
        self._wrapped_dataset = False

    # ----------------------------------------------------------
    def _wrap_dataset(self, train_loader: DataLoader):
        """Ensure underlying HousingDataset exposes a mutable `transform`."""
        if self._wrapped_dataset:
            return
        # HousingDataset already supports `.transform`; we just remember original
        self.base_ds = train_loader.dataset  # type: ignore[attr-defined]
        self.orig_transform = getattr(self.base_ds, "transform", None)
        if self.orig_transform is None:
            # identity
            self.orig_transform = lambda x: x
        self._wrapped_dataset = True

    def _apply_transform(self, aug_name: str):
        if aug_name == self.current_aug_name:
            return  # no change
        extra_tf = get_tabular_transform(aug_name)

        def _combined(x):
            return extra_tf(self.orig_transform(x))

        self.base_ds.transform = _combined  # type: ignore[attr-defined]
        self.current_aug_name = aug_name

    # ----------------------------------------------------------
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):  # type: ignore[override]
        if self.augmentation_enabled:
            self._wrap_dataset(train_loader)

        best_val_loss = float("inf")
        best_model_state = None  # type: ignore[var-annotated]
        patience = 0
        prev_metrics: Optional[Dict[str, Any]] = None
        prev_aug: str = self.current_aug_name

        for epoch in range(self.config["training"]["epochs"]):
            # ---------------- 训练一个 epoch ----------------
            train_metrics = self._run_one_epoch(train_loader, is_train=True)
            if val_loader:
                val_metrics = self._run_one_epoch(val_loader, is_train=False)
                metrics: Dict[str, Any] = {"train": train_metrics, "val": val_metrics}

                # 提前停止逻辑
                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience = 0
                else:
                    patience += 1
                if patience >= self.config["training"].get("early_stopping_patience", 5):
                    print(f"Early stopping at epoch {epoch}")
                    break
            else:
                metrics = {"train": train_metrics}

            # 记录指标
            self.metrics_logger.update(epoch, metrics, self.current_params)

            # ---------------- Chooser 环节 ----------------
            if self.augmentation_enabled:
                ctx = {
                    "current_augmentation": self.current_aug_name,
                    "epoch": epoch,
                    "dataset_size": len(train_loader.dataset) if isinstance(train_loader.dataset, Sized) else None,
                }
                choice, should_use = self.chooser.choose(metrics, ctx)
                self._log_agent_output("chooser", {"epoch": epoch, "choice": choice, "should_use": should_use})
                if should_use and choice != self.current_aug_name:
                    print(f"[Augmentation] Applying transform '{choice}' for next epoch.")
                    self._apply_transform(choice)

            # --------------- Judger 评估 augmentation 效果 ----------------
            if prev_metrics is not None:
                use_cur, reason, suggestion = self.judger.judge_optimization(
                    baseline_metrics=prev_metrics,
                    optimized_metrics=metrics,
                    baseline_params={"augmentation": prev_aug},
                    optimized_params={"augmentation": self.current_aug_name},
                    optimization_history=None,
                )
                self._log_agent_output("judger", {"epoch": epoch, "use_current": use_cur, "reason": reason, "suggestion": suggestion})

                if suggestion:
                    try:
                        refined = self.prompt_optimizer.refine_suggestion("", suggestion)
                    except Exception:
                        refined = suggestion
                    self._log_agent_output("prompt_optimizer", {"epoch": epoch, "refined_guidance": refined})

            prev_metrics = metrics
            prev_aug = self.current_aug_name

        # 保存最佳模型（若有验证集），否则保存最后一轮
        import torch
        from pathlib import Path
        save_path = Path(self.log_dir) / "model.pt"
        if best_model_state is not None:
            torch.save(best_model_state, save_path)
        else:
            torch.save(self.model.state_dict(), save_path)
        print("模型已保存到", save_path) 