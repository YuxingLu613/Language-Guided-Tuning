from __future__ import annotations

from typing import Dict, Any, Optional, Sized

from torch.utils.data import DataLoader

from .trainer import Trainer
from ..llm.augmentation_chooser import LLMAugmentationChooser
from ..utils.tabular_aug_ops import get_transform as get_tabular_transform


class TabularClassAugTrainer(Trainer):
    """Trainer for tabular *classification* tasks with LLM-driven data augmentation.

    Design mirrors `TabularAugTrainer` (regression version) but extends the
    generic `Trainer` which is oriented to classification losses / metrics.
    """

    def __init__(self, model, config_path):  # type: ignore[override]
        super().__init__(model, config_path)

        # Create independent log directory like other trainers
        from pathlib import Path
        import datetime, os
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        root_log_dir = Path(self.config["logging"]["log_dir"])
        self.log_dir = root_log_dir / f"water_aug_{ts}"
        os.makedirs(self.log_dir, exist_ok=True)
        # Update config so MetricsLogger plots go into correct folder
        self.config["logging"]["log_dir"] = str(self.log_dir)

        aug_cfg: Dict[str, Any] = self.config.get("augmentation", {})
        self.augmentation_enabled: bool = aug_cfg.get("enabled", False)
        self.current_aug_name: str = "none"

        _TABULAR_AUGS = ["gaussian_noise", "feature_dropout", "shift", "none"]
        self.chooser = LLMAugmentationChooser(self.config["llm"], allowed_augs=_TABULAR_AUGS)

        self._wrapped_dataset = False

    # ------------------------------------------------------------------
    def _wrap_dataset(self, train_loader: DataLoader):
        if self._wrapped_dataset:
            return
        self.base_ds = train_loader.dataset  # type: ignore[attr-defined]
        self.orig_transform = getattr(self.base_ds, "transform", None)
        if self.orig_transform is None:
            self.orig_transform = lambda x: x  # identity
        self._wrapped_dataset = True

    def _apply_transform(self, aug_name: str):
        if aug_name == self.current_aug_name:
            return
        extra_tf = get_tabular_transform(aug_name)

        def _combined(x):
            return extra_tf(self.orig_transform(x))

        self.base_ds.transform = _combined  # type: ignore[attr-defined]
        self.current_aug_name = aug_name

    # ------------------------------------------------------------
    def _evaluate_augmentation(self, train_loader: DataLoader, val_loader: Optional[DataLoader]):
        """与 AugTransformTrainer 保持一致的 A/B 测试流程。"""
        import torch  # 延迟导入避免循环
        from typing import Sized

        # ------ 保存当前模型 & 优化器状态 ------ #
        orig_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        orig_opt_state = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in self.optimizer.state_dict().items()}
        baseline_aug = self.current_aug_name

        # ------ 基准指标 (evaluation only) ------ #
        self.train_epoch(train_loader, is_evaluation=True)
        if val_loader is not None:
            baseline_metrics = self.validate(val_loader)
        else:
            baseline_metrics = self.train_epoch(train_loader, is_evaluation=True)

        attempt = 0
        guidance = None
        while attempt < self.judger.max_attempts:
            # 上下文信息供 Chooser
            ctx = {
                "current_augmentation": self.current_aug_name,
                "epoch": None,
                "dataset_size": len(train_loader.dataset) if isinstance(train_loader.dataset, Sized) else None,
                "metrics": baseline_metrics,
            }
            choice, should_use = self.chooser.choose(baseline_metrics, ctx, guidance)
            self._log_agent_output("chooser", {"attempt": attempt, "choice": choice})

            if not should_use or choice == self.current_aug_name:
                break  # 没有新增强可测试

            # 应用候选增强
            prev_aug = self.current_aug_name
            self._apply_transform(choice)

            # 评估候选增强
            self.train_epoch(train_loader, is_evaluation=True)
            if val_loader is not None:
                cand_metrics = self.validate(val_loader)
            else:
                cand_metrics = self.train_epoch(train_loader, is_evaluation=True)

            # Judger 判断
            use_opt, reason, suggestion = self.judger.judge_optimization(  # type: ignore[arg-type]
                baseline_metrics,  # type: ignore[arg-type]
                cand_metrics,      # type: ignore[arg-type]
                {}, {}, None
            )
            self._log_agent_output("judger", {"attempt": attempt, "use_opt": use_opt, "suggestion": suggestion})

            if use_opt:
                baseline_metrics = cand_metrics  # 采纳候选增强成为新基准
                break

            # ---------- 未采纳：回滚 ---------- #
            self._apply_transform(prev_aug)
            self.model.load_state_dict(orig_state)
            self.optimizer.load_state_dict(orig_opt_state)

            if suggestion:
                guidance = self.prompt_optimizer.refine_suggestion(choice, suggestion)
                self._log_agent_output("prompt_optimizer", {"attempt": attempt, "guidance": guidance})
            else:
                break  # 无改进建议，结束循环

            attempt += 1

        # 重置 Judger 的 attempt 计数
        self.judger.reset_attempts()
        return baseline_metrics

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):  # type: ignore[override]
        import torch

        if self.augmentation_enabled:
            self._wrap_dataset(train_loader)

        best_val = float("inf")
        patience = 0

        for epoch in range(self.config["training"]["epochs"]):
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
                    print(f"Early stopping at epoch {epoch}")
                    break
            else:
                metrics = {"train": train_metrics}

            self.metrics_logger.update(epoch, metrics, self.current_params)

            if self.augmentation_enabled:
                # 与图像版保持一致的 A/B 测试
                self._evaluate_augmentation(train_loader, val_loader)

        # 保存最终模型
        from pathlib import Path
        save_path = Path(self.log_dir) / "model.pt"
        torch.save(self.model.state_dict(), save_path)
        print("模型已保存到", save_path) 