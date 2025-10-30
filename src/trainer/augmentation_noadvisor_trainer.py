from __future__ import annotations

from typing import Optional, Dict, Any, List
from pathlib import Path

from torch.utils.data import DataLoader

from .augmentation_trainer import AugmentationTrainer


class AugmentOnlyTrainer(AugmentationTrainer):
    """Trainer variant that *omits* the hyper-parameter Advisor.

    Workflow per epoch:
    1. Train as usual.
    2. If enabled, generate K synthetic samples via LLMDataAugmentor and append.
    3. Use LLMJudger to compare *previous* epoch metrics (baseline) with the
       current metrics (after new data). The judgement + suggestion is logged
       but **no** automatic rollback is performed – the goal is observational.
    """

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> None:  # type: ignore[override]
        # Ensure dataset wrapper & dedicated log dir (helpers from parent)
        if self.augmentation_enabled:
            self._ensure_dataset_wrapped(train_loader)
            self._ensure_exclusive_logdir()

        best_val_loss = float("inf")
        patience_counter = 0
        prev_metrics: Optional[Dict[str, Any]] = None

        for epoch in range(self.config["training"]["epochs"]):
            # ---------------- Standard training -----------------
            train_metrics = self.train_epoch(train_loader, epoch)

            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                metrics = {"train": train_metrics, "val": val_metrics}

                # early stopping bookkeeping (same as parent)
                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= self.config["training"]["early_stopping_patience"]:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break
            else:
                metrics = {"train": train_metrics}

            # ---- Log metrics
            self.metrics_logger.update(epoch, metrics, self.current_params)

            # ---- Judger: compare with previous metrics (if any)
            if prev_metrics is not None:
                use_cur, reason, suggestion = self.judger.judge_optimization(
                    baseline_metrics=prev_metrics,
                    optimized_metrics=metrics,
                    baseline_params=self.current_params,
                    optimized_params=self.current_params,
                    optimization_history=None,
                )
                self._log_agent_output(
                    "judger",
                    {
                        "epoch": epoch,
                        "prev_vs_current": use_cur,
                        "reason": reason,
                        "suggestion": suggestion,
                    },
                )

            prev_metrics = metrics  # cache for next epoch comparison

            # ---- LLM data augmentation
            if self.augmentation_enabled and self.samples_per_epoch > 0:
                raw_reply, new_samples = self.augmentor.generate(
                    dataset_name=self.config["dataset"]["name"],
                    metrics=metrics,
                    num_samples=self.samples_per_epoch,
                    context_cfg=self.config,
                )
                self._log_agent_output(
                    "augmentor",
                    {"epoch": epoch, "reply": raw_reply, "n_samples": len(new_samples)},
                )

                if new_samples:
                    processed = self._process_samples(new_samples)
                    if processed:
                        train_loader.dataset.extend(processed)  # type: ignore[attr-defined]
                        print(f"[Augmentation] Added {len(processed)} synthetic samples.")
                    else:
                        print("[Augmentation] No valid samples after processing – skipped.")

        # end for epoch 