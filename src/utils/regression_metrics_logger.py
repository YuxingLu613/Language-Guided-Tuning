import json
from pathlib import Path
from typing import Dict, Any, Union
import matplotlib.pyplot as plt
from datetime import datetime

class RegressionMetricsLogger:
    """专用于回归任务 (MSE/MAE) 的指标记录器。"""

    def __init__(self, log_dir: Union[str, Path], config: Dict[str, Any]):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 记录结构
        self.metrics_history = {
            "train_loss": [],
            "val_loss": [],
            "train_mse": [],
            "val_mse": [],
            "train_mae": [],
            "val_mae": [],
            "train_r2": [],
            "val_r2": [],
            "learning_rate": [],
        }
        self.config = config
        # 存储 LLM 建议
        self.llm_suggestions = []

    def update(self, epoch: int, metrics: Dict[str, Any], params: Dict[str, float]):
        """更新历史并保存/绘图"""
        self.metrics_history["train_loss"].append(metrics["train"]["loss"])
        self.metrics_history["train_mse"].append(metrics["train"].get("mse", 0))
        self.metrics_history["train_mae"].append(metrics["train"].get("mae", 0))
        self.metrics_history["train_r2"].append(metrics["train"].get("r2", 0))

        if "val" in metrics:
            self.metrics_history["val_loss"].append(metrics["val"]["loss"])
            self.metrics_history["val_mse"].append(metrics["val"].get("mse", 0))
            self.metrics_history["val_mae"].append(metrics["val"].get("mae", 0))
            self.metrics_history["val_r2"].append(metrics["val"].get("r2", 0))

        self.metrics_history["learning_rate"].append(params["learning_rate"])

        # 保存 json
        with open(self.log_dir / "metrics_history.json", "w", encoding="utf-8") as f:
            json.dump(self.metrics_history, f, ensure_ascii=False, indent=2)

        # 绘图
        if self.config["logging"].get("plot_metrics", True):
            self._plot_metrics(epoch)

    # --------------------------------------------------------------
    # LLM suggestions
    # --------------------------------------------------------------

    def add_llm_suggestion(self, epoch: int, suggestion: str):
        """记录来自 StrategyAgent / Advisor 的文本建议。"""
        self.llm_suggestions.append({"epoch": epoch, "suggestion": suggestion})

        suggestions_file = self.log_dir / "llm_suggestions.json"
        with open(suggestions_file, "w", encoding="utf-8") as f:
            json.dump(self.llm_suggestions, f, ensure_ascii=False, indent=2)

    def _plot_metrics(self, current_epoch: int):
        try:
            import seaborn as sns  # noqa: F401
            plt.style.use("seaborn")
        except (ImportError, OSError):
            plt.style.use("default")

        epochs = list(range(current_epoch + 1))
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        # Loss
        axs[0, 0].plot(epochs, self.metrics_history["train_loss"], label="train_loss")
        if self.metrics_history["val_loss"]:
            axs[0, 0].plot(epochs, self.metrics_history["val_loss"], label="val_loss")
        axs[0, 0].set_title("Loss")
        axs[0, 0].legend(); axs[0, 0].grid(True)

        # MSE
        axs[0, 1].plot(epochs, self.metrics_history["train_mse"], label="train_mse")
        if self.metrics_history["val_mse"]:
            axs[0, 1].plot(epochs, self.metrics_history["val_mse"], label="val_mse")
        axs[0, 1].set_title("MSE")
        axs[0, 1].legend(); axs[0, 1].grid(True)

        # MAE
        axs[1, 0].plot(epochs, self.metrics_history["train_mae"], label="train_mae")
        if self.metrics_history["val_mae"]:
            axs[1, 0].plot(epochs, self.metrics_history["val_mae"], label="val_mae")
        axs[1, 0].set_title("MAE")
        axs[1, 0].legend(); axs[1, 0].grid(True)

        # LR
        axs[1, 1].plot(epochs, self.metrics_history["learning_rate"], label="lr")
        axs[1, 1].set_title("Learning Rate")
        axs[1, 1].legend(); axs[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(self.log_dir / f"metrics_epoch_{current_epoch}.png", dpi=300)
        plt.close() 