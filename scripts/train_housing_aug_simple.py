#!/usr/bin/env python
"""Housing price regression training script with LLM-driven tabular data augmentation.

与 `train_housing_augmented.py` 相比，本脚本 **完全移除** Judger 与 PromptOptimizer
逻辑——每个 epoch 结束后都会直接采纳 `LLMAugmentationChooser` 给出的增强方法，而不做
额外对比或二次确认。
"""

from __future__ import annotations

import argparse
import datetime
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, random_split

# ----------------------------------------------------------------------------
#  项目根目录放入 sys.path 方便 import
# ----------------------------------------------------------------------------
ROOT_DIR = Path(__file__).parent.parent.resolve()
sys.path.append(str(ROOT_DIR))

# ---- 项目内部依赖 ----
from src.utils.housing_dataset import HousingDataset  # noqa: E402
from src.models.custom_model import CustomModel  # noqa: E402
from src.trainer.tabular_aug_trainer import TabularAugTrainer  # noqa: E402
from src.utils.tabular_aug_ops import get_transform as get_tabular_transform  # noqa: E402


# =============================================================================
#  简化版 Trainer：继承 TabularAugTrainer 但去掉 Judger / PromptOptimizer
# =============================================================================

class SimpleTabularAugTrainer(TabularAugTrainer):
    """TabularAugTrainer without Judger / PromptOptimizer – always adopt augmentation.

    主要增强：
    1. 支持 `augment_prob`：每次仅以给定概率对样本施加增强，避免过强扰动。
    2. 支持 `epochs_per_aug`：同一增强至少持续若干 epoch，再向 LLM 询问下一步，减少震荡。
    3. 默认扩展可选增强为 ['gaussian_noise', 'feature_dropout', 'shift', 'none']。
    """

    def __init__(self, model, config_path):  # type: ignore[override]
        super().__init__(model, config_path)
        aug_cfg = self.config.get("augmentation", {})
        # 新增：增强概率与间隔
        self.augment_prob: float = float(aug_cfg.get("prob", 0.5))
        self.epochs_per_aug: int = int(aug_cfg.get("epochs_per_aug", 3))
        # 扩展 Chooser 可用操作（若尚未包含 shift）
        if "shift" not in self.chooser.available_augs:
            self.chooser.available_augs.append("shift")

    # ------------------------------------------------------------
    def _apply_transform(self, aug_name: str):  # type: ignore[override]
        """Override: 添加概率施加增强的能力。"""
        if aug_name == self.current_aug_name:
            return  # nothing to change

        # 恢复到原始 transform
        if aug_name == "none":
            self.base_ds.transform = self.orig_transform  # type: ignore[attr-defined]
            self.current_aug_name = aug_name
            return

        extra_tf = get_tabular_transform(aug_name)
        prob = self.augment_prob

        import random as _rand

        def _combined(x):
            if _rand.random() < prob:
                x = extra_tf(x)
            return self.orig_transform(x)

        self.base_ds.transform = _combined  # type: ignore[attr-defined]
        self.current_aug_name = aug_name

    # ------------------------------------------------------------
    def train(  # type: ignore[override]
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> None:
        if self.augmentation_enabled:
            self._wrap_dataset(train_loader)

        best_val_loss = float("inf")
        best_state: Optional[Dict[str, torch.Tensor]] = None
        patience = 0

        for epoch in range(self.config["training"]["epochs"]):
            # ---------------- 训练 & 验证 ----------------
            train_metrics = self._run_one_epoch(train_loader, is_train=True)
            if val_loader is not None:
                val_metrics = self._run_one_epoch(val_loader, is_train=False)
                metrics: Dict[str, Any] = {"train": train_metrics, "val": val_metrics}

                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience = 0
                else:
                    patience += 1
                if patience >= self.config["training"].get("early_stopping_patience", 5):
                    print(f"Early stopping at epoch {epoch}")
                    break
            else:
                metrics = {"train": train_metrics}

            # ---------------- 日志 ----------------
            self.metrics_logger.update(epoch, metrics, self.current_params)

            # ---------------- 选择增强 ----------------
            if self.augmentation_enabled and (epoch % self.epochs_per_aug == 0):
                ctx = {
                    "current_augmentation": self.current_aug_name,
                    "epoch": epoch,
                    "dataset_size": len(train_loader.dataset),
                }
                choice, should_use = self.chooser.choose(metrics, ctx)
                self._log_agent_output("chooser", {"epoch": epoch, "choice": choice, "should_use": should_use})
                if should_use and choice != self.current_aug_name:
                    print(f"[Augmentation] Applying transform '{choice}' for next {self.epochs_per_aug} epoch(s).")
                    self._apply_transform(choice)

        # ---------------- 保存模型 ----------------
        save_path = Path(self.log_dir) / "model.pt"
        if best_state is not None:
            torch.save(best_state, save_path)
        else:
            torch.save(self.model.state_dict(), save_path)
        print("模型已保存到", save_path)


# =============================================================================
#  辅助函数
# =============================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_housing(cfg: Dict[str, Any], seed: int) -> Tuple[DataLoader, DataLoader]:
    """加载并拆分房价数据集。"""
    csv_path = ROOT_DIR / cfg["dataset"]["path"]
    dataset = HousingDataset(str(csv_path), target_scale=1e5)

    val_ratio = float(cfg["dataset"].get("validation_split", 0.2))
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size

    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=gen)

    loader_kwargs = {
        "batch_size": cfg["training"]["batch_size"],
        "num_workers": 0,
        "pin_memory": torch.cuda.is_available(),
    }
    return (
        DataLoader(train_ds, shuffle=True, **loader_kwargs),
        DataLoader(val_ds, shuffle=False, **loader_kwargs),
    )


# =============================================================================
#  主入口
# =============================================================================

def main(cfg_path: Path, model_path: Optional[Path] = None):
    # 读取配置
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    seed = cfg["training"].get("seed", 42)
    set_seed(seed)

    # 为本次实验创建独立日志目录
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = (ROOT_DIR / cfg["logging"]["log_dir"]).joinpath(f"housing_aug_simple_{ts}")
    log_dir.mkdir(parents=True, exist_ok=True)
    cfg["logging"]["log_dir"] = str(log_dir)

    # 保存最终使用的配置
    used_cfg_path = log_dir / "used_config.yaml"
    used_cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")

    # 数据
    train_loader, val_loader = load_housing(cfg, seed)

    # 模型
    model = CustomModel(
        input_dim=cfg["dataset"]["input_dim"],
        hidden_dims=cfg["model"]["architecture"]["hidden_dims"],
        output_dim=1,
        dropout=cfg["model"]["architecture"].get("dropout", 0.1),
    )

    if model_path and model_path.exists():
        print("加载预训练模型", model_path)
        model.load_state_dict(torch.load(model_path))

    # Trainer
    trainer = SimpleTabularAugTrainer(model, used_cfg_path)
    print("\n模型结构:\n", model)
    print("\n开始房价预测训练（始终采纳增强建议）…")
    trainer.train(train_loader, val_loader)
    print("训练结束！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Housing model with LLM augmentation (no Judger)")
    parser.add_argument("--config", default="config/housing_config.yaml", help="YAML config path")
    parser.add_argument("--model_path", type=str, default=None, help="Optional pretrained model path (.pt)")
    args = parser.parse_args()

    main(Path(args.config), Path(args.model_path) if args.model_path else None) 