#!/usr/bin/env python
"""CIFAR-10 数据增强优化训练脚本（使用 AugTransformTrainer）

该脚本在每个 epoch 结束后，允许 LLM 通过 `LLMAugmentationChooser`
动态选择经典图像增强（如 RandomCrop、ColorJitter 等），并利用
`Judger` 评估是否采纳新的增强策略。

与 `train_cifar10_baseline.py` 的区别：
1. 采用 `AugTransformTrainer`，启用了 `augmentation.enabled = true`。
2. 训练集被 `AugmentedDataset` 包装，可动态追加 LLM 生成样本。
3. 基础 transform 仅含 `ToTensor + Normalize`，其余由 LLM 决定。
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import yaml
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 项目根目录
ROOT_DIR = Path(__file__).parent.parent.resolve()
sys.path.append(str(ROOT_DIR))

from src.models.cnn_model import SimpleCNN  # noqa: E402
from src.trainer.augmentation_transform_trainer import AugTransformTrainer  # noqa: E402
from src.utils.augmented_dataset import AugmentedDataset  # noqa: E402


# --------------------- 随机种子 --------------------- #

def set_seed(seed: int = 42):
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------- 数据加载 --------------------- #

_CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR_STD = (0.2023, 0.1994, 0.2010)


def _base_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
    ])
    return train_tf, test_tf


def load_data(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    tr_tf, te_tf = _base_transforms()
    data_dir = ROOT_DIR / cfg["dataset"]["path"]
    data_dir.mkdir(parents=True, exist_ok=True)

    train_ds = datasets.CIFAR10(str(data_dir), train=True, download=True, transform=tr_tf)
    train_ds = AugmentedDataset(train_ds)  # 包装，后续可追加样本
    test_ds = datasets.CIFAR10(str(data_dir), train=False, download=True, transform=te_tf)

    kwargs = {
        "batch_size": cfg["training"]["batch_size"],
        "num_workers": 0,
        "pin_memory": torch.cuda.is_available(),
    }
    return (
        DataLoader(train_ds, shuffle=True, **kwargs),
        DataLoader(test_ds, shuffle=False, **kwargs),
    )


# --------------------- 训练主流程 --------------------- #

def main(cfg_path: Path, model_path: Optional[Path] = None):
    # 读取配置
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("training", {}).get("seed", 42))

    train_loader, test_loader = load_data(cfg)

    # 构建模型
    model = SimpleCNN(
        input_shape=cfg["dataset"]["input_shape"],
        conv_layers=cfg["model"]["architecture"]["conv_layers"],
        pool_size=cfg["model"]["architecture"].get("pool_size", 2),
        fc_layers=cfg["model"]["architecture"]["fc_layers"],
        num_classes=cfg["dataset"]["num_classes"],
        dropout=cfg["model"]["architecture"].get("dropout", 0.25),
    )

    # 加载预训练权重（若提供）
    if model_path and model_path.exists():
        print(f"加载预训练模型权重: {model_path}")
        model.load_state_dict(torch.load(model_path))

    # 创建 AugTransformTrainer
    trainer = AugTransformTrainer(model, cfg_path)

    # 若配置未显式启用 augmentation，则在脚本层面打开
    trainer.config.setdefault("augmentation", {})
    trainer.config["augmentation"]["enabled"] = True

    print("\n开始 CIFAR-10 数据增强优化训练…")
    trainer.train(train_loader, test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR-10 Augmentation-Optimization Training")
    parser.add_argument(
        "--config",
        default="config/cifar10_aug.yaml",
        help="配置文件路径 (YAML)",
    )
    parser.add_argument("--model_path", type=str, default=None, help="预训练模型权重 (.pt) 路径（可选）")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    model_path: Optional[Path] = Path(args.model_path) if args.model_path else None

    main(cfg_path, model_path=model_path) 