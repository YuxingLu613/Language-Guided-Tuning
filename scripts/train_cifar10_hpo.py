#!/usr/bin/env python
"""CIFAR-10 HPO 训练脚本

使用 `Trainer`（保留 LLM Advisor / Judger / PromptOptimizer）在 CIFAR-10
上进行超参数优化实验。流程仿照 `train_water_quality_hpo.py`。
"""

from __future__ import annotations

import argparse
import datetime
import random
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import yaml
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# 项目根目录
ROOT_DIR = Path(__file__).parent.parent.resolve()
sys.path.append(str(ROOT_DIR))

from src.models.cnn_model import SimpleCNN  # noqa: E402
from src.trainer.trainer import Trainer  # noqa: E402


# ---------------- utils ---------------- #

def set_seed(seed: int):
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


_CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR_STD = (0.2023, 0.1994, 0.2010)


def _create_transforms(cfg: Dict[str, Any]) -> Tuple[transforms.Compose, transforms.Compose]:
    # 按需添加基本增强（与 baseline 保持一致）
    train_tf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
    ]
    # 测试集不含随机增强
    test_tf = [
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
    ]
    return transforms.Compose(train_tf), transforms.Compose(test_tf)


def load_dataset(cfg: Dict[str, Any], seed: int) -> Tuple[DataLoader, DataLoader]:
    tr_tf, te_tf = _create_transforms(cfg)
    data_dir = ROOT_DIR / cfg["dataset"]["path"]
    data_dir.mkdir(parents=True, exist_ok=True)

    full_train_ds = datasets.CIFAR10(str(data_dir), train=True, download=True, transform=tr_tf)

    # 使用 validation_split
    val_ratio = float(cfg["training"].get("validation_split", 0.1))
    v_size = int(len(full_train_ds) * val_ratio)
    t_size = len(full_train_ds) - v_size

    g = torch.Generator(); g.manual_seed(seed)
    train_ds, val_ds = random_split(full_train_ds, [t_size, v_size], generator=g)

    test_ds = datasets.CIFAR10(str(data_dir), train=False, download=True, transform=te_tf)

    kw = {
        "batch_size": cfg["training"]["batch_size"],
        "num_workers": 0,
        "pin_memory": torch.cuda.is_available(),
    }
    return (
        DataLoader(train_ds, shuffle=True, **kw),
        DataLoader(val_ds, shuffle=False, **kw),
    )


# ---------------- main ---------------- #

def main(cfg_path: Path, model_path: Optional[Path] = None):
    cfg: Dict[str, Any] = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    # 创建独立日志目录，防止覆盖
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg["logging"]["log_dir"] += f"_cifar10_hpo_{ts}"
    log_dir = ROOT_DIR / cfg["logging"]["log_dir"]
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "used_config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")

    seed = cfg["training"].get("seed", 42)
    set_seed(seed)

    train_loader, val_loader = load_dataset(cfg, seed)

    # 构建模型
    model = SimpleCNN(
        input_shape=cfg["dataset"]["input_shape"],
        conv_layers=cfg["model"]["architecture"]["conv_layers"],
        pool_size=cfg["model"]["architecture"].get("pool_size", 2),
        fc_layers=cfg["model"]["architecture"]["fc_layers"],
        num_classes=cfg["dataset"]["num_classes"],
        dropout=cfg["model"]["architecture"].get("dropout", 0.25),
    )

    if model_path and model_path.exists():
        print(f"加载预训练权重: {model_path}")
        model.load_state_dict(torch.load(model_path))

    # 将 (可能修改过的) cfg 写入临时文件，供 Trainer 读取
    tmp_cfg = log_dir / "temp_cfg.yaml"
    tmp_cfg.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")

    trainer = Trainer(model, tmp_cfg)
    print("开始 CIFAR-10 HPO 训练，日志目录:", cfg["logging"]["log_dir"])
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR-10 HPO Training (LLM Advisor)")
    parser.add_argument("--config", default="config/cifar10_config.yaml", help="配置文件路径")
    parser.add_argument("--model_path", default=None, help="预训练模型权重 (.pt) 路径（可选）")
    args = parser.parse_args()

    main(Path(args.config), Path(args.model_path) if args.model_path else None) 