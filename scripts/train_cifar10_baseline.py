#!/usr/bin/env python
"""Baseline 训练脚本（CIFAR-10）

本脚本在 **不** 调用任何 LLM 组件（Advisor / Judger）的情况下，
直接使用 `Trainer` 对 CIFAR-10 数据集进行训练。整体流程参考
`scripts/train_baseline.py`，差异主要体现在：
1. 数据集换为 CIFAR-10，自动解压 `data/cifar-10/cifar-10-python.tar.gz`；
2. 默认使用更深的卷积层配置以适应彩色 32×32 图像；
3. 日志目录调整为 `cifar10_log`，避免与其他实验混淆。
"""

from __future__ import annotations

import argparse
import datetime
import random
import sys
import tarfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ---------- 设置随机种子工具 ---------- #

def _set_seed(seed: int) -> None:
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


ROOT_DIR = Path(__file__).parent.parent.resolve()
sys.path.append(str(ROOT_DIR))

from src.models.cnn_model import SimpleCNN  # noqa: E402, after path append
from src.trainer.trainer import Trainer  # noqa: E402

# ----------------------- 数据集帮助函数 ----------------------- #

_CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR_STD = (0.2023, 0.1994, 0.2010)

def _create_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
    ])

    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
    ])
    return train_tf, test_tf


def _maybe_extract_cifar(data_dir: Path) -> None:
    """若未找到已解压的 `cifar-10-batches-py` 目录，则尝试从 tar.gz 解压。"""
    extracted_folder = data_dir / "cifar-10-batches-py"
    if extracted_folder.exists():
        return  # 已解压

    tar_path = data_dir / "cifar-10-python.tar.gz"
    if not tar_path.exists():
        raise FileNotFoundError(
            f"未找到 CIFAR-10 压缩包: {tar_path}. "
            "请确保已将文件置于正确路径，或将 download=True 以自动下载。"
        )
    print(f"正在解压 {tar_path} …")
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(path=data_dir)
    print("解压完成！")


def _load_cifar10(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    train_tf, test_tf = _create_transforms()
    data_dir = ROOT_DIR / config["dataset"]["path"]
    data_dir.mkdir(parents=True, exist_ok=True)

    # 如有需要先解压本地数据
    _maybe_extract_cifar(data_dir)

    use_cuda = torch.cuda.is_available()

    train_ds = datasets.CIFAR10(
        root=str(data_dir), train=True, download=False, transform=train_tf
    )
    test_ds = datasets.CIFAR10(
        root=str(data_dir), train=False, download=False, transform=test_tf
    )

    loader_kwargs: Dict[str, Any] = {
        "batch_size": config["training"]["batch_size"],
        "num_workers": 0,
        "pin_memory": use_cuda,
    }

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
    return train_loader, test_loader


# ----------------------- DummyAdvisor ----------------------- #

class _DummyAdvisor:  # pylint: disable=too-few-public-methods
    """占位 Advisor，始终返回不修改参数。"""

    def get_advice(
        self,
        current_params: Dict[str, Any],
        metrics: Dict[str, Any],
        context_cfg: Dict[str, Any],
        guidance: Optional[str] = None,
    ):
        return "no change", {}, False


# ----------------------------- 主流程 ----------------------------- #

def _main(config_path: Path, model_path: Optional[Path] = None) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # --------------------------------------------------
    # 设置随机种子，保证可比
    # --------------------------------------------------
    seed = config.get("training", {}).get("seed", 42)
    _set_seed(seed)

    # 调整日志目录，存放到 cifar10_log 的子文件夹，防止覆盖原有实验
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_log_dir = "cifar10_log"
    new_log_dir = f"{base_log_dir}/baseline_{timestamp}"
    config["logging"]["log_dir"] = new_log_dir

    # 保存修改后的 config 副本到新的日志目录，便于追溯
    log_dir_path = ROOT_DIR / new_log_dir
    log_dir_path.mkdir(parents=True, exist_ok=True)
    with open(log_dir_path / "used_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True)

    # 加载数据
    train_loader, test_loader = _load_cifar10(config)

    # 创建模型
    model = SimpleCNN(
        input_shape=config["dataset"]["input_shape"],
        conv_layers=config["model"]["architecture"]["conv_layers"],
        pool_size=config["model"]["architecture"]["pool_size"],
        fc_layers=config["model"]["architecture"]["fc_layers"],
        num_classes=config["dataset"]["num_classes"],
        dropout=config["model"]["architecture"].get("dropout", 0.25),
    )

    # 若提供预训练权重则加载
    if model_path and model_path.exists():
        print(f"加载预训练权重: {model_path}")
        model.load_state_dict(torch.load(model_path))

    # 创建 Trainer
    tmp_cfg_path = log_dir_path / "temp_cfg.yaml"
    with open(tmp_cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True)

    trainer = Trainer(model, tmp_cfg_path)

    # 替换 Advisor
    trainer.advisor = _DummyAdvisor()  # type: ignore[assignment]

    # 将 save_interval 设为极大值，且 epoch 0 不执行优化逻辑
    trainer.config["logging"]["save_interval"] = 10 ** 9

    print("\n模型结构:\n", model)
    print("\n开始 CIFAR-10 baseline 训练，日志保存到:", new_log_dir)

    trainer.train(train_loader, test_loader)

    print("训练完成！ 日志位置:", new_log_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR-10 Baseline 训练脚本 (无超参数优化)")
    parser.add_argument(
        "--config",
        type=str,
        default="config/cifar10_config.yaml",
        help="配置文件路径",
    )
    parser.add_argument("--model_path", type=str, default=None, help="预训练模型权重 (.pt) 路径（可选）")
    args = parser.parse_args()

    cfg_path = ROOT_DIR / args.config
    if not cfg_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {cfg_path}")

    model_path: Optional[Path] = Path(args.model_path) if args.model_path else None

    _main(cfg_path, model_path=model_path) 