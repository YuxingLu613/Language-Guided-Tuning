from __future__ import annotations

"""Training script for the *AdaptiveTrainer* (dynamic strategy agent).

It automatically detects the task-type via the dataset name in the YAML config
(`MNIST` → classification, `Housing` → regression). Use:

$ python scripts/train_adaptive.py --config config/config.yaml         # MNIST
$ python scripts/train_adaptive.py --config config/housing_config.yaml # Housing
"""

import sys
from pathlib import Path
import argparse
from typing import Optional
import random
from typing import Dict, Any, Tuple

import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# ----------  通用随机种子设置  ---------- #


def set_seed(seed: int = 42):
    """为可复现性设置 Python / NumPy / PyTorch 随机种子"""
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Add repo root to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

from src.trainer.adaptive_trainer import AdaptiveTrainer
from src.models.cnn_model import SimpleCNN
from src.models.custom_model import CustomModel
from src.utils.housing_dataset import HousingDataset
from src.utils.water_quality_dataset import WaterQualityDataset  # Water-quality classification
from src.utils.iris_dataset import IrisDataset  # Iris 数据集
from src.utils.winequality_dataset import WineQualityDataset  # Wine-quality regression

from importlib import import_module
import inspect
import torch.nn as _nn
# ------------------------ CIFAR-10 helpers ------------------------ #

_CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR_STD = (0.2023, 0.1994, 0.2010)


def _cifar_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    t_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
    ])

    t_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
    ])
    return t_train, t_test


def _load_cifar10(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    t_train, t_test = _cifar_transforms()
    data_dir = ROOT_DIR / cfg["dataset"]["path"]
    data_dir.mkdir(parents=True, exist_ok=True)

    train_ds = datasets.CIFAR10(str(data_dir), train=True, download=True, transform=t_train)
    test_ds = datasets.CIFAR10(str(data_dir), train=False, download=True, transform=t_test)

    kw = {
        "batch_size": cfg["training"]["batch_size"],
        "num_workers": 0,
        "pin_memory": torch.cuda.is_available(),
    }
    return (
        DataLoader(train_ds, shuffle=True, **kw),
        DataLoader(test_ds, shuffle=False, **kw),
    )


# ---------------------------------------------------------------------
# MNIST helpers
# ---------------------------------------------------------------------

def _create_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    t_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    t_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    return t_train, t_test


def _load_mnist(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    t_train, t_test = _create_transforms()
    data_dir = ROOT_DIR / cfg["dataset"]["path"]
    data_dir.mkdir(parents=True, exist_ok=True)

    train_ds = datasets.MNIST(str(data_dir), train=True, download=True, transform=t_train)
    test_ds = datasets.MNIST(str(data_dir), train=False, transform=t_test)

    loader_kw = {
        "batch_size": cfg["training"]["batch_size"],
        "num_workers": 0,
        "pin_memory": torch.cuda.is_available(),
    }
    return (
        DataLoader(train_ds, shuffle=True, **loader_kw),
        DataLoader(test_ds, shuffle=False, **loader_kw),
    )


# ---------------------------------------------------------------------
# Housing helpers
# ---------------------------------------------------------------------

def _load_housing(cfg: Dict[str, Any], seed: int) -> Tuple[DataLoader, DataLoader]:
    csv_path = ROOT_DIR / cfg["dataset"]["path"]
    ds = HousingDataset(str(csv_path), target_scale=1e5)

    val_split = float(cfg["dataset"].get("validation_split", 0.2))
    v_size = int(len(ds) * val_split)
    t_size = len(ds) - v_size

    g = torch.Generator()
    g.manual_seed(seed)
    train_ds, val_ds = random_split(ds, [t_size, v_size], generator=g)

    kw = {"batch_size": cfg["training"]["batch_size"], "num_workers": 0, "pin_memory": torch.cuda.is_available()}
    return (
        DataLoader(train_ds, shuffle=True, **kw),
        DataLoader(val_ds, shuffle=False, **kw),
    )


# ---------------------------------------------------------------------
# Water Quality helpers
# ---------------------------------------------------------------------

def _load_water_quality(cfg: Dict[str, Any], seed: int) -> Tuple[DataLoader, DataLoader]:
    csv_path = ROOT_DIR / cfg["dataset"]["path"]
    ds = WaterQualityDataset(str(csv_path))

    val_split = float(cfg["dataset"].get("validation_split", 0.2))
    v_size = int(len(ds) * val_split)
    t_size = len(ds) - v_size

    g = torch.Generator()
    g.manual_seed(seed)
    train_ds, val_ds = random_split(ds, [t_size, v_size], generator=g)

    kw = {
        "batch_size": cfg["training"]["batch_size"],
        "num_workers": 0,
        "pin_memory": torch.cuda.is_available(),
    }
    return (
        DataLoader(train_ds, shuffle=True, **kw),
        DataLoader(val_ds, shuffle=False, **kw),
    )

# ---------------------------------------------------------------------
# Wine Quality helpers
# ---------------------------------------------------------------------


def _load_winequality(cfg: Dict[str, Any], seed: int) -> Tuple[DataLoader, DataLoader]:
    csv_path = ROOT_DIR / cfg["dataset"]["path"]
    scale = float(cfg["dataset"].get("target_scale", 10.0))
    ds = WineQualityDataset(str(csv_path), target_scale=scale)

    val_split = float(cfg["dataset"].get("validation_split", 0.2))
    v_size = int(len(ds) * val_split)
    t_size = len(ds) - v_size

    g = torch.Generator(); g.manual_seed(seed)
    train_ds, val_ds = random_split(ds, [t_size, v_size], generator=g)

    kw = {
        "batch_size": cfg["training"]["batch_size"],
        "num_workers": 0,
        "pin_memory": torch.cuda.is_available(),
    }
    return (
        DataLoader(train_ds, shuffle=True, **kw),
        DataLoader(val_ds, shuffle=False, **kw),
    )

# ---------------------------------------------------------------------
# Iris helpers
# ---------------------------------------------------------------------

def _load_iris(cfg: Dict[str, Any], seed: int) -> Tuple[DataLoader, DataLoader]:
    ds = IrisDataset()

    val_split = float(cfg["dataset"].get("validation_split", 0.2))
    v_size = int(len(ds) * val_split)
    t_size = len(ds) - v_size

    g = torch.Generator()
    g.manual_seed(seed)
    train_ds, val_ds = random_split(ds, [t_size, v_size], generator=g)

    kw = {
        "batch_size": cfg["training"]["batch_size"],
        "num_workers": 0,
        "pin_memory": torch.cuda.is_available(),
    }
    return (
        DataLoader(train_ds, shuffle=True, **kw),
        DataLoader(val_ds, shuffle=False, **kw),
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main(cfg_path: Path, model_path: Optional[Path] = None):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("training", {}).get("seed", 42)
    set_seed(seed)

    ds_name = cfg["dataset"]["name"].lower()

    if ds_name == "mnist":
        task_type = "classification"
        train_loader, val_loader = _load_mnist(cfg)
        model = SimpleCNN(
            input_shape=cfg["dataset"]["input_shape"],
            conv_layers=cfg["model"]["architecture"]["conv_layers"],
            pool_size=cfg["model"]["architecture"]["pool_size"],
            fc_layers=cfg["model"]["architecture"]["fc_layers"],
            num_classes=cfg["dataset"]["num_classes"],
            dropout=cfg["model"]["architecture"].get("dropout", 0.25),
        )
    elif ds_name == "housing":
        task_type = "regression"
        train_loader, val_loader = _load_housing(cfg, seed)
        model = CustomModel(
            input_dim=cfg["dataset"]["input_dim"],
            hidden_dims=cfg["model"]["architecture"]["hidden_dims"],
            output_dim=1,
            dropout=cfg["model"]["architecture"].get("dropout", 0.1),
        )
    elif ds_name in {"waterquality", "water_quality"}:
        task_type = "classification"
        train_loader, val_loader = _load_water_quality(cfg, seed)
        # 动态加载模型，兼容优化后模型文件
        model_name = cfg["model"]["name"]
        try:
            mod = import_module(f"src.models.{model_name.lower()}")
        except ModuleNotFoundError:
            from src.models import custom_model as mod  # type: ignore

        model_cls = None
        for _n, _o in inspect.getmembers(mod, inspect.isclass):
            if issubclass(_o, _nn.Module):
                model_cls = _o
                break
        if model_cls is None:
            raise RuntimeError(f"未在 {mod} 找到 nn.Module 子类")

        try:
            model = model_cls(
                input_dim=cfg["dataset"].get("input_dim", None),
                hidden_dims=cfg["model"]["architecture"].get("hidden_dims", None),
                output_dim=cfg["dataset"].get("num_classes", 2),
                dropout=cfg["model"]["architecture"].get("dropout", 0.5),
            )
        except TypeError:
            model = model_cls()  # type: ignore
    elif ds_name == "cifar10":
        task_type = "classification"
        train_loader, val_loader = _load_cifar10(cfg)
        model = SimpleCNN(
            input_shape=cfg["dataset"]["input_shape"],
            conv_layers=cfg["model"]["architecture"]["conv_layers"],
            pool_size=cfg["model"]["architecture"]["pool_size"],
            fc_layers=cfg["model"]["architecture"]["fc_layers"],
            num_classes=cfg["dataset"]["num_classes"],
            dropout=cfg["model"]["architecture"].get("dropout", 0.25),
        )
    elif ds_name == "iris":
        task_type = "classification"
        train_loader, val_loader = _load_iris(cfg, seed)

        model_name = cfg["model"]["name"]
        try:
            mod = import_module(f"src.models.{model_name.lower()}")
        except ModuleNotFoundError:
            from src.models import custom_model as mod  # type: ignore

        model_cls = None
        for _n, _o in inspect.getmembers(mod, inspect.isclass):
            if issubclass(_o, _nn.Module):
                model_cls = _o
                break
        if model_cls is None:
            raise RuntimeError(f"未在 {mod} 找到 nn.Module 子类")

        try:
            model = model_cls(
                input_dim=cfg["dataset"].get("input_dim", None),
                hidden_dims=cfg["model"]["architecture"].get("hidden_dims", None),
                output_dim=cfg["dataset"].get("num_classes", 3),
                dropout=cfg["model"].get("architecture", {}).get("dropout", 0.3),
            )
        except TypeError:
            model = model_cls()  # type: ignore
    elif ds_name in {"winequality", "wine_quality"}:
        task_type = "regression"
        train_loader, val_loader = _load_winequality(cfg, seed)

        # 动态加载模型，根据 config.model.name
        model_name = cfg["model"]["name"]
        try:
            mod = import_module(f"src.models.{model_name.lower()}")
        except ModuleNotFoundError:
            from src.models import custom_model as mod  # type: ignore

        model_cls = None
        for _n, _o in inspect.getmembers(mod, inspect.isclass):
            if issubclass(_o, _nn.Module):
                model_cls = _o
                break
        if model_cls is None:
            raise RuntimeError(f"未在 {mod} 找到 nn.Module 子类")

        # 尝试标准构造签名
        try:
            model = model_cls(
                input_dim=cfg["dataset"].get("input_dim"),
                hidden_dims=cfg["model"]["architecture"].get("hidden_dims"),
                output_dim=1,
                dropout=cfg["model"]["architecture"].get("dropout", 0.1),
            )
        except TypeError:
            model = model_cls()  # type: ignore
    else:
        raise ValueError(f"Unsupported dataset name: {ds_name}")

    # 如果指定了权重文件则加载
    if model_path and model_path.exists():
        print(f"加载预训练权重: {model_path}")
        model.load_state_dict(torch.load(model_path))

    trainer = AdaptiveTrainer(model, task_type, cfg_path)

    print("\nModel:\n", model)
    print("\nStarting training with AdaptiveTrainer...")
    trainer.train(train_loader, val_loader)
    print("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Adaptive trainer script")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--model_path", type=str, default=None, help="预训练模型权重 (.pt) 路径（可选）")
    args = parser.parse_args()

    config_path = ROOT_DIR / args.config
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    model_path: Optional[Path] = Path(args.model_path) if args.model_path else None

    main(config_path, model_path=model_path) 