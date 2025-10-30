from __future__ import annotations

"""Train MNIST with LLM data augmentation *without* Advisor.

Run:  python scripts/train_augmented_noadvisor.py --config config/mnist_aug.yaml
"""

from pathlib import Path
import sys, random, argparse, yaml
from typing import Optional

ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets, transforms

from src.models.cnn_model import SimpleCNN
from src.trainer.augmentation_transform_trainer import AugTransformTrainer
from src.utils.augmented_dataset import AugmentedDataset


def set_seed(seed: int):
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_transforms(cfg):
    tr = [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    if "random_rotation" in cfg["dataset"].get("train_transform", []):
        tr.insert(1, transforms.RandomRotation(15))
    return transforms.Compose(tr), transforms.Compose(tr)


def load_data(cfg):
    tr_tf, te_tf = create_transforms(cfg)
    ddir = ROOT_DIR / cfg["dataset"]["path"]
    ddir.mkdir(parents=True, exist_ok=True)

    train_ds = datasets.MNIST(str(ddir), train=True, download=True, transform=tr_tf)
    train_ds = AugmentedDataset(train_ds)
    test_ds = datasets.MNIST(str(ddir), train=False, transform=te_tf)

    kwargs = dict(batch_size=cfg["training"]["batch_size"], num_workers=0, pin_memory=torch.cuda.is_available())
    return DataLoader(train_ds, shuffle=True, **kwargs), DataLoader(test_ds, shuffle=False, **kwargs)  # type: ignore[arg-type]


def create_scheduler(opt, cfg):
    if "scheduler" not in cfg["training"]:
        return None
    s = cfg["training"]["scheduler"]
    return ReduceLROnPlateau(opt, mode="min", patience=s["patience"], factor=s["factor"], min_lr=s["min_lr"], verbose=True)  # type: ignore[arg-type]


def main(cfg_path: Path, model_path: Optional[Path] = None):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg.get("training", {}).get("seed", 42))

    train_loader, test_loader = load_data(cfg)

    model = SimpleCNN(
        input_shape=cfg["dataset"]["input_shape"],
        conv_layers=cfg["model"]["architecture"]["conv_layers"],
        pool_size=cfg["model"]["architecture"].get("pool_size", 2),
        fc_layers=cfg["model"]["architecture"]["fc_layers"],
        num_classes=cfg["dataset"]["num_classes"],
        dropout=cfg["model"]["architecture"].get("dropout", 0.25),
    )

    # 如果指定了预训练模型则加载
    if model_path and model_path.exists():
        print(f"加载预训练权重: {model_path}")
        model.load_state_dict(torch.load(model_path))

    trainer = AugTransformTrainer(model, cfg_path)
    sched = create_scheduler(trainer.optimizer, cfg)
    if sched:  # type: ignore[arg-type]
        trainer.scheduler = sched  # type: ignore[assignment]

    print("Start training without Advisor …")
    trainer.train(train_loader, test_loader)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/mnist_aug.yaml")
    ap.add_argument("--model_path", type=str, default=None, help="预训练模型权重 (.pt) 路径（可选）")
    args = ap.parse_args()

    model_path: Optional[Path] = Path(args.model_path) if args.model_path else None

    main(Path(args.config), model_path=model_path) 