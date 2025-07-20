# 在日志目录中加入时间戳子目录所需
from pathlib import Path
import sys
import datetime
from typing import Tuple, Dict, Any, Optional

# 添加项目根目录到 Python 路径
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import yaml

# set_seed util

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from src.utils.housing_dataset import HousingDataset
from src.models.custom_model import CustomModel
from src.trainer.housing_trainer import HousingTrainer


# ----------------------- 数据集帮助函数 ----------------------- #

def load_housing(config: Dict[str, Any], seed: int) -> Tuple[DataLoader, DataLoader]:
    """加载房价数据集，按配置划分训练/验证集。"""
    csv_path = ROOT_DIR / config["dataset"]["path"]
    dataset = HousingDataset(str(csv_path), target_scale=1e5)

    val_split = float(config["dataset"].get("validation_split", 0.2))
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    g = torch.Generator()
    g.manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=g)

    loader_kw = {
        "batch_size": config["training"]["batch_size"],
        "num_workers": 0,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)
    return train_loader, val_loader


# ------------------------- 学习率调度器 ------------------------- #

def create_scheduler(optimizer: torch.optim.Optimizer, config: Dict[str, Any]):
    if "scheduler" not in config.get("training", {}):
        return None
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    sc_cfg = config["training"]["scheduler"]
    return ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=sc_cfg["patience"],
        factor=sc_cfg["factor"],
        min_lr=sc_cfg["min_lr"],
        verbose=True,  # type: ignore[arg-type]
    )


# ----------------------------- 主流程 ----------------------------- #

def main(config_path: Path, model_path: Optional[Path] = None):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 设定随机种子
    seed = config["training"].get("seed", 42)
    set_seed(seed)

    # 创建带时间戳的日志目录，如 logs/housing_20250719_153012/
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = (ROOT_DIR / config["logging"]["log_dir"]).joinpath(f"housing_{timestamp}")
    log_dir.mkdir(parents=True, exist_ok=True)

    # 更新配置并保存到日志目录
    config["logging"]["log_dir"] = str(log_dir)
    used_config_path = log_dir / "used_config.yaml"
    with open(used_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True)

    # 加载数据
    train_loader, val_loader = load_housing(config, seed)

    # 构建模型
    model = CustomModel(
        input_dim=config["dataset"]["input_dim"],
        hidden_dims=config["model"]["architecture"]["hidden_dims"],
        output_dim=1,
        dropout=config["model"]["architecture"].get("dropout", 0.1),
    )

    # 如指定权重文件则加载
    if model_path and model_path.exists():
        print(f"加载预训练模型权重: {model_path}")
        model.load_state_dict(torch.load(model_path))

    trainer = HousingTrainer(model, used_config_path)

    print("\n模型结构:\n", model)
    print("\n开始训练...")
    trainer.train(train_loader, val_loader)
    print("训练完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Housing 价格预测训练脚本")
    parser.add_argument("--config", type=str, default="config/housing_config.yaml", help="配置文件路径")
    parser.add_argument("--model_path", type=str, default=None, help="预训练模型权重 (.pt) 路径（可选）")
    args = parser.parse_args()

    cfg_path = ROOT_DIR / args.config
    model_path: Optional[Path] = Path(args.model_path) if args.model_path else None
    if not cfg_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {cfg_path}")

    main(cfg_path, model_path=model_path) 