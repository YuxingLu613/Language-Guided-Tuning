from pathlib import Path
import sys
import datetime
# 随机种子
import random

# ---------- 设置随机种子工具 ---------- #

def set_seed(seed: int):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 添加项目根目录到Python路径
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import yaml
import argparse
from typing import Tuple, Dict, Any, Optional

from src.models.cnn_model import SimpleCNN
from src.trainer.trainer import Trainer

# ----------------------- 数据集帮助函数 ----------------------- #

def create_transforms(config: Dict[str, Any]) -> Tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    return train_transform, test_transform


def load_mnist(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    train_tf, test_tf = create_transforms(config)
    data_dir = ROOT_DIR / config["dataset"]["path"]
    data_dir.mkdir(parents=True, exist_ok=True)

    use_cuda = torch.cuda.is_available()

    train_ds = datasets.MNIST(str(data_dir), train=True, download=True, transform=train_tf)
    test_ds = datasets.MNIST(str(data_dir), train=False, transform=test_tf)

    loader_kwargs = {
        "batch_size": config["training"]["batch_size"],
        "num_workers": 0,
        "pin_memory": use_cuda,
    }

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
    return train_loader, test_loader

# ----------------------- DummyAdvisor ----------------------- #

class DummyAdvisor:
    """占位 Advisor，始终返回不修改参数。"""

    def get_advice(self, current_params: Dict[str, Any], metrics: Dict[str, Any], context_cfg: Dict[str, Any], guidance: Optional[str] = None):
        return "no change", {}, False

# ----------------------------- 主流程 ----------------------------- #

def main(config_path: Path, model_path: Optional[Path] = None):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # --------------------------------------------------
    # 设置随机种子，保证可比
    # --------------------------------------------------
    seed = config.get('training', {}).get('seed', 42)
    set_seed(seed)

    # 调整日志目录，存放到 minist_log 的子文件夹，防止覆盖原有实验
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # 基础日志目录设置为 'minist_log'
    base_log_dir = "minist_log"
    # 最终的日志子目录，例如 'minist_log/baseline_20250718_123456'
    new_log_dir = f"{base_log_dir}/baseline_{timestamp}"
    config["logging"]["log_dir"] = new_log_dir

    # 保存修改后的 config 副本到新的日志目录，便于追溯
    log_dir_path = ROOT_DIR / new_log_dir
    log_dir_path.mkdir(parents=True, exist_ok=True)
    with open(log_dir_path / "used_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True)

    # 加载数据
    train_loader, test_loader = load_mnist(config)

    # 创建模型
    model = SimpleCNN(
        input_shape=config["dataset"]["input_shape"],
        conv_layers=config["model"]["architecture"]["conv_layers"],
        pool_size=config["model"]["architecture"]["pool_size"],
        fc_layers=config["model"]["architecture"]["fc_layers"],
        num_classes=config["dataset"]["num_classes"],
        dropout=config["model"]["architecture"].get("dropout", 0.25),
    )

    # 如果提供了预训练权重则加载
    if model_path and model_path.exists():
        print(f"加载预训练权重: {model_path}")
        model.load_state_dict(torch.load(model_path))

    # 创建 Trainer
    # 先将 config 写回临时文件，因为 Trainer 接受路径
    tmp_cfg_path = log_dir_path / "temp_cfg.yaml"
    with open(tmp_cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True)

    trainer = Trainer(model, tmp_cfg_path)

    # 用 DummyAdvisor 替换真实 Advisor，避免调用大模型
    trainer.advisor = DummyAdvisor()  # type: ignore[assignment]

    # 将 save_interval 设为极大值，且 epoch 0 不执行优化逻辑
    trainer.config["logging"]["save_interval"] = 10 ** 9

    print("\n模型结构:\n", model)
    print("\n开始 baseline 训练，日志保存到:", new_log_dir)

    trainer.train(train_loader, test_loader)

    print("训练完成！ 日志位置:", new_log_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline 训练脚本 (无超参数优化)")
    parser.add_argument(
        "--config", type=str, default="config/config.yaml", help="配置文件路径"
    )
    parser.add_argument("--model_path", type=str, default=None, help="预训练模型权重 (.pt) 路径（可选）")
    args = parser.parse_args()

    cfg_path = ROOT_DIR / args.config
    if not cfg_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {cfg_path}")

    model_path: Optional[Path] = Path(args.model_path) if args.model_path else None

    main(cfg_path, model_path=model_path) 