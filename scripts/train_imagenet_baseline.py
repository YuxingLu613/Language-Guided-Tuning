from pathlib import Path
import sys
import datetime
import random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import yaml
import argparse
from typing import Tuple, Dict, Any, Optional

# 设置随机种子
def set_seed(seed: int):
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 根目录
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

# ---------- ImageNet transforms ---------- #
def create_imagenet_transforms():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, test_transform

# ---------- ImageNet DataLoader ---------- #
def load_imagenet(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    train_transform, test_transform = create_imagenet_transforms()
    data_dir = Path(config["dataset"]["path"])
    use_cuda = torch.cuda.is_available()
    loader_kwargs = {
        "batch_size": config["training"]["batch_size"],
        "num_workers": 4,
        "pin_memory": use_cuda,
    }
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'val'
    train_ds = datasets.ImageFolder(str(train_dir), transform=train_transform)
    val_ds = datasets.ImageFolder(str(val_dir), transform=test_transform)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader

# ---------- Trainer 引用（你的Trainer/可复用基类）---------- #
from src.trainer.trainer import Trainer

# ----------------------------- 主流程 ----------------------------- #
def main(config_path: Path, model_path: Optional[Path] = None):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    seed = config.get('training', {}).get('seed', 42)
    set_seed(seed)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_log_dir = "imagenet_log"
    new_log_dir = f"{base_log_dir}/baseline_{timestamp}"
    config["logging"]["log_dir"] = new_log_dir

    log_dir_path = ROOT_DIR / new_log_dir
    log_dir_path.mkdir(parents=True, exist_ok=True)
    with open(log_dir_path / "used_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True)

    # 数据加载
    train_loader, val_loader = load_imagenet(config)

    # 模型：ResNet18（ImageNet 任务，默认1000类）
    model = models.resnet18(pretrained=config["model"].get("pretrained", True))
    model.fc = torch.nn.Linear(model.fc.in_features, config["dataset"]["num_classes"])

    # 加载权重支持
    if model_path and model_path.exists():
        print(f"加载预训练权重: {model_path}")
        model.load_state_dict(torch.load(model_path))

    # Trainer
    tmp_cfg_path = log_dir_path / "temp_cfg.yaml"
    with open(tmp_cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True)

    trainer = Trainer(model, tmp_cfg_path)
    print("\n模型结构:\n", model)
    print("\n开始 ImageNet baseline 训练，日志保存到:", new_log_dir)
    trainer.train(train_loader, val_loader)
    print("训练完成！ 日志位置:", new_log_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ImageNet Baseline 训练脚本")
    parser.add_argument(
        "--config", type=str, default="config/imagenet_config.yaml", help="配置文件路径"
    )
    parser.add_argument("--model_path", type=str, default=None, help="预训练模型权重 (.pt) 路径（可选）")
    args = parser.parse_args()

    cfg_path = ROOT_DIR / args.config
    if not cfg_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {cfg_path}")

    model_path: Optional[Path] = Path(args.model_path) if args.model_path else None

    main(cfg_path, model_path=model_path)
