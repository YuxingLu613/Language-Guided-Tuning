from pathlib import Path
import sys
import datetime

ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

import argparse
import random
from typing import Dict, Any, Tuple, Optional

import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader, random_split

from src.utils.housing_dataset import HousingDataset
from src.models.custom_model import CustomModel
from src.trainer.housing_trainer import HousingTrainer

# ----------------------- 数据集帮助函数 ----------------------- #

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_housing(config: Dict[str, Any], seed: int) -> Tuple[DataLoader, DataLoader]:
    csv_path = ROOT_DIR / config["dataset"]["path"]
    dataset = HousingDataset(str(csv_path), target_scale=1e5)

    val_ratio = float(config["dataset"].get("validation_split", 0.2))
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    g = torch.Generator()
    g.manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=g)

    kwargs = {
        "batch_size": config["training"]["batch_size"],
        "num_workers": 0,
        "pin_memory": torch.cuda.is_available(),
    }
    return DataLoader(train_ds, shuffle=True, **kwargs), DataLoader(val_ds, shuffle=False, **kwargs)

# ----------------------- DummyAdvisor ----------------------- #

class DummyAdvisor:
    """占位 Advisor：始终返回 no change。"""

    def get_advice(self, current_params, metrics, context_cfg, guidance=None):
        return "no change", {}, False

# ----------------------------- 主流程 ----------------------------- #

def main(config_path: Path, model_path: Optional[Path] = None):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 调整日志目录，防止覆盖正式实验
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    orig_log_dir = config["logging"]["log_dir"]
    new_log_dir = f"{orig_log_dir}_housing_baseline_{ts}"
    config["logging"]["log_dir"] = new_log_dir

    log_dir_path = ROOT_DIR / new_log_dir
    log_dir_path.mkdir(parents=True, exist_ok=True)

    # 保存使用的 config
    with open(log_dir_path / "used_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True)

    # 设置随机种子
    seed = config["training"].get("seed", 42)
    set_seed(seed)

    # 数据
    train_loader, val_loader = load_housing(config, seed)

    # 模型
    model = CustomModel(
        input_dim=config["dataset"]["input_dim"],
        hidden_dims=config["model"]["architecture"]["hidden_dims"],
        output_dim=1,
        dropout=config["model"]["architecture"].get("dropout", 0.1),
    )

    # 如指定权重文件则加载
    if model_path and model_path.exists():
        print(f"加载预训练权重: {model_path}")
        model.load_state_dict(torch.load(model_path))

    # 将 config 写到临时文件供 Trainer 读取
    tmp_cfg = log_dir_path / "temp_cfg.yaml"
    with open(tmp_cfg, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True)

    trainer = HousingTrainer(model, tmp_cfg)

    # 替换为 DummyAdvisor，禁用调优
    trainer.advisor = DummyAdvisor()  # type: ignore[assignment]

    print("\n[Baseline] 模型结构:\n", model)
    print("日志目录:", new_log_dir)

    trainer.train(train_loader, val_loader)

    print("Baseline 训练完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Housing Baseline 训练脚本 (无超参数调优)")
    parser.add_argument("--config", type=str, default="config/housing_config.yaml", help="配置文件路径")
    parser.add_argument("--model_path", type=str, default=None, help="预训练模型权重 (.pt) 路径（可选）")
    args = parser.parse_args()

    cfg_path = ROOT_DIR / args.config
    if not cfg_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {cfg_path}")

    model_path: Optional[Path] = Path(args.model_path) if args.model_path else None

    main(cfg_path, model_path=model_path) 