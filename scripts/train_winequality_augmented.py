from __future__ import annotations

"""Train Wine Quality regression with LLM-driven tabular data augmentation.

Usage:
    python scripts/train_winequality_augmented.py \
        --config config/winequality_config.yaml

The script mirrors `train_housing_augmented.py` but swaps in the Wine Quality
CSV dataset. It leverages `TabularAugTrainer` to let an LLM choose the best
feature-space augmentation each epoch when `augmentation.enabled=true` in the
YAML config.
"""

from pathlib import Path
import sys, argparse, random, datetime
from typing import Optional, Dict, Any, Tuple

ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import yaml

from src.utils.winequality_dataset import WineQualityDataset
from src.models.custom_model import CustomModel
from src.trainer.tabular_aug_trainer import TabularAugTrainer


# ---------------- Utils ----------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_winequality(config: Dict[str, Any], seed: int) -> Tuple[DataLoader, DataLoader]:
    csv_path = ROOT_DIR / config["dataset"]["path"]
    scale = float(config["dataset"].get("target_scale", 10.0))
    dataset = WineQualityDataset(str(csv_path), target_scale=scale)

    val_split = float(config["dataset"].get("validation_split", 0.2))
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    g = torch.Generator(); g.manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=g)

    loader_kw = {
        "batch_size": config["training"]["batch_size"],
        "num_workers": 0,
        "pin_memory": torch.cuda.is_available(),
    }
    return (
        DataLoader(train_ds, shuffle=True, **loader_kw),
        DataLoader(val_ds, shuffle=False, **loader_kw),
    )


# ---------------- Main ----------------

def main(config_path: Path, model_path: Optional[Path] = None):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    seed = config["training"].get("seed", 42)
    set_seed(seed)

    # Timestamped log dir
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = (ROOT_DIR / config["logging"]["log_dir"]).joinpath(f"wine_aug_{ts}")
    log_dir.mkdir(parents=True, exist_ok=True)
    config["logging"]["log_dir"] = str(log_dir)

    # Save used config
    used_cfg_path = log_dir / "used_config.yaml"
    used_cfg_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")

    train_loader, val_loader = load_winequality(config, seed)

    model = CustomModel(
        input_dim=config["dataset"]["input_dim"],
        hidden_dims=config["model"]["architecture"]["hidden_dims"],
        output_dim=1,
        dropout=config["model"]["architecture"].get("dropout", 0.1),
    )

    if model_path and model_path.exists():
        print("加载预训练模型权重", model_path)
        model.load_state_dict(torch.load(model_path))

    trainer = TabularAugTrainer(model, used_cfg_path)
    print("\n模型结构:\n", model)
    print("\n开始带数据增强的训练…")
    trainer.train(train_loader, val_loader)
    print("训练完成！")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/winequality_config.yaml")
    ap.add_argument("--model_path", type=str, default=None)
    args = ap.parse_args()

    main(Path(args.config), Path(args.model_path) if args.model_path else None) 