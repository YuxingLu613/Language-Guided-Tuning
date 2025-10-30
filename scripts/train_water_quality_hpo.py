from pathlib import Path
import sys, datetime, random
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

import argparse
from typing import Dict, Any, Tuple, Optional
import numpy as np
import yaml, torch
from torch.utils.data import DataLoader, random_split

from src.utils.water_quality_dataset import WaterQualityDataset
from src.models.custom_model import CustomModel
from src.trainer.trainer import Trainer

# ---------------- utils ----------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dataset(cfg: Dict[str, Any], seed: int) -> Tuple[DataLoader, DataLoader]:
    csv_path = ROOT_DIR / cfg["dataset"]["path"]
    ds = WaterQualityDataset(str(csv_path))
    val_ratio = float(cfg["dataset"].get("validation_split", 0.2))
    v_size = int(len(ds) * val_ratio)
    t_size = len(ds) - v_size
    g = torch.Generator(); g.manual_seed(seed)
    train_ds, val_ds = random_split(ds, [t_size, v_size], generator=g)
    kw = {"batch_size": cfg["training"]["batch_size"], "num_workers": 0, "pin_memory": torch.cuda.is_available()}
    return DataLoader(train_ds, shuffle=True, **kw), DataLoader(val_ds, shuffle=False, **kw)

# ---------------- main ----------------

def main(cfg_path: Path, model_path: Optional[Path] = None):
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg["logging"]["log_dir"] += f"_water_hpo_{ts}"
    log_dir = ROOT_DIR / cfg["logging"]["log_dir"]
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir/"used_config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")

    seed = cfg["training"].get("seed", 42)
    set_seed(seed)
    train_loader, val_loader = load_dataset(cfg, seed)

    model = CustomModel(
        input_dim=cfg["dataset"]["input_dim"],
        hidden_dims=cfg["model"]["architecture"]["hidden_dims"],
        output_dim=cfg["dataset"].get("num_classes", 2),
        dropout=cfg["model"]["architecture"].get("dropout", 0.5),
    )
    if model_path and model_path.exists():
        model.load_state_dict(torch.load(model_path))

    tmp_cfg = log_dir/"temp_cfg.yaml"; tmp_cfg.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")

    trainer = Trainer(model, tmp_cfg)
    print("开始 HPO 训练，日志目录", cfg["logging"]["log_dir"])
    trainer.train(train_loader, val_loader)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Water Quality HPO Training (LLM Advisor)")
    ap.add_argument("--config", default="config/water_quality_config.yaml")
    ap.add_argument("--model_path", default=None)
    args = ap.parse_args()

    main(Path(args.config), Path(args.model_path) if args.model_path else None) 