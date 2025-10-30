from pathlib import Path
import sys, datetime, random
from typing import Any, Dict, Tuple, Optional

ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

import numpy as np
import argparse, yaml, torch
from torch.utils.data import DataLoader, random_split

from src.utils.iris_dataset import IrisDataset
from src.trainer.tabular_class_aug_trainer import TabularClassAugTrainer

from importlib import import_module
import inspect
import torch.nn as _nn

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
    ds = IrisDataset()
    val_ratio = float(cfg["dataset"].get("validation_split", 0.2))
    v_size = int(len(ds) * val_ratio)
    t_size = len(ds) - v_size
    g = torch.Generator(); g.manual_seed(seed)
    train_ds, val_ds = random_split(ds, [t_size, v_size], generator=g)
    kw = {"batch_size": cfg["training"]["batch_size"], "num_workers": 0, "pin_memory": torch.cuda.is_available()}
    return DataLoader(train_ds, shuffle=True, **kw), DataLoader(val_ds, shuffle=False, **kw)

# ---------------- model loader ----------------

def build_model(cfg: Dict[str, Any]):
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
            hidden_dims=cfg["model"].get("architecture", {}).get("hidden_dims", None),
            output_dim=cfg["dataset"].get("num_classes", 3),
            dropout=cfg["model"].get("architecture", {}).get("dropout", 0.3),
        )
    except TypeError:
        model = model_cls()  # type: ignore
    return model

# ---------------- main ----------------

def main(cfg_path: Path, model_path: Optional[Path] = None):
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg["logging"]["log_dir"] += f"_iris_aug_{ts}"
    log_dir = ROOT_DIR / cfg["logging"]["log_dir"]
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "used_config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")

    seed = cfg["training"].get("seed", 42)
    set_seed(seed)
    train_loader, val_loader = load_dataset(cfg, seed)

    model = build_model(cfg)
    if model_path and model_path.exists():
        model.load_state_dict(torch.load(model_path))

    tmp_cfg = log_dir / "temp_cfg.yaml"
    tmp_cfg.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")

    trainer = TabularClassAugTrainer(model, tmp_cfg)
    print("开始 Iris 增强训练，日志目录:", cfg["logging"]["log_dir"])
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Iris Tabular Augmented Training")
    ap.add_argument("--config", default="config/iris_config.yaml")
    ap.add_argument("--model_path", default=None)
    args = ap.parse_args()

    main(Path(args.config), Path(args.model_path) if args.model_path else None) 