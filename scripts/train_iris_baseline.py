from pathlib import Path
import sys
import datetime

# 将项目根目录加入 Python 路径
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

import argparse
import random
from typing import Dict, Any, Tuple, Optional

import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader, random_split

# 本项目模块
from src.utils.iris_dataset import IrisDataset
from importlib import import_module
import inspect
import torch.nn as _nn
from src.trainer.trainer import Trainer

# ----------------------- 工具函数 ----------------------- #

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_iris(cfg: Dict[str, Any], seed: int) -> Tuple[DataLoader, DataLoader]:
    """加载 Iris 数据集并按照验证比例划分训练/验证集。"""
    dataset = IrisDataset()

    val_ratio = float(cfg["dataset"].get("validation_split", 0.2))
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size

    g = torch.Generator()
    g.manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=g)

    dl_kwargs = {
        "batch_size": cfg["training"]["batch_size"],
        "num_workers": 0,
        "pin_memory": torch.cuda.is_available(),
    }
    return (
        DataLoader(train_ds, shuffle=True, **dl_kwargs),
        DataLoader(val_ds, shuffle=False, **dl_kwargs),
    )

# ----------------------- DummyAdvisor ----------------------- #

class DummyAdvisor:
    """占位 Advisor：始终返回 no change。"""

    def get_advice(self, current_params, metrics, context_cfg, guidance=None):
        return "no change", {}, False

# ----------------------------- 主流程 ----------------------------- #

def main(config_path: Path, model_path: Optional[Path] = None):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 调整日志目录，防止覆盖正式实验
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    orig_log_dir = cfg["logging"]["log_dir"]
    new_log_dir = f"{orig_log_dir}_iris_baseline_{ts}"
    cfg["logging"]["log_dir"] = new_log_dir

    log_dir_path = ROOT_DIR / new_log_dir
    log_dir_path.mkdir(parents=True, exist_ok=True)

    # 保存实际使用的 config
    with open(log_dir_path / "used_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)

    # 设置随机种子
    seed = cfg["training"].get("seed", 42)
    set_seed(seed)

    # 加载数据
    train_loader, val_loader = load_iris(cfg, seed)

    # ---------------- 根据 config 动态加载模型 ---------------- #
    model_name: str = cfg["model"]["name"]

    try:
        mod = import_module(f"src.models.{model_name.lower()}")
    except ModuleNotFoundError:
        # 若文件名与类名存在下划线差异, 尝试回退到 custom_model
        from src.models import custom_model as mod  # type: ignore

    # 选取第一个 nn.Module 子类作为模型类
    model_cls = None
    for _name, _obj in inspect.getmembers(mod, inspect.isclass):
        if issubclass(_obj, _nn.Module):
            model_cls = _obj
            break
    if model_cls is None:
        raise RuntimeError(f"未在 {mod} 找到 nn.Module 子类用于构建模型")

    # 根据可能的 __init__ 参数尝试实例化, 不匹配则退化为无参
    try:
        model = model_cls(
            input_dim=cfg["dataset"].get("input_dim", None),
            hidden_dims=cfg["model"]["architecture"].get("hidden_dims", None),
            output_dim=cfg["dataset"].get("num_classes", 1),
            dropout=cfg["model"]["architecture"].get("dropout", 0.5),
        )
    except TypeError:
        model = model_cls()  # type: ignore[call-arg]

    # 如果提供了预训练权重则加载
    if model_path and model_path.exists():
        print(f"加载预训练权重: {model_path}")
        model.load_state_dict(torch.load(model_path))

    # 将修改后的 cfg 写入临时文件供 Trainer 使用
    tmp_cfg = log_dir_path / "temp_cfg.yaml"
    with open(tmp_cfg, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)

    trainer = Trainer(model, tmp_cfg)

    # 替换为 DummyAdvisor，避免调用真实 LLM
    trainer.advisor = DummyAdvisor()  # type: ignore[assignment]
    trainer.judger.max_attempts = 0  # type: ignore[attr-defined]

    print("\n[Iris Baseline] 模型结构:\n", model)
    print("日志目录:", new_log_dir)

    trainer.train(train_loader, val_loader)

    print("Baseline 训练完成！")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Iris Baseline 训练脚本 (无超参数调优)")
    ap.add_argument("--config", default="config/iris_config.yaml", help="配置文件路径")
    ap.add_argument("--model_path", default=None, help="预训练模型权重 (.pt) 路径（可选）")
    args = ap.parse_args()

    cfg_p = ROOT_DIR / args.config
    if not cfg_p.exists():
        raise FileNotFoundError(f"配置文件不存在: {cfg_p}")

    mdl_p: Optional[Path] = Path(args.model_path) if args.model_path else None

    main(cfg_p, model_path=mdl_p) 