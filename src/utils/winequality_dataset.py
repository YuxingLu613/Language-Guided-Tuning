import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Optional, Callable, List

class WineQualityDataset(Dataset):
    """Dataset for the UCI Wine Quality (Red) regression task.

    The raw CSV is expected to contain 11 physicochemical input features and one
    target column named ``quality`` representing the wine quality score.

    Pre-processing steps:
    1. Drop any rows with missing values.
    2. Standardise each feature to mean 0 / std 1.

    Each sample returns ``(features, quality)`` where
        - ``features`` is a float32 tensor of shape ``(11,)``
        - ``quality`` is a float32 tensor of shape ``(1,)``
    """

    def __init__(self, csv_path: str, transform: Optional[Callable] = None, *, target_column: str = "quality", target_scale: float = 10.0):
        super().__init__()
        self.transform = transform

        # 尝试自动检测分隔符（逗号或分号）
        try:
            df = pd.read_csv(csv_path, sep=None, engine="python")
        except Exception:
            # 回退为逗号分隔
            df = pd.read_csv(csv_path)

        # 清理列名空白并转小写（便于匹配）
        df.columns = df.columns.str.strip()
        lower_cols: List[str] = [c.lower() for c in df.columns]

        # ---- 目标列确定 ----
        tgt_col = target_column.lower().strip()
        if tgt_col not in lower_cols:
            # 在列名中查找包含 quality 关键字的列
            cand = [c for c in lower_cols if "quality" in c]
            if not cand:
                raise KeyError(
                    f"找不到目标列 '{target_column}'，可用列: {df.columns.tolist()}"
                )
            tgt_col = cand[0]

        # 获取真实列名（保留大小写）
        real_tgt_col = df.columns[lower_cols.index(tgt_col)]

        # 提取目标与特征，目标按 scale 缩放
        self.target_scale = target_scale
        self.targets = df[real_tgt_col].astype("float32").values / self.target_scale
        self.features = df.drop(columns=[real_tgt_col]).values.astype("float32")

        # Standardise features
        self.feature_mean = self.features.mean(axis=0, keepdims=True)
        self.feature_std = self.features.std(axis=0, keepdims=True) + 1e-8
        self.features = (self.features - self.feature_mean) / self.feature_std

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx: int):
        x = self.features[idx]
        y = self.targets[idx]
        if self.transform is not None:
            x = self.transform(x)
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32).unsqueeze(-1),
        ) 