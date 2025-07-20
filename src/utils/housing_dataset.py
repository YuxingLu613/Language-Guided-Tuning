import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Optional

class HousingDataset(Dataset):
    """基于CSV文件的房价预测数据集。返回 (features, price)。"""

    _NEIGHBORHOOD_MAP = {"Urban": 0, "Suburb": 1, "Rural": 2}

    def __init__(self, csv_path: str, transform: Optional[callable] = None, target_scale: float = 1e5):
        self.data = pd.read_csv(csv_path)
        self.transform = transform

        # 预处理：将类别特征转换为数值
        self.data["Neighborhood"] = self.data["Neighborhood"].map(self._NEIGHBORHOOD_MAP)

        # 缺失值处理：简单丢弃缺失行
        self.data.dropna(inplace=True)

        # 分离特征与目标
        self.features = self.data.drop(columns=["Price"]).values.astype("float32")
        self.targets = self.data["Price"].values.astype("float32") / target_scale  # 价格缩放

        # 特征标准化（均值 0 方差 1）
        self.feature_mean = self.features.mean(axis=0, keepdims=True)
        self.feature_std = self.features.std(axis=0, keepdims=True) + 1e-8
        self.features = (self.features - self.feature_mean) / self.feature_std

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.targets[idx]
        if self.transform:
            x = self.transform(x)
        # 转成张量
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).unsqueeze(-1) 