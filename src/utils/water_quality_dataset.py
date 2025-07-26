import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Optional, Callable

class WaterQualityDataset(Dataset):
    """基于 `water_potability.csv` 文件的水质可饮用性分类数据集。

    数据集包含 9 个连续特征与 1 个二分类标签 `Potability`：
        ph、Hardness、Solids、Chloramines、Sulfate、Conductivity、
        Organic_carbon、Trihalomethanes、Turbidity → 特征
        Potability → 0/1 标签 (0 = 不可饮用, 1 = 可饮用)

    预处理步骤：
    1. 丢弃任何包含缺失值的样本。
    2. 将所有特征标准化为均值 0、方差 1。

    返回 `(features, label)`，其中 features 为 `float32`，label 为 `int64`。
    """

    def __init__(self, csv_path: str, transform: Optional[Callable] = None):
        super().__init__()
        self.transform = transform

        # 读取 CSV
        df = pd.read_csv(csv_path)
        # 丢弃缺失值
        df = df.dropna().reset_index(drop=True)

        # 提取目标
        self.targets = df["Potability"].astype("int64").values
        # 提取特征
        self.features = df.drop(columns=["Potability"]).values.astype("float32")

        # 标准化特征
        self.feature_mean = self.features.mean(axis=0, keepdims=True)
        self.feature_std = self.features.std(axis=0, keepdims=True) + 1e-8
        self.features = (self.features - self.feature_mean) / self.feature_std

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.targets[idx]
        if self.transform is not None:
            x = self.transform(x)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long) 