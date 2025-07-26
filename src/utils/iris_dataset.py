import torch
from torch.utils.data import Dataset
from typing import Optional, Callable

from sklearn.datasets import load_iris
import numpy as np

class IrisDataset(Dataset):
    """基于 sklearn 的 Iris 数据集的 PyTorch Dataset。

    数据集包含 4 个连续特征：
        sepal length, sepal width, petal length, petal width
    以及 3 个分类标签(0, 1, 2)。

    预处理步骤：
    1. 将所有特征标准化为均值 0、方差 1。

    返回 `(features, label)`，其中 features 为 `float32`，label 为 `int64`。
    """

    def __init__(self, transform: Optional[Callable] = None):
        super().__init__()
        self.transform = transform

        data = load_iris()
        self.features: np.ndarray = data["data"].astype("float32")
        self.targets: np.ndarray = data["target"].astype("int64")

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
        # 转成 torch Tensor
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long) 