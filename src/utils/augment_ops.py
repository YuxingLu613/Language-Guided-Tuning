from __future__ import annotations

"""Simple registry of data-augmentation operations for MNIST-style images.
Extend here to add more options; keep names lowercase-snake-case.
"""

from typing import Dict
from torchvision import transforms
import torch

__all__ = ["AVAILABLE_AUGS", "get_transform"]

AVAILABLE_AUGS: Dict[str, transforms.Transform] = {
    # ---------------- Existing -----------------
    "rotation": transforms.RandomRotation(15),
    "shift": transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),

    # ---------------- New ----------------------
    # 水平翻转
    "flip": transforms.RandomHorizontalFlip(p=0.5),

    # 随机缩放 / 裁剪（保持 28x28）
    "scale": transforms.RandomResizedCrop(size=28, scale=(0.9, 1.1)),

    # 随机高斯噪声 – 用 Lambda 包装
    "noise": transforms.Lambda(lambda img: img + 0.1 * torch.randn_like(img)),
    # alias for compatibility with tabular naming (will never be chosen here)
    "gaussian_noise": transforms.Lambda(lambda img: img + 0.1 * torch.randn_like(img)),

    # 随机调整亮度 / 对比度（对灰度图也有效）
    "contrast": transforms.ColorJitter(brightness=0.2, contrast=0.2),

    # 恒等
    "none": transforms.Lambda(lambda x: x),
}


def get_transform(name: str):
    name = name.lower()
    if name not in AVAILABLE_AUGS:
        raise KeyError(f"Unknown augmentation '{name}'. Available: {list(AVAILABLE_AUGS)}")
    return AVAILABLE_AUGS[name] 