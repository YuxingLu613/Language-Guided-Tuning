from __future__ import annotations

"""Registry of augmentation operations designed for tabular / vector-style data.
Each transform is a simple callable that receives a numpy.ndarray (or torch / list)
and returns a transformed array of the same shape & dtype.

Available operations
--------------------
- gaussian_noise   : Additive Gaussian noise to continuous features.
- feature_dropout  : Randomly zero-out (mask) features with probability p.
- none             : Identity transform (no-op).

Feel free to extend with SMOTER, Mixup etc. – keep names lowercase and update
LLMAugmentationChooser.AVAILABLE_AUGS correspondingly.
"""

from typing import Dict, Callable
import numpy as np

__all__ = ["AVAILABLE_AUGS", "get_transform"]


# ------------------------- Transform helpers ------------------------- #

def _gaussian_noise(sigma: float = 0.02):
    def _t(x):
        arr = np.asarray(x, dtype=np.float32)
        return arr + sigma * np.random.randn(*arr.shape).astype(arr.dtype)
    return _t


def _feature_dropout(p: float = 0.1):
    """Randomly sets a proportion `p` of features to zero."""
    def _t(x):
        arr = np.asarray(x, dtype=np.float32)
        mask = np.random.binomial(1, 1 - p, size=arr.shape).astype(arr.dtype)
        return arr * mask
    return _t


def _shift(scale: float = 0.05):
    """Add a small uniform shift to each feature (±scale)."""
    def _t(x):
        arr = np.asarray(x, dtype=np.float32)
        shift_vals = np.random.uniform(-scale, scale, size=arr.shape).astype(arr.dtype)
        return arr + shift_vals
    return _t


def _identity(x):
    return x


AVAILABLE_AUGS: Dict[str, Callable] = {
    "gaussian_noise": _gaussian_noise(0.02),
    "feature_dropout": _feature_dropout(0.1),
    "shift": _shift(0.05),
    "none": _identity,
}


# ------------------------- Accessor ------------------------- #

def get_transform(name: str):
    name = name.lower()
    if name not in AVAILABLE_AUGS:
        raise KeyError(f"Unknown tabular augmentation '{name}'. Available: {list(AVAILABLE_AUGS)}")
    return AVAILABLE_AUGS[name] 