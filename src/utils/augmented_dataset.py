from __future__ import annotations

from typing import List, Tuple, Any, Sequence

import torch
from torch.utils.data import Dataset

__all__ = ["AugmentedDataset"]


class AugmentedDataset(Dataset):
    """Wrap an existing *base* dataset and allow appending extra samples.

    The additional samples can come from an LLM (see `LLMDataAugmentor`) or any
    other source. Each sample **must** be returned in the same format as the
    base dataset – usually a `(inputs, target)` tuple. No assumptions are made
    about the data types (NumPy, tensor, etc.) so long as the collate_fn of the
    DataLoader can handle them.

    Example
    -------
    >>> base_ds = torchvision.datasets.MNIST(root="./data", train=True, download=True)
    >>> aug_ds = AugmentedDataset(base_ds)
    >>> aug_ds.add_samples([(img_tensor, label)])
    >>> len(aug_ds)  # original + 1
    """

    def __init__(self, base_dataset: Dataset):
        self.base_dataset: Dataset = base_dataset
        self._extra_samples: List[Any] = []  # each sample mirrors base_dataset[idx]

    # --------------------------------------------------------------
    # Dataset protocol
    # --------------------------------------------------------------

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.base_dataset) + len(self._extra_samples)

    def __getitem__(self, idx: int):  # type: ignore[override]
        base_len = len(self.base_dataset)
        if idx < base_len:
            return self.base_dataset[idx]
        return self._extra_samples[idx - base_len]

    # --------------------------------------------------------------
    # Public helper
    # --------------------------------------------------------------

    def add_samples(self, samples: Sequence[Any]) -> None:
        """Append *already preprocessed* samples to the dataset.

        Parameters
        ----------
        samples: Sequence[Any]
            Each element should match the return type of `self.base_dataset.__getitem__`.
        """
        if not samples:
            return
        self._extra_samples.extend(samples)

    # convenient alias so that trainer code can remain agnostic of the wrapper
    extend = add_samples 