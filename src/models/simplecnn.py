from __future__ import annotations

"""Alias module for `SimpleCNN`.

该文件仅作为 **占位/别名**，使得当配置中 `model.name: SimpleCNN`
时，`LLMModelStructureOptimizer` 能够在路径 `src/models/simplecnn.py`
找到模型代码上下文。

实际实现复用 `cnn_model.SimpleCNN`。若将来需要定制，可
直接在此文件修改。"""

from .cnn_model import SimpleCNN  # noqa: F401 