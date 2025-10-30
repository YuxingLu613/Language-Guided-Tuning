from __future__ import annotations

from typing import Dict, Any, Tuple, List, Optional
import re, logging, requests

AVAILABLE_AUGS = [
    "rotation",
    "shift",
    "flip",
    "scale",
    "noise",
    "contrast",
    # ---- Tabular specific ----
    "gaussian_noise",
    "feature_dropout",
    "none",
]

# ---------------------------------------------------------------------------
# NOTE: By default the LLM was too conservative and often returned "none".
# To incentivise it to actually pick an augmentation we:
#   1) Raise the sampling temperature to at least `_MIN_TEMPERATURE`.
#   2) Re-phrase the system prompt to strongly *prefer* a real transform and
#      reserve "none" only for cases of clear overfitting or already optimal
#      performance.
# ---------------------------------------------------------------------------
_MIN_TEMPERATURE = 0.5

__all__ = ["LLMAugmentationChooser"]

class LLMAugmentationChooser:
    """Ask LLM to choose one augmentation method from a predefined list."""

    _SYSTEM = (
        "You are a deep-learning model-training expert. Given the current training/validation "
        "metrics and the augmentation currently in use, decide which *single* augmentation from "
        "the list will most likely improve performance in the *next* epoch. Unless the provided "
        "metrics already show strong improvement with the current setup, you SHOULD choose an "
        "augmentation method *different* from the current one. Reply 'none' *only* when applying "
        "another augmentation would clearly degrade performance or is redundant. Respond with the "
        "method name only, without any explanation."
    )

    def __init__(self, llm_cfg: Dict[str, Any], allowed_augs: Optional[List[str]] = None):
        self.api_key = llm_cfg["api_key"]
        # Guarantee a minimum amount of randomness so the LLM is less conservative.
        cfg_temp = llm_cfg.get("temperature", 0.3)
        self.temperature = max(cfg_temp, _MIN_TEMPERATURE)
        self.max_tokens = llm_cfg.get("max_tokens", 128)
        self.api_base = llm_cfg.get("api_base", "https://api.deepseek.com/v1")

        # Restrict available augmentations if provided
        self.available_augs: List[str] = allowed_augs if allowed_augs is not None else list(AVAILABLE_AUGS)

    def _build_prompt(self, metrics: Dict[str, Any], context: Dict[str, Any], guidance: str | None = None, previous_context: str | None = None) -> str:
        """Compose prompt with metrics and extra context for better decision."""
        import json
        
        # 根据数据类型确定可用的增强方法
        data_type = context.get("data_type", "unknown")
        dataset_name = context.get("dataset", "unknown")
        
        # 根据数据类型筛选适合的增强方法
        if data_type == "image":
            # 图像数据适合的增强方法
            suitable_augs = [aug for aug in self.available_augs if aug in [
                "rotation", "shift", "flip", "scale", "noise", "contrast", "none"
            ]]
        elif data_type == "tabular":
            # 表格数据适合的增强方法
            suitable_augs = [aug for aug in self.available_augs if aug in [
                "gaussian_noise", "feature_dropout", "shift", "none"
            ]]
        else:
            # 未知数据类型，使用所有可用的增强方法
            suitable_augs = self.available_augs
        
        prompt = f"Data type: {data_type}\nDataset: {dataset_name}\n"
        prompt += "Available methods: " + ", ".join(suitable_augs) + "\n\n"

        # Attach high-level context
        cur_aug = context.get("current_augmentation", "none")
        epoch = context.get("epoch")
        dataset_size = context.get("dataset_size")
        prompt += f"Epoch: {epoch}\nCurrent augmentation in use: {cur_aug}\nTrain dataset size: {dataset_size}\n\n"

        prompt += "Latest metrics (JSON):\n" + json.dumps(metrics, ensure_ascii=False) + "\n\n"
        if guidance:
            prompt += "\nREFINED GUIDANCE FROM JUDGER:\n" + guidance + "\n"
        
        if previous_context:
            prompt += "\nPREVIOUS AGENT CHANGES:\n" + previous_context + "\n"

        prompt += "Choose exactly one augmentation method (or 'none')."
        return prompt

    def _parse(self, text: str) -> Tuple[str, bool]:
        text = text.strip().lower()
        for name in self.available_augs:
            if name == text:
                return name, name != "none"
        return "none", False

    def choose(self, metrics: Dict[str, Any], context: Dict[str, Any], guidance: str | None = None, previous_context: str | None = None) -> Tuple[str, bool]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        messages = [
            {"role": "system", "content": self._SYSTEM},
            {"role": "user", "content": self._build_prompt(metrics, context, guidance, previous_context)},
        ]
        data = {
            "model": "deepseek-chat",
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        try:
            resp = requests.post(f"{self.api_base}/chat/completions", headers=headers, json=data, timeout=60)
            resp.raise_for_status()
            reply = resp.json()["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            logging.error("LLM chooser request failed: %s", e)
            return "none", False
        choice, use_aug = self._parse(reply)
        return choice, use_aug
