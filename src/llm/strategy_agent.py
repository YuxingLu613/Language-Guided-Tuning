from __future__ import annotations

"""LLM Strategy Agent
This module defines `StrategyAgent`, an LLM-powered helper that suggests **training strategies**
(loss function, optimiser, learning-rate, scheduler, etc.) between epochs based on:
1. Current strategy
2. Recent metrics
3. Task type (classification / regression / others)
4. Full user config for additional context

It is intentionally kept *generic* – nothing here assumes MNIST or Housing explicitly.

The public API mimics `LLMAdvisor.get_advice` so that trainer code can swap one for the other
with minimal changes.
"""

from typing import Dict, Any, Tuple, Optional
import re
import requests
import logging


__all__ = ["StrategyAgent"]


class StrategyAgent:
    """Ask a large language-model for next-epoch *training strategy*.

    The agent is **stateless** apart from the LLM credentials that live in the provided
    ``config`` dict under the ``llm`` key – identical to other agents in this repo so
    that the same YAML can be reused.
    """

    _KEYS_MAPPING = {
        # normalise → canonical (lr intentionally omitted; we don't accept lr modifications)
        "optimizer": "optimizer",
        "optimiser": "optimizer",  # UK spelling
        "loss": "loss_function",
        "loss_fn": "loss_function",
        "loss_function": "loss_function",
    }

    def __init__(self, llm_cfg: Dict[str, Any]):
        self.api_key: str = llm_cfg["api_key"]
        self.temperature: float = llm_cfg.get("temperature", 0.2)
        self.max_tokens: int = llm_cfg.get("max_tokens", 1024)
        self.api_base: str = llm_cfg.get("api_base", "https://api.deepseek.com/v1")

    # ---------------------------------------------------------------------
    # Prompt building helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _strategy_to_str(strategy: Dict[str, Any]) -> str:
        parts = []
        for k, v in strategy.items():
            if k == "loss_function":
                parts.append(f"loss={v}")
            elif k == "optimizer":
                parts.append(f"optimizer={v}")
            # deliberately skip learning_rate to discourage LLM from changing it
        return ", ".join(parts)

    def _build_prompt(
        self,
        task_type: str,
        current_strategy: Dict[str, Any],
        metrics: Dict[str, Any],
        context_cfg: Dict[str, Any],
        guidance: Optional[str] = None,
    ) -> str:
        """Return the text prompt for the chat completion call."""

        prompt = (
            "You are an expert machine-learning practitioner.\n"
            "At the end of each epoch you will receive: (a) task type, (b) current loss\n"
            "function & optimizer, and (c) the latest training / validation metrics.\n\n"
            "Your job: Decide the *loss function* **and** *optimizer* to use for the **next**\n"
            "epoch. If the current combination is still appropriate, reply **exactly**\n"
            "`no change` (case-insensitive). Otherwise reply with *one line* in the form:\n"
            "`loss=<loss_function>, optimizer=<optimizer>` (comma-separated, no extras).\n\n"
            "Do NOT suggest learning-rate changes or any other hyper-parameters.\n\n"
            "Task type: {task_type}\n"
        ).format(task_type=task_type)

        # Optional guidance from another agent (e.g. Judger)
        if guidance:
            prompt += f"\nREFINED GUIDANCE FROM JUDGER:\n{guidance}\n\n"

        # Attach current strategy (explicit keys)
        prompt += "Current loss function: " + current_strategy.get("loss_function", "N/A") + "\n"
        prompt += "Current optimizer: " + current_strategy.get("optimizer", "N/A") + "\n\n"

        # Attach metrics – keep raw json for ease of LLM digestion
        import json as _json
        prompt += "Recent metrics (JSON):\n" + _json.dumps(metrics, ensure_ascii=False) + "\n\n"

        # Provide static context (model / dataset) – truncated to avoid token bloat
        ds_cfg = context_cfg.get("dataset", {})
        model_cfg = context_cfg.get("model", {})
        prompt += "Dataset config: " + str(ds_cfg) + "\n"
        prompt += "Model config: " + str(model_cfg) + "\n"

        return prompt

    # ---------------------------------------------------------------------
    # Suggestion parsing helpers
    # ---------------------------------------------------------------------

    def _parse_suggestion(self, suggestion: str) -> Tuple[Dict[str, Any], bool]:
        """Parse `key=value` pairs from LLM answer -> canonical strategy dict."""
        suggestion = suggestion.strip()
        if suggestion.lower() == "no change":
            return {}, False

        # regex matches patterns like `loss=mse`, `optimizer=sgd`
        pattern = r"([a-zA-Z_]+)\s*=\s*([a-zA-Z0-9_.\-]+)"
        pairs = re.findall(pattern, suggestion)
        if not pairs:
            return {}, False

        strat: Dict[str, Any] = {}
        for raw_key, raw_val in pairs:
            key = self._KEYS_MAPPING.get(raw_key.lower())
            if not key:
                # Not an accepted key (e.g., lr) → ignore
                continue
            strat[key] = raw_val.lower()
        return strat, bool(strat)

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def decide_strategy(
        self,
        task_type: str,
        current_strategy: Dict[str, Any],
        metrics: Dict[str, Any],
        context_cfg: Dict[str, Any],
        guidance: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any], bool]:
        """Ask the language-model and return (raw_text, parsed_strategy, should_update)."""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a highly skilled ML practitioner specialising in optimising\n"
                    "training strategies across diverse tasks."
                ),
            },
            {
                "role": "user",
                "content": self._build_prompt(task_type, current_strategy, metrics, context_cfg, guidance),
            },
        ]

        data = {
            "model": "deepseek-chat",
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }

        try:
            resp = requests.post(f"{self.api_base}/chat/completions", headers=headers, json=data, timeout=30)
            resp.raise_for_status()
            suggestion = resp.json()["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            logging.error("LLM request failed: %s", e)
            return f"error: {e}", {}, False

        parsed, should = self._parse_suggestion(suggestion)
        return suggestion, parsed, should 