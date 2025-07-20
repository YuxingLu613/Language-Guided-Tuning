import requests
import logging
from typing import Dict, Any

class LLMPromptOptimizer:
    """使用 DeepSeek-Chat API 将 Judger 的反馈精炼为 Advisor 可用的提示。"""

    def __init__(self, config: Dict[str, Any]):
        self.api_key = config["api_key"]
        self.temperature = config["temperature"]
        self.max_tokens = config["max_tokens"]
        self.api_base = "https://api.deepseek.com/v1"

    def refine_suggestion(self, advisor_prompt: str, judger_suggestion: str) -> str:
        """精炼 Judger 的建议。

        Args:
            advisor_prompt: Advisor 当前使用的完整 prompt（仅作上下文，不能被大幅修改）。
            judger_suggestion: Judger 输出的原始改进建议。

        Returns:
            str: 精炼后的简短建议，供 Advisor 参考。
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        system_prompt = (
            "You are a prompt optimization assistant. "
            "Your goal is to distill feedback coming from a model performance judger into concise, actionable notes "
            "for the advisor agent. You MUST NOT rewrite the advisor prompt itself; instead, summarise the feedback "
            "in at most five short bullet points so that the advisor can understand what to adjust. "
            "Return ONLY the bullet list, without any other commentary."
        )

        user_prompt = (
            "ADVISOR_PROMPT (context):\n"
            "```\n"
            f"{advisor_prompt}\n"
            "```\n\n"
            "JUDGER_FEEDBACK:\n"
            f"{judger_suggestion}\n\n"
            "Please distill the JUDGER_FEEDBACK into concise bullet points (<5 bullets). "
            "Do NOT modify or reformat the ADVISOR_PROMPT."
        )

        data = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }

        try:
            response = requests.post(
                f"{self.api_base}/chat/completions", headers=headers, json=data, timeout=30
            )
            response.raise_for_status()
            refined = response.json()["choices"][0]["message"]["content"]
            return refined.strip()
        except requests.exceptions.RequestException as e:
            logging.error(f"PromptOptimizer 调用 LLM API 失败: {e}")
            # 返回原始建议作为备用
            return judger_suggestion.strip() 