import requests
import logging
from typing import Dict, Any, Optional, List

class LLMPromptOptimizer:
    """使用 DeepSeek-Chat API 将 Judger 的反馈精炼为 Advisor 可用的提示。"""

    def __init__(self, config: Dict[str, Any]):
        self.api_key = config["api_key"]
        self.temperature = config["temperature"]
        self.max_tokens = config["max_tokens"]
        self.api_base = "https://api.deepseek.com/v1"
        self.timeout = config.get("request_timeout", 120)

    def refine_suggestion(self, advisor_prompt: str, judger_suggestion: str, enabled_agents: Optional[List[str]] = None) -> Dict[str, str]:
        """精炼 Judger 的建议。

        Args:
            advisor_prompt: Advisor 当前使用的完整 prompt（仅作上下文，不能被大幅修改）。
            judger_suggestion: Judger 输出的原始改进建议。
            enabled_agents: 启用的agent列表 ['aug', 'adaptive', 'hpo']

        Returns:
            Dict[str, str]: 针对不同agent的精炼建议字典
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        system_prompt = (
            "You are a prompt optimization assistant. "
            "Your goal is to distill feedback coming from a model performance judger into concise, actionable notes "
            "for different agent types. You MUST NOT rewrite the advisor prompt itself; instead, summarise the feedback "
            "in agent-specific bullet points so that each agent can understand what to adjust. "
            "Return ONLY the agent-specific suggestions in the requested format, without any other commentary."
        )

        # Build agent-specific format request
        agent_format = ""
        if enabled_agents:
            agent_format = "\nAGENT_SUGGESTIONS:\n"
            for agent in enabled_agents:
                if agent == 'aug':
                    agent_format += f"AUGMENTATION_AGENT: <suggestion for data augmentation strategy>\n"
                elif agent == 'adaptive':
                    agent_format += f"ADAPTIVE_AGENT: <suggestion for training strategy (loss function, optimizer)>\n"
                elif agent == 'hpo':
                    agent_format += f"HPO_AGENT: <suggestion for hyperparameter tuning>\n"

        user_prompt = (
            "ADVISOR_PROMPT (context):\n"
            "```\n"
            f"{advisor_prompt}\n"
            "```\n\n"
            "JUDGER_FEEDBACK:\n"
            f"{judger_suggestion}\n\n"
            f"Please distill the JUDGER_FEEDBACK into agent-specific suggestions.{agent_format}"
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
                f"{self.api_base}/chat/completions", headers=headers, json=data, timeout=self.timeout
            )
            response.raise_for_status()
            refined = response.json()["choices"][0]["message"]["content"]
            
            # Parse agent-specific suggestions
            agent_suggestions = {}
            lines = [l.strip() for l in refined.strip().split("\n") if l.strip()]
            
            for line in lines:
                if line.upper().startswith("AUGMENTATION_AGENT"):
                    aug_suggestion = line.split(":", 1)[-1].strip()
                    if aug_suggestion and aug_suggestion.upper() != "N/A":
                        agent_suggestions['aug'] = aug_suggestion
                elif line.upper().startswith("ADAPTIVE_AGENT"):
                    adaptive_suggestion = line.split(":", 1)[-1].strip()
                    if adaptive_suggestion and adaptive_suggestion.upper() != "N/A":
                        agent_suggestions['adaptive'] = adaptive_suggestion
                elif line.upper().startswith("HPO_AGENT"):
                    hpo_suggestion = line.split(":", 1)[-1].strip()
                    if hpo_suggestion and hpo_suggestion.upper() != "N/A":
                        agent_suggestions['hpo'] = hpo_suggestion
            
            return agent_suggestions if agent_suggestions else {}
        except requests.exceptions.RequestException as e:
            logging.error(f"PromptOptimizer 调用 LLM API 失败: {e}")
            # 返回空字典作为备用
            return {} 