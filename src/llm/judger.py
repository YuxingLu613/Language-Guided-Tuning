import json
from typing import Dict, List, Any, Tuple, Optional
import requests
import logging
from pathlib import Path

class LLMJudger:
    def __init__(self, config: Dict[str, Any]):
        """
        初始化LLM评判器
        
        Args:
            config: LLM配置信息
        """
        self.api_key = config["api_key"]
        self.temperature = config["temperature"]
        self.max_tokens = config["max_tokens"]
        self.api_base = "https://api.deepseek.com/v1"
        self.timeout = config.get("request_timeout", 120)
        self.optimization_attempts = 0  # 记录优化尝试次数
        self.max_attempts = 5  # 最大尝试次数
        self.baseline_metrics = None  # 缓存的基准指标
        
    def reset_state(self):
        """
        重置评判器状态，确保每个epoch都有新的尝试机会
        """
        self.optimization_attempts = 0  # 重置优化尝试次数
        self.baseline_metrics = None  # 清除缓存的基准指标
        
    def _create_comparison_prompt(
        self,
        baseline_metrics: Dict[str, float],
        optimized_metrics: Dict[str, float],
        baseline_params: Dict[str, Any],
        optimized_params: Dict[str, Any],
        attempt_number: int,
        enabled_agents: Optional[List[str]] = None,
    ) -> str:
        """生成精简版比较提示词，仅关注 loss 与核心指标。"""

        def _metrics_to_str(metrics: Dict[str, Any]) -> str:
            if isinstance(metrics, dict) and "train" in metrics:
                parts = [
                    "Training:",
                    *[f"- {k}: {v:.4f}" for k, v in metrics["train"].items()],
                ]
                if "val" in metrics:
                    parts.extend(
                        [
                            "Validation:",
                            *[f"- {k}: {v:.4f}" for k, v in metrics["val"].items()],
                        ]
                    )
                return "\n".join(parts)
            else:
                return "\n".join([f"- {k}: {v:.4f}" for k, v in metrics.items()])

        # Build agent-specific suggestion format
        agent_suggestions = ""
        if enabled_agents:
            agent_suggestions = "\nAGENT_SUGGESTIONS:\n"
            for agent in enabled_agents:
                if agent == 'aug':
                    agent_suggestions += f"AUGMENTATION_AGENT: <suggestion for data augmentation strategy>\n"
                elif agent == 'adaptive':
                    agent_suggestions += f"ADAPTIVE_AGENT: <suggestion for training strategy (loss function, optimizer)>\n"
                elif agent == 'hpo':
                    agent_suggestions += f"HPO_AGENT: <suggestion for hyperparameter tuning>\n"
        
        prompt = (
            f"Attempt {attempt_number}/{self.max_attempts}.\n"
            "Compare ONLY the numeric metrics (e.g., loss, accuracy, mse, mae) between baseline and optimized models.\n"
            "Give PRIMARY importance to *validation* metrics (e.g., accuracy, precision, recall, mse, mae).\n"
            "Validation loss can be referenced but should NOT dominate the decision because loss may change substantially when switching loss functions.\n"
            "Training metrics are secondary and should be used only when validation metrics are missing or inconclusive.\n"
            "Respond in the following strict format (uppercase keywords):\n"
            "BETTER_VERSION: BASELINE / OPTIMIZED\n"
            "SUGGESTION: <short general suggestion if BASELINE is better, otherwise N/A>\n"
            f"{agent_suggestions}\n"
            "BASELINE_METRICS:\n"
            f"{_metrics_to_str(baseline_metrics)}\n\n"
            "OPTIMIZED_METRICS:\n"
            f"{_metrics_to_str(optimized_metrics)}\n"
        )

        return prompt
    
    def _create_advisor_improvement_prompt(self, 
                                         optimization_history: List[Dict[str, Any]],
                                         baseline_metrics: Dict[str, float],
                                         enabled_agents: Optional[List[str]] = None) -> str:
        """
        创建advisor改进提示词，针对不同的agent给出具体建议
        
        Args:
            optimization_history: 优化历史记录
            baseline_metrics: 基准指标
            enabled_agents: 启用的agent列表 ['aug', 'adaptive', 'hpo']
        """
        prompt = (
            "Analyze the following optimization attempts compared to a fixed baseline. "
            "Focus on understanding what parameter changes were effective or ineffective "
            "relative to the same baseline metrics.\n\n"
        )
        
        # 添加基准指标
        prompt += "Baseline metrics:\n"
        if isinstance(baseline_metrics, dict) and 'train' in baseline_metrics:
            prompt += "Training:\n"
            prompt += "\n".join([f"- {k}: {v:.4f}" for k, v in baseline_metrics['train'].items()])
            if 'val' in baseline_metrics:
                prompt += "\nValidation:\n"
                prompt += "\n".join([f"- {k}: {v:.4f}" for k, v in baseline_metrics['val'].items()])
        else:
            prompt += "\n".join([f"- {k}: {v:.4f}" for k, v in baseline_metrics.items()])
        
        # 添加优化历史
        for i, attempt in enumerate(optimization_history, 1):
            prompt += f"\n\nAttempt {i}:\n"
            prompt += "Parameter changes:\n"
            for param, values in attempt['param_changes'].items():
                prompt += f"- {param}: {values['from']} -> {values['to']}\n"
            
            prompt += "\nRelative changes from baseline:\n"
            if isinstance(attempt['optimized_metrics'], dict) and 'train' in attempt['optimized_metrics']:
                prompt += "Training:\n"
                for k in attempt['optimized_metrics']['train'].keys():
                    rel_change = ((attempt['optimized_metrics']['train'][k] - baseline_metrics['train'][k]) / 
                                abs(baseline_metrics['train'][k]) * 100)
                    prompt += f"- {k}: {rel_change:+.2f}%\n"
                
                if 'val' in attempt['optimized_metrics']:
                    prompt += "Validation:\n"
                    for k in attempt['optimized_metrics']['val'].keys():
                        rel_change = ((attempt['optimized_metrics']['val'][k] - baseline_metrics['val'][k]) / 
                                    abs(baseline_metrics['val'][k]) * 100)
                        prompt += f"- {k}: {rel_change:+.2f}%\n"
        
        prompt += "\nBased on these attempts, provide:\n"
        prompt += "1. Pattern analysis: [identify patterns in parameter changes and their effects]\n"
        prompt += "2. Effective changes: [list parameter changes that showed promise]\n"
        prompt += "3. Ineffective changes: [list parameter changes that were counterproductive]\n"
        
        # 根据启用的agent添加简化的建议格式
        if enabled_agents:
            prompt += "\nNow, provide concise recommendations for each enabled agent in the following format(The data augmentation agent is responsible for optimizing the training data, the strategy agent is responsible for optimizing the optimizer and loss function, and the HPO agent is responsible for optimizing other hyperparameters such as learning rate, weight decay, and class weights.):\n"
            
            if 'aug' in enabled_agents:
                prompt += "\nAUGMENTATION_AGENT_SUGGESTION:\n"
                prompt += "- Optimization direction: [One sentence to describe the optimization direction of data augmentation.]\n"
                prompt += "- Reasoning: [A brief explanation of why this direction is effective.]\n"
            
            if 'adaptive' in enabled_agents:
                prompt += "\nADAPTIVE_AGENT_SUGGESTION:\n"
                prompt += "- Optimization direction: [One sentence to describe the optimization direction of training strategy.]\n"
                prompt += "- Reasoning: [A brief explanation of why this direction is effective.]\n"
            
            if 'hpo' in enabled_agents:
                prompt += "\nHPO_AGENT_SUGGESTION:\n"
                prompt += "- Optimization direction: [One sentence to describe the optimization direction of hyperparameter optimization.]\n"
                prompt += "- Reasoning: [A brief explanation of why this direction is effective.]\n"
        else:
            # 如果没有指定agent，则提供通用建议
            prompt += "4. Recommended strategy: [A brief description of the optimization strategy.]\n"
        
        return prompt
    
    def _parse_comparison_result(self, result: str) -> Tuple[bool, str, Optional[Dict[str, str]]]:
        """
        解析比较结果
        
        Returns:
            Tuple[bool, str, Optional[Dict[str, str]]]:
                - 是否使用优化后的参数
                - 原因
                - 针对不同agent的建议字典（如果有）
        """
        lines = [l.strip() for l in result.strip().split("\n") if l.strip()]

        use_optimized = False
        suggestion = None
        reason = ""  # 简化，不再需要详细原因
        agent_suggestions = {}

        for line in lines:
            if line.upper().startswith("BETTER_VERSION"):
                if "OPTIMIZED" in line.upper():
                    use_optimized = True
            elif line.upper().startswith("SUGGESTION"):
                suggestion_text = line.split(":", 1)[-1].strip()
                if suggestion_text and suggestion_text.upper() != "N/A":
                    suggestion = suggestion_text
            elif line.upper().startswith("AUGMENTATION_AGENT"):
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

        return use_optimized, reason, agent_suggestions if agent_suggestions else None
    
    def judge_optimization(self, 
                         baseline_metrics: Dict[str, float],
                         optimized_metrics: Dict[str, float],
                         baseline_params: Dict[str, Any],
                         optimized_params: Dict[str, Any],
                         optimization_history: Optional[List[Dict[str, Any]]] = None,
                         enabled_agents: Optional[List[str]] = None) -> Tuple[bool, str, Optional[Dict[str, str]]]:
        """
        评判优化效果
        
        Args:
            baseline_metrics: 基准指标
            optimized_metrics: 优化后的指标
            baseline_params: 基准参数
            optimized_params: 优化后的参数
            optimization_history: 优化历史记录
            enabled_agents: 启用的agent列表 ['aug', 'adaptive', 'hpo']
            
        Returns:
            Tuple[bool, str, Optional[Dict[str, str]]]:
                - 是否使用优化后的参数
                - 判断原因
                - 针对不同agent的改进建议字典（如果需要）
        """
        # 更新尝试次数
        self.optimization_attempts += 1

        # ---- Debug: 输出收到的指标 ---- #
        print("[Judger] baseline_metrics:", baseline_metrics)
        print("[Judger] optimized_metrics:", optimized_metrics)

        # 缓存基准指标（如果还没有缓存）
        if self.baseline_metrics is None:
            self.baseline_metrics = baseline_metrics
        
        # 构建API请求
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # 创建比较提示词
        comparison_prompt = self._create_comparison_prompt(
            self.baseline_metrics,  # 使用缓存的基准指标
            optimized_metrics,
            baseline_params,
            optimized_params,
            self.optimization_attempts,
            enabled_agents
        )
        
        messages = [
            {"role": "system", "content": "You are a machine learning expert specializing in optimization evaluation."},
            {"role": "user", "content": comparison_prompt}
        ]
        
        data = {
            "model": "deepseek-chat",
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False
        }
        
        try:
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=data,
                timeout=self.timeout
            )
            response.raise_for_status()
            comparison_result = response.json()["choices"][0]["message"]["content"]
            
            use_optimized, reason, agent_suggestions = self._parse_comparison_result(comparison_result)
            
            # 如果优化效果不好且有优化历史，获取改进建议
            if not use_optimized and optimization_history:
                if self.optimization_attempts >= self.max_attempts:
                    return False, "达到最大优化尝试次数", None
                
                # 获取advisor改进建议
                improvement_prompt = self._create_advisor_improvement_prompt(
                    optimization_history,
                    self.baseline_metrics,  # 使用缓存的基准指标
                    enabled_agents  # 传入启用的agent列表
                )
                messages = [
                    {"role": "system", "content": "You are a machine learning expert specializing in optimization strategy improvement."},
                    {"role": "user", "content": improvement_prompt}
                ]
                
                data["messages"] = messages
                response = requests.post(
                    f"{self.api_base}/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=self.timeout
                )
                response.raise_for_status()
                improvement_suggestion = response.json()["choices"][0]["message"]["content"]
                
                return use_optimized, reason, improvement_suggestion
            
            return use_optimized, reason, agent_suggestions
            
        except requests.exceptions.RequestException as e:
            error_msg = f"调用LLM API时发生错误: {str(e)}"
            logging.error(error_msg)
            return False, error_msg, None
    
    def reset_attempts(self) -> None:
        """
        重置优化尝试次数和基准指标缓存
        """
        self.optimization_attempts = 0
        self.baseline_metrics = None