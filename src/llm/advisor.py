import json
from typing import Dict, List, Any, Optional, Tuple
import requests
from pathlib import Path
import logging
import re

class LLMAdvisor:
    def __init__(self, config: Dict[str, Any]):
        """
        初始化LLM顾问
        
        Args:
            config: LLM配置信息
        """
        self.api_key = config["api_key"]
        self.temperature = config["temperature"]
        self.max_tokens = config["max_tokens"]
        self.api_base = "https://api.deepseek.com/v1"
        
    def _format_param_name(self, param_name: str) -> str:
        """
        格式化参数名称，用于在prompt中显示
        
        Args:
            param_name: 原始参数名称
            
        Returns:
            str: 格式化后的参数名称
        """
        # 将weight_class_X转换为class_X_weight
        if param_name.startswith('weight_class_'):
            class_idx = param_name.split('_')[-1]
            return f"class_{class_idx}_weight"
        # 将learning_rate转换为lr
        elif param_name == 'learning_rate':
            return 'lr'
        # 其他参数保持原样
        return param_name
    
    def _create_prompt(
        self,
        current_params: Dict[str, Any],
        metrics: Dict[str, float],
        context_cfg: Dict[str, Any],
    ) -> str:
        """
        创建提示词
        """
        # 基础提示词
        prompt = (
            "You are an expert ML trainer. Given the current parameters and metrics, "
            "evaluate the training status. Suggest precise changes in hyperparameters "
            "to improve training. Output changes in the format: "
        )
        
        # 根据当前参数构建输出格式示例
        example_parts = []
        for param_name in current_params.keys():
            formatted_name = self._format_param_name(param_name)
            example_parts.append(f"{formatted_name}=X.XX")
        prompt += "'" + ", ".join(example_parts) + "'"
        
        prompt += ". If no change needed, reply 'no change'.\n\n"
        
        # 添加当前参数信息
        prompt += "Current parameters:\n"
        for param_name, value in current_params.items():
            formatted_name = self._format_param_name(param_name)
            prompt += f"- {formatted_name}: {value}\n"
        
        # 添加完整配置上下文，避免缺少关键信息
        if context_cfg:
            ds_cfg = context_cfg.get("dataset", {})
            model_cfg = context_cfg.get("model", {})
            train_cfg = context_cfg.get("training", {})

            prompt += "\nDataset config:\n"
            for k, v in ds_cfg.items():
                prompt += f"- {k}: {v}\n"

            prompt += "\nModel config:\n"
            for k, v in model_cfg.items():
                prompt += f"- {k}: {v}\n"

            prompt += "\nTraining config (static fields):\n"
            for k, v in train_cfg.items():
                if k not in current_params:  # 避免与 current_params 重复
                    prompt += f"- {k}: {v}\n"

        # 添加训练指标
        prompt += "\nTraining metrics:\n"
        if isinstance(metrics, dict) and 'train' in metrics:
            prompt += "Training:\n"
            prompt += "\n".join([f"- {k}: {v:.4f}" for k, v in metrics['train'].items()])
            if 'val' in metrics:
                prompt += "\nValidation:\n"
                prompt += "\n".join([f"- {k}: {v:.4f}" for k, v in metrics['val'].items()])
        else:
            prompt += "\n".join([f"- {k}: {v:.4f}" for k, v in metrics.items()])
        
        return prompt
    
    def _parse_suggestion(self, suggestion: str) -> Tuple[Dict[str, float], bool]:
        """
        解析LLM建议
        
        Args:
            suggestion: LLM建议文本
            
        Returns:
            Tuple[Dict[str, float], bool]: 解析后的参数和是否需要更新的标志
        """
        if suggestion.strip().lower() == 'no change':
            return {}, False
        
        params = {}
        # 使用正则表达式匹配参数值对
        pattern = r'([a-zA-Z_\d]+)\s*=\s*([\d.]+)'
        matches = re.findall(pattern, suggestion)
        
        for param_name, value in matches:
            # 将lr转换回learning_rate
            if param_name == 'lr':
                params['learning_rate'] = float(value)
            # 将class_X_weight转换回weight_class_X
            elif param_name.startswith('class_') and param_name.endswith('_weight'):
                class_idx = param_name.split('_')[1]
                params[f'weight_class_{class_idx}'] = float(value)
            # 其他参数保持原样
            else:
                params[param_name] = float(value)
        
        return params, bool(params)
    
    def get_advice(
        self,
        current_params: Dict[str, Any],
        metrics: Dict[str, float],
        context_cfg: Dict[str, Any],
    ) -> Tuple[str, Dict[str, float], bool]:
        """
        获取LLM的调优建议
        
        Args:
            current_params: 当前的超参数
            metrics: 当前epoch的评估指标
            context_cfg: 上下文配置
            
        Returns:
            Tuple[str, Dict[str, float], bool]: 
                - LLM建议原文
                - 解析后的参数更新
                - 是否需要更新参数的标志
        """
        # 构建API请求
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        messages = [
            {
                "role": "system",
                "content": "You are a machine learning expert specializing in hyperparameter optimization.",
            },
            {
                "role": "user",
                "content": self._create_prompt(current_params, metrics, context_cfg),
            },
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
                timeout=30
            )
            response.raise_for_status()
            suggestion = response.json()["choices"][0]["message"]["content"]
            
            # 解析建议
            params, should_update = self._parse_suggestion(suggestion)
            return suggestion, params, should_update
            
        except requests.exceptions.RequestException as e:
            error_msg = f"调用LLM API时发生错误: {str(e)}"
            logging.error(error_msg)
            return error_msg, {}, False 