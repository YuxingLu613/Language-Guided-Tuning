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
        self.timeout = config.get("request_timeout", 120)
        
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
        guidance: Optional[str] = None,
        previous_context: Optional[str] = None,
    ) -> str:
        """
        创建提示词 - 明确指定输出格式要求
        """
        # 构建参数名列表用于示例
        example_params = []
        for param_name in current_params.keys():
            formatted_name = self._format_param_name(param_name)
            example_params.append(f"{formatted_name}=X.XX")
        
        format_examples = [
            f"'{', '.join(example_params)}'",  # 主要格式
            "OR use multiple lines with 'param: value'",
            "OR use multiple lines with 'param=value'",
            "DO NOT use JSON format like {{'param': value}}"
        ]
        
        prompt = (
            "You are an expert ML trainer. Given the current parameters and metrics, "
            "evaluate the training status. Suggest precise changes in hyperparameters "
            "to improve training. Output changes in ONE of these formats:\n"
            f"{chr(10).join(['- ' + example for example in format_examples])}\n"
            "If no change needed, reply 'no change'.\n\n"
        )
        
        # 根据当前参数构建输出格式示例
        example_parts = []
        for param_name in current_params.keys():
            formatted_name = self._format_param_name(param_name)
            example_parts.append(f"{formatted_name}=X.XX")
        prompt += "'" + ", ".join(example_parts) + "'"
        
        prompt += ". If no change needed, reply 'no change'.\n\n"
        
        # 如有来自 Judger 的额外改进建议，放在最前面提醒但不改变既有格式
        if guidance:
            prompt += "\nNOTE FROM JUDGER (refined):\n"
            prompt += guidance.strip() + "\n"
        
        # Previous agent changes context
        if previous_context:
            prompt += "\nPREVIOUS AGENT CHANGES:\n"
            prompt += previous_context.strip() + "\n"

        # Add current parameter information
        prompt += "Current parameters:\n"
        for param_name, value in current_params.items():
            formatted_name = self._format_param_name(param_name)
            prompt += f"- {formatted_name}: {value}\n"
        
        # Add complete configuration context to avoid missing key information
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
                if k not in current_params:  # Avoid duplication with current_params
                    prompt += f"- {k}: {v}\n"

        # Add training metrics
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
        解析LLM建议 - 支持多种格式
        """
        if suggestion.strip().lower() == 'no change':
            return {}, False
        
        params = {}
        
        # 方法1: 先尝试提取XML标签内的内容
        xml_pattern = r'<(?:control|param_change)>(.*?)</(?:control|param_change)>'
        xml_match = re.search(xml_pattern, suggestion, re.DOTALL)
        
        content_to_parse = suggestion
        if xml_match:
            content_to_parse = xml_match.group(1)
            logging.debug(f"提取到XML标签内容: {content_to_parse}")
        
        # 方法2: 改进的正则表达式，支持冒号和等号两种分隔符
        patterns = [
            # 支持冒号分隔: learning_rate: 0.0003
            r'([a-zA-Z_][a-zA-Z_\d]*)\s*:\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?|\w+)',
            # 支持等号分隔: learning_rate=0.0003  
            r'([a-zA-Z_][a-zA-Z_\d]*)\s*=\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?|\w+)'
        ]
        
        all_matches = []
        for pattern in patterns:
            matches = re.findall(pattern, content_to_parse)
            all_matches.extend(matches)
        
        logging.debug(f"原始建议: {suggestion}")
        logging.debug(f"匹配到的参数对: {all_matches}")
        
        for param_name, value in all_matches:
            try:
                # 处理布尔值
                if value.lower() in {'true', 'false'}:
                    parsed_value = value.lower() == 'true'
                # 处理数字
                else:
                    parsed_value = float(value)
                
                # 参数名映射
                if param_name == 'lr':
                    params['learning_rate'] = parsed_value
                elif re.match(r'class_(\d+)_weight', param_name):
                    class_idx = re.match(r'class_(\d+)_weight', param_name).group(1)
                    params[f'weight_class_{class_idx}'] = parsed_value
                else:
                    params[param_name] = parsed_value
                    
            except (ValueError, IndexError) as e:
                logging.warning(f"解析参数失败: {param_name}={value}, 错误: {e}")
                continue
        
        logging.debug(f"解析后的参数: {params}")
        return params, bool(params)
    
    def get_advice(
        self,
        current_params: Dict[str, Any],
        metrics: Dict[str, float],
        context_cfg: Dict[str, Any],
        guidance: Optional[str] = None,
        previous_context: Optional[str] = None,
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
                "content": self._create_prompt(current_params, metrics, context_cfg, guidance, previous_context),
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
                timeout=self.timeout
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