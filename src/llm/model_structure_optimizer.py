from __future__ import annotations

import json
import re
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import requests
import yaml


class LLMModelStructureOptimizer:
    """一个使用 LLM 自动优化模型结构的通用 Agent 框架。

    该 Agent 的职责概括为三步::

        1. 在一次完整训练结束后读取 **完整的指标历史** (metrics_history) 与 **原始配置文件** (config)。
        2. 结合指标趋势分析, 通过 LLM 生成 *改进后的模型代码* 与 *仅修改 model 字段的新配置*。
        3. 将生成结果保存到磁盘, 并返回新配置文件路径, 供上层系统继续下一轮训练。

    注意: 该实现只是一个"框架", 重点放在 **接口、流程、解析** 三个方面, 而不强行限定任何任务/模型类型。
    如果需要支持更多细节 (如特殊解析格式、运行安全沙箱等) , 可以在此基础上按需扩展。
    """

    DEFAULT_API_BASE = "https://api.deepseek.com/v1"

    _CODE_RE = re.compile(r"```python(.*?)```", re.DOTALL | re.IGNORECASE)
    _YAML_RE = re.compile(r"```yaml(.*?)```", re.DOTALL | re.IGNORECASE)

    def __init__(self, llm_config: Dict[str, Any]):
        self.api_key: str = llm_config["api_key"]
        self.temperature: float = llm_config.get("temperature", 0.3)
        self.max_tokens: int = llm_config.get("max_tokens", 4096)  # 增加默认token限制
        self.api_base: str = llm_config.get("api_base", self.DEFAULT_API_BASE)

    # ---------------------------------------------------------------------
    # PUBLIC API
    # ---------------------------------------------------------------------
    def optimize(
        self,
        metrics_history_path: Path,
        original_config_path: Path,
        output_model_path: Path | None = None,
        output_config_path: Path | None = None,
    ) -> Tuple[Path, Path]:
        """主调度入口。

        Args:
            metrics_history_path: 指向 *metrics_history.json* (或其它序列化格式) 的路径, 由 MetricsLogger 输出。
            original_config_path: 原始 YAML 配置文件路径。
            output_model_path: 新模型保存路径, 例如 ``src/models/optimized_model.py`` 。
            output_config_path: 新配置保存路径, 例如 ``config/optimized_config.yaml`` 。

        Returns:
            Tuple[Path, Path]: (new_model_file, new_config_file)
        """
        metrics_history = self._load_metrics_history(metrics_history_path)
        with open(original_config_path, "r", encoding="utf-8") as f:
            original_config = yaml.safe_load(f)

        # 读取现有模型代码 (可选, 给予 LLM 充分上下文)
        model_code_context = self._extract_model_code(original_config)

        if output_model_path is None:
            output_model_path = Path("src/models/optimized_model.py")
        else:
            output_model_path = Path(output_model_path)
        target_model_name = output_model_path.stem  # 用于 prompt，要求 YAML 中保持一致

        prompt = self._build_prompt(
            metrics_history, original_config, model_code_context, target_model_name
        )
        llm_response = self._query_llm(prompt)

        model_code, new_model_cfg = self._parse_llm_response(llm_response)

        # ------------------------------------------------------------------
        # 保存文件
        # ------------------------------------------------------------------
        output_model_path.parent.mkdir(parents=True, exist_ok=True)
        output_model_path.write_text(model_code, encoding="utf-8")

        # 生成新配置
        merged_config = self._merge_model_cfg(original_config, new_model_cfg)
        if output_config_path is None:
            output_config_path = Path(original_config_path).with_name(
                f"{Path(original_config_path).stem}_optimized.yaml"
            )
        else:
            output_config_path = Path(output_config_path)
        output_config_path.write_text(yaml.safe_dump(merged_config, allow_unicode=True), encoding="utf-8")

        logging.info(
            "[ModelStructureOptimizer] 新模型与配置已保存: %s, %s",
            output_model_path,
            output_config_path,
        )

        return output_model_path, output_config_path

    # ---------------------------------------------------------------------
    # INTERNALS
    # ---------------------------------------------------------------------
    @staticmethod
    def _load_metrics_history(path: Path) -> Dict[str, Any]:
        """支持 *.json*/* .jsonl* 两种格式。"""
        if not path.exists():
            raise FileNotFoundError(f"metrics_history 文件不存在: {path}")

        if path.suffix.lower() == ".jsonl":
            # 将 jsonl 读为 list[dict]
            lines = path.read_text(encoding="utf-8").splitlines()
            return [json.loads(l) for l in lines if l.strip()]
        else:
            return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _extract_model_code(config: Dict[str, Any]) -> str:
        """尝试根据 config.model.name 定位 Python 代码, 作为 context.

        如果无法定位, 返回空字符串, 不会阻塞流程。
        """
        model_section = config.get("model", {})
        model_name: str | None = model_section.get("name")
        if not model_name:
            return ""

        # 约定: 所有模型代码位于 ``src/models/{model_name.lower()}.py``
        candidate_path = Path(__file__).parent.parent / "models" / f"{model_name.lower()}.py"
        if candidate_path.exists():
            try:
                return candidate_path.read_text(encoding="utf-8")
            except Exception:
                return ""
        return ""

    # -------------------------- prompt & LLM -----------------------------
    def _build_prompt(
        self,
        metrics_history: Dict[str, Any],
        original_config: Dict[str, Any],
        model_code_context: str,
        model_name: str,
    ) -> str:
        """构造给 LLM 的提示词。"""
        prompt = (
            "You are an advanced machine learning architect. "
            "Your task is to design an *improved* model architecture based on the provided training metrics history and current config.\n"
            "Instructions:\n"
            "1. Focus **ONLY** on changing the model structure (e.g., layer types, layer sizes, activation functions).\n"
            "2. Do NOT touch any YAML sections other than `model`. The rest of the original config must stay **byte-wise identical**.\n"
            "3. Inside the `model` section, you may *only* change numeric values (ints/floats). **Do NOT rename keys, add/remove fields, or re-indent**. Keep the original structure and formatting.\n"
            "4. The `model.name` field **MUST** be exactly '{model_name}' (must match the python file name you create without `.py`).\n"
            "5. Respond with **TWO** markdown fenced blocks in the exact order:\n"
            "   1) ```python  # improved model code\n   ...\n   ```\n"
            f"   2) ```yaml   # ONLY the new 'model' section of config\n   model:\n     name: {model_name}\n     architecture:\n       <key>: <value>\n   ```\n"
            "6. Ensure YAML is valid and preserves the original indentation; no trailing commas.\n"
            "Do NOT output anything else outside the two fenced blocks.\n"
        )

        prompt += "\n# Current config (model section only)\n"
        prompt += yaml.safe_dump({"model": original_config.get("model", {})}, allow_unicode=True)

        prompt += "\n# Metrics history (truncated)\n"
        # 为防止 prompt 过长, 只取最近 3 条记录
        if isinstance(metrics_history, list):
            recent = metrics_history[-10:]  # 减少历史记录数量
            prompt += json.dumps(recent, indent=2) + "\n"
        else:
            prompt += json.dumps(metrics_history, indent=2) + "\n"

        if model_code_context:
            prompt += "\n# Current model code\n" + model_code_context[:1000] + "\n...\n"  # 进一步限制上下文长度

        return prompt

    def _query_llm(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "You are a machine learning expert."},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }

        try:
            resp = requests.post(
                f"{self.api_base}/chat/completions", headers=headers, json=data, timeout=60
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logging.error("LLM 调用失败: %s", e)
            raise

    # -------------------------- 解析 LLM 输出 ----------------------------
    def _parse_llm_response(self, content: str) -> Tuple[str, Dict[str, Any]]:
        """解析 LLM 输出, 提取 (python_code, model_cfg)。"""
        code_match = self._CODE_RE.search(content)
        yaml_match = self._YAML_RE.search(content)

        if not code_match or not yaml_match:
            raise ValueError(
                f"LLM 返回格式不符合约定, 需同时包含 python 与 yaml 代码块。\n"
                f"实际返回内容:\n{content}\n"
                f"Python 代码块匹配: {'成功' if code_match else '失败'}\n"
                f"YAML 代码块匹配: {'成功' if yaml_match else '失败'}"
            )

        model_code = code_match.group(1).strip()
        yaml_str = yaml_match.group(1)
        try:
            model_cfg_section = yaml.safe_load(yaml_str)
            if "model" not in model_cfg_section:
                raise ValueError(
                    f"YAML 代码块必须包含 model 顶级键。\n"
                    f"实际 YAML 内容:\n{yaml_str}"
                )
        except yaml.YAMLError as e:
            raise ValueError(f"解析 YAML 失败: {e}\nYAML 内容:\n{yaml_str}") from e

        return model_code, model_cfg_section["model"]

    # ------------------------- config merge ------------------------------
    @staticmethod
    def _merge_model_cfg(original: Dict[str, Any], new_model_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """仅替换原配置的 model 字段, 其它保持不变。"""
        merged = original.copy()
        merged["model"] = new_model_cfg
        return merged 