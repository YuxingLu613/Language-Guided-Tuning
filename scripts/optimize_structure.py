import argparse
from pathlib import Path
import sys

# 将项目根目录加入路径
ROOT_DIR = Path(__file__).parent.parent.resolve()
sys.path.append(str(ROOT_DIR))

import yaml  # noqa: E402
from src.llm import LLMModelStructureOptimizer  # noqa: E402


def main(args):
    # 读取 LLM 配置 (直接从原始 config 中提取)
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    llm_cfg = cfg["llm"]

    optimizer = LLMModelStructureOptimizer(llm_cfg)
    model_path, new_cfg_path = optimizer.optimize(
        metrics_history_path=Path(args.metrics),
        original_config_path=Path(args.config),
        output_model_path=Path(args.out_model) if args.out_model else None,
        output_config_path=Path(args.out_config) if args.out_config else None,
    )

    print(f"新模型已保存至: {model_path}\n新配置已保存至: {new_cfg_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 LLM 优化模型结构并生成新配置")
    parser.add_argument("--config", required=True, help="原始 YAML 配置文件路径")
    parser.add_argument("--metrics", required=True, help="metrics_history.json 或 .jsonl 路径")
    parser.add_argument("--out_model", help="新模型输出路径, 默认 src/models/optimized_model.py")
    parser.add_argument("--out_config", help="新配置输出路径, 默认与原配置同目录")

    args = parser.parse_args()
    main(args) 