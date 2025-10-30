#!/usr/bin/env python3
"""
脚本用于合并指定log目录下aug、strategy、hpo子目录中的metrics_history.json文件
使用方法: python merge_metrics.py <log_directory_path>
"""

import json
import os
import sys
import glob
from pathlib import Path
from typing import Dict, List, Any


def find_metrics_files(log_dir: str) -> Dict[str, List[str]]:
    """
    在指定目录下查找aug、strategy、hpo目录中的metrics_history.json文件
    
    Args:
        log_dir: 日志目录路径
        
    Returns:
        包含找到的文件路径的字典，键为目录类型，值为文件路径列表
    """
    log_path = Path(log_dir)
    if not log_path.exists():
        raise FileNotFoundError(f"目录不存在: {log_dir}")
    
    # 查找所有以_aug_、_strategy_、_hpo_开头的目录
    aug_dirs = list(log_path.glob("_aug_*"))
    strategy_dirs = list(log_path.glob("_strategy_*"))
    hpo_dirs = list(log_path.glob("*_hpo_*"))
    
    metrics_files = {
        "aug": [],
        "strategy": [],
        "hpo": []
    }
    
    # 查找aug目录中的metrics_history.json
    for aug_dir in aug_dirs:
        metrics_file = aug_dir / "metrics_history.json"
        if metrics_file.exists():
            metrics_files["aug"].append(str(metrics_file))
    
    # 查找strategy目录中的metrics_history.json
    for strategy_dir in strategy_dirs:
        metrics_file = strategy_dir / "metrics_history.json"
        if metrics_file.exists():
            metrics_files["strategy"].append(str(metrics_file))
    
    # 查找hpo目录中的metrics_history.json
    for hpo_dir in hpo_dirs:
        metrics_file = hpo_dir / "metrics_history.json"
        if metrics_file.exists():
            metrics_files["hpo"].append(str(metrics_file))
    
    return metrics_files


def load_metrics_file(file_path: str) -> Dict[str, Any]:
    """
    加载单个metrics_history.json文件
    
    Args:
        file_path: 文件路径
        
    Returns:
        加载的JSON数据
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"警告: 无法加载文件 {file_path}: {e}")
        return {}


def merge_metrics_data(metrics_files: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    合并所有metrics_history.json文件的数据
    
    Args:
        metrics_files: 包含文件路径的字典
        
    Returns:
        合并后的数据
    """
    merged_data = {}
    
    # 按顺序处理: strategy -> aug -> hpo
    order = ["strategy", "aug", "hpo"]
    
    for category in order:
        if category in metrics_files and metrics_files[category]:
            print(f"处理 {category} 目录中的文件...")
            
            for file_path in sorted(metrics_files[category]):
                print(f"  加载: {file_path}")
                data = load_metrics_file(file_path)
                
                if not data:
                    continue
                
                # 如果是第一个文件，直接使用其结构
                if not merged_data:
                    merged_data = data.copy()
                else:
                    # 合并数据，将新数据追加到现有数据后面
                    for key, value_list in data.items():
                        if key in merged_data and isinstance(merged_data[key], list) and isinstance(value_list, list):
                            merged_data[key].extend(value_list)
                        else:
                            # 如果键不存在或不是列表，直接替换
                            merged_data[key] = value_list
    
    return merged_data


def save_merged_metrics(merged_data: Dict[str, Any], output_path: str):
    """
    保存合并后的数据到新文件
    
    Args:
        merged_data: 合并后的数据
        output_path: 输出文件路径
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, indent=2, ensure_ascii=False)
        print(f"合并后的数据已保存到: {output_path}")
    except Exception as e:
        print(f"错误: 无法保存文件 {output_path}: {e}")


def main():
    """主函数"""
    if len(sys.argv) != 2:
        print("使用方法: python merge_metrics.py <log_directory_path>")
        print("示例: python merge_metrics.py ./cifar10_log")
        sys.exit(1)
    
    log_dir = sys.argv[1]
    
    try:
        print(f"正在处理目录: {log_dir}")
        
        # 查找metrics文件
        metrics_files = find_metrics_files(log_dir)
        
        # 打印找到的文件
        total_files = 0
        for category, files in metrics_files.items():
            print(f"{category} 目录中找到 {len(files)} 个metrics_history.json文件")
            total_files += len(files)
            for file_path in files:
                print(f"  - {file_path}")
        
        if total_files == 0:
            print("未找到任何metrics_history.json文件")
            sys.exit(1)
        
        # 合并数据
        print("\n开始合并数据...")
        merged_data = merge_metrics_data(metrics_files)
        
        if not merged_data:
            print("没有有效数据可以合并")
            sys.exit(1)
        
        # 保存合并后的数据
        output_path = os.path.join(log_dir, "merged_metrics_history.json")
        save_merged_metrics(merged_data, output_path)
        
        # 打印合并后的数据统计
        print("\n合并后的数据统计:")
        for key, value in merged_data.items():
            if isinstance(value, list):
                print(f"  {key}: {len(value)} 个数据点")
            else:
                print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 