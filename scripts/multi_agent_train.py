import argparse
from pathlib import Path
import yaml
import torch
import sys
import re
import random
import copy
from datetime import datetime
import json
sys.path.append(str(Path(__file__).parent.parent.resolve()))

def set_seed(seed: int = 42):
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

_CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR_STD = (0.2023, 0.1994, 0.2010)

def base_cifar_transforms():
    from torchvision import transforms
    train_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
    ])
    return train_tf, test_tf

def load_cifar_data(cfg):
    from torchvision import datasets
    train_tf, test_tf = base_cifar_transforms()
    data_dir = Path(__file__).parent.parent / cfg['dataset']['path']
    data_dir.mkdir(parents=True, exist_ok=True)
    train_ds = datasets.CIFAR10(str(data_dir), train=True, download=True, transform=train_tf)
    test_ds = datasets.CIFAR10(str(data_dir), train=False, download=True, transform=test_tf)
    return train_ds, test_ds

def base_mnist_transforms():
    from torchvision import transforms
    train_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    return train_tf, test_tf

def load_mnist_data(cfg):
    from torchvision import datasets
    train_tf, test_tf = base_mnist_transforms()
    data_dir = Path(__file__).parent.parent / cfg['dataset']['path']
    data_dir.mkdir(parents=True, exist_ok=True)
    train_ds = datasets.MNIST(str(data_dir), train=True, download=True, transform=train_tf)
    test_ds = datasets.MNIST(str(data_dir), train=False, download=True, transform=test_tf)
    return train_ds, test_ds

def load_iris_data(cfg):
    from src.utils.iris_dataset import IrisDataset
    from torch.utils.data import random_split
    import torch
    
    # Load iris dataset
    dataset = IrisDataset()
    
    # Split into train/val based on config
    val_ratio = float(cfg['dataset'].get('validation_split', 0.2))
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    
    # Use seed for reproducible splits
    seed = cfg.get('training', {}).get('seed', 42)
    g = torch.Generator()
    g.manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=g)
    
    return train_ds, val_ds

def load_housing_data(cfg):
    from src.utils.housing_dataset import HousingDataset
    from torch.utils.data import random_split
    import torch
    
    # Load housing dataset
    csv_path = Path(__file__).parent.parent / cfg['dataset']['path']
    dataset = HousingDataset(str(csv_path), target_scale=1e5)
    
    # Split into train/val based on config
    val_ratio = float(cfg['dataset'].get('validation_split', 0.2))
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    
    # Use seed for reproducible splits
    seed = cfg.get('training', {}).get('seed', 42)
    g = torch.Generator()
    g.manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=g)
    
    return train_ds, val_ds

def load_water_quality_data(cfg):
    from src.utils.water_quality_dataset import WaterQualityDataset
    from torch.utils.data import random_split
    import torch
    
    # Load water quality dataset
    csv_path = Path(__file__).parent.parent / cfg['dataset']['path']
    dataset = WaterQualityDataset(str(csv_path))
    
    # Split into train/val based on config
    val_ratio = float(cfg['dataset'].get('validation_split', 0.2))
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    
    # Use seed for reproducible splits
    seed = cfg.get('training', {}).get('seed', 42)
    g = torch.Generator()
    g.manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=g)
    
    return train_ds, val_ds

def load_wine_quality_data(cfg):
    from src.utils.winequality_dataset import WineQualityDataset
    from torch.utils.data import random_split
    import torch
    
    # Load wine quality dataset
    csv_path = Path(__file__).parent.parent / cfg['dataset']['path']
    
    # Check if target_scale is provided in config
    target_scale = cfg['dataset'].get('target_scale', None)
    dataset = WineQualityDataset(str(csv_path), target_scale=target_scale)
    
    # Split into train/val based on config
    val_ratio = float(cfg['dataset'].get('validation_split', 0.2))
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    
    # Use seed for reproducible splits
    seed = cfg.get('training', {}).get('seed', 42)
    g = torch.Generator()
    g.manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=g)
    
    return train_ds, val_ds

from src.models.cnn_model import SimpleCNN
from src.trainer.adaptive_trainer import AdaptiveTrainer
from src.trainer.augmentation_transform_trainer import AugTransformTrainer
from src.trainer.trainer import Trainer
from src.utils.augmented_dataset import AugmentedDataset
from torch.utils.data import DataLoader

from src.llm.judger import LLMJudger
from src.llm.prompt_optimizer import LLMPromptOptimizer

# Helper: smart loader for config path or dict
def get_trainer_config(cfg):
    if isinstance(cfg, (str, Path)):
        with open(cfg, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return copy.deepcopy(cfg)
# ---- 1. Multi-agent sequential collaborative suggestion synthesis config (chain-based, supports ablation) ---- #

def parse_judger_suggestions(judger_output):
    """
    Parse the judger's output to extract suggestions for each agent.
    
    Args:
        judger_output: str or object - The output from the judger
        
    Returns:
        Dict[str, str] - A dictionary with agent-specific guidance
    """
    if not judger_output:
        return {}
    
    # 确保 judger_output 是字符串类型
    judger_output_str = str(judger_output) if judger_output is not None else ""
    
    agent_guidance = {}
    
    # Extract augmentation agent suggestion
    aug_match = re.search(r'AUGMENTATION_AGENT_SUGGESTION:\s*(.*?)(?=\n[A-Z_]+:|$)', judger_output_str, re.DOTALL)
    if aug_match:
        agent_guidance['aug'] = aug_match.group(1).strip()
    
    # Extract adaptive agent suggestion
    adaptive_match = re.search(r'ADAPTIVE_AGENT_SUGGESTION:\s*(.*?)(?=\n[A-Z_]+:|$)', judger_output_str, re.DOTALL)
    if adaptive_match:
        agent_guidance['adaptive'] = adaptive_match.group(1).strip()
    
    # Extract HPO agent suggestion
    hpo_match = re.search(r'HPO_AGENT_SUGGESTION:\s*(.*?)(?=\n[A-Z_]+:|$)', judger_output_str, re.DOTALL)
    if hpo_match:
        agent_guidance['hpo'] = hpo_match.group(1).strip()
    
    return agent_guidance

def multi_agent_chain_suggest(metrics, base_config, task_type, use_aug_agent, use_adaptive_agent, use_hpo_agent, log_dir, judger_output=None):
    """
    Sequential agent suggestions; returns (config, agent_logs, agent_log_file)
    Each agent will understand previous agent changes in prompt and consider decision compatibility
    
    Args:
        judger_output: str - The complete output from the judger to be parsed for agent-specific guidance
    """
    config = copy.deepcopy(base_config)
    agent_logs = []
    enabled_agents = []
    previous_changes = []  # Record previous agent changes
    
    # Parse judger suggestions for each agent
    agent_guidance = parse_judger_suggestions(judger_output) if judger_output else None
    
    # 记录每个agent的完整prompt内容
    agent_full_prompts = {}
    
    if use_aug_agent:
        aug_config = get_trainer_config(config)
        aug_trainer = AugTransformTrainer(None, aug_config)  # dummy model, for interface alignment
        aug_guidance = agent_guidance.get('aug') if agent_guidance else None
        
        # 确定数据类型和数据集名称
        dataset_name = aug_config.get('dataset', {}).get('name', 'unknown')
        # 根据数据集名称确定数据类型
        data_type = 'image'
        if dataset_name.lower() in ['housing', 'winequality', 'waterquality']:
            data_type = 'tabular'
        
        # 构建上下文信息，包含数据类型和数据集名称
        context_info = {
            'epoch': None,
            'data_type': data_type,
            'dataset': dataset_name
        }
        
        # 捕获完整的prompt内容
        aug_prompt = aug_trainer.chooser._build_prompt(metrics, context_info, aug_guidance, None)
        agent_full_prompts['aug'] = {
            'system_prompt': aug_trainer.chooser._SYSTEM,
            'user_prompt': aug_prompt
        }
        
        # 调用chooser时传递数据类型和数据集信息
        aug_result = aug_trainer.chooser.choose(metrics, context_info, previous_context=None, guidance=aug_guidance)
        if isinstance(aug_result, tuple):
            if len(aug_result) == 2:
                aug_suggestion, should_use = aug_result
                aug_params = None  # LLMAugmentationChooser doesn't return parameters
            else:
                aug_suggestion = aug_params = should_use = None
        else:
            aug_suggestion = aug_params = should_use = None
        if should_use is True and aug_suggestion and aug_suggestion != 'none':
            config.setdefault('augmentation', {})['policy'] = aug_suggestion
            previous_changes.append(f"Augmentation Agent suggested using data augmentation strategy: {aug_suggestion}")
        agent_logs.append({
            'agent': 'aug', 
            'suggestion': aug_suggestion, 
            'params': aug_params, 
            'should_use': should_use,
            'full_prompt': agent_full_prompts['aug']
        })
        enabled_agents.append('aug')
    
    if use_adaptive_agent:
        strat_config = get_trainer_config(config)
        strat_agent = AdaptiveTrainer(None, task_type, strat_config)
        # Include previous agent changes in prompt
        previous_context = ""
        if previous_changes:
            previous_context = f"Before you, other agents have made the following changes: {'; '.join(previous_changes)}. Please consider the compatibility between your decisions before making your decision."
        
        adaptive_guidance = agent_guidance.get('adaptive') if agent_guidance else None
        
        # 捕获完整的prompt内容
        adaptive_prompt = strat_agent.agent._build_prompt(task_type, {}, metrics, strat_config, adaptive_guidance, previous_context)
        agent_full_prompts['adaptive'] = {
            'system_prompt': "You are a highly skilled ML practitioner specialising in optimising training strategies across diverse tasks.",
            'user_prompt': adaptive_prompt
        }
        
        suggestion, new_strategy, _ = strat_agent.agent.decide_strategy(task_type, {}, metrics, strat_config, previous_context=previous_context, guidance=adaptive_guidance)
        if new_strategy:
            config['training'].update(new_strategy)
            changes_desc = []
            for key, value in new_strategy.items():
                changes_desc.append(f"{key}: {value}")
            previous_changes.append(f"Adaptive Agent suggested adjusting training strategy: {', '.join(changes_desc)}")
        agent_logs.append({
            'agent': 'strategy', 
            'suggestion': suggestion, 
            'params': new_strategy, 
            'previous_context': previous_context,
            'full_prompt': agent_full_prompts['adaptive']
        })
        enabled_agents.append('adaptive')
    
    if use_hpo_agent:
        hpo_config = get_trainer_config(config)
        hpo_trainer = Trainer(None, hpo_config)
        # Include previous agent changes in prompt
        previous_context = ""
        if previous_changes:
            previous_context = f"Before you, other agents have made the following changes: {'; '.join(previous_changes)}. Please consider the compatibility between your decisions before making your decision."
        
        hpo_guidance = agent_guidance.get('hpo') if agent_guidance else None
        
        # 捕获完整的prompt内容
        hpo_prompt = hpo_trainer.advisor._create_prompt({}, metrics, hpo_config, hpo_guidance, previous_context)
        agent_full_prompts['hpo'] = {
            'system_prompt': "You are a machine learning expert specializing in hyperparameter optimization.",
            'user_prompt': hpo_prompt
        }
        
        suggestion, new_params, _ = hpo_trainer.advisor.get_advice({}, metrics, hpo_config, previous_context=previous_context, guidance=hpo_guidance)
        if new_params:
            config['training'].update(new_params)
            changes_desc = []
            for key, value in new_params.items():
                changes_desc.append(f"{key}: {value}")
            previous_changes.append(f"HPO Agent suggested adjusting hyperparameters: {', '.join(changes_desc)}")
        agent_logs.append({
            'agent': 'hpo', 
            'suggestion': suggestion, 
            'params': new_params, 
            'previous_context': previous_context,
            'full_prompt': agent_full_prompts['hpo']
        })
        enabled_agents.append('hpo')
    
    agent_log_file = Path(log_dir) / f"agent_chain_epoch_{metrics.get('epoch', '?')}.json"
    with open(agent_log_file, 'w', encoding='utf-8') as f:
        json.dump({
            'enabled_agents': enabled_agents, 
            'logs': agent_logs, 
            'previous_changes': previous_changes,
            'agent_full_prompts': agent_full_prompts  # 添加完整的agent prompt内容到日志
        }, f, ensure_ascii=False, indent=2)
    return config, agent_logs, agent_log_file


def run_multichain_epoch(epoch_idx, context, judger, args):
    print(f"\n=== EPOCH {epoch_idx} ===")
    model = context['model']
    optimizer_pt = context['optimizer']
    cfg = context['config']
    train_loader = context['train_loader']
    val_loader = context['val_loader']
    log_dir = cfg['logging']['log_dir']
    
    # 重置judger的状态，确保每个epoch都有新的尝试机会
    judger.reset_state()
    
    # 确定任务类型
    dataset_name = cfg['dataset']['name'].lower()
    task_type = 'regression' if dataset_name in ['housing', 'winequality'] else 'classification'
    
    # 保存当前模型状态（状态a）
    state_a = copy.deepcopy(model.state_dict())
    optimizer_state_a = copy.deepcopy(optimizer_pt.state_dict())
    
    # 对于epoch 0，直接训练一轮并返回
    if epoch_idx == 0:
        print(f"[EPOCH 0] Direct training with initial config")
        a_trainer = AdaptiveTrainer(model, task_type, get_trainer_config(cfg))
        a_trainer.config['training']['epochs'] = 1
        metrics_train = a_trainer._run_epoch(train_loader, train=True)
        metrics_val = a_trainer._run_epoch(val_loader, train=False)
        metrics_all = {"train": metrics_train, "val": metrics_val, "epoch": epoch_idx}
        
        # 保存本轮结果到上下文，供下一轮使用
        context['prev_metrics'] = metrics_all
        context['prev_config'] = copy.deepcopy(cfg)
        
        # 保存结果
        outcome = {
            'epoch': epoch_idx,
            'flags': {
                'use_adaptive_agent': args.use_adaptive_agent,
                'use_aug_agent': args.use_aug_agent,
                'use_hpo_agent': args.use_hpo_agent,
            },
            'use_optimized': False,  # epoch 0 不使用优化
            'judger_reason': "Epoch 0: direct training without optimization",
            'final_metrics': metrics_all,
            'final_config': cfg
        }
        
        final_log_file = Path(log_dir) / f'epoch_{epoch_idx}_final_result.json'
        with open(final_log_file, 'w', encoding='utf-8') as f:
            json.dump(outcome, f, ensure_ascii=False, indent=2)
        
        return False, cfg  # epoch 0 不改变配置
    
    # 对于后续epoch，使用上一轮的结果进行优化决策
    # 从上下文中获取上一轮的指标和配置
    prev_metrics = context.get('prev_metrics')
    prev_config = context.get('prev_config')
    
    if not prev_metrics or not prev_config:
        print(f"Warning: No previous metrics or config found for epoch {epoch_idx}, skipping optimization")
        # 如果找不到上一轮结果，仍然训练一轮并保存结果
        a_trainer = AdaptiveTrainer(model, task_type, get_trainer_config(cfg))
        a_trainer.config['training']['epochs'] = 1
        metrics_train = a_trainer._run_epoch(train_loader, train=True)
        metrics_val = a_trainer._run_epoch(val_loader, train=False)
        metrics_all = {"train": metrics_train, "val": metrics_val, "epoch": epoch_idx}
        
        # 保存本轮结果
        context['prev_metrics'] = metrics_all
        context['prev_config'] = copy.deepcopy(cfg)
        
        return False, cfg
    
    print(f"[EPOCH {epoch_idx}] Using previous epoch results for optimization")
    
    # 使用上一轮的结果让agent给出优化建议
    new_cfg, agent_logs, agent_chain_path = multi_agent_chain_suggest(
        prev_metrics, prev_config, task_type,  # 使用上一轮的指标和配置
        args.use_aug_agent, args.use_adaptive_agent, args.use_hpo_agent, log_dir, None)
    
    # 使用状态a和agent建议的配置训练一轮（得到状态b）
    model.load_state_dict(state_a)
    optimizer_pt.load_state_dict(optimizer_state_a)
    a_trainer_b = AdaptiveTrainer(model, task_type, get_trainer_config(new_cfg))
    a_trainer_b.config['training']['epochs'] = 1
    metrics_b_train = a_trainer_b._run_epoch(train_loader, train=True)
    metrics_b_val = a_trainer_b._run_epoch(val_loader, train=False)
    metrics_b = {"train": metrics_b_train, "val": metrics_b_val, "epoch": epoch_idx}
    state_b = copy.deepcopy(model.state_dict())
    
    # 使用状态a和原配置训练一轮（得到状态c）
    model.load_state_dict(state_a)
    optimizer_pt.load_state_dict(optimizer_state_a)
    a_trainer_c = AdaptiveTrainer(model, task_type, get_trainer_config(cfg))
    a_trainer_c.config['training']['epochs'] = 1
    metrics_c_train = a_trainer_c._run_epoch(train_loader, train=True)
    metrics_c_val = a_trainer_c._run_epoch(val_loader, train=False)
    metrics_c = {"train": metrics_c_train, "val": metrics_c_val, "epoch": epoch_idx}
    state_c = copy.deepcopy(model.state_dict())
    
    # 获取启用的智能体列表
    enabled_agents = []
    if args.use_aug_agent: enabled_agents.append('aug')
    if args.use_adaptive_agent: enabled_agents.append('adaptive')
    if args.use_hpo_agent: enabled_agents.append('hpo')
    
    # 初始化优化历史
    optimization_history = []
    judger_rounds = []
    attempt_num = 0
    final_use = False
    final_config = cfg
    final_state = state_c  # 默认使用状态c
    judger_reason = None
    
    # 第一轮比较
    use_optimized, reason, agent_suggestions = judger.judge_optimization(
        metrics_c, metrics_b,  # 比较状态c和状态b
        cfg['training'], new_cfg['training'],
        optimization_history, enabled_agents
    )
    
    judger_rounds.append({
        'attempt': attempt_num,
        'use_optimized': use_optimized,
        'reason': reason,
        'agent_suggestions': agent_suggestions
    })
    
    optimization_history.append({
        'param_changes': {k: {'from': cfg['training'].get(k), 'to': new_cfg['training'].get(k)} 
                         for k in set(cfg['training']) | set(new_cfg['training'])},
        'original_metrics': metrics_c,
        'optimized_metrics': metrics_b
    })
    
    # 如果Judger认为b好，直接接受
    if use_optimized:
        final_use = True
        final_config = new_cfg
        final_state = state_b
        judger_reason = reason
        print(f"[EPOCH {epoch_idx}] Judger accepted optimization: {reason}")
    else:
        # 如果Judger认为c好，尝试改进建议（最多5次）
        current_cfg = new_cfg
        current_metrics_b = metrics_b
        current_state_b = state_b
        
        for attempt_num in range(1, 6):  # 最多5次尝试
            if not agent_suggestions:
                break  # 如果没有改进建议，直接退出
                
            # 直接使用judger的建议，不再使用prompt optimizer
            judger_rounds[-1]['refined'] = agent_suggestions
            
            # 基于judger建议重新生成配置
            refined_cfg, _agent_logs2, _agent_chain_path2 = multi_agent_chain_suggest(
                prev_metrics, cfg, task_type,  # 仍然使用上一轮的结果
                args.use_aug_agent, args.use_adaptive_agent, args.use_hpo_agent, 
                log_dir, agent_suggestions)
            
            # 记录优化后的agent logs
            try:
                with open(_agent_chain_path2, 'r', encoding='utf-8') as f:
                    refined_agent_payload = json.load(f)
                judger_rounds[-1]['refined_agent_logs'] = refined_agent_payload
            except Exception as e:
                print(f"Warning: Failed to read refined agent logs: {e}")
            
            # 使用状态a和新的优化配置训练一轮
            model.load_state_dict(state_a)
            optimizer_pt.load_state_dict(optimizer_state_a)
            a_trainer_refined = AdaptiveTrainer(model, task_type, get_trainer_config(refined_cfg))
            a_trainer_refined.config['training']['epochs'] = 1
            metrics_refined_train = a_trainer_refined._run_epoch(train_loader, train=True)
            metrics_refined_val = a_trainer_refined._run_epoch(val_loader, train=False)
            metrics_refined = {"train": metrics_refined_train, "val": metrics_refined_val, "epoch": epoch_idx}
            state_refined = copy.deepcopy(model.state_dict())
            
            # 比较新的优化结果与状态c
            use_refined, reason_refined, agent_suggestions_refined = judger.judge_optimization(
                metrics_c, metrics_refined,
                cfg['training'], refined_cfg['training'],
                optimization_history, enabled_agents
            )
            
            judger_rounds.append({
                'attempt': attempt_num,
                'use_optimized': use_refined,
                'reason': reason_refined,
                'agent_suggestions': agent_suggestions_refined
            })
            
            optimization_history.append({
                'param_changes': {k: {'from': cfg['training'].get(k), 'to': refined_cfg['training'].get(k)} 
                                 for k in set(cfg['training']) | set(refined_cfg['training'])},
                'original_metrics': metrics_c,
                'optimized_metrics': metrics_refined
            })
            
            # 如果改进后的建议被接受
            if use_refined:
                final_use = True
                final_config = refined_cfg
                final_state = state_refined
                judger_reason = reason_refined
                print(f"[EPOCH {epoch_idx}] Judger accepted refined optimization (attempt {attempt_num}): {reason_refined}")
                break
            
            # 准备下一次迭代
            agent_suggestions = agent_suggestions_refined
            current_cfg = refined_cfg
            current_metrics_b = metrics_refined
            current_state_b = state_refined
        
        # 如果所有尝试都失败，使用状态c和原配置
        if not final_use:
            final_use = False
            final_config = cfg
            final_state = state_c
            judger_reason = "All optimization attempts rejected by Judger"
            print(f"[EPOCH {epoch_idx}] All optimization attempts rejected, using original config")
    
    # 应用最终选择的模型状态
    model.load_state_dict(final_state)
    
    # 根据最终配置调整优化器
    if final_use and final_config != cfg:
        print(f"[EPOCH {epoch_idx}] Updating optimizer based on new configuration")
        # 获取新的学习率和其他优化器参数
        new_lr = final_config['training'].get('learning_rate', cfg['training']['learning_rate'])
        new_weight_decay = final_config['training'].get('weight_decay', 
                                                       cfg['training'].get('weight_decay', 0))
        
        # 更新优化器参数
        for param_group in optimizer_pt.param_groups:
            param_group['lr'] = new_lr
            if 'weight_decay' in param_group:
                param_group['weight_decay'] = new_weight_decay
        
        print(f"  - Learning rate updated to: {new_lr}")
        if new_weight_decay != 0:
            print(f"  - Weight decay updated to: {new_weight_decay}")
            
        # 应用其他训练参数变更
        for key, value in final_config['training'].items():
            if key not in ['learning_rate', 'weight_decay'] and key in cfg['training'] and value != cfg['training'][key]:
                cfg['training'][key] = value
                print(f"  - {key} updated to: {value}")
                
        # 确保更新后的配置被保存和使用
        context['config'] = copy.deepcopy(final_config)
    
    # 保存本轮结果到上下文，供下一轮使用
    context['prev_metrics'] = metrics_b if final_use else metrics_c
    context['prev_config'] = copy.deepcopy(final_config)
    
    # 记录详细的决策日志
    try:
        with open(agent_chain_path, 'r', encoding='utf-8') as f:
            chain_payload = json.load(f)
    except Exception:
        chain_payload = {'enabled_agents': [], 'logs': []}
    
    chain_payload['judger_rounds'] = judger_rounds
    chain_payload['prev_metrics_used'] = prev_metrics  # 记录使用了哪一轮的指标
    chain_payload['prev_config_used'] = prev_config    # 记录使用了哪一轮的配置
    
    with open(agent_chain_path, 'w', encoding='utf-8') as f:
        json.dump(chain_payload, f, ensure_ascii=False, indent=2)
    
    # 保存最终结果
    outcome = {
        'epoch': epoch_idx,
        'flags': {
            'use_adaptive_agent': args.use_adaptive_agent,
            'use_aug_agent': args.use_aug_agent,
            'use_hpo_agent': args.use_hpo_agent,
        },
        'use_optimized': final_use,
        'judger_reason': judger_reason,
        'final_metrics': metrics_b if final_use else metrics_c,
        'final_config': final_config,
        'optimization_history': optimization_history,
        'prev_metrics_used': prev_metrics,  # 记录使用了哪一轮的指标
        'prev_config_used': prev_config     # 记录使用了哪一轮的配置
    }
    
    final_log_file = Path(log_dir) / f'epoch_{epoch_idx}_final_result.json'
    with open(final_log_file, 'w', encoding='utf-8') as f:
        json.dump(outcome, f, ensure_ascii=False, indent=2)
    
    print(f"[EPOCH {epoch_idx}] Final decision: use_optimized={final_use}")
    return final_use, final_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--model_path', default=None)
    parser.add_argument('--use_adaptive_agent', action='store_true')
    parser.add_argument('--use_aug_agent', action='store_true')
    parser.add_argument('--use_hpo_agent', action='store_true')
    args = parser.parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    base_log_dir = Path(cfg['logging']['log_dir'])
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    history_dir = base_log_dir / f'history_{ts}'
    history_dir.mkdir(parents=True, exist_ok=True)
    cfg['logging']['log_dir'] = str(history_dir)
    # Disable Trainer from creating _strategy/_aug subdirectories
    cfg['logging']['disable_trainer_subdir'] = True
    set_seed(cfg.get('training', {}).get('seed', 42))
    
    # Load dataset based on config
    dataset_name = cfg['dataset']['name'].lower()
    if dataset_name == 'mnist':
        train_ds, test_ds = load_mnist_data(cfg)
    elif dataset_name == 'cifar10':
        train_ds, test_ds = load_cifar_data(cfg)
    elif dataset_name == 'iris':
        train_ds, test_ds = load_iris_data(cfg)
    elif dataset_name == 'housing':
        train_ds, test_ds = load_housing_data(cfg)
    elif dataset_name == 'waterquality':
        train_ds, test_ds = load_water_quality_data(cfg)
    elif dataset_name == 'winequality':
        train_ds, test_ds = load_wine_quality_data(cfg)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Supported datasets: MNIST, CIFAR10, Iris, Housing, WaterQuality, WineQuality")
    
    train_loader = DataLoader(train_ds, batch_size=cfg['training']['batch_size'], shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(test_ds, batch_size=cfg['training']['batch_size'], shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
    
    # Create model based on dataset type
    if dataset_name == 'iris' or dataset_name == 'waterquality':
        from src.models.custom_model import CustomModel
        model = CustomModel(
            input_dim=cfg['dataset']['input_dim'],
            hidden_dims=cfg['model']['architecture']['hidden_dims'],
            output_dim=cfg['dataset']['num_classes'],
            dropout=cfg['model']['architecture'].get('dropout', 0.3),
        )
    elif dataset_name == 'housing' or dataset_name == 'winequality':
        from src.models.custom_model import CustomModel
        # Housing and WineQuality are regression tasks, output_dim=1
        model = CustomModel(
            input_dim=cfg['dataset']['input_dim'],
            hidden_dims=cfg['model']['architecture']['hidden_dims'],
            output_dim=1,  # Regression output
            dropout=cfg['model']['architecture'].get('dropout', 0.5),
        )
    else:
        # For MNIST and CIFAR10, use SimpleCNN
        model = SimpleCNN(
            input_shape=cfg['dataset']['input_shape'],
            conv_layers=cfg['model']['architecture']['conv_layers'],
            pool_size=cfg['model']['architecture'].get('pool_size', 2),
            fc_layers=cfg['model']['architecture']['fc_layers'],
            num_classes=cfg['dataset']['num_classes'],
            dropout=cfg['model']['architecture'].get('dropout', 0.25),
        )
    optimizer_pt = torch.optim.Adam(model.parameters(), lr=cfg['training']['learning_rate'])
    context = {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'model': model,
        'optimizer': optimizer_pt,
        'config': cfg,
    }
    judger = LLMJudger(cfg['llm'])
    # Removed optimizer_agent as it's no longer needed
    epochs = args.epochs if args.epochs else cfg['training']['epochs']
    for epoch in range(epochs):
        use_optimized, final_config = run_multichain_epoch(epoch, context, judger, args)
        if use_optimized:
            context['config'] = final_config
    print("[All epochs finished]")

if __name__ == '__main__':
    main()
