import torch
import json
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Dict, Any, Optional, Union
import yaml
import copy
from tqdm import tqdm
import logging
from pathlib import Path
from ..llm.advisor import LLMAdvisor
from ..llm.judger import LLMJudger
from ..llm.prompt_optimizer import LLMPromptOptimizer
from ..utils.metrics_logger import MetricsLogger
from sklearn.metrics import precision_score, recall_score

class Trainer:
    def __init__(self, model: nn.Module, config_path: Union[str, Path]):
        """
        初始化训练器
        
        Args:
            model: PyTorch模型
            config_path: 配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
            
        # 设置设备
        if self.config['training']['device'] == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(self.config['training']['device'])
        
        print(f"\n使用设备: {self.device}")
        
        self.model = model
        self.model.to(self.device)
        
        # 初始化训练参数
        self.current_params = {
            'learning_rate': self.config['training']['learning_rate']
        }

        # 如果提供了类别权重则读取，否则使用默认 1.0（兼容非分类任务）
        default_weights = {i: 1.0 for i in range(10)}
        class_weights_cfg = self.config['training'].get('class_weights', default_weights)

        for class_idx, weight in class_weights_cfg.items():
            self.current_params[f'weight_class_{class_idx}'] = weight
        
        # 初始化优化器
        self.optimizer = self._init_optimizer()
        
        # 初始化调度器（将在train方法中设置）
        self.scheduler = None
        
        # 初始化LLM顾问、评判器以及 Prompt Optimizer
        self.advisor = LLMAdvisor(self.config['llm'])
        self.judger = LLMJudger(self.config['llm'])
        self.prompt_optimizer = LLMPromptOptimizer(self.config['llm'])
        
        # 初始化指标记录器
        self.metrics_logger = MetricsLogger(
            self.config['logging']['log_dir'],
            self.config
        )

        # --------------------------------------------------
        # Agent 输出日志简易方法
        # --------------------------------------------------

    def _log_agent_output(self, agent: str, payload: dict):
        """将每个 Agent 的输出追加到独立文件中 (JSON lines)。"""
        log_file = Path(self.config['logging']['log_dir']) / f"{agent.lower()}_outputs.jsonl"
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        
        # 优化历史记录
        self.optimization_history = []
        
        # 缓存的基准结果
        self.cached_baseline = None
        
        # 设置日志
        log_dir = Path(self.config['logging']['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            filename=log_dir / "training.log",
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )
        
        # 记录当前学习率
        self.current_lr = self.config['training']['learning_rate']
        
    def _init_optimizer(self) -> torch.optim.Optimizer:
        """
        初始化优化器
        """
        optimizer_name = self.config['training']['optimizer'].lower()
        lr = self.current_params['learning_rate']
        
        if optimizer_name == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer_name == 'sgd':
            return torch.optim.SGD(self.model.parameters(), lr=lr)
        else:
            raise ValueError(f"不支持的优化器: {optimizer_name}")
    
    def _update_parameters(self, new_params: Dict[str, float]) -> None:
        """
        更新训练参数
        
        Args:
            new_params: 新的参数值
        """
        # 更新学习率
        if 'learning_rate' in new_params:
            self.current_params['learning_rate'] = new_params['learning_rate']
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_params['learning_rate']
        
        # 更新类别权重
        for param_name, value in new_params.items():
            if param_name.startswith('weight_class_'):
                self.current_params[param_name] = value
    
    def _calculate_metrics(self, outputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        计算评估指标
        """
        # 确保数据在CPU上且转换为numpy数组
        predictions = outputs.argmax(dim=1).cpu().numpy()
        targets = targets.cpu().numpy()
        
        return {
            'accuracy': float((predictions == targets).mean()),
            'precision': float(precision_score(targets, predictions, average='macro')),
            'recall': float(recall_score(targets, predictions, average='macro'))
        }
    
    def train_epoch(self, train_loader: DataLoader, epoch: Optional[int] = None, is_evaluation: bool = False) -> Dict[str, float]:
        """
        训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
            epoch: 当前epoch数（可选，用于显示进度）
            is_evaluation: 是否是评估模式
        """
        self.model.train()
        total_loss = 0.0
        all_outputs: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []
        
        desc = f'Epoch {epoch}' if epoch is not None else 'Evaluating'
        with tqdm(train_loader, desc=desc) as pbar:
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                
                # 使用类别权重
                weights = torch.tensor(
                    [self.current_params[f'weight_class_{i}'] for i in range(10)],
                    device=self.device
                )
                loss = nn.functional.cross_entropy(output, target, weight=weights)
                
                if not is_evaluation:
                    loss.backward()
                    self.optimizer.step()
                
                total_loss += loss.item()
                all_outputs.append(output.detach())
                all_targets.append(target)
                
                # 更新进度条
                pbar.set_postfix({
                    'loss': total_loss / (batch_idx + 1),
                    'lr': self.current_params['learning_rate']
                })
        
        # 计算整体指标
        all_outputs_tensor = torch.cat(all_outputs)
        all_targets_tensor = torch.cat(all_targets)
        metrics = self._calculate_metrics(all_outputs_tensor, all_targets_tensor)
        metrics['loss'] = total_loss / len(train_loader)
        
        return metrics
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        验证模型
        """
        self.model.eval()
        total_loss = 0.0
        all_outputs: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                # 使用类别权重
                weights = torch.tensor(
                    [self.current_params[f'weight_class_{i}'] for i in range(10)],
                    device=self.device
                )
                loss = nn.functional.cross_entropy(output, target, weight=weights)
                
                total_loss += loss.item()
                all_outputs.append(output)
                all_targets.append(target)
        
        all_outputs_tensor = torch.cat(all_outputs)
        all_targets_tensor = torch.cat(all_targets)
        metrics = self._calculate_metrics(all_outputs_tensor, all_targets_tensor)
        metrics['loss'] = total_loss / len(val_loader)
        
        return metrics
    
    def _cache_baseline_results(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> Dict[str, Any]:
        """
        缓存使用当前参数的训练结果作为基准
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            
        Returns:
            Dict[str, Any]: 基准结果
        """
        # 保存当前模型状态
        model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        optimizer_state = {k: v.clone() if isinstance(v, torch.Tensor) else v 
                         for k, v in self.optimizer.state_dict().items()}
        
        # 获取基准结果
        train_metrics = self.train_epoch(train_loader, is_evaluation=True)
        if val_loader is not None:
            val_metrics = self.validate(val_loader)
            metrics = {
                'train': train_metrics,
                'val': val_metrics
            }
        else:
            metrics = {'train': train_metrics}
        
        # 恢复模型状态
        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(optimizer_state)
        
        return {
            'metrics': metrics,
            'params': self.current_params.copy(),
            'model_state': model_state,
            'optimizer_state': optimizer_state
        }

    def _evaluate_optimization(self,
                               train_loader: DataLoader,
                               val_loader: Optional[DataLoader] = None,
                               new_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        根据给定的新参数执行完整的评估流程：
        1. 以当前参数再训练 1 轮，获得 baseline（可被后续继续使用）。
        2. 在不改变权重的前提下回到 epoch t 结束时的模型状态，
           应用新的参数并训练 1 轮，获得 optimized 结果。
        3. 使用 LLMJudger 判断 optimized 与 baseline 的优劣。
           如果 optimized 更好，则采用 optimized 的参数与权重；
           否则保留 baseline，并在必要时继续尝试直至达到 max_attempts。
        """

        # ------- Step 0: 保存 epoch t 结束时的原始状态 ------- #
        original_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        original_optimizer_state = {k: v.clone() if isinstance(v, torch.Tensor) else v
                                    for k, v in self.optimizer.state_dict().items()}
        baseline_params = self.current_params.copy()

        # ------- Step 1: 使用原参数训练 1 轮，作为 baseline ------- #
        baseline_train_metrics = self.train_epoch(train_loader)
        if val_loader is not None:
            baseline_val_metrics = self.validate(val_loader)
            baseline_metrics: Dict[str, Any] = {
                'train': baseline_train_metrics,
                'val': baseline_val_metrics
            }
        else:
            baseline_metrics = {'train': baseline_train_metrics}

        # 保存 baseline 训练后的权重（可能成为最终模型）
        baseline_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        baseline_optimizer_state = {k: v.clone() if isinstance(v, torch.Tensor) else v
                                     for k, v in self.optimizer.state_dict().items()}

        # ------- Step 2: 尝试使用优化参数 ------- #
        attempt_params = new_params or {}
        attempt_idx = 0
        final_use_optimized = False
        final_reason: str = ""
        final_suggestion: Optional[str] = None
        final_metrics: Dict[str, Any] = baseline_metrics

        while attempt_idx < self.judger.max_attempts and attempt_params:
            attempt_idx += 1

            # 回到 epoch t 结束时的权重与优化器状态
            self.model.load_state_dict(original_model_state)
            self.optimizer.load_state_dict(original_optimizer_state)

            # 先恢复参数，再应用新的参数修改
            self.current_params = baseline_params.copy()
            self._update_parameters(attempt_params)

            # 使用优化后参数训练 1 轮
            opt_train_metrics = self.train_epoch(train_loader)
            if val_loader is not None:
                opt_val_metrics = self.validate(val_loader)
                optimized_metrics: Dict[str, Any] = {
                    'train': opt_train_metrics,
                    'val': opt_val_metrics
                }
            else:
                optimized_metrics = {'train': opt_train_metrics}

            # 记录本次优化尝试
            param_changes = {
                param: {
                    'from': baseline_params.get(param),
                    'to': attempt_params.get(param, baseline_params.get(param))
                }
                for param in set(baseline_params) | set(attempt_params)
            }
            self.optimization_history.append({
                'param_changes': param_changes,
                'original_metrics': baseline_metrics,
                'optimized_metrics': optimized_metrics
            })

            # 评判优化效果
            use_optimized, reason, suggestion = self.judger.judge_optimization(
                baseline_metrics,
                optimized_metrics,
                baseline_params,
                attempt_params,
                self.optimization_history
            )

            # 记录 Judger 输出
            self._log_agent_output('judger', {
                'epoch': None,
                'attempt': attempt_idx,
                'use_optimized': use_optimized,
                'reason': reason,
                'suggestion': suggestion,
            })

            if use_optimized:
                # 采用优化结果
                final_use_optimized = True
                final_reason = reason
                final_suggestion = suggestion
                final_metrics = optimized_metrics

                # 保存为新的 baseline
                self.cached_baseline = {
                    'metrics': optimized_metrics,
                    'params': self.current_params.copy(),
                    'model_state': {k: v.clone() for k, v in self.model.state_dict().items()},
                    'optimizer_state': {k: v.clone() if isinstance(v, torch.Tensor) else v
                                        for k, v in self.optimizer.state_dict().items()}
                }
                break

            # 如果优化不被采纳，看看是否还有改进建议
            final_reason = reason
            final_suggestion = suggestion

            if suggestion:
                # 使用 PromptOptimizer 精炼 Judger 的建议
                try:
                    advisor_prompt_snapshot = self.advisor._create_prompt(
                        baseline_params, baseline_metrics, self.config
                    )
                except Exception:
                    advisor_prompt_snapshot = ""

                refined_guidance = self.prompt_optimizer.refine_suggestion(
                    advisor_prompt_snapshot, suggestion
                )

                self._log_agent_output('prompt_optimizer', {
                    'attempt': attempt_idx,
                    'refined_guidance': refined_guidance,
                })

                # 让 Advisor 根据精炼后的建议重新给出参数
                suggestion_text, next_params, should_update = self.advisor.get_advice(
                    baseline_params,
                    baseline_metrics,
                    self.config,
                    guidance=refined_guidance,
                )

                self._log_agent_output('advisor', {
                    'attempt': attempt_idx,
                    'suggestion': suggestion_text,
                })

                if should_update and next_params != attempt_params:
                    attempt_params = next_params
                    continue

            # 无进一步改进，结束循环
            break

        # 如果最终没有采用优化，恢复 baseline 权重与参数
        if not final_use_optimized:
            self.model.load_state_dict(baseline_model_state)
            self.optimizer.load_state_dict(baseline_optimizer_state)
            self.current_params = baseline_params.copy()

            self.cached_baseline = {
                'metrics': baseline_metrics,
                'params': baseline_params.copy(),
                'model_state': baseline_model_state,
                'optimizer_state': baseline_optimizer_state
            }

        return {
            'use_optimized': final_use_optimized,
            'reason': final_reason,
            'suggestion': final_suggestion,
            'metrics': final_metrics
        }

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> None:
        """
        训练模型
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config['training']['epochs']):
            # 训练一个epoch
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # 验证
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                metrics = {
                    'train': train_metrics,
                    'val': val_metrics
                }
                
                # 早停检查
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= self.config['training']['early_stopping_patience']:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break
            else:
                metrics = {'train': train_metrics}
            
            # 记录日志和指标
            self.metrics_logger.update(epoch, metrics, self.current_params)
            
            # 缓存下一轮的基准结果
            if epoch % self.config['logging']['save_interval'] == 0:
                # 获取优化建议
                suggestion, new_params, should_update = self.advisor.get_advice(
                    self.current_params,
                    metrics,
                    self.config,
                    guidance=None,
                )
                
                # 记录建议
                self.metrics_logger.add_llm_suggestion(epoch, suggestion)
                self._log_agent_output('advisor', {'epoch': epoch, 'suggestion': suggestion})
                print(f"\nLLM调优建议:\n{suggestion}\n")
                
                # 如果有参数更新建议，评估优化效果
                if should_update:
                    print("评估参数更新:", new_params)
                    eval_result = self._evaluate_optimization(
                        train_loader, val_loader, new_params
                    )
                    
                    if eval_result['use_optimized']:
                        print("采用优化后的参数:", new_params)
                        print("原因:", eval_result['reason'])
                    else:
                        print("保持原参数")
                        print("原因:", eval_result['reason'])
                        if eval_result['suggestion']:
                            print("改进建议:", eval_result['suggestion'])
                            
                    # 重置优化尝试次数
                    # 每个 epoch 结束后都重置优化尝试次数，避免历史干扰
                    self.judger.reset_attempts()
                    self.optimization_history = [] 

        # ---------------- 训练结束，保存最终模型 ---------------- #
        from pathlib import Path
        save_path = Path(self.metrics_logger.log_dir) / "model.pt"
        torch.save(self.model.state_dict(), save_path)
        print("模型已保存到", save_path) 