import json
from pathlib import Path
from typing import Dict, List, Any, Union
import matplotlib.pyplot as plt
import numpy as np

class MetricsLogger:
    def __init__(self, log_dir: Union[str, Path], config: Dict[str, Any]):
        """
        初始化指标记录器
        
        Args:
            log_dir: 日志目录
            config: 配置信息
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.num_classes: int = config.get('dataset', {}).get('num_classes', 10)

        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'train_precision': [],
            'val_precision': [],
            'train_recall': [],
            'val_recall': [],
            'train_roc_auc': [],
            'val_roc_auc': [],
            'learning_rate': [],
            'weight_decay': [],
        }

        # 为每个类别权重创建历史记录
        for i in range(self.num_classes):
            self.metrics_history[f'weight_class_{i}'] = []
        
        self.config = config
        self.llm_suggestions = []
    
    def update(self, epoch: int, metrics: Dict[str, Any], params: Dict[str, float]) -> None:
        """
        更新指标历史
        
        Args:
            epoch: 当前epoch
            metrics: 训练和验证指标
            params: 当前参数值
        """
        # 记录损失与基础指标
        self.metrics_history['train_loss'].append(metrics['train']['loss'])
        self.metrics_history['train_accuracy'].append(metrics['train']['accuracy'])
        self.metrics_history['train_precision'].append(metrics['train'].get('precision'))
        self.metrics_history['train_recall'].append(metrics['train'].get('recall'))
        self.metrics_history['train_roc_auc'].append(metrics['train'].get('roc_auc'))

        if 'val' in metrics:
            self.metrics_history['val_loss'].append(metrics['val']['loss'])
            self.metrics_history['val_accuracy'].append(metrics['val']['accuracy'])
            self.metrics_history['val_precision'].append(metrics['val'].get('precision'))
            self.metrics_history['val_recall'].append(metrics['val'].get('recall'))
            self.metrics_history['val_roc_auc'].append(metrics['val'].get('roc_auc'))
        
        # 记录参数值
        self.metrics_history['learning_rate'].append(params['learning_rate'])
        self.metrics_history['weight_decay'].append(params.get('weight_decay'))
        for i in range(self.num_classes):
            self.metrics_history[f'weight_class_{i}'].append(params[f'weight_class_{i}'])
        
        # 保存到文件
        self._save_metrics()
        
        # 如果配置要求，绘制并保存图表
        if self.config['logging']['plot_metrics']:
            self._plot_metrics(epoch)
    
    def add_llm_suggestion(self, epoch: int, suggestion: str) -> None:
        """
        记录LLM建议
        
        Args:
            epoch: 当前epoch
            suggestion: LLM建议内容
        """
        self.llm_suggestions.append({
            'epoch': epoch,
            'suggestion': suggestion
        })
        
        # 保存建议到文件
        suggestions_file = self.log_dir / 'llm_suggestions.json'
        with open(suggestions_file, 'w', encoding='utf-8') as f:
            json.dump(self.llm_suggestions, f, ensure_ascii=False, indent=2)
    
    def _save_metrics(self) -> None:
        """
        保存指标历史到文件
        """
        metrics_file = self.log_dir / 'metrics_history.json'
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics_history, f, indent=2)
    
    def _plot_metrics(self, current_epoch: int) -> None:
        """
        绘制指标图表
        
        Args:
            current_epoch: 当前epoch
        """
        try:
            import seaborn as sns
            plt.style.use('seaborn')
        except (ImportError, OSError):
            # 如果seaborn不可用，使用默认样式
            plt.style.use('default')
        
        epochs = range(current_epoch + 1)
        
        # 创建多个子图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 损失曲线
        ax1.plot(epochs, self.metrics_history['train_loss'], 'b-', label='Training Loss')
        if self.metrics_history['val_loss']:
            ax1.plot(epochs, self.metrics_history['val_loss'], 'r-', label='Validation Loss')
        ax1.set_title('Loss over Epochs')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 2. 准确率曲线
        ax2.plot(epochs, self.metrics_history['train_accuracy'], 'b-', label='Training Accuracy')
        if self.metrics_history['val_accuracy']:
            ax2.plot(epochs, self.metrics_history['val_accuracy'], 'r-', label='Validation Accuracy')
        ax2.set_title('Accuracy over Epochs')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        # 3. 学习率变化
        ax3.plot(epochs, self.metrics_history['learning_rate'], 'g-')
        ax3.set_title('Learning Rate over Epochs')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.grid(True)
        
        # 4. 类别权重变化
        for i in range(self.num_classes):
            ax4.plot(epochs, self.metrics_history[f'weight_class_{i}'], 
                    label=f'Class {i}', alpha=0.7)
        ax4.set_title('Class Weights over Epochs')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Weight')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(True)
        
        # 调整布局并保存
        plt.tight_layout()
        plt.savefig(self.log_dir / f'metrics_epoch_{current_epoch}.png', 
                   bbox_inches='tight', dpi=300)
        plt.close() 