import json
from pathlib import Path
from typing import Dict, Any, Union
import matplotlib.pyplot as plt

class HousingMetricsLogger:
    """专门用于回归任务的指标记录器。记录损失、MAE、RMSE、学习率等。"""

    def __init__(self, log_dir: Union[str, Path], config: Dict[str, Any]):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': [],
            'train_rmse': [],
            'val_rmse': [],
            'train_r2': [],
            'val_r2': [],
            'learning_rate': [],
        }
        self.config = config
        self.llm_suggestions = []

    def update(self, epoch: int, metrics: Dict[str, Any], params: Dict[str, float]):
        # 记录损失和指标
        self.metrics_history['train_loss'].append(metrics['train']['loss'])
        self.metrics_history['train_mae'].append(metrics['train']['mae'])
        self.metrics_history['train_rmse'].append(metrics['train']['rmse'])
        self.metrics_history['train_r2'].append(metrics['train']['r2'])

        if 'val' in metrics:
            self.metrics_history['val_loss'].append(metrics['val']['loss'])
            self.metrics_history['val_mae'].append(metrics['val']['mae'])
            self.metrics_history['val_rmse'].append(metrics['val']['rmse'])
            self.metrics_history['val_r2'].append(metrics['val']['r2'])

        self.metrics_history['learning_rate'].append(params['learning_rate'])
        self._save_metrics()
        if self.config['logging'].get('plot_metrics', False):
            self._plot_metrics(epoch)

    def add_llm_suggestion(self, epoch: int, suggestion: str):
        self.llm_suggestions.append({'epoch': epoch, 'suggestion': suggestion})
        with open(self.log_dir / 'llm_suggestions.json', 'w', encoding='utf-8') as f:
            json.dump(self.llm_suggestions, f, ensure_ascii=False, indent=2)

    def _save_metrics(self):
        with open(self.log_dir / 'metrics_history.json', 'w', encoding='utf-8') as f:
            json.dump(self.metrics_history, f, ensure_ascii=False, indent=2)

    def _plot_metrics(self, current_epoch: int):
        epochs = list(range(current_epoch + 1))
        plt.style.use('seaborn-v0_8' if 'seaborn' in plt.style.available else 'default')
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 损失
        axes[0, 0].plot(epochs, self.metrics_history['train_loss'], label='Train Loss')
        if self.metrics_history['val_loss']:
            axes[0, 0].plot(epochs, self.metrics_history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].legend()

        # MAE
        axes[0, 1].plot(epochs, self.metrics_history['train_mae'], label='Train MAE')
        if self.metrics_history['val_mae']:
            axes[0, 1].plot(epochs, self.metrics_history['val_mae'], label='Val MAE')
        axes[0, 1].set_title('MAE')
        axes[0, 1].legend()

        # RMSE
        axes[1, 0].plot(epochs, self.metrics_history['train_rmse'], label='Train RMSE')
        if self.metrics_history['val_rmse']:
            axes[1, 0].plot(epochs, self.metrics_history['val_rmse'], label='Val RMSE')
        axes[1, 0].set_title('RMSE')
        axes[1, 0].legend()

        # 学习率
        axes[1, 1].plot(epochs, self.metrics_history['learning_rate'])
        axes[1, 1].set_title('Learning Rate')

        plt.tight_layout()
        plt.savefig(self.log_dir / f'metrics_epoch_{current_epoch}.png', dpi=300)
        plt.close() 
 