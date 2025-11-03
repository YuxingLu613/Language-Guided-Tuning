import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch.nn as nn
from typing import Optional
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
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from src.trainer.adaptive_trainer import AdaptiveTrainer
from src.trainer.augmentation_transform_trainer import AugTransformTrainer
from src.trainer.trainer import Trainer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.corpus import wordnet
from src.llm.judger import LLMJudger
from src.llm.augmentation_chooser import LLMAugmentationChooser
from src.llm.text_augmentation import TextAugmenter

def get_trainer_config(cfg):
    if isinstance(cfg, (str, Path)):
        with open(cfg, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return copy.deepcopy(cfg)


def convert_numpy_types(obj):
    """将NumPy类型转换为Python原生类型，以便JSON序列化"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(i) for i in obj)
    else:
        return obj

sys.path.append(str(Path(__file__).parent.parent.resolve()))

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------- Shakespeare Dataset ---------- #
class ShakespeareDataset(Dataset):
    def __init__(self, text_path, sequence_length):
        self.sequence_length = sequence_length
        
        # 读取文本
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # 创建字符到索引的映射
        self.chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        
        # 将文本转换为索引
        self.text_encoded = [self.char_to_idx[ch] for ch in text]
        
        # 创建序列和标签
        self.data = []
        for i in range(0, len(self.text_encoded) - sequence_length):
            seq = self.text_encoded[i:i+sequence_length]
            target = self.text_encoded[i+1:i+sequence_length+1]
            self.data.append((seq, target))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        seq, target = self.data[idx]
        return torch.tensor(seq), torch.tensor(target)
    
    def decode(self, indices):
        return ''.join([self.idx_to_char[idx.item()] for idx in indices])
        
    def encode(self, text):
        """Convert a string to a list of indices using char_to_idx mapping"""
        return [self.char_to_idx.get(ch, 0) for ch in text]

# ---------- Shakespeare LSTM Model ---------- #
class ShakespeareLSTM(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout=0.5, bidirectional=False):
        super(ShakespeareLSTM, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.lstm = torch.nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(hidden_dim * (2 if bidirectional else 1), vocab_size)
        
    def forward(self, x, hidden=None):
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = self.dropout(lstm_out)
        output = self.fc(lstm_out)
        return output, hidden
    
    def init_hidden(self, batch_size, device):
        num_directions = 2 if self.lstm.bidirectional else 1
        return (torch.zeros(self.lstm.num_layers * num_directions, batch_size, self.lstm.hidden_size, device=device),
                torch.zeros(self.lstm.num_layers * num_directions, batch_size, self.lstm.hidden_size, device=device))
    
    def generate(self, initial_seq, max_length, temperature=1.0, top_k=0, top_p=0.0, device='cpu'):
        self.eval()
        with torch.no_grad():
            current_seq = initial_seq.to(device)
            hidden = None
            generated_seq = current_seq.clone()
            
            for _ in range(max_length):
                output, hidden = self(current_seq[:, -1:], hidden)
                logits = output[:, -1, :] / temperature
                
                # Top-K 采样
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = -float('Inf')
                
                # Top-p (nucleus) 采样
                if top_p > 0.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # 移除累积概率超过阈值的token
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    logits[:, indices_to_remove] = -float('Inf')
                
                # 采样下一个token
                probs = torch.nn.functional.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # 添加到生成序列
                generated_seq = torch.cat((generated_seq, next_token), dim=1)
                current_seq = next_token
            
            return generated_seq

# ---------- 加载Shakespeare数据 ---------- #
def load_shakespeare_data(cfg):
    # 准备数据集目录
    data_dir = Path(__file__).parent.parent / cfg["dataset"]["path"]
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查数据集文件是否存在，如果不存在则下载
    data_file = data_dir / "tiny_shakespeare.txt"
    if not data_file.exists():
        print("下载Tiny Shakespeare数据集...")
        import urllib.request
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, data_file)
        print(f"数据集已下载到 {data_file}")
    
    # 创建数据集
    dataset = ShakespeareDataset(
        text_path=data_file,
        sequence_length=cfg["dataset"]["sequence_length"]
    )
    
    # 分割数据集
    train_size = int(len(dataset) * cfg["dataset"]["train_split"])
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    return train_dataset, val_dataset, dataset

# ---------- Shakespeare Trainer ---------- #
class ShakespeareTrainer:
    # Lion优化器实现
    class Lion(torch.optim.Optimizer):
        def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
            """初始化Lion优化器
            
            Args:
                params: 要优化的参数
                lr: 学习率 (默认: 1e-4)
                betas: 用于计算梯度移动平均的系数 (默认: (0.9, 0.99))
                weight_decay: 权重衰减系数 (默认: 0)
            """
            defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
            super().__init__(params, defaults)

        @torch.no_grad()
        def step(self, closure=None):
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()

            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue

                    # 梯度、参数
                    grad = p.grad
                    state = self.state[p]

                    # 状态初始化
                    if len(state) == 0:
                        state['exp_avg'] = torch.zeros_like(p)

                    # 获取超参数
                    exp_avg = state['exp_avg']
                    lr = group['lr']
                    beta1, beta2 = group['betas']
                    weight_decay = group['weight_decay']

                    # 权重衰减
                    if weight_decay != 0:
                        grad = grad + weight_decay * p

                    # 更新移动平均
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                    # 计算更新方向
                    update = exp_avg.mul(beta2).add_(grad, alpha=1 - beta2)
                    update = update.sign()

                    # 更新参数
                    p.add_(update, alpha=-lr)

            return loss
            
    # 支持的优化器
    SUPPORTED_OPTIMIZERS = {
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD,
        "rmsprop": torch.optim.RMSprop,
        "adamw": torch.optim.AdamW,
        "adagrad": torch.optim.Adagrad,
        "nadam": torch.optim.NAdam,
        "lion": Lion,
        "adafactor": "_custom_adafactor",  # 为语言模型添加Adafactor优化器
    }
    
    # Focal Loss实现
    class _FocalLoss(nn.Module):
        """Simple focal loss implementation for multi-class or binary classification."""
        def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None, reduction: str = "mean"):
            super().__init__()
            self.gamma = gamma
            self.weight = weight
            self.reduction = reduction

        def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
            ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.weight, reduction="none")
            pt = torch.exp(-ce_loss)
            focal_loss = ((1 - pt) ** self.gamma) * ce_loss
            if self.reduction == "mean":
                return focal_loss.mean()
            elif self.reduction == "sum":
                return focal_loss.sum()
            else:
                return focal_loss
    
    # Dice Loss实现
    class _DiceLoss(nn.Module):
        """Dice loss for (multi-)class classification."""

        def __init__(self, smooth: float = 1.0):
            super().__init__()
            self.smooth = smooth

        def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
            # one-hot encode targets
            num_classes = inputs.size(1)
            targets_onehot = torch.nn.functional.one_hot(targets, num_classes).float()
            inputs_prob = torch.softmax(inputs, dim=1)

            dims = (0,)
            intersection = (inputs_prob * targets_onehot).sum(dims)
            union = inputs_prob.sum(dims) + targets_onehot.sum(dims)
            dice = (2. * intersection + self.smooth) / (union + self.smooth)
            loss = 1 - dice.mean()
            return loss
    
    # Perplexity Loss实现
    class _PerplexityLoss(nn.Module):
        """Perplexity loss for language generation tasks (supports flattened inputs)."""
        def __init__(self, ignore_index=-100):
            super().__init__()
            self.ignore_index = ignore_index

        def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
            # inputs: [N, vocab_size]
            # targets: [N]
            ce_loss = nn.functional.cross_entropy(
                inputs, targets, ignore_index=self.ignore_index, reduction='none'
            )
            mask = (targets != self.ignore_index).float()
            loss = (ce_loss * mask).sum() / mask.sum().clamp(min=1.0)
            return loss
    
    # 支持的损失函数
    SUPPORTED_LOSSES = {
        # classification
        "cross_entropy": nn.CrossEntropyLoss,
        "nll": nn.NLLLoss,
        "focal": _FocalLoss,
        "dice": _DiceLoss,
        "dice_loss": _DiceLoss,
        "weighted_cross_entropy": "_custom",  # placeholder handled in _get_loss_fn
        "label_smoothing_cross_entropy": "_custom_label_smoothing",  # placeholder handled in _get_loss_fn
        # regression
        "mse": nn.MSELoss,
        "mae": nn.L1Loss,
        "huber": nn.HuberLoss,
        "smooth_l1": nn.SmoothL1Loss,
        "log_cosh": "_custom_reg",
        # language generation
        "perplexity": _PerplexityLoss,
        "language_cross_entropy": nn.CrossEntropyLoss,
        "label_smoothing_language": "_custom_language_smoothing",
    }
    
    def __init__(self, model, config):
        self.model = model
        self.config = config if isinstance(config, dict) else self._load_config(config)
        self.device = self._get_device()
        self.model.to(self.device)
        
        # 设置优化器
        self.optimizer = self._get_optimizer()
        
        # 设置学习率调度器
        self.scheduler = self._get_scheduler()
        
        # 设置损失函数
        self.criterion = self._get_loss_function()
        
        # 设置梯度裁剪
        self.clip_grad_norm = self.config['training'].get('clip_grad_norm', None)
    
    def _load_config(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _get_device(self):
        device_config = self.config['training'].get('device', 'auto')
        if device_config == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device_config)
    
    def _get_optimizer(self):
        optimizer_name = self.config['training'].get('optimizer', 'adam').lower()
        lr = self.config['training'].get('learning_rate', 0.001)
        params = self.config['training'].get('optimizer_params', {})
        
        # 处理别名
        alias_map = {
            "sgd_with_momentum": "sgd",
            "sgd_momentum": "sgd",
            "sgdm": "sgd",
        }
        optimizer_name = alias_map.get(optimizer_name, optimizer_name)
        
        # 处理Adafactor优化器（常用于语言模型）
        if optimizer_name == "adafactor":
            try:
                from transformers.optimization import Adafactor
                return Adafactor(
                    self.model.parameters(),
                    lr=lr,
                    relative_step=False,
                    scale_parameter=False,
                    warmup_init=False,
                    weight_decay=params.get('weight_decay', 0.0)
                )
            except ImportError:
                # 如果没有安装transformers库，回退到AdamW
                print("Warning: Adafactor optimizer requested but transformers library not found. Falling back to AdamW.")
                return torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=params.get('weight_decay', 0.0))
        
        # 处理nesterov作为SGD的特殊情况
        if optimizer_name == "nesterov":
            momentum = params.get('momentum', 0.9)
            wd = params.get('weight_decay', 0.0)
            return torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=wd, nesterov=True)
        
        # 使用支持的优化器字典
        if optimizer_name not in self.SUPPORTED_OPTIMIZERS:
            raise ValueError(f"不支持的优化器: {optimizer_name}")
        
        optim_cls = self.SUPPORTED_OPTIMIZERS[optimizer_name]
        wd = params.get('weight_decay', 0.0)
        
        # 确保SGD相关参数组包含momentum/dampening键
        def _ensure_sgd_keys(opt, default_m=0.0):
            for g in opt.param_groups:
                if 'momentum' not in g:
                    g['momentum'] = default_m
                if 'dampening' not in g:
                    g['dampening'] = 0.0
            return opt
        
        if optimizer_name == 'sgd':
            momentum = params.get('momentum', 0.0)
            return _ensure_sgd_keys(torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=wd), momentum)
        elif optimizer_name == 'adamw':
            return torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        elif optimizer_name == 'adagrad':
            return torch.optim.Adagrad(self.model.parameters(), lr=lr, weight_decay=wd)
        elif optimizer_name == 'nadam':
            return torch.optim.NAdam(self.model.parameters(), lr=lr, weight_decay=wd)
        elif optimizer_name == 'lion':
            return self.Lion(self.model.parameters(), lr=lr, weight_decay=wd)
        elif optimizer_name == 'adam':
            return torch.optim.Adam(
                self.model.parameters(), 
                lr=lr,
                weight_decay=wd,
                betas=(params.get('beta1', 0.9), params.get('beta2', 0.999))
            )
        elif optimizer_name == 'rmsprop':
            opt = torch.optim.RMSprop(self.model.parameters(), lr=lr, momentum=params.get('momentum', 0.0), weight_decay=wd)
            # 手动初始化 square_avg
            for group in opt.param_groups:
                for p in group['params']:
                    state = opt.state[p]
                    if 'square_avg' not in state:
                        state['square_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if 'momentum_buffer' not in state and group.get('momentum', 0.0) != 0.0:
                        state['momentum_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if 'step' not in state:
                        state['step'] = 0
            return _ensure_sgd_keys(opt, params.get('momentum', 0.0))
        
        # 通用情况
        opt_generic = optim_cls(self.model.parameters(), lr=lr, weight_decay=wd)
        return _ensure_sgd_keys(opt_generic)
    
    def _get_scheduler(self):
        scheduler_name = self.config['training'].get('scheduler', 'none').lower()
        params = self.config['training'].get('scheduler_params', {})
        
        if scheduler_name == 'step':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=params.get('step_size', 10),
                gamma=params.get('gamma', 0.5)
            )
        elif scheduler_name == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=params.get('T_max', 100)
            )
        elif scheduler_name == 'none':
            return None
        else:
            raise ValueError(f"不支持的学习率调度器: {scheduler_name}")
    
    def _get_loss_function(self):
        loss_name = self.config['training'].get('loss_function', 'cross_entropy').lower()
        
        # 标准化与别名映射
        alias_map = {
            "crossentropyloss": "cross_entropy",
            "cross_entropy_loss": "cross_entropy",
            "crossentropy": "cross_entropy",
            "ce": "cross_entropy",
            "nllloss": "nll",
            "negative_log_likelihood": "nll",
            "mse_loss": "mse",
            "l1loss": "mae",
            "mae_loss": "mae",
            "huber_loss": "huber",
            "smoothl1loss": "smooth_l1",
            "smooth_l1_loss": "smooth_l1",
            "focal_loss": "focal",
            "dice_loss": "dice",
            "label_smoothing": "label_smoothing_cross_entropy",
            "smoothed_cross_entropy": "label_smoothing_cross_entropy",
        }
        loss_name = alias_map.get(loss_name, loss_name)
        
        if loss_name not in self.SUPPORTED_LOSSES:
            # 动态添加别名或自定义加权交叉熵
            if loss_name in {"weighted_cross_entropy", "wce"}:
                def _wce(inputs: torch.Tensor, targets: torch.Tensor):
                    num_classes = inputs.size(1) if len(inputs.shape) > 1 else 2
                    weights = torch.tensor(
                        [self.config['training'].get(f"weight_class_{i}", 1.0) for i in range(num_classes)],
                        device=inputs.device,
                    )
                    return nn.functional.cross_entropy(inputs, targets, weight=weights)
                return _wce
            if loss_name == "label_smoothing_cross_entropy":
                # 获取标签平滑参数，默认为0.1
                smoothing = self.config['training'].get("label_smoothing", 0.1)
                return nn.CrossEntropyLoss(label_smoothing=smoothing)
            if loss_name == "log_cosh":
                def _log_cosh(preds: torch.Tensor, targets: torch.Tensor):
                    diff = preds.squeeze() - targets.squeeze()
                    return torch.mean(torch.log(torch.cosh(diff)))
                return _log_cosh
            if loss_name == "label_smoothing_language":
                # 获取标签平滑参数，默认为0.1
                smoothing = self.config['training'].get("label_smoothing", 0.1)
                return nn.CrossEntropyLoss(label_smoothing=smoothing)
            raise ValueError(f"不支持的损失函数: {loss_name}")
            
        if isinstance(self.SUPPORTED_LOSSES[loss_name], str):
            if self.SUPPORTED_LOSSES[loss_name] == "_custom_label_smoothing":
                smoothing = self.config['training'].get("label_smoothing", 0.1)
                return nn.CrossEntropyLoss(label_smoothing=smoothing)
            elif self.SUPPORTED_LOSSES[loss_name] == "_custom_language_smoothing":
                smoothing = self.config['training'].get("label_smoothing", 0.1)
                return nn.CrossEntropyLoss(label_smoothing=smoothing)
            elif self.SUPPORTED_LOSSES[loss_name] == "_custom_reg":
                def _log_cosh(preds: torch.Tensor, targets: torch.Tensor):
                    diff = preds.squeeze() - targets.squeeze()
                    return torch.mean(torch.log(torch.cosh(diff)))
                return _log_cosh
            # 处理其他可能的自定义损失函数
            raise ValueError(f"不支持的自定义损失函数: {loss_name}")
            
        return self.SUPPORTED_LOSSES[loss_name]()
    
    def _run_epoch(self, data_loader, train=True, dataset=None):
        if train:
            self.model.train()
            total_loss = 0.0
            
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # 应用数据增强（如果配置中有增强设置且处于训练模式）
                if hasattr(self, 'config') and 'augmentation' in self.config:
                    aug_method = self.config['augmentation'].get('method')
                    aug_params = self.config['augmentation'].get('params', {})
                    if aug_method and aug_method != 'none' and dataset:
                        # 创建文本增强器
                        text_augmenter = TextAugmenter()
                        # 应用文本增强
                        batch_size = data.size(0)
                        augmented_data = data.clone()
                        for i in range(batch_size):
                            # 解码为文本
                            text = dataset.decode(data[i])
                            # 应用增强
                            augmented_text = text_augmenter.apply_augmentation(text, aug_method, **aug_params)
                            # 编码回张量
                            augmented_tensor = dataset.encode(augmented_text)
                            seq_len = min(len(augmented_tensor), data.size(1))
                            augmented_data[i, :seq_len] = torch.tensor(augmented_tensor[:seq_len], device=self.device)
                        # 使用增强后的数据
                        data = augmented_data
                
                # 初始化隐藏状态
                hidden = self.model.init_hidden(data.size(0), self.device)
                
                # 前向传播
                self.optimizer.zero_grad()
                output, _ = self.model(data, hidden)
                
                # 计算损失
                loss = self.criterion(output.reshape(-1, output.size(-1)), target.reshape(-1))
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                if self.clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                
                # 更新参数
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(data_loader)
            return avg_loss
        else:
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            bleu_scores = []
            char_error_rates = []
            
            with torch.no_grad():
                for data, target in data_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    
                    # 初始化隐藏状态
                    hidden = self.model.init_hidden(data.size(0), self.device)
                    
                    # 前向传播
                    output, _ = self.model(data, hidden)
                    
                    # 计算损失
                    loss = self.criterion(output.reshape(-1, output.size(-1)), target.reshape(-1))
                    val_loss += loss.item()
                    
                    # 计算准确率
                    pred = output.reshape(-1, output.size(-1)).argmax(dim=1)
                    target_flat = target.reshape(-1)
                    correct += (pred == target_flat).sum().item()
                    total += target_flat.size(0)
                    
                    # 确保dataset不为None
                    if dataset is not None:
                        # 计算BLEU分数和字符错误率
                        for i in range(min(5, data.size(0))):  # 只计算一部分样本以节省时间
                            # 生成预测序列
                            pred_seq = self.model.generate(
                                data[i:i+1], 
                                max_length=20,
                                temperature=1.0,
                                device=self.device
                            )
                            
                            # 解码为字符串
                            pred_text = dataset.decode(pred_seq[0])
                            target_text = dataset.decode(target[i])
                            
                            # 计算BLEU分数
                            smoothie = SmoothingFunction().method1
                            bleu = sentence_bleu(
                                [list(target_text)],
                                list(pred_text),
                                smoothing_function=smoothie
                            )
                            bleu_scores.append(bleu)
                            
                            # 计算字符错误率
                            distance = self.levenshtein_distance(target_text, pred_text)
                            cer = distance / max(len(target_text), 1)
                            char_error_rates.append(cer)
            
            val_loss /= len(data_loader)
            perplexity = np.exp(val_loss)
            accuracy = correct / total
            avg_bleu = np.mean(bleu_scores) if bleu_scores else 0
            avg_cer = np.mean(char_error_rates) if char_error_rates else 1
            
            return val_loss, perplexity, accuracy, avg_bleu, avg_cer
    
    def levenshtein_distance(self, s1, s2):
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def generate_samples(self, dataset, epoch_idx, log_dir):
        self.model.eval()
        
        # 生成配置
        gen_config = self.config.get('generation', {})
        temperature = gen_config.get('temperature', 1.0)
        max_length = gen_config.get('max_length', 1000)
        top_k = gen_config.get('top_k', 0)
        top_p = gen_config.get('top_p', 0.0)
        
        # 随机选择初始序列
        seed_text = "ROMEO:"
        seed_encoded = torch.tensor([[dataset.char_to_idx[c] for c in seed_text]], dtype=torch.long)
        
        # 生成文本
        generated = self.model.generate(
            seed_encoded.to(self.device),
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            device=self.device
        )
        
        # 解码并保存
        generated_text = dataset.decode(generated[0])
        
        # 保存生成的文本
        samples_dir = Path(log_dir) / 'samples'
        samples_dir.mkdir(exist_ok=True)
        
        with open(samples_dir / f'sample_epoch_{epoch_idx}.txt', 'w', encoding='utf-8') as f:
            f.write(generated_text)
        
        print(f"\n生成的样本:\n{generated_text[:200]}...\n")
        return generated_text


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
        if dataset_name.lower() in ['tinyshakespeare']:
            data_type = 'text'
        
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
            config.setdefault('augmentation', {})['method'] = aug_suggestion
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
def parse_judger_suggestions(judger_output):
    """
    解析judger输出中的agent建议
    
    Args:
        judger_output: str - judger的输出
        
    Returns:
        Dict[str, str] - 各个agent的建议
    """
    if not judger_output:
        return {}
    
    judger_output_str = str(judger_output) if judger_output is not None else ""
    
    agent_guidance = {}
    
    # 提取adaptive agent建议
    adaptive_match = re.search(r'ADAPTIVE_AGENT_SUGGESTION:\s*(.*?)(?=\n[A-Z_]+:|$)', judger_output_str, re.DOTALL)
    if adaptive_match:
        agent_guidance['adaptive'] = adaptive_match.group(1).strip()
    
    # 提取HPO agent建议
    hpo_match = re.search(r'HPO_AGENT_SUGGESTION:\s*(.*?)(?=\n[A-Z_]+:|$)', judger_output_str, re.DOTALL)
    if hpo_match:
        agent_guidance['hpo'] = hpo_match.group(1).strip()
    
    return agent_guidance




# ---------- 运行多链式epoch ---------- #
def run_multichain_epoch(epoch_idx, context, judger, args):
    print(f"\n=== EPOCH {epoch_idx} ===")
    model = context['model']
    optimizer_pt = context['optimizer']
    cfg = context['config']
    train_loader = context['train_loader']
    val_loader = context['val_loader']
    dataset = context['dataset']
    log_dir = cfg['logging']['log_dir']
    
    # 重置judger的状态，确保每个epoch都有新的尝试机会
    judger.reset_state()
    
    # 确定任务类型
    task_type = 'language_generation'  # Shakespeare是语言任务
    
    # 保存当前模型状态（状态a）
    state_a = copy.deepcopy(model.state_dict())
    optimizer_state_a = copy.deepcopy(optimizer_pt.state_dict())
    
    # 对于epoch 0，直接训练一轮并返回
    if epoch_idx == 0:
        print(f"[EPOCH 0] Direct training with initial config")
        a_trainer = ShakespeareTrainer(model, get_trainer_config(cfg))
        a_trainer.config['training']['epochs'] = 1
        train_loss = a_trainer._run_epoch(train_loader, train=True, dataset=dataset)
        val_loss, perplexity, accuracy, bleu, cer = a_trainer._run_epoch(val_loader, train=False, dataset=dataset)
        metrics_train = {"loss": train_loss}
        metrics_val = {"loss": val_loss, "perplexity": perplexity, "accuracy": accuracy, "bleu": bleu, "character_error_rate": cer}
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
        a_trainer = ShakespeareTrainer(model, get_trainer_config(cfg))
        a_trainer.config['training']['epochs'] = 1
        train_loss = a_trainer._run_epoch(train_loader, train=True)
        val_loss, perplexity, accuracy, bleu, cer = a_trainer._run_epoch(val_loader, train=False)
        metrics_train = {"loss": train_loss}
        metrics_val = {"loss": val_loss, "perplexity": perplexity, "accuracy": accuracy, "bleu": bleu, "character_error_rate": cer}
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
    a_trainer_b = ShakespeareTrainer(model, get_trainer_config(new_cfg))
    a_trainer_b.config['training']['epochs'] = 1
    train_loss_b = a_trainer_b._run_epoch(train_loader, train=True, dataset=dataset)
    val_loss_b, perplexity_b, accuracy_b, bleu_b, cer_b = a_trainer_b._run_epoch(val_loader, train=False, dataset=dataset)
    metrics_b_train = {"loss": train_loss_b}
    metrics_b_val = {"loss": val_loss_b, "perplexity": perplexity_b, "accuracy": accuracy_b, "bleu": bleu_b, "character_error_rate": cer_b}
    metrics_b = {"train": metrics_b_train, "val": metrics_b_val, "epoch": epoch_idx}
    state_b = copy.deepcopy(model.state_dict())
    
    # 使用状态a和原配置训练一轮（得到状态c）
    model.load_state_dict(state_a)
    optimizer_pt.load_state_dict(optimizer_state_a)
    a_trainer_c = ShakespeareTrainer(model, get_trainer_config(cfg))
    a_trainer_c.config['training']['epochs'] = 1
    train_loss_c = a_trainer_c._run_epoch(train_loader, train=True, dataset=dataset)
    val_loss_c, perplexity_c, accuracy_c, bleu_c, cer_c = a_trainer_c._run_epoch(val_loader, train=False, dataset=dataset)
    metrics_c_train = {"loss": train_loss_c}
    metrics_c_val = {"loss": val_loss_c, "perplexity": perplexity_c, "accuracy": accuracy_c, "bleu": bleu_c, "character_error_rate": cer_c}
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
            a_trainer_refined = ShakespeareTrainer(model, get_trainer_config(refined_cfg))
            a_trainer_refined.config['training']['epochs'] = 1
            train_loss_refined = a_trainer_refined._run_epoch(train_loader, train=True, dataset=dataset)
            val_loss_refined, perplexity_refined, accuracy_refined, bleu_refined, cer_refined = a_trainer_refined._run_epoch(val_loader, train=False, dataset=dataset)
            metrics_refined_train = {"loss": train_loss_refined}
            metrics_refined_val = {"loss": val_loss_refined, "perplexity": perplexity_refined, "accuracy": accuracy_refined, "bleu": bleu_refined, "character_error_rate": cer_refined}
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
            
            # 如果优化成功，更新最终配置
            if use_refined:
                final_use = True
                final_config = refined_cfg
                final_state = state_refined
                judger_reason = reason_refined
                print(f"[EPOCH {epoch_idx}] Judger accepted refined optimization (attempt {attempt_num}): {reason_refined}")
                break
            
            # 更新agent建议，继续尝试
            agent_suggestions = agent_suggestions_refined
    
    # 应用最终选择的状态
    model.load_state_dict(final_state)
    
    # 保存本轮结果到上下文，供下一轮使用
    final_metrics = metrics_b if final_use else metrics_c
    context['prev_metrics'] = final_metrics
    context['prev_config'] = copy.deepcopy(final_config)
    
    # 保存结果
    outcome = {
        'epoch': epoch_idx,
        'flags': {
            'use_adaptive_agent': args.use_adaptive_agent,
            'use_aug_agent': args.use_aug_agent,
            'use_hpo_agent': args.use_hpo_agent,
        },
        'use_optimized': final_use,
        'judger_reason': judger_reason,
        'judger_rounds': judger_rounds,
        'optimization_history': optimization_history,
        'agent_logs': agent_logs,
        'final_metrics': final_metrics,
        'final_config': final_config
    }
    
    final_log_file = Path(log_dir) / f'epoch_{epoch_idx}_final_result.json'
    with open(final_log_file, 'w', encoding='utf-8') as f:
        json.dump(outcome, f, ensure_ascii=False, indent=2)
    
    return final_use, final_config



# ---------- 主函数 ---------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--model_path', default=None)
    parser.add_argument('--use_adaptive_agent', action='store_true')
    parser.add_argument('--use_aug_agent', action='store_true')
    parser.add_argument('--use_hpo_agent', action='store_true')
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    # 设置日志目录
    base_log_dir = Path(cfg['logging']['log_dir'])
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    history_dir = base_log_dir / f'history_{ts}'
    history_dir.mkdir(parents=True, exist_ok=True)
    cfg['logging']['log_dir'] = str(history_dir)
    # Disable Trainer from creating _strategy/_aug subdirectories
    cfg['logging']['disable_trainer_subdir'] = True
    
    # 设置随机种子
    set_seed(cfg.get('training', {}).get('seed', 42))
    
    # 创建日志目录
    log_dir = cfg['logging']['log_dir']
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    with open(log_dir_path / "used_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)
    
    # 加载数据
    train_dataset, val_dataset, dataset = load_shakespeare_data(cfg)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['training']['batch_size'],
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg['training']['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    # 创建模型
    model = ShakespeareLSTM(
        vocab_size=dataset.vocab_size,
        embedding_dim=cfg['model']['architecture']['embedding_dim'],
        hidden_dim=cfg['model']['architecture']['hidden_dim'],
        num_layers=cfg['model']['architecture']['num_layers'],
        dropout=cfg['model']['architecture']['dropout'],
        bidirectional=cfg['model']['architecture']['bidirectional']
    )
    
    # 创建优化器
    optimizer_name = cfg['training'].get('optimizer', 'adam').lower()
    lr = cfg['training'].get('learning_rate', 0.001)
    params = cfg['training'].get('optimizer_params', {})
    
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=lr,
            weight_decay=params.get('weight_decay', 0),
            betas=(params.get('beta1', 0.9), params.get('beta2', 0.999))
        )
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=params.get('momentum', 0),
            weight_decay=params.get('weight_decay', 0)
        )
    elif optimizer_name == 'rmsprop':
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=lr,
            weight_decay=params.get('weight_decay', 0)
        )
    else:
        raise ValueError(f"不支持的优化器: {optimizer_name}")
    
    # 创建judger
    judger = LLMJudger(cfg['llm'])
    
    # 创建上下文
    context = {
        'model': model,
        'optimizer': optimizer,
        'config': cfg,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'dataset': dataset,
        'prev_metrics': None,
        'prev_config': None,
        'prev_samples': None
    }
    
    # 运行多个epoch
    outcomes = []
    for epoch_idx in range(args.epochs):
        outcome = run_multichain_epoch(epoch_idx, context, judger, args)
        outcomes.append(outcome)
    
    # 保存最终结果
    final_outcome = {
        'epochs': args.epochs,
        'flags': {
            'use_adaptive_agent': args.use_adaptive_agent,
            'use_hpo_agent': args.use_hpo_agent,
            'use_aug_agent': args.use_aug_agent,
        },
        'outcomes': outcomes
    }
    
    with open(log_dir_path / "final_outcome.json", "w", encoding="utf-8") as f:
        json.dump(final_outcome, f, ensure_ascii=False, indent=2)
    
    print(f"\n训练完成！日志保存在: {log_dir}")

if __name__ == '__main__':
    main()
