from pathlib import Path
import sys
import datetime
import random
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import yaml
import argparse
from typing import Tuple, Dict, Any, Optional, List
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# 添加项目根目录到Python路径
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

# ---------- 设置随机种子工具 ---------- #
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------- Shakespeare LSTM 模型 ---------- #
class ShakespeareLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout=0.5, bidirectional=False):
        super(ShakespeareLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), vocab_size)
        
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
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # 移除累积概率超过阈值的token
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    logits[:, indices_to_remove] = -float('Inf')
                
                # 采样下一个token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # 添加到生成序列
                generated_seq = torch.cat((generated_seq, next_token), dim=1)
                current_seq = next_token
            
            return generated_seq
        
# ---------- Shakespeare 数据集 ---------- #
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

# ---------- 训练器 ---------- #
class ShakespeareTrainer:
    def __init__(self, model, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.model = model
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
        
        # 设置早停
        self.early_stopping_patience = self.config['training'].get('early_stopping_patience', float('inf'))
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # 设置日志
        self.log_dir = Path(ROOT_DIR) / self.config['logging']['log_dir']
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 指标记录
        self.train_losses = []
        self.val_losses = []
        self.perplexities = []
        self.accuracies = []
        self.bleu_scores = []
        self.char_error_rates = []
    
    def _get_device(self):
        device_config = self.config['training'].get('device', 'auto')
        if device_config == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device_config)
    
    def _get_optimizer(self):
        optimizer_name = self.config['training'].get('optimizer', 'adam').lower()
        lr = self.config['training'].get('learning_rate', 0.001)
        params = self.config['training'].get('optimizer_params', {})
        
        if optimizer_name == 'adam':
            return torch.optim.Adam(
                self.model.parameters(), 
                lr=lr,
                weight_decay=params.get('weight_decay', 0),
                betas=(params.get('beta1', 0.9), params.get('beta2', 0.999))
            )
        elif optimizer_name == 'sgd':
            return torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=params.get('momentum', 0),
                weight_decay=params.get('weight_decay', 0)
            )
        elif optimizer_name == 'rmsprop':
            return torch.optim.RMSprop(
                self.model.parameters(),
                lr=lr,
                weight_decay=params.get('weight_decay', 0)
            )
        else:
            raise ValueError(f"不支持的优化器: {optimizer_name}")
    
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
        
        if loss_name == 'cross_entropy':
            return nn.CrossEntropyLoss()
        else:
            raise ValueError(f"不支持的损失函数: {loss_name}")
    
    def train(self, train_loader, val_loader, dataset):
        epochs = self.config['training'].get('epochs', 30)
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
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
                
                train_loss += loss.item()
                
                if batch_idx % 100 == 0:
                    print(f'Epoch: {epoch+1}/{epochs} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.6f}')
            
            train_loss /= len(train_loader)
            self.train_losses.append(train_loss)
            
            # 验证阶段
            val_loss, perplexity, accuracy, bleu, cer = self.evaluate(val_loader, dataset)
            
            # 更新学习率
            if self.scheduler:
                self.scheduler.step()
            
            # 记录指标
            self.val_losses.append(val_loss)
            self.perplexities.append(perplexity)
            self.accuracies.append(accuracy)
            self.bleu_scores.append(bleu)
            self.char_error_rates.append(cer)
            
            # 打印指标
            print(f'Epoch: {epoch+1}/{epochs} Train Loss: {train_loss:.6f} Val Loss: {val_loss:.6f} '
                  f'Perplexity: {perplexity:.2f} Accuracy: {accuracy:.4f} BLEU: {bleu:.4f} CER: {cer:.4f}')
            
            # 保存模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                torch.save(self.model.state_dict(), self.log_dir / 'best_model.pt')
                print(f'模型已保存: {self.log_dir / "best_model.pt"}')
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stopping_patience:
                    print(f'早停: {self.early_stopping_patience} 个epoch没有改善')
                    break
            
            # 生成样本
            if self.config['logging'].get('save_samples', False) and (epoch + 1) % self.config['logging'].get('sample_interval', 5) == 0:
                self.generate_samples(dataset)
            
            # 保存指标图表
            if self.config['logging'].get('plot_metrics', False):
                self.plot_metrics()
    
    def evaluate(self, val_loader, dataset):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        bleu_scores = []
        char_error_rates = []
        
        with torch.no_grad():
            for data, target in val_loader:
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
        
        val_loss /= len(val_loader)
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
    
    def generate_samples(self, dataset):
        self.model.eval()
        
        # 生成配置
        gen_config = self.config.get('generation', {})
        temperature = gen_config.get('temperature', 1.0)
        max_length = gen_config.get('max_length', 1000)
        top_k = gen_config.get('top_k', 0)
        top_p = gen_config.get('top_p', 0.0)
        num_samples = gen_config.get('num_samples', 5)
        
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
        samples_dir = self.log_dir / 'samples'
        samples_dir.mkdir(exist_ok=True)
        
        with open(samples_dir / f'sample_epoch_{len(self.train_losses)}.txt', 'w', encoding='utf-8') as f:
            f.write(generated_text)
        
        print(f"\n生成的样本:\n{generated_text[:200]}...\n")
    
    def plot_metrics(self):
        plt.figure(figsize=(15, 10))
        
        # 损失曲线
        plt.subplot(2, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.title('Loss')
        plt.legend()
        
        # 困惑度曲线
        plt.subplot(2, 2, 2)
        plt.plot(self.perplexities)
        plt.title('Perplexity')
        
        # 准确率曲线
        plt.subplot(2, 2, 3)
        plt.plot(self.accuracies)
        plt.title('Accuracy')
        
        # BLEU分数曲线
        plt.subplot(2, 2, 4)
        plt.plot(self.bleu_scores, label='BLEU')
        plt.plot(self.char_error_rates, label='CER')
        plt.title('BLEU & CER')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.log_dir / 'metrics.png')
        plt.close()

# ----------------------- DummyAdvisor ----------------------- #
class DummyAdvisor:
    """占位 Advisor，始终返回不修改参数。"""
    
    def get_advice(self, current_params: Dict[str, Any], metrics: Dict[str, Any], context_cfg: Dict[str, Any], guidance: Optional[str] = None):
        return "no change", {}, False

# ----------------------------- 主流程 ----------------------------- #
def main(config_path: Path, model_path: Optional[Path] = None):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # 设置随机种子
    seed = config.get('training', {}).get('seed', 42)
    set_seed(seed)
    
    # 调整日志目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_log_dir = "shakespeare_log"
    new_log_dir = f"{base_log_dir}/baseline_{timestamp}"
    config["logging"]["log_dir"] = new_log_dir
    
    # 保存修改后的config副本
    log_dir_path = ROOT_DIR / new_log_dir
    log_dir_path.mkdir(parents=True, exist_ok=True)
    with open(log_dir_path / "used_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True)
    
    # 准备数据集目录
    data_dir = ROOT_DIR / config["dataset"]["path"]
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
        sequence_length=config["dataset"]["sequence_length"]
    )
    
    # 分割数据集
    train_size = int(len(dataset) * config["dataset"]["train_split"])
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=0
    )
    
    # 创建模型
    model = ShakespeareLSTM(
        vocab_size=dataset.vocab_size,
        embedding_dim=config["model"]["architecture"]["embedding_dim"],
        hidden_dim=config["model"]["architecture"]["hidden_dim"],
        num_layers=config["model"]["architecture"]["num_layers"],
        dropout=config["model"]["architecture"]["dropout"],
        bidirectional=config["model"]["architecture"]["bidirectional"]
    )
    
    # 如果提供了预训练权重则加载
    if model_path and model_path.exists():
        print(f"加载预训练权重: {model_path}")
        model.load_state_dict(torch.load(model_path))
    
    # 创建训练器
    tmp_cfg_path = log_dir_path / "temp_cfg.yaml"
    with open(tmp_cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True)
    
    trainer = ShakespeareTrainer(model, tmp_cfg_path)
    
    print("\n模型结构:\n", model)
    print("\n开始 baseline 训练，日志保存到:", new_log_dir)
    
    # 开始训练
    trainer.train(train_loader, val_loader, dataset)
    
    print("训练完成！ 日志位置:", new_log_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shakespeare LSTM 训练脚本")
    parser.add_argument(
        "--config", type=str, default="config/shakespeare_config.yaml", help="配置文件路径"
    )
    parser.add_argument("--model_path", type=str, default=None, help="预训练模型权重 (.pt) 路径（可选）")
    args = parser.parse_args()
    
    main(Path(args.config), Path(args.model_path) if args.model_path else None)