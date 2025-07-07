from pathlib import Path
import sys

# 添加项目根目录到Python路径
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets, transforms
import yaml
import argparse
from typing import Tuple, Dict, Any, Optional
from src.models.cnn_model import SimpleCNN
from src.trainer.trainer import Trainer

def create_transforms(config: Dict[str, Any]) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    创建数据转换
    
    Args:
        config: 配置字典
        
    Returns:
        Tuple[transforms.Compose, transforms.Compose]: 训练和测试数据转换
    """
    # 基础转换
    train_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]
    
    # 添加数据增强
    if "random_rotation" in config['dataset']['train_transform']:
        train_transforms.insert(1, transforms.RandomRotation(15))
    if "random_affine" in config['dataset']['train_transform']:
        train_transforms.insert(1, transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)))
    
    train_transform = transforms.Compose(train_transforms)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    return train_transform, test_transform

def load_mnist(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """
    加载MNIST数据集
    
    Args:
        config: 配置字典
        
    Returns:
        Tuple[DataLoader, DataLoader]: 训练和测试数据加载器
    """
    train_transform, test_transform = create_transforms(config)
    data_dir = ROOT_DIR / config['dataset']['path']
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查CUDA是否可用
    use_cuda = torch.cuda.is_available()
    
    train_dataset = datasets.MNIST(
        str(data_dir), train=True, download=True,
        transform=train_transform
    )
    
    test_dataset = datasets.MNIST(
        str(data_dir), train=False,
        transform=test_transform
    )
    
    # 根据是否使用CUDA配置数据加载器
    loader_kwargs = {
        'batch_size': config['training']['batch_size'],
        'num_workers': 0,  # Windows下默认使用单进程
        'pin_memory': use_cuda
    }
    
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_kwargs
    )
    
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **loader_kwargs
    )
    
    return train_loader, test_loader

def create_scheduler(optimizer: torch.optim.Optimizer, config: Dict[str, Any]) -> Optional[ReduceLROnPlateau]:
    """
    创建学习率调度器
    
    Args:
        optimizer: 优化器
        config: 配置字典
        
    Returns:
        Optional[ReduceLROnPlateau]: 学习率调度器，如果配置中没有scheduler则返回None
    """
    if 'scheduler' not in config['training']:
        return None
        
    scheduler_config = config['training']['scheduler']
    return ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=scheduler_config['patience'],
        factor=scheduler_config['factor'],
        min_lr=scheduler_config['min_lr'],
        verbose=True
    )

def main(config_path: Path) -> None:
    """
    主训练函数
    
    Args:
        config_path: 配置文件路径
    """
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 创建必要的目录
    log_dir = ROOT_DIR / config['logging']['log_dir']
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    train_loader, test_loader = load_mnist(config)
    
    # 创建模型
    model = SimpleCNN(
        input_shape=config['dataset']['input_shape'],
        conv_layers=config['model']['architecture']['conv_layers'],
        pool_size=config['model']['architecture']['pool_size'],
        fc_layers=config['model']['architecture']['fc_layers'],
        num_classes=config['dataset']['num_classes'],
        dropout=config['model']['architecture']['dropout']
    )
    
    # 创建训练器
    trainer = Trainer(model, config_path)
    
    # 创建学习率调度器（如果配置中有）
    scheduler = create_scheduler(trainer.optimizer, config)
    if scheduler is not None:
        trainer.scheduler = scheduler
    
    # 打印模型信息
    print("\n模型结构:")
    print(model)
    print("\n模型配置:")
    print(model.get_config())
    print("\n开始训练...")
    
    # 开始训练
    trainer.train(train_loader, test_loader)
    print("训练完成！")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN模型训练脚本')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                      help='配置文件路径')
    args = parser.parse_args()
    
    config_path = ROOT_DIR / args.config
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    main(config_path) 