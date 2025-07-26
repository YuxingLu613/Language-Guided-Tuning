import torch
from torch import nn
from typing import List

class CustomModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int] | int, output_dim: int, dropout: float = 0.1):
        """
        初始化模型
        
        Args:
            input_dim: 输入维度
            hidden_dims: 隐藏层维度列表
            output_dim: 输出维度
            dropout: Dropout比率
        """
        super().__init__()
        
        # 允许用户传入单个整数
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        
        # 构建层
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, input_dim]
            
        Returns:
            torch.Tensor: 输出张量 [batch_size, output_dim]
        """
        # 确保输入形状正确
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        return self.model(x) 