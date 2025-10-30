import torch
from torch import nn
from typing import List, Dict

class SimpleCNN(nn.Module):
    def __init__(self, input_shape: List[int], conv_layers: List[Dict], 
                 pool_size: int, fc_layers: List[int], num_classes: int, 
                 dropout: float = 0.25):
        """
        初始化CNN模型
        
        Args:
            input_shape: 输入形状 [channels, height, width]
            conv_layers: 卷积层配置列表
            pool_size: 池化层大小
            fc_layers: 全连接层神经元数量列表
            num_classes: 类别数量
            dropout: Dropout比率
        """
        super().__init__()
        
        # 保存配置
        self.input_shape = input_shape
        self.conv_configs = conv_layers
        self.pool_size = pool_size
        self.fc_sizes = fc_layers
        
        # 构建卷积层
        self.conv_layers = nn.ModuleList()
        in_channels = input_shape[0]
        
        for conv_config in conv_layers:
            conv_layer = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=conv_config['filters'],
                    kernel_size=conv_config['kernel_size'],
                    stride=conv_config['stride'],
                    padding=conv_config['padding']
                ),
                nn.ReLU(),
                nn.BatchNorm2d(conv_config['filters']),
                nn.MaxPool2d(pool_size),
                nn.Dropout2d(dropout)
            )
            self.conv_layers.append(conv_layer)
            in_channels = conv_config['filters']
        
        # 计算展平后的特征维度
        with torch.no_grad():
            x = torch.zeros(1, *input_shape)
            for conv_layer in self.conv_layers:
                x = conv_layer(x)
            flattened_size = x.numel()
        
        # 构建全连接层
        self.fc_layers = nn.ModuleList()
        in_features = flattened_size
        
        for fc_size in fc_layers:
            fc_block = nn.Sequential(
                nn.Linear(in_features, fc_size),
                nn.ReLU(),
                nn.BatchNorm1d(fc_size),
                nn.Dropout(dropout)
            )
            self.fc_layers.append(fc_block)
            in_features = fc_size
        
        # 输出层
        self.output_layer = nn.Linear(in_features, num_classes)
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """
        初始化模型权重
        """
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: 输出张量 [batch_size, num_classes]
        """
        # 卷积层
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
        
        # 输出层
        x = self.output_layer(x)
        return x
    
    def get_config(self) -> Dict:
        """
        获取模型配置
        """
        return {
            'input_shape': self.input_shape,
            'conv_layers': self.conv_configs,
            'pool_size': self.pool_size,
            'fc_layers': self.fc_sizes
        } 