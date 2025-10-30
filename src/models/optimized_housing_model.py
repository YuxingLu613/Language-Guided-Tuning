# improved model code
import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.2, hidden_dims=[256, 128, 64]):
        super().__init__()
        layers = []
        in_features = input_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(in_features, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_features = dim
            
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)