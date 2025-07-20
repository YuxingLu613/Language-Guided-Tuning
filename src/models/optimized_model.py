# improved model code
import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self, dropout=0.3, hidden_dims=[64, 32]):
        super().__init__()
        layers = []
        in_dim = hidden_dims[0]
        
        # Input layer
        layers.append(nn.Linear(1, in_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], 1))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)