import torch
import torch.nn as nn

'''
hand written adapter for finetuning
'''
class Adapter(nn.Module):
    def __init__(self, hidden_size: int, bottleneck_size: int = 64):
        super().__init__()
        self.down_proj = nn.Linear(hidden_size, bottleneck_size)
        self.activation = nn.ReLU()
        self.up_proj   = nn.Linear(bottleneck_size, hidden_size)

    def forward(self, x: torch.Tensor):
        down = self.down_proj(x)
        act  = self.activation(down)
        up   = self.up_proj(act)
        return x + up  


