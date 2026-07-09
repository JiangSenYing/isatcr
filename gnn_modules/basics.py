"""
基础模块
参考 cross_layer_opt_with_grl-main/modules/basics.py
"""

import torch as th
import torch.nn as nn


class RnnLayer(nn.Module):
    """
    RNN 层 - 用于维护时序依赖
    
    使用 GRUCell 融合历史信息，支持可选的 LayerNorm
    """
    
    def __init__(self, hidden_size: int, use_layer_norm: bool = False):
        super(RnnLayer, self).__init__()
        self._hidden_size = hidden_size
        self.rnn = nn.GRUCell(hidden_size, hidden_size)
        
        self._use_layer_norm = use_layer_norm
        if self._use_layer_norm:
            self.norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x, h):
        """
        Args:
            x: 当前输入 (batch, hidden_size)
            h: 隐状态 (batch, hidden_size)
        
        Returns:
            y: 输出 (batch, hidden_size)
            h: 新隐状态 (batch, hidden_size)
        """
        h = self.rnn(x, h)
        if self._use_layer_norm:
            y = self.norm(h)
        else:
            y = h
        return y, h


class MLP(nn.Module):
    """多层感知机"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 n_layers: int = 2, activation: str = 'relu'):
        super(MLP, self).__init__()
        
        from .activations import get_activation
        act_fn = get_activation(activation)
        
        layers = []
        if n_layers == 1:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(act_fn())
            for _ in range(n_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(act_fn())
            layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


class DuelingLayer(nn.Module):
    """
    Dueling DQN 输出层
    
    Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, n_actions: int, activation: str = 'relu'):
        super(DuelingLayer, self).__init__()
        
        from .activations import get_activation
        act_fn = get_activation(activation)
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            act_fn(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            act_fn(),
            nn.Linear(hidden_dim, n_actions)
        )
    
    def forward(self, x):
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        # Q = V + (A - mean(A))
        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q
