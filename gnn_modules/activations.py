"""
激活函数注册表
参考 cross_layer_opt_with_grl-main/modules/activations.py
"""

import torch.nn as nn

REGISTRY = {
    'relu': nn.ReLU,
    'elu': nn.ELU,
    'leaky_relu': nn.LeakyReLU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'gelu': nn.GELU,
    'silu': nn.SiLU,
    'prelu': nn.PReLU,
    'softplus': nn.Softplus,
    'identity': nn.Identity,
}


def get_activation(name: str):
    """获取激活函数类"""
    name = name.lower()
    if name not in REGISTRY:
        raise ValueError(f"Unknown activation: {name}. Available: {list(REGISTRY.keys())}")
    return REGISTRY[name]
