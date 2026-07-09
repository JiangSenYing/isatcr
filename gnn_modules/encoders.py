"""
关系编码器 (Relational Encoder)
参考 cross_layer_opt_with_grl-main/modules/encoders/rel_enc.py

使用 GATv2Conv 处理异构图观测，将邻居信息聚合到 Agent 节点。
"""

from typing import Dict, Mapping
import torch as th
from torch import Tensor
import torch.nn as nn

try:
    from torch_geometric.data import HeteroData
    from torch_geometric.nn import GATv2Conv, SAGEConv, HeteroConv
    from . import pyg_compat
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    print("Warning: PyTorch Geometric not installed. RelationalEncoder will not be available.")

# 为向后兼容保留 DGL_AVAILABLE 变量名
DGL_AVAILABLE = PYG_AVAILABLE

from .activations import get_activation


class RelationalEncoder(nn.Module):
    """
    关系型输入编码器 (PyTorch Geometric 版本)
    
    参考 cross_layer_opt_with_grl-main 的 RelationalEncoder。
    
    处理异构图观测，其中边从观测到的实体（邻居）指向 Agent。
    支持 GATv2Conv（注意力机制）或 GCNConv（GCN）。
    
    输入:
        PyG HeteroData，包含:
        - 节点类型: 'agent', 'nbr', 可能还有 'hop2_nbr'
        - 边类型: ('nbr', 'nearby', 'agent'), 可能还有 ('hop2_nbr', 'nearby', 'nbr')
        - 节点特征: graph[ntype].feat
    
    输出:
        Agent 节点的编码表示 (batch, hidden_size)
    """
    
    def __init__(
        self,
        in_feats_size_dict: Mapping[str, int],  # 各类型节点的特征维度
        hidden_size: int,                        # 隐藏层/输出维度
        n_heads: int = 4,                        # 注意力头数（仅 GAT 使用）
        conv_type: str = 'gat',                  # 卷积类型: 'gat' 或 'gcn'
        activation: str = 'relu',                # 激活函数
    ):
        super(RelationalEncoder, self).__init__()
        
        if not DGL_AVAILABLE:
            raise ImportError("PyTorch Geometric is required for RelationalEncoder")
        
        assert 'agent' in in_feats_size_dict, "agent features must be in observations"
        assert conv_type in ('gat', 'gcn'), f"conv_type must be 'gat' or 'gcn', got {conv_type}"
        
        in_feats_size_dict_ = in_feats_size_dict.copy()
        agent_feats_size = in_feats_size_dict_.pop('agent')
        self._ntypes = tuple(in_feats_size_dict_.keys())  # 除 agent 外的节点类型
        
        self._hidden_size = hidden_size
        self._n_heads = n_heads
        self._conv_type = conv_type
        self._activation = get_activation(activation)
        self._agent_feats_size = agent_feats_size
        
        # 为每种实体类型定义卷积层
        mods = {}
        conv_dict = {}
        for ntype, feats_size in in_feats_size_dict_.items():
            if conv_type == 'gat':
                # PyG GATv2Conv: 多头注意力机制
                feats_per_head = self._hidden_size // self._n_heads
                # PyG GATv2Conv 需要 in_channels, out_channels, heads
                # 对于异构图，我们需要使用 HeteroConv
                conv_dict[(ntype, 'nearby', 'agent')] = GATv2Conv(
                    (feats_size, agent_feats_size),  # (源节点维度, 目标节点维度)
                    feats_per_head,                   # 每个头的输出维度
                    heads=self._n_heads,              # 头数
                    add_self_loops=False,
                    concat=True  # 拼接多头输出
                )
            else:  # gcn -> 使用 SAGEConv（支持二分图消息传递）
                # GCNConv 不支持异构图（源和目标节点类型不同）
                # SAGEConv 支持 bipartite 消息传递
                conv_dict[(ntype, 'nearby', 'agent')] = SAGEConv(
                    (feats_size, agent_feats_size),   # (源节点维度, 目标节点维度)
                    hidden_size,                      # 输出维度
                )
            mods[ntype] = conv_dict[(ntype, 'nearby', 'agent')]
        
        self.f_conv = nn.ModuleDict(mods)
        self.hetero_conv = HeteroConv(conv_dict, aggr='sum')
        
        # MLP 聚合器：合并来自不同关系的输出
        self.f_aggr = nn.Sequential(
            nn.Linear(self._hidden_size * len(self._ntypes), self._hidden_size),
            self._activation()
        )
        
        # 激活函数
        self.act = self._activation()
    
    def forward(self, g: 'HeteroData') -> Tensor:
        """
        Args:
            g: PyG HeteroData，包含节点特征 g[ntype].feat
        
        Returns:
            Agent 节点的编码表示 (batch, hidden_size)
        """
        # 获取节点特征
        feat = pyg_compat.get_ndata(g, 'feat')
        
        # 构建 x_dict 用于 HeteroConv
        x_dict = {}
        for ntype in g.node_types:
            if ntype in feat:
                x_dict[ntype] = feat[ntype]
            elif hasattr(g[ntype], 'x') and g[ntype].x is not None:
                x_dict[ntype] = g[ntype].x
        
        # 构建 edge_index_dict
        edge_index_dict = {}
        for etype in g.edge_types:
            if hasattr(g[etype], 'edge_index'):
                edge_index_dict[etype] = g[etype].edge_index
        
        # 应用 HeteroConv
        out_dict = self.hetero_conv(x_dict, edge_index_dict)
        
        # 激活
        outputs = {}
        for ntype, out in out_dict.items():
            if ntype == 'agent':
                outputs['agent'] = self.act(out)
        
        # 如果没有 agent 输出，使用零填充
        if 'agent' not in outputs or outputs['agent'] is None:
            n_agents = x_dict.get('agent', th.zeros(1, self._agent_feats_size)).size(0)
            return th.zeros(n_agents, self._hidden_size, device=next(self.parameters()).device)
        
        return outputs['agent']
    
    def _aggr_func(self, outputs: Mapping[str, Tensor]) -> Tensor:
        """聚合来自多种关系的输出"""
        if not outputs:
            raise ValueError("No valid entity types found in graph")
        
        # 处理不同卷积输出格式
        processed = []
        for ntype in self._ntypes:
            if ntype not in outputs:
                continue
            out = outputs[ntype]
            if self._conv_type == 'gat':
                # GAT 输出: (N, n_heads * feats_per_head) -> (N, hidden_size)
                if out.dim() == 3:
                    out = out.flatten(1)
            # GCN 输出已经是 (N, hidden_size)
            processed.append(out)
        
        # 按类型顺序堆叠输出
        stacked = th.stack(processed, dim=1)
        # 展平并通过 MLP
        return self.f_aggr(stacked.flatten(1))


class FlatEncoder(nn.Module):
    """
    扁平编码器
    
    直接将扁平状态向量编码为隐藏表示，不使用图结构。
    作为 GNN 编码器的 fallback 或基准。
    """
    
    def __init__(
        self,
        input_size: int,      # 输入维度
        hidden_size: int,     # 隐藏层/输出维度
        n_layers: int = 2,    # 层数
        activation: str = 'relu',
    ):
        super(FlatEncoder, self).__init__()
        
        self._hidden_size = hidden_size
        self._activation = get_activation(activation)
        
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(self._activation())
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(self._activation())
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# 编码器注册表
ENCODER_REGISTRY = {
    'rel': RelationalEncoder,
    'relational': RelationalEncoder,
    'flat': FlatEncoder,
}
