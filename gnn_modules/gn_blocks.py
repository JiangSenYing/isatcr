"""
图神经网络块 (GN Blocks) - PyTorch Geometric 版本
参考 cross_layer_opt_with_grl-main/modules/gn_blks.py

实现节点级和边级的图网络块，用于多跳消息传递。
"""

from typing import Union, Tuple
import torch as th
import torch.nn as nn

try:
    from torch_geometric.nn import MessagePassing, GATv2Conv
    from torch_geometric.data import HeteroData
    from . import pyg_compat
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    print("Warning: PyTorch Geometric not installed. GN blocks will not be available.")

# 为向后兼容保留 DGL_AVAILABLE 变量名
DGL_AVAILABLE = PYG_AVAILABLE

from .activations import get_activation


def expand_as_pair(input_value):
    """辅助函数：将输入扩展为 (src, dst) 对"""
    if isinstance(input_value, tuple):
        return input_value
    return (input_value, input_value)

from .activations import get_activation


class NodeGNBlock(nn.Module):
    """
    节点级图网络块 (Node-focused GN Block) - PyTorch Geometric 版本
    
    参考论文: "Relational inductive biases, deep learning, and graph networks"
    
    更新规则:
    1. 边更新: f_e(源节点 + 边特征 + 目标节点) → 消息 m
    2. 消息聚合: mean/sum/max(所有邻居消息) → neigh
    3. 节点更新: f_v(聚合消息 + 自身特征) → 新节点特征
    
    用于编码阶段，将邻居信息聚合到中心节点。
    """
    
    def __init__(
        self,
        in_node_feats: Union[int, Tuple[int, int]],  # 输入节点特征维度 (src, dst) 或 int
        in_edge_feats: int,                           # 输入边特征维度
        out_node_feats: int,                          # 输出节点特征维度
        aggregator_type: str = 'mean',                # 聚合方式: mean, sum, max
        activation: str = 'relu',                     # 激活函数
    ):
        super(NodeGNBlock, self).__init__()
        
        if not DGL_AVAILABLE:
            raise ImportError("PyTorch Geometric is required for NodeGNBlock")
        
        self._in_src_node_feats, self._in_dst_node_feats = expand_as_pair(in_node_feats)
        # 例如 1跳模式：
        # 源节点: 邻居 (6维原始特征)
        # 目标节点: Agent (9维特征)
        # in_node_feats = (6, 9)

        # 在 2跳模式中，都是邻居：
        # in_node_feats = (6, 6)
        self._in_edge_feats = in_edge_feats # 3
        
        self._out_node_feats = out_node_feats # 64
        
        if aggregator_type not in ('sum', 'max', 'mean'):
            raise KeyError(f'Aggregator type {aggregator_type} not recognized.')
        self._aggr_type = aggregator_type
        self.activation = get_activation(activation)
        
        # 边更新函数: 拼接 [源节点, 边特征, 目标节点] -> 消息
        self.f_e = nn.Sequential(
            nn.Linear(self._in_src_node_feats + self._in_edge_feats + self._in_dst_node_feats, 
                      self._out_node_feats),
            self.activation()
        )
        """
            self.f_e = nn.Sequential(
            nn.Linear(6 + 3 + 6, 64),  # 输入 15维 → 输出 64维
            nn.ReLU()
        )

        # 参数量
        # 权重矩阵: (15, 64)
        # 偏置: (64,)

        作用: 为每条边计算消息(message),这个消息综合考虑了：

        源节点的状态
        边（链路）的质量
        目标节点的状态
        """
        
        # 节点更新函数: 拼接 [聚合消息, 自身特征] -> 新特征
        self.f_v = nn.Sequential(
            nn.Linear(self._out_node_feats + self._in_dst_node_feats, self._out_node_feats),
            self.activation(),
        )
        """
        self.f_v = nn.Sequential(
            nn.Linear(64 + 6, 64),  # 输入 70维 → 输出 64维
            nn.ReLU()
        )

        # 参数量
        # 权重矩阵: (70, 64)
        # 偏置: (64,)
        结合聚合的邻居信息和节点自身的原始特征，生成新的节点表示。
        """
    
    def forward(self, edge_index, node_feats, edge_feats, num_dst_nodes=None):
        """
        Args:
            edge_index: PyG 边索引 [2, num_edges]
            node_feats: 节点特征 (tensor 或 tuple of tensors)
            edge_feats: 边特征
            num_dst_nodes: 目标节点数量（可选）
        
        Returns:
            更新后的目标节点特征
        """
        if isinstance(node_feats, tuple):
            src_node_feats, dst_node_feats = node_feats
        else:
            src_node_feats = dst_node_feats = node_feats
        
        src_idx = edge_index[0] # 源节点索引
        dst_idx = edge_index[1] # 目标节点索引
        
        # 获取边的源和目标节点特征
        src_feats = src_node_feats[src_idx]  # (num_edges, src_feat_dim)
        dst_feats = dst_node_feats[dst_idx]  # (num_edges, dst_feat_dim)

        """
        # src_node_feats = dst_node_feats = feat['nbr']
        # shape: (4, 6)
        #   索引0: [0.5, 0.3, 0.8, 0.2, 0.1, 0.9]  # S2
        #   索引1: [0.4, 0.7, 0.2, 0.6, 0.5, 0.3]  # S3
        #   索引2: [0.9, 0.1, 0.4, 0.8, 0.2, 0.6]  # S4
        #   索引3: [0.3, 0.6, 0.9, 0.1, 0.7, 0.4]  # S5

        # 根据 src_idx 索引
        src_feats = tensor([
            [0.5, 0.3, 0.8, 0.2, 0.1, 0.9],  # src_idx[0]=0 → S2特征 (边0的源)
            [0.5, 0.3, 0.8, 0.2, 0.1, 0.9],  # src_idx[1]=0 → S2特征 (边1的源)
            [0.4, 0.7, 0.2, 0.6, 0.5, 0.3],  # src_idx[2]=1 → S3特征 (边2的源)
            [0.4, 0.7, 0.2, 0.6, 0.5, 0.3],  # src_idx[3]=1 → S3特征 (边3的源)
            [0.9, 0.1, 0.4, 0.8, 0.2, 0.6],  # src_idx[4]=2 → S4特征 (边4的源)
            [0.9, 0.1, 0.4, 0.8, 0.2, 0.6],  # src_idx[5]=2 → S4特征 (边5的源)
            [0.3, 0.6, 0.9, 0.1, 0.7, 0.4],  # src_idx[6]=3 → S5特征 (边6的源)
            [0.3, 0.6, 0.9, 0.1, 0.7, 0.4]   # src_idx[7]=3 → S5特征 (边7的源)
        ])
        # shape: (8, 6)

        # 根据 dst_idx 索引
        dst_feats = tensor([
            [0.4, 0.7, 0.2, 0.6, 0.5, 0.3],  # dst_idx[0]=1 → S3特征 (边0的目标)
            [0.9, 0.1, 0.4, 0.8, 0.2, 0.6],  # dst_idx[1]=2 → S4特征 (边1的目标)
            [0.5, 0.3, 0.8, 0.2, 0.1, 0.9],  # dst_idx[2]=0 → S2特征 (边2的目标)
            [0.3, 0.6, 0.9, 0.1, 0.7, 0.4],  # dst_idx[3]=3 → S5特征 (边3的目标)
            [0.5, 0.3, 0.8, 0.2, 0.1, 0.9],  # dst_idx[4]=0 → S2特征 (边4的目标)
            [0.3, 0.6, 0.9, 0.1, 0.7, 0.4],  # dst_idx[5]=3 → S5特征 (边5的目标)
            [0.4, 0.7, 0.2, 0.6, 0.5, 0.3],  # dst_idx[6]=1 → S3特征 (边6的目标)
            [0.9, 0.1, 0.4, 0.8, 0.2, 0.6]   # dst_idx[7]=2 → S4特征 (边7的目标)
        ])
        # shape: (8, 6)

        边0:
源节点(S2) ──链路──→ 目标节点(S3)
[0.5, 0.3, 0.8, 0.2, 0.1, 0.9]  ──[0.8, 0.6, 0.9]──→  [0.4, 0.7, 0.2, 0.6, 0.5, 0.3]
     src_feats[0]                  edge_feats[0]              dst_feats[0]
        """


        """
        2hop
        # 源特征（邻居特征，已编码）
        src_feats = tensor([
            [0.34, 0.56, 0.78, 0.23, 0.91, ..., 0.67],  # S2特征 (边0的源)
            [0.45, 0.67, 0.23, 0.89, 0.82, ..., 0.45],  # S3特征 (边1的源)
            [0.56, 0.34, 0.89, 0.12, 0.67, ..., 0.78],  # S4特征 (边2的源)
            [0.23, 0.78, 0.45, 0.56, 0.34, ..., 0.91]   # S5特征 (边3的源)
        ])
        # shape: (4, 64)

        # 目标特征（Agent特征，重复4次对应4条边）
        dst_feats = tensor([
            [0.5, 0.2, 0.8, 0.1, 0.6, 0.3, 0.9, 0.4, 0.7],  # Agent特征 (边0的目标)
            [0.5, 0.2, 0.8, 0.1, 0.6, 0.3, 0.9, 0.4, 0.7],  # Agent特征 (边1的目标)
            [0.5, 0.2, 0.8, 0.1, 0.6, 0.3, 0.9, 0.4, 0.7],  # Agent特征 (边2的目标)
            [0.5, 0.2, 0.8, 0.1, 0.6, 0.3, 0.9, 0.4, 0.7]   # Agent特征 (边3的目标)
        ])
        # shape: (4, 9)
        # 注意：所有边的目标都是同一个Agent，所以特征相同
        
        """
        
        # 边更新：计算消息
        messages = self.f_e(th.cat([src_feats, edge_feats, dst_feats], dim=1))
        """
            # 边0: S2 → S3
            src_feat_0 = [0.5, 0.3, 0.8, 0.2, 0.1, 0.9]      # S2特征 (6,)
            edge_feat_0 = [0.8, 0.6, 0.9]                    # S2→S3链路 (3,)
            dst_feat_0 = [0.4, 0.7, 0.2, 0.6, 0.5, 0.3]      # S3特征 (6,)

            # 拼接
            concat_0 = torch.cat([src_feat_0, edge_feat_0, dst_feat_0])
            concat_0 = [0.5, 0.3, 0.8, 0.2, 0.1, 0.9,        # S2状态
                        0.8, 0.6, 0.9,                        # 链路状态
                        0.4, 0.7, 0.2, 0.6, 0.5, 0.3]         # S3状态
            # shape: (15,)

            # 通过 f_e (MLP)
            # Linear(15 → 64)
            weights = torch.randn(64, 15)  # 权重矩阵
            bias = torch.randn(64)         # 偏置
            linear_out = concat_0 @ weights.T + bias  # (64,)
            # linear_out = [0.23, -0.45, 0.67, ..., 0.89]  (假设值)

            # ReLU激活
            message_0 = torch.relu(linear_out)
            message_0 = [0.23, 0.00, 0.67, ..., 0.89]  # 负值变0
            # shape: (64,)

            # 所有8条边的消息
            messages = tensor([
                [0.23, 0.56, 0.78, ..., 0.91],  # 边0: S2→S3 的消息 (64,)
                [0.34, 0.67, 0.45, ..., 0.82],  # 边1: S2→S4 的消息
                [0.45, 0.23, 0.89, ..., 0.67],  # 边2: S3→S2 的消息
                [0.12, 0.78, 0.34, ..., 0.56],  # 边3: S3→S5 的消息
                [0.67, 0.34, 0.56, ..., 0.45],  # 边4: S4→S2 的消息
                [0.89, 0.45, 0.23, ..., 0.78],  # 边5: S4→S5 的消息
                [0.56, 0.89, 0.12, ..., 0.34],  # 边6: S5→S3 的消息
                [0.78, 0.12, 0.67, ..., 0.23]   # 边7: S5→S4 的消息
            ])
            # shape: (8, 64)
        """

        """
        2hop
                self.f_e = nn.Sequential(
            nn.Linear(64 + 3 + 9, 64),  # 输入 76维 → 输出 64维
            nn.ReLU()
        )

        # 边0: S2 → S1(Agent)
src_feat_0 = [0.34, 0.56, 0.78, 0.23, 0.91, ..., 0.67]  # S2特征 (64,)
edge_feat_0 = [0.8, 0.6, 0.9]                            # S2→S1链路 (3,)
dst_feat_0 = [0.5, 0.2, 0.8, 0.1, 0.6, 0.3, 0.9, 0.4, 0.7]  # S1特征 (9,)

        # 拼接
        concat_0 = torch.cat([src_feat_0, edge_feat_0, dst_feat_0])
        concat_0 = [
            # S2状态（64维，包含S2及其邻居的信息）
            0.34, 0.56, 0.78, 0.23, 0.91, ..., 0.67,
            # S2→S1链路质量（3维）
            0.8, 0.6, 0.9,
            # S1(Agent)状态（9维，原始任务和资源信息）
            0.5, 0.2, 0.8, 0.1, 0.6, 0.3, 0.9, 0.4, 0.7
        ]
        # shape: (76,)

        # 通过 MLP
        # Linear(76 → 64)
        linear_out = concat_0 @ weights.T + bias
        linear_out = [0.45, -0.23, 0.78, 0.91, 0.34, ..., 0.67]  # (64,)

        # ReLU激活
        message_0 = torch.relu(linear_out)
        message_0 = [0.45, 0.00, 0.78, 0.91, 0.34, ..., 0.67]  # (64,)


        计算所有边的消息:
        # 边1: S3 → S1
        src_feat_1 = [0.45, 0.67, 0.23, ..., 0.45]  # (64,) S3+其邻居
        edge_feat_1 = [0.7, 0.5, 0.8]               # (3,)
        dst_feat_1 = [0.5, 0.2, 0.8, ..., 0.7]      # (9,)
        concat_1 = (76,)
        message_1 = [0.56, 0.89, 0.23, 0.67, ..., 0.45]  # (64,)

        # 边2: S4 → S1
        message_2 = [0.23, 0.45, 0.67, 0.34, ..., 0.89]  # (64,)

        # 边3: S5 → S1
        message_3 = [0.67, 0.34, 0.56, 0.78, ..., 0.23]  # (64,)

        # 所有消息
        messages = tensor([
            [0.45, 0.00, 0.78, 0.91, 0.34, ..., 0.67],  # S2的建议
            [0.56, 0.89, 0.23, 0.67, 0.78, ..., 0.45],  # S3的建议
            [0.23, 0.45, 0.67, 0.34, 0.56, ..., 0.89],  # S4的建议
            [0.67, 0.34, 0.56, 0.78, 0.23, ..., 0.23]   # S5的建议
        ])
        # shape: (4, 64)


        """
        
        # 消息聚合
        if num_dst_nodes is None:
            num_dst_nodes = dst_node_feats.size(0)# 4个邻居节点 # 2hop:1个Agent节点
        
        # 使用 scatter 进行聚合
        from torch_geometric.utils import scatter
        aggregated = scatter(messages, # (8, 64) 所有消息     # (4, 64) 4个邻居的消息
                             dst_idx,  # (8,) 目标节点索引    # [0, 0, 0, 0] 都指向Agent 0
                             dim=0,    # 在第0维聚合          # 在第0维聚合
                             dim_size=num_dst_nodes, # 4     # 1
                             reduce=self._aggr_type) # 'mean'  # 平均聚合
        """
        2hop
        # Agent 0 接收所有4个邻居的消息
        # dst_idx = [0, 0, 0, 0] 所有边都指向Agent 0

        messages_to_agent = [
            messages[0],  # S2的消息 [0.45, 0.00, 0.78, ..., 0.67]
            messages[1],  # S3的消息 [0.56, 0.89, 0.23, ..., 0.45]
            messages[2],  # S4的消息 [0.23, 0.45, 0.67, ..., 0.89]
            messages[3]   # S5的消息 [0.67, 0.34, 0.56, ..., 0.23]
        ]

        # 平均聚合（按维度求平均）
        aggregated[0] = mean(messages_to_agent)
        aggregated[0] = [
            (0.45 + 0.56 + 0.23 + 0.67) / 4,  # 第1维 = 0.4775
            (0.00 + 0.89 + 0.45 + 0.34) / 4,  # 第2维 = 0.42
            (0.78 + 0.23 + 0.67 + 0.56) / 4,  # 第3维 = 0.56
            (0.91 + 0.67 + 0.34 + 0.78) / 4,  # 第4维 = 0.675
            # ... 继续到第64维
            (0.67 + 0.45 + 0.89 + 0.23) / 4   # 第64维 = 0.56
        ]

        aggregated = tensor([[0.4775, 0.42, 0.56, 0.675, 0.4525, ..., 0.56]])
        # shape: (1, 64)
                S2: [0.45, 0.00, 0.78, ..., 0.67] ──┐
                                            │
        S3: [0.56, 0.89, 0.23, ..., 0.45] ──┤
                                            ├─→ mean ─→ Agent聚合表示
        S4: [0.23, 0.45, 0.67, ..., 0.89] ──┤    (1, 64)
                                            │
        S5: [0.67, 0.34, 0.56, ..., 0.23] ──┘
        """
        # aggregated[0] 是一个64维向量，代表：
        # - 来自所有邻居的综合建议
        # - 每个邻居都考虑了自己的状态、链路质量、Agent需求
        # - 这是邻居们对"如何帮助Agent完成任务"的集体智慧
        
        # 节点更新
        return self.f_v(th.cat([aggregated, dst_node_feats], dim=1))


class NodeGATBlock(nn.Module):
    """
    使用 GATv2 的节点级聚合块。

    与 NodeGNBlock 保持同样的输入/输出接口，方便在 GraphController 中直接替换。
    聚合流程:
    1. 使用边特征参与注意力打分，做邻居消息加权聚合
    2. 将聚合结果与目标节点特征拼接后更新节点表示
    """

    def __init__(
        self,
        in_node_feats: Union[int, Tuple[int, int]],
        in_edge_feats: int,
        out_node_feats: int,
        n_heads: int = 4,
        dropout: float = 0.0,
        activation: str = 'relu',
    ):
        super(NodeGATBlock, self).__init__()

        if not DGL_AVAILABLE:
            raise ImportError("PyTorch Geometric is required for NodeGATBlock")
        if n_heads <= 0:
            raise ValueError(f"n_heads must be positive, got {n_heads}")
        if out_node_feats % n_heads != 0:
            raise ValueError(
                f"out_node_feats ({out_node_feats}) must be divisible by n_heads ({n_heads})"
            )

        self._in_src_node_feats, self._in_dst_node_feats = expand_as_pair(in_node_feats)
        self._in_edge_feats = in_edge_feats
        self._out_node_feats = out_node_feats
        self._n_heads = n_heads

        self.activation = get_activation(activation)

        edge_dim = in_edge_feats if in_edge_feats > 0 else None
        self.gat = GATv2Conv(
            (self._in_src_node_feats, self._in_dst_node_feats),
            out_channels=out_node_feats // n_heads,
            heads=n_heads,
            concat=True,
            dropout=dropout,
            edge_dim=edge_dim,
            add_self_loops=False,
        )

        self.f_v = nn.Sequential(
            nn.Linear(self._out_node_feats + self._in_dst_node_feats, self._out_node_feats),
            self.activation(),
        )

    def forward(self, edge_index, node_feats, edge_feats, num_dst_nodes=None):
        """前向传播，接口与 NodeGNBlock 对齐。"""
        del num_dst_nodes  # 与 NodeGNBlock 对齐保留参数

        if isinstance(node_feats, tuple):
            src_node_feats, dst_node_feats = node_feats
        else:
            src_node_feats = dst_node_feats = node_feats

        edge_attr = edge_feats if self._in_edge_feats > 0 else None
        aggregated = self.gat((src_node_feats, dst_node_feats), edge_index, edge_attr=edge_attr)

        return self.f_v(th.cat([aggregated, dst_node_feats], dim=1))
    """
    # 节点0(S2) 接收的消息
    # dst_idx 中值为 0 的位置: [2, 4]
    messages_to_node0 = [
        messages[2],  # 边2: S3→S2 的消息 [0.45, 0.23, 0.89, ..., 0.67]
        messages[4]   # 边4: S4→S2 的消息 [0.67, 0.34, 0.56, ..., 0.45]
    ]

    # 聚合方式: 'mean' (平均)
    aggregated[0] = mean(messages_to_node0)
    aggregated[0] = [(0.45+0.67)/2, (0.23+0.34)/2, (0.89+0.56)/2, ..., (0.67+0.45)/2]
    aggregated[0] = [0.56, 0.285, 0.725, ..., 0.56]  # (64,)

    aggregated = tensor([
            [0.56, 0.28, 0.72, ..., 0.56],  # 节点0(S2) 聚合后特征 (64,)
            [0.39, 0.72, 0.45, ..., 0.62],  # 节点1(S3) 聚合后特征
            [0.56, 0.39, 0.56, ..., 0.52],  # 节点2(S4) 聚合后特征
            [0.50, 0.61, 0.28, ..., 0.67]   # 节点3(S5) 聚合后特征
        ])
        # shape: (4, 64)
    
                    边2(S3→S2)  边4(S4→S2)
                    ↓          ↓
    节点0(S2) ← 聚合 [message2, message4] → aggregated[0]

                边0(S2→S3)  边6(S5→S3)
                    ↓          ↓
    节点1(S3) ← 聚合 [message0, message6] → aggregated[1]

                边1(S2→S4)  边7(S5→S4)
                    ↓          ↓
    节点2(S4) ← 聚合 [message1, message7] → aggregated[2]

                边3(S3→S5)  边5(S4→S5)
                    ↓          ↓
    节点3(S5) ← 聚合 [message3, message5] → aggregated[3]
    """

    """
    # 节点0(S2)
    aggregated_0 = [0.56, 0.28, 0.72, ..., 0.56]       # (64,) 聚合的邻居信息
    dst_feat_0 = [0.5, 0.3, 0.8, 0.2, 0.1, 0.9]        # (6,) 自身原始特征

    # 拼接
    concat_update_0 = torch.cat([aggregated_0, dst_feat_0])
    concat_update_0 = [0.56, 0.28, 0.72, ..., 0.56,    # 聚合信息 (64维)
                    0.5, 0.3, 0.8, 0.2, 0.1, 0.9]   # 自身特征 (6维)
    # shape: (70,)

    # 通过 f_v (MLP)
    # Linear(70 → 64)
    weights = torch.randn(64, 70)
    bias = torch.randn(64)
    linear_out = concat_update_0 @ weights.T + bias
    linear_out = [0.34, -0.12, 0.78, ..., 0.91]  # (64,)

    # ReLU激活
    updated_feat_0 = torch.relu(linear_out)
    updated_feat_0 = [0.34, 0.00, 0.78, ..., 0.91]  # (64,)
    """

class EdgeGNBlock(nn.Module):
    """
    边级图网络块 (Edge-focused GN Block) - PyTorch Geometric 版本
    
    用于输出层，为每条边（每个邻居选择）计算 Q 值。
    
    输出:
    - 边级得分: 每条边对应的动作 Q 值（如选择该邻居 + 各功率档位）
    - 节点级得分: 中心节点自身的动作 Q 值（如本地计算）
    """
    
    def __init__(
        self,
        in_node_feats: Union[int, Tuple[int, int]],  # 输入节点特征维度
        in_edge_feats: int,                           # 输入边特征维度
        out_node_feats: int,                          # 节点输出维度（如1，表示本地计算Q值）
        out_edge_feats: int,                          # 边输出维度（如1，表示选择该邻居的Q值）
        hidden_size: int,                             # 隐藏层维度
        activation: str = 'relu',
    ):
        super(EdgeGNBlock, self).__init__()
        
        if not DGL_AVAILABLE:
            raise ImportError("PyTorch Geometric is required for EdgeGNBlock")
        
        self._in_src_node_feats, self._in_dst_node_feats = expand_as_pair(in_node_feats)
        self._in_edge_feats = in_edge_feats
        self._out_node_feats = out_node_feats
        self._out_edge_feats = out_edge_feats
        self._hidden_size = hidden_size
        
        self.activation = get_activation(activation)
        
        # 边更新函数: 计算每条边的 Q 值  拼接 [源节点, 边特征, 目标节点] → 边的Q值
        self.f_e = nn.Sequential(
            nn.Linear(self._in_src_node_feats + self._in_edge_feats + self._in_dst_node_feats, 
                      self._hidden_size),
            self.activation(),
            nn.Linear(self._hidden_size, self._out_edge_feats)
        )
        """
        self.f_edge = nn.Sequential(
            nn.Linear(6 + 3 + 64, 64),  # 第1层: 73维输入 → 64维隐藏
            nn.ReLU(),                  # 激活
            nn.Linear(64, 1)            # 第2层: 64维 → 1维Q值
        )
        """
        
        # 节点更新函数: 计算节点自身动作的 Q 值  目标节点特征 → 节点的Q值
        self.f_v = nn.Linear(self._in_dst_node_feats, self._out_node_feats)
    
    def forward(self, edge_index, node_feats, edge_feats):
        """
        Args:
            edge_index: PyG 边索引 [2, num_edges]
            node_feats: 节点特征 (src_feats, dst_feats)
            edge_feats: 边特征
        
        Returns:
            v2_j: 节点级输出 (batch, out_node_feats)
            e2: 边级输出 (n_edges, out_edge_feats)
        """
        if isinstance(node_feats, tuple):
            src_node_feats, dst_node_feats = node_feats # (4, 6) 邻居特征
        else:
            src_node_feats = dst_node_feats = node_feats # (1, 64) Agent特征
        
        src_idx = edge_index[0]
        dst_idx = edge_index[1]
        
        # 获取边的源和目标节点特征
        src_feats = src_node_feats[src_idx]
        dst_feats = dst_node_feats[dst_idx]
        """
        # 源特征（邻居特征）
        src_feats = tensor([
            [0.5, 0.3, 0.8, 0.2, 0.1, 0.9],  # 边0的源: S2特征
            [0.4, 0.7, 0.2, 0.6, 0.5, 0.3],  # 边1的源: S3特征
            [0.9, 0.1, 0.4, 0.8, 0.2, 0.6],  # 边2的源: S4特征
            [0.3, 0.6, 0.9, 0.1, 0.7, 0.4]   # 边3的源: S5特征
        ])
        # shape: (4, 6)

        # 目标特征（Agent特征，重复4次）
        dst_feats = tensor([
            [0.19, 0.52, 0.38, 0.71, 0.85, 0.09, 0.61, 0.76, ..., 0.72],  # 边0的目标
            [0.19, 0.52, 0.38, 0.71, 0.85, 0.09, 0.61, 0.76, ..., 0.72],  # 边1的目标
            [0.19, 0.52, 0.38, 0.71, 0.85, 0.09, 0.61, 0.76, ..., 0.72],  # 边2的目标
            [0.19, 0.52, 0.38, 0.71, 0.85, 0.09, 0.61, 0.76, ..., 0.72]   # 边3的目标
        ])
        # shape: (4, 64)
        """
        
        # 计算边级 Q 值
        e2 = self.f_e(th.cat([src_feats, edge_feats, dst_feats], dim=1))
        # 所有边的Q值
        # edge_out = tensor([
        #     [0.8523],  # 选择S2的Q值
        #     [0.9234],  # 选择S3的Q值 (最高！)
        #     [0.6789],  # 选择S4的Q值 (最低，链路差)
        #     [0.7345]   # 选择S5的Q值
        # ])
        # # shape: (4, 1)
        """
        拼接 [邻居特征 x_nbr, Agent特征 x, 链路特征 edge_feats]。
        这个评分综合考量了邻居的状态（是否空闲）、链路的状态（是否拥堵）以及我自己的状态（是否有任务要发）
        """
        """
        # === 边0的输入准备 ===
            src_feat_0 = [0.5, 0.3, 0.8, 0.2, 0.1, 0.9]  # S2特征 (6,)
            edge_feat_0 = [0.8, 0.6, 0.9]                 # S2→S1链路 (3,)
            dst_feat_0 = [0.19, 0.52, 0.38, 0.71, 0.85, 0.09, 0.61, 0.76, ..., 0.72]  # S1特征 (64,)

            # 拼接
            edge_input_0 = torch.cat([src_feat_0, edge_feat_0, dst_feat_0])
            edge_input_0 = [
                # S2的状态（6维）
                0.5, 0.3, 0.8, 0.2, 0.1, 0.9,
                # S2→S1链路质量（3维）
                0.8, 0.6, 0.9,
                # S1(Agent)状态（64维）
                0.19, 0.52, 0.38, 0.71, 0.85, 0.09, 0.61, 0.76, ..., 0.72
            ]
            # shape: (73,)
          
        通过 f_edge 第1层（Linear + ReLU）
           # 第1层: Linear(73 → 64)
            weights_1 = torch.randn(64, 73)  # 权重矩阵（实际是训练好的）
            bias_1 = torch.randn(64)         # 偏置向量

            hidden_0 = edge_input_0 @ weights_1.T + bias_1
            hidden_0 = [0.45, -0.23, 0.78, 0.91, -0.12, 0.34, 0.67, -0.56, 0.89, 0.23, ..., 0.67]
            # shape: (64,)

            # ReLU激活（负值变0）
            hidden_0 = torch.relu(hidden_0)
            hidden_0 = [0.45, 0.00, 0.78, 0.91, 0.00, 0.34, 0.67, 0.00, 0.89, 0.23, ..., 0.67]
            # shape: (64,)
            # [维度0-5] S2的状态
                # - 内存占用: 0.5
                # - CPU负载: 0.3
                # - 可用带宽: 0.8
                # - 延迟: 0.2
                # - 队列长度: 0.1
                # - 剩余能量: 0.9

                # [维度6-8] S2→S1的链路状态
                # - 传输延迟: 0.8
                # - 带宽占用率: 0.6
                # - 信号强度: 0.9

                # [维度9-72] S1(Agent)的状态
                # - 任务信息: 是否在处理、任务类型、数据包大小等
                # - 资源状态: 内存、CPU、带宽等
                # - 历史信息: RNN编码的时序状态

        通过 f_edge 第1层（Linear + ReLU）
        # 第1层: Linear(73 → 64)
            weights_1 = torch.randn(64, 73)  # 权重矩阵（实际是训练好的）
            bias_1 = torch.randn(64)         # 偏置向量

            hidden_0 = edge_input_0 @ weights_1.T + bias_1
            hidden_0 = [0.45, -0.23, 0.78, 0.91, -0.12, 0.34, 0.67, -0.56, 0.89, 0.23, ..., 0.67]
            # shape: (64,)

            # ReLU激活（负值变0）
            hidden_0 = torch.relu(hidden_0)
            hidden_0 = [0.45, 0.00, 0.78, 0.91, 0.00, 0.34, 0.67, 0.00, 0.89, 0.23, ..., 0.67]
            # shape: (64,)

            # 这是对"选择S2作为下一跳"的多维度评估
            # - 维度0-10: S2的资源充足度评分
            # - 维度11-20: S2→S1链路质量评分
            # - 维度21-30: S2与S1任务需求的匹配度
            # - 维度31-40: S2的地理位置优势
            # - 维度41-50: S2的历史性能评分
            # - 维度51-64: 综合考量的高层特征

        通过 f_edge 第2层（Linear）
        # 第2层: Linear(64 → 1)
            weights_2 = torch.randn(1, 64)  # 权重向量
            bias_2 = torch.randn(1)         # 偏置标量

            q_edge_0 = hidden_0 @ weights_2.T + bias_2
            q_edge_0 = tensor([0.8523])  # 标量Q值
            # shape: (1,)
            # Q(选择S2) = 0.8523
            # 
            # 这个值越大，表示：
            # - S2有足够的资源处理任务
            # - S2→S1的链路质量好
            # - S2能够满足S1的需求
            # - 综合来看，S2是一个好的选择


        """
        # 计算节点级 Q 值
        v2_j = self.f_v(dst_node_feats)
        """
        意义：这个评分只看我自己的状态（比如我的计算队列是否满了，我的剩余电量如何）。
        评估“我自己处理”的质量。
        
        """
        
        return v2_j, e2


class NeighborSelector(nn.Module):
    """
    邻居选择器 - PyTorch Geometric 版本
    参考 cross_layer_opt_with_grl-main/modules/graph_nn.py
    
    为每个邻居计算得分，并 padding 到固定长度。
    用于离散动作空间的 Q 值输出。
    """
    
    def __init__(
        self,
        nbr_in_feats: int,          # 邻居输入特征维度
        agent_in_feats: int,        # Agent 输入特征维度
        nbr_out_feats: int,         # 每个邻居的输出维度（如功率档位数）
        agent_out_feats: int,       # Agent 自身输出维度（如1，本地计算）
        hidden_size: int,           # 隐藏层维度
        max_nbrs: int = 4,          # 最大邻居数
        activation: str = 'relu',
        device: str = 'cpu',
    ):
        super(NeighborSelector, self).__init__()
        
        if not DGL_AVAILABLE:
            raise ImportError("PyTorch Geometric is required for NeighborSelector")
        
        self.nbr_in_feats = nbr_in_feats
        self.agent_in_feats = agent_in_feats
        self.nbr_out_feats = nbr_out_feats
        self.agent_out_feats = agent_out_feats
        self.max_nbrs = max_nbrs
        self.device = device
        
        self._hidden_size = hidden_size
        self._activation = get_activation(activation)
        
        # 邻居预测器: [邻居特征, agent特征] -> 邻居得分
        self.nbr_predictor = nn.Sequential(
            nn.Linear(nbr_in_feats + agent_in_feats, self._hidden_size),
            self._activation(),
            nn.Linear(self._hidden_size, nbr_out_feats)
        )
        
        # Agent 自身预测器: agent特征 -> 自身动作得分
        self.agent_predictor = nn.Linear(self._hidden_size, agent_out_feats)
    
    def forward(self, edge_index, x, num_agents=None):
        """
        Args:
            edge_index: PyG 边索引 [2, num_edges] (nbr -> agent 的边)
            x: 节点特征字典 {'agent': tensor, 'nbr': tensor}
            num_agents: agent 数量（可选）
        
        Returns:
            all_scores: (batch, max_nbrs * nbr_out_feats + agent_out_feats)
        """
        src_idx = edge_index[0]  # nbr indices
        dst_idx = edge_index[1]  # agent indices
        
        # 获取特征
        nbr_feats = x['nbr'][src_idx] if 'nbr' in x else x[src_idx]
        agent_feats = x['agent'][dst_idx] if 'agent' in x else x[dst_idx]
        
        # 计算邻居得分
        nbr_scores = self.nbr_predictor(th.cat([nbr_feats, agent_feats], dim=1))
        
        # 计算 agent 自身得分
        agent_x = x['agent'] if 'agent' in x else x
        own_score = self.agent_predictor(agent_x)
        
        if num_agents is None:
            num_agents = agent_x.size(0)
        
        # 计算每个 agent 的邻居数量
        from torch_geometric.utils import degree
        nbrs_per_agent = degree(dst_idx, num_nodes=num_agents, dtype=th.long).tolist()
        
        # Padding 到固定长度
        nbr_scores_split = th.split(nbr_scores, split_size_or_sections=nbrs_per_agent, dim=0)
        
        padded_nbr_scores = []
        for agent_idx, score in enumerate(nbr_scores_split):
            n_nbrs = nbrs_per_agent[agent_idx]
            if n_nbrs < self.max_nbrs:
                pad = th.zeros(self.nbr_out_feats * (self.max_nbrs - n_nbrs), 
                              dtype=th.float, device=score.device if score.numel() > 0 else self.device)
                if score.numel() > 0:
                    padded_nbr_scores.append(th.cat([score.flatten(), pad]))
                else:
                    padded_nbr_scores.append(pad)
            else:
                padded_nbr_scores.append(score.flatten()[:self.max_nbrs * self.nbr_out_feats])
        
        padded_nbr_scores = th.stack(padded_nbr_scores)
        
        all_scores = th.cat([padded_nbr_scores, own_score], dim=1)
        return all_scores


def pad_edge_output(edge_index, edge_output, max_nbrs: int, num_dst_nodes=None):
    """
    将边输出 padding 到固定长度 - PyTorch Geometric 版本
    
    Args:
        edge_index: PyG 边索引 [2, num_edges]
        edge_output: 边输出 (n_edges, out_feats)
        max_nbrs: 最大邻居数
        num_dst_nodes: 目标节点数量（可选）
    
    Returns:
        padded: (batch, max_nbrs * out_feats)
    """
    device = edge_output.device
    out_feats = edge_output.size(-1)
    
    dst_idx = edge_index[1]
    
    if num_dst_nodes is None:
        num_dst_nodes = dst_idx.max().item() + 1
    
    # 计算每个目标节点的邻居数量
    from torch_geometric.utils import degree
    nbrs_per_agent = degree(dst_idx, num_nodes=num_dst_nodes, dtype=th.long).tolist()
    
    edge_outputs = th.split(edge_output, split_size_or_sections=nbrs_per_agent, dim=0)
    
    padded_outputs = []
    for agent_idx, out in enumerate(edge_outputs):
        n_nbrs = nbrs_per_agent[agent_idx]
        if n_nbrs < max_nbrs:
            pad = th.zeros((max_nbrs - n_nbrs) * out_feats, dtype=th.float, device=device)
            if out.numel() > 0:
                padded_outputs.append(th.cat([out.flatten(), pad]))
            else:
                padded_outputs.append(pad)
        else:
            padded_outputs.append(out.flatten()[:max_nbrs * out_feats])
    
    return th.stack(padded_outputs)
