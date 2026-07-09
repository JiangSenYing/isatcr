"""
图神经网络控制器 (GNN Controllers / Agents)
参考 cross_layer_opt_with_grl-main/modules/agents/customized_agents.py

实现两种 GNN 控制器：
1. RelationalController: 使用 GATv2Conv 注意力机制
2. GraphController: 使用 NodeGNBlock + EdgeGNBlock 消息传递
"""

from typing import Dict, Mapping, Optional, Union
import torch as th
from torch import Tensor
import torch.nn as nn

try:
    from torch_geometric.data import HeteroData, Data
    from . import pyg_compat
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False

# 为向后兼容保留 DGL_AVAILABLE 变量名
DGL_AVAILABLE = PYG_AVAILABLE

from .activations import get_activation
from .basics import RnnLayer, DuelingLayer
from .encoders import RelationalEncoder, FlatEncoder
from .gn_blocks import NodeGNBlock, NodeGATBlock, EdgeGNBlock, NeighborSelector, pad_edge_output


class RelationalController(nn.Module):
    """
    关系型图神经网络控制器
    参考 cross_layer_opt_with_grl-main 的 AdHocRelationalController
    
    架构:
    1. RelationalEncoder: 使用 GATv2Conv 编码异构图观测
    2. RnnLayer: GRU 维护时序依赖
    3. NeighborSelector: 为每个邻居计算 Q 值
    
    适用于:
    - 动态邻居数量的场景
    - 需要注意力机制选择重要邻居
    """
    
    def __init__(
        self,
        obs_shape: Dict[str, int],   # 各类型节点的特征维度 {'agent': 9, 'nbr': 6}
        n_actions: int,              # 动作数 (max_nbrs + 1 for local computing)
        hidden_size: int = 64,       # 隐藏层维度
        max_nbrs: int = 4,           # 最大邻居数
        n_heads: int = 4,            # 注意力头数（仅 GAT 使用）
        conv_type: str = 'gat',      # 图卷积类型: 'gat' 或 'gcn'
        use_rnn: bool = True,        # 是否使用 RNN
        use_layer_norm: bool = False,
        dueling: bool = True,        # 是否使用 Dueling 架构
        activation: str = 'relu',
        device: str = 'cpu',
    ):
        super(RelationalController, self).__init__()
        
        if not DGL_AVAILABLE:
            raise ImportError("PyTorch Geometric is required for RelationalController")
        
        assert conv_type in ('gat', 'gcn'), f"conv_type must be 'gat' or 'gcn', got {conv_type}"
        
        self._obs_shape = obs_shape
        self._n_actions = n_actions
        self._hidden_size = hidden_size
        self._max_nbrs = max_nbrs
        self._conv_type = conv_type
        self._use_rnn = use_rnn
        self._dueling = dueling
        self.device = device
        
        # 1. 观测编码器
        self.f_enc = RelationalEncoder(
            in_feats_size_dict=obs_shape,
            hidden_size=hidden_size,
            n_heads=n_heads,
            conv_type=conv_type,
            activation=activation
        )
        
        # 1.5 邻居特征编码器（解决邻居特征未编码问题）
        # 将原始邻居特征投影到与 agent 编码相同的维度
        nbr_raw_dim = obs_shape.get('nbr', 6)
        act_fn = get_activation(activation)
        self.nbr_encoder = nn.Sequential(
            nn.Linear(nbr_raw_dim, hidden_size),
            act_fn(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # 2. RNN 层（可选）
        if use_rnn:
            self.rnn = RnnLayer(hidden_size, use_layer_norm=use_layer_norm)
        
        # 3. 输出层（邻居选择器）
        # 每个邻居输出 1 个 Q 值，agent 自身输出 1 个 Q 值（本地计算）
        # 注意：使用编码后的邻居特征维度（hidden_size），而不是原始维度
        self.f_out = NeighborSelector(
            nbr_in_feats=hidden_size,    # 使用编码后的维度
            agent_in_feats=hidden_size,
            nbr_out_feats=1,        # 每个邻居 1 个动作
            agent_out_feats=1,      # 本地计算 1 个动作
            hidden_size=hidden_size,
            max_nbrs=max_nbrs,
            activation=activation,
            device=device
        )
        
        # 4. Dueling 架构（可选）
        if dueling:
            # 状态价值函数 V(s)
            self.value_head = nn.Linear(hidden_size, 1)
            # 优势函数 A(s, a) - 注意这里输出的是标量调整，不是每个动作单独的优势
            # 实际上我们会在 forward 中对 f_out 的输出进行 dueling 组合
    
    def init_hidden(self, batch_size: int = 1) -> Tensor:
        """初始化 RNN 隐状态"""
        return th.zeros(batch_size, self._hidden_size, device=self.device)
    
    def forward(self, obs: Union[HeteroData, 'pyg_compat.PyGHeteroGraph'], h: Optional[Tensor] = None):
        """
        Args:
            obs: PyG 异构图观测 (HeteroData)
            h: RNN 隐状态 (batch, hidden_size)
        
        Returns:
            q_vals: Q 值 (batch, n_actions)
            h: 新隐状态
        """
        # 1. 编码观测
        x = self.f_enc(obs)
        
        # 2. RNN 更新
        if self._use_rnn:
            if h is None:
                h = self.init_hidden(x.size(0))
            x, h = self.rnn(x, h)
        
        # 3. 计算 Q 值
        # 获取边索引和特征
        # PyG HeteroData: obs[('nbr', 'nearby', 'agent')].edge_index 或 obs[('nbr', '1hop', 'agent')].edge_index
        edge_type = None
        for etype in [('nbr', 'nearby', 'agent'), ('nbr', '1hop', 'agent')]:
            if etype in obs.edge_types:
                edge_type = etype
                break
        
        if edge_type is None:
            raise ValueError(f"Expected edge type ('nbr', 'nearby', 'agent') or ('nbr', '1hop', 'agent'), "
                           f"but got edge types: {obs.edge_types}")
        
        edge_index = obs[edge_type].edge_index
        
        # 获取原始邻居特征并编码
        nbr_raw = obs['nbr'].feat if hasattr(obs['nbr'], 'feat') else obs['nbr'].x
        nbr_encoded = self.nbr_encoder(nbr_raw)  # [n_nbr, hidden_size]
        
        node_feats = {
            'agent': x,
            'nbr': nbr_encoded  # 使用编码后的邻居特征
        }
        q_vals = self.f_out(edge_index, node_feats)
        
        return q_vals, h


class GraphController(nn.Module):
    """
    消息传递图神经网络控制器
    参考 cross_layer_opt_with_grl-main 的 AdHocGraphController
    
    架构:
    1. NodeGNBlock/NodeGATBlock: 编码阶段，聚合邻居信息（支持 1/2/3 跳）
    2. RnnLayer: GRU 维护时序依赖
    3. EdgeGNBlock: 输出阶段，为每条边计算 Q 值
    
    适用于:
    - 需要利用边特征的场景
    - 多跳邻居信息传递（1/2/3 跳）
    - 精细的路由决策
    """
    
    def __init__(
        self,
        obs_shape: Dict[str, int],   # {'agent': 9, 'nbr': 6, 'hop': 3}
        n_actions: int,              # 动作数
        hidden_size: int = 64,
        max_nbrs: int = 4,
        n_hops: int = 1,             # GNN 跳数 (1, 2 或 3)
        enc_agg_type: str = 'gn',    # 编码聚合类型: 'gn' 或 'gat'
        gat_heads: int = 4,          # GAT 注意力头数（enc_agg_type='gat' 时生效）
        gat_dropout: float = 0.0,    # GAT dropout（enc_agg_type='gat' 时生效）
        use_rnn: bool = True,
        use_layer_norm: bool = False,
        dueling: bool = True,        # 是否使用 Dueling 架构
        activation: str = 'relu',
        device: str = 'cpu',
    ):
        super(GraphController, self).__init__()
        
        if not DGL_AVAILABLE:
            raise ImportError("PyTorch Geometric is required for GraphController")
        
        self._obs_shape = obs_shape
        self._n_actions = n_actions
        self._hidden_size = hidden_size
        self._max_nbrs = max_nbrs
        self._n_hops = n_hops
        self._enc_agg_type = enc_agg_type
        self._use_rnn = use_rnn
        self._dueling = dueling
        self.device = device

        if enc_agg_type not in ('gn', 'gat'):
            raise ValueError(f"enc_agg_type must be 'gn' or 'gat', got {enc_agg_type}")
        
        # 边特征维度（如果没有则设为 0）
        hop_feats = obs_shape.get('hop', 0)
        if hop_feats == 0:
            # 没有边特征时，创建一个虚拟的边特征维度
            hop_feats = 1
            self._use_dummy_edge_feats = True
        else:
            self._use_dummy_edge_feats = False
        """
        # 我们的场景
            obs_shape = {'agent': 9, 'nbr': 6, 'hop': 3}
            hop_feats = 3  # 边特征维度
            self._use_dummy_edge_feats = False  # 使用真实边特征

            # 边特征可能包含：
            # [传输延迟, 带宽占用率, 链路质量]
        """
        
        # 1. 编码器
        if enc_agg_type == 'gat':
            def build_enc(in_node_feats):
                return NodeGATBlock(
                    in_node_feats,
                    hop_feats,
                    hidden_size,
                    n_heads=gat_heads,
                    dropout=gat_dropout,
                    activation=activation,
                )
        else:
            def build_enc(in_node_feats):
                return NodeGNBlock(
                    in_node_feats,
                    hop_feats,
                    hidden_size,
                    activation=activation,
                )

        if n_hops == 1:
            self.enc = nn.ModuleDict({
                '1hop': build_enc((obs_shape['nbr'], obs_shape['agent']))
            })
            """
            NodeGNBlock 的作用:
            输入: 邻居特征 (6维) + Agent特征 (9维) + 边特征 (3维)
            输出: Agent的聚合表示 (64维)
            """
            """
            # 数值流动
              输入
                邻居1特征: [0.5, 0.3, 0.8, 0.2, 0.1, 0.9]  # 6维
                邻居2特征: [0.4, 0.7, 0.2, 0.6, 0.5, 0.3]
                邻居3特征: [0.9, 0.1, 0.4, 0.8, 0.2, 0.6]
                邻居4特征: [0.3, 0.6, 0.9, 0.1, 0.7, 0.4]

                Agent特征: [0.5, 0.2, 0.8, 0.1, 0.6, 0.3, 0.9, 0.4, 0.7]  # 9维

                边1特征: [0.8, 0.6, 0.9]  # 到邻居1的链路
                边2特征: [0.7, 0.5, 0.8]
                边3特征: [0.9, 0.7, 0.6]
                边4特征: [0.6, 0.8, 0.7]

                # NodeGNBlock 处理后
                Agent聚合表示: [0.21, 0.45, ..., 0.83]  # 64维 - 融合了所有邻居信息
            """
        elif n_hops == 2:
            self.enc = nn.ModuleDict({
                '2hop': build_enc((obs_shape['nbr'], obs_shape['nbr'])),
                '1hop': build_enc((hidden_size, obs_shape['agent']))
            })
            """
            第1跳: 邻居的邻居 → 邻居
                卫星S2的邻居(S5,S6) → 卫星S2
                卫星S3的邻居(S7,S8) → 卫星S3
                ...

            第2跳: 更新后的邻居 → Agent
                卫星S2(已知S5,S6信息) → 卫星S1
                卫星S3(已知S7,S8信息) → 卫星S1
                ...

            结果: Agent 现在知道 2跳范围内所有卫星的信息
            
            """
        elif n_hops == 3:
            self.enc = nn.ModuleDict({
                '3hop': build_enc((obs_shape['nbr'], obs_shape['nbr'])),
                '2hop': build_enc((hidden_size, obs_shape['nbr'])),
                '1hop': build_enc((hidden_size, obs_shape['agent']))
            })
            # 投影层：当没有3hop边时，将nbr特征投影到hidden_size维度
            self.nbr_proj = nn.Linear(obs_shape['nbr'], hidden_size)
            """
            3跳消息传递流程:
            
            第1层: 3跳邻居 → 2跳邻居
                卫星 S7,S8 的信息 → 卫星 S3
                卫星 S9,S10 的信息 → 卫星 S4
                ...
            
            第2层: 2跳邻居(已知道3跳信息) → 1跳邻居
                卫星 S3(已知S7,S8信息) → 卫星 S2
                卫星 S4(已知S9,S10信息) → 卫星 S2
                ...
            
            第3层: 1跳邻居(已知道2跳+3跳信息) → Agent
                卫星 S2(已知S3,S4,...信息) → 卫星 S1 (Agent)
                ...
            
            结果: Agent 现在知道 3跳范围内所有卫星的信息
            """
        else:
            raise ValueError(f"n_hops must be 1, 2 or 3, got {n_hops}")
        
        # 2. RNN 层
        if use_rnn:
            self.rnn = RnnLayer(hidden_size, use_layer_norm=use_layer_norm)
            """
            RnnLayer 的作用:
            # t=1 时刻
                h_0 = [0, 0, ..., 0]  # 初始隐状态 (64维)
                x_1 = [0.21, 0.45, ..., 0.83]  # 当前观测
                x_1, h_1 = rnn(x_1, h_0)
                # h_1 = [0.15, 0.32, ..., 0.67]  # 更新后的隐状态，包含历史信息

                # t=2 时刻
                x_2 = [0.34, 0.56, ..., 0.91]
                x_2, h_2 = rnn(x_2, h_1)  # 利用上一时刻的记忆
                # h_2 = [0.28, 0.41, ..., 0.75]
            
            """
        
        # 3. 输出层
        inter_nbr_feats = hidden_size if n_hops >= 2 else obs_shape['nbr']
        # 1跳模式: inter_nbr_feats = 6 (原始邻居特征)
        # 2/3跳模式: inter_nbr_feats = 64 (GNN编码后的邻居特征)

        self.f_out = EdgeGNBlock(
            (inter_nbr_feats, hidden_size),
            hop_feats,
            out_node_feats=1,    # 本地计算 Q 值
            out_edge_feats=1,    # 每个邻居选择 Q 值
            hidden_size=hidden_size,
            activation=activation
        )
        """
        EdgeGNBlock 的作用:

        为每个邻居（每条边）计算一个 Q 值
        为 Agent 本地计算一个 Q 值
        """
        
        self._max_nbrs = max_nbrs
        
        # 4. Dueling 架构（可选）
        if dueling:
            self.value_head = nn.Linear(hidden_size, 1)
    
    def init_hidden(self, batch_size: int = 1) -> Tensor:
        """初始化 RNN 隐状态"""
        return th.zeros(batch_size, self._hidden_size, device=self.device)
    
    def _get_edge_feats(self, obs: HeteroData, edge_type: tuple):
        """获取边特征，如果没有则返回虚拟特征"""
        if self._use_dummy_edge_feats:
            n_edges = obs[edge_type].edge_index.size(1)
            return th.ones(n_edges, 1, device=self.device)
        else:
            return obs[edge_type].edge_attr if hasattr(obs[edge_type], 'edge_attr') else obs[edge_type].feat
    
    def forward(self, obs: Union[HeteroData, 'pyg_compat.PyGHeteroGraph'], h: Optional[Tensor] = None):
        """
        Args:
            obs: PyG 异构图观测 (HeteroData)
            h: RNN 隐状态
        
        Returns:
            q_vals: Q 值 (batch, n_actions)
            h: 新隐状态
        """
        # 获取节点特征
        feat = {}
        for ntype in obs.node_types:
            if hasattr(obs[ntype], 'feat'):
                feat[ntype] = obs[ntype].feat
            elif hasattr(obs[ntype], 'x'):
                feat[ntype] = obs[ntype].x
        """
        obs.node_types = ['agent', 'nbr']

        # Agent 节点特征 (1个agent节点)
        feat['agent'] = tensor([[0.5, 0.2, 0.8, 0.1, 0.6, 0.3, 0.9, 0.4, 0.7]])
        # shape: (1, 9)

        # 邻居节点特征 (4个邻居节点)
        feat['nbr'] = tensor([
            [0.5, 0.3, 0.8, 0.2, 0.1, 0.9],  # 邻居0 (S2)
            [0.4, 0.7, 0.2, 0.6, 0.5, 0.3],  # 邻居1 (S3)
            [0.9, 0.1, 0.4, 0.8, 0.2, 0.6],  # 邻居2 (S4)
            [0.3, 0.6, 0.9, 0.1, 0.7, 0.4]   # 邻居3 (S5)
        ])
        # shape: (4, 6)
        
        """        
        # 1. GNN 编码
        # 定义边类型 - 支持多种命名方式
        edge_type_3hop = ('nbr', '3hop', 'nbr')
        edge_type_2hop = ('nbr', '2hop', 'nbr')
        
        # 1跳边类型可能是 'nearby' 或 '1hop'
        edge_type_1hop = None
        for etype in [('nbr', 'nearby', 'agent'), ('nbr', '1hop', 'agent')]:
            if etype in obs.edge_types:
                edge_type_1hop = etype
                break
        
        if edge_type_1hop is None:
            raise ValueError(f"Expected edge type ('nbr', 'nearby', 'agent') or ('nbr', '1hop', 'agent'), "
                           f"but got edge types: {obs.edge_types}")
        
        # 3跳模式：先处理 3hop -> 2hop -> 1hop -> agent
        if self._n_hops == 3:
            # 第1层：3跳邻居 -> 2跳邻居
            if edge_type_3hop in obs.edge_types:
                edge_index_3hop = obs[edge_type_3hop].edge_index
                edge_feats_3hop = self._get_edge_feats(obs, edge_type_3hop)
                # 3跳邻居聚合到2跳邻居，更新所有nbr特征到hidden_size维
                x_nbr = self.enc['3hop'](edge_index_3hop, feat['nbr'], edge_feats_3hop)
            else:
                # 没有3hop边时，使用投影层将nbr特征投影到hidden_size维度
                x_nbr = self.nbr_proj(feat['nbr'])
            
            # 第2层：2跳邻居(已含3跳信息) -> 1跳邻居
            if edge_type_2hop in obs.edge_types:
                edge_index_2hop = obs[edge_type_2hop].edge_index
                edge_feats_2hop = self._get_edge_feats(obs, edge_type_2hop)
                x_nbr = self.enc['2hop'](edge_index_2hop, (x_nbr, feat['nbr']), edge_feats_2hop)
            # 注意：x_nbr 现在包含了2跳和3跳邻居的信息
            
        # 2跳模式：处理 2hop -> 1hop -> agent
        elif self._n_hops == 2 and edge_type_2hop in obs.edge_types:
            edge_index_2hop = obs[edge_type_2hop].edge_index#提取 2跳边的连接关系。
            """
            # 邻居之间的连接（2跳范围）
                edge_index_2hop = tensor([
                    [0, 0, 1, 1, 2, 2, 3, 3],  # 源邻居索引
                    [1, 2, 0, 3, 1, 3, 0, 2]   # 目标邻居索引
                ])

                    # 边的含义：
                    # 边0: 邻居0(S2) → 邻居1(S3)
                    # 边1: 邻居0(S2) → 邻居2(S4)
                    # 边2: 邻居1(S3) → 邻居0(S2)
                    # 边3: 邻居1(S3) → 邻居3(S5)
                    # 边4: 邻居2(S4) → 邻居0(S2)
                    # 边5: 邻居2(S4) → 邻居3(S5)
                    # 边6: 邻居3(S5) → 邻居1(S3)
                    # 边7: 邻居3(S5) → 邻居2(S4)
            """
            edge_feats_2hop = self._get_edge_feats(obs, edge_type_2hop)#提取 2跳边的特征（如链路带宽、延迟等）。
            """
            # 数据
                edge_feats_2hop = tensor([
                    [0.7, 0.8, 0.6],  # S2→S3 链路特征
                    [0.6, 0.7, 0.5],  # S2→S4 链路特征
                    [0.7, 0.8, 0.6],  # S3→S2 链路特征
                    [0.8, 0.6, 0.7],  # S3→S5 链路特征
                    # ... 更多边
                ])
                # shape: (n_edges_2hop, 3)
            """
            x_nbr = self.enc['2hop'](edge_index_2hop, # (2, 8) 邻居间的连接
                                     feat['nbr'], # (4, 6) 原始邻居特征
                                     edge_feats_2hop) # (8, 3) 邻居间的边特征
            """
            edge_index_2hop = tensor([
                [0, 0, 1, 1, 2, 2, 3, 3],  # 源节点索引（发送消息的邻居）
                [1, 2, 0, 3, 0, 3, 1, 2]   # 目标节点索引（接收消息的邻居）
            ])
            # shape: (2, 8) - 8条边

            feat['nbr'] = tensor([
                [0.5, 0.3, 0.8, 0.2, 0.1, 0.9],  # S2 特征 (内存占用, CPU负载, 带宽, 延迟, 队列长度, 能量)
                [0.4, 0.7, 0.2, 0.6, 0.5, 0.3],  # S3 特征
                [0.9, 0.1, 0.4, 0.8, 0.2, 0.6],  # S4 特征
                [0.3, 0.6, 0.9, 0.1, 0.7, 0.4]   # S5 特征
            ])
            # shape: (4, 6) - 4个邻居，每个6维特征


            edge_feats_2hop = tensor([
                [0.8, 0.6, 0.9],  # 边0: S2→S3 链路 [传输延迟, 带宽占用, 信号强度]
                [0.7, 0.5, 0.8],  # 边1: S2→S4 链路
                [0.8, 0.6, 0.9],  # 边2: S3→S2 链路 (双向链路，可能相同或不同)
                [0.6, 0.7, 0.7],  # 边3: S3→S5 链路
                [0.7, 0.5, 0.8],  # 边4: S4→S2 链路
                [0.5, 0.8, 0.6],  # 边5: S4→S5 链路
                [0.6, 0.7, 0.7],  # 边6: S5→S3 链路
                [0.5, 0.8, 0.6]   # 边7: S5→S4 链路
            ])
            # shape: (8, 3) - 8条边，每条边3维特征

            
            """
            #在邻居节点集合内部进行消息传递。每个邻居节点会收集与它相连的其他节点的信息。
            #输出 (x_nbr)：更新后的邻居特征。
        else:
            x_nbr = feat['nbr']
            """
            x_nbr = tensor([
                [0.5, 0.3, 0.8, 0.2, 0.1, 0.9],  # 邻居0
                [0.4, 0.7, 0.2, 0.6, 0.5, 0.3],  # 邻居1
                [0.9, 0.1, 0.4, 0.8, 0.2, 0.6],  # 邻居2
                [0.3, 0.6, 0.9, 0.1, 0.7, 0.4]   # 邻居3
            ])
            # shape: (4, 6)
            """
        
        edge_index_1hop = obs[edge_type_1hop].edge_index#获取所有指向中心 Agent 的边。对应 ('nbr', '1hop', 'agent') 或 ('nbr', 'nearby', 'agent') 类型的边。
        """
           # shape: (2, 4) - 2行表示源节点和目标节点，4列表示4条边
           edge_index_1hop = tensor([
            [0, 1, 2, 3],  # 源节点索引（邻居节点）
            [0, 0, 0, 0]   # 目标节点索引（都指向agent 0）
            ])

            # 含义：
            # 边0: 邻居0 → Agent0
            # 边1: 邻居1 → Agent0
            # 边2: 邻居2 → Agent0
            # 边3: 邻居3 → Agent0       
        """
        edge_feats_1hop = self._get_edge_feats(obs, edge_type_1hop)#提取链路特征：获取邻居到 Agent 之间链路的状态。
        x = self.enc['1hop'](edge_index_1hop, (x_nbr, feat['agent']), edge_feats_1hop)
        #输出 (x)：Agent 的最终隐藏状态；这个向量 x 现在不仅包含 Agent 自己的信息，还包含了它周围所有邻居（以及邻居的邻居）的概况。
        
        # 2. RNN 更新
        if self._use_rnn:
            if h is None:
                h = self.init_hidden(x.size(0))
            x, h = self.rnn(x, h)
        
        # 3. 输出 Q 值
        agent_out, nbr_out = self.f_out(edge_index_1hop, (x_nbr, x), edge_feats_1hop)
        """
        edge_index_1hop:定义了谁是我的邻居（边的连接关系）。
        x_nbr (Source)：邻居节点的特征向量（如果开启了 2-hop,这里已经包含了邻居的邻居的信息）。
        x (Destination):Agent 自身的特征向量（已经融合了所有邻居的信息，是全局视野的浓缩）。 
        edge_feats_1hop:链路特征（如带宽、队列长度）。

        edge_index_1hop = tensor([
            [0, 1, 2, 3],  # 源节点索引（邻居索引）
            [0, 0, 0, 0]   # 目标节点索引（都指向Agent 0）
        ])

        # x_nbr: 邻居特征（可能是原始6维或编码后64维）
        # 1跳模式
        x_nbr = tensor([
            [0.5, 0.3, 0.8, 0.2, 0.1, 0.9],  # S2 原始特征
            [0.4, 0.7, 0.2, 0.6, 0.5, 0.3],  # S3 原始特征
            [0.9, 0.1, 0.4, 0.8, 0.2, 0.6],  # S4 原始特征
            [0.3, 0.6, 0.9, 0.1, 0.7, 0.4]   # S5 原始特征
        ])
        # shape: (4, 6) - 1跳模式

        # 或 2跳模式
        x_nbr = tensor([
            [0.34, 0.56, 0.78, 0.23, 0.91, ..., 0.67],  # S2 编码特征
            [0.45, 0.67, 0.23, 0.89, 0.82, ..., 0.45],  # S3 编码特征
            [0.56, 0.34, 0.89, 0.12, 0.67, ..., 0.78],  # S4 编码特征
            [0.23, 0.78, 0.45, 0.56, 0.34, ..., 0.91]   # S5 编码特征
        ])
        # shape: (4, 64) - 2跳模式

        # x: Agent 经过 GNN+RNN 更新后的特征
        x = tensor([[0.19, 0.52, 0.38, 0.71, 0.85, 0.09, 0.61, 0.76, ..., 0.72]])
        # shape: (1, 64)


        edge_feats_1hop = tensor([
            [0.8, 0.6, 0.9],  # S2→S1 链路
            [0.7, 0.5, 0.8],  # S3→S1 链路
            [0.9, 0.7, 0.6],  # S4→S1 链路
            [0.6, 0.8, 0.7]   # S5→S1 链路
        ])
        # shape: (4, 3)



        """        
        # Padding
        padded_nbr_out = pad_edge_output(edge_index_1hop, nbr_out, self._max_nbrs)
        q_vals = th.cat([padded_nbr_out, agent_out], dim=1)
        
        # Dueling 架构：Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
        if self._dueling:
            value = self.value_head(x)  # [batch, 1]
            advantage = q_vals - q_vals.mean(dim=1, keepdim=True)
            q_vals = value + advantage
        
        return q_vals, h


class SimpleQNetwork(nn.Module):
    """
    简单 MLP Q 网络（非 GNN 版本）
    
    用于:
    - 作为 GNN 的对比基准
    - 不需要图结构的简单场景
    """
    
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_size: int = 128,
        n_layers: int = 2,
        dueling: bool = False,
        activation: str = 'relu',
    ):
        super(SimpleQNetwork, self).__init__()
        
        self._state_dim = state_dim
        self._n_actions = n_actions
        self._hidden_size = hidden_size
        self._dueling = dueling
        
        act_fn = get_activation(activation)
        
        if dueling:
            self.encoder = nn.Sequential(
                nn.Linear(state_dim, hidden_size),
                act_fn(),
                nn.Linear(hidden_size, hidden_size),
                act_fn()
            )
            self.output = DuelingLayer(hidden_size, hidden_size, n_actions, activation)
        else:
            layers = [nn.Linear(state_dim, hidden_size), act_fn()]
            for _ in range(n_layers - 1):
                layers.extend([nn.Linear(hidden_size, hidden_size), act_fn()])
            layers.append(nn.Linear(hidden_size, n_actions))
            self.net = nn.Sequential(*layers)
    
    def forward(self, state: Tensor):
        if self._dueling:
            x = self.encoder(state)
            return self.output(x)
        else:
            return self.net(state)


class TaskConditionedRelationalController(nn.Module):
    """
    【任务特征分离版本】关系型图神经网络控制器
    
    与 RelationalController 的区别：
    - agent 节点特征只包含环境信息（3维），不包含任务信息
    - task_context 作为独立输入，在 GNN 编码后与环境表示融合
    
    优势：
    1. GNN 学到的是纯环境/拓扑表示，与任务无关
    2. 注意力权重只基于网络状态，更稳定
    3. 更好的泛化能力：不同任务使用相同的环境表示
    
    架构:
    1. RelationalEncoder: 编码环境图（纯拓扑信息）
    2. TaskEncoder: 编码任务上下文
    3. Fusion: 融合环境表示和任务表示
    4. RnnLayer: GRU 维护时序依赖（可选）
    5. NeighborSelector: 为每个邻居计算 Q 值
    """
    
    def __init__(
        self,
        obs_shape: Dict[str, int],   # {'agent': 3, 'nbr': 6} - agent 只有环境特征
        task_dim: int = 6,           # 任务上下文维度: mission_state(4) + routing_info(2)
        n_actions: int = 5,
        hidden_size: int = 64,
        max_nbrs: int = 4,
        n_heads: int = 4,
        conv_type: str = 'gat',
        use_rnn: bool = True,
        use_layer_norm: bool = False,
        dueling: bool = True,        # 是否使用 Dueling 架构
        activation: str = 'relu',
        device: str = 'cpu',
    ):
        super(TaskConditionedRelationalController, self).__init__()
        
        if not DGL_AVAILABLE:
            raise ImportError("PyTorch Geometric is required for TaskConditionedRelationalController")
        
        self._obs_shape = obs_shape
        self._task_dim = task_dim
        self._n_actions = n_actions
        self._hidden_size = hidden_size
        self._max_nbrs = max_nbrs
        self._use_rnn = use_rnn
        self._dueling = dueling
        self.device = device
        
        # 1. 环境编码器（GNN）- 只编码拓扑和节点状态
        self.env_encoder = RelationalEncoder(
            in_feats_size_dict=obs_shape,  # agent: 3, nbr: 6（纯环境）
            hidden_size=hidden_size,
            n_heads=n_heads,
            conv_type=conv_type,
            activation=activation
        )
        
        # 2. 任务编码器（MLP）- 编码任务上下文
        act_fn = get_activation(activation)
        self.task_encoder = nn.Sequential(
            nn.Linear(task_dim, hidden_size),
            act_fn(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # 2.5 邻居特征编码器（解决邻居特征未编码问题）
        nbr_raw_dim = obs_shape.get('nbr', 6)
        self.nbr_encoder = nn.Sequential(
            nn.Linear(nbr_raw_dim, hidden_size),
            act_fn(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # 3. 融合层 - 融合环境表示和任务表示
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            act_fn()
        )
        
        # 4. RNN 层（可选）
        if use_rnn:
            self.rnn = RnnLayer(hidden_size, use_layer_norm=use_layer_norm)
        
        # 5. 输出层（邻居选择器）
        # 使用编码后的邻居特征维度
        self.f_out = NeighborSelector(
            nbr_in_feats=hidden_size,    # 使用编码后的维度
            agent_in_feats=hidden_size,
            nbr_out_feats=1,
            agent_out_feats=1,
            hidden_size=hidden_size,
            max_nbrs=max_nbrs,
            activation=activation,
            device=device
        )
        
        # 6. Dueling 架构（可选）
        if dueling:
            self.value_head = nn.Linear(hidden_size, 1)
    
    def init_hidden(self, batch_size: int = 1) -> Tensor:
        """初始化 RNN 隐状态"""
        return th.zeros(batch_size, self._hidden_size, device=self.device)
    
    def forward(
        self, 
        obs: Union[HeteroData, 'pyg_compat.PyGHeteroGraph'], 
        task_context: Tensor,
        h: Optional[Tensor] = None
    ):
        """
        Args:
            obs: PyG 异构图观测（agent 节点只有环境特征）
            task_context: 任务上下文 [batch, task_dim] 或 [task_dim]
            h: RNN 隐状态
        
        Returns:
            q_vals: Q 值 (batch, n_actions)
            h: 新隐状态
        """
        # 处理 task_context 维度
        if task_context.dim() == 1:
            task_context = task_context.unsqueeze(0)
        task_context = task_context.to(self.device)
        
        # 1. GNN 编码环境（纯拓扑，不含任务信息）
        env_embed = self.env_encoder(obs)  # [batch, hidden_size]
        
        # 2. 编码任务上下文
        task_embed = self.task_encoder(task_context)  # [batch, hidden_size]
        
        # 3. 融合环境和任务表示
        x = th.cat([env_embed, task_embed], dim=-1)  # [batch, hidden_size * 2]
        x = self.fusion(x)  # [batch, hidden_size]
        
        # 4. RNN 更新
        if self._use_rnn:
            if h is None:
                h = self.init_hidden(x.size(0))
            x, h = self.rnn(x, h)
        
        # 5. 计算 Q 值
        edge_type = None
        for etype in [('nbr', 'nearby', 'agent'), ('nbr', '1hop', 'agent')]:
            if etype in obs.edge_types:
                edge_type = etype
                break
        
        if edge_type is None:
            raise ValueError(f"Expected edge type ('nbr', 'nearby', 'agent') or ('nbr', '1hop', 'agent'), "
                           f"but got edge types: {obs.edge_types}")
        
        edge_index = obs[edge_type].edge_index
        
        # 获取原始邻居特征并编码
        nbr_raw = obs['nbr'].feat if hasattr(obs['nbr'], 'feat') else obs['nbr'].x
        nbr_encoded = self.nbr_encoder(nbr_raw)  # [n_nbr, hidden_size]
        
        node_feats = {
            'agent': x,
            'nbr': nbr_encoded  # 使用编码后的邻居特征
        }
        q_vals = self.f_out(edge_index, node_feats)
        
        # Dueling 架构：Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
        if self._dueling:
            value = self.value_head(x)  # [batch, 1]
            advantage = q_vals - q_vals.mean(dim=1, keepdim=True)
            q_vals = value + advantage
        
        return q_vals, h


class TaskConditionedGraphController(nn.Module):
    """
    【任务特征分离版本】消息传递图神经网络控制器
    
    与 GraphController 的区别：
    - agent 节点特征只包含环境信息（3维），不包含任务信息
    - task_context 作为独立输入，在 GNN 编码后与环境表示融合
    
    架构:
    1. NodeGNBlock: 编码环境图（多跳消息传递）
    2. TaskEncoder: 编码任务上下文
    3. Fusion: 融合环境表示和任务表示
    4. RnnLayer: GRU 维护时序依赖（可选）
    5. EdgeGNBlock: 输出阶段，为每条边计算 Q 值
    """
    
    def __init__(
        self,
        obs_shape: Dict[str, int],   # {'agent': 3, 'nbr': 6, 'hop': 3}
        task_dim: int = 6,
        n_actions: int = 5,
        hidden_size: int = 64,
        max_nbrs: int = 4,
        n_hops: int = 1,
        use_rnn: bool = True,
        use_layer_norm: bool = False,
        dueling: bool = True,        # 是否使用 Dueling 架构
        activation: str = 'relu',
        device: str = 'cpu',
    ):
        super(TaskConditionedGraphController, self).__init__()
        
        if not DGL_AVAILABLE:
            raise ImportError("PyTorch Geometric is required for TaskConditionedGraphController")
        
        self._obs_shape = obs_shape
        self._task_dim = task_dim
        self._n_actions = n_actions
        self._hidden_size = hidden_size
        self._max_nbrs = max_nbrs
        self._n_hops = n_hops
        self._use_rnn = use_rnn
        self._dueling = dueling
        self.device = device
        
        # 边特征维度
        hop_feats = obs_shape.get('hop', 0)
        if hop_feats == 0:
            hop_feats = 1
            self._use_dummy_edge_feats = True
        else:
            self._use_dummy_edge_feats = False
        
        act_fn = get_activation(activation)
        
        # 1. 环境编码器（GNN）
        if n_hops == 1:
            self.enc = nn.ModuleDict({
                '1hop': NodeGNBlock(
                    (obs_shape['nbr'], obs_shape['agent']),
                    hop_feats,
                    hidden_size,
                    activation=activation
                )
            })
        elif n_hops == 2:
            self.enc = nn.ModuleDict({
                '2hop': NodeGNBlock(
                    (obs_shape['nbr'], obs_shape['nbr']),
                    hop_feats,
                    hidden_size,
                    activation=activation
                ),
                '1hop': NodeGNBlock(                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
                    (hidden_size, obs_shape['agent']),
                    hop_feats,
                    hidden_size,
                    activation=activation
                )
            })
        else:
            raise ValueError(f"n_hops must be 1 or 2, got {n_hops}")
        
        # 2. 任务编码器
        self.task_encoder = nn.Sequential(
            nn.Linear(task_dim, hidden_size),
            act_fn(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # 3. 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            act_fn()
        )
        
        # 4. RNN 层
        if use_rnn:
            self.rnn = RnnLayer(hidden_size, use_layer_norm=use_layer_norm)
        
        # 5. 输出层
        inter_nbr_feats = hidden_size if n_hops == 2 else obs_shape['nbr']
        self.f_out = EdgeGNBlock(
            (inter_nbr_feats, hidden_size),
            hop_feats,
            out_node_feats=1,
            out_edge_feats=1,
            hidden_size=hidden_size,
            activation=activation
        )
        
        # 6. Dueling 架构（可选）
        if dueling:
            self.value_head = nn.Linear(hidden_size, 1)
    
    def init_hidden(self, batch_size: int = 1) -> Tensor:
        """初始化 RNN 隐状态"""
        return th.zeros(batch_size, self._hidden_size, device=self.device)
    
    def _get_edge_feats(self, obs: HeteroData, edge_type: tuple):
        """获取边特征，如果没有则返回虚拟特征"""
        if self._use_dummy_edge_feats:
            n_edges = obs[edge_type].edge_index.size(1)
            return th.ones(n_edges, 1, device=self.device)
        else:
            return obs[edge_type].edge_attr if hasattr(obs[edge_type], 'edge_attr') else obs[edge_type].feat
    
    def forward(
        self, 
        obs: Union[HeteroData, 'pyg_compat.PyGHeteroGraph'],
        task_context: Tensor,
        h: Optional[Tensor] = None
    ):
        """
        Args:
            obs: PyG 异构图观测（agent 节点只有环境特征）
            task_context: 任务上下文 [batch, task_dim] 或 [task_dim]
            h: RNN 隐状态
        
        Returns:
            q_vals: Q 值 (batch, n_actions)
            h: 新隐状态
        """
        # 处理 task_context 维度
        if task_context.dim() == 1:
            task_context = task_context.unsqueeze(0)
        task_context = task_context.to(self.device)
        
        # 获取节点特征
        feat = {}
        for ntype in obs.node_types:
            if hasattr(obs[ntype], 'feat'):
                feat[ntype] = obs[ntype].feat
            elif hasattr(obs[ntype], 'x'):
                feat[ntype] = obs[ntype].x
        
        # 1. GNN 编码环境
        edge_type_2hop = ('nbr', '2hop', 'nbr')# 2跳邻居之间的边
        edge_type_1hop = None# 1跳邻居到agent的边
        for etype in [('nbr', 'nearby', 'agent'), ('nbr', '1hop', 'agent')]:
            if etype in obs.edge_types:
                edge_type_1hop = etype
                break
        
        if edge_type_1hop is None:
            raise ValueError(f"Expected edge type ('nbr', 'nearby', 'agent') or ('nbr', '1hop', 'agent'), "
                           f"but got edge types: {obs.edge_types}")
        
        if self._n_hops == 2 and edge_type_2hop in obs.edge_types:
            edge_index_2hop = obs[edge_type_2hop].edge_index
            edge_feats_2hop = self._get_edge_feats(obs, edge_type_2hop)
            x_nbr = self.enc['2hop'](edge_index_2hop, feat['nbr'], edge_feats_2hop)# 2跳邻居聚合到1跳邻居
        else:
            x_nbr = feat['nbr']
        
        edge_index_1hop = obs[edge_type_1hop].edge_index
        edge_feats_1hop = self._get_edge_feats(obs, edge_type_1hop)
        env_embed = self.enc['1hop'](edge_index_1hop, (x_nbr, feat['agent']), edge_feats_1hop)    # 1跳邻居聚合到agent
        
        # 2. 编码任务上下文
        task_embed = self.task_encoder(task_context)
        
        # 3. 融合环境和任务表示
        x = th.cat([env_embed, task_embed], dim=-1)
        x = self.fusion(x)
        
        # 4. RNN 更新
        if self._use_rnn:
            if h is None:
                h = self.init_hidden(x.size(0))
            x, h = self.rnn(x, h)
        
        # 5. 输出 Q 值
        agent_out, nbr_out = self.f_out(edge_index_1hop, (x_nbr, x), edge_feats_1hop)
        
        # Padding
        padded_nbr_out = pad_edge_output(edge_index_1hop, nbr_out, self._max_nbrs)
        q_vals = th.cat([padded_nbr_out, agent_out], dim=1)
        
        # Dueling 架构：Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
        if self._dueling:
            value = self.value_head(x)  # [batch, 1]
            advantage = q_vals - q_vals.mean(dim=1, keepdim=True)
            q_vals = value + advantage
        
        return q_vals, h


# 延迟导入 V2 控制器（避免循环导入）
def _get_v2_controller():
    """延迟导入 V2 控制器"""
    try:
        from .task_aware_blocks import TaskConditionedGraphControllerV2
        return TaskConditionedGraphControllerV2
    except ImportError:
        return None


class _LazyV2Controller:
    """延迟加载的 V2 控制器包装器"""
    _controller_class = None
    
    @classmethod
    def get_class(cls):
        if cls._controller_class is None:
            cls._controller_class = _get_v2_controller()
        return cls._controller_class
    
    def __new__(cls, *args, **kwargs):
        controller_class = cls.get_class()
        if controller_class is None:
            raise ImportError("TaskConditionedGraphControllerV2 not available")
        return controller_class(*args, **kwargs)


# 控制器注册表
CONTROLLER_REGISTRY = {
    'relational': RelationalController,
    'rel': RelationalController,
    'graph': GraphController,
    'gnn': GraphController,
    'simple': SimpleQNetwork,
    'mlp': SimpleQNetwork,
    # 任务特征分离版本
    'relational_separated': TaskConditionedRelationalController,
    'rel_separated': TaskConditionedRelationalController,
    'graph_separated': TaskConditionedGraphController,
    'gnn_separated': TaskConditionedGraphController,
    # V2 版本（任务感知邻居选择）
    'graph_separated_v2': _LazyV2Controller,
    'gnn_separated_v2': _LazyV2Controller,
}
