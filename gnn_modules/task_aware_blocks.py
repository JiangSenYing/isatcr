"""
任务感知的 GN Blocks - 改进版

核心改进：在邻居选择阶段，让 task_context 直接参与邻居评分计算。

问题分析：
- 原始 EdgeGNBlock: f_e(cat[nbr_feat, edge_feat, agent_feat])
  - agent_feat 包含任务信息，但 nbr_feat 不包含
  - 模型难以学习"哪个邻居适合处理当前任务"

改进方案：
- TaskAwareEdgeGNBlock: f_e(cat[nbr_feat, edge_feat, agent_feat, task_context])
  - task_context 直接参与边评分，邻居可以根据任务需求调整得分

与隐藏状态交换的兼容性：
- GNN 编码阶段（enc['2hop'], enc['1hop']）保持纯环境信息，可正常交换
- 任务信息只在最终输出阶段（f_out）使用，不影响隐藏状态交换

"""

from typing import Union, Tuple, Optional
import torch as th
import torch.nn as nn

try:
    from torch_geometric.data import HeteroData
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False

from .activations import get_activation


def expand_as_pair(input_value):
    """辅助函数：将输入扩展为 (src, dst) 对"""
    if isinstance(input_value, tuple):
        return input_value
    return (input_value, input_value)


class TaskAwareNodeGNBlock(nn.Module):
    """
    【任务感知版本】节点级图网络块（支持任务边特征）
    
    核心思想：在消息传递阶段就融入任务边特征 (hop_distance)
    
    与 NodeGNBlock 的区别：
    - 消息函数同时考虑环境边特征和任务边特征
    - 支持在聚合阶段就融入 hop_distance 信息
    
    数据流：
    - edge_feats: 环境边特征 [transmission_size] - 链路质量
    - task_edge_feats: 任务边特征 [hop_distance] - 离目标地的跳数
    - 消息函数: f_m(src_feat, edge_feat, task_edge_feat, dst_feat) -> message
    - 聚合: aggregate(messages) -> aggregated
    - 更新: f_v(aggregated, dst_feat) -> output
    
    数值示例 (2hop 模式下的 1hop 聚合):
    =====================================================
    输入:
    - src (邻居编码): [n_nbr=4, 64] 
    - dst (agent+task): [1, 9] = agent[3] + task[6]
    - edge (环境): [4, 1] = [transmission_size]
    - task_edge (hop_distance): [4, 1] = [1, 2, 1, 3] (各邻居到目标的跳数)
    
    输出:
    - [1, 64] agent的新表示，融合了邻居信息和任务需求
    """
    
    def __init__(
        self,
        in_node_feats: Union[int, Tuple[int, int]],  # 节点特征维度
        in_edge_feats: int,                           # 环境边特征维度
        out_node_feats: int,                          # 输出节点特征维度
        hidden_size: int = 64,                        # 隐藏层维度
        activation: str = 'relu',                     # 激活函数
        task_edge_dim: int = 1,                       # 任务边特征维度
    ):
        """
        初始化任务感知的图网络块
        
        示例参数 (V2Lite, n_hops=2, 1hop聚合阶段):
        - in_node_feats = (64, 9)  # src=邻居编码64维, dst=agent+task 9维
        - in_edge_feats = 1         # 环境边特征: transmission_size
        - out_node_feats = 64       # 输出64维表示
        - hidden_size = 64          # 隐藏层64维
        - task_edge_dim = 1         # hop_distance 1维
        """
        super(TaskAwareNodeGNBlock, self).__init__()
        
        # 解析源节点和目标节点的特征维度
        self._in_src_node_feats, self._in_dst_node_feats = expand_as_pair(in_node_feats)
        # 例: self._in_src_node_feats = 64 (邻居编码)
        #     self._in_dst_node_feats = 9 (agent[3] + task[6])
        
        self._in_edge_feats = in_edge_feats        # 1
        self._out_node_feats = out_node_feats      # 64
        self._hidden_size = hidden_size            # 64
        self._task_edge_dim = task_edge_dim        # 1
        
        self.activation = get_activation(activation)  # ReLU
        
        # ========== 消息函数 f_m ==========
        # 作用：为每条边计算消息，综合考虑4个因素
        # 输入: [源节点特征, 环境边特征, 任务边特征, 目标节点特征]
        msg_input_dim = self._in_src_node_feats + in_edge_feats + self._task_edge_dim + self._in_dst_node_feats
        # msg_input_dim = 64 + 1 + 1 + 9 = 75
        
        self.f_m = nn.Sequential(
            nn.Linear(msg_input_dim, hidden_size),  # 75 → 64
            self.activation(),                       # ReLU
            nn.Linear(hidden_size, hidden_size)     # 64 → 64
        )
        # 功能：计算每条边的消息，考虑：
        # - 邻居的状态 (src)
        # - 链路质量 (edge_env)
        # - 任务相关性 (hop_distance)
        # - Agent的需求 (dst with task)
        
        # ========== 更新函数 f_v ==========
        # 作用：根据聚合的消息和自身状态，更新节点表示
        # 输入: [聚合的邻居消息, 自身特征]
        self.f_v = nn.Sequential(
            nn.Linear(hidden_size + self._in_dst_node_feats, hidden_size),  # (64+9) → 64
            self.activation(),                                                # ReLU
            nn.Linear(hidden_size, out_node_feats)                           # 64 → 64
        )
        # 功能：结合邻居建议和自身状态，生成最终表示
    
    def forward(
        self,
        edge_index,
        node_feats,
        edge_feats,
        task_edge_feats: Optional[th.Tensor] = None
    ):
        """
        Args:
            edge_index: [2, num_edges]
            node_feats: (src_feats, dst_feats) 或单个特征
            edge_feats: [num_edges, in_edge_feats] 环境边特征
            task_edge_feats: [num_edges, task_edge_dim] 任务边特征
        """
        if isinstance(node_feats, tuple):
            src_node_feats, dst_node_feats = node_feats
        else:
            src_node_feats = dst_node_feats = node_feats
        
        src_idx = edge_index[0]
        dst_idx = edge_index[1]
        n_edges = edge_index.size(1)
        
        # 默认任务边特征（使用0表示未知，避免引入虚假信号）
        if task_edge_feats is None:
            task_edge_feats = th.zeros(n_edges, self._task_edge_dim, device=edge_feats.device)
        
        # 获取源节点和目标节点特征
        src_feats = src_node_feats[src_idx]  # [n_edges, src_dim]
        dst_feats = dst_node_feats[dst_idx]  # [n_edges, dst_dim]
        
        # 计算消息：融合源节点、边特征（环境+任务）、目标节点
        msg_input = th.cat([src_feats, edge_feats, task_edge_feats, dst_feats], dim=1)
        messages = self.f_m(msg_input)  # [n_edges, hidden_size]
        
        # 聚合消息
        num_dst_nodes = dst_node_feats.size(0)
        aggregated = th.zeros(num_dst_nodes, self._hidden_size, device=messages.device)
        aggregated.index_add_(0, dst_idx, messages)
        
        # 更新节点
        update_input = th.cat([aggregated, dst_node_feats], dim=1)
        output = self.f_v(update_input)
        
        return output


class TaskAwareEdgeGNBlock(nn.Module):
    """
    【任务感知版本】边级图网络块（支持分离模式）
    
    与 EdgeGNBlock 的区别：
    - f_e 的输入增加 task_context 和 task_edge_feats
    - task_edge_feats (hop_distance) 与边顺序一致，保证正确对应
    
    分离模式下的数据流：
    - edge_feats: 环境边特征 [transmission_size]，用于 GNN 编码
    - task_edge_feats: 任务边特征 [hop_distance]，只在决策时使用
    - task_context: 任务上下文 [mission_state(4), routing_info(2)]
    
    输出:
    - 边级得分: 每条边的 Q 值
    - 节点级得分: 本地计算的 Q 值
    """
    
    def __init__(
        self,
        in_node_feats: Union[int, Tuple[int, int]],  # 输入节点特征维度
        in_edge_feats: int,                           # 输入边特征维度 (环境)
        task_dim: int,                                # 任务上下文维度
        out_node_feats: int = 1,                      # 节点输出维度
        out_edge_feats: int = 1,                      # 边输出维度
        hidden_size: int = 64,                        # 隐藏层维度
        activation: str = 'relu',
        max_nbrs: int = 4,                            # 最大邻居数（保留用于兼容）
        task_edge_dim: int = 1,                       # 任务边特征维度 (hop_distance)
    ):
        super(TaskAwareEdgeGNBlock, self).__init__()
        
        self._in_src_node_feats, self._in_dst_node_feats = expand_as_pair(in_node_feats)
        self._in_edge_feats = in_edge_feats
        self._task_dim = task_dim
        self._out_node_feats = out_node_feats
        self._out_edge_feats = out_edge_feats
        self._hidden_size = hidden_size
        self._max_nbrs = max_nbrs
        self._task_edge_dim = task_edge_dim
        
        self.activation = get_activation(activation)
        
        # 边更新函数: 计算每条边的 Q 值（增加网络深度）
        # 输入: [邻居特征, 环境边特征, 任务边特征, agent特征, 任务上下文]
        edge_input_dim = (self._in_src_node_feats + self._in_edge_feats + task_edge_dim +
                          self._in_dst_node_feats + task_dim)
        self.f_e = nn.Sequential(
            nn.Linear(edge_input_dim, self._hidden_size * 2),
            self.activation(),
            nn.Linear(self._hidden_size * 2, self._hidden_size),
            self.activation(),
            nn.Linear(self._hidden_size, self._hidden_size // 2),
            self.activation(),
            nn.Linear(self._hidden_size // 2, self._out_edge_feats)
        )
        
        # 节点更新函数: 计算本地计算的 Q 值（增加网络深度）
        # 输入: [agent特征, 任务上下文]
        self.f_v = nn.Sequential(
            nn.Linear(self._in_dst_node_feats + task_dim, self._hidden_size),
            self.activation(),
            nn.Linear(self._hidden_size, self._hidden_size // 2),
            self.activation(),
            nn.Linear(self._hidden_size // 2, self._out_node_feats)
        )
    
    def forward(
        self, 
        edge_index, 
        node_feats, 
        edge_feats, 
        task_context: th.Tensor,
        task_edge_feats: Optional[th.Tensor] = None
    ):
        """
        Args:
            edge_index: PyG 边索引 [2, num_edges]
            node_feats: 节点特征 (src_feats, dst_feats)
            edge_feats: 环境边特征 [n_edges, in_edge_feats]
            task_context: 任务上下文 [batch, task_dim]
            task_edge_feats: 任务边特征 [n_edges, task_edge_dim]，与边顺序一致的 hop_distance
        
        Returns:
            v_out: 节点级输出 (batch, out_node_feats) - 本地计算Q值
            e_out: 边级输出 (n_edges, out_edge_feats) - 邻居选择Q值
        """
        if isinstance(node_feats, tuple):
            src_node_feats, dst_node_feats = node_feats
        else:
            src_node_feats = dst_node_feats = node_feats
        
        # 处理 task_context 维度
        if task_context.dim() == 1:
            task_context = task_context.unsqueeze(0)
        
        src_idx = edge_index[0]
        dst_idx = edge_index[1]
        n_edges = edge_index.size(1)
        
        # 获取边的源和目标节点特征
        src_feats = src_node_feats[src_idx]   # [n_edges, src_dim]
        dst_feats = dst_node_feats[dst_idx]   # [n_edges, dst_dim]
        
        # 将 task_context 广播到每条边
        task_broadcast = task_context[dst_idx]  # [n_edges, task_dim]
        
        # 任务边特征 (hop_distance)
        if task_edge_feats is None:
            # 如果没有提供，使用0表示未知（避免引入虚假的距离信息）
            task_edge_feats = th.zeros(n_edges, self._task_edge_dim, device=edge_feats.device)
        
        # 计算边级 Q 值（任务感知）
        # 输入: [nbr_feat, env_edge_feat, agent_feat, task_context, hop_distance]
        e_input = th.cat([src_feats, edge_feats, dst_feats, task_broadcast, task_edge_feats], dim=1)
        e_out = self.f_e(e_input)
        
        # 计算节点级 Q 值（本地计算）
        v_input = th.cat([dst_node_feats, task_context], dim=1)
        v_out = self.f_v(v_input)
        
        return v_out, e_out


class TaskConditionedGraphControllerV2(nn.Module):
    """
    【改进版 - 隐藏状态交换兼容版】任务特征分离的消息传递图神经网络控制器
    
    设计原则：
    ==========
    1. enc['2hop']：纯环境编码，输出可以作为隐藏状态交换
    2. enc['1hop']：融入任务信息，本地计算不交换
    3. 通过分层设计，同时解决收敛问题和隐藏状态交换兼容性
    
    隐藏状态交换流程：
    ==================
    训练时（上帝视角）：
        - 直接获取2跳邻居信息
        - enc['2hop'](nbr_feat) → x_nbr_hidden [hidden_size] （纯环境，可交换）
        - enc['1hop']((x_nbr_hidden, agent_with_task)) → env_embed
        - fusion([env_embed, task_embed]) → x
    
    测试时（分布式）：
        第1轮：交换原始邻居特征 raw_feat [nbr_dim]
        第2轮：每个卫星用 enc['2hop'] 聚合，交换 x_nbr_hidden [hidden_size]
        决策：收到邻居的 x_nbr_hidden 后，本地融合任务信息做决策
    
    架构:
    1. TaskEncoder: 编码任务上下文 → task_embed
    2. enc['2hop']: 纯环境编码（可交换）
    3. enc['1hop']: 融入任务的本地编码
    4. Fusion: 进一步增强
    5. TaskAwareEdgeGNBlock: 任务感知的邻居选择
    """
    
    def __init__(
        self,
        obs_shape: dict,
        task_dim: int = 6,
        n_actions: int = 5,
        hidden_size: int = 64,
        max_nbrs: int = 4,
        n_hops: int = 1,
        use_rnn: bool = True,
        use_layer_norm: bool = False,
        dueling: bool = True,
        activation: str = 'relu',
        device: str = 'cpu',
    ):
        super(TaskConditionedGraphControllerV2, self).__init__()
        
        if not PYG_AVAILABLE:
            raise ImportError("PyTorch Geometric is required")
        
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
        
        from .gn_blocks import NodeGNBlock
        
        # 1. 任务编码器
        self.task_encoder = nn.Sequential(
            nn.Linear(task_dim, hidden_size),
            act_fn(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # 2. 分层编码器
        agent_input_dim = obs_shape['agent'] + hidden_size  # agent + task_embed
        
        if n_hops == 1:
            self.enc = nn.ModuleDict({
                '1hop': TaskAwareNodeGNBlock(
                    (obs_shape['nbr'], agent_input_dim),
                    hop_feats,
                    hidden_size,
                    activation=activation
                )
            })
        elif n_hops == 2:
            self.enc = nn.ModuleDict({
                # 【可交换】纯环境的2跳聚合
                '2hop': NodeGNBlock(
                    (obs_shape['nbr'], obs_shape['nbr']),
                    hop_feats,
                    hidden_size,
                    activation=activation
                ),
                # 【不交换】融入任务的1跳聚合
                '1hop': NodeGNBlock(
                    (hidden_size, agent_input_dim),
                    hop_feats,
                    hidden_size,
                    activation=activation
                )
            })
        else:
            raise ValueError(f"n_hops must be 1 or 2, got {n_hops}")
        
        # 3. 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            act_fn()
        )
        
        # 4. RNN 层（可选）
        if use_rnn:
            from .basics import RnnLayer
            self.rnn = RnnLayer(hidden_size, use_layer_norm=use_layer_norm)
        
        # 5. 任务感知的输出层
        inter_nbr_feats = hidden_size if n_hops == 2 else obs_shape['nbr']
        self.f_out = TaskAwareEdgeGNBlock(
            (inter_nbr_feats, hidden_size),
            hop_feats,
            task_dim,
            out_node_feats=1,
            out_edge_feats=1,
            hidden_size=hidden_size,
            activation=activation,
            max_nbrs=max_nbrs
        )
        
        # 6. Dueling 架构
        if dueling:
            self.value_head = nn.Linear(hidden_size, 1)
    
    def init_hidden(self, batch_size: int = 1) -> th.Tensor:
        return th.zeros(batch_size, self._hidden_size, device=self.device)
    
    def _get_edge_feats(self, obs: HeteroData, edge_type: tuple):
        if self._use_dummy_edge_feats:
            n_edges = obs[edge_type].edge_index.size(1)
            return th.ones(n_edges, 1, device=self.device)
        else:
            return obs[edge_type].edge_attr if hasattr(obs[edge_type], 'edge_attr') else obs[edge_type].feat
    
    def encode_2hop(self, obs: HeteroData) -> th.Tensor:
        """
        【可交换】纯环境的2跳编码
        
        Returns:
            x_nbr: 聚合后的邻居表示，纯环境信息，可以交换
        """
        feat = {}
        for ntype in obs.node_types:
            if hasattr(obs[ntype], 'feat'):
                feat[ntype] = obs[ntype].feat
            elif hasattr(obs[ntype], 'x'):
                feat[ntype] = obs[ntype].x
        
        edge_type_2hop = ('nbr', '2hop', 'nbr')
        
        if self._n_hops == 2 and edge_type_2hop in obs.edge_types:
            edge_index_2hop = obs[edge_type_2hop].edge_index
            edge_feats_2hop = self._get_edge_feats(obs, edge_type_2hop)
            x_nbr = self.enc['2hop'](edge_index_2hop, feat['nbr'], edge_feats_2hop)
        else:
            x_nbr = feat['nbr']
        
        return x_nbr
    
    def forward(
        self, 
        obs: Union[HeteroData, 'pyg_compat.PyGHeteroGraph'],
        task_context: th.Tensor,
        h: Optional[th.Tensor] = None,
        task_edge_feats: Optional[dict] = None
    ):
        """
        完整的前向传播
        
        Args:
            obs: PyG 异构图观测
            task_context: 任务上下文 [batch, task_dim]
            h: RNN 隐状态
            task_edge_feats: 任务边特征 {'1hop': tensor}，包含 hop_distance，与边顺序一致
        """
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
        
        # 1. 编码任务
        task_embed = self.task_encoder(task_context)
        
        # 2. 纯环境的2跳编码（可交换）
        x_nbr = self.encode_2hop(obs)
        
        # 3. 融入任务的1跳编码
        agent_with_task = th.cat([feat['agent'], task_embed], dim=-1)
        
        edge_type_1hop = None
        for etype in [('nbr', 'nearby', 'agent'), ('nbr', '1hop', 'agent')]:
            if etype in obs.edge_types:
                edge_type_1hop = etype
                break
        
        if edge_type_1hop is None:
            raise ValueError("Expected edge type ('nbr', 'nearby', 'agent') or ('nbr', '1hop', 'agent')")
        
        edge_index_1hop = obs[edge_type_1hop].edge_index
        edge_feats_1hop = self._get_edge_feats(obs, edge_type_1hop)
        
        env_embed = self.enc['1hop'](edge_index_1hop, (x_nbr, agent_with_task), edge_feats_1hop)
        
        # 4. 融合
        x = th.cat([env_embed, task_embed], dim=-1)
        x = self.fusion(x)
        
        # 5. RNN 更新
        if self._use_rnn:
            if h is None:
                h = self.init_hidden(x.size(0))
            x, h = self.rnn(x, h)
        
        # 6. Q 值输出 - 传入任务边特征 (hop_distance)
        hop_dist_feats = None
        if task_edge_feats is not None and '1hop' in task_edge_feats:
            hop_dist_feats = task_edge_feats['1hop'].to(self.device)
        
        agent_out, nbr_out = self.f_out(
            edge_index_1hop, 
            (x_nbr, x), 
            edge_feats_1hop,
            task_context,
            task_edge_feats=hop_dist_feats
        )
        
        from .gn_blocks import pad_edge_output
        padded_nbr_out = pad_edge_output(edge_index_1hop, nbr_out, self._max_nbrs)
        q_vals = th.cat([padded_nbr_out, agent_out], dim=1)
        
        if self._dueling:
            value = self.value_head(x)
            advantage = q_vals - q_vals.mean(dim=1, keepdim=True)
            q_vals = value + advantage
        
        return q_vals, h
    
    def forward_with_exchanged_hidden(
        self,
        obs: HeteroData,
        x_nbr_exchanged: th.Tensor,
        task_context: th.Tensor,
        h: Optional[th.Tensor] = None,
        task_edge_feats: Optional[dict] = None
    ):
        """
        【测试模式】使用交换来的隐藏状态进行推理
        
        Args:
            obs: PyG 异构图观测
            x_nbr_exchanged: 从邻居交换来的隐藏状态
            task_context: 任务上下文
            h: RNN 隐状态
            task_edge_feats: 任务边特征 {'1hop': tensor}，包含 hop_distance
        """
        if task_context.dim() == 1:
            task_context = task_context.unsqueeze(0)
        task_context = task_context.to(self.device)
        
        feat = {}
        for ntype in obs.node_types:
            if hasattr(obs[ntype], 'feat'):
                feat[ntype] = obs[ntype].feat
            elif hasattr(obs[ntype], 'x'):
                feat[ntype] = obs[ntype].x
        
        # 编码任务
        task_embed = self.task_encoder(task_context)
        
        # 使用交换来的 x_nbr 进行1跳编码
        agent_with_task = th.cat([feat['agent'], task_embed], dim=-1)
        
        edge_type_1hop = None
        for etype in [('nbr', 'nearby', 'agent'), ('nbr', '1hop', 'agent')]:
            if etype in obs.edge_types:
                edge_type_1hop = etype
                break
        
        edge_index_1hop = obs[edge_type_1hop].edge_index
        edge_feats_1hop = self._get_edge_feats(obs, edge_type_1hop)
        
        env_embed = self.enc['1hop'](edge_index_1hop, (x_nbr_exchanged, agent_with_task), edge_feats_1hop)
        
        # 融合
        x = th.cat([env_embed, task_embed], dim=-1)
        x = self.fusion(x)
        
        # RNN
        if self._use_rnn:
            if h is None:
                h = self.init_hidden(x.size(0))
            x, h = self.rnn(x, h)
        
        # Q 值 - 传入任务边特征 (hop_distance)
        hop_dist_feats = None
        if task_edge_feats is not None and '1hop' in task_edge_feats:
            hop_dist_feats = task_edge_feats['1hop'].to(self.device)
        
        agent_out, nbr_out = self.f_out(
            edge_index_1hop, 
            (x_nbr_exchanged, x), 
            edge_feats_1hop,
            task_context,
            task_edge_feats=hop_dist_feats
        )
        
        from .gn_blocks import pad_edge_output
        padded_nbr_out = pad_edge_output(edge_index_1hop, nbr_out, self._max_nbrs)
        q_vals = th.cat([padded_nbr_out, agent_out], dim=1)
        
        if self._dueling:
            value = self.value_head(x)
            advantage = q_vals - q_vals.mean(dim=1, keepdim=True)
            q_vals = value + advantage
        
        return q_vals, h


# 注册到控制器注册表
def register_v2_controllers():
    """注册 V2 版本的控制器到全局注册表"""
    from .controllers import CONTROLLER_REGISTRY
    CONTROLLER_REGISTRY['graph_separated_v2'] = TaskConditionedGraphControllerV2
    CONTROLLER_REGISTRY['gnn_separated_v2'] = TaskConditionedGraphControllerV2
    # Lite 版本
    CONTROLLER_REGISTRY['graph_separated_v2_lite'] = TaskConditionedGraphControllerV2Lite
    CONTROLLER_REGISTRY['gnn_separated_v2_lite'] = TaskConditionedGraphControllerV2Lite


class TaskConditionedGraphControllerV2Lite(nn.Module):
    """
    【精简版 - 隐藏状态交换兼容版】任务特征分离的消息传递图神经网络控制器
    
    设计原则：
    ==========
    1. enc['2hop']：纯环境编码，输出可以作为隐藏状态交换
    2. enc['1hop']：融入任务信息，本地计算不交换
    3. 边特征中的 hop_distance 问题通过"任务感知注意力"解决
    
    隐藏状态交换流程：
    ==================
    训练时（上帝视角）：
        - 直接获取2跳邻居信息
        - enc['2hop'](nbr_feat) → x_nbr_hidden [hidden_size]
        - enc['1hop']((x_nbr_hidden, agent_with_task)) → x
    
    测试时（分布式）：
        第1轮：交换原始邻居特征 raw_feat [nbr_dim]
        第2轮：每个卫星用 enc['2hop'] 聚合，交换 x_nbr_hidden [hidden_size]
        决策：收到邻居的 x_nbr_hidden 后，本地用 enc['1hop'] 融合任务信息
    
    关键：x_nbr_hidden 是纯环境表示，与任务无关，可以交换！
    
    架构:
    1. enc['2hop']: 纯环境编码（nbr→nbr），输出可交换
    2. task_proj: 投影 task_context
    3. enc['1hop']: 融入任务的本地编码（不交换）
    4. f_out: 任务感知的邻居选择
    """
    
    def __init__(
        self,
        obs_shape: dict,
        task_dim: int = 6,
        n_actions: int = 5,
        hidden_size: int = 64,
        max_nbrs: int = 4,
        n_hops: int = 1,
        use_rnn: bool = False,
        use_layer_norm: bool = False,
        dueling: bool = True,
        activation: str = 'relu',
        device: str = 'cpu',
    ):
        super(TaskConditionedGraphControllerV2Lite, self).__init__()
        
        if not PYG_AVAILABLE:
            raise ImportError("PyTorch Geometric is required")
        
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
        
        from .gn_blocks import NodeGNBlock
        
        # ========== 分层编码器 ==========
        if n_hops == 1:
            # 1hop 模式：agent 输入维度 = agent_env + task_dim
            agent_input_dim = obs_shape['agent'] + task_dim
            self.enc = nn.ModuleDict({
                '1hop': TaskAwareNodeGNBlock(
                    (obs_shape['nbr'], agent_input_dim),
                    hop_feats,
                    hidden_size,
                    hidden_size=hidden_size,
                    activation=activation,
                    task_edge_dim=1
                )
            })
        elif n_hops == 2:
            # 2hop 模式：分层设计
            # enc['2hop']：纯环境编码，输出可交换
            # enc['1hop']：融入任务，本地计算（使用任务感知版本）
            agent_input_dim = obs_shape['agent'] + task_dim
            self.enc = nn.ModuleDict({
                # 【可交换】2hop→1hop 的纯环境聚合
                '2hop': NodeGNBlock(
                    (obs_shape['nbr'], obs_shape['nbr']),
                    hop_feats,
                    hidden_size,
                    activation=activation
                ),
                # 【不交换】1hop→agent 的任务感知聚合
                '1hop': TaskAwareNodeGNBlock(
                    (hidden_size, agent_input_dim),  # agent 包含任务
                    hop_feats,
                    hidden_size,
                    hidden_size=hidden_size,
                    activation=activation,
                    task_edge_dim=1
                )
            })
        else:
            raise ValueError(f"n_hops must be 1 or 2, got {n_hops}")
        
        # RNN 层（可选）
        if use_rnn:
            from .basics import RnnLayer
            self.rnn = RnnLayer(hidden_size, use_layer_norm=use_layer_norm)
        
        # 任务感知的输出层
        inter_nbr_feats = hidden_size if n_hops == 2 else obs_shape['nbr']
        self.f_out = TaskAwareEdgeGNBlock(
            (inter_nbr_feats, hidden_size),
            hop_feats,
            task_dim,
            out_node_feats=1,
            out_edge_feats=1,
            hidden_size=hidden_size,
            activation=activation,
            max_nbrs=max_nbrs
        )
        
        # Dueling 架构
        if dueling:
            self.value_head = nn.Sequential(
                nn.Linear(hidden_size + task_dim, hidden_size),
                act_fn(),
                nn.Linear(hidden_size, 1)
            )
    
    def init_hidden(self, batch_size: int = 1) -> th.Tensor:
        return th.zeros(batch_size, self._hidden_size, device=self.device)
    
    def _get_edge_feats(self, obs: HeteroData, edge_type: tuple):
        if self._use_dummy_edge_feats:
            n_edges = obs[edge_type].edge_index.size(1)
            return th.ones(n_edges, 1, device=self.device)
        else:
            return obs[edge_type].edge_attr if hasattr(obs[edge_type], 'edge_attr') else obs[edge_type].feat
    
    def encode_2hop(self, obs: HeteroData) -> th.Tensor:
        """
        【可交换】纯环境的2跳编码
        
        这个方法产生的输出可以作为隐藏状态交换
        
        Returns:
            x_nbr: 聚合后的邻居表示 [n_nbr, hidden_size]，纯环境信息
        """
        feat = {}
        for ntype in obs.node_types:
            if hasattr(obs[ntype], 'feat'):
                feat[ntype] = obs[ntype].feat
            elif hasattr(obs[ntype], 'x'):
                feat[ntype] = obs[ntype].x
        
        edge_type_2hop = ('nbr', '2hop', 'nbr')
        
        if self._n_hops == 2 and edge_type_2hop in obs.edge_types:
            edge_index_2hop = obs[edge_type_2hop].edge_index
            # tensor([[0, 1, 2, 3, 0, 1, 2, 3],
            #         [0, 0, 1, 1, 2, 2, 3, 3]])  # [2, 8]
            edge_feats_2hop = self._get_edge_feats(obs, edge_type_2hop)
            # tensor([[0.4], [0.5], [0.6], [0.3], [0.5], [0.4], [0.3], [0.5]])  # [8, 1]
            x_nbr = self.enc['2hop'](edge_index_2hop,# [2, 8]
                                      feat['nbr'],# [4, 6]
                                      edge_feats_2hop)# [8, 1]
            # 输出: x_nbr = tensor([
            #     [0.12, 0.34, 0.28, ..., 0.56],  # sat_B 聚合2跳邻居后 [64]
            #     [0.18, 0.41, 0.31, ..., 0.62],  # sat_C
            #     [0.22, 0.38, 0.25, ..., 0.58],  # sat_D
            #     [0.28, 0.45, 0.33, ..., 0.67]   # sat_E
            # ])  # [4, 64] ✅ 纯环境表示，可以交换！
        else:
            x_nbr = feat['nbr']
        
        return x_nbr
    
    def encode_1hop_with_task(
        self, 
        obs: HeteroData,
        x_nbr: th.Tensor,
        task_context: th.Tensor,
        hop_dist_feats: Optional[th.Tensor] = None
    ) -> th.Tensor:
        """
        【不交换】融入任务的1跳编码
        
        Args:
            obs: 图观测
            x_nbr: 2跳编码的输出（可以是交换来的）
            task_context: 任务上下文
            hop_dist_feats: 任务边特征 tensor [n_edges, 1]，已提取的 hop_distance
        
        Returns:
            x: agent 的最终表示 [batch, hidden_size]
        """
        if task_context.dim() == 1:
            task_context = task_context.unsqueeze(0)
        task_context = task_context.to(self.device)
        
        feat = {}
        for ntype in obs.node_types:
            if hasattr(obs[ntype], 'feat'):
                feat[ntype] = obs[ntype].feat
            elif hasattr(obs[ntype], 'x'):
                feat[ntype] = obs[ntype].x
        
        # 将任务直接拼接到 agent（无需投影）
        agent_with_task = th.cat([feat['agent'], task_context], dim=-1)
        # feat['agent'] = [[0.0, 0.8, -0.92]]  # [1, 3]
        # task_context = [[1.0, 0.5, 0.8, 0.2, 0.3, 0.0]]  # [1, 6]
        # ↓ concat
        # agent_with_task = [[0.0, 0.8, -0.92, 1.0, 0.5, 0.8, 0.2, 0.3, 0.0]]  # [1, 9]

        
        # 找到 1hop 边类型
        edge_type_1hop = None
        for etype in [('nbr', 'nearby', 'agent'), ('nbr', '1hop', 'agent')]:
            if etype in obs.edge_types:
                edge_type_1hop = etype
                break
        
        if edge_type_1hop is None:
            raise ValueError("Expected edge type ('nbr', 'nearby', 'agent') or ('nbr', '1hop', 'agent')")
        
        edge_index_1hop = obs[edge_type_1hop].edge_index
        edge_feats_1hop = self._get_edge_feats(obs, edge_type_1hop)
        # tensor([[0, 1, 2, 3],
        #         [0, 0, 0, 0]])  # [2, 4]
        
        # 融入任务的编码（同时考虑环境边特征和任务边特征）
        # hop_dist_feats 已经是提取好的 tensor
        x = self.enc['1hop'](edge_index_1hop, # [2, 4]
                              (x_nbr, agent_with_task), # ([4, 64], [1, 9])
                              edge_feats_1hop, # [4, 1] 环境边特征
                              hop_dist_feats)  # [4, 1] 任务边特征 (hop_distance)
        
        return x
    
    def forward(
        self, 
        obs: Union[HeteroData, 'pyg_compat.PyGHeteroGraph'],
        task_context: th.Tensor,
        h: Optional[th.Tensor] = None,
        task_edge_feats: Optional[dict] = None
    ):
        """
        完整的前向传播（训练模式）
        
        Args:
            obs: PyG 异构图观测
            task_context: 任务上下文 [batch, task_dim]
            h: RNN 隐状态
            task_edge_feats: 任务边特征 {'1hop': tensor}，包含 hop_distance
        
        Returns:
            q_vals: Q 值 (batch, n_actions)
            h: 新隐状态
        """
        #步骤 1：处理任务维度
        # 输入: task_context = tensor([1.0, 0.5, 0.8, 0.2, 0.3, 0.0])  # [6]
        if task_context.dim() == 1:
            task_context = task_context.unsqueeze(0)
        task_context = task_context.to(self.device)
        
        # 步骤 2：提取节点特征
        feat = {}
        for ntype in obs.node_types:
            if hasattr(obs[ntype], 'feat'):
                feat[ntype] = obs[ntype].feat
            elif hasattr(obs[ntype], 'x'):
                feat[ntype] = obs[ntype].x
        """
         # 结果:
            # feat = {
            #     'agent': tensor([[0.0, 0.8, -0.92]]),  # [1, 3]
            #     'nbr': tensor([
            #         [0.0, 0.6, 0.7, 0.0, 0.3, 0.5],  # sat_B
            #         [1.0, 0.4, 0.9, 0.0, 0.1, 0.2],  # sat_C
            #         [0.0, 0.5, 0.8, 1.0, 0.2, 0.4],  # sat_D
            #         [0.0, 0.7, 0.6, 0.0, 0.5, 0.1]   # sat_E
            #     ])  # [4, 6]
            # }
        """

        #步骤 3：2跳纯环境编码（可交换）
        
        # 1. 纯环境的2跳编码（可交换）
        x_nbr = self.encode_2hop(obs)

        #步骤 4：融入任务的1跳编码（同时考虑hop_distance）
        
        # 2. 提取任务边特征 (hop_distance)
        hop_dist_feats = None
        if task_edge_feats is not None and '1hop' in task_edge_feats:
            hop_dist_feats = task_edge_feats['1hop'].to(self.device)
        
        # 3. 融入任务的1跳编码（不交换）
        x = self.encode_1hop_with_task(obs, x_nbr, task_context, hop_dist_feats)
        
        # RNN 更新（可选）
        if self._use_rnn:
            if h is None:
                h = self.init_hidden(x.size(0))
            x, h = self.rnn(x, h)
        
        # 4. 获取边信息用于输出
        edge_type_1hop = None
        for etype in [('nbr', 'nearby', 'agent'), ('nbr', '1hop', 'agent')]:
            if etype in obs.edge_types:
                edge_type_1hop = etype
                break
        
        edge_index_1hop = obs[edge_type_1hop].edge_index
        edge_feats_1hop = self._get_edge_feats(obs, edge_type_1hop)
        
        # 5. 任务感知的 Q 值输出 - hop_dist_feats 已在前面提取好
        agent_out, nbr_out = self.f_out(
            edge_index_1hop, 
            (x_nbr, x), 
            edge_feats_1hop,
            task_context,
            task_edge_feats=hop_dist_feats
        )
        
        # Padding
        from .gn_blocks import pad_edge_output
        padded_nbr_out = pad_edge_output(edge_index_1hop, nbr_out, self._max_nbrs)
        q_vals = th.cat([padded_nbr_out, agent_out], dim=1)
        
        # 6. Dueling
        if self._dueling:
            x_with_task = th.cat([x, task_context], dim=-1)
            value = self.value_head(x_with_task)
            advantage = q_vals - q_vals.mean(dim=1, keepdim=True)
            q_vals = value + advantage
        
        return q_vals, h
    
    def forward_with_exchanged_hidden(
        self,
        obs: HeteroData,
        x_nbr_exchanged: th.Tensor,
        task_context: th.Tensor,
        h: Optional[th.Tensor] = None,
        task_edge_feats: Optional[dict] = None
    ):
        """
        【测试模式】使用交换来的隐藏状态进行推理
        
        这是隐藏状态交换后的决策接口
        
        Args:
            obs: 本地图观测（只有1跳信息）
            x_nbr_exchanged: 从邻居交换来的隐藏状态 [n_nbr, hidden_size]
            task_context: 当前任务上下文
            h: RNN 隐状态
            task_edge_feats: 任务边特征 {'1hop': tensor}，包含 hop_distance
        
        Returns:
            q_vals: Q 值
            h: 新隐状态
        """
        if task_context.dim() == 1:
            task_context = task_context.unsqueeze(0)
        task_context = task_context.to(self.device)
        
        # 提取任务边特征
        hop_dist_feats = None
        if task_edge_feats is not None and '1hop' in task_edge_feats:
            hop_dist_feats = task_edge_feats['1hop'].to(self.device)
        
        # 使用交换来的 x_nbr 进行1跳编码
        x = self.encode_1hop_with_task(obs, x_nbr_exchanged, task_context, hop_dist_feats)
        
        # RNN 更新
        if self._use_rnn:
            if h is None:
                h = self.init_hidden(x.size(0))
            x, h = self.rnn(x, h)
        
        # 获取边信息
        edge_type_1hop = None
        for etype in [('nbr', 'nearby', 'agent'), ('nbr', '1hop', 'agent')]:
            if etype in obs.edge_types:
                edge_type_1hop = etype
                break
        
        edge_index_1hop = obs[edge_type_1hop].edge_index
        edge_feats_1hop = self._get_edge_feats(obs, edge_type_1hop)
        
        # Q 值输出 - hop_dist_feats 已在前面提取好
        agent_out, nbr_out = self.f_out(
            edge_index_1hop, 
            (x_nbr_exchanged, x), 
            edge_feats_1hop,
            task_context,
            task_edge_feats=hop_dist_feats
        )
        
        from .gn_blocks import pad_edge_output
        padded_nbr_out = pad_edge_output(edge_index_1hop, nbr_out, self._max_nbrs)
        q_vals = th.cat([padded_nbr_out, agent_out], dim=1)
        
        if self._dueling:
            x_with_task = th.cat([x, task_context], dim=-1)
            value = self.value_head(x_with_task)
            advantage = q_vals - q_vals.mean(dim=1, keepdim=True)
            q_vals = value + advantage
        
        return q_vals, h
