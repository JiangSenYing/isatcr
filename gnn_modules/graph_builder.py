"""
Two-Hop GNN State Builder for Satellite Network

这个模块提供了从卫星网络拓扑直接构建 2跳 图观测的功能。
与原始的 get_current_state 相比，增加了 2跳邻居信息的获取。

使用方式：
1. 在 Satellite_with_Computing 类中集成 TwoHopGraphBuilder
2. 调用 build_graph_observation() 获取图结构化的观测
3. 将观测传入 GNNQNetwork 进行决策
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class GraphObservation:
    """
    图结构化观测
    
    包含构建 GNN 输入所需的所有信息。
    """
    # 当前节点特征
    node_feat: np.ndarray              # [node_feat_dim]
    
    # 1跳邻居特征
    neighbor_feats: np.ndarray         # [n_neighbors, neighbor_feat_dim]
    neighbor_mask: np.ndarray          # [n_neighbors] bool, True=valid
    neighbor_ids: List[str]            # 邻居节点 ID 列表
    
    # 2跳邻居特征（可选）
    hop2_neighbor_feats: Optional[np.ndarray] = None   # [n_neighbors, max_hop2, neighbor_feat_dim]
    hop2_neighbor_mask: Optional[np.ndarray] = None    # [n_neighbors, max_hop2] bool
    hop2_neighbor_ids: Optional[List[List[str]]] = None  # 2跳邻居 ID 列表
    
    # 边特征（节点到邻居的边）
    edge_feats: Optional[np.ndarray] = None  # [n_neighbors, edge_feat_dim]
    
    # 任务特征
    mission_feat: np.ndarray           # [mission_feat_dim]
    
    # 路由特征
    routing_feat: np.ndarray           # [routing_feat_dim]
    
    # 目标节点
    destination: str = ""
    
    def to_flat_state(self, mode: str = 'New') -> np.ndarray:
        """
        转换为与原始 get_current_state 兼容的平面状态向量。
        
        用于与原始 QNetwork 或训练代码兼容。
        """
        neighbors_state = []
        per_neighbor_dim = 14 if 'New' in mode else 6
        
        for i in range(len(self.neighbor_feats)):
            if self.neighbor_mask[i]:
                neighbors_state.extend(self.neighbor_feats[i].tolist())
            else:
                # 填充无效邻居
                if 'New' in mode:
                    neighbors_state.extend([1, 0, 1, 0.5, 1, 0, 1, 0.5, 1, 0, 1, 0.5, 1, 2])
                else:
                    neighbors_state.extend([1, 0, 1, 1, 1, 2])
        
        # 填充到 4 个邻居
        while len(neighbors_state) < per_neighbor_dim * 4:
            if 'New' in mode:
                neighbors_state.extend([1, 0, 1, 0.5, 1, 0, 1, 0.5, 1, 0, 1, 0.5, 1, 2])
            else:
                neighbors_state.extend([1, 0, 1, 1, 1, 2])
        
        # 拼接所有部分
        state = np.concatenate([
            np.array(neighbors_state[:per_neighbor_dim * 4]),
            self.node_feat,
            self.mission_feat,
            self.routing_feat
        ])
        
        return state
    
    def to_tensor(self, device: str = 'cpu') -> Dict[str, torch.Tensor]:
        """
        转换为 PyTorch 张量字典。
        """
        return {
            'node_feat': torch.tensor(self.node_feat, dtype=torch.float32, device=device),
            'neighbor_feats': torch.tensor(self.neighbor_feats, dtype=torch.float32, device=device),
            'neighbor_mask': torch.tensor(self.neighbor_mask, dtype=torch.bool, device=device),
            'edge_feats': torch.tensor(self.edge_feats, dtype=torch.float32, device=device) if self.edge_feats is not None else None,
            'mission_feat': torch.tensor(self.mission_feat, dtype=torch.float32, device=device),
            'routing_feat': torch.tensor(self.routing_feat, dtype=torch.float32, device=device),
            'hop2_neighbor_feats': torch.tensor(self.hop2_neighbor_feats, dtype=torch.float32, device=device) if self.hop2_neighbor_feats is not None else None,
            'hop2_neighbor_mask': torch.tensor(self.hop2_neighbor_mask, dtype=torch.bool, device=device) if self.hop2_neighbor_mask is not None else None,
        }


class TwoHopGraphBuilder:
    """
    两跳图观测构建器
    
    从卫星网络状态构建包含 1跳 和 2跳 邻居信息的图观测。
    
    使用方法:
        builder = TwoHopGraphBuilder(max_neighbors=4, max_hop2_neighbors=4)
        graph_obs = builder.build(satellite, destination, hops, is_computed, mission_state)
    """
    
    def __init__(
        self,
        max_neighbors: int = 4,
        max_hop2_neighbors: int = 4,
        include_hop2: bool = True,
        mode: str = 'New',
    ):
        """
        初始化构建器
        
        Args:
            max_neighbors: 最大 1跳邻居数
            max_hop2_neighbors: 每个 1跳邻居的最大 2跳邻居数
            include_hop2: 是否包含 2跳邻居信息
            mode: 模式 ('New' 或其他)，影响特征维度
        """
        self.max_neighbors = max_neighbors
        self.max_hop2_neighbors = max_hop2_neighbors
        self.include_hop2 = include_hop2
        self.mode = mode
        
        # 特征维度（根据 mode 确定）
        if 'New' in mode:
            self.neighbor_state_dim = 12  # 不含 edge 特征的邻居状态
            self.per_neighbor_dim = 14    # 含 edge 特征 (transmission_size, hop_distance)
        else:
            self.neighbor_state_dim = 4
            self.per_neighbor_dim = 6
        
        self.node_feat_dim = 3  # [is_producing, memory_remain, computing_remain]
        self.edge_feat_dim = 2  # [transmission_size, hop_distance]
        self.mission_feat_dim = 4  # [type, size, computing_demand, size_after_computing]
        self.routing_feat_dim = 2  # [hops, is_computed]
    
    def build(
        self,
        satellite,  # Satellite_with_Computing 实例
        destination: str,
        hops: int,
        is_computed: bool,
        mission_state: List[float],  # [type, size, computing_demand, size_after_computing]
    ) -> GraphObservation:
        """
        构建图观测
        
        Args:
            satellite: 当前卫星节点
            destination: 目标节点 ID
            hops: 当前跳数
            is_computed: 是否已计算
            mission_state: 任务状态 [type, size/max_size, computing_demand/ability, size_after/max_size]
        
        Returns:
            GraphObservation 对象
        """
        # 1. 获取当前节点特征
        node_feat = self._get_node_features(satellite)
        
        # 2. 获取邻居特征和边特征
        neighbors = satellite.neighbors  # sorted list
        n_neighbors = len(neighbors)
        
        neighbor_feats = np.zeros((self.max_neighbors, self.per_neighbor_dim), dtype=np.float32)
        neighbor_mask = np.zeros(self.max_neighbors, dtype=bool)
        edge_feats = np.zeros((self.max_neighbors, self.edge_feat_dim), dtype=np.float32)
        neighbor_ids = []
        
        for i, neighbor in enumerate(neighbors[:self.max_neighbors]):
            # 获取邻居状态
            if 'New' in self.mode:
                nbr_state = satellite.neighbor_states[neighbor]
                # 归一化处理
                nbr_state_normalized = (
                    nbr_state[0:4] + 
                    [x/4 for x in nbr_state[4:8]] + 
                    [x/12 for x in nbr_state[8:12]]
                )
            else:
                nbr_state_normalized = satellite.neighbor_states[neighbor]
            
            # 边特征
            transmission_size = satellite.transmission_size[neighbor] / satellite.memory
            if destination in satellite.neighbor_hops.get(neighbor, {}):
                hop_distance = satellite.neighbor_hops[neighbor][destination] / satellite.max_hop
            else:
                hop_distance = 2.0  # 不可达
            
            # 拼接邻居特征和边特征
            neighbor_feats[i] = np.array(nbr_state_normalized + [transmission_size, hop_distance])
            neighbor_mask[i] = True
            edge_feats[i] = np.array([transmission_size, hop_distance])
            neighbor_ids.append(neighbor)
        
        # 填充无效邻居
        for i in range(n_neighbors, self.max_neighbors):
            if 'New' in self.mode:
                neighbor_feats[i] = np.array([1, 0, 1, 0.5, 1, 0, 1, 0.5, 1, 0, 1, 0.5, 1, 2])
            else:
                neighbor_feats[i] = np.array([1, 0, 1, 1, 1, 2])
            neighbor_ids.append("")
        
        # 3. 获取 2跳邻居特征（可选）
        hop2_neighbor_feats = None
        hop2_neighbor_mask = None
        hop2_neighbor_ids = None
        
        if self.include_hop2:
            hop2_neighbor_feats, hop2_neighbor_mask, hop2_neighbor_ids = \
                self._get_hop2_neighbors(satellite, neighbors[:self.max_neighbors], destination)
        
        # 4. 任务特征
        mission_feat = np.array(mission_state, dtype=np.float32)
        
        # 5. 路由特征
        routing_feat = np.array([hops / satellite.max_hop, float(is_computed)], dtype=np.float32)
        
        return GraphObservation(
            node_feat=node_feat,
            neighbor_feats=neighbor_feats,
            neighbor_mask=neighbor_mask,
            neighbor_ids=neighbor_ids,
            hop2_neighbor_feats=hop2_neighbor_feats,
            hop2_neighbor_mask=hop2_neighbor_mask,
            hop2_neighbor_ids=hop2_neighbor_ids,
            edge_feats=edge_feats,
            mission_feat=mission_feat,
            routing_feat=routing_feat,
            destination=destination,
        )
    
    def _get_node_features(self, satellite) -> np.ndarray:
        """获取当前节点特征"""
        CT_FAC = 5  # 与原始代码一致
        
        is_producing = satellite.is_producing
        memory_remain = 1 - satellite.current_memory_occupy / satellite.memory
        computing_remain = (
            satellite.computing_remain / satellite.computing_ability -
            satellite.is_computing * (satellite.env.now - satellite.last_computing_time)
        ) / CT_FAC
        
        return np.array([is_producing, memory_remain, computing_remain], dtype=np.float32)
    
    def _get_hop2_neighbors(
        self, 
        satellite,
        hop1_neighbors: List[str],
        destination: str,
    ) -> Tuple[np.ndarray, np.ndarray, List[List[str]]]:
        """
        获取 2跳邻居特征
        
        通过邻接表 (adjacency_table) 获取每个 1跳邻居的邻居（即 2跳邻居）。
        """
        hop2_feats = np.zeros(
            (self.max_neighbors, self.max_hop2_neighbors, self.per_neighbor_dim),
            dtype=np.float32
        )
        hop2_mask = np.zeros(
            (self.max_neighbors, self.max_hop2_neighbors),
            dtype=bool
        )
        hop2_ids = [[] for _ in range(self.max_neighbors)]
        
        for i, hop1_nbr in enumerate(hop1_neighbors):
            if not hop1_nbr:
                continue
            
            # 从邻接表获取 hop1_nbr 的邻居
            if hop1_nbr in satellite.adjacency_table:
                hop1_nbr_neighbors, _ = satellite.adjacency_table[hop1_nbr]
                
                # 过滤掉当前节点和已经在 1跳邻居中的节点
                hop2_candidates = [
                    n for n in hop1_nbr_neighbors 
                    if n != satellite.name and n not in hop1_neighbors
                ]
                
                for j, hop2_nbr in enumerate(hop2_candidates[:self.max_hop2_neighbors]):
                    # 尝试获取 2跳邻居的状态
                    if hop2_nbr in satellite.neighbor_states:
                        # 如果有缓存的状态
                        if 'New' in self.mode:
                            state = satellite.neighbor_states[hop2_nbr]
                            state_normalized = (
                                state[0:4] + 
                                [x/4 for x in state[4:8]] + 
                                [x/12 for x in state[8:12]]
                            )
                        else:
                            state_normalized = satellite.neighbor_states[hop2_nbr]
                    else:
                        # 没有状态信息，使用默认值
                        if 'New' in self.mode:
                            state_normalized = [0.5] * 12
                        else:
                            state_normalized = [0.5] * 4
                    
                    # 边特征：使用估计值
                    transmission_size = 0.5  # 未知
                    if destination in satellite.routing_tables:
                        hop_distance = (satellite.routing_tables[destination][1] + 1) / satellite.max_hop
                    else:
                        hop_distance = 2.0
                    
                    hop2_feats[i, j] = np.array(state_normalized + [transmission_size, hop_distance])
                    hop2_mask[i, j] = True
                    hop2_ids[i].append(hop2_nbr)
        
        return hop2_feats, hop2_mask, hop2_ids


class TwoHopStateExtractor:
    """
    从原始 get_current_state 输出中提取 2跳信息的工具类。
    
    这个类可以在不修改原始仿真代码的情况下，
    通过后处理方式增加 2跳信息。
    """
    
    def __init__(
        self,
        per_neighbor_dim: int = 6,   # GNN 模式默认 6 维
        n_neighbors: int = 4,
        current_dim: int = 3,
        mission_dim: int = 4,
        routing_dim: int = 2,
    ):
        self.per_neighbor_dim = per_neighbor_dim
        self.n_neighbors = n_neighbors
        self.current_dim = current_dim
        self.mission_dim = mission_dim
        self.routing_dim = routing_dim
        self.neighbors_dim = per_neighbor_dim * n_neighbors
    
    def extract(
        self,
        state: np.ndarray,
        satellite,
        destination: str,
    ) -> GraphObservation:
        """
        从原始状态向量提取图观测
        
        Args:
            state: 原始 get_current_state 输出
            satellite: 当前卫星实例（用于获取 2跳信息）
            destination: 目标节点
        
        Returns:
            GraphObservation
        """
        # 解析状态向量
        neighbors_state = state[:self.neighbors_dim]
        current_state = state[self.neighbors_dim:self.neighbors_dim + self.current_dim]
        mission_state = state[self.neighbors_dim + self.current_dim:self.neighbors_dim + self.current_dim + self.mission_dim]
        routing_state = state[-self.routing_dim:]
        
        # 重塑邻居状态
        neighbor_feats = neighbors_state.reshape(self.n_neighbors, self.per_neighbor_dim)
        
        # 生成邻居掩码（通过检测默认填充值）
        neighbor_mask = np.zeros(self.n_neighbors, dtype=bool)
        for i in range(self.n_neighbors):
            # 检查是否是填充值（第一个元素为 1 表示填充）
            if neighbor_feats[i, 0] != 1 or i < len(satellite.neighbors):
                neighbor_mask[i] = True
        
        # 构建 GraphObservation
        builder = TwoHopGraphBuilder(
            max_neighbors=self.n_neighbors,
            include_hop2=True,
        )
        
        hops = int(routing_state[0] * satellite.max_hop)
        is_computed = bool(routing_state[1])
        
        # 使用 builder 获取 2跳信息
        full_obs = builder.build(
            satellite, 
            destination, 
            hops, 
            is_computed,
            mission_state.tolist()
        )
        
        # 更新邻居特征为原始值
        full_obs.neighbor_feats = neighbor_feats.astype(np.float32)
        full_obs.neighbor_mask = neighbor_mask
        full_obs.node_feat = current_state.astype(np.float32)
        full_obs.mission_feat = mission_state.astype(np.float32)
        full_obs.routing_feat = routing_state.astype(np.float32)
        
        return full_obs


def integrate_two_hop_to_satellite(satellite_class):
    """
    装饰器：为 Satellite_with_Computing 类添加 2跳图观测功能
    
    使用方式:
        @integrate_two_hop_to_satellite
        class MySatellite(Satellite_with_Computing):
            pass
    
    或者在运行时:
        Satellite_with_Computing = integrate_two_hop_to_satellite(Satellite_with_Computing)
    """
    original_init = satellite_class.__init__
    original_get_current_state = satellite_class.get_current_state
    
    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._graph_builder = TwoHopGraphBuilder(
            max_neighbors=4,
            include_hop2=True,
            mode=self.mode,
        )
    
    def get_graph_observation(
        self, 
        destination: str, 
        hops: int, 
        is_computed: bool, 
        mission_state: List[float],
    ) -> GraphObservation:
        """
        获取图结构化观测（替代 get_current_state）
        """
        return self._graph_builder.build(
            self, destination, hops, is_computed, mission_state
        )
    
    def get_current_state_with_graph(
        self,
        destination: str,
        hops: int,
        is_computed: bool,
        mission_state: List[float],
        return_graph: bool = False,
    ):
        """
        获取当前状态（增强版，可选返回图观测）
        """
        if return_graph:
            graph_obs = self.get_graph_observation(destination, hops, is_computed, mission_state)
            flat_state = graph_obs.to_flat_state(self.mode)
            return flat_state, graph_obs
        else:
            return original_get_current_state(self, destination, hops, is_computed, mission_state)
    
    satellite_class.__init__ = new_init
    satellite_class.get_graph_observation = get_graph_observation
    satellite_class.get_current_state_with_graph = get_current_state_with_graph
    
    return satellite_class
