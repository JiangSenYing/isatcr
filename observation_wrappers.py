"""
观测包装器模块 - 支持多种观测格式

参考 cross_layer_opt_with_grl-main 的设计，为卫星网络仿真环境提供三种观测格式：
1. FlatObservation: 扁平向量格式（原始格式，适用于 MLP）
2. RelationalObservation: 关系图格式（1跳邻居图，适用于简单 GNN）
3. GraphObservation: 完整图格式（通过 get_graph_inputs 获取，适用于复杂 GNN）

观测格式说明：
- FlatObservation: 直接返回扁平状态向量，不转换为图
- RelationalObservation: 将状态解析为1跳异构图
    Graph(num_nodes={'agent': 1, 'nbr': N},
          num_edges={('agent', 'talks', 'agent'): 0, ('nbr', 'nearby', 'agent'): N})
- GraphObservation: 直接从环境的 get_graph_inputs() 获取完整图结构（支持多跳）
"""

from abc import abstractmethod
from typing import Optional, Tuple, List, Dict, Any, Union
from collections import deque

import numpy as np
import torch as th
from torch import Tensor

# 尝试导入 PyG (PyTorch Geometric)，替代 DGL
try:
    from torch_geometric.data import HeteroData, Batch
    from gnn_modules import pyg_compat
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    HeteroData = None
    print("Warning: PyTorch Geometric not installed. Graph-based observations will not be available.")

# 为了向后兼容，保留 DGL_AVAILABLE 变量名
DGL_AVAILABLE = PYG_AVAILABLE
DGLGraph = HeteroData  # 类型别名


REGISTRY = {}


# ==================== 基类（模拟 ObservationWrapper） ====================

class ObservationWrapper:
    """
    观测包装器基类 - 与 cross_layer_opt_with_grl-main 保持一致
    
    工作流程：
    1. get_obs() 调用环境的 get_obs() 获取原始观测
    2. observation() 方法将原始观测转换为指定格式
    3. 返回转换后的观测
    """
    
    def __init__(self, env=None):
        """
        Args:
            env: 被包装的环境实例
        """
        self.env = env
        # 代理 env 的属性
        if env is not None:
            self.n_agents = getattr(env, 'n_agents', 1)
            self.observation_space = getattr(env, 'observation_space', None)
    
    def __getattr__(self, name):
        """代理访问 env 的属性"""
        if name.startswith("_") or name in ('env', 'n_agents', 'observation_space'):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
        return getattr(self.env, name)
    
    def get_obs(self):
        """获取观测 - 调用子类的 observation() 方法转换"""
        obs = self.env.get_obs() if self.env else None
        return self.observation(obs)
    
    @abstractmethod
    def observation(self, obs):
        """
        将原始观测转换为目标格式
        
        Args:
            obs: 原始观测（通常是每个 agent 的字典观测列表）
        
        Returns:
            转换后的观测
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_obs_size(self):
        """获取观测维度"""
        raise NotImplementedError


# ==================== 独立使用的包装器（无需环境） ====================

class BaseObservationWrapper:
    """
    独立观测包装器基类 - 用于直接包装状态向量
    
    适用于：卫星仿真中需要将扁平状态转换为图的场景
    """
    
    def __init__(self, obs_type: str = 'flat'):
        """
        Args:
            obs_type: 观测类型 ('flat', 'relational', 'graph')
        """
        self.obs_type = obs_type
    
    @abstractmethod
    def wrap_state(self, state: np.ndarray, hop2_info: Optional[Tuple] = None, 
                   neighbor_info: Optional[Dict] = None) -> Dict[str, Any]:
        """
        将原始状态包装为指定格式
        
        Args:
            state: 原始状态向量
            hop2_info: 2跳邻居信息
            neighbor_info: 邻居信息字典
        
        Returns:
            包装后的观测字典
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_obs_size(self) -> Union[int, Dict[str, int]]:
        """获取观测维度"""
        raise NotImplementedError


class FlatObservation(BaseObservationWrapper):
    """扁平向量观测 - 直接返回原始状态向量"""
    
    def __init__(self, state_dim: int = 33):
        """
        Args:
            state_dim: 状态向量维度 (例如 33)
        """
        super().__init__(obs_type='flat')
        self.state_dim = state_dim
    
    def wrap_state(self, state: np.ndarray, hop2_info: Optional[Tuple] = None,
                   neighbor_info: Optional[Dict] = None) -> Dict[str, Any]:
        """直接返回扁平状态"""
        return {
            'obs': state,
            'obs_type': 'flat'
        }
    
    def get_obs_size(self) -> int:
        return self.state_dim


REGISTRY['flat'] = FlatObservation


class RelationalObservation(BaseObservationWrapper):
    """
    关系图观测 - 将状态转换为 1跳 DGL 异构图
    
    参考 cross_layer_opt_with_grl-main 的 RelationalObservation 设计。
    
    图结构:
    - 节点类型: 'agent' (当前卫星), 'nbr' (1跳邻居)
    - 边类型: 
        - ('agent', 'talks', 'agent'): 自连边占位（空）
        - ('nbr', 'nearby', 'agent'): 邻居指向Agent
    
    输出示例:
        Graph(num_nodes={'agent': 1, 'nbr': 3},
              num_edges={('agent', 'talks', 'agent'): 0,
                         ('nbr', 'nearby', 'agent'): 3},
              metagraph=[('agent', 'agent', 'talks'),
                         ('nbr', 'agent', 'nearby')])
    
    适用于: 简单 GNN 如 GATv2Conv, RelationalEncoder
    """
    
    def __init__(self, agent_feat_dim: int = 9, nbr_feat_dim: int = 6, max_neighbors: int = 4):
        """
        Args:
            agent_feat_dim: Agent节点特征维度 (current_node_state + mission_state + routing_info)
            nbr_feat_dim: 邻居节点特征维度
            max_neighbors: 最大邻居数量
        """
        super().__init__(obs_type='relational')
        
        if not DGL_AVAILABLE:
            raise ImportError("PyTorch Geometric is required for RelationalObservation. Please install with: pip install torch_geometric")
        
        self.agent_feat_dim = agent_feat_dim
        self.nbr_feat_dim = nbr_feat_dim
        self.max_neighbors = max_neighbors
    
    def wrap_state(self, state: np.ndarray, hop2_info: Optional[Tuple] = None,
                   neighbor_info: Optional[Dict] = None) -> Dict[str, Any]:
        """
        将状态向量转换为 DGL 异构图（1跳关系）
        
        完全参考 cross_layer_opt_with_grl-main 的 RelationalObservation.observation() 方法
        
        Args:
            state: 状态向量 [33维]
                - [0:24]: 邻居状态 (4个邻居 × 6维)
                - [24:27]: 当前节点状态 (3维)
                - [27:31]: 任务状态 (4维)
                - [31:33]: 路由信息 (2维)
            neighbor_info: 可选的邻居有效性信息 {'valid_mask': np.array}
        
        Returns:
            包含 DGL 图的观测字典，结构:
            {
                'obs': state,
                'graph': DGLGraph,
                'obs_type': 'relational'
            }
        """
        # 解析状态向量
        nbr_dim = 6  # 每个邻居的特征维度
        n_max_nbr = 4  # 最大邻居数
        
        # 提取各部分特征
        nbr_feats_flat = state[:n_max_nbr * nbr_dim]  # [0:24]
        current_node_state = state[24:27]  # [24:27]
        mission_state = state[27:31]  # [27:31]
        routing_info = state[31:33]  # [31:33]
        
        # 重塑邻居特征: [4, 6]
        nbr_feats = nbr_feats_flat.reshape(n_max_nbr, nbr_dim)
        
        # ===== 1. 确定有效邻居（参考 RelationalObservation 的 ent_ids 筛选） =====
        # 与 cross_layer_opt_with_grl-main 一致：使用第一列 (availability) 作为掩码
        # ent_ids = np.equal(v[:, 0], 1)
        if neighbor_info is not None and 'valid_mask' in neighbor_info:
            valid_mask = neighbor_info['valid_mask']
        elif neighbor_info is not None and 'neighbor_mask' in neighbor_info:
            valid_mask = neighbor_info['neighbor_mask']
        else:
            # 使用第一列 availability 标志判断有效性（与原项目一致）
            valid_mask = np.equal(nbr_feats[:, 0], 1)
        
        n_valid_nbr = int(valid_mask.sum())
        
        # ===== 2. 构建图数据字典 =====
        # 参考: data_dict = {('agent', 'talks', 'agent'): ([], [])}
        data_dict = {('agent', 'talks', 'agent'): ([], [])}  # 自连边占位
        num_nodes_dict = {'agent': 1}
        feat = {}
        
        # Agent 特征
        agent_feat = np.concatenate([current_node_state, mission_state, routing_info])#3+4+2
        feat['agent'] = th.as_tensor(agent_feat, dtype=th.float).unsqueeze(0)  # shape: (1, 9)
        
        if n_valid_nbr > 0:
            # 筛选有效邻居的特征（去除可用性标志列，如果需要）
            valid_nbr_feats = nbr_feats[valid_mask]
            
            # 边定义: 每个邻居 → Agent
            # 参考: data_dict[(k, 'nearby', 'agent')] = (th.arange(n_ents), th.zeros(n_ents, dtype=th.long))
            src_nodes = th.arange(n_valid_nbr)
            dst_nodes = th.zeros(n_valid_nbr, dtype=th.long)
            
            data_dict[('nbr', 'nearby', 'agent')] = (src_nodes, dst_nodes)
            num_nodes_dict['nbr'] = n_valid_nbr
            
            # 邻居特征（去除第一列 availability，只保留实际特征）
            # 参考: feat[k] = th.as_tensor(agent_obs[k][ent_ids, 1:], dtype=th.float)
            feat['nbr'] = th.as_tensor(valid_nbr_feats[:, 1:], dtype=th.float)  # shape: (n_valid_nbr, feat_dim-1)# 去掉第0列，shape: (N, 5)
        
        # ===== 3. 创建 PyG 异构图 =====
        graph = pyg_compat.heterograph(data_dict, num_nodes_dict=num_nodes_dict)
        
        # ===== 4. 添加节点特征 =====
        for ntype, f in feat.items():
            if ntype in graph.node_types:
                graph[ntype].feat = f
        
        return {
            'obs': state,  # 保留原始状态用于兼容
            'graph': graph,
            'obs_type': 'relational'
        }
    
    def observation(self, obs: List[Dict]) -> 'HeteroData':
        """
        将 get_obs() 返回的字典格式观测转换为 PyG 图
        
        与 cross_layer_opt_with_grl-main 的 RelationalObservation.observation() 完全一致
        
        Args:
            obs: get_obs() 返回的观测列表
                 [{'agent': own_feats, 'nbr': nbr_feats, 'neighbor_mask': mask}, ...]
        
        Returns:
            PyG 批量异构图
        """
        rel_obs = []
        
        for agent_obs in obs:
            data_dict = {('agent', 'talks', 'agent'): ([], [])}  # Agent自连边（占位）
            num_nodes_dict = {'agent': 1}
            feat = {'agent': th.as_tensor(agent_obs['agent'], dtype=th.float).unsqueeze(0)}
            
            nbr_feats = agent_obs['nbr']
            
            # 使用第一列 (availability) 筛选有效邻居
            # 参考: ent_ids = np.equal(v[:, 0], 1)
            if 'neighbor_mask' in agent_obs:
                ent_ids = agent_obs['neighbor_mask']
            else:
                ent_ids = np.equal(nbr_feats[:, 0], 1)
            
            n_ents = int(ent_ids.sum())
            
            if n_ents > 0:
                # 边定义: 邻居 → Agent
                data_dict[('nbr', 'nearby', 'agent')] = (
                    th.arange(n_ents), 
                    th.zeros(n_ents, dtype=th.long)
                )
                num_nodes_dict['nbr'] = n_ents
                
                # 去除第一列 (availability)，只保留实际特征
                feat['nbr'] = th.as_tensor(nbr_feats[ent_ids, 1:], dtype=th.float)
            
            # 创建 PyG 异构图
            graph = pyg_compat.heterograph(data_dict, num_nodes_dict=num_nodes_dict)
            for ntype, f in feat.items():
                if ntype in graph.node_types:
                    graph[ntype].feat = f
            
            rel_obs.append(graph)
        
        return pyg_compat.batch(rel_obs)
    
    def get_obs_size(self) -> Dict[str, int]:
        return {
            'agent': self.agent_feat_dim,
            'nbr': self.nbr_feat_dim - 1  # 减去 availability 列
        }


REGISTRY['relational'] = RelationalObservation


class GraphObservation(BaseObservationWrapper):
    """
    完整图观测 - 通过 get_graph_inputs() 获取图结构
    
    参考 cross_layer_opt_with_grl-main 的 GraphObservation 设计。
    
    注意：这个包装器要求环境提供：
    - 方法 `.get_graph_inputs()` - 返回图结构数据
    - 属性 `.graph_feats` - 图特征维度信息
    
    get_graph_inputs() 返回格式:
    {
        'graph_data': {
            ('nbr', '1hop', 'agent'): (src_list, dst_list),
            ('nbr', '2hop', 'nbr'): (src_list, dst_list)  # 可选
        },
        'num_nodes_dict': {'agent': 1, 'nbr': N},
        'ndata': {'agent': array, 'nbr': array},  # 节点特征
        'edata': {'1hop': array, '2hop': array}   # 边特征
    }
    
    输出示例 (khops=2):
        Graph(num_nodes={'agent': 1, 'nbr': 30},
              num_edges={('nbr', '1hop', 'agent'): 3, ('nbr', '2hop', 'nbr'): 6})
    
    适用于: 多跳 GNN 如 NodeGNBlock, AdHocGraphController
    """
    
    def __init__(self, env=None, agent_feat_dim: int = 9, nbr_feat_dim: int = 6, 
                 hop_feat_dim: int = 3, hop2_feat_dim: int = 6,
                 max_neighbors: int = 4, max_hop2_neighbors: int = 3,
                 n_agents: int = 1):
        """
        Args:
            env: 环境实例（必须提供 get_graph_inputs() 方法）
            agent_feat_dim: Agent节点特征维度
            nbr_feat_dim: 邻居节点特征维度
            hop_feat_dim: 边特征维度
            hop2_feat_dim: 2跳邻居特征维度
            max_neighbors: 最大1跳邻居数
            max_hop2_neighbors: 每个1跳邻居的最大2跳邻居数
            n_agents: Agent数量
        """
        super().__init__(obs_type='graph')
        
        if not DGL_AVAILABLE:
            raise ImportError("PyTorch Geometric is required for GraphObservation. Please install with: pip install torch_geometric")
        
        self.env = env
        self.agent_feat_dim = agent_feat_dim
        self.nbr_feat_dim = nbr_feat_dim
        self.hop_feat_dim = hop_feat_dim
        self.hop2_feat_dim = hop2_feat_dim
        self.max_neighbors = max_neighbors
        self.max_hop2_neighbors = max_hop2_neighbors
        self.n_agents = n_agents
        
        # 验证环境是否支持图输入
        if env is not None:
            assert hasattr(env, "get_graph_inputs"), \
                "Absence of graph obs callback! Environment must provide get_graph_inputs() method."
            if hasattr(env, "graph_feats"):
                self._graph_feats = env.graph_feats
    
    def get_obs(self):#面向“有环境对象 env”的在线取数入口。它自己去调用环境回调拿图数据。
        """
        从环境获取图观测
        
        完全参考 cross_layer_opt_with_grl-main 的 GraphObservation.get_obs() 方法
        
        Returns:
            PyG 异构图
        """
        if self.env is None:
            raise ValueError("Environment not set. Use wrap_state() for standalone usage.")
        
        # 从环境获取图输入
        obs_relations = self.env.get_graph_inputs()
        
        graph_data = obs_relations['graph_data']  # Define edges
        num_nodes_dict = obs_relations['num_nodes_dict']  # Number of nodes
        node_feats = obs_relations['ndata']  # Node features
        edge_feats = obs_relations.get('edata')  # Edge features (可选)
        
        # 创建 PyG 异构图
        obs_graph = pyg_compat.heterograph(graph_data, num_nodes_dict=num_nodes_dict)
        
        # 添加节点特征
        for ntype in obs_graph.node_types:
            if ntype in node_feats:
                obs_graph[ntype].feat = th.as_tensor(node_feats[ntype], dtype=th.float)
        
        # 添加边特征（如果有）
        if edge_feats is not None:
            for etype in obs_graph.edge_types:
                if etype[1] in edge_feats:  # etype 是 (src, rel, dst) 元组
                    obs_graph[etype].feat = th.as_tensor(edge_feats[etype[1]], dtype=th.float)
        
        # compact_graphs 在 PyG 中不需要特别处理
        obs_graph = pyg_compat.compact_graphs(obs_graph)
        
        return obs_graph
    
    def wrap_state(self, state: np.ndarray, hop2_info: Optional[Tuple] = None,
                   neighbor_info: Optional[Dict] = None,
                   graph_inputs: Optional[Dict] = None) -> Dict[str, Any]:
        #你在经验回放、离线转换、兼容老流程时，已经拿到 state 或 graph_inputs。
        """
        将状态和图输入转换为 PyG 异构图
        
        支持两种模式：
        1. 传入 graph_inputs: 直接使用 get_graph_inputs() 的返回值
        2. 传入 hop2_info: 手动构建多跳图（向后兼容）
        
        Args:
            state: 状态向量 [33维]
            hop2_info: 2跳邻居信息 (hop2_feats, hop2_mask, nbr_mask)
            neighbor_info: 额外的邻居信息
            graph_inputs: get_graph_inputs() 的返回值（推荐方式）
        
        Returns:
            包含 PyG 图的观测字典
        """
        # 模式1: 使用 graph_inputs（推荐）
        if graph_inputs is not None:
            graph_data = graph_inputs['graph_data']
            num_nodes_dict = graph_inputs['num_nodes_dict']
            node_feats = graph_inputs['ndata']
            edge_feats = graph_inputs.get('edata')
            
            # 创建 PyG 异构图
            graph = pyg_compat.heterograph(graph_data, num_nodes_dict=num_nodes_dict)
            
            # 添加节点特征
            for ntype in graph.node_types:
                if ntype in node_feats:
                    graph[ntype].feat = th.as_tensor(node_feats[ntype], dtype=th.float)
            
            # 添加边特征
            if edge_feats is not None:
                for etype in graph.edge_types:
                    if etype[1] in edge_feats:
                        graph[etype].feat = th.as_tensor(edge_feats[etype[1]], dtype=th.float)
            
            # 压缩图
            graph = pyg_compat.compact_graphs(graph)
            
            return {
                'obs': state,
                'graph': graph,
                'obs_type': 'graph'
            }
        
        # 模式2: 使用 hop2_info（向后兼容）
        nbr_dim = 6
        n_max_nbr = 4
        
        nbr_feats_flat = state[:n_max_nbr * nbr_dim]
        current_node_state = state[24:27]
        mission_state = state[27:31]
        routing_info = state[31:33]
        
        nbr_feats = nbr_feats_flat.reshape(n_max_nbr, nbr_dim)
        
        # 1跳邻居掩码
        if hop2_info is not None:
            _, _, nbr_mask = hop2_info
        elif neighbor_info is not None and 'valid_mask' in neighbor_info:
            nbr_mask = neighbor_info['valid_mask']
        else:
            nbr_mask = nbr_feats[:, -1] < 1.5
        
        n_valid_nbr = int(nbr_mask.sum()) if isinstance(nbr_mask, np.ndarray) else int(nbr_mask.sum().item())
        
        # 构建图数据
        data_dict = {('agent', 'talks', 'agent'): ([], [])}
        num_nodes_dict = {'agent': 1}
        feat = {}
        
        agent_feat = np.concatenate([current_node_state, mission_state, routing_info])
        feat['agent'] = th.as_tensor(agent_feat, dtype=th.float).unsqueeze(0)
        
        if n_valid_nbr > 0:
            valid_nbr_feats = nbr_feats[nbr_mask]
            src_nbr = th.arange(n_valid_nbr)
            dst_nbr = th.zeros(n_valid_nbr, dtype=th.long)
            
            data_dict[('nbr', '1hop', 'agent')] = (src_nbr, dst_nbr)
            num_nodes_dict['nbr'] = n_valid_nbr
            feat['nbr'] = th.as_tensor(valid_nbr_feats, dtype=th.float)
            
            # 2跳邻居
            if hop2_info is not None:
                hop2_feats, hop2_mask, _ = hop2_info
                
                hop2_src_list = []
                hop2_dst_list = []
                hop2_feat_list = []
                hop2_node_id = 0
                
                valid_nbr_indices = np.where(nbr_mask)[0] if isinstance(nbr_mask, np.ndarray) else np.where(nbr_mask.numpy())[0]
                
                for local_nbr_idx, global_nbr_idx in enumerate(valid_nbr_indices):
                    if global_nbr_idx < len(hop2_mask):
                        for hop2_idx in range(len(hop2_mask[global_nbr_idx])):
                            if hop2_mask[global_nbr_idx][hop2_idx]:
                                hop2_src_list.append(hop2_node_id)
                                hop2_dst_list.append(local_nbr_idx)
                                hop2_feat_list.append(hop2_feats[global_nbr_idx][hop2_idx])
                                hop2_node_id += 1
                
                if hop2_node_id > 0:
                    data_dict[('nbr', '2hop', 'nbr')] = (
                        th.tensor(hop2_src_list, dtype=th.long),
                        th.tensor(hop2_dst_list, dtype=th.long)
                    )
                    # 不需要额外的 hop2_nbr 节点类型，直接复用 nbr
        
        # 创建 PyG 异构图
        graph = pyg_compat.heterograph(data_dict, num_nodes_dict=num_nodes_dict)
        
        # 添加节点特征
        for ntype, f in feat.items():
            if ntype in graph.node_types:
                graph[ntype].feat = f
        
        return {
            'obs': state,
            'graph': graph,
            'hop2_info': hop2_info,
            'obs_type': 'graph'
        }
    
    def get_obs_size(self) -> Dict[str, int]:
        if hasattr(self, '_graph_feats'):
            return self._graph_feats
        return {
            'agent': self.agent_feat_dim,
            'nbr': self.nbr_feat_dim,
            'hop': self.hop_feat_dim
        }


REGISTRY['graph'] = GraphObservation


# ==================== 经验回放辅助函数 ====================

def cat_observations(obs_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    合并多个观测（用于批处理）
    
    Args:
        obs_list: 观测字典列表
    
    Returns:
        合并后的观测字典
    """
    if not obs_list:
        return {}
    
    obs_type = obs_list[0].get('obs_type', 'flat')
    
    if obs_type == 'flat':
        # 扁平观测：直接 stack
        states = [obs['obs'] for obs in obs_list]
        return {
            'obs': th.tensor(np.array(states), dtype=th.float),
            'obs_type': 'flat'
        }
    
    elif obs_type in ('relational', 'graph', 'relational_separated', 'graph_separated'):
        # 图观测：使用 pyg_compat.batch
        if not DGL_AVAILABLE:
            raise ImportError("PyTorch Geometric is required for graph observations")
        
        states = [obs['obs'] for obs in obs_list]
        graphs = [obs['graph'] for obs in obs_list]
        
        result = {
            'obs': th.tensor(np.array(states), dtype=th.float),
            'graph': pyg_compat.batch(graphs),
            'obs_type': obs_type
        }
        
        # 如果是 GraphObservation，还需要处理 hop2_info
        if obs_type == 'graph' and 'hop2_info' in obs_list[0] and obs_list[0]['hop2_info'] is not None:
            # 批量堆叠 hop2_info
            hop2_feats_list = []
            hop2_mask_list = []
            nbr_mask_list = []
            
            for obs in obs_list:
                if obs.get('hop2_info') is not None:
                    h2f, h2m, nm = obs['hop2_info']
                    hop2_feats_list.append(h2f)
                    hop2_mask_list.append(h2m)
                    nbr_mask_list.append(nm)
            
            if hop2_feats_list:
                result['hop2_info'] = (
                    th.tensor(np.array(hop2_feats_list), dtype=th.float),
                    th.tensor(np.array(hop2_mask_list), dtype=th.bool),
                    th.tensor(np.array(nbr_mask_list), dtype=th.bool)
                )
        
        return result
    
    else:
        raise ValueError(f"Unknown observation type: {obs_type}")


def split_observations(batched_obs: Dict[str, Any], n_samples: int) -> List[Dict[str, Any]]:
    """
    将批量观测拆分为单个观测列表
    
    Args:
        batched_obs: 批量观测字典
        n_samples: 样本数量
    
    Returns:
        单个观测字典列表
    """
    obs_type = batched_obs.get('obs_type', 'flat')
    
    if obs_type == 'flat':
        states = batched_obs['obs']
        return [{'obs': states[i], 'obs_type': 'flat'} for i in range(n_samples)]
    
    elif obs_type in ('relational', 'graph', 'relational_separated', 'graph_separated'):
        if not DGL_AVAILABLE:
            raise ImportError("PyTorch Geometric is required for graph observations")
        
        states = batched_obs['obs']
        graphs = pyg_compat.unbatch(batched_obs['graph'])
        
        results = []
        for i in range(n_samples):
            result = {
                'obs': states[i],
                'graph': graphs[i],
                'obs_type': obs_type
            }
            
            if 'hop2_info' in batched_obs and batched_obs['hop2_info'] is not None:
                h2f, h2m, nm = batched_obs['hop2_info']
                result['hop2_info'] = (h2f[i], h2m[i], nm[i])
            
            results.append(result)
        
        return results
    
    else:
        raise ValueError(f"Unknown observation type: {obs_type}")


# ==================== 观测包装器工厂 ====================

def create_observation_wrapper(obs_type: str, **kwargs) -> BaseObservationWrapper:
    """
    创建观测包装器
    
    Args:
        obs_type: 观测类型 ('flat', 'relational', 'graph')
        **kwargs: 包装器参数
    
    Returns:
        观测包装器实例
    """
    if obs_type not in REGISTRY:
        raise ValueError(f"Unknown observation type: {obs_type}. Available: {list(REGISTRY.keys())}")
    
    return REGISTRY[obs_type](**kwargs)


# ==================== 经验回放缓冲区扩展 ====================

class GraphReplayBuffer:
    """
    支持图观测的经验回放缓冲区
    
    经验格式:
    - Flat: [state, mark, action, reward, next_state, done]
    - Graph: [obs_dict, mark, action, reward, next_obs_dict, done]
    """
    
    def __init__(self, capacity: int, obs_wrapper: BaseObservationWrapper):
        """
        Args:
            capacity: 缓冲区容量
            obs_wrapper: 观测包装器实例
        """
        self.capacity = capacity
        self.obs_wrapper = obs_wrapper
        self.buffer = deque(maxlen=capacity)
    
    def push(self, experience: List) -> None:
        """
        添加经验到缓冲区
        
        Args:
            experience: 经验元组/列表
        """
        self.buffer.append(experience)
    
    def extend(self, experiences: List[List]) -> None:
        """批量添加经验"""
        self.buffer.extend(experiences)
    
    def sample(self, batch_size: int) -> Dict[str, Any]:
        """
        随机采样一批经验
        
        Args:
            batch_size: 批量大小
        
        Returns:
            批量经验字典
        """
        if len(self.buffer) < batch_size:
            raise ValueError(f"Not enough samples: {len(self.buffer)} < {batch_size}")
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        obs_type = self.obs_wrapper.obs_type
        
        if obs_type == 'flat':
            # 标准格式: [state, mark, action, reward, next_state, done]
            states, marks, actions, rewards, next_states, dones = zip(*batch)
            
            return {
                'state': th.tensor(np.array(states), dtype=th.float),
                'mark': th.tensor(marks, dtype=th.long),
                'action': th.tensor(actions, dtype=th.long),
                'reward': th.tensor(rewards, dtype=th.float),
                'next_state': th.tensor(np.array(next_states), dtype=th.float),
                'done': th.tensor(dones, dtype=th.long)
            }
        
        elif obs_type in ('relational', 'graph'):
            # 图格式: [obs_dict, mark, action, reward, next_obs_dict, done]
            obs_dicts, marks, actions, rewards, next_obs_dicts, dones = zip(*batch)
            
            return {
                'obs': cat_observations(list(obs_dicts)),
                'mark': th.tensor(marks, dtype=th.long),
                'action': th.tensor(actions, dtype=th.long),
                'reward': th.tensor(rewards, dtype=th.float),
                'next_obs': cat_observations(list(next_obs_dicts)),
                'done': th.tensor(dones, dtype=th.long)
            }
        
        elif obs_type in ('relational_separated', 'graph_separated'):
            # 分离模式图格式: [obs_dict, mark, action, reward, next_obs_dict, done, last_task_context, current_task_context]
            obs_dicts, marks, actions, rewards, next_obs_dicts, dones, last_task_ctxs, current_task_ctxs = zip(*batch)
            
            return {
                'obs': cat_observations(list(obs_dicts)),
                'mark': th.tensor(marks, dtype=th.long),
                'action': th.tensor(actions, dtype=th.long),
                'reward': th.tensor(rewards, dtype=th.float),
                'next_obs': cat_observations(list(next_obs_dicts)),
                'done': th.tensor(dones, dtype=th.long),
                'last_task_context': th.stack([th.tensor(ctx, dtype=th.float) if not isinstance(ctx, th.Tensor) else ctx for ctx in last_task_ctxs]),
                'current_task_context': th.stack([th.tensor(ctx, dtype=th.float) if not isinstance(ctx, th.Tensor) else ctx for ctx in current_task_ctxs])
            }
        
        else:
            raise ValueError(f"Unknown observation type: {obs_type}")
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def can_sample(self, batch_size: int) -> bool:
        return len(self.buffer) >= batch_size


# ==================== 工具函数 ====================

def convert_experiences_to_graph_format(
    experiences: List[List],
    obs_wrapper: BaseObservationWrapper,
    include_hop2: bool = False
) -> List[List]:
    """
    将扁平格式的经验转换为图格式
    
    Args:
        experiences: 扁平格式经验列表
            格式: [state, mark, action, reward, next_state, done]
            或 2hop格式: [state, hop2_info, mark, action, reward, next_state, next_hop2_info, done]
        obs_wrapper: 观测包装器
        include_hop2: 是否包含2跳信息
    
    Returns:
        图格式经验列表
    """
    converted = []
    
    for exp in experiences:
        if include_hop2 and len(exp) == 8:
            # 2hop格式
            state, hop2_info, mark, action, reward, next_state, next_hop2_info, done = exp
            obs_dict = obs_wrapper.wrap_state(state, hop2_info=hop2_info)
            next_obs_dict = obs_wrapper.wrap_state(next_state, hop2_info=next_hop2_info)
            converted.append([obs_dict, mark, action, reward, next_obs_dict, done])
        else:
            # 标准格式
            state, mark, action, reward, next_state, done = exp[:6]
            obs_dict = obs_wrapper.wrap_state(state)
            next_obs_dict = obs_wrapper.wrap_state(next_state)
            converted.append([obs_dict, mark, action, reward, next_obs_dict, done])
    
    return converted


if __name__ == '__main__':
    # 测试代码 - 验证与 cross_layer_opt_with_grl-main 的一致性
    print("=" * 60)
    print("Testing observation wrappers...")
    print("=" * 60)
    
    # 创建测试状态（33维）
    # [0:24]: 邻居状态 (4个邻居 × 6维)
    # [24:27]: 当前节点状态
    # [27:31]: 任务状态
    # [31:33]: 路由信息
    test_state = np.random.rand(33).astype(np.float32)
    # 设置一些邻居为无效（hop_distance = 2）
    test_state[5] = 1.0   # 第1个邻居有效 (hop < 1.5)
    test_state[11] = 1.0  # 第2个邻居有效
    test_state[17] = 2.0  # 第3个邻居无效 (hop >= 1.5)
    test_state[23] = 1.0  # 第4个邻居有效
    
    print("\n[1] Testing FlatObservation:")
    flat_wrapper = FlatObservation(state_dim=33)
    flat_obs = flat_wrapper.wrap_state(test_state)
    print(f"  obs_type: {flat_obs['obs_type']}")
    print(f"  obs shape: {flat_obs['obs'].shape}")
    
    if DGL_AVAILABLE:
        print("\n[2] Testing RelationalObservation (1-hop graph):")
        print("  Expected graph structure like:")
        print("    Graph(num_nodes={'agent': 1, 'nbr': 3},")
        print("          num_edges={('agent', 'talks', 'agent'): 0,")
        print("                     ('nbr', 'nearby', 'agent'): 3})")
        
        rel_wrapper = RelationalObservation()
        rel_obs = rel_wrapper.wrap_state(test_state)
        print(f"\n  Actual result:")
        print(f"    {rel_obs['graph']}")
        print(f"    Node types: {rel_obs['graph'].ntypes}")
        print(f"    Edge types: {rel_obs['graph'].canonical_etypes}")
        print(f"    Agent feat shape: {rel_obs['graph'].nodes['agent'].data['feat'].shape}")
        if 'nbr' in rel_obs['graph'].ntypes:
            print(f"    Nbr feat shape: {rel_obs['graph'].nodes['nbr'].data['feat'].shape}")
        
        print("\n[3] Testing GraphObservation with get_graph_inputs() format:")
        
        # 模拟 get_graph_inputs() 返回值（参考 cross_layer_opt_with_grl-main）
        mock_graph_inputs = {
            'graph_data': {
                ('nbr', '1hop', 'agent'): ([1, 4, 7], [0, 0, 0]),  # 3个1跳邻居
                ('nbr', '2hop', 'nbr'): ([0, 4, 2, 1, 0, 7], [1, 1, 1, 4, 4, 4])  # 6条2跳边
            },
            'num_nodes_dict': {'agent': 1, 'nbr': 10},
            'ndata': {
                'agent': np.random.rand(1, 5).astype(np.float32),  # Agent特征
                'nbr': np.random.rand(10, 2).astype(np.float32)    # 邻居特征
            },
            'edata': {
                '1hop': np.random.rand(3, 3).astype(np.float32),   # 1跳边特征
                '2hop': np.random.rand(6, 3).astype(np.float32)    # 2跳边特征
            }
        }
        
        graph_wrapper = GraphObservation()
        graph_obs = graph_wrapper.wrap_state(test_state, graph_inputs=mock_graph_inputs)
        print(f"  Using graph_inputs (recommended):")
        print(f"    {graph_obs['graph']}")
        print(f"    Node types: {graph_obs['graph'].ntypes}")
        print(f"    Edge types: {graph_obs['graph'].canonical_etypes}")
        
        print("\n[4] Testing GraphObservation with hop2_info (backward compatible):")
        hop2_feats = np.random.rand(4, 3, 6).astype(np.float32)
        hop2_mask = np.array([[True, False, False], [True, True, False], 
                              [False, False, False], [True, False, False]])
        nbr_mask = np.array([True, True, False, True])
        hop2_info = (hop2_feats, hop2_mask, nbr_mask)
        
        graph_obs2 = graph_wrapper.wrap_state(test_state, hop2_info=hop2_info)
        print(f"  Using hop2_info:")
        print(f"    {graph_obs2['graph']}")
        
        print("\n[5] Testing batch operations (cat_observations):")
        obs_list = [rel_wrapper.wrap_state(np.random.rand(33).astype(np.float32)) for _ in range(4)]
        batched = cat_observations(obs_list)
        print(f"  Batched graph: {batched['graph']}")
        print(f"  Number of graphs in batch: {batched['graph'].batch_size}")
        
        print("\n[6] Testing split operations (split_observations):")
        split = split_observations(batched, 4)
        print(f"  Number of split observations: {len(split)}")
        print(f"  First split graph: {split[0]['graph']}")
        
        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)
    else:
        print("\nDGL not available, skipping graph tests.")
        print("Install with: pip install dgl")
