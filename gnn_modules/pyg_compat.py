"""
PyTorch Geometric 兼容层

提供 DGL API 到 PyG 的映射，使迁移更容易。
支持异构图 (HeteroData) 的创建、batch、unbatch 等操作。
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import Tensor
from torch_geometric.data import Data, HeteroData, Batch


# ==================== 类型别名 ====================

# 用于类型注解的别名，替代 dgl.DGLGraph 和 dgl.DGLHeteroGraph
PyGGraph = Union[Data, HeteroData]
PyGHeteroGraph = HeteroData


# ==================== 图创建函数 ====================

def heterograph(
    data_dict: Dict[Tuple[str, str, str], Tuple[Tensor, Tensor]],
    num_nodes_dict: Optional[Dict[str, int]] = None,
) -> HeteroData:
    """
    创建异构图 - 替代 dgl.heterograph
    
    Args:
        data_dict: 边字典 {(src_type, edge_type, dst_type): (src_ids, dst_ids)}
        num_nodes_dict: 节点数量字典 {node_type: num_nodes}
    
    Returns:
        HeteroData 对象
    
    Example:
        >>> graph = heterograph({
        ...     ('nbr', 'nearby', 'agent'): (torch.tensor([0, 1]), torch.tensor([0, 0]))
        ... }, num_nodes_dict={'agent': 1, 'nbr': 2})
    """
    hetero_data = HeteroData()
    
    # 设置节点数量
    if num_nodes_dict is not None:
        for ntype, num in num_nodes_dict.items():
            hetero_data[ntype].num_nodes = num
    
    # 设置边
    for (src_type, edge_type, dst_type), (src_ids, dst_ids) in data_dict.items():
        if isinstance(src_ids, list):
            src_ids = torch.tensor(src_ids, dtype=torch.long)
        if isinstance(dst_ids, list):
            dst_ids = torch.tensor(dst_ids, dtype=torch.long)
        
        # PyG 使用 edge_index 格式: [2, num_edges]
        edge_index = torch.stack([src_ids, dst_ids], dim=0)
        hetero_data[src_type, edge_type, dst_type].edge_index = edge_index
        
        # 自动推断节点数量（如果未指定）
        if num_nodes_dict is None:
            if src_type not in hetero_data.node_types or hetero_data[src_type].num_nodes is None:
                if len(src_ids) > 0:
                    hetero_data[src_type].num_nodes = int(src_ids.max().item()) + 1
            if dst_type not in hetero_data.node_types or hetero_data[dst_type].num_nodes is None:
                if len(dst_ids) > 0:
                    hetero_data[dst_type].num_nodes = int(dst_ids.max().item()) + 1
    
    return hetero_data


def batch(graphs: List[HeteroData]) -> HeteroData:
    """
    批量处理图 - 替代 dgl.batch
    
    Args:
        graphs: HeteroData 对象列表
    
    Returns:
        批量后的 HeteroData
    """
    if len(graphs) == 0:
        return HeteroData()
    if len(graphs) == 1:
        return graphs[0]
    
    return Batch.from_data_list(graphs)


def unbatch(batched_graph: HeteroData) -> List[HeteroData]:
    """
    拆分批量图 - 替代 dgl.unbatch
    
    Args:
        batched_graph: 批量后的 HeteroData
    
    Returns:
        HeteroData 对象列表
    """
    if isinstance(batched_graph, Batch):
        return batched_graph.to_data_list()
    else:
        return [batched_graph]


def compact_graphs(
    graphs: Union[HeteroData, List[HeteroData]],
    always_preserve: Optional[Dict[str, Tensor]] = None,
    copy_ndata: bool = True,
    copy_edata: bool = True,
) -> Union[HeteroData, List[HeteroData]]:
    """
    压缩图（移除孤立节点）- 替代 dgl.compact_graphs
    
    注意：PyG 没有直接等价的函数，这里简化处理
    
    Args:
        graphs: 输入图或图列表
        always_preserve: 每个节点类型需要保留的节点索引
        copy_ndata: 是否复制节点数据（PyG中默认包含）
        copy_edata: 是否复制边数据（PyG中默认包含）
    
    Returns:
        处理后的图
    """
    # PyG 中一般不需要这个操作，直接返回
    return graphs


# ==================== 节点/边数据访问 ====================

class NodeDataView:
    """模拟 DGL 的 ndata 访问方式"""
    
    def __init__(self, graph: HeteroData, ntype: Optional[str] = None):
        self._graph = graph
        self._ntype = ntype
    
    def __getitem__(self, key: str) -> Tensor:
        if self._ntype is not None:
            return self._graph[self._ntype][key]
        else:
            # 返回所有节点类型的字典
            result = {}
            for ntype in self._graph.node_types:
                if key in self._graph[ntype]:
                    result[ntype] = self._graph[ntype][key]
            return result
    
    def __setitem__(self, key: str, value: Tensor):
        if self._ntype is not None:
            self._graph[self._ntype][key] = value
        else:
            raise ValueError("Must specify node type for setting data")


class EdgeDataView:
    """模拟 DGL 的 edata 访问方式"""
    
    def __init__(self, graph: HeteroData, etype: Optional[Tuple[str, str, str]] = None):
        self._graph = graph
        self._etype = etype
    
    def __getitem__(self, key: str) -> Tensor:
        if self._etype is not None:
            return self._graph[self._etype][key]
        else:
            # 返回所有边类型的字典
            result = {}
            for etype in self._graph.edge_types:
                if key in self._graph[etype]:
                    result[etype] = self._graph[etype][key]
            return result
    
    def __setitem__(self, key: str, value: Tensor):
        if self._etype is not None:
            self._graph[self._etype][key] = value
        else:
            raise ValueError("Must specify edge type for setting data")


# ==================== 辅助函数 ====================

def get_ndata(graph: HeteroData, key: str = 'feat') -> Dict[str, Tensor]:
    """
    获取所有节点类型的特征 - 模拟 graph.ndata[key]
    
    Args:
        graph: HeteroData 对象
        key: 特征键名
    
    Returns:
        {node_type: features} 字典
    """
    result = {}
    for ntype in graph.node_types:
        if key in graph[ntype]:
            result[ntype] = graph[ntype][key]
        elif hasattr(graph[ntype], 'x') and graph[ntype].x is not None:
            result[ntype] = graph[ntype].x
    return result


def set_ndata(graph: HeteroData, key: str, data: Dict[str, Tensor]) -> None:
    """
    设置节点特征 - 模拟 graph.ndata[key] = data
    
    Args:
        graph: HeteroData 对象
        key: 特征键名
        data: {node_type: features} 字典
    """
    for ntype, feat in data.items():
        graph[ntype][key] = feat


def get_edata(graph: HeteroData, key: str = 'feat') -> Dict[Tuple[str, str, str], Tensor]:
    """
    获取所有边类型的特征 - 模拟 graph.edata[key]
    """
    result = {}
    for etype in graph.edge_types:
        if key in graph[etype]:
            result[etype] = graph[etype][key]
        elif hasattr(graph[etype], 'edge_attr') and graph[etype].edge_attr is not None:
            result[etype] = graph[etype].edge_attr
    return result


def set_edata(graph: HeteroData, key: str, data: Dict[Tuple[str, str, str], Tensor]) -> None:
    """
    设置边特征 - 模拟 graph.edata[key] = data
    """
    for etype, feat in data.items():
        graph[etype][key] = feat


def num_nodes(graph: HeteroData, ntype: Optional[str] = None) -> Union[int, Dict[str, int]]:
    """
    获取节点数量 - 模拟 graph.num_nodes(ntype)
    """
    if ntype is not None:
        return graph[ntype].num_nodes
    else:
        return {ntype: graph[ntype].num_nodes for ntype in graph.node_types}


def num_edges(graph: HeteroData, etype: Optional[Tuple[str, str, str]] = None) -> Union[int, Dict[Tuple[str, str, str], int]]:
    """
    获取边数量 - 模拟 graph.num_edges(etype)
    """
    if etype is not None:
        return graph[etype].edge_index.size(1)
    else:
        return {etype: graph[etype].edge_index.size(1) for etype in graph.edge_types}


def to_device(graph: HeteroData, device: Union[str, torch.device]) -> HeteroData:
    """
    将图移动到指定设备 - 模拟 graph.to(device)
    """
    return graph.to(device)


def canonical_etypes(graph: HeteroData) -> List[Tuple[str, str, str]]:
    """
    获取所有边类型 - 模拟 graph.canonical_etypes
    """
    return list(graph.edge_types)


def get_subgraph(graph: HeteroData, src_type: str, edge_type: str, dst_type: str) -> HeteroData:
    """
    获取子图 - 模拟 graph[src_type, edge_type, dst_type]
    
    返回一个只包含该边类型的子图
    """
    subgraph = HeteroData()
    
    # 复制节点
    if src_type in graph.node_types:
        for key, val in graph[src_type].items():
            subgraph[src_type][key] = val
    if dst_type in graph.node_types and dst_type != src_type:
        for key, val in graph[dst_type].items():
            subgraph[dst_type][key] = val
    
    # 复制边
    etype = (src_type, edge_type, dst_type)
    if etype in graph.edge_types:
        for key, val in graph[etype].items():
            subgraph[etype][key] = val
    
    return subgraph


# ==================== 扩展 HeteroData 使其更像 DGL ====================

class DGLCompatHeteroData(HeteroData):
    """
    扩展 HeteroData，添加 DGL 风格的属性和方法
    """
    
    @property
    def ndata(self) -> Dict[str, Tensor]:
        """模拟 DGL 的 ndata 属性"""
        return get_ndata(self, 'feat')
    
    @property
    def canonical_etypes(self) -> List[Tuple[str, str, str]]:
        """模拟 DGL 的 canonical_etypes 属性"""
        return list(self.edge_types)
    
    def num_nodes_by_type(self, ntype: Optional[str] = None) -> Union[int, Dict[str, int]]:
        """获取指定类型的节点数量"""
        return num_nodes(self, ntype)
    
    def num_edges_by_type(self, etype: Optional[Tuple[str, str, str]] = None) -> Union[int, Dict[Tuple[str, str, str], int]]:
        """获取指定类型的边数量"""
        return num_edges(self, etype)


# ==================== 导出 ====================

__all__ = [
    'PyGGraph',
    'PyGHeteroGraph',
    'heterograph',
    'batch',
    'unbatch',
    'compact_graphs',
    'get_ndata',
    'set_ndata',
    'get_edata',
    'set_edata',
    'num_nodes',
    'num_edges',
    'to_device',
    'canonical_etypes',
    'get_subgraph',
    'DGLCompatHeteroData',
]
