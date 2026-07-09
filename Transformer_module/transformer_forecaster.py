"""
Global Transformer forecasting and path planning for satellite networks.

The module has two layers:
1. SatelliteLoadTransformer predicts future global queue/link/compute loads.
2. TransformerPathPlanner scores candidate source-destination paths with the
   predicted future loads and returns the lowest-risk path.

Satellite positions and propagation delays are not predicted. Propagation
delays are used as historical input features and are also read from the
simulator/ephemeris-derived graph during planning when available.
"""

import heapq
import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import networkx as nx
import torch as th
import torch.nn as nn
import torch.nn.functional as F


EdgeName = Tuple[str, str]


@dataclass
class GlobalNetworkSnapshot:
    """One fixed-order global network state sample."""

    node_names: List[str]
    edge_names: List[EdgeName]
    queue_load: np.ndarray
    link_load: np.ndarray
    compute_queue: np.ndarray
    adjacency: Dict[str, List[str]]
    business_time: Optional[np.ndarray] = None
    propagation_delays: Optional[Dict[EdgeName, float]] = None
    memory_capacity: Optional[Dict[str, float]] = None
    computing_capacity: Optional[Dict[str, float]] = None
    node_mask: Optional[np.ndarray] = None
    link_mask: Optional[np.ndarray] = None
    sim_time: Optional[float] = None

    def aligned(self, node_names: Sequence[str], edge_names: Sequence[EdgeName]) -> "GlobalNetworkSnapshot":
        """按给定节点和边顺序重排快照，缺失节点/边用 0 特征和 0 mask 填充。"""
        node_idx = {name: i for i, name in enumerate(self.node_names)}
        edge_idx = {edge: i for i, edge in enumerate(self.edge_names)}
        source_node_mask = (
            np.asarray(self.node_mask, dtype=np.float32).reshape(len(self.node_names), 1)
            if self.node_mask is not None
            else np.ones((len(self.node_names), 1), dtype=np.float32)
        )
        source_link_mask = (
            np.asarray(self.link_mask, dtype=np.float32).reshape(len(self.edge_names), 1)
            if self.link_mask is not None
            else np.ones((len(self.edge_names), 1), dtype=np.float32)
        )

        queue_dim = int(self.queue_load.shape[-1]) if self.queue_load.ndim == 2 else 1
        compute_dim = int(self.compute_queue.shape[-1]) if self.compute_queue.ndim == 2 else 1
        business_dim = int(self.business_time.shape[-1]) if self.business_time is not None and self.business_time.ndim == 2 else 1
        link_dim = int(self.link_load.shape[-1]) if self.link_load.ndim == 2 else 1
        queue_load = np.zeros((len(node_names), queue_dim), dtype=np.float32)
        compute_queue = np.zeros((len(node_names), compute_dim), dtype=np.float32)
        business_time = np.zeros((len(node_names), business_dim), dtype=np.float32)
        link_load = np.zeros((len(edge_names), link_dim), dtype=np.float32)
        node_mask = np.zeros((len(node_names), 1), dtype=np.float32)
        link_mask = np.zeros((len(edge_names), 1), dtype=np.float32)

        for out_i, name in enumerate(node_names):
            in_i = node_idx.get(name)
            if in_i is not None:
                queue_load[out_i] = self.queue_load[in_i]
                compute_queue[out_i] = self.compute_queue[in_i]
                if self.business_time is not None:
                    business_time[out_i] = self.business_time[in_i]
                node_mask[out_i] = source_node_mask[in_i]

        for out_i, edge in enumerate(edge_names):
            in_i = edge_idx.get(edge)
            if in_i is None:
                in_i = edge_idx.get((edge[1], edge[0]))
            if in_i is not None:
                link_load[out_i] = self.link_load[in_i]
                link_mask[out_i] = source_link_mask[in_i]

        return GlobalNetworkSnapshot(
            node_names=list(node_names),
            edge_names=list(edge_names),
            queue_load=queue_load,
            link_load=link_load,
            compute_queue=compute_queue,
            business_time=business_time,
            adjacency={name: list(self.adjacency.get(name, [])) for name in node_names},
            propagation_delays=self.propagation_delays,
            memory_capacity=self.memory_capacity,
            computing_capacity=self.computing_capacity,
            node_mask=node_mask,
            link_mask=link_mask,
            sim_time=self.sim_time,
        )

    def to_digraph(self) -> nx.DiGraph:
        """把固定顺序快照转换为有向图输入/输出格式。"""
        graph = nx.DiGraph()
        node_mask = (
            np.asarray(self.node_mask, dtype=np.float32).reshape(-1)
            if self.node_mask is not None
            else np.ones(len(self.node_names), dtype=np.float32)
        )
        link_mask = (
            np.asarray(self.link_mask, dtype=np.float32).reshape(-1)
            if self.link_mask is not None
            else np.ones(len(self.edge_names), dtype=np.float32)
        )
        for node_idx, node in enumerate(self.node_names):
            if node_idx < node_mask.shape[0] and node_mask[node_idx] <= 0:
                continue
            attrs = {
                "queue_load": float(self.queue_load[node_idx, 0]) if node_idx < self.queue_load.shape[0] else 0.0,
                "compute_queue": float(self.compute_queue[node_idx, 0]) if node_idx < self.compute_queue.shape[0] else 0.0,
                "business_time": float(self.business_time[node_idx, 0]) if self.business_time is not None and node_idx < self.business_time.shape[0] else 0.0,
                "queue_features": np.asarray(self.queue_load[node_idx], dtype=np.float32) if node_idx < self.queue_load.shape[0] else np.zeros(1, dtype=np.float32),
                "compute_features": np.asarray(self.compute_queue[node_idx], dtype=np.float32) if node_idx < self.compute_queue.shape[0] else np.zeros(1, dtype=np.float32),
            }
            graph.add_node(node, **attrs)
        for edge_idx, (src, dst) in enumerate(self.edge_names):
            if edge_idx < link_mask.shape[0] and link_mask[edge_idx] <= 0:
                continue
            graph.add_edge(
                src,
                dst,
                link_load=float(self.link_load[edge_idx, 0]) if edge_idx < self.link_load.shape[0] else 0.0,
                link_features=np.asarray(self.link_load[edge_idx], dtype=np.float32) if edge_idx < self.link_load.shape[0] else np.zeros(1, dtype=np.float32),
            )
            if self.propagation_delays:
                delay = self.propagation_delays.get((src, dst), self.propagation_delays.get((dst, src)))
                if delay is not None:
                    graph.edges[src, dst]["propagation_delay"] = float(delay)
        graph.graph["node_names"] = list(self.node_names)
        graph.graph["edge_names"] = list(self.edge_names)
        if self.sim_time is not None:
            graph.graph["sim_time"] = float(self.sim_time)
        return graph

    @classmethod
    def from_digraph(
        cls,
        graph: nx.Graph,
        node_names: Optional[Sequence[str]] = None,
        edge_names: Optional[Sequence[EdgeName]] = None,
        sim_time: Optional[float] = None,
    ) -> "GlobalNetworkSnapshot":
        """从有向 NetworkX 图构造固定顺序快照。"""
        node_names = list(node_names) if node_names is not None else sorted(graph.nodes())
        edge_names = list(edge_names) if edge_names is not None else sorted(graph.edges())
        queue_load = []
        compute_queue = []
        business_time = []
        adjacency = {node: [] for node in node_names}
        propagation_delays = {}
        for node in node_names:
            attrs = graph.nodes[node] if node in graph else {}
            queue_features = attrs.get("queue_features")
            compute_features = attrs.get("compute_features")
            if queue_features is None:
                queue_features = [attrs.get("queue_load", attrs.get("predicted_queue_load", 0.0))]
            if compute_features is None:
                compute_features = [attrs.get("compute_queue", attrs.get("predicted_compute_queue", 0.0))]
            queue_load.append(np.asarray(queue_features, dtype=np.float32).reshape(-1))
            compute_queue.append(np.asarray(compute_features, dtype=np.float32).reshape(-1))
            business_time.append([float(attrs.get("business_time", attrs.get("predicted_business_time", 0.0)))])
        queue_load = _pad_feature_rows(queue_load)
        compute_queue = _pad_feature_rows(compute_queue)
        link_rows = []
        for src, dst in edge_names:
            if graph.has_edge(src, dst):
                attrs = graph.edges[src, dst]
                adjacency.setdefault(src, []).append(dst)
            else:
                attrs = {}
            link_features = attrs.get("link_features")
            if link_features is None:
                link_features = [attrs.get("link_load", attrs.get("predicted_link_load", 0.0))]
            link_rows.append(np.asarray(link_features, dtype=np.float32).reshape(-1))
            delay = attrs.get("propagation_delay", attrs.get("delay"))
            if delay is not None:
                propagation_delays[(src, dst)] = float(delay)
        snapshot_time = sim_time if sim_time is not None else graph.graph.get("sim_time")
        return cls(
            node_names=node_names,
            edge_names=edge_names,
            queue_load=queue_load,
            link_load=_pad_feature_rows(link_rows),
            compute_queue=compute_queue,
            adjacency={node: sorted(set(neighbors)) for node, neighbors in adjacency.items()},
            business_time=np.asarray(business_time, dtype=np.float32),
            propagation_delays=propagation_delays or None,
            node_mask=np.asarray([[1.0 if node in graph else 0.0] for node in node_names], dtype=np.float32),
            link_mask=np.asarray([[1.0 if graph.has_edge(*edge) else 0.0] for edge in edge_names], dtype=np.float32),
            sim_time=float(snapshot_time) if snapshot_time is not None else None,
        )


@dataclass
class PathPlan:
    """一次路径规划结果，包含路径、计算节点标记、评分和风险明细。"""

    path: List[str]
    compute_flags: List[int]
    score: float
    predicted_delay: float
    max_queue_load: float
    max_link_load: float
    max_compute_queue: float
    drop_risk: float
    details: Dict[str, float]
    compute_demands: Optional[List[float]] = None
    disappearing_links: Optional[List[EdgeName]] = None
    disappearing_link_times: Optional[Dict[EdgeName, List[float]]] = None


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for temporal inputs."""

    def __init__(self, d_model: int, max_len: int = 1024):
        """初始化固定正弦位置编码表，用于给时间序列 token 注入时序位置信息。"""
        super().__init__()
        position = th.arange(max_len).float().unsqueeze(1)
        div_term = th.exp(th.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        encoding = th.zeros(max_len, d_model)
        encoding[:, 0::2] = th.sin(position * div_term)
        if d_model > 1:
            encoding[:, 1::2] = th.cos(position * div_term[: encoding[:, 1::2].size(1)])
        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, x: th.Tensor) -> th.Tensor:
        """将位置编码加到输入张量上，保持输入形状不变。"""
        return x + self.encoding[:, : x.size(1)].to(dtype=x.dtype)


class TemporalTransformerHead(nn.Module):
    """
    Forecast one global signal family.

    Input:  [batch, history_len, item_count, input_dim]
    Output: [batch, forecast_horizon, item_count, output_dim]
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        forecast_horizon: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_history_len: int = 1024,
    ):
        """构建单类全局信号预测头，用历史序列自回归预测未来多个时间步。"""
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")

        self.output_dim = output_dim
        self.forecast_horizon = forecast_horizon
        self.input_dim = input_dim

        self.input_proj = nn.Linear(input_dim, d_model)
        self.position = SinusoidalPositionalEncoding(d_model, max_len=max_history_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.horizon_embedding = nn.Parameter(th.randn(forecast_horizon, d_model) * 0.02)
        self.decoder = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, output_dim),
        )

    def forward(self, history: th.Tensor, padding_mask: Optional[th.Tensor] = None) -> th.Tensor:
        """连续预测 forecast_horizon 步，并把每一步预测滚入历史窗口。"""
        if history.dim() != 4:
            raise ValueError("history must have shape [batch, history_len, item_count, input_dim].")

        outputs = []
        rolling_history = history
        rolling_mask = padding_mask
        for step_idx in range(self.forecast_horizon):
            prediction = self.predict_next(rolling_history, padding_mask=rolling_mask, step_idx=step_idx)
            outputs.append(prediction)
            rolling_history = self.append_prediction(rolling_history, prediction)
            rolling_mask = self._append_valid_mask(rolling_mask)
        return th.stack(outputs, dim=1)

    def predict_next(
        self,
        history: th.Tensor,
        padding_mask: Optional[th.Tensor] = None,
        step_idx: int = 0,
    ) -> th.Tensor:
        """基于当前历史窗口预测下一步信号，输出形状为 [batch, item_count, output_dim]。"""
        if history.dim() != 4:
            raise ValueError("history must have shape [batch, history_len, item_count, input_dim].")

        batch_size, history_len, item_count, input_dim = history.shape
        x = history.permute(0, 2, 1, 3).reshape(batch_size * item_count, history_len, input_dim)
        x = self.position(self.input_proj(x))

        expanded_mask = None
        if padding_mask is not None:
            expanded_mask = padding_mask.unsqueeze(1).expand(batch_size, item_count, history_len)
            expanded_mask = expanded_mask.reshape(batch_size * item_count, history_len)

        memory = self.encoder(x, src_key_padding_mask=expanded_mask)
        context = self._last_valid_context(memory, expanded_mask)
        horizon_idx = min(max(int(step_idx), 0), self.forecast_horizon - 1)
        future_token = context + self.horizon_embedding[horizon_idx].unsqueeze(0)
        y = self.decoder(future_token)
        return y.reshape(batch_size, item_count, self.output_dim)

    def append_prediction(self, history: th.Tensor, prediction: th.Tensor) -> th.Tensor:
        """
        将时间窗口向前滚动一步，并把预测值写入下一帧模板。

        保留输入特征中非预测维度的上一帧值，只更新前 output_dim 个预测维度。
        """
        if prediction.dim() != 3:
            raise ValueError("prediction must have shape [batch, item_count, output_dim].")
        if prediction.size(-1) > history.size(-1):
            raise ValueError("prediction output_dim cannot exceed history input_dim.")

        next_frame = history[:, -1].clone()
        next_frame[..., : prediction.size(-1)] = prediction
        return th.cat([history[:, 1:], next_frame.unsqueeze(1)], dim=1)

    @staticmethod
    def _append_valid_mask(padding_mask: Optional[th.Tensor]) -> Optional[th.Tensor]:
        """同步滚动 padding mask，并把新预测步标记为有效。"""
        if padding_mask is None:
            return None
        valid_step = th.zeros(
            padding_mask.size(0),
            1,
            dtype=padding_mask.dtype,
            device=padding_mask.device,
        )
        return th.cat([padding_mask[:, 1:], valid_step], dim=1)

    @staticmethod
    def _last_valid_context(memory: th.Tensor, padding_mask: Optional[th.Tensor]) -> th.Tensor:
        """从 Transformer 编码结果中取每个序列最后一个有效时间步的上下文向量。"""
        if padding_mask is None:
            return memory[:, -1]
        valid_lengths = (~padding_mask).sum(dim=1).clamp(min=1)
        batch_indices = th.arange(memory.size(0), device=memory.device)
        return memory[batch_indices, valid_lengths - 1]


class SatelliteLoadTransformer(nn.Module):
    """
    Four-head global Transformer.

    Outputs:
    - queue_forecast: future node queue/cache load.
    - link_forecast: future link traffic load.
    - compute_queue_forecast: future satellite computing-task queue load,
      predicted from compute history independently of queue history.
    - business_time_forecast: future per-satellite business generation time.
    """

    def __init__(  
        self,
        queue_input_dim: int,
        link_input_dim: int,
        compute_input_dim: int,
        forecast_horizon: int,
        queue_output_dim: int = 1,
        link_output_dim: int = 1,
        compute_output_dim: int = 1,
        business_time_output_dim: int = 1,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_history_len: int = 1024,
    ):
        """初始化队列、链路、计算队列三个预测头，并共享预测窗口等超参数。"""
        super().__init__()
        self.forecast_horizon = forecast_horizon
        head_kwargs = dict(
            forecast_horizon=forecast_horizon,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_history_len=max_history_len,
        )
        self.queue_head = TemporalTransformerHead(queue_input_dim, queue_output_dim, **head_kwargs)
        self.link_head = TemporalTransformerHead(link_input_dim, link_output_dim, **head_kwargs)
        self.compute_head = TemporalTransformerHead(compute_input_dim, compute_output_dim, **head_kwargs)
        self.business_time_head = TemporalTransformerHead(queue_input_dim, business_time_output_dim, **head_kwargs)

    def forward(
        self,
        queue_history: th.Tensor,
        link_history: th.Tensor,
        compute_history: th.Tensor,
        padding_mask: Optional[th.Tensor] = None,
    ) -> Dict[str, th.Tensor]:
        """同时预测未来节点队列、链路负载和计算队列负载。"""
        queue_outputs = []
        link_outputs = []
        compute_outputs = []
        business_time_outputs = []
        rolling_queue = queue_history
        rolling_link = link_history
        rolling_compute = compute_history
        rolling_mask = padding_mask

        for step_idx in range(self.forecast_horizon):
            queue_next = self.queue_head.predict_next(
                rolling_queue,
                padding_mask=rolling_mask,
                step_idx=step_idx,
            )
            link_next = self.link_head.predict_next(
                rolling_link,
                padding_mask=rolling_mask,
                step_idx=step_idx,
            )
            compute_next = self.compute_head.predict_next(
                rolling_compute,
                padding_mask=rolling_mask,
                step_idx=step_idx,
            )
            business_time_next = self.business_time_head.predict_next(
                rolling_queue,
                padding_mask=rolling_mask,
                step_idx=step_idx,
            )

            queue_outputs.append(queue_next)
            link_outputs.append(link_next)
            compute_outputs.append(compute_next)
            business_time_outputs.append(business_time_next)

            rolling_queue = self._append_prediction_with_load_rate(self.queue_head, rolling_queue, queue_next)
            rolling_link = self._append_prediction_with_load_rate(self.link_head, rolling_link, link_next)
            rolling_compute = self._append_prediction_with_load_rate(self.compute_head, rolling_compute, compute_next)
            rolling_mask = TemporalTransformerHead._append_valid_mask(rolling_mask)

        queue_forecast = th.stack(queue_outputs, dim=1)
        link_forecast = th.stack(link_outputs, dim=1)
        compute_queue_forecast = th.stack(compute_outputs, dim=1)
        business_time_forecast = th.stack(business_time_outputs, dim=1)
        return {
            "queue_forecast": queue_forecast,
            "link_forecast": link_forecast,
            "compute_queue_forecast": compute_queue_forecast,
            "business_time_forecast": business_time_forecast,
        }

    def predict_graphs(
        self,
        history_graphs: Sequence[nx.Graph],
        node_names: Optional[Sequence[str]] = None,
        edge_names: Optional[Sequence[EdgeName]] = None,
        device: Optional[th.device] = None,
    ) -> Dict[float, nx.DiGraph]:
        """
        以有向图序列作为输入，并返回写入预测属性的有向图序列。

        图节点可提供 queue_features/compute_features，边可提供 link_features；
        未提供时分别退化读取 queue_load、compute_queue、link_load。
        """
        if not history_graphs:
            raise ValueError("history_graphs cannot be empty.")
        node_names = list(node_names) if node_names is not None else list(history_graphs[-1].graph.get("node_names", sorted(history_graphs[-1].nodes())))
        edge_names = list(edge_names) if edge_names is not None else list(history_graphs[-1].graph.get("edge_names", sorted(history_graphs[-1].edges())))
        snapshots = [
            GlobalNetworkSnapshot.from_digraph(graph, node_names=node_names, edge_names=edge_names)
            for graph in history_graphs
        ]
        device = device or next(self.parameters()).device
        queue = th.as_tensor(np.stack([item.queue_load for item in snapshots])[None], dtype=th.float32, device=device)
        link = th.as_tensor(np.stack([item.link_load for item in snapshots])[None], dtype=th.float32, device=device)
        compute = th.as_tensor(np.stack([item.compute_queue for item in snapshots])[None], dtype=th.float32, device=device)
        self.eval()
        with th.no_grad():
            preds = self(queue, link, compute)

        latest = snapshots[-1]
        latest_step = 1.0
        if len(snapshots) >= 2 and snapshots[-1].sim_time is not None and snapshots[-2].sim_time is not None:
            latest_step = max(1e-9, float(snapshots[-1].sim_time) - float(snapshots[-2].sim_time))
        latest_time = float(latest.sim_time or 0.0)
        graphs = {}
        queue_pred = preds["queue_forecast"][0, :, :, 0].detach().cpu().numpy()
        link_pred = preds["link_forecast"][0, :, :, 0].detach().cpu().numpy()
        compute_pred = preds["compute_queue_forecast"][0, :, :, 0].detach().cpu().numpy()
        business_pred = preds["business_time_forecast"][0, :, :, 0].detach().cpu().numpy()
        for horizon_idx in range(self.forecast_horizon):
            sim_time = float(latest_time + latest_step * (horizon_idx + 1))
            graph = nx.DiGraph()
            graph.add_nodes_from(node_names)
            graph.add_edges_from(edge_names)
            graph.graph["sim_time"] = sim_time
            graph.graph["node_names"] = list(node_names)
            graph.graph["edge_names"] = list(edge_names)
            for node_idx, node in enumerate(node_names):
                graph.nodes[node]["predicted_queue_load"] = float(queue_pred[horizon_idx, node_idx])
                graph.nodes[node]["predicted_compute_queue"] = float(compute_pred[horizon_idx, node_idx])
                graph.nodes[node]["predicted_business_time"] = float(business_pred[horizon_idx, node_idx])
            for edge_idx, (src, dst) in enumerate(edge_names):
                graph.edges[src, dst]["predicted_link_load"] = float(link_pred[horizon_idx, edge_idx])
                if latest.propagation_delays:
                    delay = latest.propagation_delays.get((src, dst), latest.propagation_delays.get((dst, src)))
                    if delay is not None:
                        graph.edges[src, dst]["propagation_delay"] = float(delay)
            graphs[sim_time] = graph
        return graphs

    @staticmethod
    def _append_prediction_with_load_rate(
        head: TemporalTransformerHead,
        history: th.Tensor,
        prediction: th.Tensor,
    ) -> th.Tensor:
        """滚动历史窗口，并在存在速率特征时用预测负载差更新最后一维速率。"""
        next_history = head.append_prediction(history, prediction)
        if history.size(-1) <= prediction.size(-1):
            return next_history

        previous_load = history[:, -1, :, : prediction.size(-1)]
        predicted_rate = th.clamp(prediction - previous_load, min=-1.0, max=1.0)
        next_history[:, -1, :, -prediction.size(-1):] = predicted_rate
        return next_history

    def training_loss(
        self,
        queue_history: th.Tensor,
        link_history: th.Tensor,
        compute_history: th.Tensor,
        queue_target: th.Tensor,
        link_target: th.Tensor,
        compute_queue_target: th.Tensor,
        business_time_target: Optional[th.Tensor] = None,
        padding_mask: Optional[th.Tensor] = None,
        target_masks: Optional[Tuple[Optional[th.Tensor], Optional[th.Tensor], Optional[th.Tensor]]] = None,
    ) -> Tuple[th.Tensor, Dict[str, th.Tensor]]:
        """训练损失的兼容入口，转发到图感知版本的损失函数。"""
        return self.graph_training_loss(
            queue_history=queue_history,
            link_history=link_history,
            compute_history=compute_history,
            queue_target=queue_target,
            link_target=link_target,
            compute_queue_target=compute_queue_target,
            business_time_target=business_time_target,
            padding_mask=padding_mask,
            graph_target_masks=target_masks,
        )

    def graph_training_loss(
        self,
        queue_history: th.Tensor,
        link_history: th.Tensor,
        compute_history: th.Tensor,
        queue_target: th.Tensor,
        link_target: th.Tensor,
        compute_queue_target: th.Tensor,
        business_time_target: Optional[th.Tensor] = None,
        padding_mask: Optional[th.Tensor] = None,
        graph_target_masks: Optional[Tuple[Optional[th.Tensor], Optional[th.Tensor], Optional[th.Tensor]]] = None,
    ) -> Tuple[th.Tensor, Dict[str, th.Tensor]]:
        """
        对比预测图属性和真实未来图属性，返回总 MSE 及分项损失。

        graph_target_masks 可携带未来拓扑 mask：节点队列 mask、链路 mask、
        计算队列 mask；缺失节点或链路不会参与负载回归损失。
        """
        preds = self.forward(queue_history, link_history, compute_history, padding_mask=padding_mask)
        self.last_graph_training_predictions = {
            key: value.detach()
            for key, value in preds.items()
        }

        def write_values_to_graphs(
            graphs,
            queue_values: th.Tensor,
            link_values: th.Tensor,
            compute_values: th.Tensor,
            business_time_values: Optional[th.Tensor],
            prefix: str,
        ) -> None:
            """把预测值或真实值写回 NetworkX 图，便于调试和后续路径规划读取。"""
            queue_arr = queue_values.detach().cpu().numpy()
            link_arr = link_values.detach().cpu().numpy()
            compute_arr = compute_values.detach().cpu().numpy()
            business_arr = business_time_values.detach().cpu().numpy() if business_time_values is not None else None
            batch_graphs = graphs
            if isinstance(batch_graphs, nx.Graph):
                batch_graphs = [[batch_graphs]]
            elif isinstance(batch_graphs, dict):
                batch_graphs = [batch_graphs]
            elif batch_graphs and isinstance(batch_graphs[0], nx.Graph):
                batch_graphs = [batch_graphs]

            for batch_idx, horizon_graphs in enumerate(batch_graphs):
                if batch_idx >= queue_arr.shape[0]:
                    break
                if isinstance(horizon_graphs, dict):
                    horizon_items = list(horizon_graphs.items())
                else:
                    horizon_items = list(enumerate(horizon_graphs))
                for horizon_idx, graph_item in enumerate(horizon_items):
                    if horizon_idx >= queue_arr.shape[1]:
                        break
                    _, graph = graph_item
                    node_names = list(getattr(self, "node_names", graph.nodes()))
                    edge_names = list(getattr(self, "edge_names", graph.edges()))
                    for node_idx, node in enumerate(node_names):
                        if node_idx >= queue_arr.shape[2] or node not in graph:
                            continue
                        graph.nodes[node][f"{prefix}_queue_load"] = float(queue_arr[batch_idx, horizon_idx, node_idx, 0])
                        graph.nodes[node][f"{prefix}_compute_queue"] = float(compute_arr[batch_idx, horizon_idx, node_idx, 0])
                        if business_arr is not None:
                            graph.nodes[node][f"{prefix}_business_time"] = float(business_arr[batch_idx, horizon_idx, node_idx, 0])
                    for edge_idx, edge in enumerate(edge_names):
                        if edge_idx >= link_arr.shape[2]:
                            continue
                        src, dst = edge
                        if graph.has_edge(src, dst):
                            graph.edges[src, dst][f"{prefix}_link_load"] = float(link_arr[batch_idx, horizon_idx, edge_idx, 0])
                        elif graph.has_edge(dst, src):
                            graph.edges[dst, src][f"{prefix}_link_load"] = float(link_arr[batch_idx, horizon_idx, edge_idx, 0])

        def collect_actual_nonzero_link_edges(graphs) -> List[EdgeName]:
            """收集真实链路负载非零的边，并同步写入图的 graph 元数据。"""
            nonzero_edges = []
            batch_graphs = graphs
            if isinstance(batch_graphs, nx.Graph):
                batch_graphs = [[batch_graphs]]
            elif isinstance(batch_graphs, dict):
                batch_graphs = [batch_graphs]
            elif batch_graphs and isinstance(batch_graphs[0], nx.Graph):
                batch_graphs = [batch_graphs]

            for horizon_graphs in batch_graphs:
                if isinstance(horizon_graphs, dict):
                    horizon_items = list(horizon_graphs.items())
                else:
                    horizon_items = list(enumerate(horizon_graphs))
                for _, graph in horizon_items:
                    graph_nonzero_edges = []
                    for src, dst, edge_data in graph.edges(data=True):
                        actual_link_load = edge_data.get("actual_link_load", 0.0)
                        if float(actual_link_load) != 0.0:
                            edge = (src, dst)
                            graph_nonzero_edges.append(edge)
                            nonzero_edges.append(edge)
                    graph.graph["actual_nonzero_link_load_edges"] = graph_nonzero_edges
            return nonzero_edges

        prediction_graphs = None
        if graph_target_masks is not None and len(graph_target_masks) > 3:
            prediction_graphs = graph_target_masks[3]
        if prediction_graphs is None:
            prediction_graphs = getattr(self, "graph_training_prediction_graphs", None)
        if prediction_graphs is not None:
            if business_time_target is None:
                business_time_target = preds["business_time_forecast"].detach()
            write_values_to_graphs(
                prediction_graphs,
                preds["queue_forecast"],
                preds["link_forecast"],
                preds["compute_queue_forecast"],
                preds["business_time_forecast"],
                "predicted",
            )
            write_values_to_graphs(
                prediction_graphs,
                queue_target,
                link_target,
                compute_queue_target,
                business_time_target,
                "actual",
            )
            self.last_graph_training_nonzero_actual_links = collect_actual_nonzero_link_edges(prediction_graphs)
        queue_mask = None if graph_target_masks is None else graph_target_masks[0]
        link_mask = None if graph_target_masks is None else graph_target_masks[1]
        compute_mask = None if graph_target_masks is None else graph_target_masks[2]
        queue_loss = masked_mse_loss(preds["queue_forecast"], queue_target, queue_mask)
        link_loss = masked_mse_loss(preds["link_forecast"], link_target, link_mask)
        compute_loss = masked_mse_loss(
            preds["compute_queue_forecast"],
            compute_queue_target,
            compute_mask,
        )
        if business_time_target is None:
            business_time_loss = th.zeros((), dtype=queue_loss.dtype, device=queue_loss.device)
        else:
            business_time_loss = masked_mse_loss(
                preds["business_time_forecast"],
                business_time_target,
                queue_mask,
            )
        total_loss = queue_loss + link_loss + compute_loss + business_time_loss
        return total_loss, {
            "total_loss": total_loss.detach(),
            "graph_total_loss": total_loss.detach(),
            "graph_queue_loss": queue_loss.detach(),
            "graph_link_loss": link_loss.detach(),
            "graph_compute_queue_loss": compute_loss.detach(),
            "graph_business_time_loss": business_time_loss.detach(),
            "queue_loss": queue_loss.detach(),
            "link_loss": link_loss.detach(),
            "compute_queue_loss": compute_loss.detach(),
            "business_time_loss": business_time_loss.detach(),
        }


class GlobalStateExtractor:
    """Build fixed-order global snapshots from the current simulator state."""

    @staticmethod
    def from_env(env, previous_snapshot: Optional[GlobalNetworkSnapshot] = None) -> GlobalNetworkSnapshot:
        """从 RL 环境对象中取 simulator，并构造成 Transformer 使用的全局快照。"""
        return GlobalStateExtractor.from_simulator(env.simulator, previous_snapshot=previous_snapshot)

    @staticmethod
    def from_simulator(simulator, previous_snapshot: Optional[GlobalNetworkSnapshot] = None) -> GlobalNetworkSnapshot:
        """从仿真器当前状态提取节点、链路、队列、计算和拓扑特征。"""
        satellites = getattr(simulator, "satellites", {})
        graph = getattr(simulator, "graph", None)
        propagator = getattr(simulator, "propagator", None)

        node_names = sorted(list(satellites.keys()))
        if graph is not None:
            node_names = sorted([name for name in graph.nodes() if name in satellites])
        current_time = float(getattr(getattr(simulator, "env", None), "now", 0.0))
        edge_set = set()
        adjacency = {name: [] for name in node_names}
        if graph is not None:
            for src, dst in graph.edges():
                if src in satellites and dst in satellites:
                    edge_set.add((src, dst))
                    edge_set.add((dst, src))
                    adjacency.setdefault(src, []).append(dst)
                    adjacency.setdefault(dst, []).append(src)

        edge_names = sorted(edge_set)
        propagation_delays = None
        if propagator is not None and hasattr(propagator, "propagation_delays"):
            propagation_delays = dict(propagator.propagation_delays)

        memory_capacity = {}
        computing_capacity = {}
        memory_values = {}
        computing_values = {}
        queue_values = []
        compute_values = []
        business_time_values = []
        for node in node_names:
            sat = satellites[node]
            memory = float(getattr(sat, "memory", 1.0) or 1.0)
            computing = float(getattr(sat, "computing_ability", 1.0) or 1.0)
            memory_capacity[node] = memory
            computing_capacity[node] = computing
            memory_values[node] = memory
            computing_values[node] = computing
            memory_used = float(getattr(sat, "current_memory_occupy", 0.0))
            compute_remain = float(getattr(sat, "computing_remain", 0.0))
            is_producing = float(getattr(sat, "is_producing", 0.0) or 0.0)
            session_end = getattr(sat, "traffic_session_end_time", None)
            session_start = getattr(sat, "traffic_session_start_time", None)
            next_session_start = getattr(sat, "next_traffic_session_start_time", None)
            session_duration = float(getattr(sat, "traffic_session_duration", 0.0) or 0.0)
            remaining_time = 0.0
            if is_producing > 0.0 and session_end is not None:
                remaining_time = max(0.0, float(session_end) - current_time)
            remaining_ratio = remaining_time / max(session_duration, 1e-6) if session_duration > 0.0 else 0.0
            duration_scale = max(float(getattr(simulator, "mean_interval_time", 1.0) or 1.0) * 3.0, 1e-6)
            if is_producing > 0.0 and session_start is not None:
                time_until_business = 0.0
            elif next_session_start is not None:
                time_until_business = max(0.0, float(next_session_start) - current_time)
            else:
                time_until_business = duration_scale
            business_time_values.append([_clip01(time_until_business / duration_scale)])
            queue_values.append([
                _clip01(memory_used / memory),
                _clip01(is_producing),
                _clip01(remaining_ratio),
                _clip01(session_duration / duration_scale),
            ])
            compute_scale = max(computing, 1e-6)
            compute_values.append([_clip01(compute_remain / compute_scale)])

        node_packet_flow, link_packet_flow, compute_packet_flow = GlobalStateExtractor._packet_flow_features(
            satellites=satellites,
            node_names=node_names,
            edge_names=edge_names,
            memory_capacity=memory_values,
            computing_capacity=computing_values,
        )
        node_delay_feature, link_delay_feature = GlobalStateExtractor._delay_features(
            node_names=node_names,
            edge_names=edge_names,
            propagation_delays=propagation_delays,
        )
        node_low_delay = node_delay_feature[:, 1:2]
        queue_delay_pressure = node_low_delay * node_packet_flow[:, 2:3]
        compute_delay_pressure = node_low_delay * (
            compute_packet_flow[:, 0:1] + compute_packet_flow[:, 1:2]
        ).clip(0.0, 1.0)

        link_values = []
        for src, dst in edge_names:
            sat = satellites.get(src)
            memory = float(getattr(sat, "memory", 1.0) or 1.0) if sat is not None else 1.0
            transmission = 0.0
            if sat is not None:
                transmission = float(getattr(sat, "transmission_size", {}).get(dst, 0.0))
            link_values.append([_clip01(transmission / memory)])

        queue_array = np.concatenate(
            [
                np.asarray(queue_values, dtype=np.float32),
                node_packet_flow,
                node_delay_feature,
                queue_delay_pressure,
            ],
            axis=-1,
        )
        link_array = np.concatenate(
            [np.asarray(link_values, dtype=np.float32), link_packet_flow, link_delay_feature],
            axis=-1,
        )
        compute_array = np.concatenate(
            [
                np.asarray(compute_values, dtype=np.float32),
                compute_packet_flow,
                node_delay_feature,
                compute_delay_pressure,
            ],
            axis=-1,
        )

        snapshot = GlobalNetworkSnapshot(
            node_names=node_names,
            edge_names=edge_names,
            queue_load=queue_array,
            link_load=link_array,
            compute_queue=compute_array,
            business_time=np.asarray(business_time_values, dtype=np.float32),
            adjacency={key: sorted(set(val)) for key, val in adjacency.items()},
            propagation_delays=propagation_delays,
            memory_capacity=memory_capacity,
            computing_capacity=computing_capacity,
            node_mask=np.ones((len(node_names), 1), dtype=np.float32),
            link_mask=np.ones((len(edge_names), 1), dtype=np.float32),
            sim_time=current_time,
        )
        return GlobalStateExtractor._append_load_rate_features(snapshot, previous_snapshot)

    @staticmethod
    def _append_load_rate_features(
        snapshot: GlobalNetworkSnapshot,
        previous_snapshot: Optional[GlobalNetworkSnapshot],
    ) -> GlobalNetworkSnapshot:
        """
        为队列、链路、计算三类输入追加一个负载变化率特征。

        变化率由当前归一化主负载减去上一快照主负载并除以仿真时间差得到；
        第一帧或拓扑变化导致缺失的项使用 0，保证特征维度稳定。
        """
        queue_rate = np.zeros((len(snapshot.node_names), 1), dtype=np.float32)
        link_rate = np.zeros((len(snapshot.edge_names), 1), dtype=np.float32)
        compute_rate = np.zeros((len(snapshot.node_names), 1), dtype=np.float32)

        if previous_snapshot is not None:
            previous = previous_snapshot.aligned(snapshot.node_names, snapshot.edge_names)
            current_time = snapshot.sim_time
            previous_time = previous.sim_time
            dt = 1.0
            if current_time is not None and previous_time is not None:
                dt = max(float(current_time) - float(previous_time), 1e-6)

            current_queue = np.asarray(snapshot.queue_load[:, :1], dtype=np.float32)
            current_link = np.asarray(snapshot.link_load[:, :1], dtype=np.float32)
            current_compute = np.asarray(snapshot.compute_queue[:, :1], dtype=np.float32)
            previous_queue = np.asarray(previous.queue_load[:, :1], dtype=np.float32)
            previous_link = np.asarray(previous.link_load[:, :1], dtype=np.float32)
            previous_compute = np.asarray(previous.compute_queue[:, :1], dtype=np.float32)

            queue_rate = np.clip((current_queue - previous_queue) / dt, -1.0, 1.0)
            link_rate = np.clip((current_link - previous_link) / dt, -1.0, 1.0)
            compute_rate = np.clip((current_compute - previous_compute) / dt, -1.0, 1.0)

        snapshot.queue_load = np.concatenate(
            [np.asarray(snapshot.queue_load, dtype=np.float32), queue_rate.astype(np.float32)],
            axis=-1,
        )
        snapshot.link_load = np.concatenate(
            [np.asarray(snapshot.link_load, dtype=np.float32), link_rate.astype(np.float32)],
            axis=-1,
        )
        snapshot.compute_queue = np.concatenate(
            [np.asarray(snapshot.compute_queue, dtype=np.float32), compute_rate.astype(np.float32)],
            axis=-1,
        )
        return snapshot

    @staticmethod
    def _packet_flow_features(
        satellites,
        node_names: Sequence[str],
        edge_names: Sequence[EdgeName],
        memory_capacity: Dict[str, float],
        computing_capacity: Dict[str, float],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        从仿真器各类队列中抽取数据包轨迹压力特征。

        节点特征包括本节点转发队列字节、下行/卸载队列字节、剩余路径经过本节点的字节。
        链路特征包括链路发送队列字节、下一跳计划使用该链路的字节。
        计算特征包括本节点已排队/运行的计算需求、计划在本节点计算的转发包需求。
        """
        node_index = {name: idx for idx, name in enumerate(node_names)}
        edge_index = {edge: idx for idx, edge in enumerate(edge_names)}
        node_flow = np.zeros((len(node_names), 3), dtype=np.float32)
        link_flow = np.zeros((len(edge_names), 2), dtype=np.float32)
        compute_flow = np.zeros((len(node_names), 2), dtype=np.float32)

        def normalize_bytes(node: str, value: float) -> float:
            """按节点内存容量把字节数归一化到 [0, 1]。"""
            return _clip01(value / max(float(memory_capacity.get(node, 1.0) or 1.0), 1e-6))

        def normalize_compute(node: str, value: float) -> float:
            """按节点算力容量把计算需求归一化到 [0, 1]。"""
            return _clip01(value / max(float(computing_capacity.get(node, 1.0) or 1.0), 1e-6))

        for node in node_names:
            sat = satellites.get(node)
            if sat is None:
                continue
            n_idx = node_index[node]

            for packet in GlobalStateExtractor._store_items(getattr(sat, "forward_queue", None)):
                size = float(getattr(packet, "size", 0.0) or 0.0)
                node_flow[n_idx, 0] += normalize_bytes(node, size)
                for planned_node in GlobalStateExtractor._remaining_packet_path(packet, current_node=node):
                    p_idx = node_index.get(planned_node)
                    if p_idx is not None:
                        node_flow[p_idx, 2] += normalize_bytes(planned_node, size)

                next_hop = GlobalStateExtractor._planned_next_hop(packet, sat, current_node=node)
                e_idx = edge_index.get((node, next_hop)) if next_hop is not None else None
                if e_idx is not None:
                    link_flow[e_idx, 1] += normalize_bytes(node, size)

                compute_node = GlobalStateExtractor._planned_compute_node(packet, current_node=node)
                c_idx = node_index.get(compute_node) if compute_node is not None else None
                if c_idx is not None:
                    compute_flow[c_idx, 1] += normalize_compute(
                        compute_node,
                        GlobalStateExtractor._packet_compute_demand(packet),
                    )

            for packet in GlobalStateExtractor._store_items(getattr(sat, "offload_queue", None)):
                node_flow[n_idx, 1] += normalize_bytes(node, float(getattr(packet, "size", 0.0) or 0.0))

            for packet in GlobalStateExtractor._store_items(getattr(sat, "computing_queue", None)):
                compute_flow[n_idx, 0] += normalize_compute(
                    node,
                    GlobalStateExtractor._packet_compute_demand(packet),
                )

            transmission_queues = getattr(sat, "transmission_queue", {}) or {}
            for neighbor, store in transmission_queues.items():
                e_idx = edge_index.get((node, neighbor))
                if e_idx is None:
                    continue
                for packet in GlobalStateExtractor._store_items(store):
                    link_flow[e_idx, 0] += normalize_bytes(node, float(getattr(packet, "size", 0.0) or 0.0))

        return np.clip(node_flow, 0.0, 1.0), np.clip(link_flow, 0.0, 1.0), np.clip(compute_flow, 0.0, 1.0)

    @staticmethod
    def _delay_features(
        node_names: Sequence[str],
        edge_names: Sequence[EdgeName],
        propagation_delays: Optional[Dict[EdgeName, float]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """根据传播时延生成节点平均时延/低时延程度和链路时延特征。"""
        node_delay = np.zeros((len(node_names), 2), dtype=np.float32)
        link_delay = np.zeros((len(edge_names), 1), dtype=np.float32)
        if not propagation_delays:
            return node_delay, link_delay

        raw_delays = []
        edge_delays = []
        for edge in edge_names:
            delay = GlobalStateExtractor._edge_delay_value(edge, propagation_delays)
            edge_delays.append(delay)
            if delay is not None:
                raw_delays.append(delay)
        if not raw_delays:
            return node_delay, link_delay

        delay_scale = max(max(raw_delays), 1e-6)
        node_index = {name: idx for idx, name in enumerate(node_names)}
        node_sums = np.zeros((len(node_names), 1), dtype=np.float32)
        node_counts = np.zeros((len(node_names), 1), dtype=np.float32)

        for edge_idx, ((src, dst), delay) in enumerate(zip(edge_names, edge_delays)):
            if delay is None:
                continue
            normalized_delay = _clip01(delay / delay_scale)
            link_delay[edge_idx, 0] = normalized_delay
            for node in (src, dst):
                node_idx = node_index.get(node)
                if node_idx is not None:
                    node_sums[node_idx, 0] += normalized_delay
                    node_counts[node_idx, 0] += 1.0

        has_delay = node_counts[:, 0] > 0
        node_delay[has_delay, 0] = node_sums[has_delay, 0] / node_counts[has_delay, 0]
        node_delay[has_delay, 1] = 1.0 - node_delay[has_delay, 0]
        return node_delay, link_delay

    @staticmethod
    def _edge_delay_value(edge: EdgeName, propagation_delays: Dict[EdgeName, float]) -> Optional[float]:
        """读取一条边的非负传播时延，兼容正反两个方向的 key。"""
        delay = propagation_delays.get(edge)
        if delay is None:
            delay = propagation_delays.get((edge[1], edge[0]))
        if delay is None:
            return None
        try:
            return max(0.0, float(delay))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _store_items(store) -> List:
        """安全读取 simpy Store 等队列对象中的 items 列表。"""
        return list(getattr(store, "items", []) or [])

    @staticmethod
    def _packet_compute_demand(packet) -> float:
        """从数据包 information 字段中读取计算需求，异常或缺失时返回 0。"""
        info = getattr(packet, "information", []) or []
        if len(info) > 2:
            try:
                return max(0.0, float(info[2]))
            except (TypeError, ValueError):
                return 0.0
        return 0.0

    @staticmethod
    def _remaining_packet_path(packet, current_node: Optional[str] = None) -> List[str]:
        """返回数据包从当前节点开始的剩余规划路径；没有规划路径时退化为已走轨迹。"""
        path = list(getattr(packet, "path", None) or [])
        if current_node in path:
            return path[path.index(current_node):]
        trace = list(getattr(packet, "path_trace", None) or [])
        if current_node is not None and (not trace or trace[-1] != current_node):
            trace.append(current_node)
        return trace

    @staticmethod
    def _planned_next_hop(packet, sat, current_node: str) -> Optional[str]:
        """推断数据包在当前卫星的下一跳，优先使用显式规划路径，其次使用路由表。"""
        path = list(getattr(packet, "path", None) or [])
        if path:
            if path[0] == current_node:
                path = path[1:]
            return path[0] if path else None

        routing_tables = getattr(sat, "routing_tables", {}) or {}
        destination = getattr(packet, "destination", None)
        route = routing_tables.get(destination)
        if route and route[0]:
            return route[0][0]
        return None

    @staticmethod
    def _planned_compute_node(packet, current_node: str) -> Optional[str]:
        """推断数据包后续计划在哪个节点计算，优先读取 compute_flags。"""
        path = list(getattr(packet, "path", None) or [])
        flags = list(getattr(packet, "compute_flags", None) or [])
        if path and flags:
            if path[0] == current_node:
                path = path[1:]
                flags = flags[1:]
            for node, flag in zip(path, flags):
                if int(flag) == 1:
                    return node
        return getattr(packet, "computing_node", None)


class TransformerPathPlanner:
    """
    Global risk-aware path planner driven by Transformer predictions.

    The planner does not promise zero loss. It chooses the path with the lowest
    predicted congestion/drop risk under the current model and graph state.
    """

    def __init__(
        self,
        transformer: SatelliteLoadTransformer,
        node_names: Sequence[str],
        edge_names: Sequence[EdgeName],
        history_len: int,
        device: str = "cpu",
        max_history: int = 10000,
        score_weights: Optional[Dict[str, float]] = None,
        queue_threshold: float = 0.9,
        link_threshold: float = 0.9,
        compute_threshold: float = 0.9,
        max_candidate_expansions: int = 2000,
        max_candidate_queue_size: int = 5000,
    ):
        """初始化路径规划器，保存预测模型、固定图顺序、历史窗口和评分权重。"""
        self.transformer = transformer.to(device)
        self.node_names = list(node_names)
        self.edge_names = list(edge_names)
        self.history_len = history_len
        self.device = th.device(device)
        self.max_history = max_history
        self.queue_threshold = queue_threshold
        self.link_threshold = link_threshold
        self.compute_threshold = compute_threshold
        self.max_candidate_expansions = max(1, int(max_candidate_expansions))
        self.max_candidate_queue_size = max(1, int(max_candidate_queue_size))
        self.score_weights = {
            "link": 2.0,
            "queue": 6,
            "compute": 2.0,
            "delay": 4.0,
            "hop": 0.0,
            "drop": 8.0,
        }
        if score_weights:
            self.score_weights.update(score_weights)

        self.history: List[GlobalNetworkSnapshot] = []
        self.latest_snapshot: Optional[GlobalNetworkSnapshot] = None
        self.future_graph_builder: Optional[Callable[[List[float], float], Dict[float, nx.Graph]]] = None
        self.last_prediction_graphs: Optional[Dict[float, nx.Graph]] = None
        self.reserved_queue_load = np.zeros(len(self.node_names), dtype=np.float32)
        self.reserved_link_load = np.zeros(len(self.edge_names), dtype=np.float32)
        self.reserved_compute_load = np.zeros(len(self.node_names), dtype=np.float32)

    @classmethod
    def from_snapshot(
        cls,
        transformer: SatelliteLoadTransformer,
        snapshot: GlobalNetworkSnapshot,
        history_len: int,
        device: str = "cpu",
        **kwargs,
    ) -> "TransformerPathPlanner":
        """基于第一帧快照创建规划器，并立即把该快照加入历史缓存。"""
        planner = cls(
            transformer=transformer,
            node_names=snapshot.node_names,
            edge_names=snapshot.edge_names,
            history_len=history_len,
            device=device,
            **kwargs,
        )
        planner.add_snapshot(snapshot)
        return planner

    def add_snapshot(self, snapshot: GlobalNetworkSnapshot) -> None:
        """把新快照对齐到固定节点/边顺序后加入历史窗口。"""
        aligned = snapshot.aligned(self.node_names, self.edge_names)
        self.latest_snapshot = aligned
        self.history.append(aligned)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        self.clear_reservations()

    def set_future_graph_builder(self, builder: Optional[Callable[[List[float], float], Dict[float, nx.Graph]]]) -> None:
        """设置未来拓扑构造器，用于规划时检查预测时间片的链路是否存在。"""
        self.future_graph_builder = builder

    def clear_reservations(self) -> None:
        """清空尚未被真实快照接管的规划预约负载。"""
        self.reserved_queue_load.fill(0.0)
        self.reserved_link_load.fill(0.0)
        self.reserved_compute_load.fill(0.0)

    def reserve_plan(  # 把已经决定发送的数据包路径提前登记到预约负载中。
        self,  # 当前 TransformerPathPlanner 实例，持有预测缓存和固定节点/链路顺序。
        plan: Optional[PathPlan],  # 本次规划得到的路径方案；None 表示没有可用规划。
        packet_size: float,  # 数据包计算前的原始大小，用于估算沿途内存和链路占用。
        computing_demand: float = 0.0,  # 该数据包需要的计算量，用于估算计算节点的算力占用。
        size_after_computing: Optional[float] = None,  # 数据包完成计算后的大小；None 时认为计算前后大小不变。
    ) -> None:  # 该函数只更新内部预约负载数组，不返回值。
        """把一次已发包路径的预计占用加入预约负载，供下一次规划立即使用。"""  # 函数整体作用说明。
        if plan is None or self.latest_snapshot is None or not plan.path:  # 没有规划、没有最新快照或路径为空时无法预约。
            return  # 直接返回，避免访问空对象或无效路径。

        path = list(plan.path)  # 拷贝路径节点列表，避免后续逻辑意外修改 PathPlan 内部对象。
        compute_nodes = {  # 根据 compute_flags 找出路径中被选为计算节点的卫星。
            node  # 当前路径节点名。
            for node, flag in zip(path, plan.compute_flags or [])  # 同时遍历路径节点和对应的计算标记。
            if int(flag) == 1  # 标记为 1 的节点表示该节点承担计算任务。
        }  # 得到所有计算节点集合。
        compute_idx = min((path.index(node) for node in compute_nodes if node in path), default=len(path))  # 找到最早发生计算的位置；没有计算节点时视为路径末尾之后。
        output_size = float(packet_size if size_after_computing is None else size_after_computing)  # 确定计算完成后的数据包大小。

        for node_idx, node in enumerate(path):  # 遍历路径上的每个节点，估算该节点接收数据包带来的队列/内存负载。
            fixed_idx = self.node_names.index(node) if node in self.node_names else None  # 将节点名映射到固定节点顺序中的索引。
            if fixed_idx is None:  # 如果节点不在固定节点列表中，说明无法写入预约数组。
                continue  # 跳过这个节点。
            capacity = (self.latest_snapshot.memory_capacity or {}).get(node, 0.0)  # 读取该节点内存容量，缺失时按 0 处理。
            if capacity:  # 只有容量有效时才计算归一化占用，避免除零。
                receive_size = output_size if node_idx > compute_idx else float(packet_size)  # 计算节点之后收到的是压缩/计算后的大小，否则是原始大小。
                self.reserved_queue_load[fixed_idx] += max(0.0, receive_size / float(capacity))  # 把该节点预计新增队列负载叠加到预约队列负载中。

        for src, dst in zip(path[:-1], path[1:]):  # 遍历路径上的每一条相邻链路。
            edge_idx = self._edge_index(src, dst)  # 将链路映射到固定边顺序中的索引，兼容反向边。
            if edge_idx is None:  # 如果该链路不在固定边列表中，说明无法写入预约数组。
                continue  # 跳过这条链路。
            capacity = (self.latest_snapshot.memory_capacity or {}).get(src, 0.0)  # 用发送端节点容量作为链路负载归一化基准。
            if capacity:  # 只有容量有效时才计算归一化链路占用。
                src_idx = path.index(src)  # 获取发送端节点在路径中的位置，用于判断是否已经过计算节点。
                transmit_size = output_size if src_idx >= compute_idx else float(packet_size)  # 计算发生之后链路传输计算后大小，否则传输原始大小。
                self.reserved_link_load[edge_idx] += max(0.0, transmit_size / float(capacity))  # 把该链路预计新增传输负载叠加到预约链路负载中。

        if computing_demand > 0:  # 只有任务存在正计算需求时，才需要预约算力负载。
            for node in compute_nodes:  # 遍历所有被标记为计算节点的路径节点。
                fixed_idx = self.node_names.index(node) if node in self.node_names else None  # 将计算节点映射到固定节点顺序中的索引。
                capacity = (self.latest_snapshot.computing_capacity or {}).get(node, 0.0)  # 读取该节点计算能力，缺失时按 0 处理。
                if fixed_idx is not None and capacity:  # 只有节点索引有效且计算能力非零时才写入预约算力负载。
                    self.reserved_compute_load[fixed_idx] += max(0.0, float(computing_demand) / float(capacity))  # 把该任务的归一化计算需求叠加到预约算力负载中。

    def predict_future(self, return_graphs: bool = True):
        """用历史快照预测未来负载；需要时把预测值写入未来 NetworkX 图。"""
        if len(self.history) < self.history_len:
            return None, None
        window = self.history[-self.history_len:]
        queue = th.as_tensor(np.stack([item.queue_load for item in window])[None], dtype=th.float32, device=self.device)
        link = th.as_tensor(np.stack([item.link_load for item in window])[None], dtype=th.float32, device=self.device)
        compute = th.as_tensor(np.stack([item.compute_queue for item in window])[None], dtype=th.float32, device=self.device)
        forecast_times = self._forecast_times(self.transformer.forecast_horizon)
        future_graphs = self._build_future_graphs(forecast_times) if return_graphs else None
        self.transformer.eval()
        with th.no_grad():
            preds = self.transformer(queue, link, compute)
        preds = self._apply_reservations_to_predictions(preds)
        if not return_graphs:
            return preds, None
        self.last_prediction_graphs = self._write_predictions_to_graphs(preds, forecast_times, future_graphs)
        return preds, self.last_prediction_graphs

    def _apply_reservations_to_predictions(self, preds: Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
        """把同一快照内已规划但尚未采样进真实状态的占用叠加到预测结果。"""
        if (
            not np.any(self.reserved_queue_load)
            and not np.any(self.reserved_link_load)
            and not np.any(self.reserved_compute_load)
        ):
            return preds

        adjusted = {key: value.clone() for key, value in preds.items()}
        queue_res = th.as_tensor(self.reserved_queue_load, dtype=adjusted["queue_forecast"].dtype, device=self.device)
        link_res = th.as_tensor(self.reserved_link_load, dtype=adjusted["link_forecast"].dtype, device=self.device)
        compute_res = th.as_tensor(self.reserved_compute_load, dtype=adjusted["compute_queue_forecast"].dtype, device=self.device)

        adjusted["queue_forecast"][..., 0] = adjusted["queue_forecast"][..., 0] + queue_res.view(1, 1, -1)
        adjusted["link_forecast"][..., 0] = adjusted["link_forecast"][..., 0] + link_res.view(1, 1, -1)
        adjusted["compute_queue_forecast"][..., 0] = adjusted["compute_queue_forecast"][..., 0] + compute_res.view(1, 1, -1)
        return adjusted

    def _build_future_graphs(self, forecast_times: List[float]) -> Optional[Dict[float, nx.Graph]]:
        """调用外部 future_graph_builder 生成指定预测时间点的未来拓扑图。"""
        if self.future_graph_builder is None:
            return None
        latest_time = float(self.history[-1].sim_time or 0.0) if self.history else 0.0
        return self.future_graph_builder(forecast_times, latest_time)

    def _write_predictions_to_graphs(
        self,
        preds: Dict[str, th.Tensor],
        forecast_times: List[float],
        future_graphs: Optional[Dict[float, nx.Graph]] = None,
    ) -> Dict[float, nx.Graph]:
        """把模型输出的节点/链路预测值写入各预测时间点对应的图中。"""
        queue_pred = preds["queue_forecast"][0, :, :, 0].detach().cpu().numpy()
        link_pred = preds["link_forecast"][0, :, :, 0].detach().cpu().numpy()
        compute_pred = preds["compute_queue_forecast"][0, :, :, 0].detach().cpu().numpy()
        business_time_pred = preds.get("business_time_forecast")
        business_time_pred = (
            business_time_pred[0, :, :, 0].detach().cpu().numpy()
            if business_time_pred is not None
            else np.zeros_like(queue_pred)
        )

        graphs = {}
        for horizon_idx, sim_time in enumerate(forecast_times):
            pred_idx = min(horizon_idx, queue_pred.shape[0] - 1)
            graph = self._graph_for_prediction_time(sim_time, future_graphs)
            graph.graph["sim_time"] = float(sim_time)
            for node_idx, node in enumerate(self.node_names):
                if node not in graph:
                    continue
                graph.nodes[node]["predicted_queue_load"] = float(queue_pred[pred_idx, node_idx])
                graph.nodes[node]["predicted_compute_queue"] = float(compute_pred[pred_idx, node_idx])
                graph.nodes[node]["predicted_business_time"] = float(business_time_pred[pred_idx, node_idx])
            for src, dst in list(graph.edges()):
                edge_idx = self._edge_index(src, dst)
                if edge_idx is None:
                    continue
                graph.edges[src, dst]["predicted_link_load"] = float(link_pred[pred_idx, edge_idx])
                if "propagation_delay" not in graph.edges[src, dst] and self.latest_snapshot and self.latest_snapshot.propagation_delays:
                    delay = self.latest_snapshot.propagation_delays.get(
                        (src, dst),
                        self.latest_snapshot.propagation_delays.get((dst, src)),
                    )
                    if delay is not None:
                        graph.edges[src, dst]["propagation_delay"] = float(delay)
            graphs[float(sim_time)] = graph
        return graphs

    def _graph_for_prediction_time(
        self,
        sim_time: float,
        future_graphs: Optional[Dict[float, nx.Graph]],
    ) -> nx.Graph:
        """获取某个预测时间的拓扑图；没有外部未来图时退化为当前固定边集。"""
        if future_graphs:
            graph = future_graphs.get(float(sim_time))
            if graph is None:
                graph = future_graphs.get(sim_time)
            if graph is not None:
                return graph.copy() if graph.is_directed() else nx.DiGraph(graph)

        graph = nx.DiGraph()
        graph.add_nodes_from(self.node_names)
        graph.add_edges_from(self.edge_names)
        return graph

    def _forecast_times(self, forecast_horizon: int) -> List[float]:
        """根据历史快照的仿真时间间隔推算未来每个预测步对应的时间。"""
        if not self.history:
            return [float(idx + 1) for idx in range(forecast_horizon)]

        latest_time = self.history[-1].sim_time
        if latest_time is None:
            latest_time = 0.0

        step = self._forecast_step()
        return [float(latest_time + step * (idx + 1)) for idx in range(forecast_horizon)]

    def _forecast_step(self) -> float:
        """返回预测时间片之间的仿真时间步长。"""
        intervals = [
            later.sim_time - earlier.sim_time
            for earlier, later in zip(self.history[:-1], self.history[1:])
            if earlier.sim_time is not None and later.sim_time is not None and later.sim_time > earlier.sim_time
        ]
        return float(np.median(intervals)) if intervals else 1.0

    def plan(
        self,
        source: str,
        destination: str,
        packet_size: float = 0.0,
        computing_demand: float = 0.0,
        size_after_computing: Optional[float] = None,
        business_duration: Optional[float] = None,
        need_compute: bool = True,
        top_k: int = 16,
        delay_top_k: Optional[int] = None,
        load_top_k: Optional[int] = None,
        max_hops: int = 12,
        show_path_error: bool = False,
    ) -> PathPlan:
        """生成候选路径并评分，返回综合风险、时延和负载最低的路径方案。"""
        if self.latest_snapshot is None:
            raise ValueError("No snapshot has been added to the planner.")
        if source not in self.latest_snapshot.adjacency:
            raise ValueError(f"Unknown source node: {source}")
        if destination not in self.latest_snapshot.adjacency:
            raise ValueError(f"Unknown destination node: {destination}")

        preds, graphs = self.predict_future(return_graphs=True)
        if preds is None:
            # preds = self._current_as_prediction()
            # graphs = None
            return None
        graphs = self._extend_future_graphs_for_planning(
            preds=preds,
            graphs=graphs,
            max_hops=max_hops,
            business_duration=business_duration,
        )

        candidates = self._candidate_paths(
            source,
            destination,
            top_k=top_k,
            max_hops=max_hops,
            delay_top_k=delay_top_k,
            load_top_k=load_top_k,
            preds=preds,
            graphs=graphs,
            show_path_error=show_path_error,
        )
        if not candidates:
            raise ValueError(f"No path found from {source} to {destination}.")

        plans = [
            self._score_path(
                path,
                preds,
                graphs,
                packet_size,
                computing_demand,
                size_after_computing,
                business_duration,
                need_compute,
            )
            for path in candidates
        ]
        if plans == None:
            a = 1
        return min(plans, key=lambda item: item.score)

    def _candidate_paths(  # 生成最终送入评分器的候选路径集合。
        self,  # 当前 TransformerPathPlanner 实例，提供拓扑、预测负载和时延读取能力。
        source: str,  # 本次规划的源卫星节点。
        destination: str,  # 本次规划的目的卫星节点。
        top_k: int,  # 兼容旧配置的候选数量；未显式配置 delay_top_k 时用它作为低时延候选数量。
        max_hops: int,  # 限制候选路径最多允许经过的链路跳数，防止搜索过深。
        delay_top_k: Optional[int] = None,  # 低时延候选组数量 n；None 时退回使用 top_k。
        load_top_k: Optional[int] = None,  # 低负载候选组数量 m；None 时默认不额外生成低负载候选。
        preds: Optional[Dict[str, th.Tensor]] = None,  # Transformer 输出的未来负载张量，用于构造低负载候选。
        graphs: Optional[Dict[float, nx.Graph]] = None,  # 写入预测负载后的未来拓扑图，用于读取节点/链路预测负载。
        show_path_error: bool = False,  # 是否在候选路径为空时打印诊断原因。
    ) -> List[List[str]]:  # 返回路径列表，每条路径是按经过顺序排列的节点名列表。
        """分别按低时延和低预测负载生成候选路径，合并去重后返回。"""  #函数的整体行为说明。
        delay_count = int(top_k if delay_top_k is None else delay_top_k)  # 确定低时延候选组要保留多少条路径。
        load_count = int(0 if load_top_k is None else load_top_k)  # 确定低负载候选组要保留多少条路径。
        adjacency = self._candidate_adjacency(graphs)
        candidates = []  # 暂存两组搜索得到的候选路径，后面会统一去重。
        if delay_count > 0:  # 只有配置的低时延候选数量大于 0 时，才执行低时延路径搜索。
            candidates.extend(  # 把低时延搜索得到的路径追加到候选集合中。
                self._candidate_paths_by_cost(  # 复用通用的“按代价最小优先”路径搜索函数。
                    source,  # 传入源节点。
                    destination,  # 传入目的节点。
                    top_k=delay_count,  # 低时延候选组只取 delay_count 条。
                    max_hops=max_hops,  # 搜索过程中仍然遵守最大跳数限制。
                    initial_cost=lambda node: 0.0,  # 源节点自身不引入初始时延代价。
                    edge_cost=lambda src, dst: self._candidate_edge_delay(src, dst, graphs),  # 每扩展一条边时，用未来图传播时延作为路径增量代价。
                    adjacency=adjacency,
                )  # 低时延候选搜索结束。
            )  # 低时延候选已加入 candidates。
        if load_count > 0:  # 只有配置的低负载候选数量大于 0 时，才执行低负载路径搜索。
            load_lookup = self._prediction_load_lookup(preds=preds, graphs=graphs)  # 提取预测窗口内节点、链路、算力负载查询表。
            candidates.extend(  # 把低负载搜索得到的路径追加到候选集合中。
                self._candidate_paths_by_cost(  # 继续复用同一个按代价搜索的函数，只是代价定义换成负载。
                    source,  # 传入源节点。
                    destination,  # 传入目的节点。
                    top_k=load_count,  # 低负载候选组只取 load_count 条。
                    max_hops=max_hops,  # 搜索过程中仍然遵守最大跳数限制。
                    initial_cost=lambda node: load_lookup["node"].get(node, 1.0) + load_lookup["compute"].get(node, 1.0),  # 源节点初始代价由队列负载和算力负载组成。
                    edge_cost=lambda src, dst: (  # 每扩展到下一跳时，累计链路负载、下一跳队列负载和下一跳算力负载。
                        load_lookup["edge"].get((src, dst), load_lookup["edge"].get((dst, src), 1.0))  # 读取该链路的预测负载；找不到时按高负载 1.0 处理。
                        + load_lookup["node"].get(dst, 1.0)  # 加上下一跳节点的预测队列负载。
                        + load_lookup["compute"].get(dst, 1.0)  # 加上下一跳节点的预测算力负载。
                    ),  # 低负载路径扩展代价定义结束。
                    adjacency=adjacency,
                )  # 低负载候选搜索结束。
            )  # 低负载候选已加入 candidates。

        unique_paths = []  # 保存去重后的候选路径，保持“低时延组在前、低负载组在后”的发现顺序。
        seen = set()  # 记录已经出现过的路径 tuple，用于快速判断重复。
        for path in candidates:  # 逐条检查两组候选路径。
            key = tuple(path)  # 列表不能放进 set，所以转成 tuple 作为去重键。
            if key in seen:  # 如果这条路径已经由另一组候选生成过，就跳过。
                continue  # 不重复加入 unique_paths。
            seen.add(key)  # 把当前路径标记为已出现。
            unique_paths.append(path)  # 当前路径是新路径，加入最终候选列表。
        if not unique_paths and show_path_error:  # 如果没有任何候选路径且开启了诊断输出。
            self._print_candidate_path_error(source, destination, adjacency, max_hops)  # 打印是超过跳数还是中途无链路可走。
        return unique_paths  # 返回去重后的候选路径，后续会统一进入 _score_path 打分。

    def _candidate_paths_by_cost(
        self,
        source: str,
        destination: str,
        top_k: int,
        max_hops: int,
        initial_cost: Callable[[str], float],
        edge_cost: Callable[[str, str], float],
        adjacency: Optional[Dict[str, List[str]]] = None,
    ) -> List[List[str]]:
        """用 Dijkstra 风格搜索源到目的地的前 top_k 条低代价简单候选路径。"""
        if adjacency is None:
            adjacency = self.latest_snapshot.adjacency if self.latest_snapshot is not None else {}

        start_cost = max(0.0, float(initial_cost(source)))
        start_state = (source, frozenset([source]))
        best_state_cost = {start_state: start_cost}
        push_order = 0
        queue = [(start_cost, 0, push_order, [source])]
        paths = []
        expansions = 0
        while queue and len(paths) < top_k:
            cost, hops, _, path = heapq.heappop(queue)
            node = path[-1]
            state = (node, frozenset(path))
            if cost > best_state_cost.get(state, float("inf")):
                continue
            expansions += 1
            if expansions > self.max_candidate_expansions:
                break
            if node == destination:
                paths.append(path)
                continue
            if hops >= max_hops:
                continue
            for nbr in adjacency.get(node, []):
                if nbr in path:
                    continue
                next_path = path + [nbr]
                next_hops = hops + 1
                next_cost = cost + max(0.0, float(edge_cost(node, nbr)))
                next_state = (nbr, frozenset(next_path))
                if next_cost >= best_state_cost.get(next_state, float("inf")):
                    continue
                best_state_cost[next_state] = next_cost
                push_order += 1
                heapq.heappush(queue, (next_cost, next_hops, push_order, next_path))
                if len(queue) > self.max_candidate_queue_size:
                    queue = heapq.nsmallest(self.max_candidate_queue_size, queue)
                    heapq.heapify(queue)
        return paths

    def _print_candidate_path_error(
        self,
        source: str,
        destination: str,
        adjacency: Dict[str, List[str]],
        max_hops: int,
    ) -> None:
        """在候选路径为空时打印失败原因：超过最大跳数，或某些节点没有链路可继续走。"""
        required_hops = self._shortest_hops_by_dijkstra(source, destination, adjacency)
        if required_hops is not None and required_hops > max_hops:
            print(
                f"[Transformer path error] source={source}, destination={destination}: "
                f"path exists but exceeds max_hops={max_hops}; required_hops={required_hops}."
            )
            return
        if required_hops is not None:
            print(
                f"[Transformer path error] source={source}, destination={destination}: "
                f"path is reachable within max_hops={max_hops}; required_hops={required_hops}, "
                f"but no candidate was generated. Check search limits: "
                f"max_candidate_expansions={self.max_candidate_expansions}, "
                f"max_candidate_queue_size={self.max_candidate_queue_size}."
            )
            return

        dead_nodes = self._dead_end_nodes_from_source(source, destination, adjacency)
        if dead_nodes:
            print(
                f"[Transformer path error] source={source}, destination={destination}: "
                f"no outgoing link can continue from node(s): {', '.join(dead_nodes)}."
            )
            return

        print(
            f"[Transformer path error] source={source}, destination={destination}: "
            "no reachable path in candidate adjacency."
        )

    @staticmethod
    def _shortest_hops_by_dijkstra(
        source: str,
        destination: str,
        adjacency: Dict[str, List[str]],
    ) -> Optional[int]:
        """不限制 max_hops 做 Dijkstra 搜索，返回源点到目的点的最少跳数；不可达返回 None。"""
        queue = [(0.0, 0, source)]
        best_cost = {source: 0.0}
        best_hops = {source: 0}
        while queue:
            cost, hops, node = heapq.heappop(queue)
            if cost > best_cost.get(node, float("inf")):
                continue
            if node == destination:
                return hops
            for nbr in adjacency.get(node, []):
                next_cost = cost + 1.0
                next_hops = hops + 1
                previous_cost = best_cost.get(nbr, float("inf"))
                previous_hops = best_hops.get(nbr, float("inf"))
                if next_cost > previous_cost or (next_cost == previous_cost and next_hops >= previous_hops):
                    continue
                best_cost[nbr] = next_cost
                best_hops[nbr] = next_hops
                heapq.heappush(queue, (next_cost, next_hops, nbr))
        return None

    @staticmethod
    def _dead_end_nodes_from_source(
        source: str,
        destination: str,
        adjacency: Dict[str, List[str]],
    ) -> List[str]:
        """找出从源点可达、但无法继续向目的方向扩展的节点。"""
        queue = [source]
        visited = {source}
        dead_nodes = set()
        while queue:
            node = queue.pop(0)
            if node == destination:
                continue
            next_nodes = list(adjacency.get(node, []))
            if not next_nodes:
                dead_nodes.add(node)
                continue
            for nbr in next_nodes:
                if nbr in visited:
                    continue
                visited.add(nbr)
                queue.append(nbr)
        return sorted(dead_nodes)

    def _candidate_adjacency(self, graphs: Optional[Dict[float, nx.Graph]]) -> Dict[str, List[str]]:
        """优先从未来预测图构造候选路径搜索邻接表，缺失时退回当前快照。"""
        if graphs:
            reachable_edges = set()
            if self.latest_snapshot is not None:
                for src, neighbors in self.latest_snapshot.adjacency.items():
                    if src in self.node_names:
                        for dst in neighbors:
                            if dst in self.node_names:
                                reachable_edges.add((src, dst))
            for graph in graphs.values():
                for src, dst in graph.edges():
                    if src in self.node_names and dst in self.node_names:
                        reachable_edges.add((src, dst))
                        if not graph.is_directed():
                            reachable_edges.add((dst, src))

            adjacency = {node: set() for node in self.node_names}
            for src, dst in reachable_edges:
                adjacency[src].add(dst)
            return {node: sorted(neighbors) for node, neighbors in adjacency.items()}
        return self.latest_snapshot.adjacency if self.latest_snapshot is not None else {}

    def _extend_future_graphs_for_planning(
        self,  
        preds: Dict[str, th.Tensor],  # Transformer 已经预测出的未来负载张量，用于写回扩展后的未来图。
        graphs: Optional[Dict[float, nx.Graph]],  # 已按原始 forecast_horizon 构造并写入预测负载的未来图；可能为 None。
        max_hops: int,  # 本次路径搜索允许的最大跳数，用来估算数据包最远可能转发多久。
        business_duration: Optional[float],  # 业务持续时间；若存在，需要把拓扑检查窗口继续向后扩展这段时间。
    ) -> Optional[Dict[float, nx.Graph]]:  # 返回扩展后的未来图字典；无法扩展时返回原 graphs。
        """为路径规划额外采样更远的未来拓扑，负载预测超出 horizon 时复用最后一步。"""  # 函数用途说明。
        if self.future_graph_builder is None:  # 如果环境没有提供未来拓扑构造器，就无法额外生成未来图。
            return graphs  # 直接返回原来的预测图，保持旧行为。

        latest_time = self._latest_sim_time_for_forecast()  # 取得预测窗口的起点，也就是最新真实快照对应的仿真时间。
        step = self._forecast_step()  # 取得两个预测时间片之间的仿真时间间隔。
        existing_times = sorted(float(time) for time in (graphs or {}).keys())  # 取出原有未来图的时间点并按升序排列。
        existing_span = max((time - latest_time for time in existing_times), default=0.0)  # 计算原有未来图已经覆盖到当前时刻之后多远。

        edge_delays = []  # 暂存当前快照中可读取到的链路传播时延。
        if self.latest_snapshot is not None and self.latest_snapshot.propagation_delays:  # 只有存在最新快照和传播时延表时才读取链路时延。
            for delay in self.latest_snapshot.propagation_delays.values():  # 遍历当前已知的所有链路传播时延。
                try:  # 尝试把时延转换成非负浮点数。
                    edge_delays.append(max(0.0, float(delay)))  # 记录合法时延，负值按 0 处理。
                except (TypeError, ValueError):  # 如果时延为空、字符串异常或不能转成 float。
                    continue  # 跳过这个非法时延值，继续处理其他链路。
        edge_step = max(max(edge_delays, default=step), step)  # 用最大链路时延和预测步长中的较大值，估算每一跳最坏耗时。
        route_span = max(0, int(max_hops)) * edge_step  # 根据最大跳数估算整条候选路径可能需要覆盖的未来时间跨度。
        if business_duration is not None:  # 如果业务还有持续时间窗口要求。
            route_span += max(0.0, float(business_duration))  # 把非负业务持续时间追加到拓扑检查跨度里。
        target_span = max(existing_span, route_span)  # 最终需要覆盖的跨度取“已有覆盖”和“路径可能需要覆盖”中的较大值。
        if target_span <= existing_span + 1e-9:  # 如果原有未来图已经覆盖到目标跨度，考虑浮点误差后无需扩展。
            return graphs  # 返回原有未来图，避免重复调用环境构造拓扑。

        step_count = int(math.ceil(target_span / max(step, 1e-9)))  # 按预测步长计算需要多少个未来时间片，分母加下限避免除零。
        planning_times = [float(latest_time + step * (idx + 1)) for idx in range(step_count)]  # 生成从 t+step 到目标跨度内的规划时间点。
        future_graphs = self._build_future_graphs(planning_times)  # 调用环境提供的构造器，为这些时间点生成真实未来拓扑。
        if future_graphs is None:  # 如果构造器没有返回有效未来图。
            return graphs  # 回退到原有预测图，避免规划流程中断。
        return self._write_predictions_to_graphs(preds, planning_times, future_graphs)  # 把预测负载写入扩展后的拓扑图并返回。

    def _candidate_edge_delay(
        self,
        src: str,
        dst: str,
        graphs: Optional[Dict[float, nx.Graph]],
    ) -> float:
        """候选搜索时优先使用未来图中的边时延，多个未来切片取最大值。"""
        if not graphs:
            return self._edge_delay(src, dst)
        delays = []
        for graph in graphs.values():
            edge_data = self._graph_edge_data(graph, src, dst)
            if edge_data is not None:
                delays.append(self._edge_delay_from_graph(edge_data, default=self._edge_delay(src, dst)))
        return max(delays) if delays else self._edge_delay(src, dst)

    def _prediction_load_lookup(
        self,
        preds: Optional[Dict[str, th.Tensor]],
        graphs: Optional[Dict[float, nx.Graph]],
    ) -> Dict[str, Dict]:
        """提取每个节点/链路在预测窗口内的最大队列、链路和计算负载。"""
        node_load = {node: 1.0 for node in self.node_names}
        compute_load = {node: 1.0 for node in self.node_names}
        edge_load = {edge: 1.0 for edge in self.edge_names}

        if graphs:
            node_values = {node: [] for node in self.node_names}
            compute_values = {node: [] for node in self.node_names}
            edge_values = {edge: [] for edge in self.edge_names}
            for graph in graphs.values():
                for node in self.node_names:
                    if node in graph:
                        if "predicted_queue_load" in graph.nodes[node]:
                            node_values[node].append(float(graph.nodes[node]["predicted_queue_load"]))
                        if "predicted_compute_queue" in graph.nodes[node]:
                            compute_values[node].append(float(graph.nodes[node]["predicted_compute_queue"]))
                for src, dst in self.edge_names:
                    edge_data = self._graph_edge_data(graph, src, dst)
                    if edge_data is not None and "predicted_link_load" in edge_data:
                        edge_values[(src, dst)].append(float(edge_data["predicted_link_load"]))
            node_load = {node: max(values) if values else 1.0 for node, values in node_values.items()}
            compute_load = {node: max(values) if values else 1.0 for node, values in compute_values.items()}
            edge_load = {edge: max(values) if values else 1.0 for edge, values in edge_values.items()}
            return {"node": node_load, "edge": edge_load, "compute": compute_load}

        if preds is None:
            return {"node": node_load, "edge": edge_load, "compute": compute_load}

        queue_pred = preds["queue_forecast"][0, :, :, 0].detach().cpu().numpy()
        link_pred = preds["link_forecast"][0, :, :, 0].detach().cpu().numpy()
        compute_pred = preds["compute_queue_forecast"][0, :, :, 0].detach().cpu().numpy()
        for idx, node in enumerate(self.node_names):
            node_load[node] = float(queue_pred[:, idx].max())
            compute_load[node] = float(compute_pred[:, idx].max())
        for idx, edge in enumerate(self.edge_names):
            edge_load[edge] = float(link_pred[:, idx].max())
        return {"node": node_load, "edge": edge_load, "compute": compute_load}

    def _score_path(
        self,
        path: List[str],
        preds: Dict[str, th.Tensor],
        graphs: Optional[Dict[float, nx.Graph]],
        packet_size: float,
        computing_demand: float,
        size_after_computing: Optional[float],
        business_duration: Optional[float],
        need_compute: bool,
    ) -> PathPlan:
        """计算单条候选路径的队列、链路、计算、时延和丢包风险综合评分。"""
        if graphs:
            (
                queue_load,
                link_load,
                compute_nodes,
                compute_shares,
                compute_load,
                predicted_delay,
                topology_risk,
            ) = self._graph_path_metrics(
                path,
                graphs,
                packet_size,
                computing_demand,
                size_after_computing,
                business_duration,
                need_compute,
            )
        else:
            queue_pred = preds["queue_forecast"][0, :, :, 0].detach().cpu().numpy()
            link_pred = preds["link_forecast"][0, :, :, 0].detach().cpu().numpy()
            compute_pred = preds["compute_queue_forecast"][0, :, :, 0].detach().cpu().numpy()
            node_indices = [self.node_names.index(node) for node in path if node in self.node_names]
            edge_indices = [self._edge_index(src, dst) for src, dst in zip(path[:-1], path[1:])]
            edge_indices = [idx for idx in edge_indices if idx is not None]

            queue_load = float(queue_pred[:, node_indices].max()) if node_indices else 1.0
            link_load = float(link_pred[:, edge_indices].max()) if edge_indices else 1.0

            compute_nodes = []
            compute_shares = {}
            compute_load = 0.0
            compute_delay = 0.0
            compute_risk = 0.0
            if need_compute:
                compute_nodes, compute_shares, compute_load, compute_delay, compute_risk = self._best_compute_nodes(
                    path,
                    compute_pred,
                    computing_demand,
                    packet_size=packet_size,
                    size_after_computing=size_after_computing,
                    queue_pred=queue_pred,
                    business_duration=business_duration,
                )
            predicted_delay = self._path_delay(path) + compute_delay
            topology_risk = compute_risk
            compute_node = compute_nodes[0] if compute_nodes else None
        compute_node = compute_nodes[0] if compute_nodes else None
        if compute_node is not None:
            compute_nodes = [compute_node]
            compute_shares = {compute_node: float(computing_demand)}
        compute_flags = [1 if node == compute_node else 0 for node in path]
        compute_demands = [float(compute_shares.get(node, 0.0)) for node in path]
        disappearing_link_count, disappearing_links, disappearing_link_times = self._future_disappearing_links(
            path,
            graphs,
        )

        packet_risk = self._packet_capacity_risk(
            path,
            packet_size,
            computing_demand,
            compute_nodes[-1] if compute_nodes else None,
            size_after_computing=size_after_computing,
            graphs=graphs,
            preds=preds,
        )
        drop_risk = max(
            max(0.0, queue_load - self.queue_threshold),
            max(0.0, link_load - self.link_threshold),
            max(0.0, compute_load - self.compute_threshold),
            topology_risk,
            packet_risk,
        )

        weights = self.score_weights
        score = (
            weights["link"] * link_load
            + weights["queue"] * queue_load
            + weights["compute"] * compute_load
            + weights["delay"] * predicted_delay
            + weights["hop"] * max(0, len(path) - 1)
            + weights["drop"] * drop_risk
        )

        return PathPlan(
            path=path,
            compute_flags=compute_flags,
            score=float(score),
            predicted_delay=float(predicted_delay),
            max_queue_load=queue_load,
            max_link_load=link_load,
            max_compute_queue=float(compute_load),
            drop_risk=float(drop_risk),
            details={
                "hops": float(max(0, len(path) - 1)),
                "compute_node": -1.0 if compute_node is None else float(self.node_names.index(compute_node)),
                "compute_node_count": float(len(compute_nodes)),
                "packet_capacity_risk": float(packet_risk),
                "topology_risk": float(topology_risk),
                "future_disappearing_link_count": float(disappearing_link_count),
                "planned_compute_demand": float(sum(compute_demands)),
            },
            compute_demands=compute_demands,
            disappearing_links=disappearing_links,
            disappearing_link_times=disappearing_link_times,
        )

    def _future_disappearing_links(
        self,
        path: List[str],
        graphs: Optional[Dict[float, nx.Graph]],
    ) -> Tuple[int, List[EdgeName], Dict[EdgeName, List[float]]]:
        """统计路径上的链路在未来拓扑图中会消失的数量，并记录链路和对应时间。"""
        if not graphs:
            return 0, [], {}

        disappearing_link_times: Dict[EdgeName, List[float]] = {}
        for sim_time, graph in sorted(graphs.items(), key=lambda item: float(item[0])):
            for src, dst in zip(path[:-1], path[1:]):
                edge = (src, dst)
                if self._graph_edge_data(graph, src, dst) is None:
                    disappearing_link_times.setdefault(edge, []).append(float(sim_time))

        disappearing_links = list(disappearing_link_times.keys())
        return len(disappearing_links), disappearing_links, disappearing_link_times

    def _graph_path_metrics(
        self,
        path: List[str],
        graphs: Dict[float, nx.Graph],
        packet_size: float,
        computing_demand: float,
        size_after_computing: Optional[float],
        business_duration: Optional[float],
        need_compute: bool,
    ) -> Tuple[float, float, List[str], Dict[str, float], float, float, float]:
        """基于数据包预计到达时间，从未来预测图读取路径负载、时延和拓扑风险。"""
        graph_list = list(graphs.values())
        if not graph_list:
            return 1.0, 1.0, [], {}, 1.0 if need_compute else 0.0, self._path_delay(path), 1.0

        compute_nodes = []
        compute_shares = {}
        compute_load = 0.0
        compute_delay = 0.0
        compute_risk = 0.0
        if need_compute:
            compute_nodes, compute_shares, compute_load, compute_delay, compute_risk = self._best_compute_nodes_from_graphs(
                path,
                graph_list,
                computing_demand,
                packet_size=packet_size,
                size_after_computing=size_after_computing,
                business_duration=business_duration,
            )

        timed_graphs = self._timed_graphs(graph_list)
        queue_values = []
        link_values = []
        topology_risk = 0.0
        elapsed = 0.0
        compute_delay_by_node = {node: compute_delay for node in compute_nodes[:1]}

        for node_idx, node in enumerate(path):
            graph = self._graph_at_elapsed(timed_graphs, elapsed)
            if graph is not None and node in graph and "predicted_queue_load" in graph.nodes[node]:
                queue_values.append(max(0.0, float(graph.nodes[node]["predicted_queue_load"])))
            else:
                topology_risk = 1.0
                queue_values.append(1.0)

            if node_idx >= len(path) - 1:
                continue

            elapsed += float(compute_delay_by_node.get(node, 0.0))
            src, dst = node, path[node_idx + 1]
            graph, edge_data, wait_time = self._edge_graph_at_or_after(timed_graphs, src, dst, elapsed)
            if edge_data is None:
                topology_risk = 1.0
                link_values.append(1.0)
                elapsed += self._edge_delay(src, dst)
                continue
            elapsed += wait_time
            link_values.append(max(0.0, float(edge_data.get("predicted_link_load", self._edge_load_fallback(src, dst)))))
            elapsed += self._edge_delay_from_graph(edge_data, default=self._edge_delay(src, dst))

        queue_load = max(queue_values) if queue_values else 1.0
        link_load = max(link_values) if link_values else 1.0
        predicted_delay = elapsed + float(compute_delay_by_node.get(path[-1], 0.0) if path else 0.0)
        topology_risk = max(topology_risk, compute_risk)
        return (
            float(queue_load),
            float(link_load),
            compute_nodes,
            compute_shares,
            float(compute_load),
            float(predicted_delay),
            float(topology_risk),
        )

    def _best_compute_nodes_from_graphs(
        self,
        path: List[str],
        graphs: Sequence[nx.Graph],
        computing_demand: float,
        packet_size: float = 0.0,
        size_after_computing: Optional[float] = None,
        business_duration: Optional[float] = None,
    ) -> Tuple[List[str], Dict[str, float], float, float, float]:
        """从预测图中读取计算队列预测值，并选择该路径上的最佳单个计算节点。"""
        predicted = {}
        for node in path:
            predicted_values = [
                float(graph.nodes[node]["predicted_compute_queue"])
                for graph in graphs
                if node in graph and "predicted_compute_queue" in graph.nodes[node]
            ]
            if predicted_values:
                predicted[node] = max(predicted_values)
        return self._best_compute_plan(
            path=path,
            predicted_compute=predicted,
            computing_demand=computing_demand,
            packet_size=packet_size,
            size_after_computing=size_after_computing,
            graphs=graphs,
            business_duration=business_duration,
        )

    @staticmethod
    def _graph_edge_data(graph: nx.Graph, src: str, dst: str):
        """读取图中 src 到 dst 的边属性；有向图中也尝试反向边以兼容历史数据。"""
        if graph is None:
            return None
        data = graph.get_edge_data(src, dst)
        if data is not None:
            return data
        if not graph.is_directed():
            return None
        return graph.get_edge_data(dst, src)

    def _edge_graph_at_or_after(
        self,
        timed_graphs: Sequence[Tuple[float, nx.Graph]],
        src: str,
        dst: str,
        elapsed: float,
    ) -> Tuple[Optional[nx.Graph], Optional[Dict], float]:
        """查找到达该链路时刻或之后第一个存在链路的未来图，并返回等待时间。"""
        if not timed_graphs:
            return None, None, 0.0
        target_time = self._latest_sim_time_for_forecast(timed_graphs) + max(0.0, float(elapsed))
        fallback_graph = timed_graphs[-1][1]
        for sim_time, graph in timed_graphs:
            if sim_time < target_time - 1e-9:
                continue
            edge_data = self._graph_edge_data(graph, src, dst)
            if edge_data is not None:
                return graph, edge_data, max(0.0, float(sim_time) - target_time)
        return fallback_graph, None, 0.0

    @staticmethod
    def _edge_delay_from_graph(edge_data: Dict, default: float = 1.0) -> float:
        """从边属性中读取传播时延，缺失时返回默认值。"""
        for key in ("propagation_delay", "delay", "propagation_weight"):
            if key in edge_data and edge_data[key] is not None:
                return float(edge_data[key])
        return float(default)

    def _best_compute_nodes(
        self,
        path: List[str],
        compute_pred: np.ndarray,
        computing_demand: float,
        packet_size: float = 0.0,
        size_after_computing: Optional[float] = None,
        queue_pred: Optional[np.ndarray] = None,
        business_duration: Optional[float] = None,
    ) -> Tuple[List[str], Dict[str, float], float, float, float]:
        """
        Pick exactly one compute node by simulating the end-to-end effect on
        this path: compute waiting time, future link availability windows,
        downstream memory after shrinking, and total delay.
        """
        predicted_compute = {}
        for node in path:
            if node not in self.node_names:
                continue
            idx = self.node_names.index(node)
            predicted_compute[node] = float(compute_pred[:, idx].max())
        return self._best_compute_plan(
            path=path,
            predicted_compute=predicted_compute,
            computing_demand=computing_demand,
            packet_size=packet_size,
            size_after_computing=size_after_computing,
            queue_pred=queue_pred,
            business_duration=business_duration,
        )

    def _best_compute_node(
        self,
        path: List[str],
        compute_pred: np.ndarray,
        computing_demand: float,
        packet_size: float = 0.0,
        size_after_computing: Optional[float] = None,
        queue_pred: Optional[np.ndarray] = None,
        business_duration: Optional[float] = None,
    ) -> Tuple[List[str], Dict[str, float], float, float, float]:
        """单节点计算选择的兼容接口，内部复用 _best_compute_nodes。"""
        return self._best_compute_nodes(
            path,
            compute_pred,
            computing_demand,
            packet_size=packet_size,
            size_after_computing=size_after_computing,
            queue_pred=queue_pred,
            business_duration=business_duration,
        )

    def _best_compute_plan(
        self,
        path: List[str],
        predicted_compute: Dict[str, float],
        computing_demand: float,
        packet_size: float = 0.0,
        size_after_computing: Optional[float] = None,
        queue_pred: Optional[np.ndarray] = None,
        graphs: Optional[Sequence[nx.Graph]] = None,
        business_duration: Optional[float] = None,
    ) -> Tuple[List[str], Dict[str, float], float, float, float]:
        """枚举路径上每个可计算节点，按计算耗时、总时延和链路/内存风险选最佳节点。"""
        if computing_demand <= 0:
            return [], {}, 0.0, 0.0, 0.0

        compute_path = path[:-1] if len(path) > 1 else path
        candidates = [node for node in compute_path if node in predicted_compute]
        if not candidates:
            return [], {}, 1.0, 0.0, 1.0

        best_node: Optional[str] = None
        best_load = float("inf")
        best_delay = 0.0
        best_risk = 1.0
        best_score = float("inf")

        for node in candidates:
            share = float(computing_demand)
            compute_delay = float(predicted_compute[node]) + max(0.0, share / max(self._compute_capacity(node), 1e-6))
            shares = {node: share}
            compute_delays = {node: compute_delay}
            compute_load = compute_delay
            risk = self._compute_plan_path_risk(
                path=path,
                compute_shares=shares,
                compute_delays=compute_delays,
                total_computing_demand=computing_demand,
                packet_size=packet_size,
                size_after_computing=size_after_computing,
                queue_pred=queue_pred,
                graphs=graphs,
                business_duration=business_duration,
            )
            path_delay = self._candidate_total_delay(path, compute_delays=compute_delays, graphs=graphs)
            score = path_delay + compute_load + 1000.0 * risk
            if score < best_score:
                best_score = score
                best_node = node
                best_load = compute_load
                best_delay = compute_delay
                best_risk = risk

        if best_node is None:
            return [], {}, 1.0, 0.0, 1.0
        return [best_node], {best_node: float(computing_demand)}, float(best_load), float(best_delay), float(best_risk)

    def _compute_plan_path_risk(
        self,
        path: List[str],
        compute_shares: Dict[str, float],
        compute_delays: Dict[str, float],
        total_computing_demand: float,
        packet_size: float,
        size_after_computing: Optional[float],
        queue_pred: Optional[np.ndarray] = None,
        graphs: Optional[Sequence[nx.Graph]] = None,
        business_duration: Optional[float] = None,
    ) -> float:
        """评估某个计算方案对后续链路可达性和节点内存接收容量造成的风险。"""
        risk = 0.0
        if graphs:
            risk = max(
                risk,
                self._candidate_link_window_risk(path, compute_delays, graphs, business_duration),
            )
        if packet_size > 0:
            for node_idx, node in enumerate(path):
                receive_size = self._packet_size_after_compute_progress(
                    packet_size,
                    size_after_computing,
                    compute_shares,
                    total_computing_demand,
                    path,
                    node_idx,
                )
                risk = max(
                    risk,
                    self._memory_receive_risk(node, receive_size, queue_pred=queue_pred, graphs=graphs),
                )
        return float(risk)

    def _candidate_link_window_risk(
        self,
        path: List[str],
        compute_delays: Dict[str, float],
        graphs: Sequence[nx.Graph],
        business_duration: Optional[float],
    ) -> float:
        """检查计算等待后数据包经过每条链路的时间窗口内链路是否仍存在。"""
        timed_graphs = self._timed_graphs(graphs)
        if not timed_graphs:
            return 0.0
        latest_time = (
            float(self.latest_snapshot.sim_time)
            if self.latest_snapshot and self.latest_snapshot.sim_time is not None
            else timed_graphs[0][0]
        )
        if self.latest_snapshot is not None:
            timed_graphs.append((latest_time, self._snapshot_to_graph(self.latest_snapshot, latest_time)))
            timed_graphs.sort(key=lambda item: item[0])
        end_time = latest_time + float(business_duration) if business_duration is not None else timed_graphs[-1][0]
        elapsed = 0.0
        for edge_idx, (src, dst) in enumerate(zip(path[:-1], path[1:])):
            elapsed += float(compute_delays.get(src, 0.0))
            graph, first_edge, wait_time = self._edge_graph_at_or_after(timed_graphs, src, dst, elapsed)
            if first_edge is None:
                return 1.0
            arrival_time = latest_time + elapsed + wait_time
            if arrival_time > end_time + 1e-9:
                return 1.0
            elapsed += wait_time
            elapsed += self._edge_delay_from_graph(first_edge or {}, default=1.0)
        return 0.0

    def _memory_receive_risk(
        self,
        node: str,
        packet_size: float,
        queue_pred: Optional[np.ndarray] = None,
        graphs: Optional[Sequence[nx.Graph]] = None,
    ) -> float:
        """估算节点接收该数据包后是否会超过内存容量，返回归一化溢出风险。"""
        capacity = None
        if self.latest_snapshot and self.latest_snapshot.memory_capacity:
            capacity = self.latest_snapshot.memory_capacity.get(node)
        if not capacity:
            return 0.0
        load = None
        if graphs:
            values = [
                float(graph.nodes[node]["predicted_queue_load"])
                for graph in graphs
                if node in graph and "predicted_queue_load" in graph.nodes[node]
            ]
            if values:
                load = max(values)
        elif queue_pred is not None and node in self.node_names:
            load = float(queue_pred[:, self.node_names.index(node)].max())
        if load is None and self.latest_snapshot and node in self.node_names:
            load = float(self.latest_snapshot.queue_load[self.node_names.index(node), 0])
        used = max(0.0, float(load or 0.0)) * float(capacity)
        return max(0.0, (used + float(packet_size) - float(capacity)) / float(capacity))

    def _candidate_total_delay(
        self,
        path: List[str],
        compute_delays: Optional[Dict[str, float]] = None,
        graphs: Optional[Sequence[nx.Graph]] = None,
    ) -> float:
        """计算候选路径的端到端时延，包括传播时延和各节点计算等待/执行时间。"""
        compute_delays = compute_delays or {}
        delay = 0.0
        for src, dst in zip(path[:-1], path[1:]):
            delay += float(compute_delays.get(src, 0.0))
            if graphs:
                edge_delay = None
                for graph in graphs:
                    edge_data = self._graph_edge_data(graph, src, dst)
                    if edge_data is not None:
                        edge_delay = self._edge_delay_from_graph(edge_data, default=1.0)
                        break
                delay += float(edge_delay if edge_delay is not None else 1.0)
            else:
                delay += float(self._edge_delay(src, dst))
        if path:
            delay += float(compute_delays.get(path[-1], 0.0))
        return float(delay)

    def _compute_demand_shares(self, nodes: Sequence[str], computing_demand: float) -> Dict[str, float]:
        """旧接口兼容：计算需求不再拆分，只分配给第一个候选节点。"""
        if not nodes or computing_demand <= 0:
            return {}
        return {nodes[0]: float(computing_demand)}

    def _compute_capacity(self, node: str) -> float:
        """读取节点计算能力，缺失时返回 1 避免除零。"""
        if self.latest_snapshot and self.latest_snapshot.computing_capacity:
            capacity = self.latest_snapshot.computing_capacity.get(node)
            if capacity:
                return float(capacity)
        return 1.0

    @staticmethod
    def _packet_size_after_compute_progress(
        packet_size: float,
        size_after_computing: Optional[float],
        compute_shares: Dict[str, float],
        total_computing_demand: float,
        path: Sequence[str],
        node_idx: int,
    ) -> float:
        """根据是否已经经过计算节点，返回原始大小或计算后大小。"""
        if not compute_shares or size_after_computing is None or total_computing_demand <= 0:
            return float(packet_size)
        has_computed = any(float(compute_shares.get(node, 0.0)) > 0.0 for node in path[:node_idx])
        return float(size_after_computing) if has_computed else float(packet_size)

    @staticmethod
    def _timed_graphs(graphs: Sequence[nx.Graph]) -> List[Tuple[float, nx.Graph]]:
        """把图列表整理成按 sim_time 升序排列的 (时间, 图) 列表。"""
        timed = []
        for idx, graph in enumerate(graphs):
            if not isinstance(graph, nx.Graph):
                continue
            sim_time = graph.graph.get("sim_time", idx)
            timed.append((float(sim_time), graph))
        timed.sort(key=lambda item: item[0])
        return timed

    @staticmethod
    def _snapshot_to_graph(snapshot: GlobalNetworkSnapshot, sim_time: Optional[float] = None) -> nx.Graph:
        """把当前快照转换为 NetworkX 图，供时间序列拓扑检查使用。"""
        graph = nx.DiGraph()
        graph.add_nodes_from(snapshot.node_names)
        edge_mask = (
            np.asarray(snapshot.link_mask, dtype=np.float32).reshape(-1)
            if snapshot.link_mask is not None
            else np.ones(len(snapshot.edge_names), dtype=np.float32)
        )
        for edge_idx, (src, dst) in enumerate(snapshot.edge_names):
            if edge_idx < edge_mask.shape[0] and edge_mask[edge_idx] <= 0:
                continue
            graph.add_edge(src, dst)
            if snapshot.propagation_delays:
                delay = snapshot.propagation_delays.get(
                    (src, dst),
                    snapshot.propagation_delays.get((dst, src)),
                )
                if delay is not None:
                    graph.edges[src, dst]["propagation_delay"] = float(delay)
            if edge_idx < snapshot.link_load.shape[0]:
                graph.edges[src, dst]["predicted_link_load"] = float(snapshot.link_load[edge_idx, 0])
        for node_idx, node in enumerate(snapshot.node_names):
            if node_idx < snapshot.queue_load.shape[0]:
                graph.nodes[node]["predicted_queue_load"] = float(snapshot.queue_load[node_idx, 0])
            if node_idx < snapshot.compute_queue.shape[0]:
                graph.nodes[node]["predicted_compute_queue"] = float(snapshot.compute_queue[node_idx, 0])
            if snapshot.business_time is not None and node_idx < snapshot.business_time.shape[0]:
                graph.nodes[node]["predicted_business_time"] = float(snapshot.business_time[node_idx, 0])
        if sim_time is not None:
            graph.graph["sim_time"] = float(sim_time)
        elif snapshot.sim_time is not None:
            graph.graph["sim_time"] = float(snapshot.sim_time)
        return graph

    def _latest_sim_time_for_forecast(self, timed_graphs: Optional[Sequence[Tuple[float, nx.Graph]]] = None) -> float:
        """返回预测窗口起点对应的当前仿真时间。"""
        if self.latest_snapshot and self.latest_snapshot.sim_time is not None:
            return float(self.latest_snapshot.sim_time)
        if timed_graphs:
            if len(timed_graphs) >= 2:
                step = max(1e-9, float(timed_graphs[1][0] - timed_graphs[0][0]))
            else:
                step = 1.0
            return float(timed_graphs[0][0]) - step
        return 0.0

    def _graph_at_elapsed(
        self,
        timed_graphs: Sequence[Tuple[float, nx.Graph]],
        elapsed: float,
    ) -> Optional[nx.Graph]:
        """按包从当前时刻出发后的 elapsed，选择第一个不早于到达时刻的未来图。"""
        if not timed_graphs:
            return None
        target_time = self._latest_sim_time_for_forecast(timed_graphs) + max(0.0, float(elapsed))
        for sim_time, graph in timed_graphs:
            if sim_time >= target_time - 1e-9:
                return graph
        return timed_graphs[-1][1]

    def _prediction_value_at_elapsed(
        self,
        preds: Dict[str, th.Tensor],
        key: str,
        item_idx: int,
        elapsed: float,
    ) -> Optional[float]:
        """没有未来图时，根据 elapsed 从预测张量中取对应 horizon 的值。"""
        if key not in preds:
            return None
        tensor = preds[key]
        if tensor.dim() < 4 or tensor.size(1) <= 0 or item_idx < 0 or item_idx >= tensor.size(2):
            return None
        step = 1.0
        if len(self.history) >= 2:
            intervals = [
                later.sim_time - earlier.sim_time
                for earlier, later in zip(self.history[:-1], self.history[1:])
                if earlier.sim_time is not None and later.sim_time is not None and later.sim_time > earlier.sim_time
            ]
            if intervals:
                step = max(1e-9, float(np.median(intervals)))
        horizon_idx = min(max(int(math.ceil(max(0.0, float(elapsed)) / step)) - 1, 0), tensor.size(1) - 1)
        return float(tensor[0, horizon_idx, item_idx, 0].detach().cpu())

    def _edge_delay(self, src: str, dst: str) -> float:
        """从最新快照传播时延表中读取边时延，缺失时用 1.0。"""
        if self.latest_snapshot is None or not self.latest_snapshot.propagation_delays:
            return 1.0
        return float(
            self.latest_snapshot.propagation_delays.get(
                (src, dst),
                self.latest_snapshot.propagation_delays.get((dst, src), 1.0),
            )
        )

    def _packet_capacity_risk(
        self,
        path: List[str],
        packet_size: float,
        computing_demand: float,
        compute_node: Optional[str],
        size_after_computing: Optional[float] = None,
        graphs: Optional[Dict[float, nx.Graph]] = None,
        preds: Optional[Dict[str, th.Tensor]] = None,
    ) -> float:
        """按包到达未来节点/链路的时间片，估算新增业务造成的容量风险。"""
        if self.latest_snapshot is None:
            return 0.0
        risks = []
        timed_graphs = self._timed_graphs(list(graphs.values())) if graphs else []
        compute_idx = path.index(compute_node) if compute_node in path else len(path)
        output_size = packet_size if size_after_computing is None else float(size_after_computing)
        elapsed = 0.0

        def node_queue_load(node: str, graph: Optional[nx.Graph], at_elapsed: float) -> float:
            if graph is not None and node in graph and "predicted_queue_load" in graph.nodes[node]:
                return max(0.0, float(graph.nodes[node]["predicted_queue_load"]))
            if preds is not None and node in self.node_names:
                value = self._prediction_value_at_elapsed(preds, "queue_forecast", self.node_names.index(node), at_elapsed)
                if value is not None:
                    return max(0.0, value)
            if node in self.node_names:
                return max(0.0, float(self.latest_snapshot.queue_load[self.node_names.index(node), 0]))
            return 0.0

        def node_compute_load(node: str, graph: Optional[nx.Graph], at_elapsed: float) -> float:
            if graph is not None and node in graph and "predicted_compute_queue" in graph.nodes[node]:
                return max(0.0, float(graph.nodes[node]["predicted_compute_queue"]))
            if preds is not None and node in self.node_names:
                value = self._prediction_value_at_elapsed(preds, "compute_queue_forecast", self.node_names.index(node), at_elapsed)
                if value is not None:
                    return max(0.0, value)
            if node in self.node_names:
                return max(0.0, float(self.latest_snapshot.compute_queue[self.node_names.index(node), 0]))
            return 0.0

        def edge_link_load(src: str, dst: str, graph: Optional[nx.Graph], at_elapsed: float) -> Tuple[float, Optional[Dict], float]:
            edge_data = self._graph_edge_data(graph, src, dst) if graph is not None else None
            wait_time = 0.0
            if edge_data is None and timed_graphs:
                _, edge_data, wait_time = self._edge_graph_at_or_after(timed_graphs, src, dst, at_elapsed)
            if edge_data is not None and "predicted_link_load" in edge_data:
                return max(0.0, float(edge_data["predicted_link_load"])), edge_data, wait_time
            if edge_data is not None:
                return max(0.0, self._edge_load_fallback(src, dst)), edge_data, wait_time
            if preds is not None:
                edge_idx = self._edge_index(src, dst)
                if edge_idx is not None:
                    value = self._prediction_value_at_elapsed(preds, "link_forecast", edge_idx, at_elapsed)
                    if value is not None:
                        return max(0.0, value), edge_data, wait_time
            edge_idx = self._edge_index(src, dst)
            if edge_idx is not None:
                return max(0.0, float(self.latest_snapshot.link_load[edge_idx, 0])), edge_data, wait_time
            return 1.0, edge_data, wait_time

        for node_idx, node in enumerate(path):
            if packet_size > 0 and self.latest_snapshot.memory_capacity:
                receive_size = output_size if node_idx > compute_idx else packet_size
                capacity = self.latest_snapshot.memory_capacity.get(node)
                if capacity:
                    graph = self._graph_at_elapsed(timed_graphs, elapsed) if timed_graphs else None
                    load = node_queue_load(node, graph, elapsed)
                    risks.append(max(0.0, load + receive_size / float(capacity) - self.queue_threshold))

            if node_idx >= len(path) - 1:
                continue
            if node == compute_node and computing_demand > 0:
                compute_capacity = (self.latest_snapshot.computing_capacity or {}).get(node)
                if compute_capacity:
                    compute_load = node_compute_load(node, self._graph_at_elapsed(timed_graphs, elapsed) if timed_graphs else None, elapsed)
                    compute_share = float(computing_demand) / float(compute_capacity)
                    risks.append(max(0.0, compute_load + compute_share - self.compute_threshold))
                    elapsed += max(0.0, compute_load + compute_share)

            src, dst = node, path[node_idx + 1]
            graph = self._graph_at_elapsed(timed_graphs, elapsed) if timed_graphs else None
            link_load, edge_data, wait_time = edge_link_load(src, dst, graph, elapsed)
            if timed_graphs and edge_data is None:
                risks.append(1.0)
                elapsed += self._edge_delay(src, dst)
                continue
            elapsed += wait_time
            if packet_size > 0 and self.latest_snapshot.memory_capacity:
                src_capacity = self.latest_snapshot.memory_capacity.get(src)
                transmit_size = output_size if node_idx >= compute_idx else packet_size
                if src_capacity:
                    risks.append(max(0.0, link_load + transmit_size / float(src_capacity) - self.link_threshold))
            elapsed += self._edge_delay_from_graph(edge_data, default=self._edge_delay(src, dst)) if edge_data is not None else self._edge_delay(src, dst)

        if (
            compute_node
            and compute_idx >= len(path) - 1
            and computing_demand > 0
            and self.latest_snapshot.computing_capacity
        ):
            capacity = self.latest_snapshot.computing_capacity.get(compute_node)
            if capacity:
                graph = self._graph_at_elapsed(timed_graphs, elapsed) if timed_graphs else None
                load = node_compute_load(compute_node, graph, elapsed)
                risks.append(max(0.0, load + computing_demand / float(capacity) - self.compute_threshold))
        return max(risks) if risks else 0.0

    def _path_delay(self, path: List[str]) -> float:
        """计算路径纯传播时延；没有传播时延表时用跳数近似。"""
        if self.latest_snapshot is None or not self.latest_snapshot.propagation_delays:
            return float(max(0, len(path) - 1))
        total = 0.0
        for edge in zip(path[:-1], path[1:]):
            total += float(self.latest_snapshot.propagation_delays.get(edge, self.latest_snapshot.propagation_delays.get((edge[1], edge[0]), 1.0)))
        return total

    def _edge_load_fallback(self, src: str, dst: str) -> float:
        """未来新链路没有预测负载时，使用当前快照可用负载，否则按空闲链路处理。"""
        if self.latest_snapshot is None:
            return 0.0
        edge_idx = self._edge_index(src, dst)
        if edge_idx is not None and edge_idx < self.latest_snapshot.link_load.shape[0]:
            return float(self.latest_snapshot.link_load[edge_idx, 0])
        return 0.0

    def _edge_index(self, src: str, dst: str) -> Optional[int]:
        """在固定 edge_names 中查找边索引，兼容无向语义下的反向边。"""
        if (src, dst) in self.edge_names:
            return self.edge_names.index((src, dst))
        if (dst, src) in self.edge_names:
            return self.edge_names.index((dst, src))
        return None

    def _current_as_prediction(self) -> Dict[str, th.Tensor]:
        """历史不足无法预测时，把当前快照主负载复制成未来预测。"""
        snapshot = self.latest_snapshot
        if snapshot is None:
            raise ValueError("No snapshot available.")
        horizon = self.transformer.forecast_horizon
        queue = np.repeat(snapshot.queue_load[:, :1][None, None], horizon, axis=1)
        link = np.repeat(snapshot.link_load[:, :1][None, None], horizon, axis=1)
        compute = np.repeat(snapshot.compute_queue[:, :1][None, None], horizon, axis=1)
        business_time_source = (
            snapshot.business_time
            if snapshot.business_time is not None
            else np.zeros((len(snapshot.node_names), 1), dtype=np.float32)
        )
        business_time = np.repeat(business_time_source[:, :1][None, None], horizon, axis=1)
        return {
            "queue_forecast": th.as_tensor(queue, dtype=th.float32, device=self.device),
            "link_forecast": th.as_tensor(link, dtype=th.float32, device=self.device),
            "compute_queue_forecast": th.as_tensor(compute, dtype=th.float32, device=self.device),
            "business_time_forecast": th.as_tensor(business_time, dtype=th.float32, device=self.device),
        }


def snapshots_to_training_batch(
    snapshots: Sequence[GlobalNetworkSnapshot],
    history_len: int,
    forecast_horizon: int,
    node_names: Sequence[str],
    edge_names: Sequence[EdgeName],
    batch_size: int,
    device: str = "cpu",
    include_target_masks: bool = False,
    include_target_graphs: bool = False,
) -> Tuple[th.Tensor, ...]:
    """从连续快照中随机采样训练 batch，返回历史输入、未来目标和可选 mask/图。"""
    if len(snapshots) < history_len + forecast_horizon:
        raise ValueError("Not enough snapshots for one training sample.")

    aligned = [snapshot.aligned(node_names, edge_names) for snapshot in snapshots]
    max_start = len(aligned) - history_len - forecast_horizon
    starts = np.random.randint(0, max_start + 1, size=batch_size)

    values = [[], [], [], [], [], [], []]
    masks = [[], [], []]
    target_graphs = []
    for start in starts:
        history = aligned[start:start + history_len]
        future = aligned[start + history_len:start + history_len + forecast_horizon]
        values[0].append(np.stack([item.queue_load for item in history], axis=0))
        values[1].append(np.stack([item.link_load for item in history], axis=0))
        values[2].append(np.stack([item.compute_queue for item in history], axis=0))
        values[3].append(np.stack([item.queue_load[:, :1] for item in future], axis=0))
        values[4].append(np.stack([item.link_load[:, :1] for item in future], axis=0))
        values[5].append(np.stack([item.compute_queue[:, :1] for item in future], axis=0))
        values[6].append(np.stack([
            (item.business_time if item.business_time is not None else np.zeros((len(item.node_names), 1), dtype=np.float32))[:, :1]
            for item in future
        ], axis=0))
        if include_target_masks:
            masks[0].append(np.stack([item.node_mask for item in future], axis=0))
            masks[1].append(np.stack([item.link_mask for item in future], axis=0))
            masks[2].append(np.stack([item.node_mask for item in future], axis=0))
        if include_target_graphs:
            horizon_graphs = []
            for item in future:
                graph = nx.DiGraph()
                graph.add_nodes_from(item.node_names)
                edge_mask = (
                    np.asarray(item.link_mask, dtype=np.float32).reshape(-1)
                    if item.link_mask is not None
                    else np.ones(len(item.edge_names), dtype=np.float32)
                )
                for edge_idx, (src, dst) in enumerate(item.edge_names):
                    if edge_idx < edge_mask.shape[0] and edge_mask[edge_idx] <= 0:
                        continue
                    graph.add_edge(src, dst)
                    if item.propagation_delays:
                        delay = item.propagation_delays.get(
                            (src, dst),
                            item.propagation_delays.get((dst, src)),
                        )
                        if delay is not None:
                            graph.edges[src, dst]["propagation_delay"] = float(delay)
                if item.sim_time is not None:
                    graph.graph["sim_time"] = float(item.sim_time)
                horizon_graphs.append(graph)
            target_graphs.append(horizon_graphs)

    batch = tuple(
        th.as_tensor(np.stack(item, axis=0), dtype=th.float32, device=device)
        for item in values
    )
    if not include_target_masks:
        return batch + ((target_graphs,) if include_target_graphs else ())

    target_masks = tuple(
        th.as_tensor(np.stack(item, axis=0), dtype=th.float32, device=device)
        for item in masks
    )
    if include_target_graphs:
        return batch + (target_masks, target_graphs)
    return batch + (target_masks,)


def masked_mse_loss(pred: th.Tensor, target: th.Tensor, mask: Optional[th.Tensor] = None) -> th.Tensor:
    """计算可选 mask 加权的均方误差，mask 缺失时退化为普通 MSE。"""
    loss = F.mse_loss(pred, target, reduction="none")
    if mask is None:
        return loss.mean()

    mask = mask.to(dtype=loss.dtype, device=loss.device)
    weighted_loss = loss * mask
    normalizer = mask.expand_as(loss).sum().clamp(min=1.0)
    return weighted_loss.sum() / normalizer


def _pad_feature_rows(rows: Sequence[np.ndarray]) -> np.ndarray:
    """把变长图属性特征补齐成二维 float32 数组。"""
    if not rows:
        return np.zeros((0, 1), dtype=np.float32)
    width = max(1, max(int(row.shape[0]) for row in rows))
    output = np.zeros((len(rows), width), dtype=np.float32)
    for idx, row in enumerate(rows):
        row = np.asarray(row, dtype=np.float32).reshape(-1)
        output[idx, : min(width, row.shape[0])] = row[:width]
    return output


def _clip01(value: float) -> float:
    """把数值裁剪到 [0, 1] 区间并转成 float。"""
    return float(min(1.0, max(0.0, value)))
