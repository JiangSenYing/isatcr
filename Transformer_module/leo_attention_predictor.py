"""Attention-based LEO in-domain decision predictor.

This module predicts two supervised decisions from an in-domain LEO graph:
the next-hop satellite and the satellite where the task will be computed.
It intentionally depends only on PyTorch, NumPy, and NetworkX so it can be
used beside the existing Transformer module without requiring PyG.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import networkx as nx
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


NODE_FEATURE_KEYS = (
    "remaining_memory",
    "remaining_computing",
    "memory_occupancy_rate",
    "computing_occupancy_rate",
    "is_producing",
    "compute_queue",
    "business_time",
)
EDGE_FEATURE_KEYS = (
    "delay",
    "link_load",
    "link_queue",
    "weight",
    "target_compute_remain",
)
TASK_FEATURE_KEYS = (
    "task_type",
    "packet_size",
    "computing_demand",
    "size_after_computing",
    "is_computed",
    "hops",
)


@dataclass
class LEODomainGraphBatch:
    """Padded graph batch used by :class:`LEOAttentionDecisionPredictor`.

    Shapes:
    - node_features: [batch, max_nodes, node_feature_dim]
    - edge_index: [batch, max_edges, 2], where each row is [src_index, dst_index]
    - edge_features: [batch, max_edges, edge_feature_dim]
    - task_features: [batch, task_feature_dim]
    - source_index: [batch]
    - node_mask, edge_mask, neighbor_mask, compute_mask: boolean masks
    """

    node_features: th.Tensor
    edge_index: th.Tensor
    edge_features: th.Tensor
    task_features: th.Tensor
    source_index: th.Tensor
    node_mask: th.Tensor
    edge_mask: th.Tensor
    neighbor_mask: th.Tensor
    compute_mask: th.Tensor
    node_names: Optional[List[List[str]]] = None
    node_name_to_index: Optional[List[Dict[str, int]]] = None

    def to(self, device: Union[str, th.device]) -> "LEODomainGraphBatch":
        """Move tensor fields to a device while keeping metadata untouched."""
        return LEODomainGraphBatch(
            node_features=self.node_features.to(device),
            edge_index=self.edge_index.to(device),
            edge_features=self.edge_features.to(device),
            task_features=self.task_features.to(device),
            source_index=self.source_index.to(device),
            node_mask=self.node_mask.to(device),
            edge_mask=self.edge_mask.to(device),
            neighbor_mask=self.neighbor_mask.to(device),
            compute_mask=self.compute_mask.to(device),
            node_names=self.node_names,
            node_name_to_index=self.node_name_to_index,
        )


class EdgeAwareAttentionLayer(nn.Module):
    """One graph attention layer whose scores include edge and task context."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.src_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dst_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.edge_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.task_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.score = nn.Linear(hidden_dim, 1, bias=False)
        self.message = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_embeddings: th.Tensor,
        edge_index: th.Tensor,
        edge_embeddings: th.Tensor,
        task_embeddings: th.Tensor,
        node_mask: th.Tensor,
        edge_mask: th.Tensor,
    ) -> th.Tensor:
        batch_size, max_nodes, hidden_dim = node_embeddings.shape
        updated = node_embeddings.new_zeros(batch_size, max_nodes, hidden_dim)

        for batch_idx in range(batch_size):
            valid_edges = edge_mask[batch_idx]
            if not bool(valid_edges.any()):
                continue
            edges = edge_index[batch_idx, valid_edges].long()
            src_idx = edges[:, 0].clamp(min=0, max=max_nodes - 1)
            dst_idx = edges[:, 1].clamp(min=0, max=max_nodes - 1)
            src = node_embeddings[batch_idx, src_idx]
            dst = node_embeddings[batch_idx, dst_idx]
            edge = edge_embeddings[batch_idx, valid_edges]
            task = task_embeddings[batch_idx].unsqueeze(0).expand_as(src)

            score_input = th.tanh(
                self.src_proj(src)
                + self.dst_proj(dst)
                + self.edge_proj(edge)
                + self.task_proj(task)
            )
            scores = self.score(score_input).squeeze(-1)
            weights = _segment_softmax(scores, dst_idx, max_nodes)
            msg_input = th.cat([src, dst, edge, task], dim=-1)
            messages = self.message(msg_input) * weights.unsqueeze(-1)
            updated[batch_idx].index_add_(0, dst_idx, messages)

        output = self.norm(node_embeddings + self.dropout(updated))
        return output * node_mask.unsqueeze(-1).to(dtype=output.dtype)


class LEOAttentionDecisionPredictor(nn.Module):
    """Predict next-hop and compute-node decisions from an in-domain LEO graph."""

    def __init__(
        self,
        node_feature_dim: int = len(NODE_FEATURE_KEYS),
        edge_feature_dim: int = len(EDGE_FEATURE_KEYS),
        task_feature_dim: int = len(TASK_FEATURE_KEYS),
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_feature_dim = int(node_feature_dim)
        self.edge_feature_dim = int(edge_feature_dim)
        self.task_feature_dim = int(task_feature_dim)
        self.hidden_dim = int(hidden_dim)

        self.node_proj = nn.Sequential(
            nn.Linear(self.node_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.edge_proj = nn.Sequential(
            nn.Linear(self.edge_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.task_proj = nn.Sequential(
            nn.Linear(self.task_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.layers = nn.ModuleList(
            EdgeAwareAttentionLayer(hidden_dim, dropout=dropout)
            for _ in range(int(num_layers))
        )
        self.next_hop_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.compute_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch: LEODomainGraphBatch) -> Dict[str, th.Tensor]:
        """Return masked logits and node embeddings for a graph batch."""
        node_embeddings = self.node_proj(batch.node_features.float())
        edge_embeddings = self.edge_proj(batch.edge_features.float())
        task_embeddings = self.task_proj(batch.task_features.float())
        node_mask = batch.node_mask.bool()
        edge_mask = batch.edge_mask.bool()

        for layer in self.layers:
            node_embeddings = layer(
                node_embeddings=node_embeddings,
                edge_index=batch.edge_index,
                edge_embeddings=edge_embeddings,
                task_embeddings=task_embeddings,
                node_mask=node_mask,
                edge_mask=edge_mask,
            )

        source_embeddings = _gather_nodes(node_embeddings, batch.source_index.long())
        source_expand = source_embeddings.unsqueeze(1).expand_as(node_embeddings)
        task_expand = task_embeddings.unsqueeze(1).expand_as(node_embeddings)

        next_hop_logits = self.next_hop_head(
            th.cat([node_embeddings, source_expand, task_expand], dim=-1)
        ).squeeze(-1)
        compute_node_logits = self.compute_head(
            th.cat([node_embeddings, task_expand], dim=-1)
        ).squeeze(-1)

        next_hop_logits = next_hop_logits.masked_fill(~batch.neighbor_mask.bool(), -1e9)
        compute_node_logits = compute_node_logits.masked_fill(~batch.compute_mask.bool(), -1e9)
        return {
            "next_hop_logits": next_hop_logits,
            "compute_node_logits": compute_node_logits,
            "node_embeddings": node_embeddings,
        }

    def training_loss(
        self,
        batch: LEODomainGraphBatch,
        next_hop_target: Optional[Union[th.Tensor, Sequence[Optional[Union[int, str]]]]] = None,
        compute_node_target: Optional[Union[th.Tensor, Sequence[Optional[Union[int, str]]]]] = None,
        next_hop_mask: Optional[Union[th.Tensor, Sequence[bool]]] = None,
        compute_node_mask: Optional[Union[th.Tensor, Sequence[bool]]] = None,
        next_hop_weight: float = 1.0,
        compute_weight: float = 1.0,
        ignore_index: int = -100,
    ) -> Tuple[th.Tensor, Dict[str, th.Tensor]]:
        """Compute masked supervised losses for next-hop and compute decisions."""
        outputs = self.forward(batch)
        zero = outputs["next_hop_logits"].sum() * 0.0
        next_loss = zero
        compute_loss = zero
        if next_hop_target is not None:
            next_hop_target = _targets_to_indices(
                next_hop_target,
                batch,
                device=batch.node_features.device,
                ignore_index=ignore_index,
            )
            if next_hop_mask is not None:
                mask = _target_mask(next_hop_mask, batch.node_features.device)
                next_hop_target = next_hop_target.masked_fill(~mask, int(ignore_index))
            if bool((next_hop_target != int(ignore_index)).any()):
                next_loss = F.cross_entropy(
                    outputs["next_hop_logits"],
                    next_hop_target.long(),
                    ignore_index=int(ignore_index),
                )
        if compute_node_target is not None:
            compute_node_target = _targets_to_indices(
                compute_node_target,
                batch,
                device=batch.node_features.device,
                ignore_index=ignore_index,
            )
            if compute_node_mask is not None:
                mask = _target_mask(compute_node_mask, batch.node_features.device)
                compute_node_target = compute_node_target.masked_fill(~mask, int(ignore_index))
            if bool((compute_node_target != int(ignore_index)).any()):
                compute_loss = F.cross_entropy(
                    outputs["compute_node_logits"],
                    compute_node_target.long(),
                    ignore_index=int(ignore_index),
                )
        loss = float(next_hop_weight) * next_loss + float(compute_weight) * compute_loss
        return loss, {
            "loss": loss,
            "next_hop_loss": next_loss,
            "compute_node_loss": compute_loss,
        }

    @th.no_grad()
    def predict(
        self,
        graph_or_batch: Union[nx.Graph, LEODomainGraphBatch],
        source: Optional[Union[str, int]] = None,
        task_context: Optional[Union[Mapping[str, float], Sequence[float], np.ndarray, th.Tensor]] = None,
        device: Optional[Union[str, th.device]] = None,
        **batch_kwargs,
    ) -> Dict[str, object]:
        """Predict node names and probabilities from a graph or prepared batch."""
        was_training = self.training
        self.eval()
        if isinstance(graph_or_batch, LEODomainGraphBatch):
            batch = graph_or_batch
        else:
            if source is None:
                raise ValueError("source is required when predicting from a NetworkX graph.")
            batch = batch_from_networkx([graph_or_batch], sources=[source], task_contexts=[task_context], **batch_kwargs)
        if device is None:
            device = next(self.parameters()).device
        batch = batch.to(device)
        outputs = self.forward(batch)
        next_probs = F.softmax(outputs["next_hop_logits"], dim=-1)
        compute_probs = F.softmax(outputs["compute_node_logits"], dim=-1)
        next_indices = next_probs.argmax(dim=-1).detach().cpu().tolist()
        compute_indices = compute_probs.argmax(dim=-1).detach().cpu().tolist()

        result = {
            "next_hop_index": next_indices[0] if len(next_indices) == 1 else next_indices,
            "compute_node_index": compute_indices[0] if len(compute_indices) == 1 else compute_indices,
            "next_hop_probabilities": next_probs.detach().cpu(),
            "compute_node_probabilities": compute_probs.detach().cpu(),
            "next_hop_logits": outputs["next_hop_logits"].detach().cpu(),
            "compute_node_logits": outputs["compute_node_logits"].detach().cpu(),
        }
        if batch.node_names is not None:
            next_names = [
                batch.node_names[i][idx] if idx < len(batch.node_names[i]) else None
                for i, idx in enumerate(next_indices)
            ]
            compute_names = [
                batch.node_names[i][idx] if idx < len(batch.node_names[i]) else None
                for i, idx in enumerate(compute_indices)
            ]
            result["next_hop"] = next_names[0] if len(next_names) == 1 else next_names
            result["compute_node"] = compute_names[0] if len(compute_names) == 1 else compute_names
        if was_training:
            self.train()
        return result


def from_networkx(
    graph: nx.Graph,
    source: Union[str, int],
    task_context: Optional[Union[Mapping[str, float], Sequence[float], np.ndarray, th.Tensor]] = None,
    node_names: Optional[Sequence[str]] = None,
    node_feature_scales: Optional[Mapping[str, float]] = None,
    edge_feature_scales: Optional[Mapping[str, float]] = None,
    task_feature_scales: Optional[Mapping[str, float]] = None,
    node_feature_keys: Sequence[str] = NODE_FEATURE_KEYS,
    edge_feature_keys: Sequence[str] = EDGE_FEATURE_KEYS,
    task_feature_keys: Sequence[str] = TASK_FEATURE_KEYS,
    device: Optional[Union[str, th.device]] = None,
) -> LEODomainGraphBatch:
    """Convert one NetworkX in-domain graph to a batch of size 1."""
    return batch_from_networkx(
        [graph],
        sources=[source],
        task_contexts=[task_context],
        node_names=[node_names] if node_names is not None else None,
        node_feature_scales=node_feature_scales,
        edge_feature_scales=edge_feature_scales,
        task_feature_scales=task_feature_scales,
        node_feature_keys=node_feature_keys,
        edge_feature_keys=edge_feature_keys,
        task_feature_keys=task_feature_keys,
        device=device,
    )


def batch_from_networkx(
    graphs: Sequence[nx.Graph],
    sources: Sequence[Union[str, int]],
    task_contexts: Optional[Sequence[Optional[Union[Mapping[str, float], Sequence[float], np.ndarray, th.Tensor]]]] = None,
    node_names: Optional[Sequence[Optional[Sequence[str]]]] = None,
    node_feature_scales: Optional[Mapping[str, float]] = None,
    edge_feature_scales: Optional[Mapping[str, float]] = None,
    task_feature_scales: Optional[Mapping[str, float]] = None,
    node_feature_keys: Sequence[str] = NODE_FEATURE_KEYS,
    edge_feature_keys: Sequence[str] = EDGE_FEATURE_KEYS,
    task_feature_keys: Sequence[str] = TASK_FEATURE_KEYS,
    device: Optional[Union[str, th.device]] = None,
) -> LEODomainGraphBatch:
    """Convert NetworkX graphs into one padded tensor batch."""
    if len(graphs) != len(sources):
        raise ValueError("graphs and sources must have the same length.")
    task_contexts = task_contexts or [None] * len(graphs)
    if len(task_contexts) != len(graphs):
        raise ValueError("task_contexts must match graphs length when provided.")

    graph_items = []
    max_nodes = 1
    max_edges = 1
    names_per_graph = []
    maps_per_graph = []
    for graph_idx, graph in enumerate(graphs):
        names = _resolve_node_names(graph, None if node_names is None else node_names[graph_idx])
        name_to_idx = {name: idx for idx, name in enumerate(names)}
        source_idx = _resolve_node_index(sources[graph_idx], name_to_idx, len(names))
        node_rows = [
            _node_feature_row(graph.nodes[name] if name in graph else {}, node_feature_keys, node_feature_scales)
            for name in names
        ]
        edges, edge_rows = [], []
        for src, dst, attrs in graph.edges(data=True):
            if src not in name_to_idx or dst not in name_to_idx:
                continue
            edges.append((name_to_idx[src], name_to_idx[dst]))
            edge_rows.append(_edge_feature_row(attrs, edge_feature_keys, edge_feature_scales))
            if not graph.is_directed():
                edges.append((name_to_idx[dst], name_to_idx[src]))
                edge_rows.append(_edge_feature_row(attrs, edge_feature_keys, edge_feature_scales))
        neighbor_mask = np.zeros(len(names), dtype=np.bool_)
        if names[source_idx] in graph:
            for neighbor in graph.neighbors(names[source_idx]):
                if neighbor in name_to_idx:
                    neighbor_mask[name_to_idx[neighbor]] = True
        compute_mask = np.asarray([
            bool(graph.nodes[name].get("compute_enabled", True)) if name in graph else False
            for name in names
        ], dtype=np.bool_)
        compute_mask &= np.asarray([
            float(graph.nodes[name].get("remaining_computing", 1.0) or 0.0) > 0.0 if name in graph else False
            for name in names
        ], dtype=np.bool_)
        if not compute_mask.any():
            compute_mask = np.ones(len(names), dtype=np.bool_)
        if not neighbor_mask.any():
            neighbor_mask[source_idx] = True

        item = {
            "node_features": np.asarray(node_rows, dtype=np.float32),
            "edge_index": np.asarray(edges, dtype=np.int64) if edges else np.zeros((0, 2), dtype=np.int64),
            "edge_features": np.asarray(edge_rows, dtype=np.float32) if edge_rows else np.zeros((0, len(edge_feature_keys)), dtype=np.float32),
            "task_features": _task_feature_row(task_contexts[graph_idx], task_feature_keys, task_feature_scales),
            "source_index": source_idx,
            "neighbor_mask": neighbor_mask,
            "compute_mask": compute_mask,
        }
        graph_items.append(item)
        names_per_graph.append(names)
        maps_per_graph.append(name_to_idx)
        max_nodes = max(max_nodes, len(names))
        max_edges = max(max_edges, len(item["edge_index"]))

    batch_size = len(graph_items)
    node_features = np.zeros((batch_size, max_nodes, len(node_feature_keys)), dtype=np.float32)
    edge_index = np.zeros((batch_size, max_edges, 2), dtype=np.int64)
    edge_features = np.zeros((batch_size, max_edges, len(edge_feature_keys)), dtype=np.float32)
    task_features = np.zeros((batch_size, len(task_feature_keys)), dtype=np.float32)
    source_index = np.zeros(batch_size, dtype=np.int64)
    node_mask = np.zeros((batch_size, max_nodes), dtype=np.bool_)
    edge_mask = np.zeros((batch_size, max_edges), dtype=np.bool_)
    neighbor_mask = np.zeros((batch_size, max_nodes), dtype=np.bool_)
    compute_mask = np.zeros((batch_size, max_nodes), dtype=np.bool_)

    for batch_idx, item in enumerate(graph_items):
        n_nodes = item["node_features"].shape[0]
        n_edges = item["edge_index"].shape[0]
        node_features[batch_idx, :n_nodes] = item["node_features"]
        task_features[batch_idx] = item["task_features"]
        source_index[batch_idx] = item["source_index"]
        node_mask[batch_idx, :n_nodes] = True
        neighbor_mask[batch_idx, :n_nodes] = item["neighbor_mask"]
        compute_mask[batch_idx, :n_nodes] = item["compute_mask"]
        if n_edges:
            edge_index[batch_idx, :n_edges] = item["edge_index"]
            edge_features[batch_idx, :n_edges] = item["edge_features"]
            edge_mask[batch_idx, :n_edges] = True

    batch = LEODomainGraphBatch(
        node_features=th.as_tensor(node_features, dtype=th.float32, device=device),
        edge_index=th.as_tensor(edge_index, dtype=th.long, device=device),
        edge_features=th.as_tensor(edge_features, dtype=th.float32, device=device),
        task_features=th.as_tensor(task_features, dtype=th.float32, device=device),
        source_index=th.as_tensor(source_index, dtype=th.long, device=device),
        node_mask=th.as_tensor(node_mask, dtype=th.bool, device=device),
        edge_mask=th.as_tensor(edge_mask, dtype=th.bool, device=device),
        neighbor_mask=th.as_tensor(neighbor_mask, dtype=th.bool, device=device),
        compute_mask=th.as_tensor(compute_mask, dtype=th.bool, device=device),
        node_names=names_per_graph,
        node_name_to_index=maps_per_graph,
    )
    return batch


def _segment_softmax(scores: th.Tensor, dst_idx: th.Tensor, num_nodes: int) -> th.Tensor:
    weights = th.zeros_like(scores)
    for node_idx in dst_idx.unique():
        mask = dst_idx == node_idx
        weights[mask] = F.softmax(scores[mask], dim=0)
    return weights


def _gather_nodes(node_embeddings: th.Tensor, indices: th.Tensor) -> th.Tensor:
    batch_indices = th.arange(node_embeddings.size(0), device=node_embeddings.device)
    return node_embeddings[batch_indices, indices.clamp(min=0, max=node_embeddings.size(1) - 1)]


def _resolve_node_names(graph: nx.Graph, node_names: Optional[Sequence[str]]) -> List[str]:
    if node_names is not None:
        return list(node_names)
    graph_names = graph.graph.get("node_names")
    if graph_names is not None:
        return list(graph_names)
    return sorted(graph.nodes())


def _resolve_node_index(value: Union[str, int], name_to_idx: Mapping[str, int], node_count: int) -> int:
    if isinstance(value, str):
        if value not in name_to_idx:
            raise ValueError(f"Unknown source node: {value}")
        return int(name_to_idx[value])
    index = int(value)
    if index < 0 or index >= node_count:
        raise ValueError(f"source index {index} is outside node range [0, {node_count}).")
    return index


def _node_feature_row(
    attrs: Mapping[str, object],
    feature_keys: Sequence[str],
    scales: Optional[Mapping[str, float]],
) -> np.ndarray:
    return np.asarray([
        _scaled_value(_attr_with_aliases(attrs, key, _node_aliases(key)), key, scales)
        for key in feature_keys
    ], dtype=np.float32)


def _edge_feature_row(
    attrs: Mapping[str, object],
    feature_keys: Sequence[str],
    scales: Optional[Mapping[str, float]],
) -> np.ndarray:
    return np.asarray([
        _scaled_value(_attr_with_aliases(attrs, key, _edge_aliases(key)), key, scales)
        for key in feature_keys
    ], dtype=np.float32)


def _task_feature_row(
    task_context: Optional[Union[Mapping[str, float], Sequence[float], np.ndarray, th.Tensor]],
    feature_keys: Sequence[str],
    scales: Optional[Mapping[str, float]],
) -> np.ndarray:
    if task_context is None:
        return np.zeros(len(feature_keys), dtype=np.float32)
    if isinstance(task_context, th.Tensor):
        values = task_context.detach().cpu().float().reshape(-1).numpy()
        return _pad_or_trim(values, len(feature_keys))
    if isinstance(task_context, np.ndarray) or not isinstance(task_context, Mapping):
        values = np.asarray(task_context, dtype=np.float32).reshape(-1)
        return _pad_or_trim(values, len(feature_keys))
    return np.asarray([
        _scaled_value(task_context.get(key, 0.0), key, scales)
        for key in feature_keys
    ], dtype=np.float32)


def _pad_or_trim(values: np.ndarray, size: int) -> np.ndarray:
    output = np.zeros(size, dtype=np.float32)
    n = min(size, values.shape[0])
    if n:
        output[:n] = values[:n]
    return output


def _attr_with_aliases(attrs: Mapping[str, object], key: str, aliases: Iterable[str]) -> float:
    for candidate in (key, *aliases):
        if candidate in attrs:
            return _float(attrs.get(candidate, 0.0))
    return 0.0


def _node_aliases(key: str) -> Tuple[str, ...]:
    aliases = {
        "compute_queue": ("current_computing_queue_size", "computing_queue", "predicted_compute_queue"),
        "business_time": ("task_state", "predicted_business_time"),
        "remaining_memory": ("memory_remain",),
        "remaining_computing": ("computing_remain",),
    }
    return aliases.get(key, ())


def _edge_aliases(key: str) -> Tuple[str, ...]:
    aliases = {
        "delay": ("propagation_delay",),
        "link_queue": ("queue_occupancy", "link_queue_occupancy"),
        "link_load": ("predicted_link_load", "transmission_size"),
    }
    return aliases.get(key, ())


def _scaled_value(value: object, key: str, scales: Optional[Mapping[str, float]]) -> float:
    value = _float(value)
    scale = float((scales or {}).get(key, 1.0) or 1.0)
    if scale != 1.0:
        value = value / scale
    return float(value)


def _float(value: object) -> float:
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        return float(arr[0]) if arr.size else 0.0
    if isinstance(value, th.Tensor):
        arr = value.detach().cpu().float().reshape(-1)
        return float(arr[0].item()) if arr.numel() else 0.0
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _targets_to_indices(
    targets: Union[th.Tensor, Sequence[Optional[Union[int, str]]]],
    batch: LEODomainGraphBatch,
    device: th.device,
    ignore_index: int = -100,
) -> th.Tensor:
    if isinstance(targets, th.Tensor):
        return targets.to(device=device, dtype=th.long).reshape(-1)
    resolved = []
    for batch_idx, target in enumerate(targets):
        if target is None:
            resolved.append(int(ignore_index))
        elif isinstance(target, str):
            if batch.node_name_to_index is None:
                raise ValueError("String targets require batch.node_name_to_index metadata.")
            if target not in batch.node_name_to_index[batch_idx]:
                raise ValueError(f"Unknown target node: {target}")
            resolved.append(batch.node_name_to_index[batch_idx][target])
        else:
            resolved.append(int(target))
    return th.as_tensor(resolved, dtype=th.long, device=device)


def _target_mask(mask: Union[th.Tensor, Sequence[bool]], device: th.device) -> th.Tensor:
    if isinstance(mask, th.Tensor):
        return mask.to(device=device, dtype=th.bool).reshape(-1)
    return th.as_tensor(np.asarray(mask, dtype=np.bool_).reshape(-1), dtype=th.bool, device=device)
