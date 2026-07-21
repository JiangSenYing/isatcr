"""Linear-complexity global state and joint-action critic."""

from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F

from .types import CriticOutput


def _masked_pool(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Concatenate masked mean and max without producing NaNs for empty rows."""
    if values.shape[1] == 0:
        return values.new_zeros((values.shape[0], values.shape[-1] * 2))
    mask = mask.bool()
    weights = mask.unsqueeze(-1).to(values.dtype)
    count = weights.sum(dim=1).clamp(min=1.0)
    mean = (values * weights).sum(dim=1) / count
    masked = values.masked_fill(~mask.unsqueeze(-1), torch.finfo(values.dtype).min)
    maximum = masked.max(dim=1).values
    maximum = torch.where(mask.any(dim=1, keepdim=True), maximum, torch.zeros_like(maximum))
    return torch.cat([mean, maximum], dim=-1)


class GlobalActionCritic(nn.Module):
    """Predict global load deltas and packet outcome for an executed joint action.

    The model performs only per-node/per-edge MLP operations and masked pooling,
    so compute and memory grow linearly with constellation size.
    """

    ACTION_NODE_COUNT = 5

    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        hidden_dim: int = 128,
        task_dim: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_feature_dim = int(node_feature_dim)
        self.edge_feature_dim = int(edge_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.task_dim = int(task_dim)

        self.node_encoder = self._mlp(self.node_feature_dim, hidden_dim, hidden_dim, dropout)
        self.edge_feature_encoder = self._mlp(self.edge_feature_dim, hidden_dim, hidden_dim, dropout)
        self.edge_encoder = self._mlp(hidden_dim * 3, hidden_dim, hidden_dim, dropout)
        self.meo_encoder = self._mlp(4, hidden_dim, hidden_dim, dropout)
        self.task_encoder = self._mlp(task_dim, hidden_dim, hidden_dim, dropout)
        self.action_type_embedding = nn.Embedding(2, hidden_dim)

        fusion_dim = hidden_dim * (4 + self.ACTION_NODE_COUNT + 3)
        self.fusion = self._mlp(fusion_dim, hidden_dim * 2, hidden_dim, dropout)
        local_head_dim = hidden_dim * 2
        self.queue_head = self._mlp(local_head_dim, hidden_dim, 1, dropout)
        self.compute_head = self._mlp(local_head_dim, hidden_dim, 1, dropout)
        self.link_head = self._mlp(local_head_dim, hidden_dim, 1, dropout)
        self.success_head = self._mlp(hidden_dim, hidden_dim, 1, dropout)
        self.delay_head = self._mlp(hidden_dim, hidden_dim, 1, dropout)
        self.impact_head = self._mlp(hidden_dim, hidden_dim, 1, dropout)

    @staticmethod
    def _mlp(input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> CriticOutput:
        node_features = batch["node_features"]
        edge_features = batch["edge_features"]
        node_mask = batch["node_mask"].bool()
        edge_mask = batch["edge_mask"].bool()
        edge_index = batch["edge_index"].long()

        if node_features.ndim != 3 or edge_features.ndim != 3:
            raise ValueError("node_features and edge_features must be rank-3 batched tensors")
        node_embeddings = self.node_encoder(node_features)
        raw_edge_embeddings = self.edge_feature_encoder(edge_features)

        batch_size, node_count, hidden_dim = node_embeddings.shape
        safe_edges = edge_index.clamp(min=0, max=max(node_count - 1, 0))
        batch_indices = torch.arange(batch_size, device=node_features.device).view(-1, 1)
        src = node_embeddings[batch_indices, safe_edges[..., 0]]
        dst = node_embeddings[batch_indices, safe_edges[..., 1]]
        edge_embeddings = self.edge_encoder(torch.cat([raw_edge_embeddings, src, dst], dim=-1))

        node_global = _masked_pool(node_embeddings, node_mask)
        edge_global = _masked_pool(edge_embeddings, edge_mask)

        action_indices = batch["action_indices"].long().clamp(min=0, max=max(node_count - 1, 0))
        action_node_mask = batch["action_node_mask"].to(node_embeddings.dtype)
        selected = node_embeddings[
            torch.arange(batch_size, device=node_features.device).view(-1, 1),
            action_indices,
        ]
        selected = selected * action_node_mask.unsqueeze(-1)
        selected = selected.reshape(batch_size, -1)
        task = self.task_encoder(batch["task_features"])
        meo = self.meo_encoder(batch["meo_features"])
        action_type = self.action_type_embedding(batch["action_type"].long())
        context = self.fusion(torch.cat([
            node_global,
            edge_global,
            selected,
            meo,
            task,
            action_type,
        ], dim=-1))

        node_context = context.unsqueeze(1).expand(-1, node_count, -1)
        edge_context = context.unsqueeze(1).expand(-1, edge_embeddings.shape[1], -1)
        delta_queue = self.queue_head(torch.cat([node_embeddings, node_context], dim=-1)).squeeze(-1)
        delta_compute = self.compute_head(torch.cat([node_embeddings, node_context], dim=-1)).squeeze(-1)
        delta_link = self.link_head(torch.cat([edge_embeddings, edge_context], dim=-1)).squeeze(-1)

        return CriticOutput(
            delta_queue=delta_queue.masked_fill(~node_mask, 0.0),
            delta_compute=delta_compute.masked_fill(~node_mask, 0.0),
            delta_link=delta_link.masked_fill(~edge_mask, 0.0),
            success_logit=self.success_head(context).squeeze(-1),
            delay=F.softplus(self.delay_head(context).squeeze(-1)),
            impact=self.impact_head(context).squeeze(-1),
        )
