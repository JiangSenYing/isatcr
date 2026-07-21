"""DDQN-style MEO domain routing agent."""

from collections import deque
from collections.abc import Mapping
import os
import random
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .leo_attention_predictor import LEOAttentionDecisionPredictor, batch_from_networkx
from .meo_observation import GLOBAL_FEATURE_DIM, TASK_GLOBAL_FEATURE_DIM


class _MLPQNetwork(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class _NeighborSelfAttentionQNetwork(nn.Module):
    """Permutation-equivariant Q network over fixed-width neighbor rows."""

    CONTEXT_DIM = GLOBAL_FEATURE_DIM + TASK_GLOBAL_FEATURE_DIM

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        feedforward_dim: int,
        dropout: float,
    ):
        super().__init__()
        if int(n_actions) < 1:
            raise ValueError("MEO n_actions must be at least 1")
        if num_heads < 1:
            raise ValueError("MEO attention_heads must be at least 1")
        remaining_dim = int(state_dim) - self.CONTEXT_DIM
        if remaining_dim <= 0 or remaining_dim % int(n_actions) != 0:
            raise ValueError(
                "self_attention MEO state_dim must equal 9 context features plus "
                f"an equal-width row for each of {n_actions} neighbors; got state_dim={state_dim}"
            )
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"MEO attention hidden_dim ({hidden_dim}) must be divisible by attention_heads ({num_heads})"
            )
        if num_layers < 1:
            raise ValueError("MEO attention_layers must be at least 1")
        if feedforward_dim < 1:
            raise ValueError("MEO attention_ff_dim must be at least 1")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("MEO attention_dropout must be in [0, 1)")

        self.state_dim = int(state_dim)
        self.n_actions = int(n_actions)
        self.neighbor_dim = remaining_dim // self.n_actions
        self.context_encoder = nn.Sequential(
            nn.Linear(self.CONTEXT_DIM, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.neighbor_encoder = nn.Sequential(
            nn.Linear(self.neighbor_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.attention = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        if x.ndim != 2 or x.shape[1] != self.state_dim:
            raise ValueError(
                f"Expected MEO state batch with shape [batch, {self.state_dim}], got {tuple(x.shape)}"
            )
        context_features = x[:, : self.CONTEXT_DIM]
        neighbor_rows = x[:, self.CONTEXT_DIM :].reshape(
            x.shape[0], self.n_actions, self.neighbor_dim
        )
        padding_mask = neighbor_rows.abs().sum(dim=-1).eq(0)

        context = self.context_encoder(context_features)
        expanded_context = context.unsqueeze(1).expand(-1, self.n_actions, -1)
        neighbor_tokens = self.neighbor_encoder(
            torch.cat([neighbor_rows, expanded_context], dim=-1)
        )

        # PyTorch attention cannot safely consume a row where every token is
        # padding. Temporarily expose one dummy token, then zero every padded
        # output. This is important for terminal transitions whose next state
        # is the all-zero sentinel.
        safe_padding_mask = padding_mask.clone()
        all_padding = safe_padding_mask.all(dim=1)
        if torch.any(all_padding):
            safe_padding_mask[all_padding, 0] = False
        attended = self.attention(
            neighbor_tokens,
            src_key_padding_mask=safe_padding_mask,
        )
        attended = attended.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        q_inputs = torch.cat([attended, expanded_context], dim=-1)
        return self.q_head(q_inputs).squeeze(-1)


class MEODomainRoutingAgent:
    """Small DDQN agent whose actions are MEO neighbor indices."""

    def __init__(self, state_dim: int, cfg: Optional[dict] = None, device: str = "cpu"):
        cfg = cfg or {}
        self.state_dim = int(state_dim)
        self.n_actions = int(cfg.get("n_actions", 4))
        self.device = torch.device(device)
        self.gamma = float(cfg.get("gamma", 0.97))
        self.batch_size = int(cfg.get("batch_size", 64))
        self.target_update_freq = max(1, int(cfg.get("target_update_freq", 100)))
        self.epsilon = float(cfg.get("epsilon", 0.1))
        self.min_epsilon = float(cfg.get("min_epsilon", cfg.get("epsilon_end", 0.02)))
        self.epsilon_decay = float(cfg.get("epsilon_decay", 1.0))
        self.train_steps = 0

        hidden_dim = int(cfg.get("hidden_dim", 128))
        self.encoder_type = str(cfg.get("encoder_type", "mlp")).strip().lower()
        if self.encoder_type not in {"mlp", "self_attention"}:
            raise ValueError(
                f"Unsupported MEO encoder_type {self.encoder_type!r}; expected 'mlp' or 'self_attention'"
            )
        self.network_config = {
            "encoder_type": self.encoder_type,
            "state_dim": self.state_dim,
            "n_actions": self.n_actions,
            "hidden_dim": hidden_dim,
        }
        if self.encoder_type == "self_attention":
            self.network_config.update({
                "attention_heads": int(cfg.get("attention_heads", 4)),
                "attention_layers": int(cfg.get("attention_layers", 2)),
                "attention_ff_dim": int(cfg.get("attention_ff_dim", hidden_dim * 2)),
                "attention_dropout": float(cfg.get("attention_dropout", 0.0)),
            })
        self.online_net = self._build_q_network().to(self.device)
        self.target_net = self._build_q_network().to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=float(cfg.get("lr", 1e-4)))
        self.replay_buffer = deque(maxlen=int(cfg.get("buffer_size", 100000)))
        self.last_loss = None
        self.last_leo_rollout = None
        self.last_leo_rollout_scores = None

        leo_cfg = cfg.get("leo_policy", {}) or {}
        self.leo_policy_enabled = bool(leo_cfg.get("enabled", False))
        self.leo_policy_train_enabled = bool(leo_cfg.get("train_enabled", False))
        self.leo_policy_max_steps = max(1, int(leo_cfg.get("max_steps", 16)))
        self.leo_policy_selection_enabled = bool(leo_cfg.get("selection_enabled", True))
        self.leo_loss_gate_enabled = bool(leo_cfg.get("loss_gate_enabled", False))
        self.leo_loss_gate_window_size = max(1, int(leo_cfg.get("loss_gate_window_size", 20)))
        self.leo_loss_gate_threshold = float(leo_cfg.get("loss_gate_threshold", 0.1))
        self.leo_loss_history = deque(maxlen=self.leo_loss_gate_window_size)
        self.leo_unreachable_penalty = float(leo_cfg.get("unreachable_penalty", 10.0))
        self.leo_hop_penalty = float(leo_cfg.get("hop_penalty", 0.05))
        self.leo_delay_penalty = float(leo_cfg.get("delay_penalty", 0.05))
        self.leo_compute_penalty = float(leo_cfg.get("compute_penalty", 0.0))
        self.leo_batch_size = int(leo_cfg.get("batch_size", 32))
        self.leo_update_every = max(1, int(leo_cfg.get("update_every", 1)))
        self.leo_updates_per_step = max(1, int(leo_cfg.get("updates_per_step", 1)))
        self.leo_loss_weight = float(leo_cfg.get("loss_weight", 1.0))
        self.leo_save_path = leo_cfg.get("save_path", leo_cfg.get("model_path"))
        self.leo_replay_buffer = deque(maxlen=int(leo_cfg.get("buffer_size", 50000)))
        self.leo_train_steps = 0
        self.last_leo_loss = None
        self.last_leo_losses = {}
        self.leo_policy = None
        self.leo_optimizer = None
        if self.leo_policy_enabled:
            self.leo_policy = LEOAttentionDecisionPredictor(
                hidden_dim=int(leo_cfg.get("hidden_dim", 128)),
                num_layers=int(leo_cfg.get("num_layers", 2)),
                dropout=float(leo_cfg.get("dropout", 0.1)),
            ).to(self.device)
            if self.leo_policy_train_enabled:
                self.leo_optimizer = torch.optim.Adam(
                    self.leo_policy.parameters(),
                    lr=float(leo_cfg.get("lr", 1e-4)),
                )
            self._load_leo_policy(leo_cfg.get("model_path"))

    def _build_q_network(self) -> nn.Module:
        if self.encoder_type == "mlp":
            return _MLPQNetwork(
                self.state_dim,
                self.n_actions,
                self.network_config["hidden_dim"],
            )
        return _NeighborSelfAttentionQNetwork(
            state_dim=self.state_dim,
            n_actions=self.n_actions,
            hidden_dim=self.network_config["hidden_dim"],
            num_heads=self.network_config["attention_heads"],
            num_layers=self.network_config["attention_layers"],
            feedforward_dim=self.network_config["attention_ff_dim"],
            dropout=self.network_config["attention_dropout"],
        )

    def act(
        self,
        state: Sequence[float],
        action_mask: Optional[Sequence[bool]] = None,
        explore: bool = True,
        leo_context: Optional[Mapping] = None,
    ) -> Tuple[int, float, np.ndarray]:
        self.last_leo_rollout = None
        self.last_leo_rollout_scores = None
        valid = self._valid_indices(action_mask)
        if not valid:
            return -1, float("-inf"), np.full(self.n_actions, -np.inf, dtype=np.float32)
        q_values = self.q_values(state, action_mask)
        leo_policy_ready = self.is_leo_policy_ready()
        if leo_context is not None and leo_policy_ready:
            action, rollout, rollout_scores = self._select_action_with_leo_rollouts(q_values, valid, leo_context)
            self.last_leo_rollout = rollout
            self.last_leo_rollout_scores = rollout_scores
        elif explore and random.random() < self.epsilon:
            action = random.choice(valid)
        else:
            action = int(max(valid, key=lambda idx: q_values[idx]))
        if leo_context is not None and leo_policy_ready and self.last_leo_rollout is None:
            self.last_leo_rollout = self._run_leo_rollout(action, leo_context)
        return action, float(q_values[action]), q_values

    @property
    def leo_loss_window_average(self) -> Optional[float]:
        if not self.leo_loss_history:
            return None
        return float(np.mean(self.leo_loss_history))

    def is_leo_policy_ready(self) -> bool:
        if not (
            self.leo_policy_enabled
            and self.leo_policy_selection_enabled
            and self.leo_policy is not None
        ):
            return False
        if not self.leo_loss_gate_enabled:
            return True
        if len(self.leo_loss_history) < self.leo_loss_gate_window_size:
            return False
        return self.leo_loss_window_average < self.leo_loss_gate_threshold

    def q_values(self, state: Sequence[float], action_mask: Optional[Sequence[bool]] = None) -> np.ndarray:
        state_tensor = torch.as_tensor(np.asarray(state, dtype=np.float32), dtype=torch.float32, device=self.device).view(1, -1)
        was_training = self.online_net.training
        self.online_net.eval()
        with torch.no_grad():
            q_values = self.online_net(state_tensor).squeeze(0).detach().cpu().numpy().astype(np.float32)
        if was_training:
            self.online_net.train()
        if action_mask is not None:
            mask = np.asarray(action_mask, dtype=bool).reshape(-1)
            for idx in range(min(len(mask), self.n_actions)):
                if not mask[idx]:
                    q_values[idx] = -np.inf
            if len(mask) < self.n_actions:
                q_values[len(mask):] = -np.inf
        return q_values

    def store_experience(self, state, action, reward, next_state, done, next_action_mask=None):
        if action is None or int(action) < 0:
            return None
        # Keep the established six-field replay interface, but use a mutable
        # record so delayed packet-level credit can be added after a segment
        # has already entered replay.
        experience = [
            np.asarray(state, dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            bool(done),
            None if next_action_mask is None else np.asarray(next_action_mask, dtype=np.float32),
        ]
        self.replay_buffer.append(experience)
        return experience

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return None
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones, next_masks = zip(*batch)
        states = torch.as_tensor(np.stack(states), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device).view(-1, 1)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(np.stack(next_states), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        q = self.online_net(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            next_q_online = self.online_net(next_states)
            next_q_target = self.target_net(next_states)
            if any(mask is not None for mask in next_masks):
                mask_arr = np.ones((self.batch_size, self.n_actions), dtype=np.float32)
                for row, mask in enumerate(next_masks):
                    if mask is not None:
                        mask_arr[row, :] = 0.0
                        mask_arr[row, : min(len(mask), self.n_actions)] = mask[: self.n_actions]
                mask_tensor = torch.as_tensor(mask_arr, dtype=torch.bool, device=self.device)
                next_q_online = next_q_online.masked_fill(~mask_tensor, -1e9)
            next_actions = torch.argmax(next_q_online, dim=1, keepdim=True)
            target = rewards + (1.0 - dones) * self.gamma * next_q_target.gather(1, next_actions).squeeze(1)
        loss = F.smooth_l1_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 1.0)
        self.optimizer.step()
        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
        self.last_loss = float(loss.detach().cpu().item())
        if self.leo_policy_train_enabled:
            leo_loss = self.update_leo_policy()
            if leo_loss is not None:
                self.last_leo_loss = float(leo_loss)
        return self.last_loss

    def store_leo_experience(
        self,
        graph,
        source,
        task_context,
        destination=None,
        edge_target_distances=None,
        next_hop_target=None,
        compute_node_target=None,
    ) -> bool:
        if not self.leo_policy_enabled or not self.leo_policy_train_enabled:
            return False
        if graph is None or source is None:
            return False
        if next_hop_target is None and compute_node_target is None:
            return False
        if source not in graph:
            return False
        if next_hop_target is not None and next_hop_target not in graph:
            return False
        if compute_node_target is not None and compute_node_target not in graph:
            return False
        self.leo_replay_buffer.append({
            "graph": graph.copy() if hasattr(graph, "copy") else graph,
            "source": source,
            "task_context": dict(task_context or {}),
            "destination": destination,
            "edge_target_distances": dict(edge_target_distances or {}),
            "next_hop_target": next_hop_target,
            "compute_node_target": compute_node_target,
        })
        return True

    def update_leo_policy(self):
        if not self.leo_policy_enabled or not self.leo_policy_train_enabled:
            return None
        if self.leo_policy is None or self.leo_optimizer is None:
            return None
        if len(self.leo_replay_buffer) < self.leo_batch_size:
            return None
        if self.leo_train_steps % self.leo_update_every != 0:
            self.leo_train_steps += 1
            return None

        losses = []
        for _ in range(self.leo_updates_per_step):
            samples = random.sample(self.leo_replay_buffer, self.leo_batch_size)
            batch = batch_from_networkx(
                graphs=[item["graph"] for item in samples],
                sources=[item["source"] for item in samples],
                task_contexts=[item["task_context"] for item in samples],
                edge_target_distances=[item["edge_target_distances"] for item in samples],
                device=self.device,
            )
            next_targets = [item["next_hop_target"] for item in samples]
            compute_targets = [item["compute_node_target"] for item in samples]
            next_mask = [target is not None for target in next_targets]
            compute_mask = [target is not None for target in compute_targets]
            self.leo_policy.train()
            loss, parts = self.leo_policy.training_loss(
                batch,
                next_hop_target=next_targets,
                compute_node_target=compute_targets,
                next_hop_mask=next_mask,
                compute_node_mask=compute_mask,
                next_hop_weight=self.leo_loss_weight,
                compute_weight=self.leo_loss_weight,
            )
            self.leo_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.leo_policy.parameters(), 1.0)
            self.leo_optimizer.step()
            loss_value = float(loss.detach().cpu().item())
            losses.append(loss_value)
            self.last_leo_losses = {
                key: float(value.detach().cpu().item()) if isinstance(value, torch.Tensor) else float(value)
                for key, value in parts.items()
            }
        self.leo_train_steps += 1
        self.leo_policy.eval()
        self.last_leo_loss = float(np.mean(losses)) if losses else None
        if self.last_leo_loss is not None:
            self.leo_loss_history.append(self.last_leo_loss)
        return self.last_leo_loss

    def decay_epsilon(self) -> float:
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        return self.epsilon

    def save(self, path: str) -> None:
        if not path:
            return
        import os

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "state_dim": self.state_dim,
            "encoder_type": self.encoder_type,
            "network_config": dict(self.network_config),
            "online": self.online_net.state_dict(),
            "target": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "train_steps": self.train_steps,
            "epsilon": self.epsilon,
        }, path)
        if self.leo_policy is not None and self.leo_save_path:
            os.makedirs(os.path.dirname(self.leo_save_path) or ".", exist_ok=True)
            torch.save({
                "model": self.leo_policy.state_dict(),
                "optimizer": self.leo_optimizer.state_dict() if self.leo_optimizer is not None else None,
                "train_steps": self.leo_train_steps,
                "loss_history": list(self.leo_loss_history),
            }, self.leo_save_path)

    def load(self, path: str) -> bool:
        import os

        if not path or not os.path.exists(path):
            return False
        checkpoint = torch.load(path, map_location=self.device)
        if int(checkpoint.get("state_dim", self.state_dim)) != self.state_dim:
            return False
        checkpoint_encoder_type = str(checkpoint.get("encoder_type", "mlp")).strip().lower()
        if checkpoint_encoder_type != self.encoder_type:
            print(
                f"Warning: incompatible MEO checkpoint {path!r}: encoder_type "
                f"is {checkpoint_encoder_type!r}, expected {self.encoder_type!r}."
            )
            return False
        checkpoint_network_config = checkpoint.get("network_config")
        if (
            self.encoder_type == "self_attention"
            and checkpoint_network_config != self.network_config
        ):
            print(
                f"Warning: incompatible MEO checkpoint {path!r}: attention configuration differs."
            )
            return False
        online_state = checkpoint.get("online")
        target_state = checkpoint.get("target", online_state)
        if not self._q_state_dict_is_compatible(self.online_net, online_state):
            print(f"Warning: incompatible MEO checkpoint {path!r}: online network structure differs.")
            return False
        if not self._q_state_dict_is_compatible(self.target_net, target_state):
            print(f"Warning: incompatible MEO checkpoint {path!r}: target network structure differs.")
            return False
        self.online_net.load_state_dict(online_state)
        self.target_net.load_state_dict(target_state)
        if checkpoint.get("optimizer") is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.train_steps = int(checkpoint.get("train_steps", 0))
        self.epsilon = float(checkpoint.get("epsilon", self.epsilon))
        return True

    @staticmethod
    def _q_state_dict_is_compatible(model: nn.Module, state_dict) -> bool:
        if not isinstance(state_dict, Mapping):
            return False
        current_state = model.state_dict()
        return (
            set(state_dict.keys()) == set(current_state.keys())
            and all(
                tuple(state_dict[key].shape) == tuple(current_state[key].shape)
                for key in current_state
            )
        )

    def _load_leo_policy(self, path: Optional[str]) -> bool:
        if self.leo_policy is None or not path or not os.path.exists(path):
            return False
        checkpoint = torch.load(path, map_location=self.device)
        state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint)) if isinstance(checkpoint, dict) else checkpoint
        current_state = self.leo_policy.state_dict()
        incompatible_keys = (
            not isinstance(state_dict, Mapping)
            or set(state_dict.keys()) != set(current_state.keys())
            or any(
                key not in current_state or tuple(value.shape) != tuple(current_state[key].shape)
                for key, value in state_dict.items()
            )
        )
        if incompatible_keys:
            print(f"Warning: incompatible LEO policy checkpoint {path!r}; using a freshly initialized model.")
            self.leo_policy.eval()
            return False
        try:
            self.leo_policy.load_state_dict(state_dict)
        except (RuntimeError, ValueError, KeyError) as exc:
            print(f"Warning: incompatible LEO policy checkpoint {path!r}; using a freshly initialized model: {exc}")
            self.leo_policy.eval()
            return False
        if isinstance(checkpoint, dict):
            if self.leo_optimizer is not None and checkpoint.get("optimizer") is not None:
                self.leo_optimizer.load_state_dict(checkpoint["optimizer"])
            self.leo_train_steps = int(checkpoint.get("train_steps", self.leo_train_steps))
            restored_history = checkpoint.get("loss_history", []) or []
            self.leo_loss_history.extend(float(value) for value in restored_history)
            if self.leo_loss_history:
                self.last_leo_loss = float(self.leo_loss_history[-1])
        self.leo_policy.eval()
        return True

    def _select_action_with_leo_rollouts(self, q_values, valid, context):
        best_action = int(valid[0])
        best_score = float("-inf")
        best_rollout = None
        rollout_scores = {}
        for action in valid:
            rollout = self._run_leo_rollout(int(action), context)
            score = self._leo_adjusted_score(float(q_values[action]), rollout)
            rollout_scores[int(action)] = {
                "q_value": float(q_values[action]),
                "adjusted_score": float(score),
                "reached_target": bool(rollout.get("reached_target")) if rollout else False,
                "path_len": len(rollout.get("path", [])) if rollout else 0,
                "predicted_delay": float(rollout.get("predicted_delay", 0.0)) if rollout else 0.0,
                "compute_node": rollout.get("compute_node") if rollout else None,
                "stopped_reason": rollout.get("stopped_reason") if rollout else "no_rollout",
            }
            if score > best_score:
                best_score = score
                best_action = int(action)
                best_rollout = rollout
        return best_action, best_rollout, rollout_scores

    def _leo_adjusted_score(self, q_value: float, rollout: Optional[Dict]) -> float:
        if rollout is None:
            return q_value - self.leo_unreachable_penalty
        score = float(q_value)
        if not rollout.get("reached_target", False):
            score -= self.leo_unreachable_penalty
        path_hops = max(0, len(rollout.get("path", []) or []) - 1)
        score -= self.leo_hop_penalty * float(path_hops)
        score -= self.leo_delay_penalty * float(rollout.get("predicted_delay", 0.0) or 0.0)
        if rollout.get("compute_node") is not None:
            score -= self.leo_compute_penalty
        return score

    def _run_leo_rollout(self, action: int, context: Mapping) -> Optional[Dict]:
        if not self.leo_policy_enabled or self.leo_policy is None:
            return None
        neighbors = list(context.get("neighbors", []) or [])
        if action < 0 or action >= len(neighbors):
            return None
        intra_graph = context.get("intra_graph")
        src = context.get("src")
        next_domain = neighbors[action]
        target_node = self._candidate_exit(context.get("candidate_exits"), action, next_domain)
        distance_maps = context.get("edge_target_distances", {}) or {}
        edge_target_distances = distance_maps.get(next_domain, {}) if isinstance(distance_maps, Mapping) else {}
        if intra_graph is None or src is None or target_node is None:
            return None
        if src not in intra_graph or target_node not in intra_graph:
            return None

        task = dict(context.get("task_context", {}) or {})
        task["target_node"] = target_node
        task["next_domain"] = next_domain
        current = src
        path = [current]
        compute_flags = [0]
        events = []
        compute_node = None
        predicted_delay = 0.0
        reached_target = current == target_node
        stopped_reason = "target_reached" if reached_target else "max_steps"
        visited = {(current, bool(task.get("is_computed", False)))}

        for _ in range(self.leo_policy_max_steps):
            if current == target_node:
                reached_target = True
                stopped_reason = "target_reached"
                break
            try:
                prediction = self.leo_policy.predict(
                    intra_graph,
                    source=current,
                    task_context=task,
                    device=self.device,
                    edge_target_distances=edge_target_distances,
                )
            except Exception as exc:
                stopped_reason = f"prediction_error:{type(exc).__name__}"
                break

            predicted_compute = prediction.get("compute_node")
            if not bool(task.get("is_computed", False)) and predicted_compute == current:
                compute_node = current
                compute_flags[-1] = 1
                old_task = dict(task)
                task["is_computed"] = True
                task["computing_demand"] = 0.0
                task["packet_size"] = task.get("size_after_computing", task.get("packet_size", 0.0))
                events.append({
                    "type": "compute",
                    "node": current,
                    "task_before": old_task,
                    "task_after": dict(task),
                })
                state_key = (current, True)
                if state_key in visited:
                    stopped_reason = "loop_after_compute"
                    break
                visited.add(state_key)
                continue

            next_hop = prediction.get("next_hop")
            if next_hop is None or next_hop not in intra_graph:
                stopped_reason = "missing_next_hop"
                break
            if not intra_graph.has_edge(current, next_hop):
                stopped_reason = "invalid_next_hop"
                break
            state_key = (next_hop, bool(task.get("is_computed", False)))
            events.append({
                "type": "forward",
                "from": current,
                "to": next_hop,
                "delay": self._edge_delay(intra_graph, current, next_hop),
            })
            predicted_delay += self._edge_delay(intra_graph, current, next_hop)
            current = next_hop
            path.append(current)
            compute_flags.append(0)
            if state_key in visited and current != target_node:
                stopped_reason = "loop"
                break
            visited.add(state_key)
        else:
            reached_target = current == target_node
            stopped_reason = "target_reached" if reached_target else "max_steps"

        if current == target_node:
            reached_target = True
            stopped_reason = "target_reached"
        return {
            "action": int(action),
            "next_domain": next_domain,
            "source": src,
            "target": target_node,
            "path": path,
            "compute_flags": compute_flags,
            "compute_node": compute_node,
            "predicted_delay": float(predicted_delay),
            "reached_target": bool(reached_target),
            "events": events,
            "final_task_context": dict(task),
            "stopped_reason": stopped_reason,
        }

    @staticmethod
    def _edge_delay(graph, src, dst) -> float:
        attrs = graph[src][dst] if graph is not None and graph.has_edge(src, dst) else {}
        for key in ("delay", "propagation_delay", "propagation_weight", "predicted_delay"):
            if key in attrs:
                try:
                    return float(attrs.get(key) or 0.0)
                except (TypeError, ValueError):
                    return 0.0
        return 0.0

    @staticmethod
    def _candidate_exit(candidate_exits, action: int, next_domain):
        if isinstance(candidate_exits, Mapping):
            return candidate_exits.get(next_domain, candidate_exits.get(action))
        if candidate_exits is None:
            return None
        if action < len(candidate_exits):
            return candidate_exits[action]
        return None

    def _valid_indices(self, action_mask):
        if action_mask is None:
            return list(range(self.n_actions))
        mask = np.asarray(action_mask, dtype=bool).reshape(-1)
        return [idx for idx in range(min(len(mask), self.n_actions)) if mask[idx]]
