"""Replay, delayed-label alignment, optimization, and persistence for the critic."""

from collections import deque
from dataclasses import dataclass
import os
import random
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .model import GlobalActionCritic
from .types import CriticSample, JointAction


@dataclass
class _PendingEvent:
    pre_snapshot: object
    action: JointAction
    action_time: float
    post_snapshot: Optional[object] = None
    terminal_reward: Optional[float] = None
    success: Optional[float] = None
    delay: Optional[float] = None


class GlobalCriticTrainer:
    """Train an observational critic and optionally rerank gated LEO actions."""

    def __init__(self, cfg: Optional[dict] = None, device: str = "cpu"):
        cfg = dict(cfg or {})
        self.cfg = cfg
        self.enabled = bool(cfg.get("enabled", False))
        self.device = torch.device(device)
        self.hidden_dim = int(cfg.get("hidden_dim", 128))
        self.batch_size = int(cfg.get("batch_size", 64))
        self.warmup_samples = int(cfg.get("warmup_samples", 256))
        self.learning_rate = float(cfg.get("learning_rate", 1e-4))
        self.max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
        self.model_path = cfg.get("model_path")
        self.replay_buffer = deque(maxlen=int(cfg.get("buffer_size", 50000)))
        self.pending: Dict[str, _PendingEvent] = {}
        self.event_counter = 0
        self.discarded_pending = 0
        self.completed_samples = 0
        self.train_steps = 0
        self.model = None
        self.optimizer = None
        self.node_names = []
        self.edge_names = []
        self.last_losses = {}
        self.last_metrics = {}
        self._load_attempted = False
        self.latest_snapshot = None
        self.selection_enabled = bool(cfg.get("selection_enabled", False))
        self.selection_min_samples = max(0, int(cfg.get("selection_min_samples", 512)))
        self.selection_min_train_steps = max(0, int(cfg.get("selection_min_train_steps", 100)))
        self.selection_weight = max(0.0, float(cfg.get("selection_weight", 0.2)))
        self.selection_eps = max(float(cfg.get("selection_eps", 1e-6)), 1e-12)
        self.selection_uses = 0
        self.selection_disagreements = 0
        self.selection_fallbacks = 0
        self.selection_score_count = 0
        self.selection_q_sum = 0.0
        self.selection_risk_sum = 0.0
        self.selection_combined_sum = 0.0
        self._last_selection_ready = False
        self.feature_stats = {
            "node": {"count": 0, "mean": None, "m2": None},
            "edge": {"count": 0, "mean": None, "m2": None},
        }

        weights = cfg.get("impact_weights", {}) or {}
        self.queue_penalty = float(weights.get("queue", 0.1))
        self.link_penalty = float(weights.get("link", 0.1))
        self.compute_penalty = float(weights.get("compute", 0.1))
        self.imbalance_penalty = float(weights.get("imbalance", 0.1))
        loss_weights = cfg.get("loss_weights", {}) or {}
        self.state_loss_weight = float(loss_weights.get("state", 1.0))
        self.success_loss_weight = float(loss_weights.get("success", 1.0))
        self.delay_loss_weight = float(loss_weights.get("delay", 0.5))
        self.impact_loss_weight = float(loss_weights.get("impact", 1.0))

    def initialize(self, snapshot) -> None:
        if snapshot is not None:
            self.latest_snapshot = snapshot
        if not self.enabled or self.model is not None:
            return
        node_features, edge_features = self._snapshot_features(snapshot)
        self.node_names = list(snapshot.node_names)
        self.edge_names = list(snapshot.edge_names)
        self.model = GlobalActionCritic(
            node_feature_dim=node_features.shape[-1],
            edge_feature_dim=edge_features.shape[-1],
            hidden_dim=self.hidden_dim,
            dropout=float(self.cfg.get("dropout", 0.1)),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        if self.model_path and not self._load_attempted:
            self.load(self.model_path)

    def start_event(self, packet, snapshot, action: JointAction, action_time: float) -> Optional[str]:
        if not self.enabled or packet is None or snapshot is None:
            return None
        self.initialize(snapshot)
        self.event_counter += 1
        event_id = f"{action.decision_id}:leo-step-{self.event_counter}"
        self.pending[event_id] = _PendingEvent(
            pre_snapshot=snapshot,
            action=action,
            action_time=float(action_time),
        )
        event_ids = getattr(packet, "global_critic_event_ids", None)
        if event_ids is None:
            event_ids = []
            packet.global_critic_event_ids = event_ids
        event_ids.append(event_id)
        return event_id

    def observe_snapshot(self, snapshot) -> int:
        """Attach the first snapshot strictly later than each action."""
        if not self.enabled:
            return 0
        self.initialize(snapshot)
        self._update_feature_stats(snapshot)
        snapshot_time = getattr(snapshot, "sim_time", None)
        if snapshot_time is None:
            return 0
        matched = 0
        for event_id, event in list(self.pending.items()):
            if event.post_snapshot is None and float(snapshot_time) > event.action_time + 1e-9:
                event.post_snapshot = snapshot
                matched += 1
                self._finalize_if_ready(event_id)
        return matched

    def finish_packet(self, packet, terminal_reward: float, success: bool, delay: float) -> int:
        if not self.enabled or packet is None:
            return 0
        event_ids = list(getattr(packet, "global_critic_event_ids", None) or [])
        updated = 0
        for event_id in event_ids:
            event = self.pending.get(event_id)
            if event is None:
                continue
            event.terminal_reward = float(terminal_reward)
            event.success = float(bool(success))
            event.delay = max(0.0, float(delay))
            updated += 1
            self._finalize_if_ready(event_id)
        packet.global_critic_event_ids = []
        return updated

    def _finalize_if_ready(self, event_id: str) -> bool:
        event = self.pending.get(event_id)
        if event is None or event.post_snapshot is None or event.terminal_reward is None:
            return False
        impact = self.compute_impact_target(
            event.pre_snapshot,
            event.post_snapshot,
            event.terminal_reward,
        )
        self.replay_buffer.append(CriticSample(
            pre_snapshot=event.pre_snapshot,
            post_snapshot=event.post_snapshot,
            action=event.action,
            terminal_reward=float(event.terminal_reward),
            success=float(event.success),
            delay=float(event.delay),
            impact_target=float(impact),
        ))
        del self.pending[event_id]
        self.completed_samples += 1
        return True

    def compute_impact_target(self, pre_snapshot, post_snapshot, terminal_reward: float) -> float:
        pre = pre_snapshot.aligned(self.node_names, self.edge_names)
        post = post_snapshot.aligned(self.node_names, self.edge_names)
        queue_pre = np.asarray(pre.queue_load[:, 0], dtype=np.float32)
        queue_post = np.asarray(post.queue_load[:, 0], dtype=np.float32)
        compute_pre = np.asarray(pre.compute_queue[:, 0], dtype=np.float32)
        compute_post = np.asarray(post.compute_queue[:, 0], dtype=np.float32)
        link_pre = np.asarray(pre.link_load[:, 0], dtype=np.float32)
        link_post = np.asarray(post.link_load[:, 0], dtype=np.float32)

        queue_cost = self._masked_positive_mean(queue_post - queue_pre, pre.node_mask)
        compute_cost = self._masked_positive_mean(compute_post - compute_pre, pre.node_mask)
        link_cost = self._masked_positive_mean(link_post - link_pre, pre.link_mask)
        imbalance_growth = max(
            0.0,
            self._masked_std(queue_post, post.node_mask) - self._masked_std(queue_pre, pre.node_mask),
        )
        return float(
            terminal_reward
            - self.queue_penalty * queue_cost
            - self.compute_penalty * compute_cost
            - self.link_penalty * link_cost
            - self.imbalance_penalty * imbalance_growth
        )

    @staticmethod
    def _masked_positive_mean(values, mask) -> float:
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        valid = np.ones_like(values, dtype=bool) if mask is None else np.asarray(mask).reshape(-1) > 0
        return float(np.maximum(values[valid], 0.0).mean()) if np.any(valid) else 0.0

    @staticmethod
    def _masked_std(values, mask) -> float:
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        valid = np.ones_like(values, dtype=bool) if mask is None else np.asarray(mask).reshape(-1) > 0
        return float(values[valid].std()) if np.any(valid) else 0.0

    def update_if_ready(self) -> Optional[float]:
        if not self.enabled or self.model is None:
            return None
        needed = max(self.batch_size, self.warmup_samples)
        if len(self.replay_buffer) < needed:
            return None
        samples = random.sample(self.replay_buffer, self.batch_size)
        batch, targets = self._make_batch(samples)
        self.model.train()
        output = self.model(batch)
        node_mask = batch["node_mask"]
        edge_mask = batch["edge_mask"]

        queue_loss = self._masked_smooth_l1(output.delta_queue, targets["delta_queue"], node_mask)
        compute_loss = self._masked_smooth_l1(output.delta_compute, targets["delta_compute"], node_mask)
        link_loss = self._masked_smooth_l1(output.delta_link, targets["delta_link"], edge_mask)
        state_loss = queue_loss + compute_loss + link_loss
        success_loss = F.binary_cross_entropy_with_logits(output.success_logit, targets["success"])
        delay_loss = F.smooth_l1_loss(torch.log1p(output.delay), torch.log1p(targets["delay"]))
        impact_loss = F.smooth_l1_loss(output.impact, targets["impact"])
        loss = (
            self.state_loss_weight * state_loss
            + self.success_loss_weight * success_loss
            + self.delay_loss_weight * delay_loss
            + self.impact_loss_weight * impact_loss
        )
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.train_steps += 1
        self.last_losses = {
            "total": float(loss.detach().cpu()),
            "state": float(state_loss.detach().cpu()),
            "queue": float(queue_loss.detach().cpu()),
            "compute": float(compute_loss.detach().cpu()),
            "link": float(link_loss.detach().cpu()),
            "success": float(success_loss.detach().cpu()),
            "delay": float(delay_loss.detach().cpu()),
            "impact": float(impact_loss.detach().cpu()),
        }
        with torch.no_grad():
            self.last_metrics = {
                "impact_mae": float((output.impact - targets["impact"]).abs().mean().cpu()),
                "delay_mae": float((output.delay - targets["delay"]).abs().mean().cpu()),
                "success_accuracy": float(
                    ((output.success_logit >= 0) == (targets["success"] >= 0.5)).float().mean().cpu()
                ),
            }
        return self.last_losses["total"]

    def format_training_log(self, step=None, total_steps=None, round_idx=None) -> str:
        """Format one concise warmup/training status line for simulator logs."""
        if not self.enabled:
            return ""
        self.selection_ready()
        location = []
        if round_idx is not None:
            location.append(f"round={int(round_idx)}")
        if step is not None:
            step_text = str(int(step))
            if total_steps is not None:
                step_text += f"/{int(total_steps)}"
            location.append(f"step={step_text}")
        needed = max(self.batch_size, self.warmup_samples)
        phase = "training" if self.train_steps > 0 else "warmup"
        fields = [
            *location,
            f"phase={phase}",
            f"buffer={len(self.replay_buffer)}/{needed}",
            f"pending={len(self.pending)}",
            f"completed={self.completed_samples}",
            f"discarded={self.discarded_pending}",
            f"train_steps={self.train_steps}",
            f"selection_ready={self._last_selection_ready}",
            f"selection_uses={self.selection_uses}",
            f"selection_disagreements={self.selection_disagreements}",
            f"selection_fallbacks={self.selection_fallbacks}",
        ]
        if self.selection_score_count:
            count = float(self.selection_score_count)
            fields.extend([
                f"selection_q={self.selection_q_sum / count:.6f}",
                f"selection_risk={self.selection_risk_sum / count:.6f}",
                f"selection_score={self.selection_combined_sum / count:.6f}",
            ])
        if self.last_losses:
            fields.extend([
                f"loss={self.last_losses.get('total', 0.0):.6f}",
                f"state={self.last_losses.get('state', 0.0):.6f}",
                f"queue={self.last_losses.get('queue', 0.0):.6f}",
                f"compute={self.last_losses.get('compute', 0.0):.6f}",
                f"link={self.last_losses.get('link', 0.0):.6f}",
                f"success={self.last_losses.get('success', 0.0):.6f}",
                f"delay={self.last_losses.get('delay', 0.0):.6f}",
                f"impact={self.last_losses.get('impact', 0.0):.6f}",
            ])
        if self.last_metrics:
            fields.extend([
                f"impact_mae={self.last_metrics.get('impact_mae', 0.0):.6f}",
                f"delay_mae={self.last_metrics.get('delay_mae', 0.0):.6f}",
                f"success_acc={self.last_metrics.get('success_accuracy', 0.0):.4f}",
            ])
        return "[GlobalCritic] " + ", ".join(fields)

    @torch.no_grad()
    def predict(self, snapshot, action: JointAction) -> Optional[dict]:
        """Return a compact shadow evaluation for one proposed/executed action."""
        predictions = self.predict_many(snapshot, [action])
        return predictions[0] if predictions else None

    @torch.no_grad()
    def predict_many(self, snapshot, actions: Sequence[JointAction]):
        """Evaluate candidate actions in one model call while preserving their order."""
        if not self.enabled or snapshot is None or not actions:
            return []
        self.initialize(snapshot)
        placeholders = [
            CriticSample(
                pre_snapshot=snapshot,
                post_snapshot=snapshot,
                action=action,
                terminal_reward=0.0,
                success=0.0,
                delay=0.0,
                impact_target=0.0,
            )
            for action in actions
        ]
        batch, _ = self._make_batch(placeholders)
        was_training = self.model.training
        self.model.eval()
        output = self.model(batch)
        if was_training:
            self.model.train()

        def masked_mean(values, mask, index):
            valid = values[index][mask[index]]
            return float(valid.mean().cpu()) if valid.numel() else 0.0

        probabilities = torch.sigmoid(output.success_logit)
        return [
            {
                "impact": float(output.impact[index].cpu()),
                "success_probability": float(probabilities[index].cpu()),
                "delay": float(output.delay[index].cpu()),
                "mean_delta_queue": masked_mean(output.delta_queue, batch["node_mask"], index),
                "mean_delta_compute": masked_mean(output.delta_compute, batch["node_mask"], index),
                "mean_delta_link": masked_mean(output.delta_link, batch["edge_mask"], index),
                "train_steps": int(self.train_steps),
            }
            for index in range(len(actions))
        ]

    def selection_ready(self, snapshot=None) -> bool:
        """Return whether the critic may affect LEO greedy action selection."""
        snapshot = snapshot if snapshot is not None else self.latest_snapshot
        if self.enabled and snapshot is not None and self.model is None:
            self.initialize(snapshot)
        ready = bool(
            self.enabled
            and self.selection_enabled
            and self.model is not None
            and snapshot is not None
            and len(self.replay_buffer) >= self.selection_min_samples
            and self.train_steps >= self.selection_min_train_steps
        )
        if ready != self._last_selection_ready:
            state = "enabled" if ready else "disabled"
            print(
                f"[GlobalCritic] LEO selection gate {state}: "
                f"buffer={len(self.replay_buffer)}/{self.selection_min_samples}, "
                f"train_steps={self.train_steps}/{self.selection_min_train_steps}, "
                f"weight={self.selection_weight:.4f}"
            )
            self._last_selection_ready = ready
        return ready

    def rank_actions(self, snapshot, actions: Sequence[JointAction], q_scores: Sequence[float]):
        """Rank legal LEO candidates with normalized Q minus normalized critic risk."""
        if not self.selection_ready(snapshot):
            return None
        try:
            q_values = np.asarray(q_scores, dtype=np.float64).reshape(-1)
            if len(actions) != q_values.size or q_values.size == 0 or not np.all(np.isfinite(q_values)):
                raise ValueError("candidate actions and finite Q scores must have matching non-empty lengths")
            predictions = self.predict_many(snapshot, actions)
            impacts = np.asarray([item["impact"] for item in predictions], dtype=np.float64)
            if impacts.size != q_values.size or not np.all(np.isfinite(impacts)):
                raise ValueError("critic returned invalid candidate impacts")
            risks = -impacts
            normalized_q = self._zscore(q_values)
            normalized_risk = self._zscore(risks)
            combined = normalized_q - self.selection_weight * normalized_risk
            if not np.all(np.isfinite(combined)):
                raise ValueError("combined candidate scores are not finite")
        except Exception:
            self.selection_fallbacks += 1
            return None

        original_index = int(np.argmax(q_values))
        selected_index = int(np.argmax(combined))
        self.selection_uses += 1
        self.selection_disagreements += int(selected_index != original_index)
        self.selection_score_count += 1
        self.selection_q_sum += float(q_values[selected_index])
        self.selection_risk_sum += float(risks[selected_index])
        self.selection_combined_sum += float(combined[selected_index])
        return {
            "selected_index": selected_index,
            "original_index": original_index,
            "q_scores": q_values.astype(np.float32),
            "risks": risks.astype(np.float32),
            "combined_scores": combined.astype(np.float32),
            "predictions": predictions,
        }

    def _zscore(self, values):
        values = np.asarray(values, dtype=np.float64)
        std = float(values.std())
        if std < self.selection_eps:
            return np.zeros_like(values)
        return (values - float(values.mean())) / std

    @staticmethod
    def _masked_smooth_l1(prediction, target, mask):
        per_item = F.smooth_l1_loss(prediction, target, reduction="none")
        weights = mask.to(per_item.dtype)
        return (per_item * weights).sum() / weights.sum().clamp(min=1.0)

    def _make_batch(self, samples):
        batches = [self._sample_arrays(sample) for sample in samples]
        keys = batches[0][0].keys()
        inputs = {
            key: torch.as_tensor(np.stack([item[0][key] for item in batches]), device=self.device)
            for key in keys
        }
        target_keys = batches[0][1].keys()
        targets = {
            key: torch.as_tensor(
                np.stack([item[1][key] for item in batches]), dtype=torch.float32, device=self.device
            )
            for key in target_keys
        }
        return inputs, targets

    def _sample_arrays(self, sample: CriticSample):
        pre = sample.pre_snapshot.aligned(self.node_names, self.edge_names)
        post = sample.post_snapshot.aligned(self.node_names, self.edge_names)
        node_features, edge_features = self._snapshot_features(pre)
        node_lookup = {name: idx for idx, name in enumerate(self.node_names)}
        action_names = [
            sample.action.current_meo,
            sample.action.next_meo,
            sample.action.target_meo,
            sample.action.current_leo,
            sample.action.target_leo,
        ]
        action_indices = np.asarray([node_lookup.get(name, 0) for name in action_names], dtype=np.int64)
        action_node_mask = np.asarray([name in node_lookup for name in action_names], dtype=np.float32)
        edge_index = np.asarray([
            [node_lookup.get(src, 0), node_lookup.get(dst, 0)] for src, dst in self.edge_names
        ], dtype=np.int64).reshape(-1, 2)
        node_mask = self._flat_mask(pre.node_mask, len(self.node_names))
        edge_mask = self._flat_mask(pre.link_mask, len(self.edge_names))
        inputs = {
            "node_features": node_features.astype(np.float32),
            "edge_features": edge_features.astype(np.float32),
            "node_mask": node_mask.astype(bool),
            "edge_mask": edge_mask.astype(bool),
            "edge_index": edge_index,
            "action_indices": action_indices,
            "action_node_mask": action_node_mask,
            "action_type": np.asarray(sample.action.action_type, dtype=np.int64),
            "meo_features": sample.action.meo_features.astype(np.float32),
            "task_features": sample.action.task_features.astype(np.float32),
        }
        targets = {
            "delta_queue": np.asarray(post.queue_load[:, 0] - pre.queue_load[:, 0], dtype=np.float32),
            "delta_compute": np.asarray(post.compute_queue[:, 0] - pre.compute_queue[:, 0], dtype=np.float32),
            "delta_link": np.asarray(post.link_load[:, 0] - pre.link_load[:, 0], dtype=np.float32),
            "success": np.asarray(sample.success, dtype=np.float32),
            "delay": np.asarray(sample.delay, dtype=np.float32),
            "impact": np.asarray(sample.impact_target, dtype=np.float32),
        }
        return inputs, targets

    @staticmethod
    def _flat_mask(mask, size):
        if mask is None:
            return np.ones(size, dtype=bool)
        return np.asarray(mask, dtype=np.float32).reshape(-1)[:size] > 0

    @staticmethod
    def _snapshot_features(snapshot):
        queue = np.asarray(snapshot.queue_load, dtype=np.float32)
        compute = np.asarray(snapshot.compute_queue, dtype=np.float32)
        business = (
            np.asarray(snapshot.business_time, dtype=np.float32)
            if snapshot.business_time is not None
            else np.zeros((len(snapshot.node_names), 1), dtype=np.float32)
        )
        node_features = np.concatenate([queue, compute, business], axis=-1)
        link = np.asarray(snapshot.link_load, dtype=np.float32)
        delays = np.asarray([
            float((snapshot.propagation_delays or {}).get(edge, (snapshot.propagation_delays or {}).get((edge[1], edge[0]), 0.0)))
            for edge in snapshot.edge_names
        ], dtype=np.float32).reshape(-1, 1)
        delay_scale = max(float(delays.max()) if delays.size else 0.0, 1e-9)
        edge_features = np.concatenate([link, delays / delay_scale], axis=-1)
        return node_features, edge_features

    def _update_feature_stats(self, snapshot) -> None:
        node_features, edge_features = self._snapshot_features(snapshot)
        self._merge_feature_stats("node", node_features)
        self._merge_feature_stats("edge", edge_features)

    def _merge_feature_stats(self, key, values) -> None:
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            return
        values = values.reshape(-1, values.shape[-1])
        stats = self.feature_stats[key]
        batch_count = values.shape[0]
        batch_mean = values.mean(axis=0)
        batch_m2 = ((values - batch_mean) ** 2).sum(axis=0)
        if stats["count"] == 0:
            stats.update(count=batch_count, mean=batch_mean, m2=batch_m2)
            return
        total = stats["count"] + batch_count
        delta = batch_mean - stats["mean"]
        stats["mean"] = stats["mean"] + delta * batch_count / total
        stats["m2"] = stats["m2"] + batch_m2 + delta ** 2 * stats["count"] * batch_count / total
        stats["count"] = total

    @property
    def normalization_stats(self):
        result = {}
        for key, stats in self.feature_stats.items():
            count = int(stats["count"])
            result[key] = {
                "count": count,
                "mean": None if stats["mean"] is None else np.asarray(stats["mean"], dtype=np.float32),
                "std": None if stats["m2"] is None else np.sqrt(
                    np.asarray(stats["m2"], dtype=np.float64) / max(count - 1, 1)
                ).astype(np.float32),
            }
        return result

    def reset_round(self) -> int:
        count = len(self.pending)
        self.pending.clear()
        self.discarded_pending += count
        return count

    def save(self, path: Optional[str] = None) -> bool:
        path = path or self.model_path
        if not self.enabled or self.model is None or not path:
            return False
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict() if self.optimizer is not None else None,
            "cfg": self.cfg,
            "node_names": self.node_names,
            "edge_names": self.edge_names,
            "node_feature_dim": self.model.node_feature_dim,
            "edge_feature_dim": self.model.edge_feature_dim,
            "hidden_dim": self.model.hidden_dim,
            "train_steps": self.train_steps,
            "feature_stats": self.feature_stats,
            "selection_gate_active": self._last_selection_ready,
            "selection_stats": {
                "uses": self.selection_uses,
                "disagreements": self.selection_disagreements,
                "fallbacks": self.selection_fallbacks,
                "score_count": self.selection_score_count,
                "q_sum": self.selection_q_sum,
                "risk_sum": self.selection_risk_sum,
                "combined_sum": self.selection_combined_sum,
            },
        }, path)
        return True

    def load(self, path: Optional[str] = None) -> bool:
        path = path or self.model_path
        self._load_attempted = True
        if not self.enabled or self.model is None or not path or not os.path.exists(path):
            return False
        checkpoint = torch.load(path, map_location=self.device)
        compatible = (
            int(checkpoint.get("node_feature_dim", -1)) == self.model.node_feature_dim
            and int(checkpoint.get("edge_feature_dim", -1)) == self.model.edge_feature_dim
            and int(checkpoint.get("hidden_dim", -1)) == self.model.hidden_dim
            and checkpoint.get("node_names") == self.node_names
            and checkpoint.get("edge_names") == self.edge_names
        )
        if not compatible:
            print(f"Warning: global critic checkpoint {path!r} is incompatible; using a fresh model.")
            return False
        self.model.load_state_dict(checkpoint["model"])
        if self.optimizer is not None and checkpoint.get("optimizer") is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.train_steps = int(checkpoint.get("train_steps", 0))
        if isinstance(checkpoint.get("feature_stats"), dict):
            self.feature_stats = checkpoint["feature_stats"]
        selection_stats = checkpoint.get("selection_stats", {}) or {}
        self.selection_uses = int(selection_stats.get("uses", 0))
        self.selection_disagreements = int(selection_stats.get("disagreements", 0))
        self.selection_fallbacks = int(selection_stats.get("fallbacks", 0))
        self.selection_score_count = int(selection_stats.get("score_count", 0))
        self.selection_q_sum = float(selection_stats.get("q_sum", 0.0))
        self.selection_risk_sum = float(selection_stats.get("risk_sum", 0.0))
        self.selection_combined_sum = float(selection_stats.get("combined_sum", 0.0))
        self._last_selection_ready = bool(checkpoint.get("selection_gate_active", False))
        return True
