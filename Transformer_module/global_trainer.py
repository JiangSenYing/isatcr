"""
Training wrapper for the global Transformer forecaster and path planner.
"""

from typing import Dict, List, Optional

import numpy as np
import torch

from .transformer_forecaster import (
    GlobalStateExtractor,
    SatelliteLoadTransformer,
    TransformerPathPlanner,
    snapshots_to_training_batch,
)
from .meo_router import MEODomainRouter


def float_item(value) -> float:
    """把 torch 标量或普通数值安全转换成 Python float。"""
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def average_metric_dicts(items: List[Dict[str, float]]) -> Dict[str, float]:
    """对一组同结构指标字典逐 key 求平均，空列表返回空字典。"""
    if not items:
        return {}
    keys = items[0].keys()
    return {
        key: float(np.mean([item[key] for item in items if key in item]))
        for key in keys
    }


def format_prediction_metrics(metrics: Dict[str, float]) -> str:
    """把预测误差指标格式化成训练日志中使用的一行文本。"""
    if not metrics:
        return ""
    return (
        f"pred_mae(queue/link/compute)="
        f"{metrics.get('queue_mae', 0):.4f}/"
        f"{metrics.get('link_mae', 0):.4f}/"
        f"{metrics.get('compute_queue_mae', 0):.4f}, "
        f"business_time_mae={metrics.get('business_time_mae', 0):.4f}, "
        f"pred_rmse(queue/link/compute)="
        f"{metrics.get('queue_rmse', 0):.4f}/"
        f"{metrics.get('link_rmse', 0):.4f}/"
        f"{metrics.get('compute_queue_rmse', 0):.4f}, "
        f"business_time_rmse={metrics.get('business_time_rmse', 0):.4f}"
    )


class GlobalTransformerTrainer:
    """Train and use the global Transformer planner from simulator snapshots."""

    def __init__(self, cfg: Dict, device: torch.device):
        """读取配置并初始化训练器状态，模型会在拿到第一帧快照后懒加载。"""
        self.cfg = cfg
        self.device = device
        self.transformer_enabled = bool(cfg.get('enabled', True))
        self.meo_exit_enabled = bool(cfg.get('meo_exit_enabled', False))
        self.use_meo_aggregation = bool(cfg.get('use_meo_aggregation', True))
        self.history_len = int(cfg.get('history_len', 12))
        self.forecast_horizon = int(cfg.get('forecast_horizon', 5))
        self.batch_size = int(cfg.get('batch_size', 8))
        self.max_snapshots = int(cfg.get('max_snapshots', 20000))
        self.warmup_snapshots = int(cfg.get('warmup_snapshots', 64))
        self.update_every = max(1, int(cfg.get('update_every', 1)))
        self.updates_per_step = max(1, int(cfg.get('updates_per_step', 1)))
        self.eval_every = max(1, int(cfg.get('eval_every', 60)))
        self.plan_every = max(1, int(cfg.get('plan_every', 60)))
        self.repeat = cfg.get('repeat', 1)

        self.model = None
        self.optimizer = None
        self.planner = None
        self.snapshots = []
        self.node_names = []
        self.edge_names = []
        self.step_count = 0
        self.last_losses = {}
        self.last_metrics = {}
        self.last_plan = None
        self.meo_router = MEODomainRouter(cfg, device=str(device), transformer_enabled=self.transformer_enabled)

    def initialize(self, snapshot) -> None:
        """根据第一帧快照的特征维度创建 Transformer、优化器和路径规划器。"""
        if self.model is not None:
            return

        self.node_names = list(snapshot.node_names)
        self.edge_names = list(snapshot.edge_names)
        self.model = SatelliteLoadTransformer(
            queue_input_dim=int(snapshot.queue_load.shape[-1]),
            link_input_dim=int(snapshot.link_load.shape[-1]),
            compute_input_dim=int(snapshot.compute_queue.shape[-1]),
            forecast_horizon=self.forecast_horizon,
            d_model=int(self.cfg.get('d_model', 64)),
            nhead=int(self.cfg.get('nhead', 4)),
            num_layers=int(self.cfg.get('num_layers', 2)),
            dim_feedforward=int(self.cfg.get('dim_feedforward', 128)),
            dropout=float(self.cfg.get('dropout', 0.1)),
            max_history_len=max(self.history_len, self.forecast_horizon, 16),
        ).to(self.device)
        self.model.node_names = self.node_names
        self.model.edge_names = self.edge_names
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(self.cfg.get('learning_rate', 1e-4)),
        )
        self.planner = TransformerPathPlanner.from_snapshot(
            transformer=self.model,
            snapshot=snapshot,
            history_len=self.history_len,
            device=str(self.device),
            max_history=self.max_snapshots,
            max_candidate_expansions=int(self.cfg.get('plan_max_candidate_expansions', 2000)),
            max_candidate_queue_size=int(self.cfg.get('plan_max_candidate_queue_size', 5000)),
        )

    def add_env_snapshot(self, env, replace_same_time: bool = False) -> None:
        """从环境采集一帧全局快照，更新历史缓存、规划器和负载变化率特征。"""
        previous_snapshot = self.snapshots[-1] if self.snapshots else None
        snapshot = GlobalStateExtractor.from_env(env, previous_snapshot=previous_snapshot)
        self.initialize(snapshot)
        aligned = snapshot.aligned(self.node_names, self.edge_names)
        if previous_snapshot is not None:
            dt = 1.0
            if aligned.sim_time is not None and previous_snapshot.sim_time is not None:
                dt = max(float(aligned.sim_time) - float(previous_snapshot.sim_time), 1e-6)
            aligned.queue_load[:, -1:] = np.clip(
                (aligned.queue_load[:, :1] - previous_snapshot.queue_load[:, :1]) / dt,
                -1.0,
                1.0,
            )
            aligned.link_load[:, -1:] = np.clip(
                (aligned.link_load[:, :1] - previous_snapshot.link_load[:, :1]) / dt,
                -1.0,
                1.0,
            )
            aligned.compute_queue[:, -1:] = np.clip(
                (aligned.compute_queue[:, :1] - previous_snapshot.compute_queue[:, :1]) / dt,
                -1.0,
                1.0,
            )
        same_time = (
            replace_same_time
            and previous_snapshot is not None
            and aligned.sim_time is not None
            and previous_snapshot.sim_time is not None
            and abs(float(aligned.sim_time) - float(previous_snapshot.sim_time)) <= 1e-9
        )
        if same_time:
            self.snapshots[-1] = aligned
        else:
            self.snapshots.append(aligned)
            if len(self.snapshots) > self.max_snapshots:
                self.snapshots = self.snapshots[-self.max_snapshots:]
        if self.planner is not None:
            if same_time and self.planner.history:
                self.planner.history[-1] = aligned
                self.planner.latest_snapshot = aligned
                self.planner.clear_reservations()
            else:
                self.planner.add_snapshot(aligned)
            self.planner.set_future_graph_builder(self._make_future_graph_builder(env))
        if not same_time:
            self.step_count += 1

    @staticmethod
    def _make_future_graph_builder(env):
        """为支持未来拓扑构造的环境生成 planner 可调用的未来图构造函数。"""
        if not hasattr(env, 'build_graph_for_transformer'):
            return None

        def build_future_graphs(forecast_times, latest_sim_time):
            """按预测时间点调用环境接口，生成未来 NetworkX 拓扑图字典。"""
            graphs = {}
            for sim_time in forecast_times:
                offset_seconds = max(0.0, float(sim_time) - float(latest_sim_time))
                time_arg = GlobalTransformerTrainer._future_time_arg(env, offset_seconds)
                graphs[float(sim_time)] = env.build_graph_for_transformer(time_arg)
            return graphs

        return build_future_graphs

    @staticmethod
    def _future_time_arg(env, offset_seconds: float):
        """把相对秒数转换成环境需要的时间参数，兼容字符串时间接口和数值接口。"""
        if hasattr(env, 'current_time') and hasattr(env, 'add_time_to_str') and hasattr(env, 'time_from_str'):
            future_time = env.add_time_to_str(env.current_time, (0, int(round(offset_seconds))))
            return env.time_from_str(future_time)
        return offset_seconds

    def can_update(self) -> bool:
        """判断当前快照数量是否足够进行一次训练更新。"""
        needed = self.history_len + self.forecast_horizon
        return self.model is not None and len(self.snapshots) >= max(needed + self.batch_size, self.warmup_snapshots)

    def update_if_ready(self) -> Optional[float]:
        """在达到更新间隔且数据充足时执行训练，返回平均 loss；否则返回 None。"""
        if not self.transformer_enabled:
            return None, None
        if self.step_count % self.update_every != 0 or not self.can_update():
            return None,None

        losses = []
        for _ in range(self.updates_per_step):
            loss, parts = self._update_once()
            losses.append(loss)
            self.last_losses = parts
        return float(np.mean(losses)) if losses else None, parts

    def _update_once(self):
        """采样一个训练 batch，前向计算图损失并完成一次反向传播更新。"""
        assert self.model is not None and self.optimizer is not None
        self.model.train()
        batch = snapshots_to_training_batch(
            snapshots=self.snapshots,
            history_len=self.history_len,
            forecast_horizon=self.forecast_horizon,
            node_names=self.node_names,
            edge_names=self.edge_names,
            batch_size=self.batch_size,
            device=str(self.device),
            include_target_masks=True,
            include_target_graphs=True,
        )
        graph_target_masks = batch[7] + (batch[8],)
        loss, parts = self.model.graph_training_loss(*batch[:7], graph_target_masks=graph_target_masks)
        self.last_training_prediction_graphs = batch[8]
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return float_item(loss), {key: float_item(value) for key, value in parts.items()}

    def should_eval(self) -> bool:
        """判断当前步数是否应该执行一次预测评估。"""
        return self.transformer_enabled and self.step_count % self.eval_every == 0 and self.can_evaluate()

    def can_evaluate(self) -> bool:
        """判断是否已有足够历史和未来快照用于离线评估。"""
        return self.model is not None and len(self.snapshots) >= self.history_len + self.forecast_horizon

    @torch.no_grad()
    def evaluate_latest(self) -> Optional[Dict[str, float]]:
        """用最近一段历史预测紧随其后的未来快照，并计算 MAE/RMSE 等指标。"""
        if not self.can_evaluate():
            return None

        assert self.model is not None
        start = len(self.snapshots) - self.history_len - self.forecast_horizon
        history = self.snapshots[start:start + self.history_len]
        future = self.snapshots[start + self.history_len:start + self.history_len + self.forecast_horizon]

        queue_hist = torch.as_tensor(np.stack([item.queue_load for item in history])[None], dtype=torch.float32, device=self.device)
        link_hist = torch.as_tensor(np.stack([item.link_load for item in history])[None], dtype=torch.float32, device=self.device)
        compute_hist = torch.as_tensor(np.stack([item.compute_queue for item in history])[None], dtype=torch.float32, device=self.device)
        targets = {
            'queue': torch.as_tensor(np.stack([item.queue_load[:, :1] for item in future])[None], dtype=torch.float32, device=self.device),
            'link': torch.as_tensor(np.stack([item.link_load[:, :1] for item in future])[None], dtype=torch.float32, device=self.device),
            'compute_queue': torch.as_tensor(np.stack([item.compute_queue[:, :1] for item in future])[None], dtype=torch.float32, device=self.device),
            'business_time': torch.as_tensor(np.stack([
                (item.business_time if item.business_time is not None else np.zeros((len(item.node_names), 1), dtype=np.float32))[:, :1]
                for item in future
            ])[None], dtype=torch.float32, device=self.device),
        }
        graph_masks = {
            'queue': torch.as_tensor(np.stack([item.node_mask for item in future])[None], dtype=torch.float32, device=self.device),
            'link': torch.as_tensor(np.stack([item.link_mask for item in future])[None], dtype=torch.float32, device=self.device),
            'compute_queue': torch.as_tensor(np.stack([item.node_mask for item in future])[None], dtype=torch.float32, device=self.device),
            'business_time': torch.as_tensor(np.stack([item.node_mask for item in future])[None], dtype=torch.float32, device=self.device),
        }

        self.model.eval()
        preds = self.model(queue_hist, link_hist, compute_hist)
        metrics = {}
        metrics.update(self._metric_group('queue', preds['queue_forecast'], targets['queue'], graph_masks['queue']))
        metrics.update(self._metric_group('link', preds['link_forecast'], targets['link'], graph_masks['link']))
        metrics.update(self._metric_group('compute_queue', preds['compute_queue_forecast'], targets['compute_queue'], graph_masks['compute_queue']))
        metrics.update(self._metric_group('business_time', preds['business_time_forecast'], targets['business_time'], graph_masks['business_time']))
        self.last_metrics = metrics
        return metrics

    @staticmethod
    def _metric_group(prefix: str, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """计算一类预测目标的 MAE、RMSE、相对 MAE 和最大绝对误差。"""
        diff = pred - target
        if mask is not None:
            mask = mask.to(dtype=diff.dtype, device=diff.device)
            diff = diff * mask
            denom = mask.expand_as(diff).sum().clamp(min=1.0)
        else:
            denom = torch.as_tensor(diff.numel(), dtype=diff.dtype, device=diff.device).clamp(min=1.0)
        abs_diff = diff.abs()
        mse = torch.sum(diff * diff) / denom
        mae = torch.sum(abs_diff) / denom
        target_abs_mean = torch.sum(target.abs() * mask) / denom if mask is not None else torch.mean(target.abs())
        rel_mae = mae / target_abs_mean.clamp(min=1e-6)
        return {
            f'{prefix}_mae': float_item(mae),
            f'{prefix}_rmse': float_item(torch.sqrt(mse)),
            f'{prefix}_rel_mae': float_item(rel_mae),
            f'{prefix}_max_ae': float_item(torch.max(abs_diff)),
        }

    def should_plan(self) -> bool:
        """判断当前步数是否达到路径规划调用间隔。"""
        return self.step_count % self.plan_every == 0 and self.planner is not None

    def recommend_path(
        self,
        source,
        destination,
        packet_size,
        computing_demand,
        size_after_computing=None,
        business_duration=None,
    ):
        """调用底层 TransformerPathPlanner 推荐路径和单个计算节点。"""
        if self.planner is None or not self.snapshots:
            return None

        if source is None or destination is None:
            source, destination = self._default_source_destination()
        if source is None or destination is None or source == destination:
            return None

        try:
            self.last_plan = self.planner.plan(
                source=source,
                destination=destination,
                packet_size=packet_size,
                computing_demand=computing_demand,
                size_after_computing=size_after_computing,
                business_duration=business_duration,
                need_compute=bool(self.cfg.get('plan_need_compute', True)),
                top_k=int(self.cfg.get('plan_top_k', 16)),
                delay_top_k=int(self.cfg.get('plan_delay_top_k', self.cfg.get('plan_top_k', 16))),
                load_top_k=int(self.cfg.get('plan_load_top_k', 0)),
                max_hops=int(self.cfg.get('plan_max_hops', 12)),
                show_path_error=bool(self.cfg.get('showPathError', False)),
            )
            return self.last_plan
        except ValueError as exc:
            print(
                f"[Transformer plan ValueError] "
                f"source={source}, destination={destination}, error={exc}"
            )
            return None

    def reserve_plan(
        self,
        plan,
        packet_size,
        computing_demand=0.0,
        size_after_computing=None,
    ) -> None:
        """登记一次已实际发出的 Transformer 路径预约负载。"""
        if self.planner is None or plan is None:
            return
        self.planner.reserve_plan(
            plan=plan,
            packet_size=packet_size,
            computing_demand=computing_demand,
            size_after_computing=size_after_computing,
        )

    def predict_future_graphs(self):
        """返回写入预测负载后的未来拓扑图，供可视化或调试使用。"""
        if not self.transformer_enabled or self.planner is None or not self.snapshots:
            return None
        _, graphs = self.planner.predict_future(return_graphs=True)
        return graphs

    def recommend_meo_path(
        self,
        meo_satellite,
        src,
        dst,
        packet_size,
        task_type=0,
        computing_demand=0.0,
        size_after_computing=0.0,
        is_computed=False,
        excluded_domains=None,
    ):
        """使用 MEO 域间策略网络推荐跨多域路径。"""
        if self.meo_router is None:
            return None
        return self.meo_router.recommend_path(
            meo_satellite=meo_satellite,
            src=src,
            dst=dst,
            packet_size=packet_size,
            task_type=task_type,
            computing_demand=computing_demand,
            size_after_computing=size_after_computing,
            is_computed=is_computed,
            transformer_trainer=self,
            excluded_domains=excluded_domains,
        )

    def finish_meo_decision(self, packet, reward, done=True, meo_result=None):
        """把包级终局奖励回传给 MEO Agent。"""
        if self.meo_router is not None:
            self.meo_router.finish_decision(packet, reward, done=done, meo_result=meo_result)

    def store_leo_policy_experience(self, *args, **kwargs):
        """把 LEO 实际动作监督样本转发给 MEO router 内的 LEO predictor。"""
        if self.meo_router is None:
            return False
        return self.meo_router.store_leo_policy_experience(*args, **kwargs)

    def update_meo_if_ready(self):
        """按 MEO Agent 自身配置执行一次可选更新。"""
        if self.meo_router is None:
            return None
        return self.meo_router.update_if_ready()

    def _default_source_destination(self):
        """当调用方未提供源/目的节点时，从当前拓扑中选择默认可用节点对。"""
        if not self.snapshots:
            return None, None
        adjacency = self.snapshots[-1].adjacency
        nodes = [node for node in self.node_names if adjacency.get(node)]
        if len(nodes) < 2:
            return None, None
        return nodes[0], nodes[-1]

    def save(self, path: str) -> None:
        """保存模型参数、优化器状态、配置和固定拓扑顺序。"""
        if self.meo_router is not None:
            self.meo_router.save()
        if not self.transformer_enabled:
            return
        if self.model is None or not path:
            return
        import os

        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict() if self.optimizer is not None else None,
            'cfg': self.cfg,
            'node_names': self.node_names,
            'edge_names': self.edge_names,
        }, path)

    def load_if_compatible(self, path: str) -> None:
        """在拓扑和特征维度兼容时加载已有 Transformer checkpoint。"""
        import os

        if not self.transformer_enabled:
            return
        if self.model is None or not path or not os.path.exists(path):
            return
        checkpoint = torch.load(path, map_location=self.device)
        # if checkpoint.get('node_names') != self.node_names or checkpoint.get('edge_names') != self.edge_names:
        #     print("Warning: Transformer checkpoint topology differs from current topology; skipping load.")
        #     return
        try:
            self.model.load_state_dict(checkpoint['model'])
        except RuntimeError as exc:
            print(f"Warning: Transformer checkpoint is incompatible with current input features; skipping load. {exc}")
            return
        if self.optimizer is not None and checkpoint.get('optimizer') is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
