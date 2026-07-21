"""Data contracts used by the global action critic."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class JointAction:
    """One executed MEO decision paired with one executed LEO step."""

    decision_id: str
    current_meo: str
    next_meo: str
    target_meo: str
    current_leo: str
    target_leo: str
    action_type: int  # 0: forward, 1: compute
    meo_features: np.ndarray
    task_features: np.ndarray

    def __post_init__(self):
        features = np.asarray(self.task_features, dtype=np.float32).reshape(-1)
        if features.shape != (6,):
            raise ValueError(f"task_features must have shape (6,), got {features.shape}")
        if int(self.action_type) not in (0, 1):
            raise ValueError("action_type must be 0 (forward) or 1 (compute)")
        meo_features = np.asarray(self.meo_features, dtype=np.float32).reshape(-1)
        if meo_features.shape != (4,):
            raise ValueError(f"meo_features must have shape (4,), got {meo_features.shape}")
        object.__setattr__(self, "meo_features", meo_features.copy())
        object.__setattr__(self, "task_features", features.copy())


@dataclass
class CriticSample:
    """A completed observational training sample."""

    pre_snapshot: Any
    post_snapshot: Any
    action: JointAction
    terminal_reward: float
    success: float
    delay: float
    impact_target: float


@dataclass
class CriticOutput:
    """Multi-head prediction produced by :class:`GlobalActionCritic`."""

    delta_queue: torch.Tensor
    delta_compute: torch.Tensor
    delta_link: torch.Tensor
    success_logit: torch.Tensor
    delay: torch.Tensor
    impact: torch.Tensor
