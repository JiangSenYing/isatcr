"""
GNN 智能体 (GNN Agents)
参考 cross_layer_opt_with_grl-main 的设计，但适配卫星路由场景

实现基于 GNN 的 DDQN 智能体，支持:
1. 图观测输入 (PyG 异构图)
2. 扁平状态输入 (自动转换为图)
3. 经验回放
4. 目标网络
5. 任务特征分离模式 (separated)
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random

try:
    from torch_geometric.data import HeteroData, Data
    from . import pyg_compat
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False

# 为向后兼容保留 DGL_AVAILABLE 变量名
DGL_AVAILABLE = PYG_AVAILABLE

from .controllers import RelationalController, GraphController, SimpleQNetwork, CONTROLLER_REGISTRY
from .diffusion_policy import DiffusionActionPrior, DiffusionGuidedQNetwork

# 尝试导入 V2/V2Lite 控制器
try:
    from .task_aware_blocks import TaskConditionedGraphControllerV2, TaskConditionedGraphControllerV2Lite
    V2_AVAILABLE = True
except ImportError:
    TaskConditionedGraphControllerV2 = None
    TaskConditionedGraphControllerV2Lite = None
    V2_AVAILABLE = False


class GNNDDQNAgent:
    """
    基于 GNN 的 Double DQN 智能体
    
    支持两种观测输入:
    1. PyG 图: 直接使用图控制器
    2. 扁平状态: 使用 observation_wrappers 转换后再处理
    
    支持两种特征模式:
    1. 混合模式: agent节点包含环境+任务信息
    2. 分离模式 (separated): agent节点只有环境信息，task作为独立输入
    
    特点:
    - Double DQN: 使用在线网络选动作，目标网络估值
    - 支持 Dueling 架构
    - 支持 RNN 隐状态
    - 兼容原有训练流程
    """
    
    def __init__(
        self,
        obs_shape: Union[int, Dict[str, int]],  # 状态维度或观测形状字典
        n_actions: int = 5,                      # 动作数
        hidden_size: int = 64,                   # 隐藏层维度
        max_nbrs: int = 4,                       # 最大邻居数
        n_hops: int = 1,                         # GNN 跳数
        controller_type: str = 'relational',     # 控制器类型
        use_rnn: bool = False,                   # 是否使用 RNN
        dueling: bool = True,                    # 是否使用 Dueling 架构
        conv_type: str = 'gat',                  # 图卷积类型: 'gat' 或 'gcn'
        n_heads: int = 4,                        # GAT 注意力头数
        graph_enc_type: str = 'gn',              # GraphController 编码聚合: 'gn' 或 'gat'
        graph_gat_dropout: float = 0.0,          # GraphController 的 GAT dropout
        task_dim: int = 6,                       # 任务上下文维度（分离模式）
        gamma: float = 0.99,                     # 折扣因子
        lr: float = 1e-4,                        # 学习率
        buffer_size: int = 10000,                # 经验回放缓冲区大小
        batch_size: int = 64,                    # 批量大小
        target_update_freq: int = 100,           # 目标网络更新频率
        epsilon_start: float = 1.0,              # 初始探索率
        epsilon_end: float = 0.01,               # 最终探索率
        epsilon_decay: float = 0.995,            # 探索率衰减
        use_diffusion: bool = False,             # 是否启用扩散动作先验
        diffusion_steps: int = 20,               # 扩散去噪步数
        diffusion_hidden_size: int = 128,        # 扩散模型隐藏层
        diffusion_loss_weight: float = 0.1,      # 扩散损失权重
        diffusion_q_loss_weight: float = 0.1,    # Q 引导扩散损失权重
        diffusion_q_temperature: float = 1.0,    # Q 分布软标签温度
        diffusion_guidance_weight: float = 0.5,  # 推理时 prior 融合权重
        diffusion_guidance_warmup_updates: int = 0,  # prior 引导 warmup 更新步数
        diffusion_deterministic: bool = True,    # 推理 prior 是否确定性采样
        diffusion_normalize_prior: bool = True,  # 是否标准化 prior logits
        diffusion_prior_clip: float = 2.0,       # prior logits 限幅
        device: str = 'cpu',
    ):
        self.device = th.device(device)
        self.n_actions = n_actions
        self.max_nbrs = max_nbrs
        self.n_hops = n_hops
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.use_rnn = use_rnn
        self.dueling = dueling
        self.conv_type = conv_type
        self.n_heads = n_heads
        self.graph_enc_type = graph_enc_type
        self.graph_gat_dropout = graph_gat_dropout
        self.task_dim = task_dim
        self.controller_type = controller_type
        
        # 探索参数
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.use_diffusion = use_diffusion
        self.diffusion_loss_weight = diffusion_loss_weight
        self.diffusion_q_loss_weight = diffusion_q_loss_weight
        self.diffusion_q_temperature = diffusion_q_temperature
        self.diffusion_guidance_weight = diffusion_guidance_weight
        self.diffusion_guidance_warmup_updates = diffusion_guidance_warmup_updates
        self.max_grad_norm = 10.0
        
        # 判断是否为分离模式
        self._use_separated = 'separated' in controller_type
        
        # 判断是图输入还是扁平输入
        if isinstance(obs_shape, int):
            # 扁平状态输入，使用简单 MLP
            self._use_graph = False
            self.q_network = SimpleQNetwork(
                state_dim=obs_shape,
                n_actions=n_actions,
                hidden_size=hidden_size,
                dueling=dueling
            ).to(self.device)
            self.target_network = SimpleQNetwork(
                state_dim=obs_shape,
                n_actions=n_actions,
                hidden_size=hidden_size,
                dueling=dueling
            ).to(self.device)
        else:
            # 图观测输入
            self._use_graph = True
            if not DGL_AVAILABLE:
                raise ImportError("PyTorch Geometric is required for graph observations")
            
            # 获取控制器类
            # 优先处理 V2/V2Lite 版本
            if controller_type in ('graph_separated_v2', 'gnn_separated_v2'):
                if not V2_AVAILABLE:
                    raise ImportError("TaskConditionedGraphControllerV2 not available. "
                                      "Please check gnn_modules/task_aware_blocks.py")
                ControllerClass = TaskConditionedGraphControllerV2
            elif controller_type in ('graph_separated_v2_lite', 'gnn_separated_v2_lite'):
                if not V2_AVAILABLE:
                    raise ImportError("TaskConditionedGraphControllerV2Lite not available. "
                                      "Please check gnn_modules/task_aware_blocks.py")
                ControllerClass = TaskConditionedGraphControllerV2Lite
            else:
                ControllerClass = CONTROLLER_REGISTRY.get(controller_type, RelationalController)
            
            # 构建控制器参数
            controller_kwargs = dict(
                obs_shape=obs_shape,
                n_actions=n_actions,
                hidden_size=hidden_size,
                max_nbrs=max_nbrs,
                use_rnn=use_rnn,
                dueling=dueling,  #  传递 dueling 参数
                device=str(self.device)
            )
            
            # 分离模式需要 task_dim 参数
            if self._use_separated:
                controller_kwargs['task_dim'] = task_dim
            
            # RelationalController 支持 conv_type 和 n_heads
            if controller_type in ('relational', 'rel', 'relational_separated', 'rel_separated'):
                controller_kwargs['conv_type'] = conv_type
                controller_kwargs['n_heads'] = n_heads
            
            # GraphController 支持 n_hops（包括 V2/V2Lite 版本）
            if controller_type in ('graph', 'gnn', 'graph_separated', 'gnn_separated',
                                   'graph_separated_v2', 'gnn_separated_v2',
                                   'graph_separated_v2_lite', 'gnn_separated_v2_lite'):
                controller_kwargs['n_hops'] = n_hops

            # 仅 GraphController 使用可切换的 GN/GAT 编码块
            if controller_type in ('graph', 'gnn'):
                controller_kwargs['enc_agg_type'] = graph_enc_type
                controller_kwargs['gat_heads'] = n_heads
                controller_kwargs['gat_dropout'] = graph_gat_dropout
            
            self.q_network = ControllerClass(**controller_kwargs).to(self.device)
            self.target_network = ControllerClass(**controller_kwargs).to(self.device)

        # 同步目标网络
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optional conditional diffusion action prior. It is conditioned on the
        # Q-value vector, so it works for flat, relational, and graph controllers.
        self.diffusion_prior = None
        self.action_network = self.q_network
        if self.use_diffusion:
            initial_guidance_scale = 0.0 if diffusion_guidance_warmup_updates > 0 else 1.0
            self.diffusion_prior = DiffusionActionPrior(
                n_actions=n_actions,
                condition_dim=n_actions,
                hidden_size=diffusion_hidden_size,
                diffusion_steps=diffusion_steps,
            ).to(self.device)
            self.action_network = DiffusionGuidedQNetwork(
                self.q_network,
                self.diffusion_prior,
                guidance_weight=diffusion_guidance_weight,
                deterministic=diffusion_deterministic,
                normalize_prior=diffusion_normalize_prior,
                prior_clip=diffusion_prior_clip,
                guidance_scale=initial_guidance_scale,
            ).to(self.device)

        # 优化器
        trainable_params = list(self.q_network.parameters())
        if self.diffusion_prior is not None:
            trainable_params += list(self.diffusion_prior.parameters())
        self.trainable_params = trainable_params
        self.optimizer = th.optim.Adam(trainable_params, lr=lr)
        
        # 经验回放
        self.replay_buffer = deque(maxlen=buffer_size)
        
        # 训练计数
        self.train_step = 0
        
        # RNN 隐状态
        self._hidden = None
        self.last_losses = {}

    def get_action_network(self):
        """Return the network used by the simulator for action selection."""
        return self.action_network

    def _update_diffusion_guidance_scale(self):
        if self.diffusion_prior is None or not hasattr(self.action_network, 'set_guidance_scale'):
            return
        if self.diffusion_guidance_warmup_updates <= 0:
            self.action_network.set_guidance_scale(1.0)
            return
        scale = self.train_step / float(self.diffusion_guidance_warmup_updates)
        self.action_network.set_guidance_scale(scale)
    
    def reset_hidden(self, batch_size: int = 1):
        """重置 RNN 隐状态"""
        if self.use_rnn and self._use_graph:
            self._hidden = self.q_network.init_hidden(batch_size)
        else:
            self._hidden = None
    
    def select_action(self, obs, valid_actions: Optional[List[int]] = None, 
                      task_context: Optional[th.Tensor] = None, training: bool = True):
        """
        选择动作
        
        Args:
            obs: 观测 (PyG 图或 tensor)
            valid_actions: 有效动作列表
            task_context: 任务上下文（分离模式必需）[task_dim] 或 tensor
            training: 是否训练模式（使用 epsilon-greedy）
        
        Returns:
            action: 选择的动作
        """
        if valid_actions is None:
            valid_actions = list(range(self.n_actions))
        
        # Epsilon-greedy
        if training and random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        with th.no_grad():
            if self._use_graph:
                # 检查是否是 PyG 图类型
                if not isinstance(obs, (HeteroData, Data)):
                    raise TypeError("Expected PyTorch Geometric HeteroData observation")
                obs = obs.to(self.device)
                
                if self._use_separated:
                    # 分离模式：需要 task_context
                    if task_context is None:
                        raise ValueError("task_context is required for separated mode")
                    if not isinstance(task_context, th.Tensor):
                        task_context = th.tensor(task_context, dtype=th.float32)
                    task_context = task_context.to(self.device)
                    q_values, self._hidden = self.q_network(obs, task_context, self._hidden)
                else:
                    # 非分离模式
                    q_values, self._hidden = self.q_network(obs, self._hidden)
            else:
                if not isinstance(obs, th.Tensor):
                    obs = th.tensor(obs, dtype=th.float32)
                obs = obs.to(self.device)
                if obs.dim() == 1:
                    obs = obs.unsqueeze(0)
                q_values = self.q_network(obs)
            
            # 只考虑有效动作
            q_values = q_values.squeeze(0)
            valid_q = q_values[valid_actions]
            action_idx = valid_q.argmax().item()
            return valid_actions[action_idx]
    
    def store_experience(self, obs, action, reward, next_obs, done, 
                         task_context=None, next_task_context=None, mark=None):
        """
        存储经验到回放缓冲区
        
        Args:
            obs: 当前观测
            action: 执行的动作
            reward: 获得的奖励
            next_obs: 下一个观测
            done: 是否结束
            task_context: 当前任务上下文（分离模式）
            next_task_context: 下一个任务上下文（分离模式）
            mark: 标记（可选，用于优先级经验回放等）
        """
        mark = 0 if mark is None else int(mark)
        if self._use_separated:
            # 分离模式：存储额外的 task_context
            self.replay_buffer.append((obs, action, reward, next_obs, done, task_context, next_task_context, mark))
        else:
            self.replay_buffer.append((obs, action, reward, next_obs, done, mark))
    
    def update(self) -> Optional[float]:
        """
        更新网络
        
        Returns:
            loss: 损失值（如果更新了），否则 None
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # 采样
        batch = random.sample(self.replay_buffer, self.batch_size)
        
        if self._use_graph:
            if self._use_separated:
                return self._update_graph_separated(batch)
            else:
                return self._update_graph(batch)
        else:
            return self._update_flat(batch)
    
    def _update_flat(self, batch) -> float:
        """扁平状态的更新"""
        obs, actions, rewards, next_obs, dones, marks = zip(*batch)
        
        obs = th.tensor(np.array(obs), dtype=th.float32, device=self.device)
        actions = th.tensor(actions, dtype=th.long, device=self.device)
        rewards = th.tensor(rewards, dtype=th.float32, device=self.device)
        next_obs = th.tensor(np.array(next_obs), dtype=th.float32, device=self.device)
        dones = th.tensor(dones, dtype=th.float32, device=self.device)
        marks = th.tensor(marks, dtype=th.long, device=self.device)
        
        # 当前 Q 值
        all_q_values = self.q_network(obs)
        q_values = all_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN: 用在线网络选动作，目标网络估值
        with th.no_grad():
            next_action4_invalid = self._infer_action4_invalid_mask(next_obs)
            next_actions = self._select_next_actions(
                self.q_network(next_obs),
                action4_invalid_mask=next_action4_invalid,
            )
            next_q_values = self.target_network(next_obs)
            next_q_values = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            targets = rewards + self.gamma * next_q_values * (1 - dones)
        
        # 计算损失
        q_loss = F.smooth_l1_loss(q_values, targets)
        diffusion_loss = self._diffusion_loss(all_q_values, actions, marks)
        loss = self._combine_losses(q_loss, diffusion_loss)
        
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(self.trainable_params, self.max_grad_norm)
        self.optimizer.step()
        
        # 更新目标网络
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        self._update_diffusion_guidance_scale()
        
        # 衰减 epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        self._record_losses(q_loss, diffusion_loss, loss)
        return loss.item()
    
    def _update_graph(self, batch) -> float:
        """图观测的更新"""
        obs_list, actions, rewards, next_obs_list, dones, marks = zip(*batch)
        
        # 确保所有图都在同一设备上（CPU），然后批量处理
        obs_list_cpu = [g.cpu() if hasattr(g, 'cpu') else g for g in obs_list]
        next_obs_list_cpu = [g.cpu() if hasattr(g, 'cpu') else g for g in next_obs_list]
        
        # 批量处理图 - 使用 PyG 的 batch
        obs_batch = pyg_compat.batch(obs_list_cpu).to(self.device)
        next_obs_batch = pyg_compat.batch(next_obs_list_cpu).to(self.device)
        
        actions = th.tensor(actions, dtype=th.long, device=self.device)
        rewards = th.tensor(rewards, dtype=th.float32, device=self.device)
        dones = th.tensor(dones, dtype=th.float32, device=self.device)
        marks = th.tensor(marks, dtype=th.long, device=self.device)
        
        # 当前 Q 值
        all_q_values, _ = self.q_network(obs_batch, None)
        q_values = all_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN
        with th.no_grad():
            next_q_online, _ = self.q_network(next_obs_batch, None)
            next_action4_invalid = self._infer_action4_invalid_mask(next_obs_batch)
            next_actions = self._select_next_actions(
                next_q_online,
                action4_invalid_mask=next_action4_invalid,
            )
            next_q_target, _ = self.target_network(next_obs_batch, None)
            next_q_values = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            targets = rewards + self.gamma * next_q_values * (1 - dones)
        
        # 损失和优化
        q_loss = F.smooth_l1_loss(q_values, targets)
        diffusion_loss = self._diffusion_loss(all_q_values, actions, marks)
        loss = self._combine_losses(q_loss, diffusion_loss)
        
        self.optimizer.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(self.trainable_params, self.max_grad_norm)
        self.optimizer.step()
        
        # 更新目标网络
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        self._update_diffusion_guidance_scale()
        
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        self._record_losses(q_loss, diffusion_loss, loss)
        return loss.item()
    
    def _update_graph_separated(self, batch) -> float:
        """
        【任务特征分离模式】图观测的更新
        
        经验格式: (obs, action, reward, next_obs, done, task_context, next_task_context)
        """
        obs_list, actions, rewards, next_obs_list, dones, task_contexts, next_task_contexts, marks = zip(*batch)
        
        # 确保所有图都在同一设备上（CPU），然后批量处理
        obs_list_cpu = [g.cpu() if hasattr(g, 'cpu') else g for g in obs_list]
        next_obs_list_cpu = [g.cpu() if hasattr(g, 'cpu') else g for g in next_obs_list]
        
        # 批量处理图
        obs_batch = pyg_compat.batch(obs_list_cpu).to(self.device)
        next_obs_batch = pyg_compat.batch(next_obs_list_cpu).to(self.device)
        
        # 处理 task_context
        # 将 numpy array 或 tensor 转换为 batch tensor
        task_batch = []
        next_task_batch = []
        for tc, ntc in zip(task_contexts, next_task_contexts):
            if isinstance(tc, np.ndarray):
                task_batch.append(th.tensor(tc, dtype=th.float32))
            elif isinstance(tc, th.Tensor):
                task_batch.append(tc)
            else:
                task_batch.append(th.tensor(tc, dtype=th.float32))
            
            if isinstance(ntc, np.ndarray):
                next_task_batch.append(th.tensor(ntc, dtype=th.float32))
            elif isinstance(ntc, th.Tensor):
                next_task_batch.append(ntc)
            else:
                next_task_batch.append(th.tensor(ntc, dtype=th.float32))
        
        task_batch = th.stack(task_batch).to(self.device)
        next_task_batch = th.stack(next_task_batch).to(self.device)
        
        actions = th.tensor(actions, dtype=th.long, device=self.device)
        rewards = th.tensor(rewards, dtype=th.float32, device=self.device)
        dones = th.tensor(dones, dtype=th.float32, device=self.device)
        marks = th.tensor(marks, dtype=th.long, device=self.device)
        
        # 当前 Q 值（分离模式需要传入 task_context）
        all_q_values, _ = self.q_network(obs_batch, task_batch, None)
        q_values = all_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN
        with th.no_grad():
            next_q_online, _ = self.q_network(next_obs_batch, next_task_batch, None)
            next_action4_invalid = self._infer_action4_invalid_mask(
                next_obs_batch,
                next_task_context=next_task_batch,
            )
            next_actions = self._select_next_actions(
                next_q_online,
                action4_invalid_mask=next_action4_invalid,
            )
            next_q_target, _ = self.target_network(next_obs_batch, next_task_batch, None)
            next_q_values = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            targets = rewards + self.gamma * next_q_values * (1 - dones)
        
        # 损失和优化
        q_loss = F.smooth_l1_loss(q_values, targets)
        diffusion_loss = self._diffusion_loss(all_q_values, actions, marks)
        loss = self._combine_losses(q_loss, diffusion_loss)
        
        self.optimizer.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(self.trainable_params, self.max_grad_norm)
        self.optimizer.step()
        
        # 更新目标网络
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        self._update_diffusion_guidance_scale()
        
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        self._record_losses(q_loss, diffusion_loss, loss)
        return loss.item()

    def _diffusion_loss(
        self,
        q_values: th.Tensor,
        actions: th.Tensor,
        marks: Optional[th.Tensor] = None,
    ) -> Optional[th.Tensor]:
        if self.diffusion_prior is None:
            return None
        q_condition = q_values.detach()
        behavior_loss = self.diffusion_prior.training_loss(q_condition, actions)
        q_guided_loss = self._diffusion_q_guided_loss(q_condition, marks)
        return behavior_loss + self.diffusion_q_loss_weight * q_guided_loss

    def _select_next_actions(
        self,
        q_values: th.Tensor,
        action4_invalid_mask: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        if self.diffusion_prior is None or not hasattr(self.action_network, '_guide'):
            guided_q_values = q_values
        else:
            guided_q_values = self.action_network._guide(q_values)
        guided_q_values = self._mask_action4(guided_q_values, action4_invalid_mask)
        return guided_q_values.argmax(dim=1)

    def _mask_action4(
        self,
        q_values: th.Tensor,
        action4_invalid_mask: Optional[th.Tensor],
    ) -> th.Tensor:
        if action4_invalid_mask is None or q_values.size(1) <= 4:
            return q_values
        masked = q_values.clone()
        masked[action4_invalid_mask.bool(), 4] = -1e9
        return masked

    def _infer_action4_invalid_mask(
        self,
        obs,
        next_task_context: Optional[th.Tensor] = None,
    ) -> Optional[th.Tensor]:
        """
        Action 4 is local computation and is invalid once a packet is already
        computed. Infer that flag from the next-state observation so TD targets
        do not bootstrap through an action the simulator will later mask out.
        """
        if self.n_actions <= 4:
            return None

        if next_task_context is not None:
            return next_task_context[:, -1].bool()

        if isinstance(obs, th.Tensor):
            return obs[:, -1].bool()

        if self._use_graph and hasattr(obs, "__getitem__"):
            try:
                agent_feat = obs['agent'].feat
            except (KeyError, AttributeError, TypeError):
                try:
                    agent_feat = obs['agent'].x
                except (KeyError, AttributeError, TypeError):
                    return None
            return agent_feat[:, -1].bool()

        return None

    def _diffusion_q_guided_loss(self, q_values: th.Tensor, marks: Optional[th.Tensor]) -> th.Tensor:
        prior_logits = self.diffusion_prior.denoise_logits(q_values)
        valid_mask = th.ones_like(q_values, dtype=th.bool)
        if marks is not None and q_values.size(1) > 4:
            valid_mask[marks.bool(), 4] = False

        masked_q = q_values.masked_fill(~valid_mask, -1e9)
        masked_prior = prior_logits.masked_fill(~valid_mask, -1e9)
        temperature = max(self.diffusion_q_temperature, 1e-6)

        target_probs = F.softmax(masked_q / temperature, dim=-1)
        log_probs = F.log_softmax(masked_prior, dim=-1)
        return -(target_probs * log_probs).sum(dim=-1).mean()

    def _combine_losses(self, q_loss: th.Tensor, diffusion_loss: Optional[th.Tensor]) -> th.Tensor:
        if diffusion_loss is None:
            return q_loss
        return q_loss + self.diffusion_loss_weight * diffusion_loss

    def _record_losses(
        self,
        q_loss: th.Tensor,
        diffusion_loss: Optional[th.Tensor],
        total_loss: th.Tensor,
    ):
        self.last_losses = {
            'q_loss': q_loss.item(),
            'total_loss': total_loss.item(),
        }
        if diffusion_loss is not None:
            self.last_losses['diffusion_loss'] = diffusion_loss.item()
    
    def save(self, path: str):
        """保存模型"""
        th.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'train_step': self.train_step,
            'use_separated': self._use_separated,
            'controller_type': self.controller_type,
            'use_diffusion': self.use_diffusion,
            'diffusion_prior': self.diffusion_prior.state_dict() if self.diffusion_prior is not None else None,
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = th.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        if self.diffusion_prior is not None and checkpoint.get('diffusion_prior') is not None:
            self.diffusion_prior.load_state_dict(checkpoint['diffusion_prior'])
        try:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        except ValueError:
            print("Warning: optimizer state is incompatible with the current diffusion setting; optimizer was reinitialized.")
        self.train_step = checkpoint.get('train_step', 0)


# 兼容性别名
GNN_DDQN_Agent = GNNDDQNAgent
