"""MEO inter-domain routing policy wrapper."""

from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from .meo_agent import MEODomainRoutingAgent
from .meo_observation import AGG_FEATURE_DIM, GLOBAL_FEATURE_DIM, MAX_MEO_NEIGHBORS, TASK_GLOBAL_FEATURE_DIM, build_meo_observation, meo_state_dim


class MEODomainRewardFunction:
    """Reward dedicated to MEO inter-domain decisions."""

    def __init__(self, cfg: Optional[dict] = None):
        cfg = cfg or {}
        self.success_reward = float(cfg.get("success_reward", 1.0))
        self.failure_penalty = float(cfg.get("failure_penalty", 1.0))
        self.unfinished_penalty = float(cfg.get("unfinished_penalty", 0.0))
        self.domain_hop_penalty = float(cfg.get("domain_hop_penalty", 0.05))
        self.path_hop_penalty = float(cfg.get("path_hop_penalty", 0.0))
        self.score_penalty = float(cfg.get("score_penalty", 1.0))
        self.boundary_load_penalty = float(cfg.get("boundary_load_penalty", 0.2))
        self.boundary_delay_penalty = float(cfg.get("boundary_delay_penalty", 0.1))
        self.compute_penalty = float(cfg.get("compute_penalty", 0.1))
        self.memory_penalty = float(cfg.get("memory_penalty", 1.0))
        self.segment_success_reward = float(cfg.get("segment_success_reward", 0.2))
        self.progress_reward_weight = float(cfg.get("progress_reward_weight", 0.0))
        self.terminal_credit_weight = float(cfg.get("terminal_credit_weight", 0.0))
        self.gamma = float(cfg.get("gamma", 0.97))
        # Keep this at 0 by default so the MEO agent does not reuse the LEO
        # packet reward value as its training target.
        self.terminal_reward_weight = float(cfg.get("terminal_reward_weight", 0.0))

    def __call__(self, trace: Dict, terminal_reward: float = 0.0, done: bool = True, is_compute: bool = False) -> float:  # 根据一次 MEO 决策轨迹和包级终局结果计算 MEO 专用 reward。
        domains = trace.get("domains", []) or []  # 读取本次 MEO 规划经过的域序列，缺失时用空列表。
        transitions = trace.get("transitions", []) or []  # 读取跨域边界链路明细，用于计算边界负载和延迟惩罚。

        domain_hops = max(0, len(domains) - 1)  # 域跳数等于域数量减一，最小为 0，避免空路径得到负数。
        score = float(trace.get("score", 0.0) or 0.0)  # 读取 MEO 规划阶段给出的路径风险/代价分数。
        terminal_reward = float(terminal_reward)  # 将外部传入的包级终局 reward 转成浮点数，仅用于判断成败和可选加权。
        outcome_reward = self.terminal_outcome(
            trace,
            terminal_reward=terminal_reward,
            done=done,
            is_compute=is_compute,
        )
            
        
        boundary_load = 0.0  # 初始化跨域边界链路平均负载，默认没有边界惩罚。
        packet_delay = trace.get("packet_delay", None)  # MEO 时延惩罚优先使用包端到端时延。
        if packet_delay is not None:
            packet_delay = max(0.0, float(packet_delay or 0.0))
        boundary_delay = 0.0  # 旧数据缺少包端到端时延时，继续使用边界链路总延迟兜底。
        if transitions:  # 只有存在跨域跳转记录时，才统计边界链路属性。
            loads = [  # 收集每一段跨域边界链路的负载占用比例。
                float(item.get("link_load_ratio", item.get("normalized_link_load", 0.0)) or 0.0)
                for item in transitions  # 遍历本次 MEO 规划中的每一段跨域跳转。
                if isinstance(item, dict)  # 只处理字典格式的跳转记录，跳过异常数据。
            ]  # 边界链路负载占用比例列表构造结束。
            delays = [  # 收集每一段跨域边界链路的延迟值。
                float(item.get("delay", 0.0) or 0.0)  # 读取单段边界链路延迟，缺失或 None 时按 0 处理。
                for item in transitions  # 遍历本次 MEO 规划中的每一段跨域跳转。
                if isinstance(item, dict)  # 只处理字典格式的跳转记录，跳过异常数据。
            ]  # 边界链路延迟列表构造结束。
            boundary_load = float(np.mean(loads)) if loads else 0.0  # 用平均边界链路占用比例表示跨域出口拥塞程度，空列表时为 0。
            boundary_delay = float(np.sum(delays)) if delays else 0.0  # 用总边界延迟表示跨域传播成本，空列表时为 0。
        delay_penalty_value = packet_delay if packet_delay is not None else boundary_delay

        return (  # 返回 MEO 层专用 reward，由结果项减去各类域间决策成本。
            outcome_reward  # 成功/失败/未终止带来的基础 MEO 层结果奖励。
            - self.score_penalty * score  # 惩罚 MEO 规划器评估出的路径风险/代价。
            - self.domain_hop_penalty * domain_hops  # 惩罚跨越过多 MEO 域，鼓励域级路径更短。
            - self.boundary_load_penalty * boundary_load  # 惩罚边界链路平均负载过高，避免拥塞跨域出口。
            - self.boundary_delay_penalty * delay_penalty_value  # 惩罚包从生成到接收/终止的端到端时延。
        )  # reward 计算结束。

    def terminal_outcome(
        self,
        trace: Dict,
        terminal_reward: float = 0.0,
        done: bool = True,
        is_compute: bool = False,
    ) -> float:
        """Return the packet-level outcome component without segment costs."""
        del is_compute  # Kept for compatibility with the public reward call.
        terminal_reward = float(terminal_reward)
        meo_result = str(trace.get("meo_result", "") or "")
        if not done:
            outcome_reward = -self.unfinished_penalty
        elif meo_result == "reached_computed":
            outcome_reward = self.success_reward
        elif meo_result == "reached_uncomputed":
            outcome_reward = self.success_reward - self.compute_penalty
        elif meo_result == "memory_drop":
            outcome_reward = -self.failure_penalty - self.memory_penalty
        elif meo_result == "link_drop":
            outcome_reward = -self.failure_penalty
        elif terminal_reward > 0.0:
            outcome_reward = self.success_reward
        else:
            outcome_reward = -self.failure_penalty
        return outcome_reward + self.terminal_reward_weight * terminal_reward

    def segment_reward(
        self,
        trace: Dict,
        current_time: Optional[float] = None,
        leo_reward: float = 0.0,
    ) -> float:
        """Reward for reaching the next MEO-domain segment boundary."""
        domains = trace.get("domains", []) or []
        domain_hops = max(0, len(domains) - 1)
        path_hops = max(0, len(trace.get("path", []) or []) - 1)
        boundary_load, boundary_delay = self._boundary_costs(trace.get("transitions", []) or [])
        decision_time = float(trace.get("decision_time", 0.0) or 0.0)
        if current_time is None:
            current_time = trace.get("segment_terminal_time", None)
        if current_time is None:
            segment_delay = boundary_delay
        else:
            segment_delay = max(0.0, float(current_time) - decision_time)
        distance_before = trace.get("distance_before")
        distance_after = trace.get("distance_after")
        progress_reward = 0.0
        if distance_before is not None and distance_after is not None:
            distance_before = float(distance_before)
            distance_after = float(distance_after)
            if np.isfinite(distance_before) and np.isfinite(distance_after):
                phi_before = -distance_before
                phi_after = -distance_after
                progress_reward = self.gamma * phi_after - phi_before
        return (
            self.segment_success_reward
            + self.terminal_reward_weight * float(leo_reward)
            + self.progress_reward_weight * progress_reward
            - self.domain_hop_penalty * domain_hops
            - self.path_hop_penalty * path_hops
            - self.boundary_load_penalty * boundary_load
            - self.boundary_delay_penalty * segment_delay
        )

    @staticmethod
    def _boundary_costs(transitions) -> Tuple[float, float]:
        boundary_load = 0.0
        boundary_delay = 0.0
        if transitions:
            loads = [
                float(item.get("link_load_ratio", item.get("normalized_link_load", 0.0)) or 0.0)
                for item in transitions
                if isinstance(item, dict)
            ]
            delays = [
                float(item.get("delay", 0.0) or 0.0)
                for item in transitions
                if isinstance(item, dict)
            ]
            boundary_load = float(np.mean(loads)) if loads else 0.0
            boundary_delay = float(np.sum(delays)) if delays else 0.0
        return boundary_load, boundary_delay


class MEODomainRouter:
    """Connects MEO DDQN actions to the simulator's domain-path format."""

    def __init__(self, cfg: Optional[dict] = None, device: str = "cpu", transformer_enabled: bool = True):
        self.cfg = cfg or {}
        agent_cfg = self.cfg.get("meo_agent", {}) or {}
        reward_cfg = dict(agent_cfg.get("reward", self.cfg.get("meo_reward", {})) or {})
        reward_cfg.setdefault("gamma", float(agent_cfg.get("gamma", 0.97)))
        self.enabled = bool(self.cfg.get("meo_exit_enabled", False))
        self.transformer_enabled = bool(transformer_enabled)
        self.use_meo_aggregation = bool(self.cfg.get("use_meo_aggregation", True))
        self.max_neighbors = int(agent_cfg.get("max_neighbors", MAX_MEO_NEIGHBORS))
        self.max_domain_hops = int(agent_cfg.get("max_domain_hops", self.cfg.get("plan_max_hops", 30)))
        self.update_every = max(1, int(agent_cfg.get("update_every", 1)))
        self.updates_per_step = max(1, int(agent_cfg.get("updates_per_step", 1)))
        self.model_path = agent_cfg.get("model_path")
        state_dim = meo_state_dim(self.use_meo_aggregation, self.max_neighbors)
        self.agent = MEODomainRoutingAgent(state_dim=state_dim, cfg=agent_cfg, device=str(device))
        self.reward_function = MEODomainRewardFunction(reward_cfg)
        self.agent.load(self.model_path)
        self.decision_count = 0
        self.trace_sequence = 0
        self.update_count = 0
        self.last_update_decision_count = 0
        self.last_loss = None
        self.last_plan = None
        self.interval_rewards = []
        self.interval_losses = []
        self.interval_decisions = 0
        self.interval_experiences = 0
        self.interval_updates = 0

    def recommend_path(
        self,
        meo_satellite,
        src,
        dst,
        packet_size,
        task_type=0,
        computing_demand=0.0,
        size_after_computing=0.0,
        is_computed=False,
        transformer_trainer=None,
        excluded_domains=None,
    ):
        if not self.enabled or meo_satellite is None:
            return None
        graph = getattr(meo_satellite, "inter_domain_graph", None)
        if graph is None or graph.number_of_nodes() == 0:
            return None
        source_domain = self._domain_for(meo_satellite, src)
        target_domain = self._domain_for(meo_satellite, dst)
        if source_domain is None or target_domain is None or source_domain == target_domain:
            return None
        if self.use_meo_aggregation and not self._source_in_current_domain_graph(graph, meo_satellite, src):
            return None
        future_graphs = self._future_graph_samples(meo_satellite) if self.transformer_enabled else [None]
        candidates = []
        task_context = {
            "task_type": task_type,
            "packet_size": packet_size,
            "computing_demand": computing_demand,
            "size_after_computing": size_after_computing,
            "is_computed": is_computed,
        }
        for sample_idx, future_graph in enumerate(future_graphs):
            state, mask, neighbors, domain_distances = build_meo_observation(
                meo_satellite=meo_satellite,
                target_domain=target_domain,
                use_meo_aggregation=self.use_meo_aggregation,
                graph=graph,
                future_graph=future_graph,
                max_neighbors=self.max_neighbors,
                task_context=task_context,
                src=src,
                excluded_domains=excluded_domains,
            )
            # MEO owns only the domain-level target.  The packet-level LEO
            # agent remains responsible for choosing a concrete boundary and
            # the hops used to reach it.
            action, q_score, q_values = self.agent.act(state, mask, explore=True)
            if action < 0 or action >= len(neighbors):
                continue
            next_domain = neighbors[action]
            plan = self._assemble_local_plan(
                meo_satellite=meo_satellite,
                graph=graph,
                current_domain=source_domain,
                next_domain=next_domain,
                target_domain=target_domain,
                src=src,
                dst=dst,
                packet_size=packet_size,
            )
            if plan is None:
                continue
            policy_score = float(q_score)
            plan["meo_policy"] = {
                "state": state,
                "action": int(action),
                "action_mask": mask,
                "neighbors": list(neighbors),
                "q_score": float(q_score),
                "q_values": q_values,
                "sample_idx": int(sample_idx),
                "current_domain": source_domain,
                "next_domain": next_domain,
                "target_domain": target_domain,
                "distance_before": self._domain_distance(graph, source_domain, target_domain),
                "distance_after": domain_distances.get(next_domain),
                "excluded_domains": list(excluded_domains or []),
                "task_context": dict(task_context),
            }
            plan["policy_score"] = policy_score
            candidates.append((policy_score, plan))
        if not candidates:
            return None
        _, best_plan = max(candidates, key=lambda item: item[0])
        best_plan["candidate_count"] = len(candidates)
        best_plan["boundary_sat"] = self._target_boundary_satellites_for_direction(
            graph,
            best_plan["source_domain"],
            best_plan["next_domain"],
        )
        self.last_plan = best_plan
        self.decision_count += 1
        self.interval_decisions += 1
        return best_plan

    def attach_decision(self, packet, plan) -> None:
        policy = plan.get("meo_policy") if isinstance(plan, dict) else None
        if packet is None or policy is None:
            return
        self.trace_sequence += 1
        decision_id = str(policy.get("decision_id") or f"meo-decision-{self.trace_sequence}")
        policy["decision_id"] = decision_id
        trace = {
            "decision_id": decision_id,
            "state": np.asarray(policy["state"], dtype=np.float32),
            "action": int(policy["action"]),
            "action_mask": np.asarray(policy["action_mask"], dtype=np.float32),
            "domains": list(plan.get("domains", [])),
            "transitions": list(plan.get("transitions", [])),
            "path": list(plan.get("path", [])),
            "current_domain": policy.get("current_domain"),
            "next_domain": policy.get("next_domain"),
            "target_domain": policy.get("target_domain"),
            "neighbors": list(policy.get("neighbors", [])),
            "score": float(plan.get("score", 0.0)),
            "policy_score": float(plan.get("policy_score", 0.0)),
            "decision_time": float(plan.get("decision_time", 0.0)),
            "computing_leo": getattr(packet, "computing_leo", None),
            "computing_meo": getattr(packet, "computing_meo", None),
            "computing_waiting_time": float(getattr(packet, "computing_waiting_time", 0.0) or 0.0),
            "meo_result": getattr(packet, "meo_result", None),
            "task_context": dict(policy.get("task_context", {})),
            "distance_before": policy.get("distance_before"),
            "distance_after": policy.get("distance_after"),
        }
        traces = getattr(packet, "meo_decision_traces", None)
        if traces is None:
            traces = []
            packet.meo_decision_traces = traces
        traces.append(trace)
        packet.meo_decision_trace = trace

    def record_executed_boundary(self, packet, meo_satellite, reached_node, previous_node=None) -> bool:
        """Attach the boundary link actually used by the LEO policy to the active MEO trace."""
        if packet is None or meo_satellite is None or reached_node is None:
            return False
        trace = getattr(packet, "meo_decision_trace", None)
        if not isinstance(trace, dict):
            traces = getattr(packet, "meo_decision_traces", None) or []
            trace = traces[-1] if traces else None
        if not isinstance(trace, dict):
            return False

        current_domain = trace.get("current_domain")
        next_domain = trace.get("next_domain")
        graph = getattr(meo_satellite, "inter_domain_graph", None)
        links = self._boundary_links_for_direction(graph, current_domain, next_domain)
        if not links:
            return False

        exact_matches = [
            link for link in links
            if link.get("source_boundary") == previous_node
            and link.get("target_boundary") == reached_node
        ]
        reached_matches = [link for link in links if link.get("target_boundary") == reached_node]
        matches = exact_matches or reached_matches
        if not matches:
            return False
        link = min(matches, key=lambda item: float(item.get("quality_cost", 1.0)))
        link_load = float(link.get("link_load", 0.0) or 0.0)
        quality_cost = float(link.get("quality_cost", 1.0))
        trace["transitions"] = [{
            "from_domain": current_domain,
            "to_domain": next_domain,
            "source_boundary": link.get("source_boundary"),
            "target_boundary": link.get("target_boundary"),
            "quality_cost": quality_cost,
            "link_load": link_load,
            "link_load_ratio": self._link_load_ratio(meo_satellite, link_load),
            "delay": float(link.get("delay", 0.0) or 0.0),
        }]
        trace["score"] = quality_cost
        trace["executed_boundary_source"] = link.get("source_boundary")
        trace["executed_boundary_target"] = link.get("target_boundary")
        return True

    def finish_decision(self, packet, reward, done=True, meo_result=None):
        if packet is not None and meo_result is not None:
            packet.meo_result = meo_result
        traces = getattr(packet, "meo_decision_traces", None)
        if traces is None:
            trace = getattr(packet, "meo_decision_trace", None)
            traces = [trace] if trace else []
        packet_info = getattr(packet, "information", None) or []
        is_computed = bool(packet_info[0]) if packet_info else False
        for trace in traces:
            if not trace:
                continue
            state = trace.get("state")
            action = trace.get("action")
            if state is None or action is None:
                continue
            self._attach_packet_compute_context(packet, trace)
            self._attach_packet_delay_context(packet, trace)
            trace["meo_result"] = meo_result or trace.get("meo_result") or getattr(packet, "meo_result", None)
            meo_reward = self.reward_function(trace, terminal_reward=reward, done=done, is_compute=is_computed)
            self._store_trace_experience(trace, meo_reward, done=done)
        terminal_trace = {
            "meo_result": meo_result or getattr(packet, "meo_result", None),
        }
        terminal_outcome = self.reward_function.terminal_outcome(
            terminal_trace,
            terminal_reward=reward,
            done=done,
            is_compute=is_computed,
        )
        self._apply_terminal_credit(packet, terminal_outcome)
        packet.meo_decision_traces = []
        packet.meo_decision_trace = None

    def finish_segment(self, packet, next_plan=None, reached_node=None, leo_reward=0.0):
        """Store an intermediate MEO experience when a planned segment is reached."""
        traces = getattr(packet, "meo_decision_traces", None)
        if not traces:
            return
        trace = traces.pop()
        if not trace:
            return
        state = trace.get("state")
        action = trace.get("action")
        if state is None or action is None:
            return
        current_time = getattr(packet, "meo_segment_time", None)
        next_policy = next_plan.get("meo_policy") if isinstance(next_plan, dict) else None
        trace["meo_result"] = "segment_reached"
        trace["segment_reached_node"] = reached_node
        if current_time is not None:
            trace["segment_terminal_time"] = float(current_time)
        meo_reward = self.reward_function.segment_reward(
            trace,
            current_time=current_time,
            leo_reward=leo_reward,
        )
        done = next_policy is None
        next_state = None
        next_action_mask = None
        if next_policy is not None:
            next_state = np.asarray(next_policy["state"], dtype=np.float32)
            next_action_mask = np.asarray(next_policy["action_mask"], dtype=np.float32)
        experience = self._store_trace_experience(
            trace,
            meo_reward,
            done=done,
            next_state=next_state,
            next_action_mask=next_action_mask,
        )
        if experience is not None:
            completed = getattr(packet, "meo_completed_experiences", None)
            if completed is None:
                completed = []
                packet.meo_completed_experiences = completed
            completed.append(experience)
        packet.meo_decision_trace = traces[-1] if traces else None

    def _store_trace_experience(self, trace, reward, done, next_state=None, next_action_mask=None):
        state = np.asarray(trace.get("state"), dtype=np.float32)
        if next_state is None:
            next_state = np.zeros_like(state, dtype=np.float32)
        if next_action_mask is None:
            next_action_mask = np.zeros(self.max_neighbors, dtype=np.float32)
        experience = self.agent.store_experience(
            state=state,
            action=trace.get("action"),
            reward=float(reward),
            next_state=next_state,
            done=done,
            next_action_mask=next_action_mask,
        )
        self.interval_rewards.append(float(reward))
        self.interval_experiences += 1
        return experience

    def _apply_terminal_credit(self, packet, terminal_outcome: float) -> None:
        """Patch delayed packet-level credit into all completed segment records."""
        if packet is None:
            return
        experiences = getattr(packet, "meo_completed_experiences", None) or []
        bonus = self.reward_function.terminal_credit_weight * float(terminal_outcome)
        for experience in experiences:
            if isinstance(experience, list) and len(experience) >= 3:
                experience[2] = float(experience[2]) + bonus
        packet.meo_completed_experiences = []

    @staticmethod
    def _domain_distance(graph, source_domain, target_domain):
        if graph is None or source_domain not in graph or target_domain not in graph:
            return None
        try:
            return float(nx.shortest_path_length(graph, source_domain, target_domain))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    @staticmethod
    def _attach_packet_compute_context(packet, trace):
        if packet is None or trace is None:
            return
        computing_leo = getattr(packet, "computing_leo", None) or getattr(packet, "computing_node", None)
        trace["computing_leo"] = computing_leo
        trace["computing_meo"] = getattr(packet, "computing_meo", None)
        trace["computing_waiting_time"] = float(getattr(packet, "computing_waiting_time", 0.0) or 0.0)
        trace["meo_result"] = getattr(packet, "meo_result", trace.get("meo_result"))

    @staticmethod
    def _attach_packet_delay_context(packet, trace):
        if packet is None or trace is None:
            return
        creation_time = getattr(packet, "creation_time", None)
        terminal_time = getattr(packet, "meo_terminal_time", None)
        if creation_time is None or terminal_time is None:
            return
        trace["packet_creation_time"] = float(creation_time)
        trace["packet_terminal_time"] = float(terminal_time)
        trace["packet_delay"] = max(0.0, float(terminal_time) - float(creation_time))

    @staticmethod
    def _neighbor_compute_balance_from_trace(trace):
        state = np.asarray(trace.get("state", []), dtype=np.float32).reshape(-1)
        neighbor_values = state[GLOBAL_FEATURE_DIM + TASK_GLOBAL_FEATURE_DIM:]
        if neighbor_values.size < AGG_FEATURE_DIM or neighbor_values.size % AGG_FEATURE_DIM != 0:
            return {}
        neighbor_rows = neighbor_values.reshape(-1, AGG_FEATURE_DIM)
        action_mask = np.asarray(trace.get("action_mask", []), dtype=bool).reshape(-1)
        neighbors = list(trace.get("neighbors", []) or [])
        compute_balance = neighbor_rows[:, 3]
        result = {}
        for idx, value in enumerate(compute_balance):
            if idx >= len(neighbors):
                break
            if action_mask.size and (idx >= action_mask.size or not action_mask[idx]):
                continue
            result[str(neighbors[idx])] = float(value)
        return result

    def update_if_ready(self):
        if not self.enabled or self.decision_count <= self.last_update_decision_count:
            return None
        if self.decision_count % self.update_every != 0:
            return None
        losses = []
        for _ in range(self.updates_per_step):
            loss = self.agent.update()
            if loss is None:
                continue
            loss = float(loss)
            losses.append(loss)
            self.last_loss = loss
            self.update_count += 1
            self.interval_losses.append(loss)
            self.interval_updates += 1
            self.agent.decay_epsilon()
        self.last_update_decision_count = self.decision_count
        return float(np.mean(losses)) if losses else None

    def format_training_log(self, step=None, total_steps=None, round_idx=None):
        if (
            self.interval_decisions == 0
            and self.interval_experiences == 0
            and not self.interval_rewards
            and not self.interval_losses
        ):
            return None
        rewards = np.asarray(self.interval_rewards, dtype=np.float32)
        losses = np.asarray(self.interval_losses, dtype=np.float32)
        prefix_parts = ["MEO Agent"]
        if round_idx is not None:
            prefix_parts.append(f"round={round_idx}")
        if step is not None and total_steps is not None:
            prefix_parts.append(f"step={step}/{total_steps}")
        elif step is not None:
            prefix_parts.append(f"step={step}")
        reward_text = "reward_avg=None, reward_sum=0.0000, reward_count=0"
        if rewards.size:
            reward_text = (
                f"reward_avg={float(np.mean(rewards)):.4f}, "
                f"reward_sum={float(np.sum(rewards)):.4f}, "
                f"reward_min={float(np.min(rewards)):.4f}, "
                f"reward_max={float(np.max(rewards)):.4f}, "
                f"reward_count={int(rewards.size)}"
            )
        loss_text = "loss_avg=None, loss_latest=None, loss_count=0"
        if losses.size:
            loss_text = (
                f"loss_avg={float(np.mean(losses)):.6f}, "
                f"loss_latest={float(losses[-1]):.6f}, "
                f"loss_count={int(losses.size)}"
            )
        leo_policy_loss = getattr(self.agent, "last_leo_loss", None)
        leo_policy_buffer = len(getattr(self.agent, "leo_replay_buffer", []))
        leo_loss_window_avg = getattr(self.agent, "leo_loss_window_average", None)
        leo_loss_window_count = len(getattr(self.agent, "leo_loss_history", []))
        leo_policy_ready = self.agent.is_leo_policy_ready()
        return (
            f"{' | '.join(prefix_parts)}: "
            f"decisions={self.interval_decisions}, "
            f"experiences={self.interval_experiences}, "
            f"updates={self.interval_updates}, "
            f"{reward_text}, "
            f"{loss_text}, "
            f"replay_buffer={len(self.agent.replay_buffer)}, "
            f"leo_policy_loss_latest={leo_policy_loss}, "
            f"leo_policy_loss_window_avg={leo_loss_window_avg}, "
            f"leo_policy_loss_window_count={leo_loss_window_count}, "
            f"leo_policy_ready={leo_policy_ready}, "
            f"leo_policy_buffer={leo_policy_buffer}, "
            f"epsilon={self.agent.epsilon:.4f}"
        )

    def reset_training_log(self):
        self.interval_rewards = []
        self.interval_losses = []
        self.interval_decisions = 0
        self.interval_experiences = 0
        self.interval_updates = 0

    def save(self):
        self.agent.save(self.model_path)

    def store_leo_policy_experience(
        self,
        leo_satellite,
        task_type=0,
        packet_size=0.0,
        computing_demand=0.0,
        size_after_computing=0.0,
        is_computed=False,
        hops=0,
        destination=None,
        next_hop_target=None,
        compute_node_target=None,
    ) -> bool:
        if leo_satellite is None:
            return False
        source = getattr(leo_satellite, "name", None)
        domain = self._domain_for(leo_satellite, source)
        graph = self._leo_training_graph(leo_satellite, domain)
        if graph is None:
            return False
        task_context = self._normalize_leo_task_context(leo_satellite, {
            "task_type": task_type,
            "packet_size": packet_size,
            "computing_demand": computing_demand,
            "size_after_computing": size_after_computing,
            "is_computed": is_computed,
            "hops": hops,
        })
        edge_target_distances = self._edge_destination_distances(
            leo_satellite,
            graph,
            destination,
        )
        return self.agent.store_leo_experience(
            graph=graph,
            source=source,
            task_context=task_context,
            destination=destination,
            edge_target_distances=edge_target_distances,
            next_hop_target=next_hop_target,
            compute_node_target=compute_node_target,
        )

    def _leo_training_graph(self, leo_satellite, domain):
        graph = getattr(leo_satellite, "inter_domain_graph", None)
        if graph is None:
            master = getattr(leo_satellite, "masterMeo", None)
            propagator = getattr(leo_satellite, "propagator", None)
            master_sat = propagator.satellites.get(master) if propagator is not None and master is not None else None
            graph = getattr(master_sat, "inter_domain_graph", None)
        intra_graph = self._domain_intra_graph(graph, domain)
        if intra_graph is not None:
            return intra_graph

        propagator = getattr(leo_satellite, "propagator", None)
        topology = getattr(propagator, "graph", None)
        if topology is None:
            return None
        nodes = [
            node for node in topology.nodes
            if self._domain_for(leo_satellite, node) == domain
        ]
        if not nodes:
            return None
        return topology.subgraph(nodes).copy()

    def _build_leo_policy_context(
        self,
        meo_satellite,
        graph,
        neighbors,
        source_domain,
        target_domain,
        src,
        dst,
        task_context,
    ):
        intra_graph = self._domain_intra_graph(graph, source_domain)
        candidate_exits = {}
        edge_target_distances = {}
        src_satellite = self._satellite_by_name(meo_satellite, src) or meo_satellite
        for neighbor in neighbors:
            if graph is None or source_domain not in graph or neighbor not in graph or not graph.has_edge(source_domain, neighbor):
                candidate_exits[neighbor] = None
                continue
            link = self._best_boundary_link(graph, source_domain, neighbor)
            candidate_exits[neighbor] = link.get("source_boundary") if link is not None else None
            edge_target_distances[neighbor] = self._edge_destination_distances(
                src_satellite,
                intra_graph,
                candidate_exits[neighbor],
            )
        return {
            "graph": graph,
            "neighbors": list(neighbors),
            "source_domain": source_domain,
            "target_domain": target_domain,
            "src": src,
            "dst": dst,
            "task_context": self._normalize_leo_task_context(src_satellite, task_context),
            "candidate_exits": candidate_exits,
            "intra_graph": intra_graph,
            "edge_target_distances": edge_target_distances,
        }

    @staticmethod
    def _normalize_leo_task_context(satellite, task_context):
        task_context = dict(task_context or {})
        max_size = max(float(getattr(satellite, "max_size", 1.0) or 1.0), 1e-9)
        computing_ability = max(float(getattr(satellite, "computing_ability", 1.0) or 1.0), 1e-9)
        max_hop = max(float(getattr(satellite, "max_hop", 1.0) or 1.0), 1.0)
        return {
            "task_type": float(task_context.get("task_type", 0.0) or 0.0),
            "packet_size": float(task_context.get("packet_size", 0.0) or 0.0) / max_size,
            "computing_demand": float(task_context.get("computing_demand", 0.0) or 0.0) / computing_ability,
            "size_after_computing": float(task_context.get("size_after_computing", 0.0) or 0.0) / max_size,
            "hops": float(task_context.get("hops", 0.0) or 0.0) / max_hop,
            "is_computed": float(bool(task_context.get("is_computed", False))),
        }

    @staticmethod
    def _edge_destination_distances(reference_satellite, graph, target):
        if graph is None:
            return {}
        propagator = getattr(reference_satellite, "propagator", None)
        satellites = getattr(propagator, "satellites", {}) or {}
        directions = list(graph.edges())
        if not graph.is_directed():
            directions.extend((dst, src) for src, dst in list(graph.edges()))
        distances = {}
        for src, dst in directions:
            source_satellite = satellites.get(src)
            max_hop = max(float(getattr(source_satellite, "max_hop", getattr(reference_satellite, "max_hop", 1.0)) or 1.0), 1.0)
            distance = None
            neighbor_hops = getattr(source_satellite, "neighbor_hops", {}) or {}
            if target is not None:
                distance = (neighbor_hops.get(dst, {}) or {}).get(target)
            if distance is None and target in graph and dst in graph:
                try:
                    distance = nx.shortest_path_length(graph, dst, target)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    distance = None
            distances[(src, dst)] = float(distance) / max_hop if distance is not None else 2.0
        return distances

    @staticmethod
    def _domain_intra_graph(graph, domain):
        if graph is None or domain not in graph:
            return None
        aggregate = graph.nodes[domain].get("aggregate", {})
        if not isinstance(aggregate, dict):
            return None
        return aggregate.get("intra_graph")

    @staticmethod
    def _apply_leo_rollout_to_plan(plan, rollout):
        if not rollout or not rollout.get("reached_target"):
            if rollout:
                plan["leo_policy_rollout"] = rollout
            return
        rollout_path = list(rollout.get("path", []) or [])
        if len(rollout_path) < 1:
            plan["leo_policy_rollout"] = rollout
            return
        transition = plan.get("transitions", [{}])[0] if plan.get("transitions") else {}
        next_entry = transition.get("target_boundary")
        old_path = list(plan.get("path", []) or [])
        suffix = []
        if next_entry in old_path:
            suffix = old_path[old_path.index(next_entry):]
        elif next_entry is not None:
            suffix = [next_entry]
        new_path = list(rollout_path)
        if suffix:
            if new_path and new_path[-1] == suffix[0]:
                new_path.extend(suffix[1:])
            else:
                new_path.extend(suffix)
        plan["path"] = new_path
        compute_flags = list(rollout.get("compute_flags", []) or [])
        if len(compute_flags) < len(new_path):
            compute_flags.extend([0] * (len(new_path) - len(compute_flags)))
        plan["compute_flags"] = compute_flags[:len(new_path)]
        plan["computing_leo"] = rollout.get("compute_node")
        plan["leo_policy_rollout"] = rollout

    def _future_graph_samples(self, meo_satellite):
        if not hasattr(meo_satellite, "_predict_future_graph_samples"):
            return [None]
        future_loads_map, _ = meo_satellite._predict_future_graph_samples()
        if not future_loads_map:
            return [None]
        first_time = sorted(future_loads_map.keys())[0]
        samples = [graph for graph in future_loads_map.get(first_time, []) if isinstance(graph, nx.Graph)]
        return samples or [None]

    def _assemble_local_plan(self, meo_satellite, graph, current_domain, next_domain, target_domain, src, dst, packet_size):  # 根据当前域和策略选择的下一域，组装只包含域级目标的局部计划。
        if current_domain == next_domain or not graph.has_edge(current_domain, next_domain):  # 当前域与下一域相同或两域之间没有边时，无法执行有效的跨域转发。
            return None  # 返回空值，表示本次局部路由计划组装失败。
        boundary_satellites = self._target_boundary_satellites_for_direction(graph, current_domain, next_domain)
        if not boundary_satellites:
            return None
        domain_entries = {
            current_domain: {"entry": src, "exit": None},
            next_domain: {"entry": None, "exit": dst if next_domain == target_domain else None},
        }
        env = getattr(meo_satellite, "env", None)  # 获取仿真环境对象，以便读取当前决策发生的仿真时间。
        return {  # 返回供后续转发、策略记录和奖励计算使用的局部路由计划。
            "domains": [current_domain, next_domain],  # 保存此次局部动作覆盖的当前域和下一域。
            "transitions": [],  # 具体边界链路由 LEO 执行完成后回写。
            "domain_entries": domain_entries,  # 保存各域对应的入口与出口卫星节点。
            "path": [],  # 域内路径由 LEO Agent 逐跳决定。
            "boundary_sat": boundary_satellites,
            "source": src,  # 保存数据包的源卫星节点。
            "destination": dst,  # 保存数据包的最终目的卫星节点。
            "source_domain": current_domain,  # 保存本次局部计划开始时所在的 MEO 域。
            "target_domain": target_domain,  # 保存数据包最终需要到达的目标 MEO 域。
            "next_domain": next_domain,  # 保存 MEO 策略为本次动作选择的下一跳域。
            "packet_size": float(packet_size or 0.0),  # 保存浮点格式的数据包大小，空值按 0.0 处理。
            "score": 0.0,  # 执行前不为 MEO 预选任何边界链路。
            "decision_time": float(getattr(env, "now", 0.0) if env is not None else 0.0),  # 保存当前仿真时刻；环境不存在时使用 0.0。
            "reachable": True,  # 标记该计划已经通过必要的边界链路和域内路径可达性检查。
            "local_only": True,  # 标记该计划只描述当前域到下一域的一次局部跨域动作。
        }  # 局部路由计划组装完成。

    def _assemble_plan(self, meo_satellite, graph, domain_path, src, dst, packet_size):
        transitions = []
        score = 0.0
        for domain_a, domain_b in zip(domain_path[:-1], domain_path[1:]):
            if not graph.has_edge(domain_a, domain_b):
                return None
            link = self._best_boundary_link(graph, domain_a, domain_b)
            if link is None:
                return None
            cost = float(link.get("quality_cost", 1.0))
            score += cost
            link_load = float(link.get("link_load", 0.0) or 0.0)
            transitions.append({
                "from_domain": domain_a,
                "to_domain": domain_b,
                "source_boundary": link.get("source_boundary"),
                "target_boundary": link.get("target_boundary"),
                "quality_cost": cost,
                "link_load": link_load,
                "link_load_ratio": self._link_load_ratio(meo_satellite, link_load),
                "delay": float(link.get("delay", 0.0) or 0.0),
            })
        domain_entries = {}
        for idx, domain in enumerate(domain_path):
            entry = src if idx == 0 else transitions[idx - 1]["target_boundary"]
            exit_node = dst if idx == len(domain_path) - 1 else transitions[idx]["source_boundary"]
            domain_entries[domain] = {"entry": entry, "exit": exit_node}

        return {
            "domains": list(domain_path),
            "transitions": transitions,
            "domain_entries": domain_entries,
            "source": src,
            "destination": dst,
            "source_domain": domain_path[0],
            "target_domain": domain_path[-1],
            "packet_size": float(packet_size or 0.0),
            "score": float(score),
            "reachable": True,
        }

    def _intra_domain_path(self, meo_satellite, graph, domain, entry, exit_node):
        if entry == exit_node:
            return [entry]
        aggregate = graph.nodes[domain].get("aggregate", {}) if domain in graph else {}
        intra_graph = aggregate.get("intra_graph") if isinstance(aggregate, dict) else None
        if intra_graph is not None and entry in intra_graph and exit_node in intra_graph:
            try:
                return nx.shortest_path(intra_graph, entry, exit_node, weight="weight")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass
        topology = getattr(getattr(meo_satellite, "propagator", None), "graph", None)
        if topology is None:
            return None
        domain_nodes = [
            node for node in topology.nodes
            if self._domain_for(meo_satellite, node) == domain
        ]
        domain_graph = topology.subgraph(domain_nodes)
        if entry not in domain_graph or exit_node not in domain_graph:
            return None
        try:
            return nx.shortest_path(domain_graph, entry, exit_node)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def _edge_weight(self, domain_a, domain_b, attrs):
        links = attrs.get("boundary_links", {}) or {}
        costs = [float(link.get("quality_cost", 1.0)) for link in links.values() if isinstance(link, dict)]
        return min(costs) if costs else 1.0

    @staticmethod
    def _link_load_ratio(meo_satellite, link_load):
        capacity = float(getattr(meo_satellite, "transmission_rate", 1.0) or 1.0)
        if capacity <= 0.0:
            capacity = 1.0
        return float(np.clip(float(link_load or 0.0) / capacity, 0.0, 1.0))

    def _best_boundary_link(self, graph, domain_a, domain_b):
        candidates = self._boundary_links_for_direction(graph, domain_a, domain_b)
        if not candidates:
            return None
        return min(candidates, key=lambda item: float(item.get("quality_cost", 1.0)))

    @staticmethod
    def _boundary_links_for_direction(graph, domain_a, domain_b):
        if graph is None or domain_a is None or domain_b is None or not graph.has_edge(domain_a, domain_b):
            return []
        links = graph[domain_a][domain_b].get("boundary_links", {}) or {}
        candidates = []
        for link in links.values():
            if not isinstance(link, dict):
                continue
            src_domain = link.get("source_domain")
            dst_domain = link.get("target_domain")
            if src_domain == domain_a and dst_domain == domain_b:
                directional_link = dict(link)
            elif src_domain == domain_b and dst_domain == domain_a:
                directional_link = dict(link)
                directional_link["source_domain"] = domain_a
                directional_link["target_domain"] = domain_b
                directional_link["source_boundary"] = link.get("target_boundary")
                directional_link["target_boundary"] = link.get("source_boundary")
            else:
                continue
            if directional_link.get("source_boundary") is None or directional_link.get("target_boundary") is None:
                continue
            candidates.append(directional_link)
        return candidates

    @staticmethod
    def _target_boundary_satellites_for_direction(graph, domain_a, domain_b):
        boundary_satellites = []
        for link in MEODomainRouter._boundary_links_for_direction(graph, domain_a, domain_b):
            boundary_satellite = link.get("target_boundary")
            if boundary_satellite is not None and boundary_satellite not in boundary_satellites:
                boundary_satellites.append(boundary_satellite)
        return boundary_satellites

    @staticmethod
    def _source_in_current_domain_graph(graph, meo_satellite, src):
        current_domain = getattr(meo_satellite, "name", None)
        if graph is None or current_domain not in graph:
            return False
        aggregate = graph.nodes[current_domain].get("aggregate", {}) or {}
        intra_graph = aggregate.get("intra_graph")
        return intra_graph is not None and src in intra_graph

    def _domain_for(self, meo_satellite, name):
        if name is None:
            return None
        graph = getattr(meo_satellite, "inter_domain_graph", None)
        if graph is not None and name in graph:
            return name
        propagator = getattr(meo_satellite, "propagator", None)
        satellite = propagator.satellites.get(name) if propagator is not None else None
        if satellite is not None:
            return getattr(satellite, "masterMeo", None)
        if name == getattr(meo_satellite, "name", None):
            return name
        return None

    @staticmethod
    def _satellite_by_name(reference_satellite, name):
        propagator = getattr(reference_satellite, "propagator", None)
        satellites = getattr(propagator, "satellites", {}) or {}
        return satellites.get(name)
