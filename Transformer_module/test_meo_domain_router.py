from types import SimpleNamespace

import networkx as nx
import numpy as np
import torch

from Transformer_module import MEODomainRouter
from Transformer_module.meo_agent import MEODomainRoutingAgent
from Transformer_module.meo_router import MEODomainRewardFunction
from Transformer_module.meo_observation import (
    AGG_FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
    TASK_GLOBAL_FEATURE_DIM,
    UNREACHABLE_BOUNDARY_DELAY,
    build_meo_observation,
    meo_state_dim,
)


def _intra_graph(*nodes):
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    for src, dst in zip(nodes[:-1], nodes[1:]):
        graph.add_edge(src, dst, weight=0.1, delay=0.1)
    return graph


def _domain_graph():
    graph = nx.Graph()
    for domain, members in {
        "MEO_A": ["a0", "a1"],
        "MEO_B": ["b0", "b1"],
        "MEO_C": ["c0"],
        "MEO_D": ["d0", "d1"],
    }.items():
        graph.add_node(
            domain,
            aggregate={
                "members": members,
                "boundary_nodes": members[-1:],
                "intra_graph": _intra_graph(*members),
                "avg_memory_occupancy_rate": 0.2,
                "max_memory_occupancy_rate": 0.3,
                "avg_computing_occupancy_rate": 0.4,
                "max_computing_occupancy_rate": 0.5,
            },
            avg_memory_occupancy_rate=0.2,
            max_memory_occupancy_rate=0.3,
            avg_computing_occupancy_rate=0.4,
            max_computing_occupancy_rate=0.5,
        )
    _add_boundary(graph, "MEO_A", "MEO_B", "a1", "b0", 0.2)
    _add_boundary(graph, "MEO_B", "MEO_C", "b1", "c0", 0.2)
    _add_boundary(graph, "MEO_A", "MEO_D", "a1", "d0", 0.4)
    _add_boundary(graph, "MEO_D", "MEO_C", "d1", "c0", 0.4)
    return graph


def _add_boundary(graph, domain_a, domain_b, source_boundary, target_boundary, cost):
    graph.add_edge(domain_a, domain_b, boundary_links={
        (source_boundary, target_boundary): {
            "source_domain": domain_a,
            "target_domain": domain_b,
            "source_boundary": source_boundary,
            "target_boundary": target_boundary,
            "quality_cost": cost,
            "link_load": cost,
            "delay": cost,
        }
    })


class _FakeLEOPolicy:
    def __init__(self, responses):
        if callable(responses):
            self.responses = responses
        else:
            self.responses = dict(responses) if isinstance(responses, dict) else list(responses)
        self.calls = []

    def predict(self, graph, source, task_context=None, device=None, **kwargs):
        self.calls.append((source, dict(task_context or {}), dict(kwargs)))
        if callable(self.responses):
            return self.responses(graph, source, task_context or {})
        if isinstance(self.responses, dict) and source in self.responses:
            return self.responses[source]
        if isinstance(self.responses, list) and self.responses:
            return self.responses.pop(0)
        successors = list(graph.successors(source)) if source in graph else []
        return {
            "next_hop": successors[0] if successors else source,
            "compute_node": None,
        }


def _force_meo_action(agent, action):
    for param in agent.online_net.parameters():
        param.data.zero_()
    last_linear = agent.online_net.net[-1]
    last_linear.bias.data.fill_(-1.0)
    last_linear.bias.data[int(action)] = 1.0


def _meo_satellite():
    graph = _domain_graph()
    satellites = {}
    for domain, members in {
        "MEO_A": ["a0", "a1"],
        "MEO_B": ["b0", "b1"],
        "MEO_C": ["c0"],
        "MEO_D": ["d0", "d1"],
    }.items():
        satellites[domain] = SimpleNamespace(name=domain, isLeo=False, masterMeo=domain)
        for leo in members:
            satellites[leo] = SimpleNamespace(
                name=leo,
                isLeo=True,
                masterMeo=domain,
                memory=10.0,
                max_size=10.0,
                computing_ability=10.0,
                max_hop=10,
                neighbor_hops={},
            )
    propagator = SimpleNamespace(satellites=satellites, graph=nx.DiGraph())
    propagator.graph.add_nodes_from([name for name, sat in satellites.items() if getattr(sat, "isLeo", False)])
    propagator.graph.add_edges_from([("a0", "a1"), ("b0", "b1"), ("d0", "d1")])
    satellites["a0"].neighbor_hops = {"a1": {"a1": 0, "c0": 2}}
    leo_states = {
        "a0": {"remaining_memory": 8.0, "remaining_computing": 7.0, "is_producing": 0, "neighbors": []},
        "a1": {"remaining_memory": 6.0, "remaining_computing": 5.0, "is_producing": 1, "neighbors": [{"name": "b0", "link_load": 1.0}]},
    }
    remote = {
        "MEO_B": {
            "b0": {"remaining_memory": 7.0, "remaining_computing": 6.0, "is_producing": 0, "neighbors": []},
            "b1": {"remaining_memory": 5.0, "remaining_computing": 4.0, "is_producing": 0, "neighbors": [{"name": "c0", "link_load": 1.0}]},
        },
        "MEO_D": {
            "d0": {"remaining_memory": 7.0, "remaining_computing": 6.0, "is_producing": 0, "neighbors": []},
            "d1": {"remaining_memory": 5.0, "remaining_computing": 4.0, "is_producing": 0, "neighbors": [{"name": "c0", "link_load": 1.0}]},
        },
    }
    return SimpleNamespace(
        name="MEO_A",
        inter_domain_graph=graph,
        neighbors=["MEO_B", "MEO_D"],
        propagator=propagator,
        leoStates=leo_states,
        remote_domain_leo_states=remote,
        memory=10.0,
        max_size=10.0,
        computing_ability=10.0,
        transmission_rate=10.0,
        env=SimpleNamespace(now=0.0),
        _predict_future_graph_samples=lambda: ({1.0: [nx.Graph(), nx.Graph()]}, None),
    )


def _meo_b_satellite():
    meo = _meo_satellite()
    meo.name = "MEO_B"
    meo.neighbors = ["MEO_A", "MEO_C"]
    meo.leoStates = {
        "b0": {"remaining_memory": 7.0, "remaining_computing": 6.0, "is_producing": 0, "neighbors": []},
        "b1": {"remaining_memory": 5.0, "remaining_computing": 4.0, "is_producing": 0, "neighbors": [{"name": "c0", "link_load": 1.0}]},
    }
    meo.remote_domain_leo_states = {
        "MEO_A": {
            "a0": {"remaining_memory": 8.0, "remaining_computing": 7.0, "is_producing": 0, "neighbors": []},
            "a1": {"remaining_memory": 6.0, "remaining_computing": 5.0, "is_producing": 1, "neighbors": [{"name": "b0", "link_load": 1.0}]},
        },
        "MEO_C": {
            "c0": {"remaining_memory": 7.0, "remaining_computing": 6.0, "is_producing": 0, "neighbors": []},
        },
    }
    return meo


def test_meo_observation_shapes_and_mask():
    meo = _meo_satellite()
    for use_agg in (True, False):
        kwargs = {"src": "a0"} if use_agg else {}
        state, mask, neighbors, _ = build_meo_observation(
            meo,
            "MEO_C",
            use_meo_aggregation=use_agg,
            **kwargs,
        )
        assert state.shape == (meo_state_dim(use_agg),)
        assert mask.tolist()[:2] == [1.0, 1.0]
        assert mask.tolist()[2:] == [0.0, 0.0]
        assert neighbors == ["MEO_B", "MEO_D"]


def test_meo_observation_uses_only_inter_domain_graph_reachable_neighbors():
    meo = _meo_satellite()
    meo.neighbors = ["MEO_B", "MEO_D", "MEO_C", "Satellite_10000_4_5"]
    meo.propagator.satellites["Satellite_10000_4_5"] = SimpleNamespace(
        name="Satellite_10000_4_5",
        isLeo=False,
        masterMeo="Satellite_10000_4_5",
    )

    _, mask, neighbors, _ = build_meo_observation(
        meo,
        "MEO_C",
        use_meo_aggregation=True,
        src="a0",
    )

    assert neighbors == ["MEO_B", "MEO_D"]
    assert "MEO_C" not in neighbors
    assert "Satellite_10000_4_5" not in neighbors
    assert mask.tolist()[:2] == [1.0, 1.0]


def test_meo_reward_uses_packet_end_to_end_delay_over_boundary_delay():
    reward_fn = MEODomainRewardFunction({
        "success_reward": 1.0,
        "score_penalty": 0.0,
        "domain_hop_penalty": 0.0,
        "boundary_load_penalty": 0.0,
        "boundary_delay_penalty": 0.1,
    })
    trace = {
        "domains": ["MEO_A", "MEO_B"],
        "transitions": [{"delay": 2.0}],
        "meo_result": "reached_computed",
        "packet_delay": 10.0,
    }

    assert reward_fn(trace, terminal_reward=1.0, done=True) == 0.0


def test_meo_segment_reward_uses_segment_delay_instead_of_packet_delay():
    reward_fn = MEODomainRewardFunction({
        "segment_success_reward": 0.5,
        "score_penalty": 0.0,
        "domain_hop_penalty": 0.0,
        "boundary_load_penalty": 0.0,
        "boundary_delay_penalty": 0.1,
    })
    trace = {
        "domains": ["MEO_A", "MEO_B"],
        "transitions": [{"delay": 2.0}],
        "decision_time": 4.0,
        "packet_delay": 100.0,
    }

    assert np.isclose(reward_fn.segment_reward(trace, current_time=7.0), 0.2)


def test_meo_segment_potential_reward_prefers_progress_without_congestion():
    reward_fn = MEODomainRewardFunction({
        "segment_success_reward": 0.0,
        "progress_reward_weight": 0.2,
        "gamma": 0.97,
        "domain_hop_penalty": 0.0,
        "path_hop_penalty": 0.0,
        "boundary_load_penalty": 0.0,
        "boundary_delay_penalty": 0.0,
    })

    closer = reward_fn.segment_reward({"distance_before": 3.0, "distance_after": 2.0})
    farther = reward_fn.segment_reward({"distance_before": 3.0, "distance_after": 4.0})

    assert np.isclose(closer, 0.2 * 1.06)
    assert np.isclose(farther, 0.2 * -0.88)
    assert closer > farther


def test_meo_segment_congestion_cost_can_outweigh_distance_progress():
    reward_fn = MEODomainRewardFunction({
        "segment_success_reward": 0.0,
        "progress_reward_weight": 0.2,
        "gamma": 0.97,
        "domain_hop_penalty": 0.0,
        "path_hop_penalty": 0.0,
        "boundary_load_penalty": 0.2,
        "boundary_delay_penalty": 0.08,
    })
    congested_progress = {
        "distance_before": 3.0,
        "distance_after": 2.0,
        "decision_time": 0.0,
        "transitions": [{"link_load_ratio": 1.0}],
    }
    uncongested_detour = {
        "distance_before": 3.0,
        "distance_after": 4.0,
        "decision_time": 0.0,
        "transitions": [{"link_load_ratio": 0.0}],
    }

    congested_reward = reward_fn.segment_reward(congested_progress, current_time=10.0)
    detour_reward = reward_fn.segment_reward(uncongested_detour, current_time=0.0)

    assert detour_reward > congested_reward


def test_meo_observation_includes_task_features_and_domain_compute_pressure():
    meo = _meo_satellite()

    state, _, neighbors, _ = build_meo_observation(
        meo,
        "MEO_C",
        use_meo_aggregation=True,
        task_context={
            "task_type": 1,
            "packet_size": 2.0,
            "computing_demand": 3.0,
            "size_after_computing": 1.0,
            "is_computed": False,
        },
        src="a0",
    )

    task_start = GLOBAL_FEATURE_DIM
    task_end = task_start + TASK_GLOBAL_FEATURE_DIM
    assert np.allclose(state[task_start:task_end], [1.0, 0.2, 0.3, 0.1, 0.0])
    first_neighbor_start = task_end
    first_delay_index = first_neighbor_start + 4
    first_pressure_index = first_neighbor_start + AGG_FEATURE_DIM
    assert neighbors[0] == "MEO_B"
    assert np.isclose(state[first_delay_index], 0.3)
    assert np.isclose(state[first_pressure_index], 0.5)


def test_meo_observation_defaults_missing_task_features_to_zero():
    meo = _meo_satellite()

    state, _, _, _ = build_meo_observation(meo, "MEO_C", use_meo_aggregation=True, src="a0")

    task_start = GLOBAL_FEATURE_DIM
    task_end = task_start + TASK_GLOBAL_FEATURE_DIM
    assert np.allclose(state[task_start:task_end], np.zeros(TASK_GLOBAL_FEATURE_DIM))


def test_meo_observation_aggregation_requires_source_and_intra_graph():
    meo = _meo_satellite()

    raised = False
    try:
        build_meo_observation(meo, "MEO_C", use_meo_aggregation=True)
    except ValueError:
        raised = True
    assert raised


def test_meo_observation_masks_excluded_domains_without_reordering_neighbors():
    meo = _meo_satellite()

    _, mask, neighbors, _ = build_meo_observation(
        meo,
        "MEO_C",
        use_meo_aggregation=True,
        src="a0",
        excluded_domains={"MEO_B"},
    )

    assert neighbors == ["MEO_B", "MEO_D"]
    assert mask.tolist()[:2] == [0.0, 1.0]
    assert mask.tolist()[2:] == [0.0, 0.0]


def test_meo_observation_masks_unreachable_boundary_link_with_finite_delay():
    meo = _meo_satellite()
    meo.inter_domain_graph["MEO_A"]["MEO_B"]["boundary_links"] = {}

    state, mask, neighbors, _ = build_meo_observation(
        meo,
        "MEO_C",
        use_meo_aggregation=True,
        src="a0",
    )

    task_end = GLOBAL_FEATURE_DIM + TASK_GLOBAL_FEATURE_DIM
    first_delay_index = task_end + 4
    assert neighbors == ["MEO_B", "MEO_D"]
    assert mask.tolist()[:2] == [0.0, 1.0]
    assert np.isclose(state[first_delay_index], UNREACHABLE_BOUNDARY_DELAY)


def test_meo_observation_combines_excluded_and_unreachable_masks():
    meo = _meo_satellite()
    meo.inter_domain_graph["MEO_A"]["MEO_B"]["boundary_links"] = {}

    _, mask, neighbors, _ = build_meo_observation(
        meo,
        "MEO_C",
        use_meo_aggregation=True,
        src="a0",
        excluded_domains={"MEO_D"},
    )

    assert neighbors == ["MEO_B", "MEO_D"]
    assert mask.tolist()[:2] == [0.0, 0.0]


def test_meo_agent_does_not_select_masked_highest_q_action():
    agent = MEODomainRoutingAgent(
        state_dim=3,
        cfg={"n_actions": 4, "epsilon": 0.0},
        device="cpu",
    )
    _force_meo_action(agent, 0)

    action, q_score, q_values = agent.act(
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        explore=False,
    )

    assert action == 1
    assert q_values[0] == -np.inf
    assert q_score == q_values[1]


def test_meo_router_excluded_domain_cannot_be_selected():
    router = MEODomainRouter({
        "meo_exit_enabled": True,
        "enabled": True,
        "use_meo_aggregation": True,
        "meo_agent": {"epsilon": 0.0, "batch_size": 1, "buffer_size": 10},
    }, device="cpu", transformer_enabled=False)
    meo = _meo_satellite()

    plan = router.recommend_path(
        meo,
        "a0",
        "c0",
        packet_size=1.0,
        excluded_domains={"MEO_B"},
    )

    assert plan is not None
    assert plan["meo_policy"]["neighbors"] == ["MEO_B", "MEO_D"]
    assert plan["meo_policy"]["action_mask"].tolist()[:2] == [0.0, 1.0]
    assert plan["next_domain"] == "MEO_D"
    assert plan["domains"] == ["MEO_A", "MEO_D"]
    assert plan["boundary_sat"] == ["d0"]


def test_boundary_satellites_include_all_next_domain_candidates():
    graph = nx.Graph()
    graph.add_edge("MEO_A", "MEO_B", boundary_links={
        ("a1", "b0"): {
            "source_domain": "MEO_A",
            "target_domain": "MEO_B",
            "source_boundary": "a1",
            "target_boundary": "b0",
        },
        ("a2", "b1"): {
            "source_domain": "MEO_A",
            "target_domain": "MEO_B",
            "source_boundary": "a2",
            "target_boundary": "b1",
        },
        ("a3", "b0"): {
            "source_domain": "MEO_A",
            "target_domain": "MEO_B",
            "source_boundary": "a3",
            "target_boundary": "b0",
        },
        ("a4", None): {
            "source_domain": "MEO_A",
            "target_domain": "MEO_B",
            "source_boundary": "a4",
            "target_boundary": None,
        },
        "invalid": None,
    })

    result = MEODomainRouter._target_boundary_satellites_for_direction(graph, "MEO_A", "MEO_B")

    assert result == ["b0", "b1"]


def test_boundary_satellites_reverse_stored_link_direction():
    graph = nx.Graph()
    graph.add_edge("MEO_A", "MEO_B", boundary_links={
        ("b0", "a1"): {
            "source_domain": "MEO_B",
            "target_domain": "MEO_A",
            "source_boundary": "b0",
            "target_boundary": "a1",
        },
    })

    result = MEODomainRouter._target_boundary_satellites_for_direction(graph, "MEO_A", "MEO_B")

    assert result == ["b0"]


def test_meo_router_does_not_select_unreachable_boundary_link():
    router = MEODomainRouter({
        "meo_exit_enabled": True,
        "enabled": True,
        "use_meo_aggregation": True,
        "meo_agent": {"epsilon": 0.0, "batch_size": 1, "buffer_size": 10},
    }, device="cpu", transformer_enabled=False)
    meo = _meo_satellite()
    meo.inter_domain_graph["MEO_A"]["MEO_B"]["boundary_links"] = {}

    plan = router.recommend_path(meo, "a0", "c0", packet_size=1.0)

    assert plan is not None
    assert plan["meo_policy"]["neighbors"] == ["MEO_B", "MEO_D"]
    assert plan["meo_policy"]["action_mask"].tolist()[:2] == [0.0, 1.0]
    assert plan["next_domain"] == "MEO_D"


def test_meo_router_returns_none_when_source_missing_from_current_intra_graph():
    router = MEODomainRouter({
        "meo_exit_enabled": True,
        "enabled": True,
        "use_meo_aggregation": True,
        "meo_agent": {"epsilon": 0.0, "batch_size": 1, "buffer_size": 10},
    }, device="cpu", transformer_enabled=False)
    meo = _meo_satellite()
    meo.inter_domain_graph.nodes["MEO_A"]["aggregate"]["intra_graph"].remove_node("a0")

    plan = router.recommend_path(meo, "a0", "c0", packet_size=1.0)

    assert plan is None


def test_meo_router_second_plan_masks_previous_domain():
    router = MEODomainRouter({
        "meo_exit_enabled": True,
        "enabled": True,
        "use_meo_aggregation": True,
        "meo_agent": {"epsilon": 0.0, "batch_size": 1, "buffer_size": 10},
    }, device="cpu", transformer_enabled=False)
    meo = _meo_b_satellite()

    plan = router.recommend_path(
        meo,
        "b0",
        "c0",
        packet_size=1.0,
        excluded_domains={"MEO_A"},
    )

    assert plan is not None
    assert plan["meo_policy"]["neighbors"] == ["MEO_A", "MEO_C"]
    assert plan["meo_policy"]["action_mask"].tolist()[:2] == [0.0, 1.0]
    assert plan["next_domain"] == "MEO_C"
    assert plan["domains"] == ["MEO_B", "MEO_C"]


def test_meo_router_returns_none_when_all_actions_are_masked():
    router = MEODomainRouter({
        "meo_exit_enabled": True,
        "enabled": True,
        "use_meo_aggregation": True,
        "meo_agent": {"epsilon": 0.0, "batch_size": 1, "buffer_size": 10},
    }, device="cpu", transformer_enabled=False)
    meo = _meo_satellite()

    plan = router.recommend_path(
        meo,
        "a0",
        "c0",
        packet_size=1.0,
        excluded_domains={"MEO_B", "MEO_D"},
    )

    assert plan is None

    raw_state, _, _, _ = build_meo_observation(meo, "MEO_C", use_meo_aggregation=False)
    assert raw_state.shape == (meo_state_dim(False),)

    graph_without_intra = _domain_graph()
    graph_without_intra.nodes["MEO_A"]["aggregate"].pop("intra_graph")
    raised = False
    try:
        build_meo_observation(
            meo,
            "MEO_C",
            use_meo_aggregation=True,
            graph=graph_without_intra,
            src="a0",
        )
    except ValueError:
        raised = True
    assert raised

    raised = False
    try:
        build_meo_observation(meo, "MEO_C", use_meo_aggregation=True, src="missing")
    except ValueError:
        raised = True
    assert raised


def test_meo_router_returns_local_plan_shape_and_replay_update():
    router = MEODomainRouter({
        "meo_exit_enabled": True,
        "enabled": True,
        "use_meo_aggregation": True,
        "meo_agent": {"epsilon": 0.0, "batch_size": 1, "buffer_size": 10},
    }, device="cpu", transformer_enabled=True)
    meo = _meo_satellite()

    plan = router.recommend_path(
        meo,
        "a0",
        "c0",
        packet_size=1.0,
        task_type=1,
        computing_demand=3.0,
        size_after_computing=0.5,
        is_computed=False,
    )

    assert plan is not None
    assert plan["domains"] in (["MEO_A", "MEO_B"], ["MEO_A", "MEO_D"])
    assert plan["path"][0] == "a0"
    assert plan["path"][-1] in ("b0", "d0")
    assert plan["candidate_count"] == 2
    assert plan["meo_policy"]["state"].shape == (meo_state_dim(True),)
    assert plan["meo_policy"]["task_context"]["computing_demand"] == 3.0

    packet = SimpleNamespace(meo_decision_trace=None, meo_decision_traces=[])
    router.attach_decision(packet, plan)
    assert packet.meo_decision_trace["task_context"]["task_type"] == 1
    router.attach_decision(packet, plan)
    router.finish_decision(packet, reward=1.0, done=True)
    assert len(router.agent.replay_buffer) == 2
    assert router.agent.replay_buffer[0][2] != 1.0
    assert router.update_if_ready() is not None


def test_meo_agent_act_without_leo_context_keeps_legacy_return_shape():
    agent = MEODomainRoutingAgent(
        state_dim=3,
        cfg={"n_actions": 4, "epsilon": 0.0, "leo_policy": {"enabled": True}},
        device="cpu",
    )
    _force_meo_action(agent, 0)

    action, q_score, q_values = agent.act([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], explore=False)

    assert action == 0
    assert q_score == q_values[0]
    assert q_values.shape == (4,)
    assert agent.last_leo_rollout is None


def test_meo_agent_leo_rollout_computes_then_forwards_to_exit():
    agent = MEODomainRoutingAgent(
        state_dim=3,
        cfg={"n_actions": 4, "epsilon": 0.0, "leo_policy": {"enabled": True, "max_steps": 5}},
        device="cpu",
    )
    _force_meo_action(agent, 0)
    graph = nx.DiGraph()
    graph.add_nodes_from(["A", "B", "C"])
    graph.add_edge("A", "B")
    graph.add_edge("B", "C")
    agent.leo_policy = _FakeLEOPolicy([
        {"next_hop": "B", "compute_node": "A"},
        {"next_hop": "B", "compute_node": "A"},
        {"next_hop": "C", "compute_node": "B"},
    ])
    context = {
        "neighbors": ["MEO_B"],
        "src": "A",
        "task_context": {
            "packet_size": 10.0,
            "computing_demand": 3.0,
            "size_after_computing": 2.0,
            "is_computed": False,
        },
        "candidate_exits": {"MEO_B": "C"},
        "intra_graph": graph,
    }

    action, _, _ = agent.act([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], explore=False, leo_context=context)

    rollout = agent.last_leo_rollout
    assert action == 0
    assert rollout["reached_target"] is True
    assert rollout["path"] == ["A", "B", "C"]
    assert rollout["compute_flags"] == [1, 0, 0]
    assert rollout["compute_node"] == "A"
    assert rollout["final_task_context"]["is_computed"] is True
    assert rollout["final_task_context"]["computing_demand"] == 0.0
    assert rollout["final_task_context"]["packet_size"] == 2.0


def test_meo_agent_leo_loss_gate_requires_full_window_and_strict_threshold():
    agent = MEODomainRoutingAgent(
        state_dim=3,
        cfg={
            "n_actions": 4,
            "epsilon": 0.0,
            "leo_policy": {
                "enabled": True,
                "selection_enabled": True,
                "loss_gate_enabled": True,
                "loss_gate_window_size": 3,
                "loss_gate_threshold": 0.1,
            },
        },
        device="cpu",
    )

    agent.leo_loss_history.extend([0.01, 0.01])
    assert agent.is_leo_policy_ready() is False

    agent.leo_loss_history.append(0.28)
    assert abs(agent.leo_loss_window_average - 0.1) < 1e-12
    assert agent.is_leo_policy_ready() is False

    agent.leo_loss_history.append(0.0)
    assert agent.leo_loss_window_average < 0.1
    assert agent.is_leo_policy_ready() is True

    agent.leo_loss_history.append(0.3)
    assert agent.is_leo_policy_ready() is False


def test_meo_agent_loss_gate_blocks_all_leo_rollouts_until_ready():
    agent = MEODomainRoutingAgent(
        state_dim=3,
        cfg={
            "n_actions": 4,
            "epsilon": 0.0,
            "leo_policy": {
                "enabled": True,
                "selection_enabled": True,
                "loss_gate_enabled": True,
                "loss_gate_window_size": 2,
                "loss_gate_threshold": 0.1,
                "unreachable_penalty": 10.0,
            },
        },
        device="cpu",
    )
    _force_meo_action(agent, 0)
    graph = nx.DiGraph()
    graph.add_nodes_from(["A", "B", "C"])
    graph.add_edge("A", "B")
    agent.leo_policy = _FakeLEOPolicy({"A": {"next_hop": "B", "compute_node": None}})
    context = {
        "neighbors": ["MEO_BAD", "MEO_GOOD"],
        "src": "A",
        "task_context": {"is_computed": True},
        "candidate_exits": {"MEO_BAD": "C", "MEO_GOOD": "B"},
        "intra_graph": graph,
    }

    action, _, _ = agent.act(
        [0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], explore=False, leo_context=context
    )
    assert action == 0
    assert agent.last_leo_rollout is None
    assert agent.last_leo_rollout_scores is None

    agent.leo_loss_history.extend([0.05, 0.05])
    action, _, _ = agent.act(
        [0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], explore=False, leo_context=context
    )
    assert action == 1
    assert agent.last_leo_rollout["reached_target"] is True


def test_meo_agent_leo_loss_history_round_trips_through_checkpoint(tmp_path):
    leo_path = tmp_path / "leo_policy.pt"
    cfg = {
        "leo_policy": {
            "enabled": True,
            "selection_enabled": True,
            "loss_gate_enabled": True,
            "loss_gate_window_size": 3,
            "loss_gate_threshold": 0.1,
            "save_path": str(leo_path),
            "hidden_dim": 16,
            "num_layers": 1,
        }
    }
    agent = MEODomainRoutingAgent(state_dim=3, cfg=cfg, device="cpu")
    agent.leo_loss_history.extend([0.04, 0.05, 0.06])
    agent.save(str(tmp_path / "meo_agent.pt"))

    load_cfg = {"leo_policy": {**cfg["leo_policy"], "model_path": str(leo_path)}}
    restored = MEODomainRoutingAgent(state_dim=3, cfg=load_cfg, device="cpu")

    assert list(restored.leo_loss_history) == [0.04, 0.05, 0.06]
    assert restored.last_leo_loss == 0.06
    assert restored.is_leo_policy_ready() is True


def test_meo_agent_ignores_incompatible_old_leo_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "old_leo_policy.pt"
    original = MEODomainRoutingAgent(
        state_dim=3,
        cfg={"leo_policy": {"enabled": True, "hidden_dim": 16, "num_layers": 1}},
        device="cpu",
    )
    old_state = original.leo_policy.state_dict()
    old_state["node_proj.0.weight"] = torch.zeros((16, 7))
    torch.save({"model": old_state}, checkpoint_path)

    restored = MEODomainRoutingAgent(
        state_dim=3,
        cfg={
            "leo_policy": {
                "enabled": True,
                "hidden_dim": 16,
                "num_layers": 1,
                "model_path": str(checkpoint_path),
            }
        },
        device="cpu",
    )

    assert restored.leo_policy.node_proj[0].weight.shape == (16, 3)


def test_meo_agent_leo_rollouts_can_change_meo_action_selection():
    agent = MEODomainRoutingAgent(
        state_dim=3,
        cfg={
            "n_actions": 4,
            "epsilon": 0.0,
            "leo_policy": {
                "enabled": True,
                "max_steps": 3,
                "unreachable_penalty": 10.0,
                "hop_penalty": 0.0,
            },
        },
        device="cpu",
    )
    _force_meo_action(agent, 0)
    graph = nx.DiGraph()
    graph.add_nodes_from(["A", "B", "C"])
    graph.add_edge("A", "B")
    agent.leo_policy = _FakeLEOPolicy({"A": {"next_hop": "B", "compute_node": None}})
    context = {
        "neighbors": ["MEO_BAD", "MEO_GOOD"],
        "src": "A",
        "task_context": {"is_computed": True},
        "candidate_exits": {"MEO_BAD": "C", "MEO_GOOD": "B"},
        "intra_graph": graph,
    }

    action, q_score, q_values = agent.act(
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        explore=False,
        leo_context=context,
    )

    assert q_values[0] > q_values[1]
    assert action == 1
    assert q_score == q_values[1]
    assert agent.last_leo_rollout["next_domain"] == "MEO_GOOD"
    assert agent.last_leo_rollout["reached_target"] is True
    assert agent.last_leo_rollout_scores[0]["reached_target"] is False
    assert agent.last_leo_rollout_scores[1]["reached_target"] is True


def test_meo_agent_leo_rollout_delay_penalty_changes_action_selection():
    agent = MEODomainRoutingAgent(
        state_dim=3,
        cfg={
            "n_actions": 4,
            "epsilon": 0.0,
            "leo_policy": {
                "enabled": True,
                "max_steps": 3,
                "unreachable_penalty": 10.0,
                "hop_penalty": 0.0,
                "delay_penalty": 1.0,
            },
        },
        device="cpu",
    )
    _force_meo_action(agent, 0)
    graph = nx.DiGraph()
    graph.add_nodes_from(["A", "B", "C"])
    graph.add_edge("A", "B", delay=5.0)
    graph.add_edge("A", "C", delay=0.1)
    agent.leo_policy = _FakeLEOPolicy(
        lambda graph, source, task: {
            "next_hop": task.get("target_node"),
            "compute_node": None,
        }
    )
    context = {
        "neighbors": ["MEO_SLOW", "MEO_FAST"],
        "src": "A",
        "task_context": {"is_computed": True},
        "candidate_exits": {"MEO_SLOW": "B", "MEO_FAST": "C"},
        "intra_graph": graph,
    }

    action, _, q_values = agent.act(
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        explore=False,
        leo_context=context,
    )

    assert q_values[0] > q_values[1]
    assert action == 1
    assert agent.last_leo_rollout["predicted_delay"] == 0.1
    assert agent.last_leo_rollout_scores[0]["predicted_delay"] == 5.0
    assert agent.last_leo_rollout_scores[1]["predicted_delay"] == 0.1


def test_meo_router_recommend_path_writes_leo_rollout_into_plan():
    router = MEODomainRouter({
        "meo_exit_enabled": True,
        "enabled": True,
        "use_meo_aggregation": True,
        "meo_agent": {
            "epsilon": 0.0,
            "batch_size": 1,
            "buffer_size": 10,
            "leo_policy": {"enabled": True, "max_steps": 4},
        },
    }, device="cpu", transformer_enabled=False)
    _force_meo_action(router.agent, 0)
    router.agent.leo_policy = _FakeLEOPolicy([
        {"next_hop": "a1", "compute_node": "a0"},
        {"next_hop": "a1", "compute_node": "a0"},
    ])
    meo = _meo_satellite()

    plan = router.recommend_path(
        meo,
        "a0",
        "c0",
        packet_size=10.0,
        task_type=1,
        computing_demand=3.0,
        size_after_computing=2.0,
        is_computed=False,
    )

    assert plan is not None
    assert plan["meo_policy"]["action"] == 0
    assert plan["leo_policy_rollout"]["reached_target"] is True
    assert plan["path"][:3] == ["a0", "a1", "b0"]
    assert plan["compute_flags"][:2] == [1, 0]
    assert plan["computing_leo"] == "a0"
    first_call = router.agent.leo_policy.calls[0]
    assert first_call[1]["packet_size"] == 1.0
    assert first_call[1]["computing_demand"] == 0.3
    assert first_call[2]["edge_target_distances"][("a0", "a1")] == 0.0


def test_meo_router_store_leo_policy_experience_for_forward_and_compute():
    router = MEODomainRouter({
        "meo_exit_enabled": True,
        "use_meo_aggregation": True,
        "meo_agent": {
            "epsilon": 0.0,
            "leo_policy": {
                "enabled": True,
                "train_enabled": True,
                "batch_size": 2,
                "buffer_size": 10,
                "hidden_dim": 32,
                "num_layers": 1,
            },
        },
    }, device="cpu", transformer_enabled=False)
    meo = _meo_satellite()
    leo = meo.propagator.satellites["a0"]
    leo.propagator = meo.propagator
    leo.masterMeo = "MEO_A"

    assert router.store_leo_policy_experience(
        leo,
        task_type=1,
        packet_size=2.0,
        computing_demand=3.0,
        size_after_computing=1.0,
        is_computed=False,
        hops=1,
        destination="c0",
        next_hop_target="a1",
    )
    assert router.store_leo_policy_experience(
        leo,
        task_type=1,
        packet_size=2.0,
        computing_demand=3.0,
        size_after_computing=1.0,
        is_computed=False,
        hops=1,
        destination="c0",
        compute_node_target="a0",
    )

    assert len(router.agent.leo_replay_buffer) == 2
    assert router.agent.leo_replay_buffer[0]["next_hop_target"] == "a1"
    assert router.agent.leo_replay_buffer[1]["compute_node_target"] == "a0"
    assert router.agent.leo_replay_buffer[0]["destination"] == "c0"
    assert router.agent.leo_replay_buffer[1]["destination"] == "c0"
    assert router.agent.leo_replay_buffer[0]["task_context"] == {
        "task_type": 1.0,
        "packet_size": 0.2,
        "computing_demand": 0.3,
        "size_after_computing": 0.1,
        "hops": 0.1,
        "is_computed": 0.0,
    }
    assert router.agent.leo_replay_buffer[0]["edge_target_distances"][("a0", "a1")] == 0.2


def test_meo_agent_update_leo_policy_produces_loss():
    router = MEODomainRouter({
        "meo_exit_enabled": True,
        "use_meo_aggregation": True,
        "meo_agent": {
            "epsilon": 0.0,
            "leo_policy": {
                "enabled": True,
                "train_enabled": True,
                "batch_size": 2,
                "buffer_size": 10,
                "hidden_dim": 32,
                "num_layers": 1,
                "dropout": 0.0,
            },
        },
    }, device="cpu", transformer_enabled=False)
    meo = _meo_satellite()
    leo = meo.propagator.satellites["a0"]
    leo.propagator = meo.propagator
    leo.masterMeo = "MEO_A"
    router.store_leo_policy_experience(leo, next_hop_target="a1")
    router.store_leo_policy_experience(leo, compute_node_target="a0")

    loss = router.agent.update_leo_policy()

    assert loss is not None
    assert router.agent.last_leo_loss == loss
    assert list(router.agent.leo_loss_history) == [loss]
    assert router.agent.last_leo_losses["loss"] > 0.0


def test_meo_agent_update_leo_policy_disabled_returns_none():
    router = MEODomainRouter({
        "meo_exit_enabled": True,
        "meo_agent": {
            "epsilon": 0.0,
            "leo_policy": {
                "enabled": True,
                "train_enabled": False,
                "batch_size": 1,
                "hidden_dim": 32,
                "num_layers": 1,
            },
        },
    }, device="cpu", transformer_enabled=False)

    assert router.agent.update_leo_policy() is None


def test_meo_router_finish_segment_stores_nonterminal_experience_with_next_plan():
    router = MEODomainRouter({
        "meo_exit_enabled": True,
        "use_meo_aggregation": True,
        "meo_agent": {
            "epsilon": 0.0,
            "batch_size": 1,
            "buffer_size": 10,
            "reward": {
                "segment_success_reward": 0.5,
                "score_penalty": 0.0,
                "domain_hop_penalty": 0.0,
                "boundary_load_penalty": 0.0,
                "boundary_delay_penalty": 0.1,
            },
        },
    }, device="cpu", transformer_enabled=False)
    meo = _meo_satellite()
    plan = router.recommend_path(meo, "a0", "c0", packet_size=1.0)
    next_state = np.ones(meo_state_dim(True), dtype=np.float32)
    next_mask = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    next_plan = {
        "meo_policy": {
            "state": next_state,
            "action_mask": next_mask,
        }
    }
    packet = SimpleNamespace(meo_decision_trace=None, meo_decision_traces=[], meo_segment_time=3.0)

    router.attach_decision(packet, plan)
    router.finish_segment(packet, next_plan=next_plan, reached_node="b0")

    assert len(router.agent.replay_buffer) == 1
    stored = router.agent.replay_buffer[0]
    assert stored[4] is False
    assert np.allclose(stored[3], next_state)
    assert np.allclose(stored[5], next_mask)
    assert packet.meo_decision_traces == []
    assert packet.meo_decision_trace is None


def test_meo_router_finish_segment_without_next_plan_is_terminal():
    router = MEODomainRouter({
        "meo_exit_enabled": True,
        "use_meo_aggregation": True,
        "meo_agent": {"epsilon": 0.0, "batch_size": 1, "buffer_size": 10},
    }, device="cpu", transformer_enabled=False)
    meo = _meo_satellite()
    plan = router.recommend_path(meo, "a0", "c0", packet_size=1.0)
    packet = SimpleNamespace(meo_decision_trace=None, meo_decision_traces=[], meo_segment_time=1.0)

    router.attach_decision(packet, plan)
    router.finish_segment(packet, next_plan=None, reached_node="b0")

    assert len(router.agent.replay_buffer) == 1
    stored = router.agent.replay_buffer[0]
    assert stored[4] is True
    assert np.allclose(stored[3], np.zeros_like(stored[0]))
    assert packet.meo_decision_traces == []
    assert packet.meo_decision_trace is None


def test_meo_router_terminal_success_adds_credit_to_all_completed_segments_once():
    router = MEODomainRouter({
        "meo_exit_enabled": True,
        "use_meo_aggregation": True,
        "meo_agent": {
            "epsilon": 0.0,
            "batch_size": 1,
            "buffer_size": 10,
            "reward": {
                "segment_success_reward": 0.5,
                "terminal_credit_weight": 0.2,
                "terminal_reward_weight": 0.0,
                "success_reward": 1.5,
                "domain_hop_penalty": 0.0,
                "boundary_load_penalty": 0.0,
                "boundary_delay_penalty": 0.0,
            },
        },
    }, device="cpu", transformer_enabled=False)
    meo = _meo_satellite()
    plan = router.recommend_path(meo, "a0", "c0", packet_size=1.0)
    next_plan = {
        "meo_policy": {
            "state": np.ones(meo_state_dim(True), dtype=np.float32),
            "action_mask": np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        }
    }
    packet = SimpleNamespace(
        meo_decision_trace=None,
        meo_decision_traces=[],
        meo_segment_time=0.0,
        information=[True],
    )

    router.attach_decision(packet, plan)
    router.finish_segment(packet, next_plan=next_plan, reached_node="b0")
    stored = router.agent.replay_buffer[0]
    original_reward = stored[2]

    router.finish_decision(packet, reward=1.0, done=True, meo_result="reached_computed")

    assert isinstance(stored, list)
    assert np.isclose(stored[2], original_reward + 0.2 * 1.5)
    assert packet.meo_completed_experiences == []

    router.finish_decision(packet, reward=1.0, done=True, meo_result="reached_computed")
    assert np.isclose(stored[2], original_reward + 0.2 * 1.5)


def test_meo_router_terminal_failure_adds_negative_credit_to_completed_segments():
    router = MEODomainRouter({
        "meo_exit_enabled": True,
        "use_meo_aggregation": True,
        "meo_agent": {
            "epsilon": 0.0,
            "batch_size": 1,
            "buffer_size": 10,
            "reward": {
                "segment_success_reward": 0.5,
                "terminal_credit_weight": 0.2,
                "terminal_reward_weight": 0.0,
                "failure_penalty": 1.5,
                "domain_hop_penalty": 0.0,
                "boundary_load_penalty": 0.0,
                "boundary_delay_penalty": 0.0,
            },
        },
    }, device="cpu", transformer_enabled=False)
    meo = _meo_satellite()
    plan = router.recommend_path(meo, "a0", "c0", packet_size=1.0)
    next_plan = {
        "meo_policy": {
            "state": np.ones(meo_state_dim(True), dtype=np.float32),
            "action_mask": np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        }
    }
    packet = SimpleNamespace(
        meo_decision_trace=None,
        meo_decision_traces=[],
        meo_segment_time=0.0,
        information=[False],
    )

    router.attach_decision(packet, plan)
    router.finish_segment(packet, next_plan=next_plan, reached_node="b0")
    stored = router.agent.replay_buffer[0]
    original_reward = stored[2]

    router.finish_decision(packet, reward=-1.0, done=True, meo_result="link_drop")

    assert np.isclose(stored[2], original_reward - 0.2 * 1.5)


def test_meo_agent_epsilon_decays_after_successful_update():
    router = MEODomainRouter({
        "meo_exit_enabled": True,
        "use_meo_aggregation": True,
        "meo_agent": {
            "epsilon": 0.5,
            "min_epsilon": 0.2,
            "epsilon_decay": 0.5,
            "batch_size": 1,
            "buffer_size": 10,
        },
    }, device="cpu", transformer_enabled=False)
    meo = _meo_satellite()

    plan = router.recommend_path(meo, "a0", "c0", packet_size=1.0)
    packet = SimpleNamespace(meo_decision_trace=None, meo_decision_traces=[])
    router.attach_decision(packet, plan)
    router.finish_decision(packet, reward=1.0, done=True)

    assert router.update_if_ready() is not None
    assert router.agent.epsilon == 0.25
    assert router.update_if_ready() is None
    assert router.agent.epsilon == 0.25


def test_adjacent_domain_can_return_final_local_hop_when_called():
    router = MEODomainRouter({
        "meo_exit_enabled": True,
        "meo_agent": {"epsilon": 0.0},
    }, device="cpu", transformer_enabled=False)
    meo = _meo_satellite()

    plan = router.recommend_path(meo, "a0", "b0", packet_size=1.0)

    assert plan is not None
    assert plan["domains"] == ["MEO_A", "MEO_B"]
    assert plan["domain_entries"]["MEO_B"]["exit"] == "b0"
