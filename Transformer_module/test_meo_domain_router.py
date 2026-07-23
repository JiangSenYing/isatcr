from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest
import torch

from Transformer_module import MEODomainRouter
from Transformer_module.meo_agent import MEODomainRoutingAgent
from Transformer_module.meo_router import MEODomainRewardFunction
from Transformer_module.meo_observation import (
    AGG_FEATURE_DIM,
    DOMAIN_TOKEN_FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
    RAW_FEATURE_DIM,
    TASK_GLOBAL_FEATURE_DIM,
    UNREACHABLE_BOUNDARY_DELAY,
    build_meo_attention_observation,
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


def test_meo_attention_observation_collects_configured_hop_context_only():
    meo = _meo_satellite()
    graph = meo.inter_domain_graph
    for domain, member in (("MEO_E", "e0"), ("MEO_F", "f0"), ("MEO_G", "g0")):
        graph.add_node(
            domain,
            aggregate={
                "members": [member],
                "boundary_nodes": [member],
                "intra_graph": _intra_graph(member),
                "avg_memory_occupancy_rate": 0.2,
                "std_memory_occupancy_rate": 0.1,
                "avg_computing_occupancy_rate": 0.4,
                "std_computing_occupancy_rate": 0.1,
            },
        )
    _add_boundary(graph, "MEO_C", "MEO_E", "c0", "e0", 0.2)
    _add_boundary(graph, "MEO_E", "MEO_F", "e0", "f0", 0.2)

    expected_counts = {1: 3, 2: 4, 3: 5}
    for context_hops, expected_count in expected_counts.items():
        state, mask, neighbors, _ = build_meo_attention_observation(
            meo,
            "MEO_F",
            attention_context_hops=context_hops,
            use_meo_aggregation=True,
            src="a0",
        )

        assert state["domain_features"].shape == (expected_count, DOMAIN_TOKEN_FEATURE_DIM)
        assert state["domain_hop_matrix"].shape == (expected_count, expected_count)
        assert neighbors == ["MEO_B", "MEO_D"]
        assert mask.tolist() == [1.0, 1.0, 0.0, 0.0]
        assert state["action_domain_indices"].tolist()[:2] == [1, 2]
        assert state["action_domain_indices"].tolist()[2:] == [-1, -1]


def test_meo_attention_observation_rejects_unsupported_context_hops():
    meo = _meo_satellite()
    with pytest.raises(ValueError, match="attention_context_hops"):
        build_meo_attention_observation(
            meo,
            "MEO_C",
            attention_context_hops=4,
            use_meo_aggregation=True,
            src="a0",
        )


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


def test_meo_segment_reward_penalizes_actual_segment_hops_with_empty_plan_path():
    reward_fn = MEODomainRewardFunction({
        "segment_success_reward": 0.5,
        "domain_hop_penalty": 0.0,
        "path_hop_penalty": 0.2,
        "boundary_load_penalty": 0.0,
        "boundary_delay_penalty": 0.0,
    })

    reward = reward_fn.segment_reward({"path": [], "segment_hops": 3})

    assert np.isclose(reward, -0.1)


def test_meo_segment_reward_falls_back_to_legacy_planned_path_hops():
    reward_fn = MEODomainRewardFunction({
        "segment_success_reward": 0.5,
        "domain_hop_penalty": 0.0,
        "path_hop_penalty": 0.2,
        "boundary_load_penalty": 0.0,
        "boundary_delay_penalty": 0.0,
    })

    reward = reward_fn.segment_reward({"path": ["a0", "a1", "b0", "b1"]})

    assert np.isclose(reward, -0.1)


def test_meo_terminal_failure_penalizes_completed_segment_hops():
    reward_fn = MEODomainRewardFunction({
        "failure_penalty": 1.0,
        "score_penalty": 0.0,
        "domain_hop_penalty": 0.0,
        "path_hop_penalty": 0.2,
        "boundary_load_penalty": 0.0,
        "boundary_delay_penalty": 0.0,
    })
    trace = {
        "meo_result": "link_drop",
        "path": [],
        "segment_hops": 2,
    }

    reward = reward_fn(trace, terminal_reward=-1.0, done=True)

    assert np.isclose(reward, -1.4)


def test_meo_segment_reward_includes_weighted_leo_domain_entry_reward():
    reward_fn = MEODomainRewardFunction({
        "segment_success_reward": 0.5,
        "terminal_reward_weight": 0.25,
        "domain_hop_penalty": 0.0,
        "path_hop_penalty": 0.0,
        "boundary_load_penalty": 0.0,
    })
    trace = {
        "decision_time": 4.0,
    }

    original_reward = reward_fn.segment_reward(trace, current_time=7.0)
    weighted_reward = reward_fn.segment_reward(
        trace,
        current_time=7.0,
        leo_reward=-2.0,
    )

    assert np.isclose(weighted_reward, original_reward + 0.25 * -2.0)


def test_meo_segment_reward_ignores_leo_reward_when_weight_is_zero():
    reward_fn = MEODomainRewardFunction({
        "segment_success_reward": 0.5,
        "terminal_reward_weight": 0.0,
        "domain_hop_penalty": 0.0,
        "path_hop_penalty": 0.0,
        "boundary_load_penalty": 0.0,
    })
    trace = {
        "decision_time": 4.0,
    }

    original_reward = reward_fn.segment_reward(trace, current_time=7.0)
    reward_with_leo_value = reward_fn.segment_reward(
        trace,
        current_time=7.0,
        leo_reward=100.0,
    )

    assert np.isclose(reward_with_leo_value, original_reward)


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


def test_meo_observation_includes_unique_target_entry_node_load_statistics():
    meo = _meo_satellite()
    graph = meo.inter_domain_graph
    graph["MEO_A"]["MEO_B"]["boundary_links"] = {
        ("a0", "b0"): {
            "source_domain": "MEO_A",
            "target_domain": "MEO_B",
            "source_boundary": "a0",
            "target_boundary": "b0",
            "target_memory_occupancy_rate": 0.2,
            "target_computing_occupancy_rate": 0.3,
            "delay": 0.1,
        },
        ("a1", "b0"): {
            "source_domain": "MEO_A",
            "target_domain": "MEO_B",
            "source_boundary": "a1",
            "target_boundary": "b0",
            "target_memory_occupancy_rate": 0.6,
            "target_computing_occupancy_rate": 0.4,
            "delay": 0.1,
        },
        ("a1", "b1"): {
            "source_domain": "MEO_A",
            "target_domain": "MEO_B",
            "source_boundary": "a1",
            "target_boundary": "b1",
            "target_memory_occupancy_rate": 0.8,
            "target_computing_occupancy_rate": 0.9,
            "delay": 0.1,
        },
    }
    expected = np.asarray([0.6, 0.7, 0.8, 0.4, 0.65, 0.9], dtype=np.float32)

    for use_aggregation, base_feature_dim in ((True, 7), (False, 8)):
        state, _, neighbors, _ = build_meo_observation(
            meo,
            "MEO_C",
            use_meo_aggregation=use_aggregation,
            src="a0" if use_aggregation else None,
        )
        row_width = (AGG_FEATURE_DIM if use_aggregation else RAW_FEATURE_DIM) + 1
        first_row = state[GLOBAL_FEATURE_DIM + TASK_GLOBAL_FEATURE_DIM:][0:row_width]
        assert neighbors[0] == "MEO_B"
        assert np.allclose(first_row[base_feature_dim:base_feature_dim + 6], expected)

    attention_state, _, _, _ = build_meo_attention_observation(
        meo,
        "MEO_C",
        attention_context_hops=2,
        use_meo_aggregation=True,
        src="a0",
    )
    assert np.allclose(attention_state["action_features"][0, 7:13], expected)


def test_meo_observation_uses_new_target_endpoint_when_boundary_is_reversed():
    meo = _meo_b_satellite()
    link = meo.inter_domain_graph["MEO_A"]["MEO_B"]["boundary_links"][("a1", "b0")]
    link.update({
        "source_memory_occupancy_rate": 0.25,
        "source_computing_occupancy_rate": 0.75,
        "target_memory_occupancy_rate": 0.9,
        "target_computing_occupancy_rate": 0.1,
    })

    state, _, neighbors, _ = build_meo_observation(
        meo,
        "MEO_D",
        use_meo_aggregation=True,
        src="b0",
    )
    row_width = AGG_FEATURE_DIM + 1
    first_row = state[GLOBAL_FEATURE_DIM + TASK_GLOBAL_FEATURE_DIM:][0:row_width]

    assert neighbors[0] == "MEO_A"
    assert np.allclose(first_row[7:13], [0.25, 0.25, 0.25, 0.75, 0.75, 0.75])


def test_meo_observation_entry_load_falls_back_to_intra_graph_then_conservative_default():
    meo = _meo_satellite()
    graph = meo.inter_domain_graph
    target_intra_graph = graph.nodes["MEO_B"]["aggregate"]["intra_graph"]
    target_intra_graph.nodes["b0"].update({
        "memory_occupancy_rate": 0.33,
        "computing_occupancy_rate": 0.44,
    })

    state, _, _, _ = build_meo_observation(
        meo, "MEO_C", use_meo_aggregation=True, src="a0"
    )
    row_width = AGG_FEATURE_DIM + 1
    first_row = state[GLOBAL_FEATURE_DIM + TASK_GLOBAL_FEATURE_DIM:][0:row_width]
    assert np.allclose(first_row[7:13], [0.33, 0.33, 0.33, 0.44, 0.44, 0.44])

    target_intra_graph.nodes["b0"].clear()
    state, _, _, _ = build_meo_observation(
        meo, "MEO_C", use_meo_aggregation=True, src="a0"
    )
    first_row = state[GLOBAL_FEATURE_DIM + TASK_GLOBAL_FEATURE_DIM:][0:row_width]
    assert np.allclose(first_row[7:13], np.ones(6, dtype=np.float32))


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
    assert plan["path"] == []
    assert plan["transitions"] == []
    assert plan["score"] == 0.0
    assert plan["domain_entries"]["MEO_A"] == {"entry": "a0", "exit": None}
    assert plan["candidate_count"] == 2
    assert plan["meo_policy"]["state"].shape == (meo_state_dim(True),)
    assert plan["meo_policy"]["task_context"]["computing_demand"] == 3.0

    packet = SimpleNamespace(meo_decision_trace=None, meo_decision_traces=[])
    router.attach_decision(packet, plan)
    assert packet.meo_decision_trace["task_context"]["task_type"] == 1
    assert packet.meo_decision_trace["segment_hops"] == 0
    assert packet.meo_decision_trace["executed_path"] == ["a0"]
    router.attach_decision(packet, plan)
    router.finish_decision(packet, reward=1.0, done=True)
    assert len(router.agent.replay_buffer) == 2
    assert router.agent.replay_buffer[0][2] != 1.0
    assert router.update_if_ready() is not None


def test_meo_router_records_intra_domain_and_cross_domain_leo_hops():
    packet = SimpleNamespace(
        meo_decision_trace={
            "segment_hops": 0,
            "executed_path": ["a0"],
        },
        meo_decision_traces=[],
    )

    assert MEODomainRouter.record_leo_hop(packet, "a0", "a1") is True
    assert MEODomainRouter.record_leo_hop(packet, "a1", "b0") is True

    assert packet.meo_decision_trace["segment_hops"] == 2
    assert packet.meo_decision_trace["executed_path"] == ["a0", "a1", "b0"]


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


def test_meo_router_recommend_path_ignores_internal_leo_rollout():
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
    assert plan["boundary_sat"] == ["b0"]
    assert plan["path"] == []
    assert plan["transitions"] == []
    assert "leo_policy_rollout" not in plan
    assert "compute_flags" not in plan
    assert "computing_leo" not in plan
    assert router.agent.leo_policy.calls == []


def test_meo_router_records_exact_executed_boundary_link():
    router = MEODomainRouter({
        "meo_exit_enabled": True,
        "use_meo_aggregation": True,
        "meo_agent": {"epsilon": 0.0},
    }, device="cpu", transformer_enabled=False)
    meo = _meo_satellite()
    _force_meo_action(router.agent, 0)
    graph = meo.inter_domain_graph
    graph["MEO_A"]["MEO_B"]["boundary_links"][("a2", "b1")] = {
        "source_domain": "MEO_A",
        "target_domain": "MEO_B",
        "source_boundary": "a2",
        "target_boundary": "b1",
        "quality_cost": 0.9,
        "link_load": 2.0,
        "delay": 0.7,
    }
    plan = router.recommend_path(meo, "a0", "c0", packet_size=1.0)
    packet = SimpleNamespace(meo_decision_trace=None, meo_decision_traces=[])
    router.attach_decision(packet, plan)

    assert router.record_executed_boundary(packet, meo, "b1", previous_node="a2") is True

    trace = packet.meo_decision_trace
    assert trace["score"] == 0.9
    assert trace["transitions"] == [{
        "from_domain": "MEO_A",
        "to_domain": "MEO_B",
        "source_boundary": "a2",
        "target_boundary": "b1",
        "quality_cost": 0.9,
        "link_load": 2.0,
        "link_load_ratio": 0.2,
        "delay": 0.7,
    }]


def test_meo_router_records_reversed_and_fallback_boundary_link():
    router = MEODomainRouter({"meo_exit_enabled": True}, device="cpu", transformer_enabled=False)
    graph = nx.Graph()
    graph.add_edge("MEO_A", "MEO_B", boundary_links={
        ("b0", "a1"): {
            "source_domain": "MEO_B",
            "target_domain": "MEO_A",
            "source_boundary": "b0",
            "target_boundary": "a1",
            "quality_cost": 0.4,
            "link_load": 1.0,
            "delay": 0.2,
        },
    })
    meo = SimpleNamespace(inter_domain_graph=graph, transmission_rate=10.0)
    trace = {
        "current_domain": "MEO_A",
        "next_domain": "MEO_B",
        "transitions": [],
        "score": 0.0,
    }
    packet = SimpleNamespace(meo_decision_trace=trace, meo_decision_traces=[trace])

    assert router.record_executed_boundary(packet, meo, "b0", previous_node=None) is True
    assert trace["transitions"][0]["source_boundary"] == "a1"
    assert trace["transitions"][0]["target_boundary"] == "b0"


def test_meo_router_leaves_trace_unchanged_when_executed_boundary_is_unknown():
    router = MEODomainRouter({"meo_exit_enabled": True}, device="cpu", transformer_enabled=False)
    meo = _meo_satellite()
    trace = {
        "current_domain": "MEO_A",
        "next_domain": "MEO_B",
        "transitions": [],
        "score": 0.0,
    }
    packet = SimpleNamespace(meo_decision_trace=trace, meo_decision_traces=[trace])

    assert router.record_executed_boundary(packet, meo, "unknown", previous_node="unknown") is False
    assert trace["transitions"] == []
    assert trace["score"] == 0.0


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


def test_meo_router_finish_segment_stores_weighted_leo_entry_reward():
    router = MEODomainRouter({
        "meo_exit_enabled": True,
        "use_meo_aggregation": True,
        "meo_agent": {
            "epsilon": 0.0,
            "batch_size": 1,
            "buffer_size": 10,
            "reward": {
                "segment_success_reward": 0.5,
                "terminal_reward_weight": 0.25,
                "domain_hop_penalty": 0.0,
                "path_hop_penalty": 0.0,
                "boundary_load_penalty": 0.0,
            },
        },
    }, device="cpu", transformer_enabled=False)
    meo = _meo_satellite()
    plan = router.recommend_path(meo, "a0", "c0", packet_size=1.0)
    packet = SimpleNamespace(
        meo_decision_trace=None,
        meo_decision_traces=[],
        meo_segment_time=3.0,
    )

    router.attach_decision(packet, plan)
    trace = packet.meo_decision_traces[-1]
    original_reward = router.reward_function.segment_reward(trace, current_time=3.0)
    router.finish_segment(
        packet,
        next_plan=None,
        reached_node="b0",
        leo_reward=-2.0,
    )

    stored = router.agent.replay_buffer[0]
    assert np.isclose(stored[2], original_reward + 0.25 * -2.0)
    assert stored[4] is True


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

    assert router.updates_per_step == 1
    assert router.update_if_ready() is not None
    assert router.agent.train_steps == 1
    assert router.update_count == 1
    assert router.interval_updates == 1
    assert router.agent.epsilon == 0.25
    assert router.update_if_ready() is None
    assert router.agent.epsilon == 0.25


def test_meo_agent_runs_multiple_updates_per_step_and_returns_mean_loss():
    router = MEODomainRouter({
        "meo_exit_enabled": True,
        "use_meo_aggregation": True,
        "meo_agent": {
            "epsilon": 0.8,
            "min_epsilon": 0.1,
            "epsilon_decay": 0.5,
            "batch_size": 1,
            "buffer_size": 10,
            "updates_per_step": 8,
        },
    }, device="cpu", transformer_enabled=False)
    state = np.zeros(meo_state_dim(True), dtype=np.float32)
    action_mask = np.ones(4, dtype=np.float32)
    router.agent.store_experience(state, 0, 1.0, state, True, action_mask)
    router.decision_count = 1

    mean_loss = router.update_if_ready()

    assert router.agent.train_steps == 8
    assert router.update_count == 8
    assert router.interval_updates == 8
    assert len(router.interval_losses) == 8
    assert np.isclose(mean_loss, np.mean(router.interval_losses))
    assert np.isclose(router.last_loss, router.interval_losses[-1])
    assert np.isclose(router.agent.epsilon, 0.1)

    assert router.update_if_ready() is None
    assert router.agent.train_steps == 8
    assert router.update_count == 8
    assert np.isclose(router.agent.epsilon, 0.1)


def test_meo_agent_consumes_update_trigger_when_replay_is_insufficient():
    router = MEODomainRouter({
        "meo_exit_enabled": True,
        "use_meo_aggregation": True,
        "meo_agent": {
            "epsilon": 0.8,
            "min_epsilon": 0.1,
            "epsilon_decay": 0.5,
            "batch_size": 2,
            "buffer_size": 10,
            "updates_per_step": 8,
        },
    }, device="cpu", transformer_enabled=False)
    router.decision_count = 1

    assert router.update_if_ready() is None
    assert router.last_update_decision_count == 1
    assert router.agent.train_steps == 0
    assert router.update_count == 0
    assert router.interval_updates == 0
    assert router.interval_losses == []
    assert np.isclose(router.agent.epsilon, 0.8)

    assert router.update_if_ready() is None
    assert router.agent.train_steps == 0
    assert np.isclose(router.agent.epsilon, 0.8)


def test_meo_agent_clamps_updates_per_step_to_one():
    router = MEODomainRouter({
        "meo_exit_enabled": True,
        "meo_agent": {"updates_per_step": 0},
    }, device="cpu", transformer_enabled=False)

    assert router.updates_per_step == 1


def _attention_agent(state_dim, **overrides):
    cfg = {
        "encoder_type": "self_attention",
        "n_actions": 4,
        "hidden_dim": 16,
        "attention_heads": 4,
        "attention_layers": 1,
        "attention_ff_dim": 32,
        "attention_dropout": 0.0,
        "epsilon": 0.0,
        **overrides,
    }
    return MEODomainRoutingAgent(state_dim=state_dim, cfg=cfg, device="cpu")


def _attention_state(agent, token_count=3):
    domain_features = np.linspace(
        0.05,
        0.95,
        token_count * DOMAIN_TOKEN_FEATURE_DIM,
        dtype=np.float32,
    ).reshape(token_count, DOMAIN_TOKEN_FEATURE_DIM)
    hop_matrix = np.abs(
        np.arange(token_count)[:, None] - np.arange(token_count)[None, :]
    ).astype(np.int64)
    action_indices = np.full(agent.n_actions, -1, dtype=np.int64)
    for idx in range(min(agent.n_actions, token_count - 1)):
        action_indices[idx] = idx + 1
    return {
        "global_context": np.linspace(
            0.1, 0.9, GLOBAL_FEATURE_DIM + TASK_GLOBAL_FEATURE_DIM, dtype=np.float32
        ),
        "domain_features": domain_features,
        "domain_hop_matrix": hop_matrix,
        "action_features": np.linspace(
            0.1,
            0.8,
            agent.n_actions * agent.network_config["action_feature_dim"],
            dtype=np.float32,
        ).reshape(agent.n_actions, agent.network_config["action_feature_dim"]),
        "action_domain_indices": action_indices,
        "current_domain_index": np.asarray(0, dtype=np.int64),
    }


def test_meo_attention_encoder_supports_aggregate_raw_and_zero_states():
    for use_aggregation in (True, False):
        state_dim = meo_state_dim(use_aggregation)
        agent = _attention_agent(state_dim)
        random_states = [_attention_state(agent, 2), _attention_state(agent, 4)]
        zero_state = agent.zero_state_like(random_states[0])

        random_q = agent.online_net(agent._collate_attention_states(random_states))
        zero_q = agent.online_net(agent._collate_attention_states([zero_state, zero_state]))

        assert random_q.shape == (2, 4)
        assert zero_q.shape == (2, 4)
        assert torch.isfinite(random_q).all()
        assert torch.isfinite(zero_q).all()


def test_meo_attention_encoder_is_domain_permutation_equivariant():
    state_dim = meo_state_dim(True)
    agent = _attention_agent(state_dim)
    agent.online_net.eval()
    state = _attention_state(agent, token_count=4)
    permutation = np.asarray([2, 0, 3, 1])
    inverse = np.argsort(permutation)
    permuted_state = dict(state)
    permuted_state["domain_features"] = state["domain_features"][permutation]
    permuted_state["domain_hop_matrix"] = state["domain_hop_matrix"][permutation][:, permutation]
    valid_indices = state["action_domain_indices"] >= 0
    permuted_indices = state["action_domain_indices"].copy()
    permuted_indices[valid_indices] = inverse[permuted_indices[valid_indices]]
    permuted_state["action_domain_indices"] = permuted_indices
    permuted_state["current_domain_index"] = np.asarray(
        inverse[int(state["current_domain_index"])], dtype=np.int64
    )

    with torch.no_grad():
        q_values = agent.online_net(agent._collate_attention_states([state])).squeeze(0)
        permuted_q_values = agent.online_net(
            agent._collate_attention_states([permuted_state])
        ).squeeze(0)

    assert torch.allclose(permuted_q_values, q_values, atol=1e-6, rtol=1e-6)


def test_meo_attention_q_values_use_two_hop_domain_features():
    agent = _attention_agent(meo_state_dim(True), attention_context_hops=2)
    state = _attention_state(agent, token_count=4)
    state["action_domain_indices"] = np.asarray([1, 2, -1, -1], dtype=np.int64)
    state["domain_hop_matrix"] = np.asarray([
        [0, 1, 1, 2],
        [1, 0, 2, 1],
        [1, 2, 0, 3],
        [2, 1, 3, 0],
    ], dtype=np.int64)
    changed = agent._copy_state(state)
    changed["domain_features"][3] += 5.0

    original_q = agent.q_values(state)
    changed_q = agent.q_values(changed)

    assert not np.isclose(original_q[0], changed_q[0])


def test_meo_attention_shortest_hop_bias_changes_attention_output():
    agent = _attention_agent(meo_state_dim(True), attention_context_hops=2)
    state = _attention_state(agent, token_count=3)
    changed_hops = agent._copy_state(state)
    changed_hops["domain_hop_matrix"][0, 2] = 4
    changed_hops["domain_hop_matrix"][2, 0] = 4
    with torch.no_grad():
        for layer in agent.online_net.attention:
            layer.hop_bias.weight.copy_(
                torch.arange(layer.hop_bias.num_embeddings, dtype=torch.float32)
                .unsqueeze(1)
                .expand(-1, layer.num_heads)
            )

    original_q = agent.q_values(state)
    changed_q = agent.q_values(changed_hops)

    assert not np.allclose(original_q, changed_q)


def test_meo_attention_router_uses_k_hop_context_but_one_hop_actions():
    router = MEODomainRouter({
        "meo_exit_enabled": True,
        "use_meo_aggregation": True,
        "meo_agent": {
            "encoder_type": "self_attention",
            "attention_context_hops": 2,
            "hidden_dim": 16,
            "attention_heads": 4,
            "attention_layers": 1,
            "attention_ff_dim": 32,
            "epsilon": 0.0,
        },
    }, device="cpu", transformer_enabled=False)
    meo = _meo_satellite()

    plan = router.recommend_path(meo, "a0", "c0", packet_size=1.0)

    assert plan is not None
    assert plan["next_domain"] in {"MEO_B", "MEO_D"}
    assert plan["meo_policy"]["neighbors"] == ["MEO_B", "MEO_D"]
    state = plan["meo_policy"]["state"]
    assert isinstance(state, dict)
    assert state["domain_features"].shape[0] == 4
    assert state["action_domain_indices"].tolist() == [1, 2, -1, -1]


def test_meo_attention_agent_masks_actions_and_updates_with_terminal_zero_state():
    state_dim = meo_state_dim(True)
    agent = _attention_agent(
        state_dim,
        batch_size=2,
        buffer_size=10,
        target_update_freq=1,
    )
    state = _attention_state(agent, token_count=4)
    zero_state = agent.zero_state_like(state)
    action_mask = np.asarray([1.0, 0.0, 1.0, 0.0], dtype=np.float32)

    action, _, q_values = agent.act(state, action_mask, explore=False)
    assert action in (0, 2)
    assert np.isneginf(q_values[1])
    assert np.isneginf(q_values[3])

    agent.store_experience(state, 0, 0.5, state, False, action_mask)
    agent.store_experience(state, 2, -0.5, zero_state, True, np.zeros(4, dtype=np.float32))
    loss = agent.update()

    assert loss is not None
    assert np.isfinite(loss)
    assert agent.train_steps == 1
    for online_value, target_value in zip(
        agent.online_net.state_dict().values(), agent.target_net.state_dict().values()
    ):
        assert torch.equal(online_value, target_value)


def test_meo_q_network_checkpoints_round_trip_and_reject_other_encoder(tmp_path):
    state_dim = meo_state_dim(True)
    attention_path = tmp_path / "attention_meo.pt"
    attention = _attention_agent(state_dim)
    attention.train_steps = 7
    attention.epsilon = 0.23
    attention.save(str(attention_path))

    restored_attention = _attention_agent(state_dim)
    assert restored_attention.load(str(attention_path)) is True
    assert restored_attention.train_steps == 7
    assert np.isclose(restored_attention.epsilon, 0.23)
    for original, restored in zip(
        attention.online_net.state_dict().values(),
        restored_attention.online_net.state_dict().values(),
    ):
        assert torch.equal(original, restored)

    old_dimension_path = tmp_path / "old_dimension_attention_meo.pt"
    old_dimension_checkpoint = torch.load(attention_path, map_location="cpu")
    old_dimension_checkpoint["state_dim"] = 41
    torch.save(old_dimension_checkpoint, old_dimension_path)
    assert _attention_agent(state_dim).load(str(old_dimension_path)) is False

    legacy_path = tmp_path / "legacy_attention_meo.pt"
    legacy_checkpoint = torch.load(attention_path, map_location="cpu")
    legacy_checkpoint["network_config"] = {
        key: value
        for key, value in legacy_checkpoint["network_config"].items()
        if key not in {
            "attention_context_hops",
            "domain_feature_dim",
            "action_feature_dim",
            "topology_bias_version",
        }
    }
    torch.save(legacy_checkpoint, legacy_path)
    assert _attention_agent(state_dim).load(str(legacy_path)) is False

    mlp = MEODomainRoutingAgent(
        state_dim=state_dim,
        cfg={"encoder_type": "mlp", "n_actions": 4, "hidden_dim": 16},
        device="cpu",
    )
    mlp_before = {key: value.clone() for key, value in mlp.online_net.state_dict().items()}
    assert mlp.load(str(attention_path)) is False
    for key, value in mlp.online_net.state_dict().items():
        assert torch.equal(value, mlp_before[key])

    mlp_path = tmp_path / "mlp_meo.pt"
    mlp.save(str(mlp_path))
    restored_mlp = MEODomainRoutingAgent(
        state_dim=state_dim,
        cfg={"encoder_type": "mlp", "n_actions": 4, "hidden_dim": 16},
        device="cpu",
    )
    assert restored_mlp.load(str(mlp_path)) is True
    assert restored_attention.load(str(mlp_path)) is False


def test_adjacent_domain_can_return_final_local_hop_when_called():
    router = MEODomainRouter({
        "meo_exit_enabled": True,
        "meo_agent": {"epsilon": 0.0},
    }, device="cpu", transformer_enabled=False)
    meo = _meo_satellite()
    _force_meo_action(router.agent, 0)

    plan = router.recommend_path(meo, "a0", "b0", packet_size=1.0)

    assert plan is not None
    assert plan["domains"] == ["MEO_A", "MEO_B"]
    assert plan["domain_entries"]["MEO_B"]["exit"] == "b0"
