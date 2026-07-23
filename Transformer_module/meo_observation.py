"""Observation construction for MEO inter-domain routing."""

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np


MAX_MEO_NEIGHBORS = 4
ENTRY_NODE_LOAD_FEATURE_DIM = 6
AGG_FEATURE_DIM = 13
RAW_FEATURE_DIM = 14
GLOBAL_FEATURE_DIM = 4
TASK_GLOBAL_FEATURE_DIM = 5
TASK_PER_NEIGHBOR_FEATURE_DIM = 1
DOMAIN_TOKEN_FEATURE_DIM = 9
UNREACHABLE_BOUNDARY_DELAY = 2.0


def meo_state_dim(use_meo_aggregation: bool, max_neighbors: int = MAX_MEO_NEIGHBORS) -> int:
    per_neighbor = AGG_FEATURE_DIM if use_meo_aggregation else RAW_FEATURE_DIM
    return (
        GLOBAL_FEATURE_DIM
        + TASK_GLOBAL_FEATURE_DIM
        + int(max_neighbors) * (per_neighbor + TASK_PER_NEIGHBOR_FEATURE_DIM)
    )


def build_meo_observation(
    meo_satellite,
    target_domain: str,
    use_meo_aggregation: bool = True,
    graph: Optional[nx.Graph] = None,
    future_graph: Optional[nx.Graph] = None,
    max_neighbors: int = MAX_MEO_NEIGHBORS,
    task_context: Optional[Dict[str, float]] = None,
    src: Optional[str] = None,
    excluded_domains: Optional[Iterable[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, float]]:
    """Build fixed-size MEO policy features and action mask."""
    graph = graph if graph is not None else getattr(meo_satellite, "inter_domain_graph", None)
    current_domain = getattr(meo_satellite, "name", None)
    neighbors = _ordered_meo_neighbors(meo_satellite, graph, current_domain, max_neighbors=max_neighbors)
    mask = np.zeros(max_neighbors, dtype=np.float32)
    per_neighbor = AGG_FEATURE_DIM if use_meo_aggregation else RAW_FEATURE_DIM
    rows = np.zeros((max_neighbors, per_neighbor + TASK_PER_NEIGHBOR_FEATURE_DIM), dtype=np.float32)

    distances = _domain_distances(graph, target_domain)
    computing_demand = _float((task_context or {}).get("computing_demand", 0.0))
    excluded = {str(domain) for domain in (excluded_domains or [])}
    for idx, neighbor in enumerate(neighbors):
        mask[idx] = 0.0 if str(neighbor) in excluded else 1.0
        if use_meo_aggregation:
            neighbor_features, boundary_reachable = _aggregate_neighbor_features(
                graph,
                current_domain,
                neighbor,
                target_domain,
                distances,
                future_graph,
                src,
            )
            if not boundary_reachable:
                mask[idx] = 0.0
        else:
            neighbor_features = _raw_neighbor_features(meo_satellite, graph, neighbor, target_domain, distances, future_graph)
        entry_load_features = _entry_node_load_features(graph, current_domain, neighbor)
        rows[idx] = np.concatenate([
            neighbor_features,
            entry_load_features,
            np.asarray([_compute_pressure(meo_satellite, graph, neighbor, computing_demand)], dtype=np.float32),
        ])

    global_features = np.asarray([
        1.0,
        _safe_distance(distances.get(current_domain)),
        float(current_domain == target_domain),
        float(len(neighbors)) / max(float(max_neighbors), 1.0),
    ], dtype=np.float32)
    task_features = _task_global_features(meo_satellite, task_context)
    state = np.concatenate([global_features, task_features, rows.reshape(-1)]).astype(np.float32)
    meta = {neighbor: float(distances.get(neighbor, np.inf)) for neighbor in neighbors}
    return state, mask, neighbors, meta


def build_meo_attention_observation(
    meo_satellite,
    target_domain: str,
    attention_context_hops: int,
    use_meo_aggregation: bool = True,
    graph: Optional[nx.Graph] = None,
    future_graph: Optional[nx.Graph] = None,
    max_neighbors: int = MAX_MEO_NEIGHBORS,
    task_context: Optional[Dict[str, float]] = None,
    src: Optional[str] = None,
    excluded_domains: Optional[Iterable[str]] = None,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, List[str], Dict[str, float]]:
    """Build a variable-width K-hop observation for the MEO attention agent.

    Domain tokens provide routing context only.  ``action_features`` and
    ``action_domain_indices`` remain aligned with the physical one-hop
    neighbors returned in ``neighbors``.
    """
    context_hops = int(attention_context_hops)
    if context_hops not in (1, 2, 3):
        raise ValueError("MEO attention_context_hops must be one of 1, 2, or 3")
    graph = graph if graph is not None else getattr(meo_satellite, "inter_domain_graph", None)
    current_domain = str(getattr(meo_satellite, "name", ""))
    if graph is None or current_domain not in graph:
        raise ValueError(f"current domain {current_domain!r} is missing from inter-domain graph")

    flat_state, mask, neighbors, meta = build_meo_observation(
        meo_satellite=meo_satellite,
        target_domain=target_domain,
        use_meo_aggregation=use_meo_aggregation,
        graph=graph,
        future_graph=future_graph,
        max_neighbors=max_neighbors,
        task_context=task_context,
        src=src,
        excluded_domains=excluded_domains,
    )
    context_dim = GLOBAL_FEATURE_DIM + TASK_GLOBAL_FEATURE_DIM
    action_feature_dim = (AGG_FEATURE_DIM if use_meo_aggregation else RAW_FEATURE_DIM) + TASK_PER_NEIGHBOR_FEATURE_DIM
    action_features = flat_state[context_dim:].reshape(max_neighbors, action_feature_dim).copy()

    current_distances = nx.single_source_shortest_path_length(
        graph, current_domain, cutoff=context_hops
    )
    context_domains = sorted(
        (str(domain) for domain in current_distances),
        key=lambda domain: (int(current_distances[domain]), domain),
    )
    target_distances = _domain_distances(graph, target_domain)
    computing_demand = _float((task_context or {}).get("computing_demand", 0.0))
    domain_features = np.stack([
        _domain_token_features(
            meo_satellite=meo_satellite,
            graph=graph,
            domain=domain,
            current_domain=current_domain,
            target_domain=target_domain,
            current_distance=current_distances[domain],
            target_distances=target_distances,
            computing_demand=computing_demand,
            context_hops=context_hops,
            use_meo_aggregation=use_meo_aggregation,
        )
        for domain in context_domains
    ]).astype(np.float32)

    domain_index = {domain: idx for idx, domain in enumerate(context_domains)}
    action_domain_indices = np.full(max_neighbors, -1, dtype=np.int64)
    for idx, neighbor in enumerate(neighbors):
        action_domain_indices[idx] = domain_index.get(str(neighbor), -1)
        if action_domain_indices[idx] < 0:
            mask[idx] = 0.0

    hop_matrix = np.zeros((len(context_domains), len(context_domains)), dtype=np.int64)
    for row, domain in enumerate(context_domains):
        pair_distances = nx.single_source_shortest_path_length(graph, domain)
        for col, other in enumerate(context_domains):
            hop_matrix[row, col] = min(
                int(pair_distances.get(other, context_hops * 2)),
                context_hops * 2,
            )

    state = {
        "global_context": flat_state[:context_dim].copy(),
        "domain_features": domain_features,
        "domain_hop_matrix": hop_matrix,
        "action_features": action_features,
        "action_domain_indices": action_domain_indices,
        "current_domain_index": np.asarray(domain_index[current_domain], dtype=np.int64),
    }
    return state, mask, neighbors, meta


def _domain_token_features(
    meo_satellite,
    graph,
    domain,
    current_domain,
    target_domain,
    current_distance,
    target_distances,
    computing_demand,
    context_hops,
    use_meo_aggregation,
):
    resource_features = _domain_resource_features(
        meo_satellite, graph, domain, use_meo_aggregation
    )
    return np.concatenate([
        resource_features,
        np.asarray([
            min(float(current_distance) / max(float(context_hops), 1.0), 1.0),
            _safe_distance(target_distances.get(domain)),
            float(domain == current_domain),
            float(domain == target_domain),
            _compute_pressure(meo_satellite, graph, domain, computing_demand),
        ], dtype=np.float32),
    ])


def _domain_resource_features(meo_satellite, graph, domain, use_meo_aggregation):
    if use_meo_aggregation:
        node = graph.nodes[domain] if graph is not None and domain in graph else {}
        aggregate = node.get("aggregate", {}) or {}
        return np.asarray([
            _float(aggregate.get("avg_memory_occupancy_rate", node.get("avg_memory_occupancy_rate", 0.0))),
            _float(aggregate.get("std_memory_occupancy_rate", node.get("std_memory_occupancy_rate", node.get("max_memory_occupancy_rate", 0.0)))),
            _float(aggregate.get("avg_computing_occupancy_rate", node.get("avg_computing_occupancy_rate", 0.0))),
            _float(aggregate.get("std_computing_occupancy_rate", node.get("std_computing_occupancy_rate", node.get("max_computing_occupancy_rate", 0.0)))),
        ], dtype=np.float32)

    states = _domain_leo_states(meo_satellite, graph, domain)
    memory_capacity = getattr(meo_satellite, "memory", 1.0)
    computing_capacity = getattr(meo_satellite, "computing_ability", 1.0)
    memory_occupancy = [
        1.0 - _normalize_resource(state.get("remaining_memory", 0.0), memory_capacity)
        for state in states.values()
    ]
    computing_occupancy = [
        1.0 - _normalize_resource(state.get("remaining_computing", 0.0), computing_capacity)
        for state in states.values()
    ]
    return np.asarray([
        float(np.mean(memory_occupancy)) if memory_occupancy else 0.0,
        float(np.std(memory_occupancy)) if memory_occupancy else 0.0,
        float(np.mean(computing_occupancy)) if computing_occupancy else 0.0,
        float(np.std(computing_occupancy)) if computing_occupancy else 0.0,
    ], dtype=np.float32)


def _ordered_meo_neighbors(meo_satellite, graph, current_domain, max_neighbors):
    if graph is not None and current_domain in graph:
        candidates = [str(item) for item in graph.neighbors(current_domain)]
    else:
        candidates = []
    unique = sorted({item for item in candidates if item != current_domain})
    return unique[:max_neighbors]


def _aggregate_neighbor_features(graph, current_domain, neighbor, target_domain, distances, future_graph, src):
    aggregate = {}
    if graph is not None and neighbor in graph:
        aggregate = graph.nodes[neighbor].get("aggregate", {}) or {}
    avg_mem = _float(aggregate.get("avg_memory_occupancy_rate", graph.nodes[neighbor].get("avg_memory_occupancy_rate", 1.0) if graph is not None and neighbor in graph else 0.0))
    std_mem = _float(aggregate.get("std_memory_occupancy_rate", graph.nodes[neighbor].get("std_memory_occupancy_rate", graph.nodes[neighbor].get("max_memory_occupancy_rate", 1.0)) if graph is not None and neighbor in graph else 1.0))
    avg_comp = _float(aggregate.get("avg_computing_occupancy_rate", graph.nodes[neighbor].get("avg_computing_occupancy_rate", 1.0) if graph is not None and neighbor in graph else 1.0))
    std_comp = _float(aggregate.get("std_computing_occupancy_rate", graph.nodes[neighbor].get("std_computing_occupancy_rate", graph.nodes[neighbor].get("max_computing_occupancy_rate", 1.0)) if graph is not None and neighbor in graph else 1.0))
    source_boundary_delay, boundary_reachable = _source_to_neighbor_boundary_delay(graph, current_domain, neighbor, src)
    features = np.asarray([
        avg_mem,
        std_mem,
        avg_comp,
        std_comp,
        source_boundary_delay,
        _safe_distance(distances.get(neighbor)),
        float(neighbor == target_domain),
    ], dtype=np.float32)
    return features, boundary_reachable


def _source_to_neighbor_boundary_delay(graph, current_domain, neighbor, src):
    if src is None:
        raise ValueError("src is required when use_meo_aggregation=True")
    if graph is None:
        raise ValueError("inter-domain graph is required when use_meo_aggregation=True")
    if current_domain not in graph:
        raise ValueError(f"current domain {current_domain!r} is missing from inter-domain graph")
    if neighbor not in graph:
        raise ValueError(f"neighbor domain {neighbor!r} is missing from inter-domain graph")
    if not graph.has_edge(current_domain, neighbor):
        return UNREACHABLE_BOUNDARY_DELAY, False

    aggregate = graph.nodes[current_domain].get("aggregate", {}) or {}
    intra_graph = aggregate.get("intra_graph")
    if intra_graph is None:
        raise ValueError(f"current domain {current_domain!r} is missing aggregate.intra_graph")
    if src not in intra_graph:
        raise ValueError(f"src {src!r} is missing from intra_graph of domain {current_domain!r}")

    links = graph[current_domain][neighbor].get("boundary_links", {}) or {}
    delays = []
    for link in links.values():
        if not isinstance(link, dict):
            continue
        source_boundary = _source_boundary_for_neighbor_link(link, current_domain, neighbor)
        if source_boundary is None or source_boundary not in intra_graph:
            continue
        try:
            intra_delay = nx.shortest_path_length(intra_graph, src, source_boundary, weight=_edge_delay_weight)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        delays.append(float(intra_delay) + _float(link.get("delay", 0.0)))

    if not delays:
        return UNREACHABLE_BOUNDARY_DELAY, False
    return float(min(delays)), True


def _source_boundary_for_neighbor_link(link, current_domain, neighbor):
    source_domain = link.get("source_domain")
    target_domain = link.get("target_domain")
    if source_domain == current_domain and target_domain == neighbor:
        return link.get("source_boundary")
    if source_domain == neighbor and target_domain == current_domain:
        return link.get("target_boundary")
    return None


def _entry_node_load_features(graph, current_domain, neighbor):
    """Return target-entry memory/compute load min, mean and max for one action."""
    unavailable = np.ones(ENTRY_NODE_LOAD_FEATURE_DIM, dtype=np.float32)
    if graph is None or not graph.has_edge(current_domain, neighbor):
        return unavailable

    links = graph[current_domain][neighbor].get("boundary_links", {}) or {}
    entry_loads = {}
    for link in links.values():
        if not isinstance(link, dict):
            continue
        source_domain = link.get("source_domain")
        target_domain = link.get("target_domain")
        if source_domain == current_domain and target_domain == neighbor:
            entry = link.get("target_boundary")
            memory_key = "target_memory_occupancy_rate"
            computing_key = "target_computing_occupancy_rate"
        elif source_domain == neighbor and target_domain == current_domain:
            entry = link.get("source_boundary")
            memory_key = "source_memory_occupancy_rate"
            computing_key = "source_computing_occupancy_rate"
        else:
            continue
        if entry is None:
            continue

        memory_load = _entry_node_occupancy(
            graph, neighbor, entry, link.get(memory_key), "memory"
        )
        computing_load = _entry_node_occupancy(
            graph, neighbor, entry, link.get(computing_key), "computing"
        )
        previous = entry_loads.get(entry)
        if previous is None:
            entry_loads[entry] = (memory_load, computing_load)
        else:
            entry_loads[entry] = (
                max(previous[0], memory_load),
                max(previous[1], computing_load),
            )

    if not entry_loads:
        return unavailable
    memory_loads = np.asarray([loads[0] for loads in entry_loads.values()], dtype=np.float32)
    computing_loads = np.asarray([loads[1] for loads in entry_loads.values()], dtype=np.float32)
    return np.asarray([
        np.min(memory_loads),
        np.mean(memory_loads),
        np.max(memory_loads),
        np.min(computing_loads),
        np.mean(computing_loads),
        np.max(computing_loads),
    ], dtype=np.float32)


def _entry_node_occupancy(graph, domain, entry, boundary_value, resource):
    value = _finite_optional_float(boundary_value)
    if value is not None:
        return _clip01(value)

    aggregate = graph.nodes[domain].get("aggregate", {}) if domain in graph else {}
    intra_graph = aggregate.get("intra_graph") if isinstance(aggregate, dict) else None
    attrs = intra_graph.nodes[entry] if intra_graph is not None and entry in intra_graph else {}
    value = _finite_optional_float(attrs.get(f"{resource}_occupancy_rate"))
    if value is not None:
        return _clip01(value)
    remaining_ratio = _finite_optional_float(attrs.get(f"remaining_{resource}_ratio"))
    if remaining_ratio is not None:
        return _clip01(1.0 - remaining_ratio)
    return 1.0


def _edge_delay_weight(_u, _v, attrs):
    for key in ("delay", "propagation_delay", "weight"):
        if key in attrs:
            return _float(attrs.get(key), 0.0)
    return 0.0


def _raw_neighbor_features(meo_satellite, graph, neighbor, target_domain, distances, future_graph):
    states = _domain_leo_states(meo_satellite, graph, neighbor)
    mem_values = []
    comp_values = []
    producing = []
    link_loads = []
    boundary_count = 0
    for leo_name, state in states.items():
        mem_values.append(_normalize_resource(state.get("remaining_memory", 0.0), getattr(meo_satellite, "memory", 1.0)))
        comp_values.append(_normalize_resource(state.get("remaining_computing", 0.0), getattr(meo_satellite, "computing_ability", 1.0)))
        producing.append(float(bool(state.get("is_producing", 0))))
        is_boundary = False
        for neighbor_state in state.get("neighbors", []) or []:
            link_loads.append(_float(neighbor_state.get("link_load", 0.0)) / max(float(getattr(meo_satellite, "transmission_rate", 1.0)), 1.0))
            nbr = neighbor_state.get("name")
            nbr_sat = _satellite_by_name(meo_satellite, nbr)
            if nbr_sat is not None and getattr(nbr_sat, "masterMeo", None) != neighbor:
                is_boundary = True
        if is_boundary:
            boundary_count += 1
    count = max(len(states), 1)
    return np.asarray([
        float(np.mean(mem_values)) if mem_values else 0.0,
        float(np.min(mem_values)) if mem_values else 0.0,
        float(np.mean(comp_values)) if comp_values else 0.0,
        float(np.min(comp_values)) if comp_values else 0.0,
        float(np.mean(producing)) if producing else 0.0,
        float(np.mean(link_loads)) if link_loads else 0.0,
        float(boundary_count) / float(count),
        _safe_distance(distances.get(neighbor)),
    ], dtype=np.float32)


def _domain_leo_states(meo_satellite, graph, domain):
    if domain == getattr(meo_satellite, "name", None):
        return getattr(meo_satellite, "leoStates", None) or {}
    remote = getattr(meo_satellite, "remote_domain_leo_states", None) or {}
    states = remote.get(domain)
    if isinstance(states, dict) and "members" not in states:
        return states
    if graph is not None and domain in graph:
        aggregate = graph.nodes[domain].get("aggregate", {}) or {}
        intra_graph = aggregate.get("intra_graph")
        if intra_graph is not None:
            return {
                node: {
                    "remaining_memory": attrs.get("remaining_memory", 0.0),
                    "remaining_computing": attrs.get("remaining_computing", 0.0),
                    "is_producing": attrs.get("is_producing", 0),
                    "neighbors": [],
                }
                for node, attrs in intra_graph.nodes(data=True)
            }
    return {}


def _task_global_features(meo_satellite, task_context):
    task_context = task_context or {}
    max_size = max(_float(getattr(meo_satellite, "max_size", 1.0), 1.0), 1.0)
    return np.asarray([
        _float(task_context.get("task_type", 0.0)),
        _clip01(_float(task_context.get("packet_size", 0.0)) / max_size),
        _clip01(_float(task_context.get("computing_demand", 0.0)) / max(float(getattr(meo_satellite, "computing_ability", 1.0) or 1.0), 1.0)),
        _clip01(_float(task_context.get("size_after_computing", 0.0)) / max_size),
        float(bool(task_context.get("is_computed", False))),
    ], dtype=np.float32)


def _compute_pressure(meo_satellite, graph, domain, computing_demand):
    if computing_demand <= 0.0:
        return 0.0
    max_remaining = _max_domain_remaining_computing(meo_satellite, graph, domain)
    if max_remaining <= 0.0:
        return 1.0
    return _clip01(float(computing_demand) / max_remaining)


def _max_domain_remaining_computing(meo_satellite, graph, domain):
    remaining_values = []
    if graph is not None and domain in graph:
        aggregate = graph.nodes[domain].get("aggregate", {}) or {}
        intra_graph = aggregate.get("intra_graph")
        if intra_graph is not None:
            remaining_values.extend(
                _float(attrs["remaining_computing"])
                for _, attrs in intra_graph.nodes(data=True)
                if "remaining_computing" in attrs
            )
    if not remaining_values:
        states = _domain_leo_states(meo_satellite, graph, domain)
        remaining_values.extend(_float(state.get("remaining_computing", 0.0)) for state in states.values())
    return max(remaining_values) if remaining_values else 0.0


def _domain_distances(graph, target_domain):
    if graph is None or target_domain not in graph:
        return {}
    try:
        return nx.single_source_shortest_path_length(graph, target_domain)
    except (nx.NetworkXError, nx.NodeNotFound):
        return {}


def _satellite_by_name(meo_satellite, name):
    if name is None or getattr(meo_satellite, "propagator", None) is None:
        return None
    return meo_satellite.propagator.satellites.get(name)


def _safe_distance(value):
    if value is None or not np.isfinite(float(value)):
        return 1.0
    return min(float(value) / 8.0, 1.0)


def _normalize_resource(value, capacity):
    return max(0.0, min(float(value) / max(float(capacity), 1.0), 1.0))


def _clip01(value):
    return max(0.0, min(float(value), 1.0))


def _float(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(value):
        return float(default)
    return value


def _finite_optional_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None
