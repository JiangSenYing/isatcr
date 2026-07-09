"""Observation construction for MEO inter-domain routing."""

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np


MAX_MEO_NEIGHBORS = 4
AGG_FEATURE_DIM = 7
RAW_FEATURE_DIM = 8
GLOBAL_FEATURE_DIM = 4
TASK_GLOBAL_FEATURE_DIM = 5
TASK_PER_NEIGHBOR_FEATURE_DIM = 1
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
        rows[idx] = np.concatenate([
            neighbor_features,
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
    avg_mem = _float(aggregate.get("avg_memory_occupancy_rate", graph.nodes[neighbor].get("avg_memory_occupancy_rate", 0.0) if graph is not None and neighbor in graph else 0.0))
    std_mem = _float(aggregate.get("std_memory_occupancy_rate", graph.nodes[neighbor].get("std_memory_occupancy_rate", graph.nodes[neighbor].get("max_memory_occupancy_rate", 0.0)) if graph is not None and neighbor in graph else 0.0))
    avg_comp = _float(aggregate.get("avg_computing_occupancy_rate", graph.nodes[neighbor].get("avg_computing_occupancy_rate", 0.0) if graph is not None and neighbor in graph else 0.0))
    std_comp = _float(aggregate.get("std_computing_occupancy_rate", graph.nodes[neighbor].get("std_computing_occupancy_rate", graph.nodes[neighbor].get("max_computing_occupancy_rate", 0.0)) if graph is not None and neighbor in graph else 0.0))
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
