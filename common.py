import numpy as np
import math
import re
from typing import Optional, Tuple, Dict, Iterable

# WGS84 椭球参数
WGS84_A = 6378.137          # 地球长半轴，单位 km
WGS84_F = 1 / 298.257223563
WGS84_E2 = WGS84_F * (2 - WGS84_F)
LIGHT_SPEED_KM_S = 299792.458
DEFAULT_DELAY_KEYS = ("delay", "propagation_delay", "propagation_weight")


def _extract_tle_orbit_info(tle_filepath):
    """Extract orbit altitude and scale such as 72x22 from a TLE filename."""
    filename = str(tle_filepath).replace("\\", "/").rsplit("/", 1)[-1]
    scale_match = re.search(r"(\d+x\d+)", filename, re.IGNORECASE)
    altitude_match = re.search(r"(?:^|_)(\d+)(?=_)", filename)

    altitude = int(altitude_match.group(1)) if altitude_match else None
    scale = scale_match.group(1).lower() if scale_match else "unknown"
    return altitude, scale


def build_save_training_data_name(config):
    """
    根据配置生成训练过程数据文件名。

    文件名包含 LEO 规模、MEO 规模、数据包发包频率、训练/测试阶段、
    是否使用 transformer，各字段之间用下划线连接。
    """
    environment_cfg = config.get("environment", {})
    general_cfg = config.get("general", {})
    transformer_cfg = config.get("transformer", {})

    tle_filepaths = environment_cfg.get("tle_filepath", [])
    if isinstance(tle_filepaths, (str, bytes)):
        tle_filepaths = [tle_filepaths]

    leo_scale = "unknown"
    meo_scale = "unknown"
    for tle_filepath in tle_filepaths:
        altitude, scale = _extract_tle_orbit_info(tle_filepath)
        if altitude is None:
            continue
        if altitude <= 2000:
            leo_scale = scale
        else:
            meo_scale = scale

    packet_frequency = environment_cfg.get("packet_frequency", "unknown")
    select_mode = general_cfg.get("select_mode", "unknown")
    phase = general_cfg.get("phase", "unknown")
    transformer_text = (
        "withTransformer"
        if transformer_cfg.get("enabled", True)
        else "withoutTransformer"
    )
    use_meo_aggregation = (
        "useMeoAggregation"
        if transformer_cfg.get("use_meo_aggregation", True)
        else "noUseMeoAggregation"
    )
    meoPolicy_text = (
        "withMeoPolicy"
        if transformer_cfg.get("meo_exit_enabled", True)
        else "withoutMeoPolicy"
    )

    return (
        f"leo{leo_scale}_"
        f"meo{meo_scale}_"
        f"fre{packet_frequency}_"
        f"selectMode{select_mode}_"
        f"{phase}_"
        f"{meoPolicy_text}_"
        f"{use_meo_aggregation}_"
        f"{transformer_text}.txt"
    )


def _safe_div(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def _edge_identity(graph, u, v):
    """Return a comparable edge key, respecting directed/undirected graphs."""
    if graph.is_directed():
        return (u, v)
    return tuple(sorted((u, v), key=str))


def _numeric_array_or_none(value):
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if array.size == 0 or not np.all(np.isfinite(array)):
        return None
    return array

def z_score(data: Dict[str, float], target_key: str) -> float:
    """
    计算指定 key 对应 value 的 Z-score。

    参数:
        data: 字典，格式为 {str: float}
        target_key: 需要计算 Z-score 的 key

    返回:
        指定 key 对应 value 的 Z-score
    """

    if target_key not in data:
        raise KeyError(f"指定的 key 不存在: {target_key}")

    values = list(data.values())

    if len(values) == 0:
        raise ValueError("字典不能为空")

    mean = sum(values) / len(values)

    variance = sum((x - mean) ** 2 for x in values) / len(values)
    std = math.sqrt(variance)

    target_value = data[target_key]

    return (target_value - mean) / (std + 1e-4)

def _lookup_edge_delay(graph, u, v, data=None, delay_keys=DEFAULT_DELAY_KEYS):
    """Read edge delay from common attribute names, with position-based fallback."""
    edge_data = data if data is not None else graph.get_edge_data(u, v, default={})
    for key in delay_keys:
        value = edge_data.get(key)
        if value is not None:
            return float(value)

    if "pos_a" in edge_data and "pos_b" in edge_data:
        return float(np.linalg.norm(np.asarray(edge_data["pos_a"]) - np.asarray(edge_data["pos_b"])) / 3e5)
    if u in graph.nodes and v in graph.nodes and "pos" in graph.nodes[u] and "pos" in graph.nodes[v]:
        return float(np.linalg.norm(np.asarray(graph.nodes[u]["pos"]) - np.asarray(graph.nodes[v]["pos"])) / 3e5)
    return None


def graph_prediction_difference(
    predicted_graph,
    actual_graph,
    node_attrs: Optional[Iterable[str]] = None,
    delay_keys: Iterable[str] = DEFAULT_DELAY_KEYS,
):
    """
    比较预测图和实际图，输出节点信息、链路存在、链路时延三个层面的预测差异度。

    Parameters
    ----------
    predicted_graph:
        预测得到的 NetworkX graph。
    actual_graph:
        实际观测/仿真得到的 NetworkX graph。
    node_attrs:
        需要比较的节点属性。默认比较项目中常见的 pos、pos_0、velocity、sequence_num。
        只会计算两张图同一节点上都存在且可转成数值数组的属性。
    delay_keys:
        边上可能表示时延的属性名，默认支持 delay、propagation_delay、propagation_weight。

    Returns
    -------
    dict
        {
            "node": 节点集合和节点属性误差,
            "edge_existence": 链路存在预测指标,
            "delay": 共同链路上的时延误差
        }
    """
    if node_attrs is None:
        node_attrs = ("pos", "pos_0", "velocity", "sequence_num")
    node_attrs = tuple(node_attrs)
    delay_keys = tuple(delay_keys)

    predicted_nodes = set(predicted_graph.nodes())
    actual_nodes = set(actual_graph.nodes())
    common_nodes = predicted_nodes & actual_nodes
    missing_nodes = actual_nodes - predicted_nodes
    extra_nodes = predicted_nodes - actual_nodes

    node_attr_errors = {}
    all_node_abs_errors = []
    all_node_sq_errors = []
    for attr in node_attrs:
        attr_abs_errors = []
        attr_sq_errors = []
        compared = 0
        skipped = 0
        for node in common_nodes:
            pred_value = _numeric_array_or_none(predicted_graph.nodes[node].get(attr))
            actual_value = _numeric_array_or_none(actual_graph.nodes[node].get(attr))
            if pred_value is None or actual_value is None or pred_value.shape != actual_value.shape:
                skipped += 1
                continue
            diff = pred_value - actual_value
            abs_diff = np.abs(diff)
            attr_abs_errors.extend(abs_diff.tolist())
            attr_sq_errors.extend((diff ** 2).tolist())
            compared += 1

        if attr_abs_errors:
            abs_array = np.asarray(attr_abs_errors, dtype=float)
            sq_array = np.asarray(attr_sq_errors, dtype=float)
            node_attr_errors[attr] = {
                "compared_nodes": compared,
                "skipped_nodes": skipped,
                "mae": float(abs_array.mean()),
                "rmse": float(np.sqrt(sq_array.mean())),
                "max_error": float(abs_array.max()),
            }
            all_node_abs_errors.extend(attr_abs_errors)
            all_node_sq_errors.extend(attr_sq_errors)
        else:
            node_attr_errors[attr] = {
                "compared_nodes": 0,
                "skipped_nodes": skipped,
                "mae": None,
                "rmse": None,
                "max_error": None,
            }

    if all_node_abs_errors:
        node_abs_array = np.asarray(all_node_abs_errors, dtype=float)
        node_sq_array = np.asarray(all_node_sq_errors, dtype=float)
        node_numeric_mae = float(node_abs_array.mean())
        node_numeric_rmse = float(np.sqrt(node_sq_array.mean()))
    else:
        node_numeric_mae = None
        node_numeric_rmse = None

    predicted_edges = {
        _edge_identity(predicted_graph, u, v)
        for u, v in predicted_graph.edges()
    }
    actual_edges = {
        _edge_identity(actual_graph, u, v)
        for u, v in actual_graph.edges()
    }
    common_edges = predicted_edges & actual_edges
    missing_edges = actual_edges - predicted_edges
    extra_edges = predicted_edges - actual_edges
    all_possible_edges = len(common_nodes) * max(len(common_nodes) - 1, 0)
    if not predicted_graph.is_directed() and not actual_graph.is_directed():
        all_possible_edges //= 2
    true_negative = max(all_possible_edges - len(common_edges) - len(missing_edges) - len(extra_edges), 0)

    precision = _safe_div(len(common_edges), len(predicted_edges))
    recall = _safe_div(len(common_edges), len(actual_edges))
    f1 = _safe_div(2 * precision * recall, precision + recall)

    delay_abs_errors = []
    delay_sq_errors = []
    delay_relative_errors = []
    skipped_delay_edges = 0
    for edge in common_edges:
        u, v = edge
        pred_delay = _lookup_edge_delay(predicted_graph, u, v, delay_keys=delay_keys)
        actual_delay = _lookup_edge_delay(actual_graph, u, v, delay_keys=delay_keys)
        if pred_delay is None or actual_delay is None:
            skipped_delay_edges += 1
            continue
        if not math.isfinite(pred_delay) or not math.isfinite(actual_delay):
            skipped_delay_edges += 1
            continue
        diff = float(pred_delay - actual_delay)
        delay_abs_errors.append(abs(diff))
        delay_sq_errors.append(diff ** 2)
        if abs(actual_delay) > 1e-12:
            delay_relative_errors.append(abs(diff) / abs(actual_delay))

    if delay_abs_errors:
        delay_abs_array = np.asarray(delay_abs_errors, dtype=float)
        delay_sq_array = np.asarray(delay_sq_errors, dtype=float)
        delay_mae = float(delay_abs_array.mean())
        delay_rmse = float(np.sqrt(delay_sq_array.mean()))
        delay_max_error = float(delay_abs_array.max())
    else:
        delay_mae = None
        delay_rmse = None
        delay_max_error = None

    return {
        "node": {
            "actual_count": len(actual_nodes),
            "predicted_count": len(predicted_nodes),
            "common_count": len(common_nodes),
            "missing_count": len(missing_nodes),
            "extra_count": len(extra_nodes),
            "missing_rate": _safe_div(len(missing_nodes), len(actual_nodes)),
            "extra_rate": _safe_div(len(extra_nodes), len(predicted_nodes)),
            "numeric_mae": node_numeric_mae,
            "numeric_rmse": node_numeric_rmse,
            "attr_errors": node_attr_errors,
        },
        "edge_existence": {
            "actual_count": len(actual_edges),
            "predicted_count": len(predicted_edges),
            "correct_count": len(common_edges),
            "missing_count": len(missing_edges),
            "extra_count": len(extra_edges),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": _safe_div(len(common_edges) + true_negative, all_possible_edges),
            "jaccard_distance": 1.0 - _safe_div(len(common_edges), len(predicted_edges | actual_edges)),
        },
        "delay": {
            "compared_edge_count": len(delay_abs_errors),
            "skipped_edge_count": skipped_delay_edges,
            "mae": delay_mae,
            "rmse": delay_rmse,
            "max_error": delay_max_error,
            "mape": float(np.mean(delay_relative_errors)) if delay_relative_errors else None,
        },
    }

def isLeo(name):
    high = int(name.split('_')[1])
    if high <= 2000:
        return True
    else:
        return False

def isMeoNeighbor(src, dst, satellites):
    if src == dst or isLeo(src) or isLeo(dst):
        return False

    meo_a = satellites.get(src) if isinstance(satellites, dict) else None
    meo_b = satellites.get(dst) if isinstance(satellites, dict) else None
    if meo_a is None or meo_b is None:
        return False

    return (dst in getattr(meo_a, "neighbors", []))

def _get_satellite_lla(G, node):
    """从图节点属性中读取卫星的 (lat, lon, alt)。"""
    pos_0 = G.nodes[node].get('pos_0')
    if pos_0 is None or len(pos_0) < 3:
        raise ValueError(f"Node {node} is missing pos_0=[lat, lon, alt]")
    return float(pos_0[0]), float(pos_0[1]), float(pos_0[2])

def _get_satellite_eci( G, node):
    """从图节点属性中读取卫星的ECI坐标"""
    pos = G.nodes[node].get('pos')
    if pos is None or len(pos) < 3:
        raise ValueError(f"Node {node} is missing pos=[x, y, z]")
    return float(pos[0]), float(pos[1]), float(pos[2])


def eci_distance(pos1_km, pos2_km) -> float:
    """计算两颗卫星 ECI 坐标之间的直线距离，单位 km。"""
    pos1 = np.asarray(pos1_km, dtype=float).reshape(-1)
    pos2 = np.asarray(pos2_km, dtype=float).reshape(-1)

    if pos1.size != 3 or pos2.size != 3:
        raise ValueError("pos1_km and pos2_km must be 3D vectors.")

    return float(np.linalg.norm(pos2 - pos1))


def eci_propagation_delay(pos1_km, pos2_km, propagation_speed_km_s: float = LIGHT_SPEED_KM_S) -> float:
    """根据两颗卫星的 ECI 坐标计算传播时延，单位秒。"""
    if propagation_speed_km_s <= 0:
        raise ValueError("propagation_speed_km_s must be positive.")
    return eci_distance(pos1_km, pos2_km) / float(propagation_speed_km_s)


def free_space_path_loss_db(graph, source, destination, carrierFrequency) -> float:
    """按 L_fs=32.4+20log10(f_MHz)+20log10(d_km) 计算自由空间路径损耗。"""
    if source not in graph.nodes:
        raise ValueError(f"Node {source} is not in graph.")
    if destination not in graph.nodes:
        raise ValueError(f"Node {destination} is not in graph.")

    source_pos = np.asarray(graph.nodes[source].get("pos"), dtype=float).reshape(-1)
    destination_pos = np.asarray(graph.nodes[destination].get("pos"), dtype=float).reshape(-1)
    if source_pos.size != 3 or destination_pos.size != 3:
        raise ValueError("source and destination nodes must have pos=[x, y, z] in km.")

    distance_km = float(np.linalg.norm(destination_pos - source_pos))
    if distance_km <= 0:
        raise ValueError("source and destination distance must be positive.")

    return float(32.4 + 20 * math.log10(carrierFrequency) + 20 * math.log10(distance_km))

def computeSnr(graph, satellites)->float:
    snr = {}
    for node,next_node in graph.edges():
        send_power = satellites[node].power
        frequency = satellites[node].communication_frequency
        loss_freeSpace = free_space_path_loss_db(graph, node, next_node, frequency)
        received_power = send_power - loss_freeSpace
        noise_dbm = -174 + math.log10(satellites[node].transmission_rate) + 3
        snr[(node, next_node)] = received_power - noise_dbm
    return snr
    
def eci_speed(velocity_km_s) -> float:
    """
    根据卫星的 ECI 速度向量计算运动速度大小，单位 km/s。

    参数
    ----
    velocity_km_s:
        卫星在 ECI 坐标系下的速度向量 [vx, vy, vz]，单位 km/s。

    返回
    ----
    float
        卫星运动速度大小，单位 km/s。
    """
    velocity = np.asarray(velocity_km_s, dtype=float).reshape(-1)

    if velocity.size != 3:
        raise ValueError("velocity_km_s must be a 3D vector.")

    return float(np.linalg.norm(velocity))


def _get_satellite_velocity(G, node):
    """从图节点属性中读取卫星的 ECI 速度向量，单位 km/s。"""
    velocity = G.nodes[node].get('velocity')
    if velocity is None:
        raise ValueError(f"Node {node} is missing velocity=[vx, vy, vz]")

    velocity = np.asarray(velocity, dtype=float).reshape(-1)
    if velocity.size != 3:
        raise ValueError(
            f"Node {node} has invalid velocity shape; expected 3 values, got {velocity.size}"
        )
    return float(velocity[0]), float(velocity[1]), float(velocity[2])


def get_satellite_speed(G, node) -> float:
    """从图节点属性中读取 ECI 速度向量，并计算卫星运动速度大小，单位 km/s。"""
    return eci_speed(_get_satellite_velocity(G, node))

def geodetic_to_ecef(lat_deg, lon_deg, alt_km):
    """将大地坐标(lat, lon, alt)转换为ECEF坐标,单位 km。"""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    N = WGS84_A / np.sqrt(1 - WGS84_E2 * sin_lat ** 2)

    x = (N + alt_km) * cos_lat * cos_lon
    y = (N + alt_km) * cos_lat * sin_lon
    z = (N * (1 - WGS84_E2) + alt_km) * sin_lat
    return np.array([x, y, z], dtype=float)

def _dot(a, b) -> float:
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


def _sub(a, b):
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])


def _add(a, b):
    return (a[0]+b[0], a[1]+b[1], a[2]+b[2])


def _mul(s: float, v):
    return (s*v[0], s*v[1], s*v[2])


def _norm(v) -> float:
    return math.sqrt(_dot(v, v))


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def can_build_isl(
    pos1_km,
    vel1_km_s,
    carrier_frequency_1_ghz: float,
    bandwidth_1:float,
    power_1:float,
    pos2_km,
    vel2_km_s,
    carrier_frequency_2_ghz: float,
    doppler_threshold_mhz: float,
    bandwidth_2:float,
    power_2:float,
    *,
    earth_radius_km: float = 6371.0,
    min_clearance_km: float = 0.0,
    max_visible_distance_km: float = 3000,
    snr_thread_db: float = 10
) -> Tuple[bool, Dict[str, float]]:
    """
    根据两颗卫星的 ECI 位置、ECI 速度、载频和多普勒门限，
    判断是否能建立星间链路。

    参数
    ----
    pos1_km, pos2_km:
        两颗卫星在同一时刻的 ECI 位置向量 [x, y, z]，单位 km。
    vel1_km_s, vel2_km_s:
        两颗卫星在同一时刻的 ECI 速度向量 [vx, vy, vz]，单位 km/s。
    carrier_frequency_1_ghz, carrier_frequency_2_ghz:
        两颗卫星发送信号的载波频率，单位 GHz。
    doppler_threshold_ghz:
        可接受的多普勒频移门限，单位 GHz。
    earth_radius_km : 用于遮挡判定的等效地球半径
    min_clearance_km : 连线距地表最小净空要求，默认 0 km

    返回
    ----
    (can_link, info)
    can_link: bool
    info: {
        "distance_km": 两星直线距离,
        "clearance_km": 连线到地表的最小净空,
        "t_closest": 最近点在线段中的参数[0,1],
        "max_visible_distance_km": 由两星高度决定的最大几何可见距离,
        "doppler_1_to_2_ghz": 卫星1发、卫星2收的多普勒频移,
        "doppler_2_to_1_ghz": 卫星2发、卫星1收的多普勒频移,
        "doppler_limit_ghz": 多普勒门限
    }
    """
    doppler_threshold_ghz = doppler_threshold_mhz / 1000
    pos1 = np.asarray(pos1_km, dtype=float).reshape(-1)
    pos2 = np.asarray(pos2_km, dtype=float).reshape(-1)
    vel1 = np.asarray(vel1_km_s, dtype=float).reshape(-1)
    vel2 = np.asarray(vel2_km_s, dtype=float).reshape(-1)

    if pos1.size != 3 or pos2.size != 3:
        raise ValueError("pos1_km and pos2_km must be 3D vectors.")
    if vel1.size != 3 or vel2.size != 3:
        raise ValueError("vel1_km_s and vel2_km_s must be 3D vectors.")
    if carrier_frequency_1_ghz < 0 or carrier_frequency_2_ghz < 0:
        raise ValueError("carrier frequencies must be non-negative.")
    if doppler_threshold_ghz < 0:
        raise ValueError("doppler_threshold_ghz must be non-negative.")

    r1 = pos1
    r2 = pos2

    d = _sub(r2, r1)
    d2 = _dot(d, d)
    if d2 == 0:
        return False, {
            "distance_km": 0.0,
            "clearance_km": -earth_radius_km,
            "t_closest": 0.0,
            "max_visible_distance_km": 0.0,
            "doppler_1_to_2_ghz": 0.0,
            "doppler_2_to_1_ghz": 0.0,
            "doppler_limit_ghz": float(doppler_threshold_ghz),
        }

    # 两星距离
    distance_km = math.sqrt(d2)

    # 求线段 r(t)=r1+t(r2-r1), t∈[0,1] 上离地心最近的点
    t = _clip(-_dot(r1, d) / d2, 0.0, 1.0)
    p_closest = _add(r1, _mul(t, d))

    # 最近点到地表的最小净空
    clearance_km = _norm(p_closest) - earth_radius_km

    alt1_km = _norm(r1) - earth_radius_km
    alt2_km = _norm(r2) - earth_radius_km
    if max_visible_distance_km == 0:
        visible_distance_km = max_geometric_visible_distance(
            alt1_km,
            alt2_km,
            earth_radius_km=earth_radius_km,
        )
    else:
        visible_distance_km = max_visible_distance_km
    doppler_1_to_2_ghz, doppler_2_to_1_ghz = doppler_shift_value(
        pos1,
        vel1,
        carrier_frequency_1_ghz,
        pos2,
        vel2,
        carrier_frequency_2_ghz,
    )

    # 判定1：是否被地球遮挡
    los_ok = clearance_km >= min_clearance_km

    # 判定2：是否超过由两星海拔决定的最大几何可见距离
    range_ok = distance_km <= visible_distance_km

    # 判定3：双向多普勒频移都不能超过门限
    doppler_ok = (
        abs(doppler_1_to_2_ghz) <= doppler_threshold_ghz
        and abs(doppler_2_to_1_ghz) <= doppler_threshold_ghz
    )
    # 判定4：双向SNR都不能超过门限
    loss_1to2 = float(32.4 + 20 * math.log10(carrier_frequency_1_ghz)) + 20 * math.log10(distance_km)
    noise_1to2_dbm = -174 + math.log10(bandwidth_1) + 3
    snr_1to2 = power_2 - loss_1to2 - noise_1to2_dbm
    
    loss_2to1 = float(32.4 + 20 * math.log10(carrier_frequency_2_ghz)) + 20 * math.log10(distance_km)
    noise_2to1_dbm = -174 + math.log10(bandwidth_2) + 3
    snr_2to1 = power_1 - loss_2to1 - noise_2to1_dbm
    snr_ok = (snr_1to2 > snr_thread_db and snr_2to1 > snr_thread_db)
    
    can_link = los_ok and range_ok and doppler_ok and snr_ok

    return can_link, {
        "distance_km": distance_km,
        "clearance_km": clearance_km,
        "t_closest": t,
        "visible_distance_km": visible_distance_km,
        "doppler_1_to_2_ghz": doppler_1_to_2_ghz,
        "doppler_2_to_1_ghz": doppler_2_to_1_ghz,
        "doppler_limit_ghz": float(doppler_threshold_ghz),
        "snr_1to2": snr_1to2,
        "snr_2to1": snr_2to1,
    }
    
def max_geometric_visible_distance(
    alt1_km: float,
    alt2_km: float,
    earth_radius_km: float = 6371.0,
) -> float:
    """
    计算两颗卫星在“仅考虑地球遮挡”的情况下，
    几何可见性的最大直线距离（单位：km）。

    注意：
    1. 这个最大距离只由两颗卫星的海拔决定，经纬度不会影响结果；
    2. 保留经纬度参数，是为了和你现有的卫星输入格式保持一致。

    参数
    ----
    lat1_deg, lon1_deg : 第一颗卫星经纬度（本函数中不参与计算）
    alt1_km            : 第一颗卫星海拔 km
    lat2_deg, lon2_deg : 第二颗卫星经纬度（本函数中不参与计算）
    alt2_km            : 第二颗卫星海拔 km
    earth_radius_km    : 地球等效半径，默认 6371.0 km

    返回
    ----
    float : 最大几何可见直线距离，单位 km
    """
    if alt1_km < 0 or alt2_km < 0:
        raise ValueError("卫星海拔不能为负数。")

    r1 = earth_radius_km + alt1_km
    r2 = earth_radius_km + alt2_km

    d1 = math.sqrt(r1 * r1 - earth_radius_km * earth_radius_km)
    d2 = math.sqrt(r2 * r2 - earth_radius_km * earth_radius_km)

    return d1 + d2


def radial_relative_velocity(
    pos1_km,
    vel1_km_s,
    pos2_km,
    vel2_km_s,
) -> float:
    """
    计算两颗卫星沿星间视线方向的径向相对速度，单位 km/s。

    参数
    ----
    pos1_km, pos2_km:
        两颗卫星在同一时刻的 ECI 位置向量，形如 [x, y, z]，单位 km。
    vel1_km_s, vel2_km_s:
        两颗卫星在同一时刻的 ECI 速度向量，形如 [vx, vy, vz]，单位 km/s。

    返回
    ----
    float
        径向相对速度，单位 km/s。
        正值表示两星距离正在增大，负值表示两星距离正在减小。

    说明
    ----
    “二者之间”的径向相对速度必须结合相对位置方向来定义，
    因此仅输入两个速度向量本身是不够的。
    """
    pos1 = np.asarray(pos1_km, dtype=float).reshape(-1)
    vel1 = np.asarray(vel1_km_s, dtype=float).reshape(-1)
    pos2 = np.asarray(pos2_km, dtype=float).reshape(-1)
    vel2 = np.asarray(vel2_km_s, dtype=float).reshape(-1)

    if pos1.size != 3 or pos2.size != 3:
        raise ValueError("pos1_km and pos2_km must be 3D vectors.")
    if vel1.size != 3 or vel2.size != 3:
        raise ValueError("vel1_km_s and vel2_km_s must be 3D vectors.")

    relative_position = pos2 - pos1
    distance_km = np.linalg.norm(relative_position)
    if distance_km == 0:
        raise ValueError("The two satellites have identical positions, so the LOS direction is undefined.")

    los_unit = relative_position / distance_km
    relative_velocity = vel2 - vel1
    return float(np.dot(relative_velocity, los_unit))


def doppler_shift_value(
    pos1_km,
    vel1_km_s,
    carrier_frequency_1_ghz,
    pos2_km,
    vel2_km_s,
    carrier_frequency_2_ghz,
):
    """
    计算两颗卫星之间的多普勒频移，单位 GHz。

    参数
    ----
    pos1_km, pos2_km:
        两颗卫星在同一时刻的 ECI 位置向量 [x, y, z]，单位 km。
    vel1_km_s, vel2_km_s:
        两颗卫星在同一时刻的 ECI 速度向量 [vx, vy, vz]，单位 km/s。
    carrier_frequency_1_ghz, carrier_frequency_2_ghz:
        两颗卫星发送信号的载波频率，单位 GHz。

    返回
    ----
    tuple[float, float]
        `(doppler_1_to_2_ghz, doppler_2_to_1_ghz)`。
        第一个值表示“卫星1发射、卫星2接收”的多普勒频移，
        第二个值表示“卫星2发射、卫星1接收”的多普勒频移。

    说明
    ----
    使用经典近似公式:
        delta_f = - f_c * v_r / c
    其中:
    - f_c 为发送端载频
    - v_r 为两星沿视线方向的径向相对速度
    - c 为光速

    当 v_r > 0 时表示两星远离，因此频移为负；
    当 v_r < 0 时表示两星接近，因此频移为正。
    """
    if carrier_frequency_1_ghz < 0 or carrier_frequency_2_ghz < 0:
        raise ValueError("carrier frequencies must be non-negative.")

    radial_speed_km_s = radial_relative_velocity(pos1_km, vel1_km_s, pos2_km, vel2_km_s)
    doppler_1_to_2_ghz = -carrier_frequency_1_ghz * radial_speed_km_s / LIGHT_SPEED_KM_S
    doppler_2_to_1_ghz = carrier_frequency_2_ghz * radial_speed_km_s / LIGHT_SPEED_KM_S
    return float(doppler_1_to_2_ghz), float(doppler_2_to_1_ghz)


def satellite_adjacency_matrix(graph):
    """
    根据图结构生成卫星邻接矩阵。

    参数
    ----
    graph:
        NetworkX 图对象。

    返回
    ----
    np.ndarray
        一个 N x N 的 0/1 邻接矩阵，N 为图中的节点数。
        若卫星 i 和卫星 j 为邻居，则 matrix[i, j] = 1，否则为 0。

    说明
    ----
    1. 行列顺序与 list(graph.nodes) 保持一致；
    2. 对于无向图，结果矩阵关于主对角线对称；
    3. 默认不在主对角线上填 1。
    """
    satellite_nodes = list(graph.nodes)
    node_to_index = {node: idx for idx, node in enumerate(satellite_nodes)}
    node_count = len(satellite_nodes)
    matrix = np.zeros((node_count, node_count), dtype=int)

    for node_u, node_v in graph.edges:
        idx_u = node_to_index[node_u]
        idx_v = node_to_index[node_v]
        matrix[idx_u, idx_v] = 1
        matrix[idx_v, idx_u] = 1

    return matrix


def plot_dict_line_chart(data_dict, title,yName, xName = "Step",  save_path=None, show=True):
    """
    根据字典绘制折线图。

    参数
    ----
    data_dict:
        输入字典，key 作为横轴，value 作为纵轴。
    title:
        图标题。
    xName:
        横轴名称。
    yName:
        纵轴名称。
    save_path:
        可选，图片保存路径，例如 "result.svg"。
        为 None 时默认按照 title 名称保存为 svg。
    show:
        是否显示图片。训练脚本中只想保存图片时可设为 False。

    返回
    ----
    (fig, ax):
        matplotlib 的 Figure 和 Axes 对象，方便后续继续调整。
    """
    if not data_dict:
        raise ValueError("data_dict cannot be empty.")

    import matplotlib.pyplot as plt

    x_values = list(data_dict.keys())
    y_values = list(data_dict.values())

    fig, ax = plt.subplots()
    ax.plot(x_values, y_values, marker="o")
    ax.set_title(title)
    ax.set_xlabel(xName)
    ax.set_ylabel(yName)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    if save_path is None:
        safe_title = "".join(char if char.isalnum() or char in (" ", "_", "-") else "_" for char in str(title))
        save_path = f"{safe_title.strip() or 'line_chart'}.svg"

    fig.savefig(save_path, format="svg")
    if show:
        plt.show()

    return fig, ax
