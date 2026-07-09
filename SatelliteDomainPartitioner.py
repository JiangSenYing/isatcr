import itertools
import os
import struct
from collections import Counter, defaultdict
import statistics
import networkx as nx
import numpy as np
from common import geodetic_to_ecef,WGS84_A,WGS84_F,WGS84_E2,LIGHT_SPEED_KM_S
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
import copy
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# # WGS84 椭球参数
# WGS84_A = 6378.137          # 地球长半轴，单位 km
# WGS84_F = 1 / 298.257223563
# WGS84_E2 = WGS84_F * (2 - WGS84_F)
# LIGHT_SPEED_KM_S = 299792.458


# def geodetic_to_ecef(lat_deg, lon_deg, alt_km):
#     """将大地坐标(lat, lon, alt)转换为ECEF坐标，单位 km。"""
#     lat = np.radians(lat_deg)
#     lon = np.radians(lon_deg)

#     sin_lat = np.sin(lat)
#     cos_lat = np.cos(lat)
#     sin_lon = np.sin(lon)
#     cos_lon = np.cos(lon)

#     N = WGS84_A / np.sqrt(1 - WGS84_E2 * sin_lat ** 2)

#     x = (N + alt_km) * cos_lat * cos_lon
#     y = (N + alt_km) * cos_lat * sin_lon
#     z = (N * (1 - WGS84_E2) + alt_km) * sin_lat
#     return np.array([x, y, z], dtype=float)


def ecef_to_enu_vector(dx, dy, dz, ref_lat_deg, ref_lon_deg):
    """将ECEF中的相对向量转换到参考点(ref_lat, ref_lon)的ENU坐标系。"""
    lat = np.radians(ref_lat_deg)
    lon = np.radians(ref_lon_deg)

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(ref_lon_deg * np.pi / 180.0)
    cos_lon = np.cos(ref_lon_deg * np.pi / 180.0)

    E = -sin_lon * dx + cos_lon * dy
    N = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    U = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz
    return E, N, U


def satellite_elevation(observer_lat, observer_lon, observer_alt_km,
                        target_lat, target_lon, target_alt_km):
    """计算：从 observer 看 target 的仰角（单位：度）。"""
    p1 = geodetic_to_ecef(observer_lat, observer_lon, observer_alt_km)
    p2 = geodetic_to_ecef(target_lat, target_lon, target_alt_km)
    dx, dy, dz = p2 - p1
    E, N, U = ecef_to_enu_vector(dx, dy, dz, observer_lat, observer_lon)
    return float(np.degrees(np.arctan2(U, np.sqrt(E ** 2 + N ** 2))))



def central_angles_to_latlon(lat_angle_deg, lon_angle_deg):
    """兼容旧代码接口：当前框架里输入本身就是带符号经纬度。"""
    lat_deg = float(lat_angle_deg)
    lon_deg = (float(lon_angle_deg) + 180.0) % 360.0 - 180.0
    if not (-90.0 <= lat_deg <= 90.0):
        raise ValueError("纬度角必须在 [-90, 90] 度之间")
    return lat_deg, lon_deg



def generate_ControlDomain_Matrix(row, col):
    matrix = np.zeros((row, col), dtype=int)
    rows = np.random.randint(0, row, size=col)
    matrix[rows, np.arange(col)] = 1
    return matrix


class SatelliteDomainPartitioner:
    """
    在现有 iSatCR 框架内，对论文 Eunomia 的三步分域流程做一个可运行的工程化实现：

    1. FOV 约束下的初始分域；
    2. 对重叠区域基于 CORG 的谱聚类；
    3. 聚类-控制器匹配 + 移动感知边界微调。

    说明：
    - 当前框架没有 GS 控制器节点，也没有论文 Plotinus/NS-3 那套控制面消息实现；
      因此这里在本框架里采用 “MEO 作为控制器，LEO 作为被控交换节点” 的落地版本。
    - CORG 的三项代价（flow/sync/migration）采用与论文一致的三项结构和权重，
      但根据当前框架可获得的数据做了可执行近似：
        * flow 主要由 LEO-LIO 图上的传播时延/跳数近似；
        * sync 由拓扑局部差异 + 缓存占用差异近似；
        * migration 由方向差异 + FOV 候选差异 + 历史控制器切换近似。
    """

    def __init__(self, orbitNumber, leoSatNumberPerOrbit, episode_max,
                 controlDomainNumber, delayMap, satellite_names,
                 minimuElevationAngle, satellites,showLink,
                 partitionMethod='Eunomia', rectangular_m=None, rectangular_n=None):
        self.orbitNumber = orbitNumber
        self.leoSatNumberPerOrbit = leoSatNumberPerOrbit
        self.episode_max = episode_max
        self.controlDomainNumber = controlDomainNumber
        self.delayMap = delayMap
        self.satellite_names = satellite_names
        self.minimuElevationAngle = minimuElevationAngle
        self.satellites = satellites
        self.partitionMethod = partitionMethod
        self.rectangular_m = int(rectangular_m) if rectangular_m is not None else 1
        self.rectangular_n = int(rectangular_n) if rectangular_n is not None else 1
        self.leoSattlitesPosition = {}
        self.meoSattlitesPosition = {}

        # 当前时隙结果：{controller_name: [leo1, leo2, ...]}
        self.currentPartitionResult = {}
        self.currentAssignment = {}
        self.previousAssignment = {}
        self.activeControllers = []
        self.last_partition_summary = {}

        # 论文中的经验权重 α=0.5, β=0.3
        self.alpha = 0.5
        self.beta = 0.3
        self.gamma = 1.0 - self.alpha - self.beta

        # 近似的控制面参数，保持与论文数量级一致
        self.flow_msg_bytes = 36.0
        self.sync_msg_bytes = 24.0
        self.state_transfer_bytes = 256.0
        self.handover_notify_bytes = 16.0
        self.sync_frequency = 1.0
        self.default_bandwidth_bytes_per_s = 1.2e9 / 8.0  # 由项目 transmission_rate 的数量级近似
        self.gaussian_sigma = 0.35

        self.totalMatrix = {}
        self.partition_plot_dir = "partition_plots"
        self.partition_plot_index = 0
        self.partition_plot_enabled = True
        self._controller_color_map = {}
        self._world_land_polygons = None
        self.boundary_satellites = {}
        self.neighbors_domain = []
        self.adjacency_table = {}
        self.showLink = showLink
        self._initial_leo_state_sync_done = False

    # =========================
    # 对外主入口
    # =========================
    def calcDomainPartition(self, graph, time):
        """执行一次完整分域，并把结果写回 satellites / graph。"""
        self.neighbors_domain.clear()
        self.boundary_satellites.clear()
        self.adjacency_table.clear()
        self.updateSatllitesPosition(graph)
        if str(self.partitionMethod).lower() == 'rectangulardivision':
            return self._calc_rectangular_domain_partition(graph, time)
        else:
            self.activeControllers = self._select_active_controllers()
            if len(self.activeControllers) <= 0:
                return None
            for meo in self.activeControllers:
                self.boundary_satellites[meo] = []
                self.adjacency_table[meo] = {}
            self.currentPartitionResult = {controller: [] for controller in self.activeControllers}
            self.currentAssignment = {}

            leo_subgraph = graph.subgraph(self.leoSattlitesPosition.keys()).copy()
            self._annotate_graph_edge_delay(leo_subgraph)

            fov_candidates = self._build_fov_candidates()

            # Step 1: 非重叠区域直接分配
            overlap_nodes = []
            fallback_count = 0
            non_overlap_count = 0
            for leo_name, candidates in fov_candidates.items():
                if len(candidates) == 1:
                    controller = candidates[0]
                    self.currentAssignment[leo_name] = controller
                    self.currentPartitionResult[controller].append(leo_name)
                    non_overlap_count += 1
                else:
                    overlap_nodes.append(leo_name)
                    if len(candidates) == 0:
                        fallback_count += 1

            # Step 2 + 3: 对重叠区域做谱聚类 + 控制器匹配 + 微调
            overlap_regions = self._build_overlap_regions(leo_subgraph, overlap_nodes, fov_candidates)
            for region in overlap_regions:
                if len(region['controllers']) == 0:
                    # 没有任何控制器满足FOV，退化为最近控制器兜底。
                    for leo_name in region['nodes']:
                        controller = self._fallback_controller_for_leo(leo_name)
                        self._assign(leo_name, controller)
                    continue

                if len(region['controllers']) == 1:
                    controller = region['controllers'][0]
                    for leo_name in region['nodes']:
                        self._assign(leo_name, controller)
                    continue

                clusters = self._spectral_cluster_overlap_region(
                    leo_subgraph=leo_subgraph,
                    region_nodes=region['nodes'],
                    region_controllers=region['controllers'],
                    fov_candidates=fov_candidates,
                )

                cluster_to_controller = self._match_clusters_to_controllers(
                    clusters=clusters,
                    controllers=region['controllers'],
                    fov_candidates=fov_candidates,
                )

                for cluster_idx, nodes in enumerate(clusters):
                    controller = cluster_to_controller.get(cluster_idx)
                    if controller is None:
                        # 仍无法匹配时，再退回到单点 greedy 分配
                        for leo_name in nodes:
                            self._assign(leo_name, self._best_feasible_controller(leo_name, fov_candidates[leo_name]))
                        continue
                    for leo_name in nodes:
                        if controller in fov_candidates[leo_name]:
                            self._assign(leo_name, controller)
                        else:
                            self._assign(leo_name, self._best_feasible_controller(leo_name, fov_candidates[leo_name]))

            # 局部修复：保证每个 LEO 都被分到一个控制器
            for leo_name in self.leoSattlitesPosition:
                if leo_name not in self.currentAssignment:
                    self._assign(leo_name, self._best_feasible_controller(leo_name, fov_candidates.get(leo_name, [])))

            # 边界微调 + 连通性修复
            self._movement_aware_fine_tune(leo_subgraph, fov_candidates)
            self._repair_disconnected_domains(leo_subgraph, fov_candidates)

            self._flush_assignment_to_runtime(graph)
            self.last_partition_summary = {
                'controllers': len(self.activeControllers),
                'leo_total': len(self.leoSattlitesPosition),
                'non_overlap': non_overlap_count,
                'overlap': len(overlap_nodes),
                'fallback': fallback_count,
                'regions': len(overlap_regions),
            }
            self._finalize_partition_result(graph=graph, leo_subgraph=leo_subgraph, time=time)
            
        return self.currentPartitionResult

    def _calc_rectangular_domain_partition(self, graph, time):
        """
        RectangularDivision：按 LEO 经纬度范围均匀划分为 m x n 个互斥矩形域。

        m 对应纬度维度，n 对应经度维度。
        """
        m = max(1, int(self.rectangular_m))
        n = max(1, int(self.rectangular_n))
        domain_count = m * n

        self.activeControllers = self._select_rectangular_controllers(domain_count)
        if len(self.activeControllers) <= 0:
            return None
        if len(self.activeControllers) < domain_count:
            raise ValueError(
                f"RectangularDivision needs at least {domain_count} MEO controllers, "
                f"but only {len(self.activeControllers)} are available."
            )

        for meo in self.activeControllers:
            self.boundary_satellites[meo] = []
            self.adjacency_table[meo] = {}
        self.currentPartitionResult = {controller: [] for controller in self.activeControllers}
        self.currentAssignment = {}

        leo_subgraph = graph.subgraph(self.leoSattlitesPosition.keys()).copy()
        self._annotate_graph_edge_delay(leo_subgraph)

        leo_lat_lon = {}
        for leo_name, pos in self.leoSattlitesPosition.items():
            lat, lon = central_angles_to_latlon(pos['latitude'], pos['longitude'])
            leo_lat_lon[leo_name] = (lat, lon)

        lat_values = [lat for lat, _ in leo_lat_lon.values()]
        lon_values = [lon for _, lon in leo_lat_lon.values()]
        lat_edges = np.linspace(min(lat_values), max(lat_values), m + 1)
        lon_edges = np.linspace(min(lon_values), max(lon_values), n + 1)

        for leo_name in sorted(self.leoSattlitesPosition, key=self._parse_satellite_indices):
            lat, lon = leo_lat_lon[leo_name]
            row = self._coordinate_bin_index(lat, lat_edges, m)
            col = self._coordinate_bin_index(lon, lon_edges, n)
            domain_idx = row * n + col
            controller = self.activeControllers[domain_idx]
            self._assign(leo_name, controller)

        self._validate_rectangular_partition_no_overlap()

        self._flush_assignment_to_runtime(graph)
        self.last_partition_summary = {
            'method': 'RectangularDivision',
            'rectangular_m': m,
            'rectangular_n': n,
            'controllers': len(self.activeControllers),
            'domains': domain_count,
            'leo_total': len(self.leoSattlitesPosition),
            'domain_sizes': {
                controller: len(nodes)
                for controller, nodes in self.currentPartitionResult.items()
            },
            'latitude_range': (float(lat_edges[0]), float(lat_edges[-1])),
            'longitude_range': (float(lon_edges[0]), float(lon_edges[-1])),
        }
        self._finalize_partition_result(graph=graph, leo_subgraph=leo_subgraph, time=time)
        # return self.currentPartitionResult

    def _coordinate_bin_index(self, value, edges, group_count):
        if group_count <= 1 or edges[0] == edges[-1]:
            return 0
        return min(group_count - 1, max(0, int(np.searchsorted(edges, value, side='right') - 1)))

    def _validate_rectangular_partition_no_overlap(self):
        all_assigned = list(itertools.chain.from_iterable(self.currentPartitionResult.values()))
        duplicates = [node for node, count in Counter(all_assigned).items() if count > 1]
        expected = set(self.leoSattlitesPosition)
        assigned = set(all_assigned)
        missing = sorted(expected - assigned, key=self._parse_satellite_indices)
        extra = sorted(assigned - expected, key=self._parse_satellite_indices)
        if duplicates or missing or extra:
            raise ValueError(
                "RectangularDivision must assign each LEO to exactly one domain. "
                f"duplicates={sorted(duplicates, key=self._parse_satellite_indices)}, "
                f"missing={missing}, extra={extra}"
            )

    def _finalize_partition_result(self, graph, leo_subgraph, time=None):
        for sat_name, info in self.satellites.items():
            if not getattr(info, 'isLeo', False):
                continue
            for nbr in info.neighbors:
                if nbr not in self.satellites or not getattr(self.satellites[nbr], 'isLeo', False):
                    continue
                sat_masterMeo = info.masterMeo
                nbr_masterMeo = self.satellites[nbr].masterMeo
                if sat_masterMeo is None or nbr_masterMeo is None or sat_masterMeo == nbr_masterMeo:
                    continue
                self.boundary_satellites.setdefault(sat_masterMeo, [])
                self.boundary_satellites.setdefault(nbr_masterMeo, [])
                self.adjacency_table.setdefault(sat_masterMeo, {})
                self.adjacency_table.setdefault(nbr_masterMeo, {})
                if sat_name not in self.boundary_satellites[sat_masterMeo]:
                    self.boundary_satellites[sat_masterMeo].append(sat_name)
                if nbr not in self.boundary_satellites[nbr_masterMeo]:
                    self.boundary_satellites[nbr_masterMeo].append(nbr)
                if (sat_masterMeo, nbr_masterMeo) not in self.neighbors_domain and (nbr_masterMeo, sat_masterMeo) not in self.neighbors_domain:
                    self.neighbors_domain.append((sat_masterMeo, nbr_masterMeo))
                    self.neighbors_domain.append((nbr_masterMeo, sat_masterMeo))
                    self.adjacency_table[sat_masterMeo][nbr_masterMeo] = self.currentPartitionResult.get(nbr_masterMeo, [])
                    self.adjacency_table[nbr_masterMeo][sat_masterMeo] = self.currentPartitionResult.get(sat_masterMeo, [])
        # self._plot_partition_result_2d(graph=graph, leo_subgraph=leo_subgraph, time=time)
        self._sync_initial_domain_leo_states_to_all_meos()
        self.previousAssignment = dict(self.currentAssignment)

    def _sync_initial_domain_leo_states_to_all_meos(self):
        if self._initial_leo_state_sync_done:
            return

        domain_leo_states = {}
        for controller, leo_names in self.currentPartitionResult.items():
            states = {}
            for leo_name in leo_names:
                leo_satellite = self.satellites.get(leo_name)
                if leo_satellite is None or not getattr(leo_satellite, 'isLeo', False):
                    continue
                if not getattr(leo_satellite, 'active', True):
                    continue
                if not hasattr(leo_satellite, 'build_leo_state_for_meo'):
                    continue
                states[leo_name] = leo_satellite.build_leo_state_for_meo()
            domain_leo_states[controller] = states

        for meo_name, meo_satellite in self.satellites.items():
            if getattr(meo_satellite, 'isLeo', True):
                continue
            if not getattr(meo_satellite, 'active', True):
                continue
            if getattr(meo_satellite, 'leoStates', None) is None:
                meo_satellite.leoStates = {}
            if not hasattr(meo_satellite, 'remote_domain_leo_states'):
                meo_satellite.remote_domain_leo_states = {}

            for controller, states in domain_leo_states.items():
                states_snapshot = copy.deepcopy(states)
                if meo_name == controller:
                    meo_satellite.leoStates.update(states_snapshot)
                else:
                    meo_satellite.remote_domain_leo_states[controller] = states_snapshot

        self._initial_leo_state_sync_done = True

    def updateSatllitesPosition(self, graph):
        self.leoSattlitesPosition.clear()
        self.meoSattlitesPosition.clear()
        for sat_name, attr in graph.nodes(data=True):
            if 'pos_0' not in attr:
                continue
            lat, lon, alt = attr['pos_0']
            height = int(sat_name.split('_')[1])
            pos = {'latitude': float(lat), 'longitude': float(lon), 'altitude': float(alt)}
            if height <= 2000:
                self.leoSattlitesPosition[sat_name] = pos
            else:
                self.meoSattlitesPosition[sat_name] = pos

    def generateControlMatrix(self, K):
        for i in range(K):
            self.totalMatrix[i] = generate_ControlDomain_Matrix(
                self.controlDomainNumber * self.orbitNumber,
                self.orbitNumber * self.leoSatNumberPerOrbit,
            )

    def calcCauchy(self):
        for matrix in self.totalMatrix.values():
            _ = np.sum(matrix, axis=1)

    # =========================
    # 控制器选择 / FOV 候选
    # =========================
    def _select_active_controllers(self):
        """
        当前框架里控制器来源只有 MEO。
        为兼容原项目 controlDomainNumber 参数，按“每个 MEO 轨道选多少颗控制器”来抽样；
        当该值为空/<=0/超过轨道内节点数时，直接使用该轨道全部 MEO。
        """
        meo_by_orbit = defaultdict(list)
        for name in self.meoSattlitesPosition:
            orbit = int(name.split('_')[2])
            sat_idx = int(name.split('_')[3])
            meo_by_orbit[orbit].append((sat_idx, name))

        active = []
        per_orbit_target = int(self.controlDomainNumber) if self.controlDomainNumber else 0
        for orbit in sorted(meo_by_orbit):
            orbit_nodes = sorted(meo_by_orbit[orbit])
            names = [name for _, name in orbit_nodes]
            if per_orbit_target <= 0 or per_orbit_target >= len(names):
                active.extend(names)
                continue
            # 等间隔选点，尽量覆盖整条轨道
            positions = np.linspace(0, len(names) - 1, per_orbit_target)
            chosen_indices = sorted({int(round(pos)) for pos in positions})
            while len(chosen_indices) < per_orbit_target:
                for idx in range(len(names)):
                    if idx not in chosen_indices:
                        chosen_indices.append(idx)
                    if len(chosen_indices) == per_orbit_target:
                        break
            active.extend(names[idx] for idx in sorted(chosen_indices[:per_orbit_target]))
        return active

    def _select_rectangular_controllers(self, domain_count):
        meo_nodes = sorted(self.meoSattlitesPosition, key=self._parse_satellite_indices)
        if domain_count <= 0 or not meo_nodes:
            return []
        if domain_count >= len(meo_nodes):
            return meo_nodes

        positions = np.linspace(0, len(meo_nodes) - 1, domain_count)
        chosen_indices = sorted({int(round(pos)) for pos in positions})
        while len(chosen_indices) < domain_count:
            for idx in range(len(meo_nodes)):
                if idx not in chosen_indices:
                    chosen_indices.append(idx)
                if len(chosen_indices) == domain_count:
                    break
        return [meo_nodes[idx] for idx in sorted(chosen_indices[:domain_count])]

    def _build_fov_candidates(self):
        fov_candidates = {}
        for leo_name, leo_pos in self.leoSattlitesPosition.items():
            leo_lat, leo_lon = central_angles_to_latlon(leo_pos['latitude'], leo_pos['longitude'])
            visible = []
            max_elevation = -1e9
            best_controller = None
            for controller in self.activeControllers:
                meo_pos = self.meoSattlitesPosition[controller]
                meo_lat, meo_lon = central_angles_to_latlon(meo_pos['latitude'], meo_pos['longitude'])
                elevation = satellite_elevation(
                    leo_lat, leo_lon, leo_pos['altitude'],
                    meo_lat, meo_lon, meo_pos['altitude'],
                )
                if elevation >= self.minimuElevationAngle:
                    visible.append(controller)
                if elevation > max_elevation:
                    max_elevation = elevation
                    best_controller = controller
            if not visible and best_controller is not None:
                visible = [best_controller]
            fov_candidates[leo_name] = visible
        return fov_candidates

    # =========================
    # 重叠区域构造
    # =========================
    def _build_overlap_regions(self, leo_subgraph, overlap_nodes, fov_candidates):
        if not overlap_nodes:
            return []
        overlap_subgraph = leo_subgraph.subgraph(overlap_nodes).copy()
        regions = []
        for component in nx.connected_components(overlap_subgraph):
            component = set(component)
            # 再按候选控制器集合细分，避免把完全无关的重叠簇揉在一起
            buckets = defaultdict(set)
            for node in component:
                buckets[frozenset(fov_candidates.get(node, []))].add(node)

            for key, nodes in buckets.items():
                controller_union = set(key)
                if not controller_union:
                    # 如果 buckets key 为空，尝试从邻域候选中兜底
                    for node in nodes:
                        controller_union.update(fov_candidates.get(node, []))
                if not controller_union:
                    controller_union = set(self.activeControllers)
                regions.append({
                    'nodes': sorted(nodes),
                    'controllers': sorted(controller_union),
                })
        return regions

    # =========================
    # CORG + 谱聚类
    # =========================
    def _spectral_cluster_overlap_region(self, leo_subgraph, region_nodes, region_controllers, fov_candidates):
        if len(region_nodes) <= 1:
            return [list(region_nodes)]

        node_count = len(region_nodes)
        cluster_count = max(1, min(len(region_controllers), node_count))
        if cluster_count == 1:
            return [list(region_nodes)]

        xi = self._build_corg_cost_matrix(leo_subgraph, region_nodes, fov_candidates) # NxN
        # Gaussian kernel similarity
        similarity = np.exp(-xi / max(2.0 * (self.gaussian_sigma ** 2), 1e-8))
        np.fill_diagonal(similarity, 1.0)

        degree = np.sum(similarity, axis=1)
        safe_degree = np.where(degree > 1e-12, degree, 1e-12)
        d_inv_sqrt = np.diag(1.0 / np.sqrt(safe_degree))
        laplacian = np.eye(node_count) - d_inv_sqrt @ similarity @ d_inv_sqrt

        try:
            eigvals, eigvecs = np.linalg.eigh(laplacian)
            order = np.argsort(eigvals)[:cluster_count]
            embedding = np.real(eigvecs[:, order])
        except np.linalg.LinAlgError:
            # 数值异常时退化为按经纬度做 kmeans
            embedding = np.array([
                [self.leoSattlitesPosition[node]['latitude'], self.leoSattlitesPosition[node]['longitude']]
                for node in region_nodes
            ], dtype=float)

        row_norm = np.linalg.norm(embedding, axis=1, keepdims=True)
        row_norm[row_norm == 0] = 1.0
        embedding = embedding / row_norm

        labels = self._kmeans(embedding, cluster_count)

        clusters = [[] for _ in range(cluster_count)]
        for idx, node in enumerate(region_nodes):
            clusters[int(labels[idx])].append(node)
        clusters = [cluster for cluster in clusters if cluster]

        # 若聚类后簇数量因空簇减少，再做一次简单补偿
        if len(clusters) < cluster_count:
            remaining = sorted(region_nodes, key=lambda n: len(fov_candidates.get(n, [])), reverse=True)
            while len(clusters) < cluster_count and remaining:
                clusters.append([remaining.pop(0)])
        return clusters

    def _build_corg_cost_matrix(self, leo_subgraph, region_nodes, fov_candidates):
        n = len(region_nodes)
        xi = np.zeros((n, n), dtype=float)

        # 先算全对最短传播时延（只在当前重叠区域诱导子图上）
        region_graph = leo_subgraph.subgraph(region_nodes).copy()
        self._annotate_graph_edge_delay(region_graph)
        shortest_delay = dict(nx.all_pairs_dijkstra_path_length(region_graph, weight='delay'))
        finite_delays = []
        for src in shortest_delay:
            finite_delays.extend(shortest_delay[src].values())
        max_delay = max(finite_delays) if finite_delays else 1.0

        # 方案 A：先使用 Step 1 已经固定下来的非重叠区 seed domains，
        # 再把当前待判断的重叠节点对临时挂到共同可见的控制器名下，
        # 用这个“临时完整域”统计 |E_d| 和 |D_k|，从而避免第一次分域时没有候选域的问题。
        for i, node_i in enumerate(region_nodes):
            for j in range(i + 1, n):
                node_j = region_nodes[j]
                flow_cost = self._flow_cost(node_i, node_j, shortest_delay, max_delay)
                sync_cost = self._sync_cost(
                    leo_subgraph=leo_subgraph,
                    region_nodes=region_nodes,
                    node_i=node_i,
                    node_j=node_j,
                    fov_candidates=fov_candidates,
                )
                mig_cost = self._migration_cost(node_i, node_j, fov_candidates)
                cost = self.alpha * flow_cost + self.beta * sync_cost + self.gamma * mig_cost
                xi[i, j] = cost
                xi[j, i] = cost
        return xi

    def _flow_cost(self, node_i, node_j, shortest_delay, max_delay):
        delay = shortest_delay.get(node_i, {}).get(node_j, max_delay)
        if max_delay <= 0:
            return 0.0
        return float(delay / max_delay)

    def _sync_cost(self, leo_subgraph, region_nodes, node_i, node_j, fov_candidates):
        """
        方案 A 的冷启动实现：
        - 以 Step 1 已固定的非重叠区 seed domain 为基础；
        - 对节点对 (node_i, node_j)，在它们共同可见的控制器中，
          枚举“临时完整域 = seed_domain(controller) + {node_i, node_j}”；
        - 在这个临时完整域上统计论文同步开销公式中的 |E_d| 与 |D_k|；
        - 取最小代价作为 pairwise sync cost。

        这样第一次分域时虽然还没有最终簇，但已经有了可计算的临时候选域。
        """
        common_controllers = sorted(
            set(fov_candidates.get(node_i, [])) & set(fov_candidates.get(node_j, []))
        )
        if not common_controllers:
            return 1.0

        region_node_set = set(region_nodes)
        candidate_costs = []
        for controller in common_controllers:
            seed_nodes = self._seed_domain_nodes(controller, exclude_nodes=region_node_set)
            temp_domain_nodes = set(seed_nodes)
            temp_domain_nodes.add(node_i)
            temp_domain_nodes.add(node_j)

            domain_edge_count, domain_size = self._temporary_domain_stats(
                leo_subgraph=leo_subgraph,
                domain_nodes=temp_domain_nodes,
            )
            intra_sync = self._temporary_intra_sync_cost(
                controller=controller,
                edge_count=domain_edge_count,
                nodes=(node_i, node_j),
            )
            inter_sync = self._temporary_inter_sync_cost(
                controller=controller,
                domain_size=domain_size,
            )

            # 轻量加入一点状态异质性，防止仅靠规模统计把资源状态完全抹平。
            # heterogeneity = self._resource_heterogeneity_cost(node_i, node_j)
            # candidate_costs.append(0.75 * (intra_sync + inter_sync) + 0.25 * heterogeneity)
            candidate_costs.append((intra_sync + inter_sync))

        if not candidate_costs:
            return 1.0
        return float(min(candidate_costs))
    
    def _computing_power_disparity(self, node_i, node_j,region_nodes,fov_candidates):
        common_controllers = sorted(
            set(fov_candidates.get(node_i, [])) & set(fov_candidates.get(node_j, []))
        )
        if not common_controllers:
            return 1.0
        region_node_set = set(region_nodes)
        for controller in common_controllers:
            seed_nodes = self._seed_domain_nodes(controller, exclude_nodes=region_node_set)
            temp_domain_nodes = set(seed_nodes)
            temp_domain_nodes.add(node_i)
            temp_domain_nodes.add(node_j)
            cpu_ability_list = []
            gpu_ability_list = []
            for sat in temp_domain_nodes:
                cpu_ability_list.append(self.satellites[sat].cpu_remain_ability)
                gpu_ability_list.append(self.satellites[sat].gpu_remain_ability)
            cpu_mean = statistics.mean(cpu_ability_list)
            cpu_stdev = statistics.stdev(cpu_ability_list)
            print("cpu_mean:{}".format(cpu_mean))
            print("cpu_stdev:{}".format(cpu_stdev))
            gpu_mean = statistics.mean(gpu_ability_list)
            gpu_stdev = statistics.stdev(gpu_ability_list)
            print("gpu_mean:{}".format(gpu_mean))
            print("gpu_stdev:{}".format(gpu_stdev))
            


    def _seed_domain_nodes(self, controller, exclude_nodes=None):
        exclude_nodes = exclude_nodes or set()
        return [
            node for node in self.currentPartitionResult.get(controller, [])
            if node not in exclude_nodes
        ]

    def _temporary_domain_stats(self, leo_subgraph, domain_nodes):
        domain_nodes = [node for node in domain_nodes if node in leo_subgraph]
        if not domain_nodes:
            return 0, 0
        domain_graph = leo_subgraph.subgraph(domain_nodes)
        edge_count = domain_graph.number_of_edges()
        domain_size = domain_graph.number_of_nodes()
        return int(edge_count), int(domain_size)

    def _temporary_intra_sync_cost(self, controller, edge_count, nodes):
        if not nodes:
            return 0.0
        sync_payload = edge_count * self.sync_msg_bytes
        #需建模可用带宽
        bandwidth_term = sync_payload / max(self.default_bandwidth_bytes_per_s, 1.0)
        delays = [self._controller_propagation_delay(node, controller) for node in nodes]
        intra = self.sync_frequency * (bandwidth_term + max(delays, default=0.0))
        return self._normalize_time_cost(intra)

    def _temporary_inter_sync_cost(self, controller, domain_size):
        if domain_size <= 0 or len(self.activeControllers) <= 1:
            return 0.0
        sync_payload = domain_size * self.sync_msg_bytes
        total = 0.0
        for other in self.activeControllers:
            if other == controller:
                continue
            bandwidth_term = sync_payload / max(self.default_bandwidth_bytes_per_s, 1.0)
            propagation = self._controller_propagation_delay(controller, other)
            total += bandwidth_term + propagation
        inter = self.sync_frequency * total / max(len(self.activeControllers) - 1, 1)
        return self._normalize_time_cost(inter)

    def _resource_heterogeneity_cost(self, node_i, node_j):
        sat_i = self.satellites.get(node_i)
        sat_j = self.satellites.get(node_j)
        mem_i = getattr(sat_i, 'current_memory_occupy', 0.0) / max(getattr(sat_i, 'memory', 1.0), 1.0)
        mem_j = getattr(sat_j, 'current_memory_occupy', 0.0) / max(getattr(sat_j, 'memory', 1.0), 1.0)
        comp_i = getattr(sat_i, 'computing_remain', 0.0) / max(getattr(sat_i, 'computing_ability', 1.0), 1.0)
        comp_j = getattr(sat_j, 'computing_remain', 0.0) / max(getattr(sat_j, 'computing_ability', 1.0), 1.0)
        buffer_term = min(abs(mem_i - mem_j), 1.0)
        compute_term = min(abs(comp_i - comp_j), 1.0)
        return 0.5 * buffer_term + 0.5 * compute_term

    def _controller_propagation_delay(self, src_name, dst_name):
        src = self._satellite_ecef(src_name)
        dst = self._satellite_ecef(dst_name)
        return float(np.linalg.norm(src - dst) / LIGHT_SPEED_KM_S)

    def _normalize_time_cost(self, value):
        # 10ms 量级附近映射到 1 左右，避免同步项把其它项数值淹没。
        return float(value / (value + 0.01)) if value > 0 else 0.0

    def _migration_cost(self, node_i, node_j, fov_candidates):
        dir_i = self._estimate_direction(node_i)
        dir_j = self._estimate_direction(node_j)
        direction_term = 0.0 if dir_i == dir_j else 1.0

        cand_i = set(fov_candidates.get(node_i, []))
        cand_j = set(fov_candidates.get(node_j, []))
        union = cand_i | cand_j
        if union:
            jaccard_term = 1.0 - len(cand_i & cand_j) / len(union)
        else:
            jaccard_term = 1.0

        prev_i = self.previousAssignment.get(node_i)
        prev_j = self.previousAssignment.get(node_j)
        continuity_term = 0.0 if (prev_i is not None and prev_i == prev_j) else 1.0
        if prev_i is None and prev_j is None:
            continuity_term = 0.5

        return 0.5 * direction_term + 0.3 * jaccard_term + 0.2 * continuity_term

    def _kmeans(self, data, k, max_iter=50):
        n_samples = len(data)
        if n_samples == 0:
            return np.array([], dtype=int)
        if k >= n_samples:
            return np.arange(n_samples, dtype=int)

        # 稳定初始化：按范数排序后均匀取样
        norms = np.linalg.norm(data, axis=1)
        order = np.argsort(norms)
        init_pos = np.linspace(0, n_samples - 1, k).astype(int)
        centroids = data[order[init_pos]].copy()

        labels = np.zeros(n_samples, dtype=int)
        for _ in range(max_iter):
            distances = np.linalg.norm(data[:, None, :] - centroids[None, :, :], axis=2)
            new_labels = np.argmin(distances, axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for idx in range(k):
                members = data[labels == idx]
                if len(members) == 0:
                    farthest = int(np.argmax(np.min(distances, axis=1)))
                    centroids[idx] = data[farthest]
                else:
                    centroids[idx] = np.mean(members, axis=0)
        return labels

    # =========================
    # 聚类-控制器匹配
    # =========================
    def _match_clusters_to_controllers(self, clusters, controllers, fov_candidates):
        if not clusters or not controllers:
            return {}

        cluster_count = len(clusters)
        controller_count = len(controllers)
        cost = np.full((cluster_count, controller_count), np.inf, dtype=float)
        # for cluster in clusters:
        #     print(cluster)
        #     for sat in cluster:
        #         print(sat + " cpu:{}".format(self.satellites[sat].cpu_remain_ability) + " gpu:{}".format(self.satellites[sat].gpu_remain_ability))
        for cluster_idx, cluster_nodes in enumerate(clusters):
            cluster_centroid = self._cluster_centroid(cluster_nodes)
            majority_prev = self._majority_previous_controller(cluster_nodes)
            for controller_idx, controller in enumerate(controllers):
                invisible = sum(1 for node in cluster_nodes if controller not in fov_candidates.get(node, []))
                distance_cost = self._ecef_distance(cluster_centroid, self._controller_ecef(controller))
                continuity_discount = 0.9 if majority_prev == controller else 1.0
                cost[cluster_idx, controller_idx] = distance_cost * continuity_discount + invisible * 1e6

        if cluster_count <= controller_count:
            best = self._solve_assignment(cost)
            return {cluster_idx: controllers[ctrl_idx] for cluster_idx, ctrl_idx in best.items()}

        # 簇比控制器还多：先让每个簇贪心找到最便宜控制器
        mapping = {}
        for cluster_idx in range(cluster_count):
            ctrl_idx = int(np.argmin(cost[cluster_idx]))
            mapping[cluster_idx] = controllers[ctrl_idx]
        return mapping

    def _solve_assignment(self, cost_matrix):
        row_count, col_count = cost_matrix.shape
        best_perm = None
        best_cost = np.inf
        for cols in itertools.permutations(range(col_count), row_count):
            total = 0.0
            feasible = True
            for row_idx, col_idx in enumerate(cols):
                value = cost_matrix[row_idx, col_idx]
                if not np.isfinite(value):
                    feasible = False
                    break
                total += value
            if feasible and total < best_cost:
                best_cost = total
                best_perm = cols
        if best_perm is None:
            # 极端情况下退化为逐行贪心
            mapping = {}
            used = set()
            for row_idx in range(row_count):
                sorted_cols = np.argsort(cost_matrix[row_idx])
                chosen = None
                for col_idx in sorted_cols:
                    if col_idx not in used and np.isfinite(cost_matrix[row_idx, col_idx]):
                        chosen = int(col_idx)
                        break
                if chosen is None:
                    chosen = int(sorted_cols[0])
                used.add(chosen)
                mapping[row_idx] = chosen
            return mapping
        return {row_idx: col_idx for row_idx, col_idx in enumerate(best_perm)}

    # =========================
    # 移动感知微调 + 连通性修复
    # =========================
    def _movement_aware_fine_tune(self, leo_subgraph, fov_candidates, passes=2):
        for _ in range(passes):
            changed = False
            for leo_name in sorted(self.leoSattlitesPosition):
                candidates = fov_candidates.get(leo_name, [])
                if len(candidates) <= 1:
                    continue
                current_controller = self.currentAssignment.get(leo_name)
                best_controller = current_controller
                best_score = self._boundary_score(leo_subgraph, leo_name, current_controller)
                for controller in candidates:
                    score = self._boundary_score(leo_subgraph, leo_name, controller)
                    if score + 1e-9 < best_score:
                        best_score = score
                        best_controller = controller
                if best_controller != current_controller and best_controller is not None:
                    self._reassign(leo_name, best_controller)
                    changed = True
            if not changed:
                break

    def _boundary_score(self, leo_subgraph, leo_name, controller):
        if controller is None:
            return np.inf
        neighbors = [nbr for nbr in leo_subgraph.neighbors(leo_name)] if leo_name in leo_subgraph else []
        assigned_neighbors = [self.currentAssignment.get(nbr) for nbr in neighbors if nbr in self.currentAssignment]
        if assigned_neighbors:
            majority_controller, majority_count = Counter(assigned_neighbors).most_common(1)[0]
            cohesion_term = 1.0 - majority_count / max(len(assigned_neighbors), 1)
            controller_bonus = 0.0 if majority_controller == controller else 0.4
        else:
            cohesion_term = 0.5
            controller_bonus = 0.0

        prev_controller = self.previousAssignment.get(leo_name)
        continuity_term = 0.0 if prev_controller in (None, controller) else 0.5

        direction_term = 0.0
        my_direction = self._estimate_direction(leo_name)
        same_ctrl_neighbors = [nbr for nbr in neighbors if self.currentAssignment.get(nbr) == controller]
        if same_ctrl_neighbors:
            dir_match = sum(1 for nbr in same_ctrl_neighbors if self._estimate_direction(nbr) == my_direction)
            direction_term = 1.0 - dir_match / len(same_ctrl_neighbors)

        dist_term = self._ecef_distance(self._satellite_ecef(leo_name), self._controller_ecef(controller))
        return 0.35 * cohesion_term + 0.15 * controller_bonus + 0.2 * continuity_term + 0.15 * direction_term + 0.15 * dist_term

    def _repair_disconnected_domains(self, leo_subgraph, fov_candidates):
        for controller in list(self.currentPartitionResult.keys()):
            domain_nodes = [node for node in self.currentPartitionResult[controller] if node in leo_subgraph]
            if len(domain_nodes) <= 1:
                continue
            domain_graph = leo_subgraph.subgraph(domain_nodes)
            components = list(nx.connected_components(domain_graph))
            if len(components) <= 1:
                continue

            controller_pos = self._controller_ecef(controller)
            keep_component = max(
                components,
                key=lambda comp: -np.mean([self._ecef_distance(self._satellite_ecef(node), controller_pos) for node in comp]),
            )
            keep_component = set(keep_component)

            for component in components:
                component = set(component)
                if component == keep_component:
                    continue
                for node in component:
                    best = self._best_feasible_controller(node, fov_candidates.get(node, []), exclude={controller})
                    if best is None:
                        continue
                    self._reassign(node, best)

    # =========================
    # 运行时写回
    # =========================
    def _flush_assignment_to_runtime(self, graph):
        for controller in list(self.currentPartitionResult.keys()):
            self.currentPartitionResult[controller] = sorted(set(self.currentPartitionResult[controller]))

        # 统一写回 masterMeo / graph 属性
        for sat_name, sat in self.satellites.items():
            if getattr(sat, 'isLeo', False):
                # sat.masterMeo = self.currentAssignment.get(sat_name)
                meoName = self.currentAssignment.get(sat_name)
                sat.configMasterMeo(meoName, self.currentPartitionResult.get(meoName, []))
        for node in graph.nodes:
            if node in self.currentAssignment:
                graph.nodes[node]['domain_controller'] = self.currentAssignment[node]
            elif node in self.activeControllers:
                graph.nodes[node]['domain_controller'] = node
            else:
                graph.nodes[node]['domain_controller'] = None

        # 给控制器反向挂管理列表，并清理不再归属该 MEO 的旧 LEO 状态缓存。
        for sat_name, sat in self.satellites.items():
            if getattr(sat, 'isLeo', False):
                continue
            managed_leos = list(self.currentPartitionResult.get(sat_name, []))
            sat.my_leos = managed_leos
            if getattr(sat, 'leoStates', None) is not None:
                managed_set = set(managed_leos)
                stale_leos = [leo for leo in sat.leoStates if leo not in managed_set]
                for leo in stale_leos:
                    del sat.leoStates[leo]
                if stale_leos and hasattr(sat, 'domain_aggregate'):
                    sat.domain_aggregate = None

        # 写回 graph 属性，便于调试和后续扩展。
        for controller in self.activeControllers:
            managed_leos = list(self.currentPartitionResult.get(controller, []))
            if controller in graph.nodes:
                graph.nodes[controller]['managed_leo'] = managed_leos

    def _assign(self, leo_name, controller):
        if controller is None:
            return
        old = self.currentAssignment.get(leo_name)
        if old == controller:
            return
        if old is not None and old in self.currentPartitionResult:
            if leo_name in self.currentPartitionResult[old]:
                self.currentPartitionResult[old].remove(leo_name)
        self.currentAssignment[leo_name] = controller
        self.currentPartitionResult.setdefault(controller, []).append(leo_name)

    def _reassign(self, leo_name, controller):
        self._assign(leo_name, controller)

    # =========================
    # 基础计算工具
    # =========================
    def _annotate_graph_edge_delay(self, graph):
        for u, v in graph.edges():
            delay = self.delayMap.get((u, v), self.delayMap.get((v, u)))
            if delay is None:
                ecef_u = self._satellite_ecef(u)
                ecef_v = self._satellite_ecef(v)
                delay = np.linalg.norm(ecef_u - ecef_v) / LIGHT_SPEED_KM_S
            graph[u][v]['delay'] = float(delay)

    def _satellite_ecef(self, sat_name):
        pos = self.leoSattlitesPosition.get(sat_name) or self.meoSattlitesPosition.get(sat_name)
        return geodetic_to_ecef(pos['latitude'], pos['longitude'], pos['altitude'])

    def _controller_ecef(self, controller):
        return self._satellite_ecef(controller)

    def _cluster_centroid(self, cluster_nodes):
        points = [self._satellite_ecef(node) for node in cluster_nodes]
        return np.mean(points, axis=0)

    def _ecef_distance(self, a, b):
        dist = float(np.linalg.norm(np.asarray(a) - np.asarray(b)))
        # 归一化到 0~1 的稳定量级
        return dist / 20000.0

    def _best_feasible_controller(self, leo_name, candidates, exclude=None):
        exclude = exclude or set()
        candidates = [c for c in candidates if c not in exclude]
        if not candidates:
            candidates = [c for c in self.activeControllers if c not in exclude]
        if not candidates:
            return None
        prev = self.previousAssignment.get(leo_name)
        best = None
        best_score = np.inf
        sat_ecef = self._satellite_ecef(leo_name)
        for controller in candidates:
            score = self._ecef_distance(sat_ecef, self._controller_ecef(controller))
            if prev == controller:
                score *= 0.85
            if score < best_score:
                best_score = score
                best = controller
        return best

    def _fallback_controller_for_leo(self, leo_name):
        return self._best_feasible_controller(leo_name, [])

    def _parse_satellite_indices(self, sat_name):
        try:
            altitude, orbit, sat_idx = map(int, sat_name.split('_')[1:])
            return altitude, orbit, sat_idx
        except (ValueError, IndexError):
            return 0, 0, 0

    def _majority_previous_controller(self, cluster_nodes):
        prev = [self.previousAssignment.get(node) for node in cluster_nodes if self.previousAssignment.get(node) is not None]
        if not prev:
            return None
        return Counter(prev).most_common(1)[0][0]

    def _estimate_direction(self, leo_name):
        """
        用同轨相邻卫星的纬度差近似当前飞行方向：
        next_sat 纬度更高 -> northbound，否则 southbound。
        """
        try:
            altitude, orbit, sat_idx = map(int, leo_name.split('_')[1:])
        except ValueError:
            return 'unknown'

        same_orbit = []
        for name in self.leoSattlitesPosition:
            alt2, orbit2, sat2 = map(int, name.split('_')[1:])
            if alt2 == altitude and orbit2 == orbit:
                same_orbit.append((sat2, name))
        if len(same_orbit) <= 1:
            return 'unknown'
        same_orbit.sort()
        sat_indices = [idx for idx, _ in same_orbit]
        names = [name for _, name in same_orbit]
        pos = sat_indices.index(sat_idx)
        next_name = names[(pos + 1) % len(names)]
        lat = self.leoSattlitesPosition[leo_name]['latitude']
        next_lat = self.leoSattlitesPosition[next_name]['latitude']
        return 'northbound' if next_lat >= lat else 'southbound'

    # =========================
    # 分域结果可视化（2D 平面）
    # =========================
    def _plot_partition_result_2d(self, graph=None, leo_subgraph=None, time = None):
        """
        每次分域完成后绘制 2D 平面图：
        - 仅显示 LEO；
        - 同一控制域使用同一颜色；
        - MEO 不绘制。
        """
        if not self.partition_plot_enabled or plt is None:
            return

        os.makedirs(self.partition_plot_dir, exist_ok=True)
        self.partition_plot_index += 1

        controllers = sorted(self.currentPartitionResult.keys())
        if controllers:
            cmap = plt.get_cmap('tab20', max(len(controllers), 1))
            for idx, controller in enumerate(controllers):
                if controller not in self._controller_color_map:
                    self._controller_color_map[controller] = cmap(idx)

        fig, ax = plt.subplots(figsize=(12, 6), dpi=130)
        self._draw_earth_surface_background(ax)
        if self.showLink:
            self._draw_leo_links_with_weights(ax, graph=graph, leo_subgraph=leo_subgraph)

        for controller in controllers:
            nodes = [n for n in self.currentPartitionResult.get(controller, []) if n in self.leoSattlitesPosition]
            if not nodes:
                continue
            x = [((self.leoSattlitesPosition[n]['longitude'] + 180.0) % 360.0) - 180.0 for n in nodes]
            y = [self.leoSattlitesPosition[n]['latitude'] for n in nodes]
            ax.scatter(
                x,
                y,
                s=16,
                color=self._controller_color_map.get(controller),
                alpha=0.9,
                edgecolors='none',
                label=f"{controller} ({len(nodes)})",
            )

        unassigned = [n for n in self.leoSattlitesPosition if n not in self.currentAssignment]
        if unassigned:
            x = [((self.leoSattlitesPosition[n]['longitude'] + 180.0) % 360.0) - 180.0 for n in unassigned]
            y = [self.leoSattlitesPosition[n]['latitude'] for n in unassigned]
            ax.scatter(x, y, s=16, color='lightgray', alpha=0.8, edgecolors='none', label=f"unassigned ({len(unassigned)})")

        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_xlabel("Longitude (deg)")
        ax.set_ylabel("Latitude (deg)")
        ax.set_title(f"LEO Domain Partition #{self.partition_plot_index}")
        ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.5)
        if controllers or unassigned:
            ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)
        fig.tight_layout()

        out_path = os.path.join(self.partition_plot_dir, f"partition_{self.partition_plot_index:04d}.svg")
        fig.savefig(out_path, format='svg', bbox_inches='tight')
        plt.close(fig)

    def _draw_leo_links_with_weights(self, ax, graph=None, leo_subgraph=None):
        if leo_subgraph is None:
            if graph is None:
                return
            leo_subgraph = graph.subgraph(self.leoSattlitesPosition.keys()).copy()
            self._annotate_graph_edge_delay(leo_subgraph)

        for u, v, data in leo_subgraph.edges(data=True):
            if u not in self.leoSattlitesPosition or v not in self.leoSattlitesPosition:
                continue
            lon_u = ((self.leoSattlitesPosition[u]['longitude'] + 180.0) % 360.0) - 180.0
            lat_u = self.leoSattlitesPosition[u]['latitude']
            lon_v = ((self.leoSattlitesPosition[v]['longitude'] + 180.0) % 360.0) - 180.0
            lat_v = self.leoSattlitesPosition[v]['latitude']

            # 穿越日期变更线的链路不直连，避免出现横跨整张图的长线。
            if abs(lon_u - lon_v) > 180.0:
                continue

            ax.plot([lon_u, lon_v], [lat_u, lat_v], color='#3a3a3a', linewidth=0.45, alpha=0.55, zorder=1)

            delay = data.get('delay')
            if delay is None:
                delay = self.delayMap.get((u, v), self.delayMap.get((v, u)))
                if delay is None:
                    delay = np.linalg.norm(self._satellite_ecef(u) - self._satellite_ecef(v)) / LIGHT_SPEED_KM_S
            delay_ms = float(delay) * 1000.0

            mid_x = 0.5 * (lon_u + lon_v)
            mid_y = 0.5 * (lat_u + lat_v)
            ax.text(
                mid_x,
                mid_y,
                f"{delay_ms:.2f}ms",
                fontsize=5.5,
                color='#2b2b2b',
                ha='center',
                va='center',
                zorder=2,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.55, boxstyle='round,pad=0.12'),
            )

    def _draw_earth_surface_background(self, ax):
        # 海洋底色
        ax.set_facecolor('#cfe8ff')

        polygons = self._load_world_land_polygons()
        if not polygons:
            return

        # 陆地填充 + 海岸线
        for part in polygons:
            if len(part) < 3:
                continue
            lon = part[:, 0]
            lat = part[:, 1]
            ax.fill(lon, lat, facecolor='#e9ddbf', edgecolor='#8f8263', linewidth=0.25, alpha=0.9, zorder=0)

    def _load_world_land_polygons(self):
        """
        读取 Natural Earth shapefile（Polygon），返回若干二维点序列：
        [np.array([[lon, lat], ...]), ...]
        """
        if self._world_land_polygons is not None:
            return self._world_land_polygons

        shp_path = os.path.join(os.path.dirname(__file__), "ne_data", "ne_50m_admin_0_countries_lakes.shp")
        if not os.path.exists(shp_path):
            self._world_land_polygons = []
            return self._world_land_polygons

        polygons = []
        try:
            with open(shp_path, "rb") as f:
                _ = f.read(100)  # file header
                while True:
                    rec_header = f.read(8)
                    if len(rec_header) < 8:
                        break
                    _, content_len_words = struct.unpack(">2i", rec_header)
                    content_bytes = content_len_words * 2
                    rec_content = f.read(content_bytes)
                    if len(rec_content) < 44:
                        continue

                    shape_type = struct.unpack("<i", rec_content[0:4])[0]
                    if shape_type not in (5, 15, 25):  # Polygon / PolygonZ / PolygonM
                        continue

                    num_parts = struct.unpack("<i", rec_content[36:40])[0]
                    num_points = struct.unpack("<i", rec_content[40:44])[0]
                    if num_parts <= 0 or num_points <= 0:
                        continue

                    part_idx_start = 44
                    part_idx_end = part_idx_start + 4 * num_parts
                    points_start = part_idx_end
                    points_end = points_start + 16 * num_points
                    if points_end > len(rec_content):
                        continue

                    parts = struct.unpack("<" + "i" * num_parts, rec_content[part_idx_start:part_idx_end])
                    points_raw = rec_content[points_start:points_end]

                    all_points = np.frombuffer(points_raw, dtype="<f8").reshape(-1, 2)
                    for i, start in enumerate(parts):
                        end = parts[i + 1] if i + 1 < len(parts) else num_points
                        part = all_points[start:end]
                        if len(part) < 3:
                            continue
                        polygons.append(part.copy())
        except Exception:
            polygons = []

        self._world_land_polygons = polygons
        return self._world_land_polygons
    
    #计算域内和域间算力差异度（采用标准差/均值 --> 变异系数）
    def calcComputilityStdev(self, clusters):
        cpuCoefficientOfVariation_intra = []
        gpuCoefficientOfVariation_intra = []
        cpu_sum = []
        gpu_sum = []
        for cluster in clusters:
            cpu_reamin = []
            gpu_reamin = []
            for satName in cluster:
                cpu_reamin.append(self.satellites[satName].cpu_remain_ability)
                gpu_reamin.append(self.satellites[satName].gpu_remain_ability)
            if len(cpu_reamin) > 1:
                cpuCoefficientOfVariation_intra.append(statistics.stdev(cpu_reamin) / statistics.mean(cpu_reamin))
            else:
                cpuCoefficientOfVariation_intra.append(0)
            cpu_sum.append(sum(cpu_reamin))
            if len(gpu_reamin) > 1:
                gpuCoefficientOfVariation_intra.append(statistics.stdev(gpu_reamin) / statistics.mean(gpu_reamin))
            else:
                gpuCoefficientOfVariation_intra.append(0)
            gpu_sum.append(sum(gpu_reamin))
            
        if len(cpu_sum) > 1:
            cpuCoefficientOfVariation_inter = statistics.stdev(cpu_sum) / statistics.mean(cpu_sum)
        else:
            cpuCoefficientOfVariation_inter = 0
        if len(gpu_sum) > 1:
            gpuCoefficientOfVariation_inter = statistics.stdev(gpu_sum) / statistics.mean(gpu_sum)
        else:
            gpuCoefficientOfVariation_inter = 0
        return cpuCoefficientOfVariation_intra, gpuCoefficientOfVariation_intra,cpuCoefficientOfVariation_inter,gpuCoefficientOfVariation_inter
