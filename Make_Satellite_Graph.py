from skyfield.api import EarthSatellite
from common import eci_propagation_delay
"""
处理卫星轨道数据(TLE)并构建卫星网络的拓扑图，核心功能是将卫星的轨道信息转换为可用于仿真的网络结构（节点为卫星，边为卫星间的连接关系）
Skyfield 的核心功能之一是​​解析 TLE 数据并利用 SGP4 模型进行卫星位置计算​​
"""
class SatelliteTracker:
    def __init__(self, tle_filepath):
        # for tle_file in tle_filepath:
        #     with open(tle_file) as f:
        #         tle_data = f.read()

        #     tle_lines = tle_data.splitlines()
        #     self.satellites = [EarthSatellite(tle_lines[i + 1], tle_lines[i + 2], tle_lines[i]) for i in
        #                     range(0, len(tle_lines), 3)]
        # filepaths = self._normalize_tle_filepaths(tle_filepath)
        tle_lines = []
        for filepath in tle_filepath:
            with open(filepath) as f:
                current_lines = [line.strip() for line in f.read().splitlines() if line.strip()]
            tle_lines.extend(current_lines)

        if len(tle_lines) % 3 != 0:
            raise ValueError(
                f"Invalid TLE content: expected groups of 3 lines (name + line1 + line2), "
                f"got {len(tle_lines)} non-empty lines from {tle_filepath}"
            )

        satellite_names = tle_lines[0::3]
        name_counts = {}
        for name in satellite_names:
            name_counts[name] = name_counts.get(name, 0) + 1
        duplicate_names = sorted([name for name, count in name_counts.items() if count > 1])
        if duplicate_names:
            raise ValueError(
                f"Duplicate satellite names found across TLE files: {duplicate_names[:5]}"
                + (" ..." if len(duplicate_names) > 5 else "")
            )

        self.satellites = [
            EarthSatellite(tle_lines[i + 1], tle_lines[i + 2], tle_lines[i])
            for i in range(0, len(tle_lines), 3)
        ]

    def generate_satellite_dict(self, time):#计算指定时间 time 所有卫星的位置和属性，返回一个包含卫星详细信息的字典
        sat_dict = {}
        for sat in self.satellites:
            geocentric = sat.at(time)
            eci_position = geocentric.position.km
            subpoint = geocentric.subpoint()
            lat = subpoint.latitude.degrees
            lon = subpoint.longitude.degrees
            alt = subpoint.elevation.km
            sat_name = sat.name
            orbit_altitude, orbit_number, sat_number = [int(s) for s in sat_name.split('_')[1:]]
            sat_dict[sat_name] = [eci_position,orbit_altitude , orbit_number, sat_number,lat,lon,alt, geocentric.velocity.km_per_s]## 存储卫星信息：ECI位置、轨道高度、轨道号、卫星号、经纬度、高度、速度
        return sat_dict        

    def get_max_orbit_number(self):#统计所有卫星中最大的轨道编号（用于确定轨道总数）
        max_orbit_number = 0
        for sat in self.satellites:
            sat_name = sat.name
            orbit_number = int(sat_name.split('_')[2])
            max_orbit_number = max(max_orbit_number, orbit_number)
        return max_orbit_number

    def get_max_satellite_number(self):#统计所有卫星中最大的卫星编号（用于确定每个轨道的卫星总数）。
        max_satellite_number = 0
        for sat in self.satellites:
            sat_name = sat.name
            satellite_number = int(sat_name.split('_')[3])
            max_satellite_number = max(max_satellite_number, satellite_number)
        return max_satellite_number

    def generate_satellite_LLA_dict(self, time):#生成仅包含卫星经纬度（Latitude）、经度（Longitude）、高度（Altitude）的字典，简化位置信息存储。
        sat_LLA_dict = {}
        for sat in self.satellites:
            geocentric = sat.at(time)
            subpoint = geocentric.subpoint()
            lat = subpoint.latitude.degrees
            lon = subpoint.longitude.degrees
            alt = subpoint.elevation.km

            sat_LLA_dict[sat.name]={"latitude": lat, "longitude": lon, "altitude": alt}
        return sat_LLA_dict

import networkx as nx

class SatelliteGraph:
    def __init__(self):
        pass

    def _distance(self, pos1, pos2):#单下划线前缀表示这是一个"受保护"（protected）的方法
        return sum((a - b) ** 2 for a, b in zip(pos1, pos2)) ** 0.5

    def build_graph_with_fixed_edges(self, satellite_tracker, time, pole=False):
        # 1. 获取卫星数据并初始化图
        satellite_dict = satellite_tracker.generate_satellite_dict(time)
        graph = nx.Graph()
        graph.add_nodes_from(satellite_dict.keys())
        # 2. 设置节点属性（位置、编号等）
        for sat_name, position in satellite_dict.items():
            graph.nodes[sat_name]['pos'] = position[0]# # ECI坐标
            graph.nodes[sat_name]['sequence_num'] = position[1:4]# 轨道高度、轨道号、卫星号
            graph.nodes[sat_name]['pos_0'] = position[4:7]# 经纬度、高度
            graph.nodes[sat_name]['velocity'] = position[7:]#卫星运动速度
        # 3. 按轨道高度分别统计最大轨道号和每条轨道的最大卫星号。
        #    LEO/MEO 混合网络中，不同高度层的星座规模可能不同，不能使用全局最大值。
        max_orbit_by_altitude = {}
        max_satellite_by_altitude_orbit = {}
        for sat_data in satellite_dict.values():
            orbit_altitude, orbit_number, sat_number = sat_data[1:4]
            max_orbit_by_altitude[orbit_altitude] = max(
                max_orbit_by_altitude.get(orbit_altitude, 0),
                orbit_number
            )
            altitude_orbit = (orbit_altitude, orbit_number)
            max_satellite_by_altitude_orbit[altitude_orbit] = max(
                max_satellite_by_altitude_orbit.get(altitude_orbit, 0),
                sat_number
            )

        def add_satellite_edge(source, target):
            if target not in satellite_dict:
                return
            graph.add_edge(
                source,
                target,
                pos_a=graph.nodes[source]['pos'],
                pos_b=graph.nodes[target]['pos'],
                delay=eci_propagation_delay(graph.nodes[source]['pos'], graph.nodes[target]['pos'])
            )

        # 4. 添加同轨道邻居（同一轨道内相邻的卫星）
        for sat_name, sat_data in satellite_dict.items():
            orbit_altitude, orbit_number, sat_number = sat_data[1:4]
            max_satellite_number = max_satellite_by_altitude_orbit[(orbit_altitude, orbit_number)]
            same_orbit_neighbors = [f"Satellite_{orbit_altitude}_{orbit_number}_{sat_number - 1 if sat_number != 1 else max_satellite_number}",#生成当前卫星的 “前一个邻居” 名称
                                    f"Satellite_{orbit_altitude}_{orbit_number}_{(sat_number % max_satellite_number) + 1}"]#生成当前卫星的 “后一个邻居” 名称
            #sat_data[3] 卫星在轨道中的位置编号
            for neighbor in same_orbit_neighbors:
                add_satellite_edge(sat_name, neighbor)

        # 5. 添加相邻轨道邻居（不同轨道间的卫星连接）
        for sat_name, sat_data in satellite_dict.items():
            orbit_altitude, orbit_number, sat_number = sat_data[1:4]
            max_orbit_number = max_orbit_by_altitude[orbit_altitude]
            if pole:
                next_orbit_number = orbit_number + 1 # 极地轨道：轨道号直接+1
            else:
                next_orbit_number = (orbit_number % max_orbit_number) + 1 # 非极地：轨道号循环

            next_orbit_max_satellite = max_satellite_by_altitude_orbit.get((orbit_altitude, next_orbit_number))
            if not next_orbit_max_satellite:
                continue

            next_satellite_number = sat_number
            if next_orbit_number % 2 != 0:
                next_satellite_number = sat_number - 1 if sat_number != 1 else next_orbit_max_satellite

            next_orbit_satellite = f"Satellite_{orbit_altitude}_{next_orbit_number}_{next_satellite_number}"
            if not pole:
                # 相邻轨道的卫星编号规则：偶数轨道同编号，奇数轨道前一个编号（循环）
                add_satellite_edge(sat_name, next_orbit_satellite)
            # 极地轨道：仅当卫星不在极地附近（z坐标绝对值<6000）时添加边
            elif next_orbit_number<= max_orbit_number:
                if next_orbit_satellite in satellite_dict and abs(graph.nodes[sat_name]['pos'][2])<6000 and abs(graph.nodes[next_orbit_satellite]['pos'][2])<6000:
                    add_satellite_edge(sat_name, next_orbit_satellite)

        return graph

"""
为仿真提供基础拓扑：构建的卫星网络拓扑是后续卫星通信仿真（如数据传输、路由选择、拥塞控制等）的基础，定义了卫星之间的可达性。
动态适配时间变化:通过SatelliteTracker获取指定时间的卫星位置,确保拓扑结构随卫星运动动态更新(如在SatelliteSimulation的run方法中,每个时间步都会重新调用build_graph_with_fixed_edges更新拓扑)。
简化复杂连接逻辑：通过固定规则（同轨道循环、相邻轨道对应编号）定义连接，避免了复杂的动态邻居计算，适合仿真场景下的高效拓扑生成。


"""
