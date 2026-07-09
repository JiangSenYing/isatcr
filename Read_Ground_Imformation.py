import math
import h3


def extract_landmarks(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    landmarks = {}#字典，key=地标名称，value=经纬度+高度
    i = 0
    j = 0

    lines_list = []
    for line in lines[2:]:
        lines_list.append(line.strip())
        if ":" in line:
            break

    landmark_name = ','.join(lines_list)
    landmark_names = [name.strip() for name in landmark_name.split(':')[0].split(",")]
    while i < len(lines):
        line = lines[i]
        if "---------    ---------    --------" in line:
            i += 1
            coords_line = lines[i]
            coords = coords_line.split()
            if len(coords) == 3:
                latitude, longitude, altitude = map(float, coords)
                landmarks[landmark_names[j]] = {"latitude": latitude, "longitude": longitude, "altitude": altitude}
                j += 1
        i += 1
    return landmarks


def to_cartesian(lat, lon, alt=0):# # 将经纬度坐标转换为笛卡尔坐标（x, y, z）
    R = 6371
    lat, lon = math.radians(lat), math.radians(lon)
    x = (R + alt) * math.cos(lat) * math.cos(lon)
    y = (R + alt) * math.cos(lat) * math.sin(lon)
    z = (R + alt) * math.sin(lat)
    return x, y, z


def max_distance(elevation_angle, orbital_height):# 计算地面站在给定仰角下能看到卫星的最大距离
    elevation_angle = math.radians(elevation_angle)
    earth_radius = 6371
    distance = math.sqrt(
        orbital_height ** 2 + 2 * orbital_height * earth_radius + (math.cos(elevation_angle) * earth_radius) ** 2
        ) - math.cos(elevation_angle) * earth_radius
    return distance


def to_h3_index(lat, lon, resolution=0):
    # return h3.geo_to_h3(lat, lon, resolution)
    return h3.geo_to_h3(lat, lon, resolution)
"""
# 将经纬度转换为H3空间索引
将卫星和地面站的位置映射到 H3 网格后，可通过网格索引快速查找附近的卫星，减少全局距离计算的开销。

利用 h3 库将地理坐标（纬度、经度）转换为 H3 索引。
H3 是一种六边形网格空间索引系统，可将地球表面划分为不同精度的六边形网格，便于高效查询空间邻近的物体。
"""


def get_h3_neighbors(h3_index):# 获取H3索引的相邻网格（1阶邻居)调用 h3.k_ring(h3_index, 1)，k=1 表示只获取直接相邻的网格。
    return h3.k_ring(h3_index, 1)
"""
    用 H3 网格把卫星空间位置离散化，避免全量搜索。

    对每个用户，查找邻居格子里的卫星。

    通过几何距离判断是否可见（是否满足仰角阈值）。

    输出 {用户: [可连接卫星]}。

"""

def get_connections_h3(ground_users, satellites, elevation_angle):
    """
    ground_users:地面用户(或地面站)的字典集合,键为地面用户名称,值为包含其地理位置的字典(latitude 纬度、longitude 经度、altitude 高度）。
    satellites:卫星的字典集合,键为卫星名称，值为包含其地理位置的字典(latitude 纬度、longitude 经度、altitude 高度）。
    elevation_angle:仰角阈值（单位：度），低于该角度的卫星视为不可见（地面用户无法观测到）。
    """
    connections = {}#保存最终的用户-卫星可达关系。
    cell_satellites = {}#把卫星按照 H3 格子索引（h3_index）分类存放。

    for sat_name, sat_position in satellites.items():
        h3_index = to_h3_index(sat_position["latitude"], sat_position["longitude"])
        if h3_index not in cell_satellites:
            cell_satellites[h3_index] = []
        cell_satellites[h3_index].append(
            (sat_name, to_cartesian(sat_position["latitude"], sat_position["longitude"], sat_position["altitude"])))

    max_dist = max_distance(elevation_angle, satellites[next(iter(satellites))]["altitude"])
    #根据仰角阈值和卫星高度，算出地面用户能“看到”的卫星的 最大直线距离。
    #next(iter(satellites)) 取任意一个卫星的高度，假设所有卫星高度一样。
    for user_name, user_position in ground_users.items():
        user_h3_index = to_h3_index(user_position["latitude"], user_position["longitude"])#计算用户的 H3 网格位置 user_h3_index
        user_cartesian = to_cartesian(user_position["latitude"], user_position["longitude"],#转换成三维坐标 user_cartesian。
                                      user_position.get("altitude", 0))

        reachable_satellites = set()#用集合存储当前用户能连上的卫星。
        for neighbor_h3_index in get_h3_neighbors(user_h3_index):
            """
            获取用户所在 H3 网格及其邻居格子。
            如果邻居格子里有卫星，就逐个拿出来检查
            """
            if neighbor_h3_index in cell_satellites:
                for sat_name, sat_cartesian in cell_satellites[neighbor_h3_index]:
                    actual_distance = math.sqrt(
                        (sat_cartesian[0] - user_cartesian[0]) ** 2 +
                        (sat_cartesian[1] - user_cartesian[1]) ** 2 +
                        (sat_cartesian[2] - user_cartesian[2]) ** 2)
                    if actual_distance <= max_dist:
                        reachable_satellites.add(sat_name)
                        """
                        计算用户和卫星的直线距离。
                        如果小于等于 max_dist,说明这个卫星在用户的仰角阈值以内，可以连上。
                        """

        connections[user_name] = list(reachable_satellites)#把每个用户的可见卫星存进结果字典。

    return connections
