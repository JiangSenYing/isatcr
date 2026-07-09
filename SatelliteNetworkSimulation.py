from Make_Satellite_Graph import SatelliteTracker,SatelliteGraph
from skyfield.api import load,Topos
import random
import datetime
from Read_Ground_Imformation import extract_landmarks, get_connections_h3
from SatelliteNetworkSimulator_Beta import SatelliteNetworkSimulator,Logger
from Draw_Graph_Quiker import SatelliteVisualizer
"""
    初始化模拟环境（时间范围、卫星 / 地面站数据、网络参数等）；
    按时间步长推进模拟，动态更新卫星网络拓扑（如节点 / 边的增减）；
    模拟卫星与地面站之间的数据包传输、路由等行为；
    收集并统计网络性能指标（如丢包率、延迟、跳数等）；
    支持网络状态可视化（可选）。
"""
class SatelliteSimulation:
    def __init__(self, begin_time, end_time, time_stride, tle_filepath,SOD_file_path, mean_interarrival_time,queue_length,transmission_rate,packet_size, state_update_period,
             visualize=False,print_info=False, save_log=False, show_detail=False,random_edges_del=0,random_nodes_del=0,elevation_angle=45,pole=False):
        self.tracker = SatelliteTracker(tle_filepath)
        self.coordinates = extract_landmarks(SOD_file_path)#地面站坐标（通过extract_landmarks从 SOD 文件提取）
        self.graph_builder = SatelliteGraph()
        self.begin_time = begin_time
        self.end_time = end_time
        self.time_stride = time_stride#模拟时间步长（每次推进的时间，单位秒）
        self.mean_interarrival_time=mean_interarrival_time
        self.queue_length=queue_length
        #设置包生成的平均到达时间、队列长度。
        self.visualizer = SatelliteVisualizer(edge_color=False) if visualize else None
        self.logger = Logger(detail=show_detail, save_log=save_log, verbose=print_info)
        self.ts = load.timescale()# 创建 Timescale 对象，sat.at(time) 方法需要一个 Skyfield 的 Time 对象作为参数
        #skyfield 的时间对象，用来转 UTC 时间
        self.transmission_rate=transmission_rate
        self.packet_size=packet_size
        self.state_update_period=state_update_period
        #链路速率、包大小、状态更新周期。
        self.random_edges_del=random_edges_del
        self.random_nodes_del=random_nodes_del
        #每一步随机删除多少边/节点（模拟链路丢失或卫星故障）
        self.elevation_angle=elevation_angle
        self.pole=pole
        #地面站仰角阈值（影响可见性），是否考虑极区拓扑特殊性。
        self.staticis_list=[]
        self.time_acc=0.0
        #保存统计数据，累积时间步长。

    def time_from_str(self,time_str):
        dt = datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        return self.ts.utc(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)#解析字符串
    """
    将字符串格式的时间（如"2023-07-02 20:42:15")转换为skyfield库的时间对象,用于卫星轨道计算(skyfield需特定时间格式计算卫星位置)。
    也即后续函数中的参数(time)
    """

    def add_time_to_str(self,time_str, delta_time_tuple):
        dt = datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        delta = datetime.timedelta(minutes=delta_time_tuple[0], seconds=delta_time_tuple[1])
        updated_dt = dt + delta
        return updated_dt.strftime("%Y-%m-%d %H:%M:%S")
    """
    给时间字符串增加指定时间间隔(delta_time_tuple为(分钟, 秒)），返回更新后的时间字符串，用于推进模拟时间。
    """

    def str_to_datetime(self, time_str):
        return datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
    """
    将时间字符串转换为datetime对象,用于计算时间差。

    """

    def datetime_difference_in_seconds(self, dt1, dt2):#计算两个datetime对象的时间差（秒），用于确定模拟总时长和步数。
        diff = dt2 - dt1
        return diff.total_seconds()

    def usage_to_rgb(self, usage):
        return 'rgb('+str(int(255 * usage)) +','+str(int(255 * (1 - usage))) + ','+ str(int(255 * (1 - usage))) + ')'
        #将资源使用率（如队列使用率）转换为 RGB 颜色值，用于可视化时节点颜色的动态变化（使用率越高，红色越深）。
    def update_node_colors(self, graph, total_queue_usage):#更新图中节点的颜色（目前逻辑为默认黑色，注释中预留了根据队列使用率动态调整颜色的功能）。
        for node in graph.nodes:
            graph.nodes[node]['color'] = 'black'
            #根据队列使用率动态调整颜色的功能
            # if node in total_queue_usage:
            #     usage = total_queue_usage[node]
            #     color = self.usage_to_rgb(usage)
            #     graph.nodes[node]['color'] = color
            # else:
            #     graph.nodes[node]['color'] = 'rgb(0,255,255)'

    def remove_random_edges(self,G, n):
        if n > G.number_of_edges():
            raise ValueError("Cannot remove more edges than exist in the graph")

        edges_to_remove = random.sample(list(G.edges()), n)
        G.remove_edges_from(edges_to_remove)

        return G

    def remove_random_nodes(self,G, n):
        if n > G.number_of_nodes():
            raise ValueError("Cannot remove more nodes than exist in the graph")

        nodes_to_remove = random.sample(list(G.nodes()), n)
        G.remove_nodes_from(nodes_to_remove)

        return G

#将地面站的经纬度坐标转换为地心惯性坐标系（ECI）坐标。ECI 是卫星轨道计算的常用坐标系，便于统一计算卫星与地面站的相对位置。
    def convert_to_eci(self,landmarks, time):
        eci_landmarks = {}
        for name, coords in landmarks.items():
            topo = Topos(latitude_degrees=coords['latitude'], longitude_degrees=coords['longitude'],
                         elevation_m=coords['altitude'])
            eci_coords = topo.at(time).position.km
            eci_landmarks[name] = {"x": eci_coords[0], "y": eci_coords[1], "z": eci_coords[2]}
        return eci_landmarks

    def visualize(self,current_graph,current_time,simulator):
        G_draw = current_graph.copy()
        self.update_node_colors(G_draw, simulator.get_system_state())
        landmark_ecis = self.convert_to_eci(self.coordinates, self.time_from_str(current_time))
        for landmark_eci, eci_value in landmark_ecis.items():
            G_draw.add_node(landmark_eci)
            G_draw.nodes[landmark_eci]['pos'] = [eci_value['x'], eci_value['y'], eci_value['z']]
            G_draw.nodes[landmark_eci]['color'] = 'rgb(200,200,200)'
        self.visualizer.draw_graph(G_draw)

    def run(self):
        init_time=self.time_from_str(self.begin_time)
        current_graph = self.graph_builder.build_graph_with_fixed_edges(self.tracker,init_time,pole=self.pole)

        coordinates_s = self.tracker.generate_satellite_LLA_dict(init_time)
        connections = get_connections_h3(self.coordinates, coordinates_s, self.elevation_angle)
        #生成卫星坐标（LLA），用 H3 算法算地面站可见卫星。

        simulator = SatelliteNetworkSimulator(
            graph=current_graph,
            landmarks=connections,
            mean_interarrival_time=self.mean_interarrival_time,#数据包到达间隔均值（用于生成模拟流量）；
            queue_length=self.queue_length,
            transmission_rate=self.transmission_rate,
            packet_size=self.packet_size,
            state_update_period=self.state_update_period,
            logger=self.logger)


        current_time = self.begin_time
        total_time = self.datetime_difference_in_seconds(self.str_to_datetime(self.begin_time),
                                                         self.str_to_datetime(self.end_time))
        num_full_steps = int(total_time // self.time_stride)
        remaining_time = total_time % self.time_stride
        #计算总时长，算能跑多少完整步长，剩余多少秒。
        i = 0
        while i < num_full_steps:
            print("======"+current_time+"======")
            simulator.run(self.time_stride)
            """
            让 SatelliteNetworkSimulator 运行 time_stride 的仿真时间（例如推进 1 秒或 10 秒）。
            这一步会产生/转发包、更新队列状态、累计延迟/跳数/丢包等统计数据（具体行为依 simulator 的实现而定）。
            """
            if self.staticis_list:
                current_statics = {k: simulator.statics_data[k] - self.staticis_list[-1][k] for k in simulator.statics_data}
            else:
                current_statics=simulator.statics_data
            """
            如果之前有保存过统计快照(self.staticis_list 非空），这里计算“本步增量” = 当前累计 - 上次累计（常用于得到 step-level 的统计）。
            否则第一次就直接把累计数据当本步数据。
            """
            print("Current statics:",current_statics)
            Total, Reached, Lost_upload, Lost_relay, Total_delay, Total_hops = current_statics.values()
            """
            Total:产生/尝试发送的数据包总数
            Reached:成功到达目的地的包数
            Lost_upload:上传阶段丢失（或源头丢失）
            Lost_relay:在中继/转发过程中丢失的包数
            Total_delay:本步累计延迟（秒）
            Total_hops:本步累计跳数（跳数总和）
            """
            if Lost_relay + Reached > 0:
                packet_loss_rate= Lost_relay / (Lost_relay + Reached)
                print(f"Packet loss rate: {packet_loss_rate:.2%}")
            if Reached > 0:
                print(f"Average delay for successful transmissions: {Total_delay / Reached:.3f} second")
                print(f"Average hop count for successful transmissions: {Total_hops / Reached:.3f} hops")#打印关键性能指标（丢包率、平均延迟、平均跳数）。
            self.staticis_list.append(simulator.statics_data.copy())#把当前累计统计快照保存到 self.staticis_list，供下次计算增量使用。
            # simulator.clear_statics()
            if self.visualizer:
                self.visualize(current_graph, current_time, simulator)
            #bug更新时间：通过add_time_to_str推进当前时间；(????????????)
            self.time_acc += self.time_stride
            if self.time_acc >= 1.0:
                current_time = self.add_time_to_str(current_time, (0, int(self.time_acc)))
                self.time_acc -= int(self.time_acc)
            i += 1
            """
            更新网络拓扑：
            重新计算卫星位置，构建新的网络拓扑；
            随机删除指定数量的节点 / 边（模拟故障）；
            更新地面站与卫星的连接关系（移除已离线的卫星）；
            调用simulator.upgrade_all更新模拟器的网络拓扑和连接关系。
            """
            coordinates_s = self.tracker.generate_satellite_LLA_dict(self.time_from_str(current_time))
            connections = get_connections_h3(self.coordinates, coordinates_s,self.elevation_angle)
            old_nodes = set(current_graph.nodes())#取出当前图的节点集合 old_nodes（便于后续比较哪些节点被删掉）
            current_graph = self.graph_builder.build_graph_with_fixed_edges(self.tracker, self.time_from_str(current_time),pole=False)
            """
            使用 graph_builder(基于卫星位置和规则）重建当前时刻的卫星拓扑 current_graph(包含卫星节点和 ISL 边等）。
            """
            self.remove_random_nodes(current_graph,self.random_nodes_del)
            self.remove_random_edges(current_graph,self.random_edges_del)
            #按配置随机删除若干节点或边，用于模拟节点/链路故障或随机扰动。
            new_nodes = set(current_graph.nodes())
            lost_nodes = old_nodes - new_nodes
            for landmark, satellites in connections.items():#遍历地面站到卫星的 connections，如果某颗卫星在 lost_nodes 中，则从该地面站的卫星列表中移除。
                for lost_node in lost_nodes:
                    if lost_node in satellites:
                        connections[landmark].remove(lost_node)
            simulator.upgrade_all(current_graph,connections)
            """
            把新的拓扑 current_graph 和更新后的 connections 传给模拟器，
            更新模拟器内部的网络结构与地面可见性信息（从而下一步 simulator.run() 会基于新拓扑仿真）。
            """

        if remaining_time > 0:
            simulator.run(remaining_time)