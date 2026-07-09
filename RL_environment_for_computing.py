from SatelliteNetworkSimulator_Computing import SatelliteNetworkSimulator_OnbardComputing
from SatelliteNetworkSimulation import SatelliteSimulation
from Make_Satellite_Graph import SatelliteTracker,SatelliteGraph# 卫星跟踪与图构建
from skyfield.api import load# 时间与轨道计算工具
from Read_Ground_Imformation import extract_landmarks, get_connections_h3
from SatelliteNetworkSimulator_Beta import Logger
from Draw_Graph_Quiker import SatelliteVisualizer_geo# 可视化工具
import random
import copy
import os
from common import (can_build_isl,_get_satellite_lla,_get_satellite_eci,
                    _get_satellite_velocity,satellite_adjacency_matrix,isLeo,
                    doppler_shift_value,get_satellite_speed,eci_distance
                    )
from SatelliteDomainPartitioner import SatelliteDomainPartitioner
# 导入观测包装器
try:
    from observation_wrappers import (
        create_observation_wrapper, 
        FlatObservation, 
        RelationalObservation, 
        GraphObservation,
        REGISTRY as OBS_REGISTRY
    )
    OBS_WRAPPERS_AVAILABLE = True
except ImportError:
    OBS_WRAPPERS_AVAILABLE = False
    print("Warning: observation_wrappers not found. Only flat observations available.")

#该环境模拟了低轨（LEO）卫星星座的动态拓扑、计算任务调度和数据路由过程，为强化学习智能体提供了交互接口（如step、reset），用于训练智能体优化计算与路由的联合决策。
class SatelliteEnv(SatelliteSimulation):
    def __init__(self,mode,select_mode,q_net,epsilon,reward_factors,device,mission_possibility,poisson_rate,packet_frequency,computing_demand_factor,computing_demand_factor_2,size_after_computing_factor,size_after_computing_1,
                 begin_time, end_time, time_stride, tle_filepath, SOD_file_path, mean_interval_time,memory,
                 computing_ability, transmission_rate,downlink_rate,downstream_delays, packet_size_range, state_update_period, print_cycle,del_cycle, visualize=False,
                 print_info=False, save_log=False, show_detail=False,random_edges_del=0,random_nodes_del=0,update_cycle=1,save_training_data=None,
                 training_data_dir='./training_process_data', elevation_angle=45, controlDomainNumber = 4,minimuElevationAngle = 25,showLink = False,pole=False, n_hops=1, obs_type='flat',
                 transformer = None, business_record_enabled=False, business_record_path=None,
                 business_record_reset=True, domainPartitionMethod='Eunomia',
                 rectangular_m=1, rectangular_n=1, meo_state_update_period=None, leo_action_mask_enabled=False):
        """
        初始化卫星网络强化学习环境
        
        新增参数:
            obs_type: 观测类型 ('flat', 'relational', 'graph', 'relational_separated', 'graph_separated')
                - 'flat': 扁平向量格式，适用于 MLP 网络
                - 'relational': 关系图格式，适用于简单 GNN
                - 'graph': 完整图格式，适用于多跳 GNN
                - 'relational_separated': 分离模式关系图，agent特征仅含环境信息，任务信息独立传递
                - 'graph_separated': 分离模式完整图，agent特征仅含环境信息，任务信息独立传递
        """
        #网络与硬件配置参数
        self.tracker = SatelliteTracker(tle_filepath)# 从TLE文件跟踪卫星轨道；TLE（两行轨道元素）文件路径，存储卫星轨道数据（用于计算卫星实时位置）
        self.coordinates = extract_landmarks(SOD_file_path)#从SOD文件提取地面地标坐标，地面站信息文件路径，存储地面站的经纬度、海拔等坐标信息。
        self.graph_builder = SatelliteGraph()## 构建卫星网络拓扑图
        self.memory=memory
        self.computing_ability=computing_ability
        self.transmission_rate = transmission_rate#星间传输速率
        self.downlink_rate=downlink_rate#下行链路速率
        self.downstream_delays=downstream_delays#下行延迟
        self.elevation_angle = elevation_angle#仰角阈值;度
        self.pole = pole#控制卫星轨道范围是否覆盖极地
        self.ts = load.timescale()#来自 skyfield.api 的 load.timescale() 方法，用于创建一个时间尺度对象
                                  #用于将人类可读的时间（如 "2023-01-01 00:00:00"）转换为卫星轨道计算所需的精确时间格式。被用于 time_from_str 等方法，将字符串格式的时间转换为卫星轨道计算可用的时间对象

        #仿真时间与周期参数
        self.begin_time = begin_time
        self.end_time = end_time#仿真开始和结束时间
        self.time_stride = time_stride#仿真时间步长（单位：秒），每次调用step方法推进的时间长度。
        self.update_cycle = update_cycle#拓扑更新周期（单位：秒），每隔该时间重新构建卫星网络拓扑（因卫星运动导致连接关系变化）。
        self.current_cycle = 0.0
        self.del_cycle = del_cycle#随机删除周期（单位：秒），每隔该时间随机删除指定数量的节点 / 边（模拟网络故障）。
        self.del_update=True
        self.current_del_cycle = 0.0
        self.last_removed_nodes = set()
        self.last_removed_edges = []
        self.print_cycle = print_cycle
        self.current_print_cycle = 0.0
        self.iteration_counter = 0
        self.print_cycle_iterations = int(print_cycle / time_stride)
        self.statics = []#存储仿真过程中的累积统计数据快照；
                         #每次达到打印周期（print_cycle）时，会将当前的统计数据（如成功传输的数据包数量、丢失的数据包数量、总延迟等）存入该列表，用于后续计算周期内的增量统计（例如，计算两个快照之间的数据包丢失率、平均延迟等）。
                         #print_and_save_accumulated_data中，通过对比 self.statics 中存储的历史快照和当前统计数据，计算周期内的增量指标。
        self.latest_transformer_losses = None
        self.time_acc = 0.0#仿真中每次调用 step 方法会累加 time_stride（步长，如 1 秒），但可能存在小数部分（例如步长为 0.5 秒时）。self.time_acc 用于累积这些小数，当累计达到 1 秒时，更新 current_time（将时间字符串加 1 秒），并重置小数部分。
        
        # 观测类型配置
        self.obs_type = obs_type
        self.obs_wrapper = None
        if OBS_WRAPPERS_AVAILABLE and obs_type != 'flat':
            self._init_obs_wrapper(obs_type)
            """
            仿真器里每颗卫星产生的原始状态是一个字典或向量（内存余量、计算余量、邻居信息等），
            obs_wrapper 负责将其转换为 Q 网络能直接处理的格式：
            """
        
        #动态参数模拟
        self.state_update_period = state_update_period#卫星状态更新周期（单位通常为秒），控制卫星之间交换状态信息的频率
        self.meo_state_update_period = meo_state_update_period if meo_state_update_period is not None else state_update_period
        self.random_edges_del = random_edges_del#每次删除的随机边数，模拟星间链路中断
        self.random_nodes_del = random_nodes_del#每次删除的随机节点数，模拟卫星失效
       
        #日志与可视化参数
        self.visualizer = SatelliteVisualizer_geo(edge_color=False) if visualize else None
        self.logger = Logger(detail=show_detail, save_log=save_log, verbose=print_info)
        self.save_training_data=save_training_data#训练数据保存路径
        self.training_data_dir = training_data_dir
        
        #强化学习相关参数
        self.mode=mode
        self.select_mode=select_mode#动作选择模式（如贪心选择、随机选择等）
        self.q_net=q_net#估计动作价值，指导决策
        self.leo_action_mask_enabled = bool(leo_action_mask_enabled)
        self.epsilon=epsilon# ε-贪婪策略参数
        self.reward_factors = reward_factors# 奖励函数权重
        self.device=device
        
        #Transformer模块配置
        self.transformer_module = transformer
        
        #任务与数据生成参数
        self.packet_size_range = packet_size_range#数据包大小范围（如(1024, 4096)），任务数据包的随机大小区间。
        self.mean_interval_time = mean_interval_time#任务间隔均值（单位：秒），用于生成任务到达的平均间隔。
        self.poisson_rate=poisson_rate#泊松过程速率，用于模拟任务到达的时间间隔分布
        self.packet_frequency=packet_frequency#数据包生成频率，单位时间内生成的任务数量。
        self.computing_demand_factor=computing_demand_factor#计算需求系数，用于计算任务的计算量（如计算量 = 数据包大小 × 系数）。
        self.computing_demand_factor_2=computing_demand_factor_2
        self.size_after_computing_1=size_after_computing_1#计算后数据包大小系数，任务在卫星上计算后，输出数据的大小由原始大小乘以该系数得到（模拟计算后数据压缩 / 膨胀）
        self.size_after_computing_factor=size_after_computing_factor
        self.mission_possibility=mission_possibility# 任务生成概率
        self.n_hops=n_hops  # GNN邻居跳数配置
        self.domainPartitionMethod = domainPartitionMethod
        self.rectangular_m = rectangular_m
        self.rectangular_n = rectangular_n
        self.controlDomainNumber = controlDomainNumber
        self.minimuElevationAngle = minimuElevationAngle
        self.showLink = showLink

        

        self.step_num=0#记录强化学习环境中 step 方法被调用的总次数（即仿真步数）。标识当前仿真进度；日志打印以step_num 作为索引输出当前步数的统计信息（如丢包率、平均延迟等）。
        self.rewards=[]#每个数据包的处理结果（如成功传输、计算完成等）会产生奖励值，这些奖励会被添加到 self.rewards 中。当达到打印周期时，会计算该列表的平均值并打印，然后清空列表（self.rewards = []），为下一个周期的奖励存储做准备。

        self.current_graph = None
        self.connections = None#存储地面站与卫星之间的通信连接关系。
        self.reset(self.begin_time,controlDomainNumber = controlDomainNumber,minimuElevationAngle = minimuElevationAngle,showLink = showLink)
        self.current_time = self.begin_time
        self.satelliteDomainPartitioner = None
        
    def _init_obs_wrapper(self, obs_type):
        """
        初始化观测包装器
        
        Args:
            obs_type: 观测类型 ('flat', 'relational', 'graph')
        """
        if not OBS_WRAPPERS_AVAILABLE:
            print(f"Warning: observation_wrappers not available, falling back to 'flat'")
            self.obs_type = 'flat'
            return
        
        try:
            if obs_type == 'flat':
                # 根据模式确定状态维度
                if 'New' in self.mode:
                    state_dim = 4 * 14 + 3 + 4 + 2  # 65维
                else:
                    state_dim = 4 * 6 + 3 + 4 + 2  # 33维
                self.obs_wrapper = FlatObservation(state_dim=state_dim)
                
            elif obs_type == 'relational':
                self.obs_wrapper = RelationalObservation(
                    agent_feat_dim=9,  # current_node(3) + mission(4) + routing(2)
                    nbr_feat_dim=6,    # 每个邻居的特征维度
                    max_neighbors=4
                )
            
            elif obs_type == 'relational_separated':
                # 分离模式：agent 节点仅包含环境特征，task 单独传递
                self.obs_wrapper = RelationalObservation(
                    agent_feat_dim=3,  # current_node 环境特征: [memory_remain, computing_remain, is_producing]
                    nbr_feat_dim=6,    # 每个邻居的特征维度
                    max_neighbors=4
                )
                
            elif obs_type == 'graph':
                self.obs_wrapper = GraphObservation(
                    agent_feat_dim=9,
                    nbr_feat_dim=6,
                    hop2_feat_dim=6,
                    max_neighbors=4,
                    max_hop2_neighbors=3
                )
            
            elif obs_type == 'graph_separated':
                # 分离模式：agent 节点仅包含环境特征，task 单独传递
                self.obs_wrapper = GraphObservation(
                    agent_feat_dim=3,  # current_node 环境特征
                    nbr_feat_dim=6,
                    hop2_feat_dim=6,
                    max_neighbors=4,
                    max_hop2_neighbors=3
                )
            else:
                print(f"Warning: Unknown obs_type '{obs_type}', falling back to 'flat'")
                self.obs_type = 'flat'
                state_dim = 4 * 6 + 3 + 4 + 2
                self.obs_wrapper = FlatObservation(state_dim=state_dim)
                
        except Exception as e:
            print(f"Error initializing obs_wrapper: {e}, falling back to 'flat'")
            self.obs_type = 'flat'
            self.obs_wrapper = None

    def remove_random_edges(self,G, n,update=False):
        # if n > G.number_of_edges():
        #     raise ValueError("Cannot remove more edges than exist in the graph")
        # if update:
        #     self.last_removed_edges = random.sample(list(G.edges()), n)
        # G.remove_edges_from(self.last_removed_edges)

        """根据星间链路几何可见性删边。
        保留 n/update 参数是为了兼容现有调用；实际删除的边由 can_build_isl 判定。
        """
        edges_to_remove = []
        if n > 0:
            if n > G.number_of_edges():
                raise ValueError("Cannot remove more edges than exist in the graph")
            if update:
                edges_to_remove = random.sample(list(G.edges()), n)
        else:
            distances = []
            for node, neighbor in list(G.edges()):
                x1, y1, z1 = _get_satellite_eci(G, node)
                v_x1, v_y1, v_z1 = _get_satellite_velocity(G, node)
                x2, y2, z2 = _get_satellite_eci(G, neighbor)
                v_x2, v_y2, v_z2 = _get_satellite_velocity(G, neighbor)
                if isLeo(node) and isLeo(neighbor):
                    max_visible_distance_km = 3000
                    snr_thread = 8
                elif (isLeo(node) is True and not isLeo(neighbor)) or (isLeo(neighbor) is True and not isLeo(node)):
                    max_visible_distance_km = 15000
                    snr_thread = 10
                else:
                    max_visible_distance_km = 30000
                    snr_thread = 15
                can_link, info = can_build_isl(pos1_km=[x1, y1, z1], vel1_km_s=[v_x1, v_y1, v_z1], carrier_frequency_1_ghz=self.simulator.satellites[node].communication_frequency,bandwidth_1=self.simulator.satellites[node].transmission_rate,power_1=self.simulator.satellites[node].power,
                                               pos2_km=[x2, y2, z2], vel2_km_s=[v_x2, v_y2, v_z2], carrier_frequency_2_ghz=self.simulator.satellites[neighbor].communication_frequency,bandwidth_2=self.simulator.satellites[neighbor].transmission_rate,power_2=self.simulator.satellites[neighbor].power,
                                               doppler_threshold_mhz = 0.3, max_visible_distance_km = max_visible_distance_km,snr_thread_db = snr_thread)
                if max_visible_distance_km == 30000 and can_link == False:
                    a = 1
                if not can_link:
                    edges_to_remove.append((node, neighbor))
        self.last_removed_edges = edges_to_remove
        G.remove_edges_from(edges_to_remove)
        return G

    def remove_random_nodes(self,G, n,update=False):
        if n > G.number_of_nodes():
            raise ValueError("Cannot remove more nodes than exist in the graph")
        if update:
            self.last_removed_nodes = random.sample(list(G.nodes()), n)
        # G.remove_nodes_from(self.last_removed_edges) bug
        G.remove_nodes_from(self.last_removed_nodes)

        return G

    def step(self, epsilon):
        """
        执行一个环境时间步
        
        Args:
            epsilon: 探索率
        
        Returns:
            experiences: 经验列表（格式取决于 obs_type）
                - 'flat': [state, mark, action, reward, next_state, done]
                - 'relational'/'graph': [obs_dict, mark, action, reward, next_obs_dict, done]
        """
        self.step_num+=1
        #管理随机删除节点 / 边的周期（del_cycle），当累计的当前周期时间达到设定值后，重置周期计数，并标记需要更新随机删除的节点/边列表
        self.current_del_cycle+=self.time_stride
        if self.current_del_cycle >= self.del_cycle:#bug更正：self.del_cycle->self.current_del_cycle
            # self.del_cycle += -self.del_cycle#重置周期计数
            self.current_del_cycle += -self.current_del_cycle#重置周期计数
            self.del_update=True
        #网络拓扑更新    
        self.current_cycle+=self.time_stride
        if self.current_cycle >= self.update_cycle:
            self.current_cycle += -self.update_cycle
            # 生成当前时间的卫星经纬度高度（LLA）字典
            coordinates_s = self.tracker.generate_satellite_LLA_dict(self.time_from_str(self.current_time))
            # 获取地面站与卫星的连接关系（基于仰角筛选）
            connections = get_connections_h3(self.coordinates, coordinates_s, self.elevation_angle)
            # 记录更新前的节点集合
            old_nodes = set(self.simulator.graph.nodes())
            # 构建当前时间的卫星网络拓扑图
            current_graph = self.graph_builder.build_graph_with_fixed_edges(self.tracker, self.time_from_str(self.current_time),pole=False)

            self.remove_random_nodes(current_graph, self.random_nodes_del,self.del_update)
            self.remove_random_edges(current_graph, self.random_edges_del,self.del_update)

            self.del_update = False
            # 处理拓扑变化导致的节点丢失
            new_nodes = set(current_graph.nodes())
            lost_nodes = old_nodes - new_nodes
            # 更新地面站与卫星的连接（移除已丢失的卫星）
            for landmark, satellites in connections.items():
                for lost_node in lost_nodes:
                    if lost_node in satellites:
                        connections[landmark].remove(lost_node)
            self.current_graph=current_graph
            self.connections=connections
            dic = self.satelliteDomainPartitioner.calcDomainPartition(current_graph, self.simulator.env.now)
        # 用新的拓扑图和连接关系更新模拟器  
        self.simulator.upgrade_all(self.current_graph, self.connections)
        for satellite in self.simulator.satellites:
            self.simulator.satellites[satellite].epsilon=epsilon#更新探索率，用于动作选择（相当于到这里，已经将更新的拓扑和动作都写到模拟器了，下面让模拟器运行，执行该时间步内的模拟逻辑）
        if self.transformer_module is not None and hasattr(self.transformer_module, 'add_env_snapshot'):
            self.transformer_module.add_env_snapshot(self, replace_same_time=True)
        # 让模拟器运行 time_stride 时长（处理该时间段内的数据包传输、计算等事件）
        self.simulator.run(self.time_stride)
        
        # 根据观测类型获取经验数据
        if self.obs_type in ('relational', 'graph', 'relational_separated', 'graph_separated') and hasattr(self.simulator.propagator, 'get_experiences'):
            experiences = self.simulator.propagator.get_experiences()
            # 清空对应的经验列表
            self.simulator.propagator.experiences = []
            self.simulator.propagator.experiences_graph = []
        else:
            # 默认获取扁平格式经验
            experiences = self.simulator.propagator.experiences
            self.simulator.propagator.experiences = []
        
        self.rewards.extend(self.simulator.propagator.final_rewards)# 收集该步的最终奖励
        self.simulator.propagator.final_rewards = []
        self.iteration_counter += 1
        if self.iteration_counter >= self.print_cycle_iterations:
            self.iteration_counter = 0
            self.print_and_save_accumulated_data()
            self.current_print_cycle = 0.0
            self.rewards=[]

        #更新环境的当前时间（模拟真实时间流逝）
        self.time_acc += self.time_stride
        if self.time_acc >= 1.0:
            self.current_time = self.add_time_to_str(self.current_time, (0, int(self.time_acc)))
            self.time_acc -= int(self.time_acc)

        return experiences

    def reset(self,begin_time,controlDomainNumber=None, minimuElevationAngle=None, showLink=None):
        if controlDomainNumber is None:
            controlDomainNumber = self.controlDomainNumber
        if minimuElevationAngle is None:
            minimuElevationAngle = self.minimuElevationAngle
        if showLink is None:
            showLink = self.showLink
        self.controlDomainNumber = controlDomainNumber
        self.minimuElevationAngle = minimuElevationAngle
        self.showLink = showLink
        self.statics= []
        self.latest_transformer_losses = None
        self.begin_time=begin_time
        init_time = self.time_from_str(begin_time)
        current_graph = self.graph_builder.build_graph_with_fixed_edges(self.tracker, init_time, pole=self.pole)
        self.num_nodes = len(current_graph.nodes())
        coordinates_s = self.tracker.generate_satellite_LLA_dict(init_time)
        connections = get_connections_h3(self.coordinates, coordinates_s, self.elevation_angle)
        self.simulator = SatelliteNetworkSimulator_OnbardComputing(
            mode=self.mode,
            select_mode=self.select_mode,
            q_net=self.q_net,
            reward_factors = self.reward_factors,
            epsilon= self.epsilon,
            device= self.device,
            mission_possibility=self.mission_possibility,
            poisson_rate= self.poisson_rate,
            packet_frequency= self.packet_frequency,
            computing_demand_factor= self.computing_demand_factor,
            computing_demand_factor_2=self.computing_demand_factor_2,
            size_after_computing_factor= self.size_after_computing_factor,
            size_after_computing_1=self.size_after_computing_1,
            graph=current_graph,
            landmarks=connections,
            mean_interval_time=self.mean_interval_time,
            memory=self.memory,
            computing_ability=self.computing_ability,
            transmission_rate=self.transmission_rate,
            downstream_delays=self.downstream_delays,
            downlink_rate=self.downlink_rate,
            packet_size_range=self.packet_size_range,
            state_update_period=self.state_update_period,
            meo_state_update_period=self.meo_state_update_period,
            logger=self.logger,
            n_hops=self.n_hops,
            obs_type=self.obs_type,
            obs_wrapper=self.obs_wrapper,
            transformer_module = self.transformer_module,
            maxOrbitNumber=self.tracker.get_max_orbit_number(),
            maxLeoSatelliteNumber=self.tracker.get_max_satellite_number(),
            controlDomainNumber = controlDomainNumber,
            minimuElevationAngle = minimuElevationAngle,
            showLink = showLink,
            domainPartitionMethod = self.domainPartitionMethod,
            rectangular_m = self.rectangular_m,
            rectangular_n = self.rectangular_n,
            leo_action_mask_enabled=self.leo_action_mask_enabled,)
        self.current_time = self.begin_time
        self.time_acc = 0.0
        self.current_cycle =0.0
        self.current_print_cycle=0.0
        self.current_del_cycle = 0.0
        self.iteration_counter = 0
        self.del_update = True
        old_nodes = set(self.simulator.graph.nodes())
        current_graph = self.graph_builder.build_graph_with_fixed_edges(self.tracker,self.time_from_str(self.current_time), pole=False)

        self.remove_random_nodes(current_graph, self.random_nodes_del, True)
        self.remove_random_edges(current_graph, self.random_edges_del, True) #随机删除边15条
        new_nodes = set(current_graph.nodes())##返回图中所有节点（卫星或地面站）的列表，经过随机删除边之后的
        lost_nodes = old_nodes - new_nodes
        for landmark, satellites in connections.items():
            for lost_node in lost_nodes:
                if lost_node in satellites:
                    connections[landmark].remove(lost_node)
        self.current_graph = current_graph
        self.connections = connections
        self.satelliteDomainPartitioner = self.simulator.satelliteDomainPartitioner
        dic = self.satelliteDomainPartitioner.calcDomainPartition(current_graph, self.simulator.env.now)
        
    def render(self):
        self.visualize(self.simulator.graph, self.current_time, self.simulator)

    def visualize(self,current_graph,current_time,simulator):
        system_state=simulator.get_system_state()

        G_draw=current_graph.copy()
        self.update_node_colors(G_draw, system_state)

        for landmark, pos_value in self.coordinates.items():
            G_draw.add_node(landmark)
            G_draw.nodes[landmark]['pos_0'] = [pos_value["latitude"],pos_value ["longitude"], pos_value["altitude"]]
            G_draw.nodes[landmark]['color'] = 'purple'# 地面站节点
        self.visualizer.draw_graph(G_draw)

    def show_satellite_computing_time(self):
        satellite_computing_times={}
        for satellite in self.simulator.satellites:
            satellite_computing_times[satellite]=self.simulator.satellites[satellite].computing_time
        self.print_and_save(str(satellite_computing_times))

    def print_and_save(self, message):
        print(message)
        if self.save_training_data:
            os.makedirs(self.training_data_dir, exist_ok=True)
            file_path = os.path.join(self.training_data_dir, self.save_training_data)
            with open(file_path, 'a') as file:
                file.write(message + '\n')

    def print_and_save_accumulated_data(self):
        self.print_and_save(f"====== step {self.step_num} ======")
        self.print_and_save(f"====== {self.current_time} ======")
        # 计算当前周期的统计数据（与上一周期的差值）
        current_statics_snapshot = copy.deepcopy(self.simulator.statics_datas)
        if self.statics:
            current_statics = {k: current_statics_snapshot[k] - self.statics[-1].get(k, 0) for k in current_statics_snapshot}
        else:
            current_statics = current_statics_snapshot
        self.statics.append(current_statics_snapshot) # 保存快照用于下次计算
        # 解析统计数据并计算关键指标
        d = current_statics
        packet_loss_rates = (d['Lost_relay_0'] + d['Lost_relay_1'] + d['Lost_upload']) / (d['Lost_relay_0'] + d['Lost_relay_1'] + d['Lost_upload'] + d['Reached_0'] + d['Reached_1']) if d['Lost_relay_0'] + d['Lost_relay_1'] + d['Lost_upload'] + d['Reached_0'] + d['Reached_1'] > 0 else None
        average_delays = (d['Total_delay_0'] + d['Total_delay_1']) / (d['Reached_0'] + d['Reached_1']) if d['Reached_0'] + d['Reached_1'] > 0 else None
        average_hops = (d['Total_hops_0'] + d['Total_hops_1']) / (d['Reached_0'] + d['Reached_1']) if d['Reached_0'] + d['Reached_1'] > 0 else None
        average_computing_ratio = d['Is_computing'] / self.num_nodes / (self.print_cycle_iterations)
        average_computing_waiting_time = (d['Computing_waiting_time']) / (d['Reached_0'] + d['Reached_1']) if d['Reached_0'] + d['Reached_1'] > 0 else None
        out_memory_statics = {k: d[k] for k in sorted(d) if k.startswith('out_memory_')}
        illegal_action_statics = {k: d[k] for k in sorted(d) if k.startswith('illegal_action_')}

        self.print_and_save(f"current_statics: {current_statics}")
        self.print_and_save(f"Out of memory drops: {out_memory_statics}")
        self.print_and_save(f"Illegal action drops: {illegal_action_statics}")
        self.print_and_save(f"Packet loss rate: {'{:.2%}'.format(packet_loss_rates) if packet_loss_rates is not None else 'None'}")
        self.print_and_save(f"Average delay for successful transmissions: {'{:.3f} seconds'.format(average_delays) if average_delays is not None else 'None'}")
        self.print_and_save(f"Average hop count for successful transmissions: {'{:.3f} hops'.format(average_hops) if average_hops is not None else 'None'}")
        self.print_and_save(f"Proportion of satellites in computation: {'{:.2%}'.format(average_computing_ratio) if average_computing_ratio is not None else 'None'}")
        self.print_and_save(f"Average waiting time for computing: {'{:.3f} seconds'.format(average_computing_waiting_time) if average_computing_waiting_time is not None else 'None'}")

        rewards = sum(self.rewards) / len(self.rewards) if len(self.rewards) > 0 else None
        self.print_and_save(f"Average ending reward: {rewards if rewards is not None else 'None'}")
        transformer_losses = getattr(self, 'latest_transformer_losses', None)
        if transformer_losses:
            self.print_and_save(f"transformer_loss: {transformer_losses.get('transformer_loss')}")
            self.print_and_save(f"transformer_queue_loss: {transformer_losses.get('queue_loss')}")
            self.print_and_save(f"transformer_link_loss: {transformer_losses.get('link_loss')}")
            self.print_and_save(f"transformer_compute_queue_loss: {transformer_losses.get('compute_queue_loss')}")
            
    def build_graph_for_transformer(self, time):
        graph = self.graph_builder.build_graph_with_fixed_edges(self.tracker, time ,pole=False)
        self.remove_random_edges(graph, n = 0, update = True)
        return graph
