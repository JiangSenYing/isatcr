import simpy
import networkx as nx
import numpy as np
import logging
import random
from datetime import datetime
import heapq  # 用于实现优先队列
from common import isLeo

"""给图中所有节点和边添加后缀（如时间戳），用于区分不同时刻的拓扑图（避免动态更新时节点名称冲突）。"""
def add_suffix_to_graph(G, suffix):
    G_new = type(G)()# 创建与原图同类型的新图（如nx.Graph）
    for node in G.nodes():
        G_new.add_node(f"{node}{suffix}")
    for u, v, data in G.edges(data=True):
        G_new.add_edge(f"{u}{suffix}", f"{v}{suffix}", **data)
    return G_new

class Logger():
    def __init__(self,detail,save_log,verbose,num=None):
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.detail=detail
        self.save_log=save_log
        self.num=num
        if save_log:
            if num:
                if detail:
                    self.log_file = f"simulation_{num}_{current_time}_detail.log"
                else:
                    self.log_file = f"simulation_{num}_{current_time}.log"
            else:
                if detail:
                    self.log_file = f"simulation_{current_time}_detail.log"
                else:
                    self.log_file = f"simulation_{current_time}.log"
            if num:
                self.logger = logging.getLogger(f'SimulationLogger_{num}')
            else:
                self.logger = logging.getLogger(f'SimulationLogger')
            self.logger.setLevel(logging.INFO)
            handler = logging.FileHandler(self.log_file)
            handler.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(handler)
        self.verbose=verbose

    def log(self, message,detail=False):
        if not detail or (self.detail and detail):
            if self.verbose:
                print(message)
            if self.save_log:
                self.logger.info(message)

class Packet():
    def __init__(self,source,destination,creation_time,size):
        self.source=source
        self.destination=destination
        self.creation_time=creation_time
        self.packet_id = None
        self.business_session_id = None
        self.business_session_start_time = None
        self.business_session_duration = None
        self.path_trace = []
        self.computing_waiting_time=0
        self.hops = 0
        self.size=size
        self.information=[]
        self.routing=None
        self.computing_node=None
        self.computing_leo=None
        self.computing_meo=None
        self.path = None
        self.compute_flags = None
        self.is_intra_destination = False
        self.temporary_destination = None
        self.meo_result = None
        self.meo_terminal_time = None
        self.meo_segment_time = None
        self.meo_decision_trace = None
        self.meo_decision_traces = []

    def extra_information(self,information):
        self.information=information
        
    def add_path(self, path):
        self.path = path
    def add_temporary_destination(self, temporary_destination):
        self.temporary_destination = temporary_destination
    def add_compute_flat(self, flags):
        self.compute_flags = flags

class Propagator():
    def __init__(self,env,graph,logger,satellites,statics_data={},global_graph=False):
        self.env=env
        self.logger=logger
        self.propagation_speed=3e5
        #propagation_delay = distance / self.propagation_speed 表示信号的传播速度，其数值为 3×10⁵公里 / 秒（即 300,000 km/s），
        # 这一数值近似于真空中的光速（光速约为 299,792 km/s），在卫星通信场景中用于模拟无线电信号的传播速度。
        self.global_graph=global_graph# 是否为全局图（用于更新边属性）若为真，则把延迟写回图的边属性（用于全局权重/可视化）。
        self.graph = graph # 卫星网络拓扑图
        self.node_names=list(graph.nodes)
        self.node_positions = {node: graph.nodes[node]['pos'] for node in graph.nodes}#从 graph 的节点属性 pos 读取三维坐标（pos 应为 [x,y,z]）。
        self.node_neighbors={node: list(graph.neighbors(node)) for node in self.node_names}#['Satellite_1000_1_2', 'Satellite_1000_2_1']
        self.propagation_delays = {}
        self.calculate_delays() # 初始化传播延迟（(node1,node2)->delay）
        self.satellites=satellites
        self.meo_satellites = []
        self.statics_data=statics_data
        for name, sat in self.satellites.items():
            if sat.isLeo == False:
                self.meo_satellites.append(name)
        """
        不修改原始图的边属性：原始图（self.graph）仅用于存储卫星网络的基础拓扑结构（节点、边的存在性），不包含传播延迟（propagation_weight）、边状态（missing）等动态计算数据。
        传播延迟单独存储：传播延迟数据被保存在self.propagation_delays字典中，仅用于数据包传播时间的计算（如propagate方法中通过self.propagation_delays[(node, next_hop)]获取延迟）。
        避免冗余计算与图结构污染：卫星网络的拓扑会随时间动态变化（如update方法更新图），若global_graph=True，每次更新都需要修改图中所有边的属性，会增加计算开销并导致原始图结构被动态数据 “污染”。而False的设置让原始图保持简洁，仅作为拓扑基础，动态数据单独管理，更适合频繁更新的仿真场景。
        """
    def _distance(self, node1, node2):
        a, b = self.node_positions[node1], self.node_positions[node2]
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)

    def calculate_delays(self):
        self.propagation_delays = {}
        for node1, node2 in self.node_neighbors.items():
            for neighbor in node2:
                distance = self._distance(node1, neighbor)
                propagation_delay = distance / self.propagation_speed # 延迟=距离/速度
                self.propagation_delays[(node1, neighbor)] = propagation_delay
                self.propagation_delays[(neighbor, node1)] = propagation_delay# 双向延迟相同
                if self.global_graph:
                    self.graph[node1][neighbor]['propagation_weight']= propagation_delay
                    self.graph[node1][neighbor]['propagation_weight'] = propagation_delay
        if self.global_graph:
            for edge in self.graph.edges():
                node1, node2 = edge
                if edge in self.propagation_delays:
                    self.graph[node1][node2]['missing']=0
                else:
                    self.graph[node1][node2]['missing'] =1

    def update(self,graph):
        self.node_names=list(graph.nodes)
        self.node_positions = {node: graph.nodes[node]['pos'] for node in graph.nodes}
        self.node_neighbors={node: list(graph.neighbors(node)) for node in self.node_names}
        self.calculate_delays()
    """SimPy 进程函数(generator),被 env.process(...) 调用时执行"""
    def propagate(self,node,next_hop,packet):
        if (node, next_hop) in self.propagation_delays:#延迟信息（即链路存在）
            yield self.env.timeout(self.propagation_delays[(node, next_hop)])#模拟传播延时。
            if next_hop in self.node_names:#检查 next_hop 是否仍在节点列表 node_names（防止该节点在传播期间下线）
                success = self.satellites[next_hop].push_forward(packet)#包放进下一节点的前向队列
                if success:
                    self.logger.log(f"Time {self.env.now:.3f}: {next_hop}: Packet {(packet.source,packet.destination)} received by router. Transmission length: {self.satellites[next_hop].current_queue_length}.",detail=True)
                else:
                    if 'Lost_relay' in self.statics_data:
                        self.statics_data['Lost_relay'] += 1
                    self.logger.log(f"Time {self.env.now:.3f}: {next_hop}: Routing queue is full, discarding packet {(packet.source,packet.destination)}.")
            else:
                if 'Lost_relay' in self.statics_data:
                    self.statics_data['Lost_relay'] += 1
                self.logger.log(f"Time {self.env.now:.3f}: {next_hop} is missed, dropped 1 packet.")
        else:
            if 'Lost_relay' in self.statics_data:
                self.statics_data['Lost_relay'] += 1
            self.logger.log(f"Time {self.env.now:.3f}: connection {(node, next_hop)} is missed, dropped 1 packet.")

    """这里 push_forward 将包放入 forward_queue(而不是直接放入特定 neighbor 的 transmission_queue),这样下一步节点会在其 forward_packet() 进程中选择下一跳并放入 transmission_queue。"""

    """先 yield timeout(delay)，再把状态或邻接表通过 self.satellites[neighbor] 反向写到目标卫星对象上（用于心跳、邻居状态传播与 routing 广播）。"""
    def send_state(self, node, neighbor,value):
        if (node, neighbor) in self.propagation_delays:
            yield self.env.timeout(self.propagation_delays[(node, neighbor)])
            if neighbor in self.node_names:
                if node in self.satellites[neighbor].neighbors:
                    self.satellites[neighbor].neighbor_states[node]=value
                    self.satellites[neighbor].last_heartbeat[node]=self.env.now# 记录每个邻居最后一次心跳到达时间（用于检测邻居失效）。
                else:
                    self.satellites[neighbor].add_neighbor(node)#bug self.satellites[neighbor].add_neighbor(neighbor)------->self.satellites[neighbor].add_neighbor(node)
                    self.satellites[neighbor].neighbor_states[node] = value
            else:
                self.logger.log(f"Time {self.env.now:.3f}: {neighbor} is missed, state update failed.")
        else:
            self.logger.log(f"Time {self.env.now:.3f}: connection {(node, neighbor)} is missed, state update failed.")

    def send_adjacency_table(self, node, neighbor,table):
        if (node, neighbor) in self.propagation_delays:
            yield self.env.timeout(self.propagation_delays[(node, neighbor)])
            if neighbor in self.node_names:
                if node in self.satellites[neighbor].neighbors:
                    self.satellites[neighbor].update_adjacency_table(table)
                else:
                    self.satellites[neighbor].add_neighbor(node)#bug self.satellites[neighbor].add_neighbor(neighbor)------->self.satellites[neighbor].add_neighbor(node)
                    self.satellites[neighbor].update_adjacency_table(table)
            else:
                self.logger.log(f"Time {self.env.now:.3f}: {neighbor} is missed, state update failed.")
        else:
            self.logger.log(f"Time {self.env.now:.3f}: connection {(node, neighbor)} is missed, state update failed.")

    """
    ?????????????????
    在 else 分支里调用 add_neighbor(neighbor) 似乎有问题：如果 node 不是 neighbor 的邻居，应该调用 add_neighbor(node)（把 node 添加为 neighbor 的邻居），
    而不是 add_neighbor(neighbor)（把自己作为自己的邻居）。这看起来像拼写错误或逻辑错误，导致错误的邻居建立行为。需要把 add_neighbor(neighbor) 改成 add_neighbor(node)。
    
    """
"""模拟卫星的行为，包括数据包转发、路由计算、邻居管理等。"""
class Satellite():
    def __init__(self,env,name,neighbors,queue_length,transmission_rate,state_update_period,logger,statics_data={},processing_time=1e-9,heartbeat_timeout=0.5):
        self.name=name
        self.neighbors= neighbors
        self.env=env
        self.queue_length=queue_length# 队列最大长度（用于流量控制）
        self.transmission_rate=transmission_rate# 传输速率（单位：如kb/s，用于计算传输时间）
        self.state_update_period=state_update_period# 状态更新周期（多久向邻居发送一次状态）
        self.logger=logger
        self.transmission_queue = {neighbor: simpy.Store(self.env) for neighbor in self.neighbors}
        self.transmission_length ={neighbor: 0 for neighbor in self.neighbors}
        self.forward_queue = simpy.Store(self.env)
        self.current_queue_length=0
        self.active=True
        self.routing_tables={}
        self.neighbor_states={neighbor: 0 for neighbor in self.neighbors}
        self.propagator=None
        self.statics_data=statics_data
        self.processing_time=processing_time
        self.heartbeat_timeout = heartbeat_timeout
        self.last_heartbeat = {neighbor: env.now for neighbor in self.neighbors}
        self.hops={}
        self.adjacency_table = {self.name: (self.neighbors, self.env.now)}

    def set_propagator(self,propagator):
        self.propagator=propagator

    def push_forward(self,packet):
        if self.current_queue_length < self.queue_length:
            self.forward_queue.put(packet)
            return True
        else:
            return False
    def push_transmission(self,neighbor,packet):
        if self.current_queue_length < self.queue_length:
            self.current_queue_length += 1
            self.transmission_length[neighbor]+=1
            self.transmission_queue[neighbor].put(packet)
            return True
        else:
            return False
    def pop_transmission(self,neighbor):
        packet = yield self.transmission_queue[neighbor].get()
        self.current_queue_length-=1
        self.transmission_length[neighbor]-=1
        return packet

    def forward_packet(self):
        while self.active: # 只要卫星处于活跃状态，就持续处理数据包
            packet = yield self.forward_queue.get() # 从转发队列中获取数据包（阻塞操作，直到有数据包到达）
            packet.hops += 1
            source, destination = packet.source, packet.destination # 获取数据包的源节点和目的节点
            yield self.env.timeout(self.processing_time) # 模拟处理数据包的时间延迟（如解析包头、查询路由表等耗时）
            if not self.active:  # 如果卫星已不活跃，记录丢包并退出循环
                if 'Lost_relay' in self.statics_data:
                    self.statics_data['Lost_relay'] += 1
                self.logger.log(f"Time {self.env.now:.3f}: {self.name} is missed, dropped 1 packet")
                break
            if destination != self.name:# 若目的节点不是当前卫星，需要转发数据包
                if destination in self.routing_tables:# 检查路由_tables中是否有目的节点的路由信息
                    next_hop = random.choice(self.routing_tables[destination][0])# 从路由表中随机选择一个下一跳（可能有多个等价路径）
                    if next_hop in self.neighbors: # 若下一跳是当前卫星的邻居节点，尝试放入传输队列
                        success =self.push_transmission(next_hop, packet)
                        if not success:
                            self.logger.log(f"Time {self.env.now:.3f}: {packet} is blocked because of congestion.")
                    else: # 下一跳不是邻居（可能已离线），记录丢包
                        if 'Lost_relay' in self.statics_data:
                            self.statics_data['Lost_relay'] += 1
                        self.logger.log(f"Time {self.env.now:.3f}: {next_hop} is missed, dropped 1 packet")
                else:# 目的节点不在路由表中（无可达路径），记录丢包
                    if 'Lost_relay' in self.statics_data:
                        self.statics_data['Lost_relay'] += 1
                    self.logger.log(f"Time {self.env.now:.3f}: {destination} is missed, dropped 1 packet")
            else:# 若目的节点是当前卫星，确认数据包到达
                if 'Reached' in self.statics_data: # 更新统计信息：到达数、总跳数、总延迟
                    self.statics_data['Reached'] += 1
                if 'Total_hops' in self.statics_data:
                    self.statics_data['Total_hops'] += packet.hops
                if 'Total_delay' in self.statics_data:
                    self.statics_data['Total_delay'] += self.env.now - packet.creation_time
                self.logger.log(f"Time {self.env.now:.3f}: Packet {(source, destination)} reached its destination {self.name}.") # 记录日志：数据包到达目的地

    def transmit_packet(self,neighbor):
        while self.active:
            packet = yield self.env.process(self.pop_transmission(neighbor)) #循环通过pop_transmission从邻居的传输队列中取包；
            yield self.env.timeout(packet.size / self.transmission_rate) #按数据包大小和传输速率计算传输时间（yield self.env.timeout(packet.size / transmission_rate)）；
            if neighbor not in self.neighbors or not self.active:
                if 'Lost_relay' in self.statics_data:
                    self.statics_data['Lost_relay'] += 1
                self.logger.log(f"Time {self.env.now:.3f}: transmission stopped, dropped 1 packet")
                break
            self.logger.log(f"Time {self.env.now:.3f}: {self.name}: Packet {(packet.source,packet.destination)} departed. Transmission length: {self.current_queue_length}",detail=True)
            self.env.process(self.propagator.propagate(self.name,neighbor, packet)) #调用传播器的propagate方法（self.propagator.propagate），将数据包发送到下一跳邻居。


    def update_adjacency_dict_for_bfs(self):
        new_dict = self.adjacency_table.copy()
        for node, (neighbors, _) in self.adjacency_table.items():
            for neighbor in neighbors:
                if node not in self.adjacency_table[neighbor][0]:
                    if self.adjacency_table[node][1] > self.adjacency_table[neighbor][1]:
                        new_dict[neighbor] = (new_dict[neighbor][0] + [node], new_dict[neighbor][1])
                    else:
                        new_dict[node][0].remove(neighbor)
        return new_dict

    def build_routing_table(self):
        result_dict = {}
        queue = [(neighbor, [self.name, neighbor], 1) for neighbor in self.adjacency_table[self.name][0]]
        while queue:
            (node, path, hops) = queue.pop(0)
            if node not in result_dict:
                result_dict[node] = ([path[1]], hops)
                queue.extend((neighbor, path + [neighbor], hops + 1) for neighbor in self.adjacency_table[node][0] if
                             neighbor not in path)
            elif result_dict[node][1] == hops:
                result_dict[node][0].append(path[1])
        self.routing_tables = result_dict

    def add_neighbor(self,neighbor):
        if neighbor not in self.neighbors:
            self.neighbors.append(neighbor)
            self.transmission_queue[neighbor] = simpy.Store(self.env)
            self.transmission_length[neighbor] = 0
            self.neighbor_states[neighbor] = 0
            self.last_heartbeat[neighbor] = self.env.now
            self.adjacency_table [self.name]=(self.neighbors, self.env.now)
            self.adjacency_table_exchanger()
            self.env.process(self.monitor_single_neighbor(neighbor))
            self.env.process(self.transmit_packet(neighbor))

    def del_neighbor(self,neighbor):
        if self.active:
            if neighbor in self.neighbors:
                while self.transmission_queue[neighbor].items:
                    packet = yield self.env.process(self.pop_transmission(neighbor))
                    success= self.push_forward(packet)
                    if not success:
                        if 'Lost_relay' in self.statics_data:
                            self.statics_data['Lost_relay'] += 1
                    self.logger.log(f"Time {self.env.now:.3f}: {packet} is dropped because of satellite missing.")
                self.neighbors.remove(neighbor)
                del self.transmission_queue[neighbor]
                del self.transmission_length[neighbor]
                del self.neighbor_states[neighbor]
                self.adjacency_table[self.name]=(self.neighbors, self.env.now)
                self.update_adjacency_dict_for_bfs()
                self.build_routing_table()
                self.adjacency_table_exchanger()
                return True
            else:
                return False
        else:
            return False
    def state_exchanger(self):
        while self.active:
            yield self.env.timeout(self.state_update_period)
            if not self.active:
                break
            for neighbor in self.neighbors:
                self.env.process(self.propagator.send_state(self.name, neighbor,self.current_queue_length))
    """状态同步：让邻居实时了解当前卫星的队列负载（是否拥堵），避免向已拥堵的卫星转发数据包，减少丢包。心跳维护：间接维持邻居关系的有效性（通过更新 last_heartbeat，配合 monitor_single_neighbor 方法检测邻居是否失效）"""
    def adjacency_table_exchanger(self):
        for neighbor in self.neighbors:
            self.env.process(self.propagator.send_adjacency_table(self.name,neighbor,self.adjacency_table))

    def update_adjacency_dict(self, new_dict):
        updated = False
        for key, value in new_dict.items():
            if key not in self.adjacency_table:
                self.adjacency_table[key] = value
                updated = True
            else:
                _, old_time = self.adjacency_table[key]
                _, new_time = value
                if new_time > old_time:
                    self.adjacency_table[key] = value
                    updated = True
        return updated

    def update_adjacency_table(self, table):
        if self.update_adjacency_dict(table):
            self.update_adjacency_dict_for_bfs()
            self.build_routing_table()
            self.adjacency_table_exchanger()

    def monitor_single_neighbor(self, neighbor):
        while self.active:
            timeout_duration = self.heartbeat_timeout - (self.env.now - self.last_heartbeat[neighbor])
            if timeout_duration <= 0.01:
                yield self.env.process(self.del_neighbor(neighbor))
                break
            else:
                yield self.env.timeout(timeout_duration)

    def self_missing(self):
        self.active = False
        for neighbor in self.neighbors:
            while self.transmission_queue[neighbor].items:
                packet = yield self.env.process(self.pop_transmission(neighbor))
                if 'Lost_relay' in self.statics_data:
                    self.statics_data['Lost_relay'] += 1
                self.logger.log(f"Time {self.env.now:.3f}: {packet} is dropped because of satellite missing.")
            while self.forward_queue.items:
                packet = yield self.forward_queue.get()
                if 'Lost_relay' in self.statics_data:
                    self.statics_data['Lost_relay'] += 1
                self.logger.log(f"Time {self.env.now:.3f}: {packet} is dropped because of satellite missing.")

    def all_start(self):
        self.env.process(self.forward_packet())#将到达当前卫星的数据包放入转发队列（forward_queue），等待后续处理。是数据包进入卫星后的第一个处理环节，负责接收数据包并暂存到转发队列。
        for neighbor in self.neighbors:
            self.env.process(self.transmit_packet(neighbor)) #处理对指定邻居的数据包传输过程，包括传输延迟计算和调用传播器发送。
            self.env.process(self.monitor_single_neighbor(neighbor)) #启动一个仿真进程，用于持续监控当前卫星与指定邻居卫星的连接状态

class SatelliteNetworkSimulator:
    def __init__(self, graph,landmarks,mean_interarrival_time,queue_length,transmission_rate,packet_size,state_update_period,logger):
        self.env = simpy.Environment()
        self.graph =graph
        self.queue_length=queue_length
        self.logger=logger
        self.transmission_rate=transmission_rate
        self.state_update_period=state_update_period
        self.statics_data = {'Total': 0, 'Reached': 0, 'Lost_upload': 0, 'Lost_relay': 0, 'Total_delay': 0, 'Total_hops': 0}
        self.satellite_names=[node for node in self.graph.nodes]
        self.satellites={node : Satellite(self.env,node,list(self.graph.neighbors(node)),queue_length,transmission_rate,state_update_period,logger,self.statics_data) for node in self.graph.nodes}
        self.propagator = Propagator(self.env, graph, logger, self.satellites,self.statics_data)
        self.landmarks=landmarks
        self.mean_interarrival_time=mean_interarrival_time
        self.size = packet_size
        for satellite in self.satellites:
            self.satellites[satellite].adjacency_table=self.extract_adjacency_dict()
            self.satellites[satellite].set_propagator(self.propagator)
            self.satellites[satellite].build_routing_table()

    def extract_adjacency_dict(self):
        adjacency_dict = {}
        for node in self.satellite_names:
            neighbors = list(self.graph.neighbors(node))
            adjacency_dict[node] = (neighbors, self.env.now)
        return adjacency_dict

    def generate_traffic(self, landmark):
        def has_common_elements(list1, list2):
            set1 = set(list1)
            set2 = set(list2)
            common_elements = set1.intersection(set2)
            return len(common_elements) > 0

        while landmark in self.landmarks:
            interarrival_time = random.expovariate(1.0 / self.mean_interarrival_time)
            yield self.env.timeout(interarrival_time)
            if not landmark in self.landmarks:
                break
            if self.landmarks[landmark]:
                sources = self.landmarks[landmark]
            else:
                self.logger.log(f"Time {self.env.now:.3f}: {landmark} has no connections, packets failed to generate.")
                continue
            destination_landmark = landmark
            while destination_landmark == landmark:
                temp_landmark = random.choice(list(self.landmarks))
                temp_destinations = self.landmarks[temp_landmark]
                if self.landmarks[temp_landmark] and not has_common_elements(sources, temp_destinations):
                    destination_landmark = temp_landmark
                    destinations = temp_destinations
            min_hops=np.inf
            min_hops_pairs=[]
            for source in sources:
                for destination in destinations:
                    if destination in self.satellites[source].routing_tables:
                        hops= self.satellites[source].routing_tables[destination][1]
                    else:
                        hops = np.inf
                    if hops < min_hops:
                        min_hops = hops
                        min_hops_pairs = [(source,destination)]
                    elif hops == min_hops:
                        min_hops_pairs.append((source,destination))
            if min_hops_pairs:
                source,destination=random.choice(min_hops_pairs)
            else:
                self.logger.log(f"Time {self.env.now:.3f}: connection between {(landmark,destination_landmark)} is missed, packet failed to generate.")
                continue
            packet = Packet(source,destination,self.env.now,self.size)
            if 'Total' in self.statics_data:
                self.statics_data['Total'] += 1
            self.logger.log(f"Time {self.env.now:.3f}: {source}: Packet generated: {(source,destination)}.")
            if source in self.satellite_names:
                success = self.satellites[source].push_forward(packet)
                if success:
                    self.logger.log(f"Time {self.env.now:.3f}: {source}: Packet {(packet.source, packet.destination)} received by router. Transmission length: {self.satellites[source].current_queue_length}.",detail=True)
                else:
                    if 'Lost_upload' in self.statics_data:
                        self.statics_data['Lost_upload'] += 1
                    self.logger.log(f"Time {self.env.now:.3f}: {source}: Routing queue is full, discarding packet {(packet.source, packet.destination)}.")
            else:
                self.logger.log(f"Time {self.env.now:.3f}: {source} is missed, packet failed to generate.")
                continue

    def get_system_state(self):
        total_queue_usage = {}
        for node in self.satellite_names:
            total_usage = self.satellites[node].current_queue_length
            average_usage = total_usage / self.queue_length
            total_queue_usage[node] = average_usage
        return total_queue_usage
    def upgrade_all(self,graph,landmarks):
        old_landmarks=set(self.landmarks.keys())
        new_landmarks=set(landmarks.keys())
        self.landmarks=landmarks
        new_nodes = set(graph.nodes())
        old_nodes = set(self.graph.nodes())
        new_edges = set(graph.edges())
        old_edges = set(self.graph.edges())
        self.satellite_names=[node for node in graph]
        self.graph=graph
        self.propagator.update(graph)
        for node in new_nodes - old_nodes:
            self.satellites[node]=Satellite(self.env, node, list(self.graph.neighbors(node)), self.queue_length, self.transmission_rate,self.state_update_period, self.logger, self.statics_data)
            self.satellites[node].set_propagator(self.propagator)
            self.satellites[node].all_start()
            self.satellites[node].adjacency_table_exchanger()
        for node in old_nodes - new_nodes:
            self.env.process(self.satellites[node].self_missing())
            del self.satellites[node]
        for edge in new_edges - old_edges:
            node, neighbor = edge
            self.satellites[node].add_neighbor(neighbor)
            self.satellites[neighbor].add_neighbor(node)
        for landmark in new_landmarks-old_landmarks:
            self.env.process(self.generate_traffic(landmark))

    def clear_statics(self):
        for statics in self.statics_data:
            self.statics_data[statics]=0

    def run(self, duration):
        if self.env.now==0:
            for landmark in self.landmarks:
                self.env.process(self.generate_traffic(landmark))
            for satellite in self.satellites:
                self.satellites[satellite].all_start()
        self.env.run(until=self.env.now+duration)
        #print(self.statics_data)










