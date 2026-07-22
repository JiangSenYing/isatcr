from SatelliteNetworkSimulator_Beta import SatelliteNetworkSimulator,Packet,Satellite,Propagator
import os
import random
import time
import numpy as np
import simpy
import networkx as nx
import torch
from gnn_modules import pyg_compat  # 用于图结构观测（替代 DGL）
# from PRC import config
from torch_geometric.nn import GATConv 
import torch.nn.functional as F
from SatelliteDomainPartitioner import SatelliteDomainPartitioner
from common import isLeo,z_score
import math
import copy

PRE=True
CT_FAC=5

class Reward_Function:
    def __init__(self,reach_factor,delay_factor,loss_factor,memory_threshold,memory_factor):
        self.reach_factor=reach_factor#数据包成功到达目的地时的基础奖励权重。
        self.delay_factor=delay_factor#延迟对奖励的惩罚权重（延迟越大，惩罚越大）。
        self.loss_factor=loss_factor#数据包丢失时的惩罚权重。
        self.memory_threshold=memory_threshold#内存剩余量的阈值（用于判断内存是否紧张）
        self.memory_factor=memory_factor#内存剩余量低于阈值时的惩罚权重。

    def reach_reward(self,delay):#计算任务成功到达且被计算
        return self.reach_factor-self.delay_factor*delay#奖励 = 基础到达奖励 - 延迟惩罚

    def reach_reward_abnormal(self,delay):#计算任务成功到达，但处于异常场景？未被计算？
        return -self.delay_factor*delay#奖励 = - 延迟惩罚（获得负奖励）（没有基础奖励）

    def normal_reward(self,delay,memory_remain):
        return -self.delay_factor*delay-self.memory_factor*(memory_remain<self.memory_threshold)
    """
    常规的中间步奖惩（与延迟和内存不足惩罚相关）
    同时考虑延迟和内存状态:若剩余内存(memory_remain)低于阈值(memory_threshold),会额外叠加内存惩罚。
    """
    def loss_reward(self,delay):#计算数据包丢失时的奖励。
        return -self.loss_factor-self.delay_factor*delay#奖励 = - 丢失惩罚 - 延迟惩罚
"""
在卫星网络仿真中负责增强数据包的传播逻辑,尤其集成了强化学习(RL)的经验收集、奖励计算和任务计算相关的处理。
它在原有传播功能基础上，增加了对计算任务的跟踪、强化学习经验的记录和奖励函数的应用，适用于带计算任务的卫星网络场景。
继承 Propagator 类的核心功能（如传播延迟计算、节点间数据传输），并扩展了以下能力：

记录强化学习所需的经验数据（状态、动作、奖励等）。
结合奖励函数计算不同场景下的奖励值。
处理带计算任务的数据包（如计算需求、计算后大小等）的传播和下行逻辑。
区分不同类型任务的统计数据（如成功 / 丢失的数据包类型）。
"""    

class Propagator_Computing(Propagator):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.experiences = []  # 标准经验: [state, mark, action, reward, next_state, done]
        self.experiences_2hop = []  # 2跳经验: [state, hop2_info, mark, action, reward, next_state, next_hop2_info, done]
        self.experiences_graph = []  # 图格式经验: [obs_dict, mark, action, reward, next_obs_dict, done]
        self.final_rewards = []  # 存储最终奖励值，用于评估决策效果
        self.domain_entry_rewards = []  # 存储非终止域入口阶段奖励
        self.n_hops = 1  # 默认1跳模式
        self.obs_type = 'flat'  # 观测类型: 'flat', 'relational', 'graph'
        self.obs_wrapper = None  # 观测包装器实例


    def trans_parameters(self, max_hop, downstream_delays, reward_function, n_hops=1, 
                         obs_type='flat', obs_wrapper=None):
        """
        传递配置参数，为传播和奖励计算提供必要的参数。
        
        Args:
            max_hop: 限制数据包传播的最大跳数
            downstream_delays: 下行延迟（数据包从卫星到地面站的延迟）
            reward_function: 奖励函数实例
            n_hops: GNN邻居跳数配置
            obs_type: 观测类型 ('flat', 'relational', 'graph')
            obs_wrapper: 观测包装器实例（用于图格式）
        """
        self.max_hop = max_hop
        self.downstream_delays = downstream_delays
        self.reward_function = reward_function
        self.n_hops = n_hops
        self.obs_type = obs_type
        self.obs_wrapper = obs_wrapper

    def reset_parameters(self):
        """重置经验和奖励数据，通常用于每轮仿真或训练开始前清空历史数据。"""
        self.experiences = []
        self.experiences_2hop = []
        self.experiences_graph = []
        self.final_rewards = []
        self.domain_entry_rewards = []
    
    def add_experience(self, last_obs, mark, action, reward, current_obs, done,
                       last_task_context=None, current_task_context=None):
        """
        添加经验到缓冲区（统一接口）
        
        Args:
            last_obs: 上一个观测（根据 obs_type 可以是扁平状态向量或 PyG 图）
            mark: 标记（通常是 is_computed）
            action: 动作
            reward: 奖励
            current_obs: 当前观测（根据 obs_type 可以是扁平状态向量或 PyG 图）
            done: 是否结束
            last_task_context: 上一个任务上下文（仅分离模式使用）
            current_task_context: 当前任务上下文（仅分离模式使用）
        """
        # 根据观测类型保存不同格式的经验
        if self.obs_type == 'flat':
            # 标准扁平格式经验
            self.experiences.append([last_obs, mark, action, reward, current_obs, done])
            
        elif self.obs_type in ('relational_separated', 'graph_separated'):
            # 【分离模式】图格式经验 - 包含额外的 task_context
            # 经验格式: [obs, mark, action, reward, next_obs, done, task_context, next_task_context]
            if last_obs is not None and current_obs is not None:
                self.experiences_graph.append([
                    last_obs, mark, action, reward, current_obs, done,
                    last_task_context, current_task_context
                ])
            else:
                print(f"Warning: Observation is None, skipping experience.")
            
        elif self.obs_type in ('relational', 'graph'):
            # 普通图格式经验 - 不包含 task_context
            if last_obs is not None and current_obs is not None:
                self.experiences_graph.append([
                    last_obs, mark, action, reward, current_obs, done
                ])
            else:
                print(f"Warning: Observation is None, skipping experience.")

    def get_experiences(self):
        """
        获取当前观测类型对应的经验列表
        
        Returns:
            经验列表（根据 obs_type 返回不同格式）
            - flat: [last_obs, mark, action, reward, current_obs, done]
            - relational/graph: [last_obs, mark, action, reward, current_obs, done]
            - relational_separated/graph_separated: [last_obs, mark, action, reward, current_obs, done, 
                                                     last_task_context, current_task_context]
        """
        if self.obs_type in ('relational', 'graph', 'relational_separated', 'graph_separated') and self.experiences_graph:
            return self.experiences_graph
        else:
            return self.experiences

    def finish_meo_decision(self, packet, reward, done=True, meo_result=None):
        """Forward terminal packet rewards to the optional MEO domain router."""
        if packet is not None:
            packet.meo_terminal_time = float(getattr(self.env, "now", 0.0))
            if meo_result is not None:
                packet.meo_result = meo_result
        transformer = getattr(self, 'transformer_module', None)
        if transformer is not None and hasattr(transformer, 'finish_meo_decision'):
            transformer.finish_meo_decision(packet, reward, done=done, meo_result=meo_result)
        elif transformer is not None and getattr(transformer, 'meo_router', None) is not None:
            transformer.meo_router.finish_decision(packet, reward, done=done, meo_result=meo_result)
    
    def is_separated_mode(self):
        """检查是否为分离模式"""
        return self.obs_type in ('relational_separated', 'graph_separated')

    @staticmethod
    def leo_domain_delay(packet, current_time):
        """Return elapsed time in the packet's current LEO routing domain."""
        entry_time = getattr(packet, 'leo_domain_entry_time', None)
        if entry_time is None:
            entry_time = packet.creation_time
        return max(0.0, float(current_time) - float(entry_time))

    @staticmethod
    def completed_leo_domain_delay(packet, current_time):
        """Return elapsed time for the LEO domain segment just completed."""
        end_time = getattr(packet, 'meo_segment_time', None)
        if end_time is None:
            end_time = current_time
        start_time = getattr(packet, 'leo_previous_domain_entry_time', None)
        if start_time is None:
            start_time = packet.creation_time
        return max(0.0, float(end_time) - float(start_time))

    def leo_domain_entry_reward(self, is_computed, segment_delay):
        """Reward an MEO-directed domain entry using the packet compute state."""
        if is_computed:
            return self.reward_function.reach_reward(segment_delay)
        return self.reward_function.reach_reward_abnormal(segment_delay)

    @staticmethod
    def leo_domain_entry_done(current_node, destination):
        """Only a domain entry that is also the final LEO destination terminates."""
        return int(current_node == destination)

    def record_domain_entry_failure(self, packet, reward, current_node=None):
        """Include an in-transit failure in domain-entry monitoring.

        Failures after the packet has entered its destination MEO domain belong
        only to the terminal/ending statistic.  Before that point, the failure
        is also the outcome of the current cross-domain stage.
        """
        destination = getattr(packet, 'destination', None)
        if current_node is not None and current_node == destination:
            return False

        current_satellite = self.satellites.get(current_node)
        destination_satellite = self.satellites.get(destination)
        current_domain = getattr(current_satellite, 'masterMeo', None)
        destination_domain = getattr(destination_satellite, 'masterMeo', None)
        if (
            current_domain is not None
            and destination_domain is not None
            and current_domain == destination_domain
        ):
            return False

        self.domain_entry_rewards.append(reward)
        return True

    def _mark_cross_domain_entry(self, node, next_hop, packet):
        """Record exact domain entry after a successful cross-domain receive."""
        source_satellite = self.satellites.get(node)
        target_satellite = self.satellites.get(next_hop)
        if source_satellite is None or target_satellite is None:
            return
        source_meo = getattr(source_satellite, 'masterMeo', None)
        target_meo = getattr(target_satellite, 'masterMeo', None)
        if source_meo is None or target_meo is None or source_meo == target_meo:
            return
        previous_entry_time = getattr(packet, 'leo_domain_entry_time', None)
        if previous_entry_time is None:
            previous_entry_time = packet.creation_time
        entry_time = float(self.env.now)
        packet.leo_previous_domain_entry_time = float(previous_entry_time)
        packet.leo_domain_entry_time = entry_time
        packet.meo_segment_time = entry_time

    def propagate(self,node,next_hop,packet,toMeo = False):
        """重写父类 propagate 方法，处理带计算任务的数据包在卫星节点间的传播，同时记录强化学习经验和奖励。"""
        # 解包 information
        # 标准格式(7字段): [is_computed, type, computing_demand, size_after_computing, last_time, last_obs, last_action]
        # 分离模式(8字段): [is_computed, type, computing_demand, size_after_computing, last_time, last_obs, last_action, last_task_context]
        if toMeo == False:
            info = packet.information
            is_computed, type, computing_demand, size_after_computing, last_time, last_obs, last_action = info[:7]
            last_task_context = info[7] if len(info) > 7 else None  # 分离模式额外的任务上下文
            """
            information是额外字段,在本文项目中，将计算相关信息添加到其中
            packet.extra_information([is_computed, type, computing_demand, size_after_computing, self.env.now, current_obs, action])
            packet.information 解包：这些字段由上游路由/选择逻辑写入，含是否已计算、业务类型、计算需求、计算后大小、上次记录时间、上次观测（根据obs_type为扁平状态或DGL图）与上次动作（用于经验）
            分离模式下，last_task_context 是第8字段
            """
            if (node, next_hop) in self.propagation_delays:
                yield self.env.timeout(self.propagation_delays[(node, next_hop)])#计算节点间传播延迟
            #     self.propagation_delays 存储节点对 (node, next_hop) 的传播延迟（单位：时间）。
            #    通过 yield self.env.timeout(...) 模拟数据包在节点间传播的耗时（基于 Simpy 仿真环境）。

            #处理下一跳节点有效（存在于网络中）的情况
                if next_hop in self.node_names:
                    success = self.satellites[next_hop].push_forward(packet)#尝试将数据包加入下一跳节点的转发队列
                    if success:
                        packet.meo_previous_hop = node
                        self._mark_cross_domain_entry(node, next_hop, packet)
                        self.logger.log(f"Time {self.env.now:.3f}: {next_hop}: Packet {(packet.source,packet.destination,packet.creation_time)} received by router. Memory remain: {self.satellites[next_hop].current_memory_occupy}.",detail=True)
                    else:
                        source, destination, hops, creation_time, size = packet.source, packet.destination, packet.hops, packet.creation_time, packet.size
                        mission_state = [type, size / self.satellites[node].max_size, computing_demand / self.satellites[node].computing_ability, size_after_computing / self.satellites[node].max_size]
                        
                        # 根据 obs_type 获取对应格式的观测
                        # 分离模式下 current_hop2_info 实际是 current_task_context
                        current_obs, current_hop2_info = self.satellites[node]._get_state_for_decision(
                            destination, hops, is_computed, mission_state
                        )
                        # 分离模式：current_hop2_info 是 task_context
                        current_task_context = current_hop2_info if self.is_separated_mode() else None
                        """
                        #hops:数据包已经过的跳数。数据包的目标和传输进度(destination、hops)。
                        #is_computed:布尔值，标识数据包是否已完成计算任务。
                        #归一化的数据包属性列表（通过列表传入）：数据包的任务类型，存储占用率：数据包大小与节点最大存储容量的比值，计算资源占用率：计算需求与节点计算能力的比值，计算后的数据存储需求：计算完成后的数据大小与节点最大存储容量的比值

                            current_obs 根据 obs_type 不同：
                            - flat: 扁平状态向量（整合了节点状态和数据包属性）
                            - graph: DGL 异构图
                            - graph_separated: PyG 图 + task_context

                            数据包的目标和传输进度(destination、hops)。
                            计算任务的完成状态(is_computed)。
                            任务类型和资源需求(type、存储 / 计算资源的归一化比值）。

                            是决策的依据：
                            若存储占用率过高（接近 1),可能导致后续数据包丢失，智能体需调整路由或计算策略。
                            计算需求占比过高，可能提示节点计算资源紧张，需考虑任务卸载。
                        """
                        done = 1
                        # 计算损失奖励（因队列满丢失）
                        reward = self.reward_function.loss_reward(self.leo_domain_delay(packet, self.env.now))
                        # 记录强化学习经验（使用简化的 add_experience 方法）
                        # mark 使用 is_computed 标记
                        if last_obs is not None:
                            self.add_experience(
                                last_obs, is_computed, last_action, 
                                reward, current_obs, done,
                                last_task_context, current_task_context
                            )
                        self.finish_meo_decision(packet, reward, done=True, meo_result="memory_drop")
                        self.final_rewards.append(reward)
                        self.record_domain_entry_failure(packet, reward, current_node=node)
                        # 若数据包已计算，释放计算资源
                        if packet.computing_node and PRE:
                            self.graph.nodes[packet.computing_node]['computing_remain'] -= computing_demand
                        """当数据包因某种原因（如路由队列满、节点丢失等）被丢弃或处理完成时，释放其之前在计算节点上预留的计算资源（减少 computing_remain,即剩余计算能力），避免资源浪费。"""
                        # 更新统计数据（区分任务类型）
                        if type == 0:
                            self.statics_data['Lost_relay_0'] += 1
                        else:
                            self.statics_data['Lost_relay_1'] += 1
                
                        self.logger.log(f"Time {self.env.now:.3f}: {next_hop}: Routing queue is full, discarding packet {(packet.source,packet.destination,packet.creation_time)}.")
            #处理下一跳节点无效（不存在于网络中）的情况
                #下一跳节点不存在时，直接标记数据包丢失，更新对应类型的统计，并释放计算资源（若有）
                else:
                    reward = self.reward_function.loss_reward(self.leo_domain_delay(packet, self.env.now))
                    self.record_domain_entry_failure(packet, reward, current_node=node)
                    if last_obs is not None:
                        self.add_experience(
                            last_obs, is_computed, last_action,
                            reward, last_obs, 1,
                            last_task_context, last_task_context
                        )
                        self.final_rewards.append(reward)
                    self.finish_meo_decision(packet, reward, done=True, meo_result="link_drop")
                    # 释放计算资源（若已计算）
                    if packet.computing_node and PRE:
                        self.graph.nodes[packet.computing_node]['computing_remain'] -= computing_demand
                    if type==0:
                        self.statics_data['Lost_relay_0'] += 1
                    else:
                        self.statics_data['Lost_relay_1'] += 1
                    self.logger.log(f"Time {self.env.now:.3f}: {next_hop} is missed, dropped 1 packet.")
            #处理节点间无传播延迟（连接不存在）的情况
            #若 (node, next_hop) 不在 propagation_delays 中，说明两节点无有效连接，数据包丢失，处理逻辑与 “节点无效” 类似。
            else:
                reward = self.reward_function.loss_reward(self.leo_domain_delay(packet, self.env.now))
                self.record_domain_entry_failure(packet, reward, current_node=node)
                if last_obs is not None:
                    self.add_experience(
                        last_obs, is_computed, last_action,
                        reward, last_obs, 1,
                        last_task_context, last_task_context
                    )
                    self.final_rewards.append(reward)
                self.finish_meo_decision(packet, reward, done=True, meo_result="link_drop")
                if packet.computing_node and PRE:
                    self.graph.nodes[packet.computing_node]['computing_remain'] -= computing_demand
                if type == 0:
                    self.statics_data['Lost_relay_0'] += 1
                else:
                    self.statics_data['Lost_relay_1'] += 1
                self.logger.log(f"Time {self.env.now:.3f}: connection {(node, next_hop)} is missed, dropped 1 packet")
        else: #该分支是LEO发送给MEO的分支
            distance = self._distance(node, next_hop)
            propagation_delay = distance / self.propagation_speed # 延迟=距离/速度
            yield self.env.timeout(propagation_delay)
            if next_hop in self.node_names and next_hop in self.satellites:
                master_meo = self.satellites[next_hop]
                if master_meo.active:
                    master_meo.receive_leo_state(node, packet.information)
                    self.logger.log(
                        f"Time {self.env.now:.3f}: {next_hop} received LEO state from {node}.",
                        detail=True
                    )
                else:
                    self.logger.log(f"Time {self.env.now:.3f}: {next_hop} is inactive, LEO state update failed.")
            else:
                self.logger.log(f"Time {self.env.now:.3f}: {next_hop} is missed, LEO state update failed.")

    """#处理数据包到达目标节点（或下行业务）的逻辑，主要用于完成数据包的最终交付流程，同时记录强化学习经验和更新统计数据
    # 核心作用
    # 处理数据包到达目标节点后的收尾工作（如计算延迟、奖励）。
    # 区分数据包是否完成计算任务，更新对应统计数据。
    # 记录强化学习所需的经验（状态、动作、奖励），用于后续模型训练。"""
    def downstream(self,node,packet):
        source, destination, hops, creation_time,size = packet.source, packet.destination, packet.hops, packet.creation_time,packet.size
        # 解包 information，分离模式下有8个字段（第8个是 last_task_context）
        info = packet.information
        is_computed, type, computing_demand, size_after_computing, last_time, last_obs, last_action = info[:7]
        last_task_context = info[7] if len(info) > 7 else None  # 分离模式下的 task_context
        mission_state = [type, size/self.satellites[node].max_size, computing_demand/self.satellites[node].computing_ability, size_after_computing/self.satellites[node].max_size]
        
        # 根据 obs_type 获取当前观测（扁平状态或图结构）
        # 分离模式下返回 3 个值：obs, task_context, task_edge_feats
        result = self.satellites[node]._get_state_for_decision(
            destination, hops, is_computed, mission_state
        )
        # 分离模式下解包 3 个值
        if self.is_separated_mode():
            current_obs, current_task_context, current_task_edge_feats = result
        else:
            current_obs, current_task_context = result
            current_task_edge_feats = None
        done = 1# 标记任务结束（无论成功与否，当前数据包的流程已结束），强化学习的一个 “episode” 完成
        #处理目标节点有效（存在于卫星网络中）的情况
        if node in self.satellites:
            yield self.env.timeout(self.downstream_delays)
            domain_delay = self.leo_domain_delay(packet, self.env.now)
            if is_computed:
                # 数据包已完成计算，使用正常到达奖励
                reward = self.reward_function.reach_reward(domain_delay)
                if type==0:
                    self.statics_data['Reached_after_computed_0'] += 1
                else:
                    self.statics_data['Reached_after_computed_1'] += 1
            else:
                # 数据包未完成计算却到达目标，使用异常到达奖励
                reward = self.reward_function.reach_reward_abnormal(domain_delay)
            if type == 0:
                self.statics_data['Reached_0'] += 1
            else:
                self.statics_data['Reached_1'] += 1
            if type == 0:
                self.statics_data['Total_hops_0'] += packet.hops
            else:
                self.statics_data['Total_hops_1'] += packet.hops
            if type == 0:
                self.statics_data['Total_delay_0'] += self.env.now - packet.creation_time
            else:
                # 累计计算等待时间（所有类型通用）
                self.statics_data['Total_delay_1'] += self.env.now - packet.creation_time
            self.statics_data['Computing_waiting_time'] += packet.computing_waiting_time
            self.logger.log( f"Time {self.env.now:.3f}: Packet {(source, destination, packet.creation_time)} reached its destination {node}.")
        #处理目标节点无效（不存在于卫星网络中）的情况
        # 若目标节点 node 无效（不在卫星网络中），数据包被丢弃，奖励为 loss_reward（-loss_factor - delay_factor * 延迟，惩罚丢失和延迟）。
        # 若数据包已在某个节点完成计算（packet.computing_node 存在），则释放该节点的计算资源（computing_remain 减少）。
        else:
            reward = self.reward_function.loss_reward(self.leo_domain_delay(packet, self.env.now))
            self.record_domain_entry_failure(packet, reward, current_node=node)
            if packet.computing_node and PRE:
                self.graph.nodes[packet.computing_node]['computing_remain'] -= computing_demand
            if type == 0:
                self.statics_data['Lost_relay_0'] += 1
            else:
                self.statics_data['Lost_relay_1'] += 1
            self.logger.log(f"Time {self.env.now:.3f}: downlink of {node} is missed, dropped 1 packet")
        meo_result = "reached_computed" if node in self.satellites and is_computed else "reached_uncomputed" if node in self.satellites else "link_drop"
        self.finish_meo_decision(packet, reward, done=True, meo_result=meo_result)
        if last_obs is not None:
            self.final_rewards.append(reward)
            self.add_experience(
                last_obs, is_computed, last_action,
                reward, current_obs, done,
                last_task_context, current_task_context
            )
        else:
            pass

    # ================================================================================
    # 隐藏状态交换方法（测试模式使用）
    # ================================================================================
    
    # def send_raw_feat(self, node, neighbor, raw_feat):
    #     """
    #     发送原始特征（第1轮交换）
        
    #     模拟传播延迟后，将原始特征传递给目标邻居节点
        
    #     Args:
    #         node: 发送节点名称
    #         neighbor: 目标邻居节点名称
    #         raw_feat: 原始特征（列表或张量）
    #     """
    #     if (node, neighbor) in self.propagation_delays:
    #         yield self.env.timeout(self.propagation_delays[(node, neighbor)])
    #         if neighbor in self.node_names:
    #             if node in self.satellites[neighbor].neighbors:
    #                 self.satellites[neighbor].receive_raw_feat(node, raw_feat)
    #                 self.logger.log(f"Time {self.env.now:.3f}: {neighbor} received raw_feat from {node}", detail=True)
    #             else:
    #                 self.satellites[neighbor].add_neighbor(node)
    #                 self.satellites[neighbor].receive_raw_feat(node, raw_feat)
    #         else:
    #             self.logger.log(f"Time {self.env.now:.3f}: {neighbor} is missing, raw_feat send failed.")
    #     else:
    #         self.logger.log(f"Time {self.env.now:.3f}: Connection ({node}, {neighbor}) is missing, raw_feat send failed.")
    
    def send_hidden(self, node, neighbor, hidden):
        """
        发送隐藏状态（第2轮交换）
        
        模拟传播延迟后，将聚合后的隐藏状态传递给目标邻居节点
        
        Args:
            node: 发送节点名称
            neighbor: 目标邻居节点名称
            hidden: 聚合后的隐藏状态（张量）
        """
        if (node, neighbor) in self.propagation_delays:
            yield self.env.timeout(self.propagation_delays[(node, neighbor)])
            if neighbor in self.node_names:
                if node in self.satellites[neighbor].neighbors:
                    self.satellites[neighbor].receive_hidden(node, hidden)
                    self.logger.log(f"Time {self.env.now:.3f}: {neighbor} received hidden from {node}", detail=True)
                else:
                    self.satellites[neighbor].add_neighbor(node)
                    self.satellites[neighbor].receive_hidden(node, hidden)
            else:
                self.logger.log(f"Time {self.env.now:.3f}: {neighbor} is missing, hidden send failed.")
        else:
            self.logger.log(f"Time {self.env.now:.3f}: Connection ({node}, {neighbor}) is missing, hidden send failed.")


class Satellite_with_Computing(Satellite):
    def __init__(self,mode,select_mode,epsilon,max_hop,max_size,device,env,name,neighbors,memory,computing_ability,transmission_rate,downlink_rate,state_update_period,is_downlink,logger,statics_data={},num=None,processing_time=1e-9,heartbeat_timeout=0.25,n_hops=1,meo_state_update_period=None,leo_action_mask_enabled=False):
        self.num=num # 卫星编号
        self.mode=mode#卫星的工作模式（如强化学习模式、传统模式）和决策选择模式。
        self.select_mode=select_mode# 选择模式（路由/计算节点选择策略）
        self.q_net=None#Q网络模型（后续通过 trans_parameters 方法传入）
        self.epsilon=epsilon# ε-贪婪策略的探索率
        self.max_hop=max_hop
        self.n_hops=n_hops  # GNN邻居跳数配置：1=仅1跳邻居，2=包含2跳邻居
        self.leo_action_mask_enabled = bool(leo_action_mask_enabled)
        self.max_size=max_size# 最大数据包大小
        self.device=device
        self.name=name # 卫星名称（格式如"sat_{轨道高度}_{轨道编号}_{卫星编号}"）
        self.neighbors= sorted(neighbors)#相邻卫星节点列表
        self.env=env
        self.orbit_altitude, self.orbit_number, self.sat_number = map(int, self.name.split('_')[1:])
        self.isLeo = True
        if self.orbit_altitude > 2000:
            self.isLeo = False
        self.inter_domain_graph = None
        self.remote_domain_leo_states = {}
        self.masterMeo = None
        self.leoStates = None
        self.intra_domain_leo = None
        self.my_leos = None
        self.transformer_module = None
        self.memory=memory# 内存容量（存储数据包）
        self.computing_ability=computing_ability# 计算能力（处理任务的速率）
        self.transmission_rate=transmission_rate # 传输速率（卫星间数据传输）
        self.downlink_rate=downlink_rate#是否为下行链路节点（标记该卫星是否可直接与地面站通信）
        self.state_update_period=state_update_period# 状态更新周期
        self.meo_state_update_period = meo_state_update_period if meo_state_update_period is not None else state_update_period
        self.meo_aggregate_exchange_period = 0.5
        self.logger=logger
        self.computing_queue = simpy.Store(self.env)#本地计算任务队列
        self.offload_queue=simpy.Store(self.env)#待卸载到其他节点的任务队列
        self.offload_size=0 # 卸载队列总大小
        self.offload_length=0# 卸载队列数据包数量
        self.transmission_queue = {neighbor: simpy.Store(self.env) for neighbor in self.neighbors}
        self.transmission_size ={neighbor: 0 for neighbor in self.neighbors}# 每个邻居的传输队列大小
        self.transmission_length ={neighbor: 0 for neighbor in self.neighbors}
        self.neighbor_hops = {neighbor: {} for neighbor in self.neighbors}# 邻居到目标的跳数映射
        self.current_queue_size=0# 当前总队列大小（任务/数据总量）
        self.current_computing_queue_size=0# 本地计算队列的任务数量
        self.forward_queue = simpy.Store(self.env)
        self.current_memory_occupy=0#当前内存占用
        self.active=True
        self.routing_tables={}
         # 初始化状态（根据模式区分状态维度）
        if 'New' in self.mode:
            self.current_state = [0,1,0,0,0,4,0,0,0,12,0,0]
            self.neighbor_states = {neighbor: [0,1,0,0,0,4,0,0,0,12,0,0] for neighbor in self.neighbors}
        else:
            self.current_state = [0,1,0,0]
            self.neighbor_states={neighbor: [0,1,0,0] for neighbor in self.neighbors}
        self.propagator=None
        self.statics_data=statics_data#统计数据字典（存储任务处理、传输的统计信息，如丢包数、延迟等）
        self.processing_time=processing_time# 处理时间（单数据包）
        self.heartbeat_timeout = heartbeat_timeout#邻居节点心跳超时阈值（超过该时间未收到心跳则认为邻居不可达）
        self.last_heartbeat = {neighbor: env.now for neighbor in self.neighbors}
        self.is_downlink=is_downlink# 是否为下行链路节点（连接地面站）
        self.is_producing=0# 是否为数据产生节点（0/1）
        self.traffic_session_start_time = None
        self.traffic_session_end_time = None
        self.traffic_session_duration = 0.0
        self.next_traffic_session_start_time = None
        if self.isLeo:
            self.communication_frequency = random.uniform(23, 32) #单位：GHz
        else:
            self.communication_frequency = random.uniform(23, 30)
        self.power = 37 #单位：dbm
        self.hops={}
        self.adjacency_table = {self.name: (self.neighbors, self.env.now)} # 邻接表（自身及邻居的连接关系，带时间戳）
        self.computing_remain=0
        self.neighbor_graph=None# 邻居网络拓扑图（用于路由计算）
        self.is_computing=False
        self.computing_time=0# 累计计算时间
        self.illegal_action_records = []
        self.last_computing_time=0# 上一次计算结束时间

        # === 隐藏状态交换相关（测试模式使用）===
        # 原始特征（低级特征）[nbr_dim]
        # self._raw_feat = None
        # 聚合后的隐藏状态（高级特征）[hidden_size]
        # self._aggregated_hidden = None
        # 邻居的原始特征缓存（第1轮交换）
        # self._nbr_raw_cache = {}  # {neighbor_name: raw_feat}
        # 邻居的隐藏状态缓存（第2轮交换）
        # self._nbr_hidden_cache = {}  # {neighbor_name: hidden}
        # 测试推理器（复用训练好的模型）
        # self._test_inference = None

    def recommend_path(
        self,
        src,
        dst,
        packet_size,
        task_type=0,
        computing_demand=0.0,
        size_after_computing=0.0,
        is_computed=False,
        excluded_domains=None,
    ):  # 在 MEO 域间图视角下推荐一条域级路径。
        """  # 开始函数说明文档。
        在 MEO 域间图视角下推荐域级路径。  # 说明该函数规划的是域级路径，不是 LEO 逐跳路径。

        返回值只描述经过哪些域，以及每次跨域使用的源/目标边界 LEO。  # 返回结果只包含域和边界入口/出口。
        Transformer 未来预测在函数入口统一执行，后续域聚合与域内风险都复用  # 说明预测只在入口执行一次。
        同一批预测图，避免同一次规划中多处重复预测造成不一致。  # 说明统一预测数据可以避免前后风险估计不一致。
        """  # 结束函数说明文档。
        transformer = getattr(self, 'transformer_module', None)
        if transformer is None and self.propagator is not None:
            transformer = getattr(self.propagator, 'transformer_module', None)
        if transformer is not None and hasattr(transformer, 'recommend_meo_path'):
            meo_plan = transformer.recommend_meo_path(
                self,
                src,
                dst,
                packet_size,
                task_type=task_type,
                computing_demand=computing_demand,
                size_after_computing=size_after_computing,
                is_computed=is_computed,
                excluded_domains=excluded_domains,
            )
            meo_router = getattr(transformer, 'meo_router', None)
            if meo_router is not None and getattr(meo_router, 'enabled', False):
                return meo_plan
            if meo_plan is not None:
                return meo_plan
        future_loads_map, common_future_edges = self._predict_future_graph_samples()  # 调用 Transformer 获取多次 repeat 的未来预测图，并计算当前与未来共有链路。
        # aggregates = self._build_domain_aggregate_state(  # 基于预测图和共有链路重建域级聚合状态。
        #     future_loads_map=future_loads_map,  # 把 recommend_path 开始时得到的未来预测图传给域聚合函数复用。
        #     common_future_edges=common_future_edges,  # 把当前和未来都存在的链路集合传给域聚合函数过滤不可持续链路。
        # )  # 域聚合状态构建完成。
        graph = self.inter_domain_graph  # 读取已经刷新的域间图。
        if graph is None or graph.number_of_nodes() == 0:  # 如果域间图不存在或没有节点，说明无法做域级规划。
            return None  # 返回 None 表示没有可用域级路径。
        source_domain = self._domain_for_node_or_domain(src)  # 将业务源地址映射到所属域；如果 src 已经是域名则直接使用。
        target_domain = self._domain_for_node_or_domain(dst)  # 将业务目的地址映射到所属域；如果 dst 已经是域名则直接使用。
        if source_domain is None or target_domain is None:  # 如果源域或目的域无法识别，无法进行域间规划。
            return None  # 返回 None 表示规划失败。
        if source_domain == target_domain:  # 如果源和目的处在同一个域内，就不需要域间路径。
            return None  # 当前函数只负责域间规划，同域业务返回 None。
        excluded = {str(domain) for domain in (excluded_domains or [])}
        def directed_boundary_candidates(domain_a, domain_b, attrs):  # 定义内部函数：取出 domain_a 到 domain_b 方向可用的边界链路候选。
            links = attrs.get('boundary_links', {}) or {}  # 从域间边属性中读取所有边界链路明细。
            candidates = []  # 初始化候选边界链路列表。
            for link in links.values():  # 遍历这两个域之间记录的所有边界链路。
                if not isinstance(link, dict):  # 如果链路记录不是字典，说明数据结构异常或无效。
                    continue  # 跳过无效链路记录。
                src_domain = link.get('source_domain')  # 读取这条边界链路的源域。
                dst_domain = link.get('target_domain')  # 读取这条边界链路的目标域。
                if src_domain == domain_a and dst_domain == domain_b:  # 如果链路方向正好是当前规划方向。
                    candidates.append(link)  # 直接加入候选列表。
                elif src_domain == domain_b and dst_domain == domain_a:  # 如果链路记录方向和规划方向相反。
                    reversed_link = dict(link)  # 拷贝一份链路记录用于构造反向视角。
                    reversed_link['source_domain'] = domain_a  # 把反向候选的源域改为当前规划源域。
                    reversed_link['target_domain'] = domain_b  # 把反向候选的目标域改为当前规划目标域。
                    reversed_link['source_boundary'] = link.get('target_boundary')  # 反向时原目标边界变成源边界。
                    reversed_link['target_boundary'] = link.get('source_boundary')  # 反向时原源边界变成目标边界。
                    reversed_link['source_memory_occupancy_rate'] = link.get('target_memory_occupancy_rate')  # 反向时源端存储占用取原目标端。
                    reversed_link['target_memory_occupancy_rate'] = link.get('source_memory_occupancy_rate')  # 反向时目标端存储占用取原源端。
                    reversed_link['source_computing_occupancy_rate'] = link.get('target_computing_occupancy_rate')  # 反向时源端计算占用取原目标端。
                    reversed_link['target_computing_occupancy_rate'] = link.get('source_computing_occupancy_rate')  # 反向时目标端计算占用取原源端。
                    reversed_link['source_remaining_computing'] = link.get('target_remaining_computing')  # 反向时源端剩余算力取原目标端。
                    reversed_link['target_remaining_computing'] = link.get('source_remaining_computing')  # 反向时目标端剩余算力取原源端。
                    candidates.append(reversed_link)  # 把构造好的反向链路加入候选列表。
            return candidates  # 返回当前方向下可用的边界链路候选。

        def best_boundary_link(domain_a, domain_b, attrs):  # 定义内部函数：在两个域之间选择综合风险最低的边界链路。
            candidates = directed_boundary_candidates(domain_a, domain_b, attrs)  # 取出当前方向可用的边界链路候选。
            if not candidates:  # 如果没有候选边界链路。
                return None  # 返回 None 表示两个域之间不可用。
            return min(  # 从候选链路中选择代价最小的一条。
                candidates,  # 候选边界链路集合。
                key=lambda link: float(link.get('quality_cost', float("inf"))),  # 排序指标为已有质量代价加少量容量风险。
            )  # 返回最佳边界链路。
            
        def domain_edge_weight(domain_a, domain_b, attrs):  # 定义内部函数：给域间图上的一条边计算路径搜索权重。
            if str(domain_b) in excluded:
                return float("inf")
            link = best_boundary_link(domain_a, domain_b, attrs)  # 选择该域间边在当前方向下最好的边界链路。
            if link is None:  # 如果两个域之间没有可用边界链路。
                return float("inf")  # 返回无穷大权重，让最短路避开这条边。
            return float(link.get('quality_cost', 1.0))  # 返回域间边权重。
        
        try:  # 开始尝试在域间图上搜索最短风险路径。
            domain_path = nx.shortest_path(graph, source_domain, target_domain, weight=domain_edge_weight)  # 用自定义域间边权重搜索源域到目标域的路径。
        except (nx.NetworkXNoPath, nx.NodeNotFound):  # 如果域间无路径或源/目标域节点不存在。
            return None  # 返回 None 表示无法规划。
        transitions = []  # 初始化跨域跳转明细列表。
        score = 0.0  # 初始化整条域级路径得分。
        for domain_a, domain_b in zip(domain_path[:-1], domain_path[1:]):  # 遍历域路径中的每一段相邻域跳转。
            attrs = graph[domain_a][domain_b]  # 读取两个域之间的域间边属性。
            link = best_boundary_link(domain_a, domain_b, attrs)  # 选择这段域间跳转实际使用的边界链路。
            if link is None:  # 如果该段没有可用边界链路。
                return None  # 返回 None 表示规划失败。
            score += domain_edge_weight(domain_a, domain_b, attrs)  # 将该段域间边权重累加到总得分。
            transitions.append({  # 记录这一次跨域跳转的域和边界节点信息。
                'from_domain': domain_a,  # 保存本段跳转的源域。
                'to_domain': domain_b,  # 保存本段跳转的目标域。
                'source_boundary': link.get('source_boundary'),  # 保存本段跳转使用的源边界 LEO。
                'target_boundary': link.get('target_boundary'),  # 保存本段跳转使用的目标边界 LEO。
                'quality_cost': float(link.get('quality_cost', 1.0)),  # 保存该边界链路的质量代价。
                'link_load': float(link.get('link_load', 0.0) or 0.0),  # 保存该边界链路当前负载。
                'delay': float(link.get('delay', 0.0) or 0.0),  # 保存该边界链路时延。
            })  # 当前跨域跳转记录结束。
        domain_entries = {}  # 初始化每个域内部入口/出口节点记录。
        for idx, domain in enumerate(domain_path):  # 遍历域路径上的每个域。
            entry = src if idx == 0 else transitions[idx - 1]['target_boundary']  # 源域入口为业务源，其余域入口为上一段目标边界。
            exit_node = dst if idx == len(domain_path) - 1 else transitions[idx]['source_boundary']  # 目标域出口为业务目的，其余域出口为下一段源边界。
            domain_entries[domain] = {  # 写入当前域的入口/出口信息。
                'entry': entry,  # 当前域内部路径入口。
                'exit': exit_node,  # 当前域内部路径出口。
            }  # 当前域入口/出口记录结束。

        def intra_domain_path(domain, entry, exit_node):
            """在一个域内展开 entry 到 exit_node 的实际卫星路径。"""
            if entry == exit_node:
                return [entry]

            aggregate = graph.nodes[domain].get('aggregate', {}) if domain in graph else {}
            intra_graph = aggregate.get('intra_graph') if isinstance(aggregate, dict) else None
            if intra_graph is not None and entry in intra_graph and exit_node in intra_graph:
                try:
                    return nx.shortest_path(intra_graph, entry, exit_node, weight='weight')
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    pass

            topology = getattr(self.propagator, 'graph', None) if self.propagator is not None else None
            if topology is None:
                return None
            domain_nodes = [
                node for node in topology.nodes
                if getattr(self._satellite_by_name(node), 'masterMeo', None) == domain
            ]
            domain_graph = topology.subgraph(domain_nodes)
            if entry not in domain_graph or exit_node not in domain_graph:
                return None

            def fallback_weight(_, __, attrs):
                return float(attrs.get(
                    'weight',
                    attrs.get('propagation_weight', attrs.get('delay', 1.0)),
                ) or 1.0)

            try:
                return nx.shortest_path(domain_graph, entry, exit_node, weight=fallback_weight)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return None

        satellite_path = []
        for domain in domain_path:
            entry_exit = domain_entries[domain]
            segment = intra_domain_path(domain, entry_exit['entry'], entry_exit['exit'])
            if segment is None:
                return None
            if satellite_path and satellite_path[-1] == segment[0]:
                satellite_path.extend(segment[1:])
            else:
                satellite_path.extend(segment)
        return {  # 返回域级路径规划结果。
            'domains': domain_path,  # 域级路径经过的域列表。
            'transitions': transitions,  # 域间跳转明细列表。
            'domain_entries': domain_entries,  # 每个域内需要连接的入口和出口。
            'path': satellite_path,  # 展开后的完整卫星路径，包含域内逐跳节点和跨域边界节点。
            'source': src,  # 原始业务源地址。
            'destination': dst,  # 原始业务目的地址。
            'source_domain': source_domain,  # 业务源所属域。
            'target_domain': target_domain,  # 业务目的所属域。
            'packet_size': float(packet_size or 0.0),  # 本次业务数据包大小。
            'score': float(score),  # 域级路径总评分。
            'future_graph_times': sorted(float(time) for time in future_loads_map.keys()),  # 本次规划使用到的未来预测时间片。
            'common_future_edge_count': len(common_future_edges) if common_future_edges is not None else None,  # 当前与未来始终可用链路数量。
            'reachable': True,  # 标记规划成功且目标域可达。
        }  # 域级路径规划结果结束。

    def _domain_for_node_or_domain(self, name):
        if name is None:
            return None
        if self.inter_domain_graph is not None and name in self.inter_domain_graph:
            return name
        satellite = self._satellite_by_name(name)
        if satellite is not None:
            return getattr(satellite, 'masterMeo', None)
        if name == self.name:
            return self.name
        if name in (self.remote_domain_leo_states or {}):
            return name
        return None

    def _predict_future_graph_samples(self):
        transformer = getattr(self, 'transformer_module', None)
        if transformer is None and self.propagator is not None:
            transformer = getattr(self.propagator, 'transformer_module', None)
        repeat = max(1, int(getattr(transformer, 'repeat', 1) or 1)) if transformer is not None else 1
        future_loads_map = {}
        if transformer is not None and hasattr(transformer, 'predict_future_graphs'):
            for _ in range(repeat):
                try:
                    future_graphs = transformer.predict_future_graphs() or {}
                except Exception:
                    future_graphs = {}
                normalized = self._normalize_future_graph_samples(future_graphs)
                for sim_time, graphs in normalized.items():
                    future_loads_map.setdefault(float(sim_time), []).extend(graphs)

        common_edges = self._common_available_edges(future_loads_map)
        if common_edges is not None:
            for graphs in future_loads_map.values():
                for graph in graphs:
                    for edge in list(graph.edges()):
                        if edge not in common_edges:
                            graph.remove_edge(*edge)
        return future_loads_map, common_edges

    def _normalize_future_graph_samples(self, future_graphs):
        if not future_graphs:
            return {}
        if isinstance(future_graphs, nx.Graph):
            sim_time = future_graphs.graph.get('sim_time', self.env.now)
            return {float(sim_time): [future_graphs]}
        if isinstance(future_graphs, dict):
            normalized = {}
            for sim_time, graphs in future_graphs.items():
                if isinstance(graphs, nx.Graph):
                    graph_list = [graphs]
                elif isinstance(graphs, dict):
                    graph_list = [graph for graph in graphs.values() if isinstance(graph, nx.Graph)]
                else:
                    graph_list = [graph for graph in (graphs or []) if isinstance(graph, nx.Graph)]
                normalized[float(sim_time)] = graph_list
            return normalized
        normalized = {}
        for idx, graph in enumerate(future_graphs):
            if not isinstance(graph, nx.Graph):
                continue
            sim_time = graph.graph.get('sim_time', self.env.now + idx + 1)
            normalized.setdefault(float(sim_time), []).append(graph)
        return normalized

    def _common_available_edges(self, future_loads_map):
        def graph_edge_set(graph):
            edges = set(graph.edges())
            if not graph.is_directed():
                edges |= {(dst, src) for src, dst in edges}
            return edges

        current_edges = set()
        if self.propagator is not None and getattr(self.propagator, 'graph', None) is not None:
            current_edges = graph_edge_set(self.propagator.graph)
        edge_sets = [current_edges] if current_edges else []
        for graphs in future_loads_map.values():
            for graph in graphs:
                edge_sets.append(graph_edge_set(graph))
        if not edge_sets:
            return None
        return set.intersection(*edge_sets)

    def record_illegal_action(self, selected_neighbor, use_transformer=False):
        return
        """记录非法动作的时刻、当前卫星、选中邻居和当前邻居列表。"""
        if selected_neighbor is None:
            selected_neighbor = "None"
        neighbors_text = ", ".join(str(neighbor) for neighbor in self.neighbors)
        transformer_text = "use transformer" if use_transformer else "not use transformer"
        record = f"{self.name} {transformer_text} select {selected_neighbor}, neighbors:[{neighbors_text}]"
        timed_record = f"Time {self.env.now:.3f}: {record}"
        self.illegal_action_records.append(timed_record)

        log_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(log_dir, exist_ok=True)
        file_path = os.path.join(log_dir, "illegal_action.txt")
        with open(file_path, "a", encoding="utf-8") as file:
            file.write(timed_record + "\n")
        return timed_record
        
    def configMasterMeo(self, meoName, leoList):
        self.masterMeo = meoName
        self.intra_domain_leo = leoList
        if self.leoStates is None and self.isLeo is False:
            self.leoStates = {}

    def _remaining_memory(self):
        return max(0.0, float(self.memory - self.current_memory_occupy))

    def _current_computing_load(self):
        elapsed_computing = 0.0
        if self.is_computing and self.last_computing_time:
            elapsed_computing = max(0.0, self.env.now - self.last_computing_time) * self.computing_ability
        return max(0.0, float(self.computing_remain) - elapsed_computing)

    def _remaining_computing_resource(self):
        return max(0.0, float(self.computing_ability) - self._current_computing_load())

    def receive_leo_state(self, leo_states):
        if self.leoStates is None:
            self.leoStates = {}
        for leo_name, leo_state in leo_states.items():
            self.leoStates[leo_name] = leo_state
        expected_leos = set(self.my_leos or [])
        if not expected_leos or not expected_leos.issubset(self.leoStates.keys()):
            return
        

    def _build_domain_aggregate_state(self, future_loads_map=None, common_future_edges=None, only_aggregate_self = True):
        """把本 MEO 收齐的域内 LEO 状态聚合成一个域级节点。"""  # 说明该函数会把多个 LEO 的状态汇总为当前 MEO 代表的域状态。
        result = {}
        boundary_node_compute_map = {}
        boundary_node_link_load_map = {}
        boundary_delay_map = {}
        if only_aggregate_self == False:
            merged_dict = self.remote_domain_leo_states | {self.name:self.leoStates}
        else:
            merged_dict = {self.name:self.leoStates}
        for meo, domain_leos in merged_dict.items():
            domain_leos = sorted(domain_leos)  # 对域内 LEO 名称排序，保证后续聚合结果的顺序稳定。
            intra_graph = nx.DiGraph()  # 创建一个有向图，用来描述当前域内部的 LEO 节点和有向域内链路。
            memory_rates = []  # 保存每个域内 LEO 的存储占用率，后面用于计算平均值和最大值。
            compute_rates = []  # 保存每个域内 LEO 的计算占用率，后面用于计算平均值和最大值。
            domain_remaining_computing = {}
            domain_link_load = {}
            edge_delays_s = []
            for leo in domain_leos:  # 遍历当前 MEO 管理的每一颗 LEO。
                state = merged_dict[meo][leo] #读取该 LEO 上报的状态
                remaining_memory = float(state.get('remaining_memory', 0.0) or 0.0)  # 读取剩余存储资源，并转成浮点数。
                remaining_computing = float(state.get('remaining_computing', 0.0) or 0.0)  # 读取剩余计算资源，并转成浮点数。
                domain_remaining_computing[leo] = remaining_computing
                sat = self._satellite_by_name(leo)  # 根据 LEO 名称找到对应的卫星对象，便于读取容量参数。
                for neighbor_state in state.get('neighbors', []) or []:  # 遍历该 LEO 上报的所有邻居链路状态。
                    neighbor = neighbor_state.get('name')  # 获取邻居卫星的名称。
                    if neighbor is None:  # 如果邻居名称缺失，说明该邻居状态无效。
                        continue  # 跳过这条无效的邻居记录。
                    link_load = float(neighbor_state.get('link_load', float("inf")))
                    domain_link_load[(leo, neighbor)] = link_load
                    if common_future_edges is not None and (leo, neighbor) not in common_future_edges:
                        continue
                    if neighbor in domain_leos:#只针对域内链路做统计，用来建模域内时延的cost
                        edge_delays_s.append(self._edge_delay(leo, neighbor))
                edge_delays_s.sort()
                memory_capacity = float(getattr(sat, 'memory', self.memory) or self.memory or 1.0)  # 获取该 LEO 的总存储容量，缺失时用当前对象配置兜底。
                computing_capacity = float(getattr(sat, 'computing_ability', self.computing_ability) or self.computing_ability or 1.0)  # 获取该 LEO 的总计算能力，缺失时用当前对象配置兜底。
                memory_occupancy_rate = self._clip01(1.0 - remaining_memory / max(memory_capacity, 1e-9))  # 用剩余存储计算存储占用率，并限制在 0 到 1。
                computing_occupancy_rate = self._clip01(1.0 - remaining_computing / max(computing_capacity, 1e-9))  # 用剩余计算资源计算计算占用率，并限制在 0 到 1。
                remaining_memory_ratio = self._clip01(remaining_memory / max(memory_capacity, 1e-9))
                remaining_computing_ratio = self._clip01(remaining_computing / max(computing_capacity, 1e-9))
                memory_rates.append(memory_occupancy_rate)  # 把该 LEO 的存储占用率加入统计列表。
                compute_rates.append(computing_occupancy_rate)  # 把该 LEO 的计算占用率加入统计列表。
                intra_graph.add_node(  # 将该 LEO 加入域内图，节点属性保存它的资源状态。
                    leo,  # 节点名称使用 LEO 的名称。
                    timestamp=float(state.get('timestamp', self.env.now) or self.env.now),  # 保存该状态的时间戳，缺失时使用当前仿真时间。
                    is_producing=int(bool(state.get('is_producing', 0))),  # 保存该 LEO 是否正在产生任务，并转换成 0/1。
                    remaining_memory=remaining_memory,  # 保存该 LEO 当前剩余的存储资源。
                    remaining_computing=remaining_computing,  # 保存该 LEO 当前剩余的计算资源。
                    remaining_memory_ratio=remaining_memory_ratio,
                    remaining_computing_ratio=remaining_computing_ratio,
                    memory_occupancy_rate=memory_occupancy_rate,  #保存该 LEO 的存储占用率。
                    computing_occupancy_rate=computing_occupancy_rate,  # 保存该 LEO 的计算占用率。
                )  # 完成该 LEO 节点及其属性的添加。
            
            boundary_links = {}  # 保存从当前域连接到其他 MEO 域的边界链路信息。
            for leo in domain_leos:  # 再次遍历域内 LEO，用于分析它们的邻居链路。
                state = merged_dict[meo][leo] #读取该 LEO 上报的状态
                for neighbor_state in state.get('neighbors', []) or []:  # 遍历该 LEO 上报的所有邻居链路状态。
                    neighbor = neighbor_state.get('name')  # 获取邻居卫星的名称。
                    if neighbor is None:  # 如果邻居名称缺失，说明该邻居状态无效。
                        continue  # 跳过这条无效的邻居记录。
                    link_load = float(neighbor_state.get('link_load', float("inf")))  # 读取链路负载，缺失时按无穷大处理。
                    if common_future_edges is not None and (leo, neighbor) not in common_future_edges:
                        continue
                    if neighbor in domain_leos:  # 如果邻居也属于当前域，则这是一条域内链路。
                        remaining_computing_zScore = z_score(domain_remaining_computing, neighbor)
                        linnk_load_zScore = z_score(domain_link_load, (leo, neighbor))
                        delay = self._edge_delay(leo, neighbor)  # 计算当前 LEO 到邻居 LEO 的链路时延。
                        target_compute_remain = intra_graph.nodes[neighbor]['remaining_computing']  # 读取链路目标端 LEO 的剩余计算资源，用于衡量目标端剩余算力压力。
                        endpoint_compute_cost = 1 / (1 + math.exp(remaining_computing_zScore * 1.5))
                        link_load_cost = 1 / (1 + math.exp(linnk_load_zScore * 1.5))
                        p95_delay_tao = np.percentile(edge_delays_s, 95)
                        delay_cost = 1 - math.exp(- delay / (p95_delay_tao))
                        edge_attrs = {  # 组装域内图边上的属性。
                            'link_load': link_load,  # 保存原始链路负载。
                            'delay': delay,# 保存链路时延。
                            'link_load_cost': link_load_cost,  # 保存归一化后的链路负载。
                            'delay_cost': delay_cost,  # 保存链路cost。
                            'target_compute_remain': target_compute_remain,  # 保存目标端 LEO 的剩余计算资源。
                            'endpoint_compute_cost': endpoint_compute_cost,  # 保存由两端剩余算力换算得到的链路算力代价。
                            'weight': 0.2 * delay_cost + 0.3 * link_load_cost + 0.5 * endpoint_compute_cost,  # 使用时延、链路负载和两端算力代价之和作为边权重。
                        }  # 域内链路属性组装完成。
                        if intra_graph.has_edge(leo, neighbor):  # 如果这条同方向域内边之前已经添加过。
                            old_load = float(intra_graph[leo][neighbor].get('link_load', float("inf")) or float("inf"))  # 读取图中已有边记录的链路负载。
                            old_weight = float(intra_graph[leo][neighbor].get('weight', old_load) or old_load)  # 读取图中已有边记录的综合权重，缺失时用旧负载兜底。
                            if edge_attrs['weight'] > old_weight:  # 如果当前方向的综合权重更大，说明这条边的整体代价更高。
                                intra_graph[leo][neighbor].update(edge_attrs)  # 用当前这组属性更新该域内边。
                        else:  # 如果这条域内边还没有加入图。
                            intra_graph.add_edge(leo, neighbor, **edge_attrs)  # 将这条域内链路加入图，并写入边属性。
                    else:  # 如果邻居不在当前域内，则可能是一条跨域边界链路。
                        neighbor_sat = self._satellite_by_name(neighbor)  # 根据邻居名称查找邻居卫星对象。
                        neighbor_master = getattr(neighbor_sat, 'masterMeo', None)  # 读取邻居卫星所属的 MEO 域。
                        if neighbor_master is None or neighbor_master == self.name:  # 如果无法确定邻居域，或邻居仍属于当前 MEO。
                            continue  # 这条链路不作为有效跨域边界链路处理。
                        source_memory_rate = intra_graph.nodes[leo]['memory_occupancy_rate']  # 读取源边界 LEO 的存储占用率。
                        target_memory_rate = self._node_memory_occupancy_rate(neighbor)  # 计算目标邻居节点的存储占用率。
                        source_compute_rate = intra_graph.nodes[leo]['computing_occupancy_rate']  # 读取源边界 LEO 的计算占用率。
                        target_remaining_computing = float(neighbor_state.get('remaining_computing', 0.0) or 0.0)  # 读取目标邻居剩余计算资源。
                        target_remaining_computing = max(0, target_remaining_computing - random.uniform(10000000000, 50000000000))
                        target_compute_rate = self._node_computing_occupancy_rate(  # 根据目标邻居剩余计算资源估计目标计算占用率。
                            neighbor,  # 传入目标邻居卫星名称。
                            remaining_computing=target_remaining_computing,  # 传入目标邻居上报的剩余计算资源。
                        )  # 目标邻居计算占用率计算完成。
                        # compute_cost = 1 / (1 + math.exp(target_remaining_computing * 1.5))
                        if (self.name, neighbor_master) not in boundary_node_compute_map:
                            boundary_node_compute_map[(self.name, neighbor_master)] = {}
                            boundary_node_link_load_map[(self.name, neighbor_master)] = {}
                            boundary_delay_map[(self.name, neighbor_master)] = []
                        boundary_node_compute_map[(self.name, neighbor_master)][neighbor] = target_remaining_computing
                        boundary_node_link_load_map[(self.name, neighbor_master)][neighbor] = link_load
                        delay = self._edge_delay(leo, neighbor)  # 计算源边界 LEO 到目标邻居的链路时延。
                        boundary_delay_map[(self.name, neighbor_master)].append(delay)
                        boundary_links[(leo, neighbor)] = {  # 将这条跨域边界链路的完整信息加入列表。
                            'source_domain': self.name,  # 保存源域名称，即当前 MEO 名称。
                            'target_domain': neighbor_master,  # 保存目标域名称，即邻居卫星所属 MEO。
                            'source_boundary': leo,  # 保存源边界节点名称。
                            'target_boundary': neighbor,  # 保存目标边界节点名称。
                            'link_load': link_load,  # 保存该跨域链路的原始负载。
                            'delay': delay,  # 保存该跨域链路的时延。
                            'source_memory_occupancy_rate': source_memory_rate,  # 保存源边界节点的存储占用率。
                            'target_memory_occupancy_rate': target_memory_rate,  # 保存目标边界节点的存储占用率。
                            'source_computing_occupancy_rate': source_compute_rate,  # 保存源边界节点的计算占用率。
                            'target_computing_occupancy_rate': target_compute_rate,  # 保存目标边界节点的计算占用率。
                            'source_remaining_computing': intra_graph.nodes[leo]['remaining_computing'],  # 保存源边界节点剩余计算资源。
                            'target_remaining_computing': target_remaining_computing,  # 保存目标边界节点剩余计算资源。
                            'quality_cost': float(np.mean([  # 计算该边界链路的综合代价。
                                delay,  # 综合代价的组成项之一：链路时延。
                                source_compute_rate,  # 综合代价的组成项之一：源节点计算占用率。
                                target_compute_rate,  # 综合代价的组成项之一：目标节点计算占用率。
                            ])),  # 对多个代价指标取平均，并转成浮点数。
                        }  # 该跨域边界链路记录添加完成。

            for key, value in boundary_links.items():
                neighbor_domain = (value['source_domain'], value['target_domain'])
                compute_zScore = z_score(boundary_node_compute_map[neighbor_domain], value['target_boundary'])
                compute_cost = 1 / (1 + math.exp(compute_zScore * 1.5))
                link_load_zScore = z_score(domain_link_load, key)
                link_load_cost = 1 / (1 + math.exp(link_load_zScore * 1.5))
                p95_delay_tao = np.percentile(boundary_delay_map[(neighbor_domain)], 95)
                delay_cost = 1 - math.exp(- delay / (p95_delay_tao))
                value['quality_cost'] = 0.5 * (delay_cost + link_load_cost) + 0.5 * compute_cost

            boundary_nodes = sorted({value['source_boundary'] for key, value in boundary_links.items()})  # 从所有跨域链路中提取当前域的边界节点并排序。
            if future_loads_map is None:
                future_loads_map = {}
                transformer = getattr(self, 'transformer_module', None)
                repeat = max(1, int(getattr(transformer, 'repeat', 1) or 1)) if transformer is not None else 1
                for _ in range(repeat):
                    future_loads = self._domain_future_loads(intra_graph)  # 基于 Transformer 预测得到的未来图结构。
                    normalized_future_loads = self._normalize_future_graph_samples(future_loads)
                    for key, value in normalized_future_loads.items():
                        if future_loads_map.get(key, None) is None:
                            future_loads_map[key] = []
                        future_loads_map[key].extend(value)
            aggregate = {  # 组装当前 MEO 域的聚合状态字典。
                'domain': self.name,  # 当前聚合状态所属的 MEO 域名称。
                'timestamp': self.env.now,  # 当前聚合状态生成时的仿真时间。
                'members': domain_leos,  # 当前域包含的所有 LEO 节点。
                'member_count': len(domain_leos),  # 当前域内 LEO 节点数量。
                'boundary_nodes': boundary_nodes,  # 当前域内连接外部域的边界 LEO 节点列表。
                'boundary_links': boundary_links,  # 当前域到外部域的所有边界链路明细。
                'intra_graph': intra_graph,  # 当前域内部的 LEO 网络图。
                'avg_memory_occupancy_rate': float(np.mean(memory_rates)) if memory_rates else 0.0,  # 当前域内 LEO 的平均存储占用率。
                'max_memory_occupancy_rate': float(np.max(memory_rates)) if memory_rates else 0.0,  # 当前域内 LEO 的最大存储占用率。
                'std_memory_occupancy_rate': float(np.std(memory_rates)) if memory_rates else 0.0,  # 当前域内 LEO 存储占用率的标准差。
                'avg_computing_occupancy_rate': float(np.mean(compute_rates)) if compute_rates else 0.0,  # 当前域内 LEO 的平均计算占用率。
                'max_computing_occupancy_rate': float(np.max(compute_rates)) if compute_rates else 0.0,  # 当前域内 LEO 的最大计算占用率。
                'std_computing_occupancy_rate': float(np.std(compute_rates)) if compute_rates else 0.0,  # 当前域内 LEO 计算占用率的标准差。
            }  # 当前域聚合状态字典组装完成。
            self._refresh_inter_domain_graph(aggregate)
            result[meo] = aggregate
        return result  # 返回当前 MEO 的域级聚合状态，供域间图和状态交换使用。

    def _refresh_inter_domain_graph(self, aggregate):
        """维护 MEO 视角下的域间大节点图，边上保留多条边界链路明细。"""
        domain_name = aggregate.get('domain', self.name)
        graph = self.inter_domain_graph
        if graph is None:
            graph = nx.Graph()
        graph.add_node(
            domain_name,
            aggregate=aggregate,
            members=aggregate['members'],
            boundary_nodes=aggregate['boundary_nodes'],
            avg_memory_occupancy_rate=aggregate['avg_memory_occupancy_rate'],
            max_memory_occupancy_rate=aggregate['max_memory_occupancy_rate'],
            std_memory_occupancy_rate=aggregate['std_memory_occupancy_rate'],
            avg_computing_occupancy_rate=aggregate['avg_computing_occupancy_rate'],
            max_computing_occupancy_rate=aggregate['max_computing_occupancy_rate'],
            std_computing_occupancy_rate=aggregate['std_computing_occupancy_rate'],
            timestamp=aggregate['timestamp'],
        )

        for boundary_link in aggregate['boundary_links'].values():
            target_domain = boundary_link['target_domain']
            graph.add_node(target_domain)
            if graph.has_edge(domain_name, target_domain):
                links = graph[domain_name][target_domain].setdefault('boundary_links', {})
            else:
                graph.add_edge(domain_name, target_domain, boundary_links={})
                links = graph[domain_name][target_domain]['boundary_links']

            link_key = (
                boundary_link['source_boundary'],
                boundary_link['target_boundary'],
            )
            links[link_key] = [
                item for item in links.values()
                if (item.get('source_boundary'), item.get('target_boundary')) != link_key
            ]
            links[link_key] = boundary_link
            graph[domain_name][target_domain]['quality_costs'] = [
                item['quality_cost'] for item in links.values()
            ]
            graph[domain_name][target_domain]['link_loads'] = [
                item['link_load'] for item in links.values()
            ]
            graph[domain_name][target_domain]['memory_occupancy_rates'] = [
                (
                    item['source_memory_occupancy_rate'],
                    item['target_memory_occupancy_rate'],
                )
                for item in links.values()
            ]

        self.inter_domain_graph = graph

    def _path_risk_between_boundaries(self, intra_graph, source, future_loads):
        #计算域内入口和出口之间的路径风险度
        domain_remaining_computing = {
            node: float(data.get('remaining_computing', 0.0))
            for node, data in intra_graph.nodes(data=True)
        }
        domain_link_load = {
            (src, dst): float(attrs.get('link_load', attrs.get('normalized_link_load', float("inf"))))
            for src, dst, attrs in intra_graph.edges(data=True)
        }
        edge_delays_s = [
            attrs.get('delay', float("inf"))
            for _, _, attrs in intra_graph.edges(data=True)
        ]
        p95_delay_tao = max(
            float(np.percentile(edge_delays_s, 95)) if edge_delays_s else 0.0,
            1e-9,
        )

        def sigmoid_cost(zscore_value):
            return 1 / (1 + math.exp(zscore_value * 1.5))

        def zscore_sigmoid_cost(values, key):
            if key not in values:
                return 0.0
            return sigmoid_cost(z_score(values, key))

        def iter_future_graphs():
            future_graph_data = future_loads
            if not future_graph_data:
                return {}
            if isinstance(future_graph_data, nx.Graph):
                sim_time = future_graph_data.graph.get('sim_time', self.env.now)
                return {float(sim_time): [future_graph_data]}
            if isinstance(future_graph_data, dict):
                normalized = {}
                for sim_time, graphs in future_graph_data.items():
                    if isinstance(graphs, nx.Graph):
                        graph_list = [graphs]
                    elif isinstance(graphs, dict):
                        graph_list = [graph for graph in graphs.values() if isinstance(graph, nx.Graph)]
                    else:
                        graph_list = [graph for graph in (graphs or []) if isinstance(graph, nx.Graph)]
                    normalized[float(sim_time)] = graph_list
                return normalized
            normalized = {}
            for idx, graph in enumerate(future_graph_data):
                if not isinstance(graph, nx.Graph):
                    continue
                sim_time = graph.graph.get('sim_time', self.env.now + idx + 1)
                normalized.setdefault(float(sim_time), []).append(graph)
            return normalized

        future_graphs = iter_future_graphs()

        def predicted_edge_data(graph, src, dst):
            if graph.has_edge(src, dst):
                return graph.edges[src, dst]
            if graph.has_edge(dst, src):
                return graph.edges[dst, src]
            return None

        def future_link_load_cost(src, dst):
            """
            按预测时间片先把多张预测图聚合成单张等效图，再计算目标链路 cost。

            future_graphs 的每个时间片可能包含多张预测图。对同一个 simtime：
            1. 对 intra_graph 中每条链路收集该时间片所有预测图里的 link_load。
            2. 如果任意预测图缺少该链路，聚合 link_load 记为 inf。
            3. 否则聚合 link_load = mean_load + penalty * variance。
            4. 在该时间片的聚合单图上，对所有有限链路负载做 Z-Score + sigmoid，
               得到目标链路在该时间片下的 cost。
            """
            if not future_graphs:
                return None

            penalty = float(getattr(self, 'future_link_load_variance_penalty', 1.0) or 0.0)
            costs = {}
            aggregated_graphs = {}

            def read_link_load(edge_attrs):
                value = edge_attrs.get(
                    'predicted_link_load',
                    edge_attrs.get('link_load', edge_attrs.get('normalized_link_load')),
                )
                if value is None:
                    return None
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return None

            for simtime, graphs in future_graphs.items():
                graphs = [graph for graph in (graphs or []) if isinstance(graph, nx.Graph)]
                if not graphs:
                    costs[simtime] = float("inf")
                    continue

                aggregated_graph = nx.DiGraph()
                aggregated_graph.add_nodes_from(intra_graph.nodes(data=True))
                aggregated_values = {}

                for edge_src, edge_dst, attrs in intra_graph.edges(data=True):
                    edge_loads = []
                    edge_missing = False
                    for graph in graphs:
                        edge_data = predicted_edge_data(graph, edge_src, edge_dst)
                        if edge_data is None:
                            edge_missing = True
                            break
                        link_load = read_link_load(edge_data)
                        if link_load is None:
                            edge_missing = True
                            break
                        edge_loads.append(link_load)

                    if edge_missing or not edge_loads:
                        aggregated_load = float("inf")
                    else:
                        mean_load = float(np.mean(edge_loads))
                        variance = float(np.var(edge_loads))
                        aggregated_load = mean_load + penalty * variance

                    aggregated_attrs = dict(attrs)
                    aggregated_attrs['predicted_link_load'] = aggregated_load
                    aggregated_attrs['aggregated_link_load'] = aggregated_load
                    aggregated_attrs['link_load_mean'] = float(np.mean(edge_loads)) if edge_loads else float("inf")
                    aggregated_attrs['link_load_variance'] = float(np.var(edge_loads)) if edge_loads else float("inf")
                    aggregated_attrs['link_load_sample_count'] = len(edge_loads)
                    aggregated_graph.add_edge(edge_src, edge_dst, **aggregated_attrs)
                    aggregated_values[(edge_src, edge_dst)] = aggregated_load

                aggregated_graph.graph['sim_time'] = float(simtime)
                aggregated_graph.graph['source_graph_count'] = len(graphs)
                aggregated_graphs[float(simtime)] = aggregated_graph

                target_load = aggregated_values.get((src, dst))
                if target_load is None:
                    target_load = aggregated_values.get((dst, src))
                if target_load is None or math.isinf(float(target_load)):
                    costs[simtime] = float("inf")
                    continue

                finite_values = {
                    edge: value
                    for edge, value in aggregated_values.items()
                    if value is not None and math.isfinite(float(value))
                }
                target_key = (src, dst) if (src, dst) in finite_values else (dst, src)
                if target_key not in finite_values:
                    costs[simtime] = float("inf")
                    continue
                costs[simtime] = zscore_sigmoid_cost(finite_values, target_key)

            future_link_load_cost.aggregated_graphs = aggregated_graphs
            if not costs:
                return None
            return costs

        def link_load_cost_between(src, dst):
            future_costs = future_link_load_cost(src, dst)
            link_load_cost = sigmoid_cost(z_score(domain_link_load, (src, dst)))
            if future_costs is None:
                return link_load_cost
            if any(math.isinf(cost) for cost in future_costs.values()):
                return float("inf")
            return max([link_load_cost] + [float(cost) for cost in future_costs.values()])

        def future_endpoint_compute_cost(dst):
            """
            按预测时间片先把多张预测图聚合成单张等效节点计算负载图，
            再在每个时间片内对节点计算负载做 Z-Score + sigmoid。
            """
            if not future_graphs:
                return None

            penalty = float(getattr(self, 'future_compute_queue_variance_penalty', 1.0) or 0.0)
            costs = {}
            aggregated_graphs = {}

            def read_compute_load(node_attrs):
                value = node_attrs.get(
                    'predicted_compute_queue',
                    node_attrs.get('compute_queue', node_attrs.get('computing_occupancy_rate')),
                )
                if value is None:
                    return None
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return None

            for simtime, graphs in future_graphs.items():
                graphs = [graph for graph in (graphs or []) if isinstance(graph, nx.Graph)]
                if not graphs:
                    costs[simtime] = float("inf")
                    continue

                aggregated_graph = nx.DiGraph()
                aggregated_graph.add_edges_from(intra_graph.edges(data=True))
                aggregated_values = {}

                for node, attrs in intra_graph.nodes(data=True):
                    node_loads = []
                    node_missing = False
                    for graph in graphs:
                        if node not in graph:
                            node_missing = True
                            break
                        compute_load = read_compute_load(graph.nodes[node])
                        if compute_load is None:
                            node_missing = True
                            break
                        node_loads.append(compute_load)

                    if node_missing or not node_loads:
                        aggregated_load = float("inf")
                    else:
                        mean_load = float(np.mean(node_loads))
                        variance = float(np.var(node_loads))
                        aggregated_load = mean_load + penalty * variance

                    aggregated_attrs = dict(attrs)
                    aggregated_attrs['predicted_compute_queue'] = aggregated_load
                    aggregated_attrs['aggregated_compute_queue'] = aggregated_load
                    aggregated_attrs['compute_queue_mean'] = float(np.mean(node_loads)) if node_loads else float("inf")
                    aggregated_attrs['compute_queue_variance'] = float(np.var(node_loads)) if node_loads else float("inf")
                    aggregated_attrs['compute_queue_sample_count'] = len(node_loads)
                    aggregated_graph.add_node(node, **aggregated_attrs)
                    aggregated_values[node] = aggregated_load

                aggregated_graph.graph['sim_time'] = float(simtime)
                aggregated_graph.graph['source_graph_count'] = len(graphs)
                aggregated_graphs[float(simtime)] = aggregated_graph

                target_load = aggregated_values.get(dst)
                if target_load is None or math.isinf(float(target_load)):
                    costs[simtime] = float("inf")
                    continue

                finite_values = {
                    node: value
                    for node, value in aggregated_values.items()
                    if value is not None and math.isfinite(float(value))
                }
                if dst not in finite_values:
                    costs[simtime] = float("inf")
                    continue
                costs[simtime] = zscore_sigmoid_cost(finite_values, dst)

            future_endpoint_compute_cost.aggregated_graphs = aggregated_graphs
            if not costs:
                return None
            return costs

        def endpoint_compute_cost_between(dst):
            future_costs = future_endpoint_compute_cost(dst)
            endpoint_compute_cost = sigmoid_cost(z_score(domain_remaining_computing, dst))
            if future_costs is None:
                return endpoint_compute_cost
            if any(math.isinf(cost) for cost in future_costs.values()):
                return float("inf")
            return max([endpoint_compute_cost] + [float(cost) for cost in future_costs.values()])

        def future_delay_cost(src, dst):
            """
            按预测时间片计算目标链路的未来物理时延 cost。

            时延由物理约束/拓扑图给出，不是 Transformer 预测值；同一时间片下
            即使有多张预测 graph，也不做 mean/variance 聚合，只取该时间片的
            一张可用 graph 计算链路时延分布和目标链路 cost。
            """
            if not future_graphs:
                return None

            costs = {}

            def read_delay(edge_attrs):
                for key in ('delay', 'propagation_delay', 'propagation_weight', 'predicted_delay'):
                    value = edge_attrs.get(key)
                    if value is None:
                        continue
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        return None
                return None

            for simtime, graphs in future_graphs.items():
                graphs = [graph for graph in (graphs or []) if isinstance(graph, nx.Graph)]
                if not graphs:
                    costs[simtime] = float("inf")
                    continue

                graph = graphs[0]
                finite_delays = []
                for edge_src, edge_dst, edge_attrs in graph.edges(data=True):
                    delay_value = read_delay(edge_attrs)
                    if delay_value is None and intra_graph.has_edge(edge_src, edge_dst):
                        delay_value = intra_graph[edge_src][edge_dst].get('delay')
                    if delay_value is None:
                        continue
                    delay_value = float(delay_value)
                    if math.isfinite(delay_value):
                        finite_delays.append(delay_value)

                current_p95_delay_tao = max(
                    float(np.percentile(finite_delays, 95)) if finite_delays else 0.0,
                    1e-9,
                )

                edge_data = predicted_edge_data(graph, src, dst)
                if edge_data is None:
                    costs[simtime] = float("inf")
                    continue

                target_delay = read_delay(edge_data)
                if target_delay is None and intra_graph.has_edge(src, dst):
                    target_delay = intra_graph[src][dst].get('delay')
                if target_delay is None and intra_graph.has_edge(dst, src):
                    target_delay = intra_graph[dst][src].get('delay')
                if target_delay is None or math.isinf(float(target_delay)):
                    costs[simtime] = float("inf")
                    continue
                costs[simtime] = 1 - math.exp(-float(target_delay) / current_p95_delay_tao)

            if not costs:
                return None
            return costs

        def delay_cost_between(src, dst, attrs):
            delay = float(attrs.get('delay', 0.0) or 0.0)
            current_delay_cost = 1 - math.exp(-delay / p95_delay_tao)
            future_costs = future_delay_cost(src, dst)
            if future_costs is None:
                return current_delay_cost
            if any(math.isinf(cost) for cost in future_costs.values()):
                return float("inf")
            return max([current_delay_cost] + [float(cost) for cost in future_costs.values()])


        def edge_cost(src, dst, attrs):
            delay_cost = delay_cost_between(src, dst, attrs)
            link_load_cost = link_load_cost_between(src, dst)
            endpoint_compute_cost = endpoint_compute_cost_between(dst)
            return 0.2 * delay_cost + 0.3 * link_load_cost + 0.5 * endpoint_compute_cost

        def edge_weight(src, dst, attrs):
            return edge_cost(src, dst, attrs)

        try:
            for src, dst, attrs in intra_graph.edges(data=True):
                attrs["risk_weight"] = edge_cost(src, dst, attrs)
            lengths, paths = nx.single_source_dijkstra(
                intra_graph,
                source=source,
                weight="risk_weight",
            )

        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return {
                'path': [],
                'risk': 1.0,
                'delay': float("inf"),
                'max_node_memory_occupancy_rate': 1.0,
                'max_link_load': 1.0,
                'max_compute_occupancy_rate': 1.0,
                'reachable': False,
            }


        result = {}
        for key, path in paths.items():
            delays = []
            edge_costs = []
            node_memory_rates = []
            compute_rates = []
            for node in path:
                node_data = intra_graph.nodes[node]
                node_memory_rates.append(
                    float(node_data.get('memory_occupancy_rate', 0.0) or 0.0)
                )
                compute_rates.append(
                    float(node_data.get('computing_occupancy_rate', 0.0) or 0.0),
                )
            for src, dst in zip(path[:-1], path[1:]):
                edge_data = intra_graph[src][dst]
                delays.append(edge_data["delay"])
                edge_costs.append(edge_data["risk_weight"])
            total_delay = float(sum(delays))
            risk = self._clip01(float(np.mean(edge_costs)) if edge_costs else 0.0)
            result[key] = {
                        'path': path,
                        'risk': risk,
                        'delay': total_delay,
                        'reachable': True,
                    }
        return result

    def _domain_future_loads(self, intra_graph):
        transformer = getattr(self, 'transformer_module', None)
        if transformer is None and self.propagator is not None:
            transformer = getattr(self.propagator, 'transformer_module', None)
        try:
            graphs = transformer.predict_future_graphs() if transformer is not None else None
        except Exception:
            graphs = None
        return graphs or {}


    def _satellite_by_name(self, satellite_name):
        if self.propagator is None:
            return None
        return self.propagator.satellites.get(satellite_name)

    def _node_memory_occupancy_rate(self, satellite_name):
        satellite = self._satellite_by_name(satellite_name)
        if satellite is None:
            return 1.0
        memory = float(getattr(satellite, 'memory', self.memory) or self.memory or 1.0)
        return self._clip01(float(getattr(satellite, 'current_memory_occupy', 0.0) or 0.0) / max(memory, 1e-9))

    def _node_computing_occupancy_rate(self, satellite_name, remaining_computing=None):
        satellite = self._satellite_by_name(satellite_name)
        if satellite is None:
            return 1.0
        computing_capacity = float(getattr(satellite, 'computing_ability', self.computing_ability) or self.computing_ability or 1.0)
        if remaining_computing is None:
            remaining_computing = satellite._remaining_computing_resource()
        return self._clip01(1.0 - float(remaining_computing or 0.0) / max(computing_capacity, 1e-9))

    def _normalized_link_load(self, source_name, link_load):
        source = self._satellite_by_name(source_name)
        memory = float(getattr(source, 'memory', self.memory) or self.memory or 1.0) if source is not None else float(self.memory or 1.0)
        return self._clip01(float(link_load or 0.0) / max(memory, 1e-9))
    
    def _edge_delay(self, source_name, target_name):
        if self.propagator is None:
            return 0.0
        # delays = getattr(self.propagator, 'propagation_delays', {}) or {}
        # delay = float(delays.get((source_name, target_name), delays.get((target_name, source_name), float("inf"))) or float("inf"))
        # if delay == float("inf"):
        distance = self.propagator._distance(source_name, target_name)
        delay = float(distance) / max(float(self.propagator.propagation_speed), 1e-9)
        return delay
    @staticmethod
    def _clip01(value):
        return max(0.0, min(1.0, float(value)))
        
    #传递强化学习相关参数和直接通信卫星列表，为卫星的决策逻辑提供支持。
    def trans_parameters(self,q_net, reward_function, direct_satellites):
        self.q_net=q_net
        self.reward_function = reward_function
        self.direct_satellites = direct_satellites

    
    # def compute_raw_feat(self):
    #     """
    #     计算当前原始特征（与邻居特征维度一致）
        
    #     Returns:
    #         raw_feat: 原始特征列表 [nbr_dim]
    #     """
    #     # 基础4维特征
    #     base_feat = [
    #         self.is_producing,
    #         1 - self.current_memory_occupy / self.memory,
    #         (self.computing_remain / self.computing_ability - 
    #          self.is_computing * (self.env.now - self.last_computing_time)) / CT_FAC,
    #         sum(self.transmission_size.values()) / self.memory
    #     ]
        
    #     if 'New' in self.mode:
    #         # 12维特征：基础4维 + 邻居平均状态8维
    #         n = len(self.neighbors)
    #         if n > 0:
    #             position_sums = [sum(items) for items in zip(*self.neighbor_states.values())]
    #             av1 = 2 * position_sums[3] / n if len(position_sums) > 3 else 1
    #             av2 = 2 * position_sums[7] / n if len(position_sums) > 7 else 1
    #             av3 = 2 * position_sums[11] / n if len(position_sums) > 11 else 1
    #         else:
    #             av1, av2, av3 = 1, 1, 1
            
    #         # 补全邻居状态
    #         specified_values_list = [1, 0, 1, av1, 1, 0, 1, av2, 1, 0, 1, av3]
    #         self_values = [0, 0, 0, 0] + [x * n for x in self.current_state[:8]] + [0, 0, 0, 0]
    #         additional_values = [x * (4 - n) for x in specified_values_list]
            
    #         if n > 0:
    #             temp = [sum(tup) + add - sub for tup, add, sub in 
    #                    zip(zip(*self.neighbor_states.values()), additional_values, self_values)]
    #         else:
    #             temp = specified_values_list
            
    #         self._raw_feat = base_feat + temp[0:8]
    #     else:
    #         self._raw_feat = base_feat
        
    #     return self._raw_feat
    
    # def receive_raw_feat(self, neighbor_name, raw_feat):
    #     """
    #     接收邻居的原始特征（第1轮交换）
        
    #     Args:
    #         neighbor_name: 邻居名称
    #         raw_feat: 原始特征
    #     """
    #     self._nbr_raw_cache[neighbor_name] = raw_feat
    
    # def receive_hidden(self, neighbor_name, hidden):
    #     """
    #     接收邻居的隐藏状态（第2轮交换）
        
    #     Args:
    #         neighbor_name: 邻居名称
    #         hidden: 聚合后的隐藏状态
    #     """
    #     self._nbr_hidden_cache[neighbor_name] = hidden
    
    # def aggregate_round1(self):
    #     """
    #     第1轮聚合：用 enc['2hop'] 聚合邻居原始特征
        
    #     计算出包含2跳信息的高级特征 aggregated_hidden
    #     """
    #     if self._test_inference is not None and self._raw_feat is not None:
    #         import torch as th
            
    #         # 初始化自己的原始特征
    #         raw_tensor = th.tensor(self._raw_feat, dtype=th.float32)
    #         self._test_inference.init_state(raw_tensor)
            
    #         # 接收邻居的原始特征
    #         for nbr_name, raw_feat in self._nbr_raw_cache.items():
    #             if not isinstance(raw_feat, th.Tensor):
    #                 raw_feat = th.tensor(raw_feat, dtype=th.float32)
    #             self._test_inference.receive_neighbor_feature(nbr_name, raw_feat, 'raw')
            
    #         # 用 enc['2hop'] 聚合
    #         self._test_inference.aggregate_round1(list(self._nbr_raw_cache.keys()))
            
    #         # 保存聚合结果
    #         self._aggregated_hidden = self._test_inference.aggregated_hidden
    
    # def get_hidden_to_send(self):
    #     """
    #     获取要发送的隐藏状态（第2轮交换）
        
    #     Returns:
    #         hidden: 聚合后的隐藏状态，或 None
    #     """
    #     return self._aggregated_hidden
    
    # def make_decision_distributed(self, agent_feat, task_context, rnn_hidden=None):
    #     """
    #     分布式决策：用 enc['1hop'] 聚合邻居隐藏状态，计算Q值
        
    #     Args:
    #         agent_feat: agent节点的原始特征
    #         task_context: 任务上下文
    #         rnn_hidden: RNN隐藏状态（可选）
        
    #     Returns:
    #         q_vals: Q值张量
    #         new_rnn_hidden: 新的RNN隐藏状态
    #     """
    #     if self._test_inference is not None:
    #         import torch as th
            
    #         # 接收邻居的隐藏状态
    #         for nbr_name, hidden in self._nbr_hidden_cache.items():
    #             if not isinstance(hidden, th.Tensor):
    #                 hidden = th.tensor(hidden, dtype=th.float32)
    #             self._test_inference.receive_neighbor_feature(nbr_name, hidden, 'hidden')
            
    #         # 计算Q值
    #         q_vals, new_rnn_hidden = self._test_inference.make_decision(
    #             agent_feat=agent_feat,
    #             task_context=task_context,
    #             neighbor_names=list(self._nbr_hidden_cache.keys()),
    #             rnn_hidden=rnn_hidden
    #         )
    #         return q_vals, new_rnn_hidden
    #     return None, None
    
    # def clear_hidden_cache(self):
    #     """清空隐藏状态缓存，准备下一轮交换"""
    #     self._nbr_raw_cache.clear()
    #     self._nbr_hidden_cache.clear()
    
    # ================================================================================
    # 路由表构建
    # ================================================================================
    
    """# 作用：构建卫星的路由表，记录到达其他节点的最短路径和跳数，用于数据包转发决策。
    # 核心逻辑：
    # 使用 BFS(广度优先搜索） 计算路径：从自身节点和邻居节点出发，遍历网络拓扑（基于 adjacency_table 邻接表）。
    # 生成两个关键数据结构：
    # result_dict:记录自身到其他节点的路由信息，格式为 {目标节点: ([下一跳节点列表], 跳数)}（支持多路径，相同跳数的下一跳均会被记录）。
    # neighbor_hops:记录邻居节点到其他节点的跳数，用于辅助路由决策（如评估邻居的路径质量）。用于构造状态特征（邻居到某目的的 hop 归一化），供 RL 网络输入或启发式决策。
    # 最终将 result_dict 赋值给 self.routing_tables,作为卫星的路由表。"""
    def build_routing_table(self):
        result_dict = {}
        self.neighbor_hops = {neighbor: {} for neighbor in self.neighbors}
        for start in [self.name] + self.neighbors:
            queue = [(neighbor, [start, neighbor], 1) for neighbor in self.adjacency_table[start][0]]
            if start != self.name:#queue=(node, path, hops)  node：当前遍历到的节点（卫星或地面站）；path：从起始节点（start）到node的路径列表（例如[start, node1, node]表示从start经node1到达node）；hops：从起始节点（start）到node的跳数（路径长度，每经过一个节点跳数 + 1）。
                self.neighbor_hops[start][start] = 0
            while queue:
                (node, path, hops) = queue.pop(0)
                if start == self.name:
                    if node not in result_dict:
                        result_dict[node] = ([path[1]], hops)
                        queue.extend((neighbor, path + [neighbor], hops + 1) for neighbor in self.adjacency_table[node][0] if neighbor not in path)
                    elif result_dict[node][1] == hops:
                        result_dict[node][0].append(path[1])
                else:
                    if node not in self.neighbor_hops[start]:
                        self.neighbor_hops[start][node] = hops
                        queue.extend((neighbor, path + [neighbor], hops + 1) for neighbor in self.adjacency_table[node][0] if neighbor not in path)
        result_dict[self.name] = ([self.name], 0)
        self.routing_tables = result_dict
    """# 作用:将数据包加入转发队列(forward_queue),并检查内存是否充足。
    # 核心逻辑：
    # 先判断当前内存占用(current_memory_occupy)加上数据包大小后是否超过卫星内存容量(memory)。
    # 若内存充足：
    # 更新内存占用量，将数据包放入转发队列。
    # 若卫星工作在传统模式(Tradition)或地面模式(Ground),同步更新拓扑图(propagator.graph)中自身节点的内存占用属性，便于全局监控。
    # 返回 True 表示成功加入队列。
    # 若内存不足：返回 False,数据包可能被丢弃或转发到其他节点。
    #应用场景：处理接收到的待转发数据包，确保内存资源不溢出。"""
    def _record_out_memory_drop(self, packet):
        if len(packet.information) > 1:
            packet_type = int(packet.information[1])
            self.statics_data[f'out_memory_{packet_type}'] = self.statics_data.get(f'out_memory_{packet_type}', 0) + 1

    def push_forward(self,packet):
        if self.current_memory_occupy + packet.size < self.memory:
            self.current_memory_occupy += packet.size
            self.forward_queue.put(packet)
            if self.mode in ["Tradition","Ground"]:
                if 'current_memory_occupy' in self.propagator.graph.nodes[self.name]:
                    self.propagator.graph.nodes[self.name]['current_memory_occupy'] += packet.size
                else:
                    self.propagator.graph.nodes[self.name]['current_memory_occupy'] = packet.size
            return True
        else:
            self._record_out_memory_drop(packet)
            return False
    # 作用：将待计算的任务包加入本地计算队列（computing_queue），并更新计算资源占用。
    # 参数说明：
    # packet：待计算的任务数据包。
    # computing_demand：该任务所需的计算资源（如算力、时间）。
    # 核心逻辑：
    # 更新计算队列的大小（current_computing_queue_size）和剩余计算需求（computing_remain）。
    # 将任务包放入计算队列，等待卫星处理。
    # 若全局参数 PRE 为 False，同步更新拓扑图中自身节点的剩余计算需求属性（computing_remain），用于全局资源监控。#注意：PRE 的意义在代码中是用来决定是否提前占用全局图上的计算资源
    def push_computing(self,packet,computing_demand):
        self._record_packet_compute_assignment(packet, self.name, update_computing_node=False)
        self.current_computing_queue_size += packet.size
        self.computing_remain += computing_demand
        self.computing_queue.put(packet)
        if not PRE:
            if 'computing_remain' in self.propagator.graph.nodes[self.name]:
                self.propagator.graph.nodes[self.name]['computing_remain'] += computing_demand
            else:
                self.propagator.graph.nodes[self.name]['computing_remain'] = computing_demand

    def _record_packet_compute_assignment(self, packet, compute_node, update_computing_node=True):
        if packet is None or compute_node is None:
            return
        if update_computing_node:
            packet.computing_node = compute_node
        packet.computing_leo = compute_node
        compute_sat = None
        if self.propagator is not None:
            compute_sat = self.propagator.satellites.get(compute_node)
        packet.computing_meo = getattr(compute_sat, "masterMeo", None)

    def _planned_compute_chunk(self, packet, remaining_demand):
        remaining_demand = max(0.0, float(remaining_demand or 0.0))
        if not hasattr(packet, "remaining_computing_demand"):
            packet.remaining_computing_demand = remaining_demand
            packet.total_computing_demand = remaining_demand
        planned_chunk = getattr(packet, "current_compute_chunk", None)
        if planned_chunk is not None:
            packet.current_compute_chunk = None
            return min(max(0.0, float(planned_chunk or 0.0)), float(packet.remaining_computing_demand))
        remaining_flags = sum(1 for flag in (getattr(packet, "compute_flags", None) or []) if int(flag) == 1)
        segments_left = max(1, remaining_flags + 1)
        return max(0.0, float(packet.remaining_computing_demand) / segments_left)
    # 作用：将数据包加入到指定邻居的传输队列（transmission_queue），准备发送给该邻居。
    # 参数说明：
    # neighbor：目标邻居卫星的名称。
    # packet：待传输的数据包。
    # 核心逻辑：
    # 更新总队列大小（current_queue_size）、对应邻居的传输队列长度（transmission_length）和数据量（transmission_size）。
    # 将数据包放入该邻居的传输队列，等待发送。
    # 若卫星工作在传统模式或地面模式，同步更新拓扑图中自身与邻居连接的传输权重（transmission_weight），用于评估链路负载。
    # 应用场景：处理需要发送给邻居的数据包（如转发任务、卸载任务），跟踪星间链路的负载情况。
    def push_transmission(self,neighbor,packet):
        self.current_queue_size += packet.size
        self.transmission_length[neighbor]+=1
        self.transmission_size[neighbor]+=packet.size
        self.transmission_queue[neighbor].put(packet)
        if self.mode in ["Tradition","Ground"]:
            if 'transmission_weight' in self.propagator.graph[self.name][neighbor]:
                self.propagator.graph[self.name][neighbor]['transmission_weight']+=packet.size
            else:
                self.propagator.graph[self.name][neighbor]['transmission_weight'] = packet.size
    # 作用：从指定邻居的传输队列中取出一个数据包（准备发送），并更新相关资源状态。
    # 核心逻辑：
    # 使用 simpy.Store.get() 从 transmission_queue[neighbor] 中获取数据包（simpy 的异步操作，需用 yield）。
    # 减少与该数据包相关的队列统计：总队列大小（current_queue_size）、对应邻居的传输数据量（transmission_size）和传输长度（transmission_length），以及当前内存占用（current_memory_occupy）。
    # 若工作在 Tradition 或 Ground 模式，同步更新拓扑图（propagator.graph）中自身与邻居的链路传输权重（transmission_weight）和自身节点的内存占用，确保全局状态一致。
    # 应用场景：卫星准备向邻居发送数据包时，从队列中取出数据并释放资源，保证状态准确。
    def pop_transmission(self,neighbor):
        packet = yield self.transmission_queue[neighbor].get()
        self.current_queue_size-=packet.size
        self.transmission_size[neighbor]-=packet.size
        self.transmission_length[neighbor]-=1
        self.current_memory_occupy -= packet.size
        if self.mode in ["Tradition","Ground"]:
            self.propagator.graph[self.name][neighbor]['transmission_weight'] -= packet.size
            self.propagator.graph.nodes[self.name]['current_memory_occupy'] -= packet.size
        return packet

    def push_offload(self,packet):
        self.current_queue_size += packet.size
        self.offload_size+=packet.size
        self.offload_length+=1
        self.offload_queue.put(packet)

    def pop_offload(self):
        packet = yield self.offload_queue.get()
        self.current_queue_size-=packet.size
        self.offload_size-=packet.size
        self.offload_length-=1
        self.current_memory_occupy -= packet.size
        if self.mode in ["Tradition","Ground"]:
            self.propagator.graph.nodes[self.name]['current_memory_occupy'] -= packet.size
        return packet

    def get_current_state(self, destination, hops, is_computed, mission_state):
        current_state = []
        neighbors_state = []
        current_node_state=[self.is_producing, 1 - self.current_memory_occupy / self.memory,(self.computing_remain / self.computing_ability-self.is_computing*(self.env.now-self.last_computing_time))/ CT_FAC]
        for neighbor in self.neighbors:
            if 'New' in self.mode:
                neighbors_state.extend(self.neighbor_states[neighbor][0:4]+[x/4 for x in self.neighbor_states[neighbor][4:8]]+[x/12 for x in self.neighbor_states[neighbor][8:12]])
            else:
                neighbors_state.extend(self.neighbor_states[neighbor])
            neighbors_state.append(self.transmission_size[neighbor] / self.memory)
            if destination in self.neighbor_hops[neighbor]:
                neighbors_state.append(self.neighbor_hops[neighbor][destination] / self.max_hop)
            else:
                neighbors_state.append(2)
        if len(self.neighbors) < 4:
            if 'New' in self.mode:
                if neighbors_state:
                    av1,av2,av3=2*sum(neighbors_state[3::14])/len(self.neighbors),2*sum(neighbors_state[7::14])/len(self.neighbors),2*sum(neighbors_state[11::14])/len(self.neighbors)
                else:
                    av1,av2,av3=1,1,1
            for _ in range(4 - len(self.neighbors)):
                if 'New' in self.mode:
                    neighbors_state.extend([1, 0, 1, av1, 1, 0, 1, av2, 1, 0, 1, av3, 1, 2])
                else:
                    neighbors_state.extend([1, 0, 1, 1, 1, 2])
        current_state.extend(neighbors_state)
        current_state.extend(current_node_state)
        current_state.extend(mission_state)
        current_state.extend([hops / self.max_hop,is_computed])
        return np.array(current_state)

    def get_current_state_with_2hop(self, destination, hops, is_computed, mission_state):
        """
        获取包含2跳邻居信息的状态（用于GNN模式，配置 n_hops=2 时使用）
        
        返回:
            state: 1跳状态向量（与 get_current_state 相同）[33维]
            hop2_neighbor_feats: [4, 3, 6] 每个1跳邻居的2跳邻居特征（GNN模式固定6维）
            hop2_neighbor_mask: [4, 3] 2跳邻居掩码，True=有效
            neighbor_mask: [4] 1跳邻居掩码，True=有效
        
        特征维度说明（GNN模式，per_neighbor_dim=6）：
            - 节点状态: 4维 [is_producing, memory_remain, computing_remain, is_computing]
            - 边特征: 2维 [transmission_size, hop_distance]
            - 总计: 6维
        """
        # 1. 获取基础状态（1跳）
        state = self.get_current_state(destination, hops, is_computed, mission_state)
        
        # 2. 构建1跳邻居掩码
        neighbor_mask = np.zeros(4, dtype=bool)
        for i in range(min(len(self.neighbors), 4)):
            neighbor_mask[i] = True
        
        # 3. 获取2跳邻居信息
        max_hop2 = 3  # 每个1跳邻居最多3个2跳邻居（排除中心节点）
        
        # GNN模式: 每个邻居特征 = 节点状态(4维) + 边特征(2维) = 6维
        # New模式: 每个邻居特征 = 多跳状态(12维) + 边特征(2维) = 14维
        if 'New' in self.mode:
            node_state_dim = 12  # New模式: 3跳 × 4维状态
            per_neighbor_dim = 14
        else:
            node_state_dim = 4   # GNN模式: 仅1跳状态
            per_neighbor_dim = 6
        
        hop2_neighbor_feats = np.zeros((4, max_hop2, per_neighbor_dim), dtype=np.float32)
        hop2_neighbor_mask = np.zeros((4, max_hop2), dtype=bool)
        
        for i, hop1_neighbor in enumerate(self.neighbors[:4]):
            # 从邻接表获取该1跳邻居的邻居列表（即2跳邻居）
            if hop1_neighbor in self.adjacency_table:
                hop1_nbr_neighbors, _ = self.adjacency_table[hop1_neighbor]
                
                # 过滤掉当前节点和已在1跳邻居中的节点
                hop2_candidates = [
                    n for n in hop1_nbr_neighbors 
                    if n != self.name and n not in self.neighbors
                ]
                
                for j, hop2_neighbor in enumerate(hop2_candidates[:max_hop2]):
                    # 获取2跳邻居的节点状态
                    # 注意：neighbor_states 只包含1跳邻居，2跳邻居通常不在其中
                    if hop2_neighbor in self.neighbor_states:
                        # 如果有缓存的状态（通过状态交换获得，通常不会发生）
                        if 'New' in self.mode:
                            nbr_state = self.neighbor_states[hop2_neighbor]
                            node_state = (
                                nbr_state[0:4] + 
                                [x/4 for x in nbr_state[4:8]] + 
                                [x/12 for x in nbr_state[8:12]]
                            )
                        else:
                            # GNN模式：直接使用4维状态
                            node_state = list(self.neighbor_states[hop2_neighbor])
                    else:
                        # 没有状态信息，使用默认值（表示未知状态）
                        # [is_producing=0, memory_remain=0.5, computing_remain=0.5, is_computing=0]
                        if 'New' in self.mode:
                            node_state = [0, 0.5, 0.5, 0] * 3  # 12维
                        else:
                            node_state = [0, 0.5, 0.5, 0]  # 4维
                    
                    # 边特征
                    transmission_size = 0.5  # 未知传输队列，使用中性值
                    
                    # 估计2跳邻居到目的地的距离
                    # 优先级: 1. 从neighbor_hops获取2跳邻居的路由信息（如果有）
                    #         2. 使用1跳邻居到目的地的距离作为近似
                    #         3. 使用默认值
                    hop_distance = 0.5  # 默认中性值（归一化后）
                    
                    # 尝试方法1：从1跳邻居的neighbor_hops中获取2跳邻居到目的地的距离
                    if hop1_neighbor in self.neighbor_hops:
                        if hop2_neighbor in self.neighbor_hops[hop1_neighbor]:
                            # 如果1跳邻居知道2跳邻居到目的地的距离（通过状态交换）
                            # 这是最准确的，但通常不可用
                            hop2_to_dest = self.neighbor_hops[hop1_neighbor].get(destination, None)
                            if hop2_to_dest is not None:
                                # 2跳邻居是1跳邻居的邻居，距离关系：|hop2_to_dest - hop1_to_dest| <= 1
                                # 使用1跳邻居的距离作为近似
                                hop_distance = hop2_to_dest / self.max_hop
                        elif destination in self.neighbor_hops[hop1_neighbor]:
                            # 方法2：使用1跳邻居到目的地的距离作为近似
                            # 2跳邻居到目的地的距离约等于1跳邻居到目的地的距离（可能±1跳）
                            hop1_to_dest = self.neighbor_hops[hop1_neighbor][destination]
                            hop_distance = hop1_to_dest / self.max_hop
                    
                    # 组装特征: 节点状态 + 边特征
                    hop2_neighbor_feats[i, j] = np.array(
                        node_state + [transmission_size, hop_distance], 
                        dtype=np.float32
                    )
                    hop2_neighbor_mask[i, j] = True
        
        return state, hop2_neighbor_feats, hop2_neighbor_mask, neighbor_mask
    
    def _get_state_for_decision(self, destination, hops, is_computed, mission_state):
        """
        根据 obs_type 返回对应格式的观测
        
        Args:
            destination: 目标节点
            hops: 当前跳数
            is_computed: 是否已计算
            mission_state: 任务状态 [type, size, computing_demand, size_after_computing]
        
        Returns:
            根据 self.propagator.obs_type 返回不同格式：
            - 'flat': 返回 (state, hop2_info) —— 扁平状态向量
            - 'relational': 返回 (obs_graph, None) —— PyG 图，无边特征
            - 'graph': 返回 (obs_graph, None) —— PyG 图，有边特征
            - 'relational_separated': 返回 (obs_graph, task_context) —— 分离模式
            - 'graph_separated': 返回 (obs_graph, task_context) —— 分离模式
            
            第二个返回值：
            - flat模式: hop2_info 用于2跳邻居信息
            - 分离模式: task_context 用于任务特征
            - 其他模式: None
        """
        obs_type = getattr(self.propagator, 'obs_type', 'flat') if hasattr(self, 'propagator') else 'flat'
        
        if obs_type == 'relational_separated':
            # 【分离模式】relational：agent节点只有环境特征，task独立返回
            try:
                obs, task_context = self.get_obs_relational_separated(
                    destination, hops, is_computed, mission_state, 
                    khops=self.n_hops
                )
                return obs, task_context
            except Exception as e:
                print(f"Warning: Failed to create separated relational graph: {e}, falling back to flat state")
                obs = self.get_current_state(destination, hops, is_computed, mission_state)
                return obs, None
            
                """
                图结构中没有独立的边特征（与 relational 相同）
                agent 节点特征只有环境信息（3维）
                hop2_info 实际是 task_context，用于后续决策时融合
                边信息拼接到邻居节点特征
                
                """
        elif obs_type == 'graph_separated':
            # 【分离模式】graph：agent节点只有环境特征，task独立返回
            try:
                obs, task_context, task_edge_feats = self.get_obs_graph_separated(
                    destination, hops, is_computed, mission_state, 
                    khops=self.n_hops
                )
                """
                obs:PyG 异构图（只包含纯环境信息）
                task_context:任务上下文张量 [6维]
                    { [type, size/max_size, computing_demand/computing_ability, 
                    size_after_computing/max_size, hops/max_hop, is_computed]  }
                    这是与任务相关的全局特征，独立于图结构
                
                task_edge_feats:任务相关的边特征,独立于obs
                """
                return obs, task_context, task_edge_feats
            except Exception as e:
                print(f"Warning: Failed to create separated graph: {e}, falling back to flat state")
                obs = self.get_current_state(destination, hops, is_computed, mission_state)
                return obs, None, None
        elif obs_type == 'relational':
            # relational 模式：无边特征，边信息拼接到邻居节点特征
            try:
                obs = self.get_obs_relational(
                    destination, hops, is_computed, mission_state, 
                    khops=self.n_hops
                )
            except Exception as e:
                print(f"Warning: Failed to create relational graph: {e}, falling back to flat state")
                obs = self.get_current_state(destination, hops, is_computed, mission_state)
            return obs, None
            """
            图结构中没有独立的边特征（edata 为空）
            边信息（transmission_size, hop_distance）直接拼接到邻居节点特征
            agent 节点特征包含任务信息（9维）
            邻居节点特征：[节点状态(4/12维) + transmission_size + hop_distance]
            
            """
        elif obs_type == 'graph':
            # graph 模式：有边特征
            try:
                obs = self.get_obs_graph(
                    destination, hops, is_computed, mission_state, 
                    khops=self.n_hops
                )
            except Exception as e:
                print(f"Warning: Failed to create graph: {e}, falling back to flat state")
                obs = self.get_current_state(destination, hops, is_computed, mission_state)
            return obs, None
            """
            图结构中有独立的边特征（edata 包含边信息）
            边特征：[transmission_size, hop_distance, placeholder]（3维）
            agent 节点特征包含任务信息（9维）
            邻居节点特征：只包含节点状态（4/12维），不含边信息
            
            """
        else:
            # 扁平格式：返回状态向量
            if self.n_hops >= 2:
                state, hop2_feats, hop2_mask, nbr_mask = self.get_current_state_with_2hop(
                    destination, hops, is_computed, mission_state
                )
                hop2_info = (hop2_feats, hop2_mask, nbr_mask)
            else:
                state = self.get_current_state(destination, hops, is_computed, mission_state)
                hop2_info = None
            return state, hop2_info
            """
            最简单的观测格式
            包含 agent 状态 + 4个邻居状态 + 任务信息
            不使用图结构
            """
        
    def get_obs(self, destination, hops, is_computed, mission_state):
        """
        获取字典格式的观测，与 cross_layer_opt_with_grl-main 的 get_obs() 保持一致
        
        返回每个 agent 的观测字典，包含：
        - 'agent': agent 自身的特征向量
        - 'nbr': 邻居节点特征矩阵，每行包含 [availability, 状态特征..., hop_distance]
        
        Args:
            destination: 目标节点
            hops: 当前跳数
            is_computed: 是否已计算
            mission_state: 任务状态 [type, size, computing_demand, size_after_computing]
        
        Returns:
            list: 观测字典列表 [{'agent': own_feats, 'nbr': nbr_feats}]
        """
        # === 1. Agent 自身特征 ===
        # 当前节点状态 + 任务状态 + 路由信息
        current_node_state = np.array([
            self.is_producing, 
            1 - self.current_memory_occupy / self.memory,
            (self.computing_remain / self.computing_ability - 
             self.is_computing * (self.env.now - self.last_computing_time)) / CT_FAC
        ], dtype=np.float32)
        
        routing_info = np.array([hops / self.max_hop, is_computed], dtype=np.float32)
        
        own_feats = np.concatenate([current_node_state, mission_state, routing_info])
        
        # === 2. 邻居特征矩阵 ===
        max_nbrs = 4
        if 'New' in self.mode:
            nbr_feat_dim = 14  # 12维状态 + transmission_size + hop_distance
        else:
            nbr_feat_dim = 7   # 1(availability) + 4维状态 + transmission_size + hop_distance
        
        nbr_feats = np.zeros((max_nbrs, nbr_feat_dim), dtype=np.float32)
        
        for m, neighbor in enumerate(self.neighbors[:max_nbrs]):
            ind = 0
            # Availability 标志（1=有效邻居）
            nbr_feats[m, ind] = 1
            ind += 1
            
            # 邻居状态
            if 'New' in self.mode:
                # 新模式：12维状态（归一化）
                state = self.neighbor_states[neighbor]
                nbr_feats[m, ind:ind+4] = state[0:4]
                ind += 4
                nbr_feats[m, ind:ind+4] = [x/4 for x in state[4:8]]
                ind += 4
                nbr_feats[m, ind:ind+4] = [x/12 for x in state[8:12]]
                ind += 4
            else:
                # 旧模式：4维状态
                nbr_feats[m, ind:ind+4] = self.neighbor_states[neighbor]
                ind += 4
            
            # 传输队列大小
            nbr_feats[m, ind] = self.transmission_size[neighbor] / self.memory
            ind += 1
            
            # 到目的地的跳数
            if destination in self.neighbor_hops[neighbor]:
                nbr_feats[m, ind] = self.neighbor_hops[neighbor][destination] / self.max_hop
            else:
                nbr_feats[m, ind] = 2  # 不可达标记
        
        # === 3. 填充无效邻居 ===
        # 无效邻居的 availability=0（初始化时已经是0），设置默认填充值
        if len(self.neighbors) < max_nbrs:
            for m in range(len(self.neighbors), max_nbrs):
                # nbr_feats[m, 0] = 0  # availability=0，已经是初始值
                # 设置其他特征的默认填充值
                if 'New' in self.mode:
                    nbr_feats[m, 1:] = [1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 2][:nbr_feat_dim-1]
                else:
                    nbr_feats[m, 1:] = [1, 0, 1, 1, 1, 2][:nbr_feat_dim-1]
        
        # 返回观测字典
        # 邻居掩码可通过 nbr_feats[:, 0] == 1 获取（与 cross_layer_opt_with_grl-main 一致）
        return [dict(agent=own_feats, nbr=nbr_feats)]

    def get_obs_separated(self, destination, hops, is_computed, mission_state):
        """
        【任务特征分离版本】获取字典格式的观测
        
        与 get_obs() 的区别：
        - agent 特征只包含纯环境信息（节点状态），不包含任务信息
        - task 特征独立返回，用于在 GNN 聚合后与环境表示融合
        
        这种设计的优势：
        1. GNN 学到的是纯环境/拓扑表示，与任务无关
        2. 注意力权重只基于网络状态，更稳定
        3. 更好的泛化能力：不同任务使用相同的环境表示
        
        返回每个 agent 的观测字典，包含：
        - 'agent': agent 纯环境特征（节点状态）[3维]
        - 'nbr': 邻居节点特征矩阵（不变）
        - 'task': 任务上下文特征 [mission_state(4) + routing_info(2)] = [6维]
        
        Args:
            destination: 目标节点
            hops: 当前跳数
            is_computed: 是否已计算
            mission_state: 任务状态 [type, size, computing_demand, size_after_computing]
        
        Returns:
            list: 观测字典列表 [{'agent': agent_env_feats, 'nbr': nbr_feats, 'task': task_context}]
        """
        # === 1. Agent 环境特征（纯节点状态，不包含任务信息）===
        current_node_state = np.array([
            self.is_producing, 
            1 - self.current_memory_occupy / self.memory,
            (self.computing_remain / self.computing_ability - 
             self.is_computing * (self.env.now - self.last_computing_time)) / CT_FAC
        ], dtype=np.float32)
        
        # Agent 节点特征只包含环境信息
        agent_env_feats = current_node_state  # [3维]
        
        # === 2. 任务上下文（独立于图结构）===
        routing_info = np.array([hops / self.max_hop, is_computed], dtype=np.float32)
        task_context = np.concatenate([mission_state, routing_info])  # [6维]
        
        # === 3. 邻居特征矩阵（与 get_obs 相同）===
        max_nbrs = 4
        if 'New' in self.mode:
            nbr_feat_dim = 14  # 12维状态 + transmission_size + hop_distance
        else:
            nbr_feat_dim = 7   # 1(availability) + 4维状态 + transmission_size + hop_distance
        
        nbr_feats = np.zeros((max_nbrs, nbr_feat_dim), dtype=np.float32)
        
        for m, neighbor in enumerate(self.neighbors[:max_nbrs]):
            ind = 0
            # Availability 标志（1=有效邻居）
            nbr_feats[m, ind] = 1
            ind += 1
            
            # 邻居状态
            if 'New' in self.mode:
                state = self.neighbor_states[neighbor]
                nbr_feats[m, ind:ind+4] = state[0:4]
                ind += 4
                nbr_feats[m, ind:ind+4] = [x/4 for x in state[4:8]]
                ind += 4
                nbr_feats[m, ind:ind+4] = [x/12 for x in state[8:12]]
                ind += 4
            else:
                nbr_feats[m, ind:ind+4] = self.neighbor_states[neighbor]
                ind += 4
            
            # 传输队列大小（边相关信息）
            nbr_feats[m, ind] = self.transmission_size[neighbor] / self.memory
            ind += 1
            
            # 到目的地的跳数（边相关信息）
            if destination in self.neighbor_hops[neighbor]:
                nbr_feats[m, ind] = self.neighbor_hops[neighbor][destination] / self.max_hop
            else:
                nbr_feats[m, ind] = 2  # 不可达标记
        
        # === 4. 填充无效邻居 ===
        if len(self.neighbors) < max_nbrs:
            for m in range(len(self.neighbors), max_nbrs):
                if 'New' in self.mode:
                    nbr_feats[m, 1:] = [1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 2][:nbr_feat_dim-1]
                else:
                    nbr_feats[m, 1:] = [1, 0, 1, 1, 1, 2][:nbr_feat_dim-1]
        
        # 返回分离的观测字典
        return [dict(agent=agent_env_feats, nbr=nbr_feats, task=task_context)]

    def get_obs_relational_separated(self, destination, hops, is_computed, mission_state, khops: int = 1):
        """
        【任务特征分离版本】获取 relational 模式的 PyG 图观测
        
        与 get_obs_relational() 的区别：
        - agent 节点特征只包含纯环境信息
        - task 特征作为额外返回值，用于后续与 GNN 输出融合
        
        Args:
            destination: 目标节点
            hops: 当前跳数
            is_computed: 是否已计算
            mission_state: 任务状态
            khops: GNN 感知的邻居范围
        
        Returns:
            tuple: (HeteroData 图, task_context tensor)
        """
        # 获取分离的观测
        obs = self.get_obs_separated(destination, hops, is_computed, mission_state)
        agent_obs = obs[0]
        
        # 构建图结构（与 get_obs_relational 相同）
        data_dict = {('agent', 'talks', 'agent'): ([], [])}
        num_nodes_dict = {'agent': 1}
        feat = {'agent': torch.as_tensor(agent_obs['agent'], dtype=torch.float).unsqueeze(0)}
        
        # 处理邻居节点
        nbr_data = agent_obs['nbr']
        ent_ids = np.equal(nbr_data[:, 0], 1)
        n_ents = ent_ids.sum() # 统计有效邻居的数量
        
        if n_ents > 0:
            data_dict[('nbr', 'nearby', 'agent')] = (
                torch.arange(n_ents),#生成从 0 到 n_ents 的整数序列。 # 源节点索引
                torch.zeros(n_ents, dtype=torch.long)  # 目标节点索引（agent 节点，索引为0）
            )
            num_nodes_dict['nbr'] = n_ents
            feat['nbr'] = torch.as_tensor(nbr_data[ent_ids, 1:], dtype=torch.float)
        else:
            num_nodes_dict['nbr'] = 0
            nbr_feat_dim = nbr_data.shape[1] - 1
            feat['nbr'] = torch.zeros((0, nbr_feat_dim), dtype=torch.float)
        
        # 创建 PyG 图
        graph = pyg_compat.heterograph(data_dict, num_nodes_dict=num_nodes_dict)
        pyg_compat.set_ndata(graph, 'feat', feat)
        """
                data_dict = {
            ('nbr', 'nearby', 'agent'): ([0, 1], [0, 0])  # B→A, C→A
        }

        num_nodes_dict = {'agent': 1, 'nbr': 2}

        feat = {
            'agent': tensor([[computing, memory, state]]),     # A的特征
            'nbr': tensor([[B_feats...], [C_feats...]])        # B和C的特征
        }
        
        """
        
        # 任务上下文
        task_context = torch.as_tensor(agent_obs['task'], dtype=torch.float)
        
        return graph, task_context

    def get_obs_graph_separated(self, destination, hops, is_computed, mission_state, khops: int = 1):
        """
        【任务特征分离版本】获取 graph 模式的 PyG 图观测
        
        与 get_obs_graph() 的区别：
        - agent 节点特征只包含纯环境信息（3维）
        - 边特征分为环境特征(edata)和任务特征(task_edata)
        - task 特征作为额外返回值，用于后续与 GNN 输出融合
        
        Args:
            destination: 目标节点
            hops: 当前跳数
            is_computed: 是否已计算
            mission_state: 任务状态
            khops: GNN 感知的邻居范围（1 或 2）
        
        Returns:
            tuple: (HeteroData 图, task_context tensor, task_edge_feats dict)
                   task_edge_feats: {'1hop': tensor} - 与边顺序一致的 hop_distance
        """
        # 调用分离版本的 get_graph_inputs
        obs_relations = self.get_graph_inputs_separated(destination, hops, is_computed, mission_state, khops)
        """
        obs_relations = {
            'graph_data': {
                ('nbr', '1hop', 'agent'): ([0, 1, 2, 3], [0, 0, 0, 0]),
                ('nbr', '2hop', 'nbr'):   ([4, 5, 6, 7], [0, 0, 1, 2])
            },
            'num_nodes_dict': {'agent': 1, 'nbr': 8},
            'ndata': {
                'agent': [[0, 0.8, -0.92]],  # 纯环境
                'nbr': [[...], [...], ...]    # 8个邻居的状态
            },
            'edata': {
                '1hop': [[0.3], [0.5], [0.1], [0.2]],  # 环境：transmission_size
                '2hop': [[0.5], [0.5], [0.4], [0.5]]   # 环境：transmission_size
            },
            'task_edata': {
                '1hop': [[0.3], [0.1], [2.0], [0.5]]   # 任务：hop_distance
                # 注意：2hop 没有任务边特征！
            },
            'task': [1, 0.5, 0.8, 0.2, 0.2, 0]  # 任务全局特征
        }
        
        
        
        """
        graph_data = obs_relations['graph_data']
        num_nodes_dict = obs_relations['num_nodes_dict']
        node_feats = obs_relations['ndata']
        edge_feats = obs_relations.get('edata')        # 环境边特征
        task_edge_feats = obs_relations.get('task_edata')  # 任务边特征 (hop_distance)
        task_context = obs_relations['task']
        
        # 创建 PyG 图
        obs_graph = pyg_compat.heterograph(graph_data, num_nodes_dict=num_nodes_dict)
        """
        # PyG HeteroData 对象
        obs_graph = HeteroData(
            # 节点类型
            agent={
                'num_nodes': 1,
                'feat': None  # 稍后添加
            },
            nbr={
                'num_nodes': 8,
                'feat': None  # 稍后添加
            },
            
            # 边类型
            ('nbr', '1hop', 'agent')={
                'edge_index': tensor([[0, 1, 2, 3],    # 源节点索引（邻居）
                                    [0, 0, 0, 0]]),  # 目标节点索引（agent）
                'edge_attr': None  # 稍后添加
            },
            ('nbr', '2hop', 'nbr')={
                'edge_index': tensor([[4, 5, 6, 7],    # 源节点索引（2跳邻居）
                                    [0, 0, 1, 2]]),  # 目标节点索引（1跳邻居）
                'edge_attr': None  # 稍后添加
            }
        )       
        """
        
        # 添加节点特征
        for ntype in obs_graph.node_types:
            if ntype in node_feats:
                obs_graph[ntype].feat = torch.as_tensor(node_feats[ntype], dtype=torch.float)
        """
        obs_graph = HeteroData(
            agent={
                'num_nodes': 1,
                'feat': tensor([[0.0, 0.8, -0.92]])  # shape: (1, 3)
                #               [is_producing, memory_remain, computing_remain]
            },
            nbr={
                'num_nodes': 8,
                'feat': tensor([
                    [0.0, 0.6, 0.7, 0.0],  # sat_B (idx 0)
                    [1.0, 0.4, 0.9, 0.0],  # sat_C (idx 1)
                    [0.0, 0.5, 0.8, 1.0],  # sat_D (idx 2)
                    [0.0, 0.7, 0.6, 0.0],  # sat_E (idx 3)
                    [0.0, 0.5, 0.5, 0.0],  # sat_F (idx 4, 2跳)
                    [0.0, 0.5, 0.5, 0.0],  # sat_G (idx 5, 2跳)
                    [1.0, 0.3, 0.9, 0.0],  # sat_H (idx 6, 2跳)
                    [0.0, 0.5, 0.5, 0.0]   # sat_I (idx 7, 2跳)
                ])  # shape: (8, 4)
            },
            ...
        )
        agent 节点：3维纯环境特征（无任务信息）
        nbr 节点：4维节点状态（旧模式）或 12维（新模式）
        所有邻居（1跳+2跳）共享相同的特征维度
                
        """
        
        # 添加环境边特征
        if edge_feats is not None:
            for etype in obs_graph.edge_types: # [('nbr', '1hop', 'agent'), ('nbr', '2hop', 'nbr')]
                edge_type_key = etype[1] if isinstance(etype, tuple) else etype# 提取边类型名称：'1hop' 或 '2hop'
                if edge_type_key in edge_feats and len(edge_feats[edge_type_key]) > 0:
                    obs_graph[etype].edge_attr = torch.as_tensor(edge_feats[edge_type_key], dtype=torch.float)
        
        """
        obs_graph = HeteroData(
        ...
        ('nbr', '1hop', 'agent')={
            'edge_index': tensor([[0, 1, 2, 3],
                                [0, 0, 0, 0]]),
            'edge_attr': tensor([
                [0.3],  # sat_B -> agent: transmission_size
                [0.5],  # sat_C -> agent: transmission_size (拥塞)
                [0.1],  # sat_D -> agent: transmission_size
                [0.2]   # sat_E -> agent: transmission_size
            ])  # shape: (4, 1)
        },
        ('nbr', '2hop', 'nbr')={
            'edge_index': tensor([[4, 5, 6, 7],
                                [0, 0, 1, 2]]),
            'edge_attr': tensor([
                [0.5],  # sat_F -> sat_B
                [0.5],  # sat_G -> sat_B
                [0.4],  # sat_H -> sat_C
                [0.5]   # sat_I -> sat_D
            ])  # shape: (4, 1)
        }
    )
        只包含环境信息：transmission_size（拥塞程度）
        不包含任务信息：hop_distance（被分离出去了）
        1跳边特征：4条边 × 1维 = (4, 1)
        2跳边特征：4条边 × 1维 = (4, 1)
        """
        
        # 压缩图
        obs_graph = pyg_compat.compact_graphs(
            obs_graph, 
            always_preserve=dict(agent=torch.arange(1)),
            copy_ndata=True, 
            copy_edata=True
        )
        
        # 任务上下文
        task_tensor = torch.as_tensor(task_context, dtype=torch.float)
        """
        task_tensor = tensor([1.0, 0.5, 0.8, 0.2, 0.2, 0.0])  # shape: (6,)
                    [type, size, comp_demand, size_after, hops, is_computed]

        独立于图结构的全局任务信息
        在决策时会与每个邻居的特征拼接
        """
        
        # 任务边特征（hop_distance）- 转为 tensor
        task_edge_tensors = {}
        if task_edge_feats is not None:
            for key, val in task_edge_feats.items():
                task_edge_tensors[key] = torch.as_tensor(val, dtype=torch.float)
        """
            task_edge_tensors = {
            '1hop': tensor([
                [0.3],  # sat_B -> agent: hop_distance (3跳/10)
                [0.1],  # sat_C -> agent: hop_distance (1跳/10) 【最优】
                [2.0],  # sat_D -> agent: hop_distance (不可达)
                [0.5]   # sat_E -> agent: hop_distance (5跳/10)
            ])  # shape: (4, 1)
        }
        """
        
        return obs_graph, task_tensor, task_edge_tensors

    def get_graph_inputs_separated(self, destination, hops, is_computed, mission_state, khops: int = 1):
        """
        【任务特征分离版本】获取图结构输入
        
        与 get_graph_inputs() 的区别：
        - agent 节点特征只包含纯环境信息（3维）
        - task 特征独立返回
        
        返回:
            dict: 图结构数据
            {
                'graph_data': 边定义,
                'num_nodes_dict': 节点数量,
                'ndata': 节点特征（agent只有3维环境特征）,
                'edata': 边特征,
                'task': 任务上下文 [6维]
            }
        """
        max_nbrs = 4
        n_valid_nbr = min(len(self.neighbors), max_nbrs)
        
        # === 1. 收集所有相关节点 ===
        all_node_names = set(self.neighbors[:n_valid_nbr])
        
        nbrs2_per_nbr = None
        nbrs3_per_hop2 = None
        
        if khops >= 2 or self.n_hops >= 2:
            nbrs2_per_nbr = self._find_2hop_neighbors(max_nbrs=n_valid_nbr, max_hop2_per_nbr=4)
            for hop2_neighbors in nbrs2_per_nbr:
                all_node_names.update(hop2_neighbors)
        
        if khops >= 3 or self.n_hops >= 3:
            # 收集所有3跳邻居
            nbrs3_per_hop2 = self._find_3hop_neighbors(max_nbrs=n_valid_nbr, max_hop2_per_nbr=4, max_hop3_per_hop2=4)
            for hop1_hop3_list in nbrs3_per_hop2:
                for hop3_neighbors in hop1_hop3_list:
                    all_node_names.update(hop3_neighbors)
        
        all_node_names = sorted(list(all_node_names))
        node_name_to_idx = {name: idx for idx, name in enumerate(all_node_names)}
        n_total_nodes = len(all_node_names)
        
        # === 2. 节点数量 ===
        num_nodes_dict = {'agent': 1, 'nbr': n_total_nodes}
        
        # === 3. Agent 节点特征（只有环境信息，不包含任务）===
        current_node_state = np.array([
            self.is_producing, 
            1 - self.current_memory_occupy / self.memory,
            (self.computing_remain / self.computing_ability - 
             self.is_computing * (self.env.now - self.last_computing_time)) / CT_FAC
        ], dtype=np.float32)
        
        agent_env_feats = current_node_state  # [3维] - 纯环境特征
        
        # === 4. 任务上下文（独立返回）===
        # task_context: 只包含基础任务信息，不包含邻居的 hop_distance
        # 因为 hop_distance 是边级信息，需要单独传递以保证与边顺序一致
        routing_info = np.array([hops / self.max_hop, is_computed], dtype=np.float32)
        task_context = np.concatenate([mission_state, routing_info])  # [6维]
        
        # === 5. 邻居节点特征 ===
        if 'New' in self.mode:
            nbr_feat_dim = 12
        else:
            nbr_feat_dim = 4
        
        nbr_feats = np.zeros((n_total_nodes, nbr_feat_dim), dtype=np.float32)
        
        for idx, node_name in enumerate(all_node_names):
            if node_name in self.neighbor_states:
                if 'New' in self.mode:
                    state = self.neighbor_states[node_name]
                    nbr_feats[idx, 0:4] = state[0:4]
                    nbr_feats[idx, 4:8] = [x/4 for x in state[4:8]]
                    nbr_feats[idx, 8:12] = [x/12 for x in state[8:12]]
                else:
                    nbr_feats[idx, :] = self.neighbor_states[node_name]
            else:
                if 'New' in self.mode:
                    nbr_feats[idx, :] = [0, 0.5, 0.5, 0] * 3
                else:
                    nbr_feats[idx, :] = [0, 0.5, 0.5, 0]
        
        # === 6. 边定义与边特征 ===
        # 【分离模式】边特征分为两部分：
        #   - env_edge_feats: 纯环境信息 [transmission_size]，用于 GNN 编码，可交换
        #   - task_edge_feats: 任务相关 [hop_distance]，只在决策时使用，不交换
        
        graph_data = {}
        hop1_env_feats = []    # 环境边特征
        hop1_task_feats = []   # 任务边特征 (hop_distance)
        src_1hop = []
        dst_1hop = []
        
        for neighbor in self.neighbors[:n_valid_nbr]:
            nbr_idx = node_name_to_idx[neighbor]
            src_1hop.append(nbr_idx)
            dst_1hop.append(0)
            
            # 环境边特征: transmission_size
            hop1_env_feats.append([self.transmission_size[neighbor] / self.memory])
            
            # 任务边特征: hop_distance（与边顺序一致！）
            if destination in self.neighbor_hops[neighbor]:
                hop_dist = self.neighbor_hops[neighbor][destination] / self.max_hop
            else:
                hop_dist = 2.0  # 不可达标记
            hop1_task_feats.append([hop_dist])
        
        graph_data[('nbr', '1hop', 'agent')] = (src_1hop, dst_1hop)
        
        edge_feats = {
            '1hop': np.array(hop1_env_feats, dtype=np.float32) if hop1_env_feats else np.zeros((0, 1), dtype=np.float32)
        }
        # 任务相关的边特征（hop_distance），单独存储
        task_edge_feats = {
            '1hop': np.array(hop1_task_feats, dtype=np.float32) if hop1_task_feats else np.zeros((0, 1), dtype=np.float32)
        }
        
        # 2hop 边（只有环境特征，不需要任务特征）
        if khops >= 2 or self.n_hops >= 2:
            nbrs2_per_nbr = self._find_2hop_neighbors(max_nbrs=n_valid_nbr, max_hop2_per_nbr=3)
            max_hop2 = 3
            
            hop2_src = []
            hop2_dst = []
            hop2_env_feats = []
            
            # for nbr_idx_local, neighbor in enumerate(self.neighbors[:n_valid_nbr]):
            #     hop2_neighbors = nbrs2_per_nbr[nbr_idx_local]
            #     nbr_idx = node_name_to_idx[neighbor]
                
            #     valid_hop2_count = 0
            #     for hop2_nbr in hop2_neighbors:
            #         if hop2_nbr not in node_name_to_idx:
            #             continue
            #         if valid_hop2_count >= max_hop2:
            #             break
                    
            #         hop2_nbr_idx = node_name_to_idx[hop2_nbr]
            #         hop2_src.append(hop2_nbr_idx)
            #         hop2_dst.append(nbr_idx)
                    
            #         # 2hop 边特征只包含 transmission_size（纯环境信息）
            #         if hop2_nbr in self.transmission_size:
            #             hop2_env_feats.append([self.transmission_size[hop2_nbr] / self.memory])
            #         else:
            #             hop2_env_feats.append([0.5])
            #         valid_hop2_count += 1
            for nbr_idx_local, neighbor in enumerate(self.neighbors[:n_valid_nbr]):
                #  获取1跳邻居对象（这一步之前缺失！）
                neighbor_obj = self.propagator.satellites.get(neighbor)
                
                if neighbor_obj is None:
                    continue  # 无法获取邻居对象，跳过
                
                hop2_neighbors = nbrs2_per_nbr[nbr_idx_local]  # 名称列表
                nbr_idx = node_name_to_idx[neighbor]
                
                valid_hop2_count = 0
                for hop2_nbr in hop2_neighbors:  # hop2_nbr 是字符串名称
                    if hop2_nbr not in node_name_to_idx:
                        continue
                    if valid_hop2_count >= max_hop2:
                        break
                    
                    hop2_nbr_idx = node_name_to_idx[hop2_nbr]
                    hop2_src.append(hop2_nbr_idx)
                    hop2_dst.append(nbr_idx)
                    
                    # 正确：从1跳邻居对象获取它到2跳邻居的传输队列
                    if hop2_nbr in neighbor_obj.transmission_size:
                        hop2_env_feats.append([neighbor_obj.transmission_size[hop2_nbr] / neighbor_obj.memory])
                    else:
                        hop2_env_feats.append([0.5])
                    
                    valid_hop2_count += 1
            
            if hop2_src:
                graph_data[('nbr', '2hop', 'nbr')] = (hop2_src, hop2_dst)
                edge_feats['2hop'] = np.array(hop2_env_feats, dtype=np.float32)
            else:
                edge_feats['2hop'] = np.zeros((0, 1), dtype=np.float32)
        
        # 3hop 边（如果配置了 khops >= 3）
        if khops >= 3 or self.n_hops >= 3:
            if nbrs2_per_nbr is None:
                nbrs2_per_nbr = self._find_2hop_neighbors(max_nbrs=n_valid_nbr, max_hop2_per_nbr=4)
            if nbrs3_per_hop2 is None:
                nbrs3_per_hop2 = self._find_3hop_neighbors(max_nbrs=n_valid_nbr, max_hop2_per_nbr=4, max_hop3_per_hop2=4)
            max_hop3 = 4
            
            hop3_src = []
            hop3_dst = []
            hop3_env_feats = []
            
            for nbr_idx_local, neighbor in enumerate(self.neighbors[:n_valid_nbr]):
                # 获取1跳邻居对象
                neighbor_obj = self.propagator.satellites.get(neighbor)
                
                if neighbor_obj is None:
                    continue
                
                hop2_neighbors = nbrs2_per_nbr[nbr_idx_local]
                hop3_for_this_hop1 = nbrs3_per_hop2[nbr_idx_local] if nbr_idx_local < len(nbrs3_per_hop2) else []
                
                for hop2_idx_local, hop2_nbr in enumerate(hop2_neighbors):
                    if hop2_nbr not in node_name_to_idx:
                        continue
                    
                    # 获取2跳邻居对象
                    hop2_obj = self.propagator.satellites.get(hop2_nbr)
                    hop2_nbr_idx = node_name_to_idx[hop2_nbr]
                    
                    hop3_neighbors = hop3_for_this_hop1[hop2_idx_local] if hop2_idx_local < len(hop3_for_this_hop1) else []
                    
                    valid_hop3_count = 0
                    for hop3_nbr in hop3_neighbors:
                        if hop3_nbr not in node_name_to_idx:
                            continue
                        if valid_hop3_count >= max_hop3:
                            break
                        
                        hop3_nbr_idx = node_name_to_idx[hop3_nbr]
                        
                        # 边：3hop邻居 -> 2hop邻居
                        hop3_src.append(hop3_nbr_idx)
                        hop3_dst.append(hop2_nbr_idx)
                        
                        # 3hop 边特征: transmission_size（纯环境信息）
                        if hop2_obj is not None and hop3_nbr in hop2_obj.transmission_size:
                            hop3_env_feats.append([hop2_obj.transmission_size[hop3_nbr] / hop2_obj.memory])
                        else:
                            hop3_env_feats.append([0.5])
                        
                        valid_hop3_count += 1
            
            if hop3_src:
                graph_data[('nbr', '3hop', 'nbr')] = (hop3_src, hop3_dst)
                edge_feats['3hop'] = np.array(hop3_env_feats, dtype=np.float32)
            else:
                edge_feats['3hop'] = np.zeros((0, 1), dtype=np.float32)
        
        return {
            'graph_data': graph_data,
            'num_nodes_dict': num_nodes_dict,
            'ndata': {
                'agent': np.expand_dims(agent_env_feats, 0),  # shape: (1, 3) - 纯环境
                'nbr': nbr_feats
            },
            'edata': edge_feats,           # 环境边特征 (transmission_size)
            'task_edata': task_edge_feats, # 任务边特征 (hop_distance) - 与边顺序一致
            'task': task_context           # [6维] - 基础任务上下文
        }
    
    def get_graph_inputs(self, destination, hops, is_computed, mission_state, khops: int = 1):
        """
        获取图结构输入，与 cross_layer_opt_with_grl-main 的 get_graph_inputs() 保持一致
        
        用于 GraphObservation 包装器，返回完整的图结构数据。
        
        Args:
            destination: 目标节点
            hops: 当前跳数
            is_computed: 是否已计算
            mission_state: 任务状态
            khops: GNN 感知的邻居范围（1、2 或 3），默认1
        
        Returns:
            dict: 图结构数据
            {
                'graph_data': 边定义 {(src_type, edge_type, dst_type): (src_ids, dst_ids)},
                'num_nodes_dict': 节点数量 {'agent': 1, 'nbr': N},
                'ndata': 节点特征 {'agent': array, 'nbr': array},
                'edata': 边特征 {'1hop': array, '2hop': array, '3hop': array} (可选)
            }
        
        图结构示意（khops=3）:
        
            [Agent-0]
              ↑     ↑
           1hop   1hop
            ↑       ↑
         [Nbr-0] [Nbr-1] ...(1跳邻居包括所有卫星节点)
           ↑ ↑     ↑ ↑
          2hop    2hop
           ↑       ↑
         [2跳邻居节点]
           ↑ ↑     ↑ ↑
          3hop    3hop
           ↑       ↑
         [3跳邻居节点]
        """
        max_nbrs = 4
        n_valid_nbr = min(len(self.neighbors), max_nbrs)
        
        # === 1. 收集所有相关节点 ===
        # 包括1跳邻居和（可选的）2跳、3跳邻居
        all_node_names = set(self.neighbors[:n_valid_nbr])  # 1跳邻居
        
        nbrs2_per_nbr = None
        nbrs3_per_hop2 = None
        
        if khops >= 2 or self.n_hops >= 2:
            # 收集所有2跳邻居
            nbrs2_per_nbr = self._find_2hop_neighbors(max_nbrs=n_valid_nbr, max_hop2_per_nbr=4)
            for hop2_neighbors in nbrs2_per_nbr:
                all_node_names.update(hop2_neighbors)
        
        if khops >= 3 or self.n_hops >= 3:
            # 收集所有3跳邻居
            nbrs3_per_hop2 = self._find_3hop_neighbors(max_nbrs=n_valid_nbr, max_hop2_per_nbr=4, max_hop3_per_hop2=4)
            for hop1_hop3_list in nbrs3_per_hop2:
                for hop3_neighbors in hop1_hop3_list:
                    all_node_names.update(hop3_neighbors)
        
        # 转换为列表并排序（保证顺序稳定）
        all_node_names = sorted(list(all_node_names))
        node_name_to_idx = {name: idx for idx, name in enumerate(all_node_names)}
        n_total_nodes = len(all_node_names)
        """
        all_node_names = [
            'sat_550_0_1',  # idx 0 (sat_B)
            'sat_550_0_2',  # idx 1 (sat_C)
            'sat_550_0_3',  # idx 2 (sat_F, 2跳)
            'sat_550_0_4',  # idx 3 (sat_H, 2跳)
            'sat_550_1_0',  # idx 4 (sat_D)
            'sat_550_1_1',  # idx 5 (sat_E)
            'sat_550_1_2',  # idx 6 (sat_G, 2跳)
            'sat_550_2_0'   # idx 7 (sat_I, 2跳)
        ]

        n_total_nodes = 8

        node_name_to_idx = {
            'sat_550_0_1': 0,  # sat_B
            'sat_550_0_2': 1,  # sat_C
            'sat_550_0_3': 2,  # sat_F (2跳)
            'sat_550_0_4': 3,  # sat_H (2跳)
            'sat_550_1_0': 4,  # sat_D
            'sat_550_1_1': 5,  # sat_E
            'sat_550_1_2': 6,  # sat_G (2跳)
            'sat_550_2_0': 7   # sat_I (2跳)
        }
        """
        
        # === 2. 节点数量 ===
        num_nodes_dict = {'agent': 1, 'nbr': n_total_nodes}
        
        # === 3. Agent 节点特征 ===
        current_node_state = np.array([
            self.is_producing, 
            1 - self.current_memory_occupy / self.memory,
            (self.computing_remain / self.computing_ability - 
             self.is_computing * (self.env.now - self.last_computing_time)) / CT_FAC
        ], dtype=np.float32)
        routing_info = np.array([hops / self.max_hop, is_computed], dtype=np.float32)
        own_feats = np.concatenate([current_node_state, mission_state, routing_info])
        """
        current_node_state = [
            0,                                    # is_producing
            1 - 350/1000 = 0.65,                 # memory_remain (65%空闲)
            (0.6/1.0 - 1*(100-95))/10 = 0.01    # computing_remain
        ]

        # === 任务状态 (4维) ===
        mission_state = [
            1,                    # type (计算密集型)
            600/1000 = 0.6,      # size_norm (60%容量)
            0.7/1.0 = 0.7,       # computing_demand_norm (需要70%算力)
            250/1000 = 0.25      # size_after_norm (计算后25%)
        ]

        # === 路由信息 (2维) ===
        routing_info = [
            3/10 = 0.3,  # hops_norm (已转发3跳)
            0            # is_computed (尚未计算)
        ]

        前3维：当前卫星的环境状态（内存充足，正在计算）
        中4维：任务信息（计算密集型，中等大小，需要大量算力）
        后2维：路由信息（已转发3跳，尚未计算）
        """
        
        # === 4. 邻居节点特征 ===
        # 节点特征只包含节点自身状态，不包含边信息（transmission_size, hop_distance）
        if 'New' in self.mode:
            nbr_feat_dim = 12  # 12维状态
        else:
            nbr_feat_dim = 4   # 4维状态
        
        nbr_feats = np.zeros((n_total_nodes, nbr_feat_dim), dtype=np.float32)
        
        for idx, node_name in enumerate(all_node_names):
            if node_name in self.neighbor_states:
                if 'New' in self.mode:
                    state = self.neighbor_states[node_name]
                    nbr_feats[idx, 0:4] = state[0:4]
                    nbr_feats[idx, 4:8] = [x/4 for x in state[4:8]]
                    nbr_feats[idx, 8:12] = [x/12 for x in state[8:12]]
                else:
                    nbr_feats[idx, :] = self.neighbor_states[node_name]
            else:
                # 2跳邻居可能没有状态信息，使用默认值
                if 'New' in self.mode:
                    nbr_feats[idx, :] = [0, 0.5, 0.5, 0] * 3  # 默认状态
                else:
                    nbr_feats[idx, :] = [0, 0.5, 0.5, 0]  # 默认状态
        """
            nbr_feats = np.array([
            # === 1跳邻居（有完整状态信息）===
            [0.0, 0.7, 0.8, 0.0],  # idx 0: sat_B (空闲，资源充足)
            [1.0, 0.4, 0.9, 0.0],  # idx 1: sat_C (正在生产，内存紧张)
            
            # === 2跳邻居（无状态信息，使用默认值）===
            [0.0, 0.5, 0.5, 0.0],  # idx 2: sat_F (默认)
            [0.0, 0.5, 0.5, 0.0],  # idx 3: sat_H (默认)
            
            # === 1跳邻居 ===
            [0.0, 0.6, 0.7, 1.0],  # idx 4: sat_D (正在计算)
            [0.0, 0.8, 0.5, 0.0],  # idx 5: sat_E (算力较低)
            
            # === 2跳邻居 ===
            [0.0, 0.5, 0.5, 0.0],  # idx 6: sat_G (默认)
            [0.0, 0.5, 0.5, 0.0]   # idx 7: sat_I (默认)
        ])  # shape: (8, 4)
        #           [is_producing, memory_norm, computing_norm, is_computing]

        1跳邻居（idx 0,1,4,5）：有完整的状态信息
        2跳邻居（idx 2,3,6,7）：使用默认值 [0, 0.5, 0.5, 0]（中性状态）
        不包含边信息（transmission_size, hop_distance）
        
        """
        
        # === 5. 边定义与边特征 ===
        # 边特征维度：与 graph_feats['hop'] 一致
        hop_feat_dim = self.graph_feats.get('hop', 3)  # 默认3
        
        # 1hop 边: nbr -> agent
        graph_data = {}
        hop1_feats = []
        src_1hop = []
        dst_1hop = []
        
        for neighbor in self.neighbors[:n_valid_nbr]:#遍历1跳邻居
            nbr_idx = node_name_to_idx[neighbor]
            src_1hop.append(nbr_idx)
            dst_1hop.append(0)  # agent 索引为 0
            """
            src_1hop = [0, 1, 4, 5]  # [sat_B, sat_C, sat_D, sat_E] 的索引
            dst_1hop = [0, 0, 0, 0]  # 都指向 agent (索引0)
            """
            # 1hop 边特征: [transmission_size, hop_distance, placeholder]
            # 注意：graph 模式下边的存在本身就表示可见性，不需要 availability 标志
            h1_feats = np.zeros(hop_feat_dim, dtype=np.float32)
            h1_feats[0] = self.transmission_size[neighbor] / self.memory
            if destination in self.neighbor_hops[neighbor]:
                h1_feats[1] = self.neighbor_hops[neighbor][destination] / self.max_hop
            else:
                h1_feats[1] = 2
            h1_feats[2] = 0  # 预留（可扩展为信道质量等）
            hop1_feats.append(h1_feats)
        
        graph_data[('nbr', '1hop', 'agent')] = (src_1hop, dst_1hop)
        edge_feats = {
            '1hop': np.array(hop1_feats, dtype=np.float32) if hop1_feats else np.zeros((0, hop_feat_dim), dtype=np.float32)
        }
        """
            graph_data 
        # 1跳边：邻居 -> agent
        ('nbr', '1hop', 'agent'): (
            [0, 1, 4, 5],  # src: sat_B, sat_C, sat_D, sat_E
            [0, 0, 0, 0]   # dst: 都指向 agent
        ),
        
        # 2跳边：2跳邻居 -> 1跳邻居
        ('nbr', '2hop', 'nbr'): (
            [2, 6, 3, 7],  # src: sat_F, sat_G, sat_H, sat_I
            [0, 0, 1, 4]   # dst: sat_B, sat_B, sat_C, sat_D
        )
    }

        hop1_feats = [
            [0.4, 0.5, 0.0],  # sat_B: transmission=40%, hop_dist=5跳
            [0.7, 0.2, 0.0],  # sat_C: transmission=70%(拥塞), hop_dist=2跳【最短】
            [0.15, 2.0, 0.0], # sat_D: transmission=15%, 不可达(hop_dist=2.0)
            [0.3, 0.7, 0.0]   # sat_E: transmission=30%, hop_dist=7跳
        ]  # shape: (4, 3)

        edge_feats['1hop'] = np.array(hop1_feats)
        
        """
        
        # === 6. 2hop 边（如果配置了 khops >= 2）===
        if khops >= 2 or self.n_hops >= 2:
            nbrs2_per_nbr = self._find_2hop_neighbors(max_nbrs=n_valid_nbr, max_hop2_per_nbr=3)
            max_hop2 = 3  # 每个1跳邻居最多3个2跳邻居
            
            hop2_src = []
            hop2_dst = []
            hop2_feats = []
            
            for nbr_idx_local, neighbor in enumerate(self.neighbors[:n_valid_nbr]):
                # 获取1跳邻居对象（关键修复）
                neighbor_obj = self.propagator.satellites.get(neighbor) if hasattr(self, 'propagator') else None
                
                hop2_neighbors = nbrs2_per_nbr[nbr_idx_local]
                nbr_idx = node_name_to_idx[neighbor]
                
                # 处理有效的2跳邻居
                valid_hop2_count = 0
                for hop2_nbr in hop2_neighbors:
                    if hop2_nbr not in node_name_to_idx:
                        continue  # 跳过不在节点集合中的节点
                    if valid_hop2_count >= max_hop2:
                        break  # 限制每个1跳邻居最多3个2跳邻居
                    
                    hop2_nbr_idx = node_name_to_idx[hop2_nbr]
                    
                    # 边：2hop邻居 -> 1hop邻居
                    hop2_src.append(hop2_nbr_idx)
                    hop2_dst.append(nbr_idx)
                    
                    # 2hop 边特征: [transmission_size, hop_distance, placeholder]
                    h2_feats = np.zeros(hop_feat_dim, dtype=np.float32)
                    
                    # transmission_size: 从1跳邻居对象获取它到2跳邻居的传输队列
                    if neighbor_obj is not None and hasattr(neighbor_obj, 'transmission_size'):
                        if hop2_nbr in neighbor_obj.transmission_size:
                            h2_feats[0] = neighbor_obj.transmission_size[hop2_nbr] / neighbor_obj.memory
                        else:
                            h2_feats[0] = 0.5  # 邻居无此2跳邻居的队列信息
                    else:
                        h2_feats[0] = 0.5  # 无法访问邻居对象
                    
                    # hop_distance: 从1跳邻居对象获取2跳邻居到目标的跳数
                    if neighbor_obj is not None and hasattr(neighbor_obj, 'neighbor_hops'):
                        if hop2_nbr in neighbor_obj.neighbor_hops:
                            if destination in neighbor_obj.neighbor_hops[hop2_nbr]:
                                h2_feats[1] = neighbor_obj.neighbor_hops[hop2_nbr][destination] / self.max_hop
                            else:
                                h2_feats[1] = 2  # 2跳邻居无法到达目标
                        else:
                            h2_feats[1] = 1  # 邻居无此2跳邻居的路由信息
                    else:
                        h2_feats[1] = 1  # 无法访问邻居对象，使用默认距离
                    
                    h2_feats[2] = 0  # 预留
                    
                    hop2_feats.append(h2_feats)
                    valid_hop2_count += 1
            
            if hop2_src:
                graph_data[('nbr', '2hop', 'nbr')] = (hop2_src, hop2_dst)
                edge_feats['2hop'] = np.array(hop2_feats, dtype=np.float32)
            else:
                edge_feats['2hop'] = np.zeros((0, hop_feat_dim), dtype=np.float32)
        
        # === 7. 3hop 边（如果配置了 khops >= 3）===
        if khops >= 3 or self.n_hops >= 3:
            if nbrs2_per_nbr is None:
                nbrs2_per_nbr = self._find_2hop_neighbors(max_nbrs=n_valid_nbr, max_hop2_per_nbr=4)
            if nbrs3_per_hop2 is None:
                nbrs3_per_hop2 = self._find_3hop_neighbors(max_nbrs=n_valid_nbr, max_hop2_per_nbr=4, max_hop3_per_hop2=4)
            max_hop3 = 4  # 每个2跳邻居最多4个3跳邻居
            
            hop3_src = []
            hop3_dst = []
            hop3_feats = []
            
            for nbr_idx_local, neighbor in enumerate(self.neighbors[:n_valid_nbr]):
                # 获取1跳邻居对象
                neighbor_obj = self.propagator.satellites.get(neighbor) if hasattr(self, 'propagator') else None
                
                hop2_neighbors = nbrs2_per_nbr[nbr_idx_local]
                hop3_for_this_hop1 = nbrs3_per_hop2[nbr_idx_local] if nbr_idx_local < len(nbrs3_per_hop2) else []
                
                for hop2_idx_local, hop2_nbr in enumerate(hop2_neighbors):
                    if hop2_nbr not in node_name_to_idx:
                        continue
                    
                    # 获取2跳邻居对象
                    hop2_obj = self.propagator.satellites.get(hop2_nbr) if hasattr(self, 'propagator') else None
                    hop2_nbr_idx = node_name_to_idx[hop2_nbr]
                    
                    hop3_neighbors = hop3_for_this_hop1[hop2_idx_local] if hop2_idx_local < len(hop3_for_this_hop1) else []
                    
                    valid_hop3_count = 0
                    for hop3_nbr in hop3_neighbors:
                        if hop3_nbr not in node_name_to_idx:
                            continue
                        if valid_hop3_count >= max_hop3:
                            break
                        
                        hop3_nbr_idx = node_name_to_idx[hop3_nbr]
                        
                        # 边：3hop邻居 -> 2hop邻居
                        hop3_src.append(hop3_nbr_idx)
                        hop3_dst.append(hop2_nbr_idx)
                        
                        # 3hop 边特征: [transmission_size, hop_distance, placeholder]
                        h3_feats = np.zeros(hop_feat_dim, dtype=np.float32)
                        
                        # transmission_size: 从2跳邻居对象获取它到3跳邻居的传输队列
                        if hop2_obj is not None and hasattr(hop2_obj, 'transmission_size'):
                            if hop3_nbr in hop2_obj.transmission_size:
                                h3_feats[0] = hop2_obj.transmission_size[hop3_nbr] / hop2_obj.memory
                            else:
                                h3_feats[0] = 0.5  # 邻居无此3跳邻居的队列信息
                        else:
                            h3_feats[0] = 0.5  # 无法访问邻居对象
                        
                        # hop_distance: 从2跳邻居对象获取3跳邻居到目标的跳数
                        if hop2_obj is not None and hasattr(hop2_obj, 'neighbor_hops'):
                            if hop3_nbr in hop2_obj.neighbor_hops:
                                if destination in hop2_obj.neighbor_hops[hop3_nbr]:
                                    h3_feats[1] = hop2_obj.neighbor_hops[hop3_nbr][destination] / self.max_hop
                                else:
                                    h3_feats[1] = 2  # 3跳邻居无法到达目标
                            else:
                                h3_feats[1] = 1  # 邻居无此3跳邻居的路由信息
                        else:
                            h3_feats[1] = 1  # 无法访问邻居对象，使用默认距离
                        
                        h3_feats[2] = 0  # 预留
                        
                        hop3_feats.append(h3_feats)
                        valid_hop3_count += 1
            
            if hop3_src:
                graph_data[('nbr', '3hop', 'nbr')] = (hop3_src, hop3_dst)
                edge_feats['3hop'] = np.array(hop3_feats, dtype=np.float32)
            else:
                edge_feats['3hop'] = np.zeros((0, hop_feat_dim), dtype=np.float32)
        
        return {
            'graph_data': graph_data,
            'num_nodes_dict': num_nodes_dict,
            'ndata': {
                'agent': np.expand_dims(own_feats, 0),  # shape: (1, feat_dim)
                'nbr': nbr_feats  # shape: (n_total_nodes, feat_dim)
            },
            'edata': edge_feats
        }


        """
            return {
        # === 图拓扑结构 ===
        'graph_data': {
            ('nbr', '1hop', 'agent'): ([0, 1, 4, 5], [0, 0, 0, 0]),
            ('nbr', '2hop', 'nbr'):   ([2, 6, 3, 7], [0, 0, 1, 4])
        },
        
        # === 节点数量 ===
        'num_nodes_dict': {
            'agent': 1,
            'nbr': 8
        },
        
        # === 节点特征 ===
        'ndata': {
            # Agent 特征 (9维: 环境3 + 任务4 + 路由2)
            'agent': array([[
                0.0,   # is_producing
                0.65,  # memory_remain
                0.01,  # computing_remain
                1.0,   # task_type
                0.6,   # task_size
                0.7,   # computing_demand
                0.25,  # size_after
                0.3,   # hops
                0.0    # is_computed
            ]]),  # shape: (1, 9)
            
            # 邻居特征 (4维: 节点状态)
            'nbr': array([
                [0.0, 0.7, 0.8, 0.0],  # sat_B
                [1.0, 0.4, 0.9, 0.0],  # sat_C
                [0.0, 0.5, 0.5, 0.0],  # sat_F (2跳)
                [0.0, 0.5, 0.5, 0.0],  # sat_H (2跳)
                [0.0, 0.6, 0.7, 1.0],  # sat_D
                [0.0, 0.8, 0.5, 0.0],  # sat_E
                [0.0, 0.5, 0.5, 0.0],  # sat_G (2跳)
                [0.0, 0.5, 0.5, 0.0]   # sat_I (2跳)
            ])  # shape: (8, 4)
        },
        
        # === 边特征 ===
        'edata': {
            # 1跳边特征 (3维: transmission, hop_distance, placeholder)
            '1hop': array([
                [0.4, 0.5, 0.0],   # sat_B -> agent
                [0.7, 0.2, 0.0],   # sat_C -> agent 【最优路径】
                [0.15, 2.0, 0.0],  # sat_D -> agent (不可达)
                [0.3, 0.7, 0.0]    # sat_E -> agent
            ]),  # shape: (4, 3)
            
            # 2跳边特征
            '2hop': array([
                [0.5, 0.4, 0.0],  # sat_F -> sat_B
                [0.5, 1.0, 0.0],  # sat_G -> sat_B
                [0.5, 1.0, 0.0],  # sat_H -> sat_C
                [0.5, 1.0, 0.0]   # sat_I -> sat_D
            ])   # shape: (4, 3)
        }
    }
              
        """

    def get_obs_relational(self, destination, hops, is_computed, mission_state, khops: int = 1):
        """
        获取 relational 模式的 DGL 图观测（无边特征）
        
        与 cross_layer_opt_with_grl 的 RelationalObservation 一致：
        - 边信息（如 transmission_size, hop_distance）拼接到邻居节点特征中
        - 图的 edata 为空
        
        Args:
            destination: 目标节点
            hops: 当前跳数
            is_computed: 是否已计算
            mission_state: 任务状态
            khops: GNN 感知的邻居范围（1、2 或 3）
        
        Returns:
            torch_geometric.data.HeteroData: 异构图，只有节点特征，无边特征
        """
        # 获取包含边信息的邻居特征（get_obs 返回的格式）
        obs = self.get_obs(destination, hops, is_computed, mission_state)
        agent_obs = obs[0]  # 单智能体
        
        # 构建图结构
        data_dict = {('agent', 'talks', 'agent'): ([], [])}  # Agent 自连边（占位）
        num_nodes_dict = {'agent': 1}
        feat = {'agent': torch.as_tensor(agent_obs['agent'], dtype=torch.float).unsqueeze(0)}
        
        # 处理邻居节点
        nbr_data = agent_obs['nbr']  # shape: (max_nbrs, nbr_feat_dim)
        
        # 筛选可用邻居（第0列是 availability）
        ent_ids = np.equal(nbr_data[:, 0], 1)
        n_ents = ent_ids.sum()
        
        if n_ents > 0:
            # 边：nbr -> agent
            data_dict[('nbr', 'nearby', 'agent')] = (
                torch.arange(n_ents),  # 源节点 ID
                torch.zeros(n_ents, dtype=torch.long)  # 目标节点 ID（agent 索引为 0）
            )
            num_nodes_dict['nbr'] = n_ents
            # 邻居特征：去除第0列（availability），保留其他所有特征（包括边信息）
            feat['nbr'] = torch.as_tensor(nbr_data[ent_ids, 1:], dtype=torch.float)
        else:
            # 没有有效邻居时，创建空节点
            num_nodes_dict['nbr'] = 0
            nbr_feat_dim = nbr_data.shape[1] - 1  # 去除 availability 后的维度
            feat['nbr'] = torch.zeros((0, nbr_feat_dim), dtype=torch.float)
        
        # 创建 PyG 异构图（无边特征）
        graph = pyg_compat.heterograph(data_dict, num_nodes_dict=num_nodes_dict)
        pyg_compat.set_ndata(graph, 'feat', feat)
        # edata 为空，与 RelationalObservation 一致
        
        return graph
    
    def get_obs_graph(self, destination, hops, is_computed, mission_state, khops: int = 1):
        """
        获取 PyG 图格式的观测，参考 cross_layer_opt_with_grl 的 GraphObservation.get_obs()
        
        直接返回 PyG 异构图，可直接存入经验回放区或输入到 GNN。
        
        Args:
            destination: 目标节点
            hops: 当前跳数
            is_computed: 是否已计算
            mission_state: 任务状态
            khops: GNN 感知的邻居范围(1、2 或 3)
        
        Returns:
            torch_geometric.data.HeteroData: 异构图，包含节点特征和边特征
        """
        # 1. 获取图结构数据
        obs_relations = self.get_graph_inputs(destination, hops, is_computed, mission_state, khops)
        graph_data = obs_relations['graph_data']
        num_nodes_dict = obs_relations['num_nodes_dict']
        node_feats = obs_relations['ndata']
        edge_feats = obs_relations.get('edata')
        
        # 2. 创建 PyG 异构图
        obs_graph = pyg_compat.heterograph(graph_data, num_nodes_dict=num_nodes_dict)
        
        # 3. 添加节点特征
        for ntype in obs_graph.node_types:
            if ntype in node_feats:
                obs_graph[ntype].feat = torch.as_tensor(
                    node_feats[ntype], dtype=torch.float
                )
        
        # 4. 添加边特征
        if edge_feats is not None:
            for etype in obs_graph.edge_types:
                edge_type_key = etype[1] if isinstance(etype, tuple) else etype
                if edge_type_key in edge_feats and len(edge_feats[edge_type_key]) > 0:
                    obs_graph[etype].edge_attr = torch.as_tensor(
                        edge_feats[edge_type_key], dtype=torch.float
                    )
        
        # 5. 压缩图（移除孤立节点），保留 agent 节点
        obs_graph = pyg_compat.compact_graphs(
            obs_graph, 
            always_preserve=dict(agent=torch.arange(1)),  # 保留 agent 节点
            copy_ndata=True, 
            copy_edata=True
        )
        
        return obs_graph
    
    def _find_2hop_neighbors(self, max_nbrs: int = 4, max_hop2_per_nbr: int = 3):
        """
        获取2跳邻居（每个1跳邻居的邻居列表）
        
        参考 cross_layer_opt_with_grl-main 的 _find_2hop_neighbors()
        
        Args:
            max_nbrs: 最大1跳邻居数
            max_hop2_per_nbr: 每个1跳邻居最多保留的2跳邻居数
        
        Returns:
            nbrs2_per_nbr: list[list[str]] 
                每个1跳邻居对应一个2跳邻居名称列表
                nbrs2_per_nbr[i] = [hop2_name1, hop2_name2, ...]
        
        Note:
            2跳邻居不直接添加到路由中，因此不强制要求资格检查。
            过滤规则：
            - 排除当前节点自身
            - 排除已在1跳邻居列表中的节点（避免重复）
        """
        nbrs2_per_nbr = []
        hop1_neighbors = self.neighbors[:max_nbrs]
        
        for hop1_nbr in hop1_neighbors:
            nbrs2 = []
            # 从邻接表获取该1跳邻居的邻居列表
            if hop1_nbr in self.adjacency_table:
                hop1_nbr_neighbors, _ = self.adjacency_table[hop1_nbr]
                
                for hop2_nbr in hop1_nbr_neighbors:
                    # 过滤：排除自身和1跳邻居
                    if hop2_nbr != self.name and hop2_nbr not in hop1_neighbors:
                        nbrs2.append(hop2_nbr)
                    # 达到最大数量限制
                    if len(nbrs2) >= max_hop2_per_nbr:
                        break
            
            nbrs2_per_nbr.append(nbrs2)
        
        return nbrs2_per_nbr
    
    def _find_3hop_neighbors(self, max_nbrs: int = 4, max_hop2_per_nbr: int = 4, max_hop3_per_hop2: int = 4):
        """
        获取3跳邻居（每个2跳邻居的邻居列表）
        
        Args:
            max_nbrs: 最大1跳邻居数
            max_hop2_per_nbr: 每个1跳邻居最多保留的2跳邻居数
            max_hop3_per_hop2: 每个2跳邻居最多保留的3跳邻居数
        
        Returns:
            nbrs3_per_hop2: list[list[list[str]]]
                三层嵌套结构: nbrs3_per_hop2[i][j] = [hop3_name1, hop3_name2, ...]
                - i: 1跳邻居索引
                - j: 该1跳邻居的2跳邻居索引
                - 内层列表: 该2跳邻居的3跳邻居名称列表
        
        Note:
            过滤规则：
            - 排除当前节点自身
            - 排除1跳邻居
            - 排除当前遍历路径上的2跳邻居（可选，避免回环）
        """
        hop1_neighbors = self.neighbors[:max_nbrs]
        nbrs2_per_nbr = self._find_2hop_neighbors(max_nbrs, max_hop2_per_nbr)
        
        # 收集所有2跳邻居（用于过滤）
        all_hop2_nbrs = set()
        for nbrs2 in nbrs2_per_nbr:
            all_hop2_nbrs.update(nbrs2)
        
        nbrs3_per_hop2 = []
        
        for i, hop1_nbr in enumerate(hop1_neighbors):
            hop2_neighbors = nbrs2_per_nbr[i]
            nbrs3_for_this_hop1 = []
            
            for hop2_nbr in hop2_neighbors:
                nbrs3 = []
                # 从邻接表获取该2跳邻居的邻居列表
                if hop2_nbr in self.adjacency_table:
                    hop2_nbr_neighbors, _ = self.adjacency_table[hop2_nbr]
                    
                    for hop3_nbr in hop2_nbr_neighbors:
                        # 过滤：排除自身、1跳邻居、2跳邻居
                        if (hop3_nbr != self.name and 
                            hop3_nbr not in hop1_neighbors and
                            hop3_nbr not in all_hop2_nbrs):
                            nbrs3.append(hop3_nbr)
                        # 达到最大数量限制
                        if len(nbrs3) >= max_hop3_per_hop2:
                            break
                
                nbrs3_for_this_hop1.append(nbrs3)
            
            nbrs3_per_hop2.append(nbrs3_for_this_hop1)
        
        return nbrs3_per_hop2
    
    @property
    def graph_feats_dict(self):
        """
        图特征维度信息，与 cross_layer_opt_with_grl-main 保持一致
        
        用于 GraphObservation 包装器初始化
        
        节点特征：只包含节点自身状态
        边特征：包含边相关信息（transmission_size, hop_distance, placeholder）
        """
        if 'New' in self.mode:
            return {
                'agent': 9,   # current_node_state(3) + mission_state(4) + routing_info(2)
                'nbr': 12,    # 12维状态（不含 transmission_size 和 hop_distance）
                'hop': 3      # 边特征：transmission_size + hop_distance + placeholder
            }
        else:
            return {
                'agent': 9,
                'nbr': 4,     # 4维状态（不含 transmission_size 和 hop_distance）
                'hop': 3      # 边特征：transmission_size + hop_distance + placeholder
            }
    
    def get_obs_size(self):
        """
        获取观测维度，与 cross_layer_opt_with_grl-main 保持一致
        
        返回扁平化后的观测维度
        """
        if 'New' in self.mode:
            # agent(9) + 4邻居×(1+12+1+1) = 9 + 4×15 = 69
            return 9 + 4 * 15
        else:
            # agent(9) + 4邻居×(1+4+1+1) = 9 + 4×7 = 37  
            # 原始33维：4邻居×6 + current(3) + mission(4) + routing(2)
            return 33

    @property
    def graph_feats(self):
        """
        图特征维度信息，与 cross_layer_opt_with_grl-main 保持一致
        
        用于 GraphObservation 包装器初始化
        """
        if 'New' in self.mode:
            return {
                'agent': 9,   # current_node_state(3) + mission_state(4) + routing_info(2)
                'nbr': 14,    # 12维状态 + transmission_size + hop_distance
                'hop': 3      # 边特征维度
            }
        else:
            return {
                'agent': 9,
                'nbr': 6,     # 4维状态 + transmission_size + hop_distance
                'hop': 3
            }

    @property
    def graph_feats_separated(self):
        """
        【任务特征分离版本】图特征维度信息
        
        与 graph_feats 的区别：
        - agent 只有 3 维（纯环境特征）
        - 新增 task 维度 6 维（任务上下文）
        
        用于 TaskConditionedController 等分离架构
        """
        if 'New' in self.mode:
            return {
                'agent': 3,   # current_node_state 纯环境信息
                'nbr': 14,    # 12维状态 + transmission_size + hop_distance
                'hop': 3,     # 边特征维度
                'task': 6     # mission_state(4) + routing_info(2)
            }
        else:
            return {
                'agent': 3,   # current_node_state 纯环境信息
                'nbr': 6,     # 4维状态 + transmission_size + hop_distance
                'hop': 3,     # 边特征维度
                'task': 6     # mission_state(4) + routing_info(2)
            }

    @property
    def graph_feats_dict_separated(self):
        """
        【任务特征分离版本】用于 GraphObservation 包装器初始化
        
        与 graph_feats_dict 的区别：
        - agent 只有 3 维（纯环境特征）
        - 新增 task 维度 6 维（任务上下文）
        """
        if 'New' in self.mode:
            return {
                'agent': 3,   # current_node_state 纯环境信息
                'nbr': 12,    # 12维状态（不含边信息）
                'hop': 3,     # 边特征维度
                'task': 6     # mission_state(4) + routing_info(2)
            }
        else:
            return {
                'agent': 3,   # current_node_state 纯环境信息
                'nbr': 4,     # 4维状态（不含边信息）
                'hop': 3,     # 边特征维度
                'task': 6     # mission_state(4) + routing_info(2)
            }
 
    def tradition_routing(self,current_state,computing=True):
        def shortest_path_and_cost(source, target):
            if source == target:
                return [], 0
            else:
                try:
                    path = nx.shortest_path(self.neighbor_graph, source=source, target=target, weight='weight')[1:]
                    cost = sum(self.neighbor_graph[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))
                    return path, cost
                except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
                    return [], 10
        for edge in self.neighbor_graph.edges():
            node1,node2=edge
            self.neighbor_graph[node1][node2]['weight']=self.neighbor_graph[node1][node2]['missing']*10+self.neighbor_graph[node1][node2].get('transmission_weight', 0)/self.transmission_rate+self.neighbor_graph[node1][node2]['propagation_weight']
            if self.propagator.graph.nodes[node2].get('current_memory_occupy',0)/self.memory > 0.9:
                self.neighbor_graph[node1][node2]['weight'] += 10
        source, destination, size, computing_demand, size_after_computing,is_computed = current_state
        if is_computed:
            routing,_=shortest_path_and_cost(self.name, destination)
            return routing, None
        else:
            times={}
            routings={}
            n = max(int(len(self.neighbor_graph.nodes) * 1 / 2), 1)
            computing_nodes = [node for node in self.neighbor_graph.nodes if self.neighbor_graph.nodes[node].get('computing_remain', 0) == 0 and node!=destination]
            non_computing_nodes = sorted((node for node in self.neighbor_graph.nodes if self.neighbor_graph.nodes[node].get('computing_remain', 0) != 0 and node!=destination),key=lambda k: self.neighbor_graph.nodes[k].get('computing_remain', 0))
            computing_nodes += non_computing_nodes[:max(n - len(computing_nodes), 0)]
            for c_node in computing_nodes:
                time=self.neighbor_graph.nodes[c_node].get('computing_remain', 0)/self.computing_ability+ computing_demand/self.computing_ability + (self.propagator.graph.nodes[c_node].get('current_memory_occupy',0)/self.memory > 0.9)*10
                routing_1,time_1=shortest_path_and_cost(self.name, c_node)
                routing_2,time_2=shortest_path_and_cost(c_node, destination)
                times[c_node]=time+time_1+time_2+(len(routing_1)*size+len(routing_2)*size_after_computing)/self.transmission_rate
                routings[c_node]=routing_1+routing_2
            if times:
                computing_decision=sorted(times, key=times.get)[0]
            else:
                return [None],None
            if PRE and computing:
                if 'computing_remain' in self.propagator.graph.nodes[computing_decision]:
                    self.propagator.graph.nodes[computing_decision]['computing_remain'] += computing_demand
                else:
                    self.propagator.graph.nodes[computing_decision]['computing_remain'] = computing_demand
            if not computing:
                return routings[computing_decision], None
            else:
                return routings[computing_decision],computing_decision

    def _store_leo_policy_experience(
        self,
        packet=None,
        task_type=0,
        packet_size=0.0,
        computing_demand=0.0,
        size_after_computing=0.0,
        is_computed=False,
        hops=0,
        destination=None,
        next_hop_target=None,
        compute_node_target=None,
    ):
        transformer = getattr(self.propagator, 'transformer_module', None)
        if transformer is None or not hasattr(transformer, 'store_leo_policy_experience'):
            return False
        policy_stored = transformer.store_leo_policy_experience(
            leo_satellite=self,
            task_type=task_type,
            packet_size=packet_size,
            computing_demand=computing_demand,
            size_after_computing=size_after_computing,
            is_computed=is_computed,
            hops=hops,
            destination=destination,
            next_hop_target=next_hop_target,
            compute_node_target=compute_node_target,
        )
        critic_stored = False
        if hasattr(transformer, 'record_critic_action'):
            critic_stored = transformer.record_critic_action(
                packet=packet,
                leo_satellite=self,
                task_type=task_type,
                packet_size=packet_size,
                computing_demand=computing_demand,
                size_after_computing=size_after_computing,
                is_computed=is_computed,
                hops=hops,
                next_hop_target=next_hop_target,
                compute_node_target=compute_node_target,
            )
        return bool(policy_stored or critic_stored)
   
    def get_next_hop(
        self,
        obs,
        destination,
        is_computed,
        task_context=None,
        task_edge_feats=None,
        return_score=False,
        critic_context=None,
    ):
        """
        根据观测 obs 选择下一跳动作
        
        Args:
            obs: 观测，根据 obs_type 可以是扁平状态向量或 PyG 图
            destination: 目标节点
            is_computed: 是否已完成计算
            task_context: 任务上下文（分离模式下需要），6维张量
            task_edge_feats: 任务边特征（分离模式下需要），dict {'1hop': tensor}
        
        Returns:
            默认返回动作索引（DQN）或 [动作索引, log_prob]（PPO）。
            return_score=True 时返回 (动作结果, 所选动作得分)。
        """
        def dqn_output():
            with torch.no_grad():
                if self.propagator.obs_type in ('relational_separated', 'graph_separated'):
                    model_obs = obs.to(self.device)
                    model_task_context = task_context
                    if model_task_context is not None:
                        if not isinstance(model_task_context, torch.Tensor):
                            model_task_context = torch.tensor(model_task_context, dtype=torch.float)
                        model_task_context = model_task_context.to(self.device)
                    output, _ = self.q_net(model_obs, model_task_context, None, task_edge_feats)
                elif self.propagator.obs_type in ('relational', 'graph'):
                    model_obs = obs.to(self.device)
                    output, _ = self.q_net(model_obs, None)
                else:
                    obs_tensor = torch.tensor(obs, dtype=torch.float).unsqueeze(0).to(self.device)
                    output = self.q_net(obs_tensor)
            return output.squeeze(0)

        def dqn_action_score(output, action_index):
            if action_index < 0 or action_index >= output.size(-1):
                return float('-inf')
            if self.leo_action_mask_enabled:
                if action_index not in self._leo_valid_actions(is_computed, output.size(-1)):
                    return float('-inf')
                output = self._mask_leo_action_scores(output, is_computed)
            elif is_computed and action_index >= 4:
                return float('-inf')
            return float(output[action_index].item())

        def critic_selection(output):
            context = critic_context if isinstance(critic_context, dict) else None
            if context is None:
                return None
            transformer = getattr(self.propagator, 'transformer_module', None)
            if transformer is None or not hasattr(transformer, 'select_leo_action_with_critic'):
                return None
            valid_actions = self._leo_valid_actions(is_computed, output.size(-1))
            if not valid_actions:
                return None
            return transformer.select_leo_action_with_critic(
                packet=context.get('packet'),
                leo_satellite=self,
                q_values=output.detach().cpu().numpy(),
                valid_actions=valid_actions,
                task_type=context.get('task_type', 0),
                packet_size=context.get('packet_size', 0.0),
                computing_demand=context.get('computing_demand', 0.0),
                size_after_computing=context.get('size_after_computing', 0.0),
                is_computed=is_computed,
                hops=context.get('hops', 0),
            )

        if 'DQN' in self.mode:
            if np.random.rand() <= self.epsilon:#以ε的概率随机选择动作，具体分两种子策略（各 50% 概率）
                if np.random.rand() <= 0.5:#完全随机选择：
                    if self.leo_action_mask_enabled:
                        valid_actions = self._leo_valid_actions(is_computed, 5)
                        next_index = int(np.random.choice(valid_actions)) if valid_actions else 5
                    elif not is_computed:
                        next_index = np.random.choice(5)#0-4（4：本地计算）
                    else:
                        next_index = np.random.choice(4)
                else:#基于邻居距离的启发式探索
                    neighbor_distances = []
                    for neighbor in self.neighbors:
                        if destination in self.neighbor_hops[neighbor]:
                            neighbor_distances.append(self.neighbor_hops[neighbor][destination] / self.max_hop)
                        else:
                            neighbor_distances.append(2)
                    if len(self.neighbors) < 4:
                        for _ in range(4 - len(self.neighbors)):
                            neighbor_distances.append(2)
                    if self.leo_action_mask_enabled:
                        real_neighbor_distances = neighbor_distances[:min(len(self.neighbors), 4)]
                        if not is_computed and np.random.rand() <= 0.2:
                            next_index = 4
                        elif real_neighbor_distances:
                            min_value = min(real_neighbor_distances)
                            next_index = np.random.choice([index for index, value in enumerate(real_neighbor_distances) if value == min_value])
                        else:
                            next_index = 4 if not is_computed else 5
                    elif not is_computed:
                        if np.random.rand() <= 0.2:
                            next_index = 4
                        else:
                            min_value = min(neighbor_distances)
                            next_index = np.random.choice([index for index, value in enumerate(neighbor_distances) if value == min_value])
                    else:
                        min_value = min(neighbor_distances)
                        next_index = np.random.choice([index for index, value in enumerate(neighbor_distances) if value == min_value])
                if return_score:
                    exploration_output = dqn_output()
                    return int(next_index), dqn_action_score(exploration_output, int(next_index))
                return int(next_index)
            else:#以1-ε的概率通过 Q 网络选择最优动作
                
                # === 原方法：直接使用 q_net ===
                # 根据 obs_type 处理观测
                main_output = dqn_output()
                """这里将观测传入模型中------------------------------------------------------------------------------------"""
                # main_output 形状: (1, n_actions) 或 (n_actions,)
                critic_result = critic_selection(main_output)
                if critic_result is not None:
                    next_index = int(critic_result['action'])
                    selected_score = float(critic_result['score'])
                else:
                    selected_score = None
                    if self.leo_action_mask_enabled:
                        if not self._leo_valid_actions(is_computed, main_output.size(-1)):
                            return (5, float('-inf')) if return_score else 5
                        main_output = self._mask_leo_action_scores(main_output, is_computed)
                        next_index = torch.argmax(main_output).item()
                    elif not is_computed:
                        next_index = torch.argmax(main_output).item()#选择所有动作（0-4）中 Q 值最大的动作
                    else:
                        # 已计算时，掩码掉最后一个动作（本地计算）
                        next_index = torch.argmax(main_output[0:4]).item()
                if return_score:
                    if selected_score is not None:
                        return int(next_index), selected_score
                    return int(next_index), dqn_action_score(main_output, int(next_index))
                return next_index
            """这里动作集合通常包含：选择某个 neighbor(index)、选择本地计算(index 4)、或其他(index 5 表示错误/放弃）。"""
        else:#适用于 PPO 等策略梯度算法，通过网络输出的概率分布采样动作
            # 根据 obs_type 处理观测
            with torch.no_grad():
                if self.propagator.obs_type in ('relational_separated', 'graph_separated'):
                    # 分离模式：obs 是 PyG 图，需要额外的 task_context 和 task_edge_feats
                    obs = obs.to(self.device)
                    if task_context is not None:
                        if not isinstance(task_context, torch.Tensor):
                            task_context = torch.tensor(task_context, dtype=torch.float)
                        task_context = task_context.to(self.device)
                    main_output, _ = self.q_net(obs, task_context, None, task_edge_feats)
                elif self.propagator.obs_type in ('relational', 'graph'):
                    # 图模式：obs 是 PyG 图
                    # GNN controller 返回 (q_vals, hidden_state) 元组
                    obs = obs.to(self.device)
                    main_output, _ = self.q_net(obs, None)
                else:
                    # 扁平模式：obs 是状态向量
                    obs_tensor = torch.tensor(obs, dtype=torch.float).unsqueeze(0).to(self.device)
                    main_output = self.q_net(obs_tensor)
            # 确保是 (1, n_actions) 形状
            if main_output.dim() == 1:
                main_output = main_output.unsqueeze(0)
            if self.leo_action_mask_enabled:
                if not self._leo_valid_actions(is_computed, main_output.size(-1)):
                    invalid_action = [5, 0.0]
                    return (invalid_action, float('-inf')) if return_score else invalid_action
                main_output = self._mask_leo_action_scores(main_output, is_computed)
            elif is_computed:
                # 已计算时，掩码掉最后一个动作（本地计算）
                main_output = main_output[:, 0:4]
            main_output = torch.nn.functional.softmax(main_output, dim=-1)
            # main_output 通常是网络对每个动作的 “原始分数”（如 logits），未经过归一化，可能为任意实数值。
            # softmax 函数通过指数归一化（exp(x_i) / sum(exp(x_j))）将分数转换为 [0, 1] 区间的概率，且所有动作概率之和为 1。
            dist = torch.distributions.Categorical(main_output)
            #基于 softmax 输出的概率分布，创建一个 “类别分布” 对象（Categorical）
            # Categorical 是 PyTorch 中用于离散动作空间的概率分布类，适用于从有限个动作中按概率采样的场景。
            # 该分布以 main_output（已归一化的概率）为参数，后续可通过它进行采样或计算概率。
            action = dist.sample()
            action_result = [action.item(), dist.log_prob(action).item()]
            if return_score:
                return action_result, action_result[1]
            return action_result

    def _leo_valid_actions(self, is_computed, action_dim=5):
        action_dim = int(action_dim)
        neighbor_actions = list(range(min(len(self.neighbors), 4, action_dim)))
        if not is_computed and action_dim > 4:
            neighbor_actions.append(4)
        return neighbor_actions

    def _mask_leo_action_scores(self, scores, is_computed):
        action_dim = scores.size(-1)
        valid_actions = self._leo_valid_actions(is_computed, action_dim)
        if not valid_actions:
            return scores
        mask = torch.zeros(action_dim, dtype=torch.bool, device=scores.device)
        mask[valid_actions] = True
        while mask.dim() < scores.dim():
            mask = mask.unsqueeze(0)
        return scores.masked_fill(~mask, -1e9)

    # 功能：在候选目标节点中，筛选出基于 Q 网络评分最高的节点，用于优化转发目标选择。

    # 核心逻辑：

    # 对任务状态（mission_state）进行归一化（如数据包大小、计算需求等除以最大值）。
    # 遍历每个候选目标节点，若节点在路由表中，则通过get_current_state获取当前状态，再通过cal_score计算 Q 网络评分。
    # 筛选出评分最高的所有目标节点（可能多个节点评分相同）。

    # 输入：候选目标节点列表（destinations）、任务状态（mission_state）、当前跳数（hops）、是否已计算（is_computed）
    # 输出：评分最高的目标节点列表（best_destinations）

    def find_highest_score(self, destinations, mission_state, hops, is_computed):
        # highest_score= -2
        highest_score= -float('inf')  # 使用负无穷更安全
        mission_state=[mission_state[0], mission_state[1]/ self.max_size, mission_state[2]/ self.computing_ability, mission_state[3] / self.max_size]
        best_destinations=[]
        for destination in destinations:
            if destination in self.routing_tables:
                # 使用 _get_state_for_decision 获取观测
                if self.propagator.obs_type == 'graph_separated':
                    obs, task_context, task_edge_feats = self._get_state_for_decision(
                        destination, hops, is_computed, mission_state)
                else:
                    obs, hop2_info = self._get_state_for_decision(
                        destination, hops, is_computed, mission_state)
                    task_context = hop2_info if self.propagator.is_separated_mode() else None
                    task_edge_feats = None
                # 传入 task_context 和 task_edge_feats
                score = self.cal_score(obs, is_computed, task_context, task_edge_feats)
                if score > highest_score:
                    highest_score = score
                    best_destinations = [destination]
                elif score == highest_score:
                    best_destinations.append(destination)
        return best_destinations
    # 功能：计算当前状态下的 Q 网络评分，辅助find_highest_score进行目标节点筛选。

    # 核心逻辑：

    # 将当前状态转为张量并输入 Q 网络，获取输出。
    # 若数据包未计算，返回所有动作的最大 Q 值；若已计算，仅返回前 4 个转发动作的最大 Q 值。

    # 输入：当前状态（current_state）、是否已计算（is_computed）
    # 输出：Q 网络评分（最大 Q 值）
    def cal_score(self, obs, is_computed, task_context=None, task_edge_feats=None):
        """
        计算观测的 Q 值得分
        
        Args:
            obs: 观测，根据 obs_type 可以是扁平状态向量或 PyG 图
            is_computed: 是否已完成计算
            task_context: 任务上下文（分离模式必需），6维张量
            task_edge_feats: 任务边特征（分离模式需要），dict {'1hop': tensor}
        
        Returns:
            Q 网络评分（最大 Q 值）
        """
               
        # === 原方法：直接使用 q_net ===
        # 根据 obs_type 处理观测
        with torch.no_grad():
            if self.propagator.obs_type in ('relational_separated', 'graph_separated'):
                # 分离模式：必须传递 task_context 和 task_edge_feats
                obs = obs.to(self.device)
                if task_context is not None:
                    if not isinstance(task_context, torch.Tensor):
                        task_context = torch.tensor(task_context, dtype=torch.float)
                    task_context = task_context.to(self.device)
                # 调用 Q 网络，传入 task_context 和 task_edge_feats
                main_output, _ = self.q_net(obs, task_context, None, task_edge_feats)
            elif self.propagator.obs_type in ('relational', 'graph'):
                # 普通图模式
                obs = obs.to(self.device)
                main_output, _ = self.q_net(obs, None)
            else:
                # 扁平模式：obs 是状态向量
                obs_tensor = torch.tensor(obs, dtype=torch.float).unsqueeze(0).to(self.device)
                main_output = self.q_net(obs_tensor)
        """这里将观测传入模型中------------------------------------------------------------------------------------"""
        if not is_computed:
            score = torch.max(main_output).item()
        else:
            score = torch.max(main_output[0][0:4]).item()
        return score
    # 功能：数据包转发的主逻辑，处理数据包从队列取出后的转发、计算、丢弃等流程，集成了传统路由和强化学习决策，并记录经验和奖励。
    # 核心逻辑：
    # 循环从转发队列（forward_queue）获取数据包，更新跳数并处理。
    # 若卫星不活跃，丢弃数据包并记录统计信息。
    # 若目标节点不是自身：
        # 传统模式（Tradition/Ground）：使用预定义路由表决策下一跳。
        # 强化学习模式：通过get_next_hop或find_highest_score选择下一跳或目标节点。
        # 根据动作索引处理：
            # 索引小于邻居数：转发到对应邻居节点，记录经验和奖励。
            # 索引为 4：本地计算，将数据包放入计算队列。
            # 其他索引：丢弃数据包，记录惩罚奖励。
         # 若跳数超过最大限制（2*max_hop），丢弃数据包并记录惩罚。
    # 若目标节点是自身：将数据包放入卸载队列（offload_queue）。
    # 记录强化学习经验（状态、动作、奖励等）到经验池（propagator.experiences）。
    # 作用：是卫星处理数据包的核心流程，连接了路由决策、动作执行、奖励计算和经验收集，支撑强化学习的训练和传统路由的运行。
    """SimPy 进程(generator),由 env.process(self.forward_packet()) 启动后在仿真里并发运行。"""
    def forward_packet(self):
        if self.isLeo == False:
            return
        while self.active:
            packet = yield self.forward_queue.get()
            packet.hops += 1#将包的 hop 计数加 1（表示包经过此节点一次）注意这是在进入节点就计数。
            source, destination,hops,creation_time,size = packet.source, packet.destination,packet.hops,packet.creation_time,packet.size#包头信息与状态
                    
            # 解包 information，使用简化的 7 字段格式
            info = packet.information
            is_computed, type, computing_demand, size_after_computing, last_time, last_obs, last_action = info[:7]
            yield self.env.timeout(self.processing_time)#模拟处理（路由/决策）所需的 CPU 时间或处理延迟（非常小的值通常）
            
            # 初始化 current_task_context（所有模式都需要）
            current_task_context = None # 任务上下文（分离模式使用）
            current_task_edge_feats = None # 任务边特征（graph_separated使用
            current_obs = None # 当前观测（所有模式使用）
            reached_meo_entry = False
            leo_entry_reward = None
            
            if self.mode in ["Tradition","Ground"]:#传统模式
                current_state = [source, destination, size, computing_demand, size_after_computing,is_computed]
                hop2_info = None  # 传统模式不使用2跳信息
                current_graph = None  # 传统模式不使用图
                if not packet.routing and destination!= self.name:
                    computing = False if self.mode == "Ground" else True
                    packet.routing, computing_node = self.tradition_routing(current_state, computing)
                    if computing_node is not None:
                        self._record_packet_compute_assignment(packet, computing_node)
                    else:
                        packet.computing_node = None
            else:
                if self.select_mode == 3 and destination != self.name:#非传统模式（RL模式）
                    min_hops_destinations=self.find_min_hops_destinations(5)#筛选出跳数最少的前 5 个可达节点，作为候选目标节点
                    hightest_score_destinations=self.find_highest_score(min_hops_destinations,[type,size,computing_demand,size_after_computing],hops, is_computed)
                    #用 find_highest_score（Q 网络评分）在这些候选中挑选高分目标，并随机选一个作为当前包的 destination。
                    if hightest_score_destinations:
                        destination = np.random.choice(hightest_score_destinations)
                    else:
                        destination = 'False'
                    packet.destination = destination
                
                if packet.temporary_destination:
                    if self.name in packet.temporary_destination:
                        reached_meo_entry = True
                        segment_end_time = getattr(packet, 'meo_segment_time', None)
                        if segment_end_time is None:
                            segment_end_time = self.env.now
                            packet.meo_segment_time = float(segment_end_time)
                        segment_delay = self.propagator.completed_leo_domain_delay(
                            packet, self.env.now
                        )
                        leo_entry_reward = self.propagator.leo_domain_entry_reward(
                            is_computed, segment_delay
                        )
                        transformer = getattr(self.propagator, 'transformer_module', None)
                        meo_router = getattr(transformer, 'meo_router', None)
                        previous_domain = None
                        previous_trace = getattr(packet, 'meo_decision_trace', None)
                        if previous_trace is None:
                            traces = getattr(packet, 'meo_decision_traces', None) or []
                            previous_trace = traces[-1] if traces else None
                        if isinstance(previous_trace, dict):
                            previous_domain = previous_trace.get('current_domain')
                            if previous_domain is None:
                                previous_domains = previous_trace.get('domains', []) or []
                                if len(previous_domains) >= 2:
                                    previous_domain = previous_domains[-2]
                        excluded_domains = {previous_domain} if previous_domain is not None else None
                        if meo_router is not None:
                            meo_router.record_executed_boundary(
                                packet=packet,
                                meo_satellite=self.propagator.satellites[self.masterMeo],
                                reached_node=self.name,
                                previous_node=getattr(packet, 'meo_previous_hop', None),
                            )
                        plan = self.propagator.satellites[self.masterMeo].recommend_path(
                            src=self.name,
                            dst=destination,
                            packet_size=size,
                            task_type=type,
                            computing_demand=computing_demand,
                            size_after_computing=size_after_computing,
                            is_computed=is_computed,
                            excluded_domains=excluded_domains,
                        )
                        packet.temporary_destination = []
                        next_meo_plan = None
                        if (
                            plan is not None
                            and destination in self.propagator.satellites
                            and self.propagator.satellites[destination].masterMeo != self.masterMeo
                        ):
                            new_temps = list(plan.get('boundary_sat', []) or [])
                            packet.add_temporary_destination(new_temps)
                            if new_temps:
                                next_meo_plan = plan
                        if meo_router is not None:
                            meo_router.finish_segment(
                                packet,
                                next_plan=next_meo_plan,
                                reached_node=self.name,
                                leo_reward=leo_entry_reward,
                            )
                            if next_meo_plan is not None:
                                meo_router.attach_decision(packet, next_meo_plan)
                        packet.leo_previous_domain_entry_time = None
                # 根据 obs_type 获取当前观测
                mission_state = [type, size/self.max_size, computing_demand/self.computing_ability, size_after_computing/self.max_size]
                decision_destinations = list(packet.temporary_destination) if packet.temporary_destination else [destination]
                decision_candidates = []
                for candidate_destination in decision_destinations:
                    candidate_task_context = None
                    candidate_task_edge_feats = None
                    if self.propagator.obs_type == 'graph_separated':
                        candidate_obs, candidate_task_context, candidate_task_edge_feats = self._get_state_for_decision(
                            candidate_destination, hops, is_computed, mission_state
                        )
                    else:
                        candidate_obs, hop2_info = self._get_state_for_decision(
                            candidate_destination, hops, is_computed, mission_state)
                        candidate_task_context = hop2_info if self.propagator.is_separated_mode() else None
                    decision_candidates.append({
                        'destination': candidate_destination,
                        'obs': candidate_obs,
                        'task_context': candidate_task_context,
                        'task_edge_feats': candidate_task_edge_feats,
                    })

                destination = decision_candidates[0]['destination']
                current_obs = decision_candidates[0]['obs']
                current_task_context = decision_candidates[0]['task_context']
                current_task_edge_feats = decision_candidates[0]['task_edge_feats']
                if self.propagator.obs_type == 'graph_separated':
                    """
                    current_obs:PyG 异构图（只包含纯环境信息）
                        agent 节点特征：[3维] = [is_producing, memory_remain, computing_remain]
                        邻居节点特征：[12/4维] 节点状态
                        边特征：[transmission_size]（只有环境信息 1跳 2跳都有）
                     current_task_context：任务上下文张量 [6维] = [type, size_norm, computing_demand_norm, size_after_computing_norm, routing_info(2维)]                     
                         这是与任务相关的全局特征，独立于图结构
                     current_task_edge_feats：任务边特征 dict {'1hop': tensor}
                         1跳边特征张量 [transmission_size, hop_distance, placeholder]
                        这些任务特征用于分离架构的决策网络，帮助网络理解任务需求与环境状态的关系。


                        self.propagator.obs_type = 'graph_separated'

                        # ⚠️ 走不同的代码分支！
                        current_obs, current_task_context, current_task_edge_feats = self._get_state_for_decision(
                            destination, hops, is_computed, mission_state
                        )

                        # current_obs: PyG图（纯环境）
                        current_obs = HeteroData(
                            agent={
                                'feat': tensor([[0.0, 0.65, 0.01]])  # [1, 3] 纯环境
                            },
                            
                            nbr={
                                'feat': tensor([
                                    [0.0, 0.7, 0.8, 0.0],  # 只有节点环境状态
                                    [0.0, 0.5, 0.6, 0.0],
                                    [0.0, 0.3, 0.4, 0.0],
                                    [0.0, 0.6, 0.7, 0.0]
                                ])  # [4, 4] 不含任何任务信息
                            },
                            
                            ('nbr', '1hop', 'agent')={
                                'edge_index': tensor([[0, 1, 2, 3], [0, 0, 0, 0]]),
                                'edge_attr': tensor([
                                    [0.3],  # 只有 transmission_size！
                                    [0.5],
                                    [0.1],
                                    [0.2]
                                ])  # [4, 1] 只有环境边特征
                            },
                            
                            ('agent', '1hop', 'nbr')={
                                'edge_index': tensor([[0, 0, 0, 0], [0, 1, 2, 3]]),
                                'edge_attr': tensor([[0.3], [0.5], [0.1], [0.2]])
                            },
                            
                            # 2跳边也只有transmission_size
                            ('nbr', '2hop', 'nbr')={
                                'edge_index': tensor([[...], [...]]),
                                'edge_attr': tensor([[0.4], [0.6], ...])  # [N_2hop, 1]
                            }
                        )

                        # current_task_context: 任务全局特征
                        current_task_context = tensor([1.0, 0.6, 0.7, 0.25, 0.3, 0.0])  # [6维]
                        #                              [type, size, comp, size_after, hops, is_computed]

                        # current_task_edge_feats: 任务边特征（只有1跳！）
                        current_task_edge_feats = {
                            '1hop': tensor([
                                [0.0],  # Sat002的hop_distance
                                [1.0],  # Sat003
                                [2.0],  # Sat004
                                [1.0]   # Sat005
                            ])  # [4, 1]
                        }
                    """

                else:
                    """
                    current_obs：
                        flat 模式：扁平状态向量 [33维]
                        relational / graph 模式：PyG 图（包含任务特征）
                    hop2_info 的多重含义：
                        flat 模式：无用 None
                        relational / graph 模式：
                            - 普通模式：无用 None
                            - 分离模式：任务上下文张量 [6维] = [type, size_norm, computing_demand_norm, size_after_computing_norm, routing_info(2维)]
                    """
                    
                    current_task_context = current_task_context if self.propagator.is_separated_mode() else None
                    current_task_edge_feats = None
                # current_obs 根据 obs_type 不同：
                # - flat: 扁平状态向量:包含 agent 状态 + 4个邻居状态 + 任务信息;不使用图结构
                """
                # current_obs: 扁平状态向量
                    current_obs = array([
                        # Agent状态（9维）
                        0.0, 0.65, 0.01, 1.0, 0.6, 0.7, 0.25, 0.3, 0.0,
                        # 邻居1状态（6维）
                        1.0, 0.0, 0.7, 0.8, 0.0, 0.3, 0.3,
                        # 邻居2-4状态（6维×3）
                        ...
                    ])  # shape: (33,)

                    # hop2_info: None（flat模式不使用）
                    hop2_info = None

                    # 最终
                    current_task_context = None  # 非分离模式
                    current_task_edge_feats = None
                """
                # - relational: PyG 图（关系图，无边特征）
                """
                   图结构中没有独立的边特征（edata 为空）
                   边信息（transmission_size, hop_distance）直接拼接到邻居节点特征
                   agent 节点特征包含任务信息（9维）
                   邻居节点特征：[节点状态(4/12维) + transmission_size + hop_distance]

                   self.propagator.obs_type = 'relational'

                    # current_obs: PyG异构图
                    current_obs = HeteroData(
                        agent={
                            'feat': tensor([[
                                0.0, 0.65, 0.01,    # 环境: [is_producing, memory_remain, computing_remain]
                                1.0, 0.6, 0.7, 0.25, 0.3, 0.0  # 任务: [type, size, comp, size_after, hops, is_computed]
                            ]])  # [1, 9]
                        },
                        
                        nbr={
                            'feat': tensor([
                                # Sat002: [节点状态(4维) + transmission_size + hop_distance]
                                [0.0, 0.7, 0.8, 0.0,  0.3, 0.0],
                                # Sat003
                                [0.0, 0.5, 0.6, 0.0,  0.5, 1.0],
                                # Sat004
                                [0.0, 0.3, 0.4, 0.0,  0.1, 2.0],
                                # Sat005
                                [0.0, 0.6, 0.7, 0.0,  0.2, 1.0]
                            ])  # [4, 6]
                        },
                        
                        ('nbr', '1hop', 'agent')={
                            'edge_index': tensor([[0, 1, 2, 3], [0, 0, 0, 0]]),  # 4个邻居→agent
                            'edge_attr': None  # ⚠️ relational模式无独立边特征！
                        },
                        
                        ('agent', '1hop', 'nbr')={
                            'edge_index': tensor([[0, 0, 0, 0], [0, 1, 2, 3]]),
                            'edge_attr': None
                        }
                        # 可能还有2hop边（如果有2跳邻居）
                    )

                    # hop2_info: None（非分离模式）
                    hop2_info = None

                    # 最终
                    current_task_context = None
                    current_task_edge_feats = None`
                
                """
                # - graph: PyG 图
                """
                    图结构中有独立的边特征（edata 包含边信息）
                    边特征：[transmission_size, hop_distance, placeholder]（3维）
                    agent 节点特征包含任务信息（9维）
                    邻居节点特征：只包含节点状态（4/12维），不含边信息

                    self.propagator.obs_type = 'graph'

                    # current_obs: PyG异构图
                    current_obs = HeteroData(
                        agent={
                            'feat': tensor([[
                                0.0, 0.65, 0.01,    # 环境
                                1.0, 0.6, 0.7, 0.25, 0.3, 0.0  # 任务
                            ]])  # [1, 9]
                        },
                        
                        nbr={
                            'feat': tensor([
                                # Sat002: 只有节点状态（4维），不含边信息！
                                [0.0, 0.7, 0.8, 0.0],
                                [0.0, 0.5, 0.6, 0.0],
                                [0.0, 0.3, 0.4, 0.0],
                                [0.0, 0.6, 0.7, 0.0]
                            ])  # [4, 4]
                        },
                        
                        ('nbr', '1hop', 'agent')={
                            'edge_index': tensor([[0, 1, 2, 3], [0, 0, 0, 0]]),
                            'edge_attr': tensor([
                                [0.3, 0.0, 0.0],  # [transmission_size, hop_distance, placeholder]
                                [0.5, 1.0, 0.0],
                                [0.1, 2.0, 0.0],
                                [0.2, 1.0, 0.0]
                            ])  # [4, 3]  有独立边特征！
                        },
                        
                        ('agent', '1hop', 'nbr')={
                            'edge_index': tensor([[0, 0, 0, 0], [0, 1, 2, 3]]),
                            'edge_attr': tensor([[0.3, 0.0, 0.0], [0.5, 1.0, 0.0], ...])
                        },
                        
                        # 2跳边（如果有）
                        ('nbr', '2hop', 'nbr')={
                            'edge_index': tensor([[0, 0, 1, ...], [1, 2, 3, ...]]),  # 2跳连接
                            'edge_attr': tensor([
                                [0.4, 0.0, 0.0],  # Sat002的2跳邻居的transmission_size
                                ...
                            ])
                        }
                    )

                    # hop2_info: None
                    hop2_info = None

                    # 最终
                    current_task_context = None
                    current_task_edge_feats = None
                
                """
                # 分离模式下relational_separated，current_task_context（hop2_info） 是 task_context，current_task_edge_feats 
                """
                    图结构中没有独立的边特征（与 relational 相同）
                    agent 节点特征只有环境信息（3维）
                    hop2_info 实际是 task_context，用于后续决策时融合
                    边信息拼接到邻居节点特征

                """

                """
                    这里的else包含了四种模式：flat;relational;graph;以及分离模式(relational_separated)。
                    
               self.propagator.obs_type = 'relational_separated'

                    # current_obs: PyG图（纯环境信息！）
                    current_obs = HeteroData(
                        agent={
                            'feat': tensor([[0.0, 0.65, 0.01]])  # [1, 3] 只有环境！
                        },
                        
                        nbr={
                            'feat': tensor([
                                # 只有节点环境状态 + transmission_size（不含任务相关的hop_distance！）
                                [0.0, 0.7, 0.8, 0.0,  0.3],  # [is_producing, mem, comp, 0, trans_size]
                                [0.0, 0.5, 0.6, 0.0,  0.5],
                                [0.0, 0.3, 0.4, 0.0,  0.1],
                                [0.0, 0.6, 0.7, 0.0,  0.2]
                            ])  # [4, 5]
                        },
                        
                        ('nbr', '1hop', 'agent')={
                            'edge_index': tensor([[0, 1, 2, 3], [0, 0, 0, 0]]),
                            'edge_attr': None  # 无独立边特征
                        }
                    )

                    # hop2_info: 实际上是 task_context！
                    hop2_info = tensor([1.0, 0.6, 0.7, 0.25, 0.3, 0.0])  # [6维] 
                    #                   [type, size, comp, size_after, hops, is_computed]

                    # 最终
                    current_task_context = hop2_info  #  分离模式，赋值
                    current_task_edge_feats = None  # relational没有独立边特征
                
                """
            
            if reached_meo_entry and last_obs is not None:
                last_task_context = info[7] if len(info) > 7 else None
                leo_entry_done = self.propagator.leo_domain_entry_done(
                    self.name, destination
                )
                self.propagator.add_experience(
                    last_obs, is_computed, last_action,
                    leo_entry_reward, current_obs, leo_entry_done,
                    last_task_context, current_task_context
                )
                if leo_entry_done:
                    self.propagator.final_rewards.append(leo_entry_reward)
                else:
                    self.propagator.domain_entry_rewards.append(leo_entry_reward)
                # The previous action has been settled at the domain boundary.
                # A following action will overwrite this cleared packet state;
                # intermediate entries continue bootstrapping from current_obs.
                last_obs = None
                cleared_information = [
                    is_computed, type, computing_demand, size_after_computing,
                    self.env.now, None, None
                ]
                if self.propagator.is_separated_mode():
                    cleared_information.append(current_task_context)
                packet.extra_information(cleared_information)

            if not self.active:
                reward = self.reward_function.loss_reward(
                    self.propagator.leo_domain_delay(packet, self.env.now)
                )
                self.propagator.record_domain_entry_failure(
                    packet, reward, current_node=self.name
                )
                if last_obs is not None:
                    last_task_context = info[7] if len(info) > 7 else None
                    self.propagator.add_experience(
                        last_obs, is_computed, last_action,
                        reward, current_obs, 1,
                        last_task_context, current_task_context
                    )
                    self.propagator.final_rewards.append(reward)
                self.propagator.finish_meo_decision(packet, reward, done=True, meo_result="link_drop")
                if packet.computing_node and PRE:
                    self.propagator.graph.nodes[packet.computing_node]['computing_remain'] -= computing_demand
                if type==0:
                    self.statics_data['Lost_relay_0'] += 1
                else:
                    self.statics_data['Lost_relay_1'] += 1
                self.logger.log(f"Time {self.env.now:.3f}: {self.name} is missed, dropped 1 packet")
                break
            """
            如果在处理期间节点变为不活跃（被动清除/失效），需要：
            回滚预占计算资源(computing_remain)—— 条件是 packet.computing_node 存在并且 PRE 模式开启。
            更新丢包统计（按 type 区分）。
            记录日志并退出循环。
            """
            if destination != self.name:
                if hops <= 2 * self.max_hop:
                    selected_next_hop_name = None
                    selected_by_transformer = False
                    if self.mode in ["Tradition","Ground"]:
                        if packet.computing_node==self.name:
                            action=4
                            selected_next_hop_name = self.name
                            packet.computing_node=None
                        else:
                            if packet.routing:
                                next_hop=packet.routing.pop(0)
                                selected_next_hop_name = next_hop
                                if next_hop in self.neighbors:
                                    action=self.neighbors.index(next_hop)
                                else:
                                    action = 5
                            else:
                                action=5
                        next_index = action
                    else:
                        if packet.temporary_destination:
                            evaluated_candidates = []
                            for candidate in decision_candidates:
                                candidate_action, candidate_score = self.get_next_hop(
                                    candidate['obs'],
                                    candidate['destination'],
                                    is_computed,
                                    candidate['task_context'],
                                    candidate['task_edge_feats'],
                                    return_score=True,
                                    critic_context={
                                        'packet': packet,
                                        'task_type': type,
                                        'packet_size': size,
                                        'computing_demand': computing_demand,
                                        'size_after_computing': size_after_computing,
                                        'hops': hops,
                                    },
                                )
                                candidate_score = float(candidate_score)
                                if np.isnan(candidate_score):
                                    candidate_score = float('-inf')
                                evaluated_candidates.append((candidate_score, candidate_action, candidate))
                            _, action, selected_candidate = max(evaluated_candidates, key=lambda item: item[0])
                            destination = selected_candidate['destination']
                            current_obs = selected_candidate['obs']
                            current_task_context = selected_candidate['task_context']
                            current_task_edge_feats = selected_candidate['task_edge_feats']
                        else:
                            action = self.get_next_hop(
                                current_obs,
                                destination,
                                is_computed,
                                current_task_context,
                                current_task_edge_feats,
                                critic_context={
                                    'packet': packet,
                                    'task_type': type,
                                    'packet_size': size,
                                    'computing_demand': computing_demand,
                                    'size_after_computing': size_after_computing,
                                    'hops': hops,
                                },
                            )
                        """调用策略网络 get_next_hop，基于观测状态与目标 destination 返回 action"""
                        if 'PPO' in self.mode:
                            next_index = action[0]
                        else:
                            next_index = action
                        if 0 <= next_index < len(self.neighbors):
                            selected_next_hop_name = self.neighbors[next_index]
                        elif next_index == 4:
                            selected_next_hop_name = self.name
                        else:
                            selected_next_hop_name = f"action_{next_index}"
                            
                        # if next_index != 4 and packet.path != None :
                        #     if len(packet.path) > 0:
                        #         if packet.path[0] != self.name:
                        #             selected_next_hop_name = packet.path[0]
                        #             if packet.path[0] in self.neighbors:
                        #                 next_index = self.neighbors.index(packet.path[0])
                        #             else:
                        #                 next_index = 6
                        #         action = next_index
                        #         packet.path.pop(0)
                            
                    """如果是 PPO,action 可能是 [action_index, log_prob] 或 (index, prob)，因此取 action[0]；否则直接把 action 当索引。"""
                    if destination in self.routing_tables:
                        if 0 <= next_index < len(self.neighbors):#把包发给某邻居
                            done = 0
                            reward = self.reward_function.normal_reward(self.env.now-last_time,1-self.current_memory_occupy/self.memory)
                            next_hop = self.neighbors[next_index]
                            self._store_leo_policy_experience(
                                packet=packet,
                                task_type=type,
                                packet_size=size,
                                computing_demand=computing_demand,
                                size_after_computing=size_after_computing,
                                is_computed=is_computed,
                                hops=hops,
                                destination=destination,
                                next_hop_target=next_hop,
                                compute_node_target=None,
                            )
                            # 更新 packet.information，分离模式下使用 8 字段格式（含 task_context）
                            if self.propagator.is_separated_mode():
                                packet.extra_information([is_computed, type, computing_demand, size_after_computing, self.env.now, current_obs, action, current_task_context])
                            else:
                                packet.extra_information([is_computed, type, computing_demand, size_after_computing, self.env.now, current_obs, action])
                            self.push_transmission(next_hop, packet)#将包放入出站队列
                        elif next_index==4:#选择在本节点计算
                            done = 0
                            packet.hops-=1#把此前的 hop 增量回退 1（计算节点不应消耗 hop 或不计为转发跳）
                            reward = self.reward_function.normal_reward(self.env.now-last_time,1-self.current_memory_occupy/self.memory)
                            compute_chunk = self._planned_compute_chunk(packet, computing_demand)
                            self._store_leo_policy_experience(
                                packet=packet,
                                task_type=type,
                                packet_size=size,
                                computing_demand=computing_demand,
                                size_after_computing=size_after_computing,
                                is_computed=is_computed,
                                hops=hops,
                                destination=destination,
                                next_hop_target=None,
                                compute_node_target=self.name,
                            )
                            # 更新 packet.information，分离模式下使用 8 字段格式（含 task_context）
                            if self.propagator.is_separated_mode():
                                packet.extra_information([is_computed, type, compute_chunk, size_after_computing, self.env.now, current_obs, action, current_task_context])
                            else:
                                packet.extra_information([is_computed, type, compute_chunk, size_after_computing, self.env.now, current_obs, action])
                            self.push_computing(packet,compute_chunk)#并把计算需求传入用于预占
                        else:#非法/放弃行动（next_index其他值）
                            done = 1
                            reward = self.reward_function.loss_reward(
                                self.propagator.leo_domain_delay(packet, self.env.now)
                            )
                            self.propagator.record_domain_entry_failure(
                                packet, reward, current_node=self.name
                            )
                            if last_obs is not None:
                                last_task_context = info[7] if len(info) > 7 else None
                                previous_reward = self.reward_function.normal_reward(
                                    self.env.now-last_time,
                                    1-self.current_memory_occupy/self.memory
                                )
                                self.propagator.add_experience(
                                    last_obs, is_computed, last_action,
                                    previous_reward, current_obs, 0,
                                    last_task_context, current_task_context
                                )
                                last_obs = None
                            self.propagator.add_experience(
                                current_obs, is_computed, action,
                                reward, current_obs, 1,
                                current_task_context, current_task_context
                            )
                            self.propagator.finish_meo_decision(packet, reward, done=True, meo_result="link_drop")
                            self.propagator.final_rewards.append(reward)
                            self.current_memory_occupy-=size
                            self.statics_data[f'illegal_action_{int(type)}'] = self.statics_data.get(f'illegal_action_{int(type)}', 0) + 1
                            if packet.computing_node and PRE:
                                self.propagator.graph.nodes[packet.computing_node]['computing_remain'] -= computing_demand
                            # self.statics_data[f'illegal_action_{int(type)}'] = self.statics_data.get(f'illegal_action_{int(type)}', 0) + 1
                            self.record_illegal_action(selected_next_hop_name, selected_by_transformer)
                            if type == 0:
                                self.statics_data['Lost_relay_0'] += 1
                            else:
                                self.statics_data['Lost_relay_1'] += 1
                            self.logger.log(f"Time {self.env.now:.3f}: wrong forward decision, dropped 1 packet")
                            """把包当作丢弃,done=1(任务/episode 结束），给负的 loss_reward,追加 propagator.final_rewards,回滚内存和预占计算资源(if PRE),更新统计并记录日志"""
                    else:#如果目的地不在 routing_tables
                        #包被丢弃并统计。reward = None（没有即时 reward，可能因为这是不可控的环境/拓扑问题，不应作为 RL 信号）。
                        self.current_memory_occupy -= size
                        if packet.computing_node and PRE:
                            self.propagator.graph.nodes[packet.computing_node]['computing_remain'] -= computing_demand
                        if type == 0:
                            self.statics_data['Lost_relay_0'] += 1
                        else:
                            self.statics_data['Lost_relay_1'] += 1
                        self.logger.log(f"Time {self.env.now:.3f}: {destination} is missed, dropped 1 packet")
                        reward = self.reward_function.loss_reward(
                            self.propagator.leo_domain_delay(packet, self.env.now)
                        )
                        self.propagator.record_domain_entry_failure(
                            packet, reward, current_node=self.name
                        )
                        done = 1
                        self.propagator.final_rewards.append(reward)
                        self.propagator.finish_meo_decision(packet, reward, done=True, meo_result="link_drop")
                else:#超时（跳数过多）处理 ；如果 hops > 2 * max_hop，认为包超时或循环过久，丢弃并给负奖励，回滚资源并记录统计
                    self.current_memory_occupy -= size
                    done = 1
                    reward = self.reward_function.loss_reward(
                        self.propagator.leo_domain_delay(packet, self.env.now)
                    )
                    self.propagator.record_domain_entry_failure(
                        packet, reward, current_node=self.name
                    )
                    self.propagator.finish_meo_decision(packet, reward, done=True, meo_result="link_drop")
                    self.propagator.final_rewards.append(reward)
                    if packet.computing_node and PRE:
                        self.propagator.graph.nodes[packet.computing_node]['computing_remain'] -= computing_demand
                    if type == 0:
                        self.statics_data['Lost_relay_0'] += 1
                    else:
                        self.statics_data['Lost_relay_1'] += 1
                    self.logger.log(f"Time {self.env.now:.3f}: transmission out of time, dropped 1 packet")
            else:
                reward = None
                self.push_offload(packet)
                #（包到达其目标），把包放 offload_queue（等待下行/卸载处理）。到达后的 reward 由 downstream 进程处理（因此这里设 reward = None）
            if last_obs is not None and reward is not None:
                # 分离模式下需要传递 task_context
                last_task_context = info[7] if len(info) > 7 else None
                # current_task_context 已经在上面根据 obs_type 正确设置
                self.propagator.add_experience(
                    last_obs, is_computed, last_action,
                    reward, current_obs, done,
                    last_task_context, current_task_context
                )
            """
            如果 last_obs 存在且本次有即时 reward(即 reward 不为 None),把一条经验追加到 propagator.experiences。
            """

    def transmit_packet(self,neighbor):#这是卫星到卫星之间的链路传输过程
        if self.isLeo == False:
            return
        while self.active:
            packet = yield self.env.process(self.pop_transmission(neighbor))
            # 解包 information，分离模式下有87或第8字段（task_context）
            info = packet.information
            is_computed, type, computing_demand, size_after_computing, last_time, current_obs, next_index = info[:7]
            task_context = info[7] if len(info) > 7 else None
            # 保持观测不变，只更新时间
            if self.propagator.is_separated_mode() and task_context is not None:
                packet.extra_information([is_computed, type, computing_demand, size_after_computing, self.env.now, current_obs, next_index, task_context])
            else:
                packet.extra_information([is_computed, type, computing_demand, size_after_computing, self.env.now, current_obs, next_index])
            yield self.env.timeout(packet.size / self.transmission_rate)#模拟传输延迟
            if neighbor not in self.neighbors or not self.active:
                reward = self.reward_function.loss_reward(
                    self.propagator.leo_domain_delay(packet, self.env.now)
                )
                self.propagator.record_domain_entry_failure(
                    packet, reward, current_node=self.name
                )
                if current_obs is not None:
                    self.propagator.add_experience(
                        current_obs, is_computed, next_index,
                        reward, current_obs, 1,
                        task_context, task_context
                    )
                    self.propagator.final_rewards.append(reward)
                self.propagator.finish_meo_decision(packet, reward, done=True, meo_result="link_drop")
                if packet.computing_node and PRE:
                    self.propagator.graph.nodes[packet.computing_node]['computing_remain'] -= computing_demand
                if type == 0:
                    self.statics_data['Lost_relay_0'] += 1
                else:
                    self.statics_data['Lost_relay_1'] += 1
                self.logger.log(f"Time {self.env.now:.3f}: transmission stopped, dropped 1 packet")
            else:
                self.logger.log(f"Time {self.env.now:.3f}: {self.name}: Packet {(packet.source,packet.destination,packet.creation_time)} departed. Memory remain: {(self.memory-self.current_memory_occupy)}",detail=True)
                self.env.process(self.propagator.propagate(self.name,neighbor, packet))

    def offload_to_ground(self):#卫星到地面链路（下行）
        if self.isLeo == False:
            return
        while self.active:
            packet = yield self.env.process(self.pop_offload())
            # 灵活解包 information，支持 7 或 8 个字段
            info = packet.information
            is_computed, type, computing_demand, size_after_computing, last_time, current_state, next_index = info[:7]
            hop2_info = info[7] if len(info) > 7 else None
            packet.extra_information([is_computed,type, computing_demand, size_after_computing, self.env.now, current_state, next_index, hop2_info])
            yield self.env.timeout(packet.size / self.downlink_rate)
            if not self.active:
                reward = self.reward_function.loss_reward(
                    self.propagator.leo_domain_delay(packet, self.env.now)
                )
                self.propagator.record_domain_entry_failure(
                    packet, reward, current_node=self.name
                )
                if current_state is not None:
                    self.propagator.add_experience(
                        current_state, is_computed, next_index,
                        reward, current_state, 1,
                        hop2_info, hop2_info
                    )
                    self.propagator.final_rewards.append(reward)
                self.propagator.finish_meo_decision(packet, reward, done=True, meo_result="link_drop")
                if packet.computing_node and PRE:
                    self.propagator.graph.nodes[packet.computing_node]['computing_remain'] -= computing_demand
                if type==0:
                    self.statics_data['Lost_relay_0'] += 1
                else:
                    self.statics_data['Lost_relay_1'] += 1
                self.logger.log(f"Time {self.env.now:.3f}: offload stopped, dropped 1 packet")
                break
            self.logger.log(f"Time {self.env.now:.3f}: {self.name}: Packet {(packet.source,packet.destination,packet.creation_time)} is offloading. Memory remain: {(self.memory-self.current_memory_occupy)}",detail=True)
            self.env.process(self.propagator.downstream(self.name, packet))

    def computing_packet(self):  # 从计算队列持续取出数据包，并在本卫星上模拟计算处理过程。
        if self.isLeo == False:
            return
        while self.active:  # 只要当前卫星仍处于活动状态，就不断监听计算队列。
            packet = yield self.computing_queue.get()  # 阻塞等待一个进入本节点计算队列的数据包。
            self.is_computing = True  # 标记当前卫星正在执行计算任务，供状态观测和资源估计使用。
            self.last_computing_time = self.env.now  # 记录本次计算开始的仿真时间，用于估算进行中的计算负载。
            info = packet.information  # 读取数据包携带的任务与上一步决策信息。
            is_computed, type, computing_demand, size_after_computing, last_time, last_obs, last_action = info[:7]  # 解包标准 7 字段信息。
            computing_time_consume = computing_demand / self.computing_ability  # 按“计算需求量 / 节点计算能力”计算本次处理耗时。
            yield self.env.timeout(computing_time_consume)  # 在 SimPy 中推进计算耗时，模拟任务实际占用计算资源。
            packet.computing_waiting_time += (self.env.now - last_time)  # 累加从上次记录时间到计算完成之间的等待/处理时间。
            self.computing_time += computing_time_consume  # 累加本卫星已经消耗的总计算时间。
            remaining_before = float(getattr(packet, "remaining_computing_demand", computing_demand) or 0.0)  # 读取计算前剩余需求，缺省时认为剩余需求等于本段需求。
            remaining_after = max(0.0, remaining_before - float(computing_demand or 0.0))  # 扣除本段已完成的计算量，并避免出现负数。
            packet.remaining_computing_demand = remaining_after  # 将新的剩余计算需求写回数据包，供多计算节点计划继续使用。
            new_packet_size = size_after_computing  # 最终计算完成后，数据包大小变为 size_after_computing。
            memory_released = packet.size - new_packet_size  # 计算本次计算使数据包缩小后释放出的本地内存。
            if self.mode in ["Tradition", "Ground"]:  # 传统模式或地面模式下，还需要同步维护全局图上的节点资源属性。
                self.propagator.graph.nodes[self.name]['current_memory_occupy'] -= memory_released  # 在全局图中扣减当前节点的内存占用。
                self.propagator.graph.nodes[self.name]['computing_remain'] -= computing_demand  # 在全局图中释放本次计算预占的计算量。
            self.current_computing_queue_size -= packet.size  # 从本地计算队列占用统计中移除该包进入计算时的大小。
            self.computing_remain -= computing_demand  # 从本地剩余计算占用统计中扣除已经完成的计算需求。
            self.current_memory_occupy -= memory_released  # 本地内存占用减少本次计算释放的空间。
            packet.size = new_packet_size  # 更新数据包大小，使后续转发/下行使用计算后的大小。
            self.is_computing = False  # 标记当前计算任务结束，节点恢复非计算状态。
            self.last_computing_time = 0  # 清空计算开始时间，表示当前没有正在执行的计算。
            if not self.active:  # 如果计算完成后发现卫星已失效，则丢弃该包并做资源回滚/统计。
                self.current_memory_occupy -= packet.size  # 从本地内存占用中移除丢弃数据包当前大小。
                if self.mode in ["Tradition", "Ground"]:  # 传统模式或地面模式下同步更新全局图内存占用。
                    self.propagator.graph.nodes[self.name]['current_memory_occupy'] -= packet.size  # 在全局图中扣除该丢弃包占用的内存。
                if packet.computing_node and PRE:  # 如果该包曾经预占过计算节点资源，并且启用 PRE 预占机制，则回滚预占。
                    self.propagator.graph.nodes[packet.computing_node]['computing_remain'] -= computing_demand  # 从对应计算节点的全局预占统计中扣除本次计算需求。
                if type == 0:  # 根据任务类型记录类型 0 的中继丢包数。
                    self.statics_data['Lost_relay_0'] += 1  # 类型 0 任务丢包计数加一。
                else:  # 非类型 0 的任务按类型 1 统计丢包。
                    self.statics_data['Lost_relay_1'] += 1  # 类型 1 任务丢包计数加一。
                self.logger.log(f"Time {self.env.now:.3f}: transmission stopped, dropped 1 packet")  # 记录因节点失效导致计算后丢包的日志。
                break  # 退出计算协程循环，停止继续处理计算队列。
            info = packet.information  # 重新读取数据包信息，确保保留可能存在的第 8 个 task_context 字段。
            task_context = info[7] if len(info) > 7 else None  # 分离观测模式下，第 8 字段保存任务上下文；普通模式则为 None。
            if self.propagator.is_separated_mode() and task_context is not None:  # 分离模式且存在任务上下文时，需要把 task_context 继续写回包中。
                packet.extra_information([True, type, remaining_after, size_after_computing, last_time, last_obs, last_action, task_context])  # 更新 8 字段信息，标记是否最终完成计算并保留任务上下文。
            else:  # 普通观测模式或没有任务上下文时，使用标准 7 字段信息。
                packet.extra_information([True, type, remaining_after, size_after_computing, last_time, last_obs, last_action])  # 更新 7 字段信息，供后续转发决策和经验记录使用。
            self.logger.log(f"Time {self.env.now:.3f}: {self.name}: Packet {(packet.source,packet.destination,packet.creation_time)} finished computing. Memory remain: {(self.memory-self.current_memory_occupy)}", detail=True)  # 记录该数据包在当前节点完成本段计算后的状态。
            self.forward_queue.put(packet)  # 将计算后的数据包放回转发队列，继续按路径转发或下行。

    def build_leo_state_for_meo(self):
        neighbors_state = []
        for neighbor in self.neighbors[:4]:
            neighbor_satellite = None
            if self.propagator is not None:
                neighbor_satellite = self.propagator.satellites.get(neighbor)
            if neighbor_satellite is not None:
                neighbor_memory_remain = neighbor_satellite._remaining_memory()
                neighbor_computing_remain = neighbor_satellite._remaining_computing_resource()
            else:
                neighbor_memory_occupy = 0.0
                neighbor_computing_load = 0.0
                if self.propagator is not None and neighbor in self.propagator.graph.nodes:
                    neighbor_memory_occupy = self.propagator.graph.nodes[neighbor].get('current_memory_occupy', 0.0)
                    neighbor_computing_load = self.propagator.graph.nodes[neighbor].get('computing_remain', 0.0)
                neighbor_memory_remain = max(0.0, float(self.memory) - float(neighbor_memory_occupy))
                neighbor_computing_remain = max(0.0, float(self.computing_ability) - float(neighbor_computing_load))

            neighbors_state.append({
                'name': neighbor,
                'link_load': float(self.transmission_size.get(neighbor, float("inf"))),
                'remaining_memory': neighbor_memory_remain,
                'remaining_computing': neighbor_computing_remain,
            })

        while len(neighbors_state) < 4:
            neighbors_state.append({
                'name': None,
                'link_load': float("inf"),
                'remaining_memory': 0.0,
                'remaining_computing': 0.0,
            })

        leo_state = {
            'timestamp': self.env.now,
            'name': self.name,
            'is_producing': int(bool(self.is_producing)),
            'remaining_memory': self._remaining_memory(),
            'remaining_computing': self._remaining_computing_resource(),
            'neighbors': neighbors_state,
        }
        return leo_state

    def _leo_to_meo_delay(self, leo_name):
        if self.propagator is None:
            return 0.0
        delays = getattr(self.propagator, 'propagation_delays', {}) or {}
        if (leo_name, self.name) in delays:
            return float(delays[(leo_name, self.name)] or 0.0)
        if (self.name, leo_name) in delays:
            return float(delays[(self.name, leo_name)] or 0.0)
        if hasattr(self.propagator, '_distance') and hasattr(self.propagator, 'propagation_speed'):
            try:
                distance = self.propagator._distance(leo_name, self.name)
                return float(distance) / max(float(self.propagator.propagation_speed), 1e-9)
            except Exception:
                return 0.0
        return 0.0

    def _max_domain_leo_to_meo_delay(self, leo_names):
        delays = [self._leo_to_meo_delay(leo_name) for leo_name in leo_names]
        return max(delays) if delays else 0.0

    def collect_leo_states_from_domain(self):
        leo_states = {}
        while self.active:
            yield self.env.timeout(self.meo_state_update_period)
            managed_leos = list(self.my_leos or [])
            if not managed_leos or self.propagator is None:
                continue
            max_delay = self._max_domain_leo_to_meo_delay(managed_leos)
            yield self.env.timeout(max_delay)
            current_managed = set(self.my_leos or [])
            for leo_name in managed_leos:
                if leo_name not in current_managed:
                    continue
                leo_satellite = self.propagator.satellites.get(leo_name)
                if leo_satellite is None or not getattr(leo_satellite, 'active', False):
                    continue
                if getattr(leo_satellite, 'masterMeo', None) != self.name:
                    continue
                leo_state = leo_satellite.build_leo_state_for_meo()
                leo_states[leo_name] = leo_state
            self.receive_leo_state(leo_states)

    def add_neighbor(self,neighbor):
        if neighbor not in self.neighbors:
            self.neighbors.append(neighbor)
            self.neighbors=sorted(self.neighbors)
            self.transmission_size[neighbor] = 0
            self.transmission_length[neighbor] = 0
            if "New" in self.mode:
                self.neighbor_states[neighbor] = [0,1,0,0,0,4,0,0,0,12,0,0]
            else:
                self.neighbor_states[neighbor] = [0,1,0,0]
            self.last_heartbeat[neighbor] = self.env.now
            self.adjacency_table [self.name]=(self.neighbors, self.env.now)
            self.neighbor_hops[neighbor] = {}
            self.adjacency_table_exchanger()
            self.env.process(self.monitor_single_neighbor(neighbor))

    def del_neighbor(self,neighbor):
        if self.active:
            if neighbor in self.neighbors:
                while self.transmission_queue[neighbor].items:
                    packet = yield self.env.process(self.pop_transmission(neighbor))
                    # 灵活解包 information，支持 7 或 8 个字段
                    info = packet.information
                    is_computed, type, computing_demand, size_after_computing, last_time, last_state, last_action = info[:7]
                    last_hop2_info = info[7] if len(info) > 7 else None
                    packet.extra_information([is_computed,type, computing_demand, size_after_computing, self.env.now, None, None, None])
                    packet.routing=None
                    if packet.computing_node and PRE:
                        self.propagator.graph.nodes[packet.computing_node]['computing_remain'] -= computing_demand
                        packet.computing_node=None
                    if self.mode not in ["Tradition","Ground"]:
                        success = self.push_forward(packet)
                    else:
                        success = False
                    if not success:
                        if type == 0:
                            self.statics_data['Lost_relay_0'] += 1
                        else:
                            self.statics_data['Lost_relay_1'] += 1
                    self.logger.log(f"Time {self.env.now:.3f}: {packet} is dropped because of satellite missing.")
                if neighbor in self.neighbors:
                    self.neighbors.remove(neighbor)
                del self.transmission_size[neighbor]
                del self.neighbor_states[neighbor]
                del self.neighbor_hops[neighbor]
                del self.transmission_length[neighbor]
                self.adjacency_table[self.name]=(self.neighbors, self.env.now)
                self.update_adjacency_dict_for_bfs()
                self.build_routing_table()
                self.adjacency_table_exchanger()
                return True
            else:
                return False
        else:
            return False
    # 从 direct_satellites（可直接通信的卫星列表）中，根据路由表记录的跳数，筛选出跳数最少的前 n 个节点，确保候选目标节点是网络中可达性较好（跳数少）的节点。
    # 参数说明
    # self：卫星节点实例，包含路由表（routing_tables）和直接通信节点列表（direct_satellites）。
    # n：需要筛选的候选节点数量（如代码中调用的 find_min_hops_destinations(5) 表示筛选前 5 个）。

    def state_exchanger_meo(self):
        """把源 MEO 域内的 LEO 状态洪泛给所有可达的 MEO 邻居。"""
        while self.active:
            yield self.env.timeout(self.meo_aggregate_exchange_period)
            if self.leoStates is None or self.propagator is None:  # 如果没有可交换的 LEO 状态，或当前节点还没有绑定传播器，则无法继续交换。
                continue  # 直接退出，避免后续访问空状态或空传播器导致异常。
            start = time.perf_counter()
            aggregate_state = (self._build_domain_aggregate_state(only_aggregate_self = True))[self.name]
            visited = []
            visited.append(self.name)
            propagat_satellites = []
            propagat_satellites.append(self.name)
            while len(propagat_satellites) > 0:
                next_propagat_satellites = []
                for meo in propagat_satellites:
                    satellite = self.propagator.satellites[meo]
                    for neighbor in satellite.neighbors:
                        if neighbor in visited or neighbor == self.name:
                            continue
                        next_propagat_satellites.append(neighbor)
                        neighbor_satellite = self.propagator.satellites[neighbor]
                        visited.append(neighbor)
                        delay = float("inf")
                        if (meo, neighbor) in self.propagator.propagation_delays:
                            delay = self.propagator.propagation_delays[(meo, neighbor)]
                        elif hasattr(self.propagator, '_distance') and hasattr(self.propagator, 'propagation_speed'):
                            distance = self.propagator._distance(meo, neighbor)
                            delay = distance / float(self.propagator.propagation_speed)
                        # yield self.env.timeout(delay)
                        neighbor_satellite.remote_domain_leo_states[self.name] = copy.deepcopy(aggregate_state)
                        neighbor_satellite._refresh_inter_domain_graph(aggregate_state)
                propagat_satellites = next_propagat_satellites
                # print(self.name, "visited MEO:", [n for n in visited if not self.propagator.satellites[n].isLeo])
            end = time.perf_counter()
            # print(self.name + " state_exchanger_meo spend {}s".format(end - start))


#     候选目标节点的生成流程为：

# 卫星通过邻接表（adjacency_table）维护当前网络拓扑（邻居关系）；
# 基于邻接表，通过 BFS 算法（build_routing_table）生成路由表（routing_tables），记录所有可达节点及对应的跳数；
# find_min_hops_destinations(N)从路由表中筛选出跳数最少的前 N 个节点，作为候选目标节点（destinations），供后续评分（find_highest_score）使用。
    def find_min_hops_destinations(self, n):
        hop_counts = []
        for destination in self.direct_satellites:
            if destination in self.routing_tables:
                hops = self.routing_tables[destination][1]
            else:
                hops = np.inf
            hop_counts.append((destination, hops))
        if len(hop_counts)>n:
            hop_counts.sort(key=lambda x: x[1])
            return [a for a,b in hop_counts[:n]]
        else:
            return [a for a,b in hop_counts]

    def state_exchanger(self):
        while self.active:
            yield self.env.timeout(self.state_update_period)
            if not self.active:
                break
            for neighbor in self.neighbors:
                if 'New' in self.mode:
                    n = len(self.neighbors)
                    if n:
                        position_sums = [sum(items) for items in zip(*self.neighbor_states.values())]
                        av1, av2, av3 = 2 * position_sums[3] / n, 2 * position_sums[7] / n, 2 * position_sums[11] / n
                    else:
                        av1, av2, av3 = 1, 1, 1
                    specified_values_list = [1,0,1,av1,1,0,1,av2,1,0,1,av3]
                    """
                    这是一个 12 维的默认模板，用于补全「不足 4 个邻居时的虚拟邻居状态」（参考之前 get_current_state 中补全邻居的逻辑）；
                    模板中固定值(1、0)对应「邻居的基础属性默认值」(如是否在线、是否有任务),av1/av2/av3 对应「邻居的统计平均属性」，确保虚拟邻居状态贴近真实邻居的整体情况。
                    """
                    self_value=[self.is_producing, 1 - self.current_memory_occupy / self.memory,(self.computing_remain / self.computing_ability-self.is_computing*(self.env.now-self.last_computing_time))/ CT_FAC ,sum(self.transmission_size.values())/self.memory]
                    """
                    self_value = [
                    self.is_producing,  # 1. 是否为数据产生节点(0=否,1=是）
                    1 - self.current_memory_occupy / self.memory,  # 2. 剩余内存比例(归一化到0~1)
                    (self.computing_remain / self.computing_ability - self.is_computing*(self.env.now-self.last_computing_time))/ CT_FAC,  # 3. 剩余计算能力比例（归一化）
                    sum(self.transmission_size.values())/self.memory  # 4. 总传输队列占用比例（归一化）
                    ]

                    self_value 是当前卫星的核心自身属性(4 维），反映卫星的资源空闲情况，是状态的基础；
                    计算逻辑：所有值均归一化到 0~1 区间，避免因数值范围差异影响强化学习模型的训练。
                    """
                    self_values= [0,0,0,0]+[x * n for x in self.current_state]+[0,0,0,0]# self_values：自身当前状态 × 邻居数量（用于后续抵消邻居状态求和的影响）
                    additional_values = [x * (4 - n) for x in specified_values_list]# additional_values：默认模板 × （4-邻居数量）（用于补全不足4个邻居的状态）
                    temp = [sum(tup) + add - sub for tup, add,sub in zip(zip(*self.neighbor_states.values()), additional_values,self_values)]# temp：融合所有邻居状态 + 补全状态 - 自身状态影响（得到邻居的「平均化状态」）
                    """
                    核心目的：将「真实邻居状态 + 虚拟补全状态」融合为一个平滑的「邻居整体状态」，避免因邻居数量波动导致状态维度变化；
                    最终 temp 是 12 维，反映邻居的平均情况，后续会与 self_value 拼接成当前卫星的完整状态。
                    """
                    
                    self.current_state=self_value+temp[0:8]# 自身核心属性（4维） + 邻居平均状态（8维）= 12维
                    self.env.process(self.propagator.send_state(self.name, neighbor, self.current_state))
                    """self.current_state 最终为 12 维，包含「自身资源 + 邻居整体情况」，发送给邻居后，邻居会将其更新到自己的 self.neighbor_states 中（实现状态同步）。"""
                else:
                    #传统模式下状态简化为 4 维，无需融合邻居统计信息，直接提取自身核心属性发送
                    self.env.process(self.propagator.send_state(self.name, neighbor,[self.is_producing,1-self.current_memory_occupy/self.memory,(self.computing_remain / self.computing_ability-self.is_computing*(self.env.now-self.last_computing_time))/ CT_FAC,self.transmission_size[neighbor]/self.memory]))
    # def hidden_state_exchanger(self):
    #     """
    #     异步隐藏状态交换流程（测试模式使用）
        
    #     时序设计：
    #     ============================================================================
    #     假设 state_update_period = 0.2s（T）
        
    #     原方案（只交换原始状态）：
    #         t=0:    发送 current_state 给邻居
    #         t=T:    发送 current_state 给邻居
    #         t=2T:   发送 current_state 给邻居
    #         ...
        
    #     新方案（交换原始特征 + 隐藏状态）：
    #         采用「交替交换」策略，每个周期交换一种类型的信息
            
    #         t=0:    发送 raw_feat（原始特征）给邻居
    #                 收到邻居的 raw_feat 后，用 enc['2hop'] 聚合得到 aggregated_hidden
    #         t=T:    发送 aggregated_hidden（隐藏状态）给邻居
    #                 收到邻居的 hidden 后，缓存供决策使用
    #         t=2T:   发送 raw_feat（原始特征）给邻居
    #         t=3T:   发送 aggregated_hidden（隐藏状态）给邻居
    #         ...
        
    #     决策时：
    #         - 使用当前缓存中的邻居隐藏状态（可能是上一轮的）
    #         - 这是分布式系统中可接受的异步性
    #     ============================================================================
    #     """
    #     exchange_round = 0  # 0=发送raw_feat, 1=发送hidden
        
    #     while self.active and self._use_hidden_exchange:
    #         yield self.env.timeout(self.state_update_period)
    #         if not self.active:
    #             break
            
    #         if exchange_round == 0:
    #             # === 奇数轮：交换原始特征 ===
    #             raw_feat = self.compute_raw_feat()
    #             for neighbor in self.neighbors:
    #                 self.env.process(self.propagator.send_raw_feat(
    #                     self.name, neighbor, raw_feat))
                
    #             # 用已收到的邻居原始特征进行聚合（enc['2hop']）
    #             # 注意：这里使用的是上一轮收到的 raw_feat
    #             self.aggregate_round1()
                
    #         else:
    #             # === 偶数轮：交换隐藏状态 ===
    #             hidden = self.get_hidden_to_send()
    #             if hidden is not None:
    #                 for neighbor in self.neighbors:
    #                     self.env.process(self.propagator.send_hidden(
    #                         self.name, neighbor, hidden))
            
    #         # 切换交换类型
    #         exchange_round = 1 - exchange_round

    def all_start(self):
        super().all_start()
        if self.isLeo == True:
            self.env.process(self.computing_packet())
            self.env.process(self.offload_to_ground())
        if self.isLeo == False:
            self.env.process(self.collect_leo_states_from_domain())
            self.env.process(self.state_exchanger_meo())
        self.env.process(self.state_exchanger())
        # # 如果启用了隐藏状态交换，启动交换进程
        # if self._use_hidden_exchange:
        #     self.env.process(self.hidden_state_exchanger())

    def self_missing(self):
        self.active = False
        while self.computing_queue.items:
            packet=yield self.computing_queue.get()
            self.logger.log(f"Time {self.env.now:.3f}: {packet} is dropped because of satellite missing.")
        self.current_computing_queue_size = 0
        while self.offload_queue.items:
            packet = yield self.env.process(self.pop_offload())
            self.logger.log(f"Time {self.env.now:.3f}: {packet} is dropped because of satellite missing.")
        for neighbor in self.neighbors:
            while self.transmission_queue[neighbor].items:
                packet = yield self.env.process(self.pop_transmission(neighbor))
                self.logger.log(f"Time {self.env.now:.3f}: {packet} is dropped because of satellite missing.")
            while self.forward_queue.items:
                packet = yield self.env.process(self.forward_queue.get())
                self.logger.log(f"Time {self.env.now:.3f}: {packet} is dropped because of satellite missing.")
        self.current_memory_occupy=0
        if self.mode in ["Tradition","Ground"]:
            self.propagator.graph.nodes[self.name]['current_memory_occupy'] = 0
        self.is_producing = 0
        self.computing_remain = 0

class SatelliteNetworkSimulator_OnbardComputing(SatelliteNetworkSimulator):
    @staticmethod
    def _safe_max_hop(graph):
        """Return a robust hop upper bound for both connected and disconnected graphs."""
        if graph.number_of_nodes() <= 1:
            return 1
        if nx.is_connected(graph):
            return max(1, nx.diameter(graph))

        # For disconnected graphs, use the maximum diameter among connected components.
        max_diameter = 0
        for nodes in nx.connected_components(graph):
            subgraph = graph.subgraph(nodes)
            if subgraph.number_of_nodes() <= 1:
                component_diameter = 0
            else:
                component_diameter = nx.diameter(subgraph)
            if component_diameter > max_diameter:
                max_diameter = component_diameter
        return max(1, max_diameter)
    
    def __init__(self,mode,select_mode,q_net,epsilon,reward_factors,device,mission_possibility,poisson_rate,packet_frequency,computing_demand_factor,computing_demand_factor_2,size_after_computing_factor,size_after_computing_1,
                 graph,landmarks,mean_interval_time,memory,computing_ability,transmission_rate,downlink_rate,downstream_delays,
                 packet_size_range,state_update_period,logger,maxOrbitNumber, maxLeoSatelliteNumber,controlDomainNumber,minimuElevationAngle, showLink = False,transformer_module = None,n_hops=1,obs_type='flat',obs_wrapper=None,
                 domainPartitionMethod='Eunomia', rectangular_m=1, rectangular_n=1, meo_state_update_period=None, leo_action_mask_enabled=False):
        self.q_net = q_net
        self.mode=mode
        self.epsilon=epsilon
        self.transformer_module = transformer_module
        self.reward_function=Reward_Function(*reward_factors)
        self.device=device
        self.n_hops=n_hops  # GNN邻居跳数配置：1=仅1跳邻居，2=包含2跳邻居
        self.obs_type=obs_type  # 观测类型: 'flat', 'relational', 'graph'
        self.obs_wrapper=obs_wrapper  # 观测包装器实例
        self.leo_action_mask_enabled = bool(leo_action_mask_enabled)
        self.mission_possibility=mission_possibility# 不同类型任务的概率分布
        self.poisson_rate=poisson_rate# 数据包生成的泊松过程率（控制流量强度）
        self.packet_frequency=packet_frequency# 每个会话的数据包生成频率
        self.computing_demand_factor=computing_demand_factor # 类型0任务的计算需求系数
        self.computing_demand_factor_2=computing_demand_factor_2 # 类型1任务的计算需求系数
        self.size_after_computing_factor=size_after_computing_factor # 类型0任务计算后的大小系数    
        self.size_after_computing_1=size_after_computing_1# 类型1任务计算后的固定大小
        self.graph = graph
        self.max_hop=self._safe_max_hop(self.graph)# 网络最大跳数（图的直径）
        self.max_size=packet_size_range[1]
        self.env = simpy.Environment()
        self.memory = memory
        self.computing_ability = computing_ability
        self.logger=logger
        self.transmission_rate=transmission_rate# 卫星间传输速率
        self.downstream_delays=downstream_delays # 下行链路延迟
        self.downlink_rate=downlink_rate # 地面站与卫星的连接关系
        self.state_update_period=state_update_period
        self.meo_state_update_period = meo_state_update_period if meo_state_update_period is not None else state_update_period
        self.landmarks=landmarks
        self.direct_satellites = set(sum(self.landmarks.values(), [])) # 直接连接地面站的卫星集合
        self.statics_datas = {'Total': 0, 'Reached_0': 0, 'Reached_1': 0,'Reached_after_computed_0': 0, 'Reached_after_computed_1': 0, 'Lost_upload': 0, 'Lost_relay_0': 0, 'Lost_relay_1': 0, 'out_memory_0': 0, 'out_memory_1': 0, 'illegal_action_0': 0, 'illegal_action_1': 0, 'Total_delay_0': 0,'Total_delay_1': 0, 'Total_hops_0': 0,'Total_hops_1': 0,'Is_computing': 0,'Computing_waiting_time':0}
        self.satellite_names=[node for node in self.graph.nodes]
        self.satellites={} # 卫星节点实例字典（键：卫星名称，值：Satellite_with_Computing实例）
        self.select_mode=select_mode
        """为每个节点创建卫星对象，并传入邻居、资源等参数。
        注意：这里传给 Satellite_with_Computing 的参数顺序/数量要和类定义一致；后续 upgrade_all 中使用的构造调用不一致，可能导致错误。"""
        for node in self.graph.nodes:
            satellite_name_with_suffix = node
            self.satellites[satellite_name_with_suffix] = Satellite_with_Computing(
                self.mode,
                self.select_mode,
                self.epsilon,
                self.max_hop,
                self.max_size,
                self.device,
                self.env,
                satellite_name_with_suffix,
                [neighbor for neighbor in self.graph.neighbors(node)],
                self.memory,
                self.computing_ability,
                self.transmission_rate,
                self.downlink_rate,
                self.state_update_period,
                True if node in self.direct_satellites else False,
                self.logger,
                self.statics_datas,
                heartbeat_timeout = 0.25 if isLeo(node) else 1,
                n_hops=self.n_hops,  # 传递GNN邻居跳数配置
                meo_state_update_period=self.meo_state_update_period,
                leo_action_mask_enabled=self.leo_action_mask_enabled,
            )
        self.propagator = Propagator_Computing(self.env, graph, logger, self.satellites,self.statics_datas, True if self.mode in ["Tradition","Ground"] else False)
        self.propagator.transformer_module = self.transformer_module
        self.mean_interval_time=mean_interval_time# 任务会话的平均间隔时间
        self.size_range = packet_size_range# 数据包大小范围
        # 传递参数给传播器（包括观测类型和包装器）
        self.propagator.trans_parameters(
            self.max_hop, 
            self.downstream_delays, 
            self.reward_function,
            n_hops=self.n_hops,
            obs_type=self.obs_type,
            obs_wrapper=self.obs_wrapper
        )
        """关联各组件的依赖关系(如路由表、Q 网络）。
        为每个卫星设置邻接表、传参、设置传播器并建立路由表。adjacency_table 这里是同一份 snapshot(包含时间戳），后续应由交换函数更新。
        
        """
        for satellite in self.satellites:
            self.satellites[satellite].adjacency_table=self.extract_adjacency_dict()
            self.satellites[satellite].trans_parameters(self.q_net,self.reward_function,self.direct_satellites)
            self.satellites[satellite].set_propagator(self.propagator)  # 关联传播器
            self.satellites[satellite].transformer_module = self.transformer_module
            self.satellites[satellite].build_routing_table()
        self.maxOrbitNumber = maxOrbitNumber
        self.maxLeoSatelliteNumber = maxLeoSatelliteNumber
        self.controlDomainNumber = controlDomainNumber
        self.minimuElevationAngle = minimuElevationAngle
        self.domainPartitionMethod = domainPartitionMethod
        self.rectangular_m = rectangular_m
        self.rectangular_n = rectangular_n
        self.satelliteDomainPartitioner = SatelliteDomainPartitioner(orbitNumber = self.maxOrbitNumber, leoSatNumberPerOrbit = self.maxLeoSatelliteNumber, episode_max = 200 ,
                                                                     controlDomainNumber = self.controlDomainNumber, delayMap = self.propagator.propagation_delays, satellite_names = self.satellite_names, 
                                                                     minimuElevationAngle = self.minimuElevationAngle, satellites = self.satellites, showLink = showLink,
                                                                     partitionMethod = self.domainPartitionMethod, rectangular_m = self.rectangular_m, rectangular_n = self.rectangular_n)
    def extract_adjacency_dict(self):
        adjacency_dict = {}
        for node in self.satellite_names:
            neighbors = [f"{neighbor}" for neighbor in self.graph.neighbors(node)] # 节点的所有邻居
            adjacency_dict[f"{node}"] = (neighbors, self.env.now)# 邻接表格式：{节点: (邻居列表, 更新时间)}
        return adjacency_dict

    def generate_traffic(self, satellite):
        if self.packet_frequency <= 0 or not self.satellites[satellite].isLeo:
            return
        while satellite in self.satellite_names:# 卫星存在时持续生成流量
            self.satellites[satellite].is_producing=0# 标记为"非生成中"
            session_start_time = min(random.expovariate(1.0 / self.poisson_rate),self.poisson_rate*3)# 随机生成会话开始时间（服从指数分布，控制流量到达间隔）
            self.satellites[satellite].next_traffic_session_start_time = self.env.now + session_start_time
            #session_start_time的值的平均值是30s
            yield self.env.timeout(session_start_time)# 等待会话开始
            if not satellite in self.satellite_names: # 卫星已消失，停止生成
                self.logger.log(f"Time {self.env.now:.3f}: {satellite} is missed, packets failed to generate.")
                break
            type = random.choices([0, 1], weights=self.mission_possibility, k=1)[0]# 随机选择任务类型（0或1，按mission_possibility概率）
            if satellite in self.direct_satellites:# 直接连接地面站的卫星不生成任务（假设任务由非地面站卫星发起）
                continue
                """通过路由表找最近的直连卫星（按跳数），在并列最小跳数中随机选一个。如果没有可达的直连卫星，应该更明确地终止本次会话（例如 continue)，而不是让 destination 未定义。"""
            else:
                if self.select_mode == 1 or self.mode in ["Tradition","Ground"]:# 选择目标地面站（模式1或传统模式下，选跳数最少的地面站）
                    min_hops = np.inf
                    min_hops_destinations=[]
                    for destination in self.direct_satellites:
                        if destination in self.satellites[satellite].routing_tables:
                            hops = self.satellites[satellite].routing_tables[destination][1] # 获取跳数
                        else:
                            hops = np.inf
                         # 记录跳数最少的目标
                        if hops < min_hops:
                            min_hops = hops
                            min_hops_destinations = [destination]
                        elif hops == min_hops:#若当前目标的跳数等于已记录的最小跳数，说明它与现有最优目标等价
                            min_hops_destinations.append(destination)
                    if min_hops_destinations:
                        destination = np.random.choice(min_hops_destinations)# 从最优目标中随机选
                    else:
                        self.logger.log(f"Time {self.env.now:.3f}: connections for {satellite} is not availiable, packet failed to generate.")
            intra_traffic = False
            # if destination in self.satellites[satellite].intra_domain_leo:
            #     intra_traffic = True
            #     return
            # 会话持续时间（服从指数分布）
            session_duration = min(random.expovariate(1.0 / self.mean_interval_time),self.mean_interval_time*3)
            # session_duration = 3
            # session_duration = min(session_duration, 2)
            end_time = self.env.now + session_duration# 会话结束时间
            self.satellites[satellite].is_producing = 1# 标记为"生成中"
            self.satellites[satellite].traffic_session_start_time = self.env.now
            self.satellites[satellite].traffic_session_duration = session_duration
            self.satellites[satellite].traffic_session_end_time = end_time
            self.satellites[satellite].next_traffic_session_start_time = None
             # 传统模式下更新邻居图（用于路由计算）
            if self.mode in ["Tradition","Ground"]:
                neighbors=[]
                for _satellite in self.satellites:
                    """在特定模式下，为源卫星构建一个局部范围的 “邻居图”，仅包含那些既能到达源卫星又能到达目标节点、且总跳数在合理范围内的卫星，
                        以便后续基于该局部图进行高效的路由计算或决策（避免使用全网图带来的冗余计算）。"""
                        # 筛选距离目标较近的卫星作为邻居图节点
                    if satellite in self.satellites[_satellite].routing_tables and destination in self.satellites[_satellite].routing_tables:
                        if (self.satellites[_satellite].routing_tables[satellite][1]+self.satellites[_satellite].routing_tables[destination][1])<=min_hops+4:
                            neighbors.append(self.satellites[_satellite].name)
                self.satellites[satellite].neighbor_graph=self.propagator.graph.subgraph(neighbors)
                """基于筛选出的neighbors列表,从全局卫星图(self.propagator.graph)中提取子图(subgraph),并将该子图赋值给源卫星(satellite)的neighbor_graph属性。"""
            # 会话期间持续生成数据包

            while self.env.now < end_time:
                yield self.env.timeout(1.0 / self.packet_frequency)# 按频率生成数据包
                 # 随机生成数据包大小（在size_range范围内）
                size = random.randint(self.size_range[0], self.size_range[1])
                 # 根据任务类型设置计算需求和计算后大小
                if type == 0:
                    size_after_computing = int(random.uniform(self.size_after_computing_factor[0], self.size_after_computing_factor[1]) * size)
                    computing_demand = int(random.uniform(self.computing_demand_factor[0], self.computing_demand_factor[1]) * size)
                else:
                    size_after_computing = self.size_after_computing_1
                    computing_demand = int(random.uniform(self.computing_demand_factor_2[0], self.computing_demand_factor_2[1]) * size)\
                
                 # 非模式1时，根据评分选择目标地面站
                if self.select_mode != 1:
                    min_hops_destinations=self.satellites[satellite].find_min_hops_destinations(5)
                    hightest_score_destinations=self.satellites[satellite].find_highest_score(min_hops_destinations,[type,size,computing_demand,size_after_computing],0,0)
                    if hightest_score_destinations:
                        destination = np.random.choice(hightest_score_destinations)
                    else:
                        destination = 'False'
                 # 检查目标和源卫星是否存在
                if not (destination in self.satellite_names and satellite in self.satellite_names):
                    self.logger.log(f"Time {self.env.now:.3f}: {satellite} or {destination} is missed, packet failed to generate.")
                    break
                src_meo = self.satellites[satellite].masterMeo
                dest_meo = self.satellites[destination].masterMeo
                # if self.transformer_module != None and intra_traffic == False:# and plan == None:
                #     plan = self.transformer_module.recommend_path(
                #         source=satellite,
                #         destination=destination,
                #         packet_size=size,
                #         computing_demand=computing_demand,
                #         size_after_computing=size_after_computing,
                #         business_duration=None#max(0.0, end_time - self.env.now),
                #     )
                plan = None
                if src_meo is not None and dest_meo is not None and src_meo != dest_meo:
                    plan = self.satellites[self.satellites[satellite].masterMeo].recommend_path(
                        src=satellite,
                        dst=destination,
                        packet_size=size,
                        task_type=type,
                        computing_demand=computing_demand,
                        size_after_computing=size_after_computing,
                        is_computed=False,
                    )
                
                # 生成数据包并尝试推送至卫星的转发队列
                if destination in self.satellites[satellite].routing_tables:
                    packet = Packet(satellite, destination, self.env.now, size)
                    packet.is_intra_destination = intra_traffic
                    if intra_traffic is False and plan != None:
                        temporary_destination = []
                        temporary_destination = plan["boundary_sat"]
                        packet.add_temporary_destination(temporary_destination)
                        transformer = getattr(self, 'transformer_module', None)
                        meo_router = getattr(transformer, 'meo_router', None)
                        if meo_router is not None:
                            meo_router.attach_decision(packet, plan)
                        # packet.add_compute_flat(plan.compute_flags)
                    # 附加数据包信息（是否计算、类型、计算需求等）
                    # 7个元素: [is_computed, type, computing_demand, size_after_computing, last_time, last_obs, last_action]
                    packet.extra_information([False, type, computing_demand, size_after_computing, self.env.now, None, None])
                    self.statics_datas['Total'] += 1# 总数据包数+1
                    self.logger.log(f"Time {self.env.now:.3f}: {satellite}: Packet generated: {(satellite, destination,packet.creation_time)}.")
                     # 尝试将数据包推入卫星的转发队列
                    success = self.satellites[satellite].push_forward(packet)
                    if success:
                        # if plan != None and hasattr(self.transformer_module, "reserve_plan"):
                        #     self.transformer_module.reserve_plan(
                        #         plan=plan,
                        #         packet_size=size,
                        #         computing_demand=computing_demand,
                        #         size_after_computing=size_after_computing,
                        #     )
                        self.logger.log(f"Time {self.env.now:.3f}: {satellite}: Packet {(packet.source, packet.destination,packet.creation_time)} received by router. Memory remain: {self.satellites[satellite].current_memory_occupy}.",detail=True)
                    else:
                        self.statics_datas['Lost_upload'] += 1
                        self.logger.log(f"Time {self.env.now:.3f}: {satellite}: Routing queue is full, discarding packet ({satellite}, {destination},packet.creation_time).")
                    if self.mode in ["Tradition","Ground"]:# 传统模式下更新邻居图
                        neighbors = []
                        for _satellite in self.satellites:
                            if _satellite in self.satellites[satellite].routing_tables and destination in self.satellites[_satellite].routing_tables:
                                if (self.satellites[satellite].routing_tables[_satellite][1] + self.satellites[_satellite].routing_tables[destination][1]) <= min_hops + 4:
                                    neighbors.append(self.satellites[_satellite].name)
                        self.satellites[satellite].neighbor_graph = self.propagator.graph.subgraph(neighbors)
                else:
                    self.logger.log(f"Time {self.env.now:.3f}: {destination} is missed, packet failed to generate.")
                    break
            self.satellites[satellite].is_producing = 0# 会话结束，标记为"非生成中"
            self.satellites[satellite].traffic_session_start_time = None
            self.satellites[satellite].traffic_session_end_time = None
            self.satellites[satellite].traffic_session_duration = 0.0

    def get_system_state(self):
        total_queue_usage = {}
        total_computing_memory={}
        for node in self.satellite_names:
            average_usage = min(self.satellites[node].current_memory_occupy / self.memory,1)
            computing_memory = self.satellites[node].current_computing_queue_size / self.memory
            total_queue_usage[node] = average_usage
            total_computing_memory[node] = computing_memory
        return total_queue_usage

    def run(self, duration):
        if self.env.now==0:
            for satellite in self.satellite_names:
                self.env.process(self.generate_traffic(satellite))
            for satellite in self.satellites:
                self.satellites[satellite].all_start()

        # start_wall = time.perf_counter()
        # start_sim = self.env.now
        # until = self.env.now + duration
        # event_count = 0

        # while self.env.peek() < until:
        #     self.env.step()
        #     event_count += 1

        # if self.env.now < until:
        #     self.env.run(until=until)
        self.env.run(until=self.env.now+duration)

        if 'Is_computing' in self.statics_datas:
            for satellite in self.satellites:
                self.statics_datas['Is_computing']+= self.satellites[satellite].is_computing

    def upgrade_all(self,graph,landmarks):

        self.landmarks = landmarks
        new_nodes = set(graph.nodes())#graph是新传入的图
        old_nodes = set(self.graph.nodes())
        new_edges = set(graph.edges())
        old_edges = set(self.graph.edges())
        old_direct_satellites=self.direct_satellites
        # 1.基础参数更新：同步新拓扑的核心信息
        self.satellite_names = [node for node in graph]# 更新卫星名称列表为新拓扑的节点
        self.graph = graph# 用新拓扑图替换旧图
        self.propagator.update(graph) # 通知传播器更新拓扑（用于重新计算传播延迟）
        self.propagator.reset_parameters()# 重置传播器的参数（如延迟缓存、奖励相关参数）
        # 传递新参数给传播器（包括观测类型和包装器）
        self.propagator.trans_parameters(
            self.max_hop, 
            self.downstream_delays, 
            self.reward_function,
            n_hops=self.n_hops,
            obs_type=self.obs_type,
            obs_wrapper=self.obs_wrapper
        )
        #2.处理 “直接连接卫星” 的状态变化 # 提取新拓扑中与地面站直接连接的卫星（新直接连接卫星）
        flattened_list = set(sum(self.landmarks.values(), []))# self.landmarks是地面站-卫星连接，汇总所有连接的卫星；landmarks是传入的新的connections
        new_direct_satellites = flattened_list
        self.direct_satellites = flattened_list# 更新仿真器的直接连接卫星列表
        for node in old_direct_satellites-new_direct_satellites:# 对“不再是直接连接”的卫星：关闭下行链路标志
            self.satellites[node].is_downlink=False
        for node in new_direct_satellites-old_direct_satellites:# # 对“新增的直接连接”卫星：开启下行链路标志
            self.satellites[node].is_downlink = True
        # 3. 新增节点（卫星）的初始化与加入
        for node in new_nodes - old_nodes:
            self.satellites[node] = Satellite_with_Computing(
                self.mode,
                self.select_mode,
                self.epsilon,
                self.max_hop,
                self.max_size,
                self.device,
                self.env,
                node,
                [neighbor for neighbor in self.graph.neighbors(node)],
                self.memory,
                self.computing_ability,
                self.transmission_rate,
                self.downlink_rate,
                self.state_update_period,
                True if node in self.direct_satellites else False,
                self.logger,
                self.statics_datas,
                n_hops=self.n_hops,  # 传递GNN邻居跳数配置
                meo_state_update_period=self.meo_state_update_period,
                leo_action_mask_enabled=self.leo_action_mask_enabled,
            )# 创建新卫星实例（带计算能力的卫星）. # is_downlink标记是否直接连接地面站
            self.satellites[node].set_propagator(self.propagator)# 为新卫星绑定传播器
            self.satellites[node].all_start()# 启动卫星的所有核心进程（转发、传输、计算等）
            self.satellites[node].adjacency_table_exchanger()# 交换邻接表（获取邻居信息）
            self.env.process(self.generate_traffic(node))# 为新卫星启动流量生成进程（产生数据包）

        for node in old_nodes - new_nodes:
            for i, satellites in enumerate(self.satellites):
                self.env.process(satellites[node].self_missing())
                del satellites[node]
        for edge in new_edges - old_edges:
            node, neighbor = edge
            self.satellites[node].add_neighbor(neighbor)
            self.satellites[neighbor].add_neighbor(node)
        for satellite in self.satellites:
            self.satellites[satellite].trans_parameters(self.q_net,self.reward_function,self.direct_satellites)
