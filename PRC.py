import argparse
import yaml
import random
import torch
import numpy as np
# from torch.utils.tensorboard import SummaryWriter

# # DEFAULT_CONFIIG = "train_NewDDQN_dueling.yaml"
# def parse_args():
#     parser = argparse.ArgumentParser(description="Run the satellite simulation with specified configuration file.")
#     parser.add_argument('--config', type=str, required=True, help='Path to the configuration YAML file')
#     return parser.parse_args()
DEFAULT_CONFIG_PATH = "train/train_PurePPO_shuffle.yaml"  # 修改为你的默认配置文件路径

def parse_args():
    parser = argparse.ArgumentParser(description="Run the satellite simulation with specified configuration file.")
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG_PATH,
                        help=f'Path to the configuration YAML file (default: {DEFAULT_CONFIG_PATH})')
    return parser.parse_args()


def load_config(path):
    with open(path, 'r') as file:
        return yaml.load(file, Loader=yaml.FullLoader)


args = parse_args()
config = load_config(args.config)

# config = load_config('train_NewDQN.yaml')

random.seed(config['general']['random_seed'])
torch.manual_seed(config['general']['random_seed'])
np.random.seed(config['general']['random_seed'])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config['general']['random_seed'])

from RL_environment_for_computing import SatelliteEnv
from Base_Agents import DDQN_Agent, ShuffleEx, cal_agent_dim, PPO_Agent, DQN_Agent

phase = config['general']['phase']
# writer = SummaryWriter("./logs_train")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mode = config['agent']['mode']
if mode in ['Pure_DQN', "New_DQN", "Pure_PPO","New_PPO","Weak_DQN"]:
    state_dim, action_dim, state_mask = cal_agent_dim(neighbors_dim= config['agent']['neighbors_dim'],
                                                      edges_dim= config['agent']['edges_dim'],
                                                      distance_dim= config['agent']['distance_dim'],
                                                      mission_dim= config['agent']['mission_dim'],
                                                      current_dim= config['agent']['current_dim'],
                                                      action_dim=config['agent']['action_dim']                                                     
                                                      )

    if 'DQN' in mode:
        if 'Weak' in mode:
            Agent = DQN_Agent
        else:
            Agent = DDQN_Agent
    elif 'PPO' in mode:
        Agent = PPO_Agent
    agent = Agent(state_dim=state_dim,
                       hidden_dim=config['agent']['hidden_dim'],#神经网络隐藏层维度
                       action_dim=action_dim,
                       buffer_length=config['agent']['buffer_length'],#经验回放池大小
                       batch_size=config['agent']['batch_size'],
                       gamma=config['agent']['gamma'],
                       device=device,
                       q_mask=config['agent']['q_mask'],
                       activation=config['agent']['activation'],#激活函数
                       hidden_layers=config['agent']['hidden_layers'],#隐藏层数量
                       dueling = config['agent']['dueling'],#是否使用Dueling DQN 
                       learning_rate=config['agent']['learning_rate'],
                       repeat=config['agent']['repeat'],#重复动作次数
                       shuffle_func=ShuffleEx(state_mask).shuffle if config['agent']['shuffle'] else None)#状态打乱函数（用于数据增强）
    if phase != 'train':
        agent.load_model(config['agent']['model_path'])
else:
    agent = None

env = SatelliteEnv(mode=config['agent']['mode'],
                   select_mode=config['general']['select_mode'],#动作选择模式
                   q_net=agent.online_net if agent else None,
                   epsilon=config['general']['epsilon'],#探索率（ε-greedy策略）
                   reward_factors=config['general']['reward_factors'],#奖励函数权重
                   device=device,
                   leo_action_mask_enabled=config['agent'].get('leo_action_mask_enabled', False),
                   #环境参数（从配置文件读取）
                   mission_possibility=config['environment']['mission_possibility'],# 任务生成概率
                   poisson_rate=config['environment']['poisson_rate'],#泊松过程速率（流量生成）
                   packet_frequency=config['environment']['packet_frequency'],#数据包生成频率
                   computing_demand_factor=config['environment']['computing_demand_factor'], # 计算需求因子
                   computing_demand_factor_2=config['environment']['computing_demand_factor_2'],#第二类任务的计算需求范围（单位通常为计算资源单位，如 FLOPS 或处理时间）。
                   #区分不同类型任务的计算复杂度。例如，配置为 [300, 500] 表示第二类任务的计算需求在 300 到 500 单位之间随机取值，用于模拟多样化的任务负载。
                   size_after_computing_factor=config['environment']['size_after_computing_factor'],#计算后数据量因子
                   size_after_computing_1=config['environment']['size_after_computing_1'],#任务经过计算处理后的数据量基准值（单位通常为字节）
                   begin_time=config['general']['begin_time'],#仿真开始时间
                   end_time=None,
                   time_stride=config['general']['time_stride'],# 时间步长
                   tle_filepath=config['environment']['tle_filepath'],#卫星轨道数据（TLE）路径
                   SOD_file_path=config['environment']['SOD_file_path'],#地面站数据路径
                   mean_interval_time=config['environment']['mean_interval_time'],#任务（或数据包）生成的平均时间间隔（单位通常为秒）
                   memory=config['environment']['memory'],#卫星内存大小
                   computing_ability=config['environment']['computing_ability'],#卫星计算能力
                   transmission_rate=config['environment']['transmission_rate'],#星间传输速率
                   downlink_rate=config['environment']['downlink_rate'],#下行链路速率
                   downstream_delays=config['environment']['downstream_delays'],
                   packet_size_range=config['environment']['packet_size_range'],
                   state_update_period=config['environment']['state_update_period'],
                   print_cycle=config['general']['print_cycle'],#日志打印周期（单位：秒），每隔该时间打印一次性能指标（如丢包率、延迟）。
                   del_cycle=config['environment']['del_cycle'],#随机删除周期（单位：秒），每隔该时间随机删除指定数量的节点 / 边（模拟网络故障）。
                   visualize=config['environment']['visualize'],
                   print_info=config['environment']['print_info'],
                   show_detail=config['environment']['show_detail'],
                   save_log=config['environment']['save_log'],
                   random_edges_del=config['environment']['random_edges_del'],
                   random_nodes_del=config['environment']['random_nodes_del'],
                   update_cycle=config['environment']['update_cycle'],#网络拓扑更新周期
                   save_training_data=config['environment']['save_training_data'],
                   training_data_dir=config['environment'].get('training_data_dir', './training_process_data'),
                   elevation_angle=config['environment']['elevation_angle'],#通信仰角阈值（地面站-卫星可见性）
                   controlDomainNumber=config['environment'].get('controlDomainNumber', 4),
                   minimuElevationAngle=config['environment'].get('MinimuElevationAngle', 25),
                   showLink=config['environment'].get('ShowLink', False),
                   pole=config['environment']['pole'],
                   domainPartitionMethod=config['environment'].get('domainPartitionMethod', 'Eunomia'),
                   rectangular_m=config['environment'].get('rectangular_m', 1),
                   rectangular_n=config['environment'].get('rectangular_n', 1))

begin_time = config['general']['begin_time']
time_stride = config['general']['time_stride']#将连续时间拆分为离散的时间步
rounds = config['general']['rounds']#总训练/测试轮数
skip_time = config['general']['skip_time']#每轮结束后跳过的时间（避免数据重复）
duration = config['general']['duration']#每轮仿真持续时间
epsilon = config['general']['epsilon']#初始探索率
min_epsilon = config['general']['min_epsilon']
epsilon_decay = config['general']['epsilon_decay']#探索率衰减系数

if phase != 'train':
    epsilon = 0

episode_rewards = []   # 存每轮总奖励
moving_avg_window = 100   # 移动平均窗口  

# for k in range(rounds):
#     env.reset(begin_time)#重置环境到初始状态

#     total_reward = 0  # 本轮累计奖励

#     #单轮内的时间步循环（根据持续时间和时间步长计算步数）
#     for t in range(int(duration / time_stride)):
#         experiences = env.step(epsilon)#环境执行一步（返回智能体经验）
#         #env.render()

#         for exp in experiences:
#             # 经验格式：[last_state, mark, last_action, reward, current_state, done]
#             # 对应 Base_Agents.py 中 agent.update 对经验的解析
#             _, _, _, reward, _, _ = exp  
#             total_reward += reward

#         if phase == 'train' and agent:
#             epsilon = max(min_epsilon, epsilon * epsilon_decay)
#             agent.update(experiences)#用经验更新智能体（如DQN的经验回放）
#             #当前时间步 t 满足 “（时间步 + 1）是更新周期的整数倍” 时，触发后续操作。例如，若 update_cycle=30，则在 t=29、59、89... 时执行（即每 30 个时间步执行一次）
#             #对于 DQN（包括 DDQN、Dueling DQN 等），由于其采用 “在线网络（online_net）” 和 “目标网络（target_net）” 双网络结构，需要定期将在线网络的参数同步到目标网络，以稳定训练。
#             if (t + 1) % int(config['agent']['update_cycle']) == 0:
#                 if 'DQN' in mode:
#                     agent.target_update() # DQN同步目标网络
#                 agent.save_model(config['agent']['model_path'])


#     # 记录本轮奖励
#     episode_rewards.append(total_reward)
#     writer.add_scalar("Reward/episodic", total_reward, k)

#     if len(episode_rewards) >= moving_avg_window:
#         moving_avg = np.mean(episode_rewards[-moving_avg_window:])
#         writer.add_scalar("Reward/moving_avg", moving_avg, k)
#     else:
#         moving_avg = np.mean(episode_rewards)
#         writer.add_scalar("Reward/moving_avg", moving_avg, k)

#     if phase == 'test':
#         env.show_satellite_computing_time()
   
#     begin_time = env.add_time_to_str(begin_time, skip_time)

# writer.close()

for k in range(rounds):
    env.reset(begin_time)
    for t in range(int(duration / time_stride)):
        experiences = env.step(epsilon) #测试模式下epsilon=0，纯贪心决策；始终选最优动作
        # env.render()
        if phase == 'train' and agent:
            epsilon = max(min_epsilon, epsilon * epsilon_decay)
            agent.update(experiences)
            if (t + 1) % int(config['agent']['update_cycle']) == 0:
                if 'DQN' in mode:
                    agent.target_update()
                agent.save_model(config['agent']['model_path'])
    if phase == 'test':
        env.show_satellite_computing_time()
    begin_time = env.add_time_to_str(begin_time, skip_time)
