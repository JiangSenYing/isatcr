"""
GNN-based Training Script for Satellite Routing with Computing Tasks

使用 GNNDDQNAgent 进行训练，支持三种观测模式:
1. flat: 扁平状态向量 (使用 SimpleQNetwork)
2. relational: DGL 图，边信息编码在节点特征中 (使用 RelationalController)
3. graph: DGL 图，边信息作为边特征 (使用 GraphController)

使用方法:
    python PRC_GNN.py --config train/train_GNN_DQN2hopnew.yaml

配置文件参数说明见 train_GNN_DQN2hop.yaml
"""

import argparse
import yaml
import random
import torch
import numpy as np
import os
import sys
from datetime import datetime

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEFAULT_CONFIG_PATH = "train/test_transformer.yaml"

try:
    from Transformer_module import (
        GlobalTransformerTrainer,
        average_metric_dicts,
        format_prediction_metrics,
    )
except ImportError:
    GlobalTransformerTrainer = None
    average_metric_dicts = None
    format_prediction_metrics = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train GNN-based satellite routing agent"
    )
    parser.add_argument(
        '--config', type=str, default=DEFAULT_CONFIG_PATH,
        help=f'Path to the configuration YAML file (default: {DEFAULT_CONFIG_PATH})'
    )
    parser.add_argument(
        '--device', type=str, default=None,
        help='Device to use (cuda/cpu). If not specified, use config or auto-detect.'
    )
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Random seed. If not specified, use config.'
    )
    parser.add_argument(
        '--obs_type', type=str, default=None, choices=['flat', 'relational', 'graph', 'relational_separated', 'graph_separated'],
        help='Observation type. Overrides config if specified.'
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    """加载 YAML 配置文件"""
    if not os.path.exists(path):
        print(f"Warning: Config file {path} not found, using default values")
        return get_default_config()
    
    with open(path, 'r', encoding='utf-8') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    
    # 合并默认配置
    default_config = get_default_config()
    config = merge_configs(default_config, config)
    
    return config


def get_default_config() -> dict:
    """获取默认配置"""
    return {
        'general': {
            'random_seed': 42,
            'phase': 'train',
            'begin_time': '2024-01-01 00:00:00',
            'time_stride': 1.0,
            'rounds': 100,
            'skip_time': [0, 3600],
            'duration': 3600,
            'print_cycle': 60,
            'select_mode': 0,
            'reward_factors': [10, 0.1, 5, 0.8, 0.1],
        },
        'agent': {
            'mode': 'GNN_DQN',
            'obs_type': 'graph',           # 观测类型: 'flat', 'relational', 'graph'
            'controller_type': 'graph',    # 控制器类型: 'relational', 'graph', 'simple'
            
            # 状态维度配置
            'agent_feats': 9,              # current_node(3) + mission(4) + routing(2)
            'nbr_feats': 4,                # 邻居节点特征维度（relational模式为12含边信息）
            'edge_feats': 3,               # 边特征维度（仅graph模式使用）
            'n_actions': 5,                # 动作数: 4邻居 + 1本地计算
            'max_nbrs': 4,                 # 最大邻居数
            
            # 网络结构
            'hidden_size': 64,
            'n_hops': 2,                   # GNN 消息传递跳数
            'use_rnn': False,              # 是否使用 RNN
            'dueling': True,               # Dueling 架构
            
            # 训练参数
            'buffer_size': 100000,
            'batch_size': 64,
            'gamma': 0.99,
            'learning_rate': 1e-4,
            'target_update_freq': 100,
            'epsilon_start': 1.0,
            'epsilon_end': 0.02,
            'epsilon_decay': 0.9995,

            # Diffusion action prior
            'use_diffusion': False,
            'diffusion_steps': 10,
            'diffusion_hidden_size': 128,
            'diffusion_loss_weight': 0.02,
            'diffusion_q_loss_weight': 0.1,
            'diffusion_q_temperature': 1.0,
            'diffusion_guidance_weight': 0.1,
            'diffusion_guidance_warmup_updates': 1000,
            'diffusion_deterministic': True,
            'diffusion_normalize_prior': True,
            'diffusion_prior_clip': 2.0,
            
            # 模型保存
            'model_path': 'model_weights/gnn_ddqn_model.pth',
        },
        'transformer': {
            'enabled': False,
            'history_len': 10,
            'forecast_horizon': 5,
            'batch_size': 6,
            'learning_rate': 1e-4,
            'd_model': 64,
            'nhead': 4,
            'num_layers': 4,
            'dim_feedforward': 64,
            'dropout': 0.1,
            'max_snapshots': 20000,
            'warmup_snapshots': 40,
            'update_every': 1,
            'updates_per_step': 1,
            'eval_every': 60,
            'plan_every': 60,
            'plan_top_k': 16,
            'plan_delay_top_k': 8,
            'plan_load_top_k': 8,
            'plan_max_candidate_expansions': 2000,
            'plan_max_candidate_queue_size': 5000,
            'plan_max_hops': 30,
            'plan_packet_size': 600000000.0,
            'plan_computing_demand': 250000000000.0,
            'plan_need_compute': True,
            'plan_source': None,
            'plan_destination': None,
            'model_path': 'model_weights/TEST_transformer.pth',
        },
        'environment': {
            'mission_possibility': [0.25, 0.25, 0.25, 0.25],
            'task_profiles': [
                {
                    'name': 'small_task',
                    'probability': 0.25,
                    'packet_size_range': [200000000.0, 600000000.0],
                    'computing_demand_factor': [150, 250],
                    'size_after_computing_factor': [0.08, 0.125],
                },
                {
                    'name': 'medium_task',
                    'probability': 0.25,
                    'packet_size_range': [200000000.0, 600000000.0],
                    'computing_demand_factor': [120, 220],
                    'size_after_computing_factor': [0.1, 0.2],
                },
                {
                    'name': 'compute_intensive',
                    'probability': 0.25,
                    'packet_size_range': [200000000.0, 600000000.0],
                    'computing_demand_factor': [300, 500],
                    'size_after_computing': 40000.0,
                },
                {
                    'name': 'data_intensive',
                    'probability': 0.25,
                    'packet_size_range': [500000000.0, 1000000000.0],
                    'computing_demand_factor': [180, 300],
                    'size_after_computing_factor': [0.03, 0.08],
                },
            ],
            'poisson_rate': 30,
            'packet_frequency': 0.5,
            'computing_demand_factor': [150, 250],
            'computing_demand_factor_2': [300, 500],
            'size_after_computing_factor': [0.08, 0.125],
            'size_after_computing_1': 40000.0,
            'packet_size_range': [200000000.0, 600000000.0],
            'tle_filepath': './Satellite_Data/60Degree_500_12x24_tles.txt',
            'SOD_file_path': './Ground_Data/11_ground_stations.txt',
            'elevation_angle': 45,
            'pole': False,
            'memory': 12000000000.0,
            'computing_ability': 50000000000.0,
            'transmission_rate': 1200000000.0,
            'downlink_rate': 3000000000.0,
            'downstream_delays': 0.0016667,
            'mean_interval_time': 30,
            'state_update_period': 0.1,
            'del_cycle': 30,
            'random_edges_del': 15,
            'random_nodes_del': 0,
            'update_cycle': 10,
            'visualize': False,
            'print_info': False,
            'show_detail': False,
            'save_log': False,
            'save_training_data': 'train_GNN_DQN.txt',
            'training_data_dir': './training_process_data',
            'business_record_enabled': False,
            'business_record_path': './training_process_data/business_traffic_records.txt',
            'business_record_reset': True,
        },
    }


def merge_configs(default: dict, override: dict) -> dict:
    """递归合并配置"""
    result = default.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def get_obs_shape(config: dict) -> dict:
    """
    根据配置获取观测形状字典
    
    Returns:
        obs_shape: {'agent': int, 'nbr': int, 'hop': int} 或 int (扁平模式)
    """
    agent_cfg = config['agent']
    obs_type = agent_cfg.get('obs_type', 'graph')
    
    if obs_type == 'flat':
        # 扁平模式：返回总状态维度
        # 计算: agent_feats + max_nbrs * nbr_feats + mission_dim + current_dim
        agent_feats = agent_cfg.get('agent_feats', 14)
        nbr_feats = agent_cfg.get('nbr_feats', 4)
        max_nbrs = agent_cfg.get('max_nbrs', 4)
        # 扁平状态向量维度（与原始 New 模式兼容）
        state_dim = agent_cfg.get('state_dim', 33)  # 4邻居*6 + current(3) + mission(4) + routing(2)
        return state_dim
    
    elif obs_type == 'relational':
        # 关系模式：边信息编码在节点特征中
        # 注意：RelationalObservation 会去除 availability 列，所以实际维度是 nbr_feats_relational - 1
        nbr_feats_raw = agent_cfg.get('nbr_feats_relational', 6)
        return {
            'agent': agent_cfg.get('agent_feats', 14),
            'nbr': nbr_feats_raw - 1,  # 去除 availability 列
        }
    
    elif obs_type == 'relational_separated':
        # 分离模式：agent 节点仅包含环境特征（3维），任务特征单独传递
        nbr_feats_raw = agent_cfg.get('nbr_feats_relational', 6)
        return {
            'agent': agent_cfg.get('agent_feats', 3),  # 仅环境特征
            'nbr': nbr_feats_raw - 1,  # 去除 availability 列
        }
    
    elif obs_type == 'graph_separated':
        # 分离模式：agent 节点仅包含环境特征（3维），任务特征单独传递
        return {
            'agent': agent_cfg.get('agent_feats', 3),  # 仅环境特征
            'nbr': agent_cfg.get('nbr_feats', 4),
            'hop': agent_cfg.get('edge_feats', 3),
        }
    
    else:  # graph
        # 图模式：边信息作为边特征
        return {
            'agent': agent_cfg.get('agent_feats', 14),
            'nbr': agent_cfg.get('nbr_feats', 4),
            'hop': agent_cfg.get('edge_feats', 3),
        }


def create_gnn_agent(config: dict, device: torch.device):
    """
    创建 GNN DDQN 智能体
    
    根据 obs_type 自动选择合适的网络架构:
    - flat: SimpleQNetwork (MLP)
    - relational: RelationalController (GAT/GCN)
    - graph: GraphController (消息传递GNN)
    - relational_separated: TaskConditionedRelationalController (分离模式)
    - graph_separated: TaskConditionedGraphController (分离模式)
    """
    from gnn_modules.agents import GNNDDQNAgent
    
    agent_cfg = config['agent']
    obs_type = agent_cfg.get('obs_type', 'graph')
    
    # 获取观测形状
    obs_shape = get_obs_shape(config)
    
    # 确定控制器类型
    if obs_type == 'flat':
        controller_type = 'simple'
    else:
        controller_type = agent_cfg.get('controller_type', obs_type)
    
    # 分离模式的 task_dim 参数
    task_dim = agent_cfg.get('task_dim', None)
    if obs_type in ('relational_separated', 'graph_separated') and task_dim is None:
        task_dim = 6  # 默认值: mission(4) + routing(2)
    
    print(f"Creating GNNDDQNAgent:")
    print(f"  - obs_type: {obs_type}")
    print(f"  - controller_type: {controller_type}")
    print(f"  - obs_shape: {obs_shape}")
    print(f"  - n_hops: {agent_cfg.get('n_hops', 1)}")
    print(f"  - hidden_size: {agent_cfg.get('hidden_size', 64)}")
    print(f"  - dueling: {agent_cfg.get('dueling', True)}")
    print(f"  - conv_type: {agent_cfg.get('conv_type', 'gat')}")
    print(f"  - n_heads: {agent_cfg.get('n_heads', 4)}")
    print(f"  - graph_enc_type: {agent_cfg.get('graph_enc_type', 'gn')}")
    print(f"  - graph_gat_dropout: {agent_cfg.get('graph_gat_dropout', 0.0)}")
    print(f"  - use_diffusion: {agent_cfg.get('use_diffusion', False)}")
    if agent_cfg.get('use_diffusion', False):
        print(f"  - diffusion_steps: {agent_cfg.get('diffusion_steps', 20)}")
        print(f"  - diffusion_loss_weight: {agent_cfg.get('diffusion_loss_weight', 0.1)}")
        print(f"  - diffusion_q_loss_weight: {agent_cfg.get('diffusion_q_loss_weight', 0.1)}")
        print(f"  - diffusion_q_temperature: {agent_cfg.get('diffusion_q_temperature', 1.0)}")
        print(f"  - diffusion_guidance_weight: {agent_cfg.get('diffusion_guidance_weight', 0.5)}")
        print(f"  - diffusion_guidance_warmup_updates: {agent_cfg.get('diffusion_guidance_warmup_updates', 0)}")
    if task_dim:
        print(f"  - task_dim: {task_dim} (separated mode)")
    
    agent = GNNDDQNAgent(
        obs_shape=obs_shape,
        n_actions=agent_cfg.get('n_actions', 5),
        hidden_size=agent_cfg.get('hidden_size', 64),
        max_nbrs=agent_cfg.get('max_nbrs', 4),
        n_hops=agent_cfg.get('n_hops', 1),
        controller_type=controller_type,
        use_rnn=agent_cfg.get('use_rnn', False),
        dueling=agent_cfg.get('dueling', True),
        conv_type=agent_cfg.get('conv_type', 'gat'),
        n_heads=agent_cfg.get('n_heads', 4),
        graph_enc_type=agent_cfg.get('graph_enc_type', 'gn'),
        graph_gat_dropout=agent_cfg.get('graph_gat_dropout', 0.0),
        gamma=agent_cfg.get('gamma', 0.99),
        lr=agent_cfg.get('learning_rate', 1e-4),
        buffer_size=agent_cfg.get('buffer_size', 100000),
        batch_size=agent_cfg.get('batch_size', 64),
        target_update_freq=agent_cfg.get('target_update_freq', 100),
        device=str(device),
        task_dim=task_dim,  # 分离模式下的任务特征维度
        use_diffusion=agent_cfg.get('use_diffusion', False),
        diffusion_steps=agent_cfg.get('diffusion_steps', 20),
        diffusion_hidden_size=agent_cfg.get('diffusion_hidden_size', 128),
        diffusion_loss_weight=agent_cfg.get('diffusion_loss_weight', 0.1),
        diffusion_q_loss_weight=agent_cfg.get('diffusion_q_loss_weight', 0.1),
        diffusion_q_temperature=agent_cfg.get('diffusion_q_temperature', 1.0),
        diffusion_guidance_weight=agent_cfg.get('diffusion_guidance_weight', 0.5),
        diffusion_guidance_warmup_updates=agent_cfg.get('diffusion_guidance_warmup_updates', 0),
        diffusion_deterministic=agent_cfg.get('diffusion_deterministic', True),
        diffusion_normalize_prior=agent_cfg.get('diffusion_normalize_prior', True),
        diffusion_prior_clip=agent_cfg.get('diffusion_prior_clip', 2.0),
    )
    
    return agent


def main():
    args = parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 命令行参数覆盖配置
    if args.obs_type:
        config['agent']['obs_type'] = args.obs_type
        print(f"Overriding obs_type to: {args.obs_type}")
    
    # 设置随机种子
    seed = args.seed if args.seed is not None else config['general']['random_seed']
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed: {seed}")
    
    # 设置设备
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 导入环境
    from RL_environment_for_computing import SatelliteEnv
    
    phase = config['general']['phase']
    mode = config['agent']['mode']
    obs_type = config['agent'].get('obs_type', 'graph')
    
    # 创建智能体
    agent = create_gnn_agent(config, device)
    transformer_cfg = config.get('transformer', {})
    transformer_trainer = None
    transformer_model_path = transformer_cfg.get('model_path')
    transformer_loaded = False
    if (transformer_cfg.get('enabled', True)
            or transformer_cfg.get('meo_exit_enabled', False)
            or transformer_cfg.get('critic', {}).get('enabled', False)):
        transformer_trainer = GlobalTransformerTrainer(transformer_cfg, device)
        print("Creating Global Transformer planner:")
        print(f"  - history_len: {transformer_trainer.history_len}")
        print(f"  - forecast_horizon: {transformer_trainer.forecast_horizon}")
        print(f"  - meo_exit_enabled: {transformer_cfg.get('meo_exit_enabled', False)}")
        print(f"  - state source: global simulator snapshots")
    
    # 加载预训练模型（如果有且是测试模式）
    model_path = config['agent'].get('model_path')
    if phase != 'train' and model_path and os.path.exists(model_path):
        agent.load(model_path)
        print(f"Loaded model from {model_path}")
    
    # 创建环境
    print("\nCreating environment...")
    
    # epsilon 由环境管理（与原始代码保持一致）
    epsilon = config['general'].get('epsilon', 0.9)
    min_epsilon = config['general'].get('min_epsilon', 0.02)
    epsilon_decay = config['general'].get('epsilon_decay', 0.9995)
    
    if phase != 'train':
        epsilon = 0  # 测试模式不探索
    
    env = SatelliteEnv(
        mode=mode,
        select_mode=config['general']['select_mode'],
        q_net=agent.get_action_network(),
        epsilon=epsilon,  # 使用 general 中的 epsilon
        reward_factors=config['general']['reward_factors'],
        device=device,
        obs_type=obs_type,  # 传递观测类型
        leo_action_mask_enabled=config['agent'].get('leo_action_mask_enabled', False),
        mission_possibility=config['environment']['mission_possibility'],
        poisson_rate=config['environment']['poisson_rate'],
        packet_frequency=config['environment']['packet_frequency'],
        computing_demand_factor=config['environment']['computing_demand_factor'],
        computing_demand_factor_2=config['environment']['computing_demand_factor_2'],
        size_after_computing_factor=config['environment']['size_after_computing_factor'],
        size_after_computing_1=config['environment']['size_after_computing_1'],
        begin_time=config['general']['begin_time'],
        end_time=None,
        time_stride=config['general']['time_stride'],
        tle_filepath=config['environment']['tle_filepath'],
        SOD_file_path=config['environment']['SOD_file_path'],
        mean_interval_time=config['environment']['mean_interval_time'],
        memory=config['environment']['memory'],
        computing_ability=config['environment']['computing_ability'],
        transmission_rate=config['environment']['transmission_rate'],
        downlink_rate=config['environment']['downlink_rate'],
        downstream_delays=config['environment']['downstream_delays'],
        packet_size_range=config['environment']['packet_size_range'],
        state_update_period=config['environment']['state_update_period'],
        meo_state_update_period=config['environment'].get('meo_state_update_period'),
        print_cycle=config['general']['print_cycle'],
        del_cycle=config['environment']['del_cycle'],
        visualize=config['environment']['visualize'],
        print_info=config['environment']['print_info'],
        show_detail=config['environment']['show_detail'],
        save_log=config['environment']['save_log'],
        random_edges_del=config['environment']['random_edges_del'],
        random_nodes_del=config['environment']['random_nodes_del'],
        update_cycle=config['environment']['update_cycle'],
        save_training_data=config['environment']['save_training_data'],
        training_data_dir=config['environment'].get('training_data_dir', './training_process_data'),
        elevation_angle=config['environment']['elevation_angle'],
        pole=config['environment']['pole'],
        # n_hops: 优先使用 environment 配置，否则使用 agent 配置
        n_hops=config['environment'].get('n_hops', config['agent'].get('n_hops', 1)),
        transformer = transformer_trainer,
        controlDomainNumber = config['environment']['controlDomainNumber'],
        minimuElevationAngle = config['environment']['MinimuElevationAngle'],
        showLink = config['environment']['ShowLink'],
        domainPartitionMethod = config['environment'].get('domainPartitionMethod', 'Eunomia'),
        rectangular_m = config['environment'].get('rectangular_m', 1),
        rectangular_n = config['environment'].get('rectangular_n', 1),
        business_record_enabled=config['environment'].get('business_record_enabled', False),
        business_record_path=config['environment'].get('business_record_path'),
        business_record_reset=config['environment'].get('business_record_reset', True),
    )
    
    # 训练参数
    begin_time = config['general']['begin_time']
    time_stride = config['general']['time_stride']
    rounds = config['general']['rounds']
    skip_time = config['general']['skip_time']
    duration = config['general']['duration']
    
    print(f"\nStarting {phase} for {rounds} rounds...")
    print(f"  - Observation type: {obs_type}")
    print(f"  - Duration per round: {duration}s")
    print(f"  - Time stride: {time_stride}s")
    print(f"  - Initial epsilon: {epsilon:.4f}")
    
    # 统计
    total_losses = []
    total_transformer_losses = []
    total_transformer_metrics = []
    total_rewards_per_round = []
    
    # 训练循环
    for k in range(rounds):
        print(f"\n{'='*50}")
        print(f"Round {k+1}/{rounds} - {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*50}")
        
        if transformer_trainer is not None:
            transformer_trainer.reset_critic_round()
        env.reset(begin_time, config['environment']['controlDomainNumber'], config['environment']['MinimuElevationAngle'], config['environment']['ShowLink'],)
        agent.reset_hidden()
        if transformer_trainer is not None:
            transformer_trainer.add_env_snapshot(env)
            if not transformer_loaded and transformer_model_path:
                transformer_trainer.load_if_compatible(transformer_model_path)
                transformer_loaded = True
            if getattr(transformer_trainer, 'meo_router', None) is not None:
                transformer_trainer.meo_router.reset_training_log()
        
        # === 测试模式：每轮 reset 后启用分布式隐藏状态交换 ===
        # if phase == 'test':
        #     test_mode_enabled = env.simulator.propagator.init_test_mode(
        #         q_net=agent.q_network,
        #         hidden_size=config['agent'].get('hidden_size', 64),
        #         max_nbrs=config['agent'].get('max_nbrs', 4),
        #         device=str(device),
        #         state_update_period=config['environment'].get('state_update_period', 0.1),
        #         continuous_update=config['agent'].get('continuous_update', True),
        #     )
        #     if k == 0:  # 只在第一轮打印详细信息
        #         if test_mode_enabled:
        #             print("[Test Mode] 分布式隐藏状态交换已启用")
        #         else:
        #             print("[Test Mode] 隐藏状态交换启用失败")
        
        round_reward = 0
        round_losses = []
        round_transformer_losses = []
        round_transformer_metrics = []
        n_steps = int(duration / time_stride)
        
        # 检查是否为分离模式
        is_separated = obs_type in ('relational_separated', 'graph_separated')
        
        for t in range(n_steps):
            # 环境 step，获取经验（epsilon 由环境内部管理）
            experiences = env.step(epsilon)
            if transformer_trainer is not None:
                transformer_trainer.add_env_snapshot(env)
            
            # 存储经验并累计奖励
            for exp in experiences:
                if is_separated:
                    # 分离模式：8 字段 [obs, mark, action, reward, next_obs, done, task_context, next_task_context]
                    obs, mark, action, reward, next_obs, done, task_context, next_task_context = exp
                    agent.store_experience(obs, action, reward, next_obs, done, 
                                          task_context=task_context, next_task_context=next_task_context, mark=mark)
                else:
                    # 非分离模式：6 字段 [obs, mark, action, reward, next_obs, done]
                    obs, mark, action, reward, next_obs, done = exp
                    agent.store_experience(obs, action, reward, next_obs, done, mark=mark)
                round_reward += reward
            
            if phase == 'train':
                # epsilon 衰减（由环境管理）
                epsilon = max(min_epsilon, epsilon * epsilon_decay)
                
                # 更新智能体（repeat 次）
                repeat = config['agent'].get('repeat', 1)
                for _ in range(repeat):
                    loss = agent.update()
                    if loss is not None:
                        round_losses.append(loss)
                transformer_loss = None
                parts = None
                if transformer_trainer is not None:
                    for _ in range(repeat):
                        transformer_loss, parts = transformer_trainer.update_if_ready()
                        if transformer_loss is not None:
                            round_transformer_losses.append(transformer_loss)
                        transformer_trainer.update_meo_if_ready()

            if transformer_trainer is not None and transformer_trainer.should_eval():
                transformer_metrics = transformer_trainer.evaluate_latest()
                if transformer_metrics is not None:
                    round_transformer_metrics.append(transformer_metrics)

            # 不在主循环中调用 recommend_path；Transformer 的样本收集、
            # update_if_ready 参数更新、evaluate_latest 评估仍照常执行。
            plan_text = ""
                
            if phase != 'train' and (t + 1) % config['general']['print_cycle'] == 0:
                metrics_text = format_prediction_metrics(transformer_trainer.last_metrics) if transformer_trainer is not None else ""
                metrics_text = f", {metrics_text}" if metrics_text else ""
                print(f"  Step {t+1}/{n_steps}: buffer={len(agent.replay_buffer)}{metrics_text}{plan_text}")

            if phase == 'train':
                # 定期打印和保存
                if (t + 1) % config['general']['print_cycle'] == 0:
                    avg_loss = np.mean(round_losses[-100:]) if round_losses else 0
                    avg_transformer_loss = np.mean(round_transformer_losses[-100:]) if round_transformer_losses else 0
                    loss_parts = getattr(agent, 'last_losses', {})
                    diff_loss = loss_parts.get('diffusion_loss')
                    diff_text = f", diffusion_loss={diff_loss:.4f}" if diff_loss is not None else ""
                    transformer_text = ""
                    # if transformer_trainer is not None:
                    #     compute_loss = transformer_trainer.last_losses.get('compute_queue_loss')
                    #     compute_text = f", compute_queue_loss={compute_loss:.4f}" if compute_loss is not None else ""
                    #     metrics_text = format_prediction_metrics(transformer_trainer.last_metrics)
                    #     metrics_text = f", {metrics_text}" if metrics_text else ""
                    #     transformer_text = f", transformer_loss={avg_transformer_loss:.4f}{compute_text}{metrics_text}{plan_text}"
                    # print(f"  Step {t+1}/{n_steps}: eps={epsilon:.4f}, "
                    #       f"loss={avg_loss:.4f}{diff_text}{transformer_text}, buffer={len(agent.replay_buffer)}")
                    if transformer_loss is not None and parts is not None:
                        env.latest_transformer_losses = {
                            'transformer_loss': transformer_loss,
                            'queue_loss': parts.get('graph_queue_loss'),
                            'link_loss': parts.get('graph_link_loss'),
                            'compute_queue_loss': parts.get('graph_compute_queue_loss'),
                        }
                    meo_router = getattr(transformer_trainer, 'meo_router', None) if transformer_trainer is not None else None
                    if meo_router is not None and getattr(meo_router, 'enabled', False):
                        meo_log = meo_router.format_training_log(
                            step=t + 1,
                            total_steps=n_steps,
                            round_idx=k + 1,
                        )
                        if meo_log:
                            env.print_and_save(meo_log)
                            meo_router.reset_training_log()
                    critic = getattr(transformer_trainer, 'critic', None) if transformer_trainer is not None else None
                    if critic is not None and getattr(critic, 'enabled', False):
                        critic_log = critic.format_training_log(
                            step=t + 1,
                            total_steps=n_steps,
                            round_idx=k + 1,
                        )
                        if critic_log:
                            env.print_and_save(critic_log)
        
        # 保存模型
        if phase == 'train' and model_path:
            agent.save(model_path)
        if phase == 'train' and transformer_trainer is not None and transformer_model_path:
            transformer_trainer.save(transformer_model_path)
        
        # 统计
        avg_loss = np.mean(round_losses) if round_losses else 0
        avg_transformer_loss = np.mean(round_transformer_losses) if round_transformer_losses else 0
        avg_transformer_metrics = average_metric_dicts(round_transformer_metrics)
        total_losses.append(avg_loss)
        total_transformer_losses.append(avg_transformer_loss)
        if avg_transformer_metrics:
            total_transformer_metrics.append(avg_transformer_metrics)
        total_rewards_per_round.append(round_reward)
        
        print(f"Round {k+1} Summary:")
        print(f"  - Total reward: {round_reward:.2f}")
        print(f"  - Average loss: {avg_loss:.4f}")
        if transformer_trainer is not None:
            print(f"  - Average transformer loss: {avg_transformer_loss:.4f}")
            if avg_transformer_metrics:
                print(f"  - Transformer prediction diff: {format_prediction_metrics(avg_transformer_metrics)}")
            if transformer_trainer.last_plan is not None:
                print(f"  - Last recommended path: {' -> '.join(transformer_trainer.last_plan.path)}")
                print(f"  - Compute flags: {' -> '.join(str(flag) for flag in transformer_trainer.last_plan.compute_flags)}")
        print(f"  - Final epsilon: {epsilon:.4f}")
        
        if phase == 'test':
            env.show_satellite_computing_time()
        
        # 更新开始时间
        begin_time = env.add_time_to_str(begin_time, skip_time)
    
    # 训练完成
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    print(f"  - Total rounds: {rounds}")
    print(f"  - Average reward: {np.mean(total_rewards_per_round):.2f}")
    print(f"  - Average loss: {np.mean(total_losses):.4f}")
    if transformer_trainer is not None:
        print(f"  - Average transformer loss: {np.mean(total_transformer_losses):.4f}")
        avg_all_transformer_metrics = average_metric_dicts(total_transformer_metrics)
        if avg_all_transformer_metrics:
            print(f"  - Average transformer prediction diff: {format_prediction_metrics(avg_all_transformer_metrics)}")
    if model_path:
        print(f"  - Model saved to: {model_path}")
    if transformer_trainer is not None and transformer_model_path:
        print(f"  - Transformer model saved to: {transformer_model_path}")


if __name__ == '__main__':
    main()
