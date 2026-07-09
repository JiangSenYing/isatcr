"""
GNN Modules for Satellite Routing with Computing Tasks

参考 cross_layer_opt_with_grl-main 的模块设计，使用 DGL 处理异构图。

主要组件：
- activations: 激活函数注册表
- basics: 基础模块 (RnnLayer, MLP, DuelingLayer)
- gn_blocks: 图网络块 (NodeGNBlock, EdgeGNBlock, NeighborSelector)
- encoders: 编码器 (RelationalEncoder, FlatEncoder)
- controllers: 控制器 (RelationalController, GraphController)
- agents: 智能体 (GNNDDQNAgent)

使用示例:
    from gnn_modules import RelationalController, GNNDDQNAgent
    
    # 创建控制器
    controller = RelationalController(
        obs_shape={'agent': 9, 'nbr': 6},
        n_actions=5,
        hidden_size=64
    )
    
    # 或者使用智能体
    agent = GNNDDQNAgent(
        obs_shape={'agent': 9, 'nbr': 6},
        n_actions=5,
        controller_type='relational'
    )
"""

# ========== 新版模块（参考 cross_layer_opt_with_grl-main）==========

# 激活函数
from .activations import REGISTRY as ACT_REGISTRY, get_activation

# 基础模块
from .basics import RnnLayer, MLP, DuelingLayer

# 图网络块
from .gn_blocks import (
    NodeGNBlock,
    EdgeGNBlock,
    NeighborSelector,
    pad_edge_output,
)

# 编码器
from .encoders import (
    RelationalEncoder,
    FlatEncoder,
    ENCODER_REGISTRY,
)

# 控制器
from .controllers import (
    RelationalController,
    GraphController,
    SimpleQNetwork,
    TaskConditionedRelationalController,
    TaskConditionedGraphController,
    CONTROLLER_REGISTRY,
)

from .diffusion_policy import (
    DiffusionActionPrior,
    DiffusionGuidedQNetwork,
)

# 任务感知模块（V2 版本）
from .task_aware_blocks import (
    TaskAwareEdgeGNBlock,
    TaskConditionedGraphControllerV2,
    register_v2_controllers,
)

# 注册 V2 控制器
register_v2_controllers()

# 智能体
from .agents import (
    GNNDDQNAgent,
    GNN_DDQN_Agent,  # 兼容性别名
)

# 隐藏状态交换模块（测试模式用）
# from .hidden_state_exchange import (
#     HiddenStateAggregator,
#     LocalStateEncoder,
#     HiddenStateManager,
#     DistributedGNNInference,
#     create_hidden_exchange_modules,
# )

# 测试模式推理器
# from .test_mode_inference import (
#     TestModeInference,
#     create_test_mode_inference,
# )

# ========== 旧版模块（保持兼容性）==========

# try:
#     from .gnn_encoder import (
#         GNNStateEncoderFast,
#         GNNStateEncoderSimplified,
#         GNNStateEncoderHeterogeneous,
#         GNNStateEncoderEarlyFusion,
#         GNNStateEncoderFullHeterogeneous,
#         # 兼容性别名
#         GNNStateEncoder,
#         GNNStateEncoderSimple,
#     )
#     _OLD_ENCODERS_AVAILABLE = True
# except ImportError:
#     _OLD_ENCODERS_AVAILABLE = False

# try:
#     from .gnn_q_network import (
#         GNNQNetwork,
#         GNNActorNetwork,
#         GNNCriticNetwork,
#         GNNQNetworkEdgeLevel,
#     )
#     _OLD_NETWORKS_AVAILABLE = True
# except ImportError:
#     _OLD_NETWORKS_AVAILABLE = False

# try:
#     from .gnn_agent import GNN_PPO_Agent
#     _OLD_PPO_AVAILABLE = True
# except ImportError:
#     _OLD_PPO_AVAILABLE = False

__all__ = [
    # 新版模块 - 激活函数
    'ACT_REGISTRY',
    'get_activation',
    # 新版模块 - 基础模块
    'RnnLayer',
    'MLP',
    'DuelingLayer',
    # 新版模块 - 图网络块
    'NodeGNBlock',
    'EdgeGNBlock',
    'NeighborSelector',
    'pad_edge_output',
    # 新版模块 - 编码器
    'RelationalEncoder',
    'FlatEncoder',
    'ENCODER_REGISTRY',
    # 新版模块 - 控制器
    'RelationalController',
    'GraphController',
    'SimpleQNetwork',
    'DiffusionActionPrior',
    'DiffusionGuidedQNetwork',
    'CONTROLLER_REGISTRY',
    # 新版模块 - 智能体
    'GNNDDQNAgent',
    'GNN_DDQN_Agent',
    # 新版模块 - 隐藏状态交换
    'HiddenStateAggregator',
    'LocalStateEncoder',
    'HiddenStateManager',
    'DistributedGNNInference',
    'create_hidden_exchange_modules',
    # 新版模块 - 测试模式推理
    'TestModeInference',
    'create_test_mode_inference',
]

# # 旧版模块（如果可用）
# if _OLD_ENCODERS_AVAILABLE:
#     __all__.extend([
#         'GNNStateEncoderFast',
#         'GNNStateEncoderSimplified',
#         'GNNStateEncoderHeterogeneous',
#         'GNNStateEncoderEarlyFusion',
#         'GNNStateEncoderFullHeterogeneous',
#         'GNNStateEncoder',
#         'GNNStateEncoderSimple',
#     ])

# if _OLD_NETWORKS_AVAILABLE:
#     __all__.extend([
#         'GNNQNetwork',
#         'GNNQNetworkEdgeLevel',
#         'GNNActorNetwork',
#         'GNNCriticNetwork',
#     ])

# if _OLD_PPO_AVAILABLE:
#     __all__.append('GNN_PPO_Agent')
