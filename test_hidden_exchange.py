"""
测试模式隐藏状态交换使用示例

本示例展示如何在测试阶段使用隐藏状态交换机制，
使得用2hop训练的模型可以在只有1hop通信的情况下进行推理。

核心设计原则：
1. **聚合方式一致性**：测试时复用训练模型的编码器（RelationalEncoder/NodeGNBlock）
2. **交换次数一致性**：交换次数 = n_hops - 1（训练2hop → 测试交换1次）
3. **参数复用**：直接使用训练好的模型参数，不创建新的聚合器

隐藏状态交换流程：
    Round 0: 每个节点编码自己的本地观测 → 隐藏状态 h_0
    Round 1: 交换 h_0 → 每个节点获得1跳邻居的 h_0 → 用模型编码器聚合得到 h_1
    ...
    
经过 N 轮交换后（N = n_hops - 1），每个节点的隐藏状态就包含了 n_hops 跳邻居的信息。

使用方法：
    1. 训练阶段：正常使用 n_hops=2 训练模型
    2. 测试阶段：
       - 加载训练好的模型
       - 调用 propagator.init_test_mode(q_net, ...)
       - 仿真过程中 state_exchanger 会自动进行隐藏状态交换
       - 决策时自动使用聚合后的隐藏状态
"""

import torch as th
import numpy as np
from typing import Dict, List, Optional

# 导入隐藏状态交换模块
from gnn_modules import (
    TestModeInference,
    create_test_mode_inference,
)


def demo_test_mode_usage():
    """
    演示如何在实际测试中使用隐藏状态交换
    """
    print("=" * 70)
    print("测试模式隐藏状态交换使用说明")
    print("=" * 70)
    
    print("""
【训练阶段】（无需修改）
--------------------------
使用 n_hops=2 的配置正常训练：
    python PRC_GNN.py --config train/train_GNN_DQN2hop_relational.yaml
    
训练时模型直接获取2跳邻居信息（"上帝视角"）。


【测试阶段】（启用隐藏状态交换）
--------------------------
在测试脚本中，加载模型后调用 init_test_mode：

```python
# 1. 创建仿真环境
simulator = SatelliteNetworkSimulator_OnbardComputing(...)

# 2. 加载训练好的模型
q_net = load_model("model_weights/gnn_ddqn_2hop.pth")

# 3. 启用测试模式（关键步骤）
simulator.propagator.init_test_mode(
    q_net=q_net,
    hidden_size=64,    # 与训练配置一致
    max_nbrs=4,        # 与训练配置一致
    device='cuda'
)

# 4. 正常运行仿真
simulator.run()
```


【工作原理】
--------------------------
启用测试模式后，以下过程自动进行：

1. state_exchanger() 中：
   - 除了原有的状态交换，还会调用 _exchange_hidden_states()
   - 每个卫星编码本地观测 → 隐藏状态
   - 使用【训练模型的编码器】聚合邻居隐藏状态
   - 将自身隐藏状态发送给邻居

2. 决策时（get_next_hop / cal_score）：
   - 检测到 test_mode=True
   - 使用聚合后的隐藏状态（而不是构建2hop图）
   - 调用【训练模型的输出层】计算Q值


【关键一致性保证】
--------------------------
1. 聚合方式：测试时复用训练模型的 f_enc（RelationalEncoder）
             → 与训练时的消息传递方式完全一致

2. 交换次数：n_exchange_rounds = n_hops - 1
             训练 n_hops=2 → 测试交换 1 次
             训练 n_hops=3 → 测试交换 2 次

3. 参数复用：直接引用训练模型的权重，不创建新参数
""")
    
    print("\n" + "=" * 70)
    print("配置示例")
    print("=" * 70)
    
    print("""
【训练配置】train/train_GNN_DQN2hop_relational.yaml
```yaml
n_hops: 2
obs_type: relational_separated
controller:
  type: relational
  hidden_size: 64
  max_nbrs: 4
  conv_type: gat
  n_heads: 4
```

【测试配置】test/test_GNN_DQN2hop.yaml
```yaml
# 与训练配置保持一致
n_hops: 2
obs_type: relational_separated

# 启用测试模式
test_mode: true
hidden_size: 64
max_nbrs: 4

# 模型路径
model_path: model_weights/gnn_ddqn_2hop.pth
```
""")
    
    print("=" * 70)
    print("演示完成！")
    print("=" * 70)


def demo_hidden_exchange_mechanism():
    """
    演示隐藏状态交换的具体机制
    """
    print("\n" + "=" * 70)
    print("隐藏状态交换机制演示")
    print("=" * 70)
    
    # 创建一个简单的模拟Q网络
    class MockRelationalController(th.nn.Module):
        def __init__(self, hidden_size=64, n_actions=5):
            super().__init__()
            self._use_rnn = False
            
            # 模拟 agent_encoder
            self.agent_encoder = th.nn.Sequential(
                th.nn.Linear(3, hidden_size),
                th.nn.ReLU()
            )
            
            # 模拟 f_enc
            self.f_enc = type('FakeEncoder', (), {
                'agent_encoder': self.agent_encoder,
                'conv': None
            })()
            
            # 模拟 selector
            self.selector = th.nn.Linear(hidden_size * 2, n_actions)
        
        def forward(self, x):
            return self.selector(x)
    
    q_net = MockRelationalController(hidden_size=64, n_actions=5)
    
    # 模拟卫星网络
    satellite_names = ['sat_0', 'sat_1', 'sat_2', 'sat_3']
    adjacency = {
        'sat_0': ['sat_1', 'sat_3'],
        'sat_1': ['sat_0', 'sat_2'],
        'sat_2': ['sat_1', 'sat_3'],
        'sat_3': ['sat_0', 'sat_2'],
    }
    
    # 创建测试推理器
    inferencer = create_test_mode_inference(
        q_net=q_net,
        n_hops=2,       # 训练时用2hop
        hidden_size=64,
        max_nbrs=4,
        device='cpu'
    )
    
    print(f"\n配置:")
    print(f"  - n_hops (训练): 2")
    print(f"  - n_exchange_rounds (测试): {inferencer.n_exchange_rounds}")
    print(f"  - hidden_size: 64")
    
    # 初始化所有节点
    for name in satellite_names:
        inferencer.init_node_state(name)
    
    print("\n" + "-" * 50)
    print("Round 0: 编码本地观测")
    print("-" * 50)
    
    # 模拟本地观测
    for name in satellite_names:
        local_obs = th.randn(3)  # 3维本地观测
        hidden = inferencer.encode_local_observation(name, local_obs)
        inferencer.update_node_hidden(name, hidden, current_time=0.0)
        print(f"  {name}: 编码本地观测 → hidden (norm={hidden.norm():.3f})")
    
    print("\n" + "-" * 50)
    print("Round 1: 交换隐藏状态并聚合")
    print("-" * 50)
    
    # 交换隐藏状态
    print("  发送阶段:")
    for name in satellite_names:
        hidden, timestamp, version = inferencer.get_message_to_send(name)
        for neighbor in adjacency[name]:
            inferencer.receive_neighbor_hidden(neighbor, name, hidden, timestamp, version)
            print(f"    {name} → {neighbor}")
    
    # 聚合
    print("\n  聚合阶段:")
    for name in satellite_names:
        self_hidden = inferencer.get_node_embedding(name)
        aggregated = inferencer.aggregate_neighbor_hiddens(
            name, self_hidden, current_time=0.1, neighbor_names=adjacency[name]
        )
        inferencer.update_node_hidden(name, aggregated, current_time=0.1)
        print(f"    {name}: 聚合邻居隐藏状态 → updated (norm={aggregated.norm():.3f})")
    
    print("\n" + "-" * 50)
    print("结果: 每个节点的隐藏状态现在包含2跳信息")
    print("-" * 50)
    
    for name in satellite_names:
        embedding = inferencer.get_node_embedding(name)
        print(f"  {name}: final embedding (norm={embedding.norm():.3f})")
    
    print("\n完成！现在可以使用这些嵌入进行决策。")


if __name__ == '__main__':
    demo_test_mode_usage()
    demo_hidden_exchange_mechanism()
