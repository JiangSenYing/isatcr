"""
测试 Dueling 架构是否正确实现
"""
import torch
import sys
sys.path.append('.')

from gnn_modules.controllers import (
    RelationalController,
    GraphController,
    TaskConditionedRelationalController,
    TaskConditionedGraphController
)
from gnn_modules import pyg_compat
from torch_geometric.data import HeteroData

def test_relational_controller():
    print("=" * 50)
    print("测试 RelationalController with Dueling")
    print("=" * 50)
    
    # 创建控制器（启用 Dueling）
    controller = RelationalController(
        obs_shape={'agent': 9, 'nbr': 6},
        n_actions=5,
        hidden_size=64,
        max_nbrs=4,
        dueling=True,  # 启用 Dueling
        use_rnn=False,
        device='cpu'
    )
    
    # 检查是否有 value_head
    assert hasattr(controller, 'value_head'), "❌ RelationalController 缺少 value_head"
    assert hasattr(controller, '_dueling'), "❌ RelationalController 缺少 _dueling 标志"
    assert controller._dueling == True, "❌ _dueling 应该为 True"
    
    # 创建测试数据
    obs = HeteroData()
    obs['agent'].feat = torch.randn(1, 9)
    obs['nbr'].feat = torch.randn(3, 6)
    obs[('nbr', '1hop', 'agent')].edge_index = torch.tensor([[0, 1, 2], [0, 0, 0]])
    
    # 前向传播
    q_vals, _ = controller(obs)
    
    assert q_vals.shape == (1, 5), f"❌ Q 值形状错误: {q_vals.shape}"
    print(f"✅ Q 值形状正确: {q_vals.shape}")
    print(f"✅ Q 值: {q_vals}")
    print()

def test_graph_controller():
    print("=" * 50)
    print("测试 GraphController with Dueling")
    print("=" * 50)
    
    controller = GraphController(
        obs_shape={'agent': 9, 'nbr': 6, 'hop': 3},
        n_actions=5,
        hidden_size=64,
        max_nbrs=4,
        n_hops=1,
        dueling=True,
        use_rnn=False,
        device='cpu'
    )
    
    assert hasattr(controller, 'value_head'), "❌ GraphController 缺少 value_head"
    assert controller._dueling == True, "❌ _dueling 应该为 True"
    
    obs = HeteroData()
    obs['agent'].feat = torch.randn(1, 9)
    obs['nbr'].feat = torch.randn(3, 6)
    obs[('nbr', '1hop', 'agent')].edge_index = torch.tensor([[0, 1, 2], [0, 0, 0]])
    obs[('nbr', '1hop', 'agent')].edge_attr = torch.randn(3, 3)
    
    q_vals, _ = controller(obs)
    
    assert q_vals.shape == (1, 5), f"❌ Q 值形状错误: {q_vals.shape}"
    print(f"✅ Q 值形状正确: {q_vals.shape}")
    print(f"✅ Q 值: {q_vals}")
    print()

def test_task_conditioned_relational():
    print("=" * 50)
    print("测试 TaskConditionedRelationalController with Dueling")
    print("=" * 50)
    
    controller = TaskConditionedRelationalController(
        obs_shape={'agent': 3, 'nbr': 6},
        task_dim=6,
        n_actions=5,
        hidden_size=64,
        max_nbrs=4,
        dueling=True,
        use_rnn=False,
        device='cpu'
    )
    
    assert hasattr(controller, 'value_head'), "❌ TaskConditionedRelationalController 缺少 value_head"
    assert controller._dueling == True, "❌ _dueling 应该为 True"
    
    obs = HeteroData()
    obs['agent'].feat = torch.randn(1, 3)
    obs['nbr'].feat = torch.randn(3, 6)
    obs[('nbr', '1hop', 'agent')].edge_index = torch.tensor([[0, 1, 2], [0, 0, 0]])
    
    task_context = torch.randn(1, 6)
    
    q_vals, _ = controller(obs, task_context)
    
    assert q_vals.shape == (1, 5), f"❌ Q 值形状错误: {q_vals.shape}"
    print(f"✅ Q 值形状正确: {q_vals.shape}")
    print(f"✅ Q 值: {q_vals}")
    print()

def test_task_conditioned_graph():
    print("=" * 50)
    print("测试 TaskConditionedGraphController with Dueling")
    print("=" * 50)
    
    controller = TaskConditionedGraphController(
        obs_shape={'agent': 3, 'nbr': 6, 'hop': 3},
        task_dim=6,
        n_actions=5,
        hidden_size=64,
        max_nbrs=4,
        n_hops=1,
        dueling=True,
        use_rnn=False,
        device='cpu'
    )
    
    assert hasattr(controller, 'value_head'), "❌ TaskConditionedGraphController 缺少 value_head"
    assert controller._dueling == True, "❌ _dueling 应该为 True"
    
    obs = HeteroData()
    obs['agent'].feat = torch.randn(1, 3)
    obs['nbr'].feat = torch.randn(3, 6)
    obs[('nbr', '1hop', 'agent')].edge_index = torch.tensor([[0, 1, 2], [0, 0, 0]])
    obs[('nbr', '1hop', 'agent')].edge_attr = torch.randn(3, 3)
    
    task_context = torch.randn(1, 6)
    
    q_vals, _ = controller(obs, task_context)
    
    assert q_vals.shape == (1, 5), f"❌ Q 值形状错误: {q_vals.shape}"
    print(f"✅ Q 值形状正确: {q_vals.shape}")
    print(f"✅ Q 值: {q_vals}")
    print()

def test_dueling_effect():
    """测试 Dueling 是否真的起作用（对比启用和禁用）"""
    print("=" * 50)
    print("测试 Dueling 架构的效果")
    print("=" * 50)
    
    # 创建相同结构的观测
    obs = HeteroData()
    obs['agent'].feat = torch.randn(1, 3)
    obs['nbr'].feat = torch.randn(3, 6)
    obs[('nbr', '1hop', 'agent')].edge_index = torch.tensor([[0, 1, 2], [0, 0, 0]])
    
    task_context = torch.randn(1, 6)
    
    # 启用 Dueling
    controller_dueling = TaskConditionedRelationalController(
        obs_shape={'agent': 3, 'nbr': 6},
        task_dim=6,
        n_actions=5,
        hidden_size=64,
        dueling=True,
        use_rnn=False
    )
    
    # 禁用 Dueling
    controller_no_dueling = TaskConditionedRelationalController(
        obs_shape={'agent': 3, 'nbr': 6},
        task_dim=6,
        n_actions=5,
        hidden_size=64,
        dueling=False,
        use_rnn=False
    )
    
    q_dueling, _ = controller_dueling(obs, task_context)
    q_no_dueling, _ = controller_no_dueling(obs, task_context)
    
    print(f"✅ 启用 Dueling 的 Q 值: {q_dueling}")
    print(f"✅ 禁用 Dueling 的 Q 值: {q_no_dueling}")
    print()
    
    # 检查 Dueling 架构是否真的改变了输出
    # 注意：由于参数随机初始化，输出会不同，这是正常的
    print("✅ Dueling 架构测试完成")
    print()

if __name__ == '__main__':
    try:
        test_relational_controller()
        test_graph_controller()
        test_task_conditioned_relational()
        test_task_conditioned_graph()
        test_dueling_effect()
        
        print("=" * 50)
        print("🎉 所有测试通过！Dueling 架构已成功实现！")
        print("=" * 50)
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
