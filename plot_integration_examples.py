"""
在训练脚本中集成画图功能的示例

在训练代码末尾添加以下代码，训练结束后自动生成可视化图表
"""

# ============ 示例 1: 基本使用 ============
# 在训练脚本末尾添加

from plot_training_results import parse_training_log, plot_training_metrics, print_summary_statistics
from pathlib import Path

# 训练结束后
print("\n" + "="*60)
print("Training completed! Generating visualization plots...")
print("="*60)

# 指定日志文件路径
log_file = "training_process_data/train_NewPPO_shuffle.txt"

# 解析训练日志
data = parse_training_log(log_file)

# 打印统计摘要
print_summary_statistics(data)

# 生成图表
log_name = Path(log_file).stem
save_dir = f"training_plots/{log_name}"
plot_training_metrics(data, save_dir)

print(f"\n✓ All training plots have been saved to: {save_dir}/")


# ============ 示例 2: 在 PRC.py 中集成 ============
# 在 PRC.py 的训练循环结束后添加

"""
# 在训练循环结束后，添加以下代码
if phase == 'train':
    print("\n" + "="*60)
    print("Training completed! Generating visualization plots...")
    print("="*60)
    
    # 导入画图模块
    from plot_training_results import parse_training_log, plot_training_metrics, print_summary_statistics
    from pathlib import Path
    
    # 构建日志文件路径
    log_file = f"training_process_data/{config['general']['log_file']}"
    
    # 检查日志文件是否存在
    if Path(log_file).exists():
        # 解析日志
        data = parse_training_log(log_file)
        
        # 打印统计摘要
        print_summary_statistics(data)
        
        # 生成图表
        log_name = Path(log_file).stem
        save_dir = f"training_plots/{log_name}"
        plot_training_metrics(data, save_dir)
        
        print(f"\n✓ Training visualization completed!")
        print(f"  - Statistics: Printed above")
        print(f"  - Plots saved to: {save_dir}/")
    else:
        print(f"Warning: Log file not found at {log_file}")
"""


# ============ 示例 3: 比较多个训练结果 ============
# 创建独立脚本比较不同模型的训练结果

"""
from plot_training_results import plot_training_comparison

# 指定要比较的日志文件
log_files = [
    "training_process_data/train_NewPPO_shuffle.txt",
    "training_process_data/train_PurePPO_shuffle.txt",
    "training_process_data/train_NewDDQN_dueling.txt",
]

# 指定对应的标签
labels = [
    "NewPPO (with shuffle)",
    "PurePPO (with shuffle)",
    "NewDDQN (dueling)",
]

# 生成比较图表
print("Generating comparison plots...")
plot_training_comparison(log_files, labels, "training_plots/model_comparison")
print("✓ Comparison plots saved to: training_plots/model_comparison/")
"""


# ============ 示例 4: 自定义保存路径 ============
# 根据时间戳或其他信息自定义保存路径

"""
from plot_training_results import parse_training_log, plot_training_metrics
from datetime import datetime

log_file = "training_process_data/train_NewPPO_shuffle.txt"
data = parse_training_log(log_file)

# 使用时间戳作为目录名
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = f"training_plots/NewPPO_shuffle_{timestamp}"

plot_training_metrics(data, save_dir)
print(f"Plots saved to: {save_dir}/")
"""


# ============ 示例 5: 只生成特定指标的图表 ============
# 如果只需要某些指标，可以修改代码

"""
import matplotlib.pyplot as plt
from plot_training_results import parse_training_log
from pathlib import Path

log_file = "training_process_data/train_NewPPO_shuffle.txt"
data = parse_training_log(log_file)

# 只画平均奖励的图表
save_dir = "training_plots/reward_only"
Path(save_dir).mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(10, 6))
steps = data['steps']
rewards = data['avg_reward']

valid_indices = [i for i, v in enumerate(rewards) if v is not None]
ax.plot([steps[i] for i in valid_indices], 
        [rewards[i] for i in valid_indices], 
        color='orange', linewidth=2, marker='o', markersize=6)

ax.set_xlabel('Training Steps', fontsize=14, fontweight='bold')
ax.set_ylabel('Average Reward', fontsize=14, fontweight='bold')
ax.set_title('Training Reward Curve', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(f'{save_dir}/reward_curve.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"Reward plot saved to: {save_dir}/reward_curve.png")
"""
