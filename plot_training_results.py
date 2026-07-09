import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def parse_training_log(log_file_path):
    """
    解析训练日志文件，提取各项指标
    
    Args:
        log_file_path: 日志文件路径
        
    Returns:
        dict: 包含各项指标数据的字典
    """
    data = {
        'steps': [],
        'packet_loss_rate': [],
        'avg_delay': [],
        'avg_hop_count': [],
        'computing_proportion': [],
        'avg_waiting_time': [],
        'avg_reward': []
    }
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 按照 step 分割
    step_blocks = re.split(r'====== step \d+ ======', content)
    
    # 提取所有 step 编号
    steps = re.findall(r'====== step (\d+) ======', content)
    
    for i, (step, block) in enumerate(zip(steps, step_blocks[1:])):
        data['steps'].append(int(step))
        
        # 提取丢包率
        packet_loss_match = re.search(r'Packet loss rate: ([\d.]+)%', block)
        if packet_loss_match:
            data['packet_loss_rate'].append(float(packet_loss_match.group(1)))
        else:
            data['packet_loss_rate'].append(None)
        
        # 提取平均延迟
        delay_match = re.search(r'Average delay for successful transmissions: ([\d.]+) seconds', block)
        if delay_match:
            data['avg_delay'].append(float(delay_match.group(1)))
        else:
            data['avg_delay'].append(None)
        
        # 提取平均跳数
        hop_match = re.search(r'Average hop count for successful transmissions: ([\d.]+) hops', block)
        if hop_match:
            data['avg_hop_count'].append(float(hop_match.group(1)))
        else:
            data['avg_hop_count'].append(None)
        
        # 提取计算占比
        computing_match = re.search(r'Proportion of satellites in computation: ([\d.]+)%', block)
        if computing_match:
            data['computing_proportion'].append(float(computing_match.group(1)))
        else:
            data['computing_proportion'].append(None)
        
        # 提取平均等待时间
        waiting_match = re.search(r'Average waiting time for computing: ([\d.]+) seconds', block)
        if waiting_match:
            data['avg_waiting_time'].append(float(waiting_match.group(1)))
        else:
            data['avg_waiting_time'].append(None)
        
        # 提取平均奖励
        reward_match = re.search(r'Average ending reward: ([\d.]+)', block)
        if reward_match:
            data['avg_reward'].append(float(reward_match.group(1)))
        else:
            data['avg_reward'].append(None)
    
    return data

def plot_training_metrics(data, save_dir='training_plots'):
    """
    绘制训练指标图表
    
    Args:
        data: 包含训练指标的字典
        save_dir: 保存图表的目录
    """
    # 创建保存目录
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    steps = data['steps']
    
    # 创建一个大图，包含所有子图
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Training Metrics Overview', fontsize=16, fontweight='bold')
    
    # 1. 丢包率
    ax = axes[0, 0]
    valid_indices = [i for i, v in enumerate(data['packet_loss_rate']) if v is not None]
    if valid_indices:
        ax.plot([steps[i] for i in valid_indices], 
                [data['packet_loss_rate'][i] for i in valid_indices], 
                'b-o', linewidth=2, markersize=4)
        ax.set_xlabel('Training Steps', fontsize=12)
        ax.set_ylabel('Packet Loss Rate (%)', fontsize=12)
        ax.set_title('Packet Loss Rate', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # 2. 平均延迟
    ax = axes[0, 1]
    valid_indices = [i for i, v in enumerate(data['avg_delay']) if v is not None]
    if valid_indices:
        ax.plot([steps[i] for i in valid_indices], 
                [data['avg_delay'][i] for i in valid_indices], 
                'r-o', linewidth=2, markersize=4)
        ax.set_xlabel('Training Steps', fontsize=12)
        ax.set_ylabel('Average Delay (seconds)', fontsize=12)
        ax.set_title('Average Transmission Delay', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # 3. 平均跳数
    ax = axes[1, 0]
    valid_indices = [i for i, v in enumerate(data['avg_hop_count']) if v is not None]
    if valid_indices:
        ax.plot([steps[i] for i in valid_indices], 
                [data['avg_hop_count'][i] for i in valid_indices], 
                'g-o', linewidth=2, markersize=4)
        ax.set_xlabel('Training Steps', fontsize=12)
        ax.set_ylabel('Average Hop Count', fontsize=12)
        ax.set_title('Average Hop Count', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # 4. 计算占比
    ax = axes[1, 1]
    valid_indices = [i for i, v in enumerate(data['computing_proportion']) if v is not None]
    if valid_indices:
        ax.plot([steps[i] for i in valid_indices], 
                [data['computing_proportion'][i] for i in valid_indices], 
                'm-o', linewidth=2, markersize=4)
        ax.set_xlabel('Training Steps', fontsize=12)
        ax.set_ylabel('Computing Proportion (%)', fontsize=12)
        ax.set_title('Proportion of Satellites in Computation', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # 5. 平均等待时间
    ax = axes[2, 0]
    valid_indices = [i for i, v in enumerate(data['avg_waiting_time']) if v is not None]
    if valid_indices:
        ax.plot([steps[i] for i in valid_indices], 
                [data['avg_waiting_time'][i] for i in valid_indices], 
                'c-o', linewidth=2, markersize=4)
        ax.set_xlabel('Training Steps', fontsize=12)
        ax.set_ylabel('Average Waiting Time (seconds)', fontsize=12)
        ax.set_title('Average Computing Waiting Time', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # 6. 平均奖励
    ax = axes[2, 1]
    valid_indices = [i for i, v in enumerate(data['avg_reward']) if v is not None]
    if valid_indices:
        ax.plot([steps[i] for i in valid_indices], 
                [data['avg_reward'][i] for i in valid_indices], 
                'orange', linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Training Steps', fontsize=12)
        ax.set_ylabel('Average Reward', fontsize=12)
        ax.set_title('Average Ending Reward', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_metrics_all.png', dpi=300, bbox_inches='tight')
    print(f"Saved combined plot: {save_dir}/training_metrics_all.png")
    
    # 单独保存每个指标的大图
    metrics_config = [
        ('packet_loss_rate', 'Packet Loss Rate (%)', 'Packet Loss Rate', 'blue'),
        ('avg_delay', 'Average Delay (seconds)', 'Average Transmission Delay', 'red'),
        ('avg_hop_count', 'Average Hop Count', 'Average Hop Count', 'green'),
        ('computing_proportion', 'Computing Proportion (%)', 'Proportion of Satellites in Computation', 'magenta'),
        ('avg_waiting_time', 'Average Waiting Time (seconds)', 'Average Computing Waiting Time', 'cyan'),
        ('avg_reward', 'Average Reward', 'Average Ending Reward', 'orange')
    ]
    
    for metric_key, ylabel, title, color in metrics_config:
        fig, ax = plt.subplots(figsize=(10, 6))
        valid_indices = [i for i, v in enumerate(data[metric_key]) if v is not None]
        if valid_indices:
            ax.plot([steps[i] for i in valid_indices], 
                    [data[metric_key][i] for i in valid_indices], 
                    color=color, linewidth=2, marker='o', markersize=6, 
                    markerfacecolor='white', markeredgewidth=2)
            ax.set_xlabel('Training Steps', fontsize=14, fontweight='bold')
            ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
            ax.set_title(title, fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.tick_params(labelsize=12)
            
            plt.tight_layout()
            filename = f'{save_dir}/{metric_key}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved individual plot: {filename}")
            plt.close()

def plot_training_comparison(log_files, labels, save_dir='training_plots'):
    """
    比较多个训练日志的结果
    
    Args:
        log_files: 日志文件路径列表
        labels: 对应的标签列表
        save_dir: 保存图表的目录
    """
    # 创建保存目录
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 解析所有日志文件
    all_data = []
    for log_file in log_files:
        data = parse_training_log(log_file)
        all_data.append(data)
    
    # 创建比较图
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Training Metrics Comparison', fontsize=16, fontweight='bold')
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'cyan']
    
    metrics = [
        ('packet_loss_rate', 'Packet Loss Rate (%)', 'Packet Loss Rate Comparison', axes[0, 0]),
        ('avg_delay', 'Average Delay (seconds)', 'Average Delay Comparison', axes[0, 1]),
        ('avg_hop_count', 'Average Hop Count', 'Average Hop Count Comparison', axes[1, 0]),
        ('computing_proportion', 'Computing Proportion (%)', 'Computing Proportion Comparison', axes[1, 1]),
        ('avg_waiting_time', 'Average Waiting Time (seconds)', 'Waiting Time Comparison', axes[2, 0]),
        ('avg_reward', 'Average Reward', 'Average Reward Comparison', axes[2, 1])
    ]
    
    for metric_key, ylabel, title, ax in metrics:
        for i, (data, label) in enumerate(zip(all_data, labels)):
            valid_indices = [j for j, v in enumerate(data[metric_key]) if v is not None]
            if valid_indices:
                ax.plot([data['steps'][j] for j in valid_indices],
                       [data[metric_key][j] for j in valid_indices],
                       color=colors[i % len(colors)], linewidth=2, 
                       marker='o', markersize=4, label=label)
        
        ax.set_xlabel('Training Steps', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot: {save_dir}/training_comparison.png")
    plt.close()

def print_summary_statistics(data):
    """
    打印训练指标的统计摘要
    
    Args:
        data: 包含训练指标的字典
    """
    print("\n" + "="*60)
    print("Training Summary Statistics")
    print("="*60)
    
    metrics = [
        ('packet_loss_rate', 'Packet Loss Rate (%)', '%'),
        ('avg_delay', 'Average Delay', 'seconds'),
        ('avg_hop_count', 'Average Hop Count', 'hops'),
        ('computing_proportion', 'Computing Proportion', '%'),
        ('avg_waiting_time', 'Average Waiting Time', 'seconds'),
        ('avg_reward', 'Average Reward', '')
    ]
    
    for metric_key, name, unit in metrics:
        values = [v for v in data[metric_key] if v is not None]
        if values:
            print(f"\n{name}:")
            print(f"  Initial: {values[0]:.4f} {unit}")
            print(f"  Final: {values[-1]:.4f} {unit}")
            print(f"  Best: {min(values):.4f} {unit}" if 'loss' in metric_key or 'delay' in metric_key or 'waiting' in metric_key else f"  Best: {max(values):.4f} {unit}")
            print(f"  Mean: {np.mean(values):.4f} {unit}")
            print(f"  Std: {np.std(values):.4f} {unit}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    # 使用示例
    import sys
    
    if len(sys.argv) > 1:
        # 从命令行参数读取日志文件路径
        log_file = sys.argv[1]
    else:
        # 默认日志文件
        log_file = "training_process_data/withoutTransformer_fre0.5_withmeoDestribute.txt"
    
    print(f"Parsing training log: {log_file}")
    
    # 解析训练日志
    data = parse_training_log(log_file)
    
    # 打印统计摘要
    print_summary_statistics(data)
    
    # 获取日志文件名（不含扩展名）作为保存目录
    log_name = Path(log_file).stem
    save_dir = f"training_plots/{log_name}"
    
    # 绘制训练指标图表
    print(f"\nGenerating plots...")
    plot_training_metrics(data, save_dir)
    
    print(f"\nAll plots saved to: {save_dir}/")
    print("Done!")
    
    # 如果需要比较多个训练结果，取消下面的注释并修改
    # log_files = [
    #     "training_process_data/train_NewPPO_shuffle.txt",
    #     "training_process_data/train_PurePPO.txt",
    # ]
    # labels = ["NewPPO_shuffle", "PurePPO"]
    # plot_training_comparison(log_files, labels, "training_plots/comparison")
