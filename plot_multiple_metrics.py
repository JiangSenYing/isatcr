import os
import re
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

DEFAULT_OUTPUT_DIR = 'training_plots/metrics_comparison/withoutTransformer_fre0.5_withmeoDestribute'

def parse_training_file(file_path):
    """
    解析训练数据文件，提取各项指标
    返回: 包含各项指标列表的字典
    """
    metrics = {
        'steps': [],
        'packet_loss_rate': [],
        'avg_delay': [],
        'avg_hop_count': [],
        'computation_proportion': [],
        'avg_waiting_time': [],
        'avg_ending_reward': [],
        'meo_agent_steps': [],
        'meo_agent_reward': []
    }

    number_pattern = r'([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)'
    step_pattern = re.compile(r'^====== step (\d+) ======')
    meo_agent_reward_pattern = re.compile(
        rf'^MEO\s+Agent\b.*?\bstep=(\d+)(?:/\d+)?:.*?\breward_avg={number_pattern}',
        re.IGNORECASE
    )
    patterns = {
        'packet_loss_rate': re.compile(rf'^Packet loss rate: {number_pattern}%'),
        'avg_delay': re.compile(rf'^Average delay for successful transmissions: {number_pattern} seconds'),
        'avg_hop_count': re.compile(rf'^Average hop count for successful transmissions: {number_pattern} hops'),
        'computation_proportion': re.compile(rf'^Proportion of satellites in computation: {number_pattern}%'),
        'avg_waiting_time': re.compile(rf'^Average waiting time for computing: {number_pattern} seconds'),
        'avg_ending_reward': re.compile(rf'^Average ending reward: {number_pattern}')
    }

    current_data = {}

    def append_current_data():
        required_keys = [
            'step',
            'packet_loss_rate',
            'avg_delay',
            'avg_hop_count',
            'computation_proportion',
            'avg_waiting_time',
            'avg_ending_reward'
        ]
        if all(key in current_data for key in required_keys):
            metrics['steps'].append(current_data['step'])
            metrics['packet_loss_rate'].append(current_data['packet_loss_rate'])
            metrics['avg_delay'].append(current_data['avg_delay'])
            metrics['avg_hop_count'].append(current_data['avg_hop_count'])
            metrics['computation_proportion'].append(current_data['computation_proportion'])
            metrics['avg_waiting_time'].append(current_data['avg_waiting_time'])
            metrics['avg_ending_reward'].append(current_data['avg_ending_reward'])

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            step_match = step_pattern.match(line)
            if step_match:
                append_current_data()
                current_data = {'step': int(step_match.group(1))}
                continue

            meo_agent_match = meo_agent_reward_pattern.match(line)
            if meo_agent_match:
                metrics['meo_agent_steps'].append(int(meo_agent_match.group(1)))
                metrics['meo_agent_reward'].append(float(meo_agent_match.group(2)))
                continue

            for metric_key, pattern in patterns.items():
                match = pattern.match(line)
                if match:
                    current_data[metric_key] = float(match.group(1))
                    break

    append_current_data()
    
    return metrics

def plot_metrics(all_data, output_dir=DEFAULT_OUTPUT_DIR):
    """
    绘制所有文件的指标对比图
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 定义6个指标及其标题
    metric_configs = [
        ('packet_loss_rate', 'Packet Loss Rate (%)', 'packet_loss_rate.png'),
        ('avg_delay', 'Average Delay (seconds)', 'avg_delay.png'),
        ('avg_hop_count', 'Average Hop Count', 'avg_hop_count.png'),
        ('computation_proportion', 'Proportion of Satellites in Computation (%)', 'computation_proportion.png'),
        ('avg_waiting_time', 'Average Waiting Time (seconds)', 'avg_waiting_time.png'),
        ('avg_ending_reward', 'Average Ending Reward', 'avg_ending_reward.png')
    ]
    
    # 为每个指标创建一个图
    for metric_key, title, filename in metric_configs:
        plt.figure(figsize=(12, 7))
        
        for file_name, metrics in all_data.items():
            if len(metrics['steps']) > 0:
                plt.plot(metrics['steps'], metrics[metric_key], 
                        marker='o', linewidth=2, markersize=4, label=file_name, alpha=0.7)
        
        plt.xlabel('Training Step', fontsize=12)
        plt.ylabel(title, fontsize=12)
        plt.title(f'{title} Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc='upper right', fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"已保存: {save_path}")
        plt.close()

    has_meo_agent_data = any(
        len(metrics.get('meo_agent_steps', [])) > 0
        for metrics in all_data.values()
    )
    if has_meo_agent_data:
        plt.figure(figsize=(12, 7))

        for file_name, metrics in all_data.items():
            if len(metrics.get('meo_agent_steps', [])) > 0:
                plt.plot(
                    metrics['meo_agent_steps'],
                    metrics['meo_agent_reward'],
                    marker='o',
                    linewidth=2,
                    markersize=4,
                    label=file_name,
                    alpha=0.7
                )

        plt.xlabel('Training Step', fontsize=12)
        plt.ylabel('MEO Agent Reward', fontsize=12)
        plt.title('MEO Agent Reward Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc='upper right', fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = os.path.join(output_dir, 'meo_agent_reward.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"已保存: {save_path}")
        plt.close()

def calculate_final_averages(all_data):
    """
    计算每个文件所有step的指标平均值
    """
    final_averages = {}
    
    for file_name, metrics in all_data.items():
        if len(metrics['steps']) == 0:
            continue
        
        # 对所有step的指标求平均值
        final_averages[file_name] = {
            'Packet Loss Rate (%)': np.mean(metrics['packet_loss_rate']),
            'Average Delay (seconds)': np.mean(metrics['avg_delay']),
            'Average Hop Count': np.mean(metrics['avg_hop_count']),
            'Computation Proportion (%)': np.mean(metrics['computation_proportion']),
            'Average Waiting Time (seconds)': np.mean(metrics['avg_waiting_time']),
            'Average Ending Reward': np.mean(metrics['avg_ending_reward'])
        }
        if len(metrics.get('meo_agent_reward', [])) > 0:
            final_averages[file_name]['MEO Agent Reward'] = np.mean(metrics['meo_agent_reward'])
    
    return final_averages

def print_summary_table(final_averages):
    """
    打印汇总表格
    """
    print("\n" + "="*120)
    print("各项指标的平均值汇总表（所有step的平均）")
    print("="*120)
    
    # 表头
    metric_names = ['Packet Loss Rate (%)', 'Average Delay (seconds)', 'Average Hop Count', 
                   'Computation Proportion (%)', 'Average Waiting Time (seconds)', 'Average Ending Reward']
    if any('MEO Agent Reward' in metrics for metrics in final_averages.values()):
        metric_names.append('MEO Agent Reward')
    
    print(f"{'File Name':<30}", end="")
    for metric in metric_names:
        print(f"{metric:>20}", end="")
    print()
    print("-"*120)
    
    # 数据行
    for file_name, metrics in sorted(final_averages.items()):
        print(f"{file_name:<30}", end="")
        for metric_name in metric_names:
            value = metrics.get(metric_name)
            print(f"{value:>20.4f}" if value is not None else f"{'N/A':>20}", end="")
        print()
    
    print("="*120)

def save_summary_to_file(final_averages, output_file='metrics_summary.txt'):
    """
    将汇总数据保存到文件
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*120 + "\n")
        f.write("各项指标的平均值汇总表（所有step的平均）\n")
        f.write("="*120 + "\n\n")
        
        metric_names = ['Packet Loss Rate (%)', 'Average Delay (seconds)', 'Average Hop Count', 
                       'Computation Proportion (%)', 'Average Waiting Time (seconds)', 'Average Ending Reward']
        if any('MEO Agent Reward' in metrics for metrics in final_averages.values()):
            metric_names.append('MEO Agent Reward')
        
        # 写入表头
        f.write(f"{'File Name':<30}")
        for metric in metric_names:
            f.write(f"{metric:>20}")
        f.write("\n")
        f.write("-"*120 + "\n")
        
        # 写入数据
        for file_name, metrics in sorted(final_averages.items()):
            f.write(f"{file_name:<30}")
            for metric_name in metric_names:
                value = metrics.get(metric_name)
                f.write(f"{value:>20.4f}" if value is not None else f"{'N/A':>20}")
            f.write("\n")
        
        f.write("="*120 + "\n")
    
    print(f"\n汇总数据已保存到: {output_file}")

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='绘制多个训练日志文件的指标对比图')
    parser.add_argument(
        '-o',
        '--output-dir',
        default=DEFAULT_OUTPUT_DIR,
        help=f'图片和汇总文件的保存目录，默认: {DEFAULT_OUTPUT_DIR}'
    )
    return parser.parse_args()

def select_files(txt_files):
    """
    让用户选择要绘图的文件
    """
    print("\n可用的训练数据文件:")
    for i, f in enumerate(txt_files, 1):
        print(f"  [{i}] {f}")
    
    print("\n请选择要绘图的文件（输入选项）:")
    print("  - 输入文件编号，用逗号分隔（例如: 1,3,5）")
    print("  - 输入 'all' 选择所有文件")
    print("  - 输入 'q' 退出")
    
    while True:
        choice = input("\n请输入: ").strip()
        
        if choice.lower() == 'q':
            return None
        
        if choice.lower() == 'all':
            return txt_files
        
        try:
            indices = [int(x.strip()) for x in choice.split(',')]
            selected_files = []
            for idx in indices:
                if 1 <= idx <= len(txt_files):
                    selected_files.append(txt_files[idx - 1])
                else:
                    print(f"  ✗ 编号 {idx} 无效，请重新输入")
                    break
            else:
                return selected_files
        except ValueError:
            print("  ✗ 输入格式错误，请重新输入")

def main():
    args = parse_args()

    # 数据文件目录
    data_dir = 'training_process_data'
    output_dir = args.output_dir
    
    # 获取目录及其所有子目录下的txt文件
    data_path = Path(data_dir)
    txt_files = sorted(
        str(file_path.relative_to(data_path))
        for file_path in data_path.rglob('*.txt')
        if file_path.is_file()
    )
    
    if not txt_files:
        print("错误: 未找到任何txt文件!")
        return
    
    print(f"找到 {len(txt_files)} 个训练数据文件")
    
    # 让用户选择文件
    selected_files = select_files(txt_files)
    
    if selected_files is None:
        print("已退出")
        return
    
    if not selected_files:
        print("未选择任何文件")
        return
    
    print(f"\n已选择 {len(selected_files)} 个文件:")
    for f in selected_files:
        print(f"  - {f}")
    
    # 解析选中的文件
    all_data = {}
    print("\n开始解析文件...")
    
    for txt_file in selected_files:
        file_path = os.path.join(data_dir, txt_file)
        try:
            metrics = parse_training_file(file_path)
            if len(metrics['steps']) > 0:
                # 使用不带扩展名的文件名作为标签
                label = txt_file.replace('.txt', '').replace('train_', '')
                all_data[label] = metrics
                print(f"  ✓ {txt_file}: 解析成功，共 {len(metrics['steps'])} 个数据点")
            else:
                print(f"  ✗ {txt_file}: 未找到有效数据")
        except Exception as e:
            print(f"  ✗ {txt_file}: 解析失败 - {str(e)}")
    
    if not all_data:
        print("\n错误: 没有成功解析任何文件!")
        return
    
    # 绘制图表
    print("\n开始绘制图表...")
    plot_metrics(all_data, output_dir)
    
    # 计算最终平均值
    print("\n计算所有step的指标平均值...")
    final_averages = calculate_final_averages(all_data)
    
    # 打印汇总表格
    print_summary_table(final_averages)
    
    # 保存汇总数据到文件
    summary_file = os.path.join(output_dir, 'metrics_summary.txt')
    save_summary_to_file(final_averages, summary_file)
    
    print("\n✓ 所有任务完成!")

if __name__ == "__main__":
    main()
