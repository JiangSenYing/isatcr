# 训练结果可视化工具使用说明

## 功能说明

`plot_training_results.py` 是一个用于可视化训练过程指标的工具，可以解析训练日志文件并生成多种图表。

## 主要功能

1. **自动解析训练日志**：从文本文件中提取以下指标
   - 丢包率 (Packet Loss Rate)
   - 平均延迟 (Average Delay)
   - 平均跳数 (Average Hop Count)
   - 计算占比 (Computing Proportion)
   - 平均等待时间 (Average Waiting Time)
   - 平均奖励 (Average Reward)

2. **生成可视化图表**：
   - 综合图表：包含所有6个指标的3x2子图布局
   - 独立图表：每个指标单独的高质量大图

3. **统计摘要**：打印训练过程的统计信息（初始值、最终值、最佳值、平均值、标准差）

4. **多模型对比**：支持比较多个训练结果

## 使用方法

### 基本使用

```bash
# 使用默认日志文件（train_NewPPO_shuffle.txt）
python plot_training_results.py

# 指定日志文件
python plot_training_results.py training_process_data/train_NewPPO_shuffle.txt

# 处理其他训练日志
python plot_training_results.py training_process_data/train_PureDDQN_dueling.txt
```

### 在训练代码中调用

在你的训练脚本末尾添加：

```python
# 训练结束后，调用画图函数
from plot_training_results import parse_training_log, plot_training_metrics, print_summary_statistics

# 假设你的日志文件路径是 log_file
log_file = "training_process_data/train_NewPPO_shuffle.txt"

# 解析日志
data = parse_training_log(log_file)

# 打印统计摘要
print_summary_statistics(data)

# 生成图表
import os
log_name = os.path.splitext(os.path.basename(log_file))[0]
save_dir = f"training_plots/{log_name}"
plot_training_metrics(data, save_dir)

print(f"Training plots saved to: {save_dir}/")
```

### 比较多个训练结果

修改 `plot_training_results.py` 文件末尾的代码：

```python
# 取消注释并修改以下代码
log_files = [
    "training_process_data/train_NewPPO_shuffle.txt",
    "training_process_data/train_PurePPO_shuffle.txt",
    "training_process_data/train_NewDDQN_dueling.txt",
]
labels = ["NewPPO_shuffle", "PurePPO_shuffle", "NewDDQN_dueling"]
plot_training_comparison(log_files, labels, "training_plots/comparison")
```

然后运行：
```bash
python plot_training_results.py
```

## 输出说明

运行脚本后会生成以下文件：

```
training_plots/
└── train_NewPPO_shuffle/              # 以日志文件名命名的目录
    ├── training_metrics_all.png       # 综合图表（所有指标）
    ├── packet_loss_rate.png           # 丢包率单独图表
    ├── avg_delay.png                  # 平均延迟单独图表
    ├── avg_hop_count.png              # 平均跳数单独图表
    ├── computing_proportion.png       # 计算占比单独图表
    ├── avg_waiting_time.png           # 平均等待时间单独图表
    └── avg_reward.png                 # 平均奖励单独图表
```

## 图表说明

### 1. Packet Loss Rate（丢包率）
- 显示训练过程中数据包的丢失率
- 越低越好，表示网络传输更可靠

### 2. Average Transmission Delay（平均传输延迟）
- 显示成功传输的平均延迟时间（秒）
- 越低越好，表示数据传输更快

### 3. Average Hop Count（平均跳数）
- 显示数据包从源到目的地经过的平均跳数
- 反映路由效率

### 4. Proportion of Satellites in Computation（计算占比）
- 显示参与计算的卫星比例
- 反映计算资源利用率

### 5. Average Computing Waiting Time（平均计算等待时间）
- 显示计算任务的平均等待时间（秒）
- 越低越好，表示计算调度更高效

### 6. Average Ending Reward（平均奖励）
- 显示强化学习训练的平均奖励
- 越高越好，表示策略性能更优

## 统计信息说明

脚本会在控制台输出每个指标的统计信息：
- **Initial**: 训练开始时的指标值
- **Final**: 训练结束时的指标值
- **Best**: 训练过程中的最佳值
- **Mean**: 所有训练步骤的平均值
- **Std**: 标准差，反映指标的波动程度

## 自定义修改

### 修改图表样式

在 `plot_training_metrics()` 函数中可以修改：
- 颜色：修改 `color` 参数
- 线条样式：修改 `linewidth`, `marker`, `markersize` 等参数
- 图表大小：修改 `figsize` 参数

### 添加新的指标

1. 在 `parse_training_log()` 函数中添加新指标的解析代码
2. 在 `plot_training_metrics()` 函数中添加对应的绘图代码
3. 在 `print_summary_statistics()` 函数中添加统计信息输出

## 依赖项

```bash
pip install matplotlib numpy
```

## 常见问题

**Q: 中文字体显示不正确怎么办？**
A: 修改 `plot_training_metrics()` 函数中的字体设置：
```python
plt.rcParams['font.sans-serif'] = ['你的字体名称']
```

**Q: 图表分辨率不够怎么办？**
A: 修改 `savefig()` 函数的 `dpi` 参数，例如改为 `dpi=600`

**Q: 想要修改图表布局怎么办？**
A: 修改 `plt.subplots()` 的参数，例如从 `(3, 2)` 改为 `(2, 3)` 或其他布局

## 示例输出

成功运行后会显示：
```
Parsing training log: training_process_data/train_NewPPO_shuffle.txt

============================================================
Training Summary Statistics
============================================================

Packet Loss Rate (%):
  Initial: 4.9400 %
  Final: 0.1600 %
  Best: 0.0000 %
  Mean: 1.3375 %
  Std: 2.0818 %

...（其他指标统计）

Generating plots...
Saved combined plot: training_plots/train_NewPPO_shuffle/training_metrics_all.png
Saved individual plot: training_plots/train_NewPPO_shuffle/packet_loss_rate.png
...（其他图表）

All plots saved to: training_plots/train_NewPPO_shuffle/
Done!
```
