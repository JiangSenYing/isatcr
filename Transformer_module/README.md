# Transformer_module

该模块现在面向“全局预测 + 路径规划”：

- `SatelliteLoadTransformer`: 预测未来全局节点队列负载、链路业务负载、卫星计算任务排队量、每颗卫星距离下一次业务产生的时间。
- `GlobalStateExtractor`: 从当前仿真器读取全局状态快照，并固定节点/链路顺序。
- `TransformerPathPlanner`: 基于 Transformer 预测结果给候选路径评分，输出阻塞/丢包风险最低的路径。
- `GlobalTransformerTrainer`: 面向训练脚本的封装器，负责采样全局快照、训练 Transformer、统计预测误差和调用路径规划器。

卫星位置、星间距离、传播时延仍然不由 Transformer 预测，而是从仿真器/星历生成的图中读取。

## 预测模型

```python
from Transformer_module import SatelliteLoadTransformer

model = SatelliteLoadTransformer(
    queue_input_dim=7,
    link_input_dim=3,
    compute_input_dim=3,
    forecast_horizon=5,
)
```

输入形状：

```text
queue_history    [batch, history_len, num_nodes, queue_input_dim]
link_history     [batch, history_len, num_links, link_input_dim]
compute_history  [batch, history_len, num_nodes, compute_input_dim]
```

从仿真器采集 snapshot 时，输入会自动包含负载和包流向特征。`queue_history`
的前 4 维节点输入为：

```text
[queue_load, is_producing, session_remaining_ratio, session_duration_norm]
```

其中后 3 维来自样本收集阶段正在产生业务的卫星会话信息；训练目标仍只预测 `queue_load`。
后续维度是当前节点转发队列、下行队列以及已知剩余路径经过该节点的数据压力。
业务时间目标 `business_time` 是节点级归一化时间：正在产生业务时为 `0`，否则为距离下一次业务会话开始的时间除以时间尺度并裁剪到 `[0, 1]`。

`link_history` 的第 1 维是当前链路传输负载，后续维度表示已经排在该链路上的数据包压力、
以及转发队列中下一跳计划使用该链路的数据压力。`compute_history` 的第 1 维是当前计算队列负载，
后续维度表示已在计算队列中的计算需求、以及计划在该节点计算的数据包需求。
loss 仍然只监督 queue/link/compute 的第 1 个真实负载维度，包流向维度作为预测未来负载的辅助输入。

每个时间片还会在三类历史输入最后追加一个变化率特征：

```text
queue_history   最后一维：当前卫星队列负载相对上一时间片的变化率
link_history    最后一维：当前有向链路传输负载相对上一时间片的变化率
compute_history 最后一维：当前卫星计算队列负载相对上一时间片的变化率
```

变化率由当前归一化负载第 1 维减去上一时间片第 1 维，再除以仿真时间间隔得到，并裁剪到
`[-1, 1]`。第一个快照没有上一时间片，变化率填 `0`。这些变化率用于让模型感知最近一个时间片内
数据包经过节点、占用链路或选择计算节点造成的动态冲击。

输出：

```text
queue_forecast          [batch, forecast_horizon, num_nodes, 1]
link_forecast           [batch, forecast_horizon, num_links, 1]
compute_queue_forecast  [batch, forecast_horizon, num_nodes, 1]
business_time_forecast  [batch, forecast_horizon, num_nodes, 1]
```

`compute_queue_forecast` 由独立的 `compute_head` 预测，只读取 `compute_history`，不再拼接或依赖 `queue_history`。

模型内部按自回归方式生成这些输出：先基于历史窗口预测第 1 张未来图，
再把第 1 张预测图的 queue/link/compute 属性滚入历史窗口，继续预测第 2 张，
直到生成 `forecast_horizon` 张未来图。`queue_history` 中不属于预测目标的辅助维度
会沿用最近一帧的值作为下一步输入模板；最后一维变化率特征会根据“当前预测负载 - 上一帧负载”
重新写入下一步输入，因此每个未来预测步都会继续看到预测负载变化带来的动态趋势。

如果需要同时拿到图形式的预测结果，可以调用：

```python
preds, graph_preds = planner.predict_future(return_graphs=True)
```

如果你拿到的是训练封装器，也可以直接调用：

```python
graph_preds = trainer.predict_future_graphs()
```

模型本体也支持有向图输入/输出：

```python
graph_preds = model.predict_graphs(history_graphs)
```

其中 `history_graphs` 是按时间排序的 `networkx.DiGraph` 列表。节点可以提供
`queue_features`、`compute_features`、`business_time`，边可以提供 `link_features`；
缺失特征时分别退化读取 `queue_load`、`compute_queue`、`link_load`。

当 `GlobalTransformerTrainer.add_env_snapshot(env)` 传入的环境实现了
`build_graph_for_transformer(time)` 时，预测前会按 `forecast_horizon`
逐步调用该函数构建未来拓扑图。例如 `forecast_horizon=5` 时会构建 5 张未来图。

`graph_preds` 的格式为：

```text
{
    predicted_sim_time: networkx.Graph
}
```

其中每个预测图的节点属性包含：

```text
predicted_queue_load
predicted_compute_queue
predicted_business_time
```

每条边的属性包含：

```text
predicted_link_load
propagation_delay  # 如果当前快照里有传播时延
```

链路预测只会写到未来拓扑图中实际存在、且能在模型固定 `edge_names`
中找到的链路上；未来图中不存在的链路不会写入预测负载。

## 路径规划

```python
from Transformer_module import GlobalStateExtractor, TransformerPathPlanner

snapshot = GlobalStateExtractor.from_env(env)

planner = TransformerPathPlanner.from_snapshot(
    transformer=model,
    snapshot=snapshot,
    history_len=12,
    device="cuda",
)

# 仿真过程中持续加入全局快照
planner.add_snapshot(GlobalStateExtractor.from_env(env))

plan = planner.plan(
    source="sat_0_1",
    destination="sat_4_8",
    packet_size=600_000_000.0,
    computing_demand=2.5e11,
    need_compute=True,
    top_k=16,
    max_hops=12,
)

print(plan.path)
print(plan.compute_flags)
print(plan.score, plan.drop_risk)
```

`plan` 中包含：

```text
path                 推荐路径
compute_flags        与 path 等长，参与计算的卫星为 1，否则为 0
score                综合风险得分，越低越好
predicted_delay      预测/估计路径传播时延
max_queue_load       路径节点最大预测队列负载
max_link_load        路径链路最大预测业务负载
max_compute_queue    路径中最佳计算节点的预测计算排队量
drop_risk            阻塞/丢包风险代理值
```

注意：随机业务到达、链路删除、节点删除会导致未来不确定，因此 planner 不能数学保证“绝不丢包”，但会选择预测风险最低的路径。

## 在训练脚本中使用

`PRC_GNN_TRANSFORMER.py` 现在只需要调用封装器：

```python
from Transformer_module import (
    GlobalTransformerTrainer,
    average_metric_dicts,
    format_prediction_metrics,
)

trainer = GlobalTransformerTrainer(transformer_cfg, device)
trainer.add_env_snapshot(env)
loss = trainer.update_if_ready()
metrics = trainer.evaluate_latest() if trainer.should_eval() else None
plan = trainer.recommend_path() if trainer.should_plan() else None
```
