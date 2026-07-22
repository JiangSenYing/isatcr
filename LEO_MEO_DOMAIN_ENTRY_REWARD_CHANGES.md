# LEO–MEO 域入口子任务奖励改动说明

## 1. 改动目标

原有 LEO Agent 主要在数据包到达最终下传卫星时获得到达奖励，跨域过程中到达 MEO 指定的域入口只会结算普通逐跳奖励。

本次改动将 LEO Agent 的任务拆分为连续的域内子任务：

1. 从当前域入口出发；
2. 在域内完成计算和逐跳路由决策；
3. 到达 MEO 指定的下一域入口；
4. 结算当前域入口阶段奖励，并在新域继续联合决策；
5. 进入目的域后，继续完成“目的域入口到最终下传卫星”的最后一段任务。

普通 MEO 域入口是同一 LEO episode 内的阶段边界，最终下传或失败才是 episode 终点。这样最终计算结果可以通过 DDQN bootstrap 影响前面域的传输和计算选择。

## 2. 奖励与 episode 语义

### 2.1 到达域入口

数据包成功到达 MEO 指定的下一域入口时，上一条 LEO 动作根据计算状态获得：

```text
已计算：reward = reach_factor - delay_factor × 当前域段时延
未计算：reward =              - delay_factor × 当前域段时延
普通入口：done = 0
```

其中：

```text
当前域段时延 = 跨域接收时间 - 上一域入口时间
```

已计算的数据包调用 `reach_reward()`；未计算的数据包调用 `reach_reward_abnormal()`，即取消入口基础正奖励，但不增加额外固定惩罚。

入口阶段经验写入后会清除上一动作引用，避免同一个动作同时获得入口奖励和普通 `normal_reward`。

### 2.2 新域子任务

数据包进入新域后，以精确的跨域接收时间作为新阶段起点。入口经验以新域、新 MEO 子目标对应的观测作为 `next_state`，并通过 `done=0` 继续估计未来 Q 值。

### 2.3 最终下传奖励

进入目的域不会取消最终下传奖励。从目的域入口到下传卫星构成最后一个 LEO 子任务，下传奖励按目的域内时延计算：

```text
目的域内时延 = 当前时间 - leo_domain_entry_time
```

如果域入口卫星同时就是最终下传卫星，则该入口经验使用 `done=1`，并且只生成一条终局经验，避免对同一个 LEO 动作重复奖励。

### 2.4 失败奖励

以下 LEO 可归因的终止失败均改为按当前域时延计算 `loss_reward()`：

- 路由队列或内存溢出；
- 下一跳或链路失效；
- 节点失效；
- 非法动作；
- 跳数超时；
- 星间传输中断；
- 下传中断。

公式为：

```text
reward = -loss_factor - delay_factor × 当前域时延
done = 1
```

非法动作会作为当前动作单独写入终止经验；到达该节点的上一动作先按普通中间奖励结算，避免把当前非法决策的责任错误归给上一跳。

## 3. Packet 时间字段

在 `Packet` 中新增两个内部字段：

```python
self.leo_domain_entry_time = creation_time
self.leo_previous_domain_entry_time = None
```

字段语义如下：

| 字段 | 含义 |
|---|---|
| `leo_domain_entry_time` | 数据包进入当前域的精确时间，也是当前 LEO 子任务的起点 |
| `leo_previous_domain_entry_time` | 刚完成域段的起点，在入口奖励结算前临时保留 |
| `meo_segment_time` | 上一 MEO 分段的精确结束时间，即跨域接收时间 |

对于缺少新增字段的旧数据包，时延计算回退到 `packet.creation_time`。

## 4. 跨域时间记录

星间链路传播成功、数据包被下一跳卫星接收后，系统比较源 LEO 与目的 LEO 的 `masterMeo`：

```python
if source_meo != target_meo:
    packet.leo_previous_domain_entry_time = packet.leo_domain_entry_time
    packet.leo_domain_entry_time = env.now
    packet.meo_segment_time = env.now
```

只有成功进入不同 MEO 域时才更新时间：

- 同域转发不更新；
- 下一跳入队失败不更新；
- 链路传播失败不更新。

记录时刻是跨域传输成功并进入新域卫星队列的时间，因此新域入口排队时间归属于新域子任务。

## 5. 域入口经验结算流程

当边界卫星发现自身属于 `packet.temporary_destination` 时：

```text
跨域接收成功
    ↓
保存上一域起点并更新当前域入口时间
    ↓
边界卫星开始处理数据包
    ↓
根据计算状态调用 reach_reward 或 reach_reward_abnormal
    ↓
写入上一 LEO 动作阶段经验，普通入口 done=0
    ↓
将 LEO 入口奖励传入 MEO finish_segment()
    ↓
根据下一 MEO 计划开启新的 LEO 子任务
```

普通入口奖励加入独立的 `domain_entry_rewards`，训练日志输出 `Average domain-entry reward`；`final_rewards` 只保留最终下传和失败奖励。入口同时为最终下传卫星时，其奖励仍归入 `final_rewards`。

## 6. 相邻域处理

原逻辑在源域和目的域直接相邻时绕过 MEO 路径推荐，导致这类任务没有明确的入口子目标。

现已改为：只要源域与目的域不同，就调用 MEO 路径推荐。这样相邻域任务也会获得 MEO 指定入口及对应的 LEO 入口奖励。

MEO 分段奖励会按现有配置权重融合该入口奖励：

```python
meo_segment_reward = original_segment_reward + terminal_reward_weight * leo_entry_reward
```

`terminal_reward_weight=0.0` 时保持原有 MEO 分段奖励；设置为非零值时，已计算和未计算包的入口奖励都会影响对应的 MEO 域间决策经验。

## 7. 保持不变的行为

- MEO Agent 原有的分段基础奖励、路径进展和代价项保持不变；
- `Total_delay_0` 和 `Total_delay_1` 仍按数据包创建时间统计端到端总时延；
- 中间逐跳动作仍使用 `normal_reward()`；
- 已计算任务最终下传仍使用 `reach_reward()`；
- 未计算任务最终下传仍使用 `reach_reward_abnormal()`；
- 普通入口通过 `done=0` 跨域传播最终任务信用；
- LEO Agent 的经验格式和网络接口未改变。

## 8. 涉及文件

### `SatelliteNetworkSimulator_Beta.py`

- 为 `Packet` 初始化 LEO 当前域和上一域时间字段。

### `SatelliteNetworkSimulator_Computing.py`

- 新增当前域时延和已完成域段时延计算方法；
- 跨域成功接收时更新时间戳；
- 域入口奖励根据 `is_computed` 选择正常或异常到达公式；
- 普通入口写入 `done=0` 的阶段经验，入口等于最终目的节点时写入 `done=1`；
- 防止入口奖励与普通逐跳奖励重复写入；
- 分离记录 `domain_entry_rewards` 与 `final_rewards`；
- 下传奖励和终止失败改用当前域时延；
- 补齐非法动作、传输中断和下传中断的终止经验；
- 相邻域任务统一调用 MEO 规划。

### `test_downstream_domain_reward.py`

新增轻量测试，覆盖：

- 当前域下传时延；
- 旧数据包时间回退；
- 跨域接收时间戳更新；
- 同域转发不更新时间；
- 已完成域段时延计算；
- 已计算和未计算入口使用不同奖励公式；
- 普通入口经验使用 `done=0`，最终目的入口使用 `done=1`；
- 入口动作不会重复获得普通奖励；
- 域入口奖励和最终奖励分别统计；
- Packet 第一域时间初始化；
- 相邻域不会绕过 MEO 入口规划。

## 9. 验证结果

本次定向测试结果：

```text
14 passed
```

同时通过：

- Python AST 语法检查；
- `git diff --check`。

完整 MEO 测试在当前运行环境中无法收集，原因为环境缺少 `torch`，不是本次代码的语法或定向测试失败。

## 10. 当前奖励配置提示

LEO 的入口成功奖励继续复用 `Reward_Function.reach_reward()`：

```python
return reach_factor - delay_factor * delay
```

因此，配置中的 `delay_factor` 会直接决定域入口奖励随域内耗时衰减的速度。若 `delay_factor` 较大，即使成功到达入口，较长域段也可能得到负奖励。训练时应结合典型单域传输和计算耗时检查奖励尺度。
