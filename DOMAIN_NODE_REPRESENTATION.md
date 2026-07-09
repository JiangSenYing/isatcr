# 当前域节点表征信息整理

本文档整理当前代码中“域节点”的表征方式。这里的域节点指 MEO 控制器代表一个控制域生成的聚合节点，不是 `observation_wrappers.py` 中强化学习局部观测里的 `agent` / `nbr` 节点。

## 1. 域节点的来源

域划分由 `SatelliteDomainPartitioner` 完成，当前工程采用“以 MEO 作为控制器、LEO 作为被控交换节点”的实现方式。

一次分域后，运行时会写回这些关系：

| 信息 | 写入位置 | 含义 |
| --- | --- | --- |
| `sat.masterMeo` | LEO 卫星对象 | 当前 LEO 归属的 MEO 控制器 |
| `sat.intra_domain_leo` | LEO 卫星对象 | 与该 LEO 同域的 LEO 列表 |
| `sat.my_leos` | MEO 卫星对象 | 该 MEO 当前管理的 LEO 列表 |
| `graph.nodes[leo]['domain_controller']` | NetworkX 图节点属性 | LEO 所属域控制器 |
| `graph.nodes[meo]['managed_leo']` | NetworkX 图节点属性 | MEO 管理的 LEO 列表 |

相关代码位置：

- `SatelliteDomainPartitioner._flush_assignment_to_runtime()`
- `SatelliteNetworkSimulator_Computing.Satellite_with_Computing.configMasterMeo()`

## 2. LEO 上报给 MEO 的原始状态

每个 MEO 周期性收集自己域内 LEO 的状态。状态由 `build_leo_state_for_meo()` 构造，核心结构如下：

```python
leo_state = {
    'timestamp': self.env.now,
    'name': self.name,
    'is_producing': int(bool(self.is_producing)),
    'remaining_memory': self._remaining_memory(),
    'remaining_computing': self._remaining_computing_resource(),
    'neighbors': neighbors_state,
}
```

### LEO 自身字段

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `timestamp` | float | 状态生成时刻，即仿真时间 `env.now` |
| `name` | str | LEO 节点名称 |
| `is_producing` | int | 是否处于业务产生状态，`0/1` |
| `remaining_memory` | float | 当前剩余存储资源 |
| `remaining_computing` | float | 当前剩余计算资源 |
| `neighbors` | list[dict] | 最多 4 个邻居的链路和资源摘要 |

### LEO 邻居字段

每个 LEO 上报前 4 个邻居；不足 4 个时用虚拟邻居补齐。

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `name` | str 或 None | 邻居名称；虚拟邻居为 `None` |
| `link_load` | float | 本 LEO 发往该邻居的传输队列占用，即 `transmission_size[neighbor]` |
| `remaining_memory` | float | 邻居剩余存储资源 |
| `remaining_computing` | float | 邻居剩余计算资源 |

虚拟邻居默认值：

```python
{
    'name': None,
    'link_load': float("inf"),
    'remaining_memory': 0.0,
    'remaining_computing': float("inf"),
}
```

## 3. 域节点聚合表征

当 MEO 收齐当前 `my_leos` 中所有 LEO 的状态后，调用 `_build_domain_aggregate_state(domain_leos)` 生成域级聚合状态。

聚合结果结构如下：

```python
aggregate = {
    'domain': self.name,
    'timestamp': self.env.now,
    'members': domain_leos,
    'member_count': len(domain_leos),
    'boundary_nodes': boundary_nodes,
    'boundary_links': boundary_links,
    'intra_graph': intra_graph,
    'intra_risks': intra_risks,
    'avg_memory_occupancy_rate': ...,
    'max_memory_occupancy_rate': ...,
    'avg_computing_occupancy_rate': ...,
    'max_computing_occupancy_rate': ...,
}
```

### 域节点字段

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `domain` | str | 域节点 ID，当前等于 MEO 名称 |
| `timestamp` | float | 聚合状态生成时间 |
| `members` | list[str] | 域内 LEO 成员列表 |
| `member_count` | int | 域内 LEO 数量 |
| `boundary_nodes` | list[str] | 本域内与其他域相连的边界 LEO |
| `boundary_links` | list[dict] | 跨域边界链路明细 |
| `intra_graph` | `nx.Graph` | 域内 LEO 子图，包含节点资源和域内链路负载 |
| `intra_risks` | dict | 边界节点对之间的域内路径风险 |
| `avg_memory_occupancy_rate` | float | 域内平均存储占用率 |
| `max_memory_occupancy_rate` | float | 域内最大存储占用率 |
| `avg_computing_occupancy_rate` | float | 域内平均计算占用率 |
| `max_computing_occupancy_rate` | float | 域内最大计算占用率 |

资源占用率计算：

```text
memory_occupancy_rate = clip01(1 - remaining_memory / memory_capacity)
computing_occupancy_rate = clip01(1 - remaining_computing / computing_capacity)
```

## 4. 域内图 `intra_graph`

`intra_graph` 是域节点内部保留的 LEO 级子图，用来支持边界路径风险评估。

### 域内 LEO 节点属性

| 属性 | 类型 | 含义 |
| --- | --- | --- |
| `timestamp` | float | 该 LEO 状态时间戳 |
| `is_producing` | int | 是否产生业务 |
| `remaining_memory` | float | 剩余存储资源 |
| `remaining_computing` | float | 剩余计算资源 |
| `memory_occupancy_rate` | float | 存储占用率，范围约束到 `[0, 1]` |
| `computing_occupancy_rate` | float | 计算占用率，范围约束到 `[0, 1]` |

### 域内链路属性

只对同域邻居建立域内边。

| 属性 | 类型 | 含义 |
| --- | --- | --- |
| `link_load` | float | 原始链路负载 |
| `normalized_link_load` | float | 按源节点 memory 归一化后的链路负载 |
| `delay` | float | 链路传播时延 |
| `weight` | float | 当前为 `delay + normalized_link_load` |

如果同一条边被两个方向重复上报，代码保留更大的 `link_load` 对应的属性。

## 5. 跨域边界链路 `boundary_links`

当 LEO 的邻居不在本域，且该邻居有不同的 `masterMeo` 时，会生成一条跨域边界链路。

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `source_domain` | str | 本域 MEO |
| `target_domain` | str | 邻居所属 MEO |
| `source_boundary` | str | 本域边界 LEO |
| `target_boundary` | str | 对端域边界 LEO |
| `link_load` | float | 原始链路负载 |
| `normalized_link_load` | float | 归一化链路负载 |
| `source_memory_occupancy_rate` | float | 本域边界 LEO 存储占用率 |
| `target_memory_occupancy_rate` | float | 对端边界 LEO 存储占用率 |
| `quality_cost` | float | 跨域链路质量代价 |

`quality_cost` 当前定义为三项平均：

```text
quality_cost = mean(
    normalized_link_load,
    source_memory_occupancy_rate,
    target_memory_occupancy_rate
)
```

## 6. 域内风险 `intra_risks`

`intra_risks` 以边界 LEO 对为键：

```python
intra_risks[(entry_boundary, exit_boundary)] = {
    'path': path,
    'risk': risk,
    'delay': total_delay,
    'max_node_memory_occupancy_rate': max_memory,
    'max_link_load': max_link,
    'max_compute_occupancy_rate': max_compute,
    'reachable': True,
}
```

### 风险字段

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `path` | list[str] | 从入口边界到出口边界的域内路径 |
| `risk` | float | 综合风险，范围约束到 `[0, 1]` |
| `delay` | float | 路径总传播时延 |
| `max_node_memory_occupancy_rate` | float | 路径上最大节点存储占用率 |
| `max_link_load` | float | 路径上最大链路负载 |
| `max_compute_occupancy_rate` | float | 路径上最大计算占用率 |
| `reachable` | bool | 是否存在可达路径 |

综合风险公式：

```text
risk = clip01(
    0.35 * max_link
  + 0.30 * max_memory
  + 0.20 * max_compute
  + 0.15 * delay_risk
)
```

其中：

```text
delay_risk = clip01(total_delay / (max_hop * state_update_period))
```

如果不可达，风险字段使用保守值：

```python
{
    'path': [],
    'risk': 1.0,
    'delay': float("inf"),
    'max_node_memory_occupancy_rate': 1.0,
    'max_link_load': 1.0,
    'max_compute_occupancy_rate': 1.0,
    'reachable': False,
}
```

## 7. 域间图 `inter_domain_graph`

每个 MEO 维护一个 `inter_domain_graph`，节点是域，边是域间连接。

`_refresh_inter_domain_graph(aggregate)` 会把当前域聚合写入图节点：

| 节点属性 | 来源 | 含义 |
| --- | --- | --- |
| `aggregate` | 完整 `aggregate` | 域节点完整聚合状态 |
| `members` | `aggregate['members']` | 域内 LEO 成员 |
| `boundary_nodes` | `aggregate['boundary_nodes']` | 本域边界 LEO |
| `intra_risks` | `aggregate['intra_risks']` | 域内边界路径风险 |
| `avg_memory_occupancy_rate` | 聚合字段 | 平均存储占用率 |
| `max_memory_occupancy_rate` | 聚合字段 | 最大存储占用率 |
| `avg_computing_occupancy_rate` | 聚合字段 | 平均计算占用率 |
| `max_computing_occupancy_rate` | 聚合字段 | 最大计算占用率 |
| `timestamp` | 聚合字段 | 状态时间戳 |

域间边保留跨域边界链路列表，并派生三个摘要数组：

| 边属性 | 含义 |
| --- | --- |
| `boundary_links` | 两个域之间所有边界链路明细 |
| `quality_costs` | 每条边界链路的 `quality_cost` 列表 |
| `link_loads` | 每条边界链路的原始 `link_load` 列表 |
| `memory_occupancy_rates` | 每条边界链路两端边界节点的存储占用率二元组 |

## 8. 更新时间序

当前域节点表征的生成时序如下：

1. `SatelliteDomainPartitioner.calcDomainPartition()` 完成 LEO 到 MEO 的域划分。
2. `_flush_assignment_to_runtime()` 写回 `masterMeo`、`my_leos`、`managed_leo` 等运行时字段。
3. MEO 启动 `collect_leo_states_from_domain()`，每隔 `meo_state_update_period` 收集一次域内 LEO 状态。
4. MEO 等待域内 LEO 到 MEO 的最大传播时延后，逐个调用 LEO 的 `build_leo_state_for_meo()`。
5. MEO 通过 `receive_leo_state()` 缓存 LEO 状态；当 `my_leos` 全部收齐后生成 `domain_aggregate`。
6. MEO 调用 `_refresh_inter_domain_graph()` 更新本地域间图。
7. MEO 调用 `state_exchanger_meo()`，把域级聚合状态洪泛给相邻 MEO。

## 9. 与强化学习节点观测的区别

当前代码中还有一套面向强化学习决策的节点观测：

- `get_current_state()` 返回扁平向量。
- `RelationalObservation` / `GraphObservation` 把该向量转换成 `agent` / `nbr` 异构图。
- 在分离模式中，`agent` 仅保留当前节点环境特征，任务特征单独传递。

这套观测用于单个卫星的转发/计算决策；本文整理的域节点表征用于 MEO 汇聚域内状态并维护域间大节点图。两者共享底层资源状态，但结构和用途不同。

## 10. 关键代码索引

| 代码位置 | 作用 |
| --- | --- |
| `SatelliteDomainPartitioner.py` | 域划分、边界域关系、运行时写回 |
| `SatelliteNetworkSimulator_Computing.py::configMasterMeo` | 写入 LEO/MEO 域归属关系 |
| `SatelliteNetworkSimulator_Computing.py::build_leo_state_for_meo` | 构造 LEO 上报给 MEO 的原始状态 |
| `SatelliteNetworkSimulator_Computing.py::_build_domain_aggregate_state` | 构造域级聚合节点 |
| `SatelliteNetworkSimulator_Computing.py::_refresh_inter_domain_graph` | 更新域间图节点和边属性 |
| `SatelliteNetworkSimulator_Computing.py::_compute_intra_domain_risks` | 计算域内边界路径风险 |
| `SatelliteNetworkSimulator_Computing.py::collect_leo_states_from_domain` | 周期性收集域内 LEO 状态 |
| `SatelliteNetworkSimulator_Computing.py::state_exchanger_meo` | 在 MEO 间传播域级聚合状态 |
