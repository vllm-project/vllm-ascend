# Expert Offload 模块分析文档

本文档分析 expert offload 相关的未提交代码，包含各函数的功能、入参出参的类型及 shape。

---

## 目录

1. [ascend_config.py](#1-ascend_configpy---expert_offload_config)
2. [expert_offload_manager.py](#2-expert_offload_managerpy)
3. [hotness_tracker.py](#3-hotness_trackerpy)
4. [sliding_window_counter.py](#4-sliding_window_counterpy)
5. [fused_moe.py](#5-fused_moepy---offload-相关新增)
6. [model_runner_v1.py](#6-model_runner_v1py---offload-初始化)

---

## 1. ascend_config.py - ExpertOffloadConfig

**文件位置**: `vllm_ascend/ascend_config.py`

### ExpertOffloadConfig 类

Expert offload 配置类，继承自配置字典。

#### `__init__(user_config)`

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `user_config` | `dict \| None` | `None` | 用户配置 |

#### 配置项默认值

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `expert_offload` | `bool` | `False` | 是否启用 expert offload |
| `num_device_experts` | `int` | `32` | NPU 设备上缓存的 expert 数量 |
| `expert_map_path` | `str \| None` | `None` | Expert 映射路径 |
| `enable_prefetch` | `bool` | `True` | 是否启用预取 |
| `prefetch_hotness_top_k` | `int` | `8` | 热点 expert 数量 |
| `prefetch_min_threshold` | `int` | `2` | 热点最小阈值 |
| `prefetch_window_size` | `int` | `200` | 预取滑动窗口大小 |
| `prefetch_num_workers` | `int` | `8` | 预取线程数 |

---

## 2. expert_offload_manager.py

**文件位置**: `vllm_ascend/expert_offload/expert_offload_manager.py`

### ExpertOffloadManager 类

Expert offload 单例管理器，负责管理 CPU 侧的 expert 权重和 NPU 分页。

#### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `offload_config` | `ExpertOffloadConfig` | offload 配置 |
| `num_device_experts` | `int` | 设备上 expert 数量 |
| `w13_weights_cpu` | `list[list[torch.Tensor]]` | CPU 上的 w13 权重列表 |
| `w2_weights_cpu` | `list[list[torch.Tensor]]` | CPU 上的 w2 权重列表 |
| `moe_layers` | `list` | 注册的 AscendFusedMoE 层 |
| `_hotness_counter` | `SlidingWindowCounter \| None` | 热点计数器 |
| `_hotness_tracker` | `ExpertHotnessTracker \| None` | 热点追踪器 |
| `_transfer_thread` | `ExpertOffloadThread \| None` | 异步传输线程 |

#### `create_weights(num_moe_layers, num_total_experts, w13_up_dim, hidden_size, intermediate_size_per_partition, params_dtype)`

分配 CPU 缓冲区存储所有 MoE 层的 expert 权重。

| 参数 | 类型 | 说明 |
|------|------|------|
| `num_moe_layers` | `int` | MoE 层数量 |
| `num_total_experts` | `int` | 总 expert 数量 |
| `w13_up_dim` | `int` | w13 上投影维度 |
| `hidden_size` | `int` | 隐藏层维度 |
| `intermediate_size_per_partition` | `int` | 每分区的中间层维度 |
| `params_dtype` | `torch.dtype` | 参数数据类型 |

**CPU 权重 shape**:
- `w13`: `[hidden_size, w13_up_dim]` per expert
- `w2`: `[intermediate_size_per_partition, hidden_size]` per expert

#### `register_moe_layer(layer)`

注册 AscendFusedMoE 层。

| 参数 | 类型 | 说明 |
|------|------|------|
| `layer` | `torch.nn.Module` | AscendFusedMoE 实例 |

#### `load_w13(layer_moe_idx, expert_id, loaded_weight, shard_id)`

将 w1/w3 分片存储到 CPU 缓冲区（带 transpose 转成后置格式）。

| 参数 | 类型 | 说明 |
|------|------|------|
| `layer_moe_idx` | `int` | MoE 层索引 |
| `expert_id` | `int` | Expert ID |
| `loaded_weight` | `torch.Tensor` | 加载的权重 |
| `shard_id` | `str` | 分片 ID (`"w1"` 或 `"w3"`) |

#### `load_w2(layer_moe_idx, expert_id, loaded_weight)`

将 w2 权重存储到 CPU 缓冲区（带 transpose 转后置格式）。

| 参数 | 类型 | 说明 |
|------|------|------|
| `layer_moe_idx` | `int` | MoE 层索引 |
| `expert_id` | `int` | Expert ID |
| `loaded_weight` | `torch.Tensor` | 加载的权重 |

#### `init_device_experts()`

将前 `num_device_experts` 个 expert 从 CPU 拷贝到 NPU。

#### `init_async_offload(num_layers, window_size=200)`

初始化异步 offload 组件（预取线程、热点追踪器）。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `num_layers` | `int` | - | MoE 层数量 |
| `window_size` | `int` | `200` | 滑动窗口大小 |

#### `shutdown_async_offload()`

关闭异步 offload 线程。

#### `update_weights(layer, topk_ids, log2phy) -> int`

根据 topk_ids 增量分页加载需要的 expert，只拷贝不在设备上的 expert。

| 参数 | 类型 | 说明 |
|------|------|------|
| `layer` | `AscendFusedMoE` | MoE 层实例 |
| `topk_ids` | `torch.Tensor` | `[num_tokens, top_k]` 路由 expert 索引 |
| `log2phy` | `torch.Tensor` | `[global_num_experts]` CPU tensor，in-place 修改 |

**返回**: `int` - 执行 CPU→NPU 拷贝的次数

#### `trigger_prefetch_for_next_layer(current_layer_idx, current_expert_ids)`

Layer N 计算完成后，触发 Layer N+1 的 prefetch。

预取并集来源:
1. Layer N 激活的 expert 编号
2. Layer N+1 上个 step 的 expert 编号
3. Layer N+1 的热点 expert 编号

| 参数 | 类型 | 说明 |
|------|------|------|
| `current_layer_idx` | `int` | Layer N 的索引 |
| `current_expert_ids` | `list[int]` | Layer N 激活的 expert ID 列表 |

#### `reset_request_scope()`

请求结束时重置滑动窗口计数器和历史记录。

**调用位置**: `model_runner_v1.py` 的 `execute_model()` 方法（第 1571-1572 行）
- 当 `num_scheduled_tokens == 0` 且无 KV transfer 时，在返回 `EMPTY_MODEL_RUNNER_OUTPUT` 前调用

---

### ExpertOffloadThread 类

异步 H2D 传输线程，处理 expert 权重的异步预取。

#### `add_prefetch_task(layer_idx, expert_ids, priority=1, task_id=None) -> int`

添加预取任务到队列。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `layer_idx` | `int` | - | MoE 层索引 |
| `expert_ids` | `list[int]` | - | 需要预取的 expert ID 列表 |
| `priority` | `int` | `1` | 优先级，数值越小优先级越高 |
| `task_id` | `int \| None` | `None` | 任务 ID，None 则自动生成 |

**返回**: `int` - 任务 ID

#### `_execute_transfer(task)`

执行实际的 H2D 传输。

| 参数 | 类型 | 说明 |
|------|------|------|
| `task` | `dict` | 任务字典，包含 `layer_idx` 和 `expert_ids` |

---

## 3. hotness_tracker.py

**文件位置**: `vllm_ascend/expert_offload/hotness_tracker.py`

### ExpertHotnessTracker 类

根据 sliding window counter 输出热点 expert 编号。

#### `__init__(counter, top_k_default=8)`

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `counter` | `SlidingWindowCounter` | - | 滑动窗口计数器实例 |
| `top_k_default` | `int` | `8` | 默认返回的热点 expert 数量 |

#### `record_step_experts(layer_idx, expert_ids)`

记录当前 step 激活的 expert，作为下一个 step 的历史参考。

| 参数 | 类型 | 说明 |
|------|------|------|
| `layer_idx` | `int` | MoE 层索引 |
| `expert_ids` | `list[int]` | 该层当前激活的 expert ID 列表 |

#### `get_prev_step_experts(layer_idx) -> set[int]`

获取 layer_idx 上个 step 的 expert 编号。

| 参数 | 类型 | 说明 |
|------|------|------|
| `layer_idx` | `int` | MoE 层索引 |

**返回**: `set[int]` - 上个 step 激活的 expert ID 集合

#### `get_union_experts(layer_idx, source1_experts, hotness_top_k=None) -> set[int]`

获取 prefetch 并集。

并集来源:
- 来源 1: `source1_experts` (Layer N 激活的 expert)
- 来源 2: Layer N+1 上个 step 的 expert 编号
- 来源 3: Layer N+1 的热点 expert 编号

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `layer_idx` | `int` | - | Layer N+1 的索引 |
| `source1_experts` | `list[int]` | - | Layer N 激活的 expert 编号列表 |
| `hotness_top_k` | `int \| None` | `None` | 热点 expert 数量 |

**返回**: `set[int]` - 需要预取的 expert ID 集合

#### `reset()`

请求结束时重置，清理历史记录。

---

## 4. sliding_window_counter.py

**文件位置**: `vllm_ascend/expert_offload/sliding_window_counter.py`

### SlidingWindowCounter 类

按 layer 维护滑动窗口计数器，统计当前请求的 expert 激活频率。

#### `__init__(num_layers, window_size=200)`

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `num_layers` | `int` | - | MoE 层的数量 |
| `window_size` | `int` | `200` | 滑动窗口大小 |

#### `record(layer_idx, expert_ids)`

记录一次 expert 激活事件。

| 参数 | 类型 | 说明 |
|------|------|------|
| `layer_idx` | `int` | MoE 层索引 |
| `expert_ids` | `list[int]` | 该层激活的 expert ID 列表 |

#### `get_topk_hot_experts(layer_idx, top_k, min_threshold=2) -> list[int]`

获取热点 expert 编号，按频率排序。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `layer_idx` | `int` | - | MoE 层索引 |
| `top_k` | `int` | - | 返回前 k 个热点 expert |
| `min_threshold` | `int` | `2` | 最小激活次数阈值 |

**返回**: `list[int]` - 按频率降序排列的 expert ID 列表

#### `reset(layer_idx=None)`

重置计数器。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `layer_idx` | `int \| None` | `None` | `None` 则重置所有层 |

---

## 5. fused_moe.py - Offload 相关新增

**文件位置**: `vllm_ascend/ops/fused_moe/fused_moe.py`

### AscendFusedMoE 类新增属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `enable_expert_offload` | `bool` | 是否启用 expert offload |
| `log2phy` | `torch.Tensor` | `[global_num_experts]` CPU tensor，expert ID 到设备槽位的映射 |

### 新增方法

#### `_wrap_weight_loader_for_offload()`

包装 `weight_loader` 以拦截 w13/w2 权重并存储到 CPU。

#### `trigger_prefetch_for_next_layer(layer_idx, current_expert_ids)` (间接调用)

在 `apply()` 方法中调用，触发下一层的预取。

**调用位置**: `AscendUnquantizedFusedMoEMethod.apply()` 第 170-178 行

```python
if getattr(layer, 'enable_expert_offload', False):
    from vllm_ascend.expert_offload import ExpertOffloadManager
    manager = ExpertOffloadManager.get_instance()
    manager.update_weights(layer, topk_ids, layer.log2phy)
    log2phy = layer.log2phy = layer.log2phy.to(topk_ids.device)
    current_expert_ids = topk_ids.unique().cpu().tolist()
    layer_idx = manager.moe_layers.index(layer)
    manager.trigger_prefetch_for_next_layer(layer_idx, current_expert_ids)
```

---

## 6. model_runner_v1.py - Offload 初始化

**文件位置**: `vllm_ascend/worker/model_runner_v1.py`

### 新增导入

```python
from vllm_ascend.expert_offload.expert_offload_manager import (
    maybe_init_expert_offload_manager,
    has_expert_offload_manager,
    get_expert_offload_manager,
)
```

### 新增初始化逻辑 (约第 483-491 行)

```python
if self.ascend_config.expert_offload_config.expert_offload:
    maybe_init_expert_offload_manager(self.vllm_config)
    if has_expert_offload_manager():
        self.offload_manager = get_expert_offload_manager()
        if self.offload_manager._prefetch_config["enabled"]:
            self.offload_manager.init_async_offload(
                num_layers=len(self.offload_manager.moe_layers),
                window_size=self.ascend_config.expert_offload_config.prefetch_window_size
            )
```

### `_register_offload_layers()` 方法 (约第 3056-3077 行)

查找所有 AscendFusedMoE 层并注册到 ExpertOffloadManager。

```python
def _register_offload_layers(self):
    """Find all AscendFusedMoE layers and register with ExpertOffloadManager."""
    from vllm_ascend.ops.fused_moe.fused_moe import AscendFusedMoE

    moe_layers = [m for m in self.model.modules()
                  if isinstance(m, AscendFusedMoE)]
    if not moe_layers:
        return
    first = moe_layers[0]
    w13_up_dim = (first.w13_weight.shape[2] if hasattr(first, 'w13_weight')
                  else first.intermediate_size_per_partition * 2)
    self.offload_manager.create_weights(
        num_moe_layers=len(moe_layers),
        num_total_experts=first.global_num_experts,
        w13_up_dim=w13_up_dim,
        hidden_size=first.hidden_size,
        intermediate_size_per_partition=first.intermediate_size_per_partition,
        params_dtype=first.params_dtype,
    )
    for layer in moe_layers:
        self.offload_manager.register_moe_layer(layer)
    self.offload_manager.init_device_experts()
```

---

## 数据流总结

```
model_runner_v1.py 初始化时:
    1. maybe_init_expert_offload_manager() 创建单例
    2. _register_offload_layers() 找到所有 MoE 层
    3. create_weights() 分配 CPU 缓冲区
    4. register_moe_layer() 注册每层
    5. init_device_experts() 拷贝前 N 个 expert 到 NPU

前向传播时:
    AscendFusedMoE.forward_impl()
        -> AscendUnquantizedFusedMoEMethod.apply()
            -> ExpertOffloadManager.update_weights() [增量分页]
            -> ExpertOffloadManager.trigger_prefetch_for_next_layer() [异步预取]
                -> SlidingWindowCounter.record() [记录热点]
                -> ExpertHotnessTracker.record_step_experts() [记录历史]
                -> ExpertHotnessTracker.get_union_experts() [计算并集]
                -> ExpertOffloadThread.add_prefetch_task() [加入队列]
```

---

## 配置示例

```json
{
  "expert_offload_config": {
    "expert_offload": true,
    "num_device_experts": 32,
    "enable_prefetch": true,
    "prefetch_hotness_top_k": 8,
    "prefetch_min_threshold": 2,
    "prefetch_window_size": 200,
    "prefetch_num_workers": 8
  }
}
```