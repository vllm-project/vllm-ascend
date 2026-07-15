# DeepSeek V4 Layerwise KV Pool 设计

## 背景

AscendStore 原有的 layerwise 路径按以下假设实现：

- 只有一个 KV cache group；
- 模型执行层序号等于缓存中的物理层序号；
- 所有层共享相同的 block size 和 token 粒度；
- layerwise 请求可以通过一个全局递增的 `current_layer` 识别当前层。

DeepSeek V4 不满足这些假设。它会按 cache spec 和压缩比例把缓存拆成多个 group，典型形态包含
c4、c128 和 sliding-window cache。MTP 层也可能出现在注册的 KV cache 中。因此，同一个模型层同时具有：

- 模型执行顺序中的位置；
- KV cache group id；
- group 内的局部层序号；
- group 对应的压缩比例、block table 和物理地址布局。

如果仅删除原有的多 group `NotImplementedError`，put/get 会使用 group 0 的 block table 和地址，lookup
也会为每个 group 查询模型全部层，最终造成错误命中、地址越界或 KV 数据写入错误层。

## 目标与边界

本设计实现以下能力：

1. DeepSeek V4 的每个已注册 KV cache 层可以独立 put 和 get。
2. c4、c128、sliding-window 等多个 group 使用各自的 key 粒度、block table 和物理地址。
3. lookup 只有在每个 group 的全部层都命中时才返回公共前缀命中。
4. 保留一层 look-ahead get，使下一层传输与当前层计算重叠。
5. MTP 重复执行基础层时，不重复提交同一层的 put。
6. 单 group 模型保持原有 key 和调用行为。

本次 layerwise put 的范围是 prefill 节点：只有 `kv_producer` 和 `kv_both` 创建逐层发送线程。
`kv_consumer` 仍可逐层 get，但即使配置了 `consumer_is_to_put=true` 也不会逐层 put。该配置继续只控制
非 layerwise 的 decode 节点回灌能力，不属于本次适配范围。

本次不改变后端协议，不新增环境变量，也不修改 DeepSeek V4 模型文件和 attention kernel。唯一功能开关是
`kv_connector_extra_config.use_layerwise=true`。

## 核心数据模型

### 层映射

`register_kv_caches` 根据 `KVCacheConfig.kv_cache_groups` 建立两种独立视图：

| 视图 | 示例 | 用途 |
|------|------|------|
| 执行顺序 | layer 0、layer 1、...、MTP | 决定 get 预取和“最后一个唯一层” |
| 物理位置 | `(group_id, group_layer_id)` | 决定 key、block table 和地址 |

DeepSeek V4 的执行顺序使用 `extract_dsv4_layer_index` 排序，避免 cache group 的注册顺序被误认为模型
forward 顺序。物理位置始终保留 group 内原始顺序，以便和注册的地址数组一致。

### Layerwise key

每个 key 由以下字段共同确定：

```text
model + parallel ranks + group_id + cache_family + group_layer_id + chunk_hash
```

`group_layer_id` 是组内序号，而不是全局模型层序号。因为 key 已包含 `group_id`，不同 group 的局部层 0
不会冲突。该方案也避免为稀疏分组构造大量不存在的层 key。

### Layerwise 传输任务

单层任务增加以下信息：

- `kv_cache_group_id`：选择当前层所属 group；
- `layer_id`：当前 group 内的局部层序号；
- `key_block_ids`：每个 key 对应的真实物理 block id；
- `layer_name`：用于校验 attention hook 传入的真实层；
- `is_last_layer`：与 group 内局部序号解耦的全局完成标记；
- `completion_event` 和 `load_failed`：单层 get 的完成与失败状态。

显式携带 `key_block_ids` 是 sliding-window 正确性的必要条件。滑窗 group 可能只保留逻辑序列尾部的
block table，不能再用 `start // block_size` 直接索引这个截断后的列表。

## Lookup 设计

lookup 对每个 group 独立执行：

1. 按该 group 的 `block_size * compress_ratio` 对原始 block hash 重新分组。
2. 每个 chunk 只展开该 group 的层数，而不是模型总层数。
3. 对同一 chunk 的全部组内层执行 AND；任一层缺失则该 chunk 未命中。
4. 对 TP/PP 副本继续执行 AND。
5. 将压缩缓存位置乘以 `compress_ratio`，恢复为原始 token 坐标。
6. 在所有 group 的命中位置中取最大公共位置。

第 5 步是 DeepSeek V4 多压缩组可以求交集的关键。c4 group 的物理位置 512 和 c128 group 的物理
位置 16 都可能代表原始 token 位置 2048；不能直接比较两个物理位置。

## Put 时序

1. attention hook 调用 `save_kv_layer(layer_name, ...)`。
2. connector 把真实 `layer_name` 传给 worker。
3. worker 查出 `(group_id, group_layer_id)`，使用该 group 的 block table 生成单层任务。
4. 当前层记录独立的 NPU event，发送线程在访问该层 KV 前同步该 event。
5. 发送线程按 group 和真实 block id 计算地址并调用 backend `put`。
6. 当本轮所有唯一层都已提交时，最后一个任务携带 `is_last_layer=True`。
7. MTP 再次执行基础层时，已提交的 `layer_name` 被跳过；独立的 MTP cache 层仍正常提交。

每个调度 chunk 在其最后一层完成时都会减少一次 in-flight 计数；只有最后一个 prompt chunk 才把请求
标记为整体 put 完成。这样 chunked prefill 不会残留无法归零的发送计数。

## Get 时序

1. `start_load_kv` 收集需要 load 的请求，并预取执行顺序中的第一层。
2. attention hook 在计算当前层前调用 `wait_for_layer_load(layer_name)`。
3. worker 等待当前层所有请求的 `completion_event`。
4. 当前层成功后立即提交下一层，实现一层 look-ahead。
5. 若实际 hook 顺序与预取顺序不同，则按真实 `layer_name` 补交当前层任务，避免把错误层数据当成已加载。
6. backend 返回失败或传输线程抛异常时设置 `load_failed`，当前请求在进入 attention 计算前失败。

多 group load 失败时不把裸 block id 注入单 group 的 fallback 集合，因为不同 group 可以复用相同的
数字 block id。当前实现选择 fail-fast，防止用部分恢复的 KV 继续推理而产生静默精度错误。

## 地址计算

`prepare_value_layer` 接收 `kv_cache_group_id`、组内 `layer_id` 和可选的显式 `block_id`：

```text
addr = group_layer_base_addr + block_id * group_layer_block_stride
size = group_layer_block_len / group_block_size * transferred_tokens
```

地址数组、block length、stride 和层数都从指定 group 读取。默认参数仍指向 group 0，以兼容已有单
group 调用和测试。

## 代码修改及原因

| 文件 | 修改 | 原因 |
|------|------|------|
| `attention/sfa_v1.py` | DSA-CP full-weight o_proj 提前返回前触发当前层 save callback | `kv_both` 的 DeepSeek V4 prefill 会走该提前返回分支；原逻辑跳过 callback，导致 metadata 可保存但没有 put |
| `ascend_store_connector.py` | 透传 `layer_name`；consumer 的 layerwise save callback 直接跳过 | DSV4 不能依赖全局层计数；本次逐层 put 只覆盖 prefill 角色 |
| `pool_scheduler.py` | 允许 layerwise 使用多个 KV group；consumer 不生成逐层 save metadata | 原有保护直接阻止 DSV4 启动，同时不能让 decode 回灌配置误入逐层 put |
| `config_data.py` | 扩展按组地址接口和单层任务元数据 | 同时表达 group、局部层、尾块和完成状态 |
| `pool_worker.py` | 建立层映射，重写单层任务、预取和 lookup；只为 producer/both 创建逐层 sender | 保证 key、block table、地址和命中坐标都按 group 处理，并明确 prefill put 边界 |
| `kv_transfer.py` | 发送/接收线程按 group 取地址并处理显式最后层 | 局部层 0 也可能是模型最后层，不能再和全局层数比较 |
| `tests/ut/distributed/ascend_store/` | 增加 DSV4 多组回归 | 覆盖 c4/c128 lookup、组内层、尾块及 hook 透传 |

## 配置示例

在已有 AscendStore 配置中启用 layerwise：

```json
{
  "kv_connector": "AscendStoreConnector",
  "kv_role": "kv_producer",
  "kv_connector_extra_config": {
    "backend": "mooncake",
    "use_layerwise": true
  }
}
```

layerwise 模式需要 piecewise graph；connector 的 `requires_piecewise_for_cudagraph` 会根据
`use_layerwise` 返回对应要求。

以上是分离式部署中 prefill 节点的 put 配置。若在同一实例连续验证 put 和 get，可使用 `kv_both`；
decode consumer 保持 `kv_consumer`，并且不会因为 `consumer_is_to_put=true` 产生 layerwise put。

## 验证

CPU 单元测试覆盖：

- 单 group 原有行为；
- c4 两层与 c128 一层的多 group lookup；
- 压缩位置恢复为原始 token 坐标后的公共命中；
- group 内层序号和 cache family key；
- sliding-window 截断 block table 的显式 block id；
- 单层 get 完成事件与失败标记；
- connector 的真实层名透传；
- MTP 场景所需的显式最后层判定。

代码合入前仍需在 Ascend 环境使用真实 DeepSeek V4 权重完成以下门禁：

1. 第一次请求产生逐层 put，第二次相同请求产生逐层 get。
2. 第二次请求输出与关闭 KV pool 的 greedy 输出一致。
3. 日志中的 key 同时出现预期的 group、cache family 和局部 layer id。
4. Mooncake、TP/EP、MTP 和 piecewise ACLGraph 组合无超时或传输失败。
5. 至少验证 128K 上下文、`max_num_seqs=16`；若硬件不足需记录实际可运行上限。

### Mooncake 日志验证

验证时设置已有的 vLLM 日志级别，不新增 AscendStore 环境变量：

```bash
export VLLM_LOGGING_LEVEL=INFO
```

启动日志必须依次包含：

```text
[AscendStore][layerwise] connector enabled ... backend=mooncake
[AscendStore][layerwise] cache layout registered ... transfer_granularity=<N> ...
[AscendStore][layerwise] transfer threads ready ... sender=True receiver=True
```

测试 prompt 的 token 数必须不小于启动日志中的 `transfer_granularity`。短 prompt 没有完整可传输 chunk，
会在 enqueue 阶段得到 `keys=0`，不会调用 Mooncake put。使用一个未写入过池的长 prompt 连续请求两次。

第一次请求应出现：

```text
[AscendStore][layerwise][put] enqueue ... keys=<大于 0>
[AscendStore][layerwise][put] backend call backend=MooncakeBackend ...
[AscendStore][Mooncake][put] completed ... failed=0
[AscendStore][layerwise][put] chunk complete ... processed_layers=<总层数>/<总层数> ...
```

第二次相同请求应出现 scheduler 的 `KV pool load spec created`，以及：

```text
[AscendStore][layerwise][get] enqueue ... keys=<大于 0>
[AscendStore][layerwise][get] backend call backend=MooncakeBackend ...
[AscendStore][Mooncake][get] completed ... failed=0
[AscendStore][layerwise][get] chunk complete ... status=success
```

可用以下命令过滤关键日志：

```bash
rg "\[AscendStore\].*(layerwise|Mooncake)|KV pool load spec created" <server.log>
```

关键诊断字段含义：

| 日志 | 结论 |
|------|------|
| `sender=False` | 当前节点不是 prefill put 角色；layerwise sender 只为 `kv_producer`/`kv_both` 创建 |
| `saveable=0` | scheduler 本轮没有下发可保存请求 |
| 只有 `forward metadata ... saveable=1`，没有 `save callback` | attention 执行路径跳过了逐层 hook；重点检查 SFA DSA-CP early-return 分支 |
| `reason=request_not_registered` | 请求在发送线程中已被清理，需检查请求完成/抢占时序 |
| enqueue 阶段 `keys=0` | prompt 太短，或 store mask 过滤了所有完整 chunk |
| `total_keys>0 owned_keys=0` | 当前 TP rank 不负责该 chunk；结合 `tp_rank/put_step` 跨 rank 检查，属于正常分片 |
| `backend_call_layers=0 all_exist_layers>0` | key 已存在，跳过重复 put，属于正常行为 |
| `processed_layers` 小于总层数 | attention hook 没有覆盖所有唯一 KV 层 |
| Mooncake `failed>0` | put/get 已执行，但 Mooncake 返回失败，检查容量、metadata server 和网络 |

功能通过标准为：每个 rank 的 `processed_layers` 等于本 PP rank 的总层数；跨 TP rank 汇总后首次请求的
每一层都有 owner 执行 put；第二次请求完成全部层 get；Mooncake 的 `failed=0`；第二次请求的 greedy
输出与关闭 KV pool 时一致。仅看到 `backend call` 不能证明写入成功，必须同时检查 Mooncake
`completed` 和最终 chunk 汇总。

空 metadata 通常来自 decode 或没有 connector 任务的 forward step，不代表 put 失败。为避免 INFO 验证日志刷屏，
`requests=0` 的 metadata 汇总不再打印；判断 put 是否触发应从 `saveable=1` 后的 `save callback` 开始追踪。

### 性能收益验证

layerwise 的预期收益不是减少传输总字节数，而是把已经完成计算的层尽早交给 Mooncake，使逐层 put 与后续
层的 NPU 计算重叠，从而缩短 prefill forward 结束后的集中保存尾延迟。代价是每层增加一次 event、队列调度、
exists 和 backend 调用；当单层数据量过小或 Mooncake 更依赖大批量传输时，layerwise 可能没有收益，甚至降低
吞吐。因此不能仅凭功能日志判断性能。

在同一套 DeepSeek V4 权重、并行配置、Mooncake 节点和网络条件下做以下 A/B：

| 对照组 | 实验组 | 其余变量 |
|--------|--------|----------|
| `use_layerwise=false` | `use_layerwise=true` | 模型、并行度、block size、batch、prompt、Mooncake 配置完全一致 |

分别覆盖 2K、8K、32K、128K prompt 和并发 1、4、16；硬件不足时记录实际可运行矩阵。每组先 warmup，
再至少采集 20 轮，并分别测量：

1. 使用不同 prompt 的 cold put，防止 `all_keys_exist` 把写入跳过。
2. 使用相同 prompt 的 warm get，确认逐层预取没有增加 TTFT。
3. `kv_producer` prefill 节点只评估 put；若需在单实例同时测 put/get，使用 `kv_both`。

记录 p50/p90/p99 TTFT、prefill latency、input token throughput、请求完成到最后一次 Mooncake put 完成的
尾延迟、Mooncake 有效带宽和调用次数、CPU/NPU 利用率。建议以“cold put 尾延迟或端到端 prefill latency
至少改善 5%，且 input throughput 与 p99 TTFT 回退不超过 3%”作为初始收益门槛；最终门槛应由真实业务
流量确认。

当前验证阶段将逐层诊断日志临时提升为 `INFO`，功能验证无需开启 DEBUG。正式性能 A/B 前必须删除这些日志或
降回 `DEBUG`，并使用相同的 INFO 日志配置测试两组；否则逐层日志会放大 Python I/O 和调度开销，不能用于判断
layerwise 是否有收益。当前实现不在逐层热路径新增同步计时或 `tensor.item()`，避免为了测量引入 NPU 同步和
额外性能扰动。
