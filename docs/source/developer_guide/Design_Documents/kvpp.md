# KVPP 代码讲解文档

本文按照本分支的 KVPP 实现组织，代码路径均相对于仓库根目录。

KVPP 将当前 PP Stage 的 Target Layer KV Cache 分配给 TP 组内的 Owner Rank 持久保存，其他 Rank 复用两块 Scratch Buffer。计算前按层广播整个连续 Bundle，提前预取下一层。逻辑 Block 编号和调度器的 Cache Spec 保持完整。

适用范围为 Eager 执行下的 MLA/SFA 模型，接入 Model Runner V1 和 V2；可叠加 TP、EP、PP、Chunked Prefill、Prefix Caching、异步调度和固定步数 MTP。MTP Cache 独立保存，不参与 Target Layer 的 Owner 划分和广播。LI-C8 与 SFA-C8 按各自 Cache Spec 构造物理分量。图模式、PCP、DCP、PD 分离及变步数 MTP 不在本特性的支持范围。

目录：

```text
0. 全文统一示例假设
1. KVPP 流程总览
   1.1 开工阶段主调用栈
   1.2 运行阶段主调用栈
2. 开工阶段详解
   2.1 KVPP-1：初始化 KVPP Group
   2.2 KVPP-2：构建 Physical KV Cache Plan 与 Block 预算
   2.3 KVPP-3：分配连续 KV Cache 并绑定 Layer View
   2.4 KVPP-4：初始化 KVPP Runtime
3. 运行阶段详解
   3.1 prepare_forward：判断本轮是否有历史 KV
   3.2 wait_for_layer：Layer Forward 中的 KV Ready
   3.3 通算掩盖
4. 函数讲解（按文件组织）
   4.1 core/kv_cache_placement.py
   4.2 worker/kvpp_cache.py
   4.3 worker/v2/kvpp.py
   4.4 distributed/kv_transfer/kv_pool/broadcast_transport.py
   4.5 配置、Worker 与 Model Runner 接入
   4.6 文件之间的关系
5. 用例设计
```

## 0. 全文统一示例假设

```text
Model                : GLM-5.2
Target Layer         : 78
MTP Layer            : 1
TP                   : 8
KVPP Size            : 8
PP                   : 1
MTP                  : ON
Execution            : Eager
Block Size           : 128 Tokens
KVPP Rank            : 0（没有特别说明时）
```

开启方式：

```text
--enforce-eager
--additional-config '{"enable_kvpp": true}'
```

以下以 GLM-5.2 中带 Indexer 的 SFA Layer 为例。复用 TopK 的层可以没有 Indexer，Bundle 以 `worker_spec` 中实际存在的 Cache Entry 为准。带 Indexer 的一层可以简化理解为：

```text
Layer N
├── Main KV Cache
│   └── model.layers.N.self_attn.attn
│
└── Indexer KV Cache
    └── model.layers.N.self_attn.indexer.k_cache
```

这两个名字对应两个 Logical Cache Entry，它们一起组成 Layer N 的 Cache Bundle。

```text
一个 Transformer Layer
        ↓
一个 Cache Bundle
        ↓
Main KV 的全部 Tensor Component
+
Indexer KV 的全部 Tensor Component
```

这里区分三个层次：

- **Layer**：模型实际执行的 Transformer Layer。
- **Cache Entry**：以 Layer Name 为键的缓存项，例如 Main KV 或 Indexer KV。
- **Tensor Component**：某个 Cache Entry 内的实际分量，例如 K、RoPE K、Indexer Data、Indexer Scale。

Component 的数量和字节布局由 Cache Spec、Attention 维度及量化配置共同确定。各 View 使用的 dtype 由 Model Runner 按对应布局设置。Main KV 和 Indexer KV 的逻辑名字各自保留，物理内存按整个 Bundle 连续组织。

`worker_spec` 表示当前 Worker 所需的完整逻辑缓存信息：

```text
worker_spec
│
├── Target Model KV Cache
│   ├── Layer 0
│   │   ├── Main KV
│   │   └── Indexer KV
│   ├── ...
│   └── Layer 77
│       ├── Main KV
│       └── Indexer KV
│
└── MTP KV Cache
    └── 本 Worker 上 MTP Layer 对应的全部 Cache Entry
```

全文中，`num_blocks` 表示每个 Cache Component 包含的缓存 Block 数。一个 Block 对应 128 个 Token；每个 Component 的一页字节数根据它自己的 dtype 和维度计算。

Target Layer 的 Owner 表示：**哪一个 KVPP Rank 长期保存这一层的完整 KV Cache**。所有 Rank 仍执行本 PP Stage 的模型层计算。

## 1. KVPP 流程总览

### 1.1 开工阶段主调用栈

```text
服务启动
│
├── 初始化 Ascend 配置
│   └── KVPPConfig.from_vllm_config()
│       enable_kvpp → KVPP Size
│
├── init_ascend_model_parallel()
│   └── ★ [KVPP 插入点 1]
│       初始化 KVPP communication group
│
├── NPUWorker.get_kv_cache_spec()
│   ├── model_runner.get_kv_cache_spec()
│   └── ★ [KVPP 插入点 2]
│       create_kvpp_cache_allocation_plan()
│       保存 Owner / Bundle / Component 字节数
│       向 Engine 返回完整 Logical KV Cache Spec
│
├── NPUWorker.determine_available_memory()
│   └── _apply_kvpp_memory_budget()
│       物理显存预算 → 可容纳的 Block 数 → Logical Planner 字节预算
│
├── Engine 生成 KVCacheConfig
│   └── 确定共同使用的 num_blocks 和 Logical Cache Groups
│
└── NPUWorker.initialize_from_config()
    └── model_runner.initialize_kv_cache()
        ├── ★ [KVPP 插入点 3]
        │   allocate_kvpp_cache()
        │   分配 Persistent / Scratch / MTP
        │   reshape 并绑定各 Layer 的 KV View
        │
        └── ★ [KVPP 插入点 4]
            KVPPRuntime.create_from_kv_cache()
            创建 Broadcast Transport / Scheduler / Attention Hook
```

这个流程中有两份需要一起理解的信息：

```text
Logical Cache Spec
    描述模型有哪些层、每层使用什么缓存

KVPPPhysicalCachePlan
    描述当前 Rank 持有哪些层、各层的物理字节布局和预算
```

Engine 使用完整逻辑信息完成缓存规划；Worker 根据同一份 KVPP Plan 计算物理成本和实际分配。

### 1.2 运行阶段主调用栈

Model Runner V1 的主路径：

```text
NPUModelRunner.execute_model()
│
├── 准备输入和 Attention Metadata
│
├── ★ kvpp.prepare_forward(has_history)
│   └── 有历史 KV 时，启动第一层 Prefetch
│
├── Model Forward
│   ├── Target Layer 0 → Attention → wait_for_layer(layer_0)
│   ├── Target Layer 1 → Attention → wait_for_layer(layer_1)
│   ├── ...
│   └── Target Layer 77 → Attention → wait_for_layer(layer_77)
│
└── ★ kvpp.complete_forward()
    重置本轮调度状态
```

Model Runner V2 的主路径：

```text
NPUModelRunner.execute_model()
│
├── 设置 model_state.kvpp_is_dummy_run
│
├── super().execute_model()
│   ├── AscendModelState.prepare_attn()
│   │   └── ★ kvpp_runtime.prepare_forward(has_history)
│   │
│   └── Model Forward
│       └── 每个 Target Attention 调用 wait_for_layer(layer_name)
│
├── 复位 kvpp_is_dummy_run
└── ★ kvpp.complete_forward()
```

每一层的核心顺序是：

```text
当前层的前置计算
        ↓
等待当前层 KV Broadcast 完成
        ↓
启动下一层 KV Broadcast
        ↓
当前层写入本轮 KV，并执行 Attention
```

## 2. 开工阶段详解

### 2.1 KVPP-1：初始化 KVPP Group

开启 KVPP 后：

```text
KVPP Size = TP Size
```

在全文示例中：

```text
TP Group  : [0, 1, 2, 3, 4, 5, 6, 7]
KVPP Group: [0, 1, 2, 3, 4, 5, 6, 7]
```

两个 Group 的 Rank 成员相同，KVPP 使用自己的 `GroupCoordinator` 和 `device_group`。

创建代码位于 `init_ascend_model_parallel()`：

```python
kvpp_group_ranks = all_ranks.reshape(-1, kvpp_size).unbind(0)
_KVPP = init_model_parallel_group(
    [ranks.tolist() for ranks in kvpp_group_ranks],
    get_world_group().local_rank,
    backend,
    group_name="kvpp",
)
```

后续通过下面的入口获取：

```python
get_kvpp_group()
```

一个 Group 中常用的字段是：

```text
rank_in_group
    当前 Worker 在 KVPP Group 内的编号

ranks
    Group 内各 Rank 对应的 Global Rank

device_group
    torch.distributed.broadcast 使用的设备通信组
```

Owner Mapping 保存的是 Group 内编号。Transport 通过 `group.ranks[owner]` 找到 Broadcast 的 Global Source Rank。

使用 PP 时，每个 PP Stage 根据自己本地的 Target Layer 集合分配 Owner，KVPP Group 对应该 Stage 内的 TP Rank。MTP 是否出现在某个 Stage，以这个 Worker 实际提供的 Cache Layer Name 为准。

### 2.2 KVPP-2：构建 Physical KV Cache Plan 与 Block 预算

#### 2.2.1 详细调用链

```text
Engine 初始化 KV Cache
│
├── model_executor.get_kv_cache_specs()
│   └── collective_rpc("get_kv_cache_spec")
│       └── NPUWorker.get_kv_cache_spec()
│           ├── model_runner.get_kv_cache_spec()
│           │   得到本 Worker 的完整 Logical Spec
│           │
│           ├── ★ create_kvpp_cache_allocation_plan()
│           │   保存到 self._kvpp_cache_allocation_plan
│           │
│           └── 返回完整 Logical Spec
│
├── model_executor.determine_available_memory()
│   └── NPUWorker.determine_available_memory()
│       ├── 获得可用于 KV Cache 的物理内存预算
│       └── ★ _apply_kvpp_memory_budget()
│           ├── plan.get_num_blocks(available_bytes)
│           └── Block 数 × Logical 每 Block 总字节数
│
├── Engine 按 Logical Spec 和 Planner 字节预算计算 KVCacheConfig
│   └── 确定各 Worker 共同使用的 num_blocks
│
└── model_executor.initialize_from_config(kv_cache_configs)
    将配置发给各个 Worker
```

#### 2.2.2 Logical Spec 与 Physical Plan

`NPUWorker.get_kv_cache_spec()` 返回的逻辑信息包括本 Worker 的全部 Target 和 MTP Cache Entry。

以 Rank 0 为例：

```python
{
    "model.layers.0.self_attn.attn": AscendMLAAttentionSpec(...),
    "model.layers.0.self_attn.indexer.k_cache": AscendSFAIndexerCacheSpec(...),
    # ...
    "model.layers.77.self_attn.attn": AscendMLAAttentionSpec(...),
    "model.layers.77.self_attn.indexer.k_cache": AscendSFAIndexerCacheSpec(...),
    "model.layers.78.mtp_block.self_attn.attn": ...,  # MTP Cache Spec
}
```

`create_kvpp_cache_allocation_plan()` 根据这些信息建立：

```python
KVPPPhysicalCachePlan(
    logical_cache_spec=...,
    layer_owner_ranks=...,
    layer_bundles=...,
    tensor_sizes=...,
    kvpp_rank=0,
)
```

各字段串起来的关系是：

```text
logical_cache_spec
    Cache Name → Cache Spec
        │
        ├── layer_owner_ranks
        │   Target Cache Name → Owner KVPP Rank
        │
        ├── layer_bundles
        │   Attention Layer Name → 本层所有 Cache Name
        │
        └── tensor_sizes
            Cache Name → 各 Component 的每 Block 字节数
```

Plan 本身只保存规划信息。实际 Tensor 内存由 `allocate_kvpp_cache()` 申请。

#### 2.2.3 Owner 与 Rank 0 的物理组成

78 个 Target Layer 分给 8 个 KVPP Rank：

```text
78 // 8 = 9
78 % 8  = 6
```

前 6 个 Rank 各分到 10 层，其余 2 个 Rank 各分到 9 层：

```text
Rank 0 → Layer 0  ~ 9
Rank 1 → Layer 10 ~ 19
Rank 2 → Layer 20 ~ 29
Rank 3 → Layer 30 ~ 39
Rank 4 → Layer 40 ~ 49
Rank 5 → Layer 50 ~ 59
Rank 6 → Layer 60 ~ 68
Rank 7 → Layer 69 ~ 77
```

同一层的 Main KV 和 Indexer KV 使用同一个 Owner：

```python
{
    "model.layers.10.self_attn.attn": 1,
    "model.layers.10.self_attn.indexer.k_cache": 1,
}
```

Rank 0 的物理内存组成：

```text
Rank 0
│
├── Persistent Target Bundle
│   ├── Layer 0：Main Components + Indexer Components
│   ├── ...
│   └── Layer 9：Main Components + Indexer Components
│
├── Scratch 0：一整层 Bundle 的连续空间
├── Scratch 1：一整层 Bundle 的连续空间
│
└── Persistent MTP Bundle
    └── 本 Worker 的 MTP Components
```

每块 Scratch 都能容纳当前 PP Stage 最大的 Target Bundle。MTP 有自己的持久空间，大小由 MTP 的 Cache Spec 决定。

#### 2.2.4 Block 数怎样计算

设：

```text
B_i = Target Layer i 的一个完整 Bundle Block 的物理字节数
M   = 当前 Worker 所有 MTP Bundle 每 Block 的字节数之和
```

Rank 0 的每 Block 物理成本为：

```text
Persistent = B_0 + B_1 + ... + B_9 + M
Scratch    = 2 × max(B_0, B_1, ..., B_77)

bytes_per_block = Persistent + Scratch
```

因此：

```text
num_blocks = available_bytes // bytes_per_block
```

`get_num_blocks()` 使用 `build_kvpp_layer_layout(..., num_blocks=1)` 得到每层一个 Block 的完整布局大小，再累加当前 Rank 的持久缓存和两份 Scratch 成本。

各 Component 的长度等于每 Block 字节数乘以 `num_blocks`。Bundle 按这些长度连续拼接，因此分配成本随 Block 数线性增长。

接下来 Worker 把 Block 数换算成 Engine 使用的逻辑预算：

```python
num_blocks = plan.get_num_blocks(available_bytes)
return num_blocks * sum(
    spec.page_size_bytes for spec in plan.logical_cache_spec.values()
)
```

这两种字节数分别表示：

```text
available_bytes
    当前 Rank 可用于实际 KV Cache 分配的物理预算

返回给 Engine 的字节数
    让完整 Logical Spec 对应同一个 num_blocks 的 Planner 预算
```

为便于算数，这个预算例子取：

```text
每个 Target Bundle Block = 1 MiB
MTP Bundle Block 总量    = 1 MiB
Rank 0 可用物理预算      = 1300 MiB
```

于是：

```text
Rank 0 每 Block 物理成本
    = 10 × 1 MiB + 2 × 1 MiB + 1 MiB
    = 13 MiB

Rank 0 可容纳的 Block 数
    = 1300 MiB // 13 MiB
    = 100

完整 Logical Spec 每 Block 成本
    = 78 × 1 MiB + 1 MiB
    = 79 MiB

返回给 Engine 的 Planner 预算
    = 100 × 79 MiB
    = 7900 MiB
```

Engine 在完整逻辑拓扑下得到相应的 Block 容量，并将各 Worker 的容量约束统一到最终配置中。Worker 使用最终 `num_blocks` 分配实际的 Persistent、Scratch 和 MTP。

这里的 7900 MiB 是规划器输入，Rank 0 的实际 KVPP 分配按每 Block 13 MiB 计算。

### 2.3 KVPP-3：分配连续 KV Cache 并绑定 Layer View

#### 2.3.1 详细调用链

```text
NPUWorker.initialize_from_config(kv_cache_config)
│
└── model_runner.initialize_kv_cache()
    │
    ├── 初始化 Attention Backend / Metadata Builder
    │
    ├── 初始化 KV Cache Tensor
    │   ├── V1：_allocate_kv_cache_tensors()
    │   └── V2：allocate_kv_cache_main() → Ascend _allocate_kv_cache()
    │       │
    │       └── ★ allocate_kvpp_cache()
    │           ├── get_kvpp_cache_specs()
    │           ├── create_kvpp_cache_allocation_plan()
    │           ├── build_kvpp_layer_layout()
    │           ├── 申请两块 Scratch
    │           └── 逐层申请 Persistent / MTP 或取得 Scratch View
    │
    ├── 按 Attention Backend 需要的 dtype / shape reshape
    └── 将 KV View 绑定到对应的 Layer
```

Engine 下发的配置保留完整的 Logical Layer Name。`allocate_kvpp_cache()` 为这些名字返回对应的 Tensor View。

#### 2.3.2 一个 Bundle 的连续布局

假设某层包含三个 Component：

```text
Main KV
├── K Component
└── RoPE K Component

Indexer KV
└── Indexer Data Component
```

本例取每个 Block 的 Component 大小：

```text
K Component        : 128 KiB
RoPE K Component   : 16 KiB
Indexer Component  : 32 KiB
```

这些数值用于展开字节布局，实际执行时从 `tensor_sizes` 读取。

假设 `num_blocks=32`：

```text
K Component 总大小       = 128 KiB × 32 = 4096 KiB
RoPE K Component 总大小  = 16 KiB  × 32 = 512 KiB
Indexer Component 总大小 = 32 KiB  × 32 = 1024 KiB

Bundle 总大小 = 5632 KiB
```

布局如下：

```text
Bundle Byte Storage
│
├── [0, 4096 KiB)
│   K Component，包含全部 32 个 Block
│
├── [4096 KiB, 4608 KiB)
│   RoPE K Component，包含全部 32 个 Block
│
└── [4608 KiB, 5632 KiB)
    Indexer Component，包含全部 32 个 Block
```

Component 在 Bundle 中按顺序连续排列，后一项的起点就是前一项的末端。`build_kvpp_layer_layout()` 返回的 Offset 相对于该 Bundle 的起点。

对于这个例子：

```text
Main KV Layout:
    (offset=0,             length=4096 KiB)
    (offset=4096 KiB,      length=512 KiB)

Indexer KV Layout:
    (offset=4608 KiB,      length=1024 KiB)

Layer Span:
    5632 KiB
```

这是“每个 Component 存放全部 Block，然后排列各 Component”的布局。

#### 2.3.3 Persistent、Scratch 与 MTP 的选择

`allocate_kvpp_cache()` 遍历按执行顺序排列的 Bundle：

```text
当前 Bundle
│
├── 属于本 Rank 的 Target Layer
│   └── 申请该层自己的完整 Byte Storage
│
├── 属于其他 Rank 的 Target Layer
│   └── 使用 scratch[target_index % KVPP_SCRATCH_BUFFER_COUNT]
│
└── MTP Layer
    └── 申请该层自己的完整 Byte Storage
```

`target_index` 是本 PP Stage 的 Target Layer 执行序号：

```text
遍历一个 Target Bundle → target_index 加 1
遍历一个 MTP Bundle    → target_index 不变
```

本 Rank 持有的 Target Layer 也计入这个序号。`KVPP_SCRATCH_BUFFER_COUNT` 定义为 `2`，分配器创建相同数量的 Scratch，再用这个常量对执行序号取模。

在 Rank 0 上：

```text
Layer 0  ~ 9  → 各层自己的 Persistent Storage
Layer 10      → Scratch 0
Layer 11      → Scratch 1
Layer 12      → Scratch 0
Layer 13      → Scratch 1
...
Layer 76      → Scratch 0
Layer 77      → Scratch 1
MTP Layer 78  → 自己的 Persistent Storage
```

在 Rank 7 上，Layer 69 是自己持有的第一层，但它仍对应 Target 序号 69。后续序号继续按完整 Target 执行顺序推进。

#### 2.3.4 Cache Entry 怎样引用 Bundle

分配器通过：

```python
buffer.narrow(0, offset, length)
```

得到每个 Component 的 Byte View。

返回结果的结构是：

```python
{
    "model.layers.10.self_attn.attn": (
        main_k_byte_view,
        main_rope_byte_view,
    ),
    "model.layers.10.self_attn.indexer.k_cache": (
        indexer_byte_view,
    ),
    # ...
}
```

这些 Byte View 再由 Model Runner reshape 成 Attention 所需的 dtype 和 shape。

对于 Rank 0 的 Layer 10：

```text
Scratch 0 的连续 Byte Storage
│
├── Main K View
├── Main RoPE K View
└── Indexer View
        ↓
绑定到 Layer 10 的 Main / Indexer Cache Entry
```

Layer 12、14 等也从 Scratch 0 取得自己的 View。每层 View 的 Offset 和 Length 按该层 Bundle Layout 计算，Scratch 的总容量足以容纳其中最大的 Target Bundle。

因此：

```text
完整 Logical Layer Name
        ↓
该层的 Component Views
        ↓
本层 Persistent Storage 或共享 Scratch Storage
```

Runtime 后续构建的 Broadcast Buffer，也引用这些 Tensor 的同一块底层 Storage。

### 2.4 KVPP-4：初始化 KVPP Runtime

KV Cache 完成分配、reshape 和绑定后，创建 Runtime：

```text
KVPPRuntime.create_from_kv_cache()
│
├── 读取 KVPPConfig
├── 获取各层已绑定的 KV Tensor
├── 构建 KVPP Plan
│
├── 对每个 Target Bundle
│   ├── 取得第一个 Component 的底层 Storage
│   ├── 计算 Bundle 起点和完整 Layer Span
│   └── 创建一个覆盖整层 Bundle 的 raw Byte View
│
├── 创建 BroadcastKVPPTransport
├── 创建 KVPPScheduler
└── impl.layerwise_kv_cache_hook = scheduler
```

V1 直接传入缓存初始化得到的 `kv_caches`。V2 由 Runtime 从 `static_forward_context[name].kv_cache` 收集已绑定的缓存。

每个 Target Bundle 对应一个 Broadcast Buffer。

```text
Layer 10
    → 一个覆盖完整 Bundle 的 raw Byte View
    → 一次 Broadcast
```

这个 Byte View 覆盖该层全部 Component 连续组成的完整 Span。每层 Span 等于各 Component 长度之和，Scratch 的剩余尾部空间不属于这层的广播范围。

初始化内容与运行阶段的对应关系：

```text
layer_owner_ranks
    Layer → Owner KVPP Rank → Global Source Rank

layer_buffers
    Layer → 当前 Rank 上覆盖完整 Persistent / Scratch Bundle 的一个 Byte View

attention_layer_names
    当前 PP Stage 中 Target Attention 的执行顺序

KVPPScheduler
    prepare_forward → Prefetch → wait_for_layer → complete_forward

BroadcastKVPPTransport
    Owner Persistent → 其他 Rank 的 Scratch

Attention Hook
    Attention 调用 wait_for_layer() 后继续读写本层 KV
```

Runtime 为 Target Attention 绑定 Hook。MTP Cache 保持本地持久分配，MTP Layer 不进入 Target Prefetch 序列。

## 3. 运行阶段详解

运行阶段主要解决三个问题：

```text
prepare_forward()
    当前 Batch 是否有需要读取的历史 KV

wait_for_layer()
    当前 Layer 在什么位置等待完整缓存准备好

通算掩盖
    下一层 Broadcast 如何与当前层计算重叠
```

### 3.1 prepare_forward：判断本轮是否有历史 KV

#### 3.1.1 `has_history` 从哪里来

Model Runner V1 使用当前实际请求的 Host 侧已计算 Token 数：

```python
has_history = bool(
    np.any(self.input_batch.num_computed_tokens_cpu[:num_reqs] > 0)
)
self.kvpp.prepare_forward(has_history)
```

Model Runner V2 在 `AscendModelState.prepare_attn()` 中使用：

```python
has_history = not self.kvpp_is_dummy_run and bool(
    np.any(input_batch.num_computed_tokens_np[:num_actual_reqs] > 0)
)
self.kvpp_runtime.prepare_forward(has_history)
```

这里的 `num_computed_tokens` 表示进入本轮 Forward 时，请求已经拥有的 KV 历史长度。

例如：

```text
Batch A
num_computed_tokens = [0, 0, 0]
has_history         = False

Batch B
num_computed_tokens = [256, 0, 128]
has_history         = True
```

Batch A 中每个请求都从当前输入开始生成 KV。Batch B 中至少一个请求需要历史 KV，因此这一轮会按 Target Layer 顺序执行 Broadcast。

常见场景：

```text
首个 Prefill Chunk，所有请求 computed=0
    → has_history=False

后续 Prefill Chunk
    → 已有前面 Chunk 的 KV，has_history=True

Decode
    → 请求已有上下文 KV，has_history=True

Prefix Cache 命中
    → 命中前缀计入 computed，存在历史时 has_history=True

混合 Batch
    → 任一实际请求 computed>0，整个 Forward 启用 Broadcast
```

V1 的 Dummy Forward 显式传入 `False`。V2 在执行入口用 `dummy_run or is_profile` 设置 `kvpp_is_dummy_run`，使相应 Forward 的 `has_history=False`。

#### 3.1.2 一次 Forward 怎样启动

`KVPPRuntime.prepare_forward()` 把布尔值交给 Scheduler：

```text
prepare_forward(has_history)
        ↓
schedule_forward(has_history)
        ↓
保存 _has_history
重置 _next_attention_layer_index = 0
        ↓
has_history=True 时
        ↓
start_layer_prefetch(attention_layer_names[0])
```

`has_history=False` 时，每个 Attention Hook 直接返回。所有 Rank 为当前输入计算 KV，Owner 写入自己的持久缓存，非 Owner 写入本层的 Scratch。

`has_history=True` 时，所有 KVPP Rank 以相同层顺序参与 Collective。即使当前层的 Owner 就是本 Rank，本 Rank 也会提交该层 Broadcast。

#### 3.1.3 一层 Broadcast 具体搬运什么

假设：

```text
num_blocks = 32
Layer 10 Owner = Rank 1
```

对于 Layer 10：

```text
Rank 1 Persistent Bundle
├── Main Components：Block 0 ~ 31
└── Indexer Components：Block 0 ~ 31
                │
                │ Broadcast
                ▼
其他 Rank 对应的 Layer 10 Scratch Views
├── Main Components：Block 0 ~ 31
└── Indexer Components：Block 0 ~ 31
```

传输大小取决于 `num_blocks` 和该层的 Component Layout。当前 Batch 的请求数、历史长度或使用了几个 Page，不改变这层预先建立的 Broadcast Byte View。

Attention 仍通过 Block Table 和 Slot Mapping 定位请求所用的位置：

```text
Block Table
    决定 Attention 从哪些 Physical Block 读取历史

Slot Mapping
    决定本轮 Token 的 KV 写入哪个位置

Broadcast
    按同一 Component 的相同字节位置复制全部 Block
```

例如某请求通过 Block Table 使用 `[7, 2, 11]`，Attention 就从本层缓存的这三个 Block 读取。整层 Broadcast 完成后，Owner 的 Block 7 对应接收端的 Block 7，其余 Block 同样按位置对应。

### 3.2 wait_for_layer：Layer Forward 中的 KV Ready

初始化阶段已经建立：

```text
Target Attention Impl
└── layerwise_kv_cache_hook
    └── KVPPScheduler
```

执行到 Hook 位置时：

```python
self.layerwise_kv_cache_hook.wait_for_layer(layer_name)
```

#### 3.2.1 一层 SFA Attention 正常怎么执行

以 SFA 的 Native 路径展开：

```text
SFA Attention Forward
│
├── 准备当前输入 Hidden States
├── fused_qkv_a_proj(hidden_states)
│   └── q_c + kv_no_split
├── q_a_layernorm(q_c)
├── has_indexer=True 时，执行 indexer_select_pre_process()
│   └── 当前输入的 Indexer Projection / Norm / RoPE 等准备
│
├── ★ wait_for_layer(layer_name)
│
├── exec_kv()
│   └── Main KV 处理与 Cache 写入相关操作
├── Q Projection / RoPE
├── Main Cache 写入
├── has_indexer=True 时，写入 Indexer Cache
├── 取得 TopK
│   ├── skip_topk=True：读取缓存的 TopK
│   └── skip_topk=False：执行 Indexer 后处理并计算 TopK
├── Sparse Flash Attention
│   └── 使用本层 Main KV Cache
├── V Up Projection
└── Output Projection
```

SFA 的 Fused 路径在 `_sfa_preprocess_prolog_v3()` 或 `_sfa_preprocess_mlapo()` 前等待；存在 Indexer 时，先执行 `indexer_select_pre_process()`，再到达这个等待点。

MLA 也在进入缓存读写路径前等待：

```text
普通 MLA 路径
    _mla_preprocess()
    Q/KV Projection 与 Norm
        ↓
    wait_for_layer()
        ↓
    mla_preprocess_decode() / mla_preprocess_prefill()

Decode Fused / MLAPO 路径
    wait_for_layer()
        ↓
    mla_preprocess_only_decode()
```

这些 Hook 位于不同执行分支中。一次 Attention 按实际选中的分支执行。

#### 3.2.2 为什么 wait 放在这个位置

Hook 前的投影、Norm、Indexer 前处理主要使用当前输入，可以与当前层尚未结束的 KV Broadcast 重叠。

到达 Hook 时，需要保证：

```text
当前层的完整 Main KV 与 Indexer KV 已经准备好
Owner 的 Broadcast Source 读取已经完成
接收端的 Scratch 写入已经完成
```

然后才能：

```text
写入本轮 Token 的 KV
        ↓
读取本轮和历史 KV
        ↓
执行 Attention
```

Owner 也需要等待。它接下来会更新持久缓存中的当前 Token 位置，Broadcast 对该缓存的读取必须先完成。

#### 3.2.3 wait_for_layer 做什么

```text
wait_for_layer(N)
│
├── 本轮没有历史 KV → 返回
│
├── 等待 self._prefetch_future.result()
│   └── Layer N 的设备传输已经完成
│
├── 清空已消费的 _prefetch_future
├── _next_attention_layer_index 加 1
│
└── 还有下一层时
    └── start_layer_prefetch(N+1)
```

`layer_name` 是 Attention Hook 的调用参数，调度推进使用 `attention_layer_names` 和内部序号。每个 Target Attention 按执行顺序消费自己的 Prefetch Future。

最后一个 Target Layer 的 Hook 消费最后一次 Future，随后继续执行该层 Attention。Forward 结束后：

```python
self.kvpp.complete_forward()
```

Scheduler 执行的状态更新是：

```python
self._has_history = False
self._next_attention_layer_index = 0
```

当前层真正需要的传输等待发生在 `wait_for_layer()`。

### 3.3 通算掩盖

#### 3.3.1 Layer N 与 Layer N+1

`wait_for_layer(N)` 等到当前层的 KV Ready 后，立即提交下一层 Prefetch，然后返回 Attention。

```text
Compute Stream                         Transfer Stream

Layer N 前置计算                       Layer N Broadcast
        │                                      │
        └── wait_for_layer(N) ◄─────────────────┘
                │
                ├── 提交 Layer N+1 Prefetch ───────────┐
                │                                     │
Layer N KV Write / Attention                   Layer N+1 Broadcast
Layer N Output / Residual / MoE                         │
Layer N+1 前置计算                                      │
                │                                     │
                └── wait_for_layer(N+1) ◄───────────────┘
```

Layer N+1 Broadcast 可与 Layer N Hook 后的计算，以及 Layer N+1 Hook 前的准备工作重叠。到达下一处 Hook 时，Host 等待该层 Future 剩余的完成时间。

#### 3.3.2 两块 Scratch 怎样交替

在 Rank 0 上：

```text
Layer 10 → Scratch 0
Layer 11 → Scratch 1
Layer 12 → Scratch 0
```

因此可以形成：

```text
Layer 10 Attention 使用 Scratch 0
                ||
Layer 11 Broadcast 写入 Scratch 1
```

准备 Layer 12 时，要等 Scratch 0 的上一位使用者 Layer 10 完成。

`start_layer_prefetch()` 在调用它的当前 Compute Stream 上记录：

```python
cache_ready = torch.npu.Event()
cache_ready.record(torch.npu.current_stream())
```

启动 Layer 12 Prefetch 的位置在 Layer 11 的 Hook 内。此时 Layer 10 的计算已经提交到 Compute Stream；`cache_ready` 排在这些操作之后。

```text
Compute Stream
    Layer 10 使用 Scratch 0
        ↓
    Layer 11 前置计算
        ↓
    记录 Layer 12 的 cache_ready
        ↓
    Layer 11 使用 Scratch 1

Transfer Stream
    等待 Layer 12 的 cache_ready
        ↓
    把 Layer 12 KV 写入 Scratch 0
```

两条流通过 Event 建立依赖。CPU 记录 Event 时，前面的设备操作可以仍在执行。

这个 Event 同样覆盖该 Rank 的 Persistent Cache 先前写入。第一次 Prefetch 在 Forward 准备阶段记录 Event，后续层在各自的启动位置记录 Event。

#### 3.3.3 Future 为什么代表设备完成

后台线程执行：

```text
run_layer_prefetch()
        ↓
BroadcastKVPPTransport.prefetch()
        ↓
切换到 Transfer Stream
        ↓
transfer_stream.wait_event(cache_ready)
        ↓
取得本层完整 raw Byte View，提交一次 dist.broadcast(async_op=True)
        ↓
对本层 Work 调用 wait()
        ↓
在 Transfer Stream 记录 done Event
        ↓
done.synchronize()
        ↓
后台任务结束，Future 完成
```

这样，当 Attention 的 `Future.result()` 返回时，该层 Broadcast 已完成设备侧读写。

```text
cache_ready
    约束什么时候允许开始读写本层通信 Buffer

done
    标记本层通信已经完成设备侧读写

Future
    把这次后台传输的完成状态交给 Attention Hook
```

Scheduler 使用一个后台线程按层提交通信，并使用独立的 NPU Transfer Stream。相邻层的计算和通信通过各自的流推进，在缓存读写边界建立必要的等待关系。

## 4. 函数讲解（按文件组织）

这一章作为代码阅读辅助：

- 数据类：说明保存什么。
- 重要类：说明整体职责。
- 重点函数：按“功能 / 入参 / 出参 / 处理过程 + 例子”说明。
- 简单入口：说明它连接的上下游以及传递的数据。

---

### 4.1 `vllm_ascend/core/kv_cache_placement.py`

这个文件负责 Layer Placement、Component 字节布局和 Block 容量计算。

#### 4.1.1 `KVPPPhysicalCachePlan` 与 Component 字节信息

`tensor_sizes` 用整数记录每个 Tensor Component 的单 Block 大小：

```python
tensor_sizes: dict[str, tuple[int, ...]] = {
    "model.layers.10.self_attn.attn": (131072, 16384),
    "model.layers.10.self_attn.indexer.k_cache": (32768,),
}
```

分别表示：

```text
Main K Component
    每 Block 为 131072 Byte

Main RoPE Component
    每 Block 为 16384 Byte

Indexer Component
    每 Block 为 32768 Byte
```

Tuple 中每个整数的单位都是 Byte。计算整段长度时直接乘 `num_blocks`，各分量按顺序连续拼接。实际 Storage 使用 `int8`；Attention View 的最终 dtype 由 Model Runner 的 reshape 和量化配置确定。

`KVPPPhysicalCachePlan` 保存当前 Worker 的规划信息：

```python
KVPPPhysicalCachePlan(
    logical_cache_spec=...,
    layer_owner_ranks=...,
    layer_bundles=...,
    tensor_sizes=...,
    kvpp_rank=0,
)
```

例如 Rank 0：

```text
logical_cache_spec
    Target Layer 0 ~ 77 + 本 Worker 的 MTP Cache Entry

layer_owner_ranks
    Target 0 ~ 9  → Rank 0
    Target 10 ~ 19 → Rank 1
    ...

layer_bundles
    Layer 10 Attention Name → (Layer 10 Main Name, Layer 10 Indexer Name)

tensor_sizes
    Layer 10 Main Name → (Main K 每 Block 字节数, Main RoPE 每 Block 字节数)
    Layer 10 Indexer Name → (Indexer 每 Block 字节数, ...)

kvpp_rank
    0
```

MTP 保存在 `logical_cache_spec`、`layer_bundles` 和 `tensor_sizes` 中。`layer_owner_ranks` 只描述 Target 的分布式持有关系。

这个文件还定义 Scratch 数量：

```python
KVPP_SCRATCH_BUFFER_COUNT = 2
```

预算计算、Scratch 分配和 Target 序号取模共同使用这个常量。两块 Scratch 支持当前层计算与下一层 Prefetch 同时推进。

---

#### 4.1.2 `map_kvpp_layers_to_owners()`

**功能**

按 Transformer Layer Index，把本 PP Stage 的 Target Cache Bundle 连续分配给各 KVPP Rank。

**入参**

- `vllm_config`：提供 KVPP Size、MTP 和模型层数配置。
- `local_layer_names`：当前 Worker 的 Cache Layer Name 集合。

**出参**

```python
dict[str, int]
```

表示：

```text
Target Cache Layer Name → Owner KVPP Rank
```

**处理过程 + 例子**

```text
按 Layer Index 和 Name 排序
        ↓
find_mtp_layers() 得到本地 MTP Cache Names
        ↓
将 Target Cache Names 按 Transformer Layer Index 分组
        ↓
divmod(Target Layer 数, KVPP Size)
        ↓
按连续 Layer 区间分给每个 Owner
        ↓
同层的 Main / Indexer Name 填入相同 Owner
```

Layer 10 的两个 Cache Entry 对应：

```python
{
    "model.layers.10.self_attn.attn": 1,
    "model.layers.10.self_attn.indexer.k_cache": 1,
}
```

排序使用的是提取出的数值 Layer Index，所以 Layer 2 会排在 Layer 10 前面。

PP 场景下，输入只包含本 Stage 的层，分区以这个输入集合为范围。MTP 根据实际 Layer Name 和模型配置中的层号区间识别。

---

#### 4.1.3 `build_layer_cache_bundles()`

**功能**

把同一 Transformer Layer 的多个 Cache Entry 组织成一个 Bundle，确定其中的 Component 排列顺序。

**入参**

- `cache_spec`：Cache Layer Name 到 `KVCacheSpec` 的映射。

**出参**

```python
dict[str, tuple[str, ...]]
```

表示：

```text
Bundle 的主 Layer Name → 同层的全部 Cache Names
```

**处理过程 + 例子**

函数先按 Layer Index 排序，同一层中把 Main Cache 排在 `AscendSFAIndexerCacheSpec` 对应的 Indexer Cache 前。

输入：

```text
model.layers.10.self_attn.indexer.k_cache
model.layers.10.self_attn.attn
```

得到：

```python
{
    "model.layers.10.self_attn.attn": (
        "model.layers.10.self_attn.attn",
        "model.layers.10.self_attn.indexer.k_cache",
    ),
}
```

Bundle 的主名字用于调度 Attention；Bundle 中的全部 Cache Name 用于计算连续布局。

---

#### 4.1.4 `build_kvpp_buffer_sizes()`

**功能**

将每个 Logical Cache Spec 展开成它实际包含的 Tensor Component 字节信息。

**入参**

- `vllm_config`：提供 Attention 维度和量化配置。
- `logical_spec`：当前 Worker 的完整 Cache Spec 映射。

**出参**

```python
dict[str, tuple[int, ...]]
```

例如：

```text
Main Name
    → (K 每 Block 字节数, RoPE K 每 Block 字节数)

Indexer Name
    → (Data 每 Block 字节数, Scale 每 Block 字节数)
```

**处理过程 + 例子**

不同 Cache Spec 按以下路径处理。

```text
AscendSFAIndexerCacheSpec
    → Data Component
    → scale_dim > 0 时增加 Scale Component

Packed SFA C8 Main
    → 一个 Packed Component

其他支持的 Attention Cache
    → 根据 K / V 或 MLA K / RoPE 维度计算分量比例
    → 生成对应 Component 的每 Block 字节数
```

Indexer 的计算中，先得到：

```text
E = sfa_dcp_replicated_indexer_size × block_size × num_kv_heads
```

然后：

```text
Data Bytes Per Block
    = E × head_size × sizeof(dtype)

Scale Bytes Per Block
    = E × scale_dim × sizeof(scale_dtype)
```

例如取：

```text
replicated_size = 1
block_size      = 128
num_kv_heads    = 1
head_size       = 128
dtype           = int8
scale_dim       = 1
scale_dtype     = float16
```

得到：

```text
Data  = 1 × 128 × 1 × 128 × 1 = 16384 Byte
Scale = 1 × 128 × 1 × 1 × 2   = 256 Byte
```

于是这个 Indexer Cache 对应的字节数组是 `(16384, 256)`，两个整数分别表示 Data 和 Scale 的单 Block 大小。

普通 MLA Main 根据 `kv_lora_rank` 和 `qk_rope_head_dim` 确定 K 与 RoPE 分量。FA 量化路径使用量化配置提供的 Split Factor，按各分量的字节比例拆分 Page。

函数按 Component 顺序把每 Block 字节数保存为 Tuple，结果写入 Plan 的 `tensor_sizes`。

---

#### 4.1.5 `build_kvpp_layer_layout()`

**功能**

给一个完整 Bundle 的全部 Component 分配相对 Byte Offset，并计算这一层的完整 Span。

**入参**

- `cache_names`：同一 Bundle 内按顺序排列的 Cache Name。
- `tensor_sizes`：每个 Cache Entry 对应的 Component 每 Block 字节数 Tuple。
- `num_blocks`：每个 Component 包含的 Block 数。

**出参**

```python
(
    layout,
    total_size,
)
```

`layout` 的结构是：

```python
dict[str, tuple[tuple[int, int], ...]]
```

内层每个二元组表示：

```text
(offset_bytes, length_bytes)
```

**处理过程 + 例子**

```text
cursor = 0
        ↓
依次遍历 Cache Entry 与 Component
        ↓
length = num_blocks × size_per_block
        ↓
记录 (cursor, length)
        ↓
cursor += length
```

每个 Component 自己保存全部 Block，再顺序排列到 Bundle 中。

假设一个 Bundle 的两个 Cache Entry 为：

```text
main：两个 Component，每 Block 分别 4096 Byte、1024 Byte
indexer：一个 Component，每 Block 2048 Byte
num_blocks=4
```

则：

```python
layout = {
    "main": (
        (0, 16384),
        (16384, 4096),
    ),
    "indexer": (
        (20480, 8192),
    ),
}
total_size = 28672
```

每个 Component 从前一项末端开始。函数返回最后一个 Component 的末端作为 `total_size`，它等于所有 Component 长度之和。

这一个布局函数同时用于：

```text
Block 预算
    num_blocks=1

物理缓存分配
    num_blocks=Engine 最终配置值

Broadcast Byte View 构建
    num_blocks=Engine 最终配置值
```

---

#### 4.1.6 `KVPPPhysicalCachePlan.get_num_blocks()`

**功能**

根据当前 Rank 的实际缓存组成，计算物理预算可容纳多少个 KV Block。

**入参**

- `available_bytes`：当前 Rank 可用于 KV Cache 的物理字节预算。
- Plan 中的 Owner、Bundle、Component 每 Block 字节数和当前 Rank 信息。

**出参**

```python
int
```

表示该 Rank 根据自身预算算出的候选 Block 数。

**处理过程 + 例子**

```text
逐层 build_kvpp_layer_layout(..., num_blocks=1)
        ↓
本 Rank Owner 的 Target + MTP
    → 累加 persistent_bytes
        ↓
全部 Target 中最大 Bundle
    → scratch_bytes
        ↓
bytes_per_block = persistent_bytes + KVPP_SCRATCH_BUFFER_COUNT * scratch_bytes
        ↓
available_bytes // bytes_per_block
```

本例取 Target Bundle 每 Block 为 `P`，MTP 每 Block 总计为 `M`：

```text
Rank 0 ~ 5
    bytes_per_block = 10P + M + 2P = 12P + M

Rank 6 ~ 7
    bytes_per_block = 9P + M + 2P = 11P + M
```

每个 Rank 根据自己的物理预算得到候选容量。Engine 在各 Worker 之间统一最终 `num_blocks`。

如果 Plan 没有任何缓存字节，该函数返回 `0`。

---

#### 4.1.7 `create_kvpp_cache_allocation_plan()`

**功能**

汇总本 Worker 的逻辑拓扑、Layer Owner、Bundle 顺序和 Component 布局信息。

**入参**

- `vllm_config`：模型、并行和量化配置。
- `worker_spec`：当前 Worker 的完整 Cache Spec。
- `kvpp_rank`：当前 Worker 在 KVPP Group 内的编号。

**出参**

```python
KVPPPhysicalCachePlan
```

**处理过程 + 例子**

```text
拷贝 worker_spec 为 logical_spec
        ↓
确认 Full Attention 类型和共同 Block Size
        ↓
map_kvpp_layers_to_owners()
        ↓
build_layer_cache_bundles()
        ↓
build_kvpp_buffer_sizes()
        ↓
保存到 KVPPPhysicalCachePlan
```

Rank 0 的 Plan 既描述完整的 Layer 0～77 与 MTP，也记录其中 Layer 0～9 由本 Rank 持有。

三个阶段都会根据配置取得所需的 Plan：

```text
Worker 上报 Spec 时
    → 用于内存预算

allocate_kvpp_cache() 时
    → 用于确定实际 Allocation 和 View

create_from_kv_cache() 时
    → 用于构建 Broadcast Buffer 和 Hook 顺序
```

这些阶段使用一致的 Layer Name、Owner 分配规则和布局函数。

---

#### 4.1.8 其他函数

`find_mtp_layers()`：在当前 Worker 的 Layer Name 中，查找层号落在下面区间内的 MTP Cache Entry：

```text
[num_hidden_layers, num_hidden_layers + num_nextn_predict_layers)
```

未配置 MTP 时返回空集合。本例中对应 Layer 78。

`get_kvpp_attention_kv_dims()`：根据具体 Cache Spec 和 Attention Layer 获取分量维度。MLA 读取 `kv_lora_rank` 与 `qk_rope_head_dim`；Cache-only Layer 使用 Spec 提供的维度；其他适用 Spec 使用 `head_size` 和 `head_size_v`。

---

### 4.2 `vllm_ascend/worker/kvpp_cache.py`

这个文件连接 Engine 下发的 Logical Cache Config 与 NPU 上实际申请的 Byte Storage。

#### 4.2.1 `get_kvpp_cache_specs()`

**功能**

从 `KVCacheConfig` 中取出每个 Cache Layer 对应的 Spec。

**入参**

- `kv_cache_config`：包含 Logical Cache Groups 的缓存配置。

**出参**

```python
dict[str, KVCacheSpec]
```

**处理过程 + 例子**

```text
遍历 kv_cache_groups
        ↓
遍历 group.layer_names
        ↓
如果 group.kv_cache_spec 是 UniformTypeKVCacheSpecs
    → 从其 kv_cache_specs[name] 取得该层的 Spec

其他 Group Spec
    → 该 Group 内的 Layer 使用这个 Spec
```

例如一个 Uniform Group 中保存 Main 和 Indexer：

```text
Group
├── layer_names
│   ├── Main Name
│   └── Indexer Name
└── kv_cache_spec.kv_cache_specs
    ├── Main Name → AscendMLAAttentionSpec
    └── Indexer Name → AscendSFAIndexerCacheSpec
```

返回结果保留两种具体 Spec，使后续能够计算各自的 Component 布局。

---

#### 4.2.2 `allocate_kvpp_cache()`

**功能**

为本 Rank 持有的 Target Layer 和 MTP 分配连续缓存，为其他 Target Layer 提供交替的 Scratch View。

**入参**

- `vllm_config`：模型与并行配置。
- `kv_cache_config`：完整逻辑缓存配置及最终 `num_blocks`。
- `device`：缓存所在的 NPU Device。

**出参**

```python
dict[str, tuple[torch.Tensor, ...]]
```

每个 Tuple 中是该 Cache Entry 对应的 Component Byte Views。

**处理过程 + 例子**

```text
get_kvpp_cache_specs()
        ↓
create_kvpp_cache_allocation_plan()
        ↓
为每个 Bundle 计算完整 Layout 和 Size
        ↓
取最大 Target Size，申请 Scratch 0 和 Scratch 1
        ↓
按 Bundle 顺序逐层处理
        ↓
本 Rank Target / MTP → torch.zeros(size, dtype=int8)
其他 Target         → scratch[target_index % KVPP_SCRATCH_BUFFER_COUNT]
        ↓
按 Component Offset / Length 创建 narrow() View
        ↓
返回 Cache Name → Component Views
```

例如 Rank 0 处理 Layer 10、11、12：

```text
处理 Layer 10
    target_index = 10
    buffer = scratch[0]
    caches[Main Name] = 这层 Main Components 的 Views
    caches[Indexer Name] = 这层 Indexer Components 的 Views

处理 Layer 11
    target_index = 11
    buffer = scratch[1]

处理 Layer 12
    target_index = 12
    buffer = scratch[0]
```

对于 MTP：

```text
owner = layer_owner_ranks.get(mtp_name)
      = None
        ↓
申请自己的完整 Byte Storage
        ↓
建立 MTP Components 的 Views
```

Scratch 分配使用统一的数量常量：

```python
scratch = [
    torch.zeros(scratch_size, dtype=torch.int8, device=device)
    for _ in range(KVPP_SCRATCH_BUFFER_COUNT)
] if scratch_size else []
```

所有 Byte Storage 使用零初始化。Component Views 随后被 Model Runner 解释为 Attention 所需的类型和形状。

---

### 4.3 `vllm_ascend/worker/v2/kvpp.py`

这个文件包含 V1 和 V2 共用的 Runtime 与逐层 Prefetch Scheduler。

#### 4.3.1 `KVPPRuntime`

Model Runner 面向 KVPP 的入口类，保存：

```text
scheduler
```

运行入口是：

```text
prepare_forward(has_history)
    → scheduler.schedule_forward(has_history)

complete_forward()
    → scheduler.complete_forward()
```

`KVPPRuntime()` 可以表示未启用逐层调度的状态，此时 `scheduler=None`，这两个入口直接结束。

Model Runner 完成真实缓存初始化后，用 `create_from_kv_cache()` 创建与实际 Tensor 绑定的 Runtime。

---

#### 4.3.2 `KVPPRuntime.create_from_kv_cache()`

**功能**

以已绑定的 KV Tensor 为基础，为每个 Target Bundle 创建一个完整 Broadcast Byte View，再创建 Transport、Scheduler，并安装 Attention Hook。

**入参**

- `vllm_config`：KVPP 配置与模型信息。
- `kv_cache_config`：完整 Logical Cache Groups 和最终 Block 数。
- `static_forward_context`：Layer Name 对应的实际 Attention Module。
- `kv_caches`：可选的、已经 reshape 的缓存字典；V1 显式传入，V2 从静态上下文收集。

**出参**

```python
KVPPRuntime
```

KVPP Size 大于 1 且本 Worker 存在 Target Layer 时，返回绑定了 Scheduler 的 Runtime。

**处理过程 + 例子**

```text
读取 config
        ↓
取得 kv_caches
        ↓
建立 Plan
        ↓
逐个 Target Bundle 构建一个完整 raw Byte View
        ↓
创建 BroadcastKVPPTransport
        ↓
创建 KVPPScheduler
        ↓
按 Target 主 Attention Name 绑定 layerwise_kv_cache_hook
```

构造整层 Byte View 时，先从布局取得完整 Bundle 大小，再取得第一个 Component：

```python
_, size = build_kvpp_layer_layout(
    bundle, plan.tensor_sizes, kv_cache_config.num_blocks
)
first = kv_caches[name][0]
storage = first.untyped_storage()
base = first.storage_offset() * first.element_size()
```

`storage_offset()` 的单位是 `first` 的元素数，乘 `element_size()` 后得到 Byte Offset。

然后在同一个 Storage 上建立 `int8` View：

```python
raw = torch.empty(0, dtype=torch.int8, device=first.device).set_(
    storage, base, (size,), (1,)
)
```

`size` 是该层 Bundle 的完整字节跨度。

例如：

```text
Attention 使用的 Tensor
    Main K: bfloat16 View
    Main RoPE: bfloat16 View
    Indexer: int8 / bfloat16 View
        │
        └── 共同引用本层连续 Byte Storage
                    │
                    └── Broadcast 使用的 raw: int8 View
```

这里创建的是同一块 Storage 的视图。

Runtime 直接把整层 Byte View 保存到层名对应的位置：

```python
layer_buffers[name] = raw
```

`layer_buffers` 的结构是 `dict[str, torch.Tensor]`，每个值就是该层完整 Bundle 的 `int8` View。Attention 仍通过各 Cache Entry 的 Component Tuple 使用 K、RoPE 和 Indexer 等分量；这些 Component View 与广播用的整层 `raw` 引用相同的 Storage。

最后：

```python
static_forward_context[name].impl.layerwise_kv_cache_hook = scheduler
```

`layer_buffers` 的插入顺序同时作为 `attention_layer_names`，用于后续逐层推进。

---

#### 4.3.3 `KVPPScheduler`

负责整个 Forward 中的逐层 Prefetch：

```text
保存 has_history
维护 Target Attention 执行顺序
提交当前需要的 Prefetch
在当前层 Hook 等待完成
提交下一层 Prefetch
```

主要成员：

```text
transport
    执行本层 Broadcast

attention_layer_names
    目标 Attention 的有序名字列表

_has_history
    当前 Forward 是否启用广播

_next_attention_layer_index
    当前待消费的 Prefetch 对应的层序号

_prefetch_future
    当前后台 Prefetch 任务的 Future

_npu_device_id
    后台线程需要设置的 NPU Device

_kv_transfer_stream
    通信用 NPU Stream

_prefetch_executor
    max_workers=1 的后台线程池
```

调度器一次维护一个待消费的 Prefetch。当前层 Hook 消费它以后，立即为下一层建立新的 Future。

---

#### 4.3.4 `schedule_forward()`

**功能**

初始化本轮逐层调度状态，并按需启动第一个 Target Layer 的 Prefetch。

**入参**

- `has_history: bool`：当前 Batch 是否存在历史 KV。

**出参**

无返回值。更新本轮状态，并在需要时设置 `_prefetch_future`。

**处理过程 + 例子**

```python
self._has_history = has_history
self._next_attention_layer_index = 0
if has_history:
    self.start_layer_prefetch(self.attention_layer_names[0])
```

对于本例：

```text
prepare_forward(True)
        ↓
schedule_forward(True)
        ↓
start_layer_prefetch("model.layers.0.self_attn.attn")
```

第一层 Broadcast 从 Layer 0 Owner Rank 0 发起，其他 Rank 接收至自己对应的 Layer 0 Scratch View。

---

#### 4.3.5 `start_layer_prefetch()`

**功能**

记录缓存可用于传输的 Compute Stream 位置，并把本层通信任务提交给后台线程。

**入参**

- `layer_name`：需要 Prefetch 的 Target Attention Layer Name。

**出参**

无返回值。新任务的 Future 保存在：

```python
self._prefetch_future
```

**处理过程 + 例子**

```python
cache_ready = torch.npu.Event()
cache_ready.record(torch.npu.current_stream())
self._prefetch_future = self._prefetch_executor.submit(
    self.run_layer_prefetch, layer_name, cache_ready
)
```

例如在 Layer 11 Hook 内启动 Layer 12 Prefetch：

```text
当前 Compute Stream 已提交
    Layer 10 的缓存使用
    Layer 11 的前置计算
        ↓
记录 cache_ready
        ↓
向后台提交 Layer 12
```

后台 Transport 在自己的 Stream 上等待该 Event，之后才能读写这次广播使用的缓存。

---

#### 4.3.6 `run_layer_prefetch()`

**功能**

后台线程中一次 Prefetch 的入口。

**入参**

- `layer_name`：本次广播对应的 Target Layer。
- `cache_ready`：调用线程在 Compute Stream 上记录的 Event。

**出参**

无返回值。函数返回后，对应的 Prefetch Future 完成。

**处理过程 + 例子**

```python
torch.npu.set_device(self._npu_device_id)
self.transport.prefetch(
    layer_name, cache_ready, self._kv_transfer_stream
)
```

先设置后台线程使用的 NPU Device，再把层名、事件和 Transfer Stream 交给 Transport。

---

#### 4.3.7 `wait_for_layer()`

**功能**

在当前 Target Attention 读写 KV 前消费本层 Prefetch，并推进下一层。

**入参**

- `layer_name`：Attention Hook 提供的当前 Layer Name。

**出参**

无返回值。有历史 KV 时，返回前已消费当前层的 Future，并提交存在的下一层 Prefetch。

**处理过程 + 例子**

```text
_has_history=False
    → 返回

_has_history=True
    → _prefetch_future.result()
    → _prefetch_future = None
    → 层序号加 1
    → 从 attention_layer_names 取得下一层
    → start_layer_prefetch(next_layer)
```

例如：

```text
进入 wait_for_layer(Layer 10)
    当前 Future 对应 Layer 10
        ↓
Layer 10 KV Ready
        ↓
提交 Layer 11 Prefetch
        ↓
返回 Layer 10 Attention
```

调度使用内部的有序层列表推进。正常模型执行顺序与这个 Target 层顺序一致。

---

#### 4.3.8 `complete_forward()`

**功能**

结束本轮调度状态。

**入参**

无。

**出参**

无。

**处理过程**

```python
self._has_history = False
self._next_attention_layer_index = 0
```

最后一个 Target Hook 已经消费了本轮最后一次 Prefetch。这个方法负责把历史标志和执行序号复位。

---

### 4.4 `vllm_ascend/distributed/kv_transfer/kv_pool/broadcast_transport.py`

这个文件执行实际的跨 Rank 缓存广播。

#### 4.4.1 `BroadcastKVPPTransport`

Transport 保存三份信息：

```text
_device_group
    本 KVPP Group 使用的设备通信组

_owner_global_ranks
    Target Cache Name → Owner Global Rank

_layer_buffers
    Target Attention Name → 本 Rank 上覆盖整层 Bundle 的一个 raw Byte View
```

整个传输关系是：

```text
Layer Owner Rank
    Persistent KV Bundle
            │
            │ 同一个 Collective
            ▼
其他 KVPP Rank
    对应层的 Scratch KV Bundle Views
```

每个 Rank 都为相同 Layer 提交相同顺序的 Broadcast。`src` 标识该层 Owner，当前 Rank 传入的是自己已绑定的缓存 View。

---

#### 4.4.2 `BroadcastKVPPTransport.__init__()`

**功能**

保存通信组和缓存视图，并把 Owner 的 Group 内编号转换成 Global Rank。

**入参**

- `kvpp_group`：当前 KVPP `GroupCoordinator`。
- `layer_owner_ranks`：Target Cache Name 到 Group 内 Owner Rank 的映射。
- `layer_buffers: dict[str, torch.Tensor]`：每个 Target Attention 对应的整层 Byte View。

**出参**

无返回值（`None`）；初始化通信组、Owner 映射和缓存视图。

**处理过程 + 例子**

Global Rank 的计算为：

```python
kvpp_group.ranks[owner]
```

本例中：

```text
kvpp_group.ranks = [0, 1, 2, 3, 4, 5, 6, 7]
Layer 10 Owner  = 1
Global Source   = 1
```

如果另一个 PP Stage 的 KVPP Group 成员为：

```text
[8, 9, 10, 11, 12, 13, 14, 15]
```

则该 Group 内 Owner 1 对应 Global Rank 9。

每层的完整 Byte View 已在 Runtime 创建阶段准备好，后续 Prefetch 按 Layer Name 直接取用。

---

#### 4.4.3 `prefetch()`

**功能**

通过本 KVPP Device Group 对该层完整 Bundle 执行一次广播，并等待设备侧完成。

**入参**

- `layer_name`：需要广播的 Target Attention Layer Name。
- `cache_ready`：当前 Compute Stream 的缓存就绪事件。
- `transfer_stream`：KVPP 专用 NPU Transfer Stream。

**出参**

无返回值。函数结束时，该层 Broadcast 已完成设备侧读写。

**处理过程 + 例子**

核心代码为：

```python
with torch.npu.stream(transfer_stream):
    transfer_stream.wait_event(cache_ready)
    work = dist.broadcast(
        self._layer_buffers[layer_name],
        src=self._owner_global_ranks[layer_name],
        group=self._device_group,
        async_op=True,
    )
    work.wait()
    done = torch.npu.Event()
    done.record(transfer_stream)
done.synchronize()
```

以 Layer 10 为例：

```text
Rank 1
    buffer = Layer 10 Persistent Bundle 的完整 Byte View
    src    = 1

Rank 0
    buffer = Layer 10 在 Scratch 0 中的完整 Byte View
    src    = 1

其他 Rank
    buffer = 该 Rank 的 Layer 10 Scratch View
    src    = 1
```

所有 Rank 提交这个 Collective 后，Layer 10 Owner 的完整缓存内容写入各接收端对应的 View。

一次 Collective 覆盖整个 Bundle，其中包含 Main K、RoPE、Indexer Data 和实际存在的 Scale 等 Component。

```text
整层 raw Byte View
        ↓
一次 dist.broadcast()
        ↓
work.wait()
        ↓
done.record(transfer_stream)
        ↓
done.synchronize()
```

`done.synchronize()` 等待该层广播的设备侧读写全部完成。随后 `run_layer_prefetch()` 返回，Attention 对应的 Future 完成。

---

### 4.5 配置、Worker 与 Model Runner 接入

这些位置把配置、缓存规划、实际分配和逐层调度连接起来。

#### 4.5.1 `KVPPConfig` 与平台配置检查

`enable_kvpp` 通过主干的 `validate_additional_config_bool()` 解析，接受布尔值，也接受大小写不同的布尔字符串，例如 `"True"`、`"FALSE"`。

配置位于 `vllm_ascend/ascend_config.py`：

```python
class KVPPConfig:
    size: int = 1
```

`from_vllm_config()` 读取：

```text
enable_kvpp
    True  → size=tensor_parallel_size
    False → size=1
```

`platform._validate_parallel_config()` 在 KVPP Size 大于 1 时调用 `KVPPConfig.validate()`。

支持条件包括：

```text
执行：Eager
模型：非 Hybrid MLA
PCP：1
DCP：1
KV Transfer Connector：未配置
Speculative Decoding：未启用或固定步数 MTP
```

缓存规划使用具有共同 Block Size 的 Full Attention Spec。

#### 4.5.2 `NPUWorker.get_kv_cache_spec()`

入口位于 `vllm_ascend/worker/worker.py`。

```text
取得当前 Model Runner 的 kv_cache_spec
        ↓
KVPP Size > 1 时
        ↓
create_kvpp_cache_allocation_plan()
        ↓
保存 self._kvpp_cache_allocation_plan
        ↓
返回本 Worker 的完整 kv_cache_spec
```

Worker 保存的 Plan 用于之后的内存预算换算。

#### 4.5.3 `NPUWorker._apply_kvpp_memory_budget()`

**功能**

把当前 Worker 的物理 KV Cache 预算换算成完整 Logical Spec 对应的 Planner 输入。

**入参**

- `available_bytes`：此前内存计算得到的物理预算。

**出参**

供 Engine KV Cache Planner 使用的整数 Byte 数。

**处理过程**

```python
self.available_kv_cache_memory_bytes = available_bytes
plan = self._kvpp_cache_allocation_plan
if plan is None:
    return available_bytes
num_blocks = plan.get_num_blocks(available_bytes)
return num_blocks * sum(
    spec.page_size_bytes for spec in plan.logical_cache_spec.values()
)
```

`determine_available_memory()` 的显式 KV Cache 预算路径和内存 Profiling 路径都会经过这个换算入口。

#### 4.5.4 Model Runner V1 接入

文件：`vllm_ascend/worker/model_runner_v1.py`。

```text
构造 Model Runner
    → self.kvpp = KVPPRuntime()

_allocate_kv_cache_tensors()
    → KVPP 启用时调用 allocate_kvpp_cache()

initialize_kv_cache()
    → 完成缓存分配、reshape、绑定
    → create_from_kv_cache(..., kv_caches=kv_caches)

execute_model()
    → 根据实际请求 computed tokens 调 prepare_forward()
    → _model_forward()
    → complete_forward()

_dummy_run()
    → prepare_forward(False)
    → 执行 Dummy Forward
    → complete_forward()
```

V1 的 `kv_caches` 包含已经按 Attention 需要解释的缓存 Tensor，Runtime 从其底层 Storage 建立 Broadcast View。

#### 4.5.5 Model Runner V2 接入

涉及：

```text
worker/v2/model_runner.py
worker/v2/attn_utils.py
worker/v2/model_states/default.py
patch/worker/patch_v2/patch_attn_utils.py
```

V2 的上游 `init_kv_cache()` 调用 `allocate_kv_cache`。`patch_attn_utils` 将这个入口绑定为 Ascend 的 `allocate_kv_cache_main()`，由它组织 Ascend 分配和 reshape，再由初始化流程绑定 Layer Cache。

```text
NPUModelRunner.initialize_kv_cache()
    → super().initialize_kv_cache()
        → 上游 init_kv_cache()
            → allocate_kv_cache = allocate_kv_cache_main
                → Ascend _allocate_kv_cache()
                    → KVPP 启用：allocate_kvpp_cache()
                    → 返回各 Cache Entry 的 Component Byte Views
                → Ascend _reshape_kv_cache_v2()
            → bind_kv_cache()
    → KVPPRuntime.create_from_kv_cache()
    → self.model_state.kvpp_runtime = self.kvpp
```

`_allocate_kv_cache()` 的 KVPP 分支直接返回专用分配器生成的连续 Bundle Views。后续 reshape 保留这些 View 的底层 Storage，Runtime 在同一 Storage 上建立整层广播视图。

执行时：

```text
NPUModelRunner.execute_model()
    → 设置 kvpp_is_dummy_run = dummy_run or is_profile
    → 父类 execute_model()
        → AscendModelState.prepare_attn()
            → 从真实请求的 num_computed_tokens_np 判断 has_history
            → kvpp_runtime.prepare_forward(has_history)
        → Target Model Forward
    → kvpp_is_dummy_run = False
    → kvpp.complete_forward()
```

`AscendModelState.prepare_attn()` 使用 `input_batch.num_reqs` 对 computed 数组切片，参与判断的是实际请求。

#### 4.5.6 Attention 与 Cache Spec 接入

`attention/mla_v1.py` 和 `attention/sfa_v1.py` 的 Target Attention Impl 保存 `layerwise_kv_cache_hook`。执行到相应的缓存读写边界时调用 `wait_for_layer(layer_name)`。

`core/kv_cache_interface.py` 提供 `AscendMLAAttentionSpec` 和 `AscendSFAIndexerCacheSpec`，分别表达 Main 与 Indexer 的布局信息。两者都注册为 `FullAttentionSpec` 兼容类型，并使用 `FullAttentionManager`；满足共同 Block Size 等分组条件时，可以组成同一个 UniformType Cache Group。各 Entry 保留自己的维度、dtype 和页大小，Plan 从对应 Spec 生成 Component。各自 `merge()` 中的具体类型检查用于合并同类 Spec。

---

### 4.6 文件之间的关系

```text
ascend_config.py / platform.py / distributed/parallel_state.py
    │
    │ KVPP 配置、支持范围和通信组
    ▼
core/kv_cache_placement.py
    │
    │ Layer Owner
    │ Bundle / Component 字节布局
    │ 每 Rank 的 Block 容量
    ▼
worker/worker.py
    │
    │ 保存 Plan
    │ 返回完整 Logical Spec
    │ 把物理预算换算成 Logical Planner 预算
    ▼
Engine KV Cache Planner
    │
    │ 确定完整逻辑配置和共同 num_blocks
    ▼
worker/kvpp_cache.py
    │
    │ 分配自有 Target / MTP 连续 Storage
    │ 分配两块 Scratch
    │ 为每个 Cache Entry 创建 Component Views
    ▼
Model Runner V1 / V2
    │
    │ reshape / bind KV Cache
    │ 创建共享 KVPPRuntime
    │ 每轮调用 prepare_forward / complete_forward
    ▼
worker/v2/kvpp.py
    │
    │ 为每个 Target Bundle 创建一个完整 raw Byte View
    │ 按 has_history 启动逐层 Prefetch
    │ 当前层等待，下一层提前提交
    ▼
broadcast_transport.py
    │
    │ 在 KVPP Device Group 上对每层执行一次完整 Bundle Broadcast
    │ cache_ready → Collective → done
    ▼
Attention Hook
    当前层 KV Ready 后，继续 KV 写入与 Attention 计算
```

## 5. 用例设计

UT 按职责组织，用少量参数组合覆盖分支和边界。尺寸、Owner、Offset、别名关系和调用顺序使用明确预期值；设备和通信依赖使用现有 UT Mock，缓存布局使用真实 Tensor Storage 验证。E2E 只保留一个综合场景，不按特性展开测试矩阵。

下面列的是测试设计与断言，不代表已经执行通过。本次不包含 KVPP Nightly 用例或任务配置。

### 5.1 UT 用例

文件路径相对于仓库根目录；同一行可以包含一个职责下的多个参数化测试。

| 编号 | 看护逻辑 | 输入与场景 | 主要断言 | 用例文件 |
| --- | --- | --- | --- | --- |
| UT-01 | 配置解析和支持边界 | KVPP 布尔值与布尔字符串；开关、TP Size；图模式、非 MLA、Hybrid、PCP/DCP、Connector、非固定 MTP 等不支持配置 | 启用时 Size 等于 TP，关闭时为 1；合法配置通过；不支持组合在配置阶段报错；配置工厂保留开关 | `tests/ut/test_ascend_config.py` |
| UT-02 | PP Stage 内的通信分组 | World Size=8、TP=4、PP=2；KVPP Size 为 1 或 4 | KVPP Group 不跨 PP Stage；关闭时单 Rank；独立于 MC2 Group；销毁时清理对应 Group | `tests/ut/distributed/test_parallel_state.py` |
| UT-03 | Owner 划分和 Bundle 顺序 | 本地 Target Layer 9–16，MTP Layer 17；TP=3 与 TP=10；倒序输入 Spec；同层 Main 与 Indexer | TP=3 的 Owner 为 0/0/0/1/1/1/2/2；输入顺序不影响结果；Main 与 Indexer 同 Owner；MTP 无 Owner；逻辑 Spec 完整 | `tests/ut/core/test_kvpp_cache_placement.py` |
| UT-04 | Component 字节数及连续布局 | MLA 双分量、SFA-C8 Packed Main、LI-C8 Data/Scale；FP16/FP32 Scale；非量化 Indexer、FA 量化 Split Factor | 每个分量字节数、Offset、Bundle 总字节数与手算值一致；各分量按全部 Block 连续排列 | `tests/ut/core/test_kvpp_cache_placement.py` |
| UT-05 | 物理预算与 Block 向下取整 | 不同 Owner Rank、无 Target Owner 的 Rank、仅 MTP/空 Stage；预算处于完整 Block 边界两侧 | TP=3 各 Rank 每 Block 成本为 404/392/328 Byte；Rank 1 预算 1175/1176 Byte 得到 2/3 Block；仅 MTP 无 Scratch 成本 | `tests/ut/core/test_kvpp_cache_placement.py` |
| UT-06 | Persistent、Scratch 与 MTP 分配 | 最终 Block 数为 1 或 3；跨 Owner 边界的 Target 顺序 | 9/11/15 共用 Scratch 0，10/16 共用 Scratch 1；12/13/14 与 MTP 17 独立；物理 Storage 总量准确；写入别名可见且不污染其他 Storage | `tests/ut/worker/test_kvpp_cache.py` |
| UT-07 | Worker 逻辑 Spec 与物理预算接入 | KVPP 开关；完整本地 Cache Spec；物理预算映射到 Planner 逻辑字节数 | 不删除非 Owner 的逻辑层；Planner 接收完整逻辑 Spec 和按物理可用 Block 数换算的预算；关闭路径保持正常规划 | `tests/ut/worker/test_worker_v1.py` |
| UT-08 | V1/V2 分配与 Reshape 主入口 | 非量化和 Packed/Indexer Scale 布局；最终 KV Cache Config | 真实分配入口使用最终 Block 数；Typed View 的 dtype、shape、offset 正确；Reshape 后保留预期 Storage 别名 | `tests/ut/worker/test_model_runner_v1.py`、`tests/ut/worker/test_attn_utils_v2.py` |
| UT-09 | Runtime 整层广播范围和 Hook 绑定 | 带非零 Offset 的 Typed View；Main、Indexer 和 MTP Cache | 字节起点等于 Storage Offset × Element Size；广播范围覆盖整个 Bundle；只给 Target Main Attention 绑定 Hook | `tests/ut/worker/test_kvpp.py` |
| UT-10 | 提前一层预取与 Forward 状态 | 无历史、有历史、连续多轮 Forward；手动驱动 Future；预取 Future 失败 | 无历史不预取；有历史先预取首层，等待当前层后提交下一层；末层不越界；轮次状态重置；失败直接传递且不再提交下一层 | `tests/ut/worker/test_kvpp.py` |
| UT-11 | Broadcast 参数与设备完成顺序 | Owner/Receiver；组内 Owner 1 映射到全局 Rank 9；带边界哨兵的整层 Payload | 一次 Broadcast 传递完整 Payload；先等待 Ready Event，再 Broadcast、Work.wait、记录并同步 Done Event；设备完成后 Future 才完成；Payload 外的数据不变 | `tests/ut/distributed/kv_transfer/kv_pool/test_broadcast_transport.py` |
| UT-12 | Model Runner 历史判断与生命周期 | V1/V2 真实请求、Padding 中的历史值、Dummy/Profile Forward | 仅真实请求的 Computed Tokens 参与历史判断；Dummy/Profile 不广播；Prepare、执行、Complete 顺序正确 | `tests/ut/worker/test_model_runner_v1.py`、`tests/ut/worker/test_model_runner_v2.py` |
| UT-13 | Attention Hook 时序 | MLA/SFA Native 与 Fused 路径；有/无 Indexer；Profile 路径 | 投影之后、首次访问/写入 KV Cache 之前等待一次；Fused Prolog 之前等待；Profile 不触发实际层广播 | `tests/ut/attention/test_mla_v1.py`、`tests/ut/attention/test_sfa_v1.py` |

### 5.2 E2E 综合用例

用例使用仓库四卡场景中的 `vllm-ascend/DeepSeek-V3.2-W8A8-Pruning`，复用 `VllmRunner` 和输出对比工具。固定使用 Model Runner V1；V2 接入由 UT 看护。本用例不代表 V2 或所有 Cache 量化布局的端到端覆盖。

同一个 pytest 用例依次创建 KVPP 关闭和开启的实例，其他参数一致。每个实例顺序处理两个请求：第一个建立历史 Cache，第二个共享完整前缀并更换后缀。请求长度为 384 Token 前缀加 16 Token 后缀，单轮 Token Budget 为 128；生成长度固定为 16 Token。两次请求在同一个实例内完成，确保第二次请求能够复用第一次留下的前缀。

| 编号 | 配置与特性叠加 | 执行步骤 | 主要断言 | 用例文件 |
| --- | --- | --- | --- | --- |
| E2E-01 | A3 四卡；DeepSeek-V3.2 W8A8 Pruning；Eager；TP=2、EP 开启、PP=2；Chunked Prefill、Prefix Caching、Async Scheduling 开启；MTP 固定 1 步；Block Size=128、逻辑 Block 数=64 | KVPP 关闭/开启分别加载；顺序提交共享 384 Token 前缀、后缀不同的两个 400 Token 请求；观察真实调度输出、Worker 状态和 Speculative Metrics；比较两组结果 | 两个 PP Stage 各有两个 Worker，开启时 KVPP Group 只含本 Stage TP Rank；Target Hook 完整且排除 MTP；EP 和异步配置实际启用；首请求至少经过两次真实 Prefill 调度且缓存命中为 0；第二请求缓存命中不少于 384 Token；MTP Cache 存在且 Draft 计数大于 0；请求完成并各生成 16 Token；KVPP 开关前后 Prompt、输出 Token ID 和文本一致 | `tests/e2e/pull_request/four_card/test_kvpp.py::test_kvpp_combined_features` |
