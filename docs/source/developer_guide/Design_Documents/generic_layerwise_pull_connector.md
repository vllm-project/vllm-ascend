# 通用 Layerwise Pull Connector 设计

## 文档状态

- 状态：设计草案
- 日期：2026-08-31
- 目标模块：P/D KV Cache layerwise 传输
- 参考实现：原 MemFabric 拉取路径与原 Mooncake 分层推送路径
- 参考 PR：[Mooncake Connector V2 Pull #14068](https://github.com/vllm-project/vllm-ascend/pull/14068)

## 1. 背景

当前存在两条 layerwise P/D 传输路径：

- 原 Mooncake 路径：P 主动向 D 写入；
- 原 MemFabric 路径：P 通知数据就绪，D 主动读取。

两条路径分别维护调度、地址计算、传输线程和完成状态，且部分逻辑与具体模型或目标内存位置绑定。
随着 layerwise KV Cache 支持更多组件，继续扩展独立 connector 会增加重复代码和状态组合。

本文建议将新 connector 统一为 **pull-only**：P 只发布源数据就绪，D 始终调用 backend 的
`read` 接口完成传输。框架不区分 D2D 或 RD2H；实际传输路径由本地和远端的已注册地址以及
backend 内部实现决定。

## 2. 目标与非目标

### 2.1 目标

1. 使用同一套 layerwise pull 流程支持 Mooncake 和 MemFabric。
2. 支持一个 layer 包含多个 KV Cache 组件，例如 main K/V、indexer 和 scale。
3. P 在 KV 写入完成后立即发布数据就绪，使 D 的读取能够与后续计算重叠。
4. 只有逻辑层即将复用仍有未完成读取的物理 storage slot 时才等待。
5. 复用已有的 TP、block size 和地址映射能力，不在 connector 中加入模型专用传输分支。

### 2.2 非目标

第一版不实现：

- push 传输；
- D2D、RD2H 模式枚举或两套执行流程；
- 同一次传输在多个 backend 之间自动回退；
- 运行时动态选择最快传输路径；
- transfer graph、通用 DAG 调度器或跨请求合并框架；
- connector 内的数据格式转换或 cache allocation。

KV Cache 的物理分配与组件复用由
[组件级 Layerwise KV Cache 复用设计](component_level_layerwise_kv_cache_reuse.md)负责。connector
只消费已经确定的 tensor、地址和 storage slot 关系。

## 3. 核心原则

### 3.1 框架只处理 local/remote 地址

通用 connector 不感知目标 buffer 位于 Host 还是 NPU，也不保存 `d2d` 或 `rd2h` 字段。

例如 SFA Decode 侧：

- main K/V 的 local address 由 Decode host pool 提供；
- indexer 和可选 scale 的 local address 由 Decode NPU cache 提供；
- connector 将这些地址放入同一次或多次 `read` 调用；
- MemFabric 根据已注册的内存区域执行实际数据移动。

Mooncake D2D 使用相同流程，只是 local address 指向 Decode NPU cache。

### 3.2 P 不执行远端写入

P 侧只负责：

1. 注册并发布源 KV Cache 地址；
2. 在某层 KV 写入完成后发送 `LAYER_READY`；
3. 接收 `LAYER_DONE` 或 `LAYER_FAILED`；
4. 在物理 source slot 被复用前等待未完成读取。

所有 `batch_transfer_sync_write` 路径均不进入新 connector。

### 3.3 地址规划和传输执行分离

地址规划根据 P/D layout、block IDs 和并行拓扑生成 READ descriptor。backend 只执行已经规划好的
地址列表，不理解 layer、request、TP 或具体 KV Cache 类型。

## 4. 最小数据模型

第一版只增加一个通用 layout 类型；READ descriptor 继续使用三组地址/长度数组，不新增辅助类型。

### 4.1 Component layout

```python
@dataclass(frozen=True)
class ComponentLayout:
    name: str
    group_index: int
    block_size: int
    dtypes: tuple[str, ...]
    base_addrs: tuple[int, ...]
    block_strides: tuple[int, ...]
    block_lengths: tuple[int, ...]
    block_shapes: tuple[tuple[int, ...], ...]
    block_size_scales: tuple[int, ...]
```

`name` 用于匹配 P/D 两端语义相同的组件。一个 component 可以包含多个 tensor，例如 main K/V。
这些 tensor 的顺序是 component layout 契约的一部分，P/D 两端必须一致。`block_size` 和
`block_shapes` 用于 TP/DCP、kernel block 和 tensor 内部切片计算。worker-level metadata 另外保存
`num_blocks`，用于校验 block ID 范围。layout 不描述 Host 或 Device 类型。

`group_index` 是端侧本地索引，P/D 两端不要求相等。计算 block pairing 时，local block IDs 使用
D layout 的 `group_index`，remote block IDs 使用 P layout 的 `group_index`；两端 component 只通过
`name` 匹配。

### 4.2 READ descriptor

执行路径只消费等长的 `local_addrs`、`remote_addrs` 和 `lengths` 数组。layer name、request ID
等信息可保留在 transfer task 中用于日志和错误归属，不进入 backend 接口。

### 4.3 批量地址生成

地址 planner 不应退化成 Python 逐 block 构造 descriptor。现有 `LayerwisePullConnector` 使用
NumPy 批量计算地址，并在 local 和 remote 两端都连续时合并相邻 descriptor，这部分应保留：

```python
remote_addrs = remote_base + remote_block_ids * remote_stride
local_addrs = local_base + local_block_ids * local_stride
lengths = np.full(num_blocks, block_length, dtype=np.int64)
```

合并只能在以下条件同时满足时进行：

```text
next_remote == current_remote + current_length
next_local  == current_local  + current_length
```

常见的 whole-block main KV、MLA 和 SFA indexer 路径直接使用该 NumPy fast path。需要 head slice 或
Mamba 内部切片时，先使用 PR #14068 的布局算法生成原子 descriptor，再执行相同的连续区间合并，
不为每种 cache spec 新建一套传输执行逻辑。

## 5. 模块职责

建议保留少量模块，避免将第一版拆成过多抽象：

```text
kv_p2p/layerwise_pull/
├── connector.py     # vLLM KVConnector facade
├── scheduler.py     # request metadata 和 block 生命周期
├── worker.py        # layer hook、后台通知和读取执行
├── protocol.py      # typed 控制消息和 layout metadata
├── send_thread.py   # P 侧通知线程和 source storage slot 门控
└── read_thread.py   # D 侧地址规划、backend 适配和读取线程
```

其中：

- scheduler 不处理传输地址；
- read thread 内的地址规划不感知 scheduler；
- backend 适配只提供统一的 READ 语义，不理解 KV Cache 生命周期；
- worker 负责把 layer 生命周期与 send/read thread 串联起来。

backend 的最小接口为：

```python
class PullBackend:
    def read(
        self,
        session_id: str,
        local_addrs: list[int],
        remote_addrs: list[int],
        lengths: list[int],
    ) -> None: ...
```

worker 负责初始化 engine、注册内存和建立 session；backend adapter 只统一 READ 调用与错误码。

## 6. 控制协议与执行流程

### 6.1 PP metadata 和路由

第一版要求 P/D 的 PP size 和 layer partition 对齐。每个 D PP worker 在自己的端口段监听，P 根据
`(pp_rank, tp_rank)` 选择目标 endpoint：

```text
(pp_rank, tp_rank) -> session、endpoint、owned layer names、component layouts
```

端口范围按 `DP × PP × TP` 划分。每个 P PP worker 只发布本 stage 的实际 layer layout；MTP layer
使用模型总基础层数作为编号偏移，避免在 PP>1 时与普通 layer 编号冲突。不同 P/D PP 切分尚不支持，
初始化时会明确拒绝 PP size 不一致的配置。

### 6.2 消息和单层流程

控制面保留四类消息：

```text
LAYOUT_META       P session、并行拓扑、num_blocks 和 component layouts
READ_READY_BATCH  transfer_id、layer 和 request/block 信息
READ_DONE         transfer_id
READ_FAILED       transfer_id、错误信息
```

`transfer_id` 是 P worker 进程内单调递增的整数，用于避免上一 step 的迟到回复错误释放当前
transfer。回复的 endpoint identity 由控制连接提供。最终 request completion ACK 沿用 scheduler
side channel，不属于上述 layer worker 协议。第一版不再引入额外 session 状态机。

单层执行流程如下：

1. P 在 KV scatter 完成后记录 NPU event。
2. P 将 layer task 放入后台队列，模型计算继续执行。
3. P 后台线程等待 NPU event，然后向相关 D endpoint 发送 `READ_READY_BATCH`。
4. D 根据本地 destination layout、P 的 source layout 和请求 block IDs 批量生成 descriptors。
5. D 合并同一 layer、同一 remote session 下所有 request/component 的 descriptors，调用一次
   `backend.read()`。
6. D 读取成功后发送 `READ_DONE`；失败则发送 `READ_FAILED`。
7. P 更新 source slot 的完成状态；request block 由最终 request completion ACK 释放。

D 端只有在请求要求的所有 layer、component 和 contributor 都完成后，才将请求标记为接收完成。
不同 remote session 或不同 layer 的 descriptor 不做跨边界合并，以保持错误归属和物理 slot 完成
语义清晰。backend 如果不能在一次 READ 中混合不同类型的已注册内存，由 backend adapter 内部拆分；
connector 不增加 Host/Device 判断。

## 7. 物理 slot 复用

逻辑 layer 可能映射到相同的物理 storage slot。等待必须以物理 slot 为单位，不能只按 layer name
记录。

初始化时根据 tensor backing storage 建立：

```text
layer_index -> storage_slot_ids
storage_slot_id -> is_reused
```

P 侧每个 reused slot 维护一个完成 event，以及仍在读取该 slot 的 reader 集合：

```text
storage_slot_id -> {(transfer_id, endpoint_id), ...}
```

P 在构造每个 endpoint 的 `LAYER_READY` payload 时，根据其中实际包含且 block IDs 非空的
components 记录对应 source slots。D 只为 payload 中的 components 生成 descriptors。未包含在 payload
中的 slot 不加入 reader 集合，也不关闭完成 event。

时序要求：

1. 发送 `LAYER_READY` 前，将 `(transfer_id, endpoint_id)` 加入实际触达的 reused slots，并清除其
   完成 event；
2. 收到完成回复后，从对应 slots 的 reader 集合中移除该项；重复回复使用 `discard`，不重复计数；
3. 某个 slot 的 reader 集合为空后，立即设置该 slot 的完成 event；
4. 下一逻辑层准备覆盖该 slot 时调用 `wait_for_layer_reuse()`；
5. 非复用 slot 不进入该等待路径。

`LAYER_FAILED` 执行相同的 reader 移除操作，避免永久等待，但在 slot event 设置后仍保留错误，等待方
必须抛出该传输错误。

P 在注册 reader 之后发生的本地错误也执行相同清理，包括 task 入队失败、控制消息发送失败和通知线程
退出。此时 P 不等待 D 回复，而是主动从相关 slots 移除 `(transfer_id, endpoint_id)`、记录错误，并在
reader 集合为空时设置完成 event。

物理 slot 完成和 request block 释放是两个状态：

- slot 完成只保护 layer buffer 不被提前覆盖；
- request block 必须保留到 D 不再读取该请求，即使该 slot 没有被其他 layer 复用。

## 8. 可复用实现

[PR #14068](https://github.com/vllm-project/vllm-ascend/pull/14068) 的 Mooncake pull 实现可复用以下能力：

- 按 layer name 合并 PP/TP metadata；
- P/D 不同 TP、DCP 和 KV head 布局匹配；
- logical block、kernel block 和 block-size scale 换算；
- FullAttention、MLA、SFA indexer、Sliding Window 和 Mamba 的地址切片；
- 按 backing storage 合并内存注册区间；
- TP/DCP/block size 的单元测试矩阵。

这些逻辑应进入 `read_thread.py` 的地址规划逻辑或公共注册 helper。PR 中的请求级 worker、Mooncake 专属控制线程和
空的 layerwise hook 不直接复制到新 connector。

现有 `LayerwisePullConnector` 可复用：

- scatter 后提前发布 readiness 的 hook；
- main/indexer 分组和 unequal TP contributor 规则；
- Decode host pool 与 rank-local indexer destination 的解析；
- NumPy 批量地址生成和双端连续 descriptor 合并；
- 同一 layer 内跨 request、跨 component 的单次 batch READ；
- physical storage slot 完成门。

## 9. 迁移与验证

### 9.1 迁移顺序

1. 建立 pull backend、协议和 planner，先迁移现有 MemFabric SFA 路径。
2. 使用通用 layout planner 接入 Mooncake READ，替换 layerwise push。
3. 仅注册 `LayerwisePullConnector`，不保留旧 connector 别名。

配置只选择 backend，不配置传输方向：

```json
{
  "kv_connector": "LayerwisePullConnector",
  "kv_connector_extra_config": {
    "transfer_backend": "memfabric"
  }
}
```

### 9.2 必要测试

第一版至少覆盖：

- Mooncake 和 MemFabric 调用相同的 planner/READ 流程；
- local address 分别来自 Host 和 NPU 时 connector 行为一致；
- 无 storage 复用时 `wait_for_layer_reuse()` 不等待；
- 两个 layer 共享 slot 时，后一个 layer 等待前一个 transfer 完成；
- 多 endpoint 全部完成后才释放 slot；
- 只读取部分 component 时，仅关闭并等待实际触达的 reused slots；
- 重复完成回复不会提前释放 slot；
- 迟到的旧 `transfer_id` 不能释放新 transfer；
- P 侧入队或发送失败能够解除 slot 等待并传播错误；
- READ 失败能够解除等待并向请求报告错误；
- P/D group 顺序不同时，各自使用本地 `group_index`；
- equal/unequal TP、SFA indexer scale 和不同 block-size scale 地址映射；
- 不同 DP rank 的 PP/TP worker metadata 不会发生 key 冲突；
- P/D 使用相同 PP partition 时，各 stage 都能按 layer name 找到正确 producer；
- 不同 P PP worker 的 descriptor 不会进入同一个 batch；
- 单个 P worker 的 slot 完成不依赖无关 PP stage 的全局 barrier；
- 连续 descriptor 被正确合并，任一端不连续时不合并；
- 同一 layer/session 的多个 request 和 component 只提交一次 batch READ；
- 不同 layer 或 remote session 不会被错误合并。

需要在真实 NPU 环境验证 scatter、READ 和下一层计算能够重叠，并确认等待路径没有新增
`tensor.item()` 或 CPU-NPU 同步。
