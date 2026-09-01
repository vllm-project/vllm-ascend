# MC2 领域代码检视规则

<适用>
语言: C++
侧别: Host, Kernel
领域: true
触发: hccl_, AllGather, AllReduce, ReduceScatter, AlltoAll, SyncAll, SetFlag, WaitFlag, PipeBarrier, SyncFunc, expert, expertIds, dispatch, combine, moeExpertNum, quant, BasicQuantMode, PERTENSOR, PERCHANNEL, AlltoAllV, sendCounts, InitV2, Finalize, SetCcTiling, CCU, AICPU
默认启用: true

排除场景: 纯通信中转(有hccl_但无expert/dispatch/combine路由逻辑且无Matmul/GroupedMatmul/quant), 纯计算无集合通信(有计算无hccl_)
</适用>

<检视负载>
通用检视子 agent 检视条款容量上限: 5
</检视负载>

> **MC² = Matrix Computation & Communication**（通算融合算子框架）
>
> 将 HCCL 集合通信与计算融合为单一算子，减少 kernel launch 开销，实现通信与计算流水线并行。
>
> **规则分类概览**：
>
> | 分类 | 规则范围 | 核心检视关注点 |
> |------|---------|-------------|
> | 通信同步 | MC2-01~04 | 流同步、核间同步、全局参数一致性 |
> | MoE 专家路由 | MC2-05~08 | 专家索引边界、分发组合一致性、参数校验与日志 |
> | 量化精度 | MC2-09~12 | 量化参数类型、模式校验、精度保护、配置匹配 |
> | 硬件约束 | MC2-13~14 | CCU数据量限制、Tiling校验与结构规范 |
> | HCCL 通信与安全 | MC2-15~19 | 通信域安全、编译规范、API查阅验证、通信生命周期、跨rank参数一致性 |
>
> **核心特点**：集合通信与计算融合（Cube计算、量化计算等）、TP/SP/EP分布式推理支持、量化（pertensor/perchannel/perblock/MX pergroup）、CCU/AICPU双通信模式、多轮流水线并行

---

> **适用场景**：MC²通算融合算子的代码检视（集合通信与计算融合算子）
>
> **动态知识获取要求**：
> - 本 Skill 中的静态 API 列表、类型映射等仅作为**开发者快速参考**，非权威数据源
> - Agent 在执行检视时，应**始终通过 `/ascendc-docs-search` 获取当前 CANN 版本的官方信息**，以确保规则与目标代码版本匹配
> - 当发现代码使用未知通信原语时，同样使用该 skill 认是否属于集合通信范畴
> - 若查询到的官方文档与本 Skill 静态表不一致，以官方文档为准，并建议更新本 Skill 的静态表
>
> HCCL（Huawei Collective Communication Library）通信 API 识别方法：
> - **识别原则**：在 Ascend C 编程模型中，集合通信通过 `hccl_` 对象方法调用提供，配套初始化/销毁/配置接口。具体 API 列表和参数约束请通过 `/ascendc-docs-search` 查询获取最新信息
> - **Kernel侧通信操作**：`hccl_` 对象上的方法调用（如 `hccl_.AllGather<true>()`、`hccl_.AllReduce()`、`hccl_.ReduceScatter<false>()`、`hccl_.AlltoAllV<true>()`），属 HCOMM 数据面接口（常见示例，非穷举）
> - **HCCL 生命周期与配置**：`InitV2` / `Init`、`Finalize`、`SetCcTilingV2` / `SetCcTiling`、`Commit`、`Wait`、`GetHcclContext`（常见示例）
> - **HCCL 类型与枚举**：`HcclDataType` / `HCCL_DATA_TYPE_*`、`HcclReduceOp` / `HCCL_REDUCE_*`、`HcclServerType`（常见示例）
> - **通信原语关键词**：`AllGather`、`AllReduce`、`ReduceScatter`、`AllToAll` / `AlltoAllV`、`Broadcast`（常见示例）
> - **动态识别**：当代码中出现疑似集合通信调用但不在上述列表中，必须使用 `/ascendc-docs-search` 查询确认是否属于 HCCL 接口，确认后纳入检视范围

---

## 术语表

| 术语 | 含义 |
|------|------|
| MC² | Matrix Computation & Communication，昇腾通算融合算子框架，将集合通信与计算融合为单一算子 |
| 通算融合 | 通信与计算融合（集合通信 + Cube/Vector 计算），实现流水线并行 |
| Cube / AIC | 矩阵计算单元（AI Core Cube），执行 Matmul/GroupedMatmul 等 |
| Vector / AIV | 向量计算单元（AI Core Vector），执行 Add/Mul/Cast/Reduce 等向量操作 |
| UB | Unified Buffer，Vector 核本地存储，AIV 侧数据搬入/搬出的中间缓冲区 |
| V核 | 即 AIV（Vector 核），MC² 通算并行场景中需 SyncAll 防止 V 核间数据踩踏 |
| SyncAll | 全核全局同步屏障，所有核执行到同一 SyncAll 点后才能继续。`SyncAll<false>` 同步所有核含 AIC |
| SyncFunc | 硬件事件同步，如 `SyncFunc<HardEvent::S_MTE3>()`，精确到流水对 |
| SetFlag / WaitFlag | ISASI 精确同步机制，SetFlag 标记前序完成，WaitFlag 等待标志。必须成对、同 EVENT_ID。HardEvent 类型标识同步的流水线阶段（如 `S_MTE3` = Scalar→MTE3，`MTE2_V` = MTE2→Vector） |
| HardEvent | 硬件事件类型，用于 SetFlag/WaitFlag/PipeBarrier 指定同步的流水线对。常见类型：`S_MTE3`（Scalar→MTE3）、`MTE2_V`（MTE2→Vector）、`V_MTE3`（Vector→MTE3）、`M_V`（Cube→Vector）、`V_V`（Vector→Vector）、`MTE1_M`（MTE1搬运完成→Cube）、`MTE3_MTE2`（MTE3→MTE2 双缓冲交替）等。必须与实际数据流方向匹配 |
| PipeBarrier | 单流水屏障，如 `PipeBarrier<PIPE_V>()` 等待前序 V 操作完成。PIPE_V 内部不保证自动保序 |
| Pipe | Ascend C 流水线，包括 PIPE_S（标量）、PIPE_V（向量）、PIPE_M（Cube）、PIPE_MTE2/MTE3（数据搬运）等 |
| TP | Tensor Parallelism，张量并行，沿张量维度切分计算 |
| EP | Expert Parallelism，专家并行，MoE 架构中沿专家维度切分 |
| SP | Sequence Parallelism，序列并行 |
| 大EP | 大规模跨节点专家并行，dispatch-combine 多轮通信+计算模式 |
| CCU | AI Core 通信单元（Cube Communication Unit），高性能小数据量通信引擎，单次≤256MB |
| AICPU | AI CPU，大数据量通信引擎，无 256MB 限制 |
| HCCL | Huawei Collective Communication Library，昇腾集合通信库 |
| HCOMM | Huawei Communication，HCCL 通信基础库，提供控制面和数据面接口 |
| PTA | Process Template Architecture，算子框架层，负责通信域传递等 |
| 集合通信原语 | AllGather、AllReduce、ReduceScatter、AllToAll/AlltoAllV、Broadcast 等标准通信操作 |
| Tiling | 算子参数配置流程，将计算参数和通信参数打包传递给 Kernel |
| Mc2InitTiling | MC² HCCL 初始化 Tiling 数据结构，包含通信域、rank 信息等初始化参数 |
| Mc2CcTiling | MC² HCCL 通信配置 Tiling 数据结构，包含通信原语参数（数据量、类型等） |
| EnQue / DeQue | TPipe 队列管理，入队通知下游、出队等待上游，自带隐式同步 |
| Double Buffer | 双缓冲流水线模式，交替使用两个 buffer 实现搬入-计算-搬出重叠 |

---

## PR 差异→规则速查

> **⚠️ 本表为常见关键词示例，非穷举列表。**
>
> Agent 不得仅依赖本表判定 API 归属。涉及 HCCL 通信 API 或疑似新 API 时，必须使用 `/ascendc-docs-search` skill 动态查询：确认 API 定义、参数约束、数据类型支持列表等。优先以查询到的官方文档为准，本表仅供快速参考。
>
> **使用方法**：查看 PR 差异代码中的关键词，匹配下表，仅阅读对应分类的规则。无需一次阅读全部规则。

| 差异关键词 | 对应分类 | 应读规则 | 典型搜索命令 |
|-----------|---------|---------|------------|
| `hccl_` / `AllGather` / `AllReduce` / `ReduceScatter` / `AlltoAll` | 通信同步 + HCCL API | MC2-01~04, MC2-18~19 | `grep -rn "hccl_\|AllGather\|AllReduce\|ReduceScatter\|AlltoAll"` |
| `SyncAll` / `SetFlag` / `WaitFlag` / `PipeBarrier` / `SyncFunc` | 通信同步 | MC2-01~03 | `grep -rn "SyncAll\|SetFlag\|WaitFlag\|PipeBarrier\|SyncFunc"` |
| `InitV2` / `Finalize` / `SetCcTiling` / `GetHcclContext` / `Commit` / `Wait` | HCCL 通信 API | MC2-18 | `grep -rn "InitV2\|Finalize\|SetCcTiling\|GetHcclContext\|Commit\|Wait"` |
| `HcclDataType` / `HcclReduceOp` / `HCCL_DATA_TYPE` / `HCCL_REDUCE` | HCCL 通信 API | MC2-18 | `grep -rn "HcclDataType\|HcclReduceOp\|HCCL_DATA_TYPE\|HCCL_REDUCE"` |
| `AlltoAllV` / `AlltoAllv` / `sendCounts` / `recvCounts` / `alltoAllvSendCnt` / `alltoAllvRecvCnt` | HCCL 通信 API | MC2-19 | `grep -rn "AlltoAllV\|AlltoAllv\|sendCounts\|recvCounts\|alltoAllvSendCnt\|alltoAllvRecvCnt"` |
| `expert` / `expertIds` / `moeExpertNum` / `dispatch` / `combine` / `localMoeExpertNum` | MoE 专家路由 | MC2-05~08 | `grep -rn "expert\|expertIds\|moeExpertNum\|dispatch\|combine\|localMoeExpertNum"` |
| `GetAttrPointer` / `GetAttr` | MoE 专家路由 | MC2-08 | `grep -rn "GetAttrPointer\|GetAttr"` |
| `quant` / `BasicQuantMode` / `PERTENSOR` / `PERCHANNEL` / `MX/pergroup` / `scale` / `pertensor` / `perblock` | 量化精度 | MC2-09~12 | `grep -rn "quant\|BasicQuantMode\|PERTENSOR\|PERCHANNEL\|MX_PERGROUP\|scale\|pertensor\|perblock"` |
| `Cast` / `FP8` / `BF16` / `INT8` | 量化精度 | MC2-11 | `grep -rn "Cast.*FP8\|Cast.*BF16\|Cast.*INT8"` |
| `epWorldSize` / `tpWorldSize` / `epRankId` / `tpRankId` / `rankId` | 量化精度 + 硬件 | MC2-12, MC2-13 | `grep -rn "epWorldSize\|tpWorldSize\|epRankId\|tpRankId\|rankId"` |
| `CCU` / `AICPU` / `256.*1024.*1024` | 硬件约束 | MC2-13 | `grep -rn "CCU\|AICPU\|256.*1024.*1024"` |
| `groupName` / `hcom` / `SetGroupName` | 安全 | MC2-15 | `grep -rn "groupName\|hcom\|SetGroupName"` |
| `kernel_operator.h` / `ASC_DEVKIT_MAJOR` / `basic_api` | 编译规范 | MC2-16 | `grep -rn "kernel_operator.h\|ASC_DEVKIT_MAJOR\|basic_api"` |
| `Mc2CcTilingConfig` / `Mc2InitTiling` / `Mc2CcTiling` / `HcclTypeSelector` | API 使用（含内部实现关键词，非公开 API 规范） | MC2-17, MC2-18 | `grep -rn "Mc2CcTilingConfig\|Mc2InitTiling\|Mc2CcTiling\|HcclTypeSelector"` |
| `OP_LOGE` / `OP_LOGD` / `%ld` / `%lu` | 量化精度 + MoE | MC2-07, MC2-09, MC2-10 | `grep -rn "OP_LOGE\|OP_LOGD"` |
| `stride` / `contiguous` | 硬件约束 | MC2-14 | `grep -rn "stride\|contiguous"` |

---

## 领域判定规则

> **MC²（Matrix Computation & Communication）**是昇腾通算融合算子框架，将集合通信与计算（Cube 计算、量化计算等）融合为单一算子，实现通信与计算流水线并行。
>
> Agent 审阅代码时，先使用以下规则判定代码是否属于 MC² 领域。满足 **至少一个核心特征** 且 **不属于排除场景** 时，启用本规则集。

### 核心特征（满足任一即可）

| 特征编号 | 特征描述 | 检测方法 |
|---------|---------|---------|
| **C1** | 通信与计算流水线交替 | 同一循环或流水线结构内，`hccl_` 集合通信与融合计算（Matmul/GroupedMatmul/quant 等）交替执行，且存在同步屏障（SyncAll/SyncFunc/SetFlag+WaitFlag）衔接 |
| **C2** | 出现大EP分发/组合模式 | 存在 expert/dispatch/combine 相关标识符，且伴随 AllToAll/AlltoAllv 通信 |

### 排除场景（即使命中关键字也不启用）

| 场景 | 说明 |
|------|------|
| 纯通信中转算子（无路由逻辑） | 仅使用集合通信进行数据传输，无 expert/dispatch/combine 路由逻辑，无量化参数，无 Cube 计算融合（如分布式训练中的梯度 AllReduce） |
| 纯计算算子 | 仅使用 Matmul 等计算 API，无集合通信调用 |
| 测试/示例代码 | 文件路径包含 test/example/sample 且功能为单元测试 |

### 常见误报场景排除

| 误报场景 | 特征 | 处理方式 |
|----------|------|----------|
| 仅调用通信库做梯度同步（无计算融合） | 代码含 AllReduce 但无 Cube 计算或量化计算，无 MC² tiling 结构 | 不判定为 MC²，不启用本规则集 |
| 纯通信中转算子（无路由逻辑） | 仅有 hccl_/AllToAll 数据搬运，无 expert/dispatch/combine 路由逻辑，无量化参数，无 Matmul 融合 | 不判定为 MC² |
| 单元测试调用 MC2 接口 | 文件路径含 test，且仅为单个 API 测试 | 不启用全量规则 |

### 场景子分类（判定为 MC2 后，选取规则子集）

| 子分类 | 通算融合模式 | 应读规则 |
|--------|---------|---------|
| TP权重聚合 | AllGather + Matmul（通信→计算） | MC2-01~04, MC2-13, MC2-18~19 |
| TP输出分发 | ReduceScatter + Matmul（计算→通信） | MC2-01~04, MC2-13, MC2-18~19 |
| 全局归约 | AllReduce + Matmul | MC2-01~04, MC2-18~19 |
| EP专家分发 | AllToAllv + GroupedMatmul（通信→计算） | MC2-01~04, MC2-05~08, MC2-13, MC2-19 |
| EP反向分发 | GroupedMatmul + AllToAllv（计算→通信） | MC2-01~04, MC2-05~08, MC2-13, MC2-19 |
| 大EP分发组合 | AllToAllv + expert + GroupedMatmul（多轮） | MC2-05~08, MC2-09~12, MC2-19 |
| MoE 路由分发/组合 | AllToAllv/AllToAll + expert 路由（无 GroupedMatmul 融合，如 moe_distribute_dispatch） | MC2-01~04, MC2-05~08, MC2-13, MC2-19 |
| 复合通算融合 | 多阶段通信+计算（如 AllToAll→AllGather→Matmul） | MC2-01~04, MC2-18~19 |
| 量化+集合通信 | 量化计算 + AllReduce/ReduceScatter | MC2-09~12, MC2-18~19 |

---

## 规则快查索引

| 条例编号 | 规则名称 | 严重级别 | 适用范围 | PR 触发关键词 |
|---------|---------|---------|---------|-------------|
| MC2-01 | 核间同步必要性 | `[红线]` | Kernel侧 | `for.*round`, `hccl_.*AllGather\|AllReduce\|ReduceScatter\|AlltoAll` |
| MC2-02 | 流同步正确性 | `[红线]` | Kernel侧 | `SetFlag`, `WaitFlag`, `PipeBarrier`, `SyncFunc` |
| MC2-03 | SyncAll 同步生效 | `[红线]` | Kernel侧 | `SyncAll` |
| MC2-04 | 全局操作一致性 | `[红线]` | Host/Kernel | `hccl_`, `rankId` 条件分支 |
| MC2-05 | 专家索引边界检查 | `[红线]` | Kernel侧 | `expertIds`, `expertIdx` |
| MC2-06 | 专家分发组合一致性 | `[红线]` | Host/Kernel | `dispatch`, `combine`, `localMoeExpertNum` |
| MC2-07 | 专家参数校验与日志 | `[红线]` | Host侧 | `moeExpertNum`, `OP_LOGE` |
| MC2-08 | MoE 属性获取规范 | `[红线]` | Host侧 | `GetAttrPointer` |
| MC2-09 | 量化参数类型一致性 | `[红线]` | Host/Kernel | `quant`, `scale`, `%ld`/`%lu` |
| MC2-10 | 量化模式校验完整性 | `[红线]` | Host侧 | `BasicQuantMode`, `quantMode` |
| MC2-11 | 量化精度保护 | `[红线]` | Kernel侧 | `Cast`, `FP8`, `BF16` |
| MC2-12 | EP/TP 配置类型匹配 | `[红线]` | Host侧 | `epWorldSize`, `tpWorldSize` |
| MC2-13 | CCU 通信数据量限制 | `[红线]` | Host侧 | `CCU`, `256MB` |
| MC2-14 | Tiling 校验与结构规范 | `[设计原则]` | Host侧 | `tiling`, `struct`, `stride` |
| MC2-15 | PTA 通信域字符串深拷贝 | `[红线]` | Host侧 | `groupName`, `hcom` |
| MC2-16 | 禁止直接引用 kernel_operator.h | `[红线]` | Kernel侧 | `kernel_operator.h` |
| MC2-17 | HCCL API 参数查阅验证 | `[红线]` | Host/Kernel | `Mc2CcTilingConfig`, `GetHcclContext` |
| MC2-18 | HCCL 通信生命周期与参数 | `[红线]` | Host/Kernel | `InitV2`, `Finalize`, `HcclDataType` |
| MC2-19 | AlltoAllV 跨 rank 参数一致性 | `[红线]` | Host/Kernel | `AlltoAllV`, `sendCounts` |

---

## 一、通信同步规则

> **触发条件**：PR 差异中出现 `hccl_` / `AllGather` / `AllReduce` / `ReduceScatter` / `AlltoAll` / `SyncAll` / `SetFlag` / `WaitFlag` / `PipeBarrier` / `SyncFunc`
>
> **检视原则**：MC² 算子融合了计算与通信，多轮流水线中同步是首要风险点。检视时追踪每轮计算→通信→计算的数据流，确认同步屏障存在且正确。
>
> **同步机制速查**（参见术语表和 ascendc-sync-audit skill）：
> - **TPipe EnQue/DeQue**：Stage 间自动同步，精度最高
> - **SyncFunc<HardEvent>**：精确到流水对的硬件事件同步
> - **SetFlag/WaitFlag<HardEvent>**：ISASI 精确同步，必须成对、同 EVENT_ID
> - **PipeBarrier<PIPE>**：单流水屏障，PIPE_V 内部不保证自动保序
> - **SyncAll**：全核同步屏障，开销最大

### MC2-01: 核间同步必要性 `[Kernel]` `[红线]`

**规则**：多轮计算和集合通信之间，必须增加核间同步，防止上一轮计算结果未写入通信缓冲区就启动通信。

**检测模式**：
- 在 `for.*round` / `while.*loop` 循环体内，定位 `hccl_.*AllGather\|AllReduce\|ReduceScatter\|AlltoAll` 通信调用
- 检查每轮通信前是否有 `SyncFunc` 或 `SetFlag+WaitFlag` 或 `PipeBarrier`

**判断标准**：
- 循环中通信调用前缺少同步屏障 → **风险**（数据踩踏）
- 每轮通信前有 SyncFunc 或 SetFlag+WaitFlag → **安全**

**错误示例**（循环中通信缺少同步屏障）：
```cpp
for (int i = 0; i < roundNum; i++) {
    ComputeRound();
    hccl_.AllGather<true>(sendGM, recvGM, count, dataType, rankCnt, tileCnt);  // 缺少同步
}
```

**正确示例**：
```cpp
for (int i = 0; i < roundNum; i++) {
    ComputeRound();
    SyncFunc<HardEvent::S_MTE3>();  // 确保计算结果写入通信缓冲区
    hccl_.AllGather<true>(sendGM, recvGM, count, dataType, rankCnt, tileCnt);
}
```

---

### MC2-02: 流同步正确性 `[Kernel]` `[红线]`

**规则**：SetFlag/WaitFlag 必须顺序正确（SetFlag 先于 WaitFlag）、EVENT_ID 匹配；依赖特定 buffer 的操作必须有对应 PipeBarrier。PIPE_V 内部操作同一 tensor 重叠区域时必须有 `PipeBarrier<PIPE_V>()`。

**检测模式**：
- 定位所有 `SetFlag.*HardEvent` / `WaitFlag.*HardEvent`，检查同对使用相同 EVENT_ID 且 SetFlag 先于 WaitFlag
- 定位 `PipeBarrier`，统计同一函数中连续 PipeBarrier<PIPE_V> 数量
- 对 Add/ReduceSum 等依赖 broadcastBuffer/reduceBuffer 的操作，检查前是否有 PipeBarrier<PIPE_V>

**判断标准**：
- WaitFlag 先于 SetFlag → **风险**（死等）
- 同对 SetFlag/WaitFlag EVENT_ID 不匹配 → **风险**（同步失效）
- Add/ReduceSum 依赖 buffer 但缺少 PipeBarrier<PIPE_V> → **风险**（数据依赖未保护）
- 连续超过 3 个 PipeBarrier<PIPE_V> → **性能问题**（应分析是否可合并）
- 同一 PIPE_V 内两条指令操作同一 tensor 重叠区域但无 PipeBarrier<PIPE_V> → **风险**

**错误示例**：
```cpp
// 问题一：WaitFlag先于SetFlag → 死等
WaitFlag<HardEvent::MTE1_M>(EVENT_ID);
SetFlag<HardEvent::MTE1_M>(EVENT_ID);

// 问题二：Add依赖reduceBuffer但缺少PipeBarrier<PIPE_V>
Add(output, reduceBuffer, tempBuffer);  // reduceBuffer数据未就绪
```

**正确示例**：
```cpp
SetFlag<HardEvent::MTE2_S>(EVENT_ID7);  // SetFlag先执行
WaitFlag<HardEvent::MTE2_S>(EVENT_ID7); // 同对相同EVENT_ID

// Add依赖buffer时必须有PipeBarrier<PIPE_V>
PipeBarrier<PIPE_V>();
Add(output, reduceBuffer, tempBuffer);
```

---

### MC2-03: SyncAll 同步生效 `[Kernel]` `[红线]`

**规则**：通算并行场景下，SyncAll 必须确保所有核（含 V 核）同步生效，否则计算和后续操作在 UB 互相踩踏。

**检测模式**：定位 `SyncAll` 调用 — 在通算并行场景中，确认每轮流水线关键位置有 SyncAll。同时验证所有核从入口到出口经过的 SyncAll 总次数相等，否则有挂死风险。

**判断标准**：
- 通算并行场景中缺少 SyncAll → **风险**（V核数据踩踏，精度问题或卡死）
- 不同核的 SyncAll 总次数不一致 → **风险**（挂死）
- SyncAll 在每轮流水线关键位置存在且各核次数一致 → **安全**

**正确示例**：
```cpp
SyncAll();  // 通算并行场景中关键位置调用，确保所有核同步
```

**SyncAll 路径计数方法**：
- 列出所有从入口到 return 的独立执行路径（按核类型、coreIdx 条件分支展开）
- 对每条路径递归进入所有函数，收集 SyncAll 调用点
- 逐路径比较总数，必须一致
- **注意**：若 SyncAll 出现在循环中，需按循环次数展开后计入总数。若循环次数由变量控制，需确认该变量在所有核上取值相同

---

### MC2-04: 全局操作一致性 `[Host/Kernel]` `[红线]`

**规则**：AllGather / AllReduce / AllToAll 等全局通信操作，所有 rank 的参数必须一致。不同 rank 传不同数据量会导致集合通信卡死或结果错误。

**检测模式**：定位 `hccl_.*AllGather\|AllReduce\|ReduceScatter\|AlltoAll` 调用 — 追踪参数来源是否有 `rankId` 条件分支导致参数差异。

**判断标准**：
- 通信参数在 rank 条件分支中不同 → **风险**（集合通信卡死）
- 新增条件分支导致某些硬件平台下参数获取失败 → **风险**
- 所有 rank 参数一致且所有平台路径可达 → **安全**

**错误示例**：
```cpp
if (rankId == 0) {
    hccl_.AllGather<true>(sendGM, recvGM, dataSize, dataType, rankCnt, tileCnt);
} else {
    hccl_.AllGather<true>(sendGM, recvGM, dataSize / 2, dataType, rankCnt, tileCnt);  // 不一致 → 卡死
}
```

---

## 二、MoE 专家路由规则

> **触发条件**：PR 差异中出现 `expert` / `expertIds` / `moeExpertNum` / `dispatch` / `combine` / `localMoeExpertNum` / `GetAttrPointer`
>
> **检视原则**：MoE 架构涉及专家路由与分发组合，追踪专家索引数据流：路由→分发→计算→组合，确保边界、一致性和参数校验。

### MC2-05: 专家索引边界检查 `[Kernel]` `[红线]`

**规则**：专家索引数组访问前必须进行边界检查，防止越界访问导致 UB 访问越界或计算错误。

**检测模式**：定位 `expertIds` / `expertIdx` 数组访问，确认访问前有范围校验。

**判断标准**：
- 数组访问无范围检查 → **风险**（越界访问）
- 有 `if (moeExpertNum > MAX_EXPERT_NUM)` 校验 → **安全**

**错误示例**：
```cpp
int expertIdx = expertIds[tokenIdx];  // 无边界保护
```

**正确示例**：
```cpp
if (moeExpertNum > MAX_EXPERT_NUM) {
    OP_LOGE(opName, "moeExpertNum %ld exceeds limit %ld", moeExpertNum, MAX_EXPERT_NUM);
    return ge::GRAPH_FAILED;
}
int expertIdx = expertIds[tokenIdx];
```

---

### MC2-06: 专家分发组合一致性 `[Host/Kernel]` `[红线]`

**规则**：Dispatch 和 Combine 操作的专家对应关系必须一致，参数配置必须匹配。

**检测模式**：比对 Dispatch 和 Combine 两侧的 tiling 参数（`localMoeExpertNum`、`sharedExpertNum` 等），确认值和命名一致。

**判断标准**：
- 两侧参数值不一致 → **风险**（跨 server 通信失败）
- 参数命名不统一 → **风险**（理解歧义）
- 参数一致且命名统一 → **安全**

---

### MC2-07: 专家参数校验与日志 `[Host]` `[红线]`

**规则**：moeExpertNum、sharedExpertNum 等 MoE 参数必须进行合法性校验；日志信息必须精确——包含参数名和实际值，拼写和格式化字符串必须正确。

**检测模式**：定位 `OP_LOGE.*expert` / `OP_LOGD.*expert` — 检查日志拼写、格式化字符串与变量类型是否匹配。

**判断标准**：
- 拼写错误 → **修正**
- 格式化字符串与变量类型不匹配（int64_t 用 %lu → 应用 %ld）→ **修正**
- 错误日志缺少参数名和实际值 → **补充**
- 错误日志写明具体原因 → **安全**

**错误示例**：
```cpp
OP_LOGE(opName, "The expert Num is lager than MAX, value %lu", moeExpertNum);
// lager → larger（拼写错误）；int64_t 应使用 %ld 而非 %lu
```

**检视要点补充**：重复调用的字符串长度计算应缓存结果避免重复计算。

---

### MC2-08: MoE 属性获取规范 `[Host]` `[红线]`

**规则**：MoE 相关属性必须从 context 获取（GetAttrPointer），获取类型必须与 IR 原型定义一致。

**检测模式**：定位 `GetAttrPointer` — 对比模板参数类型与 IR 原型定义。

**判断标准**：
- `GetAttrPointer<int>` 与 IR 定义 `int64_t` 不一致 → **风险**（类型截断）
- 模板参数与 IR 定义一致 → **安全**

---

## 三、量化精度规则

> **触发条件**：PR 差异中出现 `quant` / `BasicQuantMode` / `PERTENSOR` / `PERCHANNEL` / `MX/pergroup` / `scale` / `pertensor` / `perblock` / `Cast` / `FP8` / `BF16` / `%ld` / `%lu`

### MC2-09: 量化参数类型一致性 `[Host/Kernel]` `[红线]`

**规则**：quantMode、scale 等量化参数的类型声明必须与使用一致，格式化字符串必须匹配变量类型。

**检测模式**：对每个量化参数变量，从声明追踪到所有使用点（打印、校验、赋值），定位声明再追踪使用点。

**判断标准**：
- 类型链路不一致（声明 int64_t 但打印用 %d）→ **风险**（截断）
- bool 变量做 uint32 类型转换 → **风险**（多余转换）
- 类型一致且格式化字符串匹配 → **安全**

**类型映射参考**：

| 变量类型 | 正确格式化 | 错误格式化 |
|---------|----------|----------|
| `int64_t` | `%ld` | `%d`(截断) / `%lu`(类型不匹配) |
| `uint64_t` | `%lu` | `%d`(截断) / `%ld`(类型不匹配) |
| `int32_t` | `%d` | `%ld`(无意义扩展) |
| `bool` | 直接判断 | `%d`(多余类型转换) |

---

### MC2-10: 量化模式校验完整性 `[Host]` `[红线]`

**规则**：量化模式校验必须覆盖所有支持的模式，错误信息必须包含参数名和实际值。

> **⚠️ Agent 执行此规则时，必须先通过 `/ascendc-docs-search` 查询当前 CANN 版本支持的量化模式枚举值，确保校验分支覆盖所有有效值。以下静态表仅供开发者快速参考，可能未涵盖最新模式。**

**常见量化模式枚举**（仅示例，非全集）：

| 枚举值 | 含义 | 说明 |
|--------|------|------|
| `PERTENSOR_MODE` | per-tensor 量化 | 整个 tensor 共用一个 scale |
| `PERCHANNEL_MODE` | per-channel 量化 | 每个 channel 独立 scale |
| `PERTOKEN_MODE` | per-token 量化 | 每个 token 独立 scale |
| `PERBLOCK_MODE` | per-block 量化 | 每个 block 独立 scale |
| `MX_PERGROUP_MODE` | MX per-group 量化 | 按 group 量化，MX 格式 |

> 注意：枚举值名称可能随版本迭代新增，检视时必须使用 `/ascendc-docs-search` skill 查阅官方文档确认当前支持的完整列表。

**检测模式**：
1. 使用 `/ascendc-docs-search` 查询当前 CANN 版本支持的完整 `BasicQuantMode` 枚举列表
2. 结合算子规格（ops_info）确认当前算子实际支持的量化模式子集，仅对子集内模式做校验覆盖
3. 定位 `quantMode` / `BasicQuantMode` 校验分支，与查询到的完整列表及算子规格逐项对比

**判断标准**：
- 校验分支遗漏某个量化模式 → **风险**（运行时未覆盖错误）
- 错误日志缺少参数名和实际值 → **风险**（无法诊断）
- 格式化字符串占位符数量与参数数量不匹配 → **风险**（信息丢失）

**错误示例**：
```cpp
OP_LOGE(opName, "Quant template does not support this mode.");  // 缺少实际值
OP_LOGE(opName, "value=%d", QUANT_MX, actualMode);  // 1个%d，2个参数 → actualMode丢失
```

**正确示例**：
```cpp
OP_LOGE(opName, "Quant mode %ld is not supported, valid modes: PERTENSOR(%ld), MX_PERGROUP(%ld)",
        actualMode, PERTENSOR_MODE, MX_PERGROUP_MODE);
```

**检视要点补充**：quantMode 新增时 tilingKey 设计存在冲突风险，建议使用不同设计模式避免编号冲突。

---

### MC2-11: 量化精度保护 `[Kernel]` `[红线]`

**规则**：量化/反量化操作必须考虑中间精度保护；高精度向低精度 cast 应使用饱和模式。

**检测模式**：定位 `Cast.*FP8\|Cast.*BF16\|Cast.*INT8` — 确认低精度 cast 使用饱和模式或中间精度保护。

**判断标准**：
- 非饱和 cast 导致 FP8/BF16 精度异常 → **风险**
- 量化与非量化路径精度保护不一致 → **风险**
- 使用饱和模式或中间精度保护 → **安全**

---

### MC2-12: EP/TP 配置类型匹配 `[Host]` `[红线]`

**规则**：epWorldSize、tpWorldSize 等配置参数的类型必须匹配，避免溢出。

**检测模式**：追踪 `epWorldSize` / `tpWorldSize` / `epRankId` / `tpRankId` / `rankId` 变量类型链路，确认循环变量与配置参数类型兼容。

**判断标准**：
- 循环变量 int 与 uint64_t 参数比较 → **风险**（大值溢出）
- uint32_t 偏移量在大规模推理场景 → **风险**（溢出）
- 类型匹配且足够大 → **安全**

**错误示例**：
```cpp
for (int i = 0; i < epWorldSize * tpWorldSize; i++) {  // int × uint64_t 可能溢出
uint32_t offset = baseSize * rankId;           // 大规模推理 uint32_t 可能溢出
```

---

## 四、硬件约束规则

> **触发条件**：PR 差异中出现 `CCU` / `AICPU` / `256.*1024.*1024` / `stride` / `contiguous` / `tiling.*struct`

### MC2-13: CCU 通信数据量限制 `[Host]` `[红线]`

**规则**：CCU 模式的集合通信单次传输数据量不得超过当前硬件限制，超限会导致算子卡死。

> 注意：当前已知 CCU 传输上限为 256MB，但该值可能随硬件迭代调整。Agent 应使用 `/ascendc-docs-search` 查询当前设备的实际限制值，并以查询结果为准。

**检测模式**：计算 `数据量 = 元素数 × dtype字节数 × rank数`，使用 `/ascendc-docs-search` 查询当前设备 CCU 传输上限，判断是否超限。搜索 Tiling 中是否有拆分策略。

**判断标准**：
- 单次通信数据量超过当前硬件限制且无拆分策略 → **风险**（卡死）
- 有拆分策略或使用 AICPU 模式规避 → **安全**

**示例**（CCU workspace 约束）：
```cpp
size_t sysWorkspaceSize = 256 * 1024 * 1024;  // CCU workspace 约束
```

---

### MC2-14: Tiling 校验与结构规范 `[Host]` `[设计原则]`

**规则**：校验逻辑应优先放在 Tiling 公共流程中（接口层尽量薄）；tiling struct 需对齐；MC2 算子需校验非连续输入；tiling 并发需安全。

**检测模式**：
- 定位 `worldSize.*>` / `worldSize.*<=` — 过度校验应移至 Tiling
- 定位 `struct.*TilingData` — 检查内存对齐
- 定位 `stride` / `contiguous` — 非连续输入校验
- 定位 `class.*Tiling.*{` — tiling 类成员变量读写需并发安全

**判断标准**：
- worldSize 范围校验（如 > 0 && <= 8）→ **过度校验**，应只做 != 0
- tiling struct 未对齐 → **风险**（内存布局问题）
- MC2 算子未检查非连续输入 → **风险**（通信数据错误）
- tiling 类成员变量在多线程读写 → **风险**（需并发安全保护）
- 特殊硬约束可放接口层，但 Tiling 必须兜底 → **安全**

**检视要点补充**：除0场景必须在 Tiling 中拦截，不应依赖接口层。

**检视要点补充**：tiling struct 需对齐；tilingData 类型需校验；需调整至版本对比隔离。

---

## 五、HCCL 通信与安全规范

> 本章合并所有 HCCL 相关规则（安全 MC2-15、编译 MC2-16、API 使用 MC2-17、通信生命周期 MC2-18、参数一致性 MC2-19），便于一次性获取 HCCL 全部检视要点。
>
> **触发条件**：PR 差异中出现 `groupName` / `hcom` / `kernel_operator.h` / `ASC_DEVKIT_MAJOR` / `Mc2CcTilingConfig` / `InitV2` / `Finalize` / `HcclDataType` / `AlltoAllV` / `sendCounts` / `recvCounts`

### MC2-15: PTA 通信域字符串深拷贝 `[Host]` `[红线]`

**规则**：PTA 层传递通信域字符串必须深拷贝（std::string），禁止 const char* 浅拷贝；通信域接口命名不应区分特定 SOC 类型。

**检测模式**：
- 定位 `groupName` / `hcom` / `const char*` — 确认使用 std::string 深拷贝而非 const char*
- 定位接口命名含特定 SOC 代际标识的 — 应改为通用命名

**判断标准**：
- const char* 浅拷贝 → **风险**（异步执行时字符串被释放导致乱码）
- 接口命名含特定 SOC 代际标识 → **建议修改**为通用命名
- std::string 深拷贝 → **安全**

> groupName 在 tiling struct 中为定长字符数组，长度通常由 `MAX_GROUP_NAME_LENGTH` 宏指定。Agent 检查时应以该宏实际值为准而非硬编码长度。Host 侧赋值时需深拷贝为 `std::string` 防止异步释放。

**正确示例**：
```cpp
groupEp = std::string(groupEpPtr);  // std::string 深拷贝
```

**检视要点补充**：多通信域场景下，建议使用 `std::unordered_set<std::string>` 保存 groupName，提升扩展性。

---

### MC2-16: 禁止直接引用 kernel_operator.h `[Kernel]` `[红线]`

**规则**：禁止直接引用 kernel_operator.h，应改为按需引用拆分后的 API 头文件，并使用 `ASC_DEVKIT_MAJOR` 宏进行版本隔离。

> 注意：拆分头文件策略随 CANN 版本演进可能变化，检视时应使用 `/ascendc-docs-search` skill 查阅当前版本的头文件拆分方案。

**检测模式**：定位 `#include.*kernel_operator.h` — 确认是否使用 `ASC_DEVKIT_MAJOR` 宏包裹。

**判断标准**：
- 未被宏包裹的 kernel_operator.h 引用 → **风险**（编译耗时且无版本隔离）
- 使用 ASC_DEVKIT_MAJOR 宏包裹 → **安全**

**修复方案**：
```cpp
// 方案一：完整引用（包含 cube + vec）
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

// 方案二：按需引用 - 仅需同步功能
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_operator_block_sync_intf.h"
#else
#include "kernel_operator.h"
#endif
```

**拆分头文件参考**（仅供参考，具体拆分方案以 `/ascendc-docs-search` 查询结果为准）：`kernel_basic_intf.h`（完整）、`kernel_operator_block_sync_intf.h`（同步）、`kernel_operator_data_copy_intf.h`（数据搬运）、`kernel_operator_vec_unary_intf.h`（向量计算）、`kernel_operator_mm_intf.h`（矩阵乘法）

> **⚠️ CANN 版本号会持续递增，修复方案中的 `ASC_DEVKIT_MAJOR >= 9` 仅为示例。Agent 应通过 `/ascendc-docs-search` 获取当前版本的头文件拆分策略及正确的条件编译版本号。**

---

### MC2-17: HCCL API 参数动态查阅验证 `[Host/Kernel]` `[红线]`

**规则**：检视中遇到任何 HCCL 通信 API 时，Agent 必须立即调用 `/ascendc-docs-search` 获取该 API 最新文档，并验证参数类型、数量、约束是否与代码一致；生命周期调用顺序是否正确；数据类型、归约操作是否在当前版本支持。

**禁止使用本 Skill 中任何静态映射表作为最终判断依据。** 凡未经过动态查询确认的参数使用，均视为**潜在风险**（可能因版本演进失效）。

**检测模式**：
1. 提取代码中所有 HCCL API 调用
2. 对每个 API 执行 `/ascendc-docs-search <API名称>`
3. 将文档参数与代码逐项对比

**判断标准**：
- API 参数变更但注释/文档未同步更新 → **风险**（信息不一致）
- 文档数据类型列表与代码 dtype 校验不一致 → **风险**（下游用户误用）
- 文档与代码一致且查阅了官方文档 → **安全**

---

### MC2-18: HCCL 通信生命周期与参数 `[Host/Kernel]` `[红线]`

> **⚠️ 非阻塞通信前置注意事项**：当 HCCL 通信调用模板参数为 `false`（非阻塞模式）时，**必须**在通信调用后显式执行 `hccl_.Commit(handleId)` 和 `hccl_.Wait(handleId)`。缺少 Commit 或 Wait 均视为**严重风险**——可能导致数据竞争或通信结果不可用。检视时优先排查此场景。

**规则**：HCCL 通信操作必须遵守生命周期（Init → SetCcTiling → 通信 → Commit → Wait → Finalize）；HcclDataType 必须与实际数据类型匹配；HcclReduceOp 必须与算子语义一致。

**检测模式**：
- 定位 `InitV2` / `Init` / `Finalize` / `SetCcTilingV2` / `SetCcTiling` / `Commit` / `Wait` — 确认生命周期完整且配对
- 定位 `HcclDataType` / `HCCL_DATA_TYPE` / `HcclReduceOp` / `HCCL_REDUCE` — 确认类型匹配

**判断标准**：
- Init 缺少 Finalize 配对 → **风险**（资源泄漏）
- 非阻塞通信调用（模板参数为 `false`）后，必须显式调用 `hccl_.Commit(handle)` 和 `hccl_.Wait(handle)`，否则视为**严重风险**（数据竞争）
- 阻塞通信调用（模板参数为 `true`）自动包含 Commit/Wait，但仍需确认返回值使用正确
- HcclDataType 与实际 buffer 数据类型不匹配 → **风险**（通信数据错误）
- HcclReduceOp 与算子语义不一致 → **风险**
- 生命周期完整且参数匹配 → **安全**

**HCCL 生命周期使用模式**：
```cpp
// Kernel侧完整生命周期
hccl_.InitV2(AscendC::GetHcclContext<0>(), &(tiling_->mc2InitTiling));
hccl_.SetCcTilingV2(offsetof(TilingDataType, mc2CcTiling));
// ... 通信操作 ...
hccl_.Commit(handleId);
hccl_.Wait(handleId);
hccl_.Finalize();
```

**HcclDataType 映射参考**：

> **⚠️ 以下映射表仅供开发者快速参考，可能未涵盖最新数据类型。Agent 在执行检视时必须使用 `/ascendc-docs-search` 确认实际支持的类型和枚举值，禁止将本表作为最终判断依据。**

| 实际数据类型 | 对应 HcclDataType |
|------------|-----------------|
| `float16` (half) | `HCCL_DATA_TYPE_FP16` |
| `bfloat16` | `HCCL_DATA_TYPE_BFP16` |
| `float32` | `HCCL_DATA_TYPE_FP32` |
| `int8_t` | `HCCL_DATA_TYPE_INT8` |
| `int32_t` | `HCCL_DATA_TYPE_INT32` |
| `int64_t` | `HCCL_DATA_TYPE_INT64` |
| `fp8e4m3` | `HCCL_DATA_TYPE_FP8E4M3` |
| `fp8e5m2` | `HCCL_DATA_TYPE_FP8E5M2` |

> 此表为常见类型速查，**非穷举**。新增数据类型支持时必须使用 `/ascendc-docs-search` skill 确认。若查询结果与本表不一致，以官方文档为准。

**HcclReduceOp 映射参考**：

> **⚠️ 同上，本表仅供开发者快速参考，Agent 必须以 `/ascendc-docs-search` 查询结果为准。**

| 算子语义 | 对应 HcclReduceOp |
|---------|-----------------|
| 求和归约 / 求和分发 | `HCCL_REDUCE_SUM` |
| 最大值归约 | `HCCL_REDUCE_MAX` |

**CCU/AICPU 通信引擎选择**：

> **⚠️ 通信引擎类型可能随硬件平台演进新增。Agent 应查询 `/ascendc-docs-search` 获取当前设备支持的通信引擎类型及选择策略。以下仅为常见示例。**

| 模板参数 | 通信引擎 | 适用场景 |
|---------|---------|---------|
| `HcclServerType::HCCL_SERVER_TYPE_CCU` | CCU | 小数据量高性能，单次≤256MB |
| `HcclServerType::HCCL_SERVER_TYPE_AICPU` | AICPU | 大数据量，无 256MB 限制 |

**HCCL 对象声明**（最简方式，具体引擎类型按需选择）：
```cpp
Hccl<HcclServerType::HCCL_SERVER_TYPE_CCU> hccl_;  // CCU模式
// 或
Hccl<HcclServerType::HCCL_SERVER_TYPE_AICPU> hccl_;  // AICPU模式
```
> 实际项目中可能通过模板参数或封装选择引擎，具体参考官方文档，此处仅展示最简声明方式。

**Kernel侧通信方法调用格式参考**：

> **⚠️ 参数格式可能随版本演进变化。Agent 验证通信参数时必须使用 `/ascendc-docs-search` 查询当前版本的实际 API 原型。以下仅为常见格式示例。**

| 方法 | 模板参数 | 参数格式 |
|------|---------|---------|
| `hccl_.AllGather<bool>` | `true`=阻塞, `false`=非阻塞 | `(sendGM, recvGM, count, dataType, rankCnt, repeatCount)` |
| `hccl_.AllReduce` | 无 | `(sendGM, recvGM, count, dataType, reduceOp, repeatCount)` |
| `hccl_.ReduceScatter<bool>` | `false`=非阻塞 | `(sendGM, recvGM, count, dataType, reduceOp, rankDataCnt, repeatCount)` |
| `hccl_.AlltoAll<bool>` | `false`=非阻塞 | `(sendGM, recvGM, count, dataType)` |
| `hccl_.AlltoAllV<bool>` | `true`=阻塞 | `(sendAddr, sendCounts[], sendOffsets[], sendDataType, recvAddr, recvCounts[], recvOffsets[], recvDataType)` |

> 此表为常见调用格式速查，完整参数约束需查阅官方文档。

**Tiling 数据结构**：
```cpp
// 典型 MC² Tiling Data 包含：
Mc2InitTiling mc2InitTiling;    // HCCL 初始化参数
Mc2CcTiling mc2CcTiling;        // HCCL 通信配置参数
Mc2CcTiling mc2CcTilingComm;    // 多通信域场景的额外配置（可选）
```

---

### MC2-19: AlltoAllV 跨 rank 参数一致性 `[Host/Kernel]` `[红线]`

**规则**：AlltoAllV 的 sendCounts / recvCounts / sdispls / rdispls 在所有 rank 间必须符合预期的一致性规则。

**检测模式**：定位 `AlltoAllV` / `AlltoAllv` / `sendCounts` / `recvCounts` / `alltoAllvSendCnt` / `alltoAllvRecvCnt` — 追踪参数来源，确认不同 rank 间参数是否符合预期。

**判断标准**：
- sendCounts/recvCounts 在不同 rank 间不一致且无合理理由 → **风险**（通信结果错误）
- sendType 与 recvType 不匹配 → **需确认**是否为预期设计
- 参数来源依赖 rankId 条件分支 → **需确认**所有 rank 参数总和一致

**AlltoAllV 使用模式**：
```cpp
// Commit 后必须 Wait 确保数据就绪
auto handleId = hccl_.template AlltoAllV<true>(
    (__gm__ uint8_t*)sendAddr, alltoAllvSendCnt, alltoAllvSendOffset, hcclDataType,
    (__gm__ uint8_t*)recvAddr, alltoAllvRecvCnt, alltoAllvRecvOffset, hcclDataType);
hccl_.Commit(handleId);
hccl_.Wait(handleId);  // 必须Wait确保数据就绪
```

---

## 搜索关键词总表

> **⚠️ 本表为常见关键词示例，非穷举列表。** HCCL API 随版本演进可能新增/变更，Agent 不得仅依赖本表判定 API 归属，涉及疑似新 API 时必须使用 `/ascendc-docs-search` 动态查询确认。

| 规则分类 | 搜索关键词 |
|---------|-----------|
| 通信同步 | `hccl_`, `AllGather`, `AllReduce`, `ReduceScatter`, `AlltoAll`, `SyncAll`, `SetFlag`, `WaitFlag`, `PipeBarrier`, `SyncFunc` |
| MoE 专家 | `expert`, `expertIds`, `moeExpertNum`, `dispatch`, `combine`, `localMoeExpertNum`, `sharedExpertNum`, `GetAttrPointer` |
| 量化精度 | `quant`, `BasicQuantMode`, `PERTENSOR_MODE`, `PERCHANNEL_MODE`, `MX/pergroup`, `scale`, `pertensor`, `perblock`, `Cast`, `FP8`, `BF16`, `%ld`, `%lu` |
| 硬件约束 | `CCU`, `AICPU`, `256.*1024.*1024`, `stride`, `contiguous`, `struct.*TilingData` |
| HCCL 通信与安全 | `InitV2`, `Finalize`, `SetCcTiling`, `GetHcclContext`, `Commit`, `Wait`, `HcclDataType`, `HcclReduceOp`, `AlltoAllV`, `sendCounts`, `groupName`, `hcom`, `Mc2CcTilingConfig`, `Mc2InitTiling`, `Mc2CcTiling`, `HcclTypeSelector`, `kernel_operator.h`, `ASC_DEVKIT_MAJOR` |

---

## Agent 行为指南

### 领域判定阶段
使用核心特征（C1~C2）判定代码是否属于 MC² 领域。若遇到疑似 HCCL 接口但不认识，调用 `/ascendc-docs-search` 确认是否属于集合通信范畴，确认后纳入检视范围。

### 规则匹配阶段
对照差异代码关键词，参照速查表快速定位规则子集。但所有 HCCL 参数正确性验证必须触发 `/ascendc-docs-search`，禁止以静态映射表作为最终判断依据。

### 静态表使用原则
- 仅供开发者快速参考，Agent 不得将其视为权威数据
- 若 `/ascendc-docs-search` 查询到的文档与静态表不一致，以文档为准，并建议更新本 Skill 的静态表

### 版本演进应对
- HCCL API、数据类型、枚举值可能随 CANN 版本新增/变更
- 硬件约束（如 CCU 256MB 限制）可能随硬件迭代调整
- 头文件拆分策略可能随版本变化
- 所有涉及版本敏感的信息，检视时必须查阅官方文档确认当前版本的实际约束

---

## 规则版本

- **v1.0**（2026-05-22）