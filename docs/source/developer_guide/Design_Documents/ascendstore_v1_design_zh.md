# AscendStore v1 架构与基础设施设计

AscendStore 将模型计算产生的 KV 缓存保存到外部存储，让后续请求按内容复用已有数据。v1 的业务层负责决定本次请求要查询、加载或保存什么，再组合基础设施提供的可达性计算、布局映射、消息通信、执行协调和资源管理能力完成操作。

这套划分保留两条清楚的业务主线：Scheduler 根据请求与缓存分配形成命令，Worker 根据命令选择对象、组织传输并处理结果。可复用的计算和机制集中在 Infrastructure 中，但请求归属、对象发布条件和来源释放判断仍由业务层负责。

本文说明各部分的职责、协作关系与关键约束。文中的地址和布局示例均以字节为单位。

## 1. 总体架构

![AscendStore v1 分层架构，展示接入、业务、共享契约、基础设施、Backend 适配和第三方后端的职责归属](../../assets/ascendstore_v1/architecture_layers.svg)

vLLM 负责请求调度、缓存分配与模型执行。`AscendStoreV1Connector` 适配框架钩子，在 Scheduler 和 Worker 两种角色下分别实例化。`factory.py` 根据已经解析和校验的配置装配具体组件。

业务层由 Scheduler 和 Worker 构成。Scheduler 持有请求进度、采用的命中范围和缓存分配信息；Worker 使用已经分配好的 KV Tensor，持有本地 Mapping 与执行上下文，并处理传输结果。Bulk 按完整对象传输，Layerwise 将对象的传输分解到物理层。逐层访问又分为 KeyRange 和 GVA：前者按对象 key 与字节范围访问，后者按全局地址访问。

Protocol 是 Connector、Scheduler 与 Worker 共享的业务契约，独立于基础设施目录。它定义传输命令、Lookup 请求与结果及来源释放回执，并负责 Lookup 的编解码、校验与失败时的未命中回退。

基础设施层提供可组合的能力。Configuration 与 Topology 确定静态配置和本地拓扑，RPC 管理字节消息的请求应答与连接生命周期，Coordinator 判断哪些缓存可以支持恢复，Mapping 描述对象身份与字节位置，Execution 安排调用的执行顺序，Resources 管理需要显式释放的资源。业务路径按需组合这些能力，例如 Worker 的 Lookup 使用 Coordinator、Mapping 的命名能力和 Backend。

Backend 适配层位于独立的 `backend/` 目录。它定义整对象、KeyRange 和 GVA 的访问约定，并将其转换为 Mooncake、Memcache 原生客户端调用。第三方后端层包含这些客户端、传输能力与外部存储服务。

Worker 直接调用 Backend 完成读写，Resources 持有 Backend 及其资源句柄。图中按职责分组，不表示逐层转发的调用顺序。

## 2. 代码结构

以下目录以 `ascend_store/v1/` 为根。目录树展示主要职责，省略包导出文件和部分辅助文件。

```text
v1/
├── connector.py                 vLLM 钩子适配与两侧角色入口
├── factory.py                   配置准入后的实例装配
├── protocol/
│   ├── messages.py              共享命令、Lookup 消息与来源释放回执
│   └── lookup.py                Lookup 编解码、校验与未命中回退
├── scheduler/
│   ├── scheduler.py             Lookup 采用、分配确认、每步命令规划
│   ├── lookup.py                请求侧命中候选与重算范围处理
│   ├── state.py                 请求与规划进度
│   └── source_leases.py         需要额外保护的 Store 来源引用
├── worker/
│   ├── base.py                  每步业务上下文、批次、完成与来源释放
│   ├── bulk/                   整对象 Load / Store
│   ├── layerwise/              逐层业务；KeyRange / GVA 两种访问方式
│   └── transfer/               对象行、请求归属、批次与结果证据
├── infrastructure/
│   ├── configuration.py        静态配置、缓存事实与分区约束
│   ├── topology.py             分区类型、本地 rank 与缓存层绑定
│   ├── rpc.py                  字节帧请求应答、超时与连接 / 线程生命周期
│   ├── coordinator.py          单组 / 多组缓存的恢复条件与选择掩码
│   ├── mapping.py              Key / Local / Remote 数据与 Layer 段区间关系
│   ├── execution.py            串行执行器与一步内的逐层时序协调
│   └── resources.py            执行器、对象会话、注册与客户端所有权
└── backend/
    ├── interfaces.py           整对象、KeyRange、GVA 访问契约
    ├── registration.py         注册句柄与失败回滚
    ├── results.py              原生结果校验与规范化
    ├── mooncake/               Mooncake 客户端与对象会话适配
    └── memcache/               Memcache 客户端与对象会话适配
```

`worker/transfer/` 虽然被不同 Worker 路径复用，仍属于业务侧。它需要知道一行传输属于哪个请求、原始缓存组和 Store job，以及失败应如何回报。这些对象与结果的归属决定了它的业务职责。

## 3. 业务组合

### 3.1 请求规划与传输执行

Scheduler 将外部命中视为候选。只有 vLLM 确认采用并分配目标块后，Scheduler 才拥有生成 Load 命令所需的信息。Store 按本步调度范围、已确认的 token 边界、内容 hash 和保存策略规划，并区分来源就绪状态。生成 Store 命令不代表 KV 已写入完成，Worker 必须等来源就绪后才能读取。

Worker 从命令出发，建立对象行和来源归属，调用可复用能力完成选择、映射和执行，最后将结果还原到请求或 Store job。原始 group ID 始终用于缓存组身份；紧凑的数组下标只用于本次批次内部的索引。压缩候选行不能改变远端 key 的命名空间，也不能丢失失败来源。

### 3.2 Bulk Store

Worker 按以下顺序组织一次 Bulk Store。

1. 从本步 Store 命令取得范围、hash、block 表和来源归属；结合 Coordinator 形成需要保存的候选。
2. 生成候选 key，根据 Backend 的写前存在性要求筛选对象，并按业务规则处理同批重复写入。
3. 压缩入选对象，由各组 Mapping 计算地址和长度，再由 Worker 合并为本次 Backend 参数。
4. 在后台执行路径中先准备参数，在 Backend 真正读取 KV 前等待来源就绪事件。
5. 调用 Backend，并将每个对象的结果对应回来源；分别判断写入结果和来源是否已停止被读取。

Mapping 因而不需要知道某个请求为什么要保存；Executor 不需要解释后端失败码；Backend 不决定一个 vLLM 请求何时完成。业务层组合这些能力，推进本次操作。

### 3.3 职责划分

Infrastructure 可以包含领域规则。Coordinator 必须理解不同缓存类型的恢复条件，Mapping 必须理解对象的分片身份，这些规则构成可复用能力的一部分。共享的 Load / Store 消息契约则由独立的 Protocol 承载。

应当留在业务侧的是：本次请求采用哪个候选、哪些对象参与、何时提交、如何解释失败、哪些对象已经满足发布条件，以及何时可以归还来源。基础设施返回计算结果或完成保证，业务层据此推进状态。

| 需要解决的问题 | 可复用能力 | 业务层仍需完成的决定 |
| --- | --- | --- |
| 哪些缓存能支撑恢复 | Coordinator 的范围与掩码计算 | 是否采用命中，如何与本次请求和分配对应 |
| 数据在哪里 | Mapping 的 key、地址、长度与对象偏移 | 入选对象、writer 选择、跨组组批与结果归属 |
| 调用怎样执行 | Executor / Session 的队列、预取与等待 | 提交哪个业务操作，如何处理结果及失败 |
| 资源何时释放 | Resources 的持有、顺序释放和失败保留 | 确认结果、来源安全及允许关闭的时点 |

### 3.4 共享 Protocol

Protocol 定义 Scheduler 与 Worker 之间的业务交接。Load 命令携带已采用的加载范围与目标块；Store 命令携带可保存范围或精确 checkpoint 来源，并区分来源已就绪和仍需等待的操作。每步命令与完成结果通过 vLLM Connector 机制交接，KV 字节由 Worker 直接调用 Backend 传输。

Lookup 通过独立 RPC 通道查询。Scheduler 使用 LookupClient 发出范围、缓存组和内容 hash；Worker 侧的 LookupServer 将消息交给 Worker 查询，返回可恢复终点与尾部对象的 key 边界。Protocol 负责消息校验，并将远端查询失败、通信异常或无效响应解释为本次外部缓存未命中；RPC 只负责字节帧交换，不解释命中语义。

来源释放回执表示相关 Worker 已停止读取某个 Store job 的来源，用于归还 Scheduler 持有的引用；它不代表对象已成功发布。

## 4. 基础设施

### 4.1 Configuration 与 Topology

Configuration 负责整体静态解析，将执行选项、缓存规格、传输粒度和并行分区整理为 Scheduler 与 Worker 共用的配置。缓存规格与粒度复用 vLLM 的解析结果，Backend 的对象格式要求转化为相应的分区约束。

Topology 基于静态配置，绑定当前 Worker 的实际 rank、缓存组与物理层、对象分片位置及上下文成员关系，为 Mapping 和 Worker 提供本地事实。writer 选举和对象发布由 Worker 根据这些事实决定。

初始化时先完成 Configuration 解析，由 Factory 检查支持的组合；Factory 在 Worker 侧绑定 Topology，随后申请 Backend 资源并装配 Worker。静态解析与本地绑定分开，使 Scheduler 无需依赖 Worker 的进程坐标。

### 4.2 RPC

RPC 提供同步字节帧请求应答，使用 ZMQ REQ / REP。客户端由单一调用线程使用，同一连接一次只进行一个交换；服务端在持有 socket 的线程上串行执行处理回调。Lookup 的业务适配由 Protocol 完成。

客户端延迟建连，发送和接收共用同一个截止时间。交换失败后丢弃连接并向调用方抛出异常，不自动重试；下次请求重新建连，避免迟到应答进入后续交换。Protocol 检测到无效响应时也会要求重置连接。客户端超时只结束本次等待，不能取消已经进入原生客户端的查询。

服务端的 socket 由服务线程创建、使用和关闭，启动失败通过屏障传回调用方。Connector 关闭时先等待 Lookup 服务退出，再关闭 Worker 及其 Backend 资源；等待超时则保留这些资源，允许稍后重试。服务退出保证处理回调已结束，不保证应答已送达客户端。

### 4.3 Coordinator

Coordinator 在 Token 轴上统一理解本地与外部 KV 的逻辑可用性，判断共同可恢复终点，并选择各阶段所需的逻辑块。Full Attention 依赖相应历史，滑动窗口依赖终点附近的窗口，状态缓存依赖对应位置的检查点。请求进度由 Scheduler 推进，是否采用命中也由 Scheduler 决定。

单组与多组统一使用 `Coordinator`，复用 vLLM core 的 `SpecGroup` 和注册 Manager。标准 Full Attention 在满足粒度条件时直接扫描连续命中；其他情况通过 `ExternalCachedBlockPool` 将本地已确认前缀与外部命中提供给 Manager，协调共同恢复终点。占位块只参与逻辑计算，不进入对象 key 命名。

| 阶段 | 已知条件 | Coordinator 的职责 |
| --- | --- | --- |
| Lookup | 查询上限与本地已确认前缀；查询后的外部命中 | 先选择可能支持候选终点的查询块，再根据可用性证据求各组共同支持的恢复终点 |
| Load | Scheduler 已确定的加载终点 | 选择恢复到该终点所需的逻辑块，不再求解共同终点或重复裁剪 |
| Store | 本次保存区间、prompt 边界与保留策略 | 选择应保存的逻辑块，mask 从本次起始块计数 |

Lookup 的查询候选、外部实际命中和 Load 所需块是不同结果；Store 的稀疏保留规则也不能替代 Lookup 选择，否则会漏掉仍可恢复的终点。恢复终点与尾部对象边界不一致时，`ReachablePrefix` 保留读取所需的尾部 key 边界。

Coordinator 输出逻辑终点与选择掩码，不查询 Backend、不读写 KV 字节，也不计算 Tensor 地址。Worker 执行查询，将选择对应到实际对象行，结合 Mapping 组织传输参数，再交给 Backend 执行。

### 4.4 Mapping

Mapping 将对象身份、本地字节与远端位置分别绑定为 Key、Local、Remote 三份数据，Layer 记录物理层到 segment 区间的关系。一个 segment 是可连续复制的本地字节段；各份数据通过一致的分区坐标与段顺序对应。

绑定时利用真实 Tensor view 表达切分并识别连续段，形成可复用的数值布局。运行时按内容 hash、本地 Block ID 和层区间求值，Worker 据此组织 Bulk、KeyRange 或 GVA 参数。对象选择、发布和来源安全仍由 Worker 负责。第 5 章说明具体布局关系。

### 4.5 Execution

Execution 将跨步执行与单步协调分开：Executor 提供执行队列，Session 只协调当前 step 的逐层时序。结束 Session 不会关闭执行器，也不会等待跨步 Store 完成。

`Executor` 提供一个 FIFO 单线程队列。启动屏障确认线程初始化完成，提交返回 Future，关闭等待已提交任务完成并回收线程。任务结果和异常由调用者领取。Load 与 Store 可以使用不同队列，跨队列的依赖通过事件或显式等待建立。

`Session` 协调一步内的逐层执行。模型钩子由一个调用线程推进，后台操作在传入的串行队列上运行。Session 维护 Load 顺序、有界预取窗口、计算开始 gate 和 Store 提交去重；业务回调负责实际传输和结果处理。gate 是一个同步门闩，用于让后台任务等待计算开始条件。

预取分别记录提交位置和领取位置。例如 Load 顺序为 `[2, 5, 9, 12]`，窗口深度为 2，起初提交层 2、5；模型领取层 2 后补交层 9。层 5 在后台完成，只改变它的 Future 状态，领取位置仍由模型钩子推进。同一层的重复钩子保持幂等，完成回调按队列顺序执行一次。

Store 钩子在调用线程记录来源就绪事件，再把工作交给队列。后台先准备参数，随后在读 KV 前等待该事件。中止 Load 时解除其等待的 gate，保留仅供 Store 使用的 gate；要求等待时，必须等已提交 Load 及其收尾结束。中止不能打断正在执行的原生操作，操作成功与否仍由业务层读取结果确认。

`end_step` 解除本步回调并结束 Load 协调。跨步 Store 的结果观察、来源保护和最终排空由 Worker 继续管理。

### 4.6 Resources

`Resources` 持有执行器、Backend 对象会话、内存注册句柄和客户端，并保留 KV Tensor 引用。Worker 持有 Mapping 与每步 Session。

绑定时，Resources 先保留 KV Tensor，并按真实 view 的地址跨度计算注册范围；Worker 调用 Mapping 从 Tensor 构建分区与段布局，再由 Resources 注册 buffer、启动执行器。共享 storage 的多个 view 合并覆盖范围，空 view 被跳过。注册范围包含访问跨度中的 padding，Mapping 的段长度只包含有效载荷。

关闭从 Worker 的业务收尾开始：领取结果，确认 Store 已停止读取来源，再停止执行器。Resources 随后依次关闭对象会话、执行额外清理（例如发布协调组）、释放注册句柄、关闭 Backend，最后放掉 Tensor 引用。

清理遇到失败即停止后续资源释放，保留句柄和相关引用供重试；全部清理成功后才放掉 Tensor 引用。原生调用异常导致读源状态未知时，Worker 还需保留安全状态；仅有 Future 结束或 Python 线程退出，无法证明设备和客户端已停止访问该内存。

### 4.7 Backend

整对象接口用 key 和本地地址段传输完整对象。KeyRange 接口用 key、对象大小和对象内偏移访问部分字节。GVA 接口先通过对象会话取得受保护的远端地址，再按本地地址、远端地址和长度复制。当前 Mooncake Layerwise 使用 KeyRange，Memcache Layerwise 使用 GVA。

Mooncake 的普通 TransferEngine 路径由 v1 Backend 持有进程级引擎和注册表，与旧 Connector 的全局引擎隔离。多个 v1 客户端注册同一地址和大小时共享引用，最后一个注册句柄释放时才注销内存；失败的回滚或注销保留句柄供重试。引擎初始化复用上游 `GlobalTE` 的失败策略，原生初始化失败后需要重启进程。使用 Store 独立引擎或 fabric memory 的路径仍按各自原生契约处理。

Backend 实现原生客户端调用、内存注册、对象保护和结果规范化。Worker 累计各层结果，判断对象完整性，并决定 commit 或 revoke。commit 发布完整对象；revoke 撤销未完成写入。适配器落实这些操作的原生语义。

同步 Store 正常返回时，接口保证本地来源读取已经结束，返回的失败码仍可以表示某些对象写入失败。异常缺少这一完成保证，调用者需要保留未知状态。远端发布和本地读源结束分别记录，避免用其中一个状态代替另一个。

## 5. Mapping

### 5.1 三个对象

Mapping 用 Key、Local、Remote 三份数据，分别描述对象身份、本地字节和远端位置。

| 对象 | 回答的问题 | 绑定结果 |
| --- | --- | --- |
| Key | 这些字节属于哪个对象？ | 各分区的命名前缀与后缀，运行时填入内容 hash |
| Local | 本地有哪些字节？ | 每段在 block 0 的基址、块步长和有效长度 |
| Remote | 每段放在远端对象的哪里？ | 完整对象大小与各段的对象内字节偏移；不保存运行时远端基址 |

一个 segment 是一次可以连续复制的本地字节段。Local 按物理层、缓存条目名、Tensor 分量及逻辑元素顺序排列这些段，Remote 使用相同段序。Layer 将每个物理层对应到这条段序列的半开区间；它是数组下标关系，不是字节偏移，也不生成另一份布局。

每个原始缓存组（Group）独立绑定，其上下文已经确定模型与缓存规格、原始 group ID、并行规模和当前 Worker 的本地信息。`PartitionId = (PP rank, DCP rank, head/TP rank)` 在这个绑定内唯一标识一个分区：三项分别表示目标 PP 层区间、实际 DCP 分片和 head / TP 分区；需要对齐的状态缓存以 TP rank 作为第三项。

PartitionId 是 Key 与 Local 的共同索引，跨 Group 或绑定实例时仍需要外层上下文；内容 hash 再区分该分区内不同内容的远端对象。

同一 PP 内的 head 分区共享段长度、块步长、Layer 关系和 Remote 偏移，仅本地基址及 Key 身份不同。因此 Key 与 Local 按 PartitionId 索引，Remote 与 Layer 按 PP rank 索引。

### 5.2 三种切分布局

TP、PP、CP 分别沿 head、模型层和上下文 token 划分数据，影响对象归属的不同部分。

| 切分 | 决定什么 | 当前 Mapping 的处理 |
| --- | --- | --- |
| TP / head | 每个对象包含哪些 head | TP mismatch 时将本地 head 等分，形成各自的 Key 与 Local 段；普通路径保留本地 head 范围 |
| PP / layer | 每个对象包含哪些模型层 | 将实际物理层归入目标 PP 区间，只为本地持有的层生成分区 |
| CP / token | 每个上下文分片包含哪些 token | 本地 Tensor 已完成 DCP 切分，直接保留实际 DCP 身份；按格式确定远端槽位，不再次切本地 token，PCP 也不额外切本地 Tensor |

Layerwise 在已确定的对象内逐层选择 segment，保留其在完整对象中的偏移。它不改变对象的 TP、PP、CP 归属。布局关系可以组合，但生产路径是否开放某种组合，由 Factory 的支持检查决定。

### 5.3 连续性

一个对象的有效载荷可以来自多个不连续的本地范围。连续性决定需要多少个 segment；对象字节顺序决定这些段怎样排列。两者不能靠排序本地地址来统一，也不能把其他 head 或 padding 当作有效载荷复制。

例如 NHD 布局中的两个 token 各有四个 head，每个 head 占 4 字节。选择每个 token 的前两个 head，对应本地 `[0,8)` 与 `[16,24)` 两段；紧凑对象将它们依次保存到 `[0,8)` 与 `[8,16)`。

![同一对象的两个不连续本地片段在远端紧凑排列](../../assets/ascendstore_v1/tp_head_slices.svg)

块步长与段长度也必须分开。例如每块有效载荷为 32 字节、相邻块起点相距 40 字节时，每次复制 32 字节，寻址仍按 40 字节移动。逐层访问同样保留对象偏移：第二层从对象第 32 字节开始，就不能因本次只传第二层而将偏移重置为 0。

### 5.4 用 Tensor view 建立布局

绑定阶段直接使用真实 KV Tensor 的 shape、stride 与 view 表达切分。先按 cache spec 的物理存储行数将 kernel block 归并为 group block，再沿 head 轴形成所需分区。NHD 的块内顺序是 token、head、向量；HND 及 LBHNC 的逐层 view 使用 head 优先的顺序，Mapping 保留各自的逻辑元素顺序。

连续 view 直接形成一个 segment；不连续 view 按逻辑元素顺序拆成连续子段。普通布局与 head 切片共用这一规则，padding 和非连续 stride 由实际 view 决定，不改变有效字节的对象内顺序。

这些 view 操作不复制 KV 数据，也不做布局格式转换。绑定后仅保留基址、块步长、长度和对象偏移等只读数值数组；一致的步长与长度用单元素数组共享。Tensor 的存储生命周期由 Resources 保证，更换底层分配时需要重新绑定。

### 5.5 绑定结果的使用

绑定生成 Key、Local，并按需生成 Layer 关系；Remote 由所选访问方式单独建立。Bulk 使用 Key 和 Local 传输整对象；KeyRange 为各段计算紧凑对象偏移；GVA 根据 Configuration 与 Topology 的约束，在紧凑段序上加入全局层预留、上下文槽位及对齐。Remote 的对象大小因此可以大于本地有效载荷之和。

运行时，Worker 提供已选内容 hash 和本地 Block ID。对段 j，令 βⱼ 为 block 0 基址，σⱼ 为块步长，ℓⱼ 为长度，oⱼ 为对象偏移；bᵢ 为第 i 条记录的本地 Block ID，Gᵢ 为本次对象会话提供的远端基址。地址、步长、长度和偏移均以字节计。

```text
Key       = prefix + 内容 hash + suffix
本地地址  = βⱼ + bᵢ × σⱼ
复制长度  = ℓⱼ
GVA 地址  = Gᵢ + oⱼ
```

KeyRange 直接使用 oⱼ；GVA 的 Gᵢ 由有效 Backend 对象会话提供，不属于永久绑定数据。Layerwise 用同一段区间选择 Local 与 Remote；Bulk 使用完整段序列。Lookup 的命名范围单独建立，可以覆盖本 rank 没有本地来源的分区。

Mapping 只求值位置与身份。对象选择、逐层批次复用、Backend 参数组织和结果归属由 Worker 完成；Backend 负责实际复制及对象会话。

## 6. 执行与发布

### 6.1 对象选择与结果归属

Bulk Worker 先生成候选 key，按 Backend 要求完成写前存在性检查和同批写入准入，再展开入选对象的地址。每步已经到达的工作可以合并到一个批次；Worker 保留对象、原始组、请求及 Store job 的对应关系，将后端逐对象结果还原到业务来源。

Layerwise 在各层复用同一批对象选择，并累计每个对象需要的层是否完成。Session 推进层的执行顺序，Worker 判断全部必要层的复制结果，再决定可发布的对象。批次压缩、重复行处理和逐层执行都需要保持同一套对象身份。

### 6.2 来源安全

模型写入完成、Backend 停止读源、来源允许回收是三个连续但不同的条件。Store 钩子记录就绪事件；后台完成参数准备后等待该事件，随后开始读取 KV。复制结束证据用于确认停止读源，Scheduler 的来源引用和 Worker 的 fence 共同约束后续回收或覆盖。

普通 FullAttention 请求存活期间，manager 保留相关缓存块。请求结束或抢占后，Worker 在模型可能复用这些块之前等待相关 Store。对于请求存活期间就可能回收页面的 manager，以及需要保护的 checkpoint 来源，Scheduler 额外持有引用，并等待预期 Worker 的来源释放回执。

正常 Step 收集已完成结果，只等待即将复用来源所涉及的批次。显式排空和关闭处理全部 pending Store。Backend 异常导致读源状态未知时，相关状态继续保留，并阻止危险复用。

### 6.3 共享对象发布

Memcache GVA 的共享对象可能由多个 rank 写入。Worker 从实际 `ContextRank` 成员关系出发，按 head 与 DCP 身份识别副本，为每组副本选择最小 global rank 作为 writer。当前共享协调条件包含 DCP 大于 1，且存在重复写入副本或多个 PCP rank；具体参与成员来自实际拓扑。

发布协调由 `worker/layerwise/gva/` 持有，使用覆盖相关成员的专用 Gloo 组。非 writer 和打开对象失败的成员也参与同一轮协调，保证所有参与者按相同次序进入 collective。

某个 key 发布前，需要所有必要 writer 完成各自的必要层，并确认它们看到的是同一个远端地址和对象大小。缺少 writer、任一必要层失败、地址或大小不一致，都会使该 key 不能 commit。预留空间中的空闲槽位无需虚构 writer，必要 writer 集合也必须非空。

撤销失败时保留未确认状态及会话所有权；只有原生能力能够确认安全，才能完成清理。复制结果、对象发布和本地来源释放因此分别推进。

## 7. 关键约束的验证

布局验证检查同一对象在整体与逐层视图中是否表示相同字节，并检查对象身份、顺序和结果归属在各路径间保持一致。

| 要验证的关系 | 现有测试入口 | 主要检查 |
| --- | --- | --- |
| Configuration 的解析、筛选与分区 | `infrastructure/test_configuration.py` | 输入边界、原始组 ID 与顺序、PP 默认分配 |
| Topology 的本地绑定 | `infrastructure/test_topology.py` | spec family、PP 跨度与实际上下文成员 |
| Key / Local / Remote 与 Layer 的对应 | `infrastructure/mapping/test_mapping.py` | 分区身份、段序、层区间与输入记录顺序 |
| Tensor 切分与连续段 | `infrastructure/mapping/test_mapping.py` | NHD / HND / LBHNC、padding、压缩缓存、混合类型与非连续 view 的有效字节 |
| 整体与逐层传输、TP / PP 双向恢复 | `infrastructure/mapping/test_worker_arguments.py` | 各层重组为完整对象、跨分区恢复与原始组 ID |
| GVA 对象预留与偏移 | `infrastructure/mapping/test_partition_bindings.py` | PP 层位置、上下文槽位与对齐符合既有分配公式 |
| 串行执行、预取、领取与收尾顺序 | `infrastructure/test_execution.py` | 队列排空、提交与领取分离、共享 gate 取消、跨步 Store |
| Lookup 契约与失败回退 | `protocol/test_lookup.py`、`protocol/test_rpc.py` | 消息校验、查询失败或无效响应回退为未命中 |
| RPC 生命周期与关闭顺序 | `infrastructure/test_rpc.py`、`test_connector.py` | 超时连接重建、服务线程退出、退出前保留 Worker 资源 |
| 注册与关闭所有权 | `infrastructure/test_resources.py` | 绑定与注册顺序、失败保留、执行器停止前禁止释放 |
| 共享对象发布 | `worker/layerwise/gva/test_shared_store.py`、`worker/layerwise/gva/test_collective.py` | writer 完备性、非 writer 参与、打开失败及地址分歧 |

表中路径相对于 `tests/ut/distributed/ascend_store/v1/`。CPU 单元测试与 Gloo 进程测试分别验证逻辑约束和集体通信顺序；原生 Backend 与 NPU 验证还需覆盖注册范围、读源完成、地址保护和对象发布语义。
