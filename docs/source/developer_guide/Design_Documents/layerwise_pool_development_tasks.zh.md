# LayerwisePoolConnector 开发与验收边界

本清单对应独立注册的 `LayerwisePoolConnector`，运行路径为 P 逐层写池、D 整体恢复。
P 的每个 prefill chunk 都逐层写入，不再等最后一个 chunk。完整旧块不重复写，
各 group 未满块按自身边界重新生成完整有效范围；状态使用对应 checkpoint。
worker 累积已提交对象并续租，最终清单验证完整覆盖后才交给 D。源请求取消时，
metadata 绑定阶段清理跨 chunk 保护，避免依赖 V2 Runner 的模型 forward。
协议、数据布局及生命周期设计见 [设计说明](layerwise_pool_connector.md)。
实现存在于工作分支不等于生产验收完成；下面分别列出代码边界和验收要求。

当前仍采用源端布局对象，跨 TP 的 PD 恢复依赖源 metadata；普通前缀查询按本地
布局重建 key。尚未实现上游 canonical Store TP/chunk 格式下的跨 TP 自主发现，
新 connector 的 D 写回也未对齐上游按 group 确认进度的完整生命周期。

## 模块与任务

以下路径相对 `vllm_ascend/distributed/kv_transfer/`。

| 任务 | 文件与接口 | 保留在调用方的职责 | 必须验证的行为 |
| --- | --- | --- | --- |
| T1：key 与布局 | `kv_pool/ascend_store/key_codec.py`、`cache_layout.py`；复用原 hash 构造函数 | P 提供源布局身份，D 使用源身份找 key、本地布局决定目的地址 | 完整块、尾块、不同 rank、cache group 与 target/draft 命名空间不能混淆 |
| T2：后端与读保护 | `backend/base.py`、`backend/memcache_backend.py`、`backend/executor.py` | 原生 SDK 的内存保护契约 | 分配、提交、批次返回对齐、续租、部分失败、取消排空和失败清理 |
| T3：共用逐层写入 | `layerwise_writer.py`、`backend/copy_ranges.py`；`LayerwiseAllocationPlan`、`LayerwiseWriteSession`、`LayerwiseWriteCompletion` | 见下一节 | 计算事件、逐层重叠、完整层集合、部分失败、取消、延迟回调以及旧池化回归 |
| T4：整体读取与异构恢复 | `bulk_reader.py`、`restore_planner.py`、`state_snapshot.py` | D 的本地块分配和模型执行 | 目的范围完整覆盖，TP/PP/CP 映射，副本去重，恢复完成前不可运行请求 |
| T5：CPD 控制面 | `kv_p2p/mooncake/pool_metadata.py`、`pool_control.py`、`pool_scheduler.py` | 原 proxy 的路由与透传行为 | 保持原 proxy；所有源 worker 提交后才能交接，超时或缺失不能成为成功命中 |
| T6：独立入口 | `kv_p2p/mooncake/layerwise_pool_connector.py`、注册入口 | 框架 hook 和 cache 注册 | 不继承 V2 worker，不隐式构造其传输引擎；cache 辅助组件由实际层完成回调覆盖 |
| T7：并行布局 | `cache_layout.py`、`restore_planner.py`、`state_snapshot.py` | P/D 各自的并行配置 | 保持 V2 已支持的布局条件，验证 TP 双向、PP、PCP、DCP、EP、KVPP；不能仅凭参数可启动判定适配成功 |
| T8：V2 Runner 与图 | connector hook、`platform.py` 的角色配置 | 模型 runner、target/draft 图执行 | P 逐层回调真实发生；D 整体读取在模型执行前完成；target 与 draft 均需实际 replay 证据 |
| T9：GLM 与 MTP | 共用 `utils/state_layout.py`、`utils/prefill_boundary.py`，池化的 `state_snapshot.py` | D 的本地投机配置与模型限制 | KDA、MLA/NoPE、indexer、tail、MTP 差异及 RecoverSSM；遵循 V2 支持范围 |
| T10：生产验收 | `pool_stats.py`、`examples/disaggregated_prefill_v1/layerwise_pool/`、e2e probe | 部署与验收环境 | 全量模型精度、性能、长稳、故障、资源释放与可观测性，不能用小模型测试替代 |

## T3 的具体抽取边界

当前 GVA 共享模式已进一步接入 `layerwise/object_adapter.py`：普通池化和新 PD
共同调用 `PoolCacheLayout`、`LayerwiseWriter` 与 `PoolBackendExecutor`，前缀查询
共用 `prefix_lookup.py`。普通池化继续使用原发送、接收队列；按层执行读取，保留
共享 HBM 槽位的门控。下面涉及原 `LayerBatchBuilder` 的条目描述显式 legacy
模式和非 GVA 路径；不能把这些路径的旧全块 key 当作已经证明与新格式兼容。

共享完整对象增加 `pool-full-v1` 格式身份，组件对象使用 `pool-component-v1`。
仍调用原后端 key 构造函数，但不会只因 token hash、字节数相同就借用另一布局。
P 的完整输入摘要、scheduler 计算的分组 hash 和命中对象描述经 `ReqMeta` 传给
普通 worker。普通入口与新 PD 入口已接线；全量模型和原生 SDK 验收仍单独列项。

共用的是对象布局、逐层写入和恢复计划；调度策略与模型执行仍属于各自入口。
GVA 共享模式的实际边界如下：

| 职责 | 共用实现 | 普通池化与 PD 的调用差异 |
| --- | --- | --- |
| 选择请求和已确认范围 | 各自 scheduler | 普通池化传 `ReqMeta`；PD 传稳定 prompt 的 `PoolSaveRequest` |
| 对象、key 与源地址 | `PoolCacheLayout.build_source_plan()` | 两侧使用实际 spec/tensor stride；未确认工作快照由普通 adapter 改为私有身份 |
| 分配、借用、逐层复制和提交 | `LayerwiseWriter`、`LayerwiseWriteSession` | 普通 adapter 用 `LayerwiseSenderExecutor` 投递原发送队列；PD 使用自己的 writer 执行队列 |
| 读取保护和异构映射 | `PoolBackendExecutor`、`RestorePlanner`、`state_snapshot.py` | 普通 adapter 在原接收线程按层执行；PD 在运行请求前整体恢复 |
| 查询共享前缀 | `prefix_lookup.py` | 两侧按 worker 布局查询，所有必要 worker 可读后才报告命中 |
| 计算事件与缓存槽位 | 各自 worker/runner | 普通入口保留 attention-start、save-finished 和外部槽位门控；PD 写完才能交还 P 源块 |
| 失败与对象所有权 | 共用 writer/executor + 各自请求生命周期 | 借用的共享对象不因本次失败而删除；临时私有对象只由写入者清理，I/O 排空前不能释放源内存 |

普通 GVA 写入链是 `save_kv_layer()` → `LayerwiseObjectAdapter.submit_save_layer()`
→ `LayerwiseWriter` → 原发送线程 → `LayerwiseWriteSession`；读取链是原接收线程
→ adapter 的 `load_layer()` → 共用恢复计划和 executor。没有第二个 backend
client，也没有新增一套原生复制循环。

显式 legacy 模式和非 GVA 路径继续使用原 `LayerBatchBuilder` 选择块、构造地址，
并调用共用 `LayerwiseAllocationPlan`、`LayerwiseWriteCompletion` 和
`LayerwiseWriteSession`。非 GVA 的私有 attempt 不宣称与 GVA 共享对象互通。

对取消的约定同样明确：当前原生调用返回前继续持有源内存；取消被观察到后不再提交
后续批次；部分复制不能通过完整性检查。I/O 排空也不等同于模型计算已结束。

## 模型适配由谁负责

投机接受边界不增加新的 PD 协议：P 保存稳定 prompt，D 自己运行 MTP。
共享前缀只能发布已确认 token 的数据。普通池化开启层间 HBM 复用时，还需要
覆盖前保存工作快照；未确认范围使用调度器下发的 `pool_snapshot_id` 生成
`pool-step-v1` 私有对象，不参与共享命中。下一步按上一轮记录的对象读取，
裁剪到已接受范围；独立 KDA/Mamba cache 仍交给模型维护。副本不拥有删除权，
跨步 TP barrier 后才清理旧私有对象。读取失败会先取消本轮保存。

`test_pool_prefix_schedule.py` 使用实际 Scheduler、KVCacheManager 和共享查询
端点，验证异步恢复后只调度未命中后缀，并分别覆盖 `fail`、`recompute` 和
取消后的块生命周期。CPU 字节拷贝替代传输，不能据此宣称 NPU/全模型验收通过。

| 模型相关能力 | 复用边界 | 新模型仍需提供或验证的内容 |
| --- | --- | --- |
| 普通 attention / MLA 的页、head、token 坐标 | 复用已注册 cache spec、实际 tensor stride 和通用恢复计划 | 新的 packing 或量化字节格式必须增加明确的布局描述和字节级测试 |
| KDA/GDN 的卷积与 recurrent state | V2 和池化共同调用 `mamba_component_copy_slices()` | 组件顺序与投影宽度必须符合实际注册布局，不按模型名字猜测 |
| 环形 / tail 状态 | 保持 V2 的整页复制与格式兼容检查 | 不新增跨环形容量转换；不支持的格式必须明确拒绝 |
| MTP | D 使用本地 speculator；复用 V2 的 prefill 边界和状态处理 | D 所需的源 cache 必须存在，P/D 布局需满足 V2 的兼容条件；不增加需求协商协议 |
| indexer 等辅助 cache | 按真实 attention 完成回调组装完成集合 | 没有独立回调的辅助 cache 不能单独等待；不同 namespace 的同号层不能合并 |
| RecoverSSM / KVPP | 复用模型配置限制和已有 placement / prefetch 能力 | 必须在该模式下验证；不能从普通 eager 的结果推断其正确性 |
| 多模态输入 | 继续使用模型的 processor / vision encoder；`pool_request_digest()` 包含媒体 hash、模态和占位位置 | 相同文本换图片及重复图片都要验证真实读池与输出一致性；dummy 输入的帧数和像素预算修复放在模型内，不放进 connector |

## 验收顺序与完成条件

并行能力按模型分别判断：缩小 GLM-5.3 已通过 TP 双向、PP 双向和 P PCP2
阶段验证，但当前 `Glm5NextKPoolIndexerBackend.__init__()` 明确拒绝 DCP>1。
该限制在分支基线中已存在，与 connector 类型无关；新 connector 保留它。
GLM-5.2 的 DCP2 实测结果不能外推到 GLM-5.3。如果后续要扩展 GLM-5.3 DCP，
需另行补模型/indexer 的执行能力，再验证池化映射；不能只删除 guard。

1. 对源码边界做回归：旧池化、V2 共用函数、新 connector、失败处理及 proxy 透传。
2. 用真实 SDK 检查分配、逐层写入、整体恢复、取消排空和可重复的字节结果。
3. 用缩小的 GLM 结构验证实际 cache 注册、并行映射、MTP、runner 和图执行。
4. 在目标全量 GLM 权重与部署配置上做基线对照、准确率、投机接受率、性能及长稳。
5. 在独立资源内验证服务重启、请求取消、池对象丢失、租约失效和淘汰等故障。

第 1～3 项的结果只能支持对应范围。全量权重、量化配置或多模态输入未测试时必须
保持“未验收”状态。把原生调用包在延迟返回的测试门闩中，可以证明 connector 等待
I/O 的顺序，不能证明 SDK 在真实传输跨越 TTL 或对象被淘汰时仍保护同一物理内存。
这些 SDK 契约和目标环境中的端到端验收仍是生产完成条件。

存储节点退出还需要单独验收。P/D 的推理角色不等于池内存的归属：D 如果配置了
池容量，也可能保存由 P 写入并持有租约的对象。当前三客户端实测中，D 已读完并
释放自身租约，关闭仍会在 `BmUnregister` 等待；MetaService 的原生栈停在
`MmcMetaLeaseManager::Wait()`。先释放 P 的保留租约再关闭 D，则可正常退出。
因此不能把“本地请求已排空”直接当成“可以卸载本节点池内存”。任务拆分中需
区分池客户端重启和池容量节点下线；后者须验证外部租约排空或存储迁移能力，
不能靠 connector 强删其他请求的对象，也不能将延长退出超时作为修复。
