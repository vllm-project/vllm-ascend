# DeepSeek V4 Sparse KV LRU Mooncake Requirement

Phase 0 experiment playbook: [deepseek_v4_sparse_kv_experiments.md](deepseek_v4_sparse_kv_experiments.md)

## 背景

当前仓库是 vLLM Ascend 官方仓库的 fork，分支基于 `v0.22.1rc1`
release candidate。目标是在本地 Windows 工作区先完成静态分析、方案设计和代码改造，
然后推送到 GitHub，再在有 Ascend 环境的华为服务器上拉取代码做真实验证。

本机当前没有配好的昇腾运行环境，因此本地阶段不能完成真实 NPU serve、Mooncake
传输、DeepSeek V4 权重加载和精度/性能验证。真实验证需要在服务器上闭环：
运行、收集日志和报错、回传给 Codex 继续修。

## 用户目标

针对 DeepSeek V4 的稀疏注意力推理，探索一种更激进的 KV Cache 分层方案：

1. Prefill 完成后，将请求全量 KV Cache 写入 Mooncake 对接的池化内存。
2. 每条请求、每层稀疏注意力在 NPU 本地只保留一个固定大小的 KV 工作集窗口，
   例如 2K、4K 或 8K token。
3. 该本地窗口可预装最近窗口 token 的 KV Cache。
4. Decode 时，稀疏注意力每层计算出本步需要的 top-k 历史 token
   （DeepSeek V4 Flash 模型中预期为 top-k 512）。
5. 对本地窗口缺失的 token，从 Mooncake 细粒度拉取对应 KV Cache，
   放入本地 LRU 窗口。
6. 利用相邻 decode step 之间 top-k 历史 token 高重合的性质，降低 NPU KV
   常驻内存，同时避免每步全量远端读取。

可选规范称呼：

- Demand-paged sparse KV cache
- Hierarchical KV cache with sparse-attention working-set caching
- Token-granular KV cache paging for sparse attention
- Mooncake-backed sparse KV LRU cache

中文可称为：面向稀疏注意力的分层按需换页 KV Cache，或
Mooncake 后端的稀疏 KV 工作集缓存。

## 当前仓库已有能力

基于代码和文档静态阅读，`v0.22.1rc1` 已经支持很多 DeepSeek V4 推理优化：

1. DeepSeek V4 模型路径
   - `vllm_ascend/models/deepseek_v4.py`
   - `vllm_ascend/models/deepseek_v4_mtp.py`
   - 支持 DeepSeek V4 的 mHC、MoE、DSA 稀疏注意力、MTP 权重加载和工具调用解析。

2. DSA 稀疏注意力
   - 使用 `AscendDeepseekSparseAttention` 和 `AscendDSABackend`。
   - 支持 SWA cache、compress ratio 4 和 128、compressor state cache、indexer cache。
   - Decode 中通过 LightningIndexer 生成 `cmp_sparse_indices`，再传给 sparse attention op。
   - `index_topk` 来自模型配置，典型值为 512。

3. IndexCache
   - `use_index_cache`、`index_topk_freq`、`index_topk_pattern` 可让部分 c4
     indexer 层复用已有 top-k indices。
   - 该机制优化 indexer 计算，不是 KV Cache 远端按需换入。

4. Hybrid KV cache manager
   - DeepSeek V4 使用多个 KV cache group，包含 c4、c128、SWA 等不同规格。
   - `AscendHybridKVCacheCoordinator` 对 DeepSeek V4 做专门处理。

5. KV Pool / AscendStore / Mooncake
   - `AscendStoreConnector` 支持 KV Pool。
   - Mooncake backend 支持 batch put/get，多 buffer 地址和 size。
   - 支持 prefix hash、KV cache group、cache family、compress ratio-aware key。
   - 支持 PD disaggregation 和 `MooncakeHybridConnector`。

6. 官方推荐部署优化
   - W8A8/W4A8 Ascend quantization。
   - TP、DP、EP。
   - FlashComm1。
   - Async scheduling。
   - FULL_DECODE_ONLY ACLGraph / NPU graph。
   - CPU binding。
   - Shared expert DP / shared expert multistream overlap。
   - MTP speculative decoding。
   - Safetensors prefetch 和多线程权重加载。
   - DeepSeek V4 compressor block size 32/64/128，用于改善 prefix cache hit。

## 与用户方案的差异

当前实现没有直接实现“每层 sparse attention 按 top-k token 从 Mooncake
细粒度换入本地 LRU KV 窗口”的完整机制。

主要差异：

1. 当前 KV Pool 是 prefix/block 粒度
   - lookup 基于 prompt prefix hash 和 KV block hash。
   - load/save 以 block、KV cache group、layerwise block 为主要单位。
   - 调度发生在请求 prefill/prefix cache 阶段，而不是每个 decode step 的
     sparse top-k miss 阶段。

2. DeepSeek V4 c4/c128 组目前不适合作为 KV Pool gate
   - `pool_worker.py` 中注释说明 c128 key stream 过稀疏，不适合 strict gate。
   - c4 当前是 TP-sharded key stream，可见 key 不完整。
   - 辅助组逻辑 block size 可能是 8/32，而当前 Ascend kernel 路径固定 128-token
     KV block shape。

3. 当前 SWA cache 不是 LRU sparse working set
   - SWA 是最近窗口/sliding window。
   - 用户方案需要最近窗口加稀疏 top-k 缺页换入，并在窗口满时 LRU 淘汰。

4. 当前 IndexCache 复用 top-k indices，不缓存 top-k 对应 KV
   - 它减少 indexer 计算。
   - 它不负责远端 KV 拉取、本地 KV resident set 或 LRU metadata。

5. Mooncake API 当前是多 key、多地址、多 size 的批量读写
   - 可支持 block/segment 粒度传输。
   - 若要 token 粒度随机换入，需要明确远端对象 key、地址切片、对齐和 DMA
     granularity，可能实际仍需按小 block 或压缩 token block 传输。

## 需要决策的问题

1. 缓存对象粒度
   - 真的按原始 token 粒度，还是按 c4 compressed token、KV block、或 4/8/16-token
     mini-block 粒度？
   - DeepSeek V4 的 `cmp_sparse_indices` 是 compressed stream 上的索引还是原始
     token 索引，需要在 NPU 真实日志和模型配置中确认。

2. 本地窗口缓存什么
   - 仅缓存 c4 compressed KV？
   - 是否还需要 indexer k/scale cache、compressor state、SWA KV、c128 KV？
   - 如果 attention op 同时需要 `ori_kv` 和 `cmp_kv`，只换入 `cmp_kv` 是否足够？

3. 缺页发生点
   - top-k indices 由 LightningIndexer 在 DSA decode 内部产生。
   - 如果发现 miss 后再从 Mooncake 拉取，attention op 已经在同一个 forward 内等待。
   - 需要决定是否接受同步 stall，还是提前一 token/一层预取。

4. LRU metadata 放在哪里
   - Python scheduler 维护，还是 worker/attention backend 维护？
   - 元数据在 CPU 上维护会带来 NPU/CPU 同步风险。
   - 在 NPU 上维护需要新增 kernel 或 device-side metadata 更新。

5. 与 ACLGraph/静态图的兼容性
   - Decode FULL_DECODE_ONLY graph 要求形状和控制流稳定。
   - 动态 miss 数量、远端 get、LRU 更新可能破坏图捕获。
   - 可能需要先在 eager 或 piecewise graph 模式验证，再考虑 graph 兼容。

6. 与 PD disaggregation 的关系
   - 单节点 mixed mode、PD decode node、KV Pool 三者的最小可行路径不同。
   - MooncakeHybridConnector 和 AscendStoreConnector 的职责边界需要明确。

7. 正确性和回退
   - 任何 KV miss/get 失败都必须回退到重算或全量本地 KV。
   - 不能默默用脏 KV 或缺失 KV 做 attention。

8. 性能收益前提
   - top-k 512 在相邻 decode step 的重合率必须足够高。
   - Mooncake 细粒度 get 延迟必须低于节省的 NPU 显存/带宽收益。
   - 需要记录 hit rate、miss count、transfer bytes、stall time、tokens/s、TPOT。

## 建议分阶段实施

### Phase 0: 观测和验证假设

不改语义，只加可选 instrumentation。

目标：

- 记录每层 decode 的 top-k indices。
- 统计相邻 step、相邻层的 top-k overlap。
- 统计每层所需 KV bytes 和理论工作集大小。
- 验证 `cmp_sparse_indices` 对应的真实索引空间。

输出：

- 日志或 metrics，证明 LRU 工作集有足够命中率。

### Phase 1: 本地模拟 LRU，不接 Mooncake

目标：

- 在 DSA decode 路径中，用 Python/CPU metadata 或 mock tensor 模拟
  2K/4K/8K resident set。
- 不改变 attention 输入，只统计理论 miss/hit 和需要换入的 token/block。

输出：

- 不影响精度的离线/在线观测。
- 初步容量选择建议。

### Phase 2: Mooncake 对象布局设计

目标：

- 为 DeepSeek V4 sparse cache 设计新的 key schema。
- 明确按 layer、request、cache family、token/block index 存储。
- 明确本地 block id、远端 key、NPU cache 地址之间的映射。

候选模块：

- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py`
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py`
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/mooncake_backend.py`

### Phase 3: 最小可行同步换入

目标：

- 仅支持 eager 或非 full graph decode。
- 仅支持 c4 sparse attention 的 selected compressed KV。
- 对 miss 的 compressed block 同步 batch get 到一个固定 NPU staging cache。
- attention op 使用 staging cache 和 remapped sparse indices。

需要新增：

- sparse KV resident manager。
- top-k indices 到 resident slot 的映射。
- miss list 构造和 Mooncake batch get。
- 本地 LRU 替换策略。
- 出错时回退到原始路径。

### Phase 4: 异步预取和 graph 兼容

目标：

- 使用上一 token/上一层 top-k overlap 做 prefetch。
- 将动态 miss 限制成固定上限，保持 graph shape 稳定。
- 评估 piecewise graph 或 FULL_AND_PIECEWISE graph 的可行性。

### Phase 5: 完整验证和调参

目标：

- 在 A2/A3 真实机器上验证 DeepSeek-V4-Flash-w8a8-mtp。
- 比较 baseline、IndexCache、KV Pool、Sparse KV LRU 的显存和性能。
- 输出开关默认值、失败回退策略和推荐窗口大小。

## 可能改动模块

1. 配置和开关
   - `vllm_ascend/envs.py`
   - 新增 `VLLM_ASCEND_DSV4_SPARSE_KV_LRU_*` 系列开关。

2. DeepSeek V4 DSA attention
   - `vllm_ascend/attention/dsa_v1.py`
   - 在 top-k 产生之后、sparse attention op 之前插入 resident cache 管理。

3. DeepSeek V4 模型 wiring
   - `vllm_ascend/models/deepseek_v4.py`
   - 将 per-layer cache manager 或配置传入 DSA modules。

4. KV Pool / AscendStore / Mooncake
   - `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/*`
   - 新增 token/block-granular sparse KV key 和 get/put 逻辑。

5. Cache manager
   - `vllm_ascend/core/single_type_kv_cache_manager.py`
   - `vllm_ascend/patch/platform/patch_kv_cache_coordinator.py`
   - 如果需要让 scheduler 知道 sparse resident window，需要扩展这里。

6. Custom op 或 device helper
   - 可能需要新增 NPU op 做 top-k remap、resident slot scatter/gather、LRU metadata
     更新，避免热路径 CPU 同步。

7. 测试
   - 单元测试先覆盖 key schema、LRU manager、miss/hit 计算。
   - NPU e2e 测试在服务器上跑 DeepSeek V4 Flash 小上下文和长上下文。

## 本地与服务器协作流程

1. 本地阶段
   - 静态读代码。
   - 写设计和最小代码。
   - 运行 Python 单测、ruff 或可运行的非 NPU 测试。
   - 推送到 GitHub 分支。

2. 服务器阶段
   - 拉取分支。
   - 使用官方 vLLM Ascend Docker 和真实 DeepSeek V4 权重。
   - 先跑 baseline。
   - 再跑新增开关。
   - 回传完整命令、环境变量、日志、报错、npu-smi、吞吐/延迟数据。

3. Codex 迭代阶段
   - 根据日志定位问题。
   - 修复并再次推送。
   - 重复直到启动、首请求、长上下文、性能指标都可接受。

## 并行 Code Review Agent Prompt

可以新开一个 agent，使用下面的 prompt：

```text
你是我并行的 code review agent。当前仓库是 vllm-ascend fork，分支基于
v0.22.1rc1。我正在让另一个 Codex 设计并实现 DeepSeek V4 sparse attention
的 Mooncake-backed sparse KV LRU / demand-paged KV cache 方案。

请先阅读根目录 AGENTS.md、CLAUDE.md，以及
docs/deepseek_v4_sparse_kv_lru_mooncake_requirement.md。

你的任务不是直接重写方案，而是做审查：
1. 梳理现有 DeepSeek V4、DSA attention、IndexCache、KV Pool、AscendStore、
   MooncakeHybridConnector 的真实代码路径。
2. 判断 proposed sparse KV LRU 方案是否与现有机制重复，或者会破坏哪些已有假设。
3. 特别检查：
   - top-k indices 的索引空间和粒度；
   - c4/c128/SWA/cache family 的内存布局；
   - Mooncake get/put 是否支持所需细粒度；
   - ACLGraph / async scheduling / MTP / PD disaggregation 兼容性；
   - CPU-NPU 同步风险；
   - 错误回退和精度风险。
4. 对后续代码 diff 做严格 review，优先指出 bug、性能退化、NPU 同步、内存越界、
   脏 KV、并发请求串扰、测试缺口。

输出中文 review，按严重程度列出问题，引用具体文件和行号。
不要替我做大范围重构，除非发现当前方案无法成立。
```
