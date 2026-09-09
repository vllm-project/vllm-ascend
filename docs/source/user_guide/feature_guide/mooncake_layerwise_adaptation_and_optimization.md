# Mooncake Layerwise 适配与优化分析

## 1. 适配基线与结论

- 基线：`Eric-dot/vllm-ascend:mooncake`，本地提交 `0a023b094e9e88ffaca0b1fda02529cef6277f8e`。
- 参考实现：`vllm-project/vllm-ascend#12418`。
- 本地工作分支：`eric-mooncake-pr12418-port`。
- 当前状态：核心代码、CPU mock 单测和静态检查已通过；尚未在真实昇腾 NPU、Mooncake Master/Metadata 服务和多 TP 进程上完成 E2E。

这次适配不是直接套用旧 PR。个人分支已经有更新的 `metadata.py`、多 cache entry/MTP/SFA 布局、Memcache GVA layerwise、异步线程异常传播和 layer buffer reuse，因此 Mooncake 路径复用了这些新结构，并独立增加 key-major range/session 协议。

## 2. 远端对象为什么从 `block × layer × rank` 降到 `block × rank`

旧的逐层 key 方案把每一层当作一个远端对象：

```text
model@block_hash@layer_0@rank_0
model@block_hash@layer_1@rank_0
...
```

若有 `B` 个 block、`L` 层、`R` 个实际保存 rank，对象数约为：

```text
B × L × R
```

Mooncake layerwise 改成每个 block/rank 只创建一个对象：

```text
model@block_hash@rank_0
```

对象内部按真实 cache layout 连续放置所有层：

```text
object(block, rank)
├── layer 0: [cache entry 0][cache entry 1]...
├── layer 1: [cache entry 0][cache entry 1]...
└── ...
```

每层计算完成后，只对同一对象写该层对应的 byte range。layerwise 改变的是传输时机和 range，并不要求每层成为独立对象，所以对象数变成：

```text
B × R
```

减少的是 Mooncake 元数据对象、key、session 和 `exists` 查询项，不是 KV payload 总字节数。KV 数据仍然需要保存所有层。

## 3. 当前实现的数据与控制流程

### 3.1 保存

1. Scheduler 用每个 block 的所有保存 rank key 做 `batch_is_exist`。
2. Worker 为未命中的 key 调用 `batch_put_session_start(keys, object_sizes, ReplicateConfig)`。
3. `LayerBatchBuilder` 根据当前实际 cache entry layout 计算：
   - NPU 本地 buffer 地址；
   - 该层各 cache entry 的 size；
   - 该层在远端全层对象内的 destination offset。
4. 每完成一层 attention，发送线程调用 `batch_put_from_multi_buffer_ranges`。
5. 单个 key 的 range 写失败时，只 revoke 该 key，其余 key 继续后续层。
6. 最后一层成功后调用 `batch_put_session_end`；commit 失败的 key 调用 `batch_put_session_revoke`。
7. 只有 commit 成功的 key 才进入 chunked-prefill 后续可读集合。

### 3.2 加载

1. Scheduler 只把从 block 0 开始、所有保存 rank 都为 COMPLETE 的连续前缀算作命中。
2. Worker 对去重后的远端 key 调用 `batch_get_session_start`。
3. 每层计算前，接收线程用 `batch_get_into_multi_buffer_ranges` 把该层 range 读到本地 block。
4. 单行失败会记录对应本地 block id，交回 Scheduler 触发重算，避免消费不完整 KV。
5. 最后一层或请求终止时调用 `batch_get_session_end`；异常路径按 retry/terminal 语义释放 owner。

### 3.3 Chunked prefill

`MooncakeSessionTracker` 维护三类关系：

- 尚未 commit 的 put key 与 request/block owner；
- 已 commit、可供同一请求后续 chunk 加载的 key；
- 已打开 get session 的 key 与 request owner。

它保证：

- commit 前不会把对象当成可读；
- 后续 chunk 即使没有新的 `load_spec`，仍会续租并逐层恢复此前已 commit 的 prefix；
- retry 释放 get session，但保留后续重试所需的 key/block 关系；
- terminal/preempt 清除 request 状态；
- 多个请求共享同一远端 key 时，最后一个 owner 释放后才执行 get-end。

## 4. 相对原 PR 已做的适配/修正

### 4.1 真实 layout offset，而不是固定 `layer_id × page_size`

当前分支支持一个物理层包含多个 cache entry，也支持 MTP/SFA 布局。远端 offset 使用 `group_layer_cache_entry_offsets` 和实际 `block_len` 前缀和计算：

```text
layer_object_offset = sum(block_len before this layer)
entry_offset = layer_object_offset + prefix_sum(entry sizes in this layer)
```

因此远端对象大小直接等于当前 rank 所有层 cache entry 的总字节数，不再假设每层大小完全相同，也不会错误地再乘一次 `num_layers`。

### 4.2 PutStart 继承 Mooncake 放置策略

原 PR 的 session start 没有传 `ReplicateConfig`。当前实现和 whole-key put 一致，传递：

- `preferred_segment`；
- `prefer_alloc_in_same_node`。

这样 layerwise 不会绕过已有的本地优先/同节点分配策略。

### 4.3 range 调用接入传输限流

- `layerwise_max_transfer_blocks`：限制单次 range API 的 key/block 行数；
- `layerwise_max_transfer_bytes`：把过大的单个连续 segment 拆成多个更小 range。

PutStart、GetStart、GetEnd 和 Scheduler 的 exists 查询也按 block 上限分批，避免大 prompt 产生超大 Python/C++ 参数列表和瞬时元数据峰值。

### 4.4 启动期 fail-fast

Mooncake layerwise 启动时检查所有 session/range 方法。缺少接口会直接报错，并提示需要包含 Mooncake PR #2881 的 client，而不是在首个请求的异步线程里才失败。

当前 key schema 只编码 model、block hash、TP/head rank，因此明确拒绝：

- pipeline parallel size > 1；
- prefill/decode context parallel size > 1；
- hybrid/multi-group KV cache；
- TP mismatch layerwise。

### 4.5 异常与回退

- batch 返回值必须与 key 一一对齐，拒绝缺项、布尔值和非整数结果；
- range 异常先发布 invalid block/abort，再唤醒计算线程；
- put 异常 revoke PROCESSING 对象；
- get 异常在确认 range 调用退出后才结束 session；
- 纯 consumer 在 load hook 中推进 layer cursor，不依赖不会被调用的 save hook。

## 5. 如何运行

### 5.1 Mooncake 版本

需要安装包含以下方法的 Mooncake Python client：

```text
batch_put_session_start
batch_put_from_multi_buffer_ranges
batch_put_session_end
batch_put_session_revoke
batch_get_session_start
batch_get_into_multi_buffer_ranges
batch_get_session_end
```

如果当前发布 wheel 尚未包含这些接口，需要从合入 PR #2881 后的 Mooncake 源码构建。启动时会自动检查。

### 5.2 配置示例

```bash
export MOONCAKE_CONFIG_PATH=/path/to/mooncake.json

python -m vllm.entrypoints.openai.api_server \
  --model /path/to/model \
  --tensor-parallel-size 2 \
  --enforce-eager \
  --kv-transfer-config '{
    "kv_connector": "AscendStoreConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
      "backend": "mooncake",
      "use_layerwise": true,
      "layerwise_prefetch_layers": 2,
      "layerwise_max_transfer_blocks": 64,
      "layerwise_max_transfer_bytes": 16777216
    }
  }'
```

第一轮真机验证建议先用 TP=1、`layerwise_prefetch_layers=1`、不开 hybrid/CP/PP，再逐步扩大 TP、block 数和 prefetch 深度。

联调时可临时设置 `VLLM_ASCEND_KVPOOL_RANGE_DEBUG=1`，输出 whole-key、逐层 range 和 commit 的 JSON 审计日志；正常运行保持默认 `0`，避免逐层日志开销。

## 6. 后续优化建议

### P0：上线前应补齐

1. **固定 Mooncake 最低 commit/version**
   - 仅检查方法能避免旧 client，但不能发现 ABI/返回码语义不兼容。
   - 建议在安装文档和 CI 镜像中固定包含 PR #2881 的确切 commit 或首个正式版本。

2. **key namespace 加 schema/layout fingerprint**
   - 当前兼容原 PR，仍使用 model basename。
   - 同 basename 的不同 revision、dtype、block size、KV layout 可能冲突。
   - 建议 key 加入 tenant、model revision、dtype、block size、TP layout 和 schema version 的稳定摘要。

3. **真实 NPU E2E 与故障注入**
   - 至少覆盖 TP=1/2、kv_both、P/D、chunked prefill、请求 preempt、单 key range 失败、commit 失败、Master 重启。
   - 校验加载 KV 与本地计算 logits/token 完全一致。

### P1：性能收益较高

1. **自适应 prefetch 深度**
   - 固定 `layerwise_prefetch_layers` 不能适应不同层计算时间和网络抖动。
   - 可根据最近 N 层的 `transfer_time / compute_time`、队列深度和可用 buffer 动态控制窗口。

2. **滑动 session 窗口**
   - 当前一次为本批所有 block 打开 session；长 prompt 仍可能产生大量同时活跃 lease/session。
   - 可只为未来若干层或若干 block window 开 session，完成后滚动推进，降低 Master 状态和超时压力。

3. **减少每层 Python list 构造**
   - 当前每层仍创建 `all_buffers/all_sizes/all_offsets`。
   - 可预计算每层 range template，block id 只做向量化地址偏移；进一步可把描述符缓存到 C++/pybind 层。

4. **Scheduler 增量命中查询**
   - 当前查询全部候选 block 后再找第一个 miss。
   - 可分 window 查询并在首个不完整 block 停止，长 prompt、低命中率时显著减少 Master RPC 和 key 数。

5. **基于总字节和后端反馈的动态 batch**
   - 当前 blocks/segment bytes 是静态上限。
   - 可联合限制单次总 range 数、总 bytes，并根据队列延迟、返回码和带宽自动调整。

6. **共享 key 的本地 fan-out**
   - 多请求命中同一远端 block 时，现在同一个 key 可对应多行远端读取。
   - 可先读入一个共享 staging buffer，再在本机复制到多个目标 block；是否收益取决于远端带宽与本地 H2D 带宽。

### P2：进一步演进

1. 支持 hybrid/multi-group：key 和 object header 记录 group layout，每组独立 completeness bitmap。
2. 支持 PP/PCP/DCP：key 编码并行坐标，Scheduler 按参与 rank 集合验证完整性。
3. 支持 per-layer readiness bitmap：允许消费者在整个对象 COMPLETE 前读取已经完成的早期层；需要 Mooncake 提供可见性和一致性协议，复杂度较高。
4. 用对象 header 记录 schema/version/checksum，加载前做廉价兼容性检查，避免静默读错布局。

## 7. 验证记录

- `py_compile`：核心改动文件通过。
- Ruff lint/format：通过。
- 相关 CPU mock pytest：`285 passed, 106 subtests passed`。
- `VLLM_ASCEND_KVPOOL_RANGE_DEBUG` 严格 `0/1` 取值检查：通过。
- 未完成：真实 NPU、真实 Mooncake client/Master、多节点网络和性能基准。
