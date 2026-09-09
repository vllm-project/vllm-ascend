# KVPP 方案设计

## 1. 背景

KVPP 按层划分 TP 组内的 KV Cache 持久存储，减少每张卡重复保存的历史 KV，为更多逻辑 Block 提供空间。模型计算仍按原有 TP/EP/PP 方式执行，调度器保留完整的逻辑 Cache Spec 和 Block 编号。

MLA/SFA 场景中，TP Rank 保存重复的历史 KV Cache，会限制长上下文和并发请求的容量。KVPP 将各层的持久缓存分散到 TP 组内，在计算需要时广播该层缓存。

## 2. 方案设计

方案由按层持久存储、双缓冲复用和提前一层预取组成：

```text
当前 PP Stage 的 TP 组
    Owner Rank：持久保存负责层的 Main KV + Indexer KV
         │
         └── 整层广播 ──→ 其他 Rank：接收至 Scratch Buffer
                              │
                              └── 等待完成 → 当前层计算
                                            同时预取下一层
```

### 2.1 按层分配持久缓存

每个 PP Stage 将本地 Target Layer 按层号排序，连续、均衡地分配给 TP 组内的 Owner Rank。同层的 Main KV、Indexer KV 及其量化分量组成一个 Bundle，由同一个 Owner 持久保存。MTP Cache 在各 Rank 独立保存，不参与 Target Layer 划分。

每个 Rank 为自己负责的层分配持久缓存，为其他层复用两块 Scratch Buffer。Scratch 按 Target Layer 执行序号交替使用，分别承载当前层和下一层预取的数据。

### 2.2 物理布局与预算

一个 Bundle 内的各分量连续排列，每个分量包含该层的全部逻辑 Block。各层无需共享一块全局连续内存；广播只要求当前层的 Bundle 连续。布局根据实际 Cache Spec 构造，包含 Main KV、Indexer Data 和 Scale 等分量。

每个 Rank 的物理预算按以下关系计算：

```text
每 Block 物理成本
  = 本 Rank 负责的 Target Bundle 字节数之和
  + 本 Rank 的 MTP Cache 字节数
  + 2 × 最大 Target Bundle 字节数

可用 Block 数 = floor(可用 KV 内存 / 每 Block 物理成本)
```

Worker 将物理预算对应的 Block 数换算为逻辑预算交给 Planner，实际分配使用最终确定的 Block 数。这样可以保留完整的逻辑缓存管理，同时按 KVPP 的物理布局分配内存。

### 2.3 整层广播与预取

一次 Forward 中，只要真实请求包含已计算的历史 Token，就先预取第一层。每层 Attention 在投影之后、首次访问或写入 KV Cache 之前，等待该层广播完成，并启动下一层预取。

广播在当前 PP Stage 的 KVPP Group 内进行，Owner 为源端，每层只广播一次完整 Bundle，包含全部分量和全部逻辑 Block。接收端写入该层对应的 Scratch Buffer。各 Rank 完成当前层计算后，Owner 保留更新后的 Cache，供后续 Forward 使用。

预取使用独立传输流。Ready Event 保证广播前的数据依赖，完成事件保证等待返回时设备传输已经完成。下一层广播可与当前层计算重叠；实际掩盖效果取决于通信量与计算耗时。没有历史 KV 的 Forward，以及 Dummy/Profile 路径，不执行历史缓存广播。

## 3. 支持范围与约束

| 项目 | 范围 |
| --- | --- |
| 模型与执行 | Eager 模式下的非 Hybrid MLA/SFA 模型；Model Runner V1、V2 |
| 可叠加特性 | TP、EP、PP、Chunked Prefill、Prefix Caching、异步调度、固定步数 MTP |
| Cache 布局 | 按实际 Spec 分配，包含 LI-C8 和 SFA-C8 |
| 暂不支持 | 图模式、PCP、DCP、PD 分离、变步数 MTP |

KVPP 用额外通信换取持久缓存空间。每个 Rank 仍需保留两块最大层大小的 Scratch Buffer 和独立的 MTP Cache，因此内存收益取决于层数、TP Size 和各层缓存大小。整层广播的通信量随分配的 Block 数增长，本方案不按请求或有效 Block 裁剪 Payload。

## 4. 使用方式

在模型启动参数中启用 KVPP，组大小自动取 TP Size。例如：

```bash
vllm serve <model-path> \
  --tensor-parallel-size 2 \
  --enforce-eager \
  --additional-config '{"enable_kvpp": true}'
```

`enable_kvpp` 默认为 `false`。使用 PP 时，各 Stage 在自己的 TP 组内划分 Owner 和执行广播。

## 5. 用例测试表格

UT 看护布局、预算、分配和调度逻辑；设备与通信依赖使用现有 Mock，缓存别名关系使用真实 Tensor Storage 验证。E2E 保留一个综合场景。本表描述测试设计与预期，不代表已经执行通过；本版不包含 KVPP Nightly 用例。

### 5.1 UT

文件路径相对于仓库根目录，同一行表示一类职责下的测试。

| 编号 | 测试内容 | 场景与主要预期 | 用例文件 |
| --- | --- | --- | --- |
| UT-01 | 配置与支持边界 | 开关及布尔字符串正确解析；启用时组大小等于 TP；不支持组合在配置阶段报错 | `tests/ut/test_ascend_config.py` |
| UT-02 | 通信分组 | KVPP Group 位于本地 PP Stage，关闭时为单 Rank；与 MC2 Group 独立并正确销毁 | `tests/ut/distributed/test_parallel_state.py` |
| UT-03 | Owner 与 Bundle | 不均分、Rank 多于层数及乱序 Spec 输入；分配确定、同层分量同 Owner、MTP 不参与划分 | `tests/ut/core/test_kvpp_cache_placement.py` |
| UT-04 | 分量布局 | MLA、Packed Main、Indexer Data/Scale 与不同 Scale dtype；字节数、Offset 和总长度准确 | `tests/ut/core/test_kvpp_cache_placement.py` |
| UT-05 | 物理预算 | 不同 Rank、无 Target Owner、仅 MTP、空 Stage；完整 Block 边界正确向下取整 | `tests/ut/core/test_kvpp_cache_placement.py` |
| UT-06 | 分配与别名 | 两块 Scratch 按执行序号复用；Owner/MTP 独立；Storage 总量准确，写入不污染独立缓存 | `tests/ut/worker/test_kvpp_cache.py` |
| UT-07 | Worker 预算接入 | 保留完整逻辑 Spec；物理预算正确换算为 Planner 逻辑预算；关闭路径不改预算 | `tests/ut/worker/test_worker_v1.py` |
| UT-08 | V1/V2 分配入口 | 使用最终 Block 数；Typed View 的 dtype、shape、offset 和 Storage 别名正确 | `tests/ut/worker/test_model_runner_v1.py`、`tests/ut/worker/test_attn_utils_v2.py` |
| UT-09 | Runtime 绑定 | 非零 Storage Offset 正确换算为字节起点；广播覆盖完整 Bundle；仅 Target Main 绑定 Hook | `tests/ut/worker/test_kvpp.py` |
| UT-10 | 逐层预取 | 无历史不预取；有历史提前一层；多轮状态重置；Future 失败直接传递且不再提交下一层 | `tests/ut/worker/test_kvpp.py` |
| UT-11 | 广播完成语义 | Owner 映射到正确全局 Rank；一次传递完整 Payload；设备完成后 Future 才完成，范围外数据不变 | `tests/ut/distributed/kv_transfer/kv_pool/test_broadcast_transport.py` |
| UT-12 | 历史判断与生命周期 | V1/V2 仅根据真实请求判断历史，忽略 Padding；Dummy/Profile 不广播；调用顺序正确 | `tests/ut/worker/test_model_runner_v1.py`、`tests/ut/worker/test_model_runner_v2.py` |
| UT-13 | Attention 等待位置 | MLA/SFA Native、Fused 及有/无 Indexer 路径；投影之后、首次 Cache 访问之前等待一次 | `tests/ut/attention/test_mla_v1.py`、`tests/ut/attention/test_sfa_v1.py` |

### 5.2 E2E

固定使用 Model Runner V1，V2 接入由 UT 看护。一个 pytest 用例内依次运行 KVPP 关闭和开启的实例，其他参数相同。

| 编号 | 配置 | 步骤 | 主要预期 |
| --- | --- | --- | --- |
| E2E-01 | A3 四卡；`vllm-ascend/DeepSeek-V3.2-W8A8-Pruning`；Eager；TP=2、PP=2、EP；Chunk、Prefix、异步调度；MTP 固定 1 步；Block Size=128、逻辑 Block 数=64 | 每个实例顺序处理两个共享 384 Token 前缀、各有 16 Token 不同后缀的请求；Token Budget=128，各生成 16 Token | 首请求真实触发多次 Prefill 调度且无缓存命中；第二请求至少命中 384 Token；MTP Cache 存在且 Draft 计数大于 0；KVPP 分组不跨 PP Stage、Target Hook 排除 MTP；EP/异步实际启用；KVPP 开关前后的输出 Token ID 和文本一致 |

用例位置：`tests/e2e/pull_request/four_card/test_kvpp.py::test_kvpp_combined_features`。
