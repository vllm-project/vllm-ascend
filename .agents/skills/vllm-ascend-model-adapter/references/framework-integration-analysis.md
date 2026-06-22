# Framework Integration Analysis

本文档分析 vLLM Ascend 当前在 framework 层的接线能力，聚焦 scheduler、worker/model runner、sampler、KV cache interface、ACL graph，以及这些通用模块如何被 Ascend 覆盖或 patch。

## 1. 这一层解决什么问题

framework 层当前主要负责：

- 让 upstream vLLM 的通用运行时逻辑在 Ascend 上成立
- 把模型层能力下沉到 worker / scheduler / sampler / cache interface
- 维持图模式、spec decode、KV cache、分布式通信的一致契约

## 2. 当前能力总览

当前 Ascend 在 framework 层不是简单“继承一下 worker”，而是通过：

- `worker/model_runner_v1.py`
- `worker/v2/*`
- `core/*`
- `patch/platform/*`
- `patch/worker/*`

共同构成覆盖层。

现有 baseline 文档见：

- [framework-integration-baseline.md](/home/cmq/code/vllm-ascend/.agents/skills/vllm-ascend-model-adapter/references/framework-integration-baseline.md)

## 3. 当前实现的关键能力

### 3.1 model runner 是 framework 层中枢

`NPUModelRunner` / `worker/v2/model_runner.py` 负责：

- 构造 attention metadata
- 维护 mrope/xdrope positions
- 管理 dynamic_eplb、KV scales、spec decode
- 与 ACL graph / backend 交互

见：

- [worker/model_runner_v1.py](/home/cmq/code/vllm-ascend/vllm_ascend/worker/model_runner_v1.py)
- [worker/v2/model_runner.py](/home/cmq/code/vllm-ascend/vllm_ascend/worker/v2/model_runner.py)

这说明很多“看起来像层实现”的问题，实际已经上升为 runner 级契约。

### 3.2 KV cache interface 已有 Ascend 特化

当前仓库存在：

- `core/single_type_kv_cache_manager.py`
- `patch/platform/patch_kv_cache_interface.py`
- `patch/platform/patch_kv_cache_utils.py`
- `worker/v2/attn_utils.py`

说明 KV cache 不是简单沿用 upstream；Ascend 已对 cache 形状、metadata、A5/MLA/C8 等路径进行特化。

### 3.3 scheduler 侧已有多项 Ascend 扩展

当前 framework 层并不只 patch 执行器，还包括：

- `patch/platform/patch_scheduler.py`
- `core/scheduler_dynamic_batch.py`
- `core/scheduler_profiling_chunk.py`

说明 Ascend 已经把调度策略也纳入平台能力的一部分，尤其是在动态 batch、profiling chunk 等场景。

### 3.4 sampler / spec decode 已有专门接线

当前仓库中存在：

- `sample/sampler.py`
- `sample/rejection_sampler.py`
- `spec_decode/*`
- `patch/worker/patch_spec_decode_worker.py`
- `patch/worker/patch_v2/patch_triton.py`

说明 sampler 与 spec decode 并不是外部附加功能，而是 framework 层正式覆盖范围。

### 3.5 ACL graph 已深入 framework 层

当前 ACL graph 不只是 attention 内部实现，而是贯穿：

- compilation manager
- model runner
- spec decode
- MM encoder attention

相关入口：

- [compilation/acl_graph.py](/home/cmq/code/vllm-ascend/vllm_ascend/compilation/acl_graph.py)
- [ops/mm_encoder_attention.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/mm_encoder_attention.py)
- [worker/v2/spec_decode/eagle/aclgraph.py](/home/cmq/code/vllm-ascend/vllm_ascend/worker/v2/spec_decode/eagle/aclgraph.py)

这意味着 framework 层当前已经把 graph capture/replay 看作运行时正式能力，而不是附加优化。

## 4. 当前结构假设

当前 framework 层隐含这些假设：

- 仅替换模型层 op 不足以支撑 Ascend 运行时
- attention metadata、KV cache、graph params 需要 runner 级统一管理
- upstream 通用模块常常需要 patch 才能对齐 Ascend 约束
- v1 与 v2 worker 路径都需要单独维护

## 5. 已知边界与风险

当前主要边界有：

- framework patch 面较大，upstream 漂移风险一直存在
- v1/v2/spec decode/graph 路径并不完全同构
- 很多能力已跨越单层实现，问题可能出在 metadata 契约而不是算子本身
- 某些特性只在 v1 或 v2 中成熟，不代表两边完全对齐

## 6. 分析这一层时应该看什么

建议优先看：

- 是 v1 还是 v2 runner
- attention metadata / block table / slot mapping 契约
- scheduler 是否参与问题触发
- sampler/spec decode 是否改写了运行路径
- graph capture 是否改变了输入更新方式

## 7. 相关代码

- [worker/model_runner_v1.py](/home/cmq/code/vllm-ascend/vllm_ascend/worker/model_runner_v1.py)
- [worker/v2](/home/cmq/code/vllm-ascend/vllm_ascend/worker/v2)
- [core](/home/cmq/code/vllm-ascend/vllm_ascend/core)
- [patch/platform](/home/cmq/code/vllm-ascend/vllm_ascend/patch/platform)
- [patch/worker](/home/cmq/code/vllm-ascend/vllm_ascend/patch/worker)
- [framework-integration-baseline.md](/home/cmq/code/vllm-ascend/.agents/skills/vllm-ascend-model-adapter/references/framework-integration-baseline.md)
