# K3.1 A3 MLA tiling 下沉：当前方案同步说明

更新时间：2026-11-14。状态：框架 mock 接入已实现，已提交 [Draft PR #21876](https://github.com/vllm-project/vllm-ascend/pull/21876)；真实 A3 算子尚未交付。

## 给同事的一分钟说明

K3.1 DSpark 的 MLA 路径需要使用投机验证后修正的实际 KV 长度。原有部分路径会把设备上的精确计数复制回 CPU，并在下一轮等待复制完成，可能阻塞后续任务提交。

本方案参考 A5 已有接入，提前完成 A3 的设备长度、metadata 和 attention 调用链，并用可计算的 mock 验证参数和框架逻辑。只有主模型其他消费者也能使用设备长度时，才同时跳过目标 D2H 和 event wait。目前已完成 247 项 CPU 测试；尚无 NPU 执行、完整模型精度或性能收益的结论。

## 1. 我们要解决什么

CPU 按调度计划维护的长度可能是上界。例如计划推进到 100，但投机验证拒绝部分 token 后，实际只有 97 个有效 KV。attention 必须使用 97，否则会读到无效内容。

原有同步链是：设备产生精确计数 → D2H 复制 → CPU 等待完成 → 准备后续参数。目标是让依赖实际长度的算子直接消费设备 Tensor，框架不再为了这些消费者回传精确长度。

这不代表移除所有 CPU 工作或所有 H2D/D2H，也不代表仅调用 metadata 算子就能消除全部空泡。算子侧和框架侧都需要适配。

## 2. A3 如何参考 A5

```text
DSpark 提供设备实际长度和 Query 分段
  → 复用 A5 的 FlashMLAMetadataBuilder 缓冲区生命周期
  → A3MLAAdapter 转换参数
  → get_scheduler_metadata
  → flash_attn_with_kvcache
  → 原有 DCP 合并及输出投影

同时：检查主模型消费者能力
  → 条件满足才成对跳过 runner 的长度 D2H 和 event wait
```

A3 保留原有的分离 KV 缓存和写入路径，使用独立的 `use_a3_mla_sink` 标志。复用的是 A5 的框架组织方式，不把 A3 标记成 A5 `use_flashmla`，避免误选 A5 缓存格式。

## 3. 算子接口约定

依据用户提供的社区仓库 PR #338，固定参考版本为 `7d3f7433669b524ce71c4e867b90dc3bc6f41264`。这是本次 mock 的接口契约，不代表算子团队已承诺所有生产场景。

| 接口或字段 | 框架需要的含义 |
| --- | --- |
| `get_scheduler_metadata(...)` | 接收设备长度和 Query 分段，返回同设备的不透明 int32 metadata |
| `flash_attn_with_kvcache(...)` | 接收 Query、分页 KV、设备长度及 metadata，返回 attention 结果 |
| `q` / `qv` | Query 分别为 64 维 RoPE 和 512 维内容；TND 格式 |
| `k_cache` / `v_cache` | 分离的 64/512 维 ND 分页缓存，一个 KV head；框架先写入，算子只读 |
| `cache_seqlens` | int32 `[B]`，每个请求实际有效的 KV 数量 |
| `cu_seqlens_q` | int32 `[B+1]`，包含起始 0 的 Query 分段边界 |
| `softmax_scale` | 显式沿用模型值，不按 576 重新推算 |
| 返回值 | 四项：output、FP32 LSE、out_accum、lse_accum；output 为 `[T,H,512]`，LSE 为 `[H,T]` |

接口没有 `seqused_q`。适配层把整段无效 Query 的 KV 长度置零，并屏蔽其 output/LSE；“分配一段、只使用其中部分 Query”的情况当前明确拒绝。

1536-token 管理块由已有框架转换成 12 个 128-token 算子页。不能把管理块大小直接当成算子 `page_size`。

## 4. 这版代码已经做了什么

| 改动 | 作用 |
| --- | --- |
| A3 adapter | 统一参数、布局、dtype、返回值和 provider 加载，集中保留真实算子适配点 |
| 数值 mock | 实际计算 MLA output/LSE；metadata mock 检测旧长度或旧分段被误复用 |
| MLA / DSpark 接线 | 设备长度进入新算子链，保留原有前处理、KV 写入和输出投影 |
| DCP / 缓冲区生命周期 | 计算本 rank 的有效长度，复用通信/LSE 合并；复用地址并逐轮刷新内容 |
| runner 同步策略 | 同一个判断控制精确计数 D2H 和对应 event wait |
| 主模型能力检查 | 检查实际初始化的 SFA、DCP SFA、indexer、KDA、hybrid state；未知消费者保留同步 |

mock metadata 只记录测试所需的配置与长度，不实现真实 NPU 的分核、切块调度算法。真实 tiling 逻辑由算子团队提供。

## 5. 临时配置怎么处理

| 当前配置 | 当前用途 | 正式接入后的计划 |
| --- | --- | --- |
| `a3_mla_sink_mode` | 默认 disabled；mock 数值实现；external 真实算子联调 | mock 留在测试中，生产路径按能力选择 |
| `a3_mla_sink_module` | 指定外部接口模块 | 包和 API 确定后固定导入 |
| `a3_mla_sink_allow_graph` | 显式允许实验性 graph 验证 | 验证后纳入正常图能力判断 |

`external` 调用的是真实算子，不是另一种 mock。当前代码仍需要上述配置，后续收敛时应一起修改加载和能力选择逻辑。

## 6. 当前边界与验证结论

首版契约为：K3.1 DSpark、V2 runner、A3、64/512、单 KV head、未量化 FP16/BF16、ND cache，max_model_len 不超过 16384，PP=PCP=1。DCP 已覆盖 CPU 逻辑模拟；PD connector、RecoverSSM、未知消费者、自适应验证和 sparse KV offload 等不放开整轮 CPU 长度同步。

mock 临时内存工作量限制为 `T × KV容量 × 512 <= 16M` 元素，用于小规模对接测试，不能用于完整 K3.1 吞吐压测。

| 验证层级 | 当前结果 |
| --- | --- |
| CPU 接口、数值、缓存、编排测试 | 155 项通过；独立 dense reference，覆盖因果、空请求、跨页、补齐、长度变化、DCP 等 |
| 现有 runner / PP 隔离回归 | 92 项通过 |
| NPU 上运行 mock | 尚未运行；需补 NPU 测试入口并验证 Torch 运算兼容性 |
| 真实算子 eager / ACL graph | 尚未运行，等待算子交付 |
| 完整 K3.1、多机 HCCL、PD | 尚未运行 |
| 空泡减少、吞吐和时延收益 | 尚无实测结论，需要 profiler 和对照压测 |

CPU 测试通过不等于生产验收通过。PR 保持 Draft，便于先评审接口及框架接线。

## 7. 接下来如何推进

1. 框架侧和算子侧对齐以上 ABI，确认 16K 上限、strides、空 KV、LSE、graph 支持及版本依赖。
2. 算子未到位时，先补单卡 A3 小规模 eager mock 验证，对照 CPU 参考；再检查动态长度、补齐和 trace。
3. 算子交付后用 external 模式接入；接口差异集中调整 `A3MLAAdapter`，复用现有框架链路。
4. 完成真算子数值、graph、KDA 状态回滚、DCP/PD、完整模型精度和性能验收。
5. 收敛临时配置，把 mock 限定在测试中，完成正式能力选择后再考虑合入生产路径。

## 8. 代码与参考

- [本次 Draft PR #21876](https://github.com/vllm-project/vllm-ascend/pull/21876)
- [完整实现方案及文件导航](a3_mla_tiling_sink.md)
- [A5 参考 PR #21542](https://github.com/vllm-project/vllm-ascend/pull/21542)
- [社区接口 PR #338](https://github.com/MinghuasLab/flash-attention-npu/pull/338)
- 适配点：`vllm_ascend/attention/a3_mla_sink.py`
- mock：`vllm_ascend/attention/a3_mla_mock.py`
- 同步判断：`vllm_ascend/worker/v2/model_runner.py::_can_skip_spec_cpu_length_sync`
