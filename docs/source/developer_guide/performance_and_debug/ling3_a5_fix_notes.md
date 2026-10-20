# Ling-3.1-flash-w4a8 A5 修复与性能审计

记录日期：2026-10-20。目标分支：`support_ling_3.0_for_0.30.2rc2`。
本次增量基于 `ee5e3390d7b9f575fc029a4f3a7dd974341ee125`，不替换该分支已有的 Ling 适配。

## 结论与验证范围

A5 已完成 TP8 服务启动和 2、8、32 路短请求并发验证，42/42 请求返回 HTTP 200 且内容非空。MTP、ACL graph replay、prefix cache 均有运行证据。但 W4A8 使用兼容性拆分路径，MTP 多步融合和部分 CPU 绑定优化没有启用，不能把功能验证当作性能验收。

修复最终限定到 A5：A3 保留原 W4A8 加载/融合路径，以及原 Ling KDA prefill 索引/清零/写回路径。本次没有 A3 实机结果，不能承诺 A3 全模型精度或性能完全无回归。

环境为 8 张 Ascend 950DT_9582，模型 `Ling-3.1-flash-w4a8`，镜像：

```text
m.daocloud.io/quay.io/ascend/vllm-ascend:v0.30.0rc2-a5-openeuler
sha256:6d0b1dd97cbacf336c7e4211b48aefd49aa8580c0907b29871a439d8cec8567a
```

关键运行参数为 TP8、DP1、EP、V2 runner、async scheduling、MTP 3、`FULL_DECODE_ONLY`、prefix caching、FP32 Mamba cache、`max-num-seqs=32`、`max-num-batched-tokens=8192`。主机只有八卡，原 TP16 配置应改为 TP8；独立 HCCL all-reduce/all-to-all 已通过。后续实际阻塞点是算子参数和缓存索引，不应继续把所有启动错误归因于 HCCL。

## 代码修复

### A5 ModelSlim W4A8

文件：`vllm_ascend/quantization/methods/w4a8/w4a8.py`。

原融合算子 `aclnnGroupedMatmulSwigluQuantWeightNzV2` 在 A5 拒绝非空 `weightAssistMatrix`，但直接删除 ModelSlim MSD 校正会改变计算结果。现在仅当量化来源为 `ascend` 且硬件 profile 为 `FP8_OPTIMIZED` 时：

- 在实例上禁用 fused activation，走 NPU GMM1 → activation/quantization → GMM2；不改变类级激活集合，也不回退 CPU/全精度权重。
- 保留 MSD 校正和 ND 权重，避免套用 A2/A3 的 NZ 转换；INT4 仍按接口要求打包为 INT32。
- 在加载时把原 per-channel 编码 scale 复制到 group-256 布局，保留位编码及量化数值；forward 不再反复展开 scale。这是算子输入布局适配，不是重新量化模型。
- 对非 EP、EPLB、expert weight lists、fused MC2 等未支持组合显式报错，不静默切换配置。此处 fused MC2 不等于所有普通 MC2 通信路径。
- 新版启动时通过 `logger.warning_once` 明示拆分路径及潜在吞吐代价。已采集的旧服务日志没有这条新告警；旧版的拆分行为是通过源码确认的。

非 A5 或 compressed-tensors 权重继续使用原路径。新增硬件判断在构造阶段执行，不增加每次 forward 的硬件查询。W4A8 修改位于通用量化 scheme，也适用于其他 A5 ModelSlim W4A8 使用者；当前模型级实机验证仅覆盖 Ling。

### A5 Ling KDA 状态复制

文件：`vllm_ascend/ops/bailing_moe_v3_kda.py`。

原 `recurrent_state[state_indices]` 在混合 attention/Mamba 缓存的有间隔行布局上触发 `aclnnIndex` 错误。参考 Kimi K3 的既有实现，A5 Ling 层通过 `_requires_kda_state_copy` 接入 worker 已有的初始化流程：

- cache 绑定后预编译并准备 `StateCopyPlan`；prefill 只 gather 被选中的 state 行，并根据 `has_initial_state` 清零新序列。
- chunk KDA 完成后通过同一 plan scatter 最终状态，保持未选中页不变；支持页间隔、非零 storage offset 和非连续索引。
- 同步过滤空序列对应的索引和初始状态标志，保持它们与 `keep_meta` 一致。
- 缺少 plan 或尚未准备时显式报错，不在请求中 JIT 或悄悄复制整个缓存池。

本轮审查发现最初的 Ling marker 无条件启用，会把 A3 导入尚未充分实机验证的 `StridedKDAFallbackPlan`。最终版本把 marker 与 prefill gather/scatter 同时限定为 A5，A3 继续原高级索引、`clear_ssm_states` 和写回路径。decode/MTP recurrent kernel 没有因这次增量更改。

`FP8_OPTIMIZED` 是仓库现有硬件分类：当前 A5 对应该分类，A3 对应 `STANDARD`。在运行中的 A5 镜像里也读取到了 `DeviceAdaptorFamily.FP8_OPTIMIZED`。

## A3 影响判断

| 部分 | 最终代码在 A3 上的行为 | 已验证边界 |
| --- | --- | --- |
| W4A8 权重加载 | 保留原 NZ、packing、expert-list 逻辑；不展开 A5 scale | 本地门控回归和源码比较 |
| W4A8 计算 | 保留原 fused activation 分派；不套用 A5 的 EP/EPLB/fused MC2 限制 | 本地门控回归；没有 A3 kernel benchmark |
| KDA prefill | 不注册 StateCopyPlan；保留原索引、清零和写回 | 真实 CPU Torch 的 A3/A5 分支、初始状态和缓存语义回归 |
| KDA decode/MTP | 本次增量未改动 | 不等价于重新完成 A3 全模型验收 |

因此，本修复不会主动在 A3 启用已知的 A5 拆分路径或非 950 StateCopyPlan fallback。A3 上仍应补做相同模型的生成/精度、prefix cache、长上下文与吞吐对比后再作性能保证。这里讨论的是本次增量，不代表该适配分支以前的所有变更都已经完成 A3 验证。

## 完整启动日志中的 fallback 与未启用优化

证据来自成功运行日志 `vllm-tp8-03-kimi-kda.log`，行号按 LF 计算，进度条中的 CR 不计为新行。另有两次失败日志 `vllm-tp8-01.log`（融合算子参数错误，697 行）和 `vllm-tp8-02-a5-fallback.log`（KDA Index 错误，1437 行）。

### 确实存在或需要关注的性能限制

| 项目 | 证据/实际行为 | 性能含义 |
| --- | --- | --- |
| W4A8 MLP 拆分 | A5 `fused_activations` 为空，走独立 GMM 和 activation/quantization | 多次 kernel launch 和中间张量读写；原融合路径不能正常运行，因此尚无可比较的同机融合基线，损失百分比未知 |
| MTP 多步融合未支持 | 1672 行明确指出 `ASCEND_MLA` 不支持 fused multi-step，每一步重建 attention metadata | MTP 仍工作，单步 draft graph 仍可 replay；多步 metadata 处理有额外成本 |
| CPU 绑定不完整 | 1782–1813 行八个 worker 均找不到 `uvb_poll_window_thread`；1785–1819 行仍有 worker/NUMA 绑定结果 | 跳过的是宿主机轮询线程绑定，不是全部 CPU binding。容器当前未用 `--pid=host`；宿主线程原有 affinity 和实际收益未测 |
| cache 对齐与填充 | 518–519 行 block size 从 128 调整为 640，Mamba page padding 23.29%；1652 行增加 7 个 padding layers，最多 11.11% KV 浪费 | 会影响容量和可重用前缀粒度；两项百分比不能相加，也不是吞吐下降比例 |
| draft KV group 未识别 | 1653 行告警；upstream `kv_cache_coordinator.py` 对没有 draft 标记的情况保守处理所有组的 speculative 末块 | 不代表 MTP 无 cache；可能减少 prefix reuse，需进一步验证分组/标记是否可优化 |
| FlashComm1/shared expert 优化 | 40 行 FlashComm1 disabled；当前配置的 shared expert DP、overlap 均为默认关闭 | 属于未启用的配置优化；routed experts 的 EP8 已启用，不能据此认为 EP 失效 |

### 不能仅凭告警认定模型已降级的项目

- **causal_conv PyTorch fallback（94 行）**：`patch/worker/patch_triton.py` 的通用补丁确实报告 fallback，但 Ling 自己的 KDA 文件直接导入并调用 `fla_npu.ops.ascendc` 的 conv kernel。因此这条告警不能证明当前 Ling conv 热路径用了同步 PyTorch 实现。
- **npugraph 磁盘缓存跳过（648 行起）**：含 Triton 的图不跨进程序列化，避免 process-local kernel index 失效；仍返回编译结果。614–621 行的重编译也属于启动成本。1882、1883、1901 行分别确认 draft decode、target、draft prefill 的 ACL graph replay 已发生。主模型使用明确配置的 `FULL_DECODE_ONLY`，不应声称所有 prefill 都 FULL graph；尚未测 replay 命中比例。
- **`@eager_break_during_capture`**：运行时 `VLLM_USE_BREAKABLE_CUDAGRAPH=0`，该 decorator 在此模式返回原函数，不能据其名称判定实际发生了 eager break。
- **`flashinfer_all2allv` 名称**：`ascend_config.py` 明确把它用作禁用 SP 的配置技巧，不意味着在 NPU 上执行 CUDA FlashInfer。通信器已经换成 Ascend 实现；MC2/ALLGATHER/ALLTOALL 随容量和硬件策略动态选择，当前 INFO 日志不能识别每个 batch 的实际方法。
- **cascade attention**：upstream async+spec 的关闭提示与 Ascend 对 GPU-only 参数的归一化同时出现；没有证据说明本补丁关闭了原本可用的 Ascend cascade。prefix cache 是另一功能，已有命中证据。
- **`norm_quant, act_quant`**：日志报告这些 pass 已启用，但不证明每个算子模式都命中融合，也不抵消 W4A8 MLP 拆分的事实。
- **KDA plan 的具体类型**：初始化器可选 `KDAStateCopyPlan` 或 `StridedKDAFallbackPlan`，当前 INFO 日志没有打印类型。A5、FP32、内部连续且仅页间有 padding 的布局满足前者的源码条件，独立探针也验证了该布局；但这些不等于直接观测了每个在线层的 plan 类型，需要启动诊断或 profiler 补证，不能宣称已经排除全部隐式 fallback。
- **internal format、DeepSelect、旧环境变量**：A5 base-format 提示不是 CPU 回退；可选 CUDA DeepSelect 缺失未发现用于此模型；`VLLM_VERSION` 与 `VLLM_USE_V1` 被忽略，部署脚本已经移除。V1 engine 与 V2 model runner 是不同层次，日志同时出现不矛盾。

## 实际特性与测试证据

并发测试的 `metrics-before.txt` / `metrics-after.txt` 差值如下，避免把历史累计值误算为本次结果：

| 指标 | 本次 42 请求增量 |
| --- | ---: |
| drafts | 1762 |
| draft tokens | 5286 |
| accepted draft tokens | 2387 |
| draft token 接受率 | 45.157% |
| 平均接受长度（包含每轮目标 token） | 2.3547 |
| 第 1 / 2 / 3 个位置接受率 | 75.709% / 44.041% / 15.721% |

MTP 确实执行，但接受率不能直接换算成加速比。第三步贡献较低，3 步未必是这份负载的最优设置。

先前长前缀重复请求的 `smoke-prefix-metrics.txt` 记录了 1280 个命中 token；本次 42 个短请求是 1851 queries、0 hits，不能把末尾历史累计约 28% 当作本次并发命中率。640-token 对齐粒度也限制了短 prompt 的复用机会。

三档短请求并发均达到预期峰值，测试前后 `/health` 返回 200。32 并发不是 32 条满长上下文的容量验证；日志对 262144-token 请求估算的最大并发仅 10.80×。async 在配置中开启，未发现关闭记录，但没有 overlap trace，收益尚未量化。

已完成验证：

- A5 独立 HCCL all-reduce/all-to-all；W4A8 group-256 及 MLP 数值探针；KDA 有间隔行、非零 offset、非连续索引的实机探针。
- A5 短生成、重复长前缀、2/8/32 并发；成功日志未发现实际 ERROR/FATAL/Traceback。
- 本地新增回归用例包含 scale 位编码/数值不变、硬件与量化来源门控、显式拒绝不支持组合、加载后 scale 复用、A5 告警、A3 原 KDA 路径和 A5 缺 plan 报错。
- 本机缺完整 vLLM/NPU 导入依赖，定向 CPU 回归采用真实 Torch、AST 加载实际源码及用例、mock 缺失运行时边界；不能将其报告为完整 pytest 或 A3 实机通过。

本轮新增的 A3 门控和一次性告警经过本地验证，运行中服务未热替换或重启；A5 的数学计算路径与此前通过 42 请求测试的实现一致。该区分也适用于随附适配 ZIP 的验证记录。

## 下一步性能验证建议

1. 固定 prompt/output 长度、并发和预热次数，对比 MTP 关闭与 1/2/3 步，采集 TTFT、TPOT、p50/p95、输出吞吐和接受率；不能只比较自然结束长度不同的 smoke 请求总时间。
2. profiler 分解 unfused GMM1、activation/quantization、GMM2、KDA state copy、draft metadata 和通信耗时；需要恢复融合时必须保留 MSD 校正并做数值回归，不能直接把 assist matrix 置空。
3. 在维护窗口测试带 `--pid=host` 的容器，验证轮询线程确实被找到、affinity 合适，并进行同负载 A/B。当前运行服务不因审计而自动重建。
4. 核实 draft cache group 标记与 hybrid page 对齐策略，再对长前缀复用、长上下文、显存容量进行验证。FP32→低精度 state 的改变涉及数值行为，应单独评估。
5. 用 DEBUG/trace 确认实际 MoE 通信方法、图 replay 比例和 CPU/NPU overlap，再判断 FlashComm1、shared expert DP/overlap 是否值得启用；默认关闭不等于发生故障。
