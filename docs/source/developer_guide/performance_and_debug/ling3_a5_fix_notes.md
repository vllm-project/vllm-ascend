# Ling-3.1-flash-w4a8 A5 修复与性能审计

记录日期：2026-10-20。目标分支：`support_ling_3.0_for_0.30.2rc2`。
本次增量基于 `ee5e3390d7b9f575fc029a4f3a7dd974341ee125`，不替换该分支已有的 Ling 适配。

## 结论与验证范围

A5 已完成 TP8 服务启动和 2、8、32 路短请求并发验证，42/42 请求返回 HTTP 200 且内容非空。MTP、ACL graph replay、prefix cache 均有运行证据。但 W4A8 使用兼容性拆分路径，MTP 多步融合和部分 CPU 绑定优化没有启用，不能把功能验证当作性能验收。

修复最终限定到 A5 的 Ling V3：W4A8 兼容处理只接受已解析的 `BailingMoeV3ForCausalLM` / `BailingMoeV3MTPModel` 且激活为 SiLU。后续重构使 Ling 接入 Kimi SiTU 分支使用的公共 GMM1 和激活量化入口，保留各自的激活算法。Kimi K3 和其他模型保持原算子选择；A3 保留原 W4A8 加载/融合路径，以及原 Ling KDA prefill 索引/清零/写回路径。本次没有 A3 或 Kimi K3 实机结果，不能承诺它们的全模型精度或性能完全无回归。

环境为 8 张 Ascend 950DT_9582，模型 `Ling-3.1-flash-w4a8`，镜像：

```text
m.daocloud.io/quay.io/ascend/vllm-ascend:v0.30.0rc2-a5-openeuler
sha256:6d0b1dd97cbacf336c7e4211b48aefd49aa8580c0907b29871a439d8cec8567a
```

关键运行参数为 TP8、DP1、EP、V2 runner、async scheduling、MTP 3、`FULL_DECODE_ONLY`、prefix caching、FP32 Mamba cache、`max-num-seqs=32`、`max-num-batched-tokens=8192`。主机只有八卡，原 TP16 配置应改为 TP8；独立 HCCL all-reduce/all-to-all 已通过。后续实际阻塞点是算子参数和缓存索引，不应继续把所有启动错误归因于 HCCL。

## 代码修复

### 维护原则：优先复用 Kimi K3 的公共实现

Ling 应尽量接入社区为 Kimi 优化的公共实现，避免长期维护一份复制代码。该原则已写入仓库 `AGENTS.md`。结构相似是查找复用点的依据，具体能否共用算子仍取决于激活函数、量化格式、缓存布局和输出约定。

| 部分 | 复用关系与边界 |
| --- | --- |
| KDA | 共用 `ops/kda.py` 的 chunk/recurrent kernel；本次让 A5 Ling 接入已有的 `StateCopyPlan` 与 worker 准备流程 |
| MLA | 共用 Ascend MLA attention 实现；保留各自 projection、gate 和 RoPE 语义 |
| MoE | 共用 FusedMoE runner、通信/dispatch/combine 与 `apply_moe_mlp`；不复制 Kimi 的 latent MoE 变换到 Ling |
| 普通 W4A8 | 共用量化 scheme、GMM1 反量化 helper、激活量化分派和 GMM2；保留 SiLU/SiTU 的真实区别 |
| MTP | 继续使用公共 runner/cache；Ling 与 Kimi 的 hidden/logits/循环状态输出约定不同，不能直接替换模型实现 |

复用意味着公共路径的后续优化更容易覆盖 Ling，不意味着 Kimi 的所有特有优化会自动适用。新增特例前应先检查公共接口能否表达差异，并检查目标硬件实际构建的 kernel。

### A5 Ling V3 ModelSlim W4A8

文件：`vllm_ascend/quantization/methods/w4a8/w4a8.py`。

原融合算子 `aclnnGroupedMatmulSwigluQuantWeightNzV2` 在这份 Ling A5 配置下拒绝非空 `weightAssistMatrix`，但直接删除 ModelSlim MSD 校正会改变计算结果。兼容处理仅在以下条件同时成立时启用：当前 MoE 层进入普通 `W4A8_DYNAMIC` scheme、量化来源为 `ascend`、硬件 profile 为 `FP8_OPTIMIZED`、vLLM 已解析架构为 `BailingMoeV3ForCausalLM` 或 `BailingMoeV3MTPModel`，以及文本配置的 `hidden_act` 为 `silu`。启用后：

- 不再清空实例的 `fused_activations`。Ling 和 Kimi SiTU 都通过 `apply_gmm1_act_quant`，未走全融合时调用公共 `_apply_dequantized_gmm1` 与 `_apply_activation_quant`，随后复用 GMM2。这个联合入口不保证内部只有一个 kernel。
- Ling 的 SiLU 直接复用既有 `_unified_apply_activation`，随后调用公共 `apply_act_quant`，不复制一份限幅/激活实现；保留 `npu_swiglu` → `npu_dynamic_quant`、原 BF16 中间舍入和 `swiglu_limit`。Kimi 保留 SiTU 的 `beta` / `linear_beta`、已有融合优先级和算子调用。没有用 SiTU 近似 SiLU，也没有回退 CPU/全精度权重。
- 公共 GMM1 完成后立即释放原 dispatch 输入，保持旧 Ling 的释放时机，避免重构让输入与激活/量化中间 buffer 额外重叠。
- 保留 MSD 校正和 ND 权重，避免套用 A2/A3 的 NZ 转换；INT4 仍按接口要求打包为 INT32。
- 在加载时把原 per-channel 编码 scale 复制到 group-256 布局，保留位编码及量化数值；forward 不再反复展开 scale。这是算子输入布局适配，不是重新量化模型。
- 对非 EP、EPLB、expert weight lists、fused MC2 等未支持组合显式报错，不静默切换配置。此处 fused MC2 不等于所有普通 MC2 通信路径。
- 公共 GMM helper 对已有 group-256 scale 直接复用 buffer，对 per-channel scale 仅增加一个 view 维度，不在请求中重复展开。新版启动时通过 `logger.warning_once` 明示缺少 GMM+激活+量化全融合；其吞吐代价仍需测量。已采集的旧服务日志没有这条新告警；旧版的拆分行为是通过源码确认的。

非上述 Ling 架构、非 SiLU、非 A5 或 compressed-tensors 权重继续使用原路径。新增判断在构造阶段执行，不增加每次 forward 的硬件查询。模型识别采用 `model_config.architecture` 的实际解析结果，不用模型目录名、泛化的 `bailing_hybrid` 类型或候选 architecture 列表猜测；缺少匹配信息时不启用兼容处理。文本配置优先使用 `hf_text_config`，缺失时使用 `hf_config`。

### 为什么复用公共路径，但不能直接使用 SiTU kernel

此前只使用 A5 + ModelSlim 的门控过宽：凡进入普通 W4A8 MoE scheme 的 A5 模型都会命中，Kimi K3 也不能例外。清空整个 `fused_activations` 会同时绕过 SiLU 和 SiTU 分支，并把额外的 EP/EPLB/fused MC2 限制施加给其他模型。因此已将该门控收窄到上面的 Ling V3 配置；这是一项经特定模型验证的兼容修复，不是所有 A5 W4A8 的通用规则。

相似的 KDA/MLA 模型结构不意味着 MoE 激活和算子接口相同：

| 项目 | Ling V3 本次运行 | 普通 ModelSlim W4A8_DYNAMIC 的 SiTU 路径 |
| --- | --- | --- |
| 激活 | 实际 checkpoint 配置 `hidden_act=silu` | `AscendKimiMoE` 把 `config.hidden_act` 及 SiTU 参数传给 MoE |
| 原 GMM1/激活处理 | `grouped_matmul_swiglu_quant_v2`，传入 MSD assist，触发本次参数错误 | `apply_gmm1_act_quant` 内独立的 `MoEActivation.SITU` 分支 |
| 算子选择 | 公共 GMM1 + SwiGLU + dynamic quant，保留 MSD | 满足硬件/输入条件时调用 `gmm_dequant_situ_quant`；否则为公共 GMM1 + `dequant_situ_quant` |
| 本次最终改动 | 兼容处理接入联合入口，保留已验证的计算步骤 | 抽取公共 helper，不改变原融合优先级、权重布局或配置支持范围 |

SiTU gate 包含 `beta * tanh(gate / beta) * sigmoid(gate)`，SiLU 不包含该 tanh 变换。现有 SiTU kernel 没有严格 SiLU 模式：`beta=0` 被拒绝，负值仍保留 tanh，`linear_beta` 只控制 up 分支。不能为了进入同一个 kernel 改变 checkpoint 的激活算法。

**对早期分析的更正：Python 分支存在不代表普通 SiTU 算子支持 A5。** 当前仓库的 `GMM_DEQUANT_SITU_QUANT` 能力属于 A3；`csrc/build_aclnn.sh` 仅在 A2/A3 构建 `dequant_situ_quant`，仅在 A3 构建 `gmm_dequant_situ_quant`。A5 构建的是另一套 `situ_mx_quant` / `grouped_matmul_situ_quant`。后者针对 MXFP8/MXFP4，不能直接代替本次 INT8/INT4 MSD 路径。之前根据普通 W4A8 的 Python 分支推断 Kimi A5 已有可用 SiTU 路径，证据不足，应撤回这一判断。

`--quantization ascend` 只选择 ModelSlim 配置入口，具体数值格式由每层 quant description 决定。另一种 `W4A8_MXFP` 使用独立的 `w4a8_mxfp4.py` scheme，在 A5 的 SiTU 算子选择也不同；它本来就不会经过此次修改的普通 W4A8 类。不能仅凭模型名称或命令中的 `ascend` 判断实际量化格式。

当前源码确有 Kimi A5 优化：`models/kimi_k3.py` 提供 latent RMSNorm + MXFP8 融合；`w4a8_mxfp4.py` 在 SiTU、group-32 等条件满足时选择融合；`device/device_op.py` 的 A5 adaptor 要求 `W4A8MXFP` 并调用 `grouped_matmul_situ_quant_weight_nz`。这些支持应作为 Ling 持续复用的参考，但需要相同的权重、scale、激活和结构约定，不能直接用于本次普通 INT4/INT8 MSD 权重。

本次以当前仓库的 Kimi 实现作为复用基准，没有取得一份 Kimi A5 checkpoint 或启动记录。模型级支持不能推导出任意激活、量化格式与硬件组合都使用同一个 kernel；具体路径必须结合 quant description 判断。仓库 Kimi 部署教程中的普通 W4A8 示例明确面向 A3。没有把未知的 Kimi 权重推断成 MXFP4，也没有把源码核查报告为 Kimi A5 实机验收。

### SwiGLU+量化融合候选

在目标 A5 镜像中，用 `torch_npu.npu_dequant_swiglu_quant` 的 `swiglu_mode=0`、动态 INT8 量化对模拟 GMM1 输出形状的随机 BF16 张量进行独立探测，4 个小张量场景通过。三个随机场景覆盖 4/32/128 行和 2560 列，另有全零场景。与现有 `npu_swiglu` → `npu_dynamic_quant` 相比，随机场景最大 INT8 差值为 1 个量化档，反量化输出差异的 L2 范数约为 FP32 激活参考范数的 0.67%–0.71%；零输入一致。

该结果证明融合候选在此镜像可执行，不证明全模型精度或吞吐收益。融合实现可能省略两步路径中的 BF16 中间舍入，因此本次没有默认启用。后续应在公共激活量化入口加入经过精度、边界参数、图模式和性能验证的实现，使兼容条件相同的模型共同受益，而不是再为 Ling 复制一套路径。

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
| W4A8 MLP 拆分 | 已采集服务通过空 `fused_activations` 拆分；本地新版本接入公共联合入口，内部仍保留相同 GMM / SwiGLU / dynamic quant 步骤 | 多次 kernel launch 和中间张量读写；联合入口不等于全融合。原融合路径不能正常运行，性能损失百分比未知 |
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
- 模型范围回归覆盖 Ling V3 主模型/MTP、Kimi K3、旧 Bailing 及其他模型，并检查非 Ling A5 的原 SiTU 分派、加载布局与并行配置不被兼容处理改变。
- 公共路径重构通过 15 项定向 CPU 回归，覆盖 Kimi 全融合优先级和参数、expert list、无 bias、Ling BF16 舍入/限幅及真实 `set_` 输入释放次序。
- A5 对最终公共方法完成 32 组小张量新旧对照：2 个 expert、hidden 4096、intermediate 1280，覆盖空 expert、`swiglu_limit` 0/0.75、counts/cumulative 两种 group list 和重复调用。INT8 激活结果、动态 scale、BF16 GMM2 输出均逐位一致，权重/scale/MSD bias buffer 复用；日志为 `w4a8-shared-pipeline-npu-regression.log`。探针通过 AST 加载实际新方法及通用激活函数，真实执行 NPU 算子和输入释放；不是全模型或图模式验收。
- 本机缺完整 vLLM/NPU 导入依赖，定向 CPU 回归采用真实 Torch、AST 加载实际源码及用例、mock 缺失运行时边界；不能将其报告为完整 pytest 或 A3 实机通过。

新增的硬件/模型范围门控、公共路径重构和一次性告警经过本地验证，公共计算方法另有上述 A5 小张量对照。运行中服务未热替换或重启；Ling A5 保留此前通过 42 请求测试的算子计算步骤。上述并发结果属于原服务，不能报告为新版本已重新完成全模型并发验收。该区分也适用于随附适配 ZIP 的验证记录。

## 下一步性能验证建议

1. 固定 prompt/output 长度、并发和预热次数，对比 MTP 关闭与 1/2/3 步，采集 TTFT、TPOT、p50/p95、输出吞吐和接受率；不能只比较自然结束长度不同的 smoke 请求总时间。
2. profiler 分解 unfused GMM1、activation/quantization、GMM2、KDA state copy、draft metadata 和通信耗时；需要恢复融合时必须保留 MSD 校正并做数值回归，不能直接把 assist matrix 置空。
3. 在维护窗口测试带 `--pid=host` 的容器，验证轮询线程确实被找到、affinity 合适，并进行同负载 A/B。当前运行服务不因审计而自动重建。
4. 核实 draft cache group 标记与 hybrid page 对齐策略，再对长前缀复用、长上下文、显存容量进行验证。FP32→低精度 state 的改变涉及数值行为，应单独评估。
5. 用 DEBUG/trace 确认实际 MoE 通信方法、图 replay 比例和 CPU/NPU overlap，再判断 FlashComm1、shared expert DP/overlap 是否值得启用；默认关闭不等于发生故障。
