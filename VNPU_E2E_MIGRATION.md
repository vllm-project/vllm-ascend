# vNPU E2E 理论分组与实测记录

本文件用于重新采集 A2B3 vNPU 的真实运行数据。初始分组不参考上一轮
OOM、卡住或通过结果，只根据测试文件的完整参数集、模型规模和运行配置做
理论判断。测试文件不再拆分；同一文件按其中资源需求最大的用例确定组别。

第 4、5 列已开始记录 [Actions run 29482312479](https://github.com/vllm-project/vllm-ascend/actions/runs/29482312479)
的真实结果。该运行中 1/4 卡第 2、3 桶失败，第 1、4 桶成功；1/2 卡两个桶
均成功。pytest skip 不计为通过，也不填写实测列。

[Actions run 29487123938](https://github.com/vllm-project/vllm-ascend/actions/runs/29487123938)
继续验证了迁移结果：8 个 1/4 卡桶全部成功，34 个 A2 UT 文件均被调度；
4 个 1/2 卡桶中 3 个成功，失败桶包含 reranker 和 MiniCPM。HiTest job 的
错误按本轮约定忽略，不作为迁移依据。

## Runner 与判断口径

- 1/4 卡：`linux-aarch64-a2b3-v-quarter`，8 GiB NPU 显存，6 个 CPU 核，
  `MAX_JOBS=6`，4 个分桶。
- 1/2 卡：`linux-aarch64-a2b3-v-half`，16 GiB NPU 显存，11 个 CPU 核，
  `MAX_JOBS=11`，2 个分桶。
- 整卡：当前按 32 GiB NPU 显存评估。
- BF16/FP16 权重按约 2 bytes/parameter 估算，W8A8 权重按约
  1 byte/parameter 估算；峰值还要计入 KV cache、ACL Graph、编译
  workspace、激活、草稿模型和 NPU 运行时常驻内存。
- 8 GiB 组原则上只放估算峰值不超过约 6 GiB 的文件；估算峰值约
  6–12 GiB 的文件放入 16 GiB；超过约 12 GiB、存在大批量长上下文，
  或要求物理卡能力的文件保留整卡/专用硬件。
- 第 4、5 列使用文件级结果：`✅` 表示至少有一个用例实际执行且没有失败，
  `❌` 表示至少一个用例失败；全部 skip 或未执行保持为空。

## A2 UT 的 vNPU 路由

- `tests/ut/**/a2/test_*.py` 当前共 34 个文件，统一从物理 `a2_x1` 路由到
  `a2_quarter_x1`。
- 这些文件不加载预训练模型，主要使用小型合成张量；TP/CP 多卡语义通过
  mock 模拟，未建立真实多卡 HCCL。`test_acl_graph.py` 也 mock 了
  `torch.npu.NPUGraph` 和 graph context，不会执行 E2E 那种批量图捕获。
- 静态评估未发现接近 8 GiB 的 NPU 分配；本轮 1/4 卡实测也未出现硬件或
  容量失败。若后续出现硬件能力问题，应按具体 UT 单独回退，不应把全部
  A2 UT 一次性退回物理卡。

### A2 UT 测试文件

以下路径相对于仓库根目录。run 29487123938 已调度全部 34 个 A2 UT 文件：
32 个文件存在实际执行且无失败的用例，记为 `✅`；另外 2 个文件全部 skip，
实测列保持为空。

| 测试文件 | 主要验证内容 | 目前所在组别 | 1/4卡实测结果 | 1/2卡实测结果 | 放在该 runner 的原因 |
| --- | --- | --- | --- | --- | --- |
| `tests/ut/attention/a2/test_attention_cp.py` | CP attention 的 prefill/decode、KV cache 和输出合并 | 1/4 卡（8 GiB） | ✅ |  | 不加载模型；collective 和大部分 NPU kernel 被 mock，较大的显式张量位于 CPU。 |
| `tests/ut/attention/a2/test_attention_cp_precision.py` | CP attention 的 prefill/decode/mixed/MTP 精度 | 1/4 卡（8 GiB） |  |  | 13 个参数全部 skip，没有可记录的通过用例。 |
| `tests/ut/attention/a2/test_attention_v1.py` | V1 attention backend、metadata 和 graph 参数更新 | 1/4 卡（8 GiB） | ✅ |  | 使用小型合成张量，关键 attention/graph 接口被 mock，不进行批量 ACL Graph 捕获。 |
| `tests/ut/attention/a2/test_attention_v1_precision.py` | Qwen3 causal/encoder attention 精度，模拟 TP1/2/4 | 1/4 卡（8 GiB） | ✅ |  | 不加载 Qwen3 权重；TP 仅用于配置和 head 切分，单逻辑 NPU 上运行合成 kernel。 |
| `tests/ut/attention/a2/test_common_cp.py` | CP attention 输出、LSE reshape/update 公共逻辑 | 1/4 卡（8 GiB） | ✅ |  | `all_to_all_single` 等通信被 mock，仅处理小型合成张量。 |
| `tests/ut/attention/a2/test_mla_cp.py` | MLA 的 PCP/DCP 预处理、KV 重排和 decode/prefill | 1/4 卡（8 GiB） | ✅ |  | 不加载 DeepSeek 权重；collective 与主要 NPU kernel 被 mock，验证控制和张量布局。 |
| `tests/ut/attention/a2/test_mla_cp_precision.py` | MLA CP 精度，模拟 TP1/2/4 | 1/4 卡（8 GiB） |  |  | 204 个参数全部 skip，没有可记录的通过用例。 |
| `tests/ut/attention/a2/test_mla_precision.py` | DeepSeek-V2 MLA backend 精度 | 1/4 卡（8 GiB） | ✅ |  | 仅读取模型配置并构造合成 Q/K/V、RoPE cache，不加载模型权重，显存远低于 8 GiB。 |
| `tests/ut/attention/a2/test_mla_v1.py` | MLA backend/metadata、KV cache、prefill/decode 和权重格式转换 | 1/4 卡（8 GiB） | ✅ |  | 以 mock 和小张量为主，graph update 接口被 mock，无完整模型或批量图捕获。 |
| `tests/ut/attention/a2/test_sfa_cp_precision.py` | Sparse Flash Attention 的 PCP/DCP 精度 | 1/4 卡（8 GiB） | ✅ |  | 不加载 DeepSeek 权重；HCCL group 无法在单 rank 创建，测试已用 fake gather 模拟。 |
| `tests/ut/attention/a2/test_sfa_v1.py` | SFA backend、metadata 与 prefill/decode 路径 | 1/4 卡（8 GiB） | ✅ |  | 主要验证配置和小型合成张量路径，无模型权重和真实多卡通信。 |
| `tests/ut/attention/a2/test_sfa_v1_precision.py` | DeepSeek SFA kernel 精度，模拟 TP1/2/4 | 1/4 卡（8 GiB） | ✅ |  | 只构造合成输入与 reference，TP 不要求多个逻辑设备，显存规模远低于 E2E。 |
| `tests/ut/compilation/a2/test_acl_graph.py` | `ACLGraphEntry`/`ACLGraphWrapper` 生命周期与 dispatch | 1/4 卡（8 GiB） | ✅ |  | `torch.npu.NPUGraph`、graph context 和 platform graph pool 均被 mock，不占用真实 SQ/CQ 捕获资源。 |
| `tests/ut/device_allocator/a2/test_find_loaded_library.py` | 已加载共享库的发现与未命中行为 | 1/4 卡（8 GiB） | ✅ |  | 读取进程 library mapping，不分配 NPU 张量，也不依赖模型。 |
| `tests/ut/eplb/core/a2/test_eplb_utils.py` | EPLB 配置和逻辑/物理 expert 映射 | 1/4 卡（8 GiB） | ✅ |  | 配置与映射计算为 CPU 侧逻辑，不消耗可观 NPU 显存。 |
| `tests/ut/kv_offload/a2/test_remote_decode_lifecycle.py` | remote decode KV offload connector 生命周期 | 1/4 卡（8 GiB） | ✅ |  | 使用 fake connector/request 状态验证生命周期，不加载模型或分配大 KV cache。 |
| `tests/ut/kv_offload/a2/test_remote_prefill_lifecycle.py` | remote prefill KV offload connector 生命周期 | 1/4 卡（8 GiB） | ✅ |  | 使用 mock/fake 状态验证 prefill 流程，不建立真实远端传输或大规模 KV cache。 |
| `tests/ut/ops/a2/test_gdn_chunk_meta.py` | GDN chunk metadata 的生成和更新 | 1/4 卡（8 GiB） | ✅ |  | 仅使用很小的 metadata 张量，资源需求为 MiB 级。 |
| `tests/ut/ops/a2/test_token_dispatcher.py` | MoE token dispatch/combine 与路由逻辑 | 1/4 卡（8 GiB） | ✅ |  | 测试使用小型 expert/token 张量，distributed rank 被 mock，部分依赖 SoC kernel 的参数已 skip。 |
| `tests/ut/quantization/methods/a2/test_w4a16.py` | W4A16 linear/MoE 权重创建、转换与 apply | 1/4 卡（8 GiB） | ✅ |  | 量化权重为小型合成张量，不加载 checkpoint，远低于 8 GiB。 |
| `tests/ut/quantization/methods/a2/test_w4a4_flatquant.py` | W4A4 FlatQuant linear/MoE 路径 | 1/4 卡（8 GiB） | ✅ |  | 只构造小尺寸量化参数并 mock 部分算子，无模型常驻权重。 |
| `tests/ut/quantization/methods/a2/test_w4a4_laos_dynamic.py` | W4A4 LAOS dynamic linear 路径 | 1/4 卡（8 GiB） | ✅ |  | 输入、weight 和 scale 均为小型合成张量，峰值为 MiB 级。 |
| `tests/ut/quantization/methods/a2/test_w4a8.py` | W4A8 linear/MoE 权重打包、转换与 apply | 1/4 卡（8 GiB） | ✅ |  | 使用小型 int8/int32 合成权重，rank 与关键算子被 mock，不加载真实模型。 |
| `tests/ut/quantization/methods/a2/test_w8a16.py` | W8A16 权重处理和格式转换 | 1/4 卡（8 GiB） | ✅ |  | 参数尺寸很小且格式转换接口被 mock，无显著 NPU 内存压力。 |
| `tests/ut/quantization/methods/a2/test_w8a8_dynamic.py` | Dynamic W8A8 linear/MoE 量化与执行 | 1/4 卡（8 GiB） | ✅ |  | 只有小型合成 tensor 的少量真实 NPU 运算，其余量化/矩阵算子可 mock。 |
| `tests/ut/quantization/methods/a2/test_w8a8_static.py` | Static W8A8 权重处理、offset 与格式转换 | 1/4 卡（8 GiB） | ✅ |  | 不加载 checkpoint，权重与 scale 为小尺寸合成数据，资源需求很低。 |
| `tests/ut/sample/a2/test_gumbel_sampling.py` | Gumbel sampling、temperature、seed 和大词表边界 | 1/4 卡（8 GiB） | ✅ |  | 只创建 logits 与请求状态；即使 large-vocab 参数也远小于模型权重和 KV cache。 |
| `tests/ut/spec_decode/a2/test_eagle_proposer.py` | Eagle/MTP/DFlash proposer 的输入准备、draft 执行与 graph dispatch | 1/4 卡（8 GiB） | ✅ |  | 模型、runner、distributed group 和 graph context 大量使用 mock；最大显式 hidden-state 张量约百 MiB。 |
| `tests/ut/worker/a2/test_block_table.py` | Block table 的 append、slot mapping 和提交逻辑 | 1/4 卡（8 GiB） | ✅ |  | Triton/NPU kernel 被 mock 并在 CPU 计算，小型 block metadata 不形成显存压力。 |
| `tests/ut/worker/a2/test_kvcomp_utils.py` | KV compression 配置序列化与工具函数 | 1/4 卡（8 GiB） | ✅ |  | 纯配置和小张量逻辑，无模型或 KV cache 常驻分配。 |
| `tests/ut/worker/a2/test_model_runner_v1.py` | ModelRunner KV cache 布局、采样、占位 token 和调试器流程 | 1/4 卡（8 GiB） | ✅ |  | runner/model 与 cache allocation 被 mock，验证状态和布局计算而非加载模型。 |
| `tests/ut/worker/a2/test_model_runner_v1_with_device.py` | batch execution、padding 和 graph mode 选择 | 1/4 卡（8 GiB） | ✅ |  | 仅少量真实 device 张量，不执行模型推理或完整 ACL Graph 捕获。 |
| `tests/ut/worker/a2/test_worker_multi_instance.py` | 多实例可用显存与 KV cache 预算计算 | 1/4 卡（8 GiB） | ✅ |  | 64/32 GiB 等卡容量均为 mock 数值，测试不会按这些数值实际申请显存。 |
| `tests/ut/worker/a2/test_worker_v1.py` | Worker 初始化、profile、sleep/wake、graph warmup 与权重更新 | 1/4 卡（8 GiB） | ✅ |  | `mem_get_info`、model runner、allocator、HCCL/graph manager 等均被 mock，主要验证生命周期与错误处理。 |

## E2E 单卡测试文件

以下路径相对于 `tests/e2e/pull_request/`。

| 测试文件 | 涉及模型 | 目前所在组别 | 1/4卡实测结果 | 1/2卡实测结果 | 放在该 runner 的原因 |
| --- | --- | --- | --- | --- | --- |
| `quarter_card/compile/test_graphex_norm_quant_fusion.py` | 无预训练模型；257×64 BF16 合成张量 | 1/4 卡（8 GiB） | ✅ |  | 参数和输入只有 KiB 级，主要消耗来自编译器 workspace，理论峰值远低于 6 GiB；另 1 个参数组合 skip。 |
| `quarter_card/compile/test_graphex_qknorm_rope_fusion.py` | 无预训练模型；Q/K 16/8 heads、head_dim=128 | 1/4 卡（8 GiB） | ✅ |  | 最大 QKV 与 RoPE cache 仅为 MiB 级，主要验证图融合，不加载模型权重。 |
| `quarter_card/compile/test_norm_quant_fusion.py` | 无预训练模型；257×64 BF16 合成张量 | 1/4 卡（8 GiB） | ✅ |  | 小算子与小张量测试，显存由编译 workspace 主导，理论峰值远低于 8 GiB。 |
| `quarter_card/lora/test_ilama_lora.py` | `vllm-ascend/ilama-3.2-1B` | 1/4 卡（8 GiB） | ✅ |  | 1B BF16 权重约 2 GiB，LoRA、KV cache 和小规模图捕获后预计仍低于 6 GiB。 |
| `quarter_card/lora/test_qwen3_multi_loras.py` | `Qwen/Qwen3-0.6B`，Alice/Bob LoRA | 1/4 卡（8 GiB） |  |  | 0.6B BF16 权重约 1.2 GiB，多 LoRA 增量很小；当前因本地 adapter 缓存缺失而 skip，不填写实测列。 |
| `one_card/lora/test_qwen3_reranker_lora.py` | `Qwen/Qwen3-Reranker-0.6B` | 整卡（32 GiB，物理卡） | ❌ | ❌ | 1/4 和 1/2 vNPU 均在 ACL Graph capture 出现 `Alloc sq cq fail`/`resource alloc fail`；半卡复测仍失败，说明不是增加 vNPU 显存即可解决，退回物理卡。 |
| `quarter_card/pooling/test_embedding.py` | Qwen3-Embedding-0.6B、multilingual-e5-small、bge-m3 | 1/4 卡（8 GiB） | ✅ |  | 最大约 0.6B；即使 HF 对照使用 FP32，约 2.4 GiB 权重且与 vLLM 顺序运行，预计低于 6 GiB。 |
| `quarter_card/pooling/test_scoring.py` | ms-marco-MiniLM-L6-v2、bge-reranker-v2-m3、all-MiniLM-L12-v2 | 1/4 卡（8 GiB） | ✅ |  | 模型均小于 1B且使用 FP16，HF 与 vLLM 顺序运行，权重和工作区有充足 8 GiB 余量。 |
| `quarter_card/test_attention_fa3.py` | `Qwen/Qwen3-0.6B` | 1/4 卡（8 GiB） |  |  | 权重约 1.2 GiB，max_model_len=512、gpu_memory_utilization=0.7，理论预算约 5.6 GiB。 |
| `quarter_card/test_camem.py` | `Qwen/Qwen3-0.6B` | 1/4 卡（8 GiB） | ✅ |  | 权重约 1.2 GiB，只验证 sleep/wake，小捕获尺寸不会形成明显额外峰值。 |
| `quarter_card/test_completion_with_prompt_embeds.py` | `Qwen/Qwen3-0.6B` | 1/4 卡（8 GiB） | ✅ |  | NPU 上为约 1.2 GiB 的 vLLM 权重；Transformers embedding 模型默认留在 CPU，预计 NPU 峰值低于 6 GiB。 |
| `quarter_card/test_cpu_weight_offload.py` | `Qwen/Qwen3-0.6B` | 1/4 卡（8 GiB） | ✅ |  | 权重约 1.2 GiB，max_model_len=512；部分权重主动放到 CPU，NPU 峰值理论上低于普通 0.6B 推理。 |
| `quarter_card/test_guided_decoding.py` | `Qwen/Qwen3-0.6B` | 1/4 卡（8 GiB） | ✅ |  | 结构化输出主要增加 CPU 侧解析，NPU 仍是约 1.2 GiB 权重和小捕获尺寸。 |
| `one_card/test_minicpm.py` | MiniCPM-2B-sft-bf16、MiniCPM4-0.5B | 整卡（32 GiB，物理卡） | ❌ | ❌ | 1/2 卡上 0.5B 通过，但 2B 仍在 ACL Graph capture 报 `Alloc sq cq fail`；按不拆文件约束，整文件退回物理卡。 |
| `quarter_card/test_multi_instance.py` | 两个 `Qwen/Qwen3-0.6B` 实例 | 1/4 卡（8 GiB） | ✅ |  | 两份权重约 2.4 GiB；每个实例 gpu_memory_utilization=0.4，总预算 6.4 GiB，理论上可在 8 GiB 内共存。 |
| `quarter_card/test_qwen3_0_6b.py` | `Qwen/Qwen3-0.6B` | 1/4 卡（8 GiB） | ✅ |  | 权重约 1.2 GiB，max_model_len=1024，图捕获仅 1/2/4/8，预计峰值低于 6 GiB。 |
| `quarter_card/test_qwen3_5_0_8b.py` | `Qwen/Qwen3.5-0.8B` 多模态/MTP | 1/4 卡（8 GiB） | ✅ |  | 0.8B BF16 主干约 1.6 GiB，视觉编码器、MTP 与 2048 上下文叠加后预计仍低于 6 GiB。 |
| `quarter_card/test_qwen3_embedding_0_6b.py` | `Qwen/Qwen3-Embedding-0.6B` | 1/4 卡（8 GiB） | ✅ |  | 约 1.2 GiB BF16 权重，pooling 与 capture size 4 的额外显存较小。 |
| `quarter_card/test_sampler.py` | `Qwen/Qwen3-0.6B` | 1/4 卡（8 GiB） | ✅ |  | 权重约 1.2 GiB；虽然 max_model_len=8192，但实际输入很短，gpu_memory_utilization=0.9 将总预算限制在约 7.2 GiB。 |
| `quarter_card/test_simple_cpu_offload.py` | `Qwen/Qwen3-0.6B` | 1/4 卡（8 GiB） | ✅ |  | gpu_memory_utilization=0.5，仅预算约 4 GiB NPU 显存，KV 块另行卸载到 CPU。 |
| `quarter_card/test_xlite.py` | `Qwen/Qwen3-0.6B` | 1/4 卡（8 GiB） | ✅ |  | 权重约 1.2 GiB，max_model_len=1024；FULL mode 通过，decode-only 用例 skip。 |
| `half_card/lora/test_llama32_lora.py` | `meta-llama/Llama-3.2-3B-Instruct` | 1/2 卡（16 GiB） |  |  | 3B BF16 权重约 6 GiB，LoRA、KV cache 和图运行时使峰值预计达到约 8–10 GiB，不给 8 GiB 冒险。当前模型缓存缺失时 skip。 |
| `half_card/lora/test_lora_with_spec_decode.py` | Qwen3-1.7B、Qwen3-1.7B Eagle3、LoRA | 1/2 卡（16 GiB） |  | ✅ | 主模型约 3.4 GiB；按草稿 checkpoint 最坏同规模估算，双模型权重可达约 6.8 GiB，再加 KV/LoRA/workspace 超过 8 GiB 安全线。 |
| `half_card/lora/test_qwen35_densemodel_lora.py` | `Qwen/Qwen3.5-4B` | 1/2 卡（16 GiB） |  | ✅ | 4B BF16 权重约 8 GiB，LoRA、4096 上下文和运行时使峰值预计约 10–12 GiB。 |
| `half_card/pooling/test_classification.py` | `Howeee/Qwen2.5-1.5B-apeach` | 1/2 卡（16 GiB） |  | ✅ | HF 对照明确使用 FP32，权重约 6 GiB；激活与 NPU 运行时会越过 8 GiB 的安全余量，因此放 16 GiB。 |
| `half_card/test_qwen3_8b_w8a8.py` | Qwen3-8B-W8A8、Qwen3-8B Eagle3 speculator | 1/2 卡（16 GiB） |  | ✅ | W8A8 主权重约 8 GiB，叠加 Eagle3 草稿模型、4096 KV cache 和 FULL Graph，估算峰值约 10–14 GiB，先在 16 GiB 重测。 |
| `one_card/_310p/test_classification_310p.py` | `Howeee/Qwen2.5-1.5B-apeach` | 310P 单卡（专用） |  |  | 用例验证 310P 专用实现和硬件路径，不能用 A2B3 vNPU 的显存结论替代。 |
| `one_card/_310p/test_dense_model_310p.py` | Qwen3.5-4B FP16、Qwen3-8B-W8A8 | 310P 单卡（专用） |  |  | 包含 310P 专用 dense/Mamba cache/量化路径；硬件类型不同，不进入 A2B3 1/4、1/2 卡实测。 |
| `one_card/_310p/test_embedding_310p.py` | Qwen3-Embedding-0.6B、multilingual-e5-small、bge-m3 | 310P 单卡（专用） |  |  | 模型虽小，但测试目标是 310P pooling 实现，不按 A2B3 vNPU 路由。 |
| `one_card/_310p/test_scoring_310p.py` | MiniLM、bge-reranker-v2-m3 | 310P 单卡（专用） |  |  | 测试 310P scoring 路径，必须保留在 310P runner。 |
| `one_card/_310p/test_spec_decode_mtp_310p.py` | `Qwen/Qwen3.5-4B` | 310P 单卡（专用） |  |  | 测试 310P 的 MTP 实现；硬件能力和内存管理不同，不能迁到 A2B3 vNPU。 |
| `one_card/_310p/test_vl_model_310p.py` | `Qwen/Qwen3-VL-8B-Instruct` | 310P 单卡（专用） |  |  | 既是 8B FP16 多模态模型，又明确验证 310P 路径，保留专用 runner。 |
| `one_card/lora/test_olmoe_lora.py` | `allenai/OLMoE-1B-7B-0125-Instruct` | 整卡（32 GiB） |  |  | MoE 激活约 1B但总权重约 7B，BF16 常驻约 14 GiB；LoRA、KV 与图 workspace 后 16 GiB 无可靠余量。 |
| `one_card/model_runner_v2/test_basic.py` | Qwen3-0.6B、DeepSeek-V2-Lite-W8A8、Llama/Qwen3 8B 及 Eagle/DFlash/DSpark | 整卡（32 GiB） |  |  | 整文件含多个 8B BF16 主模型及草稿模型，单主模型权重已约 16 GiB，双模型推测解码明显超过 16 GiB。 |
| `one_card/model_runner_v2/test_uva.py` | `Qwen/Qwen3-0.6B` | 整卡（32 GiB，当前 skip） |  |  | 文件当前被无条件 skip，无法形成任何 vNPU 实测数据；本轮不为无效测试改变资源路由。 |
| `one_card/spec_decode/test_dflash.py` | Qwen3-8B、Qwen3-8B-DFlash | 整卡（32 GiB） |  |  | 8B BF16 主模型约 16 GiB，再叠加 DFlash 草稿权重、KV 和图捕获，必然越过 16 GiB。 |
| `one_card/spec_decode/test_draft_parallel.py` | Llama-3.1-8B、PARD-Llama-3.2-1B | 整卡（32 GiB） |  |  | 主模型约 16 GiB，1B 草稿约 2 GiB，尚未计入 KV/PIECEWISE Graph 即已超过 16 GiB。 |
| `one_card/spec_decode/test_dspark.py` | Qwen3-8B、DSpark Qwen3-8B block7 | 整卡（32 GiB） |  |  | 8B BF16 主模型约 16 GiB，加 DSpark 草稿、4096 上下文和 max_num_seqs=256 后只能放整卡。 |
| `one_card/spec_decode/test_eagle.py` | Qwen3-8B、Qwen3-VL-8B、Eagle3 speculator | 整卡（32 GiB） |  |  | 文件包含 8B BF16 文本/视觉主模型和草稿模型，主权重本身已占约 16 GiB。 |
| `one_card/spec_decode/test_extract_hidden_states.py` | Qwen3-8B、Qwen3.5-0.8B dummy | 整卡（32 GiB） |  |  | 不拆文件后由 Qwen3-8B BF16 用例决定资源；约 16 GiB 权重外还要保存多层 hidden states 和 KV。 |
| `one_card/spec_decode/test_mtp_eagle_correctness.py` | `wemaster/deepseek_mtp_main_random_bf16` | 整卡（32 GiB） |  |  | BF16 DeepSeek/MTP checkpoint 未提供可验证的小参数上界，且 max_num_seqs=256、图捕获 20；无法证明峰值低于 16 GiB。 |
| `one_card/spec_decode/test_ngram.py` | `Meta-Llama-3.1-8B-Instruct` | 整卡（32 GiB） |  |  | fixture 主模型为 8B BF16，权重约 16 GiB，KV 和运行时使 16 GiB 无法容纳。 |
| `one_card/spec_decode/test_ngram_npu.py` | `Meta-Llama-3.1-8B-Instruct` | 整卡（32 GiB） |  |  | 8B BF16 权重约 16 GiB，另有 max_num_seqs=256、2048 上下文和 PIECEWISE Graph。 |
| `one_card/spec_decode/test_suffix.py` | `Meta-Llama-3.1-8B-Instruct` | 整卡（32 GiB） |  |  | 8B BF16 权重约 16 GiB，Suffix cache 与 KV/图运行时需要额外显存。 |
| `one_card/test_batch_invariant.py` | `Qwen/Qwen3-0.6B` | 整卡（32 GiB） |  |  | 模型虽小，但文件配置最多 144 sequences、约 1–2K prompt tokens；按约 112 KiB/token 估算 KV 峰值可达约 16–32 GiB，不能按权重下沉。 |
| `one_card/test_cpu_offloading.py` | `Qwen/Qwen3-0.6B` | 整卡（32 GiB，当前 skip） |  |  | 用例因 connector deprecated 被无条件 skip，无法产生 1/4 或 1/2 卡实测数据，本轮保持原位置。 |
| `one_card/test_multistream_overlap_shared_expert.py` | `DeepSeek-V2-Lite-W8A8` | 整卡（32 GiB） |  |  | 约 16B 总参数即使 W8A8 也接近 16 GiB 权重，再叠加 MoE workspace 与图捕获，需整卡。 |
| `one_card/test_npu_ipc_weight_transfer.py` | `Qwen/Qwen3-0.6B` 或本地 override | 整卡（32 GiB，物理卡） |  |  | 显存本身足够，但用例要求跨进程导出/导入 NPU IPC 内存句柄；这是物理设备能力约束，不应按小模型下沉到 vNPU。 |
| `one_card/test_vlm.py` | Qwen3-VL-8B、HunyuanOCR、Qwen2-Audio-7B、Whisper Large v3 Turbo | 整卡（32 GiB） |  |  | 不拆文件后由 7/8B BF16 多模态模型决定；仅语言主干已约 14–16 GiB，视觉/音频编码器、KV 和图使峰值超过 16 GiB。 |
