# vNPU 静态预测分组与测试记录

本文记录 A2B3 vNPU 的新一轮测试分组。分组只依据当前源码中的模型规模、
精度、量化方式、KV cache、ACL Graph、显式张量、并发实例和硬件语义；不读取、
引用或反推任何过往真实测试结果。

## 约束与判断口径

- A2B3 整卡按 64 GiB 计算，1/4 卡和 1/2 卡的名义显存分别为 16 GiB、
  32 GiB。
- 测试文件是最小迁移单位，不使用 pytest nodeid 拆分。同一文件由静态峰值最高
  的用例决定 runner。
- BF16/FP16 权重按约 2 bytes/parameter、W8A8 按约 1 byte/parameter
  估算；另外预留 KV cache、激活、图捕获、算子 workspace、草稿模型和运行时
  常驻显存。
- 1/4 卡主要承载不加载完整模型的合成测试和 0.5B–3B 模型；1/2 卡承载 7B/8B
  BF16、约 16B W8A8 MoE、高 batch/KV 或目标模型加草稿模型的文件。
- 只有当静态源码无法证明 vNPU 硬件语义等价时才保留物理整卡，不用旧运行结果
  作为保留理由。
- 表格的“本轮实测结果”初始统一为“待填”，用于本轮 CI 完成后记录，不预填旧数据。

## 1/4 卡 E2E（27 个文件）

| 测试文件 | 主要内容 | 静态预测依据 | 本轮实测结果 |
| --- | --- | --- | --- |
| `quarter_card/compile/test_graphex_norm_quant_fusion.py` | Norm/Quant GraphEx 融合 | 无预训练权重，显式 BF16 张量为算子级 | 待填 |
| `quarter_card/compile/test_graphex_qknorm_rope_fusion.py` | QKNorm/RoPE GraphEx 融合 | 无预训练权重，Q/K/RoPE 张量远低于 GiB 级 | 待填 |
| `quarter_card/compile/test_norm_quant_fusion.py` | Norm/Quant 编译融合 | 小型合成算子与张量 | 待填 |
| `quarter_card/lora/test_ilama_lora.py` | iLlama 1B LoRA | FP16 主权重约 2 GiB，eager、1024 上下文 | 待填 |
| `quarter_card/lora/test_llama32_lora.py` | Llama-3.2-3B LoRA | BF16 权重约 6 GiB，max_num_seqs=7、1024 上下文 | 待填 |
| `quarter_card/lora/test_lora_with_spec_decode.py` | Qwen3-1.7B LoRA + Eagle3 | 主模型约 3.4 GiB，草稿头、LoRA、KV 和图预计仍低于 16 GiB | 待填 |
| `quarter_card/lora/test_qwen3_multi_loras.py` | Qwen3-0.6B 多 LoRA | 主模型小，LoRA 增量小且 eager | 待填 |
| `quarter_card/lora/test_qwen3_reranker_lora.py` | Qwen3-Reranker-0.6B pooling + LoRA | 0.6B 单实例，无长上下文或大 batch | 待填 |
| `quarter_card/pooling/test_classification.py` | Qwen2.5-1.5B 分类 | HF FP32 与 vLLM 顺序运行，单阶段权重约 6 GiB | 待填 |
| `quarter_card/pooling/test_embedding.py` | Qwen3-0.6B/E5/BGE embedding | 小模型逐个顺序运行，对照模型不并发常驻 | 待填 |
| `quarter_card/pooling/test_scoring.py` | MiniLM/BGE scoring | 小型 FP16 pooling/cross-encoder | 待填 |
| `quarter_card/test_attention_fa3.py` | Qwen3-0.6B FA3/FIA 对比 | 0.6B、短输入、小 capture size | 待填 |
| `quarter_card/test_batch_job_aware_scheduler_e2e.py` | BatchJobAwareScheduler | 两个 Qwen3-0.6B engine 顺序创建，2048 上下文、0.7 显存比例 | 待填 |
| `quarter_card/test_camem.py` | sleep/wake 显存管理 | Qwen3-0.6B 单实例 | 待填 |
| `quarter_card/test_completion_with_prompt_embeds.py` | prompt embeddings | Qwen3-0.6B，embedding 对照与推理阶段不形成双模型 NPU 峰值 | 待填 |
| `quarter_card/test_cpu_offloading.py` | CPU KV offload connector | 文件当前整体 skip；若启用，0.6B 且 NPU 显存比例 0.5 | 待填 |
| `quarter_card/test_cpu_weight_offload.py` | 权重预取/卸载 | Qwen3-0.6B、512 上下文，部分权重驻 CPU | 待填 |
| `quarter_card/test_guided_decoding.py` | structured output | Qwen3-0.6B，额外开销主要在 CPU 解析侧 | 待填 |
| `quarter_card/test_minicpm.py` | MiniCPM 0.5B/2B | 最大 2B BF16 约 4 GiB，512 上下文、0.7 显存比例 | 待填 |
| `quarter_card/test_minimax_m3_sparse_attn.py` | MiniMax M3 稀疏 attention | 不加载完整模型，只构造生产形状的 KV/index 合成张量 | 待填 |
| `quarter_card/test_multi_instance.py` | 两个 Qwen3-0.6B 实例 | 两实例合计约 2.4 GiB 权重，单实例显存比例 0.4 | 待填 |
| `quarter_card/test_qwen3_0_6b.py` | Qwen3-0.6B 基础图模式 | 小模型、1024 上下文 | 待填 |
| `quarter_card/test_qwen3_5_0_8b.py` | Qwen3.5-0.8B + MTP | 0.8B 主模型、单 token 草稿、2048 上下文 | 待填 |
| `quarter_card/test_qwen3_embedding_0_6b.py` | Qwen3-Embedding-0.6B | 0.6B pooling，capture size=4 | 待填 |
| `quarter_card/test_sampler.py` | sampler/logprobs | Qwen3-0.6B；虽配置 8192 上下文和 capture 64，权重与 KV 仍预计小于 16 GiB | 待填 |
| `quarter_card/test_simple_cpu_offload.py` | simple CPU offload | Qwen3-0.6B、eager、显存比例 0.5 | 待填 |
| `quarter_card/test_xlite.py` | XLite eager/graph | Qwen3-0.6B、1024 上下文 | 待填 |

## 1/2 卡 E2E（15 个文件）

| 测试文件 | 主要内容 | 静态预测依据 | 本轮实测结果 |
| --- | --- | --- | --- |
| `half_card/lora/test_olmoe_lora.py` | OLMoE-1B-7B LoRA | 总权重约 7B，BF16 权重接近 14 GiB，加 LoRA 与图超过 1/4 卡安全余量 | 待填 |
| `half_card/model_runner_v2/test_basic.py` | V2 runner dense/Eagle/DFlash/DSpark/MTP | 文件原子化后由 8B BF16 主模型加草稿与图模式决定 | 待填 |
| `half_card/spec_decode/test_dflash.py` | Qwen3-8B DFlash | 约 16 GiB 主权重，加 DFlash、4096 KV、batch 256 和图捕获 | 待填 |
| `half_card/spec_decode/test_draft_parallel.py` | Llama-3.1-8B + PARD-1B | 主模型约 16 GiB、草稿约 2 GiB，加 KV/PIECEWISE Graph | 待填 |
| `half_card/spec_decode/test_dspark.py` | Qwen3-8B DSpark | 8B BF16 主模型、草稿、4096 KV、batch 256 | 待填 |
| `half_card/spec_decode/test_eagle.py` | Qwen3/Qwen3-VL-8B Eagle3 | 文件含 8B 文本和视觉主模型，单次还加载 Eagle3 草稿 | 待填 |
| `half_card/spec_decode/test_extract_hidden_states.py` | hidden-state 提取 | 文件含 Qwen3-8B 实权重，另保存多层 hidden states | 待填 |
| `half_card/spec_decode/test_mtp_eagle_correctness.py` | DeepSeek MTP smoke | BF16 MoE checkpoint、batch 256、图 capture 20，静态上不适合 16 GiB | 待填 |
| `half_card/spec_decode/test_ngram.py` | Llama-3.1-8B n-gram | 8B BF16 主权重约占满 1/4 卡名义容量 | 待填 |
| `half_card/spec_decode/test_ngram_npu.py` | Llama-3.1-8B NPU n-gram | 8B BF16、batch 256、2048 上下文、PIECEWISE Graph | 待填 |
| `half_card/spec_decode/test_suffix.py` | Llama-3.1-8B suffix decode | 8B BF16 加 suffix cache、KV 和运行时 | 待填 |
| `half_card/test_batch_invariant.py` | batch invariant/logprobs | 0.6B 权重虽小，但 batch 最高 144、8192 上下文、显存比例 0.95 | 待填 |
| `half_card/test_multistream_overlap_shared_expert.py` | DeepSeek-V2-Lite-W8A8 多流共享 expert | 约 16B 总参数 W8A8，加 MoE workspace 和 capture 32 | 待填 |
| `half_card/test_qwen3_8b_w8a8.py` | Qwen3-8B W8A8 + Eagle3 | 约 8 GiB 主权重，加草稿、4096 KV 与 FULL Graph，1/4 卡余量不足 | 待填 |
| `half_card/test_vlm.py` | 7B/8B 视觉、音频、Whisper | 文件由 7B/8B BF16 多模态模型决定，需为编码器、KV 和图保留空间 | 待填 |

## A2 UT（34 个文件）

A2 UT 仍以文件为单位。33 个文件预测放 1/4 卡；只有
`test_mla_precision.py` 因生产级 MLA 投影矩阵及初始化临时副本放入 1/2 卡。

| 测试文件 | 预测组 | 静态预测依据 | 本轮实测结果 |
| --- | --- | --- | --- |
| `tests/ut/attention/a2/test_attention_cp.py` | 1/4 卡 | 通信与 backend 以 mock 为主，小型张量 | 待填 |
| `tests/ut/attention/a2/test_attention_cp_precision.py` | 1/4 卡 | 合成 Q/K/V，不加载权重 | 待填 |
| `tests/ut/attention/a2/test_attention_v1.py` | 1/4 卡 | metadata/backend 单元测试，小张量 | 待填 |
| `tests/ut/attention/a2/test_attention_v1_precision.py` | 1/4 卡 | 单层 Qwen3-8B 形状；显式 KV cache 约 4 GiB，无模型权重 | 待填 |
| `tests/ut/attention/a2/test_common_cp.py` | 1/4 卡 | reshape/LSE 与 mock collective | 待填 |
| `tests/ut/attention/a2/test_mla_cp.py` | 1/4 卡 | mock 通信与控制路径，无完整权重 | 待填 |
| `tests/ut/attention/a2/test_mla_cp_precision.py` | 1/4 卡 | 合成 MLA 输入，无预训练权重 | 待填 |
| `tests/ut/attention/a2/test_mla_precision.py` | 1/2 卡 | 多块生产级 MLA 投影矩阵及 `randn / sqrt` 临时副本，峰值逼近 16 GiB 边界 | 待填 |
| `tests/ut/attention/a2/test_mla_v1.py` | 1/4 卡 | backend/metadata mock 和小型 cache | 待填 |
| `tests/ut/attention/a2/test_sfa_cp_precision.py` | 1/4 卡 | 合成 sparse attention 张量，通信模拟 | 待填 |
| `tests/ut/attention/a2/test_sfa_v1.py` | 1/4 卡 | 配置与小型 backend 张量 | 待填 |
| `tests/ut/attention/a2/test_sfa_v1_precision.py` | 1/4 卡 | 合成输入，无完整 DeepSeek 权重 | 待填 |
| `tests/ut/compilation/a2/test_acl_graph.py` | 1/4 卡 | NPUGraph/context/pool 均 mock | 待填 |
| `tests/ut/device_allocator/a2/test_find_loaded_library.py` | 1/4 卡 | 进程库映射检查，无 NPU 大分配 | 待填 |
| `tests/ut/eplb/core/a2/test_eplb_utils.py` | 1/4 卡 | CPU 侧 expert 映射逻辑 | 待填 |
| `tests/ut/kv_offload/a2/test_remote_decode_lifecycle.py` | 1/4 卡 | fake connector 生命周期 | 待填 |
| `tests/ut/kv_offload/a2/test_remote_prefill_lifecycle.py` | 1/4 卡 | fake connector 生命周期 | 待填 |
| `tests/ut/ops/a2/test_gdn_chunk_meta.py` | 1/4 卡 | 小型 metadata 张量 | 待填 |
| `tests/ut/ops/a2/test_gdn_layerwise_kv.py` | 1/4 卡 | 小型 layerwise KV 合成张量 | 待填 |
| `tests/ut/ops/a2/test_token_dispatcher.py` | 1/4 卡 | 小型 token/expert 张量，rank 模拟 | 待填 |
| `tests/ut/quantization/methods/a2/test_w4a16.py` | 1/4 卡 | 算子级量化矩阵 | 待填 |
| `tests/ut/quantization/methods/a2/test_w4a4_flatquant.py` | 1/4 卡 | 算子级量化矩阵 | 待填 |
| `tests/ut/quantization/methods/a2/test_w4a4_laos_dynamic.py` | 1/4 卡 | 算子级量化矩阵 | 待填 |
| `tests/ut/quantization/methods/a2/test_w4a8.py` | 1/4 卡 | 算子级量化矩阵 | 待填 |
| `tests/ut/quantization/methods/a2/test_w8a16.py` | 1/4 卡 | 算子级量化矩阵 | 待填 |
| `tests/ut/quantization/methods/a2/test_w8a8_dynamic.py` | 1/4 卡 | 算子级量化矩阵 | 待填 |
| `tests/ut/quantization/methods/a2/test_w8a8_static.py` | 1/4 卡 | 算子级量化矩阵 | 待填 |
| `tests/ut/sample/a2/test_gumbel_sampling.py` | 1/4 卡 | sampler 合成 logits | 待填 |
| `tests/ut/spec_decode/a2/test_eagle_proposer.py` | 1/4 卡 | runner/model/graph context 大量 mock | 待填 |
| `tests/ut/worker/a2/test_block_table.py` | 1/4 卡 | block table 小型 metadata | 待填 |
| `tests/ut/worker/a2/test_model_runner_v1.py` | 1/4 卡 | runner 以 `__new__`/mock 构造，小型 cache | 待填 |
| `tests/ut/worker/a2/test_model_runner_v1_with_device.py` | 1/4 卡 | 设备路径单元测试，无完整模型 | 待填 |
| `tests/ut/worker/a2/test_worker_multi_instance.py` | 1/4 卡 | worker 状态与 mock 实例 | 待填 |
| `tests/ut/worker/a2/test_worker_v1.py` | 1/4 卡 | worker 生命周期与 mock | 待填 |

## 保留物理整卡的文件

| 测试文件 | 保留原因 |
| --- | --- |
| `one_card/model_runner_v2/test_uva.py` | 验证 `pinned_mem_register`/UVA 语义；这不是显存容量问题，静态源码无法证明 vNPU 等价。 |
| `one_card/test_npu_ipc_weight_transfer.py` | 同时验证跨进程 NPU IPC 权重共享和服务生命周期，需要物理设备 IPC 语义。 |

310P 专用、A3、多卡 HCCL 测试保持原 runner，不参与本次 A2B3 vNPU 容量采样。

## 本轮临时执行方式

- PR 的选择命令临时传入 `--npu-types a2_quarter a2_half`。选择器仍先按正常
  模块依赖收集测试，再只输出 1/4 和 1/2 两类 runner；CPU、物理整卡、310P、
  A3 和多卡组本轮不执行。
- 该过滤只接入 `pr_test.yaml`，不会改变 `/e2e` 命令、定时覆盖率任务或选择器
  的默认行为。完成本轮容量采样后应删除该参数。
- `run_selected_tests.sh` 对每个文件单独调用 pytest。某个文件失败时记录状态和
  日志并继续执行同一分桶内剩余文件；全部文件结束后统一打印汇总，并以首个非零
  状态退出，确保既收集完整结果又不掩盖失败。
