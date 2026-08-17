# vNPU 静态预测分组、实测修正与测试记录

本文记录 A2B3 vNPU 的新一轮测试分组。初始分组只依据当时源码中的模型规模、
精度、量化方式、KV cache、ACL Graph、显式张量、并发实例和硬件语义；没有读取、
引用或反推任何过往真实测试结果。随后使用本轮
[Actions run 30783037208](https://github.com/vllm-project/vllm-ascend/actions/runs/30783037208?pr=12171)
的 main 矩阵结果修正分组。

[Actions run 30800319394](https://github.com/vllm-project/vllm-ascend/actions/runs/30800319394?pr=12171)
进一步确认 reranker LoRA 与 BatchJobAwareScheduler 在 1/2 卡仍会因 ACL Graph
capture 的 SQ/CQ 资源申请失败，因此整文件退回物理整卡。随后 rebase 到最新
`upstream/main`，同步上游新增、删除的测试文件；新增文件按源码静态预测分组，
实测结果保持为空。

[Actions run 31373619460](https://github.com/vllm-project/vllm-ascend/actions/runs/31373619460?pr=12171)
使用旧的 9.0.1 镜像时暴露了三类问题：多个 job 下载同一个 6 MiB csrc cache
超时；RLHF pause/sleep 的 HTTP 操作超过原 15 秒限制；MiniCPM-2B 和旧软件栈
组合在 ACL Graph capture 时触发 `Alloc sq cq fail`。本次 rebase 后 vNPU 镜像
同步到上游 A2 的 9.1.0，cache key 与 CPU producer 对齐，并为 vNPU cache restore
增加一次重试。上游同时已将 pause 超时提高到 60 秒，本分支将 sleep 超时同步为
60 秒。MiniCPM 暂留 1/2 卡，在新软件栈复测；若仍出现相同 SQ/CQ 错误，再按
整文件规则退回物理整卡。

[Actions run 32017074775](https://github.com/vllm-project/vllm-ascend/actions/runs/32017074775?pr=12171)
确认 csrc cache 和 9.1.0 安装链路已稳定。Qwen3.5-0.8B 在 1/4 卡已占用
14.31 GiB 后仍需申请 486 MiB，整文件升到 1/2 卡；MiniCPM-2B 在 9.1.0
的 1/2 卡仍因 stream 资源耗尽触发 ACL Graph `207005`，整文件退回物理整卡。
Dynamic DSpark 没有资源不足，但连续两轮在 1/2 卡得到相同的
acceptance baseline 偏差，而同一用例在上游物理 A2 的 vLLM main 和 v0.26.0
矩阵均通过；这是 vNPU 执行一致性问题，整文件退回物理整卡。另一个
1/2 卡 job 在 `apt-get update` 阶段挂起至
120 分钟超时，未开始测试；安装步骤现设置 30 分钟总超时。

## 待基础设施反馈的安装超时

- Run：[32017074775](https://github.com/vllm-project/vllm-ascend/actions/runs/32017074775?pr=12171)
- Job：[95350574280](https://github.com/vllm-project/vllm-ascend/actions/runs/32017074775/job/95350574280?pr=12171)
- Runner：`linux-aarch64-a2b3-v-half-45g6g-runner-jwgpw`（`linux-aarch64-a2b3-v-half`）
- 阶段：`Install packages` 中的 `apt-get update -y`，测试尚未开始。
- 内部源：`cache-service.nginx-pypi-cache.svc.cluster.local:8081/ubuntu-ports`。
- 时间线：2026-08-17 09:59:17 UTC 最后一次输出为拉取 `jammy-updates`
  包索引，随后约 118 分钟无新输出；11:57:41 UTC 整个 job 的
  120 分钟超时触发，并报 `Executing the custom container implementation failed`。
- 待排查：该 runner 到内部 Ubuntu mirror 的连接、mirror 后端响应及 runner
  容器网络状态；工作流侧仅设置 `Install packages` 的 30 分钟总超时，
  不额外限制单次 HTTP 连接。

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
- 表格的“本轮实测结果”只记录本轮 run；“通过（main）”表示该文件在 vLLM main
  矩阵通过。失败项同时核对 v0.26.0 矩阵，确认可复现后再迁移。

## 1/4 卡 E2E（24 个文件）

| 测试文件 | 主要内容 | 静态预测依据 | 本轮实测结果 |
| --- | --- | --- | --- |
| `quarter_card/compile/test_graphex_norm_quant_fusion.py` | Norm/Quant GraphEx 融合 | 无预训练权重，显式 BF16 张量为算子级 | 通过（main） |
| `quarter_card/compile/test_graphex_qknorm_rope_fusion.py` | QKNorm/RoPE GraphEx 融合 | 无预训练权重，Q/K/RoPE 张量远低于 GiB 级 | 通过（main） |
| `quarter_card/compile/test_norm_quant_fusion.py` | Norm/Quant 编译融合 | 小型合成算子与张量 | 通过（main） |
| `quarter_card/lora/test_ilama_lora.py` | iLlama 1B LoRA | FP16 主权重约 2 GiB，eager、1024 上下文 | 通过（main） |
| `quarter_card/lora/test_llama32_lora.py` | Llama-3.2-3B LoRA | BF16 权重约 6 GiB，max_num_seqs=7、1024 上下文 | 通过（main） |
| `quarter_card/lora/test_lora_with_spec_decode.py` | Qwen3-1.7B LoRA + Eagle3 | 主模型约 3.4 GiB，草稿头、LoRA、KV 和图预计仍低于 16 GiB | 通过（main） |
| `quarter_card/lora/test_qwen3_multi_loras.py` | Qwen3-0.6B 多 LoRA | 主模型小，LoRA 增量小且 eager | 通过（main） |
| `quarter_card/pooling/test_classification.py` | Qwen2.5-1.5B 分类 | HF FP32 与 vLLM 顺序运行，单阶段权重约 6 GiB | 通过（main） |
| `quarter_card/pooling/test_embedding.py` | Qwen3-0.6B/E5/BGE embedding | 小模型逐个顺序运行，对照模型不并发常驻 | 通过（main） |
| `quarter_card/pooling/test_scoring.py` | MiniLM/BGE scoring | 小型 FP16 pooling/cross-encoder | 通过（main） |
| `quarter_card/test_attention_fa3.py` | Qwen3-0.6B FA3/FIA 对比 | 0.6B、短输入、小 capture size | 通过（main） |
| `quarter_card/rlhf/state_transitions/test_pause_resume.py` | RLHF pause/resume 状态机 | Qwen3-0.6B BF16、eager、2048 上下文、显存比例 0.75；上游已将 pause 超时由 15 秒调为 60 秒 |  |
| `quarter_card/rlhf/state_transitions/test_sleep_wake.py` | RLHF sleep/wake 显存与输出恢复 | Qwen3-0.6B BF16、eager；v0.26.0 在 15 秒超时，现同步为 60 秒后待复测 |  |
| `quarter_card/test_completion_with_prompt_embeds.py` | prompt embeddings | Qwen3-0.6B，embedding 对照与推理阶段不形成双模型 NPU 峰值 | 通过（main） |
| `quarter_card/test_cpu_offloading.py` | CPU KV offload connector | 文件当前整体 skip；若启用，0.6B 且 NPU 显存比例 0.5 | 通过（main） |
| `quarter_card/test_cpu_weight_offload.py` | 权重预取/卸载 | Qwen3-0.6B、512 上下文，部分权重驻 CPU | 通过（main） |
| `quarter_card/test_guided_decoding.py` | structured output | Qwen3-0.6B，额外开销主要在 CPU 解析侧 | 通过（main） |
| `quarter_card/test_minimax_m3_sparse_attn.py` | MiniMax M3 稀疏 attention | 不加载完整模型，只构造生产形状的 KV/index 合成张量 | 通过（main） |
| `quarter_card/test_multi_instance.py` | 两个 Qwen3-0.6B 实例 | 两实例合计约 2.4 GiB 权重，单实例显存比例 0.4 | 通过（main） |
| `quarter_card/test_qwen3_0_6b.py` | Qwen3-0.6B 基础图模式 | 小模型、1024 上下文 | 通过（main） |
| `quarter_card/test_qwen3_embedding_0_6b.py` | Qwen3-Embedding-0.6B | 0.6B pooling，capture size=4 | 通过（main） |
| `quarter_card/test_sampler.py` | sampler/logprobs | Qwen3-0.6B；虽配置 8192 上下文和 capture 64，权重与 KV 仍预计小于 16 GiB | 通过（main） |
| `quarter_card/test_simple_cpu_offload.py` | simple CPU offload | Qwen3-0.6B、eager、显存比例 0.5 | 通过（main） |
| `quarter_card/test_xlite.py` | XLite eager/graph | Qwen3-0.6B、1024 上下文 | 通过（main） |

## 1/2 卡 E2E（11 个文件）

| 测试文件 | 主要内容 | 静态预测依据 | 本轮实测结果 |
| --- | --- | --- | --- |
| `half_card/spec_decode/test_dflash.py` | Qwen3-8B DFlash | 约 16 GiB 主权重，加 DFlash、4096 KV、batch 256 和图捕获 | 通过（main） |
| `half_card/spec_decode/test_draft_parallel.py` | Llama-3.1-8B + PARD-1B | 主模型约 16 GiB、草稿约 2 GiB，加 KV/PIECEWISE Graph | 通过（main） |
| `half_card/spec_decode/test_dspark.py` | Qwen3-8B DSpark | 8B BF16 主模型、草稿、4096 KV、batch 256 | 通过（main） |
| `half_card/spec_decode/test_eagle.py` | Qwen3/Qwen3-VL-8B Eagle3 | 文件含 8B 文本和视觉主模型，单次还加载 Eagle3 草稿 | 通过（main） |
| `half_card/spec_decode/test_mtp_eagle_correctness.py` | DeepSeek MTP smoke | BF16 MoE checkpoint、batch 256、图 capture 20，静态上不适合 16 GiB | 通过（main） |
| `half_card/spec_decode/test_ngram.py` | Llama-3.1-8B n-gram | 8B BF16 主权重约占满 1/4 卡名义容量 | 通过（main） |
| `half_card/spec_decode/test_ngram_npu.py` | Llama-3.1-8B NPU n-gram | 8B BF16、batch 256、2048 上下文、PIECEWISE Graph | 通过（main） |
| `half_card/spec_decode/test_suffix.py` | Llama-3.1-8B suffix decode | 8B BF16 加 suffix cache、KV 和运行时 | 通过（main） |
| `half_card/test_qwen3_5_0_8b.py` | Qwen3.5-0.8B + MTP | 1/4 卡上已占用 14.31 GiB 后仍需申请 486 MiB，峰值明确超过 15 GiB 可用容量 |  |
| `half_card/test_qwen3_8b_w8a8.py` | Qwen3-8B W8A8 + Eagle3 | 约 8 GiB 主权重，加草稿、4096 KV 与 FULL Graph，1/4 卡余量不足 | 通过（main） |
| `half_card/test_vlm.py` | 7B/8B 视觉、音频、Whisper | 文件由 7B/8B BF16 多模态模型决定，需为编码器、KV 和图保留空间 | 通过（main） |

## A2 UT（34 个文件）

A2 UT 仍以文件为单位。32 个文件放 1/4 卡；`test_mla_precision.py` 因生产级
MLA 投影矩阵及初始化临时副本放入 1/2 卡，`test_attention_v1_precision.py`
则根据本轮明确的 15 GiB OOM 迁入 1/2 卡。

| 测试文件 | 预测组 | 静态预测依据 | 本轮实测结果 |
| --- | --- | --- | --- |
| `tests/ut/attention/a2/test_attention_cp.py` | 1/4 卡 | 通信与 backend 以 mock 为主，小型张量 | 通过（main） |
| `tests/ut/attention/a2/test_attention_cp_precision.py` | 1/4 卡 | 合成 Q/K/V，不加载权重 | 通过（main） |
| `tests/ut/attention/a2/test_attention_v1.py` | 1/4 卡 | metadata/backend 单元测试，小张量 | 通过（main） |
| `tests/ut/attention/a2/test_attention_v1_precision.py` | 1/2 卡 | 单层生产形状叠加多块显式 KV cache，实际峰值超过 16 GiB | 1/4 卡 OOM：已分配 12.04 GiB 后仍需申请 4.00 GiB（main、v0.26.0） |
| `tests/ut/attention/a2/test_common_cp.py` | 1/4 卡 | reshape/LSE 与 mock collective | 通过（main） |
| `tests/ut/attention/a2/test_mla_cp.py` | 1/4 卡 | mock 通信与控制路径，无完整权重 | 通过（main） |
| `tests/ut/attention/a2/test_mla_cp_precision.py` | 1/4 卡 | 合成 MLA 输入，无预训练权重 | 通过（main） |
| `tests/ut/attention/a2/test_mla_precision.py` | 1/2 卡 | 多块生产级 MLA 投影矩阵及 `randn / sqrt` 临时副本，峰值逼近 16 GiB 边界 | 通过（main） |
| `tests/ut/attention/a2/test_mla_v1.py` | 1/4 卡 | backend/metadata mock 和小型 cache | 通过（main） |
| `tests/ut/attention/a2/test_sfa_cp_precision.py` | 1/4 卡 | 合成 sparse attention 张量，通信模拟 | 通过（main） |
| `tests/ut/attention/a2/test_sfa_v1.py` | 1/4 卡 | 配置与小型 backend 张量 | 通过（main） |
| `tests/ut/attention/a2/test_sfa_v1_precision.py` | 1/4 卡 | 合成输入，无完整 DeepSeek 权重 | 通过（main） |
| `tests/ut/compilation/a2/test_acl_graph.py` | 1/4 卡 | NPUGraph/context/pool 均 mock | 通过（main） |
| `tests/ut/device_allocator/a2/test_find_loaded_library.py` | 1/4 卡 | 进程库映射检查，无 NPU 大分配 | 通过（main） |
| `tests/ut/eplb/core/a2/test_eplb_utils.py` | 1/4 卡 | CPU 侧 expert 映射逻辑 | 通过（main） |
| `tests/ut/kv_offload/a2/test_remote_decode_lifecycle.py` | 1/4 卡 | fake connector 生命周期 | 通过（main） |
| `tests/ut/kv_offload/a2/test_remote_prefill_lifecycle.py` | 1/4 卡 | fake connector 生命周期 | 通过（main） |
| `tests/ut/ops/a2/test_gdn_chunk_meta.py` | 1/4 卡 | 小型 metadata 张量 | 通过（main） |
| `tests/ut/ops/a2/test_gdn_layerwise_kv.py` | 1/4 卡 | 小型 layerwise KV 合成张量 | 通过（main） |
| `tests/ut/ops/a2/test_token_dispatcher.py` | 1/4 卡 | 小型 token/expert 张量，rank 模拟 | 通过（main） |
| `tests/ut/quantization/methods/a2/test_w4a16.py` | 1/4 卡 | 算子级量化矩阵 | 通过（main） |
| `tests/ut/quantization/methods/a2/test_w4a4_flatquant.py` | 1/4 卡 | 算子级量化矩阵 | 通过（main） |
| `tests/ut/quantization/methods/a2/test_w4a4_laos_dynamic.py` | 1/4 卡 | 算子级量化矩阵 | 通过（main） |
| `tests/ut/quantization/methods/a2/test_w8a16.py` | 1/4 卡 | 算子级量化矩阵 | 通过（main） |
| `tests/ut/quantization/methods/a2/test_w8a8_dynamic.py` | 1/4 卡 | 算子级量化矩阵 | 通过（main） |
| `tests/ut/quantization/methods/a2/test_w8a8_static.py` | 1/4 卡 | 算子级量化矩阵 | 通过（main） |
| `tests/ut/sample/a2/test_gumbel_sampling.py` | 1/4 卡 | sampler 合成 logits | 通过（main） |
| `tests/ut/spec_decode/a2/test_eagle_proposer.py` | 1/4 卡 | runner/model/graph context 大量 mock | 通过（main） |
| `tests/ut/worker/a2/test_block_table.py` | 1/4 卡 | block table 小型 metadata | 通过（main） |
| `tests/ut/worker/a2/test_model_runner_v1.py` | 1/4 卡 | runner 以 `__new__`/mock 构造，小型 cache | 通过（main） |
| `tests/ut/worker/a2/test_model_runner_v1_with_device.py` | 1/4 卡 | 设备路径单元测试，无完整模型 | 通过（main） |
| `tests/ut/worker/a2/test_worker_multi_instance.py` | 1/4 卡 | worker 状态与 mock 实例 | 通过（main） |
| `tests/ut/worker/a2/test_worker_v1.py` | 1/4 卡 | worker 生命周期与 mock | 通过（main） |
| `tests/ut/worker/a2/test_worker_v2.py` | 1/4 卡 | worker v2 生命周期与 mock，无完整模型权重 | 通过（main） |

## 保留物理整卡的文件

| 测试文件 | 保留原因 |
| --- | --- |
| `one_card/model_runner_v2/test_uva.py` | 验证 `pinned_mem_register`/UVA 语义；这不是显存容量问题，静态源码无法证明 vNPU 等价。 |
| `one_card/test_npu_ipc_weight_transfer.py` | 同时验证跨进程 NPU IPC 权重共享和服务生命周期，需要物理设备 IPC 语义。 |
| `one_card/lora/test_olmoe_lora.py` | 半卡在 ACL Graph capture 阶段资源申请失败（main、v0.26.0）；单设备模型不应迁往两卡，恢复到 64 GiB 物理整卡。 |
| `one_card/model_runner_v2/test_basic.py` | 原子文件内不同 case 同时出现 ACL Graph 资源申请失败和 HCCL error code 19（main、v0.26.0），恢复到物理整卡。 |
| `one_card/spec_decode/test_extract_hidden_states.py` | 8B 默认 40960 上下文需要 6.56 GiB KV cache，但半卡仅余 3.37 GiB（main、v0.26.0），恢复到物理整卡。 |
| `one_card/test_multistream_overlap_shared_expert.py` | 半卡并非 OOM，而是 HCCL `hcclGetRootInfo` error code 19（main、v0.26.0）；该用例需要物理设备的 HCCL 语义。 |
| `one_card/lora/test_qwen3_reranker_lora.py` | 1/2 卡复测仍在 ACL Graph capture 报 `Alloc sq cq fail`；这是 vNPU SQ/CQ 资源限制，不是增加显存即可解决，退回物理整卡。 |
| `one_card/test_batch_job_aware_scheduler_e2e.py` | 1/2 卡中 4 个参数有 3 个在 ACL Graph capture 报 `Alloc sq cq fail`；按整文件迁移规则退回物理整卡。 |
| `one_card/test_minicpm.py` | 9.1.0 的 1/2 卡仍在 MiniCPM-2B 图捕获阶段报 `EE1023`/`207005`：创建 stream 过多，属于 vNPU stream 资源限制。 |
| `one_card/spec_decode/test_dynamic.py` | 物理 A2 在上游 #13819 的 vLLM main 和 v0.26.0 矩阵均通过 DSpark/DFlash；1/2 卡 vNPU 连续两轮得到相同 DSpark acceptance 偏差，功能/数值等价性未建立，因此退回物理整卡。 |

310P 专用、A3、多卡 HCCL 测试保持原 runner，不参与本次 A2B3 vNPU 容量采样。

上游本轮将原单卡 `test_batch_invariant.py` 重构为
`four_card/rlhf/consistency/test_batch_invariant_tp4.py`，因此不再列入 vNPU。
新增 UT 中没有新的 `tests/ut/**/a2/` 文件；非 A2 UT 不因本次 vNPU 试验改变
runner。

## 本轮临时执行方式

- PR 的选择命令临时传入 `--npu-types a2_quarter a2_half`。选择器仍先按正常
  模块依赖收集测试，再只输出 1/4 和 1/2 两类 runner；CPU、物理整卡、310P、
  A3 和多卡组本轮不执行。
- 该过滤只接入 `pr_test.yaml`，不会改变 `/e2e` 命令、定时覆盖率任务或选择器
  的默认行为。完成本轮容量采样后应删除该参数。
- `run_selected_tests.sh` 对每个文件单独调用 pytest。某个文件失败时记录状态和
  日志并继续执行同一分桶内剩余文件；全部文件结束后统一打印汇总，并以首个非零
  状态退出，确保既收集完整结果又不掩盖失败。
- Dynamic 文件退回物理整卡后，1/2 卡剩余文件采用 3 桶，负载为
  1360/1400/1340 秒；1/4 卡继续使用 6 桶，负载为
  1310/1390/1370/1300/1370/1360 秒。两类最长桶仅相差 10 秒，使用 3/6 桶。
