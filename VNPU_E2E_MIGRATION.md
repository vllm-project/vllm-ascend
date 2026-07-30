# vNPU E2E 迁移说明

本文记录 A2B3 vNPU 在 PR 选择性测试中的接入方式、测试分组依据和相对
[PR #12171](https://github.com/vllm-project/vllm-ascend/pull/12171) 的用例变化。

本次迁移从 `upstream/main` 的 `e462c42a4` 重新开始。旧 PR 只作为历史实测
证据，不复用它已经被 main 上新 cache 架构替代的 workflow 实现。

## 目标和边界

- 1/4 vNPU runner：`linux-aarch64-a2b3-v-quarter`，约 8 GiB NPU 显存、
  6 个 CPU 核。
- 1/2 vNPU runner：`linux-aarch64-a2b3-v-half`，约 16 GiB NPU 显存、
  11 个 CPU 核。
- 两种 vNPU 都是 A2B3、ARM64，并使用与 A2 整卡相同的
  `9.0.1-910b-ubuntu22.04-py3.12` 镜像。
- 迁移只改变测试资源路由，不拆分测试文件。同一文件按资源需求最大的用例
  选择 runner。
- 需要真实多卡通信、310P/A3 专用能力或物理 NPU 资源的测试保持原位置。

## csrc cache 接入

main 已经提供统一的 CPU csrc cache 链路：

1. `select_tests.py` 从 runner 元数据收集 `csrc_cache_target`。
2. `pr_test.yaml` 调用 `_ensure_csrc_cache.yaml` 检查目标 cache。
3. cache 缺失时，`_build_csrc_cache.yaml` 在 `linux-arm64-cpu-16` 上使用
   `SOC_VERSION=ascend910b1` 构建。
4. `_selected_tests.yaml` 在 NPU runner 上按相同 key 恢复 cache，并用
   `COMPILE_CUSTOM_KERNELS=0` 安装。

half/quarter runner 复用目标 `a2-arm64-ubuntu`，对应 key 为：

```text
vllm-ascend-build-v2-ARM64-9.0.1-910b-ubuntu22.04-py3.12-${CSRC_HASH}
```

vNPU 不再承担 cache miss 后的 csrc 编译。恢复失败时 job 直接报错退出，等待
CPU cache 构建成功后重跑，避免 6/11 核 CPU 上的长时间编译。非 vNPU runner
在 cache rollout 期间仍保留原有的 NPU 编译回退。

安装阶段仍按 runner CPU 配额设置 `MAX_JOBS`：

| runner | `SOC_VERSION` | `MAX_JOBS` | cache miss |
| --- | --- | ---: | --- |
| 1/4 vNPU | `ascend910b1` | 6 | 直接失败，不本地编译 |
| 1/2 vNPU | `ascend910b1` | 11 | 直接失败，不本地编译 |

## 分组口径

- 1/4 卡优先放无模型的合成张量测试、A2 UT，以及 0.6B/0.8B/1B 小模型。
- 1/2 卡放 1.5B FP32 对照、3B/4B BF16、8B W8A8 或主模型加草稿模型的
  用例。
- BF16/FP16 权重按约 2 bytes/parameter 估算，W8A8 按约
  1 byte/parameter 估算，并为 KV cache、ACL Graph、workspace 和运行时
  常驻内存保留余量。
- 模型虽小但依赖物理卡 SQ/CQ、NPU IPC 或真实多卡 HCCL 的用例不下沉。
- 当前分桶数为 1/4 卡 6 桶、1/2 卡 5 桶；按 `estimated_times` 做贪心均衡。

## 1/4 vNPU E2E

| 测试文件 | 资源判断 |
| --- | --- |
| `quarter_card/compile/test_graphex_norm_quant_fusion.py` | 无预训练模型，小型 BF16 合成张量 |
| `quarter_card/compile/test_graphex_qknorm_rope_fusion.py` | 无预训练模型，Q/K/RoPE 张量为 MiB 级 |
| `quarter_card/compile/test_norm_quant_fusion.py` | 无预训练模型，小算子融合测试 |
| `quarter_card/lora/test_ilama_lora.py` | iLlama 1B，FP16 权重约 2 GiB |
| `quarter_card/lora/test_qwen3_multi_loras.py` | Qwen3-0.6B，多 LoRA 增量较小 |
| `quarter_card/pooling/test_embedding.py` | 最大约 0.6B，HF 与 vLLM 顺序运行 |
| `quarter_card/pooling/test_scoring.py` | MiniLM/bge 小模型，FP16 pooling |
| `quarter_card/test_attention_fa3.py` | Qwen3-0.6B，短上下文 |
| `quarter_card/test_batch_job_aware_scheduler_e2e.py` | Qwen3-0.6B；两种 scheduler 顺序对比 |
| `quarter_card/test_camem.py` | Qwen3-0.6B，sleep/wake |
| `quarter_card/test_completion_with_prompt_embeds.py` | Qwen3-0.6B；embedding 对照主要在 CPU |
| `quarter_card/test_cpu_weight_offload.py` | Qwen3-0.6B，部分权重主动卸载到 CPU |
| `quarter_card/test_guided_decoding.py` | Qwen3-0.6B，主要增加 CPU 侧解析 |
| `quarter_card/test_minimax_m3_sparse_attn.py` | 不加载 MiniMax M3 权重；仅生产形状的合成 KV/index 张量和稀疏算子 |
| `quarter_card/test_multi_instance.py` | 两个 Qwen3-0.6B 实例，总显存预算受 0.4 比例限制 |
| `quarter_card/test_qwen3_0_6b.py` | Qwen3-0.6B，capture size 较小 |
| `quarter_card/test_qwen3_5_0_8b.py` | Qwen3.5-0.8B，预计仍低于 8 GiB |
| `quarter_card/test_qwen3_embedding_0_6b.py` | Qwen3-Embedding-0.6B |
| `quarter_card/test_sampler.py` | Qwen3-0.6B，实际输入较短 |
| `quarter_card/test_simple_cpu_offload.py` | Qwen3-0.6B，NPU 显存比例为 0.5 |
| `quarter_card/test_xlite.py` | Qwen3-0.6B，短上下文 |

其中以下两个文件是旧 PR 基线之后新增、此次重新分析后加入 1/4 vNPU 的用例：

- `test_batch_job_aware_scheduler_e2e.py`：四类用例都使用 Qwen3-0.6B，
  每次基线与目标 scheduler 顺序启动，不同时常驻两份模型。
- `test_minimax_m3_sparse_attn.py`：文件名对应 MiniMax M3，但没有加载完整模型。
  最大生产形状使用 10240 token 的分页 KV/index cache，实际分配规模远小于
  8 GiB，适合先在 1/4 vNPU 验证。

## 1/2 vNPU E2E

| 测试文件 | 资源判断 |
| --- | --- |
| `half_card/lora/test_llama32_lora.py` | Llama-3.2-3B BF16 加 LoRA，8 GiB 余量不足 |
| `half_card/lora/test_lora_with_spec_decode.py` | Qwen3-1.7B 主模型、Eagle3 草稿模型和 LoRA |
| `half_card/pooling/test_classification.py` | Qwen2.5-1.5B 的 HF FP32 对照约 6 GiB |
| `half_card/test_qwen3_8b_w8a8.py` | 8B W8A8 主模型加 Eagle3、KV cache 和 FULL Graph |

## A2 UT

`tests/ut/**/a2/test_*.py` 统一路由到 1/4 vNPU。当前共有 34 个文件，主要使用
mock、小型合成张量或配置对象，不加载完整预训练权重，也不建立真实多卡
HCCL，因此不需要占用 A2 物理整卡。

相对旧 PR 基线，数量仍为 34，但集合发生了变化：

- 新增 `tests/ut/ops/a2/test_gdn_layerwise_kv.py`：GDN layerwise KV 的小型
  metadata/cache 张量测试，适合 1/4 vNPU。
- 删除 `tests/ut/worker/a2/test_kvcomp_utils.py`：不再进入任何 runner。

其余修改过的 A2 UT 没有新增完整模型或真实多卡依赖，继续使用 1/4 vNPU。

## 保持原 runner 的新增/删除用例

旧 PR 基线之后新增的多卡 E2E 不迁入 vNPU：

- `four_card/spec_decode/test_dspark_deepseekv4.py`：保留四卡语义。
- `two_card/lora/test_qwen35_densemodel_lora_tp.py`：覆盖 TP=1/2、
  fully-sharded LoRA 与 ACL Graph 组合，保留双卡语义。
- `two_card/lora/test_qwen3moe_lora.py`：保留两卡 LoRA/TP 语义；它替代了已删除
  的 `two_card/lora/test_qwen3moe_lora_tp.py`。
- `two_card/test_gemma4.py`、`two_card/test_xlite.py`：保留两卡语义。

已删除的 `four_card/context_parallel/test_prefix_caching_cp.py` 不再调度。

以下单卡文件也继续留在物理整卡：

| 测试范围 | 保留原因 |
| --- | --- |
| `one_card/lora/test_qwen3_reranker_lora.py` | 历史 vNPU 实测在 ACL Graph capture 出现 SQ/CQ 资源分配失败 |
| `one_card/test_minicpm.py` | 文件同时包含 2B 模型，历史半卡仍在 graph capture 失败 |
| `one_card/lora/test_olmoe_lora.py` | MoE 总权重约 7B，16 GiB 缺少可靠余量 |
| `one_card/model_runner_v2/test_basic.py`、`one_card/spec_decode/` | 文件包含 8B BF16 主模型和草稿模型 |
| `one_card/test_batch_invariant.py` | 高 batch、长 prompt 的 KV cache 峰值较大 |
| `one_card/test_npu_ipc_weight_transfer.py` | 依赖物理 NPU IPC 能力，不是显存大小问题 |
| `one_card/test_vlm.py` | 文件包含 7B/8B 多模态、音频模型 |
| `one_card/model_runner_v2/test_uva.py`、`one_card/test_cpu_offloading.py` | 当前整体 skip，迁移不能产生有效 vNPU 覆盖 |

## 验证要求

本地可验证的内容：

- YAML/JSON/Python/Shell 格式和静态检查；
- `test_select_tests.py` 的 half/quarter 路由、分桶和 cache target 输出；
- `coverage.py` 对迁移后路径的完整性检查；
- main2main 固定用例路径存在。

本地 macOS 没有 Ascend NPU，以下项目必须由 CI 实测：

- CPU 构建的 A2 csrc cache 能被两种 vNPU 恢复；
- vNPU cache miss 会快速失败，不会进入本地 csrc 编译；
- 1/4 卡 21 个 E2E、34 个 A2 UT 和 1/2 卡 4 个 E2E 的真实通过率；
- 新迁入的 BatchJobAwareScheduler 与 MiniMax M3 稀疏算子用例；
- pytest skip 不计为通过，OOM、SQ/CQ 和硬件能力失败需要单独记录并回退。

旧 PR 的
[run 29482312479](https://github.com/vllm-project/vllm-ascend/actions/runs/29482312479)
和
[run 29487123938](https://github.com/vllm-project/vllm-ascend/actions/runs/29487123938)
可作为分组的历史证据，但不能替代当前 main、当前 cache artifact 和当前测试
集合上的重新验证。
