# Sequence Parallelism

## Overview

Sequence Parallelism (SP) shards the token dimension across tensor-parallel
ranks around the communication boundaries of transformer layers.

On vLLM Ascend, SP covers two paths:

- **SP MoE** (MoE models): the attention `o_proj` ends with a TP all-reduce,
  so its inputs are replicated on every TP rank. Feeding those replicated
  tokens directly into the experts duplicates compute and communication under
  expert parallelism. SP MoE keeps the expert inputs sharded by sequence and
  restores the expected layout at the MoE output boundary instead.
- **Dense SP** (Qwen3.5 dense family): dense checkpoints such as
  Qwen3.5-27B / Qwen3.6-27B / Qwen3.8-27B have no experts to shard, so SP
  applies to the whole decoder stack instead: hidden states and the residual
  stay sequence-sharded between layers, the RMSNorms run on the rank-local
  shard, attention (full attention and GDN linear attention) all-gathers the
  full sequence and reduce-scatters its sequence-parallel output projection,
  and the dense MLP all-gathers around its TP-sharded weights before
  reduce-scattering the partial output back to the shard. Each TP all-reduce
  is thereby split into a reduce-scatter plus all-gather pair around the
  layer boundaries, which reduces communication latency on prefill.

## Principle

SP MoE shards the input along the token dimension in each
Transformer layer. Different TP ranks therefore process different tokens,
avoiding duplicate expert computation for the same tokens.

The main data flow of an MoE layer is:

```text
Sequence-parallel input sharding
  -> TP all-gather: collect tokens from all ranks
  -> attention
  -> TP reduce scatter
  -> RMS Norm
  -> Router
  -> all-to-all
  -> MoE
```

Different DP ranks may have different numbers of valid tokens. Therefore, the
buffer after all-gather cannot be treated as a contiguous sequence of valid
tokens; it must be unpadded and zero-padded according to each rank's local
token size. This keeps tokens sequence-sharded during expert computation and
reduces duplicate computation and unnecessary communication.

## How to use

Steps to follow to enable SP currently:

- `tensor_parallel_size > 1`.
- `enable_expert_parallel` is set (MoE models only).
- `--additional-config '{"enable_flashcomm1": true}'` sets `flashcomm1`.

> [!NOTE]
> **Difference from upstream.** Upstream vLLM enables MoE sequence parallelism only when `data_parallel_size > 1`, together with a supported all2all backend, expert parallelism, and `tensor_parallel_size > 1`. On vLLM Ascend, `data_parallel_size > 1` is not part of the enablement condition. Ascend FlashComm also supports the TP/EP topology with `data_parallel_size = 1`, so SP MoE can be enabled when DP is 1 as long as the conditions above are met. `data_parallel_size > 1` remains supported.

### Dense models (Qwen3.5 dense family)

For dense checkpoints of the Qwen3.5 family (model type `qwen3_5_text`,
including Qwen3.5-27B, Qwen3.6-27B and Qwen3.8-27B), the FlashComm switch
enables dense SP directly — no expert parallelism is involved:

```bash
vllm serve Qwen/Qwen3.8-27B \
  --tensor-parallel-size 2 \
  --additional-config '{"enable_flashcomm1": true}'
```

Dense SP requires `pipeline_parallel_size == 1`. Pipeline parallel and
LoRA deployments keep the default all-reduce layout. When ACLGraph is
enabled, sequence parallelism aligns the graph capture sizes to multiples of
the tensor-parallel size automatically.

#### Performance notes for dense SP

Dense SP targets prefill-heavy workloads. Unlike SP MoE it removes no
duplicate expert computation — every rank still runs the full-token GEMMs —
so the expected gains are more modest and workload dependent:

- The per-layer tensor-parallel all-reduces become reduce-scatter plus
  all-gather pairs of the same total communication volume. What shrinks is
  the elementwise work: the RMSNorms and residual adds run on the rank-local
  token shard, and the between-layer activations shrink by the
  tensor-parallel factor.
- Decode-heavy, small-batch serving may see little benefit or a small
  regression (twice the per-layer collective count plus tensor-parallel
  aligned batch padding); keep the switch off for such workloads.
- Larger gains require fusing the row-parallel GEMMs with their
  reduce-scatter; see [Matmul reduce-scatter fusion](#matmul-reduce-scatter-fusion)
  and validate on the target hardware before relying on it.

### Matmul reduce-scatter fusion

#### Graph fusion

On supported Ascend devices, enable `fuse_gemm_comms` to fuse a
sequence-parallel, bias-free row-parallel projection with its token-dimension
reduce-scatter. The fused implementation supports BF16/FP16 with TP size 2,
4, or 8; unmatched graphs continue to use the unfused path.

```bash
--compilation-config '{"pass_config":{"fuse_gemm_comms":true}}'
```

#### Layer-level O-projection fusion

MiniMax-M3 and Kimi-K3 instead install a fused custom op on the attention
O-projection automatically when sequence parallelism is active, so no extra
switch is required. The op re-evaluates its gates on every forward and falls
back to the plain MM + reduce-scatter path whenever they are not met; both
paths return the same SP token shard, so the fallback is transparent.

The current eligibility matrix is:

| Gate | MiniMax-M3 | Kimi-K3 |
| --- | --- | --- |
| **Hardware platform** | Atlas A2 products, Atlas A3 products, Ascend 950PR&950DT products | Ascend 950PR&950DT products only |
| **Hardware capability gate** | `MM_REDUCE_SCATTER_BF16` | `MM_REDUCE_SCATTER_AI_CPU_INFERENCE` |
| **SP switch** | `use_sequence_parallel=True` | `use_sequence_parallel=True` |
| **Extra model condition** | None | `use_attn_residuals=True` |
| **TP size** | {2, 4, 8} | {2, 4, 8, 16, 32, 64} |
| **Weight precision** | BF16, unquantized | BF16, unquantized |
| **Input precision** | BF16 | BF16 |
| **Input shape** | Nonempty 2D tensor | Nonempty 2D tensor |
| **K dimension** | [256, 65535) | [256, 65535), and the input K must match the weight K |
| **Token count** | [256, 16384) | No fixed token-count limit |
| **Communication volume** | Bounded by the token-count limit | `padded_tokens × N × 2B < 4GiB` |
| **Weight format** | ND format | `weight_nz_mode != 2` |
| **Weight contiguity** | Must be contiguous | Must be contiguous |
| **Batch type** | Pure prefill | Pure prefill |
| **Decode / mixed batch** | Not fused | Not fused |
| **Pure decode node** | Not supported | Excluded by the dynamic batch gate |
| **LoRA** | Not supported | Not supported |
| **Bias** | Not supported | Not supported |
| **PCP** | PCP = 1 | Excludes `HeadParallelRowLinear` (PCP KDA case) |
| **DCP** | DCP = 1 | No explicit restriction |
| **Existing custom op** | Not allowed | Not allowed |
| **TP sharding contract** | `input_is_parallel=True`, `disable_tp=False` | No explicit restriction |
| **External RS contract** | `reduce_results=False` | No explicit restriction |
| **Forward context** | Must exist with nonempty `attn_metadata` | Must satisfy the pure-prefill check |
| **Underlying interface** | `npu_mm_reduce_scatter_base` | `npu_quant_mm_reduce_scatter` |

### FlashComm switch (Ascend only)

vLLM Ascend enables SP MoE through the FlashComm switch. The switch is still
required; SP MoE is not enabled from the parallel configuration alone.

To enable SP MoE, set `additional_config.enable_flashcomm1`:

```bash
# On vLLM Ascend, data-parallel-size may be 1.
# Upstream requires data-parallel-size > 1 for the same SP path.
vllm serve <moe-model> \
  --data-parallel-size 1 \
  --tensor-parallel-size 2 \
  --enable-expert-parallel \
  --additional-config '{"enable_flashcomm1": true}'
```
