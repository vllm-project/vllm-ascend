# Sequence Parallelism

## What is Sequence Parallelism

Sequence Parallelism (SP) was first introduced in [Megatron](https://arxiv.org/pdf/2205.05198), with the original intention of reducing training activation memory. The core modification was changing `Allreduce->LayerNorm` to `ReduceScatter->LayerNorm->Allgather`. This technique was later applied to inference by vllm. It should be noted that splitting Allreduce into ReduceScatter and Allgather does not inherently bring performance benefits; it reduces the computation load of LayerNorm, but this gain is minimal. The real benefits of SP come from:

1. LLM inference deployment often uses quantization. Taking INT8 quantization commonly used on NPUs as an example, after LayerNorm, a Quant operator quantizes the hidden states from BF16 to INT8. The communication volume of Allgather is halved, and the time consumption is almost halved.
2. ReduceScatter and Allgather can be fused with the preceding and following Matmul operations respectively into communication-computation parallel operators, reducing latency.

## How to Use

Currently, vllm-ascend has implemented Sequence Parallelism for VL-class models based on the Inductor pass. It can be enabled in the following way:

```bash
vllm serve Qwen/Qwen3-VL-2B-Instruct \
    --tensor-parallel-size 2 \
    --compilation-config '{"pass_config": {"enable_sp": true , "sp_min_token_num": 1000}}'
```

- `"enable_sp"`: This is the switch for SP. Since SP relies on graph mode, it is not supported in eager mode.
- `sp_min_token_num` (from upstream vllm's `pass_config`): Based on our experiments, when the number of tokens is small (empirical value is less than 1000), SP can actually bring negative impact. This is because when the communication volume is small, the fixed overhead of the communication operator becomes the dominant factor. SP will only take effect when `num_tokens >= sp_min_token_num`. **The default value is 1000 on Ascend, which generally does not need to be modified.** To customize, use `--compilation-config '{"pass_config": {"enable_sp": true, "sp_min_token_num": 512}}'`. The value will be appended into `compile_ranges_split_points`, which splits the graph compilation range and checks whether the pass is applicable per range.

Without modifying `sp_min_token_num`, the simplest way and recommended way to enable SP is:

```bash
vllm serve Qwen/Qwen3-VL-2B-Instruct \
    --tensor-parallel-size 2 \
    --compilation-config '{"pass_config": {"enable_sp": true}}'
```

## Difference Between SP and Flash Comm V1

[Flash Comm V1 (FC1)](https://gitcode.com/ascend-tribe/ascend-inference-cluster/blob/main/FlashComm/ascend-inference-cluster-flashcomm.md) is an enhanced version of Sequence Parallelism developed based on NPU. The enhancements include:

1. For models using the MLA structure, Allgather is postponed until after QKV projection, further reducing communication volume.
2. For MoE models, Allgather is postponed until after Gating+DynamicQuant, also aiming to reduce communication volume.

FC1 is a unique optimization in vllm-ascend, currently implemented based on Custom OP, but it is difficult to support VL-class models (reasons detailed in [[RFC]: support sequence parallelism by pass](https://github.com/vllm-project/vllm-ascend/issues/5712) ). Therefore, currently FC1 and SP are complementary.

## Support Matrix

### Without Quantization

|                      | VL + Dense | VL + MoE | non-VL + Dense | non-VL + MoE |
| -------------------- | ---------- | -------- | -------------- | ------------ |
| Sequence Parallelism | graph      | graph    | x              | x            |
| Flash Comm V1        | x          | x        | eager/graph    | eager/graph  |

### With Quantization

SP currently does not support quantization and is under adaptation.

|                      | VL + Dense | VL + MoE | non-VL + Dense | non-VL + MoE |
| -------------------- | ---------- | -------- | -------------- | ------------ |
| Sequence Parallelism | x          | x        | x              | x            |
| Flash Comm V1        | x          | x        | eager/graph    | eager/graph  |

## Pass Design

When SP is enabled, the following passes run in order: `SequenceParallelismPass` then `SequenceParallelismMoePass`.

### SequenceParallelismPass

Runs `NoOpEliminationPass` first to eliminate redundant view-like operations, then applies AllReduce-based patterns:

| Pattern                                | Match                            | Replacement                                                                           |
| -------------------------------------- | -------------------------------- | ------------------------------------------------------------------------------------- |
| `MiddleAllReduceRMSNormPattern`        | `all_reduce` + `layernorm`       | `reduce_scatter` + `layernorm` + `all_gather`                                         |
| `LastAllReduceRMSNormPattern`          | Same (last layer, no residual)   | Same                                                                                  |
| `Qwen3VLMiddleAllReduceRMSNormPattern` | `all_reduce` + add + `layernorm` | `reduce_scatter` + chunk(`deepstack_input_embeds`) + add + `layernorm` + `all_gather` |

**Why Qwen3 VL needs special handling by Qwen3VLMiddleAllReduceRMSNormPattern**

Qwen3-VL middle layers insert an extra add between `all_reduce` and `layernorm`: `hidden_states=hidden_states + deepstack_input_embeds`. Under SP, `hidden_states` (i.e., `input`) is reduced-scattered to shape `[seq_len/tp, hidden]` per rank, while `deepstack_input_embeds` comes from the vision/deepstack path and stays full-sequence `[seq_len, hidden]` (typically replicated across TP ranks). Simply doing `reduce_scatter(input) + deepstack_input_embeds` would cause a shape mismatch.
The fix is to chunk `deepstack_input_embeds` by `tp_size` so each rank uses `add(reduce_scatter, chunk(deepstack_input_embeds)[tp_rank])`, keeping shapes consistent before `layernorm` and `all_gather`.

### SequenceParallelismMoePass

After `SequenceParallelismPass` applies, the MoE model computation graph looks like:

![AllGather EP computation graph](../../assets/sp_moe.png)

**Overview**

1. **Postponing allgather**: Under SP, `residual` is chunked by tensor parallelism. This causes a shape mismatch between hidden states and residual in the next layer's layernorm: hidden states are gathered (full sequence) while residual remains chunked. The fix is to move `all_gather` to *after* layernorm so that layernorm operates on consistent shapes per rank. `MiddleLayerAllgatherAddRMSNormPattern`, `LastLayerAllgatherRMSNormPattern`, and `Qwen3VLMiddleLayerAllgatherAddRMSNormPattern` are designed for this purpose, each handling different layer and structure variants (see the table below).

2. **AllGatherChunkNoOp cleanup**: When MoE SP is enabled, vllm introduces a `sequence_parallel_chunk` op (corresponding to `sp_chunk` in the diagram). Together with the preceding `all_gather`, the pair forms a redundant no-op (all_gather gathers, then chunk re-splits). `AllGatherChunkNoOpPattern` replaces this pair with identity to eliminate the redundant communication and computation.

**Pattern details:**

| Pattern                            | Match                                    | Replacement                             |
| ---------------------------------- | ---------------------------------------- | --------------------------------------- |
| `MiddleLayerAllgatherAddRMSNormPattern`        | `all_gather` + slice + `layernorm`       | `layernorm` + `all_gather`              |
| `LastLayerAllgatherRMSNormPattern`             | Same (last layer, no residual)           | Same                                    |
| `Qwen3VLMiddleLayerAllgatherAddRMSNormPattern` | `all_gather` + slice + add + `layernorm` | add(chunk) + `layernorm` + `all_gather` |
| `AllGatherChunkNoOpPattern`                    | `all_gather` + `sequence_parallel_chunk_impl` | identity (no-op)                    |

### FAQ

#### Q1: Is SP enabled by default?

No, SP is not enabled by default. SP is currently in the experimental stage and will be enabled by default in the future.

The processing flow of `enable_sp` in the code is:

- In `pass_config`, `enable_sp` and `sp_min_token_num` default to `None`
- `NPUPlatform.apply_config_platform_defaults`: If `enable_sp` is `True` and `sp_min_token_num` is None, set default `sp_min_token_num` (1000 for Dense models, 1 for MoE models)
- `VllmConfig._apply_optimization_level_defaults`: `enable_sp` is set to `True` for dense models.
- `VllmConfig.__post_init__`: If `sp_min_token_num` is still `None`, then `enable_sp` is set to `False`
