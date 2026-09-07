# Sequence Parallelism

## Overview

Sequence Parallelism (SP) shards the token dimension across tensor-parallel
ranks around the communication boundaries of transformer layers.

On vLLM Ascend, SP currently covers the MoE path (SP MoE). The attention
`o_proj` ends with a TP all-reduce, so its inputs are replicated on every TP
rank. Feeding those replicated tokens directly into the experts duplicates
compute and communication under expert parallelism. SP MoE keeps the expert
inputs sharded by sequence and restores the expected layout at the MoE output
boundary instead.

## Principle

SP MoE shards the input along the token dimension at the MoE boundary in each
Transformer layer. Different TP ranks therefore process different tokens,
avoiding duplicate expert computation for the same tokens.

The main data flow of an MoE layer is:

```text
Sequence-parallel input sharding
  -> EP all-gather: collect tokens from all ranks
  -> Unpad according to each rank's actual token count
  -> Routing, dispatch, and expert computation
  -> Zero-pad according to the actual token counts
  -> EP reduce-scatter: redistribute tokens
  -> TP all-gather: restore the layout required by subsequent layers
```

Different DP ranks may have different numbers of valid tokens. Therefore, the
buffer after all-gather cannot be treated as a contiguous sequence of valid
tokens; it must be unpadded and zero-padded according to each rank's local
token size. This keeps tokens sequence-sharded during expert computation and
reduces duplicate computation and unnecessary communication.

## How to use

Upstream vLLM owns the SP MoE switch. `ParallelConfig.use_sequence_parallel_moe`
is true only when all of the following hold:

- `tensor_parallel_size > 1` and `data_parallel_size > 1`.
- `enable_expert_parallel` is set (MoE models only).
- `all2all_backend` is an SP-capable backend: `allgather_reducescatter`,
  `deepep_high_throughput`, `deepep_low_latency`, `mori_high_throughput`,
  `mori_low_latency`, or `nixl_ep`.

### Temporary FlashComm switch (Ascend only)

Until SP support is fully validated, vLLM Ascend keeps SP MoE option by original flashcomm option.

To opt into upstream SP MoE, set one of the following (the
`additional_config` form is preferred):

```bash
# Preferred.
vllm serve <moe-model> \
  --data-parallel-size 2 \
  --tensor-parallel-size 2 \
  --enable-expert-parallel \
  --additional-config '{"enable_flashcomm1": true}'
```

```bash
# Kept for compatibility.
VLLM_ASCEND_ENABLE_FLASHCOMM1=1 vllm serve <moe-model> \
  --data-parallel-size 2 \
  --tensor-parallel-size 2 \
  --enable-expert-parallel
```

This switch is temporary and deprecated. Referencing either form logs a
`FlashComm is deprecated` warning from `init_ascend_config`, and the override
carries a `TODO` to remove it once SP is supported — after that, the upstream
configuration above takes effect directly. DSA-CP also depends on this switch
(plus `pipeline_parallel_size == 1`); see the
[Context Parallel Guide](context_parallel.md) for details.
