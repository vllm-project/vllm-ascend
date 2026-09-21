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

**The original flashcomm feature overlapped functionally with the SP feature and has been deprecated since v0.27.1.**

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
  -> Moe
```

Different DP ranks may have different numbers of valid tokens. Therefore, the
buffer after all-gather cannot be treated as a contiguous sequence of valid
tokens; it must be unpadded and zero-padded according to each rank's local
token size. This keeps tokens sequence-sharded during expert computation and
reduces duplicate computation and unnecessary communication.

## How to use

Steps to follow to enable SP currently:

- `tensor_parallel_size > 1` and `data_parallel_size > 1`.
- `enable_expert_parallel` is set (MoE models only).
- `--additional-config '{"enable_flashcomm1": true}'` set `flashcomm1`

### Matmul reduce-scatter fusion

The reduce-scatter in the flow above consumes the output of the unreduced
`o_proj` matmul. CANN can run both as a single pipelined kernel
(`npu_mm_reduce_scatter_base`), which vLLM Ascend applies as an FX graph
rewrite behind upstream's `fuse_gemm_comms` switch:

```bash
vllm serve <moe-model> \
  --data-parallel-size 2 \
  --tensor-parallel-size 2 \
  --enable-expert-parallel \
  --additional-config '{"enable_flashcomm1": true}' \
  --compilation-config '{"pass_config": {"fuse_gemm_comms": true}}'
```

The rewrite is skipped, leaving the unfused matmul and reduce-scatter in place,
when any of the following holds:

- the model runs eagerly, so there is no compiled graph to rewrite,
- `tensor_parallel_size` is not 2, 4, or 8, since the kernel needs an all-mesh
  HCCS topology,
- the projection is quantized, or its dtype is neither bfloat16 nor float16,
- the contracted dimension falls outside `[256, 65535)`,
- the projection has a bias, which is applied before the reduction.

Upstream treats `fuse_gemm_comms` as async tensor parallelism built on the
sequence-parallelism Inductor pass, so enabling it also turns on
`pass_config.enable_sp` and switches the model to full-graph compilation.
Ascend does not use upstream's `AsyncTPPass`, whose replacements are CUDA
`symm_mem` collectives.

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
configuration above takes effect directly.
