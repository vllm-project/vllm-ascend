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

Upstream vLLM owns the SP MoE switch. `ParallelConfig.use_sequence_parallel_moe`
is true only when all of the following hold:

- `tensor_parallel_size > 1` and `data_parallel_size > 1`.
- `enable_expert_parallel` is set (MoE models only).
- `all2all_backend` is an SP-capable backend: `allgather_reducescatter`,
  `deepep_high_throughput`, `deepep_low_latency`, `mori_high_throughput`,
  `mori_low_latency`, or `nixl_ep`.

`vllm_ascend.utils.enable_sp()` is a thin wrapper over this flag and drives the
Ascend runtime behavior (SP padding, TP-aligned graph capture sizes, and the
MoE communication path).

Per MoE layer, the communication flow with SP MoE enabled is:

```text
sequence_parallel_chunk
  -> EP all-gather
  -> unpad to the local token sizes of each rank
  -> MoE compute
  -> zero-pad back to the local sizes
  -> EP reduce-scatter
  -> upstream TP all-gather
```

DP ranks may hold uneven token counts, so the EP all-gather buffer must be
unpadded by per-rank local sizes instead of being treated as a contiguous
valid token sequence.

At compile time, `compilation_config.pass_config.enable_sp` (implied by
`fuse_gemm_comms`) lets the torch.compile sequence-parallelism fusion pass
rewrite communication boundaries, guarded by the `sp_min_token_num` threshold
derived from hidden size, TP size, and dtype. vLLM Ascend additionally derives
`AscendConfig.enable_sp_by_pass` from this flag when graph compilation is
active. At run time, cudagraph capture sizes are filtered to multiples of the
TP size via `update_sizes_for_sequence_parallelism`.

## How to use

SP MoE needs no Ascend-specific model code. Configure the upstream parallel
options:

```bash
vllm serve <moe-model> \
  --data-parallel-size 2 \
  --tensor-parallel-size 2 \
  --enable-expert-parallel \
  --all2all-backend allgather_reducescatter
```

Constraints:

- SP requires `tensor_parallel_size > 1`; MoE models additionally require
  `enable_expert_parallel`. Both are validated in `_validate_sfa_dcp_kv_sp`.
- Cudagraph capture sizes must contain values that are multiples of the TP
  size, otherwise server initialization fails.
- Prefill Context Parallel (PCP) cannot be combined with SP.

### Temporary FlashComm switch (Ascend only)

Until SP support is fully validated, vLLM Ascend keeps SP MoE disabled by
default: unless the temporary switch described below is on,
`_setup_compile_backend` and `_setup_worker_and_scheduler` in
`vllm_ascend/platform.py` override `all2all_backend` with
`flashinfer_all2allv`, which is not an SP-capable backend, so
`use_sequence_parallel_moe` stays false.

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
