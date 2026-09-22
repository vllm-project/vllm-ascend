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

- `tensor_parallel_size > 1`.
- `enable_expert_parallel` is set (MoE models only).
- `--additional-config '{"enable_flashcomm1": true}'` set `flashcomm1`

> [!NOTE]
> **Difference from upstream.** Upstream vLLM enables MoE sequence parallelism only when `data_parallel_size > 1`, together with a supported all2all backend, expert parallelism, and `tensor_parallel_size > 1`. On vLLM Ascend, `data_parallel_size > 1` is not part of the enablement condition. Ascend FlashComm also supports the TP/EP topology with `data_parallel_size = 1`, so SP MoE can be enabled when DP is 1 as long as the conditions above are met. `data_parallel_size > 1` remains supported.

### FlashComm switch (Ascend only)

vLLM Ascend enables SP MoE through the FlashComm switch. The switch is still
required; SP MoE is not enabled from the parallel configuration alone.

To enable SP MoE, set one of the following (the `additional_config` form is
preferred):

```bash
# Preferred. On vLLM Ascend, data-parallel-size may be 1.
# Upstream requires data-parallel-size > 1 for the same SP path.
vllm serve <moe-model> \
  --data-parallel-size 1 \
  --tensor-parallel-size 2 \
  --enable-expert-parallel \
  --additional-config '{"enable_flashcomm1": true}'
```

```bash
# Kept for compatibility. data-parallel-size may be 1 on vLLM Ascend.
VLLM_ASCEND_ENABLE_FLASHCOMM1=1 vllm serve <moe-model> \
  --data-parallel-size 1 \
  --tensor-parallel-size 2 \
  --enable-expert-parallel
```
