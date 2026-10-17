# Sequence Parallelism

## Overview

Sequence Parallelism (SP) shards the token dimension across tensor-parallel
ranks around the communication boundaries of transformer layers.

On vLLM Ascend, SP covers the MoE path (SP MoE) and an opt-in dense path for
the Qwen3.5 text backbone used by Qwen3.8-27B. The attention `o_proj` normally
ends with a TP all-reduce, so its inputs are replicated on every TP rank. SP
replaces that boundary with reduce-scatter and keeps the residual stream
sharded by token.

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

For the Qwen3.8 dense path, both attention and MLP keep their tensor-parallel
weights. Each block gathers the token shards before attention and before MLP,
then reduce-scatters the partial `o_proj` or `down_proj` result. The input is
sharded once before the first decoder layer and gathered after the final norm.

Different DP ranks may have different numbers of valid tokens. Therefore, the
buffer after all-gather cannot be treated as a contiguous sequence of valid
tokens; it must be unpadded and zero-padded according to each rank's local
token size. This keeps tokens sequence-sharded during expert computation and
reduces duplicate computation and unnecessary communication.

## How to use

For SP MoE:

- `tensor_parallel_size > 1`.
- `enable_expert_parallel` is set (MoE models only).
- `--additional-config '{"enable_flashcomm1": true}'` set `flashcomm1`

For the Qwen3.8-27B dense path, the initial implementation supports this
configuration:

- Model Runner V1 (`VLLM_USE_V2_MODEL_RUNNER=0`, which is the Ascend default).
- A2, A3, or A5 hardware; the 310P compatibility path is not supported.
- `tensor_parallel_size > 1`.
- `data_parallel_size = 1`, `pipeline_parallel_size = 1`, and PCP/DCP disabled.
- Fine-grained `o_proj` and MLP tensor parallelism disabled.
- `--additional-config '{"enable_flashcomm1": true}'`.

MTP speculative decoding, prefix caching, and `FULL_DECODE_ONLY` graph mode
use the same sequence-sharded decoder path.

> [!NOTE]
> **Difference from upstream.** Upstream vLLM enables MoE sequence parallelism only when `data_parallel_size > 1`, together with a supported all2all backend, expert parallelism, and `tensor_parallel_size > 1`. On vLLM Ascend, `data_parallel_size > 1` is not part of the enablement condition. Ascend FlashComm also supports the TP/EP topology with `data_parallel_size = 1`, so SP MoE can be enabled when DP is 1 as long as the conditions above are met. `data_parallel_size > 1` remains supported.

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

For Qwen3.8-27B dense serving with TP=2:

```bash
export VLLM_USE_V2_MODEL_RUNNER=0
vllm serve <qwen3.8-27b-model> \
  --data-parallel-size 1 \
  --tensor-parallel-size 2 \
  --additional-config '{"enable_flashcomm1": true}'
```
