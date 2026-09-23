# Context Parallel Guide

## Overview

Context Parallel (CP) serves long-context requests by splitting work or KV-cache storage along the sequence dimension:

- Prefill Context Parallel (PCP) splits the prefill tokens of a long prefill request across additional ranks. Each rank computes a different part of the sequence, reducing time to first token (TTFT).
- Decode Context Parallel (DCP) shards the KV cache across ranks in a DCP group, which may reuse ranks from the PCP group, the Tensor Parallel (TP) group, or both, depending on the parallel configuration. It reduces duplicated KV-cache storage and can increase decode throughput.

For a general introduction to these two strategies, see the upstream [vLLM Context Parallel Deployment](https://docs.vllm.ai/en/latest/serving/context_parallel_deployment/) guide.

DSA-CP is a separate sparse-attention optimization controlled by `additional_config.enable_dsa_cp`. Enabling it automatically enables FlashComm as the all2all backend; there is no need to set `enable_flashcomm1` separately. It will be removed once PCP support is stable. See [Additional Configuration](../configuration/additional_config.md) for its configuration and model requirements.

## Supported Scenarios

### Prefill Context Parallel

PCP support is experimental and available only with ModelRunner V2. The following table shows the basic backend support and whether each feature can be combined with PCP:

| Attention Backend | Basic PCP | Prefix Caching + PCP | Chunked Prefill + PCP | MLAPO + PCP | Speculative Decoding + PCP | P/D Disaggregation + PCP | KV Cache Pool + PCP | Sequence Parallelism (SP) + PCP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLA | ✅ Full compatibility | ✅ Full compatibility | ✅ Full compatibility | ✅ Full compatibility | 🟠 Partial compatibility (MTP, eager and `FULL_DECODE_ONLY`) | ✅ Full compatibility (`MooncakeConnectorV1`) | 🟠 Partial compatibility (`AscendStoreConnector`, non-layerwise) | ❌ No compatibility |
| GQA | ✅ Full compatibility | ✅ Full compatibility | ✅ Full compatibility | — Not applicable | 🟠 Partial compatibility (Eagle3, eager and `FULL_DECODE_ONLY`) | ✅ Full compatibility (`MooncakeConnectorV1`) | 🟠 Partial compatibility (`AscendStoreConnector`, non-layerwise) | ❌ No compatibility |
| SFA | ✅ Full compatibility | ✅ Full compatibility | ✅ Full compatibility | ❌ No compatibility | ❌ No compatibility | ✅ Full compatibility (`MooncakeConnectorV1`) | 🟠 Partial compatibility (`AscendStoreConnector`, non-layerwise) | ❌ No compatibility |
| DSA | ✅ Full compatibility | ✅ Full compatibility | ✅ Full compatibility | — Not applicable | 🟠 Partial compatibility (MTP and DSpark, eager and `FULL_DECODE_ONLY`) | ✅ Full compatibility (`MooncakeHybridConnector`) | 🟠 Partial compatibility (`AscendStoreConnector`, non-layerwise) | ❌ No compatibility |

- ✅ **Full compatibility**: The basic path or feature combination is supported.
- 🟠 **Partial compatibility**: The basic path or feature combination is supported with the stated limitations.
- ❌ **No compatibility**: The backend or feature combination is not supported by the current MRV2 PCP implementation.
- **Not applicable**: The feature does not apply to the attention backend.

### Decode Context Parallel

DCP supports eager and graph execution, prefix caching, chunked prefill, speculative decoding, P/D disaggregation, and MLAPO on the model and hardware combinations documented by vLLM Ascend. The following table shows whether each feature can be combined with DCP across devices and attention backends:

| Device | Attention Backend | Chunked Prefill + DCP | Prefix Caching + DCP | Graph Mode + DCP | P/D Disaggregation + DCP | MLAPO + DCP | Speculative Decoding + DCP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Ascend A2/A3 | MLA/GQA | ✅ Full compatibility | ✅ Full compatibility | ✅ Full compatibility | ✅ Full compatibility | ✅ Full compatibility (MLA)<br>— Not applicable (GQA) | ✅ P/D disaggregation<br>❌ PD-mixed deployment |
| Ascend A2/A3 | SFA | ✅ Full compatibility | ✅ Full compatibility | ✅ Full compatibility | ✅ Full compatibility | ✅ Full compatibility | ✅ Full compatibility |
| 950PR&950DT Products | MLA/GQA | 🟠 Partial compatibility | 🟠 Partial compatibility | 🟠 Partial compatibility | 🟠 Partial compatibility | 🟠 Partial compatibility (MLA)<br>— Not applicable (GQA) | 🟠 P/D disaggregation<br>❌ PD-mixed deployment |
| 950PR&950DT Products | SFA | ❌ No compatibility | ❌ No compatibility | ❌ No compatibility | ❌ No compatibility | ❌ No compatibility | ❌ No compatibility |

- ✅ **Full compatibility**: Combining the feature with DCP is supported.
- 🟠 **Partial compatibility**: Combining the feature with DCP is experimentally supported; interfaces and functionality may change.
- ❌ **No compatibility**: Combining the feature with DCP is not supported.
- **Not applicable**: The feature does not apply to this attention backend.

DSA-CP supports prefix caching, chunked prefill, speculative decoding, P/D disaggregation on the model and hardware combinations documented by vLLM Ascend.

## Usage

### Prefill Context Parallel

Enable ModelRunner V2 and set `prefill_context_parallel_size` to the number of PCP ranks:

```bash
export VLLM_USE_V2_MODEL_RUNNER=1

vllm serve <supported-model> \
    --tensor-parallel-size <tp-size> \
    --prefill-context-parallel-size <pcp-size> \
    --enforce-eager
```

Unlike DCP, PCP adds extra ranks: `world_size_with_pcp = prefill_context_parallel_size * original_world_size`.

#### Decode Request Sharding

With the paired upstream PCP decode-sharding implementation, `--enable-pcp-decode-sharding` defaults to enabled. MRV2 assigns each decode request to one PCP rank when PCP > 1 and DCP = 1; the KV cache remains replicated. Setting the flag does not enable sharding for PCP = 1 or DCP > 1.

Use `--no-enable-pcp-decode-sharding` to restore replicated decode without changing PCP, DCP, TP, or the number of processes. The Python `EngineArgs`/`LLM` equivalent is `enable_pcp_decode_sharding=False`. The internal `pcp_shard_decode_requests` property combines this option with the topology constraints.

For example, keep this configuration fixed and change only the final flag between runs:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve <deepseek-model-path> \
    --tensor-parallel-size 1 \
    --prefill-context-parallel-size 2 \
    --decode-context-parallel-size 1 \
    --enable-expert-parallel --async-scheduling --enforce-eager \
    --enable-pcp-decode-sharding
# Replicated-decode control: replace the last flag with
# --no-enable-pcp-decode-sharding
```

The initial Ascend sharded-decode scope is eager, non-hybrid DeepSeek V2/V3/V3.2 MLA/SFA with RoPE; dense MLA requires unquantized KV. Graph execution, speculative decoding, KVPP and PCP O-proj weight sharding are rejected while decode sharding is enabled. Disable decode sharding to use the existing replicated-decode feature combinations documented in this guide.

For performance comparisons, keep request lengths, output lengths, concurrency and warmup identical, with EP and asynchronous scheduling enabled in both runs. Sharded decode disables fused MLA preprocessing. To isolate the cost of request sharding from that preprocessing change, also use `--additional-config '{"enable_mlapo":false}'` in both runs.

#### Speculative Decoding

MRV2 PCP supports MTP with MLA and DSA models, Eagle3 with GQA models, and
DSpark with DeepSeek-V4 DSA models. The target model runs with the
configured PCP topology, while the draft model is replicated on every PCP rank
and runs with a logical PCP size of `1`. Configure PCP only for the target
model.

For general speculative decoding configuration and model requirements, see [Speculative Decoding](speculative_decoding.md).

##### MTP with MLA

```bash
export VLLM_USE_V2_MODEL_RUNNER=1

vllm serve <mtp-capable-mla-model> \
    --tensor-parallel-size 2 \
    --prefill-context-parallel-size 2 \
    --enable-chunked-prefill \
    --enforce-eager \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}'
```

##### Eagle3 with GQA

```bash
export VLLM_USE_V2_MODEL_RUNNER=1

vllm serve <gqa-target-model> \
    --tensor-parallel-size 2 \
    --prefill-context-parallel-size 2 \
    --enable-chunked-prefill \
    --enforce-eager \
    --speculative-config '{"method": "eagle3", "model": "<eagle3-draft-model>", "num_speculative_tokens": 3}'
```

For either method, remove `--enforce-eager` and add the following option to use the supported graph mode:

```bash
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
```

#### Constraints

- PCP is supported only with ModelRunner V2.
- In P/D disaggregation, enable PCP only on the prefill (`kv_producer`) engine; the decode (`kv_consumer`) engine must use `prefill_context_parallel_size=1`.
- KV cache pooling with PCP supports only `AscendStoreConnector` with `use_layerwise=false`.
- PCP speculative decoding supports MTP with MLA and DSA models, Eagle3 with
  GQA models, and DSpark with DeepSeek-V4 DSA models.
- Draft sampling must use the greedy method.
- Full graph execution with PCP is limited to `FULL_DECODE_ONLY`.
- Pipeline parallelism, encoder-decoder models, multimodal inputs, and LoRA are not supported with MRV2 PCP.
- SFA draft attention is not supported with PCP speculative decoding.
- PCP and DCP cannot be enabled simultaneously.
- Adaptive verification is not supported with PCP speculative decoding.
- Dynamic draft lengths are outside the currently validated scope.
- PCP and [DSA-CP](#dsa-cp) cannot be enabled simultaneously with the DSA backend.

### Decode Context Parallel

```bash
vllm serve <glm-5.2-model> \
  --tensor-parallel-size <N> \
  --prefill-context-parallel-size 1 \
  --decode-context-parallel-size <N> \
  --block-size <B> \
  --cp-kv-cache-interleave-size <B>
```

DCP reuses the TP devices and does not increase the world size.

#### Constraints

- For an MLA model such as DeepSeek-R1:
    - `tensor_parallel_size >= decode_context_parallel_size`
    - `tensor_parallel_size % decode_context_parallel_size == 0`
- For a GQA model such as Qwen3-235B:
    - `(tensor_parallel_size // num_key_value_heads) >= decode_context_parallel_size`
    - `(tensor_parallel_size // num_key_value_heads) % decode_context_parallel_size == 0`
- In a KV-cache transfer scenario such as KV pooling or P/D disaggregation, set `cp_kv_cache_interleave_size` to the KV-cache `block_size` (default: 128):

    ```shell
    vllm serve deepseek-ai/DeepSeek-V2-Lite \
        --tensor-parallel-size 2 \
        --decode-context-parallel-size 2 \
        --cp-kv-cache-interleave-size 128 \
        --kv-transfer-config '{...}'
    ```

### DSA-CP

For a TP-only comparison with PCP, enable expert parallelism and use
`allgather_reducescatter` (the default EP backend). Ascend supports sequence-parallel
MoE with DP=1, TP>1 and PCP=1; DSA-CP automatically enables FlashComm1. With neither
DSA-CP nor FlashComm1 requested, sequence-parallel MoE remains disabled.

DSA-CP will be fully deprecated once PCP is ready. PCP is currently experimental,
with support for some feature combinations still in progress.

To try PCP with the same world size, replace TP size `N > 1` with
`--tensor-parallel-size 1 --prefill-context-parallel-size N` and remove
`enable_dsa_cp` from `additional_config`. With TP size 1, PCP requires additional
ranks. Check the compatibility and limitations above before migrating.

```bash
vllm serve <glm-5.2-model> \
  --tensor-parallel-size <N> \
  --enable-expert-parallel \
  --block-size <B> \
  --additional-config '{"enable_dsa_cp": true}'
```

For implementation details, see the [Context Parallel design document](../../developer_guide/Design_Documents/context_parallel.md).
