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

| Attention Backend | Basic PCP | Prefix Caching + PCP | Chunked Prefill + PCP | MLAPO + PCP | Speculative Decoding + PCP | P/D Disaggregation + PCP | Sequence Parallelism (SP) + PCP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MLA | ✅ Full compatibility | ✅ Full compatibility | ✅ Full compatibility | ✅ Full compatibility | 🟠 Partial compatibility (MTP, eager and `FULL_DECODE_ONLY`) | ❌ No compatibility | ❌ No compatibility |
| GQA | ✅ Full compatibility | ✅ Full compatibility | ✅ Full compatibility | — Not applicable | 🟠 Partial compatibility (Eagle3, eager and `FULL_DECODE_ONLY`) | ❌ No compatibility | ❌ No compatibility |
| SFA | ✅ Full compatibility | ✅ Full compatibility | ✅ Full compatibility | ❌ No compatibility | ❌ No compatibility | ❌ No compatibility | ❌ No compatibility |
| DSA | ✅ Full compatibility | ✅ Full compatibility | ✅ Full compatibility | — Not applicable | 🟠 Partial compatibility (MTP, eager and `FULL_DECODE_ONLY`) | ❌ No compatibility | ❌ No compatibility |

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
| Ascend 950 | MLA/GQA | 🟠 Partial compatibility | 🟠 Partial compatibility | 🟠 Partial compatibility | 🟠 Partial compatibility | 🟠 Partial compatibility (MLA)<br>— Not applicable (GQA) | 🟠 P/D disaggregation<br>❌ PD-mixed deployment |
| Ascend 950 | SFA | ❌ No compatibility | ❌ No compatibility | ❌ No compatibility | ❌ No compatibility | ❌ No compatibility | ❌ No compatibility |

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

#### Speculative Decoding

MRV2 PCP supports MTP with MLA models and Eagle3 with GQA models. The target model runs with the configured PCP topology, while the draft model is replicated on every PCP rank and runs with a logical PCP size of `1`. Configure PCP only for the target model.

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
- PCP speculative decoding supports only MTP with MLA models and Eagle3 with GQA models.
- Draft sampling must use the greedy method.
- Full graph execution with PCP is limited to `FULL_DECODE_ONLY`.
- Pipeline parallelism, encoder-decoder models, multimodal inputs, and LoRA are not supported with MRV2 PCP.
- DSA and SFA draft attention are not supported with PCP speculative decoding.
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

```bash
vllm serve <glm-5.2-model> \
  --tensor-parallel-size <N> \
  --block-size <B> \
  --additional-config '{"enable_dsa_cp": true}'
```

For implementation details, see the [Context Parallel design document](../../developer_guide/Design_Documents/context_parallel.md).
