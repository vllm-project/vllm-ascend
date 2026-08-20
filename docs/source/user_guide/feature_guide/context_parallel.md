# Context Parallel Guide

## Overview

Context Parallel (CP) serves long-context requests by splitting work or KV-cache storage along the sequence dimension:

- Prefill Context Parallel (PCP) splits the new tokens of a long prefill request across additional ranks. Each rank computes a different part of the sequence, reducing time to first token (TTFT).
- Decode Context Parallel (DCP) shards the KV cache across ranks in an existing Tensor Parallel (TP) group. It reduces duplicated KV-cache storage and can increase decode throughput.

For a general introduction to these two strategies, see the upstream [vLLM Context Parallel Deployment](https://docs.vllm.ai/en/latest/serving/context_parallel_deployment/) guide.

DSA-CP is a separate sparse-attention optimization controlled by `additional_config.enable_dsa_cp`. It does not use the PCP process layout. See [Additional Configuration](../configuration/additional_config.md) for its configuration and model requirements.

## Supported Scenarios

### Prefill Context Parallel

PCP support is experimental and available only with ModelRunner V2. The initial implementation supports the following attention backends:

| Attention Backend | Status | Notes |
| --- | --- | --- |
| GQA | Experimental | Basic unquantized eager-mode prefill |
| DeepSeek-V4 DSA | Experimental | Eager-mode support introduced by [#14037](https://github.com/vllm-project/vllm-ascend/pull/14037) |

The DeepSeek-V4 DSA support documented here depends on #14037. Other attention backends and feature combinations are not covered by this initial MRV2 PCP path.

### Decode Context Parallel

DCP supports eager and graph execution, prefix caching, chunked prefill, speculative decoding, P/D disaggregation, and MLAPO on the model and hardware combinations documented by vLLM Ascend. The following table shows whether each feature can be combined with DCP across devices and attention backends:

| Device | Attention Backend | Chunked Prefill + DCP | Prefix Caching + DCP | Graph Mode + DCP | P/D Disaggregation + DCP | MLAPO + DCP | Speculative Decoding + DCP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Ascend A2/A3 | MLA/GQA | 🟢 Supported | 🟢 Supported | 🟢 Supported | 🟢 Supported | 🟢 Supported (MLA)<br>— Not applicable (GQA) | 🟢 P/D disaggregation<br>🔴 PD-mixed deployment |
| Ascend A2/A3 | SFA | 🟢 Supported | 🟢 Supported | 🟢 Supported | 🟢 Supported | 🟢 Supported | 🟢 Supported |
| Ascend 950 | MLA/GQA | 🔵 Experimental | 🔵 Experimental | 🔵 Experimental | 🔵 Experimental | 🔵 Experimental (MLA)<br>— Not applicable (GQA) | 🔵 P/D disaggregation<br>🔴 PD-mixed deployment |
| Ascend 950 | SFA | 🔴 Not supported | 🔴 Not supported | 🔴 Not supported | 🔴 Not supported | 🔴 Not supported | 🔴 Not supported |

- 🟢 **Supported**: Combining the feature with DCP is supported.
- 🔵 **Experimental**: Combining the feature with DCP is experimentally supported; interfaces and functionality may change.
- 🔴 **Not supported**: Combining the feature with DCP is not supported.
- **Not applicable**: The feature does not apply to this attention backend.

## Prefill Context Parallel Usage

Enable ModelRunner V2 and set `prefill_context_parallel_size` to the number of PCP ranks:

```bash
export VLLM_USE_V2_MODEL_RUNNER=1

vllm serve <supported-model> \
    --tensor-parallel-size <tp-size> \
    --prefill-context-parallel-size <pcp-size> \
    --enforce-eager
```

Unlike DCP, PCP adds ranks to the process world. With pipeline parallelism disabled, the process world size is `tensor_parallel_size * prefill_context_parallel_size`.

For a DeepSeek-V4 DSA model, setting `prefill_context_parallel_size` selects the DSA PCP backend automatically. Do not also set `additional_config.enable_dsa_cp`.

### PCP Constraints

- PCP and DCP cannot be enabled simultaneously on Ascend MRV2.
- Pipeline parallelism must remain disabled (`pipeline_parallel_size=1`).
- Encoder-decoder models, multimodal inputs, and LoRA are not supported.
- Use eager mode for the initial GQA and DeepSeek-V4 DSA implementations.
- DeepSeek-V4 DSA PCP and legacy DSA-CP (`additional_config.enable_dsa_cp`) are mutually exclusive.

## Decode Context Parallel Usage

Offline example:

```python
from vllm import LLM, SamplingParams

prompts = ["The future of AI is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="deepseek-ai/DeepSeek-V2-Lite",
    tensor_parallel_size=2,
    decode_context_parallel_size=2,
)
outputs = llm.generate(prompts, sampling_params)
```

Online example:

```bash
vllm serve deepseek-ai/DeepSeek-V2-Lite \
    --tensor-parallel-size 2 \
    --decode-context-parallel-size 2
```

DCP reuses the TP devices and does not increase the world size.

### DCP Constraints

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

For implementation details, see the [Context Parallel design document](../../developer_guide/Design_Documents/context_parallel.md).
