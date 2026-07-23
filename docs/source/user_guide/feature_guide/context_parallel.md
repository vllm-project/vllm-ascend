# Decode Context Parallel Guide

## Overview

Decode Context Parallel (DCP) shards the KV cache along the sequence dimension across devices in a Tensor Parallel (TP) group. It removes redundant KV-cache copies and can increase the batch size available for long-context decoding.

Prefill Context Parallel is not supported by vLLM Ascend. The upstream `prefill_context_parallel_size` option must remain at its default value of `1`.

DSA-CP is a separate sparse-attention optimization controlled by `additional_config.enable_dsa_cp`. See [Additional Configuration](../configuration/additional_config.md) for its configuration and model requirements.

## Supported Scenarios

DCP supports eager and graph execution, prefix caching, chunked prefill, speculative decoding, P/D disaggregation, and MLAPO on the model and hardware combinations documented by vLLM Ascend. The SFA attention backend supports speculative decoding. In the MLA and GQA attention backend, speculative decoding is supported in the P/D disaggregation deployment scenario and not supported in the mixed deployment scenario.

## Usage

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

## Constraints

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

For implementation details, see the [Decode Context Parallel design document](../../developer_guide/Design_Documents/context_parallel.md).