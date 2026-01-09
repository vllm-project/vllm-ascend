# Context Parallel Guide

## Overview

This guide shows how to use Context Parallel, a long sequence inference optimization technique. Context Parallel includes `PCP` (Prefill Context Parallel) and `DCP` (Decode Context Parallel), which reduces NPU memory usage and improves inference speed in long sequence LLM inference.

## Benefits of Context Parallel
Context parallel mainly solves the problem of serving long context requests. As prefill and decode present quite different characteristics and have quite different SLO (service level objectives), we need to implement context parallel separately for them. The major considerations are:

- For long context prefill, we can use context parallel to reduce TTFT (time to first token) by amortizing the computation time of the prefill across query tokens.
- For long context decode, we can use context parallel to reduce KV cache duplication and offer more space for KV cache to increase the batchsize (and hence the throughput).

To learn more about the theory and implementation details of context parallel, please refer to the [context parallel developer guide](../../developer_guide/feature_guide/context_parallel.md).

## Supported Scenarios
Currently context parallel can be used together with most other features, supported features are as follows:
|         | Eager | Graph | Prefix <br> Cache | Chunked <br> Prefill | SpecDecode <br> (MTP) | PD <br> disaggregation | MLAPO |
| ------- | ----- | ----- | ------ | ------ | ----- | ----- | ----- |
| **PCP** | ✅    | ✅     | ✅      | ✅       | ✅      | ✅ | ✅|
| **DCP** | ✅    | ✅     | ✅      | ✅       | ✅      | ✅ | ✅ |

## How to use Context Parallel
You can enable `PCP` and `DCP` by `prefill_context_parallel_size` and `decode_context_parallel_size`, refer to the following example:

- Offline example:

    ```python
    from vllm import LLM, SamplingParams

    prompts = [
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    llm = LLM(
        model="deepseek-ai/DeepSeek-V2-Lite",
        tensor_parallel_size=2,
        decode_context_parallel_size=2,
        prefill_context_parallel_size=2,
    )
    outputs = llm.generate(prompts, sampling_params)
    ```

- Online example:

    ```bash
    vllm serve deepseek-ai/DeepSeek-V2-Lite \
        --tensor-parallel-size 2 \
        --decode-context-parallel-size 2 \
        --prefill-context-parallel-size 2 \
    ```

The total world_size is `tensor_parallel_size` * `prefill_context_parallel_size`, so the examples above need 4 NPUs for each.

## Constraints
- While using DCP, the following constraints must be met:
    - For MLA based model, such as Deepseek-R1:
        - `tensor_parallel_size >= decode_context_parallel_size`
        - `tensor_parallel_size % decode_context_parallel_size == 0`
    - For GQA based model, such as Qwen3-235B:
        - `(tensor_parallel_size // num_key_value_heads) >= decode_context_parallel_size`
        - `(tensor_parallel_size // num_key_value_heads) % decode_context_parallel_size == 0`

- While using Context Parallel in KV cache transfer needed scenario (e.g. KV pooling, PD-disaggregation), to simplify KV cache transmission, `cp_kv_cache_interleave_size` must be set to the same value of KV cache `block_size`(default: 128), which specify cp to split KV cache in a block-interleave style. For example:

    ```
    vllm serve deepseek-ai/DeepSeek-V2-Lite \
        --tensor-parallel-size 2 \
        --decode-context-parallel-size 2 \
        --prefill-context-parallel-size 2 \
        --cp-kv-cache-interleave-size 128 \
        --kv-transfer-config {...} \
    ```

## Experimental Results
To evaluate the effectiveness of Context Parallel in in long sequence LLM inference scenarios, we use **DeepSeek-R1-W8A8** and **Qwen3-235B**, deploy PD-disaggregate instances in the environment of 64 cards Ascend 910C*64G (A3), the configuration and performance data are as follows.

- DeepSeek-R1-W8A8:
    | Configuration | Input length <br> 32k | Input length <br> 64k | Input length <br> 128k |
    | ----------------------------- | ------------------------- | ------------------------- | ------------------------- |
    | P node: (DP2 TP8 EP16) *2 <br> D node: (DP32 EP32) *1       | TTFT: 9.3s <br> TPOT: 72ms | TTFT: 22.8s <br> TPOT: 74ms | TTFT: 73.2s <br> TPOT: 82ms |
    | P node: (PCP2 TP8 DCP8 EP16) *2 <br> D node: (DP32 EP32) *1 | TTFT: 7.9s <br> TPOT: 74ms | TTFT: 15.9s <br> TPOT: 78ms | TTFT: 46.0s <br> TPOT: 83ms |

- Qwen3-235B:
    | Configuration | Input length <br> 32k | Input length <br> 64k | Input length <br> 120k |
    | ----------------------------- | ------------------------- | ------------------------- | ------------------------- |
    | P node: (DP2 TP8 EP16) *2 <br> D node: (DP32 EP32) *1       | TTFT: 5.1s <br> TPOT: 65ms | TTFT: 13.1s <br> TPOT: 85ms | TTFT: 33.9s <br> TPOT: 120ms |
    | P node: (PCP2 TP8 DCP2 EP16) *2 <br> D node: (DP32 EP32) *1 | TTFT: 3.0s <br> TPOT: 66ms | TTFT: 8.9s <br> TPOT: 86ms | TTFT: 22.7s <br> TPOT: 121ms |
