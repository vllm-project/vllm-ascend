# Suffix Speculative Decoding

## **Introduction**

Suffix Decoding proposes draft tokens by matching patterns in the prompt and generated content, without an additional draft model. Model Runner V1 uses the CPU-based Arctic Inference implementation. The experimental Model Runner V2 implementation performs matching and cross-request retrieval on the NPU; see [Model Runner V2](#model-runner-v2-experimental).

The benchmark results and deployment instructions below describe the V1 implementation. They are not performance measurements of the experimental V2 path.

## Model Runner V2 (experimental)

This path adapts upstream [vLLM #60339](https://github.com/vllm-project/vllm/pull/60339)
to Ascend. It requires a vLLM version containing the complete upstream change
(merge `190df0b90007eead441d97d3abf6270f84f2c63e` or a descendant), including
suffix configuration and the `BaseSpeculator` request lifecycle hooks. The
repository's older verified pin does not contain these interfaces. This change
does not update that pin or backport the upstream runner protocol.

Suffix decoding uses V2 by default unless another unsupported configuration
requires V1. Set `VLLM_USE_V2_MODEL_RUNNER=0` to explicitly use the legacy V1
implementation. A V2 configuration with an older upstream version raises an
actionable dependency error instead of silently falling back to V1.
NPU generation, async scheduling, TP consistency and throughput qualification
are still pending. On a compatible upstream version:

```bash
vllm serve /data/Qwen3-8B \
  --async-scheduling \
  --speculative-config '{"method":"suffix","num_speculative_tokens":8,"suffix_decoding_corpus_tokens":1048576}'
```

Arctic Inference is not used by the V2 speculator. Matching works as follows:

- Each request searches its own prompt and generated tokens for the longest
  matching suffix, up to `suffix_decoding_max_tree_depth` (default 24).
- Completed responses, with up to three preceding prompt tokens, are added to
  an NPU-resident ring. Two hash indexes use the last two and four tokens to
  locate candidates. Candidate matches are extended and their continuations
  are selected by prefix-consistent frequency voting. The longer history or
  corpus match wins; history wins ties.
- Finished requests are ingested before their state slots are released.
  Preempted requests are not ingested. Sorted request IDs and integer index
  updates preserve the same corpus state across TP ranks. Upstream warmup
  resets the corpus and acceptance statistics before serving.

The default corpus capacity is 1,048,576 tokens. Setting
`suffix_decoding_corpus_tokens: 0` or `suffix_decoding_max_cached_requests: 0`
disables cross-request retrieval. On V2 the latter is only a disable switch;
positive values do not limit the number of cached requests. Corpus matching
needs a maximum match depth of at least two; depth one uses history only.
Default corpus and index storage consume approximately 36 MiB, in addition to
dense token history (`max_num_seqs * max_model_len * 4` bytes) and scratch buffers.

V2 always returns `num_speculative_tokens` draft slots. Positions without a
continuation contain the last sampled token and are verified as ordinary
drafts. Upstream metrics include these filler slots; they are not a measure
of the number of genuine suffix matches. `suffix_decoding_max_spec_factor`
and `suffix_decoding_min_token_prob` apply only to V1.

Optional `enable_adaptive_verification: true` supplies per-position confidence
from online acceptance statistics grouped by draft source, match length and
position. It requires full decode graphs and an Ascend target attention backend
that supports variable query lengths. It is disabled by default. The NPU
observer reduces batch-local integer counts per statistics bucket rather than
using the GPU observer's contended int64 atomic additions. This implementation
has not yet been tested on NPU; backend compatibility and throughput remain
to be qualified. At high concurrency, verifying every draft can cost more than
the accepted tokens save.

The corpus is shared across requests, including different `cache_salt` values
and LoRA adapters. Disable the corpus where requests must not influence each
other's drafting. As upstream documents, a fixed sampling seed alone does not
make outputs reproducible with a shared corpus or adaptive verification. For
that use case, disable both. Under async scheduling, a finished request may
contribute tokens beyond the scheduler's final stop point; ingestion follows
upstream device-visible lengths.

The Ascend speculator inherits upstream `SuffixSpeculator` initialization,
slot bookkeeping, reset and finished-response ingestion. Its proposal wrapper
only guards empty batches before calling `super()`. NPU launch wrappers in
`ops/triton/spec_decode/suffix.py` own stride, grid and compilation-parameter
handling. The worker's `patch_triton.py` installs the four launch adapters at
startup; they remain bound for that worker process, including direct upstream
suffix calls. Per-call replacement, restoration and locking are unnecessary.
Older runtimes without the suffix module skip registration; selecting MRV2
suffix still requires its complete upstream protocol. Tests that directly
instantiate the speculator install the same worker bindings explicitly.
Remove these rebinds when upstream supports device kernel dispatch.

## Model Runner V1 deployment and benchmarks

This document provides step-by-step guidance on how to deploy and benchmark the Suffix Decoding speculative inference technology supported by `vllm-ascend` on Atlas A2 hardware. The setup utilizes a single Atlas 800T A2 node with a 4-card deployment of the Qwen3-32B model instance. Benchmarking is conducted using authentic open-source datasets covering the following categories:

| **Dataset Category**           | **Dataset Name** |
| ------------------------------ | ---------------- |
| Code Generation                | HumanEval        |
| Common Sense Reasoning         | ARC              |
| Mathematical Reasoning         | gsm8k            |
| Natural Language Understanding | SuperGLUE_BoolQ  |
| Comprehensive Examination      | AGIEval          |
| Multi-turn Dialogue            | ShareGPT         |

The benchmarking tool used in this tutorial is AISBench, which supports performance testing for all the datasets listed above. The final section of this tutorial presents a performance comparison between enabling and disabling Suffix Decoding under the condition of satisfying an SLO TPOT < 50ms across different datasets and concurrency levels. Validations demonstrate that the Qwen3-32B model achieves a throughput improvement of approximately 20% to 80% on various real-world datasets when Suffix Decoding is enabled.

## **Download vllm-ascend Image**

This tutorial uses the official image, version `{{ vllm_ascend_version }}`. Use the following command to download:

```bash

docker pull quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}
```

## **Run with Docker**

Container startup command:

```bash

# Update the vllm-ascend image
export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}
export NAME=vllm-ascend

# Run the container using the defined variables
# This test uses four Atlas A2 NPU cards to create the container.
# Mount the hccn.conf file from the host node into the container.

docker run --rm \
--name $NAME \
--net=host \
--shm-size=1g \
--device /dev/davinci0 \
--device /dev/davinci1 \
--device /dev/davinci2 \
--device /dev/davinci3 \
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--device /dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/Ascend/driver/tools/hccn_tool:\
/usr/local/Ascend/driver/tools/hccn_tool \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /etc/hccn.conf:/etc/hccn.conf \
-v /root/.cache:/root/.cache \
-it $IMAGE bash
```

## **Install arctic-inference**

Before enabling Suffix Decoding speculative inference on Ascend, the Arctic Inference plugin must be installed. Arctic Inference is an open-source plugin launched by Snowflake specifically to optimize LLM inference speed. For detailed technical principles, please refer to the following article: [Fastest Speculative Decoding in vLLM with Arctic Inference and Arctic Training](https://www.snowflake.com/en/engineering-blog/fast-speculative-decoding-vllm-arctic/). Install it within the container using the following command:

```bash
pip install arctic-inference
```

## **vLLM Instance Deployment**

Use the following command to start the container service instance. Speculative inference is enabled via the `--speculative-config` parameter, where `method` is set to `suffix`. For this test, `num_speculative_tokens` is uniformly set to `3`.

```bash
# set the NPU device number:
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
# Set the operator dispatch pipeline level to 1 and disable manual memory control in ACLGraph
export TASK_QUEUE_ENABLE=1
# Enable the AIVector core to directly schedule ROCE communication.
export HCCL_OP_EXPANSION_MODE="AIV"
# This section documents the legacy CPU suffix implementation.
export VLLM_USE_V2_MODEL_RUNNER=0
vllm serve /data/Qwen3-32B \
  --served-model-name qwen3 \
  --trust-remote-code \
  --distributed-executor-backend mp \
  --tensor-parallel-size 4 \
  --max-model-len 5500 \
  --max-num-batched-tokens 40960 \
  --speculative-config '{"method": "suffix", "num_speculative_tokens": 3}' \
  --gpu-memory-utilization 0.9 \
  --additional-config '{"pa_shape_list":[48,64,72,80], "weight_prefetch_config":{"enable":true}}' \
  --port 8011
```

## **AISbench Benchmark Testing**

Performance for all open-source datasets is tested using AISbench. For specific instructions, refer to [Using AISBench for performance evaluation](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/evaluation/using_ais_bench.html#execute-performance-evaluation).

**Model Configuration**:

```bash
# "ignore_eos" must be set to "False", and "max_out_len" should be set to a large value to allow the model to output completely and naturally.

from ais_bench.benchmark.models import VLLMCustomAPIChatStream

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChatStream,
        abbr='vllm-api-stream-chat',
        path="<path_to_your_model>/Qwen3-32B",
        model="qwen3",
        request_rate = 0,
        retry = 2,
        host_ip = "<your_server_ip>",
        host_port = 8011,
        max_out_len = 4000,
        batch_size= 16,
        trust_remote_code=False,
        generation_kwargs = dict(
            temperature = 0,
            ignore_eos = False
        )
    )
]
```

**Performance Benchmarking Commands**:

```bash
# Example command to test gsm8k dataset performance using the first 100 prompts. Commands for other datasets are similar.
ais_bench --models vllm-api-stream-chat \
  --datasets gsm8k_gen_0_shot_cot_str_perf \
  --debug --summarizer default_perf --mode perf --num-prompts 100
```

## **Test Results**

Below are the detailed test results of the six open-source datasets in this evaluation. Compared to the baseline, the improvement in TPOT and throughput performance at different concurrency levels after enabling Suffix Decoding varies across datasets. Below is a summary of the results:

| **Dataset Category** | **Typical Representative** | **Throughput Improvement (BS=1-10)** | **SLO TPOT** |
| -------------------- | -------------------------- | ------------------------------------ | ------------ |
| **High Gain**        | AGIEval, GSM8K             | **> 50%**                            | < 50ms       |
| **Medium-Low Gain**  | ARC, ShareGPT              | **20% ~ 30%**                        | < 50ms       |

Below is the raw detailed test results:

| Concurrency         | Avg Input | Avg Output | Requests | Base TPOT(ms) | Base Throughput(TPS) | Suffix TPOT(ms) | Suffix Throughput(TPS) | Accept Rate | TPOT Gain | TPS Gain |
| ------------------- | --------- | ---------- | -------- | ------------- | -------------------- | --------------- | ---------------------- | ----------- | --------- | -------- |
| **HumanEval**       |           |            |          |               |                      |                 |                        |             |           |          |
| 1                   | 150       | 2700       | 100      | 55.1          | 18.1                 | 37.9            | 26.3                   | 27.0%       | 45.2%     | 45.1%    |
| 15                  | 150       | 2700       | 100      | 61.6          | 233.8                | 45.8            | 318.2                  | 27.0%       | 34.6%     | 36.1%    |
| 26                  | 150       | 2700       | 100      | 64.7          | 403.8                | 50.9            | 519.2                  | 27.0%       | 27.2%     | 28.6%    |
| **ARC**             |           |            |          |               |                      |                 |                        |             |           |          |
| 1                   | 76        | 960        | 100      | 52.8          | 18.9                 | 39.5            | 25.4                   | 23.9%       | 33.7%     | 34.6%    |
| 8                   | 76        | 960        | 100      | 59.1          | 125.4                | 47.0            | 163.1                  | 23.9%       | 25.7%     | 30.0%    |
| 15                  | 76        | 960        | 100      | 59.8          | 245.8                | 48.9            | 311.7                  | 23.9%       | 22.3%     | 26.8%    |
| **GSM8K**           |           |            |          |               |                      |                 |                        |             |           |          |
| 1                   | 67        | 1570       | 100      | 55.5          | 18.0                 | 35.7            | 28.5                   | 31.1%       | 55.6%     | 58.4%    |
| 17                  | 67        | 1570       | 100      | 61.5          | 279.8                | 45.4            | 403.0                  | 31.1%       | 35.6%     | 44.0%    |
| 26                  | 67        | 1570       | 100      | 63.9          | 396.4                | 50.0            | 527.6                  | 31.1%       | 27.8%     | 33.1%    |
| **ShareGPT**        |           |            |          |               |                      |                 |                        |             |           |          |
| 1                   | 666       | 231        | 327      | 54.1          | 18.3                 | 39.2            | 24.1                   | 23.9%       | 37.9%     | 31.5%    |
| 8                   | 666       | 231        | 327      | 58.8          | 125.0                | 46.2            | 153.2                  | 23.9%       | 27.1%     | 22.5%    |
| 14                  | 666       | 231        | 327      | 61.8          | 227.0                | 49.9            | 273.9                  | 23.9%       | 23.8%     | 20.7%    |
| **SuperGLUE_BoolQ** |           |            |          |               |                      |                 |                        |             |           |          |
| 1                   | 207       | 314        | 100      | 54.1          | 18.4                 | 36.1            | 26.8                   | 33.4%       | 49.8%     | 45.6%    |
| 16                  | 207       | 314        | 100      | 60.0          | 229.7                | 43.5            | 303.9                  | 33.4%       | 38.0%     | 32.3%    |
| 32                  | 207       | 314        | 100      | 62.7          | 396.4                | 47.8            | 507.5                  | 33.4%       | 31.3%     | 28.0%    |
| **AGIEval**         |           |            |          |               |                      |                 |                        |             |           |          |
| 1                   | 735       | 1880       | 100      | 53.1          | 18.7                 | 31.8            | 34.1                   | 50.3%       | 66.8%     | 81.9%    |
| 24                  | 735       | 1880       | 100      | 64.0          | 381.2                | 43.3            | 629.0                  | 50.3%       | 47.8%     | 65.0%    |
| 34                  | 735       | 1880       | 100      | 70.0          | 494.6                | 50.2            | 768.4                  | 50.3%       | 39.4%     | 55.3%    |
