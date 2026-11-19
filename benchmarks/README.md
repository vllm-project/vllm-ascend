# vLLM Ascend Benchmarks

## Introduction

This document outlines the benchmarking methodology for vllm-ascend, aimed at evaluating the performance under a variety of workloads. The primary goal is to help developers assess whether their pull requests improve or degrade vllm-ascend's performance.

## Overview

**Benchmarking Coverage**: We measure latency, throughput, and fixed-QPS serving on the Atlas 800I A2 (see [quick_start](../docs/source/getting_started/quick_start.md) to learn more supported devices list), with different models(coming soon).

- Latency tests
    - Input length: 32 tokens.
    - Output length: 128 tokens.
    - Batch size: fixed (8).
    - Models: Qwen2.5-7B-Instruct, Qwen3-8B.
    - Evaluation metrics: end-to-end latency (mean, median, p99).

- Throughput tests
    - Input length: randomly sample 200 prompts from ShareGPT dataset (with fixed random seed).
    - Output length: the corresponding output length of these 200 prompts.
    - Batch size: dynamically determined by vllm to achieve maximum throughput.
    - Models: Qwen2.5-VL-7B-Instruct, Qwen2.5-7B-Instruct, Qwen3-8B.
    - Evaluation metrics: throughput.
- Serving tests
    - Input length: randomly sample 200 prompts from ShareGPT dataset (with fixed random seed).
    - Output length: the corresponding output length of these 200 prompts.
    - Batch size: dynamically determined by vllm and the arrival pattern of the requests.
    - **Average QPS (query per second)**: 1, 4, 16 and inf. QPS = inf means all requests come at once. For other QPS values, the arrival time of each query is determined using a random Poisson process (with fixed random seed).
    - Models: Qwen2.5-VL-7B-Instruct, Qwen2.5-7B-Instruct, Qwen3-8B.
    - Evaluation metrics: throughput, TTFT (time to the first token, with mean, median and p99), ITL (inter-token latency, with mean, median and p99).

**Benchmarking Duration**: about 800 seconds for single model.

## Quick Use

### Prerequisites

Before running the benchmarks, ensure the following:

- vllm and vllm-ascend are installed and properly set up in an NPU environment, as these scripts are specifically designed for NPU devices.

- Install necessary dependencies for benchmarks:
  
  ```shell
  pip install -r benchmarks/requirements-bench.txt
  ```
  
- For performance benchmark, it is recommended to set the [load-format](https://github.com/vllm-project/vllm-ascend/blob/5897dc5bbe321ca90c26225d0d70bff24061d04b/benchmarks/tests/latency-tests.json#L7) as `dummy`, It will construct random weights based on the passed model without downloading the weights from internet, which can greatly reduce the benchmark time.
- If you want to run a customized benchmark, feel free to add your own models and parameters in the [JSON](https://github.com/vllm-project/vllm-ascend/tree/main/benchmarks/tests), let's take `Qwen2.5-VL-7B-Instruct`as an example:

  ```json
  [
  {
    "test_name": "serving_qwen2_5vl_7B_tp1",
    "qps_list": [
      1,
      4,
      16,
      "inf"
    ],
    "server_parameters": {
      "model": "Qwen/Qwen2.5-VL-7B-Instruct",
      "tensor_parallel_size": 1,
      "swap_space": 16,
      "disable_log_stats": "",
      "disable_log_requests": "",
      "trust_remote_code": "",
      "max_model_len": 16384
    },
    "client_parameters": {
      "model": "Qwen/Qwen2.5-VL-7B-Instruct",
      "backend": "openai-chat",
      "dataset_name": "hf",
      "hf_split": "train",
      "endpoint": "/v1/chat/completions",
      "dataset_path": "lmarena-ai/vision-arena-bench-v0.1",
      "num_prompts": 200
    }
  }
  ]
  ```
  
this Json will be structured and parsed into server parameters and client parameters by the benchmark script. This configuration defines a test case named `serving_qwen2_5vl_7B_tp1`, designed to evaluate the performance of the `Qwen/Qwen2.5-VL-7B-Instruct` model under different request rates. The test includes both server and client parameters, for more parameters details, see vllm benchmark [cli](https://github.com/vllm-project/vllm/tree/main/vllm/benchmarks).

- **Test Overview**
    - Test Name: serving_qwen2_5vl_7B_tp1

    - Queries Per Second (QPS): The test is run at four different QPS levels: 1, 4, 16, and inf (infinite load, typically used for stress testing).

- Server Parameters
    - Model: Qwen/Qwen2.5-VL-7B-Instruct

    - Tensor Parallelism: 1 (no model parallelism is used; the model runs on a single device or node)

    - Swap Space: 16 GB (used to handle memory overflow by swapping to disk)

    - disable_log_stats: disables logging of performance statistics.

    - disable_log_requests: disables logging of individual requests.

    - Trust Remote Code: enabled (allows execution of model-specific custom code)

    - Max Model Length: 16,384 tokens (maximum context length supported by the model)

- Client Parameters

    - Model: Qwen/Qwen2.5-VL-7B-Instruct (same as the server)

    - Backend: openai-chat (suggests the client uses the OpenAI-compatible chat API format)

    - Dataset Source: Hugging Face (hf)

    - Dataset Split: train

    - Endpoint: /v1/chat/completions (the REST API endpoint to which chat requests are sent)

    - Dataset Path: lmarena-ai/vision-arena-bench-v0.1 (the benchmark dataset used for evaluation, hosted on Hugging Face)

    - Number of Prompts: 200 (the total number of prompts used during the test)

### Run benchmarks

#### Use benchmark script

The provided scripts automatically execute performance tests for serving, throughput, and latency. To start the benchmarking process, run command in the vllm-ascend root directory:

```shell
bash benchmarks/scripts/run-performance-benchmarks.sh
```

Once the script completes, you can find the results in the benchmarks/results folder. The output files may resemble the following:

```shell
.
|-- serving_qwen2_5_7Bvl_tp1_qps_1.json
|-- serving_qwen2_5_7Bvl_tp1_qps_16.json
|-- serving_qwen2_5_7Bvl_tp1_qps_4.json
|-- serving_qwen2_5_7Bvl_tp1_qps_inf.json
|-- throughput_qwen2_5_7Bvl_tp1.json
```

These files contain detailed benchmarking results for further analysis.

#### Use benchmark cli

For more flexible and customized use, benchmark cli is also provided to run online/offline benchmarks
Similarly, let's take `Qwen2.5-VL-7B-Instruct` benchmark as an example:

##### Online serving

1. Launch the server:

    ```shell
    vllm serve Qwen2.5-VL-7B-Instruct --max-model-len 16789
    ```

2. Running performance tests using cli
  
    ```shell
    vllm bench serve --model Qwen2.5-VL-7B-Instruct\
    --endpoint-type "openai-chat" --dataset-name hf \
    --hf-split train --endpoint "/v1/chat/completions" \
    --dataset-path "lmarena-ai/vision-arena-bench-v0.1" \
    --num-prompts 200 \
    --request-rate 16
    ```

##### Offline

- **Throughput**

  ```shell
  vllm bench throughput --output-json results/throughput_qwen2_5_7B_tp1.json \
  --model Qwen/Qwen2.5-7B-Instruct --tensor-parallel-size 1 --load-format dummy \
  --dataset-path /github/home/.cache/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 200 --backend vllm
  ```

- **Latency**
  
  ```shell
  vllm bench latency --output-json results/latency_qwen2_5_7B_tp1.json \
  --model Qwen/Qwen2.5-7B-Instruct --tensor-parallel-size 1 \
  --load-format dummy --num-iters-warmup 5 --num-iters 15
  ```

## KV-pool CPU metadata microbenchmark

[`benchmark_kv_pool_metadata.py`](benchmark_kv_pool_metadata.py) compares two
AscendStore implementations of scheduler request filtering and layerwise leased
GVA (global virtual address) preparation. It loads the actual source modules and
uses real NumPy arrays with an immediate mock store. It does not start a model,
contact a storage service, or execute NPU operations. Its timings cover CPU
metadata preparation, not KV transfer, serving throughput, TTFT, or a complete
scheduler step.

### Environment and usage

Use a configured vLLM/vLLM-Ascend environment with NumPy and the dependencies
needed to import the scheduler and worker modules. Both source revisions must
be compatible with the same installed vLLM and dependency versions. A standalone
Python installation is insufficient even though the measured work runs on CPU.

Run from the candidate vLLM-Ascend checkout. The following Linux shell example
uses the revision before the metadata lookup optimization as its baseline;
replace `BASELINE_REF` to compare another compatible revision:

```shell
BASELINE_REF=24a76e84394ae3427195e3715fc08e90769d398c
git worktree add --detach ../vllm-ascend-kv-pool-baseline "$BASELINE_REF"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 VLLM_TARGET_DEVICE=empty
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python3 benchmarks/benchmark_kv_pool_metadata.py \
  --baseline-root ../vllm-ascend-kv-pool-baseline \
  --candidate-root . \
  --warmup 3 --rounds 5 --iterations 3 \
  --output benchmarks/results/kv-pool-metadata.json
```

`--baseline-root`, `--candidate-root`, and `--output` are required. The positive
integer options `--warmup`, `--rounds`, and `--iterations` default to 3, 5, and 3.
Each round runs baseline, candidate, candidate, baseline (ABBA), giving 30 timed
samples per variant and scenario with the defaults. Reset and result validation
run outside the timing window; only the target method call is timed. The
scheduler's asynchronous-load callback is replaced with an ID recorder to
isolate filtering work.

### Scenarios and results

The script runs 26 scenarios:

- Eight filtering cases: 128/512/1,024/4,096 unfinished requests, with all or half
  already present in the scheduled cached-request list. This is a scheduling
  relationship, not a measured external KV-cache hit rate.
- Fourteen sparse filtering cases: 128 or 4,096 scheduled cached IDs and 0, 1, or
  4 unfinished IDs. Nonempty cases match the head or tail of the cached list, or
  remain pending because their IDs are absent from it.
- Four layerwise GVA cases: 128/512/2,048/8,192 restored blocks.

Read `cases[].baseline` and `cases[].candidate` in the output JSON for
`median_ms`, interpolated `p95_ms`, `samples`, and every raw sample in `raw_ms`.
Each case also records `verified_operations`: callback count/order fingerprints
or GVA fingerprints and store operation counts. Assertions check equivalent
results between variants and validate each timed call; there is no automatic
performance pass/fail threshold.

The JSON records Python/platform information, source roots, target-file SHA256
hashes, loaded module paths, and Git revisions when available. Use these fields
to verify which implementations were compared. Compare results within the same
environment and workload, including sparse cases that may have constant-factor
tradeoffs; local metadata speedups do not establish end-to-end serving gains.
