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

### Eager-only DeepSeek V4.1 SWA comparison

On an Ascend 910B server, run:

```bash
python benchmark_dsv41_swa_triton.py --quick --json swa-quick.json
```

This compares only the three fused versions at 1024 and 4096 rows, plus the
original RMSNorm → in-place interleaved partial RoPE → SK scatter path as the
correctness and latency reference. The frozen #21183 version uses one program
per row. The single-layer #21581 adaptation keeps its partial RoPE `gather64`,
bounded slot check and split store, with one cyclic program per vector core.
This is not a timing of #21581's multi-layer DSpark launch. The benchmark's
current candidate mirrors the production persistent kernel: it slices the
BF16 RoPE tail, uses `gather64` and split stores, and reuses weights across rows.
Both cyclic candidates launch one program per vector core (40 at both sizes
on the target device). The script reads the selected device's `num_vectorcore`
by default; `--vector-cores` is an optional override.

The script does not import vLLM or vLLM Ascend. For the original path it needs
`vllm_ascend_C*.so` and the matching CANN custom operators. It finds the binary
in an installed vLLM Ascend package or accepts `--native-library /path/to/.so`.
It discovers the vendor directory if packaged next to that binary; otherwise
the CANN environment must already expose the operators. It does not substitute
a CPU or PyTorch approximation for the original path.

Each row reports median event/wall latency per invocation and NoPE/RoPE exact
match rates and maximum absolute errors against the original cache write.
`PASS` allows numerical error within `atol=0.035, rtol=0.01`; untouched cache
rows must match exactly. `--check-only` skips timing. No graphs are captured.

### Standalone DeepSeek V4.1 SWA Triton benchmark

Copy `benchmark_dsv41_swa_triton.py` to an Ascend 910B server with CANN, PyTorch,
`torch-npu`, and a compatible Triton-Ascend. It does not import vLLM or vLLM
Ascend and does not load a model. It benchmarks the full post-WKV RMSNorm →
partial interleaved RoPE → paged store. The default focuses on 1024 and 4096
rows in the page-local (`paged`) layout. It tests 25 configurations per size:
the frozen baseline and four variants (`masked512`, `gather64-bounded`,
`slice-gather64-split`, `slice-pair64`) across two schedules and three grids.

```bash
python benchmark_dsv41_swa_triton.py --vector-cores 40
```

Copy the four `RESULT` lines to share a run; no JSON file is needed. Each line
shows the timed count, the baseline, the fastest `masked512` scheduling control,
and the best candidate from each of the two fastest other variant families
(ranked by event median). Values are event/wall medians in microseconds,
followed by paired event speedup p50/p10. Use `--verbose` for the ranked tables
or `--json` for all samples.

- `pr21183-masked512/direct` is the frozen PR #21183 fused kernel. `masked512`
  reuses its approach in the configurable scheduler, so it can also be compared
  with a capped grid. `gather64` adapts the partial RoPE path from PR #21581 to
  a single layer. `gather64-bounded` and `gather512-bounded` add #21581's
  upper-bound checks and branch around the store for partial and full-width
  paths, respectively. All variants use the same two-column physical mapping;
  the DSpark-specific layer dispatch, flat slot format and 2D grid are excluded.
- `gather512`, `gather64`, `load64` and `slice-gather64-split` compare ways to
  obtain the adjacent RoPE values. The two `slice-gather64` variants compare
  split versus merged stores; `slice-pair64` replaces gather with even/odd
  split and join, while `pair512` uses even/odd split and join over the full row
  with masked cos/sin. `slice-pair64-skip` tests skipping the row on a negative
  slot; `tile2`, `tile4`, and `tile8` test batched row reductions and stores.
  `tile4` and `tile8` are opt-in via `--variants` because larger tiles can hit
  Ascend UB capacity and alignment constraints. A candidate compilation
  failure is shown as `SKIP` and recorded in `--json`; it is not timed. A
  baseline failure or incorrect cache contents still result in a nonzero exit.
- `direct` uses one program per row/tile. `cyclic` strides by the program count;
  `blocked` assigns each single-row program a contiguous range. Both scan
  `--programs 20 40 80` by default, capped by the number of work items. On 40
  vector cores, 40 programs give one program per core on average; 80 programs
  do not add cores. An optional one-layer grid of 13 probes #21581's per-layer
  token fan-out, not its 39 total programs.
- Layouts include page-local token order, shuffled slots with padded strides,
  and trailing negative slots for graph padding. The script checks cache holes
  and padding as well as both feature slices against a BF16-rounded CPU
  reference. Captured graphs replay with changed KV and slot contents before
  being timed. Use `--layouts shuffled-padded` for a padded-stride follow-up
  or `--layouts graph-tail --modes graph` for a graph-padding regression check.
  `--slot-dtype int64` and `--row-padding 32` can stress additional stride and
  slot cases. The default cache block size is 128.
- The default compares fused implementations. Optionally pass `--native-library`
  with a trusted local shared library that registers `inplace_partial_rotary_mul`
  and `npu_scatter_nd_update_sk` and whose matching CANN kernels are installed.
  This adds the native three-operator comparison; it is not needed to run the
  benchmark or choose among Triton variants.

`eager` measures a burst of Python launches; its event interval can include
stream gaps from host submission. `graph` repeatedly replays a one-call graph.
`graph-batch` puts all repeats in one graph replay to amortize submission.
Compile, warm up, validate and capture outside the timed samples. The JSON
contains interleaved raw samples, median/p90, and speedup paired by sample round
against PR #21183. These are microbenchmarks on reused buffers, not serving
latency or isolated device task durations.

Compare the 1024- and 4096-row eager and graph-batch medians and paired p10
speedups. Use `--tokens`, `--variants`, `--schedules`, and `--programs` to
retest finalists or check specific crossover points.
If variants are close or eager timing is dominated by
dispatch, profile a reduced case on the target CANN version, for example:

```bash
msprof --output=./swa-profile --task-time=on --application="python benchmark_dsv41_swa_triton.py --tokens 4096 --layouts paged --variants slice-pair64 --schedules cyclic --programs 40 --modes eager"
```

Use the exported `op_summary` task durations and timeline to separate device
execution from launch gaps. Confirm any selected production kernel with the
operator's NPU tests and the real serving workload, including stream overlap.

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
