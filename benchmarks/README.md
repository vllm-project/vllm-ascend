# Introduction
This document outlines the benchmarking methodology for vllm-ascend, aimed at evaluating the performance under a variety of workloads. The primary goal is to help developers assess whether their pull requests improve or degrade vllm-ascend's performance. To maintain alignment with vLLM, we use the [benchmark](https://github.com/vllm-project/vllm/tree/main/benchmarks) script provided by the vllm project.

# Overview
**Benchmarking Coverage**: We measure latency, throughput, and fixed-QPS serving on the Atlas800I A2 (see [quick_start](../docs/source/quick_start.md) to learn more supported devices list), with different models(coming soon).
- Latency tests
    - Input length: 32 tokens.
    - Output length: 128 tokens.
    - Batch size: fixed (8).
    - Models: Meta-Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct.
    - Evaluation metrics: end-to-end latency (mean, median, p99).

- Throughput tests
    - Input length: randomly sample 200 prompts from ShareGPT dataset (with fixed random seed).
    - Output length: the corresponding output length of these 200 prompts.
    - Batch size: dynamically determined by vllm to achieve maximum throughput.
    - Models: Meta-Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct.
    - Evaluation metrics: throughput.
- Serving tests
    - Input length: randomly sample 200 prompts from ShareGPT dataset (with fixed random seed).
    - Output length: the corresponding output length of these 200 prompts.
    - Batch size: dynamically determined by vllm and the arrival pattern of the requests.
    - **Average QPS (query per second)**: 1, 4, 16 and inf. QPS = inf means all requests come at once. For other QPS values, the arrival time of each query is determined using a random Poisson process (with fixed random seed).
    - Models: Meta-Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct.
    - Evaluation metrics: throughput, TTFT (time to the first token, with mean, median and p99), ITL (inter-token latency, with mean, median and p99).

**Benchmarking Duration**: about 800 senond for single model.


# Quick Use
## Prerequisites
Before running the benchmarks, ensure the following:

- vllm and vllm-ascend are installed and properly set up in an NPU environment, as these scripts are specifically designed for NPU devices.

- Install necessary dependencies for benchmarks:
    ```
    pip install -r benchmarks/requirements-bench.txt
    ```
    
- For performance benchmark, it is recommended to set the [load-format](https://github.com/vllm-project/vllm-ascend/blob/5897dc5bbe321ca90c26225d0d70bff24061d04b/benchmarks/tests/latency-tests.json#L7) as `dummy`, It will construct random weights based on the passed model without downloading the weights from internet, which can greatly reduce the benchmark time. feel free to add your own models and parameters in the JSON to run your customized benchmarks.

## Run benchmarks
The provided scripts automatically execute performance tests for serving, throughput, and latency. To start the benchmarking process, run command in the vllm-ascend root directory:
```
bash benchmarks/scripts/run-performance-benchmarks.sh
```
Once the script completes, you can find the results in the benchmarks/results folder. The output files may resemble the following:
```
|-- latency_llama8B_tp1.json
|-- serving_llama8B_tp1_sharegpt_qps_1.json
|-- serving_llama8B_tp1_sharegpt_qps_16.json
|-- serving_llama8B_tp1_sharegpt_qps_4.json
|-- serving_llama8B_tp1_sharegpt_qps_inf.json
|-- throughput_llama8B_tp1.json
```
These files contain detailed benchmarking results for further analysis.
