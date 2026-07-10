# Qwen3.6-35B-A3B Deployment Tutorial

## 1 Introduction

Qwen3.6-35B-A3B is a sparse MoE model in the Qwen3.6 family, with 35B total parameters and about 3B activated parameters per token. It uses the hybrid attention architecture used by Qwen3.5-style models, and is suitable for long-context online serving on Ascend hardware.

This document describes the main validation steps for the model, including supported features, prerequisites, installation, single-node online deployment, functional verification, accuracy and performance evaluation, performance tuning, and FAQs.

The `Qwen3.6-35B-A3B` model is first supported in `vllm-ascend:v0.18.0rc1`. Use `v0.18.0rc1` or later for this model. The examples below use the version placeholder configured by the documentation build system.

## 2 Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix, including BF16, W8A8 quantization, chunked prefill, automatic prefix caching, asynchronous scheduling, tensor parallelism, expert parallelism, and ACLGraph support.

Refer to [feature guide](../../user_guide/feature_guide/index.md) to get feature configuration details.

## 3 Prerequisites

### 3.1 Model Weight

- `Qwen3.6-35B-A3B` (BF16 version): requires 1 Atlas 800 A3 (64G x 16) node or 1 Atlas 800 A2 (64G x 8) node. [Download model weight](https://modelscope.cn/models/Qwen/Qwen3.6-35B-A3B).
- `Qwen3.6-35B-A3B-w8a8` (quantized version): requires 1 Atlas 800 A3 (64G x 16) node or 1 Atlas 800 A2 (64G x 8) node. [Download model weight](https://www.modelscope.cn/models/Eco-Tech/Qwen3.6-35B-A3B-w8a8).

It is recommended to download the model weight to `/root/.cache/`.

## 4 Installation

### 4.1 Docker Image Installation

Select an image based on your machine type. For example, use `quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}` for Atlas 800 A2 and `quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-a3` for Atlas 800 A3.

Refer to [using docker](../../installation.md#set-up-using-docker) for the complete installation guide.

```bash

# Update --device according to your device.
# Atlas A2: /dev/davinci[0-7]
# Atlas A3: /dev/davinci[0-15]
# Download the model weight to /root/.cache in advance.
export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}
export NAME=vllm-ascend

docker run --rm \
  --name $NAME \
  --net=host \
  --shm-size=1g \
  --device /dev/davinci0 \
  --device /dev/davinci1 \
  --device /dev/davinci2 \
  --device /dev/davinci3 \
  --device /dev/davinci4 \
  --device /dev/davinci5 \
  --device /dev/davinci6 \
  --device /dev/davinci7 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /root/.cache:/root/.cache \
  -it $IMAGE bash
```

After entering the container, verify that vLLM and vLLM-Ascend can be imported:

```shell
python -c "import vllm, vllm_ascend; print('vllm and vllm_ascend are ready')"
```

### 4.2 Source Code Installation

You can also build and install `vllm-ascend` from source. Refer to [set up using python](../../installation.md#set-up-using-python).

## 5 Online Service Deployment

### 5.1 Single-Node Online Deployment

Single-node deployment runs both Prefill and Decode on the same node. `Qwen3.6-35B-A3B-w8a8` can be deployed on 1 Atlas 800 A3 (64G x 16) or 1 Atlas 800 A2 (64G x 8). The W8A8 version needs `--quantization ascend`.

Run the following script to execute online inference with up to 262144 context length on 1 Atlas 800 A3 (64G x 16).

```shell
#!/bin/sh

# Load model from ModelScope to speed up download.
export VLLM_USE_MODELSCOPE=True

# Reduce memory fragmentation and avoid out-of-memory errors.
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=1024
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl kernel.sched_migration_cost_ns=50000
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD

vllm serve Eco-Tech/Qwen3.6-35B-A3B-w8a8 \
  --host 0.0.0.0 \
  --port 8000 \
  --data-parallel-size 1 \
  --tensor-parallel-size 2 \
  --enable-expert-parallel \
  --seed 1024 \
  --quantization ascend \
  --served-model-name qwen3.6 \
  --max-num-seqs 128 \
  --max-model-len 262144 \
  --max-num-batched-tokens 16384 \
  --trust-remote-code \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --additional-config '{"enable_cpu_binding":true, "enable_flashcomm1":true, "multistream_overlap_shared_expert": true}' \
  --async-scheduling
```

Common Issues Tip: If the service fails to start, HBM is insufficient, or requests are not scheduled as expected, refer to [FAQs](../../faqs.md) first, and then check the model-specific FAQ in Section 10.

**Key parameters:**

- `--data-parallel-size 1` and `--tensor-parallel-size 2` set DP and TP for the default single-node serving example.
- `--enable-expert-parallel` enables expert parallelism for MoE layers. Do not mix MoE tensor parallelism and expert parallelism in the same MoE layer.
- `--max-model-len` is the maximum input plus output length for a single request. Increase it only when enough KV cache is available.
- `--max-num-seqs` is the maximum number of active requests scheduled by each DP group. For performance tests, set `--max-num-seqs * --data-parallel-size` greater than or equal to the test concurrency.
- `--max-num-batched-tokens` is the maximum number of tokens processed in one scheduler step. A larger value can improve prefill efficiency but consumes more activation memory.
- `--gpu-memory-utilization` controls how much HBM vLLM can use to calculate KV cache capacity. A higher value increases KV cache size but can trigger OOM if runtime memory is higher than the profile run.
- `--enable-prefix-caching` enables prefix caching. For long-context serving, monitor memory usage because prefix caching can increase KV cache pressure.
- `--quantization ascend` enables Ascend quantization for the W8A8 model. Remove this option when deploying the BF16 model.
- `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'` enables full decode ACLGraph replay to reduce dispatch overhead.
- `--additional-config` enables Ascend-specific optimizations. `enable_flashcomm1` enables FlashComm1, `multistream_overlap_shared_expert` overlaps shared expert computation, and `enable_cpu_binding` enables Ascend-native CPU binding.
- `--async-scheduling` enables asynchronous scheduling, which can improve high-concurrency throughput.

## 6 Functional Verification

After the server is started, send a request to verify basic model functionality.

```shell
curl http://<server_ip>:<port>/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6",
    "prompt": "The future of AI is",
    "max_tokens": 50,
    "temperature": 0
  }'
```

Expected result: the HTTP status is 200 and the JSON response contains a `choices` field with generated text.

## 7 Accuracy Evaluation

Here are two accuracy evaluation methods.

### 7.1 Using AISBench

Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) for details. After execution, you can get the accuracy result of `Qwen3.6-35B-A3B-w8a8`.

### 7.2 Using Language Model Evaluation Harness

Refer to [Using lm_eval](../../developer_guide/evaluation/using_lm_eval.md) for installation and usage details. When using online serving, set `base_url` to the endpoint started in Section 5.

```shell
lm_eval \
  --model local-completions \
  --model_args model=qwen3.6,base_url=http://127.0.0.1:8000/v1/completions,tokenized_requests=False,trust_remote_code=True \
  --tasks gsm8k \
  --output_path ./
```

## 8 Performance Evaluation

### 8.1 Using AISBench

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

### 8.2 Using vLLM Benchmark

Run performance evaluation of `Qwen3.6-35B-A3B-w8a8` as an example. Refer to [vLLM benchmark](https://docs.vllm.ai/en/latest/benchmarking/) for more details.

There are three `vllm bench` subcommands:

- `latency`: benchmark the latency of a single batch of requests.
- `serve`: benchmark online serving throughput.
- `throughput`: benchmark offline inference throughput.

Take `serve` as an example:

```shell
export VLLM_USE_MODELSCOPE=True

vllm bench serve \
  --model Eco-Tech/Qwen3.6-35B-A3B-w8a8 \
  --served-model-name qwen3.6 \
  --dataset-name random \
  --random-input 200 \
  --num-prompts 200 \
  --request-rate 1 \
  --save-result \
  --result-dir ./
```

After several minutes, you can get the performance evaluation result.

## 9 Performance Tuning

### 9.1 Recommended Configurations

The following configurations are validated in specific test environments and are for reference only. The optimal configuration depends on hardware type, maximum input/output length, request concurrency, prefix cache hit rate, and quantization. Tune the parameters in Section 9.2 based on your actual workload.

| Scenario | Deployment Mode | Total NPUs | Weight Version | Key Considerations |
| -------- | --------------- | ---------- | -------------- | ------------------ |
| Long context | Single-node online serving | 2 or more NPUs | W8A8 | Use larger `--max-model-len` and reserve enough KV cache. Lower `--max-num-seqs` if OOM occurs. |
| High throughput | Single-node online serving | 8 or more NPUs | W8A8 | Increase local DP groups within one node and tune `--max-num-batched-tokens`. |
| Low latency | Single-node online serving | 2 or more NPUs | W8A8 | Use smaller `--max-num-batched-tokens`, full decode ACLGraph, and disable speculative decoding by default. |

| Scenario | Node Role | NPUs | TP | DP | Max Num Seqs | Max Model Len | Max Num Batched Tokens | Prefix Cache | Main Optimizations |
| -------- | --------- | ---- | -- | -- | ------------ | ------------- | ---------------------- | ------------ | ------------------ |
| Long context | Single node | 2 or more | 2 | 1 | 128 | 262144 | 16384 | On | FullGraph, FlashComm1, shared expert overlap, CPU binding |
| High throughput | Single node | 8 or more | 2 | 4 or more | 32 per DP | 65536 | 8192 | On | FullGraph, FlashComm1, async scheduling, shared expert overlap |
| Low latency | Single node | 2 or more | 2 | 1 | Tune by concurrency | 32768 or 65536 | 1024 to 4096 | Workload dependent | FullGraph, CPU binding, speculative decoding disabled |

### 9.2 Tuning Guidelines

Refer to [public performance tuning documentation](../../developer_guide/performance_and_debug/optimization_and_tuning.md) for general tuning methods, and refer to [feature matrix](../../user_guide/support_matrix/feature_matrix.md) for feature descriptions.

Recommended tuning order:

1. Use single-node deployment. If more throughput is required, increase local DP groups within the same node.
2. Choose the maximum context length with `--max-model-len`. Long context increases KV cache usage, so reduce `--max-num-seqs` or `--gpu-memory-utilization` if OOM occurs.
3. Tune `--max-num-batched-tokens`. Larger values usually improve prefill throughput but increase activation memory. Decode-heavy workloads usually need smaller values.
4. Tune `--max-num-seqs` according to service concurrency. Requests above this value wait in the queue and the waiting time is counted in TTFT and TPOT.
5. Tune `--gpu-memory-utilization`. Increase it to provide more KV cache, but leave headroom for runtime memory fluctuation and expert imbalance.
6. Tune ACLGraph capture. `FULL_DECODE_ONLY` is recommended for decode. If you set `cudagraph_capture_sizes` manually, include common decode batch sizes. With FlashComm1, use capture sizes that are multiples of TP size.

### 9.3 Model-Specific Optimizations

| Optimization | Enablement | Benefit | Notes |
| ------------ | ---------- | ------- | ----- |
| Hybrid attention support | Enabled by model implementation | Supports Qwen3.6 long-context inference. | Tune context length based on KV cache capacity. |
| Full decode ACLGraph | `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'` | Reduces operator dispatch overhead and stabilizes decode performance. | Recommended for decode-heavy serving. |
| FlashComm1 | `--additional-config '{"enable_flashcomm1": true}'` | Reduces communication overhead in TP and high-concurrency scenarios. | May not help low-concurrency workloads. |
| Shared expert overlap | `--additional-config '{"multistream_overlap_shared_expert": true}'` | Overlaps shared expert computation in MoE workloads. | Recommended for throughput scenarios. |
| Asynchronous scheduling | `--async-scheduling` | Improves high-concurrency throughput by using non-blocking scheduling. | Disable it and compare if the workload is latency-sensitive. |
| Prefix caching | `--enable-prefix-caching` | Improves repeated-prefix workloads. | Monitor HBM usage for long-context workloads. |
| Qwen3.6 MTP speculative decoding | `--speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": true}'` | Can improve decode throughput when stable and accepted tokens are high. | Validate stability, TTFT, TPOT, and throughput for your workload. |

## 10 FAQ

For common environment, installation, and general parameter issues, refer to [FAQs](../../faqs.md). This section only covers model-specific issues for Qwen3.6-35B-A3B.

### Q1: Why does the service report OOM during startup or soon after accepting requests?

**Phenomenon:** The service fails during profile run, or it starts successfully but reports OOM when real traffic arrives.

**Cause:** Qwen3.6 long-context serving consumes a large KV cache. Large `--max-model-len`, large `--max-num-seqs`, large `--max-num-batched-tokens`, or high `--gpu-memory-utilization` can leave insufficient HBM headroom.

**Solution:** Use the W8A8 model with `--quantization ascend` when possible, lower `--max-model-len`, lower `--max-num-seqs`, lower `--max-num-batched-tokens`, or reduce `--gpu-memory-utilization`. Keep `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`.

### Q2: Why does enabling prefix caching not improve performance?

**Phenomenon:** Prefix caching is enabled, but throughput or latency does not improve.

**Cause:** Prefix caching only helps when requests share reusable prefixes. For random prompts or low cache hit rates, it may add memory pressure without visible gains.

**Solution:** Enable prefix caching for repeated-prefix workloads. For random benchmark datasets or memory-constrained long-context workloads, compare with `--no-enable-prefix-caching`.

### Q3: How should I tune async scheduling for Qwen3.6?

**Phenomenon:** Throughput improves in high-concurrency scenarios, but some latency-sensitive workloads may not benefit.

**Cause:** Asynchronous scheduling reduces blocking overhead, but the benefit depends on concurrency, prompt/output length, and graph capture shape.

**Solution:** Use `--async-scheduling` for high-throughput serving. For low-latency serving, compare TTFT and TPOT with and without this option.
