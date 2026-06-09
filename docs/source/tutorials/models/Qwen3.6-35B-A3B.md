# Qwen3.6-35B-A3B Deployment Tutorial

## 1 Introduction

Qwen3.6-35B-A3B is a sparse MoE model in the Qwen3.6 family, with 35B total parameters and about 3B activated parameters per token. It uses the hybrid attention architecture used by Qwen3.5-style models, and is suitable for long-context online serving on Ascend hardware.

This document describes the main validation steps for the model, including supported features, prerequisites, installation, single-node online deployment, optional multi-node deployment, functional verification, accuracy and performance evaluation, performance tuning, and FAQs.

The `Qwen3.6-35B-A3B` model is first supported in `vllm-ascend:v0.18.0rc1`. Use `v0.18.0rc1` or later for this model. The examples below use the version placeholder configured by the documentation build system.

:::{note}
Qwen3.6-35B-A3B has a known issue when MTP/speculative decoding is enabled. If the service shuts down with `numAcceptedTokens[0]=4 exceeds varlen segment length=3`, disable speculative decoding and refer to [#9956](https://github.com/vllm-project/vllm-ascend/issues/9956). The default startup commands in this document keep speculative decoding disabled.
:::

## 2 Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix, including BF16, W8A8 quantization, chunked prefill, automatic prefix caching, asynchronous scheduling, tensor parallelism, expert parallelism, data parallelism, and ACLGraph support.

Refer to [feature guide](../../user_guide/feature_guide/index.md) to get feature configuration details.

## 3 Prerequisites

### 3.1 Model Weight

- `Qwen3.6-35B-A3B` (BF16 version): requires 1 Atlas 800 A3 (64G x 16) node or 1 Atlas 800 A2 (64G x 8) node. [Download model weight](https://modelscope.cn/models/Qwen/Qwen3.6-35B-A3B).
- `Qwen3.6-35B-A3B-w8a8` (quantized version): requires 1 Atlas 800 A3 (64G x 16) node or 1 Atlas 800 A2 (64G x 8) node. [Download model weight](https://www.modelscope.cn/models/Eco-Tech/Qwen3.6-35B-A3B-w8a8).

It is recommended to download the model weight to `/root/.cache/`. If you use multi-node deployment, use a shared directory or keep the same model path on each node.

### 3.2 Verify Multi-node Communication (Optional)

If you want to deploy the model in a multi-node environment, verify the communication environment according to [verify multi-node communication environment](../../installation.md#verify-multi-node-communication).

## 4 Installation

### 4.1 Docker Image Installation

Select an image based on your machine type. For example, use `quay.io/ascend/vllm-ascend:|vllm_ascend_version|` for Atlas 800 A2 and `quay.io/ascend/vllm-ascend:|vllm_ascend_version|-a3` for Atlas 800 A3.

Refer to [using docker](../../installation.md#set-up-using-docker) for the complete installation guide.

```{code-block} bash
:substitutions:

# Update --device according to your device.
# Atlas A2: /dev/davinci[0-7]
# Atlas A3: /dev/davinci[0-15]
# Download the model weight to /root/.cache in advance.
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
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

If you want to deploy a multi-node service, set up the same environment on each node.

### 4.2 Source Code Installation

You can also build and install `vllm-ascend` from source. Refer to [set up using python](../../installation.md#set-up-using-python).

If you want to deploy a multi-node service, install the same version of vLLM and vLLM-Ascend on each node.

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

:::{note}
MTP/speculative decoding is not enabled in the default command because of [#9956](https://github.com/vllm-project/vllm-ascend/issues/9956). If you want to test it, add `--speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": true}'` and compare stability, TTFT, TPOT, and throughput.
:::

### 5.2 Multi-Node Deployment with MP

Qwen3.6-35B-A3B fits in a single node, so multi-node MP deployment is mainly used to scale throughput with more DP groups. The following example uses 2 Atlas 800 A2 (64G x 8) nodes with `DP8TP2`: 4 local DP groups per node and TP2 per DP group.

Replace `nic_name`, `local_ip`, and `node0_ip` with the actual network interface and IP addresses in your environment.

Run the following script on node 0.

```shell
#!/bin/sh

export VLLM_USE_MODELSCOPE=True
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# Get these values through ifconfig.
# nic_name is the network interface name corresponding to local_ip.
nic_name="xxxx"
local_ip="xxxx"

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=1024
export TASK_QUEUE_ENABLE=1

vllm serve Eco-Tech/Qwen3.6-35B-A3B-w8a8 \
  --host 0.0.0.0 \
  --port 8000 \
  --data-parallel-size 8 \
  --api-server-count 4 \
  --data-parallel-size-local 4 \
  --data-parallel-address $local_ip \
  --data-parallel-rpc-port 13389 \
  --seed 1024 \
  --served-model-name qwen3.6 \
  --tensor-parallel-size 2 \
  --enable-expert-parallel \
  --max-num-seqs 32 \
  --max-model-len 65536 \
  --max-num-batched-tokens 8192 \
  --trust-remote-code \
  --gpu-memory-utilization 0.9 \
  --enable-prefix-caching \
  --quantization ascend \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --additional-config '{"enable_cpu_binding":true, "enable_flashcomm1":true, "multistream_overlap_shared_expert": true}' \
  --async-scheduling
```

Common Issues Tip: If node 1 cannot join the service or HCCL initialization times out, refer to [verify multi-node communication environment](../../installation.md#verify-multi-node-communication) and [FAQs](../../faqs.md). Make sure the network interface names, IP addresses, and RPC ports are consistent across nodes.

Run the following script on node 1.

```shell
#!/bin/sh

export VLLM_USE_MODELSCOPE=True
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# Get these values through ifconfig.
# nic_name is the network interface name corresponding to local_ip.
nic_name="xxxx"
local_ip="xxxx"

# The value of node0_ip must be consistent with local_ip on node 0.
node0_ip="xxxx"

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=1024
export TASK_QUEUE_ENABLE=1

vllm serve Eco-Tech/Qwen3.6-35B-A3B-w8a8 \
  --host 0.0.0.0 \
  --port 8000 \
  --headless \
  --data-parallel-size 8 \
  --data-parallel-size-local 4 \
  --data-parallel-start-rank 4 \
  --data-parallel-address $node0_ip \
  --data-parallel-rpc-port 13389 \
  --seed 1024 \
  --tensor-parallel-size 2 \
  --served-model-name qwen3.6 \
  --max-num-seqs 32 \
  --max-model-len 65536 \
  --max-num-batched-tokens 8192 \
  --enable-expert-parallel \
  --trust-remote-code \
  --gpu-memory-utilization 0.9 \
  --enable-prefix-caching \
  --quantization ascend \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --additional-config '{"enable_cpu_binding":true, "enable_flashcomm1":true, "multistream_overlap_shared_expert": true}' \
  --async-scheduling
```

Common Issues Tip: If the headless node exits immediately, check whether node 0 is already running, whether `--data-parallel-address` points to node 0, and whether `--data-parallel-start-rank` is unique for each node.

**Key parameters for MP deployment:**

- `--data-parallel-size` is the global DP size across all nodes. In the example, 8 DP ranks are used.
- `--data-parallel-size-local` is the number of DP ranks on the current node. In the example, each A2 node has 4 local DP ranks.
- `--data-parallel-start-rank` is the first DP rank on the current node. Node 0 starts from 0 by default, and node 1 starts from 4.
- `--data-parallel-address` must point to the master DP node. Use node 0 `local_ip` on node 0 and `node0_ip` on other nodes.
- `--data-parallel-rpc-port` is the DP RPC port. Use the same value on all nodes and ensure the port is available.
- `--api-server-count` controls how many API server processes are started on the master node.
- `--headless` starts a worker node without exposing an API server. Use it on non-master nodes.
- `--tensor-parallel-size 2` maps one TP group to 2 NPUs. With `--data-parallel-size-local 4`, each A2 node uses 8 NPUs.
- `HCCL_IF_IP`, `GLOO_SOCKET_IFNAME`, `TP_SOCKET_IFNAME`, and `HCCL_SOCKET_IFNAME` bind HCCL, Gloo, and TP communication to the selected network.

### 5.3 Prefill-Decode Disaggregation

Qwen3.6-35B-A3B does not require PD disaggregation for capacity because both BF16 and W8A8 deployments can fit in one node. If your production workload needs separate prefill and decode resource pools, refer to [Mooncake](../features/pd_disaggregation_mooncake_multi_node.md) for the general PD disaggregation workflow.

When adapting the Mooncake scripts for Qwen3.6-35B-A3B, keep the following model-specific settings:

- Use `Eco-Tech/Qwen3.6-35B-A3B-w8a8` with `--quantization ascend` for the W8A8 model.
- Keep `--enable-expert-parallel` for MoE layers.
- Use larger `--max-num-batched-tokens` on prefill nodes and smaller values on decode nodes.
- Use `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'` on decode nodes.
- Use `--additional-config '{"recompute_scheduler_enable": true, "enable_cpu_binding": true}'` only when `kv_role` is `kv_producer` or `kv_consumer`.
- Keep MTP/speculative decoding disabled unless you are explicitly validating [#9956](https://github.com/vllm-project/vllm-ascend/issues/9956).

Common Issues Tip: If PD requests reach the proxy but no output is returned, check that the proxy host list includes all healthy prefill and decode endpoints, and verify that the service verification request in Section 6 succeeds through the proxy port.

## 6 Functional Verification

After the server is started, send a request to verify basic model functionality. For single-node and MP deployment, use the API endpoint on node 0. For PD disaggregation, use the proxy endpoint.

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

The following configurations are validated in specific test environments and are for reference only. The optimal configuration depends on hardware type, maximum input/output length, request concurrency, prefix cache hit rate, quantization, and whether you need multi-node throughput scaling. Tune the parameters in Section 9.2 based on your actual workload.

| Scenario | Deployment Mode | Total NPUs | Weight Version | Key Considerations |
| -------- | --------------- | ---------- | -------------- | ------------------ |
| Long context | Single-node online serving | 2 or more NPUs | W8A8 | Use larger `--max-model-len` and reserve enough KV cache. Lower `--max-num-seqs` if OOM occurs. |
| High throughput | Single-node or multi-node MP | 8 or more NPUs | W8A8 | Increase throughput through DP groups and tune `--max-num-batched-tokens`. |
| Low latency | Single-node online serving | 2 or more NPUs | W8A8 | Use smaller `--max-num-batched-tokens`, full decode ACLGraph, and disable speculative decoding by default. |

| Scenario | Node Role | NPUs | TP | DP | Max Num Seqs | Max Model Len | Max Num Batched Tokens | Prefix Cache | Main Optimizations |
| -------- | --------- | ---- | -- | -- | ------------ | ------------- | ---------------------- | ------------ | ------------------ |
| Long context | Single node | 2 or more | 2 | 1 | 128 | 262144 | 16384 | On | FullGraph, FlashComm1, shared expert overlap, CPU binding |
| High throughput | MP node | 8 per A2 node | 2 | 4 per node | 32 per DP | 65536 | 8192 | On | FullGraph, FlashComm1, async scheduling, shared expert overlap |
| Low latency | Single node | 2 or more | 2 | 1 | Tune by concurrency | 32768 or 65536 | 1024 to 4096 | Workload dependent | FullGraph, CPU binding, speculative decoding disabled |

### 9.2 Tuning Guidelines

Refer to [public performance tuning documentation](../../developer_guide/performance_and_debug/optimization_and_tuning.md) for general tuning methods, and refer to [feature matrix](../../user_guide/support_matrix/feature_matrix.md) for feature descriptions.

Recommended tuning order:

1. Set the deployment topology first. Use single-node deployment for validation and multi-node MP only when you need more DP groups for throughput.
2. Choose the maximum context length with `--max-model-len`. Long context increases KV cache usage, so reduce `--max-num-seqs` or `--gpu-memory-utilization` if OOM occurs.
3. Tune `--max-num-batched-tokens`. Larger values usually improve prefill throughput but increase activation memory. Decode-heavy workloads usually need smaller values.
4. Tune `--max-num-seqs` according to service concurrency. Requests above this value wait in the queue and the waiting time is counted in TTFT and TPOT.
5. Tune `--gpu-memory-utilization`. Increase it to provide more KV cache, but leave headroom for runtime memory fluctuation and expert imbalance.
6. Keep MTP/speculative decoding disabled by default. Enable it only for controlled validation until [#9956](https://github.com/vllm-project/vllm-ascend/issues/9956) is resolved.
7. Tune ACLGraph capture. `FULL_DECODE_ONLY` is recommended for decode. If you set `cudagraph_capture_sizes` manually, include common decode batch sizes. With FlashComm1, use capture sizes that are multiples of TP size.

### 9.3 Model-Specific Optimizations

| Optimization | Enablement | Benefit | Notes |
| ------------ | ---------- | ------- | ----- |
| Hybrid attention support | Enabled by model implementation | Supports Qwen3.6 long-context inference. | Tune context length based on KV cache capacity. |
| Full decode ACLGraph | `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'` | Reduces operator dispatch overhead and stabilizes decode performance. | Recommended for decode-heavy serving. |
| FlashComm1 | `--additional-config '{"enable_flashcomm1": true}'` | Reduces communication overhead in TP and high-concurrency scenarios. | May not help low-concurrency workloads. |
| Shared expert overlap | `--additional-config '{"multistream_overlap_shared_expert": true}'` | Overlaps shared expert computation in MoE workloads. | Recommended for throughput scenarios. |
| Asynchronous scheduling | `--async-scheduling` | Improves high-concurrency throughput by using non-blocking scheduling. | Disable it and compare if the workload is latency-sensitive. |
| Prefix caching | `--enable-prefix-caching` | Improves repeated-prefix workloads. | Monitor HBM usage for long-context workloads. |
| Qwen3.6 MTP speculative decoding | `--speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": true}'` | Can improve decode throughput when stable and accepted tokens are high. | Not enabled by default because of [#9956](https://github.com/vllm-project/vllm-ascend/issues/9956). |

## 10 FAQ

For common environment, installation, and general parameter issues, refer to [FAQs](../../faqs.md). This section only covers model-specific issues for Qwen3.6-35B-A3B.

### Q1: Why does the service report OOM during startup or soon after accepting requests?

**Phenomenon:** The service fails during profile run, or it starts successfully but reports OOM when real traffic arrives.

**Cause:** Qwen3.6 long-context serving consumes a large KV cache. Large `--max-model-len`, large `--max-num-seqs`, large `--max-num-batched-tokens`, or high `--gpu-memory-utilization` can leave insufficient HBM headroom.

**Solution:** Use the W8A8 model with `--quantization ascend` when possible, lower `--max-model-len`, lower `--max-num-seqs`, lower `--max-num-batched-tokens`, or reduce `--gpu-memory-utilization`. Keep `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`.

### Q2: Why does Qwen3.6 shut down after enabling MTP/speculative decoding?

**Phenomenon:** The service may shut down and report `numAcceptedTokens[0]=4 exceeds varlen segment length=3` during shape/dtype processing.

**Cause:** This is a known issue for Qwen3.6-35B-A3B with MTP/speculative decoding, tracked in [#9956](https://github.com/vllm-project/vllm-ascend/issues/9956).

**Solution:** Disable `--speculative-config` for production serving. If you need to test MTP, run it in a controlled validation environment and compare stability, TTFT, TPOT, and throughput.

### Q3: Why does multi-node MP deployment hang during initialization?

**Phenomenon:** One node waits for other ranks, HCCL initialization times out, or the headless node exits.

**Cause:** Network interface names, IP addresses, DP ranks, or RPC ports are inconsistent across nodes.

**Solution:** Verify multi-node communication first. Ensure `HCCL_IF_IP`, `GLOO_SOCKET_IFNAME`, `TP_SOCKET_IFNAME`, and `HCCL_SOCKET_IFNAME` match the selected NIC. Ensure all nodes use the same `--data-parallel-rpc-port`, non-master nodes use `--headless`, and `--data-parallel-start-rank` does not overlap.

### Q4: Why does enabling prefix caching not improve performance?

**Phenomenon:** Prefix caching is enabled, but throughput or latency does not improve.

**Cause:** Prefix caching only helps when requests share reusable prefixes. For random prompts or low cache hit rates, it may add memory pressure without visible gains.

**Solution:** Enable prefix caching for repeated-prefix workloads. For random benchmark datasets or memory-constrained long-context workloads, compare with `--no-enable-prefix-caching`.

### Q5: How should I tune async scheduling for Qwen3.6?

**Phenomenon:** Throughput improves in high-concurrency scenarios, but some latency-sensitive workloads may not benefit.

**Cause:** Asynchronous scheduling reduces blocking overhead, but the benefit depends on concurrency, prompt/output length, and graph capture shape.

**Solution:** Use `--async-scheduling` for high-throughput serving. For low-latency serving, compare TTFT and TPOT with and without this option.
