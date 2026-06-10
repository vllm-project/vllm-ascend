# Qwen3-235B-A22B

## 1 Introduction

Qwen3 is the latest generation of large language models in Qwen series, offering a comprehensive suite of dense and mixture-of-experts (MoE) models. Built upon extensive training, Qwen3 delivers groundbreaking advancements in reasoning, instruction-following, agent capabilities, and multilingual support. Qwen3-235B-A22B is the largest MoE variant, featuring 235B total parameters with 22B activated per token.

This document will demonstrate the main validation steps for Qwen3-235B-A22B in the vLLM-Ascend environment, including supported features, environment preparation, single-node and multi-node deployment, accuracy and performance evaluation.

The Qwen3-235B-A22B model is first supported in **v0.8.4rc2**. This document is validated and written based on **vLLM-Ascend v0.13.0**. All **v0.13.0 and later versions** can run stably. To use the latest features (e.g., PD separation, fused MC2), it is recommended to use v0.13.0 or a later version.

## 2 Supported Features

Please refer to the [Supported Features List](../../user_guide/support_matrix/supported_models.md) for the model support matrix.

Please refer to the [Feature Guide](../../user_guide/feature_guide/index.md) for feature configuration information.

## 3 Prerequisites

### 3.1 Model Weight

The following model variants are available. It is recommended to download the model weight to a shared directory across multiple nodes (e.g., `/root/.cache/`).

**BF16 Version:**

| Model | Hardware Requirement | Download |
|-------|---------------------|----------|
| Qwen3-235B-A22B (BF16) | 1 Atlas 800I A3 (64G × 16) node, 1 Atlas 800I A2 (64G × 8) node, or 2 Atlas 800I A2 (32G × 8) nodes | [Download](https://www.modelscope.cn/models/Qwen/Qwen3-235B-A22B) |

**Quantized Version (Pre-converted):**

| Model | Quantization | Hardware Requirement | Download |
|-------|-------------|---------------------|----------|
| Qwen3-235B-A22B-W8A8 | W8A8 | 1 Atlas 800I A3 (64G × 16) node, 1 Atlas 800I A2 (64G × 8) node, or 2 Atlas 800I A2 (32G × 8) nodes | [Download](https://modelers.cn/models/Modelers_Park/Qwen3-235B-A22B-w8a8) |

These are the recommended numbers of cards, which can be adjusted according to the actual situation.

### 3.2 Model Quantization (Optional)

If you wish to quantize the model yourself, refer to the [MindStudio ModelSlim documentation](https://gitcode.com/Ascend/msmodelslim) for W8A8 quantization procedures. Pre-converted quantized weights are also available for download (see [3.1 Model Weight](#31-model-weight)).

### 3.3 Verify Multi-node Communication

If multi-node deployment is required, please follow the [Verify Multi-node Communication Environment](../../installation.md#verify-multi-node-communication) guide for communication verification.

## 4 Installation

### 4.1 Docker Image Installation

You can use the official all-in-one Docker image for Qwen3 MoE models.

**Docker Pull:**

```{code-block} bash
   :substitutions:

docker pull quay.io/ascend/vllm-ascend:|vllm_ascend_version|
```

**Docker Run:**

```{code-block} bash
   :substitutions:

# Update --device according to your device (Atlas A2: /dev/davinci[0-7], Atlas A3:/dev/davinci[0-15]).
# For Atlas A2 machines:
# export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
# For Atlas A3 machines:
# export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|-a3
docker run --rm \
    --name vllm-ascend-env \
    --shm-size=1g \
    --net=host \
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

The default workdir is `/workspace`. vLLM and vLLM-Ascend code are placed in `/vllm-workspace` and installed in [development mode](https://setuptools.pypa.io/en/latest/userguide/development_mode.html) (`pip install -e`), so changes take effect immediately without requiring a new installation.

To verify the successful installation of the environment, please refer to [installation](../../installation.md).

If deploying a multi-node environment, set up the environment on each node.

Installation Verification:
After starting the container, run the following command to verify the installation:
```bash
docker ps | grep vllm-ascend-env
```
Expected result: The container is listed with status . You can also verify the vllm-ascend version inside the container:Up
```bash
pip show vllm-ascend
```
Expected result: The version information is displayed, matching the pulled image version.

### 4.2 Source Code Installation

If you prefer not to use the Docker image, you can build from source:

1. Clone the repository:
```bash
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
```
2. Install in development mode:
```bash
pip install -e .
```
**Installation Verification:**
```bash
pip show vllm-ascend
```

Expected result: The version information is displayed, confirming a successful installation.

:::{note}
If you want to deploy a multi-node environment, you need to set up environment on each node.
:::

## 5 Online Service Deployment

### 5.1 Single-Node Online Deployment

Single-node deployment completes both Prefill and Decode within the same node, suitable for development, testing, and small-to-medium scale inference scenarios.

**Start the server:**

```shell
export VLLM_USE_MODELSCOPE=True
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=512
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export TASK_QUEUE_ENABLE=1

vllm serve vllm-ascend/Qwen3-235B-A22B-w8a8 \
    --host <host_ip> \
    --port <port> \
    --tensor-parallel-size 8 \
    --data-parallel-size 1 \
    --seed 1024 \
    --quantization ascend \
    --served-model-name qwen3 \
    --max-num-seqs 32 \
    --max-model-len 133000 \
    --max-num-batched-tokens 8096 \
    --enable-expert-parallel \
    --trust-remote-code \
    --gpu-memory-utilization 0.95 \
    --hf-overrides '{"rope_parameters": {"rope_type":"yarn","rope_theta":1000000,"factor":4,"original_max_position_embeddings":32768}}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --async-scheduling
```
:::{note}
- Replace your_model_path with the actual model path (e.g., Modelscope ID or local path).

- To enable quantization for Ascend, the quantization method must be `"ascend"`. If the model is not a quantized model, remove the `--quantization ascend` parameter.

- --enable-expert-parallel enables Expert Parallelism, which is required for MoE models. vLLM does not support mixing ETP and EP; MoE layers use either pure EP or pure TP.

- If you are already inside the container (see [Section 4.1](#41-docker-image-installation)), skip the Docker run step and proceed directly to **Start the server**.
:::

:::{note}
- For additional parameter details, refer to the [vLLM Serving Arguments documentation](https://docs.vllm.com.cn/en/latest/cli/serve/?h=block+size#arguments).
:::

**Service Verification:**

If the service starts successfully, the following startup log will be displayed:

```text
(APIServer pid=<pid>) INFO:     Started server process [<pid>]
(APIServer pid=<pid>) INFO:     Waiting for application startup.
(APIServer pid=<pid>) INFO:     Application startup complete.
```

### 5.2 Multi-Node PD Separation Deployment

The complete PD disaggregation deployment process involves multi-node communication, KV cache transfer, and proxy configuration. For the detailed deployment guide, please refer to [Prefill-Decode Disaggregation Mooncake Verification (Qwen)](../features/pd_disaggregation_mooncake_multi_node.md).

The following example shows the parameter configuration for a three-node A3 PD disaggregation scenario (one Prefill node + two Decode nodes):

**Prefill Node:**

```shell
#!/bin/sh
export HCCL_IF_IP=prefill_node_1_ip

# Set ifname according to your network setting
ifname=""

export GLOO_SOCKET_IFNAME=${ifname}
export TP_SOCKET_IFNAME=${ifname}
export HCCL_SOCKET_IFNAME=${ifname}

# Load model from ModelScope to speed up download
export VLLM_USE_MODELSCOPE=True
# To reduce memory fragmentation and avoid out of memory
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=512
export HCCL_OP_EXPANSION_MODE="AIV"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_ENABLE_FUSED_MC2=2
export TASK_QUEUE_ENABLE=1
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH

vllm serve vllm-ascend/Qwen3-235B-A22B-w8a8 \
    --host <host_ip> \
    --port <port> \
    --tensor-parallel-size 8 \
    --data-parallel-size 2 \
    --data-parallel-size-local 2 \
    --data-parallel-start-rank 0 \
    --data-parallel-address prefill_node_1_ip \
    --data-parallel-rpc-port prefill_node_dp_port \
    --seed 1024 \
    --quantization ascend \
    --served-model-name qwen3 \
    --max-num-seqs 24 \
    --max-model-len 40960 \
    --max-num-batched-tokens 16384 \
    --enable-expert-parallel \
    --enforce-eager \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --no-enable-prefix-caching \
    --kv-transfer-config \
        '{"kv_connector": "MooncakeConnectorV1",
          "kv_role": "kv_producer",
          "kv_port": "30000",
          "engine_id": "0",
          "kv_connector_extra_config": {
              "prefill": {"dp_size": 2, "tp_size": 8},
              "decode": {"dp_size": 8, "tp_size": 4}
          }}'
```

**Decode Node 1:**

```shell
#!/bin/sh
export HCCL_IF_IP=decode_node_1_ip

ifname=""

export GLOO_SOCKET_IFNAME=${ifname}
export TP_SOCKET_IFNAME=${ifname}
export HCCL_SOCKET_IFNAME=${ifname}

# Load model from ModelScope to speed up download
export VLLM_USE_MODELSCOPE=True
# To reduce memory fragmentation and avoid out of memory
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1024
export HCCL_OP_EXPANSION_MODE="AIV"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_ENABLE_FUSED_MC2=2
export TASK_QUEUE_ENABLE=1
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH

vllm serve vllm-ascend/Qwen3-235B-A22B-w8a8 \
    --host <host_ip> \
    --port <port> \
    --tensor-parallel-size 4 \
    --data-parallel-size 8 \
    --data-parallel-size-local 4 \
    --data-parallel-start-rank 0 \
    --data-parallel-address decode_node_1_ip \
    --data-parallel-rpc-port decode_node_dp_port \
    --seed 1024 \
    --quantization ascend \
    --served-model-name qwen3 \
    --max-num-seqs 128 \
    --max-model-len 40960 \
    --max-num-batched-tokens 256 \
    --enable-expert-parallel \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --async-scheduling \
    --no-enable-prefix-caching \
    --kv-transfer-config \
        '{"kv_connector": "MooncakeConnectorV1",
          "kv_role": "kv_consumer",
          "kv_port": "30100",
          "engine_id": "1",
          "kv_connector_extra_config": {
              "prefill": {"dp_size": 2, "tp_size": 8},
              "decode": {"dp_size": 8, "tp_size": 4}
          }}'
```

**Decode Node 2:**

```shell
#!/bin/sh
export HCCL_IF_IP=decode_node_2_ip

ifname=""

export GLOO_SOCKET_IFNAME=${ifname}
export TP_SOCKET_IFNAME=${ifname}
export HCCL_SOCKET_IFNAME=${ifname}

# Load model from ModelScope to speed up download
export VLLM_USE_MODELSCOPE=True
# To reduce memory fragmentation and avoid out of memory
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1024
export HCCL_OP_EXPANSION_MODE="AIV"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_ENABLE_FUSED_MC2=2
export TASK_QUEUE_ENABLE=1
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH

vllm serve vllm-ascend/Qwen3-235B-A22B-w8a8 \
    --host <host_ip> \
    --port <port> \
    --headless \
    --tensor-parallel-size 4 \
    --data-parallel-size 8 \
    --data-parallel-size-local 4 \
    --data-parallel-start-rank 4 \
    --data-parallel-address decode_node_1_ip \
    --data-parallel-rpc-port decode_node_dp_port \
    --seed 1024 \
    --quantization ascend \
    --served-model-name qwen3 \
    --max-num-seqs 128 \
    --max-model-len 40960 \
    --max-num-batched-tokens 256 \
    --enable-expert-parallel \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --async-scheduling \
    --no-enable-prefix-caching \
    --kv-transfer-config \
        '{"kv_connector": "MooncakeConnectorV1",
          "kv_role": "kv_consumer",
          "kv_port": "30100",
          "engine_id": "1",
          "kv_connector_extra_config": {
              "prefill": {"dp_size": 2, "tp_size": 8},
              "decode": {"dp_size": 8, "tp_size": 4}
          }}'
```
:::{note}
- For additional parameter details, refer to the [vLLM Serving Arguments documentation](https://docs.vllm.com.cn/en/latest/cli/serve/?h=block+size#arguments).
:::

**Service Verification:**

If the service starts successfully, the following startup log will be displayed:

```text
(APIServer pid=<pid>) INFO:     Started server process [<pid>]
(APIServer pid=<pid>) INFO:     Waiting for application startup.
(APIServer pid=<pid>) INFO:     Application startup complete.
```

## 6 Functional Verification

After the service is started, the model can be invoked by sending a prompt:

```shell
curl http://<node0_ip>:<port>/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen3",
        "prompt": "The future of AI is",
        "max_completion_tokens": 50,
        "temperature": 0
    }'
```

Expected result: HTTP 200 with a JSON response containing the `choices` field with generated text.

## 7 Accuracy Evaluation

### Using AISBench

For details, please refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md).


## 8 Performance

### Using AISBench

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

### Using vLLM Benchmark

Refer to [vLLM benchmark](https://docs.vllm.ai/en/latest/benchmarking/) for more details.

There are three `vllm bench` subcommands:

- `latency`: Benchmark the latency of a single batch of requests.
- `serve`: Benchmark the online serving throughput.
- `throughput`: Benchmark offline inference throughput.

Take `serve` as an example:

```shell
export VLLM_USE_MODELSCOPE=True
vllm bench serve \
    --model vllm-ascend/Qwen3-235B-A22B-w8a8 \
    --dataset-name random \
    --random-input 200 \
    --num-prompts 200 \
    --request-rate 1 \
    --save-result \
    --result-dir ./
```

After several minutes, you will get the performance evaluation result.

## 9 Performance Tuning

### 9.1 Recommended Configurations

> **Important**: The configurations provided in this section are validated in specific test environments and are **not** guaranteed to be globally optimal. Actual performance depends on factors such as input/output length distribution, request rate, prefix cache hit rate, hardware configuration, and precision requirements. It is strongly recommended to use the following as a starting point and refer to [Section 9.2](#92-tuning-guidelines) for tuning based on your own workload.

#### Table 1: Scenario Overview

| Scenario | Deployment Mode | *Total NPUs | Weight Version | Key Considerations |
|----------|----------------|-------------|----------------|---------------------|
| High Throughput | Single-Node (TP4, DP4) | 16 (A3) | W8A8 | DP and TP distribute MoE experts across 16 NPUs for maximum throughput |
| High Throughput | PD Disaggregation (3 nodes) | 48 (3×A3) | W8A8 | 3-node PD separation balances prefill and decode resources for high throughput |
| Low Latency | PD Hybrid (TP16) | 16 (A3) | W8A8 | 16-NPU TP minimizes per-token latency with speculative decoding |
| Long Context | PD Hybrid (TP8, CP2) | 16 (A3) | W8A8 | 8-NPU TP with Context Parallelism extends context to 135K tokens |

> **Note**: `*Total NPUs` indicates the total number of NPUs used across all nodes.

#### Table 2: Detailed Node Configuration

| Scenario | Configuration | #NPUs | TP | DP | BS | Concurrency | Max Context Length | MTP Speculation Num | FUSED_MC2 | EP Switch | FC+CP Switch | Async Scheduling |
|----------|---------------|-------|----|----|----|----|-------------|--------------------|-----------|-----------|--------------|------------------|
| High Throughput | Single-Node (TP4, DP4) | 16 | 4 | 4 | 128 | 128 | 40960 | none | On | On | On | On |
| Low Latency | PD Hybrid (TP16) | 16 | 16 | 1 | 128 | 128 | 32768 | 3 | Off | On | On | On |
| Long Context | PD Hybrid (TP8, CP2) | 16 | 8 | 1 | 32 | 32 | 135000 | none | On | On | On | On |

> **Note**: For additional parameter details, refer to the [vLLM Serving Arguments documentation](https://docs.vllm.com.cn/en/latest/cli/serve/?h=block+size#arguments).

<u>Single-node PD Hybrid — High Throughput (TPOT ~50ms):</u>

Single-node PD hybrid deployment optimized for maximum throughput on Atlas 800I A3 (64G × 16):

```bash
export HCCL_IF_IP=<node_ip>
export GLOO_SOCKET_IFNAME=<ifname>
export TP_SOCKET_IFNAME=<ifname>
export HCCL_SOCKET_IFNAME=<ifname>

export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=1024
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

export OMP_NUM_THREADS=1
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl kernel.sched_migration_cost_ns=50000
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export TASK_QUEUE_ENABLE=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

export VLLM_ASCEND_ENABLE_FUSED_MC2=1

vllm serve /mnt/share/weight/Qwen3-235B-A22B-w8a8-rot/ \
    --served-model-name "qwen" \
    --host <host_ip> \
    --port <port> \
    --async-scheduling \
    --tensor-parallel-size 4 \
    --data-parallel-size 4 \
    --data-parallel-size-local 4 \
    --data-parallel-start-rank 0 \
    --data-parallel-address <node_ip> \
    --data-parallel-rpc-port <rpc_port> \
    --enable-expert-parallel \
    --max-num-seqs 128 \
    --max-model-len 32768 \
    --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --quantization ascend \
    --no-enable-prefix-caching \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --additional-config '{"enable_cpu_binding":true}'
```

<u>Single-node PD Hybrid — Low Latency (TPOT ~20ms):</u>

Single-node PD hybrid deployment optimized for low latency with speculative decoding (Eagle3):

```bash
export HCCL_IF_IP=<node_ip>
export GLOO_SOCKET_IFNAME=<ifname>
export TP_SOCKET_IFNAME=<ifname>
export HCCL_SOCKET_IFNAME=<ifname>

export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=1024
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

export OMP_NUM_THREADS=1
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl kernel.sched_migration_cost_ns=50000
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export TASK_QUEUE_ENABLE=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

vllm serve /mnt/share/weight/Qwen3-235B-A22B-w8a8-rot/ \
    --served-model-name "qwen" \
    --host <host_ip> \
    --port <port> \
    --async-scheduling \
    --tensor-parallel-size 16 \
    --data-parallel-size 1 \
    --data-parallel-size-local 1 \
    --data-parallel-start-rank 0 \
    --data-parallel-address <node_ip> \
    --data-parallel-rpc-port <rpc_port> \
    --enable-expert-parallel \
    --max-num-seqs 128 \
    --max-model-len 32768 \
    --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --quantization ascend \
    --no-enable-prefix-caching \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --speculative-config '{"method": "eagle3", "model":"/mnt/share/weight/Qwen3-235B-A22B-EAGLE3-rotated/", "num_speculative_tokens": 3}' \
    --additional-config '{"enable_cpu_binding":true}'
```

<u>Single-node PD Hybrid — Long Context (up to 135K):</u>

Single-node PD hybrid deployment optimized for long context with Context Parallelism and yarn rope-scaling:

```bash
export HCCL_IF_IP=<node_ip>
export GLOO_SOCKET_IFNAME=<ifname>
export TP_SOCKET_IFNAME=<ifname>
export HCCL_SOCKET_IFNAME=<ifname>

export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=1024
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

export OMP_NUM_THREADS=1
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl kernel.sched_migration_cost_ns=50000
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export TASK_QUEUE_ENABLE=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

export VLLM_ASCEND_ENABLE_FUSED_MC2=1

vllm serve /mnt/share/weight/Qwen3-235B-A22B-w8a8-rot/ \
    --served-model-name "qwen" \
    --host <host_ip> \
    --port <port> \
    --tensor-parallel-size 8 \
    --data-parallel-size 1 \
    --decode-context-parallel-size 2 \
    --prefill-context-parallel-size 2 \
    --enable-expert-parallel \
    --cp-kv-cache-interleave-size 128 \
    --max-num-seqs 32 \
    --max-model-len 135000 \
    --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.85 \
    --trust-remote-code \
    --quantization ascend \
    --no-enable-prefix-caching \
    --hf-overrides '{"rope_parameters": {"rope_type":"yarn","rope_theta":1000000,"factor":4,"original_max_position_embeddings":131072}}' \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --additional-config '{"enable_cpu_binding":true}'
```

### 9.2 Tuning Guidelines

#### 9.2.1 General Tuning Reference

Please refer to the [Public Performance Tuning Documentation](../../developer_guide/performance_and_debug/optimization_and_tuning.md) for tuning methods.
Please refer to the [Feature Guide](../../user_guide/support_matrix/feature_matrix.md) for detailed feature descriptions


This section provides complete, annotated launch commands for representative model variants. These configurations have been validated in production environments and can serve as a starting point for your deployment.

## 10 FAQ

For common environment, installation, and general parameter issues, please refer to the [vLLM-Ascend FAQs](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html). This section only covers issues specific to Qwen3-235B-A22B.

### Q: What hardware is required for Qwen3-235B-A22B?

For BF16: 1 Atlas 800I A3 (64G × 16) node, 1 Atlas 800I A2 (64G × 8) node, or 2 Atlas 800I A2 (32G × 8) nodes. For W8A8 quantized version, the hardware requirements are similar.

### Q: How do I enable long context beyond 40K?

Use yarn rope-scaling. For vLLM >= v0.12.0: `--hf-overrides '{"rope_parameters": {"rope_type":"yarn","rope_theta":1000000,"factor":4,"original_max_position_embeddings":32768}}'`. For older versions, use `--rope_scaling`. Model variants like Qwen3-235B-A22B-Instruct-2507 natively support long contexts and don't need this parameter.

### Q: When should I use PD disaggregation vs single-node deployment?

Single-node deployment is simpler and recommended when the model fits within a single node. PD disaggregation separates Prefill and Decode across nodes, enabling higher throughput for large-scale serving. For Qwen3-235B-A22B, three A3 nodes with PD disaggregation can achieve ~3× the throughput of single-node deployment.

### Q: What is the difference between `VLLM_ASCEND_ENABLE_FUSED_MC2=1` and `=2`?

Value `1` enables the base MoE fused operator, suitable for typical EP configurations. Value `2` enables an alternative fusion strategy optimized for large-scale EP (e.g., EP32 in PD disaggregation scenarios). Both are experimental and currently only support W8A8 quantization on Atlas A3 servers.

### Q: When should I use Expert Parallelism?

Expert Parallelism (EP) should always be enabled for Qwen3-235B-A22B (an MoE model) via `--enable-expert-parallel`. It distributes FFN experts across NPUs to reduce per-device computation. EP works alongside TP, where MoE layers use EP and non-MoE layers use TP.

### Q: How do I choose between Context Parallelism and PD Disaggregation?

Context Parallelism (CP) splits the KV cache of a single request across multiple NPUs, suitable for long context scenarios on a single node. PD Disaggregation separates Prefill and Decode across nodes, suitable for high-throughput serving with many concurrent requests. 
