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

### 3.2 Verify Multi-node Communication (Optional)

If multi-node deployment is required, please follow the [Verify Multi-node Communication Environment](../../installation.md#verify-multi-node-communication) guide for communication verification.

## 4 Installation

### 4.1 Docker Image Installation

You can use the official all-in-one Docker image for Qwen3-235B-A22B deployment.

**Docker Pull:**

```{code-block} bash
   :substitutions:

docker pull quay.io/ascend/vllm-ascend:|vllm_ascend_version|
```

**Docker Run:**

```{code-block} bash
   :substitutions:

# Update --device according to your device (Atlas A2: /dev/davinci[0-7], Atlas A3:/dev/davinci[0-15]).
# Update the vllm-ascend image according to your environment.
# Note: download the weight to /root/.cache in advance.
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
export NAME=vllm-ascend

# Note: If using bridge network with docker, expose available ports for multi-node communication in advance.
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
    -it $IMAGE bash
```

The default workdir is `/workspace`. vLLM and vLLM-Ascend code are placed in `/vllm-workspace` and installed in [development mode](https://setuptools.pypa.io/en/latest/userguide/development_mode.html) (`pip install -e`), so changes take effect immediately without requiring a new installation.

To verify the successful installation of the environment, please refer to [installation](../../installation.md).

If deploying a multi-node environment, set up the environment on each node.

### 4.2 Source Code Installation

In addition, if you don't want to use the Docker image as above, you can also build all from source:

- Install `vllm-ascend` from source, refer to [installation](../../installation.md).

If you want to deploy a multi-node environment, you need to set up environment on each node.

## 5 Online Service Deployment

### 5.1 Single-Node Online Deployment

Qwen3-235B-A22B and Qwen3-235B-A22B-W8A8 can both be deployed on 1 Atlas 800I A3 (64G × 16) or 1 Atlas 800I A2 (64G × 8). Quantized versions require `--quantization ascend`.

#### BF16 Models (e.g., Qwen3-235B-A22B)

```shell
# Load model from ModelScope to speed up download
export VLLM_USE_MODELSCOPE=True
# To reduce memory fragmentation and avoid out of memory
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

vllm serve Qwen/Qwen3-235B-A22B \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 8 \
    --served-model-name qwen3 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --async-scheduling
```

#### W8A8 Quantized Model (e.g., Qwen3-235B-A22B-W8A8)

The following example demonstrates best practices for Qwen3-235B-A22B-W8A8 on a single node, targeting 128K context with DP=1, TP=8:

```shell
#!/bin/sh
# Load model from ModelScope to speed up download
export VLLM_USE_MODELSCOPE=True
# To reduce memory fragmentation and avoid out of memory
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=512
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export TASK_QUEUE_ENABLE=1

vllm serve vllm-ascend/Qwen3-235B-A22B-w8a8 \
    --host 0.0.0.0 \
    --port 8000 \
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
- `vllm-ascend/Qwen3-235B-A22B-w8a8` is the default model path, replace this with your actual path.
- [Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B#processing-long-texts) originally only supports 40960 context (`max_position_embeddings`). To run long sequences (e.g., 128K context), yarn rope-scaling is required:
  - For vLLM >= v0.12.0: `--hf-overrides '{"rope_parameters": {"rope_type":"yarn","rope_theta":1000000,"factor":4,"original_max_position_embeddings":32768}}'`
  - For vLLM < v0.12.0: `--rope_scaling '{"rope_type":"yarn","factor":4,"original_max_position_embeddings":32768}'`
- If using weights like [Qwen3-235B-A22B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507) which natively support long contexts, no rope-scaling parameter is needed.
- If the model is not a quantized model, remove the `--quantization ascend` parameter.
:::

After the service is started, you can verify the deployment by sending a request:

```bash
curl http://<node0_ip>:<port>/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen3",
        "prompt": "The future of AI is",
        "max_tokens": 50,
        "temperature": 0
    }'
```

Expected result: HTTP 200 with a JSON response containing the `choices` field with generated text.

### 5.2 Multi-Node MP Deployment (Recommended)

Assume you have Atlas 800I A3 (64G × 16) nodes (or 2× A2 nodes), and want to deploy the Qwen3-235B-A22B model across multiple nodes with Data Parallelism.

The key parameters are explained as follows:

| Parameter | Description |
|-----------|-------------|
| `--tensor-parallel-size` | Degrades the model across multiple NPUs within a single node. For A3 (16 cards) or A2 (8 cards), set to 8 for optimal performance. |
| `--data-parallel-size` | Number of data parallel groups across all nodes. For 2-node deployment, set to 2. |
| `--data-parallel-size-local` | Number of data parallel groups per node (typically 1 per node). |
| `--data-parallel-address` | IP address of the master node (Node 0) for DP communication. |
| `--data-parallel-rpc-port` | RPC port for DP communication across nodes. |
| `--data-parallel-start-rank` | Starting rank for DP. Node 0 uses 0, Worker Node uses 1. |
| `--enable-expert-parallel` | Enables Expert Parallelism for MoE models to distribute experts across GPUs. |
| `HCCL_BUFFSIZE` | HCCL communication buffer size. For MoE models, 1024 improves communication efficiency for multi-node. |
| `--headless` | Worker nodes use this flag to join the server without starting their own API endpoints. |

**Node 0 (Master):**

```shell
#!/bin/sh
# Load model from ModelScope to speed up download
export VLLM_USE_MODELSCOPE=True
# To reduce memory fragmentation and avoid out of memory
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
# nic_name is the network interface name corresponding to local_ip of the current node
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

vllm serve Qwen/Qwen3-235B-A22B \
    --host 0.0.0.0 \
    --port 8000 \
    --data-parallel-size 2 \
    --api-server-count 2 \
    --data-parallel-size-local 1 \
    --data-parallel-address $local_ip \
    --data-parallel-rpc-port 13389 \
    --seed 1024 \
    --served-model-name qwen3 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --max-num-seqs 16 \
    --max-model-len 32768 \
    --max-num-batched-tokens 4096 \
    --trust-remote-code \
    --async-scheduling \
    --gpu-memory-utilization 0.9
```

**Node 1 (Worker):**

```shell
#!/bin/sh
# Load model from ModelScope to speed up download
export VLLM_USE_MODELSCOPE=True
# To reduce memory fragmentation and avoid out of memory
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
# nic_name is the network interface name corresponding to local_ip of the current node
nic_name="xxxx"
local_ip="xxxx"

# The value of node0_ip must be consistent with the value of local_ip set in node0 (master node)
node0_ip="xxxx"

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=1024
export TASK_QUEUE_ENABLE=1

vllm serve Qwen/Qwen3-235B-A22B \
    --host 0.0.0.0 \
    --port 8000 \
    --headless \
    --data-parallel-size 2 \
    --data-parallel-size-local 1 \
    --data-parallel-start-rank 1 \
    --data-parallel-address $node0_ip \
    --data-parallel-rpc-port 13389 \
    --seed 1024 \
    --tensor-parallel-size 8 \
    --served-model-name qwen3 \
    --max-num-seqs 16 \
    --max-model-len 32768 \
    --max-num-batched-tokens 4096 \
    --enable-expert-parallel \
    --trust-remote-code \
    --async-scheduling \
    --gpu-memory-utilization 0.9 \
```

If the service starts successfully, the following information will be displayed on node 0:

```shell
INFO:     Started server process [44610]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Started server process [44611]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

### 5.3 Multi-Node Ray Deployment

Refer to [Ray Distributed (Qwen/Qwen3-235B-A22B)](../features/ray.md).

### 5.4 Prefill-Decode Disaggregation

Refer to [Prefill-Decode Disaggregation Mooncake Verification (Qwen)](../features/pd_disaggregation_mooncake_multi_node.md) for the detailed deployment guide.

For the recommended configuration and startup scripts, see [Section 9.3.2 Three Node A3 — PD Disaggregation](#932-three-node-a3--pd-disaggregation).

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

This section describes the standardized methods and tools for evaluating the output quality (accuracy) of the Qwen3-235B-A22B model. [AISBench](https://gitcode.com/ascend/aisbench) is used as the primary evaluation tool.

### 7.1 Using AISBench

1. Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) for details.

2. For reference, the following are the accuracy results of `Qwen3-235B-A22B-W8A8` on `vllm-ascend:v0.11.0rc0`:

| dataset | version | metric | mode | vllm-api-general-chat |
|----- | ----- | ----- | ----- | -----|
| cevaldataset | - | accuracy | gen | 91.16 |

### 7.2 Using Language Model Evaluation Harness

Using the `gsm8k` dataset as an example, run the accuracy evaluation in online mode:

1. For `lm_eval` installation, please refer to [Using lm_eval](../../developer_guide/evaluation/using_lm_eval.md).
2. Run `lm_eval`:

```shell
lm_eval \
  --model local-completions \
  --model_args model=/root/.cache/vllm-ascend/Qwen3-235B-A22B-W8A8,base_url=http://127.0.0.1:8000/v1/completions,tokenized_requests=False,trust_remote_code=True \
  --tasks gsm8k \
  --output_path ./
```

## 8 Performance

This section describes the standardized methods and tools for evaluating the performance of the Qwen3-235B-A22B model. [AISBench](https://gitcode.com/ascend/aisbench) and the [vLLM Benchmark](https://github.com/vllm-project/vllm/tree/main/benchmarks) suite are used for performance evaluation.

### 8.1 Using AISBench

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

### 8.2 Using vLLM Benchmark

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

> **Important**: The configurations provided in this section are validated in specific test environments and are **not** guaranteed to be globally optimal. Actual performance depends on factors such as input/output length distribution, request rate, prefix cache hit rate, hardware configuration, and precision requirements. It is strongly recommended to use the following as a starting point and refer to [Section 9.2](#92-tuning-guidelines) for tuning based on your own workload.

### 9.1 Recommended Configurations

#### Table 1: Scenario Overview

| Scenario | Deployment Mode | Total NPUs | Weight Version | Key Considerations |
|----------|----------------|-------------|----------------|---------------------|
| High Throughput<br>(3.5K → 1.5K) | Single-node | 16 (A3) | Qwen3-235B-A22B-W8A8 | For high throughput with MoE models, try tuning DP and TP sizes (DP=4, TP=4), enabling FlashComm (`VLLM_ASCEND_ENABLE_FLASHCOMM1=1`), Fused MC2 (`VLLM_ASCEND_ENABLE_FUSED_MC2=1`), and expanding `max-num-seqs` for larger batch concurrency |
| High Throughput<br>(3.5K → 1.5K) | PD Disaggregation (3 nodes) | 48 (3×A3) | Qwen3-235B-A22B-W8A8 | For PD disaggregation, try tuning prefill/decode TP and DP ratios, using Fused MC2=2 for large-scale EP, and tuning `max-num-batched-tokens` for prefill vs decode balance |
| Low Latency<br>(TPOT bound) | PD Hybrid (single node) | 16 (A3) | Qwen3-235B-A22B-W8A8 | For low latency, try reducing TP to 16 with DP=1 (full model on all NPUs), enabling speculative decoding (Eagle3), and disabling unnecessary optimizations |
| Long Context<br>(up to 135K) | PD Hybrid (single node) | 16 (A3) | Qwen3-235B-A22B-W8A8 | For long context scenarios, try enabling Context Parallelism (`--decode-context-parallel-size 2`), yarn rope-scaling (`--hf-overrides`), and reducing `max-num-seqs` to fit KV cache |

#### Table 2: Detailed Node Configuration

| Scenario | Total NPUs | Tensor Parallel | Data Parallel | Expert Parallel | Context Parallel | Max Num Seqs | Max Model Len | Max Batched Tokens | Speculative | FlashComm | Fused MC2 |
|----------|-----------|----------------|---------------|----------------|-----------------|-------------|--------------|-------------------|-------------|-----------|-----------|
| Single-Node High Throughput | 16 | 4 | 4 | Yes | - | 128 | 40960 | 16384 | No | Yes | 1 |
| PD Disaggregation High Throughput | 48 | P:8 / D:4 | P:2 / D:8 | Yes | - | P:24 / D:128 | 40960 | P:16384 / D:256 | No | Yes | 2 |
| PD Hybrid Low Latency | 16 | 16 | 1 | Yes | - | 128 | 32768 | 16384 | Eagle3 | Yes | No |
| PD Hybrid Long Context | 16 | 8 | 1 | Yes | 2 | 32 | 135000 | 16384 | No | Yes | 1 |

### 9.2 Tuning Guidelines

#### 9.2.1 General Guidelines

- **DP vs TP ratio**: For MoE models, increasing DP reduces inter-device communication per replica but increases total memory usage. A good starting point is DP × TP = total NPUs per node.
- **Expert Parallelism**: Always enable `--enable-expert-parallel` for MoE models to distribute experts efficiently.
- **FlashComm**: Use `VLLM_ASCEND_ENABLE_FLASHCOMM1=1` when TP > 1 and concurrency is high (e.g., `max-num-seqs` > 32).
- **Fused MC2**: Use `VLLM_ASCEND_ENABLE_FUSED_MC2=1` for W8A8 quantized MoE on Atlas A3. Use `=2` for large-scale EP (EP ≥ 16).
- **CPU Binding**: Use `--additional-config '{"enable_cpu_binding":true}'` for better NUMA locality.
- **JEMalloc**: Preload `libjemalloc.so.2` via `LD_PRELOAD` for better memory management.

#### 9.2.2 Optimization Highlights

The following optimizations are enabled by default and require no additional configuration:

| Optimization Technique | Technical Principle | Performance Benefit |
|------------------------|--------------------|---------------------|
| Rope Optimization | The cos_sin_cache and indexing operations of positional encoding are executed only in the first layer, and subsequent layers reuse them directly | Reduces redundant computation during the decoding phase, accelerating inference |
| FullGraph Optimization | Captures and replays the entire decoding graph at once using `compilation_config={"cudagraph_mode":"FULL_DECODE_ONLY"}` | Significantly reduces scheduling latency, stabilizes multi-device performance |

The following advanced optimizations require explicit enablement:

| Optimization Technique | Enablement Method | Applicable Scenarios | Precautions |
|------------------------|-------------------|----------------------|-------------|
| FlashComm_v1 | `export VLLM_ASCEND_ENABLE_FLASHCOMM1=1` | High-concurrency, TP > 1 scenarios with MoE models | Currently only supported for MoE in scenarios where TP > 1 |
| Fused MC2 | `export VLLM_ASCEND_ENABLE_FUSED_MC2=1` (or `=2`) | W8A8 quantization on Atlas A3 servers | Experimental. Value `2` is recommended for EP32 scenarios (e.g., PD disaggregation) |
| Asynchronous Scheduling | `--async-scheduling` | Large-scale models, high-concurrency scenarios | Should be used in coordination with FullGraph optimization |
| Weight Prefetch | `--additional-config '{"weight_prefetch_config":{"enabled":true}}'` | High-throughput decode with MoE | Reduces weight loading latency during expert routing |
| CPU Binding | `--additional-config '{"enable_cpu_binding":true}'` | All NPU deployments | Improves NUMA locality and reduces memory access latency |
| Context Parallelism | `--decode-context-parallel-size N` / `--prefill-context-parallel-size N` | Long context scenarios beyond 64K | Splits KV cache across NPUs for extended context |

##### Fused MC2

Setting `VLLM_ASCEND_ENABLE_FUSED_MC2=1` enables MoE fused operators that reduce time consumption in both prefill and decode. For large-scale EP scenarios (e.g., EP32 in PD disaggregation), use `VLLM_ASCEND_ENABLE_FUSED_MC2=2` which enables a different fusion strategy optimized for higher EP degrees. This is an experimental feature currently only supporting W8A8 quantization on Atlas A3 servers.

##### Long Context with Yarn Rope-Scaling

Qwen3-235B-A22B originally supports 40960 context. For 128K or longer contexts, yarn rope-scaling is required:

- For vLLM >= v0.12.0: `--hf-overrides '{"rope_parameters": {"rope_type":"yarn","rope_theta":1000000,"factor":4,"original_max_position_embeddings":32768}}'`
- For vLLM < v0.12.0: `--rope_scaling '{"rope_type":"yarn","factor":4,"original_max_position_embeddings":32768}'`

If using weights like [Qwen3-235B-A22B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507) which natively support long contexts, no rope-scaling parameter is needed.

##### HCCL_BUFFSIZE

For large MoE models like Qwen3-235B-A22B, setting `HCCL_BUFFSIZE=512` or `1024` can improve communication efficiency. Smaller values (512) are typically sufficient for single-node deployments; larger values (1024) may benefit multi-node scenarios.

### 9.3 Reference Configurations

#### 9.3.1 Single Node A3 (64G × 16) — High Throughput

This configuration targets maximum throughput on a single Atlas 800I A3 node with W8A8 quantization, DP=4, TP=4, and FlashComm:

```shell
#!/bin/sh
# Load model from ModelScope to speed up download
export VLLM_USE_MODELSCOPE=True
# To reduce memory fragmentation and avoid out of memory
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=512
export HCCL_OP_EXPANSION_MODE="AIV"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_ENABLE_FUSED_MC2=1
export TASK_QUEUE_ENABLE=1

vllm serve vllm-ascend/Qwen3-235B-A22B-w8a8 \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 4 \
    --data-parallel-size 4 \
    --seed 1024 \
    --quantization ascend \
    --served-model-name qwen3 \
    --max-num-seqs 128 \
    --max-model-len 40960 \
    --max-num-batched-tokens 16384 \
    --enable-expert-parallel \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --no-enable-prefix-caching \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --async-scheduling
```

Benchmark:

```shell
vllm bench serve --model qwen3 \
    --tokenizer vllm-ascend/Qwen3-235B-A22B-w8a8 \
    --ignore-eos \
    --dataset-name random \
    --random-input-len 3584 \
    --random-output-len 1536 \
    --num-prompts 800 \
    --max-concurrency 160 \
    --request-rate 24 \
    --host 0.0.0.0 \
    --port 8000
```

Reference results (input 3584 / output 1536, max concurrency 160):

| num_requests | concurrency | mean TTFT(ms) | mean TPOT(ms) | output token throughput (tok/s) |
|----- | ----- | ----- | ----- | -----|
| 720 | 144 | 4717.45 | 48.69 | 2761.72 |

:::{note}
- `export VLLM_ASCEND_ENABLE_FUSED_MC2=1` enables MoE fused operators. This is an experimental feature which only supports W8A8 quantization on Atlas A3 servers.
- Prefix cache is disabled because of random datasets. Enable prefix cache if requests have long common prefixes.
:::

#### 9.3.2 Three Node A3 — PD Disaggregation

On three Atlas 800I A3 (64G × 16) servers, the recommended setup uses one node as the Prefill instance and two nodes as the Decode instance.

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
    --host 0.0.0.0 \
    --port 8000 \
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
    --host 0.0.0.0 \
    --port 8000 \
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
    --host 0.0.0.0 \
    --port 8000 \
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

**PD Proxy:**

```shell
python load_balance_proxy_server_example.py \
    --port 12347 \
    --prefiller-hosts prefill_node_1_ip \
    --prefiller-port 8000 \
    --decoder-hosts decode_node_1_ip \
    --decoder-ports 8000
```

Benchmark:

```shell
vllm bench serve --model qwen3 \
    --tokenizer vllm-ascend/Qwen3-235B-A22B-w8a8 \
    --ignore-eos \
    --dataset-name random \
    --random-input-len 3584 \
    --random-output-len 1536 \
    --num-prompts 2880 \
    --max-concurrency 576 \
    --request-rate 8 \
    --host 0.0.0.0 \
    --port 12347
```

Reference results (input 3584 / output 1536, max concurrency 576):

| num_requests | concurrency | mean TTFT(ms) | mean TPOT(ms) | output token throughput (tok/s) |
|----- | ----- | ----- | ----- | -----|
| 2880 | 576 | 3735.98 | 52.07 | 8593.44 |

:::{note}
- Setting `export VLLM_ASCEND_ENABLE_FUSED_MC2=2` is recommended for this scenario (EP32) for optimized MoE fusion.
:::

#### 9.3.3 PD Hybrid — High Throughput (TPOT ~50ms)

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
    --host 0.0.0.0 \
    --port 20002 \
    --async-scheduling \
    --tensor-parallel-size 4 \
    --data-parallel-size 4 \
    --data-parallel-size-local 4 \
    --data-parallel-start-rank 0 \
    --data-parallel-address <node_ip> \
    --data-parallel-rpc-port 13395 \
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

#### 9.3.4 PD Hybrid — Low Latency (TPOT ~20ms)

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
    --host 0.0.0.0 \
    --port 20002 \
    --async-scheduling \
    --tensor-parallel-size 16 \
    --data-parallel-size 1 \
    --data-parallel-size-local 1 \
    --data-parallel-start-rank 0 \
    --data-parallel-address <node_ip> \
    --data-parallel-rpc-port 13395 \
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

#### 9.3.5 PD Hybrid — Long Context (up to 135K)

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
    --host 0.0.0.0 \
    --port 20002 \
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

:::{note}
- **High Throughput**: Enable FlashComm, Fused MC2, CPU binding. Use DP=4, TP=4 for balanced throughput on A3.
- **Low Latency**: Use TP=16 to maximize single-replica compute. Enable Eagle3 speculative decoding to reduce effective TPOT.
- **Long Context**: Use Context Parallelism (CP=2) with TP=8 to split KV cache across NPUs. Yarn rope-scaling (`--hf-overrides`) is required for 135K context. Reduce `gpu-memory-utilization` to 0.85 to accommodate larger KV cache.
- Adjust paths (`/mnt/share/weight/...`, `--host`, `--port`) and network interface names to match your environment.
:::  

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
