# Qwen3-235B-A22B

## 1 Introduction

Qwen3 is the latest generation of large language models in Qwen series, offering a comprehensive suite of dense and mixture-of-experts (MoE) models. Built upon extensive training, Qwen3 delivers groundbreaking advancements in reasoning, instruction-following, agent capabilities, and multilingual support. Qwen3-235B-A22B is the largest MoE variant, featuring 235B total parameters with 22B activated per token.

This document will demonstrate the main validation steps for Qwen3-235B-A22B in the vLLM-Ascend environment, including supported features, environment preparation, single-node and multi-node deployment, accuracy and performance evaluation.

The Qwen3-235B-A22B model is first supported in **v0.8.4rc2**. This document is validated and written based on **vLLM-Ascend v0.13.0**. All **v0.13.0 and later versions** can run stably. To use the latest features (e.g., PD separation, fused MC2), it is recommended to use v0.13.0 or a later version.

## 2 Supported Features

Please refer to the [Supported Features List](../../user_guide/support_matrix/supported_models.md) for the model support matrix.

Please refer to the [Feature Guide](../../user_guide/feature_guide/index.md) for feature configuration information.

## 3 Environment Preparation

### 3.1 Model Weight

The following model variants are available. It is recommended to download the model weight to a shared directory across multiple nodes (e.g., `/root/.cache/`).

| Model | Hardware Requirement | Download |
|-------|---------------------|----------|
| Qwen3-235B-A22B (BF16) | 1 Atlas 800I A3 (64G × 16) node, 1 Atlas 800I A2 (64G × 8) node, or 2 Atlas 800I A2 (32G × 8) nodes | [Download](https://www.modelscope.cn/models/Qwen/Qwen3-235B-A22B) |
| Qwen3-235B-A22B-W8A8 | 1 Atlas 800I A3 (64G × 16) node, 1 Atlas 800I A2 (64G × 8) node, or 2 Atlas 800I A2 (32G × 8) nodes | [Download](https://modelscope.cn/models/vllm-ascend/Qwen3-235B-A22B-W8A8) |

These are the recommended numbers of cards, which can be adjusted according to the actual situation.

### 3.2 Verify Multi-node Communication (Optional)

If multi-node deployment is required, please follow the [Verify Multi-node Communication Environment](../../installation.md#verify-multi-node-communication) guide for communication verification.

## 4 Installation

:::::{tab-set}
::::{tab-item} Use Docker image

Select an image based on your machine type and start the Docker image on your node. Refer to [Using Docker](../../installation.md#set-up-using-docker) for details.

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

::::
::::{tab-item} Build from source

You can build all from source.

- Install `vllm-ascend`, refer to [set up using python](../../installation.md#set-up-using-python).

::::
:::::

If deploying a multi-node environment, set up the environment on each node.

## 5 Online Service Deployment

### 5.1 Single-Node Online Deployment

Qwen3-235B-A22B and Qwen3-235B-A22B-W8A8 can both be deployed on 1 Atlas 800I A3 (64G × 16) or 1 Atlas 800I A2 (64G × 8). Quantized versions require `--quantization ascend`.

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

### 5.2 Multi-Node MP Deployment (Recommended)

Assume you have Atlas 800I A3 (64G × 16) nodes (or 2× A2 nodes), and want to deploy the Qwen3-235B-A22B model across multiple nodes with Data Parallelism.

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

Refer to [Prefill-Decode Disaggregation Mooncake Verification (Qwen)](../features/pd_disaggregation_mooncake_multi_node.md).

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

1. Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) for details.

2. For reference, the following are the accuracy results of `Qwen3-235B-A22B-W8A8` on `vllm-ascend:v0.11.0rc0`:

| dataset | version | metric | mode | vllm-api-general-chat |
|----- | ----- | ----- | ----- | -----|
| cevaldataset | - | accuracy | gen | 91.16 |

### Using Language Model Evaluation Harness

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

## 9 Best Practices

The following reference configurations have been validated for optimal performance with Qwen3-235B-A22B.

### 9.1 Single Node A3 (64G × 16) — High Throughput

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
- `export VLLM_ASCEND_ENABLE_FUSED_MC2=1` enables MoE fused operators that reduce time consumption of MoE in both prefill and decode. This is an experimental feature which only supports W8A8 quantization on Atlas A3 servers. If you encounter problems, disable it by setting `VLLM_ASCEND_ENABLE_FUSED_MC2=0`.
- Prefix cache is disabled because of random datasets. Enable prefix cache if requests have long common prefixes.
:::

### 9.2 Three Node A3 — PD Disaggregation

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
- We recommend setting `export VLLM_ASCEND_ENABLE_FUSED_MC2=2` for this scenario (typically EP32 for Qwen3-235B). This enables a different MoE fusion operator optimized for large-scale EP.
:::

## 10 Performance Tuning

### 10.1 Key Optimization Points

In this section, we introduce the key optimization points that can significantly improve the performance of Qwen3-235B-A22B.

#### 10.1.1 Basic Optimizations

The following optimizations are enabled by default and require no additional configuration:

| Optimization Technique | Technical Principle | Performance Benefit |
|------------------------|--------------------|---------------------|
| Rope Optimization | The cos_sin_cache and indexing operations of positional encoding are executed only in the first layer, and subsequent layers reuse them directly | Reduces redundant computation during the decoding phase, accelerating inference |
| FullGraph Optimization | Captures and replays the entire decoding graph at once using `compilation_config={"cudagraph_mode":"FULL_DECODE_ONLY"}` | Significantly reduces scheduling latency, stabilizes multi-device performance |

#### 10.1.2 Advanced Optimizations (Require Explicit Enablement)

| Optimization Technique | Technical Principle | Enablement Method | Applicable Scenarios | Precautions |
|------------------------|--------------------|-------------------|----------------------|-------------|
| FlashComm_v1 | Decomposes traditional Allreduce into Reduce-Scatter and All-Gather, reducing RMSNorm computation dimensions | `export VLLM_ASCEND_ENABLE_FLASHCOMM1=1` | High-concurrency, TP > 1 scenarios with MoE models | Currently only supported for MoE in scenarios where TP > 1 |
| Fused MC2 | MoE fused operators that reduce time consumption in both prefill and decode | `export VLLM_ASCEND_ENABLE_FUSED_MC2=1` (or `2` for large-scale EP) | W8A8 quantization on Atlas A3 servers | Experimental feature. Value `2` is recommended for EP32 scenarios (e.g., Qwen3-235B PD disaggregation) |
| Asynchronous Scheduling | Non-blocking task scheduling to improve concurrent processing capability | `--async-scheduling` | Large-scale models, high-concurrency scenarios | Should be used in coordination with FullGraph optimization |

### 10.2 Parameter Reference

The key parameters used in deployment commands are explained below:

- **`--data-parallel-size` and `--tensor-parallel-size`**: Common settings for Data Parallelism (DP) and Tensor Parallelism (TP). DP replicates the model across groups; TP shards layers within a group.
- **`--max-model-len`**: The context length — the maximum value of input plus output for a single request.
- **`--max-num-seqs`**: The maximum number of requests that each DP group can process concurrently. Requests exceeding this limit will wait in the queue. For performance benchmarking, ensure `max-num-seqs` × `data-parallel-size` >= the actual target concurrency.
- **`--max-num-batched-tokens`**: The maximum number of tokens processed in a single step. With Chunked Prefill / SplitFuse enabled by default in vLLM V1:
  - Requests with input longer than this value are split into multiple rounds.
  - Decode requests are prioritized; prefill requests are scheduled only if capacity is available.
  - Larger values reduce overall latency but increase activation memory pressure.
- **`--gpu-memory-utilization`**: The proportion of HBM used for inference. KV cache size = `gpu-memory-utilization` × HBM size − peak memory usage (from warm-up profiling). Higher values allow more KV cache but risk OOM due to EP load imbalance. Default is `0.9`.
- **`--enable-expert-parallel`**: Enables Expert Parallelism (EP) for MoE models. vLLM does not support mixing ETP and EP; MoE layers use either pure EP or pure TP.
- **`--no-enable-prefix-caching`**: Disables prefix caching. Remove this option to enable it for workloads with shared prefixes.
- **`--quantization ascend`**: Enables Ascend quantization. Remove for BF16 models.
- **`--compilation-config`**: Contains graph mode configurations. `"cudagraph_mode": "FULL_DECODE_ONLY"` is recommended. `"cudagraph_capture_sizes"` specifies graph capture levels; the default (evenly distributed values from 1 to `max-num-seqs`) is usually sufficient.

### 10.3 Optimization Highlights

#### Fused MC2

Setting `VLLM_ASCEND_ENABLE_FUSED_MC2=1` enables MoE fused operators that reduce time consumption in both prefill and decode. For large-scale EP scenarios (e.g., EP32 in PD disaggregation), use `VLLM_ASCEND_ENABLE_FUSED_MC2=2` which enables a different fusion strategy optimized for higher EP degrees. This is an experimental feature currently only supporting W8A8 quantization on Atlas A3 servers.

#### Long Context with Yarn Rope-Scaling

Qwen3-235B-A22B originally supports 40960 context. For 128K or longer contexts, yarn rope-scaling is required. The appropriate `--hf-overrides` or `--rope_scaling` parameter must be set based on your vLLM version. See the notes in [5.1](#51-single-node-online-deployment) for exact parameters.

#### HCCL_BUFFSIZE

For large MoE models like Qwen3-235B-A22B, setting `HCCL_BUFFSIZE=512` or `1024` can improve communication efficiency. Smaller values (512) are typically sufficient for single-node deployments; larger values (1024) may benefit multi-node scenarios.

## 11 FAQ

### Q: What hardware is required for Qwen3-235B-A22B?

For BF16: 1 Atlas 800I A3 (64G × 16) node, 1 Atlas 800I A2 (64G × 8) node, or 2 Atlas 800I A2 (32G × 8) nodes. For W8A8 quantized version, the hardware requirements are similar.

### Q: How do I enable long context beyond 40K?

Use yarn rope-scaling. For vLLM >= v0.12.0: `--hf-overrides '{"rope_parameters": {"rope_type":"yarn","rope_theta":1000000,"factor":4,"original_max_position_embeddings":32768}}'`. For older versions, use `--rope_scaling`. Model variants like Qwen3-235B-A22B-Instruct-2507 natively support long contexts and don't need this parameter.

### Q: When should I use PD disaggregation vs single-node deployment?

Single-node deployment is simpler and recommended when the model fits within a single node. PD disaggregation separates Prefill and Decode across nodes, enabling higher throughput for large-scale serving. For Qwen3-235B-A22B, three A3 nodes with PD disaggregation can achieve ~3× the throughput of single-node deployment.

### Q: What is the difference between `VLLM_ASCEND_ENABLE_FUSED_MC2=1` and `=2`?

Value `1` enables the base MoE fused operator, suitable for typical EP configurations. Value `2` enables an alternative fusion strategy optimized for large-scale EP (e.g., EP32 in PD disaggregation scenarios). Both are experimental and currently only support W8A8 quantization on Atlas A3 servers.
