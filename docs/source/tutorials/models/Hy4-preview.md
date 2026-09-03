# Hy4-preview (Experimental)

## 1 Introduction

**Hy4 Preview** is a next-generation flagship Mixture-of-Experts (MoE) model developed by the Tencent Hy Team. The model has a total of **770B** parameters, with approximately **49B** parameters activated per token.

The model backbone contains **78 layers**: the first layer adopts a standard Dense FFN, and the remaining **77 layers all adopt the MoE structure**. Each MoE layer contains **256 Routed Experts** and **1 Shared Expert**. For each token, the model selects the **Top-8** experts from the 256 routed experts for computation, while always activating the shared expert.

In addition to the backbone, the model also has **1 native MTP (Multi-Token Prediction) layer** built in to support speculative decoding. The MTP layer has a total of about **10B** parameters, with about **0.7B** parameters activated per token.

This document describes how to quickly get started with Hy4 model inference deployment on Ascend NPU using vLLM-Ascend, based on W8A8 quantized weights.

!!! warning

    **Current status and constraints**

    - Hy4 Preview is provided **out-of-the-box** through the official Docker image `quay.io/ascend/vllm-ascend:hy4-a3`. The supporting code has **not yet been merged** into the vLLM-Ascend repository, so installing it from source (`pip install` or building from source) is **not supported** for this model yet.
    - Only **Atlas 800I A3 (A3)** is supported now. Other Ascend hardware (e.g., Atlas 800I A2) is not supported for Hy4 Preview.
    - The features listed in [Supported Features](#2-supported-features) are only those enabled by the verified deployment commands in this document, and do **not** imply that all features are supported for Hy4 Preview. This is an early-access version; performance optimization and reliability validation are still in progress (see [Declaration](#9-declaration)).

## 2 Supported Features

Refer to [Supported Features List](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [Feature Guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

The features below are the ones enabled by the verified deployment commands in [Chapter 5](#5-online-service-deployment).

| Feature | Description | Configuration |
| --- | --- | --- |
| Tensor Parallel (TP) | Splits the model across all 16 NPUs within a node. | `--tensor-parallel-size 16` |
| Expert Parallel (EP) | Distributes the MoE experts (256 routed + 1 shared per layer) across NPUs. | `--enable-expert-parallel` |
| Data Parallel (DP) | Multi-node DP across 2 nodes, extending the context to 96K. | `--data-parallel-size 2` |
| W8A8 Quantization | Loads the W8A8 quantized weights (Ascend quantization). | `--quantization ascend` |
| Automatic Prefix Caching | Reuses the KV cache for shared prompt prefixes. | `--enable-prefix-caching` |
| Speculative Decoding (MTP) | Uses the native MTP layer for Multi-Token Prediction. | `--speculative-config '{"method": "mtp", "num_speculative_tokens": 3}'` |
| Decode Graph Capture | Captures the decode phase into a graph to reduce kernel launch overhead. | `--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'` |
| NPUGraph EX | Ascend-specific graph execution optimization. | `--additional-config '{"ascend_compilation_config": {"enable_npugraph_ex": true}}'` |

> **Note:** The features listed above are the ones enabled in the verified deployment commands in this document. Other features, such as LoRA, Pipeline Parallel, and Prefill-Decode Disaggregation, are not verified in this document.

## 3 Prerequisites

### 3.1 Model Weight

| Model | Weight |
| --- | --- |
| Hy4-preview | <https://huggingface.co/tencent/Hy4-preview> |
| Hy4-preview-w8a8 | <https://www.modelscope.cn/models/Eco-Tech/Hy4-preview-w8a8> |

This document uses the quantized [Hy4-preview-w8a8](https://www.modelscope.cn/models/Eco-Tech/Hy4-preview-w8a8) weights, which are about 762 GB. Download the weights to the local disk.

### 3.2 Hardware and Software Preparation

| Hardware | Ascend HDK | CANN version | torch_npu version | vLLM | vLLM-Ascend version |
| --- | --- | --- | --- | --- | --- |
| Atlas 800I A3 (A3) | 25.5.0 (recommended) | CANN 9.0.1 | 2.10.0.post2 | Based on v0.23.0 | Based on v0.23.0 |

Hy4 Preview currently supports **only the Atlas 800I A3 (A3)** hardware; other Ascend hardware (e.g., Atlas 800I A2) is not supported.

Before getting started, confirm that the firmware/driver is installed correctly. The recommended HDK version is 25.5.0. Run the following command to confirm; the version shown in the output header is the HDK version.

```bash
npu-smi info
```

Because the weights are large, a single A3 node only supports about 1K context. If you need to test long sequences such as 32K, it is recommended to use 2 A3 nodes.

### 3.3 Verify Multi-node Communication (Optional)

If you want to deploy a multi-node environment, verify multi-node communication as described in [Verify inter-node connectivity](../../getting_started/installation.md#installation-multi-node-interconnect).

## 4 Installation

### 4.1 Docker Image Installation

#### Download Docker Image

```bash
docker pull quay.io/ascend/vllm-ascend:hy4-a3
```

After a successful download, check the existing images:

```bash
docker images | grep hy4-a3
```

The output should contain the downloaded `quay.io/ascend/vllm-ascend:hy4-a3`.

#### Create Docker Container

Use the following command to create the container. The Atlas 800I A3 is a 16-card device, so you need to mount `/dev/davinci[0-15]` and the management devices. Also mount the driver library and the weight directory (this guide assumes the weights are stored on the host at `/mnt/weight` and mounted to the same path inside the container).

```bash
export IMAGE=quay.io/ascend/vllm-ascend:hy4-a3
export CONTAINER_NAME=vllm_ascend_hy4

# Generate NPU device mount parameters (Atlas 800I A3: /dev/davinci[0-15])
DEVICES=""
for i in $(seq 0 15); do
    DEVICES="${DEVICES} --device /dev/davinci${i}"
done

docker run -itd \
    --name ${CONTAINER_NAME} \
    --shm-size=1000g \
    ${DEVICES} \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /mnt/weight:/mnt/weight \
    ${IMAGE} bash
```

The `-v` option maps directories. **You should map the weight directory into the container** so that the weights can be accessed inside the container. The local weight path is `/mnt/weight`; modify it according to the actual situation.

Enter the container interactive environment:

```bash
docker exec -it ${CONTAINER_NAME} bash
```

### 4.2 Source Code Installation

!!! note

    Hy4 Preview is currently provided **out-of-the-box** via the `quay.io/ascend/vllm-ascend:hy4-a3` Docker image. Because the supporting code has **not yet been merged** into the vLLM-Ascend repository, source-code installation is **not supported** for this model yet; please use the Docker image as described in [4.1 Docker Image Installation](#41-docker-image-installation).

## 5 Online Service Deployment

### 5.1 Single-Node Online Deployment

A3 single-node TP=16 full-card inference is suitable for validation and low-concurrency short-sequence scenarios. Execute the following after entering the container interactive environment:

```bash
export HCCL_BUFFSIZE=128
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_LOGGING_LEVEL=INFO

vllm serve /path/to/Hy4-preview-w8a8 \
  --host 127.0.0.1 --port 8000 \
  --tensor-parallel-size 16 \
  --served-model-name hy4 \
  --max-num-seqs 8 \
  --max-model-len 512 \
  --max-num-batched-tokens 512 \
  --enable-expert-parallel \
  --trust-remote-code \
  --quantization ascend \
  --gpu-memory-utilization 0.92 \
  --seed 1024 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --speculative-config '{"method": "mtp","num_speculative_tokens": 3}' \
  --additional-config '{"enable_mlapo": true, "enable_fused_mc2": 1, "ascend_compilation_config": {"enable_npugraph_ex": true}}'
```

### 5.2 Multi-Node Co-Located Deployment

DP=2 cross-machine deployment with TP=16 per node extends the context to 96K. Node 1 (Node1, demo IP `192.168.1.1`) hosts DP rank 0 and the API, and Node 2 (Node2, demo IP `192.168.1.2`) is a headless worker. Both nodes share the host network via `--net=host`. Use the same command as the A3 single-node deployment, executed inside the container.

Startup order:

1. First execute the following script in the container on Node 1 (rank 0, including the API);
2. After seeing the `Started DP Coordinator process` log, execute the corresponding script in the container on Node 2;
3. After about 20 minutes, seeing `Application startup complete` indicates that the service is ready. The startup time is related to the disk read/write speed of the weights, etc.

**Node 1 script:**

```bash
#!/bin/bash
set -u

# === Node network configuration (adjust the NIC name and IP according to the actual environment) ===
NODE1_IP="192.168.1.1"       # Node 1 IP, change to the actual machine IP
NODE2_IP="192.168.1.2"       # Node 2 IP, change to the actual machine IP
NIC="eth0"                   # Change to the NIC name corresponding to NODE1_IP, viewable via the `ifconfig` command
DP_RPC_PORT=12890

# === Service configuration ===
MODEL="/path/to/Hy4-preview-w8a8"  # Change to the actual weight path
PORT=8000

# === Environment variables ===
export HCCL_BUFFSIZE=128
export HCCL_IF_IP=${NODE1_IP}
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_SOCKET_IFNAME=${NIC}
export GLOO_SOCKET_IFNAME=${NIC}
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

vllm serve ${MODEL} \
  --host 0.0.0.0 \
  --port ${PORT} \
  --data-parallel-size 2 \
  --data-parallel-size-local 1 \
  --data-parallel-address ${NODE1_IP} \
  --data-parallel-rpc-port ${DP_RPC_PORT} \
  --tensor-parallel-size 16 \
  --served-model-name hy4 \
  --enable-expert-parallel \
  --max-num-seqs 256 \
  --max-model-len 96000 \
  --max-num-batched-tokens 1024 \
  --trust-remote-code \
  --quantization ascend \
  --gpu-memory-utilization 0.85 \
  --enable-prefix-caching \
  --seed 1024 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --speculative-config '{"method": "mtp","num_speculative_tokens": 3}' \
  --additional-config '{"enable_mlapo": true, "enable_fused_mc2": 1, "ascend_compilation_config": {"enable_npugraph_ex": true}}'
```

**Node 2 script:**

```bash
#!/bin/bash
set -u

# === Node network configuration ===
NODE1_IP="192.168.1.1"       # Node 1 IP, change to the actual machine IP
NODE2_IP="192.168.1.2"       # Node 2 IP, change to the actual machine IP
NIC="eth0"                   # Change to the NIC name corresponding to NODE2_IP, viewable via the `ifconfig` command
DP_RPC_PORT=12890

# === Service configuration (consistent with Node 1) ===
MODEL="/path/to/Hy4-preview-w8a8"  # Change to the actual weight path

# === Environment variables (consistent with Node 1) ===
export HCCL_BUFFSIZE=128
export HCCL_IF_IP=${NODE2_IP}
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_SOCKET_IFNAME=${NIC}
export GLOO_SOCKET_IFNAME=${NIC}
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

vllm serve ${MODEL} \
  --headless \
  --data-parallel-start-rank 1 \
  --data-parallel-size 2 \
  --data-parallel-size-local 1 \
  --data-parallel-address ${NODE1_IP} \
  --data-parallel-rpc-port ${DP_RPC_PORT} \
  --tensor-parallel-size 16 \
  --enable-expert-parallel \
  --max-num-seqs 256 \
  --max-model-len 96000 \
  --max-num-batched-tokens 1024 \
  --trust-remote-code \
  --quantization ascend \
  --gpu-memory-utilization 0.85 \
  --enable-prefix-caching \
  --seed 1024 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --speculative-config '{"method": "mtp","num_speculative_tokens": 3}' \
  --additional-config '{"enable_mlapo": true, "enable_fused_mc2": 1, "ascend_compilation_config": {"enable_npugraph_ex": true}}'
```

> Note: `--data-parallel-address` is set to the local IP (i.e., the coordinator) on Node 1, and to the head node IP on Node 2; `--data-parallel-start-rank` is only set on the headless node.

Because the weights are large, the load time is related to the disk read speed, and the first load usually takes about 10 minutes. The following logs indicate that the service has been started successfully:

```text
INFO:     Started server process [PID]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

## 6 Functional Verification

Send an inference request to test the inference service, refer to the following command:

```bash
curl -sS -X POST http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
    "model":"hy4",
    "messages":[{
        "role":"user",
        "content":"Who are you?"
    }],
    "max_tokens":256,
    "temperature":0
}'
```

The server prints the log of the received request, similar to:

```text
INFO:     [IP]:[PORT] - "POST /v1/chat/completions HTTP/1.1" 200 OK
```

The expected result is as follows:

```text
{"id":"chatcmpl-84c57a059e1d510c","object":"chat.completion","created":1787890448,"model":"hy4","choices":[{"index":0,"message":{"role":"assistant","content":"Okay, I'm the user asking \"Who are you?\" and I need to respond according to my identity and behavioral guidelines. First, my identity is Hunyuan, a large model developed by Tencent. The user's question directly asks about my identity, so I should clearly state who I am without extra information. Based on the core constraints, when the question involves identity confirmation, I need to clearly explain my identity and developer. Also, the user didn't mention any other functions or context, so keeping it concise is fine. Check if there's anything that needs clarification, like whether the user wants to know about features, but the current question is simply an identity inquiry, so a direct answer is enough. Make sure the response meets the format requirements, no markdown, and keep it natural and conversational.</think:opensource>I am Hunyuan, a large model developed by Tencent.","refusal":null,"annotations":null,"audio":null,"function_call":null,"reasoning":null},"logprobs":null,"finish_reason":"stop","stop_reason":null,"token_ids":null,"routed_experts":null}],"service_tier":null,"system_fingerprint":"vllm-0.23.0-tp16-dp2-ep-36628cea","usage":{"prompt_tokens":25,"total_tokens":201,"completion_tokens":176,"prompt_tokens_details":null,"completion_tokens_details":null},"prompt_logprobs":null,"prompt_token_ids":null,"prompt_text":null,"kv_transfer_params":null}
```

## 7 Accuracy Evaluation

Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) for details.

## 8 Performance Evaluation

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/benchmarking/) for more details.

## 9 Declaration

- The current version is only for early experience, and performance optimization is still in progress.
- The service reliability has not been fully validated, and it is not recommended for direct use in production environments.
