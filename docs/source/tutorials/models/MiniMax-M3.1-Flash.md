# MiniMax-M3.1-Flash (Experimental)

## 1 Introduction

!!! note

    The "PD Separation Deployment" mentioned in this document is provided only as an example/reference design to demonstrate the basic process of deploying Prefill and Decode in separated mode. It is not recommended for use in the production environment.

    **For the production environment on the customer side, [MindIE Motor](https://www.hiascend.com/document/detail/en/mindie/latest/quickstart/docs/en/user_guide/quick_start.md) + vLLM-Ascend is recommended.**

MiniMax-M3.1-Flash is a multimodal Mixture-of-Experts model with approximately 428B total parameters and 23B activated parameters per token. It uses sparse attention throughout the model.

This document covers supported features, environment and model preparation, single-node deployment, PD separation, functional verification, accuracy and performance evaluation, performance tuning, and troubleshooting.

!!! warning

    **Current status and constraints**

    - `MiniMax-M3.1-Flash` is provided **out-of-the-box** through the Docker images described in [Installation](#4-installation). The supporting code has **not yet been merged** into the vLLM-Ascend repository, so installing it from source (`pip install` or building from source) is **not supported** for this model yet.
    - The features listed in [Supported Features](#2-supported-features) are only those enabled by the verified deployment commands in this document, and do **not** imply that all features are supported for `MiniMax-M3.1-Flash`. This is an early-access version; performance optimization and reliability validation are still in progress.

## 2 Supported Features

Refer to the [Supported Models](../../user_guide/support_matrix/supported_models.md) page for the support matrix and the [Feature Guide](../../user_guide/feature_guide/index.md) for feature configuration.

## 3 Prerequisites

### 3.1 Model Weight

The official release of MiniMax-M3.1-Flash is available on [Hugging Face](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash). However, its routed MoE experts use the NVFP4 data format, which cannot run directly on Ascend hardware. The following quantized weights are adapted for Ascend:

| Weight Version | Deployment | Model Path |
| --- | --- | --- |
| MiniMax-M3.1-Flash-W8A8 | Atlas 800I A2, tow-node PD colocation <br> Atlas 800I A3, three-node PD disaggregation (1P2D) | [ModelScope](https://www.modelscope.cn/models/Eco-Tech/Minimax-M3.1-Flash-w8a8) |
| MiniMax-M3.1-Flash-MXFP8 | 950DT products, three-node PD disaggregation (1P2D) | [ModelScope](https://www.modelscope.cn/models/Eco-Tech/Minimax-M3.1-Flash-w8a8-mxfp8) |

Download the weight variant used by the serving commands to the same model directory on all participating nodes. It is recommended to place the model weight in a shared cache directory.

### 3.2 Verify Multi-node Communication

Prepare the business network and the NPU interconnect by following
[Verify inter-node connectivity](../../getting_started/installation.md#installation-multi-node-interconnect).

On each host, use `ip addr` or `ifconfig` to identify the host communication IP and the interface that owns it. Set `LOCAL_IP` to that IP and `NIC_NAME` to the interface name; for example, if the IP is assigned to `eth0`, set `NIC_NAME="eth0"`. The scripts use this interface for both `HCCL_SOCKET_IFNAME` and `GLOO_SOCKET_IFNAME`. See [multi-node host configuration](../../developer_guide/contribution/multi_node_test.md#2-test-without-kubernetes) for selecting mutually reachable host IPs.

## 4 Installation

### 4.1 A2 Docker Container

Run the following on each A2 host. Set the model path to the W8A8 checkpoint.

```bash
export IMAGE="quay.io/ascend/vllm-ascend:minimax-m3.1-flash"
export CONTAINER_NAME=minimax_m31_flash
export MODEL_DIR="<MINIMAX_M31_FLASH_W8A8_PATH>"
export WORK_DIR="<DEPLOYMENT_DIRECTORY>"
mkdir -p "$WORK_DIR"

docker run -it --name "$CONTAINER_NAME" \
    --network host --privileged --shm-size=100g \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    --device /dev/davinci0 --device /dev/davinci1 \
    --device /dev/davinci2 --device /dev/davinci3 \
    --device /dev/davinci4 --device /dev/davinci5 \
    --device /dev/davinci6 --device /dev/davinci7 \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro \
    -v /usr/local/sbin:/usr/local/sbin:ro \
    -v /etc/hccn.conf:/etc/hccn.conf:ro \
    -v /usr/local/dcmi:/usr/local/dcmi:ro \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
    -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
    -v "$MODEL_DIR":/models/MiniMax-M3.1-Flash-W8A8:ro \
    -v "$WORK_DIR":/workspace \
    -w /workspace "$IMAGE" bash
```

### 4.2 A3 Docker Container

Run the following on each A3 host. Set the model path to the W8A8 checkpoint.

```bash
export IMAGE="quay.io/ascend/vllm-ascend:minimax-m3.1-flash-a3-openeuler"
export CONTAINER_NAME=minimax_m31_flash
export MODEL_DIR="<MINIMAX_M31_FLASH_W8A8_PATH>"
export WORK_DIR="<DEPLOYMENT_DIRECTORY>"
mkdir -p "$WORK_DIR"

docker run -it --name "$CONTAINER_NAME" \
    --network host --privileged --shm-size=100g \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    --device /dev/davinci0 --device /dev/davinci1 \
    --device /dev/davinci2 --device /dev/davinci3 \
    --device /dev/davinci4 --device /dev/davinci5 \
    --device /dev/davinci6 --device /dev/davinci7 \
    --device /dev/davinci8 --device /dev/davinci9 \
    --device /dev/davinci10 --device /dev/davinci11 \
    --device /dev/davinci12 --device /dev/davinci13 \
    --device /dev/davinci14 --device /dev/davinci15 \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro \
    -v /usr/local/sbin:/usr/local/sbin:ro \
    -v /etc/hccn.conf:/etc/hccn.conf:ro \
    -v /usr/local/dcmi:/usr/local/dcmi:ro \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
    -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
    -v "$MODEL_DIR":/models/MiniMax-M3.1-Flash-W8A8:ro \
    -v "$WORK_DIR":/workspace \
    -w /workspace "$IMAGE" bash
```

### 4.3 950DT products Docker Container

Run the following on each 950DT products host. Set the model path to the W8A8-MXFP8 checkpoint.

```bash
export IMAGE="quay.io/ascend/vllm-ascend:minimax-m3.1-flash-a5-openeuler"
export CONTAINER_NAME=minimax_m31_flash
export MODEL_DIR="<MINIMAX_M31_FLASH_W8A8_MXFP8_PATH>"
export WORK_DIR="<DEPLOYMENT_DIRECTORY>"

docker run -it --name "$CONTAINER_NAME" \
    --network host --ipc host --privileged --shm-size=100g \
    --device /dev/davinci_manager --device /dev/hisi_hdc \
    --device /dev/ummu --device /dev/uburma \
    --device /dev/davinci0 --device /dev/davinci1 \
    --device /dev/davinci2 --device /dev/davinci3 \
    --device /dev/davinci4 --device /dev/davinci5 \
    --device /dev/davinci6 --device /dev/davinci7 \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro \
    -v /usr/local/sbin:/usr/local/sbin:ro \
    -v /etc/hccn.conf:/etc/hccn.conf:ro \
    -v /usr/local/dcmi:/usr/local/dcmi:ro \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
    -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
    -v /etc/hixlep:/etc/hixlep:ro \
    -v /etc/hccl_rootinfo.json:/etc/hccl_rootinfo.json:ro \
    -v "$MODEL_DIR":/models/MiniMax-M3.1-Flash-W8A8-MXFP8:ro \
    -v "$WORK_DIR":/workspace \
    -w /workspace "$IMAGE" bash
```

## 5 Online Service Deployment

Use the container and model weights for your hardware from [Installation](#4-installation).

For descriptions of the standard `vllm serve` arguments used in the deployment examples, refer to the [vLLM Serving Arguments documentation](https://docs.vllm.ai/en/latest/cli/serve/#arguments). For Ascend-specific options passed through `--additional-config`, refer to [Additional Configuration](../../user_guide/configuration/additional_config.md). For Ascend-specific environment variables, refer to [Environment Variables](../../user_guide/configuration/env_vars.md).

**Note**: In the scripts below, `max-num-seqs` represents the maximum number of sequences the scheduler can process in a single iteration. Adjust the `max-num-seqs` parameter dynamically based on actual business.

`MAX_MODEL_LEN` is a shell variable whose value is passed to the vLLM argument `--max-model-len`. The examples use the following context budgets:

| Deployment | MAX_MODEL_LEN |
| --- | --- |
| Single-node, A3 or 950DT products | 133000 |
| PD colocation, A2 | 133000 |
| PD disaggregation, A3 | 263000 |
| PD disaggregation, 950DT products | 133000 |

These values belong to different deployment profiles. Weights, communication buffers, parallelism, and KV caches share device memory, so a context budget that fits one profile may not fit another. Select a value that covers the prompt and generated tokens together, and check available KV cache capacity at the intended concurrency. The `1M` model limit in the support matrix does not mean every example configuration reserves enough memory for 1M-token requests. To use the `263000` setting reported in [Text Evaluation](#72-text-evaluation), update `MAX_MODEL_LEN` in the serving scripts, including both roles for PD deployment, and recheck memory capacity before evaluation.

For text-only deployment, `--limit-mm-per-prompt` can be omitted. For multimodal deployment, configure this parameter according to the actual request shape. For example, use `--limit-mm-per-prompt '{"image":2,"video":0}'` for two-image requests, and use `--limit-mm-per-prompt '{"image":0,"video":1}'` for one-video requests.

### 5.1 Single-Node Deployment

Single-node deployment runs Prefill and Decode on the same host. Both examples use W8A8 weights and [DSpark speculative decoding](../../user_guide/feature_guide/speculative_decoding.md#speculating-using-dspark), which generates draft tokens and verifies them with the target model. They use all 16 NPUs on A3 or all eight NPUs on 950DT products.

=== "A3 series (W8A8)"

    **Start the Service**

    Run the following inside the A3 container. It starts one API service with DP4 x TP4 / EP16 across all 16 NPUs, using W8A8 weights and DSpark speculative decoding with seven draft tokens.

    Set `MODEL_PATH` to your model weight directory; the following example uses the container mount path configured in [Section 4](#4-installation).

    ```bash
    export MODEL_PATH="/models/MiniMax-M3.1-Flash-W8A8"
    export MAX_MODEL_LEN=133000

    export VLLM_USE_V2_MODEL_RUNNER=1
    export HCCL_BUFFSIZE=1024
    export HCCL_OP_EXPANSION_MODE="AIV"
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

    vllm serve ${MODEL_PATH} \
      --host 0.0.0.0 \
      --port 11223 \
      --served-model-name minimax-m3.1-flash \
      --trust-remote-code \
      --max-model-len "$MAX_MODEL_LEN" \
      --tensor-parallel-size 4 \
      --data-parallel-size 4 \
      --enable-expert-parallel \
      --max-num-seqs 32 \
      --disable-hybrid-kv-cache-manager \
      --gpu-memory-utilization 0.92 \
      --quantization ascend \
      --reasoning-parser minimax_m3 \
      --enable-auto-tool-choice \
      --tool-call-parser minimax_m3 \
      --limit-mm-per-prompt '{"image":1,"video":0}' \
      --enable-chunked-prefill \
      --enable-prefix-caching \
      --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
      --additional-config '{
          "enable_cpu_binding": true,
          "enable_flashcomm1": true,
          "multistream_overlap_shared_expert": true,
          "enable_shared_expert_dp":true,
          "weight_nz_mode": 2
      }' \
      --speculative-config '{
          "method":"dspark",
          "num_speculative_tokens":7,
          "enforce_eager":true,
          "draft_sample_method":"probabilistic"
      }' \
      --safetensors-load-strategy prefetch
    ```

    Wait for the following startup messages, then check the health endpoint from another terminal in the container:

    ```text
    INFO:     Started server process
    INFO:     Waiting for application startup.
    INFO:     Application startup complete.
    ```

    ```bash
    curl --noproxy '*' --fail --silent --show-error --output /dev/null \
        --write-out '%{http_code}\n' http://127.0.0.1:11223/health
    ```

    Expected output: `200`. Send requests to port `11223` with model name `minimax-m3.1-flash`. If startup is still in progress or fails, follow [Startup Troubleshooting](#102-model-loading-is-still-in-progress-or-startup-fails).

=== "950DT products"

    **Start the Service**

    Run the following inside the 950DT products container. It uses DP2 x TP4 / EP8 across all eight NPUs, W8A8-MXFP8 weights, and DSpark speculative decoding with seven draft tokens.

    Set `MODEL_PATH` to your model weight directory; the following example uses the container mount path configured in [Section 4](#4-installation).

    ```bash
    export MODEL_PATH="/models/MiniMax-M3.1-Flash-W8A8-MXFP8"
    export MAX_MODEL_LEN=133000

    export VLLM_USE_V2_MODEL_RUNNER=1
    export HCCL_BUFFSIZE=1024
    export HCCL_OP_EXPANSION_MODE="CCU_SCHED"
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

    vllm serve ${MODEL_PATH} \
      --host 0.0.0.0 \
      --port 11223 \
      --served-model-name minimax-m3.1-flash \
      --trust-remote-code \
      --max-model-len "$MAX_MODEL_LEN" \
      --tensor-parallel-size 4 \
      --data-parallel-size 2 \
      --enable-expert-parallel \
      --max-num-seqs 32 \
      --disable-hybrid-kv-cache-manager \
      --gpu-memory-utilization 0.92 \
      --quantization ascend \
      --reasoning-parser minimax_m3 \
      --limit-mm-per-prompt '{"image":1,"video":0}' \
      --enable-chunked-prefill \
      --enable-prefix-caching \
      --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
      --additional-config '{
          "enable_cpu_binding": true,
          "enable_flashcomm1": true,
          "multistream_overlap_shared_expert": true,
          "enable_shared_expert_dp":true
      }' \
      --speculative-config '{
          "method":"dspark",
          "num_speculative_tokens":7,
          "enforce_eager":true,
          "kv_cache_dtype":"bfloat16",
          "draft_sample_method":"probabilistic"
      }' \
      --safetensors-load-strategy prefetch
    ```

    Wait for the following startup messages, then check the health endpoint from another terminal in the container:

    ```text
    INFO:     Started server process
    INFO:     Waiting for application startup.
    INFO:     Application startup complete.
    ```

    ```bash
    curl --noproxy '*' --fail --silent --show-error --output /dev/null \
        --write-out '%{http_code}\n' http://127.0.0.1:11223/health
    ```

    Expected output: `200`. Send requests to port `11223` with model name `minimax-m3.1-flash`. If startup is still in progress or fails, follow [Startup Troubleshooting](#102-model-loading-is-still-in-progress-or-startup-fails).

### 5.2 Multi-Node Deployment

Multi-Node deployment runs Prefill and Decode on the same host, but there are more than one host node. Both nodes use DSpark speculative decoding with seven draft tokens and probabilistic draft sampling. Configure the host-specific values at the beginning of each node script before starting the service.

Monitor node log while weights load and graph capture completes. Startup duration depends on storage, hardware, and the selected configuration; use continuing log progress and `Application startup complete` to judge readiness. If logs stop progressing or report an error, follow [Startup Troubleshooting](#102-model-loading-is-still-in-progress-or-startup-fails).

=== "A2 series (W8A8)"

    There are two A2 nodes, DP4 × TP4 / EP16 in total. The following values match the configurations in the node scripts below and summarize the differences between the two nodes.

    | Parameter | Node 0 | Node 1 |
    | --- | --- | --- |
    | Role | API server | Headless worker |
    | DP × TP | 2 × 4 | 2 × 4 |
    | Data-parallel rank | 0, 1 | 2, 3 |
    | Tool-call parsing | Enabled | — |
    | max-num-seqs | 32 | 32 |

    **Node 0**

    Save the following as `run_node0.sh` on node 0.

    ```bash
    #!/usr/bin/env bash
    set -e

    # Configure these values for the current host.
    MODEL_PATH="/models/MiniMax-M3.1-Flash-W8A8"
    LOCAL_IP="<CURRENT_NODE_IP>"
    NIC_NAME="<CURRENT_NODE_NIC>"
    SERVICE_PORT=11223
    MAX_MODEL_LEN=133000
    MAX_NUM_SEQS=32
    MAX_NUM_BATCHED_TOKENS=4096
    GPU_MEMORY_UTILIZATION=0.92

    # Fixed runtime configuration.
    export VLLM_USE_V2_MODEL_RUNNER=1
    export HCCL_BUFFSIZE=1024
    export HCCL_OP_EXPANSION_MODE=AIV
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export HCCL_IF_IP="$LOCAL_IP"
    export HCCL_SOCKET_IFNAME="$NIC_NAME"
    export GLOO_SOCKET_IFNAME="$NIC_NAME"
    export VLLM_ENGINE_READY_TIMEOUT_S=3600
    export HCCL_CONNECT_TIMEOUT=600

    vllm serve "$MODEL_PATH" \
        --host 0.0.0.0 \
        --port "$SERVICE_PORT" \
        --served-model-name minimax-m3.1-flash \
        --trust-remote-code \
        --quantization ascend \
        --max-model-len "$MAX_MODEL_LEN" \
        --tensor-parallel-size 4 \
        --enable-expert-parallel \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
        --distributed-executor-backend mp \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --reasoning-parser minimax_m3 \
        --limit-mm-per-prompt '{"image":1,"video":0}' \
        --enable-chunked-prefill \
        --enable-prefix-caching \
        --disable-hybrid-kv-cache-manager \
        --additional-config '{
            "enable_cpu_binding": true,
            "enable_flashcomm1": true,
            "multistream_overlap_shared_expert": true,
            "enable_shared_expert_dp": true,
            "weight_nz_mode": 2
        }' \
        --safetensors-load-strategy prefetch \
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
        --enable-auto-tool-choice \
        --tool-call-parser minimax_m3 \
        --data-parallel-size 4 \
        --data-parallel-size-local 2 \
        --data-parallel-address "$LOCAL_IP" \
        --speculative-config '{
            "method":"dspark",
            "num_speculative_tokens":7,
            "enforce_eager":true,
            "draft_sample_method":"probabilistic"
        }'
    ```

    Start the engine inside the node 0 container:

    ```bash
    nohup bash run_node0.sh > node0.log 2>&1 &
    ```

    **Node 1**

    Save the following as `run_node1.sh` on node 1. `HEAD_NODE_IP` is the business network IP of node 0.

    ```bash
    #!/usr/bin/env bash
    set -e

    # Configure these values for the current host.
    MODEL_PATH="/models/MiniMax-M3.1-Flash-W8A8"
    LOCAL_IP="<CURRENT_NODE_IP>"
    HEAD_NODE_IP="<NODE0_IP>"
    NIC_NAME="<CURRENT_NODE_NIC>"
    SERVICE_PORT=11223
    MAX_MODEL_LEN=133000
    MAX_NUM_SEQS=32
    MAX_NUM_BATCHED_TOKENS=4096
    GPU_MEMORY_UTILIZATION=0.92

    # Fixed runtime configuration.
    export VLLM_USE_V2_MODEL_RUNNER=1
    export HCCL_BUFFSIZE=1024
    export HCCL_OP_EXPANSION_MODE=AIV
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export HCCL_IF_IP="$LOCAL_IP"
    export HCCL_SOCKET_IFNAME="$NIC_NAME"
    export GLOO_SOCKET_IFNAME="$NIC_NAME"
    export VLLM_ENGINE_READY_TIMEOUT_S=3600
    export HCCL_CONNECT_TIMEOUT=600

    vllm serve "$MODEL_PATH" \
        --host 0.0.0.0 \
        --port "$SERVICE_PORT" \
        --served-model-name minimax-m3.1-flash \
        --trust-remote-code \
        --quantization ascend \
        --max-model-len "$MAX_MODEL_LEN" \
        --tensor-parallel-size 4 \
        --enable-expert-parallel \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
        --distributed-executor-backend mp \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --reasoning-parser minimax_m3 \
        --limit-mm-per-prompt '{"image":1,"video":0}' \
        --enable-chunked-prefill \
        --enable-prefix-caching \
        --disable-hybrid-kv-cache-manager \
        --additional-config '{
            "enable_cpu_binding": true,
            "enable_flashcomm1": true,
            "multistream_overlap_shared_expert": true,
            "enable_shared_expert_dp": true,
            "weight_nz_mode": 2
        }' \
        --safetensors-load-strategy prefetch \
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
        --headless \
        --data-parallel-size 4 \
        --data-parallel-size-local 2 \
        --data-parallel-start-rank 2 \
        --data-parallel-address "$HEAD_NODE_IP" \
        --speculative-config '{
            "method":"dspark",
            "num_speculative_tokens":7,
            "enforce_eager":true,
            "draft_sample_method":"probabilistic"
        }'
    ```

    Start the engine inside the node 1 container:

    ```bash
    nohup bash run_node1.sh > node1.log 2>&1 &
    ```

    Node 1 is headless and does not open an HTTP server. After node 0 prints the following startup messages, check the health endpoint from another terminal in the node 0 container:

    ```text
    INFO:     Started server process
    INFO:     Waiting for application startup.
    INFO:     Application startup complete.
    ```

    ```bash
    curl --noproxy '*' --fail --silent --show-error --output /dev/null \
        --write-out '%{http_code}\n' http://127.0.0.1:11223/health
    ```

    Expected output: `200`. Send requests to port `11223` on node 0 with model name `minimax-m3.1-flash`.

### 5.3 PD Disaggregation (1P2D)

Both platforms use one Prefill node and two Decode nodes (1P2D), with Prefill and Decode on separate hosts. Both platforms use DSpark speculative decoding with seven draft tokens and probabilistic draft sampling. Configure the host-specific values at the beginning of each Prefill and Decode script before starting the service.

Monitor the role logs while weights load and graph capture completes. Startup duration depends on storage, hardware, and the selected configuration; use continuing log progress to monitor initialization, and wait for `Application startup complete` on each HTTP-serving instance to judge readiness. Headless workers join the distributed Decode service without exposing their own HTTP API. If logs stop progressing or report an error, follow [Startup Troubleshooting](#102-model-loading-is-still-in-progress-or-startup-fails) before starting the proxy.

=== "A3 series (W8A8)"

    Use one Prefill node and two Decode nodes, each with all 16 NPUs. Prefill uses DP1 × TP4 × PP4. Decode uses DP8 × TP4 × PP1 across the two nodes: D1 runs global DP ranks 0–3, and D2 runs ranks 4–7. Each rank starts one TP4 service on four local NPUs. All eight Decode ranks share the D1 communication IP as their DP master address and use the same RPC port. Use the same image and weights on all three nodes.

    The following values match the configurations in the Prefill and Decode scripts below and summarize the differences between the two roles.

    | Parameter | Prefill | Decode |
    | --- | --- | --- |
    | DP × TP × PP | 1 × 4 × 4 | 8 × 4 × 1 across two nodes |
    | Global DP ranks | 0 | D1: 0–3; D2: 4–7 |
    | PP layer partition | 15,15,15,15 | — |
    | max-num-seqs | 32 | 128 per replica |
    | Execution mode | Eager | FULL_DECODE_ONLY |
    | HCCL_BUFFSIZE (MB) | 512 | 2048 |

    **Prefill Node (P)**

    Save the following as `run_prefill.sh` on the Prefill node.

    ```bash
    #!/usr/bin/env bash
    set -e

    # Configure these values for the current host.
    MODEL_PATH="/models/MiniMax-M3.1-Flash-W8A8"
    LOCAL_IP="<CURRENT_NODE_IP>"
    NIC_NAME="<CURRENT_NODE_NIC>"
    SERVICE_PORT=19100
    KV_PORT=24000
    MAX_NUM_SEQS=32
    MAX_MODEL_LEN=263000
    MAX_BATCHED_TOKENS=32768
    GPU_MEMORY_UTILIZATION=0.92

    # Fixed runtime configuration.
    export VLLM_PP_LAYER_PARTITION=15,15,15,15
    export VLLM_USE_V2_MODEL_RUNNER=1
    export HCCL_BUFFSIZE=512
    export HCCL_IF_IP="$LOCAL_IP"
    unset HCCL_INTRA_ROCE_ENABLE
    export HCCL_OP_EXPANSION_MODE=AIV
    export HCCL_SOCKET_IFNAME="$NIC_NAME"
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
    export GLOO_SOCKET_IFNAME="$NIC_NAME"
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

    vllm serve "$MODEL_PATH" \
        --host 0.0.0.0 \
        --port "$SERVICE_PORT" \
        --served-model-name minimax-m3.1-flash \
        --trust-remote-code \
        --hf-overrides '{"architectures":["MiniMaxM3SparseForCausalLM"]}' \
        --language-model-only \
        --quantization ascend \
        --dtype bfloat16 \
        --tensor-parallel-size 4 \
        --pipeline-parallel-size 4 \
        --enable-expert-parallel \
        --distributed-executor-backend mp \
        --seed 1024 \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --block-size 128 \
        --kv-cache-dtype bfloat16 \
        --enable-chunked-prefill \
        --enable-prefix-caching \
        --reasoning-parser minimax_m3 \
        --enable-auto-tool-choice \
        --tool-call-parser minimax_m3 \
        --data-parallel-size 1 \
        --enforce-eager \
        --long-prefill-token-threshold 2048 \
        --disable-hybrid-kv-cache-manager \
        --speculative-config '{
            "method":"dspark",
            "num_speculative_tokens":7,
            "enforce_eager":true,
            "draft_sample_method":"probabilistic"
        }' \
        --additional-config '{
            "enable_cpu_binding":true,
            "multistream_overlap_shared_expert":true,
            "enable_shared_expert_dp":true,
            "weight_nz_mode":2,
            "enable_flashcomm1":true
        }' \
        --kv-transfer-config '{
            "kv_connector":"MooncakeConnectorV1",
            "kv_role":"kv_producer",
            "kv_port":'"$KV_PORT"',
            "engine_id":"m31flash-p-dp0",
            "kv_connector_extra_config":{
                "use_ascend_direct":true,
                "prefill":{"dp_size":1,"tp_size":4,"pp_size":4,"pp_layer_partition":"15,15,15,15"},
                "decode":{"dp_size":8,"tp_size":4,"pp_size":1}
            }
        }'
    ```

    Start the Prefill engine:

    ```bash
    nohup bash run_prefill.sh > prefill.log 2>&1 &
    ```

    **Decode Node 1 (D1, DP ranks 0–3)**

    Save the following as `run_decode_node1_rank.sh` on D1. Set `LOCAL_IP` and `NIC_NAME` for this host, and set `DECODE_HEAD_IP` to D1's reachable communication IP. Keep the image, weights, global parallelism, and RPC port identical on both Decode nodes. Each invocation starts one TP4 service; global DP ranks identify services across both nodes, while local ranks select the NPU group on the current node.

    ```bash
    #!/usr/bin/env bash
    set -e

    # Configure these values for the current host.
    MODEL_PATH="/models/MiniMax-M3.1-Flash-W8A8"
    LOCAL_IP="<CURRENT_NODE_IP>"
    NIC_NAME="<CURRENT_NODE_NIC>"
    DECODE_HEAD_IP="<DECODE1_NODE_IP>"
    DP_RANK=${1:-0}
    LOCAL_DP_RANK=$DP_RANK
    SERVICE_PORT=$((19200 + DP_RANK))
    DP_RPC_PORT=16920
    KV_PORT=$((25000 + DP_RANK*32))
    MAX_NUM_SEQS=128
    MAX_MODEL_LEN=263000
    MAX_BATCHED_TOKENS=32768
    GPU_MEMORY_UTILIZATION=0.92

    # Fixed runtime configuration.
    export VLLM_USE_V2_MODEL_RUNNER=1
    export HCCL_BUFFSIZE=2048
    export HCCL_IF_IP="$LOCAL_IP"
    unset HCCL_INTRA_ROCE_ENABLE
    export HCCL_OP_EXPANSION_MODE=AIV
    export HCCL_SOCKET_IFNAME="$NIC_NAME"
    export ASCEND_RT_VISIBLE_DEVICES="$((LOCAL_DP_RANK*4)),$((LOCAL_DP_RANK*4+1)),$((LOCAL_DP_RANK*4+2)),$((LOCAL_DP_RANK*4+3))"
    export GLOO_SOCKET_IFNAME="$NIC_NAME"
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

    vllm serve "$MODEL_PATH" \
        --host 0.0.0.0 \
        --port "$SERVICE_PORT" \
        --served-model-name minimax-m3.1-flash \
        --trust-remote-code \
        --hf-overrides '{"architectures":["MiniMaxM3SparseForCausalLM"]}' \
        --language-model-only \
        --quantization ascend \
        --dtype bfloat16 \
        --tensor-parallel-size 4 \
        --pipeline-parallel-size 1 \
        --enable-expert-parallel \
        --distributed-executor-backend mp \
        --seed 1024 \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --block-size 128 \
        --kv-cache-dtype bfloat16 \
        --enable-chunked-prefill \
        --enable-prefix-caching \
        --reasoning-parser minimax_m3 \
        --enable-auto-tool-choice \
        --tool-call-parser minimax_m3 \
        --data-parallel-size 8 \
        --data-parallel-size-local 1 \
        --data-parallel-rank "$DP_RANK" \
        --data-parallel-address "$DECODE_HEAD_IP" \
        --data-parallel-rpc-port "$DP_RPC_PORT" \
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
        --disable-hybrid-kv-cache-manager \
        --speculative-config '{
            "method":"dspark",
            "num_speculative_tokens":7,
            "enforce_eager":true,
            "draft_sample_method":"probabilistic"
        }' \
        --additional-config '{
            "enable_cpu_binding":true,
            "multistream_overlap_shared_expert":true,
            "enable_shared_expert_dp":true,
            "enable_flashcomm1":true,
            "weight_nz_mode":2
        }' \
        --kv-transfer-config '{
            "kv_connector":"MooncakeConnectorV1",
            "kv_role":"kv_consumer",
            "kv_port":'"$KV_PORT"',
            "engine_id":"m31flash-d-dp'"$DP_RANK"'",
            "kv_connector_extra_config":{
                "use_ascend_direct":true,
                "prefill":{"dp_size":1,"tp_size":4,"pp_size":4,"pp_layer_partition":"15,15,15,15"},
                "decode":{"dp_size":8,"tp_size":4,"pp_size":1}
            }
        }'
    ```

    Start all four D1 services inside the D1 container, then start D2 without waiting for D1 to become ready:

    ```bash
    for rank in 0 1 2 3; do
        nohup bash run_decode_node1_rank.sh "$rank" > "decode-$rank.log" 2>&1 &
    done
    ```

    **Decode Node 2 (D2, DP ranks 4–7)**

    Save the following as `run_decode_node2_rank.sh` on D2. Set `LOCAL_IP` and `NIC_NAME` for this host, and set `DECODE_HEAD_IP` to D1's reachable communication IP. Keep the image, weights, global parallelism, and RPC port identical on both Decode nodes. Each invocation starts one TP4 service; global DP ranks identify services across both nodes, while local ranks select the NPU group on the current node.

    ```bash
    #!/usr/bin/env bash
    set -e

    # Configure these values for the current host.
    MODEL_PATH="/models/MiniMax-M3.1-Flash-W8A8"
    LOCAL_IP="<CURRENT_NODE_IP>"
    NIC_NAME="<CURRENT_NODE_NIC>"
    DECODE_HEAD_IP="<DECODE1_NODE_IP>"
    DP_RANK=${1:-4}
    LOCAL_DP_RANK=$((DP_RANK - 4))
    SERVICE_PORT=$((19200 + DP_RANK))
    DP_RPC_PORT=16920
    KV_PORT=$((25000 + DP_RANK*32))
    MAX_NUM_SEQS=128
    MAX_MODEL_LEN=263000
    MAX_BATCHED_TOKENS=32768
    GPU_MEMORY_UTILIZATION=0.92

    # Fixed runtime configuration.
    export VLLM_USE_V2_MODEL_RUNNER=1
    export HCCL_BUFFSIZE=2048
    export HCCL_IF_IP="$LOCAL_IP"
    unset HCCL_INTRA_ROCE_ENABLE
    export HCCL_OP_EXPANSION_MODE=AIV
    export HCCL_SOCKET_IFNAME="$NIC_NAME"
    export ASCEND_RT_VISIBLE_DEVICES="$((LOCAL_DP_RANK*4)),$((LOCAL_DP_RANK*4+1)),$((LOCAL_DP_RANK*4+2)),$((LOCAL_DP_RANK*4+3))"
    export GLOO_SOCKET_IFNAME="$NIC_NAME"
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

    vllm serve "$MODEL_PATH" \
        --host 0.0.0.0 \
        --port "$SERVICE_PORT" \
        --served-model-name minimax-m3.1-flash \
        --trust-remote-code \
        --hf-overrides '{"architectures":["MiniMaxM3SparseForCausalLM"]}' \
        --language-model-only \
        --quantization ascend \
        --dtype bfloat16 \
        --tensor-parallel-size 4 \
        --pipeline-parallel-size 1 \
        --enable-expert-parallel \
        --distributed-executor-backend mp \
        --seed 1024 \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --block-size 128 \
        --kv-cache-dtype bfloat16 \
        --enable-chunked-prefill \
        --enable-prefix-caching \
        --reasoning-parser minimax_m3 \
        --enable-auto-tool-choice \
        --tool-call-parser minimax_m3 \
        --data-parallel-size 8 \
        --data-parallel-size-local 1 \
        --data-parallel-rank "$DP_RANK" \
        --data-parallel-address "$DECODE_HEAD_IP" \
        --data-parallel-rpc-port "$DP_RPC_PORT" \
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
        --disable-hybrid-kv-cache-manager \
        --speculative-config '{
            "method":"dspark",
            "num_speculative_tokens":7,
            "enforce_eager":true,
            "draft_sample_method":"probabilistic"
        }' \
        --additional-config '{
            "enable_cpu_binding":true,
            "multistream_overlap_shared_expert":true,
            "enable_shared_expert_dp":true,
            "enable_flashcomm1":true,
            "weight_nz_mode":2
        }' \
        --kv-transfer-config '{
            "kv_connector":"MooncakeConnectorV1",
            "kv_role":"kv_consumer",
            "kv_port":'"$KV_PORT"',
            "engine_id":"m31flash-d-dp'"$DP_RANK"'",
            "kv_connector_extra_config":{
                "use_ascend_direct":true,
                "prefill":{"dp_size":1,"tp_size":4,"pp_size":4,"pp_layer_partition":"15,15,15,15"},
                "decode":{"dp_size":8,"tp_size":4,"pp_size":1}
            }
        }'
    ```

    Immediately start all four D2 services inside the D2 container. All eight global DP ranks must join before Decode initialization can complete:

    ```bash
    for rank in 4 5 6 7; do
        nohup bash run_decode_node2_rank.sh "$rank" > "decode-$rank.log" 2>&1 &
    done
    ```

    **Load-balance Proxy**

    Download [load_balance_proxy_server_example.py](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py) from the official repository and save it as `/workspace/load_balance_proxy_server_example.py` inside the Prefill container.

    After Prefill and all eight Decode services print `Application startup complete`, run the following
    inside the Prefill container:

    ```bash
    PREFILL_IP="<PREFILL_NODE_IP>"
    DECODE1_IP="<DECODE1_NODE_IP>"
    DECODE2_IP="<DECODE2_NODE_IP>"

    python3 /workspace/load_balance_proxy_server_example.py \
        --host 0.0.0.0 \
        --port 19080 \
        --prefiller-hosts "$PREFILL_IP" \
        --prefiller-ports 19100 \
        --decoder-hosts \
            "$DECODE1_IP" "$DECODE1_IP" "$DECODE1_IP" "$DECODE1_IP" \
            "$DECODE2_IP" "$DECODE2_IP" "$DECODE2_IP" "$DECODE2_IP" \
        --decoder-ports \
            19200 19201 19202 19203 \
            19204 19205 19206 19207
    ```

=== "950DT products"

    Use one Prefill node and two Decode nodes (1P2D), each with all eight NPUs. Prefill uses DP1 × TP4 × PP2. The two Decode nodes form one distributed DP4 × TP4 × PP1 / EP16 instance, with two local DP ranks per node. The Decode head runs ranks 0–1 and provides the HTTP API; the headless worker runs ranks 2–3.

    The [reference deployment](https://github.com/CXY-Katrina/workspace/tree/6a00e8b246ebb9274f649956c59ccb2c5ec66adf/minimax_m3/minimax_m3.1_flash/deployments/2026-10-27-a5-1p2d) used `quay.io/atlas-ci/vllm-atlas-temp:minimax-m3.1-flash-a5-openeuler-40260585905-1-arm64-temp` and W8A8-MXFP8 QuaRot l32 weights. Set `IMAGE` and `MODEL_DIR` in Section 4.3 accordingly, keeping the documented container mount path. Use the same image and weights on all three nodes.

    The following values match the configurations in the Prefill and Decode scripts below and summarize the differences between the two roles.

    | Parameter | Prefill | Decode |
    | --- | --- | --- |
    | DP × TP × PP | 1 × 4 × 2 | 4 × 4 × 1 across two nodes |
    | Local DP ranks | 0 | Head: 0–1; worker: 2–3 |
    | max-num-seqs | 128 | 256 per DP engine |
    | max-num-batched-tokens | 16384 | 32768 |
    | gpu-memory-utilization | 0.85 | 0.92 |
    | HCCL_BUFFSIZE (MB) | 512 | 1024 |
    | Main-model execution | Eager | FULL_DECODE_ONLY |

    **Prefill Node (P)**

    Save the following as `run_prefill.sh` on the Prefill node. Configure the model path, host IP address, and network interface at the beginning of the script.

    ```bash
    #!/usr/bin/env bash
    set -e

    # Configure these values for the current host.
    MODEL_PATH="/models/MiniMax-M3.1-Flash-W8A8-MXFP8"
    LOCAL_IP="<CURRENT_NODE_IP>"
    NIC_NAME="<CURRENT_NODE_NIC>"
    SERVICE_PORT=31050
    DP_RPC_PORT=6884
    KV_PORT=30000
    MAX_NUM_SEQS=128
    MAX_MODEL_LEN=133000
    MAX_BATCHED_TOKENS=16384
    GPU_MEMORY_UTILIZATION=0.85

    export VLLM_USE_V2_MODEL_RUNNER=1
    export VLLM_ENGINE_READY_TIMEOUT_S=1800
    export HCCL_BUFFSIZE=512
    export HCCL_IF_IP="$LOCAL_IP"
    export HCCL_OP_EXPANSION_MODE="CCU_SCHED"
    export HCCL_SOCKET_IFNAME="$NIC_NAME"
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export GLOO_SOCKET_IFNAME="$NIC_NAME"
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

    vllm serve "$MODEL_PATH" \
        --host 0.0.0.0 \
        --port "$SERVICE_PORT" \
        --data-parallel-size 1 \
        --data-parallel-size-local 1 \
        --data-parallel-address "$LOCAL_IP" \
        --data-parallel-rpc-port "$DP_RPC_PORT" \
        --tensor-parallel-size 4 \
        --pipeline-parallel-size 2 \
        --served-model-name minimax-m3.1-flash \
        --trust-remote-code \
        --dtype bfloat16 \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
        --max-model-len "$MAX_MODEL_LEN" \
        --disable-hybrid-kv-cache-manager \
        --enable-expert-parallel \
        --quantization ascend \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --distributed-executor-backend mp \
        --kv-cache-dtype fp8 \
        --reasoning-parser minimax_m3 \
        --enable-auto-tool-choice \
        --tool-call-parser minimax_m3 \
        --language-model-only \
        --safetensors-load-strategy prefetch \
        --enforce-eager \
        --enable-chunked-prefill \
        --enable-prefix-caching \
        --speculative-config '{
            "method":"dspark",
            "num_speculative_tokens":7,
            "enforce_eager":true,
            "kv_cache_dtype":"bfloat16",
            "draft_sample_method":"probabilistic"
        }' \
        --additional-config '{
            "enable_cpu_binding": true,
            "multistream_overlap_shared_expert": true,
            "enable_shared_expert_dp": true,
            "enable_flashcomm1": true
        }' \
        --kv-transfer-config '{
            "kv_connector": "MooncakeConnectorV1",
            "kv_role": "kv_producer",
            "kv_port": "'"$KV_PORT"'",
            "kv_connector_extra_config": {
                "use_ascend_direct": true,
                "prefill": {
                    "dp_size": 1,
                    "tp_size": 4,
                    "pp_size": 2
                },
                "decode": {
                    "dp_size": 4,
                    "tp_size": 4,
                    "pp_size": 1
                }
            }
        }'
    ```

    Start the Prefill engine inside the Prefill container:

    ```bash
    nohup bash run_prefill.sh > prefill.log 2>&1 &
    ```

    **Decode Head Node (D1, DP ranks 0–1)**

    Save the following as `run_decode_head.sh` on D1. Set `LOCAL_IP` and `NIC_NAME` for this host, and set `DECODE_HEAD_IP` to its reachable communication IP. This node provides the Decode HTTP API. Ensure all three nodes' communication IPs resolve promptly; configure matching `/etc/hosts` entries if needed.

    ```bash
    #!/usr/bin/env bash
    set -e

    # Configure these values for the current host.
    MODEL_PATH="/models/MiniMax-M3.1-Flash-W8A8-MXFP8"
    LOCAL_IP="<CURRENT_NODE_IP>"
    NIC_NAME="<CURRENT_NODE_NIC>"
    DECODE_HEAD_IP="<DECODE_HEAD_NODE_IP>"
    SERVICE_PORT=31060
    DP_RPC_PORT=6885
    KV_PORT=26900
    MAX_NUM_SEQS=256
    MAX_MODEL_LEN=133000
    MAX_BATCHED_TOKENS=32768
    GPU_MEMORY_UTILIZATION=0.92

    export VLLM_USE_V2_MODEL_RUNNER=1
    export VLLM_ENGINE_READY_TIMEOUT_S=1800
    export HCCL_BUFFSIZE=1024
    export HCCL_IF_IP="$LOCAL_IP"
    export HCCL_OP_EXPANSION_MODE="CCU_SCHED"
    export HCCL_SOCKET_IFNAME="$NIC_NAME"
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export GLOO_SOCKET_IFNAME="$NIC_NAME"
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

    vllm serve "$MODEL_PATH" \
        --host 0.0.0.0 \
        --port "$SERVICE_PORT" \
        --data-parallel-size 4 \
        --data-parallel-size-local 2 \
        --data-parallel-start-rank 0 \
        --data-parallel-address "$DECODE_HEAD_IP" \
        --data-parallel-rpc-port "$DP_RPC_PORT" \
        --tensor-parallel-size 4 \
        --pipeline-parallel-size 1 \
        --served-model-name minimax-m3.1-flash \
        --trust-remote-code \
        --dtype bfloat16 \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
        --max-model-len "$MAX_MODEL_LEN" \
        --disable-hybrid-kv-cache-manager \
        --enable-expert-parallel \
        --quantization ascend \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --distributed-executor-backend mp \
        --kv-cache-dtype fp8 \
        --reasoning-parser minimax_m3 \
        --enable-auto-tool-choice \
        --tool-call-parser minimax_m3 \
        --language-model-only \
        --safetensors-load-strategy prefetch \
        --enable-prefix-caching \
        --seed 1024 \
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","max_cudagraph_capture_size":1024}' \
        --speculative-config '{
            "method":"dspark",
            "num_speculative_tokens":7,
            "enforce_eager":true,
            "kv_cache_dtype":"bfloat16",
            "draft_sample_method":"probabilistic"
        }' \
        --additional-config '{
            "enable_cpu_binding": true,
            "ascend_compilation_config": {
                "fuse_norm_quant": false,
                "fuse_qknorm_rope": true
            },
            "multistream_overlap_shared_expert": true,
            "enable_shared_expert_dp": true,
            "enable_flashcomm1": true
        }' \
        --kv-transfer-config '{
            "kv_connector": "MooncakeConnectorV1",
            "kv_role": "kv_consumer",
            "kv_port": "'"$KV_PORT"'",
            "kv_connector_extra_config": {
                "use_ascend_direct": true,
                "prefill": {
                    "dp_size": 1,
                    "tp_size": 4,
                    "pp_size": 2
                },
                "decode": {
                    "dp_size": 4,
                    "tp_size": 4,
                    "pp_size": 1
                }
            }
        }'
    ```

    Start the Decode head inside its container:

    ```bash
    nohup bash run_decode_head.sh > decode.log 2>&1 &
    ```

    **Decode Worker Node (D2, DP ranks 2–3, headless)**

    Save the following as `run_decode_worker.sh` on D2. Set `LOCAL_IP` and `NIC_NAME` for this host, and set `DECODE_HEAD_IP` to the same D1 communication IP used above. Keep the image, weights, RPC/KV ports, and global parallelism settings identical to D1. This headless node does not expose an HTTP API.

    ```bash
    #!/usr/bin/env bash
    set -e

    # Configure these values for the current host.
    MODEL_PATH="/models/MiniMax-M3.1-Flash-W8A8-MXFP8"
    LOCAL_IP="<CURRENT_NODE_IP>"
    NIC_NAME="<CURRENT_NODE_NIC>"
    DECODE_HEAD_IP="<DECODE_HEAD_NODE_IP>"
    SERVICE_PORT=31060
    DP_RPC_PORT=6885
    KV_PORT=26900
    MAX_NUM_SEQS=256
    MAX_MODEL_LEN=133000
    MAX_BATCHED_TOKENS=32768
    GPU_MEMORY_UTILIZATION=0.92

    export VLLM_USE_V2_MODEL_RUNNER=1
    export VLLM_ENGINE_READY_TIMEOUT_S=1800
    export HCCL_BUFFSIZE=1024
    export HCCL_IF_IP="$LOCAL_IP"
    export HCCL_OP_EXPANSION_MODE="CCU_SCHED"
    export HCCL_SOCKET_IFNAME="$NIC_NAME"
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export GLOO_SOCKET_IFNAME="$NIC_NAME"
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

    vllm serve "$MODEL_PATH" \
        --host 0.0.0.0 \
        --port "$SERVICE_PORT" \
        --data-parallel-size 4 \
        --data-parallel-size-local 2 \
        --data-parallel-start-rank 2 \
        --headless \
        --data-parallel-address "$DECODE_HEAD_IP" \
        --data-parallel-rpc-port "$DP_RPC_PORT" \
        --tensor-parallel-size 4 \
        --pipeline-parallel-size 1 \
        --served-model-name minimax-m3.1-flash \
        --trust-remote-code \
        --dtype bfloat16 \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
        --max-model-len "$MAX_MODEL_LEN" \
        --disable-hybrid-kv-cache-manager \
        --enable-expert-parallel \
        --quantization ascend \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --distributed-executor-backend mp \
        --kv-cache-dtype fp8 \
        --reasoning-parser minimax_m3 \
        --enable-auto-tool-choice \
        --tool-call-parser minimax_m3 \
        --language-model-only \
        --safetensors-load-strategy prefetch \
        --enable-prefix-caching \
        --seed 1024 \
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","max_cudagraph_capture_size":1024}' \
        --speculative-config '{
            "method":"dspark",
            "num_speculative_tokens":7,
            "enforce_eager":true,
            "kv_cache_dtype":"bfloat16",
            "draft_sample_method":"probabilistic"
        }' \
        --additional-config '{
            "enable_cpu_binding": true,
            "ascend_compilation_config": {
                "fuse_norm_quant": false,
                "fuse_qknorm_rope": true
            },
            "multistream_overlap_shared_expert": true,
            "enable_shared_expert_dp": true,
            "enable_flashcomm1": true
        }' \
        --kv-transfer-config '{
            "kv_connector": "MooncakeConnectorV1",
            "kv_role": "kv_consumer",
            "kv_port": "'"$KV_PORT"'",
            "kv_connector_extra_config": {
                "use_ascend_direct": true,
                "prefill": {
                    "dp_size": 1,
                    "tp_size": 4,
                    "pp_size": 2
                },
                "decode": {
                    "dp_size": 4,
                    "tp_size": 4,
                    "pp_size": 1
                }
            }
        }'
    ```

    Immediately start the worker inside its container; do not wait for D1 to become ready first:

    ```bash
    nohup bash run_decode_worker.sh > decode.log 2>&1 &
    ```

    Both Decode nodes must join before the distributed Decode service can become ready.

    **Load-balance Proxy**

    Download [load_balance_proxy_server_example.py](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py) from the official repository and save it as `/workspace/load_balance_proxy_server_example.py` inside the Prefill container.

    After Prefill and the Decode head print `Application startup complete`, run the following
    inside the Prefill container:

    ```bash
    PREFILL_IP="<PREFILL_NODE_IP>"
    DECODE1_IP="<DECODE1_NODE_IP>"
    DECODE2_IP="<DECODE2_NODE_IP>"

    python3 /workspace/load_balance_proxy_server_example.py \
        --host 0.0.0.0 \
        --port 19080 \
        --prefiller-hosts "$PREFILL_IP" \
        --prefiller-ports 19100 \
        --decoder-hosts \
            "$DECODE1_IP" "$DECODE1_IP" "$DECODE1_IP" "$DECODE1_IP" \
            "$DECODE2_IP" "$DECODE2_IP" "$DECODE2_IP" "$DECODE2_IP" \
        --decoder-ports \
            19200 19201 19202 19203 \
            19204 19205 19206 19207 \
        --max-retries 0 \
        --workers 4
    ```

## 6 Functional Verification

Replace `{ip}` and `{port}` with the service endpoint. All examples use `minimax-m3.1-flash` as the served model name.

### 6.1 Text

  ```bash
  curl http://{ip}:{port}/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d @- <<EOF
  {
    "model": "minimax-m3.1-flash",
    "messages": [
      {
        "role": "user",
        "content": "Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.\n\nA student regrets that he fell asleep during a lecture in electrochemistry, facing the following incomplete statement in a test:\nThermodynamically, oxygen is a …oxidant in basic solutions. Kinetically, oxygen reacts …in acidic solutions.\nWhich combination of weaker/stronger and faster/slower is correct?\n\nA) weaker —faster\nB) stronger —faster\nC) weaker - slower\nD) stronger —slower"
      }
    ],
    "max_tokens": 8000,
    "temperature": 1.0
  }
  EOF
  ```

  Expected result: the answer is C.

### 6.2 Single Image

  Start the service with image input enabled, for example `--limit-mm-per-prompt '{"image":1,"video":0}'`. Replace `${IMAGE_PATH}` with a local image path on the client side.

  ```bash
  IMAGE_PATH=/path/to/image.jpg
  IMAGE_BASE64="$(base64 -w 0 "${IMAGE_PATH}")"

  curl http://{ip}:{port}/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d @- <<EOF
  {
    "model": "minimax-m3.1-flash",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,${IMAGE_BASE64}"}},
          {"type": "text", "text": "Briefly describe this image."}
        ]
      }
    ],
    "max_tokens": 512,
    "temperature": 0
  }
  EOF
  ```

  Expected result: HTTP 200 response with a JSON body containing non-empty `choices` and generated text describing the image.

### 6.3 Single Video

  Start the service with video input enabled, for example `--limit-mm-per-prompt '{"image":0,"video":1}'`. If the request uses `file://` local video paths, also add `--allowed-local-media-path /` or a narrower allowed directory. If `media_io_kwargs.video.num_frames` is not specified, vLLM samples 32 frames by default.

  ```bash
  curl http://{ip}:{port}/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "minimax-m3.1-flash",
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "video_url",
              "video_url": {
                "url": "file:///path/to/video.mp4"
              }
            },
            {
              "type": "text",
              "text": "Briefly describe the main content of this video."
            }
          ]
        }
      ],
      "max_tokens": 512,
      "temperature": 0
    }'
  ```

  Expected result: HTTP 200 response with a JSON body containing non-empty `choices` and generated text describing the video content.

### 6.4 Mixed Image and Video Request

  Start the service with both image and video input enabled. For the following request, use `--limit-mm-per-prompt '{"image":1,"video":1}'`. If the request uses `file://` local video paths, also add `--allowed-local-media-path /` or a narrower allowed directory.

  ```bash
  IMAGE_BASE64="$(base64 -w 0 /path/to/image.jpg)"

  curl http://{ip}:{port}/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d @- <<EOF
  {
    "model": "minimax-m3.1-flash",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,${IMAGE_BASE64}"}},
          {"type": "video_url", "video_url": {"url": "file:///path/to/video.mp4"}},
          {"type": "text", "text": "Describe the image and video separately, and explain whether they are related."}
        ]
      }
    ],
    "max_tokens": 512,
    "temperature": 0
  }
  EOF
  ```

  Expected result: HTTP 200 response with a JSON body containing non-empty `choices` and generated text that describes the image and video separately and explains whether they are related.

## 7 Accuracy Evaluation

### 7.1 Using AISBench

For detailed instructions, refer to [Using AISBench for accuracy evaluation](../../developer_guide/evaluation/using_ais_bench.md).

### 7.2 Text Evaluation

| Dataset | Hardware | Score | max-model-len | max-num-seqs | max_out_len | batch_size | generation_kwargs |
|---------|----------|-------|---------------|--------------|-------------|------------|-------------------|
| AIME2026 | Ascend 950DT products | 93.33@repeat4 | 263000 | 32 | 81920 | 32 | temperature=1.0, top_p=0.95, reasoning_effort="max", chat_template_kwargs={"thinking_mode":"enabled"} |
| AIME2026 | A3 | 93.33@repeat1 | 263000 | 32 | 81920 | 32 | temperature=1.0, top_p=0.95, reasoning_effort="max", chat_template_kwargs={"thinking_mode":"enabled"} |
| AIME2026 | A2 | 93.33@repeat1 | 133000 | 32 | 81920 | 32 | temperature=1.0, top_p=0.95, reasoning_effort="max", chat_template_kwargs={"thinking_mode":"enabled"} |

## 8 Performance Evaluation

### 8.1 Using AISBench

For setup details, including installation, dataset download, and configuration, please refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

The following is an example configuration for the performance evaluation config file:

```python
# Example configuration: benchmarks/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py
from ais_bench.benchmark.models import VLLMCustomAPIChat
from ais_bench.benchmark.utils.postprocess.model_postprocessors import extract_non_reasoning_content

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr="vllm-api-stream-chat",
        path="your_model_path",
        model="minimax-m3.1-flash",
        stream=True,
        request_rate=0,
        use_timestamp=False,
        retry=2,
        host_ip="127.0.0.1",
        host_port=8009,
        max_out_len=1024,
        batch_size=32,
        trust_remote_code=False,
        generation_kwargs=dict(
            temperature=0,
            ignore_eos=True
        ),
    )
]
```

**Run the performance evaluation using the GSM8K dataset as an example:**

```bash
ais_bench --models vllm_api_stream_chat --datasets gsm8k_gen_0_shot_cot_str_perf --debug --summarizer default_perf --mode perf --num-prompts 128
```

## 9 Performance Tuning

### 9.1 Recommended Configurations

Use the commands in [Section 5](#5-online-service-deployment) as starting configurations. They are not globally optimal: the appropriate settings depend on request lengths, concurrency, prefix-cache hit rate, and the available device memory. Evaluate each adjustment with the same workload and compare latency, throughput, and memory use.

| Scenario | Starting Configuration | Parameters to Evaluate | Validation |
| --- | --- | --- | --- |
| Long context | A3 PD example with `MAX_MODEL_LEN=263000`; the other examples start at `133000`. | Set `MAX_MODEL_LEN` for the combined input/output budget and reduce `--max-num-seqs` if KV cache capacity is insufficient. | Confirm the requested context fits at the target concurrency before increasing load. |
| Low latency | Single-node examples in Section 5.1. | Reduce request concurrency and compare `--max-num-seqs` and `--max-num-batched-tokens` settings. | Compare TTFT and TPOT alongside output-token throughput. |
| High throughput | PD examples in Section 5.3. | Adjust `MAX_NUM_SEQS` and `MAX_BATCHED_TOKENS` on Prefill and Decode separately while keeping both connector topology declarations consistent. | Increase offered load gradually and check throughput, latency, and failed requests. |

### 9.2 Tuning Guidelines

The examples explicitly enable FlashComm1 and seven-token DSpark. On 950DT products, the PD main model uses FP8 KV cache while the DSpark draft uses BF16 KV cache. Preserve that distinction when tuning; the draft cache dtype is configured inside `--speculative-config`.

Change one parameter at a time and retain the command and benchmark result for each run. For general methods, refer to [Optimization and Tuning](../../developer_guide/performance_and_debug/optimization_and_tuning.md). For feature descriptions and configuration constraints, refer to the [Feature Matrix](../../user_guide/support_matrix/feature_matrix.md) and [Speculative Decoding Guide](../../user_guide/feature_guide/speculative_decoding.md#speculating-using-dspark).

## 10 FAQ

For common environment, installation, and general parameter issues, refer to the [Public FAQs](../../faqs.md). This section covers MiniMax-M3.1-Flash-specific deployment issues.

### 10.1 DSpark Draft Initialization Fails with FP8 KV Cache

**Symptom:** A 950DT products PD service fails while initializing the DSpark draft because FP8 KV quantization parameters are missing.

**Cause:** The draft inherits the main model's FP8 KV cache configuration unless a draft cache dtype is specified.

**Resolution:** Keep the main-model argument `--kv-cache-dtype fp8`, and set `kv_cache_dtype` to `bfloat16` inside `--speculative-config` on both Prefill and Decode:

```json
{
    "method": "dspark",
    "num_speculative_tokens": 7,
    "enforce_eager": true,
    "kv_cache_dtype": "bfloat16",
    "draft_sample_method": "probabilistic"
}
```

The scripts in Section 5.3 already include this setting. It changes only the draft KV cache dtype.

### 10.2 Model Loading Is Still in Progress or Startup Fails

**Symptom:** The API is not ready while loading the approximately 428B-parameter checkpoint or capturing graphs.

**Cause:** Weight loading, initialization, and graph capture depend on storage speed, the hardware, and the selected configuration. Elapsed time alone does not distinguish ongoing initialization from a failure.

**Resolution:** Follow the foreground logs for single-node deployment. For PD deployment, use `tail -f prefill.log` on Prefill; use `tail -f decode-0.log decode-1.log decode-2.log decode-3.log` on A3 D1 and `tail -f decode-4.log decode-5.log decode-6.log decode-7.log` on A3 D2; use `tail -f decode.log` on each 950DT products Decode node. Allow initialization to continue while logs show progress. Send requests or start the PD proxy after every required service prints `Application startup complete`; the single-node health check must return `200`.

If logs stop progressing or the process exits, inspect the first error in every affected role log. Check that `MODEL_PATH` points to the matching weight variant mounted in Section 4, confirm the expected devices are visible with `npu-smi info`, and compare the actual startup command with the corresponding platform example. For PD communication failures, also recheck `LOCAL_IP`, `NIC_NAME`, and [inter-node connectivity](#32-verify-multi-node-communication). For memory errors, review the context budget and concurrency described in [Section 5](#5-online-service-deployment) and the [out-of-memory FAQ](../../faqs.md#12-how-to-handle-the-out-of-memory-issue).
