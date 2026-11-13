# MiniMax-M3.1-Flash

## 1 Introduction

!!! note

    The "PD Separation Deployment" mentioned in this document is provided only as an example/reference design to demonstrate the basic process of deploying Prefill and Decode in separated mode. It is not recommended for use in the production environment.

    **For the production environment on the customer side, [MindIE Motor](https://www.hiascend.com/document/detail/en/mindie/latest/quickstart/docs/en/user_guide/quick_start.md) + vLLM-Ascend is recommended.**

MiniMax-M3.1-Flash is a multimodal Mixture-of-Experts model with approximately 428B total parameters and 23B activated parameters per token. It uses sparse attention throughout the model.

This document covers supported features, environment and model preparation, single-node deployment, PD separation, functional verification, accuracy and performance evaluation, performance tuning, and troubleshooting.

MiniMax-M3.1-Flash is supported on the vLLM Ascend `main` branch and in `v0.32.0rc`. This guide is written based on the `main` branch. Use the deployment image for your hardware from [Installation](#4-installation).

The features described below are those enabled by the deployment commands in this guide. Refer to [Supported Features](#2-supported-features) for the model support matrix.

## 2 Supported Features

Refer to the [Supported Models](../../user_guide/support_matrix/supported_models.md) page for the support matrix and the [Feature Guide](../../user_guide/feature_guide/index.md) for feature configuration.

## 3 Prerequisites

### 3.1 Model Weight

The official release of MiniMax-M3.1-Flash is available on [Hugging Face](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash). However, its routed MoE experts use the NVFP4 data format, which cannot run directly on Ascend hardware. The following quantized weights are adapted for Ascend:

| Weight Version | Deployment | Model Path |
| --- | --- | --- |
| MiniMax-M3.1-Flash-W8A8 | Atlas 800I A2, two-node PD colocation <br> Atlas 800I A3, single-node deployment or three-host PD disaggregation (1P1D, Decode spans two hosts) | [ModelScope](https://www.modelscope.cn/models/Eco-Tech/Minimax-M3.1-Flash-w8a8) |
| MiniMax-M3.1-Flash-MXFP8 | 950DT products, single-node deployment or three-host PD disaggregation (1P1D, Decode spans two hosts) | [ModelScope](https://www.modelscope.cn/models/Eco-Tech/Minimax-M3.1-Flash-w8a8-mxfp8) |

Download the weight variant used by the serving commands to the same model directory on all participating nodes. It is recommended to place the model weight in a shared cache directory.

The Ascend-adapted W8A8 and W8A8-MXFP8 weights above are published on ModelScope only; the Hugging Face repository contains the original NVFP4 checkpoint, which cannot be loaded on Ascend.

Record the downloaded model directories, for example `/models/MiniMax-M3.1-Flash-W8A8` and `/models/MiniMax-M3.1-Flash-W8A8-MXFP8`, and ensure they are accessible inside the container. The commands in this guide use the placeholders `<MINIMAX_M31_FLASH_W8A8_PATH>` and `<MINIMAX_M31_FLASH_W8A8_MXFP8_PATH>` in Section 4.1, and the `MODEL_PATH` variable in Section 5. Replace them with the paths you recorded.

### 3.2 Verify Multi-node Communication

Prepare the business network and the NPU interconnect by following
[Verify inter-node connectivity](../../getting_started/installation.md#installation-multi-node-interconnect).

On each host, use `ip addr` or `ifconfig` to identify the host communication IP and the interface that owns it. Set `LOCAL_IP` to that IP and `NIC_NAME` to the interface name; for example, if the IP is assigned to `eth0`, set `NIC_NAME="eth0"`. The scripts use this interface for both `HCCL_SOCKET_IFNAME` and `GLOO_SOCKET_IFNAME`. See [multi-node host configuration](../../developer_guide/contribution/multi_node_test.md#2-test-without-kubernetes) for selecting mutually reachable host IPs.

## 4 Installation

### 4.1 Docker Image Installation

Choose the tab for your hardware and run the commands on each participating host.

The image tags below are fixed to the release that matches this guide. If your vLLM-Ascend version differs, replace the tag with the one for your release; see [Using Docker](../../getting_started/installation.md#installation-prebuilt-image).

=== "A2 series (W8A8)"

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

=== "A3 series (W8A8)"

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

=== "950DT products"

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

After the container starts, check that it is running and that the expected NPUs are visible:

```bash
docker ps --filter name=minimax_m31_flash
docker exec minimax_m31_flash bash -lc 'npu-smi info'
```

Expected result: the container status is `Up`; `npu-smi info` lists 8 NPUs on A2 and 950DT products, and 16 NPUs on A3.

### 4.2 Source Code Installation

If you prefer not to use the Docker image, build both packages from source. MiniMax-M3.1-Flash support is available on the vLLM Ascend `main` branch, so install that branch rather than an older release.

1. Clone and install vLLM:

    ```bash
    git clone https://github.com/vllm-project/vllm.git
    cd vllm
    pip install -v -e .
    ```

2. Clone and install vLLM Ascend (`main` branch):

    ```bash
    git clone https://github.com/vllm-project/vllm-ascend.git
    cd vllm-ascend
    git checkout main
    pip install -v -e .
    ```

3. Verify the installation:

    ```bash
    pip show vllm vllm-ascend
    ```

    Expected result: both packages are listed with the expected versions.

When the service spans multiple nodes, install the same vLLM and vLLM Ascend revisions in every container so that the distributed group uses identical code.

## 5 Online Service Deployment

Use the container and model weights for your hardware from [Installation](#4-installation).

For descriptions of the standard `vllm serve` arguments used in the deployment examples, refer to the [vLLM Serving Arguments documentation](https://docs.vllm.ai/en/latest/cli/serve/#arguments). For Ascend-specific options passed through `--additional-config`, refer to [Additional Configuration](../../user_guide/configuration/additional_config.md). For Ascend-specific environment variables, refer to [Environment Variables](../../user_guide/configuration/env_vars.md).

**Note**: In the scripts below, `max-num-seqs` represents the maximum number of sequences the scheduler can process in a single iteration. Adjust the `max-num-seqs` parameter dynamically based on actual business.

`MAX_MODEL_LEN` is a shell variable whose value is passed to the vLLM argument `--max-model-len`. The examples use the following context budgets:

| Deployment | MAX_MODEL_LEN |
| --- | --- |
| Single-node, A3 or 950DT products | 133000 |
| PD colocation, A2 | 133000 |
| PD disaggregation, A3 | 133000 |
| PD disaggregation, 950DT products | 133000 |
| 1M PD disaggregation, A3 ([Section 5.5](#55-1m-token-context-deployment)) | 1050624 |

These values belong to different deployment profiles. Weights, communication buffers, parallelism, and KV caches share device memory, so a context budget that fits one profile may not fit another. Select a value that covers the prompt and generated tokens together, and check available KV cache capacity at the intended concurrency. The `1M` model limit in the support matrix does not mean every example configuration reserves enough memory for 1M-token requests. To use the `263000` setting reported in [Text Evaluation](#72-text-evaluation), update `MAX_MODEL_LEN` in the serving scripts, including both roles for PD deployment, and recheck memory capacity before evaluation.

For text-only deployment, `--limit-mm-per-prompt` can be omitted. For multimodal deployment, configure this parameter according to the actual request shape. For example, use `--limit-mm-per-prompt '{"image":2,"video":0}'` for two-image requests, and use `--limit-mm-per-prompt '{"image":0,"video":1}'` for one-video requests.

### 5.1 Single-Node Online Deployment

Single-node deployment runs Prefill and Decode on the same host. The A3 example uses W8A8 weights on all 16 NPUs, and the 950DT products example uses W8A8-MXFP8 weights on all eight NPUs. Both use [DSpark speculative decoding](../../user_guide/feature_guide/speculative_decoding.md#speculating-using-dspark), which generates draft tokens and verifies them with the target model.

=== "A3 series (W8A8)"

    **Start the Service**

    Run the following inside the A3 container. It starts one API service with DP4 × TP4 / EP16 across all 16 NPUs, using W8A8 weights and DSpark speculative decoding with seven draft tokens.

    Set `MODEL_PATH` to your model weight directory; the following example uses the container mount path configured in [Section 4](#4-installation). Set `HCCL_IF_IP` to the communication IP of the current host, that is, the IP used as `LOCAL_IP` in [Section 3.2](#32-verify-multi-node-communication).

    ```bash
    export MODEL_PATH="/models/MiniMax-M3.1-Flash-W8A8"
    export MAX_MODEL_LEN=133000
    export HCCL_IF_IP="<CURRENT_NODE_IP>"

    export VLLM_USE_V2_MODEL_RUNNER=1
    export HCCL_BUFFSIZE=1024
    export HCCL_IF_BASE_PORT=61000
    export HCCL_OP_EXPANSION_MODE=AIV
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

    vllm serve "$MODEL_PATH" \
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
          "weight_nz_mode": 1
      }' \
      --speculative-config '{
          "method":"dspark",
          "num_speculative_tokens":7,
          "enforce_eager":true,
          "kv_cache_dtype":"bfloat16"
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

    Expected result: `200`. Send requests to port `11223` with model name `minimax-m3.1-flash`. If startup is still in progress or fails, follow [Startup Troubleshooting](#102-model-loading-is-still-in-progress-or-startup-fails).

=== "950DT products"

    **Start the Service**

    Run the following inside the 950DT products container. It uses DP2 × TP4 / EP8 across all eight NPUs, W8A8-MXFP8 weights, and DSpark speculative decoding with seven draft tokens.

    Set `MODEL_PATH` to your model weight directory; the following example uses the container mount path configured in [Section 4](#4-installation).

    ```bash
    export MODEL_PATH="/models/MiniMax-M3.1-Flash-W8A8-MXFP8"
    export MAX_MODEL_LEN=133000

    export VLLM_USE_V2_MODEL_RUNNER=1
    export HCCL_BUFFSIZE=1024
    export HCCL_OP_EXPANSION_MODE="CCU_SCHED"
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

    vllm serve "$MODEL_PATH" \
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
          "kv_cache_dtype":"bfloat16"
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

    Expected result: `200`. Send requests to port `11223` with model name `minimax-m3.1-flash`. If startup is still in progress or fails, follow [Startup Troubleshooting](#102-model-loading-is-still-in-progress-or-startup-fails).

### 5.2 Multi-Node Deployment

Multi-node deployment spreads one data-parallel group across multiple hosts. Every host runs Prefill and Decode in the same engine instance (PD colocation), and the hosts join one distributed group. Both nodes use DSpark speculative decoding with seven draft tokens. Configure the host-specific values at the beginning of each node script before starting the service.

Monitor node log while weights load and graph capture completes. Startup duration depends on storage, hardware, and the selected configuration; use continuing log progress and `Application startup complete` to judge readiness. If logs stop progressing or report an error, follow [Startup Troubleshooting](#102-model-loading-is-still-in-progress-or-startup-fails).

There are two A2 nodes, DP4 × TP4 / EP16 in total. The following values match the configurations in the node scripts below and summarize the differences between the two nodes.

| Parameter | Node 0 | Node 1 |
| --- | --- | --- |
| Role | API server | Headless worker |
| DP × TP | 2 × 4 | 2 × 4 |
| Data-parallel rank | 0, 1 | 2, 3 |
| Tool-call parsing | Enabled | — |
| max-num-seqs | 32 | 32 |

=== "A2 series (W8A8)"

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
            "enforce_eager":true
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
            "enforce_eager":true
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

    Expected result: `200`. Send requests to port `11223` on node 0 with model name `minimax-m3.1-flash`.

### 5.3 PD Disaggregation (1P1D)

Both platforms use a 1P1D topology: one Prefill service on one host and one distributed Decode service spanning two hosts. The two Decode hosts share one global data-parallel group; D1 and D2 identify physical hosts within that service. Multiple Decode API endpoints represent ranks or local rank groups within the same distributed service. Prefill processes the prompt, Decode generates tokens, and a proxy forwards requests between them while Mooncake transfers the KV cache. For the general workflow, refer to [Mooncake](../features/pd_disaggregation_mooncake_multi_node.md). Both platforms use DSpark speculative decoding with seven draft tokens. Configure the host-specific values at the beginning of each Prefill and Decode script before starting the service.

Start Prefill, D1, and D2 within a few seconds of each other, and start the proxy only after every engine prints `Application startup complete`. Do not wait for one role to become ready before starting the next: a role that is still missing blocks the distributed rendezvous, and the roles started earlier fail when `VLLM_ENGINE_READY_TIMEOUT_S` expires before the group is complete.

Monitor the role logs while weights load and graph capture completes. Startup duration depends on storage, hardware, and the selected configuration; use continuing log progress to monitor initialization, and wait for `Application startup complete` on every HTTP-serving instance to judge readiness. If logs stop progressing or report an error, follow [Startup Troubleshooting](#102-model-loading-is-still-in-progress-or-startup-fails) before starting the proxy.

**A3 series (W8A8) deployment configuration**

Use one Prefill host and two hosts for the distributed Decode service (1P1D), each with all 16 NPUs and the W8A8 weights. Prefill uses DP1 × TP4 × PP4. Decode uses one global DP8 × TP4 × PP1 group across the two hosts: D1 runs global DP ranks 0–3 and D2 runs ranks 4–7, and each rank starts one TP4 replica on four local NPUs. Every Decode rank exposes its own HTTP API, and the proxy registers all eight Decode endpoints. Use the same image and weights on all three hosts.

The following values match the configurations in the Prefill and Decode scripts below and summarize the differences between the two roles.

| Parameter | Prefill | Decode |
| --- | --- | --- |
| DP × TP × PP | 1 × 4 × 4 | 8 × 4 × 1 across two nodes |
| Global DP ranks | 0 | D1: 0–3; D2: 4–7 |
| API port | 19100 | D1: 19200–19203; D2: 19300–19303 |
| DP RPC port | 16920 | 16920, D1 is the DP master |
| KV port | 24000 | 25000 + rank × 32 (D1); 26000 + local rank × 32 (D2) |
| PP layer partition | 15,15,15,15 | — |
| max-num-seqs | 8 | 128 per replica |
| max-num-batched-tokens | 8192 | 2048 |
| gpu-memory-utilization | 0.85 | 0.92 |
| Execution mode | Eager | FULL_DECODE_ONLY |
| HCCL_BUFFSIZE (MB) | 512 | 1024 |

**950DT products deployment configuration**

Use one Prefill host and two hosts for the distributed Decode service (1P1D), each with all eight NPUs and the W8A8-MXFP8 weights. Prefill uses DP1 × TP4 × PP2. The two Decode hosts form one global DP4 × TP4 × PP1 / EP16 group with two local DP ranks per host: D1 runs ranks 0–1 and D2 runs ranks 2–3. Both Decode hosts serve the HTTP API and are registered separately as proxy endpoints, so do not add `--headless` to the D2 script. The verified configuration uses `--data-parallel-size-local 2` on both hosts and `--data-parallel-start-rank 2` on D2. Each host exposes one API endpoint serving its two local DP ranks; retaining the API on D2 does not change its local DP size. D1 is the Data Parallel master and the two Decode scripts must use the same `DECODE_HEAD_IP`, DP RPC port, and KV port.

The following values match the configurations in the Prefill and Decode scripts below and summarize the differences between the two roles. They use 128K input and 1K output; see the configuration table after the role summary for the 16K input variant.

| Parameter | Prefill | Decode |
| --- | --- | --- |
| DP × TP × PP | 1 × 4 × 2 | 4 × 4 × 1 across two nodes |
| Local DP ranks | 0 | D1: 0–1; D2: 2–3 |
| API port | 31050 | 31060 on both nodes |
| DP RPC port | 6884 | 6885, D1 is the DP master |
| KV port | 30000 | 26900 |
| max-num-seqs | 128 | 128 per DP engine |
| max-num-batched-tokens | 8000 | 8000 |
| gpu-memory-utilization | 0.85 | 0.92 |
| HCCL_BUFFSIZE (MB) | 512 | 1024 |
| Main-model execution | Eager | FULL_DECODE_ONLY with 1024-token graph capture |
| KV cache dtype | fp8 | fp8 (DSpark draft uses bfloat16) |

All three scripts use `MAX_MODEL_LEN=133000`, which covers a 128K prompt plus 1K generated tokens. For 16K input workloads, only the Decode side changes: lower the sequence limit and fall back to the default graph capture size, which is enough for 64 sequences × (1 + 7 draft tokens) = 512 tokens per step.

| Parameter | 128K input / 1K output | 16K input / 1K output |
| --- | --- | --- |
| Decode `MAX_NUM_SEQS` | 128 | 64 |
| Decode `--compilation-config` | `{"cudagraph_mode":"FULL_DECODE_ONLY","max_cudagraph_capture_size":1024}` | `{"cudagraph_mode":"FULL_DECODE_ONLY"}` |
| All other parameters | Unchanged | Unchanged; keep `MAX_MODEL_LEN=133000` and `MAX_BATCHED_TOKENS=8000` |

The capture size is not cosmetic: with 128 sequences and seven draft tokens, one decode step holds 1024 tokens, and the default capture limit of 512 makes the scheduler fall back to eager execution for the oversized batch, which raises TPOT by an order of magnitude. With 64 sequences the default limit is sufficient.

**Deployment scripts**

=== "A3 series (W8A8)"

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
    MAX_NUM_SEQS=8
    MAX_MODEL_LEN=133000
    MAX_BATCHED_TOKENS=8192
    GPU_MEMORY_UTILIZATION=0.85

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
    export ASCEND_LOCAL_COMM_RES='{"version":"1.2"}'
    export VLLM_USE_FASTOKENS=1

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
        --disable-hybrid-kv-cache-manager \
        --speculative-config '{"method":"dspark","num_speculative_tokens":7,"enforce_eager":true,"kv_cache_dtype":"bfloat16"}' \
        --reasoning-parser minimax_m3 \
        --data-parallel-size 1 \
        --enforce-eager \
        --long-prefill-token-threshold 2048 \
        --additional-config '{
            "enable_cpu_binding":true,
            "multistream_overlap_shared_expert":true,
            "enable_shared_expert_dp":true,
            "weight_nz_mode":1,
            "enable_flashcomm1":true,
            "enable_fused_mc2":0
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

    Save the following as `run_decode_d1.sh` on D1. Set `LOCAL_IP` and `NIC_NAME` for this host. The script also uses this IP as the Data Parallel master address of the global DP8 group, so D2 must point at the same IP. Keep the image, weights, global parallelism, and RPC port identical on both Decode nodes.

    ```bash
    #!/usr/bin/env bash
    set -e

    # Configure these values for the current host.
    MODEL_PATH="/models/MiniMax-M3.1-Flash-W8A8"
    LOCAL_IP="<CURRENT_NODE_IP>"
    NIC_NAME="<CURRENT_NODE_NIC>"
    DP_RANK=${1:-0}
    SERVICE_PORT=$((19200 + DP_RANK))
    DP_RPC_PORT=16920
    KV_PORT=$((25000 + DP_RANK*32))
    MAX_NUM_SEQS=128
    MAX_MODEL_LEN=133000
    MAX_BATCHED_TOKENS=2048
    GPU_MEMORY_UTILIZATION=0.92

    # Fixed runtime configuration.
    export VLLM_USE_V2_MODEL_RUNNER=1
    export HCCL_BUFFSIZE=1024
    export HCCL_IF_IP="$LOCAL_IP"
    unset HCCL_INTRA_ROCE_ENABLE
    export HCCL_OP_EXPANSION_MODE=AIV
    export HCCL_SOCKET_IFNAME="$NIC_NAME"
    export ASCEND_RT_VISIBLE_DEVICES="$((DP_RANK*4)),$((DP_RANK*4+1)),$((DP_RANK*4+2)),$((DP_RANK*4+3))"
    export GLOO_SOCKET_IFNAME="$NIC_NAME"
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export ASCEND_LOCAL_COMM_RES='{"version":"1.2"}'

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
        --disable-hybrid-kv-cache-manager \
        --speculative-config '{"method":"dspark","num_speculative_tokens":7,"enforce_eager":true,"kv_cache_dtype":"bfloat16"}' \
        --reasoning-parser minimax_m3 \
        --data-parallel-size 8 \
        --data-parallel-size-local 1 \
        --data-parallel-rank "$DP_RANK" \
        --data-parallel-address "$LOCAL_IP" \
        --data-parallel-rpc-port "$DP_RPC_PORT" \
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","max_cudagraph_capture_size":512}' \
        --additional-config '{
            "enable_cpu_binding":true,
            "multistream_overlap_shared_expert":true,
            "enable_shared_expert_dp":true,
            "enable_flashcomm1":true,
            "weight_nz_mode":1
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

    Start all four D1 replicas; each replica serves one global DP rank:

    ```bash
    for rank in 0 1 2 3; do
        nohup bash run_decode_d1.sh "$rank" > "decode-$rank.log" 2>&1 &
    done
    ```

    **Decode Node 2 (D2, DP ranks 4–7)**

    Save the following as `run_decode_d2.sh` on D2. Set `LOCAL_IP` and `NIC_NAME` for this host, and set `DECODE_HEAD_IP` to the D1 IP used above. The argument is the local rank on this node; global DP ranks are offset by four.

    ```bash
    #!/usr/bin/env bash
    set -e

    # Configure these values for the current host.
    MODEL_PATH="/models/MiniMax-M3.1-Flash-W8A8"
    LOCAL_IP="<CURRENT_NODE_IP>"
    NIC_NAME="<CURRENT_NODE_NIC>"
    DECODE_HEAD_IP="<DECODE1_NODE_IP>"
    LOCAL_RANK=${1:-0}
    DP_RANK=$((LOCAL_RANK + 4))
    SERVICE_PORT=$((19300 + LOCAL_RANK))
    DP_RPC_PORT=16920
    KV_PORT=$((26000 + LOCAL_RANK*32))
    MAX_NUM_SEQS=128
    MAX_MODEL_LEN=133000
    MAX_BATCHED_TOKENS=2048
    GPU_MEMORY_UTILIZATION=0.92

    # Fixed runtime configuration.
    export VLLM_USE_V2_MODEL_RUNNER=1
    export HCCL_BUFFSIZE=1024
    export HCCL_IF_IP="$LOCAL_IP"
    unset HCCL_INTRA_ROCE_ENABLE
    export HCCL_OP_EXPANSION_MODE=AIV
    export HCCL_SOCKET_IFNAME="$NIC_NAME"
    export ASCEND_RT_VISIBLE_DEVICES="$((LOCAL_RANK*4)),$((LOCAL_RANK*4+1)),$((LOCAL_RANK*4+2)),$((LOCAL_RANK*4+3))"
    export GLOO_SOCKET_IFNAME="$NIC_NAME"
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export ASCEND_LOCAL_COMM_RES='{"version":"1.2"}'

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
        --disable-hybrid-kv-cache-manager \
        --speculative-config '{"method":"dspark","num_speculative_tokens":7,"enforce_eager":true,"kv_cache_dtype":"bfloat16"}' \
        --reasoning-parser minimax_m3 \
        --data-parallel-size 8 \
        --data-parallel-size-local 1 \
        --data-parallel-rank "$DP_RANK" \
        --data-parallel-address "$DECODE_HEAD_IP" \
        --data-parallel-rpc-port "$DP_RPC_PORT" \
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","max_cudagraph_capture_size":512}' \
        --additional-config '{
            "enable_cpu_binding":true,
            "multistream_overlap_shared_expert":true,
            "enable_shared_expert_dp":true,
            "enable_flashcomm1":true,
            "weight_nz_mode":1
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

    Start all four D2 replicas:

    ```bash
    for rank in 0 1 2 3; do
        nohup bash run_decode_d2.sh "$rank" > "decode-$((rank+4)).log" 2>&1 &
    done
    ```

    **Load-balance Proxy**

    Download [load_balance_proxy_server_example.py](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py) from the official repository and save it as `/workspace/load_balance_proxy_server_example.py` on a node that can reach all Prefill and Decode API ports.

    Save the following as `proxy_server.sh` and start it after Prefill and all eight Decode replicas print `Application startup complete`:

    ```bash
    #!/usr/bin/env bash
    set -e

    export no_proxy="${no_proxy:+${no_proxy},}<PREFILL_NODE_IP>,<DECODE1_NODE_IP>,<DECODE2_NODE_IP>"
    export NO_PROXY="$no_proxy"

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
            19300 19301 19302 19303
    ```

    ```bash
    bash proxy_server.sh
    ```

    Send requests to port `19080` with model name `minimax-m3.1-flash`.

=== "950DT products"

    **Prefill Node (P)**

    Save the following as `run_prefill.sh` on the Prefill node.

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
    MAX_BATCHED_TOKENS=8000
    GPU_MEMORY_UTILIZATION=0.85

    # Fixed runtime configuration.
    export VLLM_USE_FASTOKENS=1
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
        --additional-config '{
            "enable_cpu_binding": true,
            "multistream_overlap_shared_expert": true,
            "enable_shared_expert_dp": true,
            "enable_flashcomm1": true,
            "enable_fused_mc2": 0
        }' \
        --speculative-config '{
            "method":"dspark",
            "num_speculative_tokens":7,
            "enforce_eager":true,
            "kv_cache_dtype":"bfloat16"
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

    Start the Prefill engine:

    ```bash
    nohup bash run_prefill.sh > prefill.log 2>&1 &
    ```

    **Decode Head Node (D1, DP ranks 0–1)**

    Save the following as `run_decode_head.sh` on D1. Set `LOCAL_IP` and `NIC_NAME` for this host. D1 is the Data Parallel master, so the script uses its own IP as `DECODE_HEAD_IP`; D2 must be configured with the same value.

    ```bash
    #!/usr/bin/env bash
    set -e

    # Configure these values for the current host.
    MODEL_PATH="/models/MiniMax-M3.1-Flash-W8A8-MXFP8"
    LOCAL_IP="<CURRENT_NODE_IP>"
    NIC_NAME="<CURRENT_NODE_NIC>"
    # D1 is the Data Parallel master; D2 must use the same value.
    DECODE_HEAD_IP="$LOCAL_IP"
    SERVICE_PORT=31060
    DP_RPC_PORT=6885
    KV_PORT=26900
    MAX_NUM_SEQS=128
    MAX_MODEL_LEN=133000
    MAX_BATCHED_TOKENS=8000
    GPU_MEMORY_UTILIZATION=0.92

    # Fixed runtime configuration.
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
        --speculative-config '{
            "method":"dspark",
            "num_speculative_tokens":7,
            "enforce_eager":true,
            "kv_cache_dtype":"bfloat16"
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

    Start the Decode head:

    ```bash
    nohup bash run_decode_head.sh > decode.log 2>&1 &
    ```

    **Decode Worker Node (D2, DP ranks 2–3)**

    Save the following as `run_decode_worker.sh` on D2. Set `LOCAL_IP` and `NIC_NAME` for this host, and set `DECODE_HEAD_IP` to the same D1 IP used above. Keep the image, weights, parallelism, and RPC/KV ports identical to D1. This node also serves the HTTP API and is registered as the second Decode endpoint of the proxy; do not add `--headless`.

    ```bash
    #!/usr/bin/env bash
    set -e

    # Configure these values for the current host.
    MODEL_PATH="/models/MiniMax-M3.1-Flash-W8A8-MXFP8"
    LOCAL_IP="<CURRENT_NODE_IP>"
    NIC_NAME="<CURRENT_NODE_NIC>"
    DECODE_HEAD_IP="<DECODE1_NODE_IP>"
    SERVICE_PORT=31060
    DP_RPC_PORT=6885
    KV_PORT=26900
    MAX_NUM_SEQS=128
    MAX_MODEL_LEN=133000
    MAX_BATCHED_TOKENS=8000
    GPU_MEMORY_UTILIZATION=0.92

    # Fixed runtime configuration.
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
        --speculative-config '{
            "method":"dspark",
            "num_speculative_tokens":7,
            "enforce_eager":true,
            "kv_cache_dtype":"bfloat16"
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

    Start the Decode worker at the same time as D1:

    ```bash
    nohup bash run_decode_worker.sh > decode.log 2>&1 &
    ```

    Both Decode nodes must join before the distributed Decode group can become ready.

    **Load-balance Proxy**

    Download [load_balance_proxy_server_example.py](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py) from the official repository and save it as `/workspace/load_balance_proxy_server_example.py` on a node that can reach the Prefill and both Decode API ports.

    After Prefill, D1, and D2 print `Application startup complete`, save the following as `proxy_server.sh` and start it:

    ```bash
    #!/usr/bin/env bash
    set -e

    export no_proxy="${no_proxy:+${no_proxy},}<PREFILL_NODE_IP>,<DECODE1_NODE_IP>,<DECODE2_NODE_IP>"
    export NO_PROXY="$no_proxy"

    PREFILL_IP="<PREFILL_NODE_IP>"
    DECODE1_IP="<DECODE1_NODE_IP>"
    DECODE2_IP="<DECODE2_NODE_IP>"

    python3 /workspace/load_balance_proxy_server_example.py \
        --host 0.0.0.0 \
        --port 19080 \
        --prefiller-hosts "$PREFILL_IP" \
        --prefiller-ports 31050 \
        --decoder-hosts "$DECODE1_IP" "$DECODE2_IP" \
        --decoder-ports 31060 31060 \
        --max-retries 0 \
        --workers 4
    ```

    ```bash
    bash proxy_server.sh
    ```

    Each Decode node registers one endpoint that fronts its two local DP ranks, so the proxy lists two Decode hosts even though the DP group has four ranks. Send requests to port `19080` with model name `minimax-m3.1-flash`.

### 5.4 Prefill-Decode Disaggregation with KV Cache Pool

This section builds on [Section 5.3](#53-pd-disaggregation-1p1d) and keeps its 1P1D topology, role scripts, proxy mapping, and MiniMax-M3.1-Flash `vllm serve` settings on both platforms. On Atlas 800I A3, one Prefill service runs with `DP1 × TP4 × PP4` and one distributed Decode service runs with a global `DP8 × TP4 × PP1` group spanning two hosts, with four local ranks per host. On 950DT products, one Prefill service runs with `DP1 × TP4 × PP2` and one distributed Decode service runs with a global `DP4 × TP4 × PP1` group spanning two hosts, with two local ranks per host. Prefill and Decode switch to `MultiConnector` so live P→D KV transfer and Mooncake KV Cache Pool work together:

- `MooncakeConnectorV1` transfers KV from Prefill to Decode in real time (same role as Section 5.3).
- `AscendStoreConnector` stores KV in the Mooncake KV Cache Pool so later Prefills with the same prefix can hit the pool instead of recomputing.
- By default the Prefill node performs pool lookup, load, and write.

The examples below enable three-tier KV storage: the active NPU KV cache, the Mooncake memory pool, and SSD offload.

For backend selection, `mooncake.json`, Mooncake Master, eviction, tenant options, and hardware-specific pooling exports, refer to the [KV Cache Pool Deployment Guide](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md). For MiniMax-M3.1-Flash-specific issues, refer to [Chapter 10 FAQ](#10-faq).

**Compared with normal PD in Section 5.3, pay attention to these pooling-only requirements:**

| Item | Normal PD (5.3) | Pooled PD (this section) |
| ---- | --------------- | ------------------------ |
| Connector | Single `MooncakeConnectorV1` | `MultiConnector` wrapping `MooncakeConnectorV1` + `AscendStoreConnector` |
| Mooncake Master | Not required | Start `mooncake_master --enable_offload=true` before Decode / Prefill |
| `mooncake.json` | Not required | Required on every rank; A3 P and both D nodes use `4GB`, 950DT P and both D nodes use `64GB`; enable SSD offload on all nodes |
| `lookup_rpc_port` | Not used | **Must be unique per DP rank** (`37000/37100 + DP_RANK`); on 950DT products use one distinct base port per host (D1 `37100`, D2 `37102`) to avoid IPC endpoint collisions |
| Extra env | Section 5.3 `HCCL_*` only | Keep 5.3 env, then add A3 fabric-memory or 950DT Device UB exports from [KV Cache Pool Deployment Guide](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md) |
| Container mounts | `/etc/hccn.conf`; 950DT products also keep `/etc/hixlep/` | Also mount the local SSD offload directory |
| Startup order | Decode → Prefill → Proxy | **Mooncake Master → Decode → Prefill → Proxy** |
| Verification | P→D KV transfer only | Also check Prefill pool lookup/get/put hits after a repeated-prefix warmup |

#### 5.4.1 Prerequisites

Mount the host HCCN config into every A3 container that participates in pooling:

```bash
-v /etc/hccn.conf:/etc/hccn.conf:ro
```

On 950DT products, keep the `/etc/hixlep/` mount from Section 5.3 as well. The pooled 950DT scripts export `ASCEND_LOCAL_COMM_RES_PATH=/etc/hixlep/`, and the KV pool path uses that directory for Device UB communication. The validated 950DT deployment used only that mount; `/etc/hccn.conf` was not present in its containers and was not required.

Use a local SSD directory on each Prefill and Decode host to store KV cache data offloaded from the memory pool. `/nvme/mooncake_offload` is an example directory inside the container, not a required SSD mount point or a directory that already exists on every host.

On each host, first identify the local SSD and its mount point:

```bash
lsblk -o NAME,TYPE,SIZE,ROTA,FSTYPE,MOUNTPOINTS,MODEL
df -hT
```

Use `MODEL` to identify the SSD and `MOUNTPOINTS` to find its mounted filesystem. `ROTA=0` indicates a non-rotational device; confirm the storage type with the machine's disk information, especially for RAID or virtual disks. Choose an existing local SSD filesystem with enough free space. A shared network filesystem is not the local SSD tier.

For example, if the SSD is mounted at `/data/nvme`, create the offload directory there before starting the container. Replace `/data/nvme` below with the actual SSD mount point:

```bash
mkdir -p /data/nvme/mooncake_offload
findmnt -T /data/nvme/mooncake_offload
df -hT /data/nvme/mooncake_offload
```

Confirm that `findmnt` reports the selected SSD filesystem and that `df` shows enough available space. Creating a directory named `/nvme` alone does not mount an SSD or ensure that data is stored on one.

Add this bind mount when creating each model container:

```bash
-v /data/nvme/mooncake_offload:/nvme/mooncake_offload
```

The left side is the directory on that host's SSD; the right side is the directory visible inside the container. With this mapping, keep `"ssd_offload_path": "/nvme/mooncake_offload"` in `mooncake.json`. If the container path changes, update `ssd_offload_path` to match. Use an absolute path that exists and is writable by the vLLM process; do not use a symlink. Repeat this setup on P and every D host: identical path strings refer to separate local SSD storage on each host. The backend creates per-rank subdirectories beneath it. The installed Mooncake package must support the `enable_ssd_offload` and `ssd_offload_path` arguments in `MooncakeDistributedStore.setup()`; use the dependencies in the [KV Cache Pool Deployment Guide](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md).

Place `mooncake.json` next to the role scripts on every node (`run_prefill.sh` and the two Decode scripts). Use the same reachable Mooncake Master address and `tenant_id` across P and D. `global_segment_size` is allocated per worker rank, not per node; a non-zero value must be aligned to `1GB`. If `MOONCAKE_GLOBAL_SEGMENT_SIZE` is already exported, update it to the intended size or unset it before startup because it overrides the JSON value.

**A3: Prefill and Decode configuration**

Use the following `mooncake.json` on P, D1, and D2. Replace `<MOONCAKE_MASTER_IP>` with the actual Master IP.

```json
{
  "metadata_server": "P2PHANDSHAKE",
  "protocol": "ascend",
  "device_name": "",
  "master_server_address": "<MOONCAKE_MASTER_IP>:50088",
  "global_segment_size": "4GB",
  "preferred_segment": false,
  "prefer_alloc_in_same_node": true,
  "enable_ssd_offload": true,
  "ssd_offload_path": "/nvme/mooncake_offload",
  "tenant_id": "default"
}
```

The A3 example uses `4GB` per worker rank on P and both D nodes. Adjust this value to each machine's available allocatable memory and fragmentation after reserving memory for vLLM and other processes. Allow additional space for the SSD I/O buffer and allocation headroom, and account for all contributing worker ranks on the node.

**950DT: Prefill configuration**

Use the following `mooncake.json` on Prefill, with `preferred_segment` set to `false`. Replace `<MOONCAKE_MASTER_IP>` with the actual Master IP.

```json
{
  "metadata_server": "P2PHANDSHAKE",
  "protocol": "ascend",
  "device_name": "",
  "master_server_address": "<MOONCAKE_MASTER_IP>:50088",
  "global_segment_size": "64GB",
  "preferred_segment": false,
  "prefer_alloc_in_same_node": true,
  "enable_ssd_offload": true,
  "ssd_offload_path": "/data/mooncake_offload",
  "tenant_id": "default"
}
```

**950DT: Decode configuration**

Use the following `mooncake.json` on both D1 and D2, with `preferred_segment` set to `true`. Keep the Master address and `tenant_id` identical to Prefill.

```json
{
  "metadata_server": "P2PHANDSHAKE",
  "protocol": "ascend",
  "device_name": "",
  "master_server_address": "<MOONCAKE_MASTER_IP>:50088",
  "global_segment_size": "64GB",
  "preferred_segment": true,
  "prefer_alloc_in_same_node": true,
  "enable_ssd_offload": true,
  "ssd_offload_path": "/data/mooncake_offload",
  "tenant_id": "default"
}
```

The 950DT examples use `64GB` per worker rank on Prefill and both Decode hosts. The validated deployment used `/data/mooncake_offload` because those hosts have no `/nvme` mount; `/data` is a local ext4 filesystem on each host and is visible inside the container with the same path. If your hosts use a different local SSD directory, bind-mount it into the container and set `ssd_offload_path` to the container path. As with the A3 example, adjust the segment size to each machine's allocatable memory and fragmentation instead of treating `64GB` as a fixed requirement.

#### 5.4.2 Environment Variables

Start from the Section 5.3 environment block for your platform (`HCCL_IF_IP`, `HCCL`/`GLOO` socket IFNAME, `HCCL_BUFFSIZE`, `PYTORCH_NPU_ALLOC_CONF`, `ASCEND_RT_VISIBLE_DEVICES`, and, on A3 Prefill, `VLLM_PP_LAYER_PARTITION="15,15,15,15"`). Then add the pooling-required variables below. Choose the hardware/communication exports from the [Mooncake backend startup examples](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md#step-4-start-vllm-inference-service) according to your machine and link type.

Common pooling exports on Prefill node and DP ranks:

```bash
# Optional: only if Mooncake is installed in a non-standard path.
if [ -n "${MOONCAKE_LIB_DIRS:-}" ]; then
    export LD_LIBRARY_PATH="${MOONCAKE_LIB_DIRS}:${LD_LIBRARY_PATH:-}"
fi
export MOONCAKE_CONFIG_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/mooncake.json"
unset MOONCAKE_MASTER MOONCAKE_GLOBAL_SEGMENT_SIZE
export PYTHONHASHSEED=0
# SSD I/O buffer per worker: 1GB.
export MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES=1073741824
# SSD disk budget per NPU worker rank: 10 GiB.
export MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES=$((10 * 1024 * 1024 * 1024))
export MOONCAKE_OFFLOAD_BUCKET_MAX_TOTAL_SIZE=$((10 * 1024 * 1024 * 1024))
export MOONCAKE_OFFLOAD_BUCKET_EVICTION_POLICY=lru
```

On Atlas 800I A3 with the HCCS fabric link, also export `ASCEND_ENABLE_USE_FABRIC_MEM=1`.

On 950DT products, export the Device UB settings and clear the global resource override instead:

```bash
export ASCEND_LOCAL_COMM_RES_PATH=/etc/hixlep/
export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'
unset ASCEND_GLOBAL_RESOURCE_CONFIG
```

The SSD I/O buffer is allocated separately from `global_segment_size`. The `1GB` value above is an aligned starting point; size it for the KV read/write batches and include it in the per-rank memory budget.

For this three-tier (NPU, host memory, and SSD) example, the recommended starting SSD budget is **10 GiB per NPU worker rank** on every Prefill and Decode node. The two capacity limits describe the same SSD budget; they do not add up to 20 GiB:

- `MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES` sets the per-worker SSD capacity reported to Mooncake Master. Its default is `2199023255552` bytes (2 TiB) per worker; this is advertised capacity, not an immediate disk allocation.
- `MOONCAKE_OFFLOAD_BUCKET_MAX_TOTAL_SIZE` sets the local bucket size limit. Its default is `0`, meaning 90% of the physical disk capacity. Set it explicitly alongside the reported capacity.
- `MOONCAKE_OFFLOAD_BUCKET_EVICTION_POLICY=lru` enables least-recently-used eviction, removing old SSD cache entries to make room for new writes within the configured bucket limit.

Adjust both values to the workload and usable SSD space, allowing for all workers sharing that disk and leaving room for other data. Count TP and PP worker ranks, not just DP ranks or physical cards: an A3 node with 16 logical NPU workers has a combined budget of 160 GiB at 10 GiB per worker. These limits control SSD usage; they do not change the host-memory pool or SSD I/O buffer size. For details, see [Mooncake SSD Offload Parameter Tuning](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md#mooncake-ssd-offload-parameter-tuning).

| Hardware | Dependencies | Required pooling exports | Notes |
| -------- | ------------- | ------------------------ | ----- |
| 800I/T A3 (HCCS, recommended) | HDK >= 26.0, or HDK >= 25.5 with mooncake >= v0.3.11; CANN >= 9.0.0 | `export ASCEND_ENABLE_USE_FABRIC_MEM=1` | Keep the Section 5.3 `HCCL_IF_IP` / socket IFNAME exports. |
| 950PR/DT (Device UB) | HDK >= 25.6 with mooncake >= v0.3.11; CANN >= 9.1.0 | `export ASCEND_LOCAL_COMM_RES_PATH=/etc/hixlep/` and `export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'`; `unset ASCEND_GLOBAL_RESOURCE_CONFIG` | Mount `/etc/hixlep/`. For UBOE instead, use `ASCEND_GLOBAL_RESOURCE_CONFIG` as described in the KV Cache Pool guide. |

#### 5.4.3 Prefill / Decode Scripts

**Key parameters**

`MultiConnector` creates a separate `KVTransferConfig` for each entry in `connectors`; child connectors do not inherit the outer `kv_role`. Both child connectors must specify `kv_connector` and `kv_role`. The scripts below set `KV_ROLE=kv_producer` on Prefill and `KV_ROLE=kv_consumer` on Decode, and explicitly pass that role to both children.

| Child connector | Port configuration in this example |
| --- | --- |
| `MooncakeConnectorV1` | Set `kv_port` explicitly for the P→D connection, using the role-specific `KV_PORT` from Section 5.3. |
| `AscendStoreConnector` | No explicit `kv_port` is needed: this connector does not use it. Set `kv_connector_extra_config.lookup_rpc_port` for its lookup IPC endpoint and `backend` to `mooncake`; configure the Mooncake Master address separately in `mooncake.json`. |

The shared `KVTransferConfig` schema provides a default `kv_port`, so omitting it is not equivalent to omitting `kv_role`. In particular, `AscendStoreConnector` still requires its own `kv_role`; do not remove that field from either child.

Reuse the Section 5.3 role scripts and launch commands. Replace each role script with the pooled version below. Keep the Section 5.3 `vllm serve` flag style; only `--kv-transfer-config` switches to `MultiConnector`. Derive the Decode `lookup_rpc_port` from `DP_RANK` so each rank uses a distinct IPC endpoint; Prefill has one DP rank and uses `37000`. On 950DT products, one `vllm serve` process per host serves two local DP ranks and the lookup endpoint is namespaced per local DP rank from the base port, so each host uses one distinct base port (D1 `37100`, D2 `37102`).

**A3 series (W8A8) deployment configuration**

Prefill-Decode disaggregation with KV Cache Pool on 3 Atlas 800I A3 (64GB × 16) for `MiniMax-M3.1-Flash-W8A8` with DSpark speculative decoding.

The 1P1D topology uses one Prefill service with `DP1 × TP4 × PP4` on one host and one distributed Decode service with a global `DP8 × TP4 × PP1` group spanning two hosts, with four local ranks per Decode host. Service settings match Section 5.3; `MultiConnector` adds the KV pool.

**950DT products deployment configuration**

Prefill-Decode disaggregation with KV Cache Pool on 950DT products (96GB × 8) for `MiniMax-M3.1-Flash-W8A8-MXFP8` with DSpark speculative decoding.

The 1P1D topology uses one Prefill service with `DP1 × TP4 × PP2` on one host and one distributed Decode service with a global `DP4 × TP4 × PP1` group spanning two hosts, with two local ranks per Decode host. Both Decode hosts serve the HTTP API on port `31060` and join one data-parallel group; do not add `--headless`. Service settings match Section 5.3; `MultiConnector` adds the KV pool.

**Deployment scripts**

=== "A3 series (W8A8)"

    **1. Prefill node**

    Save as `run_prefill.sh` alongside `mooncake.json`.

    ```bash
    #!/usr/bin/env bash
    set -e

    # Configure these values for the current host.
    MODEL_PATH="/models/MiniMax-M3.1-Flash-W8A8"
    LOCAL_IP="<CURRENT_NODE_IP>"
    NIC_NAME="<CURRENT_NODE_NIC>"
    SERVICE_PORT=19100
    KV_PORT=24000
    MAX_NUM_SEQS=8
    MAX_MODEL_LEN=133000
    MAX_BATCHED_TOKENS=8192
    GPU_MEMORY_UTILIZATION=0.85

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
    export ASCEND_LOCAL_COMM_RES='{"version":"1.2"}'
    export VLLM_USE_FASTOKENS=1

    # Optional: only if Mooncake is installed in a non-standard path.
    if [ -n "${MOONCAKE_LIB_DIRS:-}" ]; then
        export LD_LIBRARY_PATH="${MOONCAKE_LIB_DIRS}:${LD_LIBRARY_PATH:-}"
    fi
    export MOONCAKE_CONFIG_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/mooncake.json"
    unset MOONCAKE_MASTER MOONCAKE_GLOBAL_SEGMENT_SIZE
    export ASCEND_ENABLE_USE_FABRIC_MEM=1
    export PYTHONHASHSEED=0
    # SSD I/O buffer per worker: 1GB, aligned for A3 fabric memory.
    export MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES=1073741824
    # SSD disk budget per NPU worker rank: 10 GiB.
    export MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES=$((10 * 1024 * 1024 * 1024))
    export MOONCAKE_OFFLOAD_BUCKET_MAX_TOTAL_SIZE=$((10 * 1024 * 1024 * 1024))
    export MOONCAKE_OFFLOAD_BUCKET_EVICTION_POLICY=lru
    KV_ROLE=kv_producer
    LOOKUP_RPC_PORT=37000

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
        --disable-hybrid-kv-cache-manager \
        --speculative-config '{"method":"dspark","num_speculative_tokens":7,"enforce_eager":true,"kv_cache_dtype":"bfloat16"}' \
        --reasoning-parser minimax_m3 \
        --data-parallel-size 1 \
        --enforce-eager \
        --long-prefill-token-threshold 2048 \
        --additional-config '{
            "enable_cpu_binding":true,
            "multistream_overlap_shared_expert":true,
            "enable_shared_expert_dp":true,
            "weight_nz_mode":1,
            "enable_flashcomm1":true,
            "enable_fused_mc2":0
        }' \
        --kv-transfer-config '{
            "kv_connector": "MultiConnector",
            "kv_role": "'"$KV_ROLE"'",
            "kv_connector_extra_config": {
                "connectors": [
                    {
                        "kv_connector": "MooncakeConnectorV1",
                        "kv_buffer_device": "npu",
                        "kv_role": "'"$KV_ROLE"'",
                        "kv_port": '"$KV_PORT"',
                        "kv_connector_extra_config": {
                            "use_ascend_direct": true,
                            "prefill": {"dp_size": 1, "tp_size": 4, "pp_size": 4, "pp_layer_partition": "15,15,15,15"},
                            "decode": {"dp_size": 8, "tp_size": 4, "pp_size": 1}
                        }
                    },
                    {
                        "kv_connector": "AscendStoreConnector",
                        "kv_role": "'"$KV_ROLE"'",
                        "kv_connector_extra_config": {
                            "backend": "mooncake",
                            "lookup_rpc_port": '"$LOOKUP_RPC_PORT"'
                        }
                    }
                ]
            }
        }'
    ```

    **2. First Decode node**

    Save as `run_decode_d1.sh` alongside `mooncake.json`.

    ```bash
    #!/usr/bin/env bash
    set -e

    # Configure these values for the current host.
    MODEL_PATH="/models/MiniMax-M3.1-Flash-W8A8"
    LOCAL_IP="<CURRENT_NODE_IP>"
    NIC_NAME="<CURRENT_NODE_NIC>"
    DP_RANK=${1:-0}
    SERVICE_PORT=$((19200 + DP_RANK))
    DP_RPC_PORT=16920
    KV_PORT=$((25000 + DP_RANK*32))
    MAX_NUM_SEQS=128
    MAX_MODEL_LEN=133000
    MAX_BATCHED_TOKENS=2048
    GPU_MEMORY_UTILIZATION=0.92

    # Fixed runtime configuration.
    export VLLM_USE_V2_MODEL_RUNNER=1
    export HCCL_BUFFSIZE=1024
    export HCCL_IF_IP="$LOCAL_IP"
    unset HCCL_INTRA_ROCE_ENABLE
    export HCCL_OP_EXPANSION_MODE=AIV
    export HCCL_SOCKET_IFNAME="$NIC_NAME"
    export ASCEND_RT_VISIBLE_DEVICES="$((DP_RANK*4)),$((DP_RANK*4+1)),$((DP_RANK*4+2)),$((DP_RANK*4+3))"
    export GLOO_SOCKET_IFNAME="$NIC_NAME"
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export ASCEND_LOCAL_COMM_RES='{"version":"1.2"}'

    # Optional: only if Mooncake is installed in a non-standard path.
    if [ -n "${MOONCAKE_LIB_DIRS:-}" ]; then
        export LD_LIBRARY_PATH="${MOONCAKE_LIB_DIRS}:${LD_LIBRARY_PATH:-}"
    fi
    export MOONCAKE_CONFIG_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/mooncake.json"
    unset MOONCAKE_MASTER MOONCAKE_GLOBAL_SEGMENT_SIZE
    export ASCEND_ENABLE_USE_FABRIC_MEM=1
    export PYTHONHASHSEED=0
    # SSD I/O buffer per worker: 1GB, aligned for A3 fabric memory.
    export MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES=1073741824
    # SSD disk budget per NPU worker rank: 10 GiB.
    export MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES=$((10 * 1024 * 1024 * 1024))
    export MOONCAKE_OFFLOAD_BUCKET_MAX_TOTAL_SIZE=$((10 * 1024 * 1024 * 1024))
    export MOONCAKE_OFFLOAD_BUCKET_EVICTION_POLICY=lru
    KV_ROLE=kv_consumer
    LOOKUP_RPC_PORT=$((37100 + DP_RANK))

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
        --disable-hybrid-kv-cache-manager \
        --speculative-config '{"method":"dspark","num_speculative_tokens":7,"enforce_eager":true,"kv_cache_dtype":"bfloat16"}' \
        --reasoning-parser minimax_m3 \
        --data-parallel-size 8 \
        --data-parallel-size-local 1 \
        --data-parallel-rank "$DP_RANK" \
        --data-parallel-address "$LOCAL_IP" \
        --data-parallel-rpc-port "$DP_RPC_PORT" \
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","max_cudagraph_capture_size":512}' \
        --additional-config '{
            "enable_cpu_binding":true,
            "multistream_overlap_shared_expert":true,
            "enable_shared_expert_dp":true,
            "enable_flashcomm1":true,
            "weight_nz_mode":1
        }' \
        --kv-transfer-config '{
            "kv_connector": "MultiConnector",
            "kv_role": "'"$KV_ROLE"'",
            "kv_connector_extra_config": {
                "connectors": [
                    {
                        "kv_connector": "MooncakeConnectorV1",
                        "kv_buffer_device": "npu",
                        "kv_role": "'"$KV_ROLE"'",
                        "kv_port": '"$KV_PORT"',
                        "kv_connector_extra_config": {
                            "use_ascend_direct": true,
                            "prefill": {"dp_size": 1, "tp_size": 4, "pp_size": 4, "pp_layer_partition": "15,15,15,15"},
                            "decode": {"dp_size": 8, "tp_size": 4, "pp_size": 1}
                        }
                    },
                    {
                        "kv_connector": "AscendStoreConnector",
                        "kv_role": "'"$KV_ROLE"'",
                        "kv_connector_extra_config": {
                            "backend": "mooncake",
                            "lookup_rpc_port": '"$LOOKUP_RPC_PORT"'
                        }
                    }
                ]
            }
        }'
    ```

    **3. Second Decode node**

    Save as `run_decode_d2.sh` alongside `mooncake.json`.

    ```bash
    #!/usr/bin/env bash
    set -e

    # Configure these values for the current host.
    MODEL_PATH="/models/MiniMax-M3.1-Flash-W8A8"
    LOCAL_IP="<CURRENT_NODE_IP>"
    NIC_NAME="<CURRENT_NODE_NIC>"
    DECODE_HEAD_IP="<DECODE1_NODE_IP>"
    LOCAL_RANK=${1:-0}
    DP_RANK=$((LOCAL_RANK + 4))
    SERVICE_PORT=$((19300 + LOCAL_RANK))
    DP_RPC_PORT=16920
    KV_PORT=$((26000 + LOCAL_RANK*32))
    MAX_NUM_SEQS=128
    MAX_MODEL_LEN=133000
    MAX_BATCHED_TOKENS=2048
    GPU_MEMORY_UTILIZATION=0.92

    # Fixed runtime configuration.
    export VLLM_USE_V2_MODEL_RUNNER=1
    export HCCL_BUFFSIZE=1024
    export HCCL_IF_IP="$LOCAL_IP"
    unset HCCL_INTRA_ROCE_ENABLE
    export HCCL_OP_EXPANSION_MODE=AIV
    export HCCL_SOCKET_IFNAME="$NIC_NAME"
    export ASCEND_RT_VISIBLE_DEVICES="$((LOCAL_RANK*4)),$((LOCAL_RANK*4+1)),$((LOCAL_RANK*4+2)),$((LOCAL_RANK*4+3))"
    export GLOO_SOCKET_IFNAME="$NIC_NAME"
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export ASCEND_LOCAL_COMM_RES='{"version":"1.2"}'

    # Optional: only if Mooncake is installed in a non-standard path.
    if [ -n "${MOONCAKE_LIB_DIRS:-}" ]; then
        export LD_LIBRARY_PATH="${MOONCAKE_LIB_DIRS}:${LD_LIBRARY_PATH:-}"
    fi
    export MOONCAKE_CONFIG_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/mooncake.json"
    unset MOONCAKE_MASTER MOONCAKE_GLOBAL_SEGMENT_SIZE
    export ASCEND_ENABLE_USE_FABRIC_MEM=1
    export PYTHONHASHSEED=0
    # SSD I/O buffer per worker: 1GB, aligned for A3 fabric memory.
    export MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES=1073741824
    # SSD disk budget per NPU worker rank: 10 GiB.
    export MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES=$((10 * 1024 * 1024 * 1024))
    export MOONCAKE_OFFLOAD_BUCKET_MAX_TOTAL_SIZE=$((10 * 1024 * 1024 * 1024))
    export MOONCAKE_OFFLOAD_BUCKET_EVICTION_POLICY=lru
    KV_ROLE=kv_consumer
    LOOKUP_RPC_PORT=$((37100 + DP_RANK))

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
        --disable-hybrid-kv-cache-manager \
        --speculative-config '{"method":"dspark","num_speculative_tokens":7,"enforce_eager":true,"kv_cache_dtype":"bfloat16"}' \
        --reasoning-parser minimax_m3 \
        --data-parallel-size 8 \
        --data-parallel-size-local 1 \
        --data-parallel-rank "$DP_RANK" \
        --data-parallel-address "$DECODE_HEAD_IP" \
        --data-parallel-rpc-port "$DP_RPC_PORT" \
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","max_cudagraph_capture_size":512}' \
        --additional-config '{
            "enable_cpu_binding":true,
            "multistream_overlap_shared_expert":true,
            "enable_shared_expert_dp":true,
            "enable_flashcomm1":true,
            "weight_nz_mode":1
        }' \
        --kv-transfer-config '{
            "kv_connector": "MultiConnector",
            "kv_role": "'"$KV_ROLE"'",
            "kv_connector_extra_config": {
                "connectors": [
                    {
                        "kv_connector": "MooncakeConnectorV1",
                        "kv_buffer_device": "npu",
                        "kv_role": "'"$KV_ROLE"'",
                        "kv_port": '"$KV_PORT"',
                        "kv_connector_extra_config": {
                            "use_ascend_direct": true,
                            "prefill": {"dp_size": 1, "tp_size": 4, "pp_size": 4, "pp_layer_partition": "15,15,15,15"},
                            "decode": {"dp_size": 8, "tp_size": 4, "pp_size": 1}
                        }
                    },
                    {
                        "kv_connector": "AscendStoreConnector",
                        "kv_role": "'"$KV_ROLE"'",
                        "kv_connector_extra_config": {
                            "backend": "mooncake",
                            "lookup_rpc_port": '"$LOOKUP_RPC_PORT"'
                        }
                    }
                ]
            }
        }'
    ```

=== "950DT products"

    **1. Prefill node**

    Save as `run_prefill.sh` alongside `mooncake.json`.

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
    MAX_BATCHED_TOKENS=8000
    GPU_MEMORY_UTILIZATION=0.85

    # Fixed runtime configuration.
    export VLLM_USE_FASTOKENS=1
    export VLLM_USE_V2_MODEL_RUNNER=1
    export VLLM_ENGINE_READY_TIMEOUT_S=1800
    export HCCL_BUFFSIZE=512
    export HCCL_IF_IP="$LOCAL_IP"
    export HCCL_OP_EXPANSION_MODE="CCU_SCHED"
    export HCCL_SOCKET_IFNAME="$NIC_NAME"
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export GLOO_SOCKET_IFNAME="$NIC_NAME"
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

    # Optional: only if Mooncake is installed in a non-standard path.
    if [ -n "${MOONCAKE_LIB_DIRS:-}" ]; then
        export LD_LIBRARY_PATH="${MOONCAKE_LIB_DIRS}:${LD_LIBRARY_PATH:-}"
    fi
    export MOONCAKE_CONFIG_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/mooncake.json"
    unset MOONCAKE_MASTER MOONCAKE_GLOBAL_SEGMENT_SIZE
    export PYTHONHASHSEED=0
    # SSD I/O buffer per worker: 1GB.
    export MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES=1073741824
    # SSD disk budget per NPU worker rank: 10 GiB.
    export MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES=$((10 * 1024 * 1024 * 1024))
    export MOONCAKE_OFFLOAD_BUCKET_MAX_TOTAL_SIZE=$((10 * 1024 * 1024 * 1024))
    export MOONCAKE_OFFLOAD_BUCKET_EVICTION_POLICY=lru
    export ASCEND_LOCAL_COMM_RES_PATH=/etc/hixlep/
    export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'
    unset ASCEND_GLOBAL_RESOURCE_CONFIG
    KV_ROLE=kv_producer
    LOOKUP_RPC_PORT=37000

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
        --additional-config '{
            "enable_cpu_binding": true,
            "multistream_overlap_shared_expert": true,
            "enable_shared_expert_dp": true,
            "enable_flashcomm1": true,
            "enable_fused_mc2": 0
        }' \
        --speculative-config '{
            "method":"dspark",
            "num_speculative_tokens":7,
            "enforce_eager":true,
            "kv_cache_dtype":"bfloat16"
        }' \
        --kv-transfer-config '{
            "kv_connector": "MultiConnector",
            "kv_role": "'"$KV_ROLE"'",
            "kv_connector_extra_config": {
                "connectors": [
                    {
                        "kv_connector": "MooncakeConnectorV1",
                        "kv_buffer_device": "npu",
                        "kv_role": "'"$KV_ROLE"'",
                        "kv_port": '"$KV_PORT"',
                        "kv_connector_extra_config": {
                            "use_ascend_direct": true,
                            "ascend_local_comm_res_path": "/etc/hixlep",
                            "prefill": {"dp_size": 1, "tp_size": 4, "pp_size": 2},
                            "decode": {"dp_size": 4, "tp_size": 4, "pp_size": 1}
                        }
                    },
                    {
                        "kv_connector": "AscendStoreConnector",
                        "kv_role": "'"$KV_ROLE"'",
                        "kv_connector_extra_config": {
                            "backend": "mooncake",
                            "lookup_rpc_port": '"$LOOKUP_RPC_PORT"'
                        }
                    }
                ]
            }
        }'
    ```

    **2. First Decode node (D1)**

    Save as `run_decode_d1.sh` alongside `mooncake.json`. This is the pooled replacement of the Section 5.3 950DT `run_decode_head.sh`. D1 is the Data Parallel master, serves local DP ranks 0–1, and exposes API port `31060`. D2 must use the same `DECODE_HEAD_IP` and `DP_RPC_PORT`.

    ```bash
    #!/usr/bin/env bash
    set -e

    # Configure these values for the current host.
    MODEL_PATH="/models/MiniMax-M3.1-Flash-W8A8-MXFP8"
    LOCAL_IP="<CURRENT_NODE_IP>"
    NIC_NAME="<CURRENT_NODE_NIC>"
    # D1 is the Data Parallel master; D2 must use the same value.
    DECODE_HEAD_IP="$LOCAL_IP"
    SERVICE_PORT=31060
    DP_RPC_PORT=6885
    KV_PORT=26900
    MAX_NUM_SEQS=128
    MAX_MODEL_LEN=133000
    MAX_BATCHED_TOKENS=8000
    GPU_MEMORY_UTILIZATION=0.92

    # Fixed runtime configuration.
    export VLLM_USE_V2_MODEL_RUNNER=1
    export VLLM_ENGINE_READY_TIMEOUT_S=1800
    export HCCL_BUFFSIZE=1024
    export HCCL_IF_IP="$LOCAL_IP"
    export HCCL_OP_EXPANSION_MODE="CCU_SCHED"
    export HCCL_SOCKET_IFNAME="$NIC_NAME"
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export GLOO_SOCKET_IFNAME="$NIC_NAME"
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

    # Optional: only if Mooncake is installed in a non-standard path.
    if [ -n "${MOONCAKE_LIB_DIRS:-}" ]; then
        export LD_LIBRARY_PATH="${MOONCAKE_LIB_DIRS}:${LD_LIBRARY_PATH:-}"
    fi
    export MOONCAKE_CONFIG_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/mooncake.json"
    unset MOONCAKE_MASTER MOONCAKE_GLOBAL_SEGMENT_SIZE
    export PYTHONHASHSEED=0
    # SSD I/O buffer per worker: 1GB.
    export MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES=1073741824
    # SSD disk budget per NPU worker rank: 10 GiB.
    export MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES=$((10 * 1024 * 1024 * 1024))
    export MOONCAKE_OFFLOAD_BUCKET_MAX_TOTAL_SIZE=$((10 * 1024 * 1024 * 1024))
    export MOONCAKE_OFFLOAD_BUCKET_EVICTION_POLICY=lru
    export ASCEND_LOCAL_COMM_RES_PATH=/etc/hixlep/
    export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'
    unset ASCEND_GLOBAL_RESOURCE_CONFIG
    KV_ROLE=kv_consumer
    # One base port per host; D1 and D2 must differ.
    LOOKUP_RPC_PORT=37100

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
        --speculative-config '{
            "method":"dspark",
            "num_speculative_tokens":7,
            "enforce_eager":true,
            "kv_cache_dtype":"bfloat16"
        }' \
        --kv-transfer-config '{
            "kv_connector": "MultiConnector",
            "kv_role": "'"$KV_ROLE"'",
            "kv_connector_extra_config": {
                "connectors": [
                    {
                        "kv_connector": "MooncakeConnectorV1",
                        "kv_buffer_device": "npu",
                        "kv_role": "'"$KV_ROLE"'",
                        "kv_port": '"$KV_PORT"',
                        "kv_connector_extra_config": {
                            "use_ascend_direct": true,
                            "ascend_local_comm_res_path": "/etc/hixlep",
                            "prefill": {"dp_size": 1, "tp_size": 4, "pp_size": 2},
                            "decode": {"dp_size": 4, "tp_size": 4, "pp_size": 1}
                        }
                    },
                    {
                        "kv_connector": "AscendStoreConnector",
                        "kv_role": "'"$KV_ROLE"'",
                        "kv_connector_extra_config": {
                            "backend": "mooncake",
                            "lookup_rpc_port": '"$LOOKUP_RPC_PORT"'
                        }
                    }
                ]
            }
        }'
    ```

    **3. Second Decode node (D2)**

    Save as `run_decode_d2.sh` alongside `mooncake.json`. This is the pooled replacement of the Section 5.3 950DT `run_decode_worker.sh`. Set `LOCAL_IP` and `NIC_NAME` for this host, keep `DECODE_HEAD_IP` equal to D1's IP, and keep every other value identical to D1 except `--data-parallel-start-rank` (`2` here) and `LOOKUP_RPC_PORT` (`37102` here).

    ```bash
    #!/usr/bin/env bash
    set -e

    # Configure these values for the current host.
    MODEL_PATH="/models/MiniMax-M3.1-Flash-W8A8-MXFP8"
    LOCAL_IP="<CURRENT_NODE_IP>"
    NIC_NAME="<CURRENT_NODE_NIC>"
    DECODE_HEAD_IP="<DECODE1_NODE_IP>"
    SERVICE_PORT=31060
    DP_RPC_PORT=6885
    KV_PORT=26900
    MAX_NUM_SEQS=128
    MAX_MODEL_LEN=133000
    MAX_BATCHED_TOKENS=8000
    GPU_MEMORY_UTILIZATION=0.92

    # Fixed runtime configuration.
    export VLLM_USE_V2_MODEL_RUNNER=1
    export VLLM_ENGINE_READY_TIMEOUT_S=1800
    export HCCL_BUFFSIZE=1024
    export HCCL_IF_IP="$LOCAL_IP"
    export HCCL_OP_EXPANSION_MODE="CCU_SCHED"
    export HCCL_SOCKET_IFNAME="$NIC_NAME"
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export GLOO_SOCKET_IFNAME="$NIC_NAME"
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

    # Optional: only if Mooncake is installed in a non-standard path.
    if [ -n "${MOONCAKE_LIB_DIRS:-}" ]; then
        export LD_LIBRARY_PATH="${MOONCAKE_LIB_DIRS}:${LD_LIBRARY_PATH:-}"
    fi
    export MOONCAKE_CONFIG_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/mooncake.json"
    unset MOONCAKE_MASTER MOONCAKE_GLOBAL_SEGMENT_SIZE
    export PYTHONHASHSEED=0
    # SSD I/O buffer per worker: 1GB.
    export MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES=1073741824
    # SSD disk budget per NPU worker rank: 10 GiB.
    export MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES=$((10 * 1024 * 1024 * 1024))
    export MOONCAKE_OFFLOAD_BUCKET_MAX_TOTAL_SIZE=$((10 * 1024 * 1024 * 1024))
    export MOONCAKE_OFFLOAD_BUCKET_EVICTION_POLICY=lru
    export ASCEND_LOCAL_COMM_RES_PATH=/etc/hixlep/
    export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'
    unset ASCEND_GLOBAL_RESOURCE_CONFIG
    KV_ROLE=kv_consumer
    # One base port per host; D1 and D2 must differ.
    LOOKUP_RPC_PORT=37102

    vllm serve "$MODEL_PATH" \
        --host 0.0.0.0 \
        --port "$SERVICE_PORT" \
        --data-parallel-size 4 \
        --data-parallel-size-local 2 \
        --data-parallel-start-rank 2 \
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
        --speculative-config '{
            "method":"dspark",
            "num_speculative_tokens":7,
            "enforce_eager":true,
            "kv_cache_dtype":"bfloat16"
        }' \
        --kv-transfer-config '{
            "kv_connector": "MultiConnector",
            "kv_role": "'"$KV_ROLE"'",
            "kv_connector_extra_config": {
                "connectors": [
                    {
                        "kv_connector": "MooncakeConnectorV1",
                        "kv_buffer_device": "npu",
                        "kv_role": "'"$KV_ROLE"'",
                        "kv_port": '"$KV_PORT"',
                        "kv_connector_extra_config": {
                            "use_ascend_direct": true,
                            "ascend_local_comm_res_path": "/etc/hixlep",
                            "prefill": {"dp_size": 1, "tp_size": 4, "pp_size": 2},
                            "decode": {"dp_size": 4, "tp_size": 4, "pp_size": 1}
                        }
                    },
                    {
                        "kv_connector": "AscendStoreConnector",
                        "kv_role": "'"$KV_ROLE"'",
                        "kv_connector_extra_config": {
                            "backend": "mooncake",
                            "lookup_rpc_port": '"$LOOKUP_RPC_PORT"'
                        }
                    }
                ]
            }
        }'
    ```

#### 5.4.4 Start the Services

Start in this order:

```text
Mooncake Master
    ↓
First Decode node → Second Decode node
    ↓
Prefill
    ↓
Proxy (:19080)
```

1. Start `mooncake_master` on the Prefill node or another reachable host, and use its actual IP in every `mooncake.json`. Confirm port `50088` is reachable from all model nodes:

    ```bash
    mooncake_master \
    --port 50088 \
    --eviction_high_watermark_ratio 0.9 \
    --eviction_ratio 0.1 \
    --default_kv_lease_ttl 11000 \
    --enable_offload=true \
    --client_ttl=120
    ```

    These are example Master settings: `default_kv_lease_ttl=11000` is in milliseconds, while `client_ttl=120` is in seconds. The eviction ratios are dimensionless; see the [Mooncake Master parameter table](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md#step-2-configure-mooncakejson-and-start-mooncake_master) for their meanings and supported ranges. Keep `enable_offload=true` for this SSD-offload deployment.

    Before starting the model services, run this TCP connectivity check from each model container, replacing the Master IP. Continue only after it succeeds and the Master logs show successful startup. If it fails, check the Master process, bind address, and network access, then retry:

    ```bash
    python3 - <<'PYCODE'
    import socket

    with socket.create_connection(("<MOONCAKE_MASTER_IP>", 50088), timeout=5):
        print("Mooncake Master TCP endpoint is reachable")
    PYCODE
    ```

    This checks the TCP listener; successful worker registration and KV operations are verified in Section 5.4.5.

2. Start the first A3 Decode node (global DP ranks 0–3). Use the pooled `run_decode_d1.sh`, not the normal-PD script:

    ```bash
    for rank in 0 1 2 3; do
        nohup bash run_decode_d1.sh "$rank" > "decode-$rank.log" 2>&1 &
    done
    ```

    Wait for rank 0 to print `Started DP Coordinator process`. Start the second node from another shell; initialization needs both D nodes.

    On 950DT products, start one pooled Decode process on each Decode host instead of one process per rank. D1 serves local DP ranks 0–1 and D2 serves local DP ranks 2–3; start both before waiting, because the distributed group needs both hosts:

    ```bash
    # On the first Decode host (D1).
    nohup bash run_decode_d1.sh > decode.log 2>&1 &
    # On the second Decode host (D2).
    nohup bash run_decode_d2.sh > decode.log 2>&1 &
    ```

    Both hosts expose API port `31060`. Then skip step 3 on 950DT products.

3. Start the second A3 Decode node (global DP ranks 4–7; A3 only). `LOCAL_IP` is the second node's IP, but `DECODE_HEAD_IP` remains the first Decode node's IP:

    ```bash
    for rank in 0 1 2 3; do
        nohup bash run_decode_d2.sh "$rank" > "decode-$((rank+4)).log" 2>&1 &
    done
    ```

    D1 exposes API ports `19200`–`19203`; D2 exposes `19300`–`19303`. Wait for all eight DP ranks to print `Application startup complete`.

4. Start Prefill using the pooled `run_prefill.sh`:

    ```bash
    nohup bash run_prefill.sh > prefill.log 2>&1 &
    ```

    Wait for Prefill to print `Application startup complete` on port `19100`.

    On 950DT products, start the pooled Prefill with `nohup bash run_prefill.sh > prefill.log 2>&1 &` and wait for `Application startup complete` on port `31050`.

5. Start the proxy. The A3 proxy registers one Prefill endpoint and eight Decode endpoints. Use the unchanged `proxy_server.sh` from Section 5.3:

    ```bash
    bash proxy_server.sh
    ```

    The service is accessible at `<proxy_ip>:19080`; use this endpoint with model name `minimax-m3.1-flash` in [Section 6](#6-functional-verification).

    On 950DT products, use the unchanged Section 5.3 950DT `proxy_server.sh`: it registers one Prefill endpoint on `31050` and two Decode endpoints on `31060`, and the pooled scripts keep those ports.

#### 5.4.5 Verification

1. Confirm Mooncake Master port `50088` is reachable, Master runs with `--enable_offload=true`, and every node uses the same Master address and `tenant_id`. Confirm the Prefill and Decode nodes load the `global_segment_size` from their `mooncake.json` (`4GB` on A3, `64GB` on 950DT products). Check worker logs for successful segment registration.
2. Confirm every Decode engine port is ready, then every Prefill engine port. A3 has nine APIs (one P plus eight D); 950DT products have three (one Prefill on `31050` and two Decode on `31060`).
3. Send requests only to the proxy on port `19080`.
4. Warm up with a repeated-prefix request, then send again and check Prefill logs for KV Pool lookup/get/put and hit information.
5. Confirm Decode logs still show a normal Mooncake P→D KV transfer.
6. Confirm SSD storage initializes under `ssd_offload_path` for each contributing worker. Use a working set larger than the memory pool to verify SSD offload and subsequent reads; a memory-pool hit alone does not exercise the SSD tier.

### 5.5 1M-Token Context Deployment

This text-only W8A8 profile uses a 1P1D topology: one Prefill service on one host and one distributed Decode service spanning two hosts. The two Decode hosts share one global DP4 × TP8 × PP1 group. All three hosts are Atlas 800I A3 machines, each with 16 visible 64 GB NPU devices. Here, 1M means **1,048,576 input tokens**. The total context limit is `1050624`, reserving 2048 positions for generation; the verification request below caps generation at 256 tokens. Use the model image and W8A8 checkpoint from Sections 3 and 4 on every node. The validated runtime uses vLLM 0.32.0, torch_npu 2.10.0.post7, and CANN 9.2.0, with the M3.1 operators and selected-parameter prefetch options shown below available in the image.

| Host | Global DP / TP / PP | Local DP ranks | API ports | Batched-token budget |
| --- | --- | --- | --- | --- |
| Prefill host | 1 / 8 / 2 | 0 | 31050 | 8192 |
| Decode host 1 | 4 / 8 / 1 | 0–1 | 31260, 31261 | 1 |
| Decode host 2 | 4 / 8 / 1 | 2–3 | 31260, 31261 | 1 |

This is a functional long-context configuration with one sequence per Decode replica. Both roles use eager execution, Model Runner V2, BF16 KV caches, and chunked prefill. Speculative decoding, prefix caching, asynchronous scheduling, and static kernels are disabled. The batched-token budget limits each iteration; it does not limit the total input length.

Decode reserves `48421142528` bytes (about 45.1 GiB) of KV cache per worker. To retain workspace for the attention operators, it prefetches router and attention-projection weights from CPU using `gate.weight`, `qkv_proj.weight`, and `o_proj.weight`. Other weights remain on NPU. Decode uses `weight_nz_mode=0` so the selected projection weights use ordinary layouts for prefetching. Allow for these CPU-to-NPU transfers when evaluating latency; tune performance separately.

#### 5.5.1 Preparation

Complete the communication checks in [Section 3.2](#32-verify-multi-node-communication), and prepare the containers, W8A8 weights, and Mooncake dependencies from Sections 4 and 5.3. Replace the IP/NIC placeholders with each host's communication-network IP and matching interface. Adjust the CANN environment-script path if needed. The templates set `VLLM_HOST_IP` as well as the HCCL/GLOO interfaces so KV-transfer endpoints advertise the intended network.

Use the same `launch_online_dp.py` as [MiniMax-M3 Section 5.3.1](MiniMax-M3.md#531-prefill-decode-disaggregation): copy that section's complete Python launcher, including its `--pp-size` option. Create separate role directories inside the containers:

```bash
mkdir -p /workspace/m31-1m/prefill /workspace/m31-1m/decode
```

Save a copy of `launch_online_dp.py` in each role directory that will be used. Save the Prefill template below as `/workspace/m31-1m/prefill/run_dp_template.sh` on P, and the Decode template as `/workspace/m31-1m/decode/run_dp_template.sh` on both D hosts. Keep P and D directories separate even on shared storage: the launcher reads `./run_dp_template.sh` from its working directory.

The templates use named environment variables. Set host-specific values in the launch blocks in Section 5.5.3; the launcher supplies each engine's `DP_RANK`, `SERVICE_PORT`, and `VISIBLE_DEVICES`. Each engine occupies `TP_SIZE × PP_SIZE` NPUs. Each Decode engine serves one DP rank with `--data-parallel-size-local 1`; the launcher's `--dp-size-local 2` starts two engines per D host. Both D hosts use the first D host's IP as `DP_ADDRESS` and share one global DP4 group.

#### 5.5.2 Service Scripts

=== "Atlas 800I A3 (W8A8)"

    **1. Prefill node: `prefill/run_dp_template.sh`**

    ```bash
    #!/usr/bin/env bash
    set -eo pipefail

    unset http_proxy https_proxy ftp_proxy all_proxy HTTP_PROXY HTTPS_PROXY FTP_PROXY ALL_PROXY
    source /usr/local/Ascend/ascend-toolkit/set_env.sh

    # Set this host's communication IP, NIC and W8A8 weight path.
    export LOCAL_IP="${LOCAL_IP:-<CURRENT_NODE_IP>}"
    export NIC_NAME="${NIC_NAME:-<CURRENT_NODE_NIC>}"
    export MODEL_PATH="${MODEL_PATH:-/models/MiniMax-M3.1-Flash-W8A8}"

    # The launcher supplies these values for the Prefill engine.
    export DP_SIZE=${DP_SIZE:-1}
    export TP_SIZE=${TP_SIZE:-8}
    export PP_SIZE=${PP_SIZE:-2}
    export SERVICE_PORT=${SERVICE_PORT:-31050}
    export VISIBLE_DEVICES="${VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"

    export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200
    export VLLM_PP_LAYER_PARTITION="30,30"
    export VLLM_USE_V2_MODEL_RUNNER=1
    export HCCL_BUFFSIZE=256
    export HCCL_OP_EXPANSION_MODE=AIV
    export HCCL_IF_IP="$LOCAL_IP"
    export VLLM_HOST_IP="$LOCAL_IP"
    export HCCL_CONNECT_TIMEOUT=120
    export HCCL_EXEC_TIMEOUT=1800
    export HCCL_SOCKET_IFNAME="$NIC_NAME"
    export ASCEND_ENABLE_USE_FABRIC_MEM=1
    export ASCEND_RT_VISIBLE_DEVICES="$VISIBLE_DEVICES"
    if [ -n "${MOONCAKE_LIB_DIRS:-}" ]; then
        export LD_LIBRARY_PATH="${MOONCAKE_LIB_DIRS}:${LD_LIBRARY_PATH:-}"
    fi
    export GLOO_SOCKET_IFNAME="$NIC_NAME"
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

    vllm serve "$MODEL_PATH" \
        --enforce-eager \
        --compilation-config '{"cudagraph_mode":"NONE"}' \
        --host 0.0.0.0 \
        --port "$SERVICE_PORT" \
        --data-parallel-size "$DP_SIZE" \
        --tensor-parallel-size "$TP_SIZE" \
        --pipeline-parallel-size "$PP_SIZE" \
        --distributed-executor-backend mp \
        --served-model-name minimax-m3.1-flash \
        --enable-expert-parallel \
        --seed 1024 \
        --max-model-len 1050624 \
        --max-num-seqs 1 \
        --max-num-batched-tokens 8192 \
        --long-prefill-token-threshold 8192 \
        --enable-chunked-prefill \
        --no-enable-prefix-caching \
        --no-async-scheduling \
        --trust-remote-code \
        --quantization ascend \
        --dtype bfloat16 \
        --block-size 128 \
        --hf-overrides '{"architectures":["MiniMaxM3SparseForCausalLM"]}' \
        --gpu-memory-utilization 0.95 \
        --kv-cache-dtype bfloat16 \
        --language-model-only \
        --additional-config '{
        "enable_cpu_binding": true,
        "ascend_compilation_config": {
            "enable_static_kernel": false,
            "fuse_norm_quant": false,
            "fuse_qknorm_rope": false
        },
        "multistream_overlap_shared_expert": true,
        "weight_nz_mode": 1,
        "enable_shared_expert_dp": false,
        "enable_flashcomm1": true
    }' \
        --kv-transfer-config '{
        "kv_connector": "MooncakeConnectorV1",
        "kv_buffer_device": "npu",
        "kv_role": "kv_producer",
        "kv_port": 20000,
        "engine_id": "m31-1m-prefill-0",
        "kv_connector_extra_config": {
            "use_ascend_direct": true,
            "prefill": {
                "dp_size": 1,
                "tp_size": 8,
                "pp_size": 2,
                "pp_layer_partition": "30,30"
            },
            "decode": {
                "dp_size": 4,
                "tp_size": 8,
                "pp_size": 1
            }
        }
    }'
    ```

    **2. Decode nodes: `decode/run_dp_template.sh`**

    ```bash
    #!/usr/bin/env bash
    set -eo pipefail

    unset http_proxy https_proxy ftp_proxy all_proxy HTTP_PROXY HTTPS_PROXY FTP_PROXY ALL_PROXY
    source /usr/local/Ascend/ascend-toolkit/set_env.sh

    # Set this host's communication IP, NIC and W8A8 weight path.
    export LOCAL_IP="${LOCAL_IP:-<CURRENT_NODE_IP>}"
    export NIC_NAME="${NIC_NAME:-<CURRENT_NODE_NIC>}"
    export MODEL_PATH="${MODEL_PATH:-/models/MiniMax-M3.1-Flash-W8A8}"

    # Both D hosts use the first D host's IP as the global DP coordinator.
    export DP_SIZE=${DP_SIZE:-4}
    export TP_SIZE=${TP_SIZE:-8}
    export PP_SIZE=${PP_SIZE:-1}
    export DP_RANK=${DP_RANK:-0}
    export DP_ADDRESS="${DP_ADDRESS:-<FIRST_DECODE_NODE_IP>}"
    export DP_RPC_PORT=${DP_RPC_PORT:-31422}
    export SERVICE_PORT=${SERVICE_PORT:-31260}
    export VISIBLE_DEVICES="${VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

    unset VLLM_PP_LAYER_PARTITION

    export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200
    export VLLM_USE_V2_MODEL_RUNNER=1
    export HCCL_BUFFSIZE=8
    export HCCL_IF_IP="$LOCAL_IP"
    export VLLM_HOST_IP="$LOCAL_IP"
    export HCCL_CONNECT_TIMEOUT=120
    export HCCL_EXEC_TIMEOUT=1800
    export HCCL_OP_EXPANSION_MODE="AIV"
    export HCCL_SOCKET_IFNAME="$NIC_NAME"
    export ASCEND_ENABLE_USE_FABRIC_MEM=1
    export ASCEND_RT_VISIBLE_DEVICES="$VISIBLE_DEVICES"
    if [ -n "${MOONCAKE_LIB_DIRS:-}" ]; then
        export LD_LIBRARY_PATH="${MOONCAKE_LIB_DIRS}:${LD_LIBRARY_PATH:-}"
    fi
    export GLOO_SOCKET_IFNAME="$NIC_NAME"
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

    KV_PORT=$((21000 + DP_RANK * 64))

    vllm serve "$MODEL_PATH" \
        --data-parallel-rank "$DP_RANK" \
        --data-parallel-size-local 1 \
        --data-parallel-address "$DP_ADDRESS" \
        --data-parallel-rpc-port "$DP_RPC_PORT" \
        --enforce-eager \
        --compilation-config '{"cudagraph_mode":"NONE"}' \
        --host 0.0.0.0 \
        --port "$SERVICE_PORT" \
        --data-parallel-size "$DP_SIZE" \
        --tensor-parallel-size "$TP_SIZE" \
        --pipeline-parallel-size "$PP_SIZE" \
        --distributed-executor-backend mp \
        --served-model-name minimax-m3.1-flash \
        --enable-expert-parallel \
        --seed 1024 \
        --max-model-len 1050624 \
        --max-num-seqs 1 \
        --max-num-batched-tokens 1 \
        --long-prefill-token-threshold 1 \
        --enable-chunked-prefill \
        --no-enable-prefix-caching \
        --no-async-scheduling \
        --trust-remote-code \
        --quantization ascend \
        --dtype bfloat16 \
        --block-size 128 \
        --hf-overrides '{"architectures":["MiniMaxM3SparseForCausalLM"]}' \
        --gpu-memory-utilization 0.99 \
        --kv-cache-memory-bytes 48421142528 \
        --offload-backend prefetch \
        --offload-group-size 1 \
        --offload-num-in-group 1 \
        --offload-prefetch-step 1 \
        --offload-params gate.weight qkv_proj.weight o_proj.weight \
        --kv-cache-dtype bfloat16 \
        --language-model-only \
        --additional-config '{
        "enable_cpu_binding": true,
        "ascend_compilation_config": {
            "enable_static_kernel": false,
            "fuse_norm_quant": false,
            "fuse_qknorm_rope": false
        },
        "multistream_overlap_shared_expert": false,
        "weight_nz_mode": 0,
        "enable_shared_expert_dp": false,
        "enable_flashcomm1": false
    }' \
        --kv-transfer-config '{
        "kv_connector": "MooncakeConnectorV1",
        "kv_buffer_device": "npu",
        "kv_role": "kv_consumer",
        "kv_port": '"$KV_PORT"',
        "engine_id": "m31-1m-decode-'"$DP_RANK"'",
        "kv_connector_extra_config": {
            "use_ascend_direct": true,
            "prefill": {
                "dp_size": 1,
                "tp_size": 8,
                "pp_size": 2,
                "pp_layer_partition": "30,30"
            },
            "decode": {
                "dp_size": 4,
                "tp_size": 8,
                "pp_size": 1
            }
        }
    }'
    ```

Keep the tested Decode settings on both hosts: `HCCL_BUFFSIZE=8`, token budget and long-prefill threshold `1`, and `48421142528` bytes of KV cache per worker. The KV handshake port is `21000 + DP_RANK × 64`, giving `21000`, `21064`, `21128`, and `21192`; Prefill uses `20000`. These ports are separate from the HTTP and DP RPC ports. If changing TP/PP/DP, update both roles' KV-transfer layouts and all launch mappings together.

#### 5.5.3 Start the Services

**1. Prefill node**

```bash
cd /workspace/m31-1m/prefill

export LOCAL_IP="<PREFILL_NODE_IP>"
export NIC_NAME="<PREFILL_NODE_NIC>"
export MODEL_PATH="/models/MiniMax-M3.1-Flash-W8A8"
export DP_SIZE=1
export TP_SIZE=8
export PP_SIZE=2
export LOCAL_DP_SIZE=1
export DP_RANK_START=0
export DP_ADDRESS="$LOCAL_IP"
export DP_RPC_PORT=6884
export SERVICE_PORT=31050

python launch_online_dp.py \
    --dp-size "$DP_SIZE" --tp-size "$TP_SIZE" --pp-size "$PP_SIZE" \
    --dp-size-local "$LOCAL_DP_SIZE" --dp-rank-start "$DP_RANK_START" \
    --dp-address "$DP_ADDRESS" --dp-rpc-port "$DP_RPC_PORT" \
    --vllm-start-port "$SERVICE_PORT"
```

**2. First Decode node: global DP ranks 0–1**

```bash
cd /workspace/m31-1m/decode

export LOCAL_IP="<FIRST_DECODE_NODE_IP>"
export NIC_NAME="<FIRST_DECODE_NODE_NIC>"
export MODEL_PATH="/models/MiniMax-M3.1-Flash-W8A8"
export DP_SIZE=4
export TP_SIZE=8
export PP_SIZE=1
export LOCAL_DP_SIZE=2
export DP_RANK_START=0
export DP_ADDRESS="$LOCAL_IP"
export DP_RPC_PORT=31422
export SERVICE_PORT=31260

python launch_online_dp.py \
    --dp-size "$DP_SIZE" --tp-size "$TP_SIZE" --pp-size "$PP_SIZE" \
    --dp-size-local "$LOCAL_DP_SIZE" --dp-rank-start "$DP_RANK_START" \
    --dp-address "$DP_ADDRESS" --dp-rpc-port "$DP_RPC_PORT" \
    --vllm-start-port "$SERVICE_PORT"
```

This starts the DP coordinator and the first two engines. Keep this command running and start the second D host from another shell; do not wait for the first D host to become fully healthy before starting the second.

**3. Second Decode node: global DP ranks 2–3**

```bash
cd /workspace/m31-1m/decode

export LOCAL_IP="<SECOND_DECODE_NODE_IP>"
export NIC_NAME="<SECOND_DECODE_NODE_NIC>"
export MODEL_PATH="/models/MiniMax-M3.1-Flash-W8A8"
export DP_SIZE=4
export TP_SIZE=8
export PP_SIZE=1
export LOCAL_DP_SIZE=2
export DP_RANK_START=2
export DP_ADDRESS="<FIRST_DECODE_NODE_IP>"
export DP_RPC_PORT=31422
export SERVICE_PORT=31260

python launch_online_dp.py \
    --dp-size "$DP_SIZE" --tp-size "$TP_SIZE" --pp-size "$PP_SIZE" \
    --dp-size-local "$LOCAL_DP_SIZE" --dp-rank-start "$DP_RANK_START" \
    --dp-address "$DP_ADDRESS" --dp-rpc-port "$DP_RPC_PORT" \
    --vllm-start-port "$SERVICE_PORT"
```

The launcher's device assignment is:

| Host / DP rank | Local logical NPUs | API port |
| --- | --- | --- |
| Prefill / 0 | 0–15 | 31050 |
| First D / 0 | 0–7 | 31260 |
| First D / 1 | 8–15 | 31261 |
| Second D / 2 | 0–7 | 31260 |
| Second D / 3 | 8–15 | 31261 |

Wait for all five APIs to report `Application startup complete` and return HTTP 200 from `/health`. Run the two D launch commands concurrently until their shared initialization completes.

Check that each engine reports enough KV-cache tokens for `1050624` total tokens and that Decode logs confirm weight prefetching. Cache capacity alone does not establish functional readiness: retain operator workspace and complete the long-input test below when using a different image, checkpoint, or memory budget.

**4. Proxy: one Prefill endpoint and four Decode endpoints**

Reuse the proxy prepared in Section 5.3. Set the three host IPs and start it inside the Prefill container:

```bash
PREFILL_IP="<PREFILL_NODE_IP>"
DECODE1_IP="<FIRST_DECODE_NODE_IP>"
DECODE2_IP="<SECOND_DECODE_NODE_IP>"
python3 /workspace/load_balance_proxy_server_example.py \
    --host 0.0.0.0 --port 19080 \
    --prefiller-hosts "$PREFILL_IP" --prefiller-ports 31050 \
    --decoder-hosts "$DECODE1_IP" "$DECODE1_IP" "$DECODE2_IP" "$DECODE2_IP" \
    --decoder-ports 31260 31261 31260 31261 \
    --workers 1
```

#### 5.5.4 Verify the 1M Input

Send requests to the proxy so that the test exercises KV transfer. First run the text request in Section 6.1 and check that the response is coherent. Then generate an exact 1M-token input with three distinct records near 5%, 50%, and 95% of the context. Run the following from `/workspace/m31-1m` in a container with the checkpoint tokenizer:

```bash
export MODEL_PATH="/models/MiniMax-M3.1-Flash-W8A8"
export SERVED_MODEL_NAME=minimax-m3.1-flash
python3 - <<'PYCODE'
import json
import os
import random
from pathlib import Path
from transformers import AutoTokenizer

n = 1048576
tokenizer = AutoTokenizer.from_pretrained(
    os.environ["MODEL_PATH"], trust_remote_code=True, local_files_only=True)
marker = "UNIQUE_LONG_ARCHIVE_PLACEHOLDER_741b"
rendered = tokenizer.apply_chat_template(
    [{"role": "user", "content": marker}], tokenize=False,
    add_generation_prompt=True, thinking_mode="disabled")
prefix, suffix = rendered.split(marker)
encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
values = {"alpha": "8f3a31a57f3ea87f", "beta": "5c4af4dc5228aa06",
          "gamma": "5345ade0b95e93dc"}
head = encode(prefix) + encode(
    "Read the archive. Retrieve the exact secret values for alpha, beta, gamma.\n")
tail = encode("\nReturn only a JSON object with keys alpha, beta, gamma.\n") + encode(suffix)
needles = [encode(f"\nUNIQUE SECRET RECORD: {key} = {value}\n")
           for key, value in values.items()]
rng = random.Random(1104)
pool = encode("".join(
    f"Ordinary archive record {i:06d}: storage code {rng.randrange(10**8):08d}; "
    "this record contains no secret value.\n" for i in range(4096)))
budget = n - len(head) - len(tail) - sum(map(len, needles))
body, previous = [], 0
for index, stop in enumerate([budget // 20, budget // 2, budget * 19 // 20, budget]):
    size = stop - previous
    body.extend((pool * (size // len(pool) + 1))[:size])
    if index < len(needles):
        body.extend(needles[index])
    previous = stop
ids = head + body + tail
assert len(ids) == n
Path("request-1m.json").write_text(json.dumps({
    "model": os.environ["SERVED_MODEL_NAME"], "prompt": ids,
    "max_tokens": 256, "temperature": 0, "stream": False,
    "add_special_tokens": False,
}))
Path("expected-1m.json").write_text(json.dumps(values))
print(f"Prepared {len(ids)} input tokens")
PYCODE

curl --noproxy '*' --fail-with-body --max-time 1800 \
    "http://<PROXY_IP>:19080/v1/completions" \
    -H 'Content-Type: application/json' \
    --data-binary @request-1m.json -o response-1m.json

python3 - <<'PYCODE'
import json
from pathlib import Path
response = json.loads(Path("response-1m.json").read_text())
expected = json.loads(Path("expected-1m.json").read_text())
text = response["choices"][0]["text"]
assert response["usage"]["prompt_tokens"] == 1048576
assert response["choices"][0]["finish_reason"] == "stop"
assert all(value in text for value in expected.values())
assert "\ufffd" not in text
print("PASS: 1,048,576 input tokens; all three records retrieved")
PYCODE
```

Also compare backend `/metrics` snapshots before and after the request: the Prefill `vllm:prompt_tokens_total` counter and the receiving Decode `vllm:external_prefix_cache_hits_total` counter should both increase by `1048576`. Repeat against a replica on the other Decode host when validating a new deployment.

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
| MMMU PRO | Ascend 950DT products | 80.35 | 133000 | 128 | 81920 | 32 | temperature=1, top_p=0.95, reasoning_effort="max" |
| MMMU PRO | A3 | 79.6 | 133000 | 128 | 81920 | 32 | temperature=1, top_p=0.95, reasoning_effort="max" |

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
        host_port=11223,
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

`host_ip` and `host_port` must point at the endpoint you deployed: `127.0.0.1` and `11223` for the single-node or A2 colocated service, and the proxy host and `19080` for the PD and KV-pool topologies in Sections 5.3 and 5.4.

**Run the performance evaluation using the GSM8K dataset as an example:**

```bash
ais_bench --models vllm_api_stream_chat --datasets gsm8k_gen_0_shot_cot_str_perf --debug --summarizer default_perf --mode perf --num-prompts 128
```

Expected result: the run prints a performance summary that includes request throughput (req/s), output-token throughput (token/s), mean and P99 TTFT, mean and P99 TPOT, and the number of completed and failed requests. A run is valid when the failed-request count is zero and the sample count matches `--num-prompts`.

## 9 Performance Tuning

### 9.1 Recommended Configurations

Use the commands in [Section 5](#5-online-service-deployment) as starting configurations. They are not globally optimal: the appropriate settings depend on request lengths, concurrency, prefix-cache hit rate, and the available device memory. Evaluate each adjustment with the same workload and compare latency, throughput, and memory use.

| Scenario | Starting Configuration | Parameters to Evaluate | Validation |
| --- | --- | --- | --- |
| Long context | All examples set `MAX_MODEL_LEN` to `133000`. | Set `MAX_MODEL_LEN` for the combined input/output budget and reduce `--max-num-seqs` if KV cache capacity is insufficient. | Confirm the requested context fits at the target concurrency before increasing load. |
| Low latency | Single-node examples in Section 5.1. | Reduce request concurrency and compare `--max-num-seqs` and `--max-num-batched-tokens` settings. | Compare TTFT and TPOT alongside output-token throughput. |
| High throughput | PD examples in Section 5.3. | Adjust `MAX_NUM_SEQS` and `MAX_BATCHED_TOKENS` on Prefill and Decode separately while keeping both connector topology declarations consistent. | Increase offered load gradually and check throughput, latency, and failed requests. |

### 9.2 Tuning Guidelines

#### 9.2.1 DSpark and KV Cache Configuration

The examples explicitly enable FlashComm1 and seven-token DSpark. On 950DT products, the PD main model uses FP8 KV cache while the DSpark draft uses BF16 KV cache. Preserve that distinction when tuning; the draft cache dtype is configured inside `--speculative-config`.

#### 9.2.2 Workload-Specific Tuning

Change one parameter at a time and retain the command and benchmark result for each run. For general methods, refer to [Optimization and Tuning](../../developer_guide/performance_and_debug/optimization_and_tuning.md). For feature descriptions and configuration constraints, refer to the [Feature Matrix](../../user_guide/support_matrix/feature_matrix.md) and [Speculative Decoding Guide](../../user_guide/feature_guide/speculative_decoding.md#speculating-using-dspark).

#### 9.2.3 Tool-Assisted Tuning with msmodeling

The msmodeling tool (MindStudio-Modeling) optimizes serving and benchmark parameters together, combining measured results with knowledge-base experience to determine a better parameter set in one run and push performance toward the hardware limit.

For the complete, step-by-step usage guide please refer to **[Optix-for-PD-disaggregation](https://gitcode.com/Ascend/msmodeling/blob/master/docs/en/quick_start/optix_agent_pd_disaggregation.md)**.

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
    "kv_cache_dtype": "bfloat16"
}
```

The scripts in Section 5.3 already include this setting. It changes only the draft KV cache dtype.

### 10.2 Model Loading Is Still in Progress or Startup Fails

**Symptom:** The API is not ready while loading the approximately 428B-parameter checkpoint or capturing graphs.

**Cause:** Weight loading, initialization, and graph capture depend on storage speed, the hardware, and the selected configuration. Elapsed time alone does not distinguish ongoing initialization from a failure.

**Resolution:** Follow the foreground logs for single-node deployment. For PD deployment, use `tail -f prefill.log` on Prefill; use `tail -f decode-0.log decode-1.log decode-2.log decode-3.log` on A3 D1 and `tail -f decode-4.log decode-5.log decode-6.log decode-7.log` on A3 D2; use `tail -f decode.log` on each 950DT products Decode node. Allow initialization to continue while logs show progress. Send requests or start the PD proxy after every required service prints `Application startup complete`; the single-node health check must return `200`.

If logs stop progressing or the process exits, inspect the first error in every affected role log. Check that `MODEL_PATH` points to the matching weight variant mounted in Section 4, confirm the expected devices are visible with `npu-smi info`, and compare the actual startup command with the corresponding platform example. For PD communication failures, also recheck `LOCAL_IP`, `NIC_NAME`, and [inter-node connectivity](#32-verify-multi-node-communication). For memory errors, review the context budget and concurrency described in [Section 5](#5-online-service-deployment) and the [out-of-memory FAQ](../../faqs.md#12-how-to-handle-the-out-of-memory-issue).
