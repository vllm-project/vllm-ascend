# Kimi-K3 (Experimental)

## 1 Introduction

!!! warning "Experimental support"
    Kimi K3 support in vLLM-Ascend 0.26.0rc is an initial experimental release. It is intended for evaluation and validation with the fixed deployment configurations in this guide. Supported scenarios, performance, and configuration interfaces may evolve in later releases; do not treat this guide as a production support commitment.

Kimi K3 is a native multimodal Mixture-of-Experts (MoE) model. Its language backbone combines Kimi Delta Attention (KDA) with periodic Gated Multi-head Latent Attention (MLA), and uses Stable LatentMoE for expert computation. The model also integrates a MoonViT vision encoder and supports text, image understanding, reasoning, and tool calling.

This document will show the main verification steps of the model, including supported features, feature configuration, environment preparation, multi-node deployment on Atlas 800 A3 and Atlas 800 A2, functional verification, and AISBench evaluation.

This document is validated and written based on **vLLM-Ascend 0.26.0rc**.

The current release includes a subset of the Kimi K3 optimization features that have been validated for this version. To provide a reproducible and supportable baseline, this guide uses fixed deployment configurations instead of exposing every tunable optimization.

These configurations position Kimi K3 for native multimodal inference, reasoning and tool calling, and multi-node mixed Prefill/Decode or PD separation deployments on Atlas 800 A3 and A2. Additional optimization features and configuration guidance will be added in later releases after validation.

## 2 Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [feature guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## 3 Prerequisites

### 3.1 Model Weight

Download the [Eco-Tech/Kimi-K3-w4a8](https://www.modelscope.cn/models/Eco-Tech/Kimi-K3-w4a8) ModelSlim W4A8 quantized weight from ModelScope. This guide includes the following validated deployment configurations:

| Platform                     | Deployment                                 | Topology                  |
| ---------------------------- | ------------------------------------------ | ------------------------- |
| 4 × Atlas 800 A3 (64G × 16)  | Mixed Prefill/Decode deployment            | DP4/TP16/EP64             |
| 16 × Atlas 800 A3 (64G × 16) | Eight Prefill nodes and eight Decode nodes | DP8/TP16/PP1 on each side |
| 8 × Atlas 800 A2 (64G × 8)   | Mixed Prefill/Decode deployment            | DP8/TP8/EP64              |

The checkpoint directory must contain the model configuration, tokenizer, image processor, and model weight files required by the published Kimi K3 package.

For PD separation with DSpark speculative decoding, download the [RadixArk/Kimi-K3-DSpark](https://huggingface.co/RadixArk/Kimi-K3-DSpark) MLA draft-model checkpoint in addition to the target-model checkpoint.

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`.

### 3.2 Verify Multi-node Communication (Optional)

If you want to deploy multi-node environment, you need to verify multi-node communication according to [verify multi-node communication environment](../../installation.md#verify-multi-node-communication).

## 4 Installation

### 4.1 Docker Image Installation

Select an image based on your host operating system and start it on every node. For more container options, refer to [using docker](../../installation.md#set-up-using-docker).

=== "A3 series"

    Kimi K3 is validated on Atlas 800 A3 (64G × 16).

    === "Ubuntu"

        ```bash
        export IMAGE=quay.io/ascend/vllm-ascend:kimi-k3-a3
        docker run --rm \
            --name vllm-ascend \
            --shm-size=1g \
            --net=host \
            --privileged=true \
            --device /dev/davinci0 \
            --device /dev/davinci1 \
            --device /dev/davinci2 \
            --device /dev/davinci3 \
            --device /dev/davinci4 \
            --device /dev/davinci5 \
            --device /dev/davinci6 \
            --device /dev/davinci7 \
            --device /dev/davinci8 \
            --device /dev/davinci9 \
            --device /dev/davinci10 \
            --device /dev/davinci11 \
            --device /dev/davinci12 \
            --device /dev/davinci13 \
            --device /dev/davinci14 \
            --device /dev/davinci15 \
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

    === "openEuler"

        ```bash
        export IMAGE=quay.io/ascend/vllm-ascend:kimi-k3-a3-openeuler
        docker run --rm \
            --name vllm-ascend \
            --shm-size=1g \
            --net=host \
            --privileged=true \
            --device /dev/davinci0 \
            --device /dev/davinci1 \
            --device /dev/davinci2 \
            --device /dev/davinci3 \
            --device /dev/davinci4 \
            --device /dev/davinci5 \
            --device /dev/davinci6 \
            --device /dev/davinci7 \
            --device /dev/davinci8 \
            --device /dev/davinci9 \
            --device /dev/davinci10 \
            --device /dev/davinci11 \
            --device /dev/davinci12 \
            --device /dev/davinci13 \
            --device /dev/davinci14 \
            --device /dev/davinci15 \
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

    After a successful `docker run`, verify the container with `docker ps`.

=== "A2 series"

    Kimi K3 is validated on Atlas 800 A2 (64G × 8).

    === "Ubuntu"

        ```bash
        export IMAGE=quay.io/ascend/vllm-ascend:kimi-k3
        docker run --rm \
            --name vllm-ascend \
            --shm-size=1g \
            --net=host \
            --privileged=true \
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

    === "openEuler"

        ```bash
        export IMAGE=quay.io/ascend/vllm-ascend:kimi-k3-openeuler
        docker run --rm \
            --name vllm-ascend \
            --shm-size=1g \
            --net=host \
            --privileged=true \
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

    After a successful `docker run`, verify the container with `docker ps`.

### 4.2 Source Code Installation

If you don't want to use the docker image as above, you can also build all from source:

- Install `vllm-ascend` from source, refer to [installation](../../installation.md).

If you want to deploy multi-node environment, you need to set up environment on each node.

Kimi K3 configuration, multimodal processing, reasoning parsing, and tool parsing are registered by vLLM-Ascend. Use a vLLM and vLLM-Ascend source revision that matches the validated version in this document.

## 5 Online Service Deployment

### 5.1 Mixed Prefill/Decode Deployment

=== "Atlas 800 A3 (four-node)"

    The validated mixed deployment uses four Atlas 800 A3 (64G × 16) nodes. vLLM data parallelism spans the four nodes, each node runs one DP rank, and tensor parallelism uses all 16 NPUs in the node. The resulting topology is DP4/TP16/EP64.

    Before starting the service:

    - Replace the model path, local IP address, network interface, service port, and DP RPC port with values from the target environment.
    - `NIC_NAME` must be the interface that owns `LOCAL_IP`.
    - Start Node 0 first. The `NODE0_IP` configured on Nodes 1 through 3 must equal `LOCAL_IP` on Node 0.
    - Assign `--data-parallel-start-rank` values `1`, `2`, and `3` to Nodes 1, 2, and 3 respectively.

    === "Node 0"

        ```shell
        # Values that must be adapted to the target environment.
        export MODEL_PATH=<KIMI_K3_MODEL_PATH>
        export LOCAL_IP=<NODE0_LOCAL_IP>
        export NIC_NAME=<NODE0_NIC_NAME>
        export PORT=<SERVICE_PORT>
        export RPC_PORT=<DP_RPC_PORT>
        export VLLM_VERSION=0.26.0
        export DRAFT_MODEL_PATH=<KIMI_K3_DSPARK_MODEL_PATH>

        export HCCL_BUFFSIZE=800
        export HCCL_IF_IP=$LOCAL_IP
        export HCCL_SOCKET_IFNAME=$NIC_NAME
        export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
        export GLOO_SOCKET_IFNAME=$NIC_NAME
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

        SPECULATIVE_CONFIG="$(
          printf \
          '{"method":"dspark","model":"%s","num_speculative_tokens":7,"draft_tensor_parallel_size":16,"max_model_len":4096,"draft_sample_method":"greedy","enforce_eager":true}' \
          "$DRAFT_MODEL_PATH"
        )"

        vllm serve $MODEL_PATH \
            --served-model-name kimi-k3 \
            --port $PORT \
            --allowed-local-media-path / \
            --trust-remote-code \
            --tensor-parallel-size 16 \
            --data-parallel-size 4 \
            --data-parallel-size-local 1 \
            --data-parallel-address $LOCAL_IP \
            --data-parallel-rpc-port $RPC_PORT \
            --enable-prefix-caching \
            --enable-expert-parallel \
            --max-num-seqs 16 \
            --max-model-len 131072 \
            --max-num-batched-tokens 24576 \
            --gpu-memory-utilization 0.9 \
            --speculative-config "$SPECULATIVE_CONFIG" \
            --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
            --mm-processor-cache-gb 0 \
            --additional-config '{"enable_cpu_binding":true, "enable_flashcomm1":true, "multistream_overlap_shared_expert":true}' \
            --mm-encoder-tp-mode data \
            --limit-mm-per-prompt '{"vision_chunk": 2}' \
            --enable-auto-tool-choice \
            --reasoning-parser kimi_k3 \
            --tool-call-parser kimi_k3 \
            --tokenizer-mode kimi_k3
        ```

    === "Nodes 1-3"

        Run this command on every worker node. Set `LOCAL_IP` and `NIC_NAME` to the current node and set `DP_START_RANK` to `1`, `2`, or `3`.

        ```shell
        # Values that must be adapted to the target environment.
        export MODEL_PATH=<KIMI_K3_MODEL_PATH>
        export LOCAL_IP=<WORKER_LOCAL_IP>
        export NODE0_IP=<NODE0_LOCAL_IP>
        export NIC_NAME=<WORKER_NIC_NAME>
        export PORT=<SERVICE_PORT>
        export RPC_PORT=<DP_RPC_PORT>
        export DP_START_RANK=<1_OR_2_OR_3>
        export VLLM_VERSION=0.26.0
        export DRAFT_MODEL_PATH=<KIMI_K3_DSPARK_MODEL_PATH>

        export HCCL_BUFFSIZE=800
        export HCCL_IF_IP=$LOCAL_IP
        export HCCL_SOCKET_IFNAME=$NIC_NAME
        export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
        export GLOO_SOCKET_IFNAME=$NIC_NAME
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

        SPECULATIVE_CONFIG="$(
          printf \
          '{"method":"dspark","model":"%s","num_speculative_tokens":7,"draft_tensor_parallel_size":16,"max_model_len":4096,"draft_sample_method":"greedy","enforce_eager":true}' \
          "$DRAFT_MODEL_PATH"
        )"

        vllm serve $MODEL_PATH \
            --headless \
            --served-model-name kimi-k3 \
            --port $PORT \
            --allowed-local-media-path / \
            --trust-remote-code \
            --tensor-parallel-size 16 \
            --data-parallel-size 4 \
            --data-parallel-size-local 1 \
            --data-parallel-start-rank $DP_START_RANK \
            --data-parallel-address $NODE0_IP \
            --data-parallel-rpc-port $RPC_PORT \
            --enable-prefix-caching \
            --enable-expert-parallel \
            --max-num-seqs 16 \
            --max-model-len 131072 \
            --max-num-batched-tokens 24576 \
            --gpu-memory-utilization 0.9 \
            --speculative-config "$SPECULATIVE_CONFIG" \
            --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
            --mm-processor-cache-gb 0 \
            --additional-config '{"enable_cpu_binding":true, "enable_flashcomm1":true, "multistream_overlap_shared_expert":true}' \
            --mm-encoder-tp-mode data \
            --limit-mm-per-prompt '{"vision_chunk": 2}' \
            --enable-auto-tool-choice \
            --reasoning-parser kimi_k3 \
            --tool-call-parser kimi_k3 \
            --tokenizer-mode kimi_k3
        ```

    The following values differ between the master and worker nodes:

    | Setting                       | Node 0         | Nodes 1-3         | Description                                            |
    | ----------------------------- | -------------- | ----------------- | ------------------------------------------------------ |
    | `LOCAL_IP`                    | Node 0 IP      | Current worker IP | Each node uses its own IP address.                     |
    | `NODE0_IP`                    | Not required   | Node 0 IP         | Workers use this address to join the DP group.         |
    | `--headless`                  | Omitted        | Enabled           | Workers do not expose the API endpoint.                |
    | `--data-parallel-address`     | `$LOCAL_IP`    | `$NODE0_IP`       | Always resolves to Node 0.                             |
    | `--data-parallel-start-rank`  | `0` by default | `1`, `2`, or `3`  | Every node must own a unique DP rank.                  |

    Key deployment parameters:

    | Parameter                                   | Description                                                      |
    | ------------------------------------------- | ---------------------------------------------------------------- |
    | `--tensor-parallel-size 16`                 | Uses all 16 NPUs in one A3 node for tensor parallelism.          |
    | `--data-parallel-size 4`                    | Creates four global DP ranks across four nodes.                  |
    | `--data-parallel-size-local 1`              | Runs one DP rank on the current node.                            |
    | `--data-parallel-start-rank`                | Selects the global starting DP rank for a worker node.           |
    | `--data-parallel-rpc-port`                  | Must be identical and reachable on every node.                   |
    | `--enable-expert-parallel`                  | Enables expert parallelism for the MoE layers.                   |
    | `--max-model-len 131072`                    | Sets the maximum combined input and output length.               |
    | `--max-num-seqs 16`                         | Sets the maximum active sequences for each DP group.             |
    | `--max-num-batched-tokens 24576`            | Controls the scheduler token budget.                             |
    | `--enable-prefix-caching`                   | Enables automatic prefix caching.                                |
    | `--compilation-config`                      | Uses `FULL_DECODE_ONLY` ACL Graph replay.                        |
    | `--tokenizer-mode kimi_k3`                  | Uses the Kimi K3 tokenizer mode.                                 |
    | `--additional-config`                       | Enables Ascend CPU binding and FlashComm1.                       |
    | `HCCL_IF_IP` and socket interface variables | Bind HCCL and Gloo communication to the selected interface.      |

    !!! note
        Serving a 1M-token context requires at least eight Atlas 800 A3 (64G × 16) nodes. Change the following parameters on every node:

        | Parameter                  | Four-node default | Eight-node (1M context) |
        | -------------------------- | ----------------- | ----------------------- |
        | `--data-parallel-size`     | `4`               | `8`                     |
        | `--max-model-len`          | `131072`          | `1048576`               |
        | `--max-num-batched-tokens` | `24576`           | `8192`                  |

        Run the worker command on Nodes 1 through 7 and assign each node a unique `--data-parallel-start-rank` from `1` through `7`.

    If a worker exits immediately, confirm that Node 0 is already running, `--data-parallel-address` resolves to Node 0, and every worker uses a unique `--data-parallel-start-rank`.

    Verify the service through Node 0:

    ```shell
    curl http://<NODE0_LOCAL_IP>:<SERVICE_PORT>/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "kimi-k3",
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": "The future of AI is"
                }]
            }],
            "max_tokens": 1024,
            "temperature": 1.0,
            "top_p": 0.95
        }'
    ```

    The service should return HTTP 200 and a `choices` field containing generated text.

=== "Atlas 800 A2 (eight-node)"

    The validated Atlas 800 A2 deployment uses eight nodes with eight NPUs per node. Each node runs one DP rank and uses all eight local NPUs for tensor parallelism. Every DP rank handles both Prefill and Decode, resulting in a DP8/TP8/EP64 topology. Node 0 runs the API server and DP rank 0, while Nodes 1 through 7 run headless DP workers. This baseline serves the language model only.

    Before starting the service:

    - Replace the model path, local IP address, network interface, service port, and DP RPC port with values from the target environment.
    - `NIC_NAME` must be the interface that owns `LOCAL_IP`.
    - Start Node 0 first. The `NODE0_IP` configured on Nodes 1 through 7 must equal `LOCAL_IP` on Node 0.
    - Assign a unique `DP_START_RANK` from `1` through `7` to each worker node.
    - Ensure proxy bypass settings include all API and communication IP addresses used by the eight nodes.

    === "Node 0"

        ```shell
        # Values that must be adapted to the target environment.
        export MODEL_PATH=<KIMI_K3_MODEL_PATH>
        export LOCAL_IP=<NODE0_LOCAL_IP>
        export NIC_NAME=<NODE0_NIC_NAME>
        export PORT=<SERVICE_PORT>
        export RPC_PORT=<DP_RPC_PORT>

        source /usr/local/Ascend/ascend-toolkit/set_env.sh
        source /vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash

        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
        export HCCL_BUFFSIZE=256
        export HCCL_IF_IP=$LOCAL_IP
        export HCCL_INTRA_ROCE_ENABLE=1
        export HCCL_SOCKET_IFNAME=$NIC_NAME
        export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        export GLOO_SOCKET_IFNAME=$NIC_NAME
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export VLLM_LOGGING_LEVEL=INFO
        export TIKTOKEN_CACHE_DIR=/root/.cache/tiktoken-k3

        vllm serve $MODEL_PATH \
            --host 0.0.0.0 \
            --port $PORT \
            --served-model-name kimi-k3 \
            --trust-remote-code \
            --language-model-only \
            --mm-encoder-tp-mode data \
            --skip-mm-profiling \
            --limit-mm-per-prompt '{"vision_chunk":2}' \
            --data-parallel-size 8 \
            --data-parallel-size-local 1 \
            --data-parallel-start-rank 0 \
            --data-parallel-address $LOCAL_IP \
            --data-parallel-rpc-port $RPC_PORT \
            --tensor-parallel-size 8 \
            --enable-expert-parallel \
            --dtype bfloat16 \
            --max-model-len 262144 \
            --gpu-memory-utilization 0.90 \
            --enable-prefix-caching \
            --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8]}' \
            --tokenizer-mode kimi_k3 \
            --enable-auto-tool-choice \
            --reasoning-parser kimi_k3 \
            --tool-call-parser kimi_k3 \
            --additional-config '{"enable_flashcomm1":false,"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},"enable_cpu_binding":true}'
        ```

    === "Nodes 1-7"

        Run this command on every worker node. Set `LOCAL_IP` and `NIC_NAME` to the current node and set `DP_START_RANK` to a unique value from `1` through `7`.

        ```shell
        # Values that must be adapted to the target environment.
        export MODEL_PATH=<KIMI_K3_MODEL_PATH>
        export LOCAL_IP=<WORKER_LOCAL_IP>
        export NODE0_IP=<NODE0_LOCAL_IP>
        export NIC_NAME=<WORKER_NIC_NAME>
        export PORT=<SERVICE_PORT>
        export RPC_PORT=<DP_RPC_PORT>
        export DP_START_RANK=<1_TO_7>

        source /usr/local/Ascend/ascend-toolkit/set_env.sh
        source /vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash

        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
        export HCCL_BUFFSIZE=256
        export HCCL_IF_IP=$LOCAL_IP
        export HCCL_INTRA_ROCE_ENABLE=1
        export HCCL_SOCKET_IFNAME=$NIC_NAME
        export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        export GLOO_SOCKET_IFNAME=$NIC_NAME
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export VLLM_LOGGING_LEVEL=INFO
        export TIKTOKEN_CACHE_DIR=/root/.cache/tiktoken-k3

        vllm serve $MODEL_PATH \
            --headless \
            --host 0.0.0.0 \
            --port $PORT \
            --served-model-name kimi-k3 \
            --trust-remote-code \
            --language-model-only \
            --mm-encoder-tp-mode data \
            --skip-mm-profiling \
            --limit-mm-per-prompt '{"vision_chunk":2}' \
            --data-parallel-size 8 \
            --data-parallel-size-local 1 \
            --data-parallel-start-rank $DP_START_RANK \
            --data-parallel-address $NODE0_IP \
            --data-parallel-rpc-port $RPC_PORT \
            --tensor-parallel-size 8 \
            --enable-expert-parallel \
            --dtype bfloat16 \
            --max-model-len 262144 \
            --gpu-memory-utilization 0.90 \
            --enable-prefix-caching \
            --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8]}' \
            --tokenizer-mode kimi_k3 \
            --enable-auto-tool-choice \
            --reasoning-parser kimi_k3 \
            --tool-call-parser kimi_k3 \
            --additional-config '{"enable_flashcomm1":false,"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},"enable_cpu_binding":true}'
        ```

    The following values differ between the master and worker nodes:

    | Setting                      | Node 0       | Nodes 1-7                         | Description                                      |
    | ---------------------------- | ------------ | --------------------------------- | ------------------------------------------------ |
    | `LOCAL_IP`                   | Node 0 IP    | Current worker IP                 | Each node uses its own communication IP address. |
    | `NODE0_IP`                   | Not required | Node 0 IP                         | Workers use this address to join the DP group.   |
    | `--headless`                 | Omitted      | Enabled                           | Workers do not expose an API endpoint.           |
    | `--data-parallel-address`    | `$LOCAL_IP`  | `$NODE0_IP`                       | Always resolves to Node 0.                       |
    | `--data-parallel-start-rank` | `0`          | Unique value from `1` through `7` | Every node owns one global DP rank.              |

    Key A2 deployment parameters:

    | Parameter                      | Description                                                                     |
    | ------------------------------ | ------------------------------------------------------------------------------- |
    | `--tensor-parallel-size 8`     | Uses all eight NPUs in one A2 node for tensor parallelism.                      |
    | `--data-parallel-size 8`       | Creates eight global DP ranks across eight nodes.                               |
    | `--data-parallel-size-local 1` | Runs one DP rank on the current node.                                           |
    | `--language-model-only`        | Disables the multimodal encoder for this validated A2 baseline.                 |
    | `--max-model-len 262144`       | Sets a 256K combined input and output context limit.                            |
    | `--compilation-config`         | Uses `FULL_DECODE_ONLY` graph replay with capture sizes `1`, `2`, `4`, and `8`. |
    | `--additional-config`          | Enables NPU graph execution and CPU binding while keeping FlashComm1 disabled.  |

    Do not set `HCCL_OP_EXPANSION_MODE=AIV` for this baseline. Start Node 0 first, then start Nodes 1 through 7 as soon as possible. If a worker exits immediately, verify that Node 0 is running, all nodes use the same RPC port, `--data-parallel-address` resolves to Node 0, and every worker has a unique DP start rank.

### 5.2 Sixteen-Node PD Separation Deployment

The validated PD separation topology uses 16 Atlas 800 A3 (64G × 16) nodes: eight Prefill nodes and eight Decode nodes. Both sides use DP8/TP16/PP1. Prefill nodes additionally use a memcache-backed KV pool.

Refer to [PD Disaggregation with Mooncake](../features/pd_disaggregation_mooncake_multi_node.md) for the general service workflow and [KV Pool](../../user_guide/feature_guide/kv_pool.md) for memcache pool concepts.

This deployment supports DSpark speculative decoding. Configure the same draft-model path and `num_speculative_tokens` on both Prefill and Decode nodes. The validated configuration uses a Kimi K3 MLA draft model, TP16, greedy drafting, and seven speculative tokens.

#### 5.2.1 Start the memcache MetaService

Start one MetaService instance before the Prefill engines:

```shell
export MMC_META_CONFIG_PATH=<PATH_TO_MMC_META_CONF>
python -c "from memcache_hybrid import MetaService; MetaService.main()"
```

`mmc-meta.conf` configures MetaService and `mmc-local.conf` is loaded by every Prefill inference process. Run `pip show memcache_hybrid` to locate the installed package, copy the example files from `memcache_hybrid/config/`, and adapt them to the target environment.

#### 5.2.2 Create the engine templates

=== "Prefill"

    ```shell
    KV_PORT=36000

    unset ftp_proxy FTP_PROXY
    unset https_proxy HTTPS_PROXY
    unset http_proxy HTTP_PROXY

    nic_name=<PREFILL_NIC_NAME>
    local_ip=<PREFILL_LOCAL_IP>

    export VLLM_VERSION=0.26.0
    export DRAFT_MODEL_PATH=<KIMI_K3_DSPARK_MODEL_PATH>
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
    export HCCL_BUFFSIZE=1024
    export HCCL_IF_IP=${local_ip}
    export HCCL_SOCKET_IFNAME=${nic_name}
    export ASCEND_ENABLE_USE_FABRIC_MEM=1
    export ASCEND_RT_VISIBLE_DEVICES=$1
    export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH
    export GLOO_SOCKET_IFNAME=${nic_name}
    export MMC_LOCAL_CONFIG_PATH=<PATH_TO_MMC_LOCAL_CONF>
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export PYTHONHASHSEED=0

    SPECULATIVE_CONFIG="$(
      printf \
      '{"method":"dspark","model":"%s","num_speculative_tokens":7,"draft_tensor_parallel_size":16,"max_model_len":4096,"draft_sample_method":"greedy","enforce_eager":true}' \
      "$DRAFT_MODEL_PATH"
    )"

    vllm serve <KIMI_K3_MODEL_PATH> \
        --host 0.0.0.0 \
        --port $2 \
        --enable-auto-tool-choice \
        --reasoning-parser kimi_k3 \
        --tool-call-parser kimi_k3 \
        --tokenizer-mode kimi_k3 \
        --data-parallel-size $3 \
        --data-parallel-rank $4 \
        --data-parallel-address $5 \
        --data-parallel-rpc-port $6 \
        --tensor-parallel-size $7 \
        --enable-expert-parallel \
        --seed 1024 \
        --served-model-name kimi-k3 \
        --max-model-len 133120 \
        --max-num-batched-tokens 8192 \
        --max-num-seqs 16 \
        --enforce-eager \
        --trust-remote-code \
        --gpu-memory-utilization 0.9 \
        --speculative-config "$SPECULATIVE_CONFIG" \
        --quantization ascend \
        --mm-encoder-tp-mode data \
        --skip-mm-profiling \
        --safetensors_load_strategy prefetch \
        --mamba-cache-mode align \
        --enable-prefix-caching \
        --additional-config '{"recompute_scheduler_enable":false,"enable_flashcomm1":true,"enable_mlapo":true,"multistream_overlap_shared_expert":true}' \
        --limit-mm-per-prompt '{"vision_chunk": 2}' \
        --kv-transfer-config \
        '{
          "kv_connector": "MultiConnector",
          "kv_role": "kv_producer",
          "kv_connector_extra_config": {
            "connectors": [
              {
                "kv_connector": "MooncakeConnectorV1",
                "kv_role": "kv_producer",
                "kv_port": "'"$KV_PORT"'",
                "kv_connector_extra_config": {
                  "prefill": {"dp_size": 8, "tp_size": 16},
                  "decode": {"dp_size": 8, "tp_size": 16}
                }
              },
              {
                "kv_connector": "AscendStoreConnector",
                "kv_role": "kv_producer",
                "kv_connector_extra_config": {
                  "backend": "memcache",
                  "lookup_rpc_port": "0"
                }
              }
            ]
          }
        }'
    ```

=== "Decode"

    ```shell
    KV_PORT=36200

    unset ftp_proxy FTP_PROXY
    unset https_proxy HTTPS_PROXY
    unset http_proxy HTTP_PROXY

    nic_name=<DECODE_NIC_NAME>
    local_ip=<DECODE_LOCAL_IP>

    export VLLM_VERSION=0.26.0
    export DRAFT_MODEL_PATH=<KIMI_K3_DSPARK_MODEL_PATH>
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
    export HCCL_BUFFSIZE=1024
    export HCCL_IF_IP=${local_ip}
    export HCCL_OP_EXPANSION_MODE="AIV"
    export HCCL_SOCKET_IFNAME=${nic_name}
    export ASCEND_ENABLE_USE_FABRIC_MEM=1
    export ASCEND_RT_VISIBLE_DEVICES=$1
    export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH
    export GLOO_SOCKET_IFNAME=${nic_name}
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

    SPECULATIVE_CONFIG="$(
      printf \
      '{"method":"dspark","model":"%s","num_speculative_tokens":7,"draft_tensor_parallel_size":16,"max_model_len":4096,"draft_sample_method":"greedy","enforce_eager":true}' \
      "$DRAFT_MODEL_PATH"
    )"

    vllm serve <KIMI_K3_MODEL_PATH> \
        --host 0.0.0.0 \
        --port $2 \
        --enable-auto-tool-choice \
        --reasoning-parser kimi_k3 \
        --tool-call-parser kimi_k3 \
        --tokenizer-mode kimi_k3 \
        --data-parallel-size $3 \
        --data-parallel-rank $4 \
        --data-parallel-address $5 \
        --data-parallel-rpc-port $6 \
        --tensor-parallel-size $7 \
        --enable-expert-parallel \
        --seed 1024 \
        --served-model-name kimi-k3 \
        --max-model-len 133120 \
        --max-num-batched-tokens 8192 \
        --max-num-seqs 16 \
        --trust-remote-code \
        --gpu-memory-utilization 0.9 \
        --speculative-config "$SPECULATIVE_CONFIG" \
        --quantization ascend \
        --mm-encoder-tp-mode data \
        --skip-mm-profiling \
        --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
        --safetensors_load_strategy prefetch \
        --mamba-cache-mode align \
        --enable-prefix-caching \
        --additional-config '{"recompute_scheduler_enable":false,"enable_flashcomm1":true,"enable_mlapo":true,"multistream_overlap_shared_expert":true}' \
        --limit-mm-per-prompt '{"vision_chunk":2}' \
        --kv-transfer-config \
        '{
          "kv_connector": "MooncakeConnectorV1",
          "kv_role": "kv_consumer",
          "kv_port": "'"$KV_PORT"'",
          "kv_connector_extra_config": {
            "prefill": {"dp_size": 8, "tp_size": 16},
            "decode": {"dp_size": 8, "tp_size": 16}
          }
        }'
    ```

#### 5.2.3 Start the engines

Deploy `launch_online_dp.py` and the corresponding engine template on every node. The following example starts one local DP rank in a DP8/TP16/PP1 group:

```shell
python launch_online_dp.py \
    --dp-size 8 \
    --tp-size 16 \
    --pp-size 1 \
    --dp-size-local 1 \
    --dp-rank-start <LOCAL_DP_RANK> \
    --dp-address <PD_MASTER_IP> \
    --dp-rpc-port <DP_RPC_PORT> \
    --vllm-start-port <VLLM_START_PORT>
```

Use ranks `0` through `7` for each eight-node side. Configure independent master addresses, RPC ports, and vLLM port ranges for the Prefill and Decode groups.

After the engines start, configure and start the load-balancing proxy as described in [PD Disaggregation with Mooncake](../features/pd_disaggregation_mooncake_multi_node.md#start-the-service).

Key PD settings:

| Setting                      | Value                        | Description                                             |
| ---------------------------- | ---------------------------- | ------------------------------------------------------- |
| Topology                     | 8P8D                         | Eight Prefill and eight Decode nodes.                   |
| `--dp-size`                  | `8`                          | Eight DP ranks on each side.                            |
| `--tp-size`                  | `16`                         | Uses all 16 NPUs in a node.                             |
| `--pp-size`                  | `1`                          | One pipeline stage per engine.                          |
| `--dp-size-local`            | `1`                          | One DP rank per node.                                   |
| `KV_PORT`                    | `36000` for P, `36200` for D | Separates producer and consumer KV traffic.             |
| `MMC_LOCAL_CONFIG_PATH`      | Prefill only                 | Connects the producer to the memcache KV pool.          |
| `recompute_scheduler_enable` | `false`                      | Matches the validated Prefill and Decode configuration. |

## 6 Functional Verification

### 6.1 Mixed Deployment Functional Verification

=== "Atlas 800 A3"

    After an A3 mixed or PD service is ready, send a multimodal request to the API endpoint:

    ```shell
    curl http://<SERVICE_IP>:<SERVICE_PORT>/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "kimi-k3",
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "<IMAGE_URL_OR_DATA_URL>"}
                    },
                    {
                        "type": "text",
                        "text": "Describe the image."
                    }
                ]
            }],
            "max_tokens": 1024,
            "temperature": 1.0,
            "top_p": 0.95
        }'
    ```

    The service should return HTTP 200 and a `choices` field containing the image description. The current implementation supports image inputs but does not support video inputs.

=== "Atlas 800 A2"

    The validated A2 deployment uses `--language-model-only`. After all eight DP ranks are ready, send a text request to the Node 0 API endpoint:

    ```shell
    curl http://<NODE0_LOCAL_IP>:<SERVICE_PORT>/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "kimi-k3",
            "messages": [{
                "role": "user",
                "content": "Explain data parallelism in one sentence."
            }],
            "max_tokens": 64
        }'
    ```

    The service should return HTTP 200 and a `choices` field containing generated text. Nodes 1 through 7 are headless workers and do not accept HTTP requests directly.

    `X-data-parallel-rank` is an optional HTTP request header that pins a request to a specific DP rank. Without this header, the internal vLLM load balancer on Node 0 selects an available rank. For this DP8 deployment, use an integer from `0` through `7` only when validating one rank, troubleshooting a worker, or testing rank-local prefix-cache behavior:

    ```shell
    -H "X-data-parallel-rank: 0" \
    ```

    Production traffic should normally omit this header so that requests remain balanced across all DP ranks. The request is always sent to the Node 0 API endpoint, even when a worker rank is selected.

## 7 AISBench Evaluation

The following GPQA accuracy and performance evaluation procedures use AISBench with the four-node DP4/TP16/EP64 service.

### 7.1 Install AISBench

Run AISBench in a separate environment or container on the master node so the load generator does not affect the serving processes:

```shell
git clone https://github.com/AISBench/benchmark
cd benchmark
pip3 install -e ./ --use-pep517
pip3 install -r requirements/api.txt
pip3 install -r requirements/extra.txt
pip3 install -r requirements/hf_vl_dependency.txt
```

### 7.2 GPQA Accuracy Evaluation

GPQA Diamond is a four-option scientific question-answering benchmark. Use the standard four-node mixed deployment in [Section 5.1.1](#511-four-node-mixed-deployment) for this test; do not apply the performance-specific service settings in the next section.

#### 7.2.1 Download the GPQA Dataset

Download and extract the dataset in the AISBench dataset directory:

```shell
cd <AISBENCH_BENCHMARK_DIRECTORY>/ais_bench/datasets
wget http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/gpqa.zip
unzip gpqa.zip
```

This creates the `gpqa/` directory containing `gpqa_diamond.csv`, which is the subset used by the configured benchmark.

#### 7.2.2 Configure AISBench

Configure the following files in the AISBench source tree:

- `<AISBENCH_BENCHMARK_DIRECTORY>/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py`
    - Set `path` to `<KIMI_K3_MODEL_PATH>`.
    - Set `model` to `kimi-k3`.
    - Set `host_ip` and `host_port` to the Node 0 service endpoint.
    - Set `max_out_len` to `65536` and `batch_size` to `32`.
    - Configure `generation_kwargs` with the Kimi K3 settings used for the evaluation.
- `<AISBENCH_BENCHMARK_DIRECTORY>/ais_bench/benchmark/configs/datasets/gpqa/gpqa_gen_0_shot_cot_chat_prompt.py`
    - Set `path` to the directory that contains `gpqa_diamond.csv`. The default is `ais_bench/datasets/gpqa/`.

The command below uses `gpqa_gen_0_shot_cot_chat_prompt`; do not configure `gpqa_gen_0_shot_str` for this run. The CoT prompt requires the response to end with `Answer: <A|B|C|D>`.

#### 7.2.3 Run GPQA

After the service is ready, run:

```shell
ais_bench --models vllm_api_stream_chat --datasets gpqa_gen_0_shot_cot_chat_prompt --debug --dump-eval-details
```

#### 7.2.4 View Results

AISBench writes output to `outputs/default/<timestamp>/` by default. Read the `accuracy` field from `results/vllm-api-stream-chat/GPQA_diamond.json`. The corresponding `predictions/` output and evaluation details from `--dump-eval-details` can be used to inspect individual answers.

### 7.3 Performance Service Configuration

Change these values from the standard Section 5.1.1 deployment on all four nodes:

| Parameter                  | Standard deployment | Performance test |
| -------------------------- | ------------------- | ---------------- |
| `--max-model-len`          | 131072              | 250000           |
| `--max-num-batched-tokens` | 24576               | 8192             |
| `--gpu-memory-utilization` | 0.9                 | 0.95             |

The master-node `vllm serve` command is:

```shell
vllm serve <KIMI_K3_MODEL_PATH> \
    --served-model-name kimi-k3 \
    --port <SERVICE_PORT> \
    --allowed-local-media-path / \
    --trust-remote-code \
    --tensor-parallel-size 16 \
    --data-parallel-size 4 \
    --data-parallel-size-local 1 \
    --data-parallel-address <NODE0_LOCAL_IP> \
    --data-parallel-rpc-port <DP_RPC_PORT> \
    --enable-prefix-caching \
    --enable-expert-parallel \
    --max-num-seqs 16 \
    --max-model-len 250000 \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.95 \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --mm-processor-cache-gb 0 \
    --additional-config '{"enable_cpu_binding":true, "enable_flashcomm1":true}' \
    --mm-encoder-tp-mode data \
    --limit-mm-per-prompt '{"vision_chunk": 2}' \
    --enable-auto-tool-choice \
    --reasoning-parser kimi_k3 \
    --tool-call-parser kimi_k3 \
    --tokenizer-mode kimi_k3
```

Worker nodes use the same performance values and the worker-specific arguments from Section 5.1.1.

### 7.4 Configure the Load Generator

Before running `aisbench_test.py`, create its dataset directory and configure the validation helper:

```shell
mkdir -p <DATASET_DIRECTORY>
```

```python
DATASET_PATH = "<DATASET_DIRECTORY>"
WORK_PATH = "<AISBENCH_BENCHMARK_DIRECTORY>"
MODEL_NAME = "kimi-k3"
MODEL_PATH = "<KIMI_K3_MODEL_PATH>"
HOST_IP = "<SERVICE_IP>"
HOST_PORT = "<SERVICE_PORT>"
DEFAULT_PERFORMANCE_TEST = "default_perf"
OUTPUT_DIR = "./outputs/default"

# Set the serving endpoints when collecting per-DP prefix-cache metrics.
# PD deployments should list every relevant endpoint.
POD_INFO = []
```

Disable proxies before the test:

```shell
env | grep -i proxy
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
```

### 7.5 Run the Performance Tests

8K input, 1K output, and no prefix-cache hit:

```shell
python3 aisbench_test.py \
    --input_len 8192 \
    --output_len 1024 \
    --data_num 16 \
    --concurrency 4 \
    --request_rate 0 \
    --repeat_rate 0 \
    --prefix_test
```

128K input, 1K output, and a 99% prefix-cache hit rate:

```shell
python3 aisbench_test.py \
    --input_len 131024 \
    --output_len 1024 \
    --data_num 16 \
    --concurrency 4 \
    --request_rate 0 \
    --dataset_type prefix_cache \
    --repeat_rate 0.99 \
    --prefix_test
```

`request_rate=0` sends requests as quickly as the configured concurrency permits. `repeat_rate=0.99` makes 99% of requests reuse the same prefix.

### 7.6 Enabled Optimizations

| Feature                                    | Description                                                                                                             |
| ------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------- |
| Chunked Prefill                            | Splits long prefill inputs into chunks to reduce per-step memory peaks.                                                 |
| Asynchronous scheduling                    | Decouples scheduling and execution.                                                                                     |
| Prefix Cache                               | Reuses KV state for repeated prefixes.                                                                                  |
| DP + TP + EP                               | Combines data, tensor, and expert parallelism for the MoE model.                                                        |
| ACL Graph                                  | Uses `FULL_DECODE_ONLY` replay to reduce decode scheduling overhead.                                                    |
| KDA + MLA cache management                 | Manages the heterogeneous recurrent and KV states.                                                                      |
| FlashComm1                                 | Enables communication optimization.                                                                                     |
| CPU Binding                                | Reduces cross-core scheduling overhead.                                                                                 |
| DSpark speculative decoding (PD only)      | Uses the Kimi K3 MLA draft model to generate seven greedy draft tokens per forward pass.                                |
| KDA fused norm gate and attention residual | Uses Triton fused paths when Triton is available. No server option is required.                                         |
| KDA fused QKV projection                   | Fuses the KDA Q, K, and V projections automatically for supported quantized linear paths. No server option is required. |
| Stride-aware recurrent KDA state update    | Updates non-contiguous KDA state views in place without copying the full state. No server option is required.           |

The automatic operator optimizations require a serving image that includes their corresponding vLLM-Ascend implementation.

## 8 Performance Tuning

Use the validated deployment values above as a baseline. Adjust `max-model-len`, `max-num-seqs`, `max-num-batched-tokens`, and `gpu-memory-utilization` together for the target workload.

Refer to the [performance tuning guide](../../developer_guide/performance_and_debug/optimization_and_tuning.md) and the [feature matrix](../../user_guide/support_matrix/feature_matrix.md) for additional guidance.

## 9 FAQ

For common environment, installation, and general parameter issues, refer to the [Public FAQ](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html).

- **Q: Which multimodal inputs are supported by the current Kimi K3 implementation?**
A: The current local processor accepts image inputs. Video inputs are not supported.
- **Q: Which server options are required for Kimi K3 reasoning and tool calling?**
A: Configure `--tokenizer-mode kimi_k3`, `--enable-auto-tool-choice`, `--reasoning-parser kimi_k3`, and `--tool-call-parser kimi_k3` together.
- **Q: How should TP size be selected?**
A: TP size must divide the checkpoint's attention-head count. It also affects KDA state layout and expert placement, so validate memory capacity and communication performance together.
- **Q: How is DSpark enabled in PD separation?**
A: Download a Kimi K3 MLA DSpark draft checkpoint, set `DRAFT_MODEL_PATH` on both Prefill and Decode nodes, and pass the same `--speculative-config` to both. Prefill must retain `--enforce-eager`.
