# Kimi-K3 (Experimental)

## 1 Introduction

!!! warning "Experimental support"
    Kimi K3 support in vLLM-Ascend 0.26.0rc is an initial experimental release. It is intended for evaluation and validation with the fixed deployment configurations in this guide. Supported scenarios, performance, and configuration interfaces may evolve in later releases; do not treat this guide as a production support commitment.

Kimi K3 is a native multimodal Mixture-of-Experts (MoE) model. Its language backbone combines Kimi Delta Attention (KDA) with periodic Gated Multi-head Latent Attention (MLA), and uses Stable LatentMoE for expert computation. The model also integrates a MoonViT vision encoder and supports text, image understanding, reasoning, and tool calling.

This document will show the main verification steps of the model, including supported features, feature configuration, environment preparation, multi-node deployment on Atlas 800 A3, Atlas 800 A2, and Atlas 950DT, functional verification, and AISBench evaluation.

This document is validated and written based on **vLLM-Ascend 0.26.0rc**.

The current release includes a subset of the Kimi K3 optimization features that have been validated for this version. To provide a reproducible and supportable baseline, this guide uses fixed deployment configurations instead of exposing every tunable optimization.

These configurations position Kimi K3 for native multimodal inference, reasoning and tool calling, and multi-node mixed Prefill/Decode or PD separation deployments on Atlas 800 A3, A2, and Atlas 950DT. Additional optimization features and configuration guidance will be added in later releases after validation.

## 2 Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [feature guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## 3 Prerequisites

### 3.1 Model Weight

Download the [Eco-Tech/Kimi-K3-w4a8](https://www.modelscope.cn/models/Eco-Tech/Kimi-K3-w4a8) ModelSlim W4A8 quantized weight from ModelScope. This guide includes the following validated deployment configurations:

| Platform                     | Deployment                                 | Topology                  |
| ---------------------------- | ------------------------------------------ | ------------------------- |
| 4 × Atlas 800 A3 (64G × 16)  | Mixed Prefill/Decode deployment            | DP4/TP16/EP64             |
| 8 × Atlas 800 A3 (64G × 16)  | Four Prefill nodes and four Decode nodes   | DP4/TP16/PP1 on each side |
| 8 × Atlas 800 A2 (64G × 8)   | Mixed Prefill/Decode deployment            | DP8/TP8/EP64              |
| 4 × Atlas 950DT (8 devices)   | Mixed Prefill/Decode deployment            | DP4/TP8/EP32              |
| 8 × Atlas 950DT (8 devices)   | Four Prefill nodes and four Decode nodes   | DP4/TP8/PP1 on each side  |

The checkpoint directory must contain the model configuration, tokenizer, image processor, and model weight files required by the published Kimi K3 package.

For DSpark speculative decoding in mixed or PD separation deployments, download the [Inferact/Kimi-K3-DSpark](https://huggingface.co/Inferact/Kimi-K3-DSpark) MLA draft-model checkpoint in addition to the target-model checkpoint.

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

=== "Atlas 950DT"

    Kimi K3 is validated on Atlas 950DT with eight devices per node.

    === "Ubuntu"

        ```bash
        export IMAGE=quay.io/ascend/vllm-ascend:kimi-k3-a5
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
        export IMAGE=quay.io/ascend/vllm-ascend:kimi-k3-a5-openeuler
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

!!! warning "DSpark long-context limitation"
    For the current `Inferact/Kimi-K3-DSpark` draft weights, DSpark acceptance is low for approximately 20k–40k-token inputs and remains low for longer contexts. Consequently, 128k inputs may receive no effective speculative-decoding benefit. This is a limitation of the current draft weights, not of the DSpark framework; validate acceptance and end-to-end performance for the target workload before enabling the draft model.

The A2 capabilities have not changed in this release and remain consistent with **vLLM-Ascend 0.23.0**; no iterative updates have been made.

### 5.1 Mixed Prefill/Decode Deployment

=== "Atlas 800 A3 (four-node)"

    The validated mixed deployment uses four Atlas 800 A3 (64G × 16) nodes. vLLM data parallelism spans the four nodes, each node runs one DP rank, and tensor parallelism uses all 16 NPUs in the node. The resulting topology is DP4/TP16/EP64.

    On Atlas 800 A3, a mixed Prefill/Decode deployment follows the Prefill execution mode. HCCL therefore uses AICPU by default; leave `HCCL_OP_EXPANSION_MODE` unset.

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
            --additional-config '{"enable_cpu_binding":true, "enable_flashcomm1":true}' \
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
            --additional-config '{"enable_cpu_binding":true, "enable_flashcomm1":true}' \
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
    | `--additional-config`                       | Enables Ascend CPU binding and FlashComm1.                         |
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
    | `--max-model-len 262144`       | Sets a 256k combined input and output context limit.                            |
    | `--compilation-config`         | Uses `FULL_DECODE_ONLY` graph replay with capture sizes `1`, `2`, `4`, and `8`. |
    | `--additional-config`          | Enables NPU graph execution and CPU binding while keeping FlashComm1 disabled.  |

    Do not set `HCCL_OP_EXPANSION_MODE=AIV` for this baseline. Start Node 0 first, then start Nodes 1 through 7 as soon as possible. If a worker exits immediately, verify that Node 0 is running, all nodes use the same RPC port, `--data-parallel-address` resolves to Node 0, and every worker has a unique DP start rank.

=== "Atlas 950DT (four-node)"

    The validated mixed deployment uses four Atlas 950DT nodes with eight devices per node. vLLM data parallelism spans the four nodes, each node runs one DP rank, and tensor parallelism uses all eight devices in the node. The resulting topology is DP4/TP8/EP32.

    The Atlas 950DT feature set and service options are the same as A3, except that Atlas 950DT uses TP8 and exposes eight devices per node. Leave `HCCL_OP_EXPANSION_MODE` unset for this mixed Prefill/Decode deployment.

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
        export DRAFT_MODEL_PATH=<KIMI_K3_DSPARK_MODEL_PATH>

        export HCCL_BUFFSIZE=800
        export HCCL_IF_IP=$LOCAL_IP
        export HCCL_SOCKET_IFNAME=$NIC_NAME
        export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        export GLOO_SOCKET_IFNAME=$NIC_NAME
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

        SPECULATIVE_CONFIG="$(
          printf \
          '{"method":"dspark","model":"%s","num_speculative_tokens":7,"draft_tensor_parallel_size":8,"max_model_len":4096,"draft_sample_method":"greedy","enforce_eager":true}' \
          "$DRAFT_MODEL_PATH"
        )"

        vllm serve $MODEL_PATH \
            --served-model-name kimi-k3 \
            --port $PORT \
            --allowed-local-media-path / \
            --trust-remote-code \
            --tensor-parallel-size 8 \
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
            --additional-config '{"enable_cpu_binding":true, "enable_flashcomm1":true}' \
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
        export DRAFT_MODEL_PATH=<KIMI_K3_DSPARK_MODEL_PATH>

        export HCCL_BUFFSIZE=800
        export HCCL_IF_IP=$LOCAL_IP
        export HCCL_SOCKET_IFNAME=$NIC_NAME
        export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        export GLOO_SOCKET_IFNAME=$NIC_NAME
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

        SPECULATIVE_CONFIG="$(
          printf \
          '{"method":"dspark","model":"%s","num_speculative_tokens":7,"draft_tensor_parallel_size":8,"max_model_len":4096,"draft_sample_method":"greedy","enforce_eager":true}' \
          "$DRAFT_MODEL_PATH"
        )"

        vllm serve $MODEL_PATH \
            --headless \
            --served-model-name kimi-k3 \
            --port $PORT \
            --allowed-local-media-path / \
            --trust-remote-code \
            --tensor-parallel-size 8 \
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
            --additional-config '{"enable_cpu_binding":true, "enable_flashcomm1":true}' \
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
    | `--tensor-parallel-size 8`                  | Uses all eight devices in one Atlas 950DT node for tensor parallelism. |
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
    | `--additional-config`                       | Enables Ascend CPU binding and FlashComm1.                        |
    | `HCCL_IF_IP` and socket interface variables | Bind HCCL and Gloo communication to the selected interface.      |

    !!! note
        Serving a 1M-token context requires at least eight Atlas 950DT nodes. Change the following parameters on every node:

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

### 5.2 Eight-Node PD Separation Deployment

The validated PD separation topology uses eight nodes: four Prefill nodes and four Decode nodes. A3 uses DP4/TP16/PP1 on each side, while Atlas 950DT uses DP4/TP8/PP1 on each side.

Refer to [PD Disaggregation with Mooncake](../features/pd_disaggregation_mooncake_multi_node.md) for the general service workflow.

On Atlas 800 A3 and Atlas 950DT, Prefill uses AICPU by default, so leave `HCCL_OP_EXPANSION_MODE` unset in the Prefill command. Decode uses AIV; explicitly set `HCCL_OP_EXPANSION_MODE=AIV` in the Decode command.

This deployment supports DSpark speculative decoding. Configure the same `Inferact/Kimi-K3-DSpark` draft-model path and `num_speculative_tokens` on both Prefill and Decode nodes. The validated configuration uses draft TP16 on A3 or draft TP8 on Atlas 950DT, greedy drafting, and seven speculative tokens. The seventh argument of the engine template is the tensor-parallel size: `16` for A3 and `8` for Atlas 950DT.

#### 5.2.1 Create the engine templates

=== "Prefill"

    ```shell
    KV_PORT=36000

    unset ftp_proxy FTP_PROXY
    unset https_proxy HTTPS_PROXY
    unset http_proxy HTTP_PROXY

    nic_name=<PREFILL_NIC_NAME>
    local_ip=<PREFILL_LOCAL_IP>

    export DRAFT_MODEL_PATH=<KIMI_K3_DSPARK_MODEL_PATH>
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
    export HCCL_BUFFSIZE=1024
    export HCCL_IF_IP=${local_ip}
    export HCCL_SOCKET_IFNAME=${nic_name}
    export ASCEND_ENABLE_USE_FABRIC_MEM=1
    export ASCEND_RT_VISIBLE_DEVICES=$1
    export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH
    export GLOO_SOCKET_IFNAME=${nic_name}
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export PYTHONHASHSEED=0

    SPECULATIVE_CONFIG="$(
      printf \
      '{"method":"dspark","model":"%s","num_speculative_tokens":7,"draft_tensor_parallel_size":'"$7"',"max_model_len":4096,"draft_sample_method":"greedy","enforce_eager":true}' \
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
        --additional-config '{"recompute_scheduler_enable":false,"enable_flashcomm1":true,"multistream_overlap_shared_expert":true}' \
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
                    "prefill": {
                        "dp_size": 4,
                        "tp_size": '"$7"'
                    },
                    "decode": {
                        "dp_size": 4,
                        "tp_size": '"$7"'
                   }
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
      '{"method":"dspark","model":"%s","num_speculative_tokens":7,"draft_tensor_parallel_size":'"$7"',"max_model_len":4096,"draft_sample_method":"greedy","enforce_eager":true}' \
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
        --additional-config '{"recompute_scheduler_enable":false,"multistream_overlap_shared_expert":true}' \
        --limit-mm-per-prompt '{"vision_chunk":2}' \
        --kv-transfer-config \
        '{
          "kv_connector": "MooncakeConnectorV1",
          "kv_role": "kv_consumer",
          "kv_port": "'"$KV_PORT"'",
          "kv_connector_extra_config": {
            "prefill": {
                "dp_size": 4,
                "tp_size": '"$7"'
            },
            "decode": {
                "dp_size": 4,
                "tp_size": '"$7"'
            }
          }
        }'
    ```

#### 5.2.2 Start the engines

Deploy `launch_online_dp.py` and the corresponding engine template on every node. Pass `0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15` as the template's first argument on A3 and `0,1,2,3,4,5,6,7` on Atlas 950DT.

=== "Atlas 800 A3"

    ```shell
    python launch_online_dp.py \
        --dp-size 4 \
        --tp-size 16 \
        --pp-size 1 \
        --dp-size-local 1 \
        --dp-rank-start <LOCAL_DP_RANK> \
        --dp-address <PD_MASTER_IP> \
        --dp-rpc-port <DP_RPC_PORT> \
        --vllm-start-port <VLLM_START_PORT>
    ```

=== "Atlas 950DT"

    ```shell
    python launch_online_dp.py \
        --dp-size 4 \
        --tp-size 8 \
        --pp-size 1 \
        --dp-size-local 1 \
        --dp-rank-start <LOCAL_DP_RANK> \
        --dp-address <PD_MASTER_IP> \
        --dp-rpc-port <DP_RPC_PORT> \
        --vllm-start-port <VLLM_START_PORT>
    ```

Use ranks `0` through `3` for each four-node side. Configure independent master addresses, RPC ports, and vLLM port ranges for the Prefill and Decode groups.

After the engines start, configure and start the load-balancing proxy as described in [PD Disaggregation with Mooncake](../features/pd_disaggregation_mooncake_multi_node.md#start-the-service).

Key PD settings:

| Setting                      | Value                        | Description                                             |
| ---------------------------- | ---------------------------- | ------------------------------------------------------- |
| Topology                     | 4P4D                         | Four Prefill and four Decode nodes.                     |
| `--dp-size`                  | `4`                          | Four DP ranks on each side.                             |
| `--tp-size`                  | `16` on A3; `8` on Atlas 950DT | Uses all devices in a node.                           |
| `--pp-size`                  | `1`                          | One pipeline stage per engine.                          |
| `--dp-size-local`            | `1`                          | One DP rank per node.                                   |
| `KV_PORT`                    | `36000` for P, `36200` for D | Separates producer and consumer KV traffic.             |
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

=== "Atlas 950DT"

    After an Atlas 950DT mixed or PD service is ready, send a multimodal request to the API endpoint:

    ```shell
    curl http://<NODE0_LOCAL_IP>:<SERVICE_PORT>/v1/chat/completions \
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

## 7 Accuracy Evaluation

Here is one accuracy evaluation method for Kimi K3.

| dataset | version | metric | mode | vllm-api-general-chat | note |
| ----- | ----- | ----- | ----- | ----- | ----- |
| GPQA | - | accuracy | gen | 92.42 | 8 Atlas 800 A3 (64GB × 16) |
| OCRBench | - | accuracy | gen | 0.88 | 8 Atlas 800 A3 (64GB × 16) |
| GPQA | - | accuracy | gen | 93.5 | 1 Atlas 950DT |
| OCRBench | - | accuracy | gen | 0.891 | 1 Atlas 950DT |

### Using AISBench

1. Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) for the environment setup and evaluation procedure.

2. Run AISBench against the Kimi K3 service and collect the generated result files. Keep the model and tokenizer revisions, chat template, sampling settings, dataset, and evaluator revision fixed when comparing results.

## 8 Performance Evaluation

### Using AISBench

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for the environment setup and evaluation procedure.

### Using vLLM Benchmark

Use `vllm bench serve` to measure the online serving performance of the Kimi K3 service. The following is a minimal example for eight 128k-input, 1k-output requests. Replace the service address, model path, dataset path, and result directory for the target environment.

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/benchmarking/) for more details.

```shell
export DATA_NUM=8
export CONCURRENCY=8

vllm bench serve \
  --base-url http://<SERVICE_HOST>:<SERVICE_PORT> \
  --endpoint /v1/completions \
  --model kimi-k3 \
  --tokenizer <KIMI_K3_MODEL_PATH> \
  --tokenizer-mode kimi_k3 \
  --trust-remote-code \
  --dataset-name custom \
  --dataset-path <DATASET_PATH> \
  --custom-output-len 1024 \
  --skip-chat-template \
  --num-prompts ${DATA_NUM} \
  --request-rate inf \
  --max-concurrency ${CONCURRENCY} \
  --ignore-eos \
  --temperature 0 \
  --seed 0 \
  --disable-shuffle \
  --save-result \
  --result-dir <RESULT_DIR>
```

## 9 Performance Tuning

Use the validated deployment values above as a baseline. Adjust `max-model-len`, `max-num-seqs`, `max-num-batched-tokens`, and `gpu-memory-utilization` together for the target workload.

Refer to the [performance tuning guide](../../developer_guide/performance_and_debug/optimization_and_tuning.md) and the [feature matrix](../../user_guide/support_matrix/feature_matrix.md) for additional guidance.

## 10 FAQ

For common environment, installation, and general parameter issues, refer to the [Public FAQ](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html).

- **Q: Which multimodal inputs are supported by the current Kimi K3 implementation?**
A: The current local processor accepts image inputs. Video inputs are not supported.
- **Q: Which server options are required for Kimi K3 reasoning and tool calling?**
A: Configure `--tokenizer-mode kimi_k3`, `--enable-auto-tool-choice`, `--reasoning-parser kimi_k3`, and `--tool-call-parser kimi_k3` together.
- **Q: How should TP size be selected?**
A: TP size must divide the checkpoint's attention-head count. It also affects KDA state layout and expert placement, so validate memory capacity and communication performance together.
- **Q: How is DSpark enabled in PD separation?**
A: Download `Inferact/Kimi-K3-DSpark`, set `DRAFT_MODEL_PATH` on both Prefill and Decode nodes, and pass the same `--speculative-config` to both. Prefill must retain `--enforce-eager`.
