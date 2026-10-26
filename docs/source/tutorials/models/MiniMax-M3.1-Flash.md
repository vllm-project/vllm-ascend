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
| MiniMax-M3.1-Flash-W8A8 | Atlas 800I A3, two-node PD disaggregation | [ModelScope](https://www.modelscope.cn/models/Eco-Tech/Minimax-M3.1-Flash-w8a8) |
| MiniMax-M3.1-Flash-MXFP8 | 950DT products, two-node PD disaggregation | [ModelScope](https://www.modelscope.cn/models/Eco-Tech/Minimax-M3.1-Flash-w8a8-mxfp8) |

Download the weight variant used by the serving commands to the same model directory on both nodes. It is recommended to place the model weight in a shared cache directory.

### 3.2 Verify Multi-node Communication

Prepare the business network and the NPU interconnect by following
[Verify inter-node connectivity](../../getting_started/installation.md#installation-multi-node-interconnect).

On each host, use `ip addr` or `ifconfig` to identify the host communication IP and the interface that owns it. Set `LOCAL_IP` to that IP and `NIC_NAME` to the interface name; for example, if the IP is assigned to `eth0`, set `NIC_NAME="eth0"`. The scripts use this interface for both `HCCL_SOCKET_IFNAME` and `GLOO_SOCKET_IFNAME`. See [multi-node host configuration](../../developer_guide/contribution/multi_node_test.md#2-test-without-kubernetes) for selecting mutually reachable host IPs.

## 4 Installation

### 4.1 A3 Docker Container

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

### 4.2 950DT products Docker Container

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

### 5.2 PD Disaggregation

PD disaggregation runs Prefill and Decode on separate hosts. Both platforms use DSpark speculative decoding with seven draft tokens and probabilistic draft sampling. Configure the host-specific values at the beginning of each Prefill and Decode script before starting the service.

Monitor the role logs while weights load and graph capture completes. Startup duration depends on storage, hardware, and the selected configuration; use continuing log progress and `Application startup complete` to judge readiness. If logs stop progressing or report an error, follow [Startup Troubleshooting](#102-model-loading-is-still-in-progress-or-startup-fails) before starting the proxy.

=== "A3 series (W8A8)"

    The following values match the configurations in the Prefill and Decode scripts below and summarize the differences between the two roles.

    | Parameter | Prefill | Decode |
    | --- | --- | --- |
    | DP × TP × PP | 1 × 4 × 4 | 4 × 4 × 1 |
    | PP layer partition | 15,15,15,15 | — |
    | max-num-seqs | 32 | 128 per replica |
    | Execution mode | Eager | FULL_DECODE_ONLY |
    | HCCL_BUFFSIZE (MB) | 512 | 2048 |

    **Prefill Node**

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
                "decode":{"dp_size":4,"tp_size":4,"pp_size":1}
            }
        }'
    ```

    Start the Prefill engine:

    ```bash
    nohup bash run_prefill.sh > prefill.log 2>&1 &
    ```

    **Decode Node**

    Save the following as `run_decode_rank.sh` on the Decode node. Each invocation
    starts one TP4 replica with a distinct DP rank, port range, and device group.

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
    export ASCEND_RT_VISIBLE_DEVICES="$((DP_RANK*4)),$((DP_RANK*4+1)),$((DP_RANK*4+2)),$((DP_RANK*4+3))"
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
        --data-parallel-size 4 \
        --data-parallel-size-local 1 \
        --data-parallel-rank "$DP_RANK" \
        --data-parallel-address "$LOCAL_IP" \
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
                "decode":{"dp_size":4,"tp_size":4,"pp_size":1}
            }
        }'
    ```

    Start all four Decode replicas:

    ```bash
    for rank in 0 1 2 3; do
        nohup bash run_decode_rank.sh "$rank" > "decode-$rank.log" 2>&1 &
    done
    ```

    **Load-balance Proxy**

    Download [load_balance_proxy_server_example.py](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py) from the official repository and save it as `/workspace/load_balance_proxy_server_example.py` inside the Prefill container.

    After the five engines print `Application startup complete`, run the following
    inside the Prefill container:

    ```bash
    PREFILL_IP="<PREFILL_NODE_IP>"
    DECODE_IP="<DECODE_NODE_IP>"

    python3 /workspace/load_balance_proxy_server_example.py \
        --host 0.0.0.0 --port 19080 \
        --prefiller-hosts "$PREFILL_IP" --prefiller-ports 19100 \
        --decoder-hosts "$DECODE_IP" "$DECODE_IP" "$DECODE_IP" "$DECODE_IP" \
        --decoder-ports 19200 19201 19202 19203
    ```

=== "950DT products"

    Use one Prefill node and one Decode node, each with all eight NPUs. Prefill uses DP1 × TP4 × PP2, and Decode uses DP2 × TP4 × PP1. Each node starts one vLLM API service.

    The following values match the configurations in the Prefill and Decode scripts below and summarize the differences between the two roles.

    | Parameter | Prefill | Decode |
    | --- | --- | --- |
    | DP × TP × PP | 1 × 4 × 2 | 2 × 4 × 1 |
    | max-num-seqs | 128 | 256 |
    | HCCL_BUFFSIZE (MB) | 512 | 2048 |
    | Main-model execution | Eager | FULL_DECODE_ONLY |

    **Prefill Node**

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
    MAX_BATCHED_TOKENS=32768
    GPU_MEMORY_UTILIZATION=0.92

    export VLLM_USE_V2_MODEL_RUNNER=1
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
                    "dp_size": 2,
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

    **Decode Node**

    Save the following as `run_decode.sh` on the Decode node. Configure the model path, host IP address, and network interface at the beginning of the script.

    ```bash
    #!/usr/bin/env bash
    set -e

    # Configure these values for the current host.
    MODEL_PATH="/models/MiniMax-M3.1-Flash-W8A8-MXFP8"
    LOCAL_IP="<CURRENT_NODE_IP>"
    NIC_NAME="<CURRENT_NODE_NIC>"
    SERVICE_PORT=31060
    DP_RPC_PORT=6885
    KV_PORT=26900
    MAX_NUM_SEQS=256
    MAX_MODEL_LEN=133000
    MAX_BATCHED_TOKENS=32768
    GPU_MEMORY_UTILIZATION=0.92

    export VLLM_USE_V2_MODEL_RUNNER=1
    export HCCL_BUFFSIZE=2048
    export HCCL_IF_IP="$LOCAL_IP"
    export HCCL_OP_EXPANSION_MODE="CCU_SCHED"
    export HCCL_SOCKET_IFNAME="$NIC_NAME"
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export GLOO_SOCKET_IFNAME="$NIC_NAME"
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

    vllm serve "$MODEL_PATH" \
        --host 0.0.0.0 \
        --port "$SERVICE_PORT" \
        --data-parallel-size 2 \
        --data-parallel-size-local 2 \
        --data-parallel-address "$LOCAL_IP" \
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
        --seed 1024 \
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
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
                    "dp_size": 2,
                    "tp_size": 4,
                    "pp_size": 1
                }
            }
        }'
    ```

    Start the Decode engine inside the Decode container:

    ```bash
    nohup bash run_decode.sh > decode.log 2>&1 &
    ```

    **Load-balance Proxy**

    Download [load_balance_proxy_server_example.py](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py) from the official repository and save it as `/workspace/load_balance_proxy_server_example.py` inside the Prefill container.

    After both engines print `Application startup complete`, run the following
    inside the Prefill container:

    ```bash
    PREFILL_IP="<PREFILL_NODE_IP>"
    DECODE_IP="<DECODE_NODE_IP>"

    python3 /workspace/load_balance_proxy_server_example.py \
        --host 0.0.0.0 --port 8009 \
        --prefiller-hosts "$PREFILL_IP" --prefiller-ports 31050 \
        --decoder-hosts "$DECODE_IP" --decoder-ports 31060
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

## 8 Performance Evaluation

Use [vLLM serving benchmarks](../../developer_guide/performance_and_debug/performance_benchmark.md#321-online-serving) to measure request throughput, output-token throughput, time to first token (TTFT), and time per output token (TPOT). Start the matching service in [Section 5](#5-online-service-deployment) and wait until it is ready before running the benchmark from a container with the model tokenizer available.

Set `SERVICE_HOST` to the reachable API or proxy IP. Use port `11223` for single-node deployment, `19080` for the A3 PD proxy, or `8009` for the 950DT products PD proxy. Set `MODEL_PATH` to the corresponding mounted weight directory from [Section 4](#4-installation); the following example uses the A3 path.

```bash
SERVICE_HOST="<SERVICE_IP>"
SERVICE_PORT=11223
MODEL_PATH="/models/MiniMax-M3.1-Flash-W8A8"
mkdir -p ./benchmark_results

vllm bench serve \
    --backend vllm \
    --base-url "http://${SERVICE_HOST}:${SERVICE_PORT}" \
    --endpoint /v1/completions \
    --model "$MODEL_PATH" \
    --served-model-name minimax-m3.1-flash \
    --trust-remote-code \
    --dataset-name random \
    --random-input-len 1024 \
    --random-output-len 128 \
    --num-prompts 32 \
    --max-concurrency 4 \
    --request-rate inf \
    --ignore-eos \
    --save-result \
    --result-dir ./benchmark_results
```

This is an initial benchmark workload, not a peak-performance configuration. A successful run completes all 32 requests with no failed requests and saves a result JSON file under `./benchmark_results`. The output has the following form; `<measured>` fields are placeholders for the values produced by your run, not MiniMax-M3.1-Flash performance measurements:

```text
============ Serving Benchmark Result ============
Successful requests:                     32
Failed requests:                         0
Benchmark duration (s):                  <measured>
Total input tokens:                      <measured>
Total generated tokens:                  <measured>
Request throughput (req/s):              <measured>
Output token throughput (tok/s):         <measured>
Peak output token throughput (tok/s):    <measured>
Peak concurrent requests:                <measured>
Total Token throughput (tok/s):          <measured>
---------------Time to First Token----------------
Mean TTFT (ms):                          <measured>
Median TTFT (ms):                        <measured>
P99 TTFT (ms):                           <measured>
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          <measured>
Median TPOT (ms):                        <measured>
P99 TPOT (ms):                           <measured>
---------------Inter-token Latency----------------
Mean ITL (ms):                           <measured>
Median ITL (ms):                         <measured>
P99 ITL (ms):                            <measured>
==================================================
```

Keep the hardware, image tag, weight variant, serving command, input/output lengths, request rate, and concurrency with each result. Increase the number of requests for sustained measurements and repeat the same workload when comparing configurations. Refer to the [benchmark CLI arguments](https://docs.vllm.ai/en/latest/cli/bench/serve/) for workload and result options.

## 9 Performance Tuning

### 9.1 Recommended Configurations

Use the commands in [Section 5](#5-online-service-deployment) as starting configurations. They are not globally optimal: the appropriate settings depend on request lengths, concurrency, prefix-cache hit rate, and the available device memory. Evaluate each adjustment with the same workload and compare latency, throughput, and memory use.

| Scenario | Starting Configuration | Parameters to Evaluate | Validation |
| --- | --- | --- | --- |
| Long context | A3 PD example with `MAX_MODEL_LEN=263000`; the other examples start at `133000`. | Set `MAX_MODEL_LEN` for the combined input/output budget and reduce `--max-num-seqs` if KV cache capacity is insufficient. | Confirm the requested context fits at the target concurrency before increasing load. |
| Low latency | Single-node examples in Section 5.1. | Reduce request concurrency and compare `--max-num-seqs` and `--max-num-batched-tokens` settings. | Compare TTFT and TPOT alongside output-token throughput. |
| High throughput | PD examples in Section 5.2. | Adjust `MAX_NUM_SEQS` and `MAX_BATCHED_TOKENS` on Prefill and Decode separately while keeping both connector topology declarations consistent. | Increase offered load gradually and check throughput, latency, and failed requests. |

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

The scripts in Section 5.2 already include this setting. It changes only the draft KV cache dtype.

### 10.2 Model Loading Is Still in Progress or Startup Fails

**Symptom:** The API is not ready while loading the approximately 428B-parameter checkpoint or capturing graphs.

**Cause:** Weight loading, initialization, and graph capture depend on storage speed, the hardware, and the selected configuration. Elapsed time alone does not distinguish ongoing initialization from a failure.

**Resolution:** Follow the foreground logs for single-node deployment. For PD deployment, use `tail -f prefill.log` on Prefill; use `tail -f decode-0.log decode-1.log decode-2.log decode-3.log` for A3 Decode, or `tail -f decode.log` for 950DT products Decode. Allow initialization to continue while logs show progress. Send requests or start the PD proxy after every required service prints `Application startup complete`; the single-node health check must return `200`.

If logs stop progressing or the process exits, inspect the first error in every affected role log. Check that `MODEL_PATH` points to the matching weight variant mounted in Section 4, confirm the expected devices are visible with `npu-smi info`, and compare the actual startup command with the corresponding platform example. For PD communication failures, also recheck `LOCAL_IP`, `NIC_NAME`, and [inter-node connectivity](#32-verify-multi-node-communication). For memory errors, review the context budget and concurrency described in [Section 5](#5-online-service-deployment) and the [out-of-memory FAQ](../../faqs.md#12-how-to-handle-the-out-of-memory-issue).
