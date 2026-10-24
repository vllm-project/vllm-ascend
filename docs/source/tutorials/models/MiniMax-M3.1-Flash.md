# MiniMax-M3.1-Flash

## 1 Introduction

!!! note

    The "PD Separation Deployment" mentioned in this document is provided only as an example/reference design to demonstrate the basic process of deploying Prefill and Decode in separated mode. It is not recommended for use in the production environment.

    **For the production environment on the customer side, [MindIE Motor](https://www.hiascend.com/document/detail/en/mindie/latest/quickstart/docs/en/user_guide/quick_start.md) + vLLM-Ascend is recommended.**

MiniMax-M3.1-Flash is a multimodal Mixture-of-Experts model with approximately 428B total parameters and 23B activated parameters per token. It uses sparse attention throughout the model.

This document covers supported features, environment and model preparation, single-node deployment, PD separation, and accuracy evaluation.

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
| MiniMax-M3.1-Flash-W8A8 | Atlas 800I A3, two-node PD disaggregation | [ModelScope](https://www.modelscope.cn/models/models/Eco-Tech/Minimax-M3.1-Flash-w8a8) |
| MiniMax-M3.1-Flash-MXFP8 | 950DT products, two-node PD disaggregation | [ModelScope](https://www.modelscope.cn/models/models/Eco-Tech/Minimax-M3.1-Flash-w8a8-mxfp8) |

Download the weight variant used by the serving commands to the same model directory on both nodes. It is recommended to place the model weight in a shared cache directory.

### 3.2 Verify Multi-node Communication

Prepare the business network and the NPU interconnect by following
[Verify inter-node connectivity](../../getting_started/installation.md#installation-multi-node-interconnect).

## 4 Installation

### 4.1 A3 Docker Container

Run the following on each A3 host. Set the model path to the W8A8 checkpoint.

```bash
export IMAGE="quay.io/ascend/vllm-ascend:minimax-m3.1-a3-openeuler"
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

### 4.2 A5 Docker Container

Run the following on each A5 host. Set the model path to the W8A8-MXFP8 checkpoint.

```bash
export IMAGE="quay.io/ascend/vllm-ascend:minimax-m3.1-a5-openeuler"
export CONTAINER_NAME=minimax_m31_flash
export MODEL_DIR="<MINIMAX_M31_FLASH_W8A8_PATH>"
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

**Note**: In the script above, `max-num-seqs` represents the maximum number of sequences the scheduler can process in a single iteration. Adjust the `max-num-seqs` parameter dynamically based on actual business.

For text-only deployment, `--limit-mm-per-prompt` can be omitted. For multimodal deployment, configure this parameter according to the actual request shape. For example, use `--limit-mm-per-prompt '{"image":2,"video":0}'` for two-image requests, and use `--limit-mm-per-prompt '{"image":0,"video":1}'` for one-video requests.

### 5.1 Single-Node Deployment

Single-node deployment runs Prefill and Decode on the same host. Both examples use W8A8 weights and DSpark speculative decoding, with all 16 NPUs on A3 or all eight NPUs on A5.

=== "A3 series (W8A8)"

    **Start the Service**

    Run the following inside the A3 container. It starts one API service with DP4 x TP4 / EP16 across all 16 NPUs, using W8A8 weights and DSpark speculative decoding with seven draft tokens.

    Set `MODEL_PATH` to your model weight directory; the following example uses the container mount path configured in [Section 4](#4-installation).

    ```bash
    export MODEL_PATH="/models/MiniMax-M3.1-Flash-W8A8"

    export VLLM_USE_V2_MODEL_RUNNER=1
    export HCCL_BUFFSIZE=1024
    export HCCL_OP_EXPANSION_MODE="AIV"
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

    vllm serve ${MODEL_PATH} \
      --host 0.0.0.0 \
      --port 11223 \
      --served-model-name minimax-m3.1-flash \
      --trust-remote-code \
      --max-model-len 133000 \
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

    Send requests directly to port `11223` with model name `minimax-m3.1-flash`.

=== "950DT products"

    **Start the Service**

    Run the following inside the A5 container. It uses DP2 x TP4 / EP8 across all eight NPUs, W8A8-MXFP8 weights, and DSpark speculative decoding with seven draft tokens.

    Set `MODEL_PATH` to your model weight directory; the following example uses the container mount path configured in [Section 4](#4-installation).

    ```bash
    export MODEL_PATH="/models/MiniMax-M3.1-Flash-W8A8-MXFP8"

    export VLLM_USE_V2_MODEL_RUNNER=1
    export HCCL_BUFFSIZE=1024
    export HCCL_OP_EXPANSION_MODE="AIV"
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

    vllm serve ${MODEL_PATH} \
      --host 0.0.0.0 \
      --port 11223 \
      --served-model-name minimax-m3.1-flash \
      --trust-remote-code \
      --max-model-len 133000 \
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

    Send requests directly to port `11223` with model name `minimax-m3.1-flash`.

### 5.2 PD Disaggregation

PD disaggregation runs Prefill and Decode on separate hosts. On both hosts, prepare the common environment from the matching platform tab below, using each host's own IP address and network interface. Use `/workspace/env.sh` for A3 or `/workspace/runtime_env.sh` for A5, then start the corresponding Prefill and Decode services.

=== "A3"

    Here, 1P1D means one Prefill host and one Decode host. The Decode host runs four
    TP4 replicas in one DP4 group.

    | Parameter | Prefill | Decode |
    | --- | --- | --- |
    | DP × TP × PP | 1 × 4 × 4 | 4 × 4 × 1 |
    | Visible NPU IDs | 0–15 | 0–3, 4–7, 8–11, 12–15 |
    | PP layer partition | 15,15,15,15 | — |
    | API ports | 19100 | 19200–19203 |
    | DP RPC port | — | 16920 |
    | KV base ports | 24000 | 25000 + 32 × DP rank |
    | max-model-len | 263000 | 263000 |
    | max-num-batched-tokens | 32768 | 32768 |
    | max-num-seqs | 32 | 128 per replica |
    | KV cache dtype | BF16 | BF16 |
    | Execution mode | Eager | FULL_DECODE_ONLY |

    **Common Environment**

    Save the following as `/workspace/env.sh` in the A3 container. Set `LOCAL_IP` and `NIC_NAME` to the current host's business IP address and network interface.

    ```bash
    #!/usr/bin/env bash
    unset http_proxy https_proxy ftp_proxy all_proxy HTTP_PROXY HTTPS_PROXY FTP_PROXY ALL_PROXY
    export NO_PROXY='*' no_proxy='*'

    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    source /usr/local/Ascend/nnal/atb/set_env.sh

    export LOCAL_IP="<CURRENT_NODE_IP>"
    export NIC_NAME="<CURRENT_NODE_NIC>"
    export MODEL_PATH="/models/MiniMax-M3.1-Flash-W8A8"

    export VLLM_USE_V2_MODEL_RUNNER=1
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    export VLLM_BATCH_INVARIANT=0
    export OMP_NUM_THREADS=1
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export VLLM_ENGINE_READY_TIMEOUT_S=3600
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
    export HCCL_CONNECT_TIMEOUT=600
    export HCCL_EXEC_TIMEOUT=600
    export HCCL_IF_IP="$LOCAL_IP"
    export VLLM_HOST_IP="$LOCAL_IP"
    export HCCL_SOCKET_IFNAME="$NIC_NAME"
    export GLOO_SOCKET_IFNAME="$NIC_NAME"
    export TP_SOCKET_IFNAME="$NIC_NAME"
    export HCCL_OP_EXPANSION_MODE=AIV
    unset HCCL_INTRA_ROCE_ENABLE
    export ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.protocol_desc":["hccs:device"]}'
    export ASCEND_CONNECT_TIMEOUT=30000
    export ASCEND_TRANSFER_TIMEOUT=30000

    export ASCEND_SOURCE=/opt/m31flash/code/vllm-ascend
    export OP_VENDOR="$ASCEND_SOURCE/vllm_ascend/_cann_ops_custom/vendors/custom_transformer"
    export ASCEND_CUSTOM_OPP_PATH="$OP_VENDOR"
    export LD_LIBRARY_PATH="$OP_VENDOR/op_api/lib:${LD_LIBRARY_PATH:-}"
    export LD_PRELOAD="/usr/lib64/libjemalloc.so.2${LD_PRELOAD:+:$LD_PRELOAD}"
    export PYTHONPATH="$ASCEND_SOURCE:/opt/m31flash/code/vllm:${PYTHONPATH:-}"
    ```

    **Prefill Node**

    Save the following as `run_prefill.sh` on the Prefill node.

    ```bash
    #!/usr/bin/env bash
    source ./env.sh

    # P: one PP4 x TP4 engine, all 16 chips.
    export TP_SIZE=4
    export PP_SIZE=4
    export DP_SIZE=1
    export SERVICE_PORT=19100
    export VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
    export ASCEND_RT_VISIBLE_DEVICES="$VISIBLE_DEVICES"
    export VLLM_PP_LAYER_PARTITION=15,15,15,15
    export HCCL_BUFFSIZE=1024
    export HCCL_NPU_SOCKET_PORT_RANGE=28000-28255
    export KV_PORT=24000
    export MAX_NUM_SEQS=${P_MAX_NUM_SEQS:-32}

    vllm serve "$MODEL_PATH" \
        --host 0.0.0.0 \
        --port "$SERVICE_PORT" \
        --served-model-name minimax-m31-flash \
        --trust-remote-code \
        --hf-overrides '{"architectures":["MiniMaxM3SparseForCausalLM"]}' \
        --language-model-only \
        --quantization ascend \
        --dtype bfloat16 \
        --tensor-parallel-size "$TP_SIZE" \
        --pipeline-parallel-size "$PP_SIZE" \
        --enable-expert-parallel \
        --distributed-executor-backend mp \
        --seed 1024 \
        --max-model-len "${MAX_MODEL_LEN:-263000}" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --max-num-batched-tokens "${MAX_BATCHED_TOKENS:-32768}" \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.92}" \
        --block-size 128 \
        --kv-cache-dtype bfloat16 \
        --enable-chunked-prefill \
        --enable-prefix-caching \
        --reasoning-parser minimax_m3 \
        --enable-auto-tool-choice \
        --tool-call-parser minimax_m3 \
        --data-parallel-size "$DP_SIZE" \
        --enforce-eager \
        --long-prefill-token-threshold 2048 \
        --additional-config '{
            "enable_cpu_binding":true,
            "multistream_overlap_shared_expert":true,
            "enable_shared_expert_dp":true,
            "weight_nz_mode":2,
            "ascend_compilation_config":{"fuse_norm_quant":false,"enable_static_kernel":false},
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
    source ./env.sh

    # D: this script starts ONE of four TP4 replicas. The host launcher starts 0..3.
    export DP_RANK=${1:-${DP_RANK:-0}}
    export TP_SIZE=4
    export PP_SIZE=1
    export DP_SIZE=4
    export DP_ADDRESS="$LOCAL_IP"
    export DP_RPC_PORT=16920
    export SERVICE_PORT=$((19200 + DP_RANK))
    export VISIBLE_DEVICES="$((DP_RANK*4)),$((DP_RANK*4+1)),$((DP_RANK*4+2)),$((DP_RANK*4+3))"
    export ASCEND_RT_VISIBLE_DEVICES="$VISIBLE_DEVICES"
    unset VLLM_PP_LAYER_PARTITION
    export HCCL_BUFFSIZE=2048
    export HCCL_NPU_SOCKET_PORT_RANGE="$((30000+DP_RANK*128))-$((30127+DP_RANK*128))"
    export KV_PORT=$((25000 + DP_RANK*32))
    export MAX_NUM_SEQS=${D_MAX_NUM_SEQS:-128}

    vllm serve "$MODEL_PATH" \
        --host 0.0.0.0 \
        --port "$SERVICE_PORT" \
        --served-model-name minimax-m31-flash \
        --trust-remote-code \
        --hf-overrides '{"architectures":["MiniMaxM3SparseForCausalLM"]}' \
        --language-model-only \
        --quantization ascend \
        --dtype bfloat16 \
        --tensor-parallel-size "$TP_SIZE" \
        --pipeline-parallel-size "$PP_SIZE" \
        --enable-expert-parallel \
        --distributed-executor-backend mp \
        --seed 1024 \
        --max-model-len "${MAX_MODEL_LEN:-263000}" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --max-num-batched-tokens "${MAX_BATCHED_TOKENS:-32768}" \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.92}" \
        --block-size 128 \
        --kv-cache-dtype bfloat16 \
        --enable-chunked-prefill \
        --enable-prefix-caching \
        --reasoning-parser minimax_m3 \
        --enable-auto-tool-choice \
        --tool-call-parser minimax_m3 \
        --data-parallel-size "$DP_SIZE" \
        --data-parallel-size-local 1 \
        --data-parallel-rank "$DP_RANK" \
        --data-parallel-address "$DP_ADDRESS" \
        --data-parallel-rpc-port "$DP_RPC_PORT" \
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
        --additional-config '{
            "enable_cpu_binding":true,
            "multistream_overlap_shared_expert":true,
            "enable_shared_expert_dp":true,
            "enable_flashcomm1":true,
            "weight_nz_mode":2,
            "ascend_compilation_config":{"fuse_norm_quant":false,"enable_static_kernel":false}
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

    After the five engines print `Application startup complete`, run the following
    inside the Prefill container:

    ```bash
    export PREFILL_IP="<PREFILL_NODE_IP>"
    export DECODE_IP="<DECODE_NODE_IP>"

    python3 "$ASCEND_SOURCE/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py" \
        --host 0.0.0.0 --port 19080 \
        --prefiller-hosts "$PREFILL_IP" --prefiller-ports 19100 \
        --decoder-hosts "$DECODE_IP" "$DECODE_IP" "$DECODE_IP" "$DECODE_IP" \
        --decoder-ports 19200 19201 19202 19203
    ```

=== "A5"

    Use one Prefill node and one Decode node, each with all eight NPUs and DP2 × TP4 × PP1 / EP8. Each node starts one vLLM API service, which manages both local DP ranks. Speculative decoding is disabled in this configuration.

    | Parameter | Prefill | Decode |
    | --- | --- | --- |
    | Hardware | Ascend 950DT9582, 8 NPUs | Ascend 950DT9582, 8 NPUs |
    | DP × TP × PP | 2 × 4 × 1 | 2 × 4 × 1 |
    | Expert parallel size | 8 | 8 |
    | Visible NPU IDs | 0–7 | 0–7 |
    | API port | 31050 | 31060 |
    | DP RPC port | 6884 | 6885 |
    | KV transfer port | 30000 | 26900 |
    | Weight quantization | MXFP8 | MXFP8 |
    | Compute / KV cache dtype | BF16 / FP8 | BF16 / FP8 |
    | max-model-len | 133000 | 133000 |
    | max-num-batched-tokens | 32768 | 32768 |
    | max-num-seqs | 128 | 256 |
    | HCCL_BUFFSIZE (MB) | 1024 | 2048 |
    | KV cache manager | `--disable-hybrid-kv-cache-manager` | `--disable-hybrid-kv-cache-manager` |
    | Main-model execution | Eager | FULL_DECODE_ONLY |
    | fuse_norm_quant | true | false |
    | fuse_qknorm_rope | true | true |
    | Speculative decoding | Disabled | Disabled |

    **Common Environment**

    A5 uses MXFP8 main-model weights and FP8 KV cache. The environment uses CANN 9.2, torch 2.10.0, and Triton/Triton-Ascend 3.6.0, with the Python environment at `/workspace/.venv`.

    Save the following as `/workspace/runtime_env.sh` in the A5 container and set `NIC_NAME` to the current host's network interface. The launch commands set `LOCAL_IP` before sourcing this file.

    ```bash
    #!/usr/bin/env bash
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    if [ -f /usr/local/Ascend/cann-9.2.0/share/info/ascendnpu-ir/bin/set_env.sh ]; then
        source /usr/local/Ascend/cann-9.2.0/share/info/ascendnpu-ir/bin/set_env.sh
    fi
    if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
        source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=1
    fi
    unset ftp_proxy https_proxy http_proxy all_proxy FTP_PROXY HTTPS_PROXY HTTP_PROXY ALL_PROXY
    export NIC_NAME="<CURRENT_NODE_NIC>"
    export PATH=/workspace/.venv/bin:/workspace/bin:$PATH
    export PYTHONPATH=/workspace/src/vllm:/workspace/src/vllm-ascend:${PYTHONPATH:-}
    export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/python3.12.13/lib/python3.12/site-packages/mooncake:${LD_LIBRARY_PATH:-}
    export VLLM_USE_V2_MODEL_RUNNER=1
    export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'
    export HCCL_IF_IP=$LOCAL_IP
    export HCCL_OP_EXPANSION_MODE=AIV
    export HCCL_SOCKET_IFNAME="$NIC_NAME"
    export GLOO_SOCKET_IFNAME="$NIC_NAME"
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    ```

    **Prefill Node**

    Save the following as `/workspace/prefill/run_dp_template.sh`.

    ```bash
    #!/usr/bin/env bash
    set -e
    source /workspace/runtime_env.sh
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export HCCL_BUFFSIZE=1024
    DP_RANK_ARGS=(--data-parallel-size-local "$DP_SIZE")
    mkdir -p /workspace/logs
    exec /workspace/.venv/bin/vllm serve "$MODEL_PATH" \
        --host 0.0.0.0 \
        --port "$SERVICE_PORT" \
        --data-parallel-size "$DP_SIZE" \
        "${DP_RANK_ARGS[@]}" \
        --data-parallel-address "$DP_ADDRESS" \
        --data-parallel-rpc-port "$DP_RPC_PORT" \
        --tensor-parallel-size "$TP_SIZE" \
        --pipeline-parallel-size "$PP_SIZE" \
        --served-model-name minimax-m3 \
        --trust-remote-code \
        --dtype bfloat16 \
        --max-num-seqs 128 \
        --max-num-batched-tokens 32768 \
        --max-model-len 133000 \
        --disable-hybrid-kv-cache-manager \
        --enable-expert-parallel \
        --quantization mxfp8 \
        --gpu-memory-utilization 0.92 \
        --distributed-executor-backend mp \
        --kv-cache-dtype fp8 \
        --reasoning-parser minimax_m3 \
        --enable-auto-tool-choice \
        --tool-call-parser minimax_m3 \
        --limit-mm-per-prompt '{"image":1,"video":1}' \
        --safetensors-load-strategy prefetch \
        --enforce-eager \
        --enable-chunked-prefill \
        --enable-prefix-caching \
        --additional-config '{"enable_cpu_binding":true,"ascend_compilation_config":{"fuse_norm_quant":true,"fuse_qknorm_rope":true},"multistream_overlap_shared_expert":true,"enable_shared_expert_dp":true,"enable_flashcomm1":true}' \
        --kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_producer","kv_port":"30000","kv_connector_extra_config":{"use_ascend_direct":true,"prefill":{"dp_size":2,"tp_size":4,"pp_size":1},"decode":{"dp_size":2,"tp_size":4,"pp_size":1}}}' > /workspace/logs/prefill-rank${DP_RANK}.log 2>&1
    ```

    Save the following as `/workspace/prefill/start.sh` and set `LOCAL_IP` to this node's compute-network IP. `MODEL_PATH` matches the A5 checkpoint mounted in the container.

    ```bash
    #!/usr/bin/env bash
    set -e
    cd /workspace/prefill
    export LOCAL_IP="<CURRENT_NODE_IP>"
    export DP_SIZE=2 TP_SIZE=4 PP_SIZE=1 DP_RANK=0
    export DP_ADDRESS="$LOCAL_IP" DP_RPC_PORT=6884 SERVICE_PORT=31050
    export MODEL_PATH="/models/MiniMax-M3.1-Flash-W8A8-MXFP8"
    exec bash /workspace/prefill/run_dp_template.sh
    ```

    Start it on the Prefill host:

    ```bash
    docker exec -d minimax_m31_flash bash /workspace/prefill/start.sh
    ```

    Check health after model loading completes:

    ```bash
    curl --noproxy '*' -sS -o /dev/null -w '%{http_code}\n' http://127.0.0.1:31050/health
    ```

    **Decode Node**

    Save the following as `/workspace/decode/run_dp_template.sh`.

    ```bash
    #!/usr/bin/env bash
    set -e
    source /workspace/runtime_env.sh
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export HCCL_BUFFSIZE=2048
    DP_RANK_ARGS=(--data-parallel-size-local "$DP_SIZE")
    mkdir -p /workspace/logs
    exec /workspace/.venv/bin/vllm serve "$MODEL_PATH" \
        --host 0.0.0.0 \
        --port "$SERVICE_PORT" \
        --data-parallel-size "$DP_SIZE" \
        "${DP_RANK_ARGS[@]}" \
        --data-parallel-address "$DP_ADDRESS" \
        --data-parallel-rpc-port "$DP_RPC_PORT" \
        --tensor-parallel-size "$TP_SIZE" \
        --pipeline-parallel-size "$PP_SIZE" \
        --served-model-name minimax-m3 \
        --trust-remote-code \
        --dtype bfloat16 \
        --max-num-seqs 256 \
        --max-num-batched-tokens 32768 \
        --max-model-len 133000 \
        --disable-hybrid-kv-cache-manager \
        --enable-expert-parallel \
        --quantization mxfp8 \
        --gpu-memory-utilization 0.92 \
        --distributed-executor-backend mp \
        --kv-cache-dtype fp8 \
        --reasoning-parser minimax_m3 \
        --enable-auto-tool-choice \
        --tool-call-parser minimax_m3 \
        --limit-mm-per-prompt '{"image":1,"video":1}' \
        --safetensors-load-strategy prefetch \
        --seed 1024 \
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
        --additional-config '{"enable_cpu_binding":true,"ascend_compilation_config":{"fuse_norm_quant":false,"fuse_qknorm_rope":true},"multistream_overlap_shared_expert":true,"enable_shared_expert_dp":true,"enable_flashcomm1":true}' \
        --kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_consumer","kv_port":"26900","kv_connector_extra_config":{"use_ascend_direct":true,"prefill":{"dp_size":2,"tp_size":4,"pp_size":1},"decode":{"dp_size":2,"tp_size":4,"pp_size":1}}}' > /workspace/logs/decode-rank${DP_RANK}.log 2>&1
    ```

    Save the following as `/workspace/decode/start.sh` and set `LOCAL_IP` to this node's compute-network IP. `MODEL_PATH` matches the A5 checkpoint mounted in the container.

    ```bash
    #!/usr/bin/env bash
    set -e
    cd /workspace/decode
    export LOCAL_IP="<CURRENT_NODE_IP>"
    export DP_SIZE=2 TP_SIZE=4 PP_SIZE=1 DP_RANK=0
    export DP_ADDRESS="$LOCAL_IP" DP_RPC_PORT=6885 SERVICE_PORT=31060
    export MODEL_PATH="/models/MiniMax-M3.1-Flash-W8A8-MXFP8"
    exec bash /workspace/decode/run_dp_template.sh
    ```

    Start it on the Decode host:

    ```bash
    docker exec -d minimax_m31_flash bash /workspace/decode/start.sh
    ```

    Check health after model loading completes:

    ```bash
    curl --noproxy '*' -sS -o /dev/null -w '%{http_code}\n' http://127.0.0.1:31060/health
    ```

    **Load-balance Proxy**

    After both health checks return `200`, save the following as `/workspace/start_proxy.sh` in the Prefill container and set the two compute-network IPs. Configure one API endpoint per role.

    ```bash
    #!/usr/bin/env bash
    set -e
    export PREFILL_IP="<PREFILL_NODE_IP>"
    export DECODE_IP="<DECODE_NODE_IP>"
    export LOCAL_IP="$PREFILL_IP"
    source /workspace/runtime_env.sh
    mkdir -p /workspace/logs
    exec /workspace/.venv/bin/python /workspace/src/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
        --host 0.0.0.0 --port 8009 \
        --prefiller-hosts "$PREFILL_IP" --prefiller-ports 31050 \
        --decoder-hosts "$DECODE_IP" --decoder-ports 31060 \
        > /workspace/logs/proxy.log 2>&1
    ```

    Start the proxy on the Prefill host:

    ```bash
    docker exec -d minimax_m31_flash bash /workspace/start_proxy.sh
    ```

    Service logs are `/workspace/logs/prefill-rank0.log`, `/workspace/logs/decode-rank0.log`, and `/workspace/logs/proxy.log`.

    **Send a Request**

    The A5 proxy listens on port `8009`; the model name is `minimax-m3`.

    ```bash
    curl --noproxy '*' "http://<PREFILL_NODE_IP>:8009/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d '{
            "model": "minimax-m3",
            "messages": [{"role": "user", "content": "Explain sparse attention."}],
            "temperature": 0,
            "max_tokens": 2048,
            "stream": false
        }'
    ```

    This section adapts the [original A5 launch scripts](https://github.com/CXY-Katrina/workspace/tree/main/minimax_m3/minimax_m3.1_flash) with speculative decoding removed.

## 6 Send a Request

For single-node deployment, send requests directly to port `11223` using the model name from the matching platform tab in [Single-Node Deployment](#51-single-node-deployment). The following example uses the A3 PD proxy on port `19080`. For the A5 PD proxy on port `8009`, use the request example in the A5 tab of [PD Disaggregation](#52-pd-disaggregation).

```bash
curl --noproxy '*' http://127.0.0.1:19080/v1/models

curl --noproxy '*' http://127.0.0.1:19080/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{
        "model": "minimax-m31-flash",
        "messages": [{"role": "user", "content": "Explain sparse attention."}],
        "temperature": 0,
        "max_tokens": 2048,
        "stream": false
    }'
```
