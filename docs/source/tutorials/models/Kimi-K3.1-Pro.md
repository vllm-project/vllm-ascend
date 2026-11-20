---
render_macros: false
---

# Kimi K3.1 Pro (Experimental)

This guide covers feature configuration, environment preparation, multi-node Prefill/Decode (PD) separation, functional verification, and accuracy and performance evaluation for Kimi K3.1 Pro. Deployment scenarios include basic PD separation, 1M long-context PD separation, and pooled PD separation.

This guide covers the validated Atlas 800I A3 and 950DT Products platforms. A2 and 950PR Products are not currently supported.

## 1 Introduction {#1-introduction}

!!! note

    The PD separation deployment (including the Proxy) in this document is provided only as an example/reference design to demonstrate the basic process of deploying Prefill and Decode separately. It is not recommended for use in production environments.

    **For the production environment on the customer side, [MindIE Motor](https://www.hiascend.com/document/detail/en/mindie/latest/quickstart/docs/en/user_guide/quick_start.md) + vLLM-Ascend is recommended.**

Kimi K3.1 Pro is a multimodal Mixture-of-Experts (MoE) model developed by Moonshot AI as an evolution of Kimi K3. It retains the hybrid attention and Latent MoE architecture and introduces DSA sparse attention. The model supports visual understanding and a one-million-token context, combining language understanding, reasoning, and tool calling for coding, long-document processing, and multi-step agent tasks.

This guide provides eight-node 4P4D basic and 1M configurations for Atlas 800I A3 and 950DT Products. Use the parallelism, cache types, and parsers specified by each platform's templates. [Section 5.3](#53-pooled-pd) provides the Atlas 800I A3 pooled reference and the 950DT Products pooled configuration.

This guide is based on **vLLM-Ascend 0.30.0**, which introduces experimental support for Kimi K3.1 Pro for evaluation and validation with the configurations provided here. Supported scenarios, performance, and configuration interfaces may evolve in later releases.

## 2 Supported Features {#2-supported-features}

Refer to [supported features](../../user_guide/support_matrix/supported_models.md#generative-models) for the model's supported feature matrix.

Refer to the [feature guide](../../user_guide/feature_guide/index.md) for feature configuration.

## 3 Prerequisites {#3-prerequisites}

### 3.1 Model Weights {#31-model-weights}

| Weight Version | Hardware Requirements | Download Links | Notes |
|---|---|---|---|
| Kimi K3.1 Pro target model (A3)<br>`Kimi-K3.1-Pro-w4a8` | PD separation:<br>8 Atlas 800I A3 servers, with 16 devices per node (64 GB × 16). | [ModelScope](https://www.modelscope.cn/models/Eco-Tech/Kimi-K3.1-Pro-w4a8) | Use the W4A8 quantized checkpoint for A3 with `--quantization ascend`. See [Section 5](#5-online-service-deployment) for the topology and loading options. |
| Kimi K3.1 Pro target model (950DT Products)<br>`Kimi-K3.1-Pro-w4a8-mxfp` | PD separation:<br>8 950DT servers, with 8 devices per node. | [ModelScope](https://www.modelscope.cn/models/Eco-Tech/Kimi-K3.1-Pro-w4a8-mxfp) | Use the W4A8_MXFP quantized checkpoint for 950DT Products with `--quantization ascend`. See [Section 5](#5-online-service-deployment) for the topology and loading options. |

The target checkpoint directory must contain the model configuration, tokenizer, image processor, and model weight files required by the Kimi K3.1 Pro release. DSpark is a speculative decoding method: a draft model proposes candidate tokens, which the target model then verifies. For DSpark in PD separation or pooled PD deployment, prepare a draft checkpoint compatible with the platform and target model and specify its local directory.

For each role, `draft_tensor_parallel_size` must match the target model's `--tensor-parallel-size`; Prefill and Decode may use different TP sizes.

**Paths:** Download the target checkpoint for your platform from the table above. The commands below use example local directories: `/root/.cache/Kimi-K3.1-Pro-w4a8` on A3 and `/root/.cache/Kimi-K3.1-Pro-w4a8-mxfp` on 950DT Products. Download the complete checkpoint into the corresponding directory; if you use another directory, update the `vllm serve` path in every template for that platform. `<YOUR_DRAFT_MODEL_PATH>` still denotes the local directory of the compatible DSpark draft checkpoint. Use the same paths on every node and mount the weight directories inside every container. Weight directory names are independent of the API service name; keep `kimi-k3.1-pro` for both the served model name and the request `model` field.

### 3.2 Verify Multi-node Communication {#32-verify-multi-node-communication}

Before multi-node deployment, follow the [multi-node communication verification guide](../../getting_started/installation.md#installation-multi-node) to check communication within and between nodes.

## 4 Installation {#4-installation}

### 4.1 Docker Image Installation {#41-docker-image-installation}

Select an image for the deployment platform and host operating system, and start a container on every node. Refer to the [installation guide](../../getting_started/installation.md) for additional container options.

=== "A3 series"

    Kimi K3.1 Pro is validated on Atlas 800I A3 (64 GB × 16).

    Select the complete command for the host operating system and run it on every node. If the model directory is outside `/root/.cache/`, update the mount path accordingly.

    === "Ubuntu"

        ```bash
        export IMAGE="quay.io/ascend/vllm-ascend:kimi-k3.1-pro-a3"
        docker run --rm \
            --name kimi-k31 \
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
            -it "$IMAGE" bash
        ```

    === "openEuler"

        ```bash
        export IMAGE="quay.io/ascend/vllm-ascend:kimi-k3.1-pro-a3-openeuler"
        docker run --rm \
            --name kimi-k31 \
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
            -it "$IMAGE" bash
        ```

=== "950DT Products"

    Select the complete command for the host operating system and run it on every node. If the model directory is outside `/root/.cache/`, update the mount path accordingly.

    === "Ubuntu"

        ```bash
        export IMAGE="quay.io/ascend/vllm-ascend:kimi-k3.1-pro-a5"
        docker run --rm \
            --name kimi-k31 \
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
            --device /dev/hisi_hdc \
            -v /usr/local/dcmi:/usr/local/dcmi \
            -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
            -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
            -v /etc/ascend_install.info:/etc/ascend_install.info \
            -v /root/.cache:/root/.cache \
            -it "$IMAGE" bash
        ```

    === "openEuler"

        ```bash
        export IMAGE="quay.io/ascend/vllm-ascend:kimi-k3.1-pro-a5-openeuler"
        docker run --rm \
            --name kimi-k31 \
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
            --device /dev/hisi_hdc \
            -v /usr/local/dcmi:/usr/local/dcmi \
            -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
            -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
            -v /etc/ascend_install.info:/etc/ascend_install.info \
            -v /root/.cache:/root/.cache \
            -it "$IMAGE" bash
        ```

After starting the container, run `docker ps` in another terminal and confirm that the container specified by `--name` is listed with status `Up`. The commands above use `--rm`, which removes the container when it exits; if startup fails, inspect the output in the launch terminal first. To retain the exit status and logs, rerun without `--rm`, then inspect the container with `docker ps -a` and `docker logs <CONTAINER_NAME>`.

The containers above use privileged mode, host networking, and writable host mounts, and are intended for a controlled deployment environment. Services listen on `0.0.0.0`, and the examples do not configure access authentication; restrict network access to Proxy, engine, and cache service ports. Save scripts, configuration, and logs that must persist in explicitly mounted directories. `--rm` removes files in the container writable layer but does not delete host files in bind-mounted directories.

### 4.2 Source Code Installation {#42-source-code-installation}

If you do not use the Docker images above, you can build from source. Follow the [installation guide](../../getting_started/installation.md) to install vLLM-Ascend and prepare the same runtime environment on every deployment node.

Kimi K3.1 Pro model configuration, multimodal processing, reasoning parsing, and tool-call parsing are registered by vLLM-Ascend. Use vLLM and vLLM-Ascend source revisions compatible with **vLLM-Ascend 0.30.0**, the version used in this guide.

## 5 Online Service Deployment {#5-online-service-deployment}

This section covers basic PD separation, 1M long-context PD separation, and pooled PD separation. Enable `VLLM_USE_V2_MODEL_RUNNER=1` on every node, use `kimi-k3.1-pro` as the served model name, and fill in model paths as described in [Section 3.1](#31-model-weights). The A3 and 950DT Products basic and 1M configurations follow their respective deployment scripts; the A3 pooled reference and 950DT Products pooled configurations are listed in [Section 5.3](#53-pooled-pd).

If startup or a request fails, inspect the relevant engine and Proxy logs, then consult the [public FAQ](../../faqs.md) for environment, installation, and common parameter issues.

MegaMoE refers to the fused MoE computation path controlled by `enable_fused_mc2`. Use the setting specified in each deployment template.

### 5.1 Basic Multi-node PD Separation Deployment {#51-pd-deployment}

#### 5.1.1 Create the Engine Templates {#511-create-the-engine-templates}

=== "A3 series"

    A3 uses eight nodes in a 4P4D deployment: **PP4 / PCP4 / TP4** on Prefill, with layer partition `25,25,24,19`, and **DP4 / DCP16 / TP16** on Decode. Only P0 exposes the Prefill API; each of the four Decode nodes exposes its own API.

    Both roles use Model Runner V2 and MooncakeConnectorV2, with a maximum model length of `204800`. Prefill uses eager execution and a GPU memory utilization setting of `0.90`, enables PCP4, and disables MegaMoE. Decode uses `FULL_DECODE_ONLY` and a GPU memory utilization setting of `0.80`, with shared-expert multistream overlap, scheduler recomputation, and MLAPO enabled. Shared-expert DP is disabled on both roles.

    Leave `HCCL_OP_EXPANSION_MODE` unset on Prefill and set it to `AIV` on Decode. Both roles use `HCCL_BUFFSIZE=1024`. The main attention cache and indexer K cache use INT8, with a block size of `768`; Decode also uses an interleave size of `768`. The Prefill topology in the connector configuration on both roles is TP4 / PCP4 / PP4.

    DSpark3 is optional. Draft TP is 4 on Prefill and 16 on Decode and must match the target TP on each role; both roles use three speculative tokens. Refer to the [Mooncake PD disaggregation guide](../../tutorials/features/pd_disaggregation_mooncake_multi_node.md) for the general service workflow.

    MooncakeConnectorV2 exchanges actual layer metadata through its handshake and does not read `pp_layer_partition` from `kv_connector_extra_config`. Keep `VLLM_PP_LAYER_PARTITION=25,25,24,19` on every Prefill node for model layer partitioning; leave the environment variable unset on Decode. This applies to the basic, 1M, and pooled templates.



    To enable DSpark3, replace `<YOUR_DRAFT_MODEL_PATH>` with the local directory of a draft model compatible with the A3 target model. Use the same draft checkpoint on both roles and the same directory path on every node.

    === "Prefill"

        Save the Prefill template below as `run_dp_template.sh`. Fill in the local IP, NIC, model paths, KV/HCCL ports, and the top-level `API_PORT`, `NODE_RANK`, `MASTER_ADDR`, and `MASTER_PORT` variables, then start it directly with Bash. The script automatically sets `--headless` on P1–P3.

        ```bash
        #!/usr/bin/env bash
        set -euo pipefail

        API_PORT="<PREFILL_API_PORT>"
        NODE_RANK="<PREFILL_NODE_RANK>"
        MASTER_ADDR="<PREFILL_MASTER_IP>"
        MASTER_PORT="<PREFILL_MASTER_PORT>"
        headless_args=()
        if (( NODE_RANK > 0 )); then
            headless_args=(--headless)
        fi

        KV_PORT='<PREFILL_KV_PORT>'
        nic_name='<PREFILL_NIC_NAME>'
        local_ip='<PREFILL_LOCAL_IP>'

        unset ftp_proxy FTP_PROXY http_proxy HTTP_PROXY https_proxy HTTPS_PROXY
        unset LD_PRELOAD HCCL_OP_EXPANSION_MODE VLLM_PP_LAYER_PARTITION

        export HCCL_IF_IP="$local_ip"
        export HCCL_SOCKET_IFNAME="$nic_name"
        export GLOO_SOCKET_IFNAME="$nic_name"
        export TP_SOCKET_IFNAME="$nic_name"
        export HCCL_BUFFSIZE=1024
        export HCCL_CONNECT_TIMEOUT=1200
        export HCCL_IF_BASE_PORT='<PREFILL_HCCL_BASE_PORT>'
        export HCCL_HOST_SOCKET_PORT_RANGE='<PREFILL_HCCL_PORT_RANGE>'
        export HCCL_NPU_SOCKET_PORT_RANGE=auto
        export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
        export LD_LIBRARY_PATH="/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:${LD_LIBRARY_PATH:-}"
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export PYTHONHASHSEED=0
        export OMP_NUM_THREADS=1
        export OMP_PROC_BIND=false
        export VLLM_USE_V2_MODEL_RUNNER=1
        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
        export VLLM_HOST_IP="$local_ip"
        export VLLM_PP_LAYER_PARTITION="25,25,24,19"
        export FLA_STABLE_DEBUG_DESC=0

        vllm serve '/root/.cache/Kimi-K3.1-Pro-w4a8' \
            --host 0.0.0.0 \
            --port "$API_PORT" \
            --quantization ascend \
            --enable-auto-tool-choice \
            --reasoning-parser kimi_k3 \
            --tool-call-parser kimi_k3 \
            --tokenizer-mode kimi_k3 \
            --data-parallel-size "1" \
            --pipeline-parallel-size 4 \
            --prefill-context-parallel-size 4 \
            --decode-context-parallel-size 1 \
            --nnodes 4 \
            --node-rank "$NODE_RANK" \
            "${headless_args[@]}" \
            --distributed-executor-backend mp \
            --data-parallel-backend mp \
            --master-addr "$MASTER_ADDR" \
            --master-port "$MASTER_PORT" \
            --tensor-parallel-size "4" \
            --enable-expert-parallel \
            --seed 1024 \
            --served-model-name kimi-k3.1-pro \
            --max-model-len 204800 \
            --max-num-batched-tokens 8192 \
            --max-num-seqs 12 \
            --enforce-eager \
            --trust-remote-code \
            --gpu-memory-utilization 0.90 \
            --block-size 768 \
            --kv-cache-dtype int8 \
            --attention_config.indexer_kv_dtype int8 \
            --mm-encoder-tp-mode data \
            --no-mm-device-do-normalize \
            --mm-processor-cache-gb 1 \
            --skip-mm-profiling \
            --enable-prefix-caching \
            --additional-config '{"enable_flashcomm1":false,"multistream_overlap_shared_expert":false,"enable_fused_mc2":0,"enable_shared_expert_dp":false}' \
            --limit-mm-per-prompt '{"vision_chunk":2}' \
            --kv-transfer-config \
            '{
              "kv_connector": "MooncakeConnectorV2",
              "kv_role": "kv_producer",
              "kv_port": "'"$KV_PORT"'",
              "kv_connector_extra_config": {
                "prefill": {
                  "dp_size": 1,
                  "tp_size": 4,
                  "pp_size": 4,
                  "pcp_size": 4,
                  "dcp_size": 1
                },
                "decode": {
                  "dp_size": 4,
                  "tp_size": 16,
                  "pp_size": 1,
                  "pcp_size": 1,
                  "dcp_size": 16
                }
              }
            }'
        ```

        **Optional: Enable DSpark3**

        Add the following option to the `vllm serve` command above, retaining a trailing backslash if it is inserted before the last line. Enable it on both roles, using the same draft checkpoint and three speculative tokens.

        ```bash
        --speculative-config '{"method":"dspark","model":"<YOUR_DRAFT_MODEL_PATH>","num_speculative_tokens":3,"draft_tensor_parallel_size":4,"draft_sample_method":"greedy","kv_cache_dtype":"bfloat16","enforce_eager":true}'
        ```

    === "Decode"

        Save the Decode template below as `run_dp_template.sh` on each Decode node. Fill in the local IP, network interface, and model path. Fill in the KV port and the HCCL base port and port range for your environment (range format: `start-end`), avoiding port conflicts.

        ```bash
        #!/usr/bin/env bash
        set -euo pipefail

        KV_PORT='<DECODE_KV_PORT>'
        nic_name='<DECODE_NIC_NAME>'
        local_ip='<DECODE_LOCAL_IP>'

        unset ftp_proxy FTP_PROXY http_proxy HTTP_PROXY https_proxy HTTPS_PROXY
        unset LD_PRELOAD VLLM_PP_LAYER_PARTITION

        export HCCL_IF_IP="$local_ip"
        export HCCL_SOCKET_IFNAME="$nic_name"
        export GLOO_SOCKET_IFNAME="$nic_name"
        export TP_SOCKET_IFNAME="$nic_name"
        export HCCL_OP_EXPANSION_MODE=AIV
        export HCCL_BUFFSIZE=1024
        export HCCL_CONNECT_TIMEOUT=1200
        export HCCL_IF_BASE_PORT='<DECODE_HCCL_BASE_PORT>'
        export HCCL_HOST_SOCKET_PORT_RANGE='<DECODE_HCCL_PORT_RANGE>'
        export HCCL_NPU_SOCKET_PORT_RANGE=auto
        export ASCEND_RT_VISIBLE_DEVICES="$1"
        export LD_LIBRARY_PATH="/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:${LD_LIBRARY_PATH:-}"
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export PYTHONHASHSEED=0
        export OMP_NUM_THREADS=1
        export OMP_PROC_BIND=false
        export VLLM_USE_V2_MODEL_RUNNER=1
        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
        export VLLM_HOST_IP="$local_ip"
        export FLA_STABLE_DEBUG_DESC=0

        vllm serve '/root/.cache/Kimi-K3.1-Pro-w4a8' \
            --host 0.0.0.0 \
            --port "$2" \
            --quantization ascend \
            --enable-auto-tool-choice \
            --reasoning-parser kimi_k3 \
            --tool-call-parser kimi_k3 \
            --tokenizer-mode kimi_k3 \
            --data-parallel-size "$3" \
            --data-parallel-size-local 1 \
            --data-parallel-rank "$4" \
            --data-parallel-address "$5" \
            --data-parallel-rpc-port "$6" \
            --distributed-executor-backend mp \
            --tensor-parallel-size "$7" \
            --pipeline-parallel-size 1 \
            --prefill-context-parallel-size 1 \
            --decode-context-parallel-size 16 \
            --cp-kv-cache-interleave-size 768 \
            --block-size 768 \
            --enable-expert-parallel \
            --seed 1024 \
            --served-model-name kimi-k3.1-pro \
            --max-model-len 204800 \
            --max-num-batched-tokens 8192 \
            --max-num-seqs 12 \
            --trust-remote-code \
            --gpu-memory-utilization 0.80 \
            --kv-cache-dtype int8 \
            --attention_config.indexer_kv_dtype int8 \
            --mm-encoder-tp-mode data \
            --no-mm-device-do-normalize \
            --mm-processor-cache-gb 1 \
            --skip-mm-profiling \
            --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
            --enable-prefix-caching \
            --additional-config '{"enable_fused_mc2":0,"multistream_overlap_shared_expert":true,"enable_shared_expert_dp":false,"enable_mlapo":true,"scheduler_config":{"recompute_scheduler_enable":true}}' \
            --limit-mm-per-prompt '{"vision_chunk":2}' \
            --kv-transfer-config \
            '{
              "kv_connector": "MooncakeConnectorV2",
              "kv_role": "kv_consumer",
              "kv_port": "'"$KV_PORT"'",
              "kv_connector_extra_config": {
                "prefill": {
                  "dp_size": 1,
                  "tp_size": 4,
                  "pp_size": 4,
                  "pcp_size": 4,
                  "dcp_size": 1
                },
                "decode": {
                  "dp_size": 4,
                  "tp_size": 16,
                  "pp_size": 1,
                  "pcp_size": 1,
                  "dcp_size": 16
                }
              }
            }'
        ```

        **Optional: Enable DSpark3**

        Add the following option to the `vllm serve` command above, retaining a trailing backslash if it is inserted before the last line. Enable it on both roles, using the same draft checkpoint and three speculative tokens.

        ```bash
        --speculative-config '{"method":"dspark","model":"<YOUR_DRAFT_MODEL_PATH>","num_speculative_tokens":3,"draft_tensor_parallel_size":16,"draft_sample_method":"greedy","kv_cache_dtype":"bfloat16","enforce_eager":false}'
        ```

=== "950DT Products"

    950DT Products use eight nodes in a 4P4D topology: DP1 / PP4 / PCP8 / TP1 on Prefill, with layer partition `24,24,24,21`, and DP4 / TP8 / DCP8 on Decode. Only P0 exposes a Prefill API; P1–P3 use `--headless`, and all four Decode nodes expose APIs.

    Both roles use `--max-model-len 140000` in the basic configuration. The target model uses W4A8_MXFP weights with `--quantization ascend`; the main attention and indexer K caches use FP8. Both roles enable Model Runner V2 and CCU communication (`HCCL_OP_EXPANSION_MODE=CCU_SCHED`), with `HCCL_BUFFSIZE` set to `2000/800` on Prefill/Decode.

    Prefill uses eager execution, memory utilization `0.90`, a batched-token limit of `16384`, and a sequence limit of `64`, with `enable_flashcomm1=true`, `enable_shared_expert_dp=true`, and MegaMoE disabled. Decode uses `FULL_DECODE_ONLY`, memory utilization `0.92`, a batched-token limit of `768`, and a sequence limit of `64`, with MLAPO, shared-expert multistream overlap, and scheduler recomputation enabled. Both roles use block size and interleave size `1536`.

    Prefill retains `enable_flashcomm1=true` in this section. The current TP1 configuration does not meet the TP > 1 requirement for generic MoE sequence parallelism or shared-expert DP, so the configured values do not imply that these paths are active; check startup logs for the matching version. All 950DT Products templates configure MLAPO through `--additional-config`: `enable_mlapo=false` on Prefill and `enable_mlapo=true` on Decode.

    The 950DT Products basic and 1M templates enable DSpark3 with the same draft checkpoint on both roles, draft TP `1/8` on Prefill/Decode matching the target TP on each role, three speculative tokens, draft `enforce_eager=false`, and `kv_cache_dtype=auto`. The Prefill target model retains `--enforce-eager`. Replace `<YOUR_DRAFT_MODEL_PATH>` with the local directory of a draft checkpoint compatible with the target model on this platform.

    The templates use `--tokenizer-mode kimi_k3`. For reasoning parsing and tool calling, also configure `--reasoning-parser kimi_k3`, `--tool-call-parser kimi_k3`, and `--enable-auto-tool-choice` together as described in [Section 10](#10-faq).

    Fill in a unique `<PREFILL_ENGINE_ID>` or `<DECODE_ENGINE_ID>` for each engine. Use distinct node ranks for P0–P3; the template automatically adds `--headless` on P1–P3.

    === "Prefill"

        Save the Prefill template below as `run_dp_template.sh`. Fill in the local IP, NIC, model paths, KV/HCCL ports, and the top-level `API_PORT`, `NODE_RANK`, `MASTER_ADDR`, and `MASTER_PORT` variables, then start it directly with Bash. The script automatically sets `--headless` on P1–P3.

        ```bash
        #!/usr/bin/env bash
        set -e

        API_PORT="<PREFILL_API_PORT>"
        NODE_RANK="<PREFILL_NODE_RANK>"
        MASTER_ADDR="<PREFILL_MASTER_IP>"
        MASTER_PORT="<PREFILL_MASTER_PORT>"

        unset ftp_proxy FTP_PROXY
        unset https_proxy HTTPS_PROXY
        unset http_proxy HTTP_PROXY

        nic_name="<PREFILL_NIC_NAME>"
        local_ip="<PREFILL_LOCAL_IP>"
        KV_PORT="<PREFILL_KV_PORT>"
        ENGINE_ID="<PREFILL_ENGINE_ID>"
        DRAFT_MODEL_PATH="<YOUR_DRAFT_MODEL_PATH>"

        headless_args=()
        if (( NODE_RANK > 0 )); then
            headless_args=(--headless)
        fi

        export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        export HCCL_IF_IP="$local_ip"
        export VLLM_HOST_IP="$local_ip"
        export GLOO_SOCKET_IFNAME="$nic_name"
        export TP_SOCKET_IFNAME="$nic_name"
        export HCCL_SOCKET_IFNAME="$nic_name"
        export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'
        export VLLM_RPC_TIMEOUT=3600000
        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
        export HCCL_EXEC_TIMEOUT=2040
        export HCCL_CONNECT_TIMEOUT=1200
        export HCCL_BUFFSIZE=2000
        export OMP_PROC_BIND=false
        export OMP_NUM_THREADS=10
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export TASK_QUEUE_ENABLE=1

        SPECULATIVE_CONFIG="$(
          printf \
          '{"method":"dspark","model":"%s","num_speculative_tokens":3,"draft_tensor_parallel_size":1,"draft_sample_method":"greedy","enforce_eager":false,"kv_cache_dtype":"auto"}' \
          "$DRAFT_MODEL_PATH"
        )"

        export HCCL_OP_EXPANSION_MODE="CCU_SCHED"
        export VLLM_USE_V2_MODEL_RUNNER=1
        export VLLM_PP_LAYER_PARTITION=24,24,24,21

        vllm serve "/root/.cache/Kimi-K3.1-Pro-w4a8-mxfp" \
            --host 0.0.0.0 \
            --port "$API_PORT" \
            --data-parallel-size "1" \
            --tensor-parallel-size "1" \
            --prefill-context-parallel-size 8 \
            --nnodes 4 \
            --node-rank "$NODE_RANK" \
            --master-addr "$MASTER_ADDR" \
            --master-port "$MASTER_PORT" \
            --pipeline-parallel-size 4 \
            "${headless_args[@]}" \
            --block-size 1536 \
            --cp-kv-cache-interleave-size 1536 \
            --served-model-name kimi-k3.1-pro \
            --allowed-local-media-path / \
            --trust-remote-code \
            --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":128}' \
            --enable-expert-parallel \
            --max-num-seqs 64 \
            --max-model-len 140000 \
            --max-num-batched-tokens 16384 \
            --gpu-memory-utilization 0.90 \
            --enable-prefix-caching \
            --kv-cache-dtype fp8 \
            --quantization ascend \
            --attention_config.indexer_kv_dtype fp8 \
            --tokenizer-mode kimi_k3 \
            --mm-processor-cache-gb 0 \
            --mm-encoder-tp-mode data \
            --no-mm-device-do-normalize \
            --enforce-eager \
            --additional-config '{"enable_mlapo":false,"enable_flashcomm1":true,"enable_shared_expert_dp":true,"multistream_overlap_shared_expert":false,"enable_fused_mc2":0,"enable_force_eplb":false}' \
            --profiler-config '{"profiler":"torch","torch_profiler_dir":"<PROFILER_OUTPUT_DIR>","torch_profiler_with_stack":false}' \
            --kv-transfer-config '{
          "kv_connector": "MooncakeConnectorV2",
          "kv_role": "kv_producer",
          "kv_port": "'"$KV_PORT"'",
          "engine_id": "'"$ENGINE_ID"'",
          "kv_connector_extra_config": {
            "prefill": {
              "pp_size": 4,
              "pcp_size": 8,
              "dp_size": 1,
              "tp_size": 1
            },
            "decode": {
              "dp_size": 4,
              "tp_size": 8
            }
          }
        }' \
            --speculative-config "$SPECULATIVE_CONFIG"
        ```

    === "Decode"

        Save the Decode template below as `run_dp_template.sh` on the corresponding nodes. Fill in the local IP, network interface, KV port, and target and draft model paths, and replace `<PROFILER_OUTPUT_DIR>` with a writable profiling output directory.

        ```bash
        #!/usr/bin/env bash
        set -e

        unset ftp_proxy FTP_PROXY
        unset https_proxy HTTPS_PROXY
        unset http_proxy HTTP_PROXY

        nic_name="<DECODE_NIC_NAME>"
        local_ip="<DECODE_LOCAL_IP>"
        KV_PORT="<DECODE_KV_PORT>"
        ENGINE_ID="<DECODE_ENGINE_ID>"
        DRAFT_MODEL_PATH="<YOUR_DRAFT_MODEL_PATH>"

        export ASCEND_RT_VISIBLE_DEVICES="$1"
        export HCCL_IF_IP="$local_ip"
        export VLLM_HOST_IP="$local_ip"
        export GLOO_SOCKET_IFNAME="$nic_name"
        export TP_SOCKET_IFNAME="$nic_name"
        export HCCL_SOCKET_IFNAME="$nic_name"
        export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'
        export VLLM_RPC_TIMEOUT=3600000
        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
        export HCCL_EXEC_TIMEOUT=2040
        export HCCL_CONNECT_TIMEOUT=1200
        export HCCL_BUFFSIZE=800
        export OMP_PROC_BIND=false
        export OMP_NUM_THREADS=10
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export TASK_QUEUE_ENABLE=1

        SPECULATIVE_CONFIG="$(
          printf \
          '{"method":"dspark","model":"%s","num_speculative_tokens":3,"draft_tensor_parallel_size":8,"draft_sample_method":"greedy","enforce_eager":false,"kv_cache_dtype":"auto"}' \
          "$DRAFT_MODEL_PATH"
        )"

        export HCCL_OP_EXPANSION_MODE="CCU_SCHED"
        export VLLM_USE_V2_MODEL_RUNNER=1

        vllm serve "/root/.cache/Kimi-K3.1-Pro-w4a8-mxfp" \
            --host 0.0.0.0 \
            --port "$2" \
            --data-parallel-size "$3" \
            --data-parallel-rank "$4" \
            --data-parallel-address "$5" \
            --data-parallel-rpc-port "$6" \
            --tensor-parallel-size "$7" \
            --decode-context-parallel-size 8 \
            --block-size 1536 \
            --cp-kv-cache-interleave-size 1536 \
            --served-model-name kimi-k3.1-pro \
            --allowed-local-media-path / \
            --trust-remote-code \
            --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":128}' \
            --enable-expert-parallel \
            --max-num-seqs 64 \
            --max-model-len 140000 \
            --max-num-batched-tokens 768 \
            --gpu-memory-utilization 0.92 \
            --enable-prefix-caching \
            --kv-cache-dtype fp8 \
            --quantization ascend \
            --attention_config.indexer_kv_dtype fp8 \
            --tokenizer-mode kimi_k3 \
            --mm-processor-cache-gb 0 \
            --mm-encoder-tp-mode data \
            --no-mm-device-do-normalize \
            --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
            --additional-config '{"enable_mlapo":true,"multistream_overlap_shared_expert":true,"scheduler_config":{"recompute_scheduler_enable":true}}' \
            --profiler-config '{"profiler":"torch","torch_profiler_dir":"<PROFILER_OUTPUT_DIR>","torch_profiler_with_stack":false}' \
            --kv-transfer-config '{
          "kv_connector": "MooncakeConnectorV2",
          "kv_role": "kv_consumer",
          "kv_port": "'"$KV_PORT"'",
          "engine_id": "'"$ENGINE_ID"'",
          "kv_connector_extra_config": {
            "prefill": {
              "pp_size": 4,
              "pcp_size": 8,
              "dp_size": 1,
              "tp_size": 1
            },
            "decode": {
              "dp_size": 4,
              "tp_size": 8
            }
          }
        }' \
            --speculative-config "$SPECULATIVE_CONFIG"
        ```

**Key deployment parameters:**

| Parameter | Description and tuning guidance |
|---|---|
| `--tensor-parallel-size` / `--data-parallel-size` / `--pipeline-parallel-size` | Control tensor, data and pipeline parallelism. Update node assignments, launcher arguments and the P/D connector topology together when changing these values. |
| `--prefill-context-parallel-size` / `--decode-context-parallel-size` | Control Prefill and Decode context parallelism. Choose values for the device topology and long-sequence cache requirements, keeping TP, PP and connector settings consistent. |
| `--max-model-len` | Limit the total input and output tokens per request. Check KV cache capacity before increasing this limit and reserve room for generated output. |
| `--max-num-seqs` | Limit the number of sequences processed in a scheduling iteration. Increasing this value can improve concurrent throughput but also increase cache usage and request latency. |
| `--max-num-batched-tokens` | Set the token budget per scheduling iteration. Tune it for input length and concurrency, comparing throughput, time to first token and time per output token. |
| `--gpu-memory-utilization` | Set the fraction of device memory budgeted for the model executor. A larger budget can leave more space for KV cache; retain runtime headroom and check for out-of-memory errors. |
| `--enable-prefix-caching` | Allow KV cache reuse for matching prefixes. Record cold or warm cache state when benchmarking to make results comparable. |
| `--enforce-eager` / `--compilation-config` | Select eager or graph execution. The templates use eager Prefill and `FULL_DECODE_ONLY` Decode; recheck memory usage and performance after changing modes. |
| `--kv-cache-dtype` / `--attention_config.indexer_kv_dtype` | Set the main attention KV cache and indexer K cache types. Keep a platform-supported combination and validate accuracy and cache capacity after changes. |
| `--block-size` / `--cp-kv-cache-interleave-size` | Control KV cache block size and token interleaving granularity for context parallelism. Use the platform template values and check P/D cache layout and connector compatibility before changing them. |
| `--speculative-config` | Configure the DSpark draft model and speculative token count. Draft TP must equal target-model TP for each role; evaluate gains using acceptance rate and end-to-end throughput. |
| `--enable-expert-parallel` / `--additional-config` | Control expert parallelism and platform optimization options. Start from the template for each platform and role, then change and measure one option at a time. |

#### 5.1.2 Start the Engines and Proxy {#512-start-the-engines-and-proxy}

The Decode launcher is located at [`vllm-ascend/examples/external_online_dp/launch_online_dp.py`](https://github.com/vllm-project/vllm-ascend/blob/main/examples/external_online_dp/launch_online_dp.py) in the vLLM-Ascend repository. Place it in the same directory as the corresponding `run_dp_template.sh` on each Decode node.

=== "A3 series"

    **Start Prefill (P0–P3)**

    Save the matching Prefill template as `run_dp_template.sh` on each P node, fill in its variables, and run it directly without positional arguments. This deployment uses DP=1 and PP=4 on Prefill, with one PP stage per node.

    This A3 deployment uses DP=1, PP=4, TP=4, and PCP=4 on Prefill, with 16 NPU devices per node.

    ```bash
    bash run_dp_template.sh
    ```

    | Prefill script variable | Configuration |
    |---|---|
    | `NODE_RANK` | Set to `0, 1, 2, 3` on P0–P3 respectively; this is the PP node rank. |
    | `MASTER_ADDR` | Use the P0 IP on all four P nodes. |
    | `MASTER_PORT` | Use the same Prefill master port on all four P nodes. |
    | `API_PORT` | P0 API listening port; register the P0 IP and port with the Proxy. P1–P3 use `--headless` and do not expose an API service. |

    The template automatically adds `--headless` on P1–P3. Start P0 before P1–P3, keeping model paths, layer partitioning, and parallelism settings consistent across all four P nodes.

    **Start Decode (D0–D3)**

    Deploy `launch_online_dp.py` on the D nodes and save the matching Decode template as `run_dp_template.sh` in the same directory. Run the following command from that directory to start one DP instance per node:

    This guide uses the repository launcher that passes seven positional arguments. DCP is explicitly set in the Decode template: `16` for A3 and `8` for 950DT Products. Use a launcher that matches the template; do not substitute versions that pass parameters through `$8` or `AICLUSTERHUB_PCP_SIZE` / `AICLUSTERHUB_DCP_SIZE` without adapting the template.

    ```bash
    python launch_online_dp.py \
        --dp-size 4 \
        --tp-size 16 \
        --dp-size-local 1 \
        --dp-rank-start '<DECODE_DP_RANK>' \
        --dp-address '<DECODE_MASTER_IP>' \
        --dp-rpc-port '<DECODE_DP_RPC_PORT>' \
        --vllm-start-port '<DECODE_API_PORT>'
    ```

    Set `<DECODE_DP_RANK>` to `0, 1, 2, 3` on D0–D3 respectively. Use the D0 IP for `<DECODE_MASTER_IP>` and the same `<DECODE_DP_RPC_PORT>` on all four D nodes. Set `<DECODE_API_PORT>` to each node's actual API port. Start D0 before D1–D3.

    The launcher still passes Decode arguments `$1`–`$7`: device list, API port, DP size, DP rank, Decode master IP, DP RPC port, and TP size. Configure master addresses and ports separately for P and D, and register the actual API endpoints with the Proxy.

    **Start the Proxy**

    Once the P0 API and all four Decode APIs are ready, run the following command in the configured environment on the Proxy node:


    ```bash
    python '<VLLM_ASCEND_SOURCE_DIR>/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py' \
        --host 0.0.0.0 \
        --port '<PROXY_PORT>' \
        --workers 1 \
        --prefiller-hosts '<P0_IP>' \
        --prefiller-ports '<P0_API_PORT>' \
        --decoder-hosts '<D0_IP>' '<D1_IP>' '<D2_IP>' '<D3_IP>' \
        --decoder-ports '<D0_API_PORT>' '<D1_API_PORT>' '<D2_API_PORT>' '<D3_API_PORT>'
    ```

    Register only P0 as the Prefill service. Send client inference requests to the Proxy, which forwards them to P0 and the four Decode nodes. The Decode APIs also support health checks, metrics queries, and profiling.

    **Key PD Settings**

    | Setting | A3 Prefill | A3 Decode |
    |---|---|---|
    | DP / TP / PP | `1 / 4 / 4` | `4 / 16 / 1` |
    | PCP / DCP | `4 / 1` | `1 / 16` |
    | `VLLM_PP_LAYER_PARTITION` | `25,25,24,19` | Unset |
    | `--block-size` | `768` | `768` |
    | `--cp-kv-cache-interleave-size` | Not explicitly set | `768` |
    | `--max-num-batched-tokens` | `8192` | `8192` |
    | `--max-num-seqs` | `12` | `12` |
    | `HCCL_BUFFSIZE` | `1024` | `1024` |
    | `HCCL_OP_EXPANSION_MODE` | Unset | `AIV` |
    | `--max-model-len` | `204800` | `204800` |
    | `--gpu-memory-utilization` | `0.90` | `0.80` |
    | Graph mode | Eager | `FULL_DECODE_ONLY` |
    | `--quantization` | `ascend` | `ascend` |
    | `--kv-cache-dtype` | `int8` | `int8` |
    | `--attention_config.indexer_kv_dtype` (indexer K cache) | `int8` | `int8` |
    | `enable_fused_mc2` | `0` (MegaMoE disabled) | `0` |
    | `enable_flashcomm1` | `false` | Not explicitly set |
    | `enable_shared_expert_dp` | `false` | `false` |
    | `multistream_overlap_shared_expert` | `false` | `true` |
    | `enable_mlapo` | Not explicitly set | `true` |
    | `scheduler_config.recompute_scheduler_enable` | Not explicitly set | `true` |
    | Prefix Cache | Enabled | Enabled |
    | Model Runner | V2 | V2 |
    | KV connector | `MooncakeConnectorV2` | `MooncakeConnectorV2` |
    | Draft TP (optional DSpark3) | `4` | `16` |
    | Draft `enforce_eager` (optional DSpark3) | `true` | `false` |
    | Draft KV dtype (optional DSpark3) | `bfloat16` | `bfloat16` |
    | `num_speculative_tokens` (optional DSpark3) | `3` | `3` |

    For 1M context, change only `--max-model-len` to `1048576` on both roles and keep all other settings unchanged. See [Section 5.2](#52-long-context).

=== "950DT Products"

    **Start Prefill (P0–P3)**

    Save the matching Prefill template as `run_dp_template.sh` on each P node, fill in its variables, and run it directly without positional arguments. This deployment uses DP=1 and PP=4 on Prefill, with one PP stage per node.

    This 950DT Products deployment uses DP=1, PP=4, TP=1, and PCP=8 on Prefill, with eight NPU devices per node.

    ```bash
    bash run_dp_template.sh
    ```

    | Prefill script variable | Configuration |
    |---|---|
    | `NODE_RANK` | Set to `0, 1, 2, 3` on P0–P3 respectively; this is the PP node rank. |
    | `MASTER_ADDR` | Use the P0 IP on all four P nodes. |
    | `MASTER_PORT` | Use the same Prefill master port on all four P nodes. |
    | `API_PORT` | P0 API listening port; register the P0 IP and port with the Proxy. P1–P3 use `--headless` and do not expose an API service. |

    The template automatically adds `--headless` on P1–P3. Start P0 before P1–P3, keeping model paths, layer partitioning, and parallelism settings consistent across all four P nodes.

    **Start Decode (D0–D3)**

    Deploy `launch_online_dp.py` on the D nodes and save the matching Decode template as `run_dp_template.sh` in the same directory. Run the following command from that directory to start one DP instance per node:

    This guide uses the repository launcher that passes seven positional arguments. DCP is explicitly set in the Decode template: `16` for A3 and `8` for 950DT Products. Use a launcher that matches the template; do not substitute versions that pass parameters through `$8` or `AICLUSTERHUB_PCP_SIZE` / `AICLUSTERHUB_DCP_SIZE` without adapting the template.

    ```bash
    python launch_online_dp.py \
        --dp-size 4 \
        --tp-size 8 \
        --dp-size-local 1 \
        --dp-rank-start '<DECODE_DP_RANK>' \
        --dp-address '<DECODE_MASTER_IP>' \
        --dp-rpc-port '<DECODE_DP_RPC_PORT>' \
        --vllm-start-port '<DECODE_API_PORT>'
    ```

    Set `<DECODE_DP_RANK>` to `0, 1, 2, 3` on D0–D3 respectively. Use the D0 IP for `<DECODE_MASTER_IP>` and the same `<DECODE_DP_RPC_PORT>` on all four D nodes. Set `<DECODE_API_PORT>` to each node's actual API port. Start D0 before D1–D3.

    The launcher still passes Decode arguments `$1`–`$7`: device list, API port, DP size, DP rank, Decode master IP, DP RPC port, and TP size. Configure master addresses and ports separately for P and D, and register the actual API endpoints with the Proxy.

    **Start the Proxy**

    After the P0 API and all four Decode APIs are ready, register these five addresses with the Proxy. Each API port below must match that node's launch command. Send client requests to the Proxy using model name `kimi-k3.1-pro`.

    ```bash
    python '<VLLM_ASCEND_SOURCE_DIR>/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py' \
        --host 0.0.0.0 \
        --port '<PROXY_PORT>' \
        --workers 1 \
        --prefiller-hosts '<P0_IP>' \
        --prefiller-ports '<P0_API_PORT>' \
        --decoder-hosts '<D0_IP>' '<D1_IP>' '<D2_IP>' '<D3_IP>' \
        --decoder-ports '<D0_API_PORT>' '<D1_API_PORT>' '<D2_API_PORT>' '<D3_API_PORT>'
    ```

    **Key PD Settings**

    | Setting | 950DT Products Prefill | 950DT Products Decode |
    |---|---|---|
    | DP / TP / PP | `1 / 1 / 4` | `4 / 8 / 1` |
    | PCP / DCP | `8 / 1` | `1 / 8` |
    | VLLM_PP_LAYER_PARTITION | `24,24,24,21` | — |
    | `--block-size` / `--cp-kv-cache-interleave-size` | `1536 / 1536` | `1536 / 1536` |
    | `--max-model-len` | `140000` | `140000` |
    | `--max-num-batched-tokens` | `16384` | `768` |
    | `--max-num-seqs` | `64` | `64` |
    | `--gpu-memory-utilization` | `0.90` | `0.92` |
    | `--kv-cache-dtype` / `--attention_config.indexer_kv_dtype` | `fp8 / fp8` | `fp8 / fp8` |
    | `HCCL_OP_EXPANSION_MODE` / `HCCL_BUFFSIZE` | `CCU_SCHED / 2000` | `CCU_SCHED / 800` |
    | Graph mode | Eager | `FULL_DECODE_ONLY` |
    | Draft TP / num_speculative_tokens | `1 / 3` | `8 / 3` |
    | Draft enforce_eager / kv_cache_dtype | `false / auto` | `false / auto` |
    | `--tokenizer-mode` | `kimi_k3` | `kimi_k3` |

### 5.2 Long-Context and 1M PD Separation Deployment {#52-long-context}

#### 5.2.1 Create the Engine Templates {#521-create-the-engine-templates}

=== "A3 series"

    A3 1M deployment uses the same eight-node 4P4D topology as [Section 5.1](#51-pd-deployment): PP4 / PCP4 / TP4 on Prefill and DP4 / DCP16 / TP16 on Decode. The basic configuration already enables DCP16. Change only `--max-model-len` to `1048576` on both roles and keep all other settings unchanged.

    The 1M limit includes both input and output tokens. Reserve space for the generated response.

    To enable DSpark3, replace `<YOUR_DRAFT_MODEL_PATH>` with the local directory of a draft model compatible with the A3 target model and add the role-specific options from [Section 5.1](#51-pd-deployment). Use the same draft checkpoint on both roles and the same directory path on every node. Complete Prefill and Decode templates and launch and verification instructions are provided below.

    === "Prefill"

        Save the Prefill template below as `run_dp_template.sh`. Fill in the local IP, NIC, model paths, KV/HCCL ports, and the top-level `API_PORT`, `NODE_RANK`, `MASTER_ADDR`, and `MASTER_PORT` variables, then start it directly with Bash. The script automatically sets `--headless` on P1–P3.

        ```bash
        #!/usr/bin/env bash
        set -euo pipefail

        API_PORT="<PREFILL_API_PORT>"
        NODE_RANK="<PREFILL_NODE_RANK>"
        MASTER_ADDR="<PREFILL_MASTER_IP>"
        MASTER_PORT="<PREFILL_MASTER_PORT>"
        headless_args=()
        if (( NODE_RANK > 0 )); then
            headless_args=(--headless)
        fi

        KV_PORT='<PREFILL_KV_PORT>'
        nic_name='<PREFILL_NIC_NAME>'
        local_ip='<PREFILL_LOCAL_IP>'

        unset ftp_proxy FTP_PROXY http_proxy HTTP_PROXY https_proxy HTTPS_PROXY
        unset LD_PRELOAD HCCL_OP_EXPANSION_MODE VLLM_PP_LAYER_PARTITION

        export HCCL_IF_IP="$local_ip"
        export HCCL_SOCKET_IFNAME="$nic_name"
        export GLOO_SOCKET_IFNAME="$nic_name"
        export TP_SOCKET_IFNAME="$nic_name"
        export HCCL_BUFFSIZE=1024
        export HCCL_CONNECT_TIMEOUT=1200
        export HCCL_IF_BASE_PORT='<PREFILL_HCCL_BASE_PORT>'
        export HCCL_HOST_SOCKET_PORT_RANGE='<PREFILL_HCCL_PORT_RANGE>'
        export HCCL_NPU_SOCKET_PORT_RANGE=auto
        export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
        export LD_LIBRARY_PATH="/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:${LD_LIBRARY_PATH:-}"
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export PYTHONHASHSEED=0
        export OMP_NUM_THREADS=1
        export OMP_PROC_BIND=false
        export VLLM_USE_V2_MODEL_RUNNER=1
        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
        export VLLM_HOST_IP="$local_ip"
        export VLLM_PP_LAYER_PARTITION="25,25,24,19"
        export FLA_STABLE_DEBUG_DESC=0

        vllm serve '/root/.cache/Kimi-K3.1-Pro-w4a8' \
            --host 0.0.0.0 \
            --port "$API_PORT" \
            --quantization ascend \
            --enable-auto-tool-choice \
            --reasoning-parser kimi_k3 \
            --tool-call-parser kimi_k3 \
            --tokenizer-mode kimi_k3 \
            --data-parallel-size "1" \
            --pipeline-parallel-size 4 \
            --prefill-context-parallel-size 4 \
            --decode-context-parallel-size 1 \
            --nnodes 4 \
            --node-rank "$NODE_RANK" \
            "${headless_args[@]}" \
            --distributed-executor-backend mp \
            --data-parallel-backend mp \
            --master-addr "$MASTER_ADDR" \
            --master-port "$MASTER_PORT" \
            --tensor-parallel-size "4" \
            --enable-expert-parallel \
            --seed 1024 \
            --served-model-name kimi-k3.1-pro \
            --max-model-len 1048576 \
            --max-num-batched-tokens 8192 \
            --max-num-seqs 12 \
            --enforce-eager \
            --trust-remote-code \
            --gpu-memory-utilization 0.90 \
            --block-size 768 \
            --kv-cache-dtype int8 \
            --attention_config.indexer_kv_dtype int8 \
            --mm-encoder-tp-mode data \
            --no-mm-device-do-normalize \
            --mm-processor-cache-gb 1 \
            --skip-mm-profiling \
            --enable-prefix-caching \
            --additional-config '{"enable_flashcomm1":false,"multistream_overlap_shared_expert":false,"enable_fused_mc2":0,"enable_shared_expert_dp":false}' \
            --limit-mm-per-prompt '{"vision_chunk":2}' \
            --kv-transfer-config \
            '{
              "kv_connector": "MooncakeConnectorV2",
              "kv_role": "kv_producer",
              "kv_port": "'"$KV_PORT"'",
              "kv_connector_extra_config": {
                "prefill": {
                  "dp_size": 1,
                  "tp_size": 4,
                  "pp_size": 4,
                  "pcp_size": 4,
                  "dcp_size": 1
                },
                "decode": {
                  "dp_size": 4,
                  "tp_size": 16,
                  "pp_size": 1,
                  "pcp_size": 1,
                  "dcp_size": 16
                }
              }
            }'
        ```

    === "Decode"

        Save the 1M Decode template below as `run_dp_template.sh` on each Decode node. Fill in the local IP, network interface, model path, KV port, and HCCL port settings as described in [Section 5.1.1](#511-create-the-engine-templates).

        ```bash
        #!/usr/bin/env bash
        set -euo pipefail

        KV_PORT='<DECODE_KV_PORT>'
        nic_name='<DECODE_NIC_NAME>'
        local_ip='<DECODE_LOCAL_IP>'

        unset ftp_proxy FTP_PROXY http_proxy HTTP_PROXY https_proxy HTTPS_PROXY
        unset LD_PRELOAD VLLM_PP_LAYER_PARTITION

        export HCCL_IF_IP="$local_ip"
        export HCCL_SOCKET_IFNAME="$nic_name"
        export GLOO_SOCKET_IFNAME="$nic_name"
        export TP_SOCKET_IFNAME="$nic_name"
        export HCCL_OP_EXPANSION_MODE=AIV
        export HCCL_BUFFSIZE=1024
        export HCCL_CONNECT_TIMEOUT=1200
        export HCCL_IF_BASE_PORT='<DECODE_HCCL_BASE_PORT>'
        export HCCL_HOST_SOCKET_PORT_RANGE='<DECODE_HCCL_PORT_RANGE>'
        export HCCL_NPU_SOCKET_PORT_RANGE=auto
        export ASCEND_RT_VISIBLE_DEVICES="$1"
        export LD_LIBRARY_PATH="/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:${LD_LIBRARY_PATH:-}"
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export PYTHONHASHSEED=0
        export OMP_NUM_THREADS=1
        export OMP_PROC_BIND=false
        export VLLM_USE_V2_MODEL_RUNNER=1
        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
        export VLLM_HOST_IP="$local_ip"
        export FLA_STABLE_DEBUG_DESC=0

        vllm serve '/root/.cache/Kimi-K3.1-Pro-w4a8' \
            --host 0.0.0.0 \
            --port "$2" \
            --quantization ascend \
            --enable-auto-tool-choice \
            --reasoning-parser kimi_k3 \
            --tool-call-parser kimi_k3 \
            --tokenizer-mode kimi_k3 \
            --data-parallel-size "$3" \
            --data-parallel-size-local 1 \
            --data-parallel-rank "$4" \
            --data-parallel-address "$5" \
            --data-parallel-rpc-port "$6" \
            --distributed-executor-backend mp \
            --tensor-parallel-size "$7" \
            --pipeline-parallel-size 1 \
            --prefill-context-parallel-size 1 \
            --decode-context-parallel-size 16 \
            --cp-kv-cache-interleave-size 768 \
            --block-size 768 \
            --enable-expert-parallel \
            --seed 1024 \
            --served-model-name kimi-k3.1-pro \
            --max-model-len 1048576 \
            --max-num-batched-tokens 8192 \
            --max-num-seqs 12 \
            --trust-remote-code \
            --gpu-memory-utilization 0.80 \
            --kv-cache-dtype int8 \
            --attention_config.indexer_kv_dtype int8 \
            --mm-encoder-tp-mode data \
            --no-mm-device-do-normalize \
            --mm-processor-cache-gb 1 \
            --skip-mm-profiling \
            --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
            --enable-prefix-caching \
            --additional-config '{"enable_fused_mc2":0,"multistream_overlap_shared_expert":true,"enable_shared_expert_dp":false,"enable_mlapo":true,"scheduler_config":{"recompute_scheduler_enable":true}}' \
            --limit-mm-per-prompt '{"vision_chunk":2}' \
            --kv-transfer-config \
            '{
              "kv_connector": "MooncakeConnectorV2",
              "kv_role": "kv_consumer",
              "kv_port": "'"$KV_PORT"'",
              "kv_connector_extra_config": {
                "prefill": {
                  "dp_size": 1,
                  "tp_size": 4,
                  "pp_size": 4,
                  "pcp_size": 4,
                  "dcp_size": 1
                },
                "decode": {
                  "dp_size": 4,
                  "tp_size": 16,
                  "pp_size": 1,
                  "pcp_size": 1,
                  "dcp_size": 16
                }
              }
            }'
        ```

=== "950DT Products"

    950DT Products 1M deployment uses the basic eight-node 4P4D topology: DP1 / PP4 / PCP8 / TP1 on Prefill and DP4 / TP8 / DCP8 on Decode. Set `--max-model-len 1048576` on both Prefill and Decode and retain the other basic settings.

    The 1M limit includes both input and output tokens. Reserve space for the generated response.

    The 950DT Products basic and 1M templates enable DSpark3 with the same draft checkpoint on both roles, draft TP `1/8` on Prefill/Decode matching the target TP on each role, three speculative tokens, draft `enforce_eager=false`, and `kv_cache_dtype=auto`. The Prefill target model retains `--enforce-eager`. Replace `<YOUR_DRAFT_MODEL_PATH>` with the local directory of a draft checkpoint compatible with the target model on this platform.

    The templates use `--tokenizer-mode kimi_k3`. For reasoning parsing and tool calling, also configure `--reasoning-parser kimi_k3`, `--tool-call-parser kimi_k3`, and `--enable-auto-tool-choice` together as described in [Section 10](#10-faq).

    Fill in a unique `<PREFILL_ENGINE_ID>` or `<DECODE_ENGINE_ID>` for each engine. Use distinct node ranks for P0–P3; the template automatically adds `--headless` on P1–P3.

    === "Prefill"

        Save the Prefill template below as `run_dp_template.sh`. Fill in the local IP, NIC, model paths, KV/HCCL ports, and the top-level `API_PORT`, `NODE_RANK`, `MASTER_ADDR`, and `MASTER_PORT` variables, then start it directly with Bash. The script automatically sets `--headless` on P1–P3.

        ```bash
        #!/usr/bin/env bash
        set -e

        API_PORT="<PREFILL_API_PORT>"
        NODE_RANK="<PREFILL_NODE_RANK>"
        MASTER_ADDR="<PREFILL_MASTER_IP>"
        MASTER_PORT="<PREFILL_MASTER_PORT>"

        unset ftp_proxy FTP_PROXY
        unset https_proxy HTTPS_PROXY
        unset http_proxy HTTP_PROXY

        nic_name="<PREFILL_NIC_NAME>"
        local_ip="<PREFILL_LOCAL_IP>"
        KV_PORT="<PREFILL_KV_PORT>"
        ENGINE_ID="<PREFILL_ENGINE_ID>"
        DRAFT_MODEL_PATH="<YOUR_DRAFT_MODEL_PATH>"

        headless_args=()
        if (( NODE_RANK > 0 )); then
            headless_args=(--headless)
        fi

        export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        export HCCL_IF_IP="$local_ip"
        export VLLM_HOST_IP="$local_ip"
        export GLOO_SOCKET_IFNAME="$nic_name"
        export TP_SOCKET_IFNAME="$nic_name"
        export HCCL_SOCKET_IFNAME="$nic_name"
        export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'
        export VLLM_RPC_TIMEOUT=3600000
        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
        export HCCL_EXEC_TIMEOUT=2040
        export HCCL_CONNECT_TIMEOUT=1200
        export HCCL_BUFFSIZE=2000
        export OMP_PROC_BIND=false
        export OMP_NUM_THREADS=10
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export TASK_QUEUE_ENABLE=1

        SPECULATIVE_CONFIG="$(
          printf \
          '{"method":"dspark","model":"%s","num_speculative_tokens":3,"draft_tensor_parallel_size":1,"draft_sample_method":"greedy","enforce_eager":false,"kv_cache_dtype":"auto"}' \
          "$DRAFT_MODEL_PATH"
        )"

        export HCCL_OP_EXPANSION_MODE="CCU_SCHED"
        export VLLM_USE_V2_MODEL_RUNNER=1
        export VLLM_PP_LAYER_PARTITION=24,24,24,21

        vllm serve "/root/.cache/Kimi-K3.1-Pro-w4a8-mxfp" \
            --host 0.0.0.0 \
            --port "$API_PORT" \
            --data-parallel-size "1" \
            --tensor-parallel-size "1" \
            --prefill-context-parallel-size 8 \
            --nnodes 4 \
            --node-rank "$NODE_RANK" \
            --master-addr "$MASTER_ADDR" \
            --master-port "$MASTER_PORT" \
            --pipeline-parallel-size 4 \
            "${headless_args[@]}" \
            --block-size 1536 \
            --cp-kv-cache-interleave-size 1536 \
            --served-model-name kimi-k3.1-pro \
            --allowed-local-media-path / \
            --trust-remote-code \
            --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":128}' \
            --enable-expert-parallel \
            --max-num-seqs 64 \
            --max-model-len 1048576 \
            --max-num-batched-tokens 16384 \
            --gpu-memory-utilization 0.90 \
            --enable-prefix-caching \
            --kv-cache-dtype fp8 \
            --quantization ascend \
            --attention_config.indexer_kv_dtype fp8 \
            --tokenizer-mode kimi_k3 \
            --mm-processor-cache-gb 0 \
            --mm-encoder-tp-mode data \
            --no-mm-device-do-normalize \
            --enforce-eager \
            --additional-config '{"enable_mlapo":false,"enable_flashcomm1":true,"enable_shared_expert_dp":true,"multistream_overlap_shared_expert":false,"enable_fused_mc2":0,"enable_force_eplb":false}' \
            --profiler-config '{"profiler":"torch","torch_profiler_dir":"<PROFILER_OUTPUT_DIR>","torch_profiler_with_stack":false}' \
            --kv-transfer-config '{
          "kv_connector": "MooncakeConnectorV2",
          "kv_role": "kv_producer",
          "kv_port": "'"$KV_PORT"'",
          "engine_id": "'"$ENGINE_ID"'",
          "kv_connector_extra_config": {
            "prefill": {
              "pp_size": 4,
              "pcp_size": 8,
              "dp_size": 1,
              "tp_size": 1
            },
            "decode": {
              "dp_size": 4,
              "tp_size": 8
            }
          }
        }' \
            --speculative-config "$SPECULATIVE_CONFIG"
        ```

    === "Decode"

        Save the 1M Decode template below as `run_dp_template.sh` on the corresponding nodes. Fill in the local IP, network interface, KV port, and target and draft model paths, and replace `<PROFILER_OUTPUT_DIR>` with a writable profiling output directory.

        ```bash
        #!/usr/bin/env bash
        set -e

        unset ftp_proxy FTP_PROXY
        unset https_proxy HTTPS_PROXY
        unset http_proxy HTTP_PROXY

        nic_name="<DECODE_NIC_NAME>"
        local_ip="<DECODE_LOCAL_IP>"
        KV_PORT="<DECODE_KV_PORT>"
        ENGINE_ID="<DECODE_ENGINE_ID>"
        DRAFT_MODEL_PATH="<YOUR_DRAFT_MODEL_PATH>"

        export ASCEND_RT_VISIBLE_DEVICES="$1"
        export HCCL_IF_IP="$local_ip"
        export VLLM_HOST_IP="$local_ip"
        export GLOO_SOCKET_IFNAME="$nic_name"
        export TP_SOCKET_IFNAME="$nic_name"
        export HCCL_SOCKET_IFNAME="$nic_name"
        export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'
        export VLLM_RPC_TIMEOUT=3600000
        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
        export HCCL_EXEC_TIMEOUT=2040
        export HCCL_CONNECT_TIMEOUT=1200
        export HCCL_BUFFSIZE=800
        export OMP_PROC_BIND=false
        export OMP_NUM_THREADS=10
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export TASK_QUEUE_ENABLE=1

        SPECULATIVE_CONFIG="$(
          printf \
          '{"method":"dspark","model":"%s","num_speculative_tokens":3,"draft_tensor_parallel_size":8,"draft_sample_method":"greedy","enforce_eager":false,"kv_cache_dtype":"auto"}' \
          "$DRAFT_MODEL_PATH"
        )"

        export HCCL_OP_EXPANSION_MODE="CCU_SCHED"
        export VLLM_USE_V2_MODEL_RUNNER=1

        vllm serve "/root/.cache/Kimi-K3.1-Pro-w4a8-mxfp" \
            --host 0.0.0.0 \
            --port "$2" \
            --data-parallel-size "$3" \
            --data-parallel-rank "$4" \
            --data-parallel-address "$5" \
            --data-parallel-rpc-port "$6" \
            --tensor-parallel-size "$7" \
            --decode-context-parallel-size 8 \
            --block-size 1536 \
            --cp-kv-cache-interleave-size 1536 \
            --served-model-name kimi-k3.1-pro \
            --allowed-local-media-path / \
            --trust-remote-code \
            --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":128}' \
            --enable-expert-parallel \
            --max-num-seqs 64 \
            --max-model-len 1048576 \
            --max-num-batched-tokens 768 \
            --gpu-memory-utilization 0.92 \
            --enable-prefix-caching \
            --kv-cache-dtype fp8 \
            --quantization ascend \
            --attention_config.indexer_kv_dtype fp8 \
            --tokenizer-mode kimi_k3 \
            --mm-processor-cache-gb 0 \
            --mm-encoder-tp-mode data \
            --no-mm-device-do-normalize \
            --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
            --additional-config '{"enable_mlapo":true,"multistream_overlap_shared_expert":true,"scheduler_config":{"recompute_scheduler_enable":true}}' \
            --profiler-config '{"profiler":"torch","torch_profiler_dir":"<PROFILER_OUTPUT_DIR>","torch_profiler_with_stack":false}' \
            --kv-transfer-config '{
          "kv_connector": "MooncakeConnectorV2",
          "kv_role": "kv_consumer",
          "kv_port": "'"$KV_PORT"'",
          "engine_id": "'"$ENGINE_ID"'",
          "kv_connector_extra_config": {
            "prefill": {
              "pp_size": 4,
              "pcp_size": 8,
              "dp_size": 1,
              "tp_size": 1
            },
            "decode": {
              "dp_size": 4,
              "tp_size": 8
            }
          }
        }' \
            --speculative-config "$SPECULATIVE_CONFIG"
        ```

#### 5.2.2 Start and Verify the Service {#522-start-and-verify-the-service}

=== "A3 series"

    Using the 1M templates above, follow [Section 5.1.2](#512-start-the-engines-and-proxy) to run Bash directly on each P node and the Python launcher on each D node, then start the Proxy.

    Check the logs for model length `1048576`, PCP4 / TP4 on Prefill, and DCP16 / TP16 on Decode. First check the service and complete a short request as described in [Section 6](#6-functional-verification), then validate long requests through the Proxy. Total input and output tokens must remain within the configured limit. Record the actual lengths, concurrency, memory usage, and request completion status.

=== "950DT Products"

    Using the 1M templates above, follow [Section 5.1.2](#512-start-the-engines-and-proxy) to run Bash directly on each P node and the Python launcher on each D node, then start the Proxy. Check maximum length `1048576` on both roles and DCP8 on Decode; validate a short request before testing long sequences through the Proxy.

### 5.3 Pooled PD Separation Deployment {#53-pooled-pd}

The pooling flow differs by platform: A3 uses a separate MetaService, eight standalone MemCache providers, and zero-capacity engine clients. 950DT Products uses the same MetaService startup procedure without standalone providers. See [Section 5.3.2](#532-prepare-the-memcache-configuration-files) for configuration.

The supplied A3 reference uses eight nodes with **DP1 / PP4 / PCP4 / TP4 / DCP1** on Prefill and **DP4 / PP1 / PCP1 / TP16 / DCP16** on Decode. Both roles use a maximum model length of `1048576`, INT8 attention/indexer caches, DSpark3, and non-layerwise pool transfer (`use_layerwise=false`). The pooled example uses a separate MetaService host and one 1 GiB DRAM provider per P/D node. The 950DT Products engine templates below follow [Section 5.1.1](#511-create-the-engine-templates); see [Section 5.3.5](#535-create-the-engine-templates) for their lookup identifier assignments.

These are supplied deployment references. Environment readiness does not establish model-side pool hits, output correctness, or performance. Verify those on the deployed runtime as described in [Section 5.3.7](#537-verify-the-pool-and-model-service). The A3 reference retains different P/D TP sizes; the equal-TP settings in the K3 templates are not a universal pooling requirement. Do not infer support for another cache layout from this reference.

#### 5.3.1 Prepare the MF Components {#531-prepare-the-mf-components}

=== "A3 series"

    Use the model environment prepared in [Section 4](#4-installation). The supplied A3 environment uses `memcache_hybrid==1.3.0` and `memfabric_hybrid==1.3.0`. Check the packages with the same Python environment used by vLLM:

    ```bash
    python3 -m pip show memcache_hybrid memfabric_hybrid
    ```

    If these compatible versions are missing, install them in that environment:

    ```bash
    python3 -m pip install "memfabric_hybrid==1.3.0" "memcache_hybrid==1.3.0"
    ```

    For offline installation or a source build, follow the [MemCache installation guide](https://gitcode.com/Ascend/memcache/blob/master/doc/install_run.md). See the [KV Cache Pool deployment guide](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md) for additional backend and platform requirements.

    Choose an absolute working directory, represented by `<MEMCACHE_CONFIG_DIR>`, on every provider and engine node and on the MetaService host. Keep the following files there; this need not be the package installation directory:

    | File | Purpose |
    |---|---|
    | `runtime_env.sh` | Select the Python interpreter, add package libraries to `LD_LIBRARY_PATH`, and set `PYTHONHASHSEED=0` |
    | `mmc-meta.conf` | MetaService configuration; needed only on the MetaService host |
    | `mmc-local-standalone.conf` | Standalone provider configuration, with DRAM capacity |
    | `mmc-local.conf` | Engine client configuration, with `dram.size=0GB` |
    | `memcache_standalone.py` | Start and keep the standalone provider running |

    Save the following as `<MEMCACHE_CONFIG_DIR>/runtime_env.sh`. Set `MEMC_PYTHON` before sourcing it if the interpreter is not `python3`. Both packages must be installed in the selected interpreter's environment:

    ```bash
    #!/usr/bin/env bash
    export MEMC_PYTHON="${MEMC_PYTHON:-python3}"
    MEMFABRIC_PACKAGE_DIR="$("$MEMC_PYTHON" -c 'import importlib.util; print(next(iter(importlib.util.find_spec("memfabric_hybrid").submodule_search_locations)))')"
    MEMCACHE_PACKAGE_DIR="$("$MEMC_PYTHON" -c 'import importlib.util; print(next(iter(importlib.util.find_spec("memcache_hybrid").submodule_search_locations)))')"
    export LD_LIBRARY_PATH="${MEMFABRIC_PACKAGE_DIR}/lib:${MEMCACHE_PACKAGE_DIR}/lib:${LD_LIBRARY_PATH:-}"
    export PYTHONHASHSEED=0
    ```

    Check imports from the actual engine launch directory as well as a neutral directory such as `/tmp`. A local directory named `vllm_ascend` can shadow an editable installation:

    ```bash
    source '<MEMCACHE_CONFIG_DIR>/runtime_env.sh'
    "$MEMC_PYTHON" -c 'import vllm, vllm_ascend, memcache_hybrid, memfabric_hybrid; print(vllm.__file__); print(vllm_ascend.__file__)'
    ```

    For A3, expose all 16 logical NPU devices to each node's container.

=== "950DT Products"

    Use MF package versions compatible with 950DT Products. On the MetaService host, prepare `runtime_env.sh` as shown in the A3 tab and select a Python environment with the matching packages installed. Standalone-provider files and startup steps apply only to A3. Use the prepared model and MF runtime environment on P/D nodes; set each node configuration path explicitly in [Section 5.3.6](#536-start-the-engines-and-proxy).

#### 5.3.2 Prepare the MemCache Configuration Files {#532-prepare-the-memcache-configuration-files}

This guide uses `ock.mmc.local_service.dram.size=1GB` by default. It specifies the host DRAM capacity contributed to the cache pool by one MemCache LocalService instance, not NPU memory or total cluster capacity. Size it for the available memory on your machine, reserving resources for the model, operating system, and other processes. See the [MemCache configuration documentation](https://gitcode.com/Ascend/memcache) for adjustments. A3 standalone providers use this default, while engine clients retain `dram.size=0GB` to avoid duplicate allocations. 950DT Products node configurations use `1GB`.

=== "A3 series"

    Select one MetaService host, which may be separate from the eight model nodes. Create `<MEMCACHE_CONFIG_DIR>/mmc-meta.conf` on that host. The supplied environment uses ports `5000`, `6000`, and `8099`:

    ```ini
    ock.mmc.meta_service_url = tcp://<META_SERVICE_IP>:5000
    ock.mmc.meta_service.config_store_url = tcp://<META_SERVICE_IP>:6000
    ock.mmc.meta_service.metrics_url = http://<META_SERVICE_IP>:8099
    ock.mmc.log_level = info
    ```

    On every P/D node, create `<MEMCACHE_CONFIG_DIR>/mmc-local.conf` for the **engine clients**. Use the same MetaService address everywhere:

    ```ini
    ock.mmc.meta_service_url = tcp://<META_SERVICE_IP>:5000
    ock.mmc.local_service.config_store_url = tcp://<META_SERVICE_IP>:6000
    ock.mmc.log_level = info
    ock.mmc.local_service.world_size = 256
    ock.mmc.local_service.protocol = device_sdma
    ock.mmc.local_service.dram.size = 0GB
    ock.mmc.local_service.max.dram.size = 1024GB
    ```

    Use [mem_scan.py](https://gitcode.com/Ascend/memfabric_hybrid/blob/master/src/smem/python/memfabric_hybrid/memfabric_hybrid/mem_scan.py) to determine each node's available memory after reserving resources for the model and operating system. Create `<MEMCACHE_CONFIG_DIR>/mmc-local-standalone.conf` for the **standalone provider** on every node. The example below assumes at least 1 GiB of memory is available for the pool; adjust its DRAM capacity to the memory available on that node:

    ```ini
    ock.mmc.meta_service_url = tcp://<META_SERVICE_IP>:5000
    ock.mmc.local_service.config_store_url = tcp://<META_SERVICE_IP>:6000
    ock.mmc.log_level = info
    ock.mmc.local_service.world_size = 256
    ock.mmc.local_service.protocol = device_sdma
    ock.mmc.local_service.dram.size = 1GB
    ock.mmc.local_service.max.dram.size = 1024GB
    ```

    `world_size=256` is the LocalService communication capacity, not the number of standalone providers to start. The reference starts one provider on each of the eight P/D nodes. Its eight 1 GiB DRAM segments total `8589934592` bytes (8 GiB).

=== "950DT Products"

    Prepare `nodes/<NODE_NAME>/mmc-local.conf` on each of the eight nodes, setting DRAM capacity as described in this section. At startup, set `MMC_LOCAL_CONFIG_PATH` to the absolute path of the local node configuration file, as shown in [Section 5.3.6](#536-start-the-engines-and-proxy).

    Set the MetaService address, transport protocol, and other fields to match the deployment environment.

    Create the MetaService `mmc-meta.conf` using the A3 tab as a reference. Set addresses and ports for your environment and keep them consistent across all node configurations.

#### 5.3.3 Start MetaService {#533-start-metaservice}

=== "A3 series"

    Check for an existing MetaService before starting a new one. On the selected MetaService host, source the runtime environment and select the metadata configuration:

    ```bash
    source '<MEMCACHE_CONFIG_DIR>/runtime_env.sh'
    export MMC_META_CONFIG_PATH='<MEMCACHE_CONFIG_DIR>/mmc-meta.conf'
    "$MEMC_PYTHON" -u -c 'from memcache_hybrid import MetaService; MetaService.main()'
    ```

    Keep the process running. All P/D nodes must reach the configured MetaService on TCP ports `5000` and `6000`; query metrics on port `8099`. Update every configuration if the addresses or ports change. Reuse an existing healthy MetaService without restarting it as part of a single provider's startup.

=== "950DT Products"

    Following the A3 procedure, load the runtime environment and select the metadata configuration on the chosen MetaService host, then start the service:

    ```bash
    source '<MEMCACHE_CONFIG_DIR>/runtime_env.sh'
    export MMC_META_CONFIG_PATH='<MEMCACHE_CONFIG_DIR>/mmc-meta.conf'
    "$MEMC_PYTHON" -u -c 'from memcache_hybrid import MetaService; MetaService.main()'
    ```

    Keep the process running and ensure all P/D nodes can reach the configured MetaService address and ports. Reuse an existing healthy service. This step starts only MetaService, without per-node standalone MMC processes.

#### 5.3.4 Start the Standalone MemCache Processes {#534-start-the-standalone-memcache-processes}

=== "A3 series"

    Save the following as `<MEMCACHE_CONFIG_DIR>/memcache_standalone.py` on every P/D node:

    ```python
    import time

    from memcache_hybrid import DistributedObjectStore


    if __name__ == "__main__":
        store = DistributedObjectStore()
        result = store.init(0)
        if result != 0:
            print(f"Failed to initialize MemCache, result = {result}", flush=True)
            raise SystemExit(1)
        print("Successfully initialized memcache.", flush=True)

        while True:
            time.sleep(1)
    ```

    After confirming that no existing provider uses the same resources, start **one** standalone provider per node in a separate shell:

    ```bash
    source '<MEMCACHE_CONFIG_DIR>/runtime_env.sh'
    export MMC_LOCAL_CONFIG_PATH='<MEMCACHE_CONFIG_DIR>/mmc-local-standalone.conf'
    "$MEMC_PYTHON" -u '<MEMCACHE_CONFIG_DIR>/memcache_standalone.py'
    ```

    Wait for `Successfully initialized memcache.` on all eight nodes. Confirm that each process is still alive and that MetaService reports the expected registered DRAM capacity before starting the engines.

    The provider uses `mmc-local-standalone.conf`; each vLLM engine must use `mmc-local.conf`. Passing the provider configuration to engine workers can allocate DRAM repeatedly.

=== "950DT Products"

    Skip this step for 950DT Products. Once MetaService is ready, start the P/D engines as described in [Section 5.3.6](#536-start-the-engines-and-proxy).

#### 5.3.5 Create the Engine Templates {#535-create-the-engine-templates}

=== "A3 series"

    Prefill sets its PP node rank and master address/port through script variables and starts directly with Bash. Decode retains the launcher's seven positional arguments. Use the pooled template for each role and consistent model paths on all nodes.

    Fill in the A3 pooled templates' KV ports, lookup identifiers, HCCL ports, and profiling directory for your environment. Reference values are `33100/33500` for `<PREFILL_KV_PORT>` / `<DECODE_KV_PORT>`, `33101/33501` for `<PREFILL_LOOKUP_RPC_PORT>` / `<DECODE_LOOKUP_RPC_PORT>`, and Prefill HCCL base port `45000` with range `45000-45511`. Avoid service-port conflicts on the same host and use a writable `<PROFILER_OUTPUT_DIR>`.

    The reference uses the same compatible DSpark draft checkpoint on both roles, with three speculative tokens and BF16 draft caches. Draft TP is 4 on Prefill and 16 on Decode; the Prefill draft uses eager execution and the Decode draft uses `enforce_eager=false`. Fill in the local directory of a draft checkpoint compatible with the A3 target model.



    === "Prefill"

        ```bash
        #!/usr/bin/env bash
        set -euo pipefail

        API_PORT="<PREFILL_API_PORT>"
        NODE_RANK="<PREFILL_NODE_RANK>"
        MASTER_ADDR="<PREFILL_MASTER_IP>"
        MASTER_PORT="<PREFILL_MASTER_PORT>"

        local_ip='<PREFILL_LOCAL_IP>'
        dp_rank=0
        PREFILL_DP_RPC_PORT='<PREFILL_DP_RPC_PORT>'
        worker_args=()
        if [[ "$NODE_RANK" != 0 ]]; then
            worker_args+=(--headless)
        fi
        nic_name='<PREFILL_NIC_NAME>'

        unset ftp_proxy FTP_PROXY https_proxy HTTPS_PROXY http_proxy HTTP_PROXY
        unset LD_PRELOAD HCCL_OP_EXPANSION_MODE VLLM_PP_LAYER_PARTITION

        export HCCL_IF_IP="$local_ip"
        export HCCL_SOCKET_IFNAME="$nic_name"
        export GLOO_SOCKET_IFNAME="$nic_name"
        export HCCL_BUFFSIZE=1024
        export HCCL_IF_BASE_PORT='<PREFILL_HCCL_BASE_PORT>'
        export HCCL_HOST_SOCKET_PORT_RANGE='<PREFILL_HCCL_PORT_RANGE>'
        export HCCL_NPU_SOCKET_PORT_RANGE=auto
        export HCCL_CONNECT_TIMEOUT=1200
        export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
        export LD_LIBRARY_PATH="/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:${LD_LIBRARY_PATH:-}"

        source '<MEMCACHE_CONFIG_DIR>/runtime_env.sh'
        export MMC_LOCAL_CONFIG_PATH='<MEMCACHE_CONFIG_DIR>/mmc-local.conf'
        test -r "$MMC_LOCAL_CONFIG_PATH"
        export VLLM_PP_LAYER_PARTITION="25,25,24,19"
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export PYTHONHASHSEED=0
        export VLLM_HOST_IP="$local_ip"
        export VLLM_USE_V2_MODEL_RUNNER=1
        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
        export FLA_STABLE_DEBUG_DESC=0

        exec vllm serve '/root/.cache/Kimi-K3.1-Pro-w4a8' \
            --host 0.0.0.0 --port "$API_PORT" \
            --quantization ascend \
            --tensor-parallel-size "4" \
            --nnodes 4 \
            --node-rank "$NODE_RANK" \
            --master-addr "$MASTER_ADDR" \
            --master-port "$MASTER_PORT" \
            "${worker_args[@]}" \
            --prefill-context-parallel-size 4 \
            --pipeline-parallel-size 4 \
            --decode-context-parallel-size 1 \
            --data-parallel-size "1" \
            --data-parallel-size-local 1 \
            --data-parallel-rank "$dp_rank" \
            --data-parallel-address "$MASTER_ADDR" \
            --data-parallel-rpc-port "$PREFILL_DP_RPC_PORT" \
            --distributed-executor-backend mp \
            --enable-expert-parallel --seed 1024 \
            --tokenizer-mode kimi_k3 \
            --enable-auto-tool-choice \
            --reasoning-parser kimi_k3 \
            --tool-call-parser kimi_k3 \
            --served-model-name kimi-k3.1-pro \
            --max-model-len 1048576 \
            --max-num-batched-tokens 8192 \
            --max-num-seqs 32 \
            --enforce-eager --trust-remote-code \
            --gpu-memory-utilization 0.9 \
            --block-size 768 \
            --kv-cache-dtype int8 \
            --attention_config.indexer_kv_dtype int8 \
            --speculative-config '{"method":"dspark","model":"<YOUR_DRAFT_MODEL_PATH>","num_speculative_tokens":3,"draft_tensor_parallel_size":4,"draft_sample_method":"greedy","kv_cache_dtype":"bfloat16","enforce_eager":true}' \
            --mm-encoder-tp-mode data \
            --no-mm-device-do-normalize \
            --mm-processor-cache-gb 1 \
            --skip-mm-profiling \
            --enable-prefix-caching \
            --limit-mm-per-prompt '{"vision_chunk":2}' \
            --profiler-config '{"profiler":"torch","torch_profiler_dir":"<PROFILER_OUTPUT_DIR>","torch_profiler_with_stack":false}' \
            --additional-config '{"enable_flashcomm1":true,"multistream_overlap_shared_expert":false,"enable_fused_mc2":0,"enable_shared_expert_dp":true}' \
            --kv-transfer-config '{
              "kv_connector": "MultiConnector",
              "kv_role": "kv_producer",
              "kv_connector_extra_config": {
                "connectors": [
                  {
                    "kv_connector": "MooncakeConnectorV2",
                    "kv_role": "kv_producer",
                    "kv_port": "<PREFILL_KV_PORT>",
                    "kv_connector_extra_config": {
                      "prefill": {
                        "dp_size": 1,
                        "tp_size": 4,
                        "pp_size": 4,
                        "pcp_size": 4,
                        "dcp_size": 1
                      },
                      "decode": {
                        "dp_size": 4,
                        "tp_size": 16,
                        "pp_size": 1,
                        "pcp_size": 1,
                        "dcp_size": 16
                      }
                    }
                  },
                  {
                    "kv_connector": "AscendStoreConnector",
                    "kv_role": "kv_producer",
                    "kv_connector_extra_config": {
                      "backend": "memcache",
                      "lookup_rpc_port": "<PREFILL_LOOKUP_RPC_PORT>",
                      "use_layerwise": false
                    }
                  }
                ]
              }
            }'
        ```

    === "Decode"

        ```bash
        #!/usr/bin/env bash
        set -euo pipefail

        local_ip='<DECODE_LOCAL_IP>'
        nic_name='<DECODE_NIC_NAME>'

        unset ftp_proxy FTP_PROXY https_proxy HTTPS_PROXY http_proxy HTTP_PROXY
        unset LD_PRELOAD VLLM_PP_LAYER_PARTITION

        export HCCL_IF_IP="$local_ip"
        export HCCL_SOCKET_IFNAME="$nic_name"
        export GLOO_SOCKET_IFNAME="$nic_name"
        export HCCL_BUFFSIZE=1024
        export HCCL_OP_EXPANSION_MODE=AIV
        export HCCL_CONNECT_TIMEOUT=1200
        export ASCEND_RT_VISIBLE_DEVICES="$1"
        export LD_LIBRARY_PATH="/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:${LD_LIBRARY_PATH:-}"

        source '<MEMCACHE_CONFIG_DIR>/runtime_env.sh'
        export MMC_LOCAL_CONFIG_PATH='<MEMCACHE_CONFIG_DIR>/mmc-local.conf'
        test -r "$MMC_LOCAL_CONFIG_PATH"

        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
        export PYTHONHASHSEED=0
        export VLLM_HOST_IP="$local_ip"
        export VLLM_USE_V2_MODEL_RUNNER=1
        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
        export FLA_STABLE_DEBUG_DESC=0

        exec vllm serve '/root/.cache/Kimi-K3.1-Pro-w4a8' \
            --host 0.0.0.0 --port "$2" \
            --quantization ascend \
            --tensor-parallel-size "$7" \
            --data-parallel-size "$3" \
            --data-parallel-rank "$4" \
            --data-parallel-address "$5" \
            --data-parallel-rpc-port "$6" \
            --decode-context-parallel-size 16 \
            --cp-kv-cache-interleave-size 768 \
            --block-size 768 \
            --enable-expert-parallel --seed 1024 \
            --tokenizer-mode kimi_k3 \
            --enable-auto-tool-choice \
            --reasoning-parser kimi_k3 \
            --tool-call-parser kimi_k3 \
            --served-model-name kimi-k3.1-pro \
            --max-model-len 1048576 \
            --max-num-batched-tokens 8192 \
            --max-num-seqs 32 \
            --trust-remote-code \
            --gpu-memory-utilization 0.80 \
            --kv-cache-dtype int8 \
            --attention_config.indexer_kv_dtype int8 \
            --speculative-config '{"method":"dspark","model":"<YOUR_DRAFT_MODEL_PATH>","num_speculative_tokens":3,"draft_tensor_parallel_size":16,"draft_sample_method":"greedy","kv_cache_dtype":"bfloat16","enforce_eager":false}' \
            --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
            --mm-encoder-tp-mode data \
            --no-mm-device-do-normalize \
            --mm-processor-cache-gb 1 \
            --skip-mm-profiling \
            --enable-prefix-caching \
            --limit-mm-per-prompt '{"vision_chunk":2}' \
            --profiler-config '{"profiler":"torch","torch_profiler_dir":"<PROFILER_OUTPUT_DIR>","torch_profiler_with_stack":false}' \
            --additional-config '{"multistream_overlap_shared_expert":true,"enable_mlapo":true,"scheduler_config":{"recompute_scheduler_enable":true}}' \
            --kv-transfer-config '{
              "kv_connector": "MultiConnector",
              "kv_role": "kv_consumer",
              "kv_connector_extra_config": {
                "connectors": [
                  {
                    "kv_connector": "MooncakeConnectorV2",
                    "kv_role": "kv_consumer",
                    "kv_port": "<DECODE_KV_PORT>",
                    "kv_connector_extra_config": {
                      "prefill": {
                        "dp_size": 1,
                        "tp_size": 4,
                        "pp_size": 4,
                        "pcp_size": 4,
                        "dcp_size": 1
                      },
                      "decode": {
                        "dp_size": 4,
                        "tp_size": 16,
                        "pcp_size": 1,
                        "dcp_size": 16
                      }
                    }
                  },
                  {
                    "kv_connector": "AscendStoreConnector",
                    "kv_role": "kv_consumer",
                    "kv_connector_extra_config": {
                      "backend": "memcache",
                      "lookup_rpc_port": "<DECODE_LOOKUP_RPC_PORT>",
                      "use_layerwise": false
                    }
                  }
                ]
              }
            }'
        ```

=== "950DT Products"

    The 950DT Products pooled templates use `MultiConnector` with `MooncakeConnectorV2` and `AscendStoreConnector`, and set `use_layerwise=false`. They inherit `MMC_LOCAL_CONFIG_PATH` from the launch environment, preserving the node configuration path set in [Section 5.3.2](#532-prepare-the-memcache-configuration-files).

    See [Section 5.3.7](#537-verify-the-pool-and-model-service) for configuration and cache verification after startup.

    !!! note "Lookup Identifier Assignments"

        The 950DT Products pooled launch commands follow [Section 5.1.1](#511-create-the-engine-templates): DP1 / PP4 / PCP8 / TP1 on Prefill and DP4 / TP8 / DCP8 on Decode, with matching connector topology. Both roles use maximum length `172032`. All four PP4 Prefill nodes use `LOOKUP_RPC_PORT=0`. The four DP4 Decode nodes use `4/5/6/7` for DP ranks `0/1/2/3`, respectively, calculated as `$4 + 4` in the template. These lookup identifiers identify local IPC endpoints, not PP ranks or MemCache registration ranks.

    The 950DT Products pooled templates enable DSpark3. Fill in the local directory of a draft checkpoint compatible with the 950DT Products target model. Use the same draft checkpoint on both roles, with draft TP `1/8` on Prefill/Decode matching each target TP, three speculative tokens, draft `enforce_eager=false`, and `kv_cache_dtype=auto`. The Prefill target model retains `--enforce-eager`. Model, cache, quantization, and scheduling settings follow [Section 5.1.1](#511-create-the-engine-templates); only maximum length and pool configuration change. Configure optional reasoning and tool-call flags as described in [Section 5.1.1](#511-create-the-engine-templates), using `kimi_k3` parsers.

    === "Prefill"

        Save the Prefill template below as `run_dp_template.sh`. Fill in the local IP, NIC, model paths, KV/HCCL ports, and the top-level `API_PORT`, `NODE_RANK`, `MASTER_ADDR`, and `MASTER_PORT` variables, then start it directly with Bash. The script automatically sets `--headless` on P1–P3.

        ```bash
        #!/usr/bin/env bash
        set -e

        API_PORT="<PREFILL_API_PORT>"
        NODE_RANK="<PREFILL_NODE_RANK>"
        MASTER_ADDR="<PREFILL_MASTER_IP>"
        MASTER_PORT="<PREFILL_MASTER_PORT>"

        unset ftp_proxy FTP_PROXY
        unset https_proxy HTTPS_PROXY
        unset http_proxy HTTP_PROXY

        nic_name="<PREFILL_NIC_NAME>"
        local_ip="<PREFILL_LOCAL_IP>"
        KV_PORT="<PREFILL_KV_PORT>"
        ENGINE_ID="<PREFILL_ENGINE_ID>"
        DRAFT_MODEL_PATH="<YOUR_DRAFT_MODEL_PATH>"

        headless_args=()
        if (( NODE_RANK > 0 )); then
            headless_args=(--headless)
        fi

        export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        export HCCL_IF_IP="$local_ip"
        export VLLM_HOST_IP="$local_ip"
        export GLOO_SOCKET_IFNAME="$nic_name"
        export TP_SOCKET_IFNAME="$nic_name"
        export HCCL_SOCKET_IFNAME="$nic_name"
        export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'
        export VLLM_RPC_TIMEOUT=3600000
        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
        export HCCL_EXEC_TIMEOUT=2040
        export HCCL_CONNECT_TIMEOUT=1200
        export HCCL_BUFFSIZE=2000
        export OMP_PROC_BIND=false
        export OMP_NUM_THREADS=10
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export TASK_QUEUE_ENABLE=1

        SPECULATIVE_CONFIG="$(
          printf \
          '{"method":"dspark","model":"%s","num_speculative_tokens":3,"draft_tensor_parallel_size":1,"draft_sample_method":"greedy","enforce_eager":false,"kv_cache_dtype":"auto"}' \
          "$DRAFT_MODEL_PATH"
        )"

        export HCCL_OP_EXPANSION_MODE="CCU_SCHED"
        export VLLM_USE_V2_MODEL_RUNNER=1
        export VLLM_PP_LAYER_PARTITION=24,24,24,21

        LOOKUP_RPC_PORT=0
        : "${MMC_LOCAL_CONFIG_PATH:?Load the node-specific pool environment before starting vLLM}"
        test -r "$MMC_LOCAL_CONFIG_PATH"
        export LD_LIBRARY_PATH="/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:${LD_LIBRARY_PATH:-}"

        vllm serve "/root/.cache/Kimi-K3.1-Pro-w4a8-mxfp" \
            --host 0.0.0.0 \
            --port "$API_PORT" \
            --data-parallel-size "1" \
            --tensor-parallel-size "1" \
            --prefill-context-parallel-size 8 \
            --nnodes 4 \
            --node-rank "$NODE_RANK" \
            --master-addr "$MASTER_ADDR" \
            --master-port "$MASTER_PORT" \
            --pipeline-parallel-size 4 \
            "${headless_args[@]}" \
            --block-size 1536 \
            --cp-kv-cache-interleave-size 1536 \
            --served-model-name kimi-k3.1-pro \
            --allowed-local-media-path / \
            --trust-remote-code \
            --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":128}' \
            --enable-expert-parallel \
            --max-num-seqs 64 \
            --max-model-len 172032 \
            --max-num-batched-tokens 16384 \
            --gpu-memory-utilization 0.90 \
            --enable-prefix-caching \
            --kv-cache-dtype fp8 \
            --quantization ascend \
            --attention_config.indexer_kv_dtype fp8 \
            --tokenizer-mode kimi_k3 \
            --mm-processor-cache-gb 0 \
            --mm-encoder-tp-mode data \
            --no-mm-device-do-normalize \
            --enforce-eager \
            --additional-config '{"enable_mlapo":false,"enable_flashcomm1":true,"enable_shared_expert_dp":true,"multistream_overlap_shared_expert":false,"enable_fused_mc2":0,"enable_force_eplb":false}' \
            --profiler-config '{"profiler":"torch","torch_profiler_dir":"<PROFILER_OUTPUT_DIR>","torch_profiler_with_stack":false}' \
            --kv-transfer-config '{
          "kv_connector": "MultiConnector",
          "kv_role": "kv_producer",
          "kv_connector_extra_config": {
            "connectors": [
              {
                "kv_connector": "MooncakeConnectorV2",
                "kv_role": "kv_producer",
                "kv_port": "'"$KV_PORT"'",
                "engine_id": "'"$ENGINE_ID"'",
                "kv_connector_extra_config": {
                  "prefill": {
                    "pp_size": 4,
                    "pcp_size": 8,
                    "dp_size": 1,
                    "tp_size": 1
                  },
                  "decode": {
                    "dp_size": 4,
                    "tp_size": 8
                  }
                }
              },
              {
                "kv_connector": "AscendStoreConnector",
                "kv_role": "kv_producer",
                "kv_connector_extra_config": {
                  "backend": "memcache",
                  "use_layerwise": false,
                  "lookup_rpc_port": "'"$LOOKUP_RPC_PORT"'"
                }
              }
            ]
          }
        }' \
            --speculative-config "$SPECULATIVE_CONFIG"
        ```

    === "Decode"

        Save the Pooled Decode template below as `run_dp_template.sh` on the corresponding nodes. Fill in the local IP, network interface, KV port, and target and draft model paths, and replace `<PROFILER_OUTPUT_DIR>` with a writable profiling output directory.

        ```bash
        #!/usr/bin/env bash
        set -e

        unset ftp_proxy FTP_PROXY
        unset https_proxy HTTPS_PROXY
        unset http_proxy HTTP_PROXY

        nic_name="<DECODE_NIC_NAME>"
        local_ip="<DECODE_LOCAL_IP>"
        KV_PORT="<DECODE_KV_PORT>"
        ENGINE_ID="<DECODE_ENGINE_ID>"
        DRAFT_MODEL_PATH="<YOUR_DRAFT_MODEL_PATH>"

        export ASCEND_RT_VISIBLE_DEVICES="$1"
        export HCCL_IF_IP="$local_ip"
        export VLLM_HOST_IP="$local_ip"
        export GLOO_SOCKET_IFNAME="$nic_name"
        export TP_SOCKET_IFNAME="$nic_name"
        export HCCL_SOCKET_IFNAME="$nic_name"
        export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'
        export VLLM_RPC_TIMEOUT=3600000
        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
        export HCCL_EXEC_TIMEOUT=2040
        export HCCL_CONNECT_TIMEOUT=1200
        export HCCL_BUFFSIZE=800
        export OMP_PROC_BIND=false
        export OMP_NUM_THREADS=10
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export TASK_QUEUE_ENABLE=1

        SPECULATIVE_CONFIG="$(
          printf \
          '{"method":"dspark","model":"%s","num_speculative_tokens":3,"draft_tensor_parallel_size":8,"draft_sample_method":"greedy","enforce_eager":false,"kv_cache_dtype":"auto"}' \
          "$DRAFT_MODEL_PATH"
        )"

        export HCCL_OP_EXPANSION_MODE="CCU_SCHED"
        export VLLM_USE_V2_MODEL_RUNNER=1

        LOOKUP_RPC_PORT=$(($4 + 4))
        : "${MMC_LOCAL_CONFIG_PATH:?Load the node-specific pool environment before starting vLLM}"
        test -r "$MMC_LOCAL_CONFIG_PATH"
        export LD_LIBRARY_PATH="/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:${LD_LIBRARY_PATH:-}"

        vllm serve "/root/.cache/Kimi-K3.1-Pro-w4a8-mxfp" \
            --host 0.0.0.0 \
            --port "$2" \
            --data-parallel-size "$3" \
            --data-parallel-rank "$4" \
            --data-parallel-address "$5" \
            --data-parallel-rpc-port "$6" \
            --tensor-parallel-size "$7" \
            --decode-context-parallel-size 8 \
            --block-size 1536 \
            --cp-kv-cache-interleave-size 1536 \
            --served-model-name kimi-k3.1-pro \
            --allowed-local-media-path / \
            --trust-remote-code \
            --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":128}' \
            --enable-expert-parallel \
            --max-num-seqs 64 \
            --max-model-len 172032 \
            --max-num-batched-tokens 768 \
            --gpu-memory-utilization 0.92 \
            --enable-prefix-caching \
            --kv-cache-dtype fp8 \
            --quantization ascend \
            --attention_config.indexer_kv_dtype fp8 \
            --tokenizer-mode kimi_k3 \
            --mm-processor-cache-gb 0 \
            --mm-encoder-tp-mode data \
            --no-mm-device-do-normalize \
            --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
            --additional-config '{"enable_mlapo":true,"multistream_overlap_shared_expert":true,"scheduler_config":{"recompute_scheduler_enable":true}}' \
            --profiler-config '{"profiler":"torch","torch_profiler_dir":"<PROFILER_OUTPUT_DIR>","torch_profiler_with_stack":false}' \
            --kv-transfer-config '{
          "kv_connector": "MultiConnector",
          "kv_role": "kv_consumer",
          "kv_connector_extra_config": {
            "connectors": [
              {
                "kv_connector": "MooncakeConnectorV2",
                "kv_role": "kv_consumer",
                "kv_port": "'"$KV_PORT"'",
                "engine_id": "'"$ENGINE_ID"'",
                "kv_connector_extra_config": {
                  "prefill": {
                    "pp_size": 4,
                    "pcp_size": 8,
                    "dp_size": 1,
                    "tp_size": 1
                  },
                  "decode": {
                    "dp_size": 4,
                    "tp_size": 8
                  }
                }
              },
              {
                "kv_connector": "AscendStoreConnector",
                "kv_role": "kv_consumer",
                "kv_connector_extra_config": {
                  "backend": "memcache",
                  "use_layerwise": false,
                  "lookup_rpc_port": "'"$LOOKUP_RPC_PORT"'"
                }
              }
            ]
          }
        }' \
            --speculative-config "$SPECULATIVE_CONFIG"
        ```

#### 5.3.6 Start the Engines and Proxy {#536-start-the-engines-and-proxy}

=== "A3 series"

    Before starting pooled engines, ensure MetaService and all eight standalone providers are ready. Use the pooled templates in this section, which load `runtime_env.sh` and the client configuration. Keep A3 Prefill `<PREFILL_DP_RPC_PORT>` identical on all four P nodes and distinct from `MASTER_PORT`; Prefill DP rank remains `0`.

    **Start Prefill (P0–P3)**

    Save the matching Prefill template as `run_dp_template.sh` on each P node, fill in its variables, and run it directly without positional arguments. This deployment uses DP=1 and PP=4 on Prefill, with one PP stage per node.

    This A3 deployment uses DP=1, PP=4, TP=4, and PCP=4 on Prefill, with 16 NPU devices per node.

    ```bash
    bash run_dp_template.sh
    ```

    | Prefill script variable | Configuration |
    |---|---|
    | `NODE_RANK` | Set to `0, 1, 2, 3` on P0–P3 respectively; this is the PP node rank. |
    | `MASTER_ADDR` | Use the P0 IP on all four P nodes. |
    | `MASTER_PORT` | Use the same Prefill master port on all four P nodes. |
    | `API_PORT` | P0 API listening port; register the P0 IP and port with the Proxy. P1–P3 use `--headless` and do not expose an API service. |

    The template automatically adds `--headless` on P1–P3. Start P0 before P1–P3, keeping model paths, layer partitioning, and parallelism settings consistent across all four P nodes.

    **Start Decode (D0–D3)**

    Deploy `launch_online_dp.py` on the D nodes and save the matching Decode template as `run_dp_template.sh` in the same directory. Run the following command from that directory to start one DP instance per node:

    This guide uses the repository launcher that passes seven positional arguments. DCP is explicitly set in the Decode template: `16` for A3 and `8` for 950DT Products. Use a launcher that matches the template; do not substitute versions that pass parameters through `$8` or `AICLUSTERHUB_PCP_SIZE` / `AICLUSTERHUB_DCP_SIZE` without adapting the template.

    ```bash
    python launch_online_dp.py \
        --dp-size 4 \
        --tp-size 16 \
        --dp-size-local 1 \
        --dp-rank-start '<DECODE_DP_RANK>' \
        --dp-address '<DECODE_MASTER_IP>' \
        --dp-rpc-port '<DECODE_DP_RPC_PORT>' \
        --vllm-start-port '<DECODE_API_PORT>'
    ```

    Set `<DECODE_DP_RANK>` to `0, 1, 2, 3` on D0–D3 respectively. Use the D0 IP for `<DECODE_MASTER_IP>` and the same `<DECODE_DP_RPC_PORT>` on all four D nodes. Set `<DECODE_API_PORT>` to each node's actual API port. Start D0 before D1–D3.

    The launcher still passes Decode arguments `$1`–`$7`: device list, API port, DP size, DP rank, Decode master IP, DP RPC port, and TP size. Configure master addresses and ports separately for P and D, and register the actual API endpoints with the Proxy.

    The supplied lookup identifiers are `33101` on Prefill and `33501` on Decode, separate from the Mooncake KV ports `33100` and `33500`. `lookup_rpc_port` identifies the local pool scheduler/worker IPC endpoint; it is not the MetaService address, PP node rank, or an automatically assigned MemCache registration rank. One instance per separate host can reuse its role's identifier. Give additional instances on the same host distinct identifiers and non-conflicting service ports.

    Once P0 and all four Decode APIs are ready, start the Proxy with the command in [Section 5.1.2](#512-start-the-engines-and-proxy), registering only P0 as the prefiller and D0–D3 as decoders. Use their actual API ports. Clients use the same `kimi-k3.1-pro` name through the Proxy.

    | Setting | A3 pooled Prefill | A3 pooled Decode |
    |---|---|---|
    | DP / TP / PP | `1 / 4 / 4` | `4 / 16 / 1` |
    | PCP / DCP | `4 / 1` | `1 / 16` |
    | Maximum model length | `1048576` | `1048576` |
    | Block size / CP interleave | `768` / not explicitly set | `768 / 768` |
    | Main / indexer cache dtype | `int8 / int8` | `int8 / int8` |
    | Batched tokens / maximum sequences | `8192 / 32` | `8192 / 32` |
    | GPU memory utilization | `0.90` | `0.80` |
    | Graph mode | Eager | `FULL_DECODE_ONLY` |
    | DSpark tokens / draft TP | `3 / 4` | `3 / 16` |
    | Draft cache / eager | `bfloat16 / true` | `bfloat16 / false` |
    | `enable_flashcomm1` / shared-expert DP | `true / true` | Not explicitly set |
    | Shared-expert overlap / MLAPO | `false` / not explicitly set | `true / true` |
    | Allocator expandable segments | `True` | `False` |
    | Pool backend / layerwise | `memcache / false` | `memcache / false` |
    | Pool role | `kv_producer` | `kv_consumer` |

    The pooled reference has different sequence limits, FlashComm/shared-expert settings, and allocator settings from the basic A3 templates. Keep the pooled template's settings together rather than replacing only the connector in [Section 5.1](#51-pd-deployment).

=== "950DT Products"

    Start MetaService as described in [Section 5.3.3](#533-start-metaservice). In the prepared model and MF runtime environment on each P/D node, replace the path below with the absolute path of the local configuration prepared in [Section 5.3.2](#532-prepare-the-memcache-configuration-files) and run it in the actual launch shell:

    ```bash
    export MMC_LOCAL_CONFIG_PATH='<ABSOLUTE_NODE_CONFIG_PATH>/mmc-local.conf'
    ```

    In the same shell, follow [Section 5.1.2](#512-start-the-engines-and-proxy) to start P/D and the Proxy using the pooled templates: direct Bash for Prefill and the matching Python launcher for Decode. Child processes inherit the environment variable. If an existing environment script sets it, reuse that setting after verifying that it points to the local node configuration.

    Keep maximum length `172032` on both roles. Lookup identifiers are `0` on P0–P3 and `4, 5, 6, 7` on D0–D3. P1–P3 use `--headless`; register only P0 and the four Decode APIs with the Proxy.

#### 5.3.7 Verify the Pool and Model Service {#537-verify-the-pool-and-model-service}

=== "A3 series"

    The registered capacities below assume the reference configuration of 1 GiB per provider. After changing DRAM capacity, verify registration against the actual configuration rather than a fixed 8 GiB total.


    Check the current provider PID and its current initialization log on every node. Query the configured MetaService, bypassing HTTP proxies:

    ```bash
    curl --noproxy '*' -fsS --max-time 10 \
        'http://<META_SERVICE_IP>:8099/metrics' \
        | grep -E '^memcache_(segment_capacity_bytes|total_capacity_bytes)'
    ```

    For the eight-provider 1 GiB reference, expect eight DRAM segments of `1073741824` bytes and a total of `8589934592` bytes. MemCache registration ranks are dynamic; do not map a segment rank directly to a model P/D rank. MetaService metrics establish provider registration and capacity, not model-side cache hits.

    After the engines and Proxy are ready, send the text and multimodal requests in [Section 6](#6-functional-verification) using `kimi-k3.1-pro`. Inspect the active engine configuration for `MultiConnector`, `backend=memcache`, `use_layerwise=false`, and the client `MMC_LOCAL_CONFIG_PATH`. Repeat a request with the same sufficiently long prefix and correlate the engine's external-cache lookup/load/save logs or metrics with the request and the returned output. Local prefix caching is also enabled; a faster repeated request alone does not prove a MemCache hit.

    Validate model-side pool reuse, output correctness, and accuracy before recording the deployment as accepted. Measure performance separately with the same runtime, topology, and cache state. Starting the provider or passing `bash -n` does not establish those results.

=== "950DT Products"

    Do not apply the A3 eight-standalone-provider or 8 GiB registration criteria to 950DT Products. Inspect MetaService and actual engine/worker logs for the effective `MMC_LOCAL_CONFIG_PATH`, the matching node configuration adjusted to available capacity (reference value: `1GB`), cache initialization, and observed registered capacity. Do not infer registration from node count alone. Then validate request outputs and external-cache reuse through the Proxy, recording accuracy and performance separately.

## 6 Functional Verification {#6-functional-verification}

### 6.1 Text and Multimodal Requests {#61-text-and-multimodal-requests}

Once the P/D engines and Proxy are ready, replace `<PROXY_IP>` and `<PROXY_PORT>` with the actual service address and the port used to start the Proxy. Both the request's `model` field and `--served-model-name` are `kimi-k3.1-pro`.

```bash
curl "http://<PROXY_IP>:<PROXY_PORT>/v1/models"

curl "http://<PROXY_IP>:<PROXY_PORT>/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "kimi-k3.1-pro",
    "messages": [{"role": "user", "content": "Introduce yourself in one sentence."}],
    "max_tokens": 1024
  }'
```

Image request example (replace `<IMAGE_URL_OR_DATA_URL>` with an accessible image URL or a valid image data URL):

```bash
curl "http://<PROXY_IP>:<PROXY_PORT>/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "kimi-k3.1-pro",
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

The request should return HTTP 200, with generated content in the response's `choices` field. For image requests, the response should match the image. If generation stops at the output token limit, adjust the output budget for the reasoning mode in use.

## 7 Accuracy Evaluation {#7-accuracy-evaluation}

The following accuracy results were reported for Kimi K3.1 Pro with PD separation.

| dataset | version | metric | mode | vllm-api-general-chat | note |
|---|---|---|---|---|---|
| GPQA | - | accuracy | gen | 92.42 | Atlas 800I A3, PD separation, 64 cards |
| MMMU PRO | - | accuracy | gen | 80.75 | Atlas 800I A3, PD separation, 64 cards |
| GPQA | - | accuracy | gen | 94.44 | 950DT Products, PD separation, 64 cards |
| MMMU PRO | - | accuracy | gen | 81.40 | 950DT Products, PD separation, 64 cards |

### 7.1 Using AISBench {#71-using-aisbench}

1. Refer to the [AISBench guide](../../developer_guide/evaluation/using_ais_bench.md) to prepare the environment and evaluation configuration.
2. Run evaluation through the Proxy API and save the configuration, generated outputs, and evaluation report.
3. When comparing results, keep the model and tokenizer revisions, chat template, sampling settings, output length, dataset, and evaluation-tool version fixed.

## 8 Performance Evaluation {#8-performance-evaluation}

### 8.1 Using AISBench {#81-using-aisbench}

Refer to the [AISBench performance evaluation guide](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) to prepare the environment and test configuration.

### 8.2 Using vLLM Benchmark {#82-using-vllm-benchmark}

The following minimal random-text benchmark sends 16 requests through the Proxy with concurrency 1, 1024 input tokens, and 128 output tokens. Replace the service address with the actual Proxy endpoint, set `<LOCAL_MODEL_PATH>` to the platform checkpoint directory accessible to the benchmark client, and choose a writable results directory.

```bash
vllm bench serve \
    --base-url 'http://<PROXY_IP>:<PROXY_PORT>' \
    --endpoint /v1/completions \
    --model kimi-k3.1-pro \
    --tokenizer '<LOCAL_MODEL_PATH>' \
    --tokenizer-mode kimi_k3 \
    --trust-remote-code \
    --dataset-name random \
    --random-input-len 1024 \
    --random-output-len 128 \
    --num-prompts 16 \
    --max-concurrency 1 \
    --request-rate inf \
    --ignore-eos \
    --seed 0 \
    --save-result \
    --result-dir '<BENCHMARK_RESULT_DIR>'
```

After a successful run, check that 16 requests succeeded and record request throughput (requests/s), output-token throughput (tokens/s), time to first token (TTFT, ms), and time per output token after the first (TPOT, ms). Result files are saved under `--result-dir`. This small workload verifies the benchmark procedure; adjust lengths, concurrency, and request count to represent the target workload.

See the [vLLM Benchmark guide](../../developer_guide/performance_and_debug/performance_benchmark.md) for more parameters, and check options using `vllm bench serve --help` from the installed version. Keep test data, concurrency, lengths, cache state, and metric definitions consistent when comparing results, and save the commands used.

## 9 Performance Tuning {#9-performance-tuning}

### 9.1 Recommended Configurations {#91-recommended-configurations}

Use the configurations in [Chapter 5, Online Service Deployment](#5-online-service-deployment) as the starting point for tuning. The optimal configuration depends on the workload and hardware environment.

### 9.2 Tuning Guide {#92-tuning-guide}

Use the deployment settings above as a baseline. Adjust `max-model-len`, `max-num-seqs`, `max-num-batched-tokens`, and `gpu-memory-utilization` together for the target workload.

Refer to the [performance tuning guide](../../developer_guide/performance_and_debug/optimization_and_tuning.md) and [feature matrix](../../user_guide/support_matrix/feature_matrix.md) for additional guidance.

## 10 FAQ {#10-faq}

For common environment, installation, and parameter questions, refer to the [public FAQ](../../faqs.md).

- **Q: Which multimodal inputs are supported by the current Kimi K3.1 Pro implementation?** A: This guide provides configuration and verification examples for image input. See [Section 6](#6-functional-verification).

- **Q: Which server options are required for Kimi K3.1 Pro reasoning and tool calling?** A: Configure `--tokenizer-mode`, `--enable-auto-tool-choice`, `--reasoning-parser`, and `--tool-call-parser` together. Use `kimi_k3` for the tokenizer and parsers on both A3 and 950DT Products.

- **Q: How should TP size be selected?** A: The attention-head count in the checkpoint must be divisible by TP size. TP also changes the KDA state layout and expert placement; check memory requirements and communication performance together.

- **Q: How is DSpark enabled in PD separation?** A: Prepare a DSpark draft checkpoint compatible with the target model on this platform and replace `<YOUR_DRAFT_MODEL_PATH>` in both templates with its local directory. Apply the role-specific `--speculative-config` from Section [5.1](#51-pd-deployment), [5.2](#52-long-context), or [5.3](#53-pooled-pd). Keep the draft checkpoint and `num_speculative_tokens` identical on both sides, and retain `--enforce-eager` for the Prefill target model.

- **Q: Which option is required for multimodal image input?** A: Add `--no-mm-device-do-normalize` to every engine startup command, including Prefill, Decode, and headless workers. This keeps image pixel rescaling and normalization on the CPU, because the current Ascend Kimi vision wrapper does not apply device-side normalization. Omitting this option can degrade image accuracy.
