# Hy4 (Experimental)

## 1 Introduction

!!! note

    PD separation is provided as a reference deployment. For customer production deployments, follow the [MindIE Motor](https://www.hiascend.com/document/detail/en/mindie/latest/quickstart/docs/en/user_guide/quick_start.md) + vLLM Ascend deployment guidance.

This release has been validated only on 950DT products. A2 and A3 have not yet been validated; support is planned for future releases.

This guide describes Hy4 FP8 deployment with vLLM Ascend on 950DT products. It covers single-node co-located serving, two-node co-located serving, and four-node Prefill/Decode (PD) separation. Each host uses eight NPUs.

The serving examples use compressed-tensors FP8 weights, expert parallelism, prefix caching, and MTP speculative decoding. The four-node PD configuration has been run successfully on 950DT products and used for measurement. It remains an experimental reference deployment.

Hy4 uses the DFly2 architecture for its speculative module, not native MTP. It reuses vLLM's MTP method name for configuration: set `"method": "mtp"` in the speculative config to enable DFly2 speculative decoding.

## 2 Supported Features

Refer to the [Supported Models](../../user_guide/support_matrix/supported_models.md) page for the support matrix and the [Feature Guide](../../user_guide/feature_guide/index.md) for feature configuration.

## 3 Prerequisites

### 3.1 Model Weight

Use the `tencent/Hy4-FP8` checkpoint from [Hugging Face](https://huggingface.co/tencent/Hy4-FP8) or [ModelScope](https://modelscope.cn/models/Tencent-Hunyuan/Hy4-FP8). Download the complete checkpoint, including its tokenizer, model configuration, multimodal processor files, and weights required by MTP. Make it accessible at `/path/to/Hy4-FP8` inside every serving container. Use the same checkpoint revision on every host.

### 3.2 Verify Multi-node Communication

For multi-node deployment, follow [multi-node interconnect preparation](../../getting_started/installation.md#installation-multi-node-interconnect). Choose mutually reachable host IPs and the corresponding local network interfaces. Ensure HCCL connectivity and access to the API, DP coordinator, and KV transfer ports.

## 4 Installation

### 4.1 Docker Image Installation

Use the Hy4 openEuler image `quay.io/ascend/vllm-ascend:hy4-1120-a5-openeuler` on 950DT products. The deployment was validated with the following software versions:

| Hardware | Ascend HDK | CANN version | torch_npu version | vLLM | vLLM-Ascend version |
| --- | --- | --- | --- | --- | --- |
| 950DT products | 25.1.rc3 | CANN 9.1.0 | 2.10.0.post4 | v0.30.0 | Based on v0.30.0 |

Use matching vLLM, vLLM Ascend, CANN, and torch-npu versions on every host.

Follow [Installation](../../getting_started/installation.md) for the container setup appropriate to the hardware. On every host:

- Expose NPU devices 0–7 and the required management devices.
- Mount the Ascend driver, firmware, and communication configuration required by the hardware.
- Use host networking and sufficient shared memory.
- Mount the weights at /path/to/Hy4-FP8.
- Make the matching repositories accessible at /path/to/vllm-ascend and /path/to/vllm. PD scripts use both through PYTHONPATH.

The following container template uses the Hy4 openEuler image for 950DT products. Use the matching device and communication mounts for your environment. The host model and repository directories must already contain the complete checkpoint and matching code.

```bash
export IMAGE="quay.io/ascend/vllm-ascend:hy4-1120-a5-openeuler"
export CONTAINER_NAME=hy4
export MODEL_DIR="/path/to/Hy4-FP8"
export WORK_DIR="/path/to/hy4-workspace"
mkdir -p "$WORK_DIR"

docker run \
    -it \
    --name "$CONTAINER_NAME" \
    --network host \
    --ipc host \
    --shm-size=100g \
    --device /dev/davinci_manager \
    --device /dev/hisi_hdc \
    --device /dev/ummu \
    --device /dev/uburma \
    --device /dev/davinci0 \
    --device /dev/davinci1 \
    --device /dev/davinci2 \
    --device /dev/davinci3 \
    --device /dev/davinci4 \
    --device /dev/davinci5 \
    --device /dev/davinci6 \
    --device /dev/davinci7 \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro \
    -v /usr/local/sbin:/usr/local/sbin:ro \
    -v /etc/hccn.conf:/etc/hccn.conf:ro \
    -v /usr/local/dcmi:/usr/local/dcmi:ro \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
    -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
    -v /etc/hixlep:/etc/hixlep:ro \
    -v /etc/hccl_rootinfo.json:/etc/hccl_rootinfo.json:ro \
    -v "$MODEL_DIR":/path/to/Hy4-FP8:ro \
    -v "$WORK_DIR":/workspace \
    -v /path/to/vllm-ascend:/path/to/vllm-ascend \
    -v /path/to/vllm:/path/to/vllm \
    -w /workspace \
    "$IMAGE" \
    bash
```

Run the following commands inside the prepared containers. Replace paths beginning with /path/to/, all address placeholders, and interface placeholders before execution. Use the same weights and software versions on every participating host.

## 5 Online Service Deployment

### 5.1 Single-Node Co-located Deployment

=== "950DT products"

    Run on the single serving host. Provide sufficient host RAM for prefetch weight offloading; the RAM requirement remains to be measured.

    ```bash
    set -euo pipefail

    set +u
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    set -u
    cd /path/to/vllm-ascend

    MODEL=/path/to/Hy4-FP8
    PORT=8000

    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    export HCCL_BUFFSIZE=128
    export HCCL_OP_EXPANSION_MODE=AIV
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export VLLM_ENGINE_READY_TIMEOUT_S=10800

    exec vllm serve "${MODEL}" \
        --host 0.0.0.0 \
        --port "${PORT}" \
        --data-parallel-size 1 \
        --tensor-parallel-size 8 \
        --served-model-name Hy4 \
        --enable-expert-parallel \
        --distributed-executor-backend mp \
        --quantization compressed-tensors \
        --safetensors-load-strategy prefetch \
        --max-num-seqs 4 \
        --max-model-len 4096 \
        --max-num-batched-tokens 2048 \
        --gpu-memory-utilization 0.95 \
        --kv-cache-memory-bytes 536870912 \
        --offload-backend prefetch \
        --offload-group-size 4 \
        --offload-num-in-group 1 \
        --offload-prefetch-step 1 \
        --enable-prefix-caching \
        --no-async-scheduling \
        --seed 0 \
        --compilation-config '{
          "cudagraph_mode": "FULL_DECODE_ONLY"
        }' \
        --speculative-config '{
          "method": "mtp",
          "num_speculative_tokens": 5,
          "draft_load_config": {}
        }' \
        --additional-config '{
          "weight_nz_mode": 0
        }'
    ```

### 5.2 Two-Node Co-located Deployment

=== "950DT products"

    Save the following as start_colocated.sh on both hosts. Set NODE0_IP to the same coordinator address on both hosts. Set NODE_IP and NIC for the local host. Select the indicated FRONTEND_ARGS for each node: node 0 exposes port 8000 and node 1 is headless.

    Start node 0 first. Once its DP coordinator has started, start node 1 in another terminal. Both processes must remain running; increasing the readiness timeout does not replace starting the peer.

    ```bash
    #!/usr/bin/env bash
    set -euo pipefail

    set +u
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    set -u
    cd /path/to/vllm-ascend

    NODE0_IP="<NODE0_IP>"
    NODE_IP="<LOCAL_IP>"
    NIC="<NIC_NAME>"
    MODEL=/path/to/Hy4-FP8

    # DP2 x TP8 across two hosts, one DP replica per host, EP16.
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    export VLLM_HOST_IP=${NODE_IP}
    export HCCL_IF_IP=${NODE_IP}
    export HCCL_SOCKET_IFNAME=${NIC}
    export GLOO_SOCKET_IFNAME=${NIC}
    export HCCL_BUFFSIZE=1024
    unset HCCL_OP_EXPANSION_MODE
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    # Frontend readiness includes the remote engine; this does not fix a missing peer.
    export VLLM_ENGINE_READY_TIMEOUT_S=10800

    # Node 0:
    FRONTEND_ARGS=(--api-server-count 1 --host 0.0.0.0 --port 8000)
    # Node 1: replace the line above with:
    # FRONTEND_ARGS=(--headless --data-parallel-start-rank 1)

    exec vllm serve "${MODEL}" \
        "${FRONTEND_ARGS[@]}" \
        --tensor-parallel-size 8 \
        --data-parallel-size 2 \
        --data-parallel-size-local 1 \
        --data-parallel-backend mp \
        --data-parallel-address "${NODE0_IP}" \
        --data-parallel-rpc-port 12890 \
        --enable-expert-parallel \
        --served-model-name Hy4 \
        --distributed-executor-backend mp \
        --quantization compressed-tensors \
        --safetensors-load-strategy prefetch \
        --max-num-seqs 256 \
        --max-model-len 163840 \
        --max-num-batched-tokens 8192 \
        --gpu-memory-utilization 0.9 \
        --enable-prefix-caching \
        --enable-chunked-prefill \
        --no-async-scheduling \
        --seed 0 \
        --compilation-config '{
          "cudagraph_mode": "FULL_DECODE_ONLY"
        }' \
        --speculative-config '{
          "method": "mtp",
          "num_speculative_tokens": 5,
          "draft_load_config": {},
          "draft_sample_method": "probabilistic"
        }' \
        --additional-config '{
          "weight_nz_mode": 1,
          "sparse_kv_offload_config": {
            "enabled": false
          },
          "enable_mlapo": true,
          "enable_fused_mc2": 1,
          "ascend_compilation_config": {
            "enable_npugraph_ex": false
          }
        }'
    ```

### 5.3 Four-Node PD Separation

=== "950DT products"

    Use four distinct hosts. P0/P1 form one DP2 × TP8 group, and D0/D1 form another. Only P0 and D0 expose HTTP APIs. The proxy routes requests to those group frontends.

    | Host | DP rank | API port | DP coordinator | KV role | KV rank | KV port |
    | --- | --- | --- | --- | --- | --- | --- |
    | P0 | 0 | 18010 | P0:12890 | kv_producer | 0 | 23000 |
    | P1 | 1, headless | — | P0:12890 | kv_producer | 0 | 23000 |
    | D0 | 0 | 18020 | D0:12891 | kv_consumer | 1 | 24000 |
    | D1 | 1, headless | — | D0:12891 | kv_consumer | 1 | 24000 |

    The KV ranks follow the supplied scripts and identify P/D groups separately from DP ranks. The reference deployment uses five MTP speculative tokens and HCCL_BUFFSIZE=1024 on all four hosts.

    Save the following files in the same directory on every host.

    **config.sh**

    ```bash
    #!/usr/bin/env bash
    ASCEND_REPO=/path/to/vllm-ascend
    VLLM_REPO=/path/to/vllm
    MODEL=/path/to/Hy4-FP8
    QUANTIZATION=compressed-tensors
    NIC="<NIC_NAME>"
    ```

    Set NIC separately on each host if interface names differ.

    **start_common.sh**

    ```bash
    #!/usr/bin/env bash
    # Run the corresponding role script inside each serving container.
    set -eo pipefail
    SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
    source "$SCRIPT_DIR/config.sh"
    if [[ ${1:-} != --dry-run ]]; then source /usr/local/Ascend/ascend-toolkit/set_env.sh; fi
    set -u
    export PYTHONPATH="${ASCEND_REPO}:${VLLM_REPO}:${PYTHONPATH:-}"
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_ENGINE_READY_TIMEOUT_S=3600
    export VLLM_HOST_IP="$NODE_IP" HCCL_IF_IP="$NODE_IP"
    export HCCL_SOCKET_IFNAME=${NIC} GLOO_SOCKET_IFNAME=${NIC} TP_SOCKET_IFNAME=${NIC}
    export HCCL_BUFFSIZE=1024
    unset HCCL_OP_EXPANSION_MODE
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export NO_PROXY='*' no_proxy='*' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    args=(python3 -m vllm.entrypoints.cli.main serve "$MODEL"
        --host 0.0.0.0 --port "$API_PORT"
        --data-parallel-size 2 --data-parallel-size-local 1
        --data-parallel-address "$MASTER_IP" --data-parallel-rpc-port "$DP_PORT"
        --tensor-parallel-size 8 --served-model-name Hy4 --enable-expert-parallel
        --distributed-executor-backend mp --quantization "$QUANTIZATION"
        --max-num-seqs 64 --max-model-len 147456 --max-num-batched-tokens 8192
        --gpu-memory-utilization 0.85 --enable-prefix-caching --async-scheduling --seed 0
        --compilation-config '{
          "cudagraph_mode": "FULL_DECODE_ONLY"
        }'
        --speculative-config '{
          "method": "mtp",
          "num_speculative_tokens": 5,
          "draft_load_config": {}
        }'
        --additional-config '{
          "weight_nz_mode": 1,
          "sparse_kv_offload_config": {
            "enabled": false
          },
          "enable_mlapo": true,
          "enable_fused_mc2": 1,
          "ascend_compilation_config": {
            "enable_npugraph_ex": false
          }
        }'
        --kv-transfer-config "{\"kv_connector\":\"MooncakeConnectorV1\",\"kv_buffer_device\":\"npu\",\"kv_role\":\"$KV_ROLE\",\"kv_rank\":$KV_RANK,\"kv_port\":$KV_PORT,\"kv_connector_extra_config\":{\"prefill\":{\"dp_size\":2,\"tp_size\":8},\"decode\":{\"dp_size\":2,\"tp_size\":8}}}")
    if [ "$RANK" = 1 ]; then args+=(--headless --data-parallel-start-rank 1); fi
    printf '%q ' "${args[@]}"; printf '\n'
    if [ "${1:-}" = --dry-run ]; then exit 0; fi
    exec "${args[@]}" "$@"
    ```

    **Role scripts**

    Create one script per host using the following template and the values in the table. NODE_IP is the local host address; MASTER_IP is P0 for P roles and D0 for D roles.

    ```bash
    #!/usr/bin/env bash
    NODE_IP="<LOCAL_IP>" MASTER_IP="<GROUP_MASTER_IP>" RANK=0 API_PORT=18010 DP_PORT=12890
    KV_ROLE=kv_producer KV_RANK=0 KV_PORT=23000
    source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/start_common.sh"
    ```

    | Script | NODE_IP | MASTER_IP | RANK | API_PORT | DP_PORT | KV_ROLE | KV_RANK | KV_PORT |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | start_p0.sh | P0 IP | P0 IP | 0 | 18010 | 12890 | kv_producer | 0 | 23000 |
    | start_p1.sh | P1 IP | P0 IP | 1 | 18010 | 12890 | kv_producer | 0 | 23000 |
    | start_d0.sh | D0 IP | D0 IP | 0 | 18020 | 12891 | kv_consumer | 1 | 24000 |
    | start_d1.sh | D1 IP | D0 IP | 1 | 18020 | 12891 | kv_consumer | 1 | 24000 |

    Start P0, wait for its DP coordinator, then start P1. Start D0, wait for its DP coordinator, then start D1. Check both frontend health endpoints before starting the proxy:

    ```bash
    curl --fail "http://<P0_IP>:18010/health"
    curl --fail "http://<D0_IP>:18020/health"
    ```

    **Proxy**

    Run on a host that can reach P0 and D0. The proxy script must be present in the matching vLLM Ascend checkout.

    ```bash
    source ./config.sh
    export PYTHONPATH="${ASCEND_REPO}:${VLLM_REPO}:${PYTHONPATH:-}"
    export NO_PROXY='*' no_proxy='*'
    python3 "$ASCEND_REPO/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py" \
        --host 0.0.0.0 --port 8000 \
        --prefiller-hosts "<P0_IP>" --prefiller-ports 18010 \
        --decoder-hosts "<D0_IP>" --decoder-ports 18020
    ```

    After preparing the files, use bash start_p0.sh --dry-run (or the corresponding role script) to inspect commands without starting an engine. A dry run does not validate model loading, graph capture, communication, or KV transfer.

## 6 Functional Verification

### 6.1 Text

For co-located deployment, use the API host. For PD deployment, use the proxy host.

```bash
SERVER_IP="<API_OR_PROXY_IP>"
curl --fail "http://${SERVER_IP}:8000/v1/models"
curl --fail "http://${SERVER_IP}:8000/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Hy4",
        "messages": [{"role": "user", "content": "What is 2 + 2?"}],
        "max_tokens": 64,
        "temperature": 0
    }'
```

Confirm that the response uses model Hy4 and contains a valid answer. For PD, also confirm from both groups' logs that Prefill and Decode execute and KV transfer completes. An HTTP response alone is insufficient evidence of successful PD operation.

### 6.2 Single Image

Use a local image and send it as a base64 data URL to the same chat completions endpoint. The image is read on the curl client; it does not need to be mounted in the serving container. The example uses a PNG image. Change `IMAGE_MIME` to `image/jpeg` for a JPEG image, and replace the question with one appropriate to the image. For a HallusionBench sample, use its image and the corresponding dataset question.

The four-node PD deployment supports multimodal requests and has been tested with image inputs. Send the following request to the proxy host to verify the deployed service.

```bash
SERVER_IP="<API_OR_PROXY_IP>"
IMAGE_PATH=/path/to/image.png
IMAGE_MIME=image/png
IMAGE_BASE64=$(base64 < "$IMAGE_PATH" | tr -d '\n')

curl --fail --show-error --noproxy '*' --max-time 180 \
    "http://${SERVER_IP}:8000/v1/chat/completions" \
    -H "Content-Type: application/json" \
    --data-binary @- <<EOF
{
    "model": "Hy4",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:${IMAGE_MIME};base64,${IMAGE_BASE64}"
                    }
                },
                {
                    "type": "text",
                    "text": "Describe the objects visible in this image."
                }
            ]
        }
    ],
    "max_tokens": 512,
    "temperature": 0,
    "stream": false,
    "chat_template_kwargs": {
        "reasoning_effort": "no_think"
    }
}
EOF
```

Check that `choices[0].message.content` answers the image question and is consistent with the image. Inspect `finish_reason` and `usage` for truncation and token consumption. This single-image request is a functional check; HallusionBench accuracy requires running the dataset with its matching evaluator.

## 7 Accuracy Evaluation

### 7.1 Using AISBench

Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) for accuracy and performance evaluation. Use the serving model name Hy4 and direct PD requests to the proxy.

Record the checkpoint revision, software versions, topology, dataset, and sampling settings with the evaluation results.

### 7.2 Text Evaluation

The following results use the DP2 + TP8 + EP service with FP8 weights, DFly 5, with prefix caching, graph and asynchronous scheduling enabled.

| Dataset | Hardware | Score | max-model-len | max-num-seqs | max_out_len | batch_size | generation_kwargs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GPQA-Diamond | Ascend 950DT | 182/198 (91.92%) | 131072 | 256 | 131072 | 256 | model default config |
| Hullusionbench | Ascend 950DT | 960/1129 (85.03%) | 131072 | 256 | 131072 | 256 | temperature=0.8 top_k=15 top_p=0.95 repetition_penalty=1.0 |

## 8 Performance Evaluation

### 8.1 Using vLLM Bench

Run the following from a prepared client environment with access to the complete checkpoint tokenizer. Set the server address to the API host for co-located serving or the proxy host for PD separation. This initial workload fits the single-node example's context limit; adapt the workload to the intended use case after functional verification.

```bash
MODEL_PATH=/path/to/Hy4-FP8
SERVER_IP="<API_OR_PROXY_IP>"
vllm bench serve \
    --backend openai \
    --base-url "http://${SERVER_IP}:8000" \
    --endpoint /v1/completions \
    --model "$MODEL_PATH" \
    --served-model-name Hy4 \
    --dataset-name random \
    --random-input-len 1024 \
    --random-output-len 128 \
    --random-range-ratio 1.0 \
    --num-prompts 16 \
    --max-concurrency 1 \
    --request-rate inf \
    --ignore-eos \
    --save-result \
    --result-dir /path/to/hy4-benchmark-results
```

For additional evaluation methods, refer to [AISBench performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation). No measured throughput, latency, or MTP acceptance results are provided in this guide.

## 9 Performance Tuning

### 9.1 Reference Configurations

Use the configuration table in [Supported Features](#2-supported-features) as the starting point for each topology. Single-node serving uses weight offloading and a small KV cache to reduce device memory requirements. Two-node serving distributes experts across both hosts without weight offloading. The PD example uses separate two-host groups for Prefill and Decode.

### 9.2 Tuning Guidelines

- Tune concurrency and token budgets against the actual available memory. A configured maximum model length does not establish that every request at that length can run at the configured maximum concurrency.
- Account for host RAM and transfer bandwidth when using single-node prefetch weight offloading.
- Compare MTP acceptance and latency with the same workload before changing the speculative token count. Keep the topology-specific scheduling settings until changes have been validated.
- Keep the Prefill and Decode topology declarations consistent in both groups' KV connector configuration.

## 10 FAQ

### 10.1 Multi-node Startup Waits for a Peer

Confirm that every DP rank has started, that the group coordinator address is consistent, and that each local IP belongs to the selected network interface. Check inter-node connectivity and coordinator ports. Increasing the engine readiness timeout does not resolve a missing or unreachable peer.

### 10.2 Single-Node Deployment Runs Out of Memory

Check both device memory and host RAM. The example uses prefetch weight offloading and an explicit KV cache allocation. Confirm that the weight-offload configuration is active before increasing sequence length, concurrency, or KV cache allocation.

### 10.3 PD Requests Do Not Complete

Check P0 and D0 health independently, then verify the proxy's frontend addresses and ports. Check both groups' logs for KV connector initialization and transfer errors. Confirm matching Mooncake versions and the DP/TP topology in both connector configurations.
