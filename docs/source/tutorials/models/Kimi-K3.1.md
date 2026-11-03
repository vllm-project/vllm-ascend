# Kimi-K3.1

## 1 Introduction

This document describes Kimi K3.1 Prefill/Decode (PD) separation on 950DT products (A5), using eight nodes with eight devices per node.

!!! note

    This is a reference configuration requiring a K3.1-enabled vLLM/vLLM-Ascend build. The model registration, sparse-attention operators, and connector support must be available in that build. End-to-end eight-node validation is still required.

## 2 Supported Features

!!! important

    Use Model Runner V2 by setting `VLLM_USE_V2_MODEL_RUNNER=1` on every node before starting the service.

The configuration below uses 140000-token PD separation and DSpark speculative decoding on A5. KV caches are transferred through `MooncakeConnectorV2`. Prefill and Decode each use DP4/TP8; Decode enables DCP8. Validate the selected target and draft checkpoints together with the K3.1-enabled build.

## 3 Prerequisites

### 3.1 Model Weight

| Weight Version | Hardware Requirements | Download Links | Notes |
| --- | --- | --- | --- |
| Kimi K3.1 W4A8 target checkpoint | 8 × 950DT products, 8 devices per node | Use your K3.1 checkpoint distribution | ModelSlim W4A8; architecture `KimiK31ForConditionalGeneration`. |
| K3.1-compatible DSpark checkpoint | The same eight-node A5 deployment | Use the draft checkpoint validated with the target model | Draft TP8, three speculative tokens, greedy drafting. |

The target checkpoint directory must contain the model configuration, tokenizer, image processor, quantization metadata, and weight shards. The reference checkpoint has 93 text layers. Verify its architecture and quantization metadata before starting the service.

**Path description**: The commands below use `<YOUR_MODEL_PATH>` for the target model and `<YOUR_DRAFT_MODEL_PATH>` for the DSpark draft. Replace these placeholders with the same absolute paths on every node and mount both directories inside each container.

### 3.2 Verify Multi-node Communication (Optional)

Follow [verify multi-node communication environment](../../getting_started/installation.md#installation-multi-node) and the [950 SuperPod communication configuration](../features/pd_disaggregation_mooncake_multi_node.md#atlas-850850e950-superpod-communication-configuration). All eight containers must reach the Prefill and Decode service-network addresses, and device-to-device communication must be working.

## 4 Installation

### 4.1 Docker Image Installation

Use the same K3.1-enabled A5 image on every node. Configure host networking, devices, driver, firmware, and communication mounts according to the [A5 container requirements](../features/pd_disaggregation_mooncake_multi_node.md#atlas-850850e950-superpod-950pr950dt-container-requirements). Also mount the target and draft weight directories. Set `<YOUR_RUNTIME_ENV_SCRIPT>` to the environment script for the installed K3.1 build, including its CANN environment and matching custom operators, and use the same build on every node.

### 4.2 Source Code Installation

Build and install a K3.1-enabled source revision using the [installation guide](../../getting_started/installation.md). Keep that revision and its compiled operators consistent across all eight nodes. Check the model registration inside the runtime container:

```shell
python - <<'PY'
from vllm import ModelRegistry
from vllm_ascend.models import register_model

register_model()
assert "KimiK31ForConditionalGeneration" in ModelRegistry.get_supported_archs(), \
    "Install a K3.1-enabled vLLM-Ascend build before continuing."
PY
```

Registration alone does not validate the required operators or distributed execution.

## 5 Online Service Deployment {: #5-online-service-deployment }

### 5.1 Basic Eight-Node PD Separation Deployment

This A5 deployment uses eight nodes with eight devices per node. Four Prefill nodes form a DP4 group, and the other four nodes form a separate DP4 Decode group. Each node runs one TP8 instance with PP1 and exposes its own HTTP endpoint.

The following templates provide the complete PD configuration: a 140000-token context window, DCP8 on Decode, three-token DSpark, and `MooncakeConnectorV2` for P-to-D KV transfer. Both roles must use the same target and compatible draft checkpoints, TP8, and speculative-token count. Both roles use FP8 target KV and indexer caches, Ascend quantization, and an automatic draft KV-cache dtype. The 140000-token limit includes input and output tokens; reserve space for the generated response.

Keep the P/D master addresses, RPC ports, HTTP ports, and KV port ranges independent.

#### 5.1.1 Create the engine templates

Save the corresponding Prefill or Decode engine template as `run_dp_template.sh` on each node. The templates accept the seven positional arguments listed in [Section 5.1.2](#512-start-the-engines). Replace each node's `local_ip` and `nic_name`, the role's KV base port, and the target and draft model paths.

=== "950DT products"

    Set `<YOUR_DRAFT_MODEL_PATH>` to the draft checkpoint validated with the K3.1 target model. The target model uses eager execution on Prefill; the draft uses `enforce_eager:false` on both roles. Validate this combination with the selected runtime.

    === "Prefill"

        ```shell
        KV_PORT=<PREFILL_KV_PORT>

        unset ftp_proxy FTP_PROXY
        unset https_proxy HTTPS_PROXY
        unset http_proxy HTTP_PROXY
        unset all_proxy ALL_PROXY LD_PRELOAD
        nic_name=<PREFILL_NIC_NAME>
        local_ip=<PREFILL_LOCAL_IP>

        source <YOUR_RUNTIME_ENV_SCRIPT>
        export DRAFT_MODEL_PATH=<YOUR_DRAFT_MODEL_PATH>
        export HCCL_IF_IP=${local_ip}
        export VLLM_HOST_IP=${local_ip}
        export GLOO_SOCKET_IFNAME=${nic_name}
        export TP_SOCKET_IFNAME=${nic_name}
        export HCCL_SOCKET_IFNAME=${nic_name}
        export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'
        export ASCEND_RT_VISIBLE_DEVICES=$1
        export VLLM_RPC_TIMEOUT=3600000
        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
        export HCCL_EXEC_TIMEOUT=2040
        export HCCL_CONNECT_TIMEOUT=1200
        export HCCL_BUFFSIZE=800
        export OMP_PROC_BIND=false
        export OMP_NUM_THREADS=10
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export TASK_QUEUE_ENABLE=1
        export HCCL_OP_EXPANSION_MODE="CCU_SCHED"
        export VLLM_USE_V2_MODEL_RUNNER=1
        export PYTHONHASHSEED=0
        export LD_LIBRARY_PATH="/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:${LD_LIBRARY_PATH:-}"
        export VLLM_ASCEND_ENABLE_MLAPO=0
        unset VLLM_PP_LAYER_PARTITION

        SPECULATIVE_CONFIG="$(
          printf \
          '{"method":"dspark","model":"%s","num_speculative_tokens":3,"draft_tensor_parallel_size":'"$7"',"draft_sample_method":"greedy","enforce_eager":false,"kv_cache_dtype":"auto"}' \
          "$DRAFT_MODEL_PATH"
        )"

        vllm serve <YOUR_MODEL_PATH> \
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
            --served-model-name kimi \
            --max-model-len 140000 \
            --max-num-batched-tokens 16384 \
            --max-num-seqs 64 \
            --gpu-memory-utilization 0.85 \
            --enforce-eager \
            --allowed-local-media-path / \
            --trust-remote-code \
            --kv-cache-dtype fp8 \
            --quantization ascend \
            --attention_config.indexer_kv_dtype fp8 \
            --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":128}' \
            --speculative-config "$SPECULATIVE_CONFIG" \
            --mm-encoder-tp-mode data \
            --mm-processor-cache-gb 0 \
            --enable-prefix-caching \
            --additional-config '{"enable_flashcomm1":true,"enable_shared_expert_dp":true,"multistream_overlap_shared_expert":false,"enable_fused_mc2":2,"enable_force_eplb":false}' \
            --profiler-config '{"profiler":"torch","torch_profiler_dir":"./vllm_profile","torch_profiler_with_stack":false}' \
            --kv-transfer-config \
            '{
              "kv_connector": "MooncakeConnectorV2",
              "kv_role": "kv_producer",
              "kv_port": "'"$KV_PORT"'",
              "engine_id": "kimi-k31-prefill-'"$4"'",
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

    === "Decode"

        ```shell
        KV_PORT=<DECODE_KV_PORT>

        unset ftp_proxy FTP_PROXY
        unset https_proxy HTTPS_PROXY
        unset http_proxy HTTP_PROXY
        unset all_proxy ALL_PROXY LD_PRELOAD
        nic_name=<DECODE_NIC_NAME>
        local_ip=<DECODE_LOCAL_IP>

        source <YOUR_RUNTIME_ENV_SCRIPT>
        export DRAFT_MODEL_PATH=<YOUR_DRAFT_MODEL_PATH>
        export HCCL_IF_IP=${local_ip}
        export VLLM_HOST_IP=${local_ip}
        export GLOO_SOCKET_IFNAME=${nic_name}
        export TP_SOCKET_IFNAME=${nic_name}
        export HCCL_SOCKET_IFNAME=${nic_name}
        export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'
        export ASCEND_RT_VISIBLE_DEVICES=$1
        export VLLM_RPC_TIMEOUT=3600000
        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
        export HCCL_EXEC_TIMEOUT=2040
        export HCCL_CONNECT_TIMEOUT=1200
        export HCCL_BUFFSIZE=800
        export OMP_PROC_BIND=false
        export OMP_NUM_THREADS=10
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export TASK_QUEUE_ENABLE=1
        export HCCL_OP_EXPANSION_MODE="CCU_SCHED"
        export VLLM_USE_V2_MODEL_RUNNER=1
        export PYTHONHASHSEED=0
        export LD_LIBRARY_PATH="/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:${LD_LIBRARY_PATH:-}"
        export VLLM_ASCEND_ENABLE_MLAPO=1
        unset VLLM_PP_LAYER_PARTITION

        SPECULATIVE_CONFIG="$(
          printf \
          '{"method":"dspark","model":"%s","num_speculative_tokens":3,"draft_tensor_parallel_size":'"$7"',"draft_sample_method":"greedy","enforce_eager":false,"kv_cache_dtype":"auto"}' \
          "$DRAFT_MODEL_PATH"
        )"

        vllm serve <YOUR_MODEL_PATH> \
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
            --decode-context-parallel-size 8 \
            --block-size 768 \
            --cp-kv-cache-interleave-size 768 \
            --enable-expert-parallel \
            --served-model-name kimi \
            --max-model-len 140000 \
            --max-num-batched-tokens 768 \
            --max-num-seqs 40 \
            --gpu-memory-utilization 0.91 \
            --allowed-local-media-path / \
            --trust-remote-code \
            --kv-cache-dtype fp8 \
            --quantization ascend \
            --attention_config.indexer_kv_dtype fp8 \
            --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":128}' \
            --speculative-config "$SPECULATIVE_CONFIG" \
            --mm-encoder-tp-mode data \
            --mm-processor-cache-gb 0 \
            --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
            --enable-prefix-caching \
            --additional-config '{"multistream_overlap_shared_expert":true,"scheduler_config":{"recompute_scheduler_enable":true}}' \
            --profiler-config '{"profiler":"torch","torch_profiler_dir":"./vllm_profile","torch_profiler_with_stack":false}' \
            --kv-transfer-config \
            '{
              "kv_connector": "MooncakeConnectorV2",
              "kv_role": "kv_consumer",
              "kv_port": "'"$KV_PORT"'",
              "engine_id": "kimi-k31-decode-'"$4"'",
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

#### 5.1.2 Start the engines

Deploy [`launch_online_dp.py`](https://github.com/vllm-project/vllm-ascend/blob/main/examples/external_online_dp/launch_online_dp.py) and save the corresponding Prefill or Decode template as `run_dp_template.sh` on every node.

=== "950DT products"

    ```shell
    python launch_online_dp.py \
        --dp-size <DP_SIZE> \
        --tp-size 8 \
        --dp-size-local 1 \
        --dp-rank-start <LOCAL_RANK> \
        --dp-address <PD_MASTER_IP> \
        --dp-rpc-port <DP_RPC_PORT> \
        --vllm-start-port <VLLM_START_PORT>
    ```

Set `<DP_SIZE>` to `4` for both Prefill and Decode. Set `<LOCAL_RANK>` to the DP rank (`0` through `3`) within the corresponding group. The launcher passes these as the template's third and fourth arguments; `--dp-size-local 1` starts one local template process using devices `0,1,2,3,4,5,6,7`.

| Template Argument | Meaning | Prefill | Decode |
| --- | --- | --- | --- |
| `$1` | Visible devices | `0,1,2,3,4,5,6,7` | `0,1,2,3,4,5,6,7` |
| `$2` | HTTP port | `8000` | `18200` |
| `$3` | DP size | `4` | `4` |
| `$4` | Local rank | DP rank `0`–`3` | DP rank `0`–`3` |
| `$5` | Group master IP | Prefill rank-0 IP | Decode rank-0 IP |
| `$6` | DP RPC port | `29501` | `29581` |
| `$7` | TP size | `8` | `8` |

Use Prefill KV base port `30100` and Decode KV base port `30300`, with free ranges for the connector's per-rank sockets. See the [KV-port configuration guide](../features/pd_disaggregation_mooncake_multi_node.md#kv_port-configuration-guide). Start rank 0 before ranks 1–3 in each group. All eight engines accept HTTP requests; no rank is launched with `--headless` in this DP4 configuration.

After the engines are ready, start the [example load-balancing proxy](../features/pd_disaggregation_mooncake_multi_node.md#example-proxy-for-deployment). Register all four Prefill and all four Decode endpoints:

```shell
python <VLLM_ASCEND_SOURCE_DIR>/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
    --host 0.0.0.0 \
    --port 18080 \
    --prefiller-hosts <P0_IP> <P1_IP> <P2_IP> <P3_IP> \
    --prefiller-ports 8000 8000 8000 8000 \
    --decoder-hosts <D0_IP> <D1_IP> <D2_IP> <D3_IP> \
    --decoder-ports 18200 18200 18200 18200 \
    --workers 1 \
    --log-level INFO
```

## 6 Functional Verification

### 6.1 PD Separation Functional Verification

=== "950DT products"

    Check `/health` on all four Prefill and all four Decode HTTP endpoints before sending a request through the proxy:

    ```shell
    curl --fail --noproxy '*' http://<PROXY_IP>:18080/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "kimi",
            "messages": [{
                "role": "user",
                "content": "What is the capital of China?"
            }],
            "max_tokens": 128,
            "stream": false
        }'
    ```

    Confirm that the response contains generated text, and inspect both roles' logs for successful KV transfer. Healthy HTTP endpoints alone do not validate PD transfer, accuracy, or long-context behavior. Verify these on the actual eight-node environment, keeping input plus requested output within `140000` tokens.

## 7 Accuracy Evaluation

### 7.1 Using AISBench

Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) for the setup and evaluation procedure. Run against the proxy endpoint with served model name `kimi`. Record target/draft revisions, tokenizer, chat template, dataset, evaluator revision, and sampling settings. No accuracy results are claimed for this reference configuration.

## 8 Performance Evaluation

### 8.1 Using AISBench

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation), and use the same proxy endpoint for all requests.

### 8.2 Using vLLM Benchmark

Use `vllm bench serve` against the proxy endpoint, with served model name `kimi`, the K3.1 checkpoint as the tokenizer path, and tokenizer mode `kimi_k3`. See the [performance benchmark guide](../../developer_guide/performance_and_debug/performance_benchmark.md) for workload preparation and measurement. Record request lengths, concurrency, cache state, and draft acceptance alongside latency and throughput. No performance results are claimed for this reference configuration.

## 9 Performance Tuning

### 9.1 Recommended Configurations

The following reference values match the engine templates in [Section 5.1](#51-basic-eight-node-pd-separation-deployment). Validate memory capacity and concurrency for the selected checkpoints and workload.

| Setting | 950DT Prefill | 950DT Decode |
| --- | --- | --- |
| DP / TP / PP | `4` / `8` / `1` | `4` / `8` / `1` |
| Model Runner | V2 | V2 |
| Maximum model length | `140000` | `140000` |
| Maximum batched tokens | `16384` | `768` |
| Maximum sequences | `64` | `40` |
| Device memory utilization | `0.85` | `0.91` |
| DCP | `1` | `8` |
| KV block / interleave size | Model defaults | `768` / `768` |
| HCCL expansion mode / buffer size | `CCU_SCHED` / `800` | `CCU_SCHED` / `800` |
| DSpark tokens / draft TP | `3` / `8` | `3` / `8` |
| Draft `enforce_eager` | `false` | `false` |
| Target execution | Eager | `FULL_DECODE_ONLY` |
| KV connector | `MooncakeConnectorV2` | `MooncakeConnectorV2` |
| Target KV / indexer cache dtype | `fp8` / `fp8` | `fp8` / `fp8` |
| Draft KV cache dtype | `auto` | `auto` |

### 9.2 Tuning Guide

Adjust `max-model-len`, `max-num-seqs`, `max-num-batched-tokens`, and `gpu-memory-utilization` together for the target workload. Keep target/draft paths, speculative-token count, TP, and connector topology consistent between roles. Refer to the [performance tuning guide](../../developer_guide/performance_and_debug/optimization_and_tuning.md) for the general procedure.

## 10 FAQ

- **Q: Why does the model registration check fail?**
A: The runtime needs K3.1 model registration and its matching operators and distributed support. Verify that all required components are installed in the runtime container.
- **Q: Which nodes should be registered with the proxy?**
A: Register all four Prefill and all four Decode HTTP endpoints. Each group uses DP ranks 0–3.
- **Q: Which DSpark settings must match?**
A: Both roles use the same compatible draft checkpoint, three speculative tokens, TP8, greedy drafting, and `enforce_eager:false`. Keep the Prefill target's `--enforce-eager` enabled and validate the combination with the selected K3.1 runtime.
