# Kimi-K3.1

## 1 Introduction

This guide describes eight-node Prefill/Decode (PD) separation for Kimi K3.1 on 950DT products (A5), with eight devices per node. It follows the [Kimi K3 eight-node topology](Kimi-K3.md#52-basic-eight-node-pd-separation-deployment) and [pooled PD deployment](Kimi-K3.md#54-pooled-pd-separation-deployment), using a ModelSlim W4A8 checkpoint, a 1M context window, DSpark, and a MemCache-backed KV Cache Pool.

!!! note
    This is a reference configuration. The commands require a K3.1-enabled vLLM/vLLM-Ascend build with the model registration, sparse-attention operators, AttnRes pipeline-parallel transport, and Mooncake support described below. They do not establish support in an unmodified K3 release image. End-to-end eight-node validation is still required for this configuration.

## 2 Prerequisites

- Eight 950DT nodes, each exposing devices `0,1,2,3,4,5,6,7`, with working device-to-device communication.
- The same K3.1-enabled software build on all eight nodes. Use Model Runner V2 by setting `VLLM_USE_V2_MODEL_RUNNER=1` on every node.
- A Kimi K3.1 ModelSlim W4A8 checkpoint containing the model configuration, tokenizer, image processor, quantization metadata, and weight shards. Mount it at the same absolute path inside every container.
- A DSpark draft checkpoint validated with the K3.1 target checkpoint. Mount the same draft-model path on every node; the templates use three speculative tokens and draft TP8 on both roles.
- Containers with host networking and the A5 devices, driver, firmware, and communication configuration described in [the Mooncake multi-node guide](../features/pd_disaggregation_mooncake_multi_node.md#atlas-850850e950-superpod-communication-configuration). Initialize the Ascend environment before running the scripts.
- Mooncake and the Python dependencies used by the [example proxy](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py) installed in the runtime environment.
- MemCache installed and configured. Follow [Kimi K3 Sections 5.4.1–5.4.4](Kimi-K3.md#541-install-mf) to configure the shared MetaService, node-local configuration, and standalone MemCache processes. Start MetaService and verify that the standalone process on every node reports successful initialization before starting vLLM. Set `MMC_LOCAL_CONFIG_PATH` for the engines to `mmc-local.conf`, rather than the standalone process's configuration file.

Check the installed model registration inside a container:

```bash
python - <<'PY'
from vllm import ModelRegistry
from vllm_ascend.models import register_model

register_model()
assert "KimiK31ForConditionalGeneration" in ModelRegistry.get_supported_archs(), \
    "Install a K3.1-enabled vLLM-Ascend build before continuing."
PY
```

The reference checkpoint has 93 text layers and an AttnRes block size of 12. The Prefill partition `24,24,24,21` covers all 93 layers and places the first three cuts at AttnRes block boundaries. Verify these values against the checkpoint's `text_config` before starting the engines. Registration alone does not validate the required operators or PP transport.

## 3 Eight-node PD separation

### 3.1 Topology and configuration

Four nodes form **one** Prefill instance with DP1/TP8/PP4. Four other nodes form the Decode DP4 group, with TP8 on each node and PP1. Prefill and Decode must use the same TP size, target checkpoint, draft checkpoint, speculative-token count, and compatible cache settings. The 1M limit includes both input and output tokens.

| Setting | Prefill | Decode |
| --- | --- | --- |
| Number of nodes | 4 | 4 |
| DP / TP / PP | 1 / 8 / 4 | 4 / 8 / 1 |
| Node rank | PP rank 0–3 | DP rank 0–3 |
| Layer partition | `24,24,24,21` | Full model on each DP rank |
| HTTP endpoint | Rank 0 only | One per DP rank |
| Model Runner | V2 | V2 |
| Maximum model length | `1048576` | `1048576` |
| Maximum batched tokens | `16384` | `768` |
| Maximum sequences | `96` | `40` |
| Device memory utilization | `0.76` | `0.91` |
| KV block / interleave size | Model defaults | `768` / `768` |
| DCP | `1` | `8` |
| HCCL expansion mode / buffer size | `CCU_SCHED` / `800` | `CCU_SCHED` / `800` |
| DSpark tokens / draft TP | `3` / `8` | `3` / `8` |
| Target eager execution | Enabled | `FULL_DECODE_ONLY` graphs |
| Draft `enforce_eager` | `false` | `false` |
| KV connector | `MultiConnector` | `MultiConnector` |
| AscendStore `use_layerwise` | `true` | `false` |

`MultiConnector` combines `MooncakeConnectorV2` for P-to-D transfer with `AscendStoreConnector` for the MemCache-backed pool. The memory utilization and concurrency values are reference settings that require validation against the chosen checkpoint and workload. Cache dtypes are left at their model defaults in these templates.

Save the following as `pd_config.sh` in the same working directory on every node. Replace the target-model path, draft-model path, MemCache configuration directory, and the two master IPs. Use the service-network IPs that are reachable from all eight containers.

```bash
#!/usr/bin/env bash
export MODEL_PATH="<YOUR_KIMI_K31_W4A8_MODEL_PATH>"
export DRAFT_MODEL_PATH="<YOUR_KIMI_K31_COMPATIBLE_DSPARK_MODEL_PATH>"
export MMC_LOCAL_CONFIG_PATH="<MEMCACHE_CONFIG_DIR>/mmc-local.conf"
export P_MASTER_IP="<PREFILL_RANK_0_IP>"
export D_MASTER_IP="<DECODE_RANK_0_IP>"
export P_MASTER_PORT=29580
export D_RPC_PORT=29581
export P_HTTP_PORT=18100
export D_HTTP_PORT=18200
export P_KV_PORT=30100
export D_KV_PORT=30300
export LOOKUP_RPC_ID=0
export PP_PARTITION="24,24,24,21"
export MAX_MODEL_LEN=1048576
export SERVED_MODEL_NAME="kimi"
```

Keep the P/D master ports and HTTP/KV port ranges independent. Each KV base port needs a free range for the connector's per-rank sockets; follow the [KV-port configuration guide](../features/pd_disaggregation_mooncake_multi_node.md#kv_port-configuration-guide). `lookup_rpc_port` is an IPC identifier, rather than a TCP port. This example uses `0` with one local engine on each node; use distinct identifiers if multiple local engine instances share the same IPC namespace.

Save the shared environment as `pd_env.sh` alongside `pd_config.sh`. The engine scripts pass each node's own IP and interface to this file.

```bash
#!/usr/bin/env bash
# Initialize CANN before enabling nounset in the caller.
source /usr/local/Ascend/cann/set_env.sh
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY
unset ftp_proxy FTP_PROXY all_proxy ALL_PROXY LD_PRELOAD
unset VLLM_PP_LAYER_PARTITION
export HCCL_IF_IP="$LOCAL_IP"
export VLLM_HOST_IP="$LOCAL_IP"
export HCCL_SOCKET_IFNAME="$NIC_NAME"
export GLOO_SOCKET_IFNAME="$NIC_NAME"
export TP_SOCKET_IFNAME="$NIC_NAME"
export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HCCL_OP_EXPANSION_MODE=CCU_SCHED
export HCCL_BUFFSIZE=800
export VLLM_RPC_TIMEOUT=3600000
export HCCL_EXEC_TIMEOUT=2040
export HCCL_CONNECT_TIMEOUT=1200
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export TASK_QUEUE_ENABLE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export PYTHONHASHSEED=0
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export LD_LIBRARY_PATH="/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:${LD_LIBRARY_PATH:-}"
SPECULATIVE_CONFIG=$(python - "$DRAFT_MODEL_PATH" <<'PY'
import json
import sys

print(json.dumps({"method":"dspark", "model":sys.argv[1],
                  "num_speculative_tokens":3, "draft_tensor_parallel_size":8,
                  "draft_sample_method":"greedy", "enforce_eager":False}))
PY
)
```

### 3.2 Prefill engine template

Save this script as `run_prefill.sh` on all four Prefill nodes. Rank 0 exposes the HTTP endpoint. Ranks 1–3 run with `--headless`. Every rank uses its own `LOCAL_IP` and the same Prefill master address and port.

```bash
#!/usr/bin/env bash
set -e -o pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source ./pd_config.sh
NODE_RANK=${1:?Usage: bash run_prefill.sh NODE_RANK LOCAL_IP NIC_NAME}
LOCAL_IP=${2:?Set the local service-network IP}
NIC_NAME=${3:?Set the local service-network interface}
[[ "$NODE_RANK" =~ ^[0-3]$ ]] || { echo "NODE_RANK must be 0-3" >&2; exit 2; }
source ./pd_env.sh
set -u
export VLLM_PP_LAYER_PARTITION="$PP_PARTITION"
export VLLM_ASCEND_ENABLE_MLAPO=0
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

HEADLESS=()
if (( NODE_RANK > 0 )); then HEADLESS+=(--headless); fi

exec vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 --port "$P_HTTP_PORT" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --data-parallel-size 1 \
    --tensor-parallel-size 8 --pipeline-parallel-size 4 \
    --nnodes 4 --node-rank "$NODE_RANK" \
    --distributed-executor-backend mp \
    --master-addr "$P_MASTER_IP" --master-port "$P_MASTER_PORT" \
    "${HEADLESS[@]}" \
    --enable-expert-parallel --seed 1024 \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens 16384 --max-num-seqs 96 \
    --gpu-memory-utilization 0.76 --enforce-eager \
    --trust-remote-code \
    --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":128}' \
    --tokenizer-mode kimi_k3 --reasoning-parser kimi_k3 --tool-call-parser kimi_k3 \
    --enable-auto-tool-choice --enable-prefix-caching \
    --mm-encoder-tp-mode data --mm-processor-cache-gb 0 \
    --additional-config '{"enable_shared_expert_dp":true,"multistream_overlap_shared_expert":false,"enable_fused_mc2":2,"enable_force_eplb":false}' \
    --profiler-config '{"profiler":"torch","torch_profiler_dir":"./vllm_profile","torch_profiler_with_stack":false}' \
    --speculative-config "$SPECULATIVE_CONFIG" \
    --kv-transfer-config "{
      \"kv_connector\": \"MultiConnector\",
      \"kv_role\": \"kv_producer\",
      \"engine_id\": \"kimi-k31-prefill-$NODE_RANK\",
      \"kv_connector_extra_config\": {
        \"connectors\": [
          {\"kv_connector\":\"MooncakeConnectorV2\",\"kv_role\":\"kv_producer\",\"kv_port\":$P_KV_PORT,
           \"kv_connector_extra_config\":{
             \"prefill\":{\"dp_size\":1,\"tp_size\":8,\"pp_size\":4},
             \"decode\":{\"dp_size\":4,\"tp_size\":8}}},
          {\"kv_connector\":\"AscendStoreConnector\",\"kv_role\":\"kv_producer\",
           \"kv_connector_extra_config\":{\"backend\":\"memcache\",\"use_layerwise\":true,\"lookup_rpc_port\":\"$LOOKUP_RPC_ID\"}}
        ]
      }
    }"
```

Following the A5 K3 template, set the Prefill partition through `VLLM_PP_LAYER_PARTITION`. Use the same partition on all four Prefill nodes. The target model remains in eager mode; the DSpark draft uses `enforce_eager:false`. Validate this combination with the selected K3.1-enabled build and draft checkpoint.

### 3.3 Decode engine template

Save this script as `run_decode.sh` on all four Decode nodes. Each node runs one TP8 process group with its own DP rank and HTTP endpoint. All ranks use the Decode rank-0 IP as `--data-parallel-address`; this address is independent of the Prefill master.

```bash
#!/usr/bin/env bash
set -e -o pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source ./pd_config.sh
DP_RANK=${1:?Usage: bash run_decode.sh DP_RANK LOCAL_IP NIC_NAME}
LOCAL_IP=${2:?Set the local service-network IP}
NIC_NAME=${3:?Set the local service-network interface}
[[ "$DP_RANK" =~ ^[0-3]$ ]] || { echo "DP_RANK must be 0-3" >&2; exit 2; }
source ./pd_env.sh
set -u
export VLLM_ASCEND_ENABLE_MLAPO=1

exec vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 --port "$D_HTTP_PORT" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --data-parallel-size 4 --data-parallel-rank "$DP_RANK" \
    --data-parallel-address "$D_MASTER_IP" --data-parallel-rpc-port "$D_RPC_PORT" \
    --tensor-parallel-size 8 --decode-context-parallel-size 8 \
    --enable-expert-parallel --seed 1024 \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens 768 --max-num-seqs 40 \
    --gpu-memory-utilization 0.91 \
    --trust-remote-code \
    --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":128}' \
    --block-size 768 --cp-kv-cache-interleave-size 768 \
    --tokenizer-mode kimi_k3 --reasoning-parser kimi_k3 --tool-call-parser kimi_k3 \
    --enable-auto-tool-choice --enable-prefix-caching \
    --mm-encoder-tp-mode data --mm-processor-cache-gb 0 \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --additional-config '{"multistream_overlap_shared_expert":true,"scheduler_config":{"recompute_scheduler_enable":true}}' \
    --profiler-config '{"profiler":"torch","torch_profiler_dir":"./vllm_profile","torch_profiler_with_stack":false}' \
    --speculative-config "$SPECULATIVE_CONFIG" \
    --kv-transfer-config "{
      \"kv_connector\": \"MultiConnector\",
      \"kv_role\": \"kv_consumer\",
      \"engine_id\": \"kimi-k31-decode-$DP_RANK\",
      \"kv_connector_extra_config\": {
        \"connectors\": [
          {\"kv_connector\":\"MooncakeConnectorV2\",\"kv_role\":\"kv_consumer\",\"kv_port\":$D_KV_PORT,
           \"kv_connector_extra_config\":{
             \"prefill\":{\"dp_size\":1,\"tp_size\":8,\"pp_size\":4},
             \"decode\":{\"dp_size\":4,\"tp_size\":8}}},
          {\"kv_connector\":\"AscendStoreConnector\",\"kv_role\":\"kv_consumer\",
           \"kv_connector_extra_config\":{\"backend\":\"memcache\",\"use_layerwise\":false,\"lookup_rpc_port\":\"$LOOKUP_RPC_ID\"}}
        ]
      }
    }"
```

### 3.4 Start the engines

Run each command in its corresponding node's container. Replace the IP and interface placeholders with local values. Start Prefill rank 0 before ranks 1–3, and Decode rank 0 before the other Decode ranks. Keep rank 0 running while starting its workers from separate shells; it waits for the distributed group to form.

| Node | Command |
| --- | --- |
| Prefill 0 | `bash run_prefill.sh 0 <P0_IP> <P0_NIC>` |
| Prefill 1 | `bash run_prefill.sh 1 <P1_IP> <P1_NIC>` |
| Prefill 2 | `bash run_prefill.sh 2 <P2_IP> <P2_NIC>` |
| Prefill 3 | `bash run_prefill.sh 3 <P3_IP> <P3_NIC>` |
| Decode 0 | `bash run_decode.sh 0 <D0_IP> <D0_NIC>` |
| Decode 1 | `bash run_decode.sh 1 <D1_IP> <D1_NIC>` |
| Decode 2 | `bash run_decode.sh 2 <D2_IP> <D2_NIC>` |
| Decode 3 | `bash run_decode.sh 3 <D3_IP> <D3_NIC>` |

Before starting the proxy, check `/health` on Prefill rank 0 and all four Decode HTTP endpoints. Prefill ranks 1–3 are headless and have no HTTP health endpoint.

```bash
curl --fail --noproxy '*' "http://<P0_IP>:18100/health"
curl --fail --noproxy '*' "http://<D0_IP>:18200/health"
curl --fail --noproxy '*' "http://<D1_IP>:18200/health"
curl --fail --noproxy '*' "http://<D2_IP>:18200/health"
curl --fail --noproxy '*' "http://<D3_IP>:18200/health"
```

### 3.5 Start the proxy

Run the proxy from a container that can reach all engine endpoints. Replace the source directory and IP placeholders. Register only Prefill rank 0 as a Prefill endpoint, and register all four Decode endpoints. The PP workers are part of the single Prefill instance.

```bash
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
python <VLLM_ASCEND_SOURCE_DIR>/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
    --host 0.0.0.0 --port 18080 \
    --prefiller-hosts <P0_IP> --prefiller-ports 18100 \
    --decoder-hosts <D0_IP> <D1_IP> <D2_IP> <D3_IP> \
    --decoder-ports 18200 18200 18200 18200 \
    --workers 1 --log-level INFO
```

## 4 Functional verification

Send the request through the proxy, using the same served model name configured for both roles:

```bash
curl --fail --noproxy '*' "http://<PROXY_IP>:18080/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{
      "model": "kimi",
      "messages": [{"role":"user","content":"What is the capital of China?"}],
      "max_tokens": 128,
      "stream": false
    }'
```

Confirm that the response contains generated text and inspect the Prefill and Decode logs for successful KV transfer. Healthy HTTP endpoints alone do not validate PD transfer, output accuracy, or long-context behavior. Validate these on the actual eight-node environment before treating the configuration as a deployment baseline.
