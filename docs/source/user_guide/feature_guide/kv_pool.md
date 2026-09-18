# KV Cache Pool (Ascend Store)

## Feature Introduction

KV Cache Pool (Ascend Store) is a cross-node KV Cache pooling storage feature provided by vLLM-Ascend. It writes KV Cache to external storage backends (Mooncake / Memcache / Yuanrong) via `AscendStoreConnector`, and works with the PD (Prefill/Decode) disaggregation architecture to share and reuse KV Cache across multiple vLLM instances, avoiding redundant recomputation of identical prefixes.

Key benefits include: memory decoupling between Prefill and Decode nodes in PD disaggregation deployments with independent scaling; cross-request/cross-node KV reuse reducing redundant Prefill computation; SSD Offload support for extended KV capacity; and transfer QoS priority control.

Supported since vLLM-Ascend v0.23.0 (using the latest version is recommended).

### Constraints and Limitations

| Category | Constraints |
| :--- | :--- |
| Common | Hardware:<br>• Supported hardware series A2/A3/950PR&950DT Products; for HDK/CANN requirements of each series, see the [Hardware Dependency Quick Reference table](#ascend_global_resource_config)<br>• 950PR&950DT Products requires extra mounts `/dev/ummu`, `/dev/uburma`, `/usr/bin/urma_admin`, `/lib/route.conf`, `/etc/hccl_rootinfo.json`<br>Software Dependencies:<br>• CANN >= 9.1.0<br>Model:<br>• `kv_load_failure_policy=recompute` does not yet support hybrid attention models (e.g., DeepSeekV4, Qwen 3.5)<br>• MLA models can use `consumer_is_to_put` for Decode node to store KV for Prefill node<br>Feature Mutex:<br>• `use_layerwise` only supported on the Prefill node of the Mooncake/Memcache backends |
| Mooncake backend | Software Dependencies:<br>• mooncake >= 0.3.11.post1 (non-default tenant requires >= 0.3.12); Mooncake wheel requires glibc >= 2.35<br>Deployment:<br>• Store/PD traffic separation applies to A3 and 950PR&950DT Products |
| Memcache backend | Software Dependencies:<br>• Requires `memfabric-hybrid` and `memcache-hybrid` (SSD Cache needs `memcache_hybrid >= 1.2.0`)<br>Hardware:<br>• On A3, using `device_sdma` protocol requires LingQu Computing Network >= 1.5<br>Deployment:<br>• Additionally supports separated deployment of MemCache and vLLM |
| Yuanrong backend | Software Dependencies:<br>• Requires `openyuanrong-datasystem`<br>Feature Mutex:<br>• Worker cannot configure both Coordinator and etcd discovery backends simultaneously<br>• Under `P2P_TRANSFER` or FabricMem mode, client-side device memory pre-registration is always skipped regardless of `enable_dev_mem_pregister` value |

## Feature Usage

### Usage Scenarios

Choose the storage backend according to your workload requirements:

| Scenario | Selection Guidance |
| :--- | :--- |
| Mooncake backend | Default backend. Mooncake is Moonshot AI's open-source distributed KVCache system for LLM inference; choose it when SSD Offload for extended KV capacity or multi-tenant quota management is needed |
| Memcache backend | Based on MemFabric; choose it when A3 HCCS high-speed interconnect for lower transfer latency, or layer-by-layer KV transfer (`use_layerwise`), is needed |
| Yuanrong backend | Based on openyuanrong-datasystem (an openEuler open-source distributed foundation for integrated training and inference); choose it when multi-node deployment, Remote H2D transfer, or openEuler ecosystem integration is needed |

### Environment Preparation

1. Verify that `hccn.conf` exists in the environment. When using Docker, mount it into the container:
   ```bash
   cat /etc/hccn.conf
   ```
2. 950PR&950DT Products requires extra device and config mounts; see Common in Constraints and Limitations for the list.
3. Synchronize `PYTHONHASHSEED` across all nodes:
   ```bash
   export PYTHONHASHSEED=0
   ```
4. Install software for the selected backend (see installation steps in each scenario).

### Scenario 1: Mooncake Backend

#### Step 1: Software Installation

Check the Mooncake wheel dependency (for the glibc version requirement, see Software Dependencies in Constraints and Limitations):

```shell
ldd --version
```

Install Mooncake:

```shell
python3 -m pip install mooncake-transfer-engine-npu==0.3.11.post1 --extra-index-url https://mirrors.aliyun.com/pypi/web/simple
```

#### Step 2: Configure mooncake.json and Start mooncake_master

> The following Mooncake guides are optional in-depth references; you can complete the deployment in this document without reading them:
>
> - [Mooncake Store Deployment Guide](https://github.com/kvcache-ai/Mooncake/blob/main/docs/source/deployment/mooncake-store-deployment-guide.md): in-depth reference on Mooncake Store deployment architecture and operations, useful when customizing the deployment topology or troubleshooting the Store layer
> - [SSD Offload](https://github.com/kvcache-ai/Mooncake/blob/main/docs/source/deployment/ssd/ssd-offload.md): advanced configuration reference for SSD Offload capacity planning and eviction policies

Configure `mooncake.json` and point `MOONCAKE_CONFIG_PATH` to its full path:

```json
{
    "metadata_server": "P2PHANDSHAKE",
    "protocol": "ascend",
    "device_name": "",
    "master_server_address": "<master_ip>:50088",
    "global_segment_size": "1GB",
    "preferred_segment": false,
    "prefer_alloc_in_same_node": true,
    "enable_ssd_offload": false,
    "ssd_offload_path": "/nvme/mooncake_offload",
    "tenant_id": "default"
}
```

<details markdown="1">
<summary>mooncake.json Parameters</summary>

| `Parameter` | Type | Default | Required | Value Range | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `metadata_server` | str | No default | Yes | P2PHANDSHAKE | Configured as P2PHANDSHAKE. |
| `protocol` | str | No default | Yes | ascend | Must be set to `ascend` on NPU. |
| `device_name` | str | "" (empty string) | No | Empty string | The ascend protocol does not use device names; leave empty. |
| `master_server_address` | str | No default | Yes | <ip>:<port> | Master service IP and port. Can be overridden by the `MOONCAKE_MASTER` environment variable (takes precedence, useful for injecting the master address through Kubernetes). |
| `global_segment_size` | str | No default | Yes | Must align to 1 GB (1024 MB / 1048576 KB / 1073741824 B) | Registered memory size per card to the KV Pool. Can be overridden by the `MOONCAKE_GLOBAL_SEGMENT_SIZE` environment variable (takes precedence). |
| `preferred_segment` | bool | false | No | true / false | Whether to prefer storing KV on the local segment. |
| `prefer_alloc_in_same_node` | bool | true | No | true / false | Whether to prefer allocating KV on the same node. |
| `enable_ssd_offload` | bool | false | No | true / false | Whether to enable SSD offload. Environment variables are not supported. |
| `ssd_offload_path` | str | No default | Required when SSD offload enabled | Absolute path | Absolute path for SSD offload data storage (for example, /nvme/mooncake_offload). The directory must exist and be writable by the vLLM process; create it before startup (`mkdir -p <path>`). Relative paths, symlinks, and paths containing `..` are rejected. |
| `tenant_id` | str | default | No | String | Mooncake tenant namespace. Missing, `null`, empty, or whitespace-only values use `default`; surrounding whitespace is removed. All Prefill, Decode, scheduler, and replica instances that share KV entries must use the same tenant ID. |

</details>

Start `mooncake_master` (only needs to run on one node):

```shell
mooncake_master --port 50088 --eviction_high_watermark_ratio 0.9 --eviction_ratio 0.1 --default_kv_lease_ttl 11000 --enable_offload=false --client_ttl=120
```

To enable strict multi-tenant isolation, start with:

```shell
mooncake_master \
    --port 50088 \
    --enable_multi_tenants=true \
    --tenant_quota_connector_type=file \
    --tenant_quota_connector_uri=/etc/mooncake/tenant_quotas.yaml
```

Example tenant quota file (`/etc/mooncake/tenant_quotas.yaml`):

```yaml
version: 1

tenants:
  - name: tenant-a
    quota: 200GB
  - name: tenant-b
    quota: 200GB
  - name: default
    quota: 100GB
```

Notes on multi-tenant mode:

- While strict multi-tenant mode is disabled, tenant IDs are ignored for object placement and objects remain in the `default` namespace.
- Strict mode rejects writes for unregistered tenants, including `default`, so every tenant used by vLLM-Ascend must appear in the policy.
- The file connector can be replaced with `etcd` when Mooncake is built with `STORE_USE_ETCD=ON`; in that case, set `tenant_quota_connector_uri` to the etcd endpoints.
- `tenant_id` is an instance-level namespace and quota identity, not an authentication mechanism. A client that can access Mooncake can still declare a tenant ID. Keep incompatible models, model versions, quantization formats, and KV layouts in separate model or release namespaces even when tenant isolation is enabled.

Mooncake exposes tenant quota snapshots through the master metrics HTTP port (default `9003`):

```shell
curl -s http://<master_host>:9003/api/v1/tenant_quotas
curl -s "http://<master_host>:9003/api/v1/tenant_quotas?tenant_id=tenant-a"
```

<details markdown="1">
<summary>mooncake_master Parameters</summary>

| `Parameter` | Type | Default | Required | Value Range | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `port` | int | No default | Yes | Port number, must match the port in `master_server_address` in mooncake.json | Master service listening port. |
| `eviction_high_watermark_ratio` | float | No default | No | Ratio, [0, 1] | Watermark where Mooncake Store performs eviction. |
| `eviction_ratio` | float | No default | No | Ratio, [0, 1] | Portion of stored objects to evict. |
| `default_kv_lease_ttl` | int | No default | No | Milliseconds; must be larger than `ASCEND_CONNECT_TIMEOUT` and `ASCEND_TRANSFER_TIMEOUT` | Default lease TTL for KV objects (milliseconds). |
| `enable_offload` | bool | false | Required when SSD offload enabled | true / false | Set to true to enable SSD offload in master. |
| `client_ttl` | int | 10 | No | Integer (seconds) | Seconds a client stays alive after the last Ping. See "FAQ - Issue 2: SEGMENT_NOT_FOUND (SSD Offload)". |
| `enable_multi_tenants` | bool | false | No | true / false | Enable strict multi-tenant mode. |
| `tenant_quota_connector_type` | str | No default | Required when multi-tenant enabled | file / etcd | Tenant quota connector type. |
| `tenant_quota_connector_uri` | str | No default | Required when multi-tenant enabled | Quota file path or etcd endpoints | Tenant quota file path or etcd endpoints. |

</details>

#### Step 3: PD Disaggregation Scenario

Using `MultiConnector` to simultaneously utilize both `MooncakeConnectorV1` and `AscendStoreConnector`: `MooncakeConnectorV1` performs kv_transfer, while `AscendStoreConnector` serves as the prefix-cache node.

For A3 and 950PR&950DT Products Store/PD traffic separation, set `ASCEND_GLOBAL_RESOURCE_CONFIG` on both the prefill and decode nodes. The top-level resource configuration controls `MooncakeConnectorV1` PD traffic, and the `store` section controls `AscendStoreConnector` Mooncake Store traffic, so the two traffic classes do not compete on the same physical link.

##### run_prefill.sh / run_decode.sh

```shell
#!/bin/bash

# prefill / decode
ROLE="prefill"
# A2 (800I/800T A2) or A3 (800I/800T A3) or 950PR&950DT Products
HARDWARE_SERIES="A2"
# Link type: ROCE or HCCS in A3 series.
LINK_TYPE="ROCE"
LOCAL_IP="<local_ip>"
NIC_NAME="<nic_name>"

MODEL_PATH="<model_path>/Qwen3-32B"
SERVED_MODEL_NAME="qwen3"
DATA_PARALLEL_SIZE=1
TENSOR_PARALLEL_SIZE=8
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# parameters required for kv pool and mooncake
export PYTHONHASHSEED=0
export MOONCAKE_CONFIG_PATH="<config_dir>/mooncake.json"
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH

if [ "$ROLE" == "prefill" ]; then
    KV_ROLE="kv_producer"
    KV_PORT="20001"
    LOOKUP_RPC_PORT="0"
    API_PORT="8100"
else
    KV_ROLE="kv_consumer"
    KV_PORT="20002"
    LOOKUP_RPC_PORT="1"
    API_PORT="8200"
fi

echo "Starting vLLM on Series: $HARDWARE_SERIES, Role: $ROLE"

rm -rf /root/ascend/log/*
rm -rf ./connector.log

# See Configuration Parameters section for detailed parameter descriptions
if [ "$HARDWARE_SERIES" == "A2" ] || { [ "$HARDWARE_SERIES" == "A3" ] && [ "$LINK_TYPE" == "ROCE" ]; }; then
    echo 200000 > /proc/sys/vm/nr_hugepages
    export HCCL_IF_IP=$LOCAL_IP
    export GLOO_SOCKET_IFNAME=$NIC_NAME
    export TP_SOCKET_IFNAME=$NIC_NAME
    export HCCL_SOCKET_IFNAME=$NIC_NAME
    export HCCL_INTRA_ROCE_ENABLE=1

elif [ "$HARDWARE_SERIES" == "A3" ] && [ "$LINK_TYPE" == "HCCS" ]; then
    export ACL_OP_INIT_MODE=1
    export ASCEND_ENABLE_USE_FABRIC_MEM=1
elif [ "$HARDWARE_SERIES" == "A5" ]; then
    # 950PR&950DT Products UBOE
    export ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.protocol_desc":["uboe:device"]}'
    # 950PR&950DT Products UB
    export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'
else
    echo "Error: Invalid HARDWARE_SERIES. Set to 'A2', 'A3', or 'A5'."
    exit 1
fi

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

KV_CONFIG='{
  "kv_connector": "MultiConnector",
  "kv_role": "'$KV_ROLE'",
  "kv_connector_extra_config": {
    "connectors": [
      {
        "kv_connector": "MooncakeConnectorV1",
        "kv_role": "'$KV_ROLE'",
        "kv_port": "'$KV_PORT'",
        "kv_connector_extra_config": {
          "prefill": {
            "dp_size": '$DATA_PARALLEL_SIZE',
            "tp_size": '$TENSOR_PARALLEL_SIZE'
          },
          "decode": {
            "dp_size": '$DATA_PARALLEL_SIZE',
            "tp_size": '$TENSOR_PARALLEL_SIZE'
          }
        }
      },
      {
        "kv_connector": "AscendStoreConnector",
        "kv_role": "'$KV_ROLE'",
        "kv_connector_extra_config": {
          "backend": "mooncake",
          "lookup_rpc_port": "'$LOOKUP_RPC_PORT'"
        }
      }
    ]
  }
}'

CMD_ARGS=(
  --model "$MODEL_PATH"
  --served-model-name "$SERVED_MODEL_NAME"
  --trust-remote-code
  --enforce-eager
  --data-parallel-size "$DATA_PARALLEL_SIZE"
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
  --port "$API_PORT"
  --max-num-seqs 20
  --max-model-len 32768
  --max-num-batched-tokens 16384
  --gpu-memory-utilization 0.9
  --kv-transfer-config "$KV_CONFIG"
)

python -m vllm.entrypoints.openai.api_server "${CMD_ARGS[@]}" > log_${ROLE}.log 2>&1

echo "vLLM started. Log file: log_${ROLE}.log"
```

Start the proxy_server (connecting Prefill and Decode nodes):

```shell
python vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
    --host localhost \
    --prefiller-hosts localhost \
    --prefiller-ports 8100 \
    --decoder-hosts localhost \
    --decoder-ports 8200
```

Replace localhost with the actual IP address.

Run inference:

Short question:

```shell
curl -s http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{ "model": "qwen3", "prompt": "Hello. I have a question. The president of the United States is", "max_completion_tokens": 200, "temperature":0.0 }'
```

Long question:

```shell
curl -s http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{ "model": "qwen3", "prompt": "Given the accelerating impacts of climate change\u2014including rising sea levels, increasing frequency of extreme weather events, loss of biodiversity, and adverse effects on agriculture and human health\u2014there is an urgent need for a robust, globally coordinated response. However, international efforts are complicated by a range of factors: economic disparities between high-income and low-income countries, differing levels of industrialization, varying access to clean energy technologies, and divergent political systems that influence climate policy implementation. In this context, how can global agreements like the Paris Accord be redesigned or strengthened to not only encourage but effectively enforce emission reduction targets? Furthermore, what mechanisms can be introduced to promote fair and transparent technology transfer, provide adequate financial support for climate adaptation in vulnerable regions, and hold nations accountable without exacerbating existing geopolitical tensions or disproportionately burdening those with historically lower emissions?", "max_completion_tokens": 256, "temperature":0.0 }'
```

To enable Decode node KV Cache storage for Prefill use with MLA models, add `consumer_is_to_put: true` to `AscendStoreConnector`; if Prefill enables PP, also set `prefill_pp_size` or `prefill_pp_layer_partition`:

```json
{
    "kv_connector": "AscendStoreConnector",
    "kv_role": "kv_consumer",
    "kv_load_failure_policy": "recompute",
    "kv_connector_extra_config": {
        "lookup_rpc_port": "0",
        "backend": "mooncake",
        "consumer_is_to_put": true,
        "prefill_pp_size": 2,
        "prefill_pp_layer_partition": "30,31"
    }
}
```

Expected output: Returns a JSON response conforming to the OpenAI Completions API specification, containing `id`, `choices` (with `text` and `finish_reason`), `usage`, and other fields.

#### Step 4: PD-Mixed Scenario

##### pd_mix.sh

```shell
#!/bin/bash

# A2 (800I/800T A2) or A3 (800I/800T A3) or 950PR&950DT Products
HARDWARE_SERIES="A2"
# Link type: ROCE or HCCS in A3 series.
LINK_TYPE="ROCE"
LOCAL_IP="<local_ip>"
NIC_NAME="<nic_name>"

MODEL_PATH="<model_path>/Qwen3-32B"
SERVED_MODEL_NAME="qwen3"
DATA_PARALLEL_SIZE=1
TENSOR_PARALLEL_SIZE=8
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# parameters required for kv pool and mooncake
export PYTHONHASHSEED=0
export MOONCAKE_CONFIG_PATH="<config_dir>/mooncake.json"
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH

echo "Starting vLLM on Series: $HARDWARE_SERIES"

rm -rf /root/ascend/log/*
rm -rf ./connector.log

# See Configuration Parameters section for detailed parameter descriptions
if [ "$HARDWARE_SERIES" == "A2" ] || { [ "$HARDWARE_SERIES" == "A3" ] && [ "$LINK_TYPE" == "ROCE" ]; }; then
    echo 200000 > /proc/sys/vm/nr_hugepages
    export HCCL_IF_IP=$LOCAL_IP
    export GLOO_SOCKET_IFNAME=$NIC_NAME
    export TP_SOCKET_IFNAME=$NIC_NAME
    export HCCL_SOCKET_IFNAME=$NIC_NAME
    export HCCL_INTRA_ROCE_ENABLE=1

elif [ "$HARDWARE_SERIES" == "A3" ] && [ "$LINK_TYPE" == "HCCS" ]; then
    export ACL_OP_INIT_MODE=1
    export ASCEND_ENABLE_USE_FABRIC_MEM=1
elif [ "$HARDWARE_SERIES" == "A5" ]; then
    # 950PR&950DT Products UBOE
    export ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.protocol_desc":["uboe:device"]}'
    # 950PR&950DT Products UB
    export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'
else
    echo "Error: Invalid HARDWARE_SERIES. Set to 'A2', 'A3', or 'A5'."
    exit 1
fi

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

KV_CONFIG='{
  "kv_connector": "AscendStoreConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
     "backend": "mooncake",
     "lookup_rpc_port": "0"
     }
}'

CMD_ARGS=(
  --model "$MODEL_PATH"
  --served-model-name "$SERVED_MODEL_NAME"
  --trust-remote-code
  --enforce-eager
  --data-parallel-size "$DATA_PARALLEL_SIZE"
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
  --port 8100
  --max-num-seqs 20
  --max-model-len 32768
  --max-num-batched-tokens 16384
  --gpu-memory-utilization 0.9
  --kv-transfer-config "$KV_CONFIG"
)

python -m vllm.entrypoints.openai.api_server "${CMD_ARGS[@]}" > log_mix.log 2>&1

echo "vLLM started. Log file: log_mix.log"
```

Run inference (no separate proxy needed — requests go directly to the mixed deployment port):

Short question:

```shell
curl -s http://localhost:8100/v1/completions -H "Content-Type: application/json" -d '{ "model": "qwen3", "prompt": "Hello. I have a question. The president of the United States is", "max_completion_tokens": 200, "temperature":0.0 }'
```

Long question:

```shell
curl -s http://localhost:8100/v1/completions -H "Content-Type: application/json" -d '{ "model": "qwen3", "prompt": "Given the accelerating impacts of climate change\u2014including rising sea levels, increasing frequency of extreme weather events, loss of biodiversity, and adverse effects on agriculture and human health\u2014there is an urgent need for a robust, globally coordinated response. However, international efforts are complicated by a range of factors: economic disparities between high-income and low-income countries, differing levels of industrialization, varying access to clean energy technologies, and divergent political systems that influence climate policy implementation. In this context, how can global agreements like the Paris Accord be redesigned or strengthened to not only encourage but effectively enforce emission reduction targets? Furthermore, what mechanisms can be introduced to promote fair and transparent technology transfer, provide adequate financial support for climate adaptation in vulnerable regions, and hold nations accountable without exacerbating existing geopolitical tensions or disproportionately burdening those with historically lower emissions?", "max_completion_tokens": 256, "temperature":0.0 }'
```

Expected output: Returns a JSON response in OpenAI Completions API format.

**Note:** For MooncakeStore with `ASCEND_BUFFER_POOL` enabled, it is recommended to perform a warm-up phase before running actual performance benchmarks. Because HCCS one-sided communication connections are created lazily after instance launch, full-mesh connections require a one-time overhead (4 MB device memory per connection). Warm-up recommendation: input sequence length 8k, output sequence length 1, total requests 2-3x the number of devices.

```shell
# Example warm-up request
curl -s http://localhost:8100/v1/completions -H "Content-Type: application/json" -d '{ "model": "qwen3", "prompt": "Hello.", "max_completion_tokens": 1, "temperature":0.0 }'
```

#### Step 5: MooncakeStore SSD Offload (Embedded Real Client Mode)

In Mode A (Embedded Real Client), Mooncake is embedded in vLLM. When vLLM starts, `AscendStoreConnector`/`MooncakeBackend` automatically calls `MooncakeDistributedStore.setup()` using the settings in `mooncake.json`, with no separate `mooncake_client` process required.

SSD disk usage control environment variables:

```shell
# 800 GB total disk, 8 TP ranks, ~100 GB per rank
export MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES=$((100 * 1024 * 1024 * 1024))
export MOONCAKE_OFFLOAD_BUCKET_MAX_TOTAL_SIZE=$((100 * 1024 * 1024 * 1024))
export MOONCAKE_OFFLOAD_BUCKET_EVICTION_POLICY=lru
export MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES=1073741824   # 1 GB
```

`MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES` risk: If not aligned to 1 GB (A3 + FabricMem scenario), it may cause `adxl MallocMem` failures or `FileStorage init` segfaults. Always set to a multiple of 1 GB.

Note: `--max-num-batched-tokens` only chunks prefill compute; it does not reduce the memory required by `MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES`.

<details markdown="1">
<summary>Mooncake SSD Environment Variables</summary>

| `Parameter` | Type | Default | Required | Value Range | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES` | int (bytes) | 1342177280 (1280 MB) | No | Must align to 1 GB when A3 + `ASCEND_ENABLE_USE_FABRIC_MEM=1` | Per-rank SSD read/write buffer size (bytes). Not configurable in mooncake.json. Increase when `BUFFER_OVERFLOW` occurs. |
| `MOONCAKE_OFFLOAD_BUCKET_MAX_TOTAL_SIZE` | int (bytes) | 0 | No | Bytes; 0 means 90% of physical disk capacity | Eviction threshold (bytes). Set an explicit value to control disk usage precisely. |
| `MOONCAKE_OFFLOAD_BUCKET_EVICTION_POLICY` | str | none | No | none / fifo / lru | SSD eviction policy: `none` (writes fail when full), `fifo`, or `lru`. |
| `MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES` | int (bytes) | 2199023255552 (2 TB) | Strongly recommended to set explicitly | Must match actual disk capacity | Per-rank maximum disk usage reported to Mooncake master, aggregated by master across clients (roughly 2 TB x rank count). Default far exceeds real disk capacity; must be overridden. |

</details>

### Scenario 2: Memcache Backend

Before starting the installation, complete the prerequisite checks in Step 1 (memory scan, 950PR&950DT Products signature verification disabling and container mounts, SSD disk status checks).

#### Step 1: Prerequisite Checks

**Check memory:**

```shell
free -h
```

If `free -h` shows that excessive cache usage affects the available KV cache pool size, you can optionally run the following commands to release the cache and compact memory fragments:

```shell
# Release pagecache/dentry/inode to free contiguous physical memory
echo 3 > /proc/sys/vm/drop_caches
# Trigger memory compaction to reduce fragmentation
echo 1 > /proc/sys/vm/compact_memory
```

**A3 only: scan available memory:**

```shell
python3 mem_scan.py                   # 1GB specification scan
python3 mem_scan.py -m 2              # 2MB huge page scan
```

Script location: [mem_scan.py](https://gitcode.com/Ascend/memfabric_hybrid/blob/develop/src/smem/python/memfabric_hybrid/memfabric_hybrid/mem_scan.py)

**950PR&950DT Products only (disable signature verification + mount key paths + install kernel package):**

```shell
# Step 1: Disable HDK signature verification (only needs to be executed once per machine)
for i in {0..7}; do npu-smi set -t custom-op-secverify-enable -i $i -d 1; done;
for i in {0..7}; do npu-smi set -t custom-op-secverify-mode -i $i -d 0; done;
```

Docker containers need to mount key paths. Example command:

```shell
docker run -u root -it -d --name ${NAME} --net=host --privileged=true \
    --device=/dev/davinci_manager --device=/dev/hisi_hdc --device=/dev/ummu --device=/dev/uburma \
    --device=/dev/davinci0 --device=/dev/davinci1 --device=/dev/davinci2 --device=/dev/davinci3 \
    --device=/dev/davinci4 --device=/dev/davinci5 --device=/dev/davinci6 --device=/dev/davinci7 \
    -v /usr/bin/urma_admin:/usr/bin/urma_admin \
    -v /lib/route.conf:/lib/route.conf \
    -v /etc/hccl_rootinfo.json:/etc/hccl_rootinfo.json \
    -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
    -v /usr/local/sbin:/usr/local/sbin \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /var/log/npu/:/usr/slog \
    -v /etc/hccn.conf:/etc/hccn.conf \
    -v /etc/hixlep:/etc/hixlep \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -w /home \
    ${IMAGES_ID} \
    bash
```

Update `/lib/route.conf` inside the container.

**Check disk status before enabling SSD:**

```shell
lsblk /dev/nvme1n1                    # No partitions expected
mount | grep nvme1n1                  # No mount points expected
blkid /dev/nvme1n1                    # No filesystem signature expected
```

If no physical disk is available, simulate using a loop device:

```shell
dd if=/dev/zero of=/data/boostio_disk.img bs=1G count=640 status=progress
LOOP_DEV=$(losetup --find --show --direct-io=on /data/boostio_disk.img)
echo "${LOOP_DEV}"
```

#### Step 2: Software Installation

```shell
pip install memfabric-hybrid
pip install memcache-hybrid
```

#### Step 3: Configure Memcache Config File

Find the installation path:

```shell
pip show memcache_hybrid
```

Use `{INSTALL_PATH}` to denote the `Location` value in the output.

##### mmc-meta.conf

```ini
ock.mmc.meta_service_url = tcp://<meta_service_ip>:5000
ock.mmc.meta_service.config_store_url = tcp://<meta_service_ip>:6000
ock.mmc.meta_service.metrics_url = http://<meta_service_ip>:8000
ock.mmc.log_level = info
# Tune the following parameters when SSD is enabled to improve SSD cache hit rate
ock.mmc.evict_threshold_high = 70
ock.mmc.evict_threshold_low = 60
```

<details markdown="1">
<summary>mmc-meta.conf Parameters</summary>

| `Parameter` | Type | Default | Required | Value Range | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `ock.mmc.meta_service_url` | str | tcp://127.0.0.1:5000 | No | tcp://<host>:<port>, port [1025, 65535] | MetaService address. P and D nodes must use the same endpoint. Host supports IP and domain. |
| `ock.mmc.meta_service.config_store_url` | str | tcp://127.0.0.1:6000 | No | tcp://<host>:<port>, port [1025, 65535] | Config store URL. |
| `ock.mmc.meta_service.metrics_url` | str | http://127.0.0.1:8000 | No | http(s)://<host>:<port>, port [1025, 65535] | Metrics URL. |
| `ock.mmc.log_level` | str | info | No | debug / info / warn / error | Log level; shared by MetaService and LocalService. |
| `ock.mmc.evict_threshold_high` | int | 90 | No | [1, 99] (percentage) | High watermark of the secondary (L2) pool; eviction is triggered when this watermark is reached, and only when `usedSize * 100 > totalSize * threshold`. When SSD is enabled, 70 is recommended to improve SSD cache hit rate. |
| `ock.mmc.evict_threshold_low` | int | 80 | No | [1, 98] (percentage) | Low watermark of the secondary (L2) pool; eviction stops at this watermark. When SSD is enabled, 60 is recommended. |

</details>

##### mmc-local.conf

```ini
ock.mmc.meta_service_url = tcp://<meta_service_ip>:5000
ock.mmc.local_service.config_store_url = tcp://<meta_service_ip>:6000
ock.mmc.log_level = info
ock.mmc.local_service.world_size = 256
ock.mmc.local_service.protocol = device_sdma
ock.mmc.local_service.dram.size = 1GB
ock.mmc.local_service.max.dram.size = 1024GB
# SSD feature related parameters below
ock.mmc.local_service.storage.enabled = false
ubsio.disk.path = /dev/nvmexn1:/dev/nvmexn2p1:/dev/loopX
ubsio.mem.size_in_gb = 10
ubsio.standalone.device_count = 8
ubsio.standalone.force_new_disk = true
```

<details markdown="1">
<summary>mmc-local.conf Parameters</summary>

| `Parameter` | Type | Default | Required | Value Range | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `ock.mmc.meta_service_url` | str | tcp://127.0.0.1:5000 | No | tcp://<host>:<port>, port [1025, 65535] | Must match mmc-meta.conf (host is cluster-ip or domain in HA deployments). |
| `ock.mmc.local_service.config_store_url` | str | tcp://127.0.0.1:6000 | No | tcp://<host>:<port>, port [1025, 65535] | Must match `ock.mmc.meta_service.config_store_url` in mmc-meta.conf. |
| `ock.mmc.local_service.world_size` | int | 256 | No | [1, 1024] | Maximum number of LocalServices supported (including future additions). Once ranks are connected, no further modifications are allowed — a meta restart is required. |
| `ock.mmc.local_service.protocol` | str | host_rdma | Yes | host_rdma / host_urma / host_tcp / host_shm / device_sdma / device_rdma / device_urma / device_uboe | Communication protocol. A2 recommended: `device_rdma` (RoCE); A3 HCCS recommended: `device_sdma` (requires LingQu Computing Network >= 1.5); 950PR&950DT Products UB: `device_urma`; UBOE: `device_uboe`. `host_shm` requires DRAM > 0 and HBM = 0. |
| `ock.mmc.local_service.dram.size` | int | 1GB | Yes | [0, 1TB], auto-aligned to 2MB (host_* protocols) or 1GB (device_* protocols) | DRAM size allocated per die; supports formats such as 134217728, 2048KB, 200MB, 2.5GB, or 1TB. For example, on A3, to allocate 640GB as KV pool, set this parameter to 640/16=40GB. Set 0GB for A3 when HCCS is available. |
| `ock.mmc.local_service.max.dram.size` | int | 1TB | No | [0, 1TB] | Maximum DRAM size. Needed when ranks contribute different sizes of DRAM. The default 1TB is binary 1024^4 bytes. |
| `ock.mmc.local_service.storage.enabled` | bool | false | No | true / false | Enable SSD caching. |
| `ubsio.disk.path` | str | No default | Required when SSD enabled | Absolute paths, multiple paths separated by `:` | SSD block device, partition, or loop device paths. Devices must be dedicated with no mount points or filesystem signatures. `/dev/sd*` not recommended. |
| `ubsio.mem.size_in_gb` | int | 10 | No | Integer [0, 3072]; SSD cache requires >= 5 | DRAM size requested by UBS IO per die (i.e., per LocalService/process), in GB. Data storage flow: HBM → second-level pooled DRAM → third-level pooled DRAM → SSD. The total allocation must not exceed the node memory available after reserving memory for the operating system, vLLM, and the Memcache DRAM pool. Typically 10; 50 recommended for separated deployment. |
| `ubsio.standalone.device_count` | int | No default | Yes | Positive integer | Number of DRAM-enabled LocalServices, i.e., the number of vLLM processes on the environment (8 on A2, 16 on A3; if fewer cards are actually used, configure the actual process count, e.g., 4 when only 4 cards are occupied). |
| `ubsio.standalone.force_new_disk` | bool | true | No | true / false | Clears the mounted disks each time the service is started. The current version does not support fault recovery; keeping true is recommended. |

</details>

#### Step 4: Run MetaService

```shell
export MMC_META_CONFIG_PATH={INSTALL_PATH}/memcache_hybrid/config/mmc-meta.conf

python -c "from memcache_hybrid import MetaService; MetaService.main()"
```

Expected output: MetaService starts successfully with no errors.

#### Step 5: PD Disaggregation Scenario

##### run_prefill.sh / run_decode.sh

```shell
#!/bin/bash

# prefill / decode
ROLE="prefill"
# A2 (800I/800T A2) or A3 (800I/800T A3) or 950PR&950DT Products
HARDWARE_SERIES="A2"
# Link type: ROCE or HCCS in A3 series.
LINK_TYPE="ROCE"
LOCAL_IP="<local_ip>"
NIC_NAME="<nic_name>"

MODEL_PATH="<model_path>/Qwen3-32B"
SERVED_MODEL_NAME="qwen3"
DATA_PARALLEL_SIZE=1
TENSOR_PARALLEL_SIZE=8
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# parameters required for kv pool and memcache
export PYTHONHASHSEED=0
export MMC_LOCAL_CONFIG_PATH={INSTALL_PATH}/memcache_hybrid/config/mmc-local.conf
export LD_LIBRARY_PATH={INSTALL_PATH}/memcache_hybrid/lib:${PYTHON_LIB_DIR}:${LD_LIBRARY_PATH}

if [ "$ROLE" == "prefill" ]; then
    KV_ROLE="kv_producer"
    KV_PORT="20001"
    LOOKUP_RPC_PORT="0"
    API_PORT="8100"
else
    KV_ROLE="kv_consumer"
    KV_PORT="20002"
    LOOKUP_RPC_PORT="1"
    API_PORT="8200"
fi

echo "Starting vLLM on Series: $HARDWARE_SERIES, Role: $ROLE"

rm -rf /root/ascend/log/*
rm -rf ./connector.log

# See Configuration Parameters section for detailed parameter descriptions
if [ "$HARDWARE_SERIES" == "A2" ] || { [ "$HARDWARE_SERIES" == "A3" ] && [ "$LINK_TYPE" == "ROCE" ]; }; then
    echo 200000 > /proc/sys/vm/nr_hugepages
    export HCCL_IF_IP=$LOCAL_IP
    export GLOO_SOCKET_IFNAME=$NIC_NAME
    export TP_SOCKET_IFNAME=$NIC_NAME
    export HCCL_SOCKET_IFNAME=$NIC_NAME
    export HCCL_INTRA_ROCE_ENABLE=1

elif [ "$HARDWARE_SERIES" == "A3" ] && [ "$LINK_TYPE" == "HCCS" ]; then
    export ACL_OP_INIT_MODE=1
    export ASCEND_ENABLE_USE_FABRIC_MEM=1
elif [ "$HARDWARE_SERIES" == "A5" ]; then
    # 950PR&950DT Products UBOE
    export ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.protocol_desc":["uboe:device"]}'
    # 950PR&950DT Products UB
    export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'
else
    echo "Error: Invalid HARDWARE_SERIES. Set to 'A2', 'A3', or 'A5'."
    exit 1
fi

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

KV_CONFIG='{
  "kv_connector": "MultiConnector",
  "kv_role": "'$KV_ROLE'",
  "kv_connector_extra_config": {
    "connectors": [
      {
        "kv_connector": "MooncakeConnectorV1",
        "kv_role": "'$KV_ROLE'",
        "kv_port": "'$KV_PORT'",
        "kv_connector_extra_config": {
          "prefill": {
            "dp_size": '$DATA_PARALLEL_SIZE',
            "tp_size": '$TENSOR_PARALLEL_SIZE'
          },
          "decode": {
            "dp_size": '$DATA_PARALLEL_SIZE',
            "tp_size": '$TENSOR_PARALLEL_SIZE'
          }
        }
      },
      {
        "kv_connector": "AscendStoreConnector",
        "kv_role": "'$KV_ROLE'",
        "kv_connector_extra_config": {
          "backend": "memcache",
          "lookup_rpc_port": "'$LOOKUP_RPC_PORT'",
          "use_layerwise": false
        }
      }
    ]
  }
}'

CMD_ARGS=(
  --model "$MODEL_PATH"
  --served-model-name "$SERVED_MODEL_NAME"
  --trust-remote-code
  --enforce-eager
  --data-parallel-size "$DATA_PARALLEL_SIZE"
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
  --port "$API_PORT"
  --max-num-seqs 20
  --max-model-len 32768
  --max-num-batched-tokens 16384
  --gpu-memory-utilization 0.9
  --kv-transfer-config "$KV_CONFIG"
)

python -m vllm.entrypoints.openai.api_server "${CMD_ARGS[@]}" > log_${ROLE}.log 2>&1

echo "vLLM started. Log file: log_${ROLE}.log"
```

`use_layerwise` can be set to `true` to enable per-layer KV access, supported by both the Mooncake and Memcache backends; it is only supported on the Prefill node. `consumer_is_to_put` and `consumer_is_to_load` can also be configured via `kv_connector_extra_config`.

To start proxy_server and run inference, refer to the corresponding sub-steps in "Step 3: PD Disaggregation Scenario" of [Scenario 1: Mooncake Backend](#scenario-1-mooncake-backend).

Expected output: Same as Mooncake scenario, returns OpenAI Completions API JSON response.

#### Step 6: PD-Mixed Scenario

##### pd_mix.sh

```shell
#!/bin/bash

# A2 (800I/800T A2) or A3 (800I/800T A3) or 950PR&950DT Products
HARDWARE_SERIES="A2"
# Link type: ROCE or HCCS in A3 series.
LINK_TYPE="ROCE"
LOCAL_IP="<local_ip>"
NIC_NAME="<nic_name>"

MODEL_PATH="<model_path>/Qwen3-32B"
SERVED_MODEL_NAME="qwen3"
DATA_PARALLEL_SIZE=1
TENSOR_PARALLEL_SIZE=8
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# parameters required for kv pool and memcache
export PYTHONHASHSEED=0
export MMC_LOCAL_CONFIG_PATH={INSTALL_PATH}/memcache_hybrid/config/mmc-local.conf
export LD_LIBRARY_PATH={INSTALL_PATH}/memcache_hybrid/lib:${PYTHON_LIB_DIR}:${LD_LIBRARY_PATH}

echo "Starting vLLM on Series: $HARDWARE_SERIES"

rm -rf /root/ascend/log/*
rm -rf ./connector.log

# See Configuration Parameters section for detailed parameter descriptions
if [ "$HARDWARE_SERIES" == "A2" ] || { [ "$HARDWARE_SERIES" == "A3" ] && [ "$LINK_TYPE" == "ROCE" ]; }; then
    echo 200000 > /proc/sys/vm/nr_hugepages
    export HCCL_IF_IP=$LOCAL_IP
    export GLOO_SOCKET_IFNAME=$NIC_NAME
    export TP_SOCKET_IFNAME=$NIC_NAME
    export HCCL_SOCKET_IFNAME=$NIC_NAME
    export HCCL_INTRA_ROCE_ENABLE=1

elif [ "$HARDWARE_SERIES" == "A3" ] && [ "$LINK_TYPE" == "HCCS" ]; then
    export ACL_OP_INIT_MODE=1
    export ASCEND_ENABLE_USE_FABRIC_MEM=1
elif [ "$HARDWARE_SERIES" == "A5" ]; then
    # 950PR&950DT Products UBOE
    export ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.protocol_desc":["uboe:device"]}'
    # 950PR&950DT Products UB
    export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'
else
    echo "Error: Invalid HARDWARE_SERIES. Set to 'A2', 'A3', or 'A5'."
    exit 1
fi

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

KV_CONFIG='{
  "kv_connector": "AscendStoreConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
     "backend": "memcache",
     "lookup_rpc_port": "0",
     "use_layerwise": false
  }
}'

CMD_ARGS=(
  --model "$MODEL_PATH"
  --served-model-name "$SERVED_MODEL_NAME"
  --trust-remote-code
  --enforce-eager
  --data-parallel-size "$DATA_PARALLEL_SIZE"
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
  --port 8100
  --max-num-seqs 20
  --max-model-len 32768
  --max-num-batched-tokens 16384
  --gpu-memory-utilization 0.9
  --kv-transfer-config "$KV_CONFIG"
)

python -m vllm.entrypoints.openai.api_server "${CMD_ARGS[@]}" > log_mix.log 2>&1

echo "vLLM started. Log file: log_mix.log"
```

For inference commands, refer to "Step 4: PD-Mixed Scenario" of [Scenario 1: Mooncake Backend](#scenario-1-mooncake-backend).

#### Step 7: Memcache and vLLM Separated Deployment

This deployment mode runs MemCache and vLLM in different processes (distinct from vLLM PD disaggregation). In the default co-located mode, vLLM loads the model weights before the KV connector initializes MemCache, so MemCache may not be able to reserve sufficient memory from the remaining available space. Starting a standalone MemCache process before vLLM allows MemCache to reserve a larger memory pool.

Prepare two LocalService configuration files with the same connection and protocol settings: the configuration used by the vLLM process does not contribute DRAM (`dram.size = 0GB`), and the configuration used by the standalone MemCache process specifies the amount of DRAM to contribute. Set `ock.mmc.local_service.max.dram.size` to accommodate the maximum `dram.size` used by the LocalService processes.

Steps:

1. Start MetaService (same as above).
2. Use the following `mmc-local-standalone.conf` configuration to start an independent Memcache process on each node:

   ```ini
   ock.mmc.local_service.dram.size = 600GB
   ock.mmc.local_service.max.dram.size = 1024GB
   ```

3. Wait for all nodes to report successful initialization.
4. Use the following `mmc-local.conf` configuration (`dram.size = 0GB`) to start vLLM:

   ```ini
   ock.mmc.local_service.dram.size = 0GB
   ock.mmc.local_service.max.dram.size = 1024GB
   ```

Startup script reference: [Memcache + vLLM + A3 Separated Deployment Case](https://gitcode.com/Ascend/memcache/wiki/MemCache+vLLM+A3%E5%88%86%E7%A6%BB%E9%83%A8%E7%BD%B2%E6%A1%88%E4%BE%8B.md)

#### Step 8: Enable Memcache SSD Cache

When enabling SSD Cache in the separated deployment scenario, refer to the following configurations:

##### mmc-local.conf (used by the vLLM process, contributing no DRAM)

```ini
ock.mmc.meta_service_url = tcp://<meta_service_ip>:5000
ock.mmc.local_service.config_store_url = tcp://<meta_service_ip>:6000
ock.mmc.log_level = info
ock.mmc.local_service.world_size = 256
ock.mmc.local_service.protocol = device_sdma
ock.mmc.local_service.dram.size = 0GB
ock.mmc.local_service.max.dram.size = 1024GB
```

##### mmc-local-standalone.conf (used by the standalone MemCache process)

```ini
ock.mmc.meta_service_url = tcp://<meta_service_ip>:5000
ock.mmc.local_service.config_store_url = tcp://<meta_service_ip>:6000
ock.mmc.log_level = info
ock.mmc.local_service.world_size = 256
ock.mmc.local_service.protocol = device_sdma
ock.mmc.local_service.dram.size = 600GB
ock.mmc.local_service.max.dram.size = 1024GB
# SSD feature related parameters below
ock.mmc.local_service.storage.enabled = true
ubsio.disk.path = /dev/nvmexn1:/dev/nvmexn2p1:/dev/loopX
ubsio.mem.size_in_gb = 50
ubsio.standalone.device_count = 1
ubsio.standalone.force_new_disk = true
```

When adjusting `ubsio.mem.size_in_gb`, calculate the maximum permitted per-process value by dividing the node memory available to UBS IO by the number of DRAM-enabled local services, rounding down, and capping the result at `3072`:

```text
maximum ubsio.mem.size_in_gb = min(3072, floor(available node memory for UBS IO (GB) / number of DRAM-enabled local services))
```

For example, if `200` GB is available to UBS IO and four local services have DRAM enabled, the upper limit is `50` GB per process. If the calculated upper limit is less than `5`, free more node memory or reduce the number of DRAM-enabled local services.

For the scenario of separate deployment of MemCache, it is recommended to configure a single process with `50` GB. In other scenarios, the recommended value is `10` GB. If you want to use the L2.5 memory caching capability, increase `ubsio.mem.size_in_gb` within the limits above and adjust [ubsio.wcache.evict_water_level](https://gitcode.com/Ascend/memcache/wiki/DRAM%20+%20SSD%20%E5%A4%9A%E7%BA%A7%E6%B1%A0%E5%8C%96%E9%85%8D%E7%BD%AE%E6%8C%87%E5%8D%97.md#ubsiowcacheevict_water_level) accordingly.

For disk config, eviction watermarks, and other UBS IO parameters, see the [DRAM + SSD Multi-level Pooling Configuration Guide](https://gitcode.com/Ascend/memcache/wiki/DRAM%20+%20SSD%20%E5%A4%9A%E7%BA%A7%E6%B1%A0%E5%8C%96%E9%85%8D%E7%BD%AE%E6%8C%87%E5%8D%97.md).

### Scenario 3: Yuanrong Backend

#### Step 1: Install Yuanrong Datasystem

```bash
pip install openyuanrong-datasystem
python -c "import yr.datasystem; print('Yuanrong Datasystem is ready')"
dscli --version
```

Expected output:
- `Yuanrong Datasystem is ready`
- Version number displayed by `dscli`.

If the prebuilt package does not match the CANN or driver version, build Yuanrong Datasystem from source: [Yuanrong Datasystem](https://atomgit.com/openeuler/yuanrong-datasystem).

#### Step 2: Choose a Service Discovery Backend

##### Option 1: Start Coordinator

```bash
COORDINATOR_ADDRESS="<coordinator_ip>:31511"

dscli start -c \
  --coordinator_address "${COORDINATOR_ADDRESS}"
```

Expected output: `Start coordinator service ... success`.

Single-node quick start (Coordinator + Worker in one step):

```bash
dscli start -a \
  --coordinator_address "127.0.0.1:31511" \
  --worker_address "127.0.0.1:31501" \
  --shared_memory_size_mb 4096
```

##### Option 2: Start etcd

```bash
ETCD_VERSION="v3.5.12"
ETCD_IP="127.0.0.1"
if [ "$(uname -m)" = "aarch64" ]; then
  ETCD_ARCH="linux-arm64"
else
  ETCD_ARCH="linux-amd64"
fi
wget https://github.com/etcd-io/etcd/releases/download/${ETCD_VERSION}/etcd-${ETCD_VERSION}-${ETCD_ARCH}.tar.gz
tar -xvf etcd-${ETCD_VERSION}-${ETCD_ARCH}.tar.gz
cd etcd-${ETCD_VERSION}-${ETCD_ARCH}
sudo cp etcd etcdctl /usr/local/bin/

etcd \
  --name etcd-single \
  --data-dir /tmp/etcd-data \
  --listen-client-urls http://0.0.0.0:2379 \
  --advertise-client-urls http://${ETCD_IP}:2379 \
  --listen-peer-urls http://0.0.0.0:2380 \
  --initial-advertise-peer-urls http://${ETCD_IP}:2380 \
  --initial-cluster etcd-single=http://${ETCD_IP}:2380 &

etcdctl --endpoints "${ETCD_IP}:2379" put key "value"
etcdctl --endpoints "${ETCD_IP}:2379" get key
```

Expected output: `etcdctl put key "value"` returns `OK`; `etcdctl get key` returns `key`->`value`.

For production environments, refer to the official etcd clustering documentation: [etcd clustering guide](https://etcd.io/docs/v3.7/op-guide/clustering/).

##### Multi-node deployment

Install Yuanrong Datasystem on every node. Each node runs one Datasystem Worker with a unique, reachable `worker_address`. All Workers must use the same service discovery backend and backend address.

Multi-node deployment with Coordinator (example with Coordinator node address `192.168.1.10`):

```bash
# Run once on the Coordinator node.
dscli start -c \
  --coordinator_address "192.168.1.10:31511"
```

```bash
# Run on every Worker node. Set WORKER_IP to that node's own IP;
# keep COORDINATOR_ADDRESS identical on all nodes.
WORKER_IP="<this_node_ip>"
COORDINATOR_ADDRESS="192.168.1.10:31511"

dscli start -w \
  --worker_address "${WORKER_IP}:31501" \
  --coordinator_address "${COORDINATOR_ADDRESS}" \
  --shared_memory_size_mb 4096
```

Multi-node deployment with etcd: start an etcd service or cluster that every Worker can reach, then start one Worker on every node. Set `WORKER_IP` to that node's own IP and keep `ETCD_ADDRESS` identical on all nodes:

```bash
# Run on every Worker node.
WORKER_IP="<this_node_ip>"
ETCD_ADDRESS="192.168.1.10:2379"

dscli start -w \
  --worker_address "${WORKER_IP}:31501" \
  --etcd_address "${ETCD_ADDRESS}" \
  --shared_memory_size_mb 4096
```

For both backends:

* Do not use `127.0.0.1` or `0.0.0.0` as a Worker address in a multi-node deployment. Other Workers must be able to connect to the advertised IP.
* Allow network access to the Coordinator port (`31511`) or etcd client port (`2379`), and to every Worker port (`31501` in these examples).
* On each node, set `worker_addr` in `yuanrong.json` to that node's local `WORKER_IP:31501`. The configuration file therefore differs by node.

For production control-plane high availability, deploy multiple Coordinators with static Raft peers, a unique `coordinator_address` and `coordinator_raft_data_dir` for each Coordinator, and the same `coordinator_raft_initial_peers` list. See the [Yuanrong Datasystem dscli documentation](https://atomgit.com/openeuler/yuanrong-datasystem/blob/master/docs/source_zh_cn/deployment/dscli.md#coordinator-%E5%A4%9A%E8%8A%82%E7%82%B9%E9%83%A8%E7%BD%B2).

#### Step 3: Start Datasystem Worker

```bash
COORDINATOR_ADDRESS="<coordinator_ip>:31511"
WORKER_IP="<worker_ip>"
WORKER_LOG_DIR="/var/log/yuanrong/worker"
sudo mkdir -p "${WORKER_LOG_DIR}"
sudo chown "$(id -u):$(id -g)" "${WORKER_LOG_DIR}"

dscli start -w \
  --worker_address "${WORKER_IP}:31501" \
  --coordinator_address "${COORDINATOR_ADDRESS}" \
  --log_dir "${WORKER_LOG_DIR}" \
  --shared_memory_size_mb 40960 \
  --arena_per_tenant 1 \
  --enable_huge_tlb true \
  --enable_fallocate false \
  --rpc_thread_num 64 \
  --oc_thread_num 64 \
  --enable_worker_worker_batch_get true \
  --sc_regular_socket_num 0 \
  --sc_stream_socket_num 0
```

Expected output: Worker starts successfully with no errors. Worker logs, including files whose base name is normally `datasystem_worker`, are written under the `--log_dir` directory. Use an absolute path so the log location does not depend on the worker process's current directory.

The `--worker_address` value is consumed later as `worker_addr` in `yuanrong.json`, so keep the host and port identical on the same node. Configure only one coordination backend for a Worker: when using Coordinator, do not also set `etcd_address` or `metastore_address`.

Stop the Worker when it is no longer needed. Stop the selected service discovery backend only after all Workers have stopped. For an independent etcd cluster, follow its normal cluster maintenance procedure:

```bash
dscli stop --worker_address "${WORKER_IP}:31501"
# Coordinator option only:
dscli stop --coordinator_address "${COORDINATOR_ADDRESS}"
```

<details markdown="1">
<summary>Yuanrong dscli Worker Parameters</summary>

| `Parameter` | Type | Default | Required | Value Range | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `worker_address` | str | No default | Yes | `<host>:<port>` | Worker address, must match `yuanrong.json` `worker_addr`. |
| `coordinator_address` | str | No default | Option 1 | `<ip>:31511` | Coordinator address. |
| `etcd_address` | str | No default | Option 2 | `<ip>:2379` | etcd address. |
| `log_dir` | str | No default | No | Absolute path | Worker log directory, use absolute path. Create the directory and grant the worker process write permission before startup. |
| `shared_memory_size_mb` | int | No default | Yes | Positive integer (MB) | Shared memory size (MB). Example: 40960 (40 GB). |
| `arena_per_tenant` | int | No default | No | Positive integer | Shared memory arena count per tenant. Conservative starting point: 1. |
| `enable_huge_tlb` | bool | No default | No | true / false | Use HugeTLB pages for shared memory. Reserve enough 2 MiB huge pages before starting the worker. |
| `enable_fallocate` | bool | No default | No | true / false | Execute fallocate for shared memory file. Recommended false with HugeTLB. |
| `rpc_thread_num` | int | No default | No | Positive integer | RPC/ZMQ service concurrency. |
| `oc_thread_num` | int | No default | No | Positive integer | Object Cache business thread pool size. |
| `enable_worker_worker_batch_get` | bool | No default | No | true / false | Enable batched Object Cache reads between Workers. |
| `sc_regular_socket_num` | int | 0 | No | >= 0 | Stream Cache regular socket count. Keep at 0 when KV Pool does not use Stream Cache. |
| `sc_stream_socket_num` | int | 0 | No | >= 0 | Stream Cache stream socket count. Keep at 0 when KV Pool does not use Stream Cache. |
| `remote_h2d_device_ids` | str | empty | No | Comma-separated device IDs, e.g., `"0,1,2,3,4,5,6,7"` | Non-empty enables worker-side Remote H2D. Using multiple available NPU device IDs is recommended. |
| `remote_h2d_link_type` | str | ROCE | No | ROCE / HCCS (case-sensitive) | Link type. `ROCE` maps to client `P2P_TRANSFER`; `HCCS` maps to client `HIXL` (covers buffer-pool, HIXL RoCE direct, and FabricMem sub-modes). For `HCCS`, the client process must also export `DS_RH2D_LINK_TYPE=HCCS` before starting vLLM (the backend does not export it automatically); `ROCE` is the datasystem default and needs no env var. |
| `remote_h2d_hccs_buffer_pool` | str | 4:8 | No | `<count>:<size>` | HIXL buffer-pool parameter, only used when `link_type=HCCS`. Ignored under HIXL RoCE direct mode (`HCCL_INTRA_ROCE_ENABLE=1`). |

</details>

#### Step 4: Configure Environment Variables and `yuanrong.json`

```bash
export PYTHONHASHSEED=0
export DS_WORKER_ADDR="${WORKER_IP}:31501"
export DATASYSTEM_CLIENT_LOG_DIR="/var/log/yuanrong/client"
export DS_ENABLE_EXCLUSIVE_CONNECTION=0
export DS_ENABLE_REMOTE_H2D=0

mkdir -p "${DATASYSTEM_CLIENT_LOG_DIR}"
```

<details markdown="1">
<summary>Yuanrong Environment Variables</summary>

| `Variable` | Type | Default | Required | Value Range | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `PYTHONHASHSEED` | int | 0 | Yes | Integer (consistent across all nodes) | Must be consistent across all nodes to guarantee uniform hash generation. |
| `DS_WORKER_ADDR` | str | N/A | Yes | `<host>:<port>` | Datasystem Worker address, must match local `dscli start --worker_address` value. |
| `DATASYSTEM_CLIENT_LOG_DIR` | str | ~/.datasystem/logs | No | Directory path | Directory for Yuanrong client SDK logs (base name normally `ds_client`). Must be set before starting vLLM; use a directory separate from the worker logs. |
| `DS_ENABLE_EXCLUSIVE_CONNECTION` | int | 0 | No | 0 / 1 | Passed to Yuanrong `HeteroClient.enable_exclusive_connection`. Use `1` to enable the exclusive connection mode when required by your deployment. |
| `DS_ENABLE_REMOTE_H2D` | int | 0 | No | 0 / 1 | Passed to Yuanrong `HeteroClient.enable_remote_h2d`. Use `1` only after the Remote H2D requirements are met. |

</details>

`DATASYSTEM_CLIENT_LOG_DIR` before starting vLLM because the Yuanrong client reads it during logging initialization. Client SDK logs, whose base name is normally `ds_client`, are written to this directory. Use a directory separate from the worker logs.

Configure `yuanrong.json` (pointed to by `YR_CONFIG_PATH`):

```json
{
    "worker_addr": "<worker_ip>:31501",
    "connect_timeout_ms": 9000,
    "request_timeout_ms": 0,
    "get_sub_timeout_ms": 0,
    "enable_remote_h2d": false,
    "remote_h2d_transport_backend": "HIXL",
    "enable_fabric_mem": false,
    "enable_dev_mem_pregister": false,
    "use_layerwise": false
}
```

<details markdown="1">
<summary>yuanrong.json Fields</summary>

| `Field` | Type | Default | Required | Value Range | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `worker_addr` | str | No default | Yes | `<host>:<port>` | Datasystem Worker address, must match `dscli start --worker_address`. |
| `connect_timeout_ms` | int | 9000 | No | Integer >= 500 | Connection establishment timeout (ms). |
| `request_timeout_ms` | int | 0 | No | 0 or positive integer (ms) | Request timeout (ms). `0` preserves the Yuanrong SDK behavior of using `connect_timeout_ms` as the request timeout; set a positive value to control request timeout independently. |
| `get_sub_timeout_ms` | int | 0 | No | 0 or positive integer (ms) | Timeout for `mget_h2d_from_multi_buffers` to wait for objects to become ready (ms); `0` means no waiting. It may be greater than `request_timeout_ms`; the Yuanrong Get path expands that call's RPC timeout to accommodate the configured object-ready wait. |
| `enable_remote_h2d` | bool | false | No | true / false | Passed to Yuanrong `HeteroClient.enable_remote_h2d`. Use `true` only after the Remote H2D requirements are met. |
| `remote_h2d_transport_backend` | str | HIXL | No | HIXL / P2P_TRANSFER | vLLM-side transport name, corresponds to Worker `--remote_h2d_link_type` (HCCS ↔ HIXL, ROCE ↔ P2P_TRANSFER). Under `HIXL` the backend pre-registers device memory unless `enable_fabric_mem` is `true`; under `P2P_TRANSFER` it skips pre-registration. |
| `enable_fabric_mem` | bool | false | No | true / false | Selects HIXL FabricMem mode, where HIXL `OPTION_ENABLE_USE_FABRIC_MEM` handles Fabric shareable handle exchange automatically and the backend skips client-side `pre_register_device_memory`. Only meaningful when `remote_h2d_transport_backend="HIXL"`. FabricMem requires datasystem-side support (HIXL FabricMem build and the corresponding datasystem environment variable); check the datasystem documentation before enabling this flag. |
| `enable_dev_mem_pregister` | bool | false | No | true / false | Master toggle for client-side device memory pre-registration (`pre_register_device_memory`). To actually pre-register, this flag must be `true` and the automatic conditions must hold: `enable_remote_h2d=true`, `remote_h2d_transport_backend="HIXL"`, and `enable_fabric_mem=false`. Under `P2P_TRANSFER` or FabricMem mode pre-registration is always skipped regardless of this toggle. Set this to `true` for HIXL HCCS Remote H2D deployments that require client-side device memory registration. |
| `use_layerwise` | bool | false | No | true / false | Must match `kv_connector_extra_config.use_layerwise`. When `false`, the scheduler-side Yuanrong store skips initialization because non-layerwise lookup is delegated to the TP0 worker. When `true`, the scheduler initializes a metadata-only Yuanrong client with Remote H2D disabled, so it does not create a HIXL engine. |

</details>

`worker_addr` must match the local `dscli start --worker_address` value.

##### Remote H2D Requirements and Verification

Set `DS_ENABLE_REMOTE_H2D` to `1` (or `enable_remote_h2d` to `true` in `yuanrong.json`) only when Remote Host-to-Device transfer is enabled and verified in the Yuanrong Datasystem deployment:

* Reserve enough 2 MiB HugeTLB pages before starting the worker. For 40 GiB shared memory, reserve at least 20480 2 MiB huge pages.
* Start each Datasystem worker with Remote H2D enabled. The worker start command must include `--remote_h2d_device_ids`, `--enable_huge_tlb true`, `--arena_per_tenant 1`, and `--enable_fallocate false`. Using multiple available NPU device IDs is recommended, for example `"0,1,2,3,4,5,6,7"` on an 8-NPU node.

```bash
dscli start -w \
  --worker_address "${WORKER_IP}:31501" \
  --coordinator_address "${COORDINATOR_ADDRESS}" \
  --log_dir "/var/log/yuanrong/worker" \
  --shared_memory_size_mb 40960 \
  --arena_per_tenant 1 \
  --enable_huge_tlb true \
  --enable_fallocate false \
  --rpc_thread_num 64 \
  --oc_thread_num 64 \
  --enable_worker_worker_batch_get true \
  --sc_regular_socket_num 0 \
  --sc_stream_socket_num 0 \
  --remote_h2d_device_ids "0,1,2,3,4,5,6,7"
```

For HIXL HCCS links (Atlas A3 with HCCS reachability), set `--remote_h2d_link_type "HCCS"` and the HIXL buffer-pool parameter. The IP in `--worker_address` is also used as the HIXL endpoint IP, so use a reachable address rather than `127.0.0.1` or `0.0.0.0`. HIXL RoCE direct mode is a sub-mode of HCCS selected by `HCCL_INTRA_ROCE_ENABLE=1` on both sides and additionally requires a reachable RoCE link:

```bash
dscli start --interleave 0-7 -w \
  --worker_address "${WORKER_IP}:31501" \
  --coordinator_address "${COORDINATOR_ADDRESS}" \
  --log_dir "/var/log/yuanrong/worker" \
  --shared_memory_size_mb 40960 \
  --arena_per_tenant 1 \
  --enable_huge_tlb true \
  --enable_fallocate false \
  --rpc_thread_num 64 \
  --oc_thread_num 64 \
  --enable_worker_worker_batch_get true \
  --sc_regular_socket_num 0 \
  --sc_stream_socket_num 0 \
  --remote_h2d_device_ids "0,1,2,3,4,5,6,7" \
  --remote_h2d_link_type "HCCS" \
  --remote_h2d_hccs_buffer_pool "4:8"
```

* Make sure the NPU driver, firmware, and CANN toolkit required by Yuanrong Remote H2D are installed and visible to the worker process. In containers, mount the Ascend driver path, `npu-smi`, `hccn_tool`, `/etc/hccn.conf`, `/etc/ascend_install.info`, and the required `/dev/davinci*` devices.
* When the worker uses `--remote_h2d_link_type "HCCS"`, the client process must also export `DS_RH2D_LINK_TYPE=HCCS` before starting vLLM (the backend does not export it automatically); `ROCE` is the datasystem default and needs no env var.
* Verify the NPU and RoCE environment before enabling the client flag:

```bash
# Check the current 2 MiB HugeTLB page size, total count, and free count.
grep -E "HugePages_Total|HugePages_Free|Hugepagesize" /proc/meminfo

# Optional: check 2 MiB HugeTLB pages on each NUMA node.
for node in /sys/devices/system/node/node*/hugepages/hugepages-2048kB; do
  echo "$node total=$(cat "$node/nr_hugepages") free=$(cat "$node/free_hugepages")"
done

# Check that NPU devices and the driver are visible to the worker environment.
npu-smi info

# Check that the NPU topology is visible.
npu-smi info -t topo

# Check optical module detection on the selected local NPU.
hccn_tool -i <local_npu_id> -optical -g

# Check RoCE physical link status. The expected link status is UP.
for i in {0..7}; do hccn_tool -i $i -link -g; done

# Check the selected NPU IP address and reachability to the remote NPU.
hccn_tool -i <local_npu_id> -ip -g
hccn_tool -i <local_npu_id> -ping -g address <remote_npu_ip>
```

If these checks fail, keep `DS_ENABLE_REMOTE_H2D=0` and use the default Datasystem transfer path.

#### Step 5: Run AscendStoreConnector (Yuanrong Backend)

```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model /xxxxx/Qwen2.5-7B-Instruct \
    --port 8100 \
    --trust-remote-code \
    --enforce-eager \
    --no-enable-prefix-caching \
    --tensor-parallel-size 1 \
    --data-parallel-size 1 \
    --max-model-len 10000 \
    --block-size 128 \
    --max-num-batched-tokens 4096 \
    --kv-transfer-config \
    '{
    "kv_connector": "AscendStoreConnector",
    "kv_role": "kv_both",
    "kv_load_failure_policy": "recompute",
    "kv_connector_extra_config": {
        "lookup_rpc_port": "1",
        "backend": "yuanrong",
        "use_layerwise": false
    }
}'
```

`lookup_rpc_port` is the RPC port used between the pooling scheduler process and the worker process. Each instance must use a unique port value.

**Note:** The Yuanrong backend normalizes KV keys before calling Datasystem. Supported ASCII keys up to 1024 bytes are preserved. Longer keys or keys containing unsupported characters are rewritten to a maximum of 1024 characters with a hash suffix, so do not rely on the raw key string when debugging backend storage. No extra buffer pre-registration step is required.

Expected output: vLLM starts successfully, OpenAI API service is ready.

### ASCEND_GLOBAL_RESOURCE_CONFIG

`ASCEND_GLOBAL_RESOURCE_CONFIG` is a JSON string passed to HIXL. Common fields are:

| Field | Type | Default | Required | Value Range | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `comm_resource_config.protocol_desc` | Array of strings | No default | No | e.g., `["hccs:device"]`, `["roce:device"]`, `["uboe:device"]` | Protocol descriptor for the top-level Mooncake transfer engine. In PD disaggregation, this controls the `MooncakeConnectorV1` PD transfer path. |
| `store.comm_resource_config.protocol_desc` | Array of strings | No default | No | e.g., `["roce:device"]` | Protocol descriptor for Mooncake Store traffic used by `AscendStoreConnector`. On A3, this can be set to `["roce:device"]` while PD transfer uses HCCS. |
| `comm_resource_config.listen_port` | int | 16666 | No | Port number | One-sided communication listen port. Use a different port for standalone `mooncake_client` processes to avoid conflicts with embedded clients. |
| `fabric_memory.max_capacity` | int | No default | Only when fabric mem budget is insufficient | Integer (GB per process) | Fabric memory quota. |

**Hardware Dependency Quick Reference:**

| Hardware Series | HDK Requirement | CANN Requirement | Other Dependencies |
| :--- | :--- | :--- | :--- |
| 950PR&950DT Products | >= 25.6 (with mooncake >= v0.3.11) | >= 9.1.0 | Requires UBOE/UB device and config mounts |
| 800 I/T A3 | >= 26.0 or >= 25.5 (with mooncake >= v0.3.11) | >= 9.1.0 | LingQu Computing Network >= 1.5 (required for both Mooncake FabricMem and Memcache `device_sdma`); recommended `ASCEND_ENABLE_USE_FABRIC_MEM=1` |
| 800 I/T A2 | >= 25.5 recommended | >= 9.1.0 | `HCCL_INTRA_ROCE_ENABLE=1` direct transfer |

### QoS Configuration

Both the Mooncake and Memcache backends support configuring the transfer QoS. The valid range is **0-4 (integers only)**, and the default value is 0 when not configured. A larger value means a higher transfer priority. Invalid values (non-integer, out of range) cause startup to fail fast with a validation error.

QoS can be configured through `kv_connector_extra_config`, which is injected into the backend-specific configuration automatically:

```json
{
    "kv_connector": "AscendStoreConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
        "qos_priority": 1,
        "lookup_rpc_port": "1",
        "backend": "mooncake",
        "use_layerwise": false
    }
}
```

| Parameter | Type | Default | Required | Value Range | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `qos_priority` | int | 0 | No | [0, 4] (integer, larger = higher priority) | Supported by both Mooncake and Memcache backends. Invalid values (non-integer, out of range) cause fast startup failure. |

Values in `kv_connector_extra_config` take precedence over environment variables; a WARN log is emitted on override. For the Mooncake backend, `qos_priority` is merged into existing `ASCEND_GLOBAL_RESOURCE_CONFIG` (other fields preserved). When `ASCEND_GLOBAL_RESOURCE_CONFIG` was not set, configuring `qos_priority` also creates it.

### Verification

| Verification Item | Command | Expected Output |
| :--- | :--- | :--- |
| vLLM inference service | `curl -s http://localhost:<port>/v1/completions ...` | Returns JSON response with `choices` field, `finish_reason` is `stop` or `length` |
| Yuanrong Datasystem ready | `python -c "import yr.datasystem; print('Yuanrong Datasystem is ready')"` | Prints `Yuanrong Datasystem is ready` |
| dscli availability | `dscli --version` | Displays version number |
| Coordinator startup | `dscli start -c ...` | Prints `Start coordinator service ... success` |
| etcd availability | `etcdctl put key "value"; etcdctl get key` | put returns `OK`, get returns `key`->`value` |
| Worker logs | Check logs under `--log_dir` | No errors/abnormal exits |
| vLLM startup logs | `cat log_<role>.log` | Contains `Available KV cache memory` etc., no stack traces |
| SSD Offload buffer | Startup logs | Each rank prints `AlignedClientBufferAllocator: allocated <N> bytes` |
| Mooncake master | `curl -s http://<master_host>:9003/api/v1/tenant_quotas` | Returns tenant quota JSON |

## Configuration Parameters

<details markdown="1">
<summary>kv-transfer-config Common Parameters</summary>

| `Parameter` | Type | Default | Required | Value Range | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `kv_load_failure_policy` | str | fail | No | recompute / fail | Behavior when KV loading fails: `recompute` rolls back and recomputes (does not yet support hybrid attention models like DeepSeekV4, Qwen 3.5), `fail` terminates the request. When using MultiConnector, configure on the top-level `kv-transfer-config`. |
| `lookup_rpc_port` | str | No default | Yes | Unique port number (e.g., "0", "1") | RPC port between pooling scheduler and worker processes. Each instance must use a unique port. The legacy name `mooncake_rpc_port` is deprecated; use `lookup_rpc_port`. |
| `load_async` | bool | false | No | true / false | Whether to enable asynchronous loading. |
| `backend` | str | mooncake | No | mooncake / memcache / yuanrong | KV Pool storage backend. |
| `consumer_is_to_put` | bool | false | No | true / false | Whether Decode node puts KV Cache into KV Pool. |
| `consumer_is_to_load` | bool | false | No | true / false | Whether Decode node loads KV Cache from KV Pool. |
| `use_layerwise` | bool | false | No | true / false | Layer-by-layer KV save/load, supported by both Mooncake and Memcache backends. Only supported on the Prefill node. |
| `prefill_pp_size` | int | 1 | Required when PP + `consumer_is_to_put` | Positive integer | Prefill PP size. |
| `prefill_pp_layer_partition` | str | No default | No | Comma-separated layer numbers, e.g., "30,31" | Prefill PP layer partition. If not configured, layers are evenly divided by `prefill_pp_size`. |
| `qos_priority` | int | 0 | No | [0, 4] (integer, larger = higher priority) | KV Pool transfer QoS priority. |

</details>

## Tuning Recommendations

### Mooncake SSD Offload Parameter Tuning

- Increase `MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES` when `BUFFER_OVERFLOW` occurs, but do not exceed the `Available KV cache memory` value in vLLM Worker logs. Use byte literals only (e.g., `10737418240`); `10G`/`10GB` is not supported.
- On A3 with `ASCEND_ENABLE_USE_FABRIC_MEM=1`, `MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES` must be aligned to 1 GB (multiple of 1073741824); avoid unaligned values such as `1280MB`, `512MB`, or `1.5GB`. `local_buffer_size` in `mooncake.json` is not used under fabric mem mode.
- Fabric mem budget formula (per rank):
  ```text
  fabric_memory.max_capacity >= global_segment_size + MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES (+ headroom)
  ```
  If quota is insufficient, some ranks may fail with `Memory_Allocation_Failure(EL0004)` after `global_segment_size` succeeds but `MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES` allocation fails.
- `MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES` defaults to 2 TB which far exceeds real disk capacity. Always set it to the actual per-rank budget. For example, with 800 GB disk and 8 TP ranks:
  ```shell
  export MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES=$((100 * 1024 * 1024 * 1024))
  export MOONCAKE_OFFLOAD_BUCKET_MAX_TOTAL_SIZE=$((100 * 1024 * 1024 * 1024))
  export MOONCAKE_OFFLOAD_BUCKET_EVICTION_POLICY=lru
  export MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES=1073741824   # 1 GB
  ```
- Host memory budget:
  ```text
  host_memory_for_mooncake ~ TP x (global_segment_size + MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES + local_buffer_size)
  ```
- Verify after tuning: each rank prints `AlignedClientBufferAllocator: allocated <N> bytes` at startup; no `BUFFER_OVERFLOW` / `Failed to get ... keys out of ... error_codes=[-10]` under load. If failures persist with a large buffer, check overlapping loads (`load_async`).

### MooncakeStore Warm-up

For MooncakeStore with `ASCEND_BUFFER_POOL` enabled, perform a warm-up phase before running actual performance benchmarks. Since HCCS one-sided communication connections are created lazily after instance launch, full-mesh connections incur a one-time overhead (4 MB device memory per connection). Warm-up recommendation: input sequence length 8k, output sequence length 1, total requests 2-3x number of devices.

### Memcache UBS IO Memory Pool Tuning

- `ubsio.mem.size_in_gb` upper limit formula:
  ```text
  maximum ubsio.mem.size_in_gb = min(3072, floor(available node memory for UBS IO (GB) / number of DRAM-enabled local services))
  ```
- Separated deployment recommended: 50 GB per process; other scenarios: 10 GB.
- To use L2.5 memory caching, increase `ubsio.mem.size_in_gb` within the limit and adjust [ubsio.wcache.evict_water_level](https://gitcode.com/Ascend/memcache/wiki/DRAM%20+%20SSD%20%E5%A4%9A%E7%BA%A7%E6%B1%A0%E5%8C%96%E9%85%8D%E7%BD%AE%E6%8C%87%E5%8D%97.md#ubsiowcacheevict_water_level).

### Huge Page Cleanup and Setup

On 950PR&950DT Products, residual huge pages can degrade both Mooncake and Memcache performance. Before starting the service, clean up 1GB and 2MB huge pages:

```shell
# Clean up 1GB huge pages
for n in 0 1 2 3; do
    echo 0 > /sys/devices/system/node/node${n}/hugepages/hugepages-1048576kB/nr_hugepages
done

# Clean up 2MB huge pages
for n in 0 1 2 3; do
    echo 0 > /sys/devices/system/node/node${n}/hugepages/hugepages-2048kB/nr_hugepages
done
```

For the Memcache backend, additionally ensure `ock.mmc.local_service.max.dram.size` matches the actual available memory (e.g., `2028GB`). Verify after cleanup:

```shell
# Verify 1GB huge pages are clean
for n in 0 1 2 3; do
    echo -n "node$n: "
    cat /sys/devices/system/node/node${n}/hugepages/hugepages-1048576kB/nr_hugepages
done

# Verify 2MB huge pages are clean
for n in 0 1 2 3; do
    echo -n "node$n: "
    cat /sys/devices/system/node/node${n}/hugepages/hugepages-2048kB/nr_hugepages
done
```

### Yuanrong Worker Parameter Tuning

- Thread counts such as `rpc_thread_num` and `oc_thread_num` are tuning starting points. Adjust them according to available CPU cores and measured request throughput.
- With `shared_memory_size_mb=40960`, reserve at least 20480 2 MiB huge pages:
  ```bash
  grep -E "HugePages_Total|HugePages_Free|Hugepagesize" /proc/meminfo
  ```
- Worker `-w` consumes subsequent command-line arguments. All `dscli start` options (e.g., `--timeout`) must be placed before `-w`.

## FAQ

For common environment, installation, and general parameter issues, see the [Public FAQ](../../faqs.md). This chapter only covers issues specific to this feature.

Public FAQ references:
- [Mooncake Store Deployment Guide](https://github.com/kvcache-ai/Mooncake/blob/main/docs/source/deployment/mooncake-store-deployment-guide.md)
- [SSD Offload](https://github.com/kvcache-ai/Mooncake/blob/main/docs/source/deployment/ssd/ssd-offload.md)
- [HIXL Common Issue Localization Guide](https://gitcode.com/cann/hixl/wiki/HIXL%E5%B8%B8%E8%A7%81%E9%97%AE%E9%A2%98%E5%AE%9A%E4%BD%8D%E6%89%8B%E5%86%8C.md)
- [Memcache FAQ](https://gitcode.com/Ascend/memcache/wiki/FAQ.md)

### Issue 1: failed to put/get key

**Problem Description:** vLLM reports failed `put` or `get` operations.

**Root Cause Analysis:** First determine whether the error is reported by Mooncake itself:
- `put` failure: Mooncake log shows `NO_AVAILABLE_HANDLE` or `BatchPut failed ... due to insufficient space`. This usually means the remaining space after eviction is not enough for one `BatchPut` request.
- `get` failure: Mooncake log shows `lease_expired_before_data_transfer_completed key=...` or returns `LEASE_EXPIRED`. The KV object lease expired before data transfer completed.

**Resolution Steps:**
1. Identify the error source. If Mooncake-reported `put` failure, ensure eviction policy remaining space (e.g., `1 - eviction_ratio`) can hold one batch put, or increase capacity, increase eviction headroom, or reduce batch size.
2. If `get` failure, increase `mooncake_master` `--default_kv_lease_ttl` and keep it larger than `ASCEND_CONNECT_TIMEOUT` and `ASCEND_TRANSFER_TIMEOUT`.
3. If not Mooncake-reported, it is likely an HIXL (ascend_direct) transfer-layer issue. Collect plog files under `/root/ascend/log/debug/plog` for investigation.

### Issue 2: `SEGMENT_NOT_FOUND` (SSD Offload)

**Problem Description:** Client logs show `OffloadObjectHeartbeat failed, error code is SEGMENT_NOT_FOUND`. The rank's SSD Offload stops until the segment is registered again.

**Root Cause Analysis:** Master has unmounted the rank's `LOCAL_DISK` segment (typically after `client_expired` when Ping stops refreshing TTL). Common trigger when `enable_cpu_binding=true`: Mooncake starts Ping during init, then vLLM-Ascend `bind_cpus()` runs `migratepages`/IRQ binding; the Ping thread is not pinned and misses beats under default `client_ttl=10`.

**Resolution Steps:**
1. Temporary: raise Master TTL, e.g., `mooncake_master ... --client_ttl=120`. Tune to your init/warmup window (60-120 is often enough).
2. Recovery: upgrade Mooncake to > v0.3.11 (main branch) which can remount `LOCAL_DISK` and rescan metadata.
3. Root fix: pin the storage Ping thread to a release/isolated CPU (Mooncake-side change).
4. When debugging restarts, restart Master together with vLLM to avoid stale `segment_already_exists` state.

### Issue 3: Fabric Memory Misalignment Causing Allocation Failure

**Problem Description:** `adxl MallocMem` / `aclrtMapMem` reports `Invalid_Argument`. With SSD offload enabled, `MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES` allocation failure may cause `FileStorage init` segfault and abort vLLM startup.

**Root Cause Analysis:** On A3 with `ASCEND_ENABLE_USE_FABRIC_MEM=1`, each fabric mem allocation must be an integer multiple of 1 GB. Mooncake does not round sizes up, and the default 1280 MB (1.25 GB) is not aligned.

**Resolution Steps:** For buffer alignment and fabric memory quota configuration, see "Tuning Recommendations - Mooncake SSD Offload Parameter Tuning".

### Issue 4: `MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES` Too Small (`BUFFER_OVERFLOW`)

**Problem Description:** SSD reads fail with `BUFFER_OVERFLOW` (`error_code=-10`) during `FileStorage::AllocateBatch`, and vLLM may fail when `kv_load_failure_policy=fail`.

**Root Cause Analysis:** With `enable_ssd_offload=true`, Mooncake allocates a separate per-rank SSD read/write buffer. This buffer is independent of `global_segment_size` in `mooncake.json` — increasing the segment does not fix `BUFFER_OVERFLOW`.

**Resolution Steps:** For how to increase the buffer and how to verify after tuning, see "Tuning Recommendations - Mooncake SSD Offload Parameter Tuning".

### Issue 5: Memcache Related Issues

Pre-operation checks (memory inspection, A3 available memory scanning, 950PR&950DT Products signature verification disabling and container mounts, SSD disk status checks) are described in "Step 1: Prerequisite Checks" of [Scenario 2: Memcache Backend](#scenario-2-memcache-backend). For Memcache troubleshooting, refer to the [Memcache FAQ](https://gitcode.com/Ascend/memcache/wiki/FAQ.md).

### Issue 6: DSv4 Deployment Notes

Note the following when enabling KV Pool for DSv4. Other historical known issues have been fixed (see [vllm-ascend issue #9975](https://github.com/vllm-project/vllm-ascend/issues/9975) for details):

- You must add `--no-disable-hybrid-kv-cache-manager` at startup; otherwise the service runs into an OOM problem during startup.
- DSv4 stores the states of all compression ratio families in full, and fully storing a sequence of 1M tokens takes about 300GB of space. This is expected and is the same behavior as upstream vLLM; further optimization depends on the vLLM community.

### Issue 7: AscendStore KV Pool Known Issues (v0.23.0)

For the v0.23.0 AscendStore KV Pool known issues, see [vllm-ascend issue #12390](https://github.com/vllm-project/vllm-ascend/issues/12390).
