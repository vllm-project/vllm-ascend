# KV Cache Pool（Ascend Store）Deployment Guide

## 1. Environmental Dependencies

* Software:
    * CANN >= 8.5.0
    * vLLM：main branch
    * vLLM-Ascend：main branch
    * mooncake：>= 0.3.11.post1

### KV Pool Parameter Description

#### `kv_load_failure_policy`: KV Load Failure Handling Policy

`kv_load_failure_policy` is a top-level field in `kv-transfer-config`.

* `recompute`: When KV loading fails, vLLM rolls the request back to the last valid prefix and reschedules it to recompute the failed KV blocks. Hybrid attention models (e.g. DeepSeekV4, Qwen 3.5) are not supported yet.
* `fail`: When KV loading fails, the affected request is terminated directly with an error.

The default value in vLLM is `fail`. If you want the request to fall back to recomputation after a KV load failure, set it to `recompute`.

When `MultiConnector` is used, configure `kv_load_failure_policy` on the `MultiConnector` top-level `kv-transfer-config` instead of the child connectors.

#### `kv_connector_extra_config`: Additional Configurable Parameters for Pooling

| Parameter | Description |
| :--- | :--- |
| `lookup_rpc_port` | Port for RPC Communication Between Pooling Scheduler Process and Worker Process: Each Instance Requires a Unique Port Configuration. |
| `load_async` | Whether to Enable Asynchronous Loading. The default value is false. |
| `backend` | Set the storage backend for kvpool (`mooncake`, `memcache`, `yuanrong`), with the default being `mooncake`. |
| `consumer_is_to_put` | Whether Decode node put KV Cache into KV Pool. The default value is false. |
| `consumer_is_to_load` | Whether Decode node load KV cache from KV Pool. The default value is false. |
| `use_layerwise` | Enable layer-by-layer KV save/load. Only supported on the Prefill node and requires the `memcache` backend. The default value is false. |
| `prefill_pp_size` | Prefill PP size, needs to be set when Prefill node enables PP. |
| `prefill_pp_layer_partition` | Prefill PP layer partition, needs to be set when Prefill node enables PP. |

### Environment Variable Configuration

To guarantee uniform hash generation, it is required to synchronize the PYTHONHASHSEED environment variable across all nodes upon enabling KV Pool.

```bash
export PYTHONHASHSEED=0
```

For non-layerwise `AscendStoreConnector` deployments using the Mooncake
backend, `VLLM_MOONCAKE_LOAD_RECV_THREADS` controls the number of KV-load
receiver threads (default: `1`):

```bash
export VLLM_MOONCAKE_LOAD_RECV_THREADS=4
```

Values greater than `1` enable request-level parallel loading for both
`load_async: true` and synchronous loading. Each request is handled by one
receiver thread, so a single request is not split across the pool. The setting
is ignored by layerwise transfer and non-Mooncake AscendStore backends.

## 2. Example of using Mooncake as a KV Pool backend

### Step 2.1: Software Installation

* Software:
    * Check Configuration:

        Ensure that the hccn.conf file exists in the environment. If using Docker, mount it into the container.

        ```bash
        cat /etc/hccn.conf
        ```

        For Ascend 950 Products, additionally mount:
        * devices: `/dev/ummu`, `/dev/uburma`
        * commands: `/usr/bin/urma_admin`
        * configurations: `/lib/route.conf`, `/etc/hccl_rootinfo.json`

    * Install Mooncake

        Mooncake is the serving platform for Kimi, a leading LLM service provided by Moonshot AI.
        The Mooncake wheel requires glibc 2.35 or later. Check the installed glibc version before installation:

        ```shell
        ldd --version
        ```

        Install Mooncake with pip:

        ```shell
        python3 -m pip install mooncake-transfer-engine-npu==0.3.11.post1 --extra-index-url https://mirrors.aliyun.com/pypi/web/simple
        ```

        Mooncake `0.3.11.post1` remains supported when `tenant_id` is omitted or resolves to `default`. A non-default tenant requires a Mooncake version whose `MooncakeDistributedStore.setup()` accepts `tenant_id`; use Mooncake `0.3.12` or later for multi-tenant deployments.

### Step 2.2: Run Mooncake Master

**Note:** Before proceeding, review the following Mooncake guides:

* [Mooncake Store Deployment Guide](https://github.com/kvcache-ai/Mooncake/blob/main/docs/source/deployment/mooncake-store-deployment-guide.md)
* [SSD Offload](https://github.com/kvcache-ai/Mooncake/blob/main/docs/source/deployment/ssd/ssd-offload.md)
* [Tenant Quota Management](https://github.com/kvcache-ai/Mooncake/blob/main/docs/source/deployment/mooncake-store-deployment-guide.md#tenant-quota-management)

#### Step 2.2.1: Configure mooncake.json

The environment variable **MOONCAKE_CONFIG_PATH** is configured to the full path where mooncake.json is located.

```shell
{
    "metadata_server": "P2PHANDSHAKE",
    "protocol": "ascend",
    "device_name": "",
    "master_server_address": "xx.xx.xx.xx:50088",
    "global_segment_size": "1GB" (1024MB/1048576KB/1073741824B/1073741824),
    "preferred_segment": false,
    "prefer_alloc_in_same_node": true,
    "enable_ssd_offload": false, # only required when the SSD offload feature is enabled
    "ssd_offload_path": "/nvme/mooncake_offload", # only required when the SSD offload feature is enabled)
    "tenant_id": "default"
}
```

| Parameter | Description |
| :--- | :--- |
| `metadata_server` | Configured as **P2PHANDSHAKE**. |
| `protocol` | Must be set to `ascend` on the NPU. |
| `device_name` | Leave as empty string `""`.The ascend protocol does not use device names. |
| `master_server_address` | IP and port of the master service. It can also be set via the **MOONCAKE_MASTER** environment variable, which takes precedence over this configuration item (useful for injecting the master address through Kubernetes). |
| `global_segment_size` | Registered memory size per card to the KV Pool. **Needs to be aligned to 1GB.** It can also be set via the **MOONCAKE_GLOBAL_SEGMENT_SIZE** environment variable, which takes precedence over this configuration item. |
| `preferred_segment` | Whether to prefer storing KV on the local segment when putting objects to the KV Pool. Defaults to **false**. |
| `prefer_alloc_in_same_node` | Whether to prefer allocating KV on the same node. Defaults to **true**. |
| `enable_ssd_offload` | Set to `true` to enable SSD offload. Environment variables are not supported. |
| `ssd_offload_path` | **Required when `enable_ssd_offload` is `true`.** Absolute path to a local directory where Mooncake stores offloaded KV data (for example, `/nvme/mooncake_offload`). The directory must exist and be writable by the vLLM process; create it before startup (`mkdir -p <path>`). Relative paths, symbolic links, and paths containing `..` are rejected by Mooncake. |
| `tenant_id` | Optional Mooncake tenant namespace. Missing, `null`, empty, or whitespace-only values use `default`; surrounding whitespace is removed. All Prefill, Decode, scheduler, and replica instances that share KV entries must use the same tenant ID. Non-default tenants require Mooncake `0.3.12` or later. |

#### Step 2.2.2: Start mooncake_master

As a standalone process, the master Service only needs to be launched on one node.

Under the mooncake folder:

```shell
mooncake_master --port 50088 --eviction_high_watermark_ratio 0.9 --eviction_ratio 0.1 --default_kv_lease_ttl 11000 --enable_offload=false --client_ttl=120
```

| Field | Description |
| :--- | :--- |
| `eviction_high_watermark_ratio` | Determines the watermark where Mooncake Store will perform eviction. |
| `eviction_ratio` | Determines the portion of stored objects that would be evicted. |
| `default_kv_lease_ttl` | Controls the default lease TTL for KV objects (milliseconds). Keep it larger than `ASCEND_CONNECT_TIMEOUT` and `ASCEND_TRANSFER_TIMEOUT`. |
| `enable_offload` | Set to `true` to enable SSD offload in Mooncake master. Keep the master port aligned with `master_server_address` in `mooncake.json`. Only required when SSD offload is enabled. |
| `client_ttl` | Seconds a client stays alive after the last Ping. CLI default is `10`; see [SEGMENT_NOT_FOUND with SSD offload](#5321-segment_not_found-with-ssd-offload). Only required when SSD offload is enabled. |

#### Step 2.2.3: Enable Strict Multi-Tenant Mode

Tenant IDs are ignored for object placement while strict multi-tenant mode is disabled, and objects remain in the `default` namespace. To enable isolated namespaces and per-tenant memory quota admission, start Mooncake master with strict multi-tenant mode and a policy connector:

```shell
mooncake_master \
    --port 50088 \
    --enable_multi_tenants=true \
    --tenant_quota_connector_type=file \
    --tenant_quota_connector_uri=/etc/mooncake/tenant_quotas.yaml
```

For example, `/etc/mooncake/tenant_quotas.yaml` can contain:

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

The file connector can be replaced with `etcd` when Mooncake is built with `STORE_USE_ETCD=ON`; in that case, set `tenant_quota_connector_uri` to the etcd endpoints. Strict mode rejects writes for unregistered tenants, including `default`, so every tenant used by vLLM-Ascend must appear in the policy.

Mooncake exposes tenant quota snapshots through the master metrics HTTP port (default `9003`):

```shell
curl -s http://<master_host>:9003/api/v1/tenant_quotas
curl -s "http://<master_host>:9003/api/v1/tenant_quotas?tenant_id=tenant-a"
```

`tenant_id` is an instance-level namespace and quota identity, not an authentication mechanism. A client that can access Mooncake can still declare a tenant ID. Keep incompatible models, model versions, quantization formats, and KV layouts in separate model or release namespaces even when tenant isolation is enabled.

### Step 2.3: PD Disaggregation Scenario

#### Step 2.3.1: Run `prefill` Node and `decode` Node

Using `MultiConnector` to simultaneously utilize both `MooncakeConnectorV1` and `AscendStoreConnector`. `MooncakeConnectorV1` performs kv_transfer, while `AscendStoreConnector` serves as the prefix-cache node.

**run_prefill.sh/run_decode.sh:**

```shell
#!/bin/bash

# prefill / decode
ROLE="prefill"
# A2 (800I/800T A2) or A3 (800I/800T A3) or A5 (950PR/950DT)
HARDWARE_SERIES="A2"
# Link type: ROCE or HCCS in A3 series.
LINK_TYPE="ROCE"
LOCAL_IP="xx.xx.xx.xx"
NIC_NAME="xxxxxx"

MODEL_PATH="xxxxxxx/Qwen3-32B"
SERVED_MODEL_NAME="qwen3"
DATA_PARALLEL_SIZE=1
TENSOR_PARALLEL_SIZE=8
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# parameters required for kv pool and mooncake
export PYTHONHASHSEED=0
export MOONCAKE_CONFIG_PATH="/xxxxxx/mooncake.json"
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH


if [ "$ROLE" == "prefill" ]; then
    KV_ROLE="kv_producer"
    KV_PORT="20001"
    LOOKUP_RPC_PORT="0"
else
    KV_ROLE="kv_consumer"
    KV_PORT="20002"
    LOOKUP_RPC_PORT="1"
fi

echo "Starting vLLM on Series: $HARDWARE_SERIES, Role: $ROLE"

rm -rf /root/ascend/log/*
rm -rf ./connector.log

# For detailed parameter descriptions, see 5.1 Environment Variables Description
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
    # A5 UBOE
    export ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.protocol_desc":["uboe:device"]}'
    # A5 UB
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
  --port 30050
  --max-num_seqs 20
  --max-model-len 32768
  --max-num-batched-tokens 16384
  --gpu-memory-utilization 0.9
  --kv-transfer-config "$KV_CONFIG"
)

python -m vllm.entrypoints.openai.api_server "${CMD_ARGS[@]}" > log_${ROLE}.log 2>&1

echo "vLLM started. Log file: log_${ROLE}.log"
```

Currently, the key-value pool in PD Disaggregate only stores the kv cache generated by the Prefill node by default. In models using MLA, it is now supported that the Decode node stores the kv cache for use by the Prefill node, enabled by adding `consumer_is_to_put: true` to the AscendStoreConnector. If the Prefill node enables PP, `prefill_pp_size` or `prefill_pp_layer_partition` also needs to be set. Example as follows:

```python
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

#### Step 2.3.2: Start proxy_server

```shell
python vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
    --host localhost \
    --prefiller-hosts localhost \
    --prefiller-ports 8100 \
    --decoder-hosts localhost \
    --decoder-ports 8200 \
```

Change localhost to your actual IP address.

#### Step 2.3.3: Run Inference

Configure the localhost, port, and model weight path in the command to your own settings.

Short question:

```shell
curl -s http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{ "model": "/xxxxx/Qwen2.5-7B-Instruct", "prompt": "Hello. I have a question. The president of the United States is", "max_completion_tokens": 200, "temperature":0.0 }'
```

Long question:

```shell
curl -s http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{ "model": "/xxxxx/Qwen2.5-7B-Instruct", "prompt": "Given the accelerating impacts of climate change—including rising sea levels, increasing frequency of extreme weather events, loss of biodiversity, and adverse effects on agriculture and human health—there is an urgent need for a robust, globally coordinated response. However, international efforts are complicated by a range of factors: economic disparities between high-income and low-income countries, differing levels of industrialization, varying access to clean energy technologies, and divergent political systems that influence climate policy implementation. In this context, how can global agreements like the Paris Accord be redesigned or strengthened to not only encourage but effectively enforce emission reduction targets? Furthermore, what mechanisms can be introduced to promote fair and transparent technology transfer, provide adequate financial support for climate adaptation in vulnerable regions, and hold nations accountable without exacerbating existing geopolitical tensions or disproportionately burdening those with historically lower emissions?", "max_completion_tokens": 256, "temperature":0.0 }'
```

### Step 2.4: PD-Mixed Inference

#### Step 2.4.1: Run Mixed Deployment Script

```shell
bash pd_mix.sh
```

Content of pd_mix.sh:

```shell
# A2 (800I/800T A2) or A3 (800I/800T A3) or A5 (950PR/950DT)
HARDWARE_SERIES="A2"
# Link type: ROCE or HCCS in A3 series.
LINK_TYPE="ROCE"
LOCAL_IP="xx.xx.xx.xx"
NIC_NAME="xxxxxx"

MODEL_PATH="xxxxxxx/Qwen3-32B"
SERVED_MODEL_NAME="qwen3"
DATA_PARALLEL_SIZE=1
TENSOR_PARALLEL_SIZE=8
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# parameters required for kv pool and mooncake
export PYTHONHASHSEED=0
export MOONCAKE_CONFIG_PATH="/xxxxxx/mooncake.json"
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH


echo "Starting vLLM on Series: $HARDWARE_SERIES"

rm -rf /root/ascend/log/*
rm -rf ./connector.log

# For detailed parameter descriptions, see 5.1 Environment Variables Description
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
    # A5 UBOE
    export ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.protocol_desc":["uboe:device"]}'
    # A5 UB
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
  --port 30050
  --max-num_seqs 20
  --max-model-len 32768
  --max-num-batched-tokens 16384
  --gpu-memory-utilization 0.9
  --kv-transfer-config "$KV_CONFIG"
)

python -m vllm.entrypoints.openai.api_server "${CMD_ARGS[@]}" > log_mix.log 2>&1

echo "vLLM started. Log file: log_mix.log"
```

#### Step 2.4.2: Run Inference

Configure the localhost, port, and model weight path in the command to your own settings. The requests sent will only go to the port where the mixed deployment script is located, and there is no need to start a separate proxy.

Short question:

```shell
curl -s http://localhost:8100/v1/completions -H "Content-Type: application/json" -d '{ "model": "/xxxxx/Qwen2.5-7B-Instruct", "prompt": "Hello. I have a question. The president of the United States is", "max_completion_tokens": 200, "temperature":0.0 }'
```

Long question:

```shell
curl -s http://localhost:8100/v1/completions -H "Content-Type: application/json" -d '{ "model": "/xxxxx/Qwen2.5-7B-Instruct", "prompt": "Given the accelerating impacts of climate change—including rising sea levels, increasing frequency of extreme weather events, loss of biodiversity, and adverse effects on agriculture and human health—there is an urgent need for a robust, globally coordinated response. However, international efforts are complicated by a range of factors: economic disparities between high-income and low-income countries, differing levels of industrialization, varying access to clean energy technologies, and divergent political systems that influence climate policy implementation. In this context, how can global agreements like the Paris Accord be redesigned or strengthened to not only encourage but effectively enforce emission reduction targets? Furthermore, what mechanisms can be introduced to promote fair and transparent technology transfer, provide adequate financial support for climate adaptation in vulnerable regions, and hold nations accountable without exacerbating existing geopolitical tensions or disproportionately burdening those with historically lower emissions?", "max_completion_tokens": 256, "temperature":0.0 }'
```

Note: For MooncakeStore with `ASCEND_BUFFER_POOL` enabled, it is recommended to perform a warm-up phase before running actual performance benchmarks.

This is because HCCL one-sided communication connections are created lazily after the instance is launched when Device-to-Device communication is involved. Currently, full-mesh connections between all devices are required. Establishing these connections introduces a one-time time overhead and persistent device memory consumption (4 MB of device memory per connection).

**For warm-up, it is recommended to issue requests with an input sequence length of 8K and an output sequence length of 1, with the total number of requests being 2–3× the number of devices (cards/dies).**

### Step 2.5: Enable MooncakeStore SSD Offload with Embedded Real Client Mode

For detailed configuration, refer to [2.2.1](#step-221-configure-mooncakejson) and [2.2.2](#step-222-start-mooncake_master).

#### Step 2.5.1: Running the Embedded Real Client

With Mode A (Embedded Real Client), Mooncake is embedded in vLLM. When the vLLM service starts, `AscendStoreConnector` / `MooncakeBackend` automatically calls `MooncakeDistributedStore.setup()` using the settings in `mooncake.json` (including `enable_ssd_offload` and `ssd_offload_path` when SSD offload is enabled). No separate `mooncake_client` process is required.

#### Step 2.5.2: SSD Disk Usage Control

The following environment variables control disk space usage for SSD offload (bucket backend):

| Environment Variable | Default | Description |
| :--- | :--- | :--- |
| `MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES` | `1342177280` (1280 MB) | Per-rank SSD read/write buffer size in bytes. **Not** configurable in `mooncake.json`. If you hit `BUFFER_OVERFLOW`, increase this value — see [Sizing MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES](#5323-sizing-mooncake_offload_local_buffer_size_bytes). **On A3 with `ASCEND_ENABLE_USE_FABRIC_MEM=1`, must be aligned to 1GB and counts toward per-rank fabric mem quota (see [Fabric memory size alignment](#5322-fabric-memory-size-alignment-a3--ascend_enable_use_fabric_mem1))**. |
| `MOONCAKE_OFFLOAD_BUCKET_MAX_TOTAL_SIZE` | `0` | Eviction threshold in bytes. When set to `0`, the backend uses **90% of the physical disk capacity** as the quota. Set an explicit value to control disk usage precisely. |
| `MOONCAKE_OFFLOAD_BUCKET_EVICTION_POLICY` | `none` | Eviction policy: `none` (writes fail when full), `fifo`, or `lru`. |
| `MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES` | `2199023255552` (2 TB) | **Per-rank** maximum disk usage reported to Mooncake master. Master aggregates this across clients (roughly **2 TB × rank count** in the `SSD Storage` total). **Always override** to match real disk capacity — the default often exceeds available space. |

**`MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES` risk:** If left at the 2 TB default, master shows a total SSD quota far larger than the physical disk (e.g. 16 ranks → ~32 TB displayed on a 1 TB NVMe). Offload still fails when the disk fills, while monitoring looks healthy. Set this to your actual per-rank budget before production use.

Since each TP rank uses an independent SSD subdirectory (`rank_0/`, `rank_1/`, ...) under `ssd_offload_path`, all ranks share the same physical disk. To prevent a single rank from consuming excessive space, set an explicit per-rank quota. For example, with an 800 GB disk and 8 TP ranks:

```shell
# 800 GB total disk, 8 ranks, ~100 GB per rank
export MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES=$((100 * 1024 * 1024 * 1024))
export MOONCAKE_OFFLOAD_BUCKET_MAX_TOTAL_SIZE=$((100 * 1024 * 1024 * 1024))
export MOONCAKE_OFFLOAD_BUCKET_EVICTION_POLICY=lru
export MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES=1073741824   # 1 GB
```

## 3. Example of using Memcache as a KV Pool backend

### Step 3.1: Prerequisites

Before installing and configuring Memcache, perform the necessary environment checks including memory inspection[5.2.1](#521-check-memory), A3 available memory scanning[5.2.2](#522-a3-only-scan-available-memory), Ascend 950 Products signature verification/container mounting by see [5.2.3](#523-ascend-950-products-only-disable-signature-verification--mount-key-paths-in-container--install-kernel-package). If also want to enable SSD feature, see [5.2.4](#524-checks-before-enabling-ssd).

### Step 3.2: Software Installation

**MemCache depends on MemFabric. Therefore, MemFabric must be installed. Installing the memcache after the memfabric is installed.**

```shell
pip install memfabric-hybrid
pip install memcache-hybrid
```

Enabling Memcache SSD Cache requires `memcache_hybrid >= 1.2.0`.

### Step 3.3: Configuring the memcache Config File

Run `pip show memcache_hybrid` and find the `Location` value in the output. Use that value as `{INSTALL_PATH}` below.

```shell
pip show memcache_hybrid
```

The configuration file is located at {INSTALL_PATH}/memcache_hybrid/config.

**mmc-meta.conf：**

```shell
ock.mmc.meta_service_url = tcp://xx.xx.xx.xx:5000
ock.mmc.meta_service.config_store_url = tcp://xx.xx.xx.xx:6000
ock.mmc.meta_service.metrics_url = http://xx.xx.xx.xx:8000
ock.mmc.log_level = info
# If SSD is enabled, modify the following parameters to improve SSD cache hit rate
ock.mmc.evict_threshold_high = 70
ock.mmc.evict_threshold_low = 60
ock.mmc.rewarm.dram_watermark = 95
```

**mmc-local.conf：**

```shell
ock.mmc.meta_service_url = tcp://xx.xx.xx.xx:5000
ock.mmc.local_service.config_store_url = tcp://xx.xx.xx.xx:6000
ock.mmc.log_level = info
ock.mmc.local_service.world_size = 256
ock.mmc.local_service.protocol = device_sdma
ock.mmc.local_service.dram.size = 1GB
ock.mmc.local_service.max.dram.size = 1024GB
# SSD feature related parameters below
ock.mmc.local_service.storage.enabled = false  # Set to true to enable SSD storage
ubsio.disk.path = /dev/nvmexn1:/dev/nvmexn2p1:/dev/loopX
ubsio.mem.size_in_gb = 10
ubsio.standalone.device_count = 8
ubsio.standalone.force_new_disk = true
```

**Key Focuses：**

| Parameter | Description |
| :--- | :--- |
| `ock.mmc.meta_service_url` | The P node and D node should be configured with the same MetaService endpoint. |
| `ock.mmc.local_service.config_store_url` | Its value must be the same as `ock.mmc.meta_service.config_store_url` in `mmc-meta.conf`. |
| `ock.mmc.local_service.world_size` | Maximum number of supported LocalService, including services that will be added in the future. |
| `ock.mmc.local_service.protocol` | The recommended protocols are `device_rdma` (RDMA over device, supported for A2 and A3 when device RoCE is available, recommended for A2) and `device_sdma` (SDMA over device, supported for A3 when HCCS is available, recommended for A3). For Ascend 950 Products UB scenarios, set to `device_urma`. For Ascend 950 Products UBOE scenarios, set to `device_uboe`. For details about other supported protocols, see the [MemCache LocalService configuration file](https://gitcode.com/Ascend/memcache/blob/master/config/mmc-local.conf). |
| `ock.mmc.local_service.dram.size` | DRAM size allocated per die. For example, on A3, to allocate 640GB as KV pool, this parameter should be set to 640/16=40GB. set 0GB for A3 when HCCS is available. |
| `ock.mmc.local_service.max.dram.size` | The MAX size of ock.mmc.local_service.dram.size in all local processes, necessary if ranks contribute different sizes of DRAM. |
| `ock.mmc.local_service.storage.enabled` | Set to `true` to enable SSD caching. |
| `ubsio.disk.path` | **Required when SSD caching is enabled. Specify the target SSD block devices, partitions, or loop devices directly. The configured devices must be exclusively used by UBS IO and must not have any mount points.** Separate multiple paths with colons (`:`). /dev/sdx do not recommend.|
| `ubsio.mem.size_in_gb` | Per-process UBS IO memory pool size in GB. The recommended value is `10`. The supported range is an integer from `0` to `3072`; SSD caching requires at least `5` GB per process. The total allocation must not exceed the node memory available after reserving memory for the operating system, vLLM, and the Memcache DRAM pool. |
| `ubsio.standalone.device_count` | Number of local services whose `ock.mmc.local_service.dram.size` is not `0`. |
| `ubsio.standalone.force_new_disk` | Controls whether UBS IO initializes the configured SSD devices as new disks instead of recovering their existing metadata. Set to `true` because the current version does not support fault recovery. |

### Step 3.4: Run Memcache MetaService

As a standalone process, the meta Service only needs to be launched on one node.

Starting the MetaService service.

```shell
export MMC_META_CONFIG_PATH={INSTALL_PATH}/memcache_hybrid/config/mmc-meta.conf

python -c "from memcache_hybrid import MetaService; MetaService.main()"
```

### Step 3.5: PD Disaggregation Scenario

#### Step 3.5.1: Run `prefill` Node and `decode` Node

Using `MultiConnector` to simultaneously utilize both `MooncakeConnectorV1` and `AscendStoreConnector`. `MooncakeConnectorV1` performs kv_transfer, while `AscendStoreConnector` enables KV Cache Pool

#### 800I A2/800T A2/800I A3/800T A3/950PR Ascend 950 Products/950DT Ascend 950 Products Series

**run_prefill.sh/run_decode.sh:**

```shell
#!/bin/bash

# prefill / decode
ROLE="prefill"
# A2 (800I/800T A2) or A3 (800I/800T A3) or A5 (950PR/950DT)
HARDWARE_SERIES="A2"
# Link type: ROCE or HCCS in A3 series.
LINK_TYPE="ROCE"
LOCAL_IP="xx.xx.xx.xx"
NIC_NAME="xxxxxx"

MODEL_PATH="xxxxxxx/Qwen3-32B"
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
else
    KV_ROLE="kv_consumer"
    KV_PORT="20002"
    LOOKUP_RPC_PORT="1"
fi

echo "Starting vLLM on Series: $HARDWARE_SERIES, Role: $ROLE"

rm -rf /root/ascend/log/*
rm -rf ./connector.log

# For detailed parameter descriptions, see 5.1 Environment Variables Description
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
    # A5 UBOE
    export ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.protocol_desc":["uboe:device"]}'
    # A5 UB
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
          "use_layerwise":false  # Set to true only on the Prefill node to enable layerwise
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
  --port 30050
  --max-num_seqs 20
  --max-model-len 32768
  --max-num-batched-tokens 16384
  --gpu-memory-utilization 0.9
  --kv-transfer-config "$KV_CONFIG"
)

python -m vllm.entrypoints.openai.api_server "${CMD_ARGS[@]}" > log_${ROLE}.log 2>&1

echo "vLLM started. Log file: log_${ROLE}.log"
```

#### Step 3.5.2: Start proxy_server

Refer to [Start proxy_server](#step-232-start-proxy_server) in the MooncakeStore deployment section.

#### Step 3.5.3: Run Inference

Refer to [Run Inference](#step-233-run-inference) in the MooncakeStore deployment section.

### Step 3.6: PD-Mixed Scenario

#### Step 3.6.1: Run Mixed Deployment Script

#### 800I A2/800T A2/800I A3/800T A3/950PR Ascend 950 Products/950DT Ascend 950 Products  Series

**Run_pd_mix.sh:**

```shell
#!/bin/bash

# A2 (800I/800T A2) or A3 (800I/800T A3) or A5 (950PR/950DT)
HARDWARE_SERIES="A2"
# Link type: ROCE or HCCS in A3 series.
LINK_TYPE="ROCE"
LOCAL_IP="xx.xx.xx.xx"
NIC_NAME="xxxxxx"

MODEL_PATH="xxxxxxx/Qwen3-32B"
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

# For detailed parameter descriptions, see 5.1 Environment Variables Description
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
    # A5 UBOE
    export ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.protocol_desc":["uboe:device"]}'
    # A5 UB
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
     "use_layerwise":false  # Set to true to enable layerwise
  }
}'

CMD_ARGS=(
  --model "$MODEL_PATH"
  --served-model-name "$SERVED_MODEL_NAME"
  --trust-remote-code
  --enforce-eager
  --data-parallel-size "$DATA_PARALLEL_SIZE"
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
  --port 30050
  --max-num_seqs 20
  --max-model-len 32768
  --max-num-batched-tokens 16384
  --gpu-memory-utilization 0.9
  --kv-transfer-config "$KV_CONFIG"
)

python -m vllm.entrypoints.openai.api_server "${CMD_ARGS[@]}" > log_mix.log 2>&1

echo "vLLM started. Log file: log_mix.log"

```

#### Step 3.6.2: Run Inference

Refer to [Run Inference](#step-242-run-inference) in the MooncakeStore deployment section.

### Step 3.7: Separated Deployment of MemCache and vLLM

This deployment mode runs MemCache and vLLM in different processes. It is different from vLLM PD disaggregation. In the default co-located mode, vLLM loads the model weights before the KV connector initializes MemCache. As a result, MemCache may not be able to reserve sufficient memory from the remaining available space. Starting a standalone MemCache process before vLLM allows MemCache to reserve a larger memory pool. **This mode currently only support the A3 HCCS scenario**.

Prepare two LocalService configuration files with the same connection and protocol settings. The configuration used by the vLLM process does not contribute DRAM:

```ini
# mmc-local.conf used by the vLLM process
ock.mmc.local_service.dram.size = 0GB
ock.mmc.local_service.max.dram.size = 1024GB
```

The configuration used by the standalone MemCache process specifies the amount of DRAM to contribute:

```ini
# mmc-local-standalone.conf used by the standalone MemCache process
ock.mmc.local_service.dram.size = 600GB
ock.mmc.local_service.max.dram.size = 1024GB
```

The preceding sizes are examples. Adjust them according to the available memory, and set `ock.mmc.local_service.max.dram.size` to accommodate the maximum `dram.size` used by the LocalService processes.

Deploy the services in the following order:

1. Start MetaService as described above.
2. Before starting vLLM, start a standalone MemCache process with `mmc-local-standalone.conf` on every node. These processes contribute the configured DRAM to the memory pool.
3. Wait until the standalone MemCache process reports successful initialization on every node.
4. Set `MMC_LOCAL_CONFIG_PATH` to `mmc-local.conf`, and then start the vLLM inference processes as described above. MemCache in the vLLM processes connects to the existing memory pool without contributing additional DRAM.

For the standalone MemCache startup script and the complete A3 deployment procedure, see [Memcache + vLLM + A3](https://gitcode.com/Ascend/memcache/wiki/MemCache+vLLM+A3%E5%88%86%E7%A6%BB%E9%83%A8%E7%BD%B2%E6%A1%88%E4%BE%8B.md).

### Step 3.8: Enable Memcache SSD Cache

#### Step 3.8.1: Configuration

For detailed configuration, refer to [Configuring the memcache Config File](#step-33-configuring-the-memcache-config-file), and have to see [5.2.4](#524-checks-before-enabling-ssd).

#### Step 3.8.2: enable SSD when separated deployment of MemCache and vLLM

Refer to the following configuration：

**mmc-local.conf：**

```shell
ock.mmc.meta_service_url = tcp://xx.xx.xx.xx:5000
ock.mmc.local_service.config_store_url = tcp://xx.xx.xx.xx:6000
ock.mmc.log_level = info
ock.mmc.local_service.world_size = 256
ock.mmc.local_service.protocol = device_sdma
ock.mmc.local_service.dram.size = 0GB
ock.mmc.local_service.max.dram.size = 1024GB
```

**mmc-local-standalone.conf：**

```shell
ock.mmc.meta_service_url = tcp://xx.xx.xx.xx:5000
ock.mmc.local_service.config_store_url = tcp://xx.xx.xx.xx:6000
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

#### Step 3.8.3: UBS IO Memory Pool Sizing

When adjusting the recommended value, calculate the maximum permitted per-process value by dividing the node memory available to UBS IO by the number of DRAM-enabled local services, rounding down, and capping the result at `3072`:

```text
maximum ubsio.mem.size_in_gb = min(3072, floor(available node memory for UBS IO (GB) / number of DRAM-enabled local services))
```

For example, if `200` GB is available to UBS IO and four local services have DRAM enabled, the upper limit is `50` GB per process, so the recommended value `ubsio.mem.size_in_gb = 10` is valid. If the calculated upper limit is less than `5`, free more node memory or reduce the number of DRAM-enabled local services.

For the scenario of separate deployment of MemCache, it is recommended to configure a single process with `50` GB. In other scenarios, it is recommended to configure `10` GB. If you want to use the L2.5 memory caching capability, increase `ubsio.mem.size_in_gb` within the limits above and adjust [`ubsio.wcache.evict_water_level`](https://gitcode.com/Ascend/memcache/wiki/DRAM%20+%20SSD%20%E5%A4%9A%E7%BA%A7%E6%B1%A0%E5%8C%96%E9%85%8D%E7%BD%AE%E6%8C%87%E5%8D%97.md#ubsiowcacheevict_water_level) accordingly.

For disk config, eviction watermarks, and other UBS IO parameters, see the [DRAM + SSD Multi-level Pooling Configuration Guide](https://gitcode.com/Ascend/memcache/wiki/DRAM%20+%20SSD%20%E5%A4%9A%E7%BA%A7%E6%B1%A0%E5%8C%96%E9%85%8D%E7%BD%AE%E6%8C%87%E5%8D%97.md).

## 4. Example of using Yuanrong as a KV Pool backend

* Software:
    * Install `openyuanrong-datasystem` on all nodes (`yr.datasystem` must be importable).

### Step 4.1: Install Yuanrong Datasystem

```bash
pip install openyuanrong-datasystem
```

If the prebuilt package does not match the CANN or Ascend driver version in
your environment, build Yuanrong Datasystem from source in the vLLM Ascend
image. Follow the official Yuanrong Datasystem build instructions:
<https://atomgit.com/openeuler/yuanrong-datasystem>

### Step 4.2: Start etcd

Yuanrong Datasystem uses etcd for service discovery. The following example
starts a single-node etcd cluster:

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

For production environments, refer to the official etcd clustering
documentation: <https://etcd.io/docs/v3.7/op-guide/clustering/>

### Step 4.3: Start Datasystem Worker

Start a Datasystem worker on each node by using `dscli`. The following
configuration is a recommended starting point for high-throughput KV Pool
workloads:

```bash
WORKER_LOG_DIR="/var/log/yuanrong/worker"
sudo mkdir -p "${WORKER_LOG_DIR}"
sudo chown "$(id -u):$(id -g)" "${WORKER_LOG_DIR}"

dscli start -w \
  --worker_address "${WORKER_IP}:31501" \
  --etcd_address "${ETCD_IP}:2379" \
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

The `--worker_address` value is consumed later by `DS_WORKER_ADDR`, so keep
the host and port identical on the same node.

The tuning parameters above have the following effects:

| Parameter | Description |
| :--- | :--- |
| `log_dir` | Sets the Datasystem worker log directory. Create the directory and grant the worker process write permission before startup. |
| `arena_per_tenant=1` | Uses one shared-memory arena per tenant as a conservative starting point for memory and file-descriptor usage. |
| `enable_huge_tlb=true` | Backs worker shared memory with HugeTLB pages. Reserve enough 2 MiB huge pages before starting the worker. |
| `enable_fallocate=false` | Disables `fallocate` for the shared-memory file; use this setting with the HugeTLB configuration above. |
| `rpc_thread_num=64` | Sets the RPC/ZMQ service concurrency. |
| `oc_thread_num=64` | Sets the Object Cache business-thread pool size. |
| `enable_worker_worker_batch_get=true` | Enables batched Object Cache reads between Datasystem workers. |
| `sc_regular_socket_num=0`, `sc_stream_socket_num=0` | Disables the Stream Cache service. Both values must be greater than zero to enable it; keep them at zero when KV Pool does not use Stream Cache. |

For `shared_memory_size_mb=40960`, reserve at least 20480 2 MiB huge pages and
verify that they are available before starting the worker:

```bash
grep -E "HugePages_Total|HugePages_Free|Hugepagesize" /proc/meminfo
```

Worker logs, including files whose base name is normally
`datasystem_worker`, are written under the `--log_dir` directory. Use an
absolute path so the log location does not depend on the worker process's
current directory.

These thread counts are tuning starting points rather than universal defaults.
Adjust them according to the available CPU cores and measured request
throughput. Because `-w` consumes the remaining command-line arguments, place
any `dscli start` options such as `--timeout` before `-w`.

For more parameters, refer to the `dscli` usage documentation on the Yuanrong
Datasystem official site:
<https://atomgit.com/openeuler/yuanrong-datasystem>

To stop the worker:

```bash
dscli stop --worker_address "${WORKER_IP}:31501"
```

### Step 4.4: Environment Variable Configuration

Set the following environment variables on each node before starting vLLM:

| Variable | Required | Default | Description |
| :--- | :--- | :--- | :--- |
| `PYTHONHASHSEED` | Yes | `0` | Must be consistent across all nodes to guarantee uniform hash generation. |
| `DS_WORKER_ADDR` | Yes | N/A | Datasystem worker address in `<host>:<port>` format. This must match the local `dscli start --worker_address` value. |
| `DATASYSTEM_CLIENT_LOG_DIR` | No | `~/.datasystem/logs` | Directory for Yuanrong client SDK logs created by the vLLM process. Use a directory separate from the worker logs. |
| `DS_ENABLE_EXCLUSIVE_CONNECTION` | No | `0` | Passed to Yuanrong `HeteroClient.enable_exclusive_connection`. Use `1` to enable the exclusive connection mode when required by your deployment. |
| `DS_ENABLE_REMOTE_H2D` | No | `0` | Passed to Yuanrong `HeteroClient.enable_remote_h2d`. Use `1` only after the Remote H2D requirements below are met. |

```bash
export PYTHONHASHSEED=0
export DS_WORKER_ADDR="${WORKER_IP}:31501"
export DATASYSTEM_CLIENT_LOG_DIR="/var/log/yuanrong/client"
export DS_ENABLE_EXCLUSIVE_CONNECTION=0
export DS_ENABLE_REMOTE_H2D=0

mkdir -p "${DATASYSTEM_CLIENT_LOG_DIR}"
```

Set `DATASYSTEM_CLIENT_LOG_DIR` before starting vLLM because the Yuanrong
client reads it during logging initialization. Client SDK logs, whose base
name is normally `ds_client`, are written to this directory.

#### Step 4.4.1: Remote H2D Requirements

Set `DS_ENABLE_REMOTE_H2D=1` only when Remote Host-to-Device transfer is
enabled and verified in the Yuanrong Datasystem deployment:

* Reserve enough 2 MiB HugeTLB pages before starting the worker. For 40 GiB
  shared memory, reserve at least 20480 2 MiB huge pages.
* Start each Datasystem worker with Remote H2D enabled. The worker start
  command must include `--remote_h2d_device_ids`, `--enable_huge_tlb true`,
  `--arena_per_tenant 1`, and `--enable_fallocate false`. Using multiple
  available NPU device IDs is recommended, for example `"0,1,2,3,4,5,6,7"` on
  an 8-NPU node.

```bash
dscli start -w \
  --worker_address "${WORKER_IP}:31501" \
  --etcd_address "${ETCD_IP}:2379" \
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

* Make sure the NPU driver, firmware, and CANN toolkit required by Yuanrong
  Remote H2D are installed and visible to the worker process. In containers,
  mount the Ascend driver path, `npu-smi`, `hccn_tool`, `/etc/hccn.conf`,
  `/etc/ascend_install.info`, and the required `/dev/davinci*` devices.
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

If these checks fail, keep `DS_ENABLE_REMOTE_H2D=0` and use the default
Datasystem transfer path.

### Step 4.5: Run AscendStoreConnector with Yuanrong backend

Use `AscendStoreConnector` with `backend: "yuanrong"`:

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
        "backend": "yuanrong"
    }
}'
```

`lookup_rpc_port` is the RPC port used between the pooling scheduler process
and the worker process. Each instance must use a unique port value.

### Notes

* The Yuanrong backend normalizes KV keys before calling Datasystem. Supported
  ASCII keys up to 1024 bytes are preserved. Longer keys or keys containing
  unsupported characters are rewritten to a maximum of 1024 characters with a
  hash suffix, so do not rely on the raw key string when debugging backend
  storage.
* No extra buffer pre-registration step is required for Yuanrong. The backend
  uses device pointers directly when building blob lists.

## 5. Appendix and FAQ

### 5.1. Environment Variables Description

This section describes hardware-specific environment variables required by both Mooncake and Memcache backends.

| Hardware | Dependencies | Export Command | Description |
| :--- | :--- | :--- | :--- |
| 950PR/DT Ascend 950 Products series | HDK >=25.6 with mooncake >= v0.3.11 <br>CANN >= 9.1.0 | # UBOE<br> `export ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.protocol_desc":["uboe:device"]}'` <br> # UB<br>`export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'` | Configure the required environment variables based on the communication protocol to use. |
| 800 I/T A3 series | HDK >= 26.0<br>or HDK >= 25.5 with mooncake >= v0.3.11<br>CANN >= 9.0.0<br>LingQu Computing Network >= 1.5 | `export ASCEND_ENABLE_USE_FABRIC_MEM=1` | **Recommended**. Enables unified memory address direct transmission scheme. With SSD offload, see [Fabric memory size alignment](#5322-fabric-memory-size-alignment-a3--ascend_enable_use_fabric_mem1) — memory sizes must be aligned to 1GB. |
| 800 I/T A2 series | HDK >= 25.5 is recommended | `export HCCL_INTRA_ROCE_ENABLE=1` | Required by direct transmission scheme on 800 I/T A2 series|

### 5.2. Memcache Prerequisites

#### 5.2.1. Check memory

Use `free -h` to check the memory. If excessive cache affects the KV cache pool size, clean the cache and defragment memory:

```shell
# Release pagecache/dentry/inode to free contiguous physical memory
echo 3 > /proc/sys/vm/drop_caches
# Trigger memory compaction to reduce fragmentation
echo 1 > /proc/sys/vm/compact_memory
```

#### 5.2.2. (A3 only) Scan available memory

Scan the available memory on the environment by running a script. Script address: [mem_scan.py](https://gitcode.com/Ascend/memfabric_hybrid/blob/develop/script/mem_scan.py). Execute command:

```shell
python3 mem_scan.py
```

MemFabric uses 2MB and 1GB huge pages. The script scans by 1GB specification by default. To scan available memory for 2MB huge pages, run:

```shell
python3 mem_scan.py -m 2
```

#### 5.2.3. (Ascend 950 Products only) Disable signature verification + mount key paths in container + install kernel package

**Step 1:** Disable HDK signature verification on the bare metal (only needs to be executed once per machine):

```shell
for i in {0..7}; do npu-smi set -t custom-op-secverify-enable -i $i -d 1; done;
for i in {0..7}; do npu-smi set -t custom-op-secverify-mode -i $i -d 0; done;
```

**Step 2:** Refer to the following `docker run` command to start the container. Ensure `/usr/bin/urma_admin`, `/lib/route.conf`, `/etc/hccl_rootinfo.json` are mounted into the container:

```shell
docker run -u root -it -d --name ${NAME} --net=host --privileged=true \
    --device=/dev/davinci_manager --device=/dev/hisi_hdc --device=/dev/ummu --device=/dev/uburma \
    --device=/dev/davinci0 \
    --device=/dev/davinci1 \
    --device=/dev/davinci2 \
    --device=/dev/davinci3 \
    --device=/dev/davinci4 \
    --device=/dev/davinci5 \
    --device=/dev/davinci6 \
    --device=/dev/davinci7 \
    -v /usr/bin/urma_admin:/usr/bin/urma_admin \
    -v /lib/route.conf:/lib/route.conf \
    -v /root/host:/root/host  \
    -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
    -v /usr/local/sbin:/usr/local/sbin \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /var/log/npu/:/usr/slog \
    -v /mnt:/mnt \
    -v /etc/hccn.conf:/etc/hccn.conf \
    -v /usr/lib64:/usr/lib64   \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
    -v /etc/hccl_rootinfo.json:/etc/hccl_rootinfo.json \
    -v /home/:/home/ \
    -v /etc/hixlep:/etc/hixlep \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -w /home \
    ${IMAGES_ID} \
    bash
```

**Step 3:** Update `/lib/route.conf` inside the container.

#### 5.2.4. Checks before enabling SSD

Check the disk status using `lsblk`. Taking `nvme1n1` as an example, ensure the disk has no partitions, no mount points, and no filesystem signatures:

```shell
lsblk /dev/nvme1n1                    # No partitions expected
mount | grep nvme1n1                  # No mount points expected
blkid /dev/nvme1n1                    # No filesystem signature expected
```

If no physical disk is available, you can simulate a disk using a loop device. Refer to the following commands:

```shell
# Create a 640GB image file (adjust count as needed)
dd if=/dev/zero of=/data/boostio_disk.img bs=1G count=640 status=progress

# Mount the loop device with direct_io enabled; the command outputs the actual device path
LOOP_DEV=$(losetup --find --show --direct-io=on /data/boostio_disk.img)
echo "${LOOP_DEV}"
# Example output: /dev/loop3, where /dev/loop3 is the simulated disk
```

### 5.3. Mooncake FAQ

#### 5.3.1. failed to put/get key

When vLLM reports failed `put` or `get` operations, first check whether the error is reported by Mooncake itself.

* If the error is reported by Mooncake:
    * For `put` failures, check whether the Mooncake log contains `NO_AVAILABLE_HANDLE` or `BatchPut failed ... due to insufficient space`. This usually means the remaining space after eviction is not enough for one `BatchPut` request. Ensure the space left by the eviction policy (for example, the capacity implied by `1 - eviction_ratio`) can hold one batch put, or consider increasing the available capacity, increasing eviction headroom, or reducing the batch size.
    * For `get` failures, check whether the Mooncake log contains `lease_expired_before_data_transfer_completed key=...` or returns `LEASE_EXPIRED`. This means the KV object lease expired before the data transfer completed. Increase `--default_kv_lease_ttl` for `mooncake_master` as needed, and keep it larger than `ASCEND_CONNECT_TIMEOUT` and `ASCEND_TRANSFER_TIMEOUT`.
* If the error is not reported by Mooncake, it is likely an HIXL (ascend_direct) transfer-layer issue. Collect plog files under `/root/ascend/log/debug/plog` and check whether the issue matches a known HIXL problem.

For common troubleshooting and issue localization guidance for HIXL (ascend_direct), see:
<https://gitcode.com/cann/hixl/wiki/HIXL%E5%B8%B8%E8%A7%81%E9%97%AE%E9%A2%98%E5%AE%9A%E4%BD%8D%E6%89%8B%E5%86%8C.md>

#### 5.3.2. SSD FAQ

##### 5.3.2.1. SEGMENT_NOT_FOUND with SSD offload

If client logs show `OffloadObjectHeartbeat failed, error code is SEGMENT_NOT_FOUND`, Master has unmounted the rank's `LOCAL_DISK` segment (usually after `client_expired` when Ping stops refreshing TTL). SSD offload on that rank stops until the segment is registered again.

**Typical trigger (with `enable_cpu_binding=true`):** Mooncake starts Ping during init, then vLLM-Ascend `bind_cpus()` runs `migratepages`/IRQ binding; the Ping thread is not pinned and can miss beats under the default `client_ttl=10`.

| Mitigation | Notes |
| :--- | :--- |
| **Temporary:** raise Master TTL | e.g. `mooncake_master ... --client_ttl=120`. Tune to your init/warmup window (often `60`–`120` is enough). Does not fix the root cause. |
| **Recovery:** upgrade Mooncake | Versions **> v0.3.11** (main branch) can remount `LOCAL_DISK` and rescan metadata after `SEGMENT_NOT_FOUND`. This **recovers after** cleanup; it does **not** prevent expiry or in-flight request failures while metadata is gone. |
| **Root fix:** Mooncake Ping CPU affinity | Pin the storage Ping thread to a release/isolated CPU (Mooncake-side change). Optional vLLM-Ascend cooperation to pass the release CPU per rank. |

Also restart Master together with vLLM to avoid stale `segment_already_exists` state when debugging restarts.

##### 5.3.2.2. Fabric memory size alignment (A3 + `ASCEND_ENABLE_USE_FABRIC_MEM=1`) {: #5322-fabric-memory-size-alignment-a3--ascend_enable_use_fabric_mem1}

On A3 with fabric memory enabled, **each** fabric mem allocation must be an integer multiple of **1 GB** (1073741824 bytes). Mooncake does not round sizes up automatically.

| Parameter | Config source | Alignment |
| :--- | :--- | :--- |
| `global_segment_size` | `mooncake.json` or export `MOONCAKE_GLOBAL_SEGMENT_SIZE` | Each rank's segment size must be aligned to 1GB (e.g. `"1GB"`, `"20GB"`). |
| `MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES` | export `MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES` (only when `enable_ssd_offload=true`) | Must be aligned to 1GB. Default is 1280 MB (1.25 GB), which is **not** aligned and is too small for long-context SSD loads — size with [Sizing MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES](#5323-sizing-mooncake_offload_local_buffer_size_bytes). |

`local_buffer_size` in `mooncake.json` is **not** used under fabric mem (vLLM-Ascend passes `0` to `setup()`).

**Risk if misaligned:** `adxl MallocMem` / `aclrtMapMem` fails with `Invalid_Argument`. With SSD offload enabled, a failed `MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES` allocation can segfault during `FileStorage` init and abort vLLM startup. Avoid values such as `"1280MB"`, `"512MB"`, or `"1.5GB"`.

**Fabric mem quota:** Both `global_segment_size` and `MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES` are separate fabric mem allocations **per rank**. Their sizes add up against the HIXL fabric mem limit configured via `ASCEND_GLOBAL_RESOURCE_CONFIG` (e.g. `"fabric_memory.max_capacity":32`, unit GB per process — see HIXL docs). Rough budget per rank:

```text
fabric_memory.max_capacity  ≥  global_segment_size + MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES  (+ headroom)
```

**Risk if quota is too low:** Some ranks fail with `Memory_Allocation_Failure(EL0004)` after `global_segment_size` succeeds but `MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES` allocation fails. Increase `fabric_memory.max_capacity`, reduce `global_segment_size` or `MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES`, or ensure the node has enough host memory.

Example (add to your vLLM startup script when SSD offload is on):

```bash
export ASCEND_ENABLE_USE_FABRIC_MEM=1
export MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES=1073741824   # 1 GB, fabric-mem aligned
```

**set ASCEND_GLOBAL_RESOURCE_CONFIG only if fabric mem is too low.**

```bash
# Per-rank fabric mem budget: 20 GB segment + 1 GB MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES → set max_capacity ≥ 22 (GB)
export ASCEND_GLOBAL_RESOURCE_CONFIG='{"fabric_memory.max_capacity":32}'
```

##### 5.3.2.3. Sizing MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES

When `enable_ssd_offload=true`, Mooncake allocates a **separate per-rank SSD read/write buffer** sized by `MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES`. This buffer is **independent** of `global_segment_size` in `mooncake.json` — increasing the segment does **not** fix `BUFFER_OVERFLOW` caused by an undersized `MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES`.

If the buffer is too small, SSD reads fail with `BUFFER_OVERFLOW` (`error_code=-10`) during `FileStorage::AllocateBatch`, and vLLM may fail when `kv_load_failure_policy=fail`.

If you encounter `BUFFER_OVERFLOW` during use, try increasing `MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES`. Do not set it higher than the **Available KV cache memory** value shown in vLLM worker logs:

```text
(Worker_TP0_EP0 pid=21240) INFO 06-23 17:41:09 [worker.py:552] Available KV cache memory: XX
```

Example:

```bash
export MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES=10737418240   # 10 GB
```

Use **byte literals only** (`10737418240`). `10G` / `10GB` are ignored and fall back to the 1280 MB default.

<details>
<summary>Notes</summary>

* `--max-num-batched-tokens` only chunks prefill compute; it does **not** reduce the memory required by `MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES`.

</details>

###### Host memory budget (single node)

`MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES` is allocated **per rank**, in addition to `global_segment_size`:

```text
host_memory_for_mooncake ≈ TP × (global_segment_size + MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES + local_buffer_size)
```

Ensure `free -h` **available** on the host exceeds this sum plus vLLM overhead. `MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES` does **not** need to fit inside `global_segment_size`.

###### Verify after tuning

1. Startup: each rank logs `AlignedClientBufferAllocator: allocated <N> bytes` with your configured size.
2. Under load: no `BUFFER_OVERFLOW` / `Failed to get ... keys out of ... error_codes=[-10]`.
3. If failures persist with a large buffer, check overlapping loads (`load_async`).

### 5.4. Memcache FAQ

1. Pre-operation steps:
2. For Memcache troubleshooting, see:
<https://gitcode.com/Ascend/memcache/wiki/FAQ.md>

### 5.5. DSv4 known issue (temporary)

For the temporary DSv4 known issue, see:
<https://github.com/vllm-project/vllm-ascend/issues/9975>
