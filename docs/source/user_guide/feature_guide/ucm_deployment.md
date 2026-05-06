# UCM Store Deployment Guide

## Why Use UCM

Unified Cache Manager (UCM) provides an external KV-cache storage layer designed for prefix-caching scenarios in vLLM/vLLM-Ascend. Unlike KV Pooling, which expands prefix-cache capacity only by aggregating device memory and therefore remains limited by HBM/DRAM size and lacks persistence, UCM decouples compute from storage and adopts a tiered design. Each node uses local DRAM as a fast cache, while a shared backend—such as NFS, 3FS, or enterprise-grade storage—serves as the persistent KV store.

**Key benefits of using UCM:**

1. **Breaks Device Memory Capacity Limits**: Traditional prefix caching is constrained by HBM/DRAM size. UCM removes this ceiling by offloading KV cache to external storage, enabling cache capacity to scale with the storage system rather than with compute resources.

2. **Persistent and Reliable KV Cache**: UCM provides durable KV cache storage, ensuring that cached prefix blocks survive across service restarts, instance failures, or scheduling migrations. This is critical for production-grade inference systems.

3. **Multi-Scenario Acceleration**: UCM not only supports prefix caching but also offers training-free sparse attention methods (e.g., GSA, CacheBlend) for handling extremely long sequence inference tasks. Additionally, UCM provides PD disaggregation solutions based on storage-compute separation architecture, enabling flexible management of heterogeneous computing resources.

4. **Significant Performance Improvement**: When integrated with vLLM, UCM achieves **3-10x reduction** in inference latency across various scenarios, including multi-turn dialogue and long-context reasoning tasks. Benchmarks show up to **8x improvement in TTFT** for prefix caching scenarios.

## How UCM Works

### Architecture

UCM adopts a **centralized architecture** for KV cache management, constructing a three-tier cache hierarchy:

```bash
HBM (GPU Memory) → DRAM (Local Cache) → Storage Backend (SSD/NFS/3FS)
```

This three-tier design enables:

- **HBM (Tier 1)**: Fastest access for active inference computation
- **DRAM (Tier 2)**: High-speed local cache for frequently accessed KV blocks
- **Storage Backend (Tier 3)**: Persistent storage layer including local SSD, NFS-mounted storage, or dedicated systems like 3FS for unlimited capacity scaling

UCM chose the centralized approach (similar to DeepSeek's 3FS) over decentralized designs for several reasons:

1. **Simplicity**: Avoids complex affinity scheduling required in decentralized architectures
2. **Decoupling**: Keeps inference instances independent without reporting KV cache status to schedulers
3. **No Data Silos**: Centralized storage prevents redundant KV cache accumulation across isolated instances
4. **Better Compatibility**: Superior compatibility with PD disaggregation and large-scale deployment

### Capabilities

UCM currently provides the following capabilities:

| Capability | Description |
|------------|-------------|
| **Prefix Cache** | Persistent KV cache storage with support for NFS Store, 3FS Store, and Pipeline Store |
| **Sparse Attention** | Training-free sparse attention methods including GSA (Graph-based Sparse Attention) and CacheBlend for long-context acceleration |
| **PD Disaggregation** | Prefill-Decode disaggregation with multiple modes: P2P, Centralized PD, NPGD, and xPYD |
| **ReRoPE** | Support for Rotary Position Embedding extensions |

**Supported Platforms:**

- CUDA (NVIDIA H100, H20, L40, L20)
- CANN (Ascend 910C, 910B)
- MUSA (Mthreads S5000)
- MACA (MetaX C500)

**Supported Frameworks:**

- vLLM (main branch)
- vLLM-Ascend (main branch)
- SGLang (main branch)

> **Note**: For the complete and latest support matrix, refer to [UCM Support Matrix](https://ucm.readthedocs.io/en/latest/user-guide/support-matrix/support_matrix.html).

## Deployment Guide

### Prerequisites

- OS: Linux
- Hardware with Ascend NPUs. It is typically the Atlas 800 A2 series.
- vLLM: main branch
- vLLM Ascend: main branch

### UCM Installation

**Please refer to the [official UCM installation guide for Ascend NPU](https://ucm.readthedocs.io/en/latest/getting-started/quickstart_vllm_ascend.html)**

### PD Disaggregation Scenario

UCM supports two types of PD disaggregation architectures:

| Type | KV Transfer Method | Characteristics |
|------|-------------------|-----------------|
| **Centralized PD** | Via unified storage backend (NFS/3FS) | Simple architecture, complete decoupling, stateless instances |
| **Distributed PD (P2P)** | Direct transfer via Mooncake + UCM prefix cache | Lower latency, suitable for homogeneous P/D nodes |

#### Centralized PD Disaggregation

In centralized PD disaggregation, KV cache is transmitted via a unified storage pool. The Prefill node offloads KV cache to the storage backend, and the Decode node retrieves it with high prefix cache hit rates. This approach achieves the highest degree of decoupling and simplifies scheduling logic.

> **Important**: For cross-node deployment, all Prefill and Decode nodes must have access to a **shared storage backend** (e.g., NFS-mounted directory or 3FS). Ensure the storage path is accessible from all nodes before proceeding.

**Example: 2P2D Setup**

Assume 2 Prefill instances on node 192.168.10.1 (ports 7800, 7801) and 2 Decode instances on node 192.168.10.2 (ports 7802, 7803), with a shared NFS storage at `/mnt/test1`.

**Step 1: Prepare UCM Configuration File**

Create a configuration file (e.g., `ucm_config_example.yaml`) with PipelineStore:

```yaml
ucm_connectors:
  - ucm_connector_name: "UcmPipelineStore"
    ucm_connector_config:
      store_pipeline: "Cache|Posix"
      storage_backends: "/mnt/test1"
      cache_buffer_capacity_gb: 64
enable_event_sync: true
use_layerwise: false
```

Key configuration parameters:

- **storage_backends**: The shared storage directory accessible from all nodes (e.g., NFS-mounted path or 3FS). For cross-node PD disaggregation, this must be a shared storage path.

> **Note**: PipelineStore is the recommended connector for UCM. It chains Cache Store (Device ↔ Host) and Posix Store (Host ↔ Storage backend) for optimal transfer performance. For more configuration options, refer to [UCM PipelineStore Documentation](https://ucm.readthedocs.io/en/latest/user-guide/prefix-cache/pipeline_store.html).

**Step 2: Run Prefill Servers**

```bash
export PYTHONHASHSEED=123456
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
vllm serve /models/QwQ-32B \
    --host 0.0.0.0 \
    --port 7800 \
    --gpu-memory-utilization 0.92 \
    --data-parallel-size 1 \
    --tensor-parallel-size 4 \
    --seed 1024 \
    --max-model-len 17000 \
    --max-num-batched-tokens 8000 \
    --max-num-seqs 20 \
    --trust-remote-code \
    --enforce-eager \
    --kv-transfer-config \
    '{
        "kv_connector": "UCMConnector",
        "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {"UCM_CONFIG_FILE": "/path/to/ucm_config_example.yaml"}
    }'
```

To start the second Prefill instance on the same node, modify `--port` (e.g., port 7801) and `ASCEND_RT_VISIBLE_DEVICES` accordingly.

**Step 3: Run Decode Servers**

```bash
export PYTHONHASHSEED=123456
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
vllm serve /models/QwQ-32B \
    --host 0.0.0.0 \
    --port 7802 \
    --gpu-memory-utilization 0.92 \
    --data-parallel-size 1 \
    --tensor-parallel-size 4 \
    --seed 1024 \
    --max-model-len 17000 \
    --max-num-batched-tokens 8000 \
    --max-num-seqs 20 \
    --trust-remote-code \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --kv-transfer-config \
    '{
        "kv_connector": "UCMConnector",
        "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {"UCM_CONFIG_FILE": "/path/to/ucm_config_example.yaml"}
    }'
```

To start the second Decode instance on the same node, modify `--port` (e.g., port 7803) and `ASCEND_RT_VISIBLE_DEVICES` accordingly.

**Step 4: Run Load Balancing Service**

```bash
python /vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
    --port 7805 \
    --host 0.0.0.0 \
    --prefiller-hosts 192.168.10.1 192.168.10.1 \
    --prefiller-ports 7800 7801 \
    --decoder-hosts 192.168.10.2 192.168.10.2 \
    --decoder-ports 7802 7803
```

**Step 5: Performance Testing**

```bash
vllm bench serve \
    --backend vllm \
    --model /models/QwQ-32B \
    --host 192.168.10.1 \
    --port 7805 \
    --seed 123456 \
    --dataset-name random \
    --num-prompts 10 \
    --random-input-len 8000 \
    --random-output-len 1000 \
    --request-rate inf \
    --ignore-eos
```

#### Distributed PD Disaggregation (P2P)

In P2P distributed PD disaggregation, Mooncake handles direct KV cache transfer from Prefill to Decode nodes via high-speed network, while UCM provides prefix cache on Prefill nodes for KV cache reuse. This mode is suitable for scenarios with homogeneous P/D nodes and lower latency requirements.

> **Note**: From vLLM-Ascend 0.11.0, the official image includes pre-installed Mooncake. For installation details, refer to [kvcache-ai/Mooncake](https://github.com/kvcache-ai/Mooncake).

**Example: 2P2D Setup**

Assume 2 Prefill instances on node 192.168.10.1 (ports 9000, 9001) and 2 Decode instances on node 192.168.10.2 (ports 9000, 9001).

**Step 1: Run Mooncake Master Service**

Run Mooncake master on any node (e.g., 192.168.10.1):

```bash
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
mooncake_master --port 50088 \
    --eviction_high_watermark_ratio 0.9 \
    --eviction_ratio 0.1 \
    --default_kv_lease_ttl 11000
```

Prepare `mooncake.json` on each node:

```json
{
    "metadata_server": "P2PHANDSHAKE",
    "protocol": "ascend",
    "device_name": "",
    "master_server_address": "192.168.10.1:50088",
    "global_segment_size": "1GB"
}
```

Also prepare a UCM configuration file (`ucm_config_example.yaml`) for prefix cache on Prefill nodes:

```yaml
ucm_connectors:
  - ucm_connector_name: "UcmPipelineStore"
    ucm_connector_config:
      store_pipeline: "Cache|Posix"
      storage_backends: "/mnt/test1"
      cache_buffer_capacity_gb: 64
enable_event_sync: true
use_layerwise: true
```

> **Note**: For more configuration options, refer to [UCM PipelineStore Documentation](https://ucm.readthedocs.io/en/latest/user-guide/prefix-cache/pipeline_store.html).

**Step 2: Run Prefill Service**

```bash
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$LD_LIBRARY_PATH
export PYTHONHASHSEED=0
export PYTHONPATH=$PYTHONPATH:/vllm-workspace/vllm
export MOONCAKE_CONFIG_PATH="./mooncake.json"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

vllm serve /models/QwQ-32B \
    --host 0.0.0.0 \
    --port 9000 \
    --gpu-memory-utilization 0.92 \
    --data-parallel-size 1 \
    --tensor-parallel-size 4 \
    --seed 1024 \
    --max-model-len 17000 \
    --max-num-batched-tokens 8000 \
    --max-num-seqs 20 \
    --trust-remote-code \
    --enforce-eager \
    --kv-transfer-config \
    '{
        "kv_connector": "MultiConnector",
        "kv_role": "kv_producer",
        "kv_connector_extra_config": {
            "connectors": [
                {
                    "kv_connector": "MooncakeConnectorV1",
                    "kv_role": "kv_producer",
                    "kv_port": 20001,
                    "kv_connector_extra_config": {
                        "prefill": {"dp_size": 1, "tp_size": 4},
                        "decode": {"dp_size": 1, "tp_size": 4}
                    }
                },
                {
                    "kv_connector": "UCMConnector",
                    "kv_role": "kv_both",
                    "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
                    "kv_connector_extra_config": {"UCM_CONFIG_FILE": "/vllm-workspace/unified-cache-management/examples/ucm_config_example.yaml"}
                }
            ]
        }
    }'
```

To start multiple Prefill instances on the same node, modify `--port`, `kv_port`, and `ASCEND_RT_VISIBLE_DEVICES` for each instance (e.g., port 9001 with kv_port 20002 for the second instance).

**Step 3: Run Decode Service**

```bash
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$LD_LIBRARY_PATH
export PYTHONHASHSEED=0
export PYTHONPATH=$PYTHONPATH:/vllm-workspace/vllm
export MOONCAKE_CONFIG_PATH="./mooncake.json"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

vllm serve /models/QwQ-32B \
    --host 0.0.0.0 \
    --port 9000 \
    --data-parallel-size 1 \
    --tensor-parallel-size 4 \
    --seed 1024 \
    --max-model-len 17000 \
    --max-num-batched-tokens 8000 \
    --trust-remote-code \
    --max-num-seqs 4 \
    --gpu-memory-utilization 0.92 \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --kv-transfer-config \
    '{
        "kv_connector": "MooncakeConnectorV1",
        "kv_role": "kv_consumer",
        "kv_port": 20001,
        "kv_connector_extra_config": {
            "prefill": {"dp_size": 1, "tp_size": 4},
            "decode": {"dp_size": 1, "tp_size": 4}
        }
    }'
```

To start multiple Decode instances on the same node, modify `--port`, `kv_port`, and `ASCEND_RT_VISIBLE_DEVICES` for each instance (e.g., port 9001 with kv_port 20002 for the second instance).

**Step 4: Run Load Balancing Service**

```bash
python /vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
    --port 7850 \
    --host 0.0.0.0 \
    --prefiller-hosts 192.168.10.1 192.168.10.1 \
    --prefiller-ports 9000 9001 \
    --decoder-hosts 192.168.10.2 192.168.10.2 \
    --decoder-ports 9000 9001
```

**Step 5: Performance Testing**

```bash
vllm bench serve \
    --backend vllm \
    --model /models/QwQ-32B \
    --host 192.168.10.1 \
    --port 7850 \
    --seed 123456 \
    --dataset-name random \
    --num-prompts 10 \
    --random-input-len 8000 \
    --random-output-len 1000 \
    --request-rate inf \
    --ignore-eos
```

### PD-Mixed Inference

PD-Mixed Inference refers to the standard vLLM serving mode where Prefill and Decode phases for different requests are processed concurrently within the same instance. Unlike PD Disaggregation which physically separates Prefill and Decode into dedicated instances, PD-Mixed handles both phases in a unified scheduler, allowing interleaved execution: while one request is in Decode phase, another request can simultaneously undergo Prefill.

UCM enhances PD-Mixed by providing persistent KV cache storage, enabling:

- Prefix cache reuse across requests with shared prefixes
- KV cache persistence across service restarts
- Offloading KV cache to external storage to reduce GPU memory pressure

**Step 1: Prepare UCM Configuration File**

Create a configuration file (e.g., `ucm_config_example.yaml`) with PipelineStore:

```yaml
ucm_connectors:
  - ucm_connector_name: "UcmPipelineStore"
    ucm_connector_config:
      store_pipeline: "Cache|Posix"
      storage_backends: "/mnt/test1"
      cache_buffer_capacity_gb: 64
enable_event_sync: true
use_layerwise: true
```

Key configuration parameters:

- **storage_backends**: Directory for KV cache storage. Can be local SSD or NFS-mounted path.

> **Note**: For more configuration options, refer to [UCM PipelineStore Documentation](https://ucm.readthedocs.io/en/latest/user-guide/prefix-cache/pipeline_store.html).

**Step 2: Run PD-Mixed Service**

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
vllm serve /models/QwQ-32B \
    --host 0.0.0.0 \
    --port 7800 \
    --gpu-memory-utilization 0.92 \
    --data-parallel-size 2 \
    --tensor-parallel-size 4 \
    --seed 1024 \
    --max-model-len 17000 \
    --max-num-batched-tokens 8000 \
    --max-num-seqs 20 \
    --trust-remote-code \
    --enforce-eager \
    --block-size 128 \
    --kv-transfer-config \
    '{
        "kv_connector": "UCMConnector",
        "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {"UCM_CONFIG_FILE": "/path/to/ucm_config_example.yaml"}
    }'
```

**Step 3: Performance Testing**

Run the benchmark twice to observe the prefix cache effect:

```bash
# First run - no cache hit
vllm bench serve \
    --backend vllm \
    --model /models/QwQ-32B \
    --host localhost \
    --port 7800 \
    --seed 123456 \
    --dataset-name random \
    --num-prompts 10 \
    --random-input-len 8000 \
    --random-output-len 1000 \
    --request-rate inf \
    --ignore-eos

# Second run - observe cache hit improvement
vllm bench serve \
    --backend vllm \
    --model /models/QwQ-32B \
    --host localhost \
    --port 7800 \
    --seed 123456 \
    --dataset-name random \
    --num-prompts 10 \
    --random-input-len 8000 \
    --random-output-len 1000 \
    --request-rate inf \
    --ignore-eos
```

After the second run, a significant reduction in TTFT should be observed due to UCM prefix cache hits. Review the vLLM logs for cache hit information:

```bash
INFO ucm_connector.py:xxx: request_id: xxx, total_blocks_num: xxx, hit hbm: 0, hit external: xxx
```

## Example: PD Disaggregation with Large Scale Expert Parallelism

This section demonstrates PD disaggregation for MoE models with large-scale Expert Parallelism. MoE models require enabling data parallelism to distribute expert weights across multiple nodes.

**Deployment Configuration:**

- **Prefill Instance**: 4 nodes (192.168.10.1-4), DP4TP8 (4 DP processes, each with TP8)
- **Decode Instance**: 4 nodes (192.168.10.5-8), DP8TP4 (8 DP processes, each with TP4)
- **Total**: 8 Atlas 800T A2 servers with 8 Ascend 910B3 NPU cards each
- **Storage**: 8 servers connected to AI storage device A800 via CE8875 switch

> **Note**: For external load balancing data parallelism, refer to the vLLM official documentation: [Data Parallel Deployment of external Load Balancing](https://docs.vllm.ai/en/latest/serving/data_parallel_deployment/#external-load-balancing).

### Deployment Steps

**Step 1: Run Mooncake Master Service**

Run Mooncake master on any node (e.g., 192.168.10.1):

```bash
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
mooncake_master --port 50088 \
    --eviction_high_watermark_ratio 0.9 \
    --eviction_ratio 0.1 \
    --default_kv_lease_ttl 11000
```

Prepare `mooncake.json` on all 8 nodes:

```json
{
    "metadata_server": "P2PHANDSHAKE",
    "protocol": "ascend",
    "device_name": "",
    "master_server_address": "192.168.10.1:50088",
    "global_segment_size": "1GB"
}
```

**Step 2: Run Prefill Service (DP4TP8)**

First, prepare a UCM configuration file (`ucm_config_example.yaml`) for prefix cache on Prefill nodes (192.168.10.1-4):

```yaml
ucm_connectors:
  - ucm_connector_name: "UcmPipelineStore"
    ucm_connector_config:
      store_pipeline: "Cache|Posix"
      storage_backends: "/mnt/test1"
      cache_buffer_capacity_gb: 64
enable_event_sync: true
use_layerwise: true
```

Key configuration parameters:

- **storage_backends**: The shared storage directory accessible from all nodes (e.g., NFS-mounted path or 3FS).

> **Note**: For more configuration options, refer to [UCM PipelineStore Documentation](https://ucm.readthedocs.io/en/latest/user-guide/prefix-cache/pipeline_store.html).

Prepare `prefill.sh` on Prefill nodes (192.168.10.1-4):

```bash
#!/bin/sh

export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$LD_LIBRARY_PATH
export PYTHONHASHSEED=0
export PYTHONPATH=$PYTHONPATH:/vllm-workspace/vllm
export MOONCAKE_CONFIG_PATH="./mooncake.json"

device_list=$1
local_ip=$2
nic_name=$3
server_port=$4
tp_size=$5
dp_size=$6
dp_rank=$7
dp_address=$8
dp_rpc_port=$9
mooncake_port=${10}

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export HCCL_BUFFSIZE=256
export ASCEND_RT_VISIBLE_DEVICES=$device_list

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export VLLM_USE_MODELSCOPE="True"

vllm serve /models/GLM-5.1-w4a8 \
    --host 0.0.0.0 \
    --port $server_port \
    --data-parallel-size $dp_size \
    --data-parallel-address $dp_address \
    --data-parallel-rpc-port $dp_rpc_port \
    --data-parallel-rank $dp_rank \
    --tensor-parallel-size $tp_size \
    --enable-expert-parallel \
    --seed 1024 \
    --max-model-len 17000 \
    --max-num-batched-tokens 8000 \
    --trust-remote-code \
    --max-num-seqs 4 \
    --gpu-memory-utilization 0.92 \
    --quantization ascend \
    --enforce-eager \
    --additional-config '{"enable_weight_nz_layout":true,"enable_prefill_optimizations":true}' \
    --kv-transfer-config \
    '{
        "kv_connector": "MultiConnector",
        "kv_role": "kv_producer",
        "kv_connector_extra_config": {
            "connectors": [
                {
                    "kv_connector": "MooncakeConnectorV1",
                    "kv_role": "kv_producer",
                    "kv_port": '$mooncake_port',
                    "kv_connector_extra_config": {
                        "prefill": {"dp_size": '$dp_size', "tp_size": '$tp_size'},
                        "decode": {"dp_size": 8, "tp_size": 4}
                    }
                },
                {
                    "kv_connector": "UCMConnector",
                    "kv_role": "kv_both",
                    "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
                    "kv_connector_extra_config": {"UCM_CONFIG_FILE": "/path/to/ucm_config_example.yaml"}
                }
            ]
        }
    }' 2>&1 | tee "prefiller_dp_$dp_rank.log"
```

Prepare `run_multi_dp.sh` for Prefill nodes:

```bash
#!/bin/bash

local_ip="xxxx"           # IP of current node (192.168.10.1/2/3/4)
nic_name="xxxx"           # Network interface name corresponding to local_ip
tp_size=8
dp_size=4                 # Total DP engines for Prefill
dp_size_local=1           # 1 DP process per node (TP8 uses all 8 cards)
dp_rank_start=xxxx        # 0 for node1, 1 for node2, 2 for node3, 3 for node4
dp_address="192.168.10.1" # Master node for DP communication
dp_rpc_port=13395
server_port=9000
mooncake_port=20001
template_path="./prefill.sh"
cards_per_node=8

cards_per_process=$((cards_per_node / dp_size_local))

for ((i=0; i<dp_size_local; i++)); do
  dp_rank=$((dp_rank_start + i))
  server_port=$((server_port + i))
  mooncake_port=$((mooncake_port + i * tp_size))
  
  start_card=$((i * cards_per_process))
  device_list=$(seq -s, $start_card $((start_card + cards_per_process - 1)))
  
  bash $template_path $device_list $local_ip $nic_name $server_port $tp_size $dp_size $dp_rank $dp_address $dp_rpc_port $mooncake_port &
done

wait
```

Execute `run_multi_dp.sh` on each Prefill node (192.168.10.1-4) with appropriate `local_ip` and `dp_rank_start`:

- 192.168.10.1: `dp_rank_start=0`
- 192.168.10.2: `dp_rank_start=1`
- 192.168.10.3: `dp_rank_start=2`
- 192.168.10.4: `dp_rank_start=3`

**Step 3: Run Decode Service (DP8TP4)**

Prepare `decode.sh` on Decode nodes (192.168.10.5-8):

```bash
#!/bin/sh

export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$LD_LIBRARY_PATH
export PYTHONHASHSEED=0
export PYTHONPATH=$PYTHONPATH:/vllm-workspace/vllm
export MOONCAKE_CONFIG_PATH="./mooncake.json"

device_list=$1
local_ip=$2
nic_name=$3
server_port=$4
tp_size=$5
dp_size=$6
dp_rank=$7
dp_address=$8
dp_rpc_port=$9
mooncake_port=${10}

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export HCCL_BUFFSIZE=256
export ASCEND_RT_VISIBLE_DEVICES=$device_list

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export VLLM_USE_MODELSCOPE="True"

vllm serve /models/GLM-5.1-w4a8 \
    --host 0.0.0.0 \
    --port $server_port \
    --data-parallel-size $dp_size \
    --data-parallel-address $dp_address \
    --data-parallel-rpc-port $dp_rpc_port \
    --data-parallel-rank $dp_rank \
    --tensor-parallel-size $tp_size \
    --enable-expert-parallel \
    --seed 1024 \
    --max-model-len 17000 \
    --max-num-batched-tokens 8000 \
    --trust-remote-code \
    --max-num-seqs 4 \
    --gpu-memory-utilization 0.92 \
    --quantization ascend \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --kv-transfer-config \
    '{
        "kv_connector": "MooncakeConnectorV1",
        "kv_role": "kv_consumer",
        "kv_port": '$mooncake_port',
        "kv_connector_extra_config": {
            "prefill": {"dp_size": 4, "tp_size": 8},
            "decode": {"dp_size": '$dp_size', "tp_size": '$tp_size'}
        }
    }' 2>&1 | tee "decoder_dp_$dp_rank.log"
```

Prepare `run_multi_dp.sh` for Decode nodes:

```bash
#!/bin/bash

local_ip="xxxx"           # IP of current node (192.168.10.5/6/7/8)
nic_name="xxxx"           # Network interface name corresponding to local_ip
tp_size=4
dp_size=8                 # Total DP engines for Decode
dp_size_local=2           # 2 DP processes per node (TP4 uses 4 cards each)
dp_rank_start=xxxx        # 0 for node5, 2 for node6, 4 for node7, 6 for node8
dp_address="192.168.10.5" # Master node for DP communication
dp_rpc_port=13395
server_port=9000
mooncake_port=20001
template_path="./decode.sh"
cards_per_node=8

cards_per_process=$((cards_per_node / dp_size_local))

for ((i=0; i<dp_size_local; i++)); do
  dp_rank=$((dp_rank_start + i))
  server_port=$((server_port + i))
  mooncake_port=$((mooncake_port + i * tp_size))
  
  start_card=$((i * cards_per_process))
  device_list=$(seq -s, $start_card $((start_card + cards_per_process - 1)))
  
  bash $template_path $device_list $local_ip $nic_name $server_port $tp_size $dp_size $dp_rank $dp_address $dp_rpc_port $mooncake_port &
done

wait
```

Execute `run_multi_dp.sh` on each Decode node (192.168.10.5-8) with appropriate `local_ip` and `dp_rank_start`:

- 192.168.10.5: `dp_rank_start=0`
- 192.168.10.6: `dp_rank_start=2`
- 192.168.10.7: `dp_rank_start=4`
- 192.168.10.8: `dp_rank_start=6`

**Step 4: Run Load Balancing Service**

```bash
python /vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
    --port 7850 \
    --host 0.0.0.0 \
    --prefiller-hosts 192.168.10.1 192.168.10.2 192.168.10.3 192.168.10.4 \
    --prefiller-ports 9000 9000 9000 9000 \
    --decoder-hosts 192.168.10.5 192.168.10.5 192.168.10.6 192.168.10.6 192.168.10.7 192.168.10.7 192.168.10.8 192.168.10.8 \
    --decoder-ports 9000 9001 9000 9001 9000 9001 9000 9001
```

### Benchmark Results

The following benchmark demonstrates UCM prefix cache effectiveness in large-scale Expert Parallelism PD disaggregation scenarios.

**Test Configuration:**

- Total requests: 128
- Request concurrency: 128
- Constraint: Total requests kept within Prefill instance's available HBM capacity for KV cache storage

**KV Cache Pre-seeding Procedure:**

Before each test, KV cache must be pre-seeded with a prefix ratio of **0.8**:

1. **Pre-seed Phase**: Send 128 requests with input length = `target_input_length × 0.8` and output length = 1 to establish the KV cache prefix
2. **Test Phase**: Send 128 requests with full target input length and output length = 1000

Example for 32K input scenario:

- Pre-seed: 128 requests with 25600 (32K × 0.8) input tokens + 1 output token
- Test: 128 requests with 32000 input tokens + 1000 output tokens

This procedure ensures the prefix portion (80% of input) is cached before measuring performance, simulating real-world prefix reuse scenarios.

**Test Commands:**

```bash
# Step 1: Pre-seed KV cache (25600 = 32000 * 0.8)
vllm bench serve \
    --backend vllm \
    --model /models/GLM-5.1-w4a8 \
    --host 192.168.10.1 \
    --port 7850 \
    --seed 123456 \
    --dataset-name random \
    --num-prompts 128 \
    --random-input-len 25600 \
    --random-output-len 1 \
    --request-rate inf \
    --ignore-eos

# Step 2: Run performance test
vllm bench serve \
    --backend vllm \
    --model /models/GLM-5.1-w4a8 \
    --host 192.168.10.1 \
    --port 7850 \
    --seed 123456 \
    --dataset-name random \
    --num-prompts 128 \
    --random-input-len 32000 \
    --random-output-len 1000 \
    --request-rate inf \
    --ignore-eos
```

**Test Scenarios:**

| Scenario | Description |
|----------|-------------|
| **Recalculation** | Baseline without UCM, HBM prefix cache disabled (full recomputation) |
| **HBM PC** | Without UCM, HBM prefix cache enabled |
| **UCM PC** | With UCM prefix cache enabled |

**Performance Results:**

<table>
  <thead>
    <tr>
      <th rowspan="2">Input Length</th>
      <th rowspan="2">Output Length</th>
      <th colspan="3">Recalculation</th>
      <th colspan="3">HBM PC</th>
      <th colspan="3">UCM PC</th>
    </tr>
    <tr>
      <th>TTFT (ms)</th>
      <th>TPOT (ms)</th>
      <th>E2EL (ms)</th>
      <th>TTFT (ms)</th>
      <th>TPOT (ms)</th>
      <th>E2EL (ms)</th>
      <th>TTFT (ms)</th>
      <th>TPOT (ms)</th>
      <th>E2EL (ms)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>32K</strong></td>
      <td><strong>1K</strong></td>
      <td>140730</td>
      <td>64</td>
      <td>173820</td>
      <td>108879</td>
      <td>65</td>
      <td>142228</td>
      <td>51861</td>
      <td>66</td>
      <td>85615</td>
    </tr>
    <tr>
      <td><strong>64K</strong></td>
      <td><strong>1K</strong></td>
      <td>181864</td>
      <td>64</td>
      <td>214988</td>
      <td>144444</td>
      <td>65</td>
      <td>177561</td>
      <td>69718</td>
      <td>66</td>
      <td>103752</td>
    </tr>
    <tr>
      <td><strong>128K</strong></td>
      <td><strong>1K</strong></td>
      <td>268016</td>
      <td>65</td>
      <td>301648</td>
      <td>267680</td>
      <td>65</td>
      <td>301135</td>
      <td>105083</td>
      <td>66</td>
      <td>138946</td>
    </tr>
  </tbody>
</table>

> **Note**: Due to data parallelism, requests during the test phase may not be routed to the same DP process that was used for KV cache pre-seeding. As a result, HBM PC achieves an actual cache hit rate lower than the intended 0.8. UCM addresses this limitation by storing all KV cache in shared external storage, ensuring that requests can hit cached data regardless of which DP process handles them. This guarantees a true cache hit rate equal to the pre-seeding ratio of 0.8, significantly reducing TTFT compared to HBM PC. The improved TTFT effectively increases Prefill instance throughput, thereby boosting the overall system throughput.
