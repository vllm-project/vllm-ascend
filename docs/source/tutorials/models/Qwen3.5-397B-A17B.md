# Qwen3.5-397B-A17B Deployment Tutorial

## 1 Introduction

Qwen3.5-397B-A17B is a large-scale Qwen3.5 MoE model that combines multimodal capability, long-context inference, MTP speculative decoding, and W8A8 quantized deployment for production serving on Ascend hardware.

This document describes the main validation steps for the model, including supported features, prerequisites, installation, single-node online deployment, multi-node deployment, Prefill-Decode (PD) disaggregation, functional verification, accuracy and performance evaluation, performance tuning, and FAQs.

The `Qwen3.5-397B-A17B` model is first supported in `vllm-ascend:v0.17.0rc1`. Use `v0.17.0rc1` or later for this model. The examples below use the version placeholder configured by the documentation build system.

## 2 Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix, including BF16, W8A8 quantization, chunked prefill, automatic prefix caching, speculative decoding, asynchronous scheduling, tensor parallelism, expert parallelism, data parallelism, PD disaggregation, and ACLGraph support.

Refer to [feature guide](../../user_guide/feature_guide/index.md) to get feature configuration details.

:::{note}
The support matrix records the maximum verified capability for this model. The startup examples in this document use practical validation settings for online serving and performance testing. Adjust `--max-model-len`, `--max-num-seqs`, and `--max-num-batched-tokens` based on your service workload and available KV cache.
:::

## 3 Prerequisites

### 3.1 Model Weight

- `Qwen3.5-397B-A17B` (BF16 version): requires 2 Atlas 800 A3 (64G x 16) nodes or 4 Atlas 800 A2 (64G x 8) nodes. [Download model weight](https://www.modelscope.cn/models/Qwen/Qwen3.5-397B-A17B).
- `Qwen3.5-397B-A17B-w8a8` (quantized version): requires 1 Atlas 800 A3 (64G x 16) node or 2 Atlas 800 A2 (64G x 8) nodes. [Download model weight](https://www.modelscope.cn/models/Eco-Tech/Qwen3.5-397B-A17B-w8a8-mtp).

It is recommended to download the model weight to a shared directory across multiple nodes, such as `/root/.cache/`, so that all serving nodes can load the same path.

### 3.2 Verify Multi-node Communication (Optional)

If you want to deploy the model in a multi-node environment, verify the communication environment according to [verify multi-node communication environment](../../installation.md#verify-multi-node-communication).

## 4 Installation

### 4.1 Docker Image Installation

Select an image based on your machine type. For example, use `quay.io/ascend/vllm-ascend:|vllm_ascend_version|` for Atlas 800 A2 and `quay.io/ascend/vllm-ascend:|vllm_ascend_version|-a3` for Atlas 800 A3.

Refer to [using docker](../../installation.md#set-up-using-docker) for the complete installation guide.

```{code-block} bash
:substitutions:

# Update --device according to your device.
# Atlas A2: /dev/davinci[0-7]
# Atlas A3: /dev/davinci[0-15]
# Download the model weight to /root/.cache in advance.
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
export NAME=vllm-ascend

docker run --rm \
  --name $NAME \
  --net=host \
  --shm-size=1g \
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

After entering the container, verify that vLLM and vLLM-Ascend can be imported:

```shell
python -c "import vllm, vllm_ascend; print('vllm and vllm_ascend are ready')"
```

If you want to deploy a multi-node service, set up the same environment on each node.

### 4.2 Source Code Installation

You can also build and install `vllm-ascend` from source. Refer to [set up using python](../../installation.md#set-up-using-python).

If you want to deploy a multi-node service, install the same version of vLLM and vLLM-Ascend on each node.

## 5 Online Service Deployment

### 5.1 Single-Node Online Deployment

Single-node deployment runs both Prefill and Decode on the same node. It is suitable for functional validation, long-context single-cluster serving, and W8A8 deployment on 1 Atlas 800 A3 (64G x 16) node. The W8A8 version needs `--quantization ascend`.

Run the following script to execute online 128K inference on 1 Atlas 800 A3 (64G x 16).

```shell
#!/bin/sh

# Load model from ModelScope to speed up download.
export VLLM_USE_MODELSCOPE=True

# Reduce memory fragmentation and avoid out-of-memory errors.
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=1024
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl kernel.sched_migration_cost_ns=50000
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD

vllm serve Eco-Tech/Qwen3.5-397B-A17B-w8a8-mtp \
  --host 0.0.0.0 \
  --port 8000 \
  --data-parallel-size 1 \
  --tensor-parallel-size 16 \
  --enable-expert-parallel \
  --seed 1024 \
  --quantization ascend \
  --served-model-name qwen3.5 \
  --max-num-seqs 128 \
  --max-model-len 133000 \
  --max-num-batched-tokens 16384 \
  --trust-remote-code \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": true}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --additional-config '{"enable_cpu_binding":true, "enable_fused_mc2":1, "enable_flashcomm1":true}'
```

Common Issues Tip: If the service fails to start, HBM is insufficient, or requests are not scheduled as expected, refer to [FAQs](../../faqs.md) first, and then check the model-specific FAQ in Section 10.

**Key parameters:**

- `--data-parallel-size 1` and `--tensor-parallel-size 16` set DP and TP for one 16-NPU A3 node.
- `--enable-expert-parallel` enables expert parallelism for MoE layers. Do not mix MoE tensor parallelism and expert parallelism in the same MoE layer.
- `--max-model-len` is the maximum input plus output length for a single request. Increase it only when enough KV cache is available.
- `--max-num-seqs` is the maximum number of active requests scheduled by each DP group. For performance tests, set `--max-num-seqs * --data-parallel-size` greater than or equal to the test concurrency.
- `--max-num-batched-tokens` is the maximum number of tokens processed in one scheduler step. A larger value can improve prefill efficiency but consumes more activation memory.
- `--gpu-memory-utilization` controls how much HBM vLLM can use to calculate KV cache capacity. A higher value increases KV cache size but can trigger OOM if runtime memory is higher than the profile run.
- `--enable-prefix-caching` enables prefix caching. For Qwen3.5, short prefixes may not be cached when the hybrid KV cache manager adjusts the block size to a large value.
- `--quantization ascend` enables Ascend quantization for the W8A8 model. Remove this option when deploying the BF16 model.
- `--speculative-config` enables Qwen3.5 MTP speculative decoding. Reduce `num_speculative_tokens` or remove this option if the workload is sensitive to first-token latency or if MTP is unstable in your environment.
- `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'` enables full decode ACLGraph replay to reduce dispatch overhead.
- `--additional-config` enables Ascend-specific optimizations. `enable_fused_mc2` enables MoE fused operators, `enable_flashcomm1` enables FlashComm1, and `enable_cpu_binding` enables Ascend-native CPU binding.

### 5.2 Multi-Node Deployment with MP (Recommended)

Multi-node MP deployment uses vLLM data parallelism across nodes and tensor parallelism within each node. It is recommended for the W8A8 model on 2 Atlas 800 A2 (64G x 8) nodes.

Assume you have 2 Atlas 800 A2 nodes and want to deploy `Qwen3.5-397B-A17B-w8a8-mtp` across them. Replace `nic_name`, `local_ip`, and `node0_ip` with the actual network interface and IP addresses in your environment.

Run the following script on node 0.

```shell
#!/bin/sh

export VLLM_USE_MODELSCOPE=True
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# Get these values through ifconfig.
# nic_name is the network interface name corresponding to local_ip.
nic_name="xxxx"
local_ip="xxxx"

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=1024
export TASK_QUEUE_ENABLE=1

vllm serve Eco-Tech/Qwen3.5-397B-A17B-w8a8-mtp \
  --host 0.0.0.0 \
  --port 8000 \
  --data-parallel-size 2 \
  --api-server-count 2 \
  --data-parallel-size-local 1 \
  --data-parallel-address $local_ip \
  --data-parallel-rpc-port 13389 \
  --seed 1024 \
  --served-model-name qwen3.5 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --max-num-seqs 16 \
  --max-model-len 32768 \
  --max-num-batched-tokens 4096 \
  --trust-remote-code \
  --gpu-memory-utilization 0.9 \
  --no-enable-prefix-caching \
  --quantization ascend \
  --speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": true}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --additional-config '{"enable_cpu_binding":true, "multistream_overlap_shared_expert": true}'
```

Common Issues Tip: If node 1 cannot join the service or HCCL initialization times out, refer to [verify multi-node communication environment](../../installation.md#verify-multi-node-communication) and [FAQs](../../faqs.md). Make sure the network interface names, IP addresses, and RPC ports are consistent across nodes.

Run the following script on node 1.

```shell
#!/bin/sh

export VLLM_USE_MODELSCOPE=True
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# Get these values through ifconfig.
# nic_name is the network interface name corresponding to local_ip.
nic_name="xxxx"
local_ip="xxxx"

# The value of node0_ip must be consistent with local_ip on node 0.
node0_ip="xxxx"

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=1024
export TASK_QUEUE_ENABLE=1

vllm serve Eco-Tech/Qwen3.5-397B-A17B-w8a8-mtp \
  --host 0.0.0.0 \
  --port 8000 \
  --headless \
  --data-parallel-size 2 \
  --data-parallel-size-local 1 \
  --data-parallel-start-rank 1 \
  --data-parallel-address $node0_ip \
  --data-parallel-rpc-port 13389 \
  --seed 1024 \
  --tensor-parallel-size 8 \
  --served-model-name qwen3.5 \
  --max-num-seqs 16 \
  --max-model-len 32768 \
  --max-num-batched-tokens 4096 \
  --enable-expert-parallel \
  --trust-remote-code \
  --gpu-memory-utilization 0.9 \
  --no-enable-prefix-caching \
  --quantization ascend \
  --speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": true}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --additional-config '{"enable_cpu_binding":true, "multistream_overlap_shared_expert": true}'
```

Common Issues Tip: If the headless node exits immediately, check whether node 0 is already running, whether `--data-parallel-address` points to node 0, and whether `--data-parallel-start-rank` is unique for each node.

If the service starts successfully, the following information is displayed on node 0:

```shell
INFO:     Started server process [44610]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Started server process [44611]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

**Key parameters for MP deployment:**

- `--data-parallel-size` is the global DP size across all nodes. In the example, 2 DP ranks are used.
- `--data-parallel-size-local` is the number of DP ranks on the current node. In the example, each A2 node has 1 local DP rank.
- `--data-parallel-start-rank` is the first DP rank on the current node. Node 0 starts from 0 by default, and node 1 starts from 1.
- `--data-parallel-address` must point to the master DP node. Use node 0 `local_ip` on node 0 and `node0_ip` on other nodes.
- `--data-parallel-rpc-port` is the DP RPC port. Use the same value on all nodes and ensure the port is available.
- `--api-server-count` controls how many API server processes are started on the master node.
- `--headless` starts a worker node without exposing an API server. Use it on non-master nodes.
- `--tensor-parallel-size 8` maps one TP group to the 8 NPUs on each A2 node.
- `HCCL_IF_IP`, `GLOO_SOCKET_IFNAME`, `TP_SOCKET_IFNAME`, and `HCCL_SOCKET_IFNAME` bind HCCL, Gloo, and TP communication to the selected network.
- `multistream_overlap_shared_expert` overlaps shared expert computation for better throughput on MoE workloads.

### 5.3 Multi-Node Deployment with Ray

For Ray-based distributed deployment, refer to [Ray Distributed (Qwen/Qwen3-235B-A22B)](../features/ray.md). The same model weight, communication verification, and parameter tuning principles apply to Qwen3.5-397B-A17B.

Common Issues Tip: If Ray workers cannot discover each other, check Ray cluster status first, then verify the same HCCL and network interface settings used in MP deployment.

### 5.4 Prefill-Decode Disaggregation

PD disaggregation separates Prefill and Decode into different service groups. Prefill nodes process large prompt chunks, Decode nodes serve token generation, and a proxy forwards requests between them. This mode is suitable for production serving scenarios where prefill and decode resource ratios need to be tuned separately.

We recommend using Mooncake for deployment. Refer to [Mooncake](../features/pd_disaggregation_mooncake_multi_node.md) for the general PD disaggregation workflow.

Take Atlas 800 A3 (64G x 16) as an example. We recommend deploying 1P1D with 3 nodes for `Qwen3.5-397B-A17B-w8a8-mtp`:

- 1 Prefill node: 1 Atlas 800 A3 (64G x 16).
- 2 Decode nodes: 2 Atlas 800 A3 (64G x 16).

Deploy `run_p.sh`, `run_d0.sh`, and `run_d1.sh` on the corresponding nodes, and deploy a proxy script on the prefill master node to forward requests.

#### 5.4.1 Prefill Node

Create `run_p.sh` on the prefill node.

```shell
#!/bin/bash

unset ftp_proxy
unset https_proxy
unset http_proxy

# Get these values through ifconfig.
# nic_name is the network interface name corresponding to local_ip.
nic_name="xxx"
local_ip="xxx"

export VLLM_ENGINE_READY_TIMEOUT_S=30000
export VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT=480
export IP_ADDRESS=$local_ip
export NETWORK_CARD_NAME=$nic_name
export HCCL_IF_IP=$IP_ADDRESS
export GLOO_SOCKET_IFNAME=$NETWORK_CARD_NAME
export TP_SOCKET_IFNAME=$NETWORK_CARD_NAME
export HCCL_SOCKET_IFNAME=$NETWORK_CARD_NAME
export VLLM_USE_V1=1
export HCCL_BUFFSIZE=1536
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$LD_LIBRARY_PATH
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export VLLM_TORCH_PROFILER_WITH_STACK=0
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

vllm serve Eco-Tech/Qwen3.5-397B-A17B-w8a8-mtp \
  --host ${IP_ADDRESS} \
  --port 30060 \
  --no-enable-prefix-caching \
  --enable-expert-parallel \
  --data-parallel-size 8 \
  --data-parallel-size-local 8 \
  --api-server-count 1 \
  --data-parallel-address ${IP_ADDRESS} \
  --max-num-seqs 64 \
  --data-parallel-rpc-port 6884 \
  --tensor-parallel-size 2 \
  --seed 1024 \
  --distributed-executor-backend mp \
  --served-model-name qwen3.5 \
  --max-model-len 16384 \
  --max-num-batched-tokens 4096 \
  --trust-remote-code \
  --quantization ascend \
  --no-disable-hybrid-kv-cache-manager \
  --speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": true}' \
  --additional-config '{"recompute_scheduler_enable": true, "enable_cpu_binding": true, "enable_fused_mc2":1}' \
  --gpu-memory-utilization 0.9 \
  --enforce-eager \
  --kv-transfer-config \
  '{"kv_connector": "MooncakeLayerwiseConnector",
    "kv_role": "kv_producer",
    "kv_port": "23010",
    "kv_connector_extra_config": {
      "prefill": {
        "dp_size": 8,
        "tp_size": 2
      },
      "decode": {
        "dp_size": 16,
        "tp_size": 2
      }
    }
  }'
```

Common Issues Tip: If the prefill service is not ready for a long time, check whether the model path is shared, `ASCEND_RT_VISIBLE_DEVICES` contains all 16 NPUs, and the Mooncake `kv_port` is available.

#### 5.4.2 Decode Node 0

Create `run_d0.sh` on the first decode node.

```shell
#!/bin/bash

unset ftp_proxy
unset https_proxy
unset http_proxy

# Get these values through ifconfig.
# nic_name is the network interface name corresponding to local_ip.
nic_name="xxx"
local_ip="xxx"

# The value of node0_ip must be consistent with local_ip on the first decode node.
node0_ip="xxxx"

export VLLM_ENGINE_READY_TIMEOUT_S=30000
export VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT=480
export MASTER_IP_ADDRESS=$node0_ip
export IP_ADDRESS=$local_ip
export NETWORK_CARD_NAME=$nic_name
export HCCL_IF_IP=$IP_ADDRESS
export GLOO_SOCKET_IFNAME=$NETWORK_CARD_NAME
export TP_SOCKET_IFNAME=$NETWORK_CARD_NAME
export HCCL_SOCKET_IFNAME=$NETWORK_CARD_NAME
export VLLM_USE_V1=1
export HCCL_BUFFSIZE=1536
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$LD_LIBRARY_PATH
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export VLLM_TORCH_PROFILER_WITH_STACK=0
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

vllm serve Eco-Tech/Qwen3.5-397B-A17B-w8a8-mtp \
  --host ${IP_ADDRESS} \
  --port 30050 \
  --no-enable-prefix-caching \
  --enable-expert-parallel \
  --data-parallel-size 16 \
  --data-parallel-size-local 8 \
  --data-parallel-start-rank 0 \
  --api-server-count 1 \
  --data-parallel-address ${MASTER_IP_ADDRESS} \
  --max-num-seqs 32 \
  --data-parallel-rpc-port 6884 \
  --tensor-parallel-size 2 \
  --seed 1024 \
  --distributed-executor-backend mp \
  --served-model-name qwen3.5 \
  --max-model-len 16384 \
  --max-num-batched-tokens 128 \
  --trust-remote-code \
  --quantization ascend \
  --no-disable-hybrid-kv-cache-manager \
  --speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": true}' \
  --additional-config '{"recompute_scheduler_enable": true, "enable_cpu_binding": true, "enable_fused_mc2":1}' \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --gpu-memory-utilization 0.96 \
  --kv-transfer-config \
  '{"kv_connector": "MooncakeLayerwiseConnector",
    "kv_buffer_device": "npu",
    "kv_role": "kv_consumer",
    "kv_port": "36010",
    "kv_connector_extra_config": {
      "prefill": {
        "dp_size": 8,
        "tp_size": 2
      },
      "decode": {
        "dp_size": 16,
        "tp_size": 2
      }
    }
  }'
```

Common Issues Tip: If decode node 0 fails to initialize, check that `MASTER_IP_ADDRESS` points to decode node 0 itself, `--data-parallel-start-rank` is 0, and `kv_connector_extra_config.decode.dp_size` matches the global decode DP size.

#### 5.4.3 Decode Node 1

Create `run_d1.sh` on the second decode node.

```shell
#!/bin/bash

unset ftp_proxy
unset https_proxy
unset http_proxy

# Get these values through ifconfig.
# nic_name is the network interface name corresponding to local_ip.
nic_name="xxx"
local_ip="xxx"

# The value of node0_ip must be consistent with local_ip on the first decode node.
node0_ip="xxxx"

export VLLM_ENGINE_READY_TIMEOUT_S=30000
export VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT=480
export MASTER_IP_ADDRESS=$node0_ip
export IP_ADDRESS=$local_ip
export NETWORK_CARD_NAME=$nic_name
export HCCL_IF_IP=$IP_ADDRESS
export GLOO_SOCKET_IFNAME=$NETWORK_CARD_NAME
export TP_SOCKET_IFNAME=$NETWORK_CARD_NAME
export HCCL_SOCKET_IFNAME=$NETWORK_CARD_NAME
export VLLM_USE_V1=1
export HCCL_BUFFSIZE=1536
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$LD_LIBRARY_PATH
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export VLLM_TORCH_PROFILER_WITH_STACK=0
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

vllm serve Eco-Tech/Qwen3.5-397B-A17B-w8a8-mtp \
  --host ${IP_ADDRESS} \
  --port 30050 \
  --headless \
  --no-enable-prefix-caching \
  --enable-expert-parallel \
  --data-parallel-size 16 \
  --data-parallel-size-local 8 \
  --data-parallel-start-rank 8 \
  --data-parallel-address ${MASTER_IP_ADDRESS} \
  --max-num-seqs 32 \
  --data-parallel-rpc-port 6884 \
  --tensor-parallel-size 2 \
  --seed 1024 \
  --distributed-executor-backend mp \
  --served-model-name qwen3.5 \
  --max-model-len 16384 \
  --max-num-batched-tokens 128 \
  --trust-remote-code \
  --quantization ascend \
  --no-disable-hybrid-kv-cache-manager \
  --speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": true}' \
  --additional-config '{"recompute_scheduler_enable": true, "enable_cpu_binding": true, "enable_fused_mc2":1}' \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --gpu-memory-utilization 0.96 \
  --kv-transfer-config \
  '{"kv_connector": "MooncakeLayerwiseConnector",
    "kv_buffer_device": "npu",
    "kv_role": "kv_consumer",
    "kv_port": "36010",
    "kv_connector_extra_config": {
      "prefill": {
        "dp_size": 8,
        "tp_size": 2
      },
      "decode": {
        "dp_size": 16,
        "tp_size": 2
      }
    }
  }'
```

Common Issues Tip: If decode node 1 cannot join decode node 0, check that `--headless` is set, `--data-parallel-start-rank` is 8, and `--data-parallel-address` points to decode node 0.

**Key parameters for PD disaggregation:**

- `--distributed-executor-backend mp` uses multiprocessing on each node for the local workers.
- Prefill uses `--data-parallel-size 8`, `--data-parallel-size-local 8`, and `--tensor-parallel-size 2`. This creates 8 local DP groups, each with TP2.
- Decode uses `--data-parallel-size 16`, `--data-parallel-size-local 8`, and `--tensor-parallel-size 2`. Decode node 0 starts from DP rank 0 and decode node 1 starts from DP rank 8.
- `--data-parallel-address` and `--data-parallel-rpc-port` define the DP control plane. For decode nodes, the address must point to decode node 0.
- `--max-num-batched-tokens` is larger on the prefill node and smaller on decode nodes because prefill is prompt-token intensive while decode is latency sensitive.
- `recompute_scheduler_enable` sends requests back to the prefill side to recompute KV cache when decode KV cache is insufficient. Enable it on both prefill and decode nodes in PD mode.
- `--kv-transfer-config` sets the Mooncake connector. `kv_role` is `kv_producer` on prefill and `kv_consumer` on decode.
- `kv_connector_extra_config.prefill.dp_size/tp_size` and `decode.dp_size/tp_size` must match the actual global DP and TP layout.
- `--no-enable-prefix-caching` disables prefix caching. For PD disaggregation, the D-node prefix-cache known issue is tracked in [#7944](https://github.com/vllm-project/vllm-ascend/issues/7944).
- `VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT` is the timeout in seconds for automatically releasing the prefiller KV cache for a request.
- `--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'` is recommended on decode nodes to reduce decode dispatch overhead.

#### 5.4.4 Request Forwarding

Run a proxy server on the same node as the prefiller service instance. You can get the proxy program in the repository examples: [load_balance_proxy_server_example.py](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py).

```shell
unset ftp_proxy
unset https_proxy
unset http_proxy
python3 load_balance_proxy_layerwise_server_example.py \
  --prefiller-hosts 141.xx.xx.1 \
  --prefiller-ports 30060 \
  --decoder-hosts 141.xx.xx.2 141.xx.xx.3 \
  --decoder-ports 30050 30050 \
  --host 141.xx.xx.1 \
  --port 8010
```

For example:

```shell
cd vllm-ascend/examples/disaggregated_prefill_v1/
bash proxy.sh
```

Common Issues Tip: If requests reach the proxy but no output is returned, check that the proxy host list includes all healthy prefill and decode endpoints, and verify that the service verification request in Section 6 succeeds through the proxy port.

## 6 Functional Verification

After the server is started, send a request to verify basic model functionality. For single-node and MP deployment, use the API endpoint on node 0. For PD disaggregation, use the proxy endpoint.

```shell
curl http://<server_ip>:<port>/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5",
    "prompt": "The future of AI is",
    "max_tokens": 50,
    "temperature": 0
  }'
```

Expected result: the HTTP status is 200 and the JSON response contains a `choices` field with generated text.

## 7 Accuracy Evaluation

Here are two accuracy evaluation methods.

### 7.1 Using AISBench

1. Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) for details.
2. After execution, you can get the result. The following result of `Qwen3.5-397B-A17B-w8a8` on `vllm-ascend:v0.17.0rc1` is for reference only.

| dataset | version | metric | mode | vllm-api-general-chat |
| ------- | ------- | ------ | ---- | --------------------- |
| gsm8k | - | accuracy | gen | 96.74 |

### 7.2 Using Language Model Evaluation Harness

Refer to [Using lm_eval](../../developer_guide/evaluation/using_lm_eval.md) for installation and usage details. When using online serving, set `base_url` to the endpoint started in Section 5.

```shell
lm_eval \
  --model local-completions \
  --model_args model=qwen3.5,base_url=http://127.0.0.1:8000/v1/completions,tokenized_requests=False,trust_remote_code=True \
  --tasks gsm8k \
  --output_path ./
```

## 8 Performance Evaluation

### 8.1 Using AISBench

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

### 8.2 Using vLLM Benchmark

Run performance evaluation of `Qwen3.5-397B-A17B-w8a8` as an example. Refer to [vLLM benchmark](https://docs.vllm.ai/en/latest/benchmarking/) for more details.

There are three `vllm bench` subcommands:

- `latency`: benchmark the latency of a single batch of requests.
- `serve`: benchmark online serving throughput.
- `throughput`: benchmark offline inference throughput.

Take `serve` as an example:

```shell
export VLLM_USE_MODELSCOPE=True

vllm bench serve \
  --model Eco-Tech/Qwen3.5-397B-A17B-w8a8-mtp \
  --served-model-name qwen3.5 \
  --dataset-name random \
  --random-input 200 \
  --num-prompts 200 \
  --request-rate 1 \
  --save-result \
  --result-dir ./
```

After several minutes, you can get the performance evaluation result.

## 9 Performance Tuning

### 9.1 Recommended Configurations

The following configurations are validated in specific test environments and are for reference only. The optimal configuration depends on hardware type, maximum input/output length, request concurrency, prefix cache hit rate, quantization, and prefill/decode ratio. Tune the parameters in Section 9.2 based on your actual workload.

| Scenario | Deployment Mode | Total NPUs | Weight Version | Key Considerations |
| -------- | --------------- | ---------- | -------------- | ------------------ |
| Long context | Single-node online serving | 16 A3 NPUs | W8A8 MTP | Use larger `--max-model-len` and reserve enough KV cache. Lower `--max-num-seqs` if OOM occurs. |
| High throughput | Multi-node MP | 16 A2 NPUs | W8A8 MTP | Increase concurrency through DP and tune `--max-num-batched-tokens` for prefill throughput. |
| Low latency | 1P1D PD disaggregation | 48 A3 NPUs | W8A8 MTP | Use separate prefill and decode DP/TP layouts and enable full decode ACLGraph on decode nodes. |

| Scenario | Node Role | NPUs | TP | DP | Max Num Seqs | Max Model Len | Max Num Batched Tokens | MTP Tokens | Prefix Cache | Main Optimizations |
| -------- | --------- | ---- | -- | -- | ------------ | ------------- | ---------------------- | ---------- | ------------ | ------------------ |
| Long context | Single node | 16 | 16 | 1 | 128 | 133000 | 16384 | 3 | On | FullGraph, FlashComm1, Fused MC2, CPU binding |
| High throughput | MP node | 8 per node | 8 | 1 per node, 2 global | 16 per DP | 32768 | 4096 | 3 | Off | FullGraph, shared expert overlap, CPU binding |
| Low latency | Prefill node | 16 | 2 | 8 | 64 | 16384 | 4096 | 3 | Off | Recompute scheduler, Fused MC2, CPU binding |
| Low latency | Decode node | 16 per node | 2 | 8 per node, 16 global | 32 | 16384 | 128 | 3 | Off | FullGraph, recompute scheduler, Fused MC2, CPU binding |

### 9.2 Tuning Guidelines

Refer to [public performance tuning documentation](../../developer_guide/performance_and_debug/optimization_and_tuning.md) for general tuning methods, and refer to [feature matrix](../../user_guide/support_matrix/feature_matrix.md) for feature descriptions.

Recommended tuning order:

1. Set the deployment topology first. Use single-node deployment for validation, MP deployment for simple multi-node serving, and PD disaggregation when prefill and decode need different resource ratios.
2. Choose the maximum context length with `--max-model-len`. Long context increases KV cache usage, so reduce `--max-num-seqs` or `--gpu-memory-utilization` if OOM occurs.
3. Tune `--max-num-batched-tokens`. Larger values usually improve prefill throughput but increase activation memory. Decode-heavy workloads usually need smaller values.
4. Tune `--max-num-seqs` according to service concurrency. Requests above this value wait in the queue and the waiting time is counted in TTFT and TPOT.
5. Tune `--gpu-memory-utilization`. Increase it to provide more KV cache, but leave headroom for runtime memory fluctuation and expert imbalance.
6. Tune `--speculative-config`. MTP can improve decode throughput, but the best `num_speculative_tokens` depends on acceptance rate and workload.
7. Tune ACLGraph capture. `FULL_DECODE_ONLY` is recommended for decode. If you set `cudagraph_capture_sizes` manually, include common decode batch sizes. With FlashComm1, use capture sizes that are multiples of TP size.

### 9.3 Model-Specific Optimizations

| Optimization | Enablement | Benefit | Notes |
| ------------ | ---------- | ------- | ----- |
| Rope optimization | Enabled by default | Reuses position encoding work across layers to reduce decode overhead. | No extra configuration is required. |
| AddRMSNormQuant fusion | Enabled by default | Fuses normalization and quantization to reduce memory access. | Applies to quantized paths. |
| Zero-like elimination | Enabled by default | Removes unnecessary zero-like tensor operations in attention. | No extra configuration is required. |
| Qwen3.5 MTP speculative decoding | `--speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": true}'` | Improves decode throughput when acceptance rate is good. | Reduce speculative tokens if latency or stability regresses. |
| Full decode ACLGraph | `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'` | Reduces operator dispatch overhead and stabilizes decode performance. | Recommended for decode-heavy serving. |
| FlashComm1 | `--additional-config '{"enable_flashcomm1": true}'` | Reduces communication overhead in large TP and high-concurrency scenarios. | May not help low-concurrency workloads. |
| Fused MC2 | `--additional-config '{"enable_fused_mc2": 1}'` | Enables MoE fused operators to improve MoE prefill/decode efficiency. | If accuracy or performance regresses in multi-DP large-token scenarios, disable it and compare. |
| Shared expert overlap | `--additional-config '{"multistream_overlap_shared_expert": true}'` | Overlaps shared expert computation in MoE workloads. | Recommended for MP throughput scenarios. |
| Recompute scheduler | `--additional-config '{"recompute_scheduler_enable": true}'` | Recomputes KV through prefill when decode KV cache is insufficient in PD mode. | Only valid when `kv_role` is `kv_producer` or `kv_consumer`. |
| CPU binding | `--additional-config '{"enable_cpu_binding": true}'` | Improves CPU affinity and reduces scheduling jitter on ARM servers. | Enabled by default in many configurations, but explicitly setting it keeps the recipe clear. |

## 10 FAQ

For common environment, installation, and general parameter issues, refer to [FAQs](../../faqs.md). This section only covers model-specific issues for Qwen3.5-397B-A17B.

### Q1: Why does the service report OOM during startup or soon after accepting requests?

**Phenomenon:** The service fails during profile run, or it starts successfully but reports OOM when real traffic arrives.

**Cause:** Qwen3.5-397B-A17B has high weight and KV cache memory requirements. Large `--max-model-len`, large `--max-num-seqs`, large `--max-num-batched-tokens`, or high `--gpu-memory-utilization` can leave insufficient HBM headroom. Runtime expert load imbalance can also make real inference use more memory than the profile run.

**Solution:** Use the W8A8 model with `--quantization ascend` when possible, lower `--max-model-len`, lower `--max-num-seqs`, lower `--max-num-batched-tokens`, or reduce `--gpu-memory-utilization`. Keep `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`. For BF16 or larger context, use the required number of A2/A3 nodes.

### Q2: Why does multi-node MP deployment hang during initialization?

**Phenomenon:** One node waits for other ranks, HCCL initialization times out, or the headless node exits.

**Cause:** Network interface names, IP addresses, DP ranks, or RPC ports are inconsistent across nodes.

**Solution:** Verify multi-node communication first. Ensure `HCCL_IF_IP`, `GLOO_SOCKET_IFNAME`, `TP_SOCKET_IFNAME`, and `HCCL_SOCKET_IFNAME` match the selected NIC. Ensure all nodes use the same `--data-parallel-rpc-port`, non-master nodes use `--headless`, and `--data-parallel-start-rank` does not overlap.

### Q3: Why is prefix caching disabled in the PD disaggregation examples?

**Phenomenon:** PD disaggregation may show abnormal behavior when prefix caching is enabled on decode nodes.

**Cause:** The D-node prefix-cache issue is a known limitation tracked in [#7944](https://github.com/vllm-project/vllm-ascend/issues/7944).

**Solution:** Use `--no-enable-prefix-caching` for PD disaggregation until the limitation is resolved. For non-PD single-node serving, enable prefix caching only when the workload has repeated prefixes and the cache hit rate is meaningful.

### Q4: Why does performance regress after enabling FlashComm1 or Fused MC2?

**Phenomenon:** Throughput decreases, latency increases, or MoE load becomes unstable after enabling communication or MoE fusion optimizations.

**Cause:** These optimizations are workload dependent. FlashComm1 is most useful in high-concurrency TP scenarios. Fused MC2 may not be suitable for some multi-DP large-token cases where padded tokens overload certain experts.

**Solution:** Compare with `enable_flashcomm1` disabled and `enable_fused_mc2` set to 0. If FlashComm1 is enabled and you tune `cudagraph_capture_sizes`, use values that are multiples of TP size. Keep the better setting for your actual concurrency and prompt length distribution.

### Q5: How should I tune MTP speculative decoding for this model?

**Phenomenon:** MTP improves throughput in some workloads, but may increase first-token latency or provide limited benefit in others.

**Cause:** The benefit depends on speculative token acceptance rate, request length, and decode concurrency.

**Solution:** Start with `num_speculative_tokens` set to 3 as shown in this document. If the service is latency-sensitive or the acceptance rate is low, reduce the value or remove `--speculative-config` and compare TTFT, TPOT, and throughput.
