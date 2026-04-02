# Qwen3.5-397B-A17B

## Introduction

Qwen3.5 represents a significant leap forward, integrating breakthroughs in multimodal learning, architectural efficiency, reinforcement learning scale, and global accessibility to empower developers and enterprises with unprecedented capability and efficiency.

This document will show the main verification steps of the model, including supported features, feature configuration, environment preparation, single-node and multi-node deployment, accuracy and performance evaluation.

The `Qwen3.5-397B-A17B` model is first supported in `vllm-ascend:v0.17.0rc1`.

## Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [feature guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## Environment Preparation

### Model Weight

- `Qwen3.5-397B-A17B`(BF16 version): require 2 Atlas 800 A3 (64G × 16) nodes or 4 Atlas 800 A2 (64G × 8) nodes. [Download model weight](https://www.modelscope.cn/models/Qwen/Qwen3.5-397B-A17B)
- `Qwen3.5-397B-A17B-w8a8`(Quantized version): require 1 Atlas 800 A3 (64G × 16) node or 2 Atlas 800 A2 (64G × 8) nodes. [Download model weight](https://www.modelscope.cn/models/Eco-Tech/Qwen3.5-397B-A17B-w8a8-mtp)

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`.

### Verify Multi-node Communication(Optional)

If you want to deploy multi-node environment, you need to verify multi-node communication according to [verify multi-node communication environment](../../installation.md#verify-multi-node-communication).

### Installation

:::::{tab-set}
::::{tab-item} Use docker image

For example, using images `quay.io/ascend/vllm-ascend:v0.17.0rc1`(for Atlas 800 A2) and `quay.io/ascend/vllm-ascend:v0.17.0rc1-a3`(for Atlas 800 A3).

Select an image based on your machine type and start the docker image on your node, refer to [using docker](../../installation.md#set-up-using-docker).

```{code-block} bash
  :substitutions:
  # Update --device according to your device (Atlas A2: /dev/davinci[0-7] Atlas A3:/dev/davinci[0-15]).
  # Update the vllm-ascend image according to your environment.
  # Note you should download the weight to /root/.cache in advance.
  # Update the vllm-ascend image
  export IMAGE=m.daocloud.io/quay.io/ascend/vllm-ascend:|vllm_ascend_version|
  export NAME=vllm-ascend

  # Run the container using the defined variables
  # Note: If you are running bridge network with docker, please expose available ports for multiple nodes communication in advance.
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
  -it $IMAGE bash
```

::::
::::{tab-item} Build from source

You can build all from source.

- Install `vllm-ascend`, refer to [set up using python](../../installation.md#set-up-using-python).

::::
:::::

If you want to deploy multi-node environment, you need to set up environment on each node.

## Deployment

### Single-node Deployment

`Qwen3.5-397B-A17B` can be deployed on 2 Atlas 800 A3(64G*16) or 4 Atlas 800 A2(64G*8).
`Qwen3.5-397B-A17B-w8a8` can be deployed on 1 Atlas 800 A3(64G*16) or 2 Atlas 800 A2(64G*8), need to start with parameter `--quantization ascend`.

Run the following script to execute online 128k inference On 1 Atlas 800 A3(64G*16).

```shell
#!/bin/sh
# Load model from ModelScope to speed up download
export VLLM_USE_MODELSCOPE=true
# To reduce memory fragmentation and avoid out of memory
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
export VLLM_ASCEND_ENABLE_FUSED_MC2=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

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
--speculative_config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": true}' \
--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
--additional-config '{"enable_cpu_binding":true}' \
--async-scheduling
```

**Notice:**

The parameters are explained as follows:

- `--data-parallel-size` 1 and `--tensor-parallel-size` 16 are common settings for data parallelism (DP) and tensor parallelism (TP) sizes.
- `--max-model-len` represents the context length, which is the maximum value of the input plus output for a single request.
- `--max-num-seqs` indicates the maximum number of requests that each DP group is allowed to process. If the number of requests sent to the service exceeds this limit, the excess requests will remain in a waiting state and will not be scheduled. Note that the time spent in the waiting state is also counted in metrics such as TTFT and TPOT. Therefore, when testing performance, it is generally recommended that `--max-num-seqs` * `--data-parallel-size` >= the actual total concurrency.
- `--max-num-batched-tokens` represents the maximum number of tokens that the model can process in a single step. Currently, vLLM v1 scheduling enables ChunkPrefill/SplitFuse by default, which means:
    - (1) If the input length of a request is greater than `--max-num-batched-tokens`, it will be divided into multiple rounds of computation according to `--max-num-batched-tokens`;
    - (2) Decode requests are prioritized for scheduling, and prefill requests are scheduled only if there is available capacity.
    - Generally, if `--max-num-batched-tokens` is set to a larger value, the overall latency will be lower, but the pressure on GPU memory (activation value usage) will be greater.
- `--gpu-memory-utilization` represents the proportion of HBM that vLLM will use for actual inference. Its essential function is to calculate the available kv_cache size. During the warm-up phase (referred to as profile run in vLLM), vLLM records the peak GPU memory usage during an inference process with an input size of `--max-num-batched-tokens`. The available kv_cache size is then calculated as: `--gpu-memory-utilization` * HBM size - peak GPU memory usage. Therefore, the larger the value of `--gpu-memory-utilization`, the more kv_cache can be used. However, since the GPU memory usage during the warm-up phase may differ from that during actual inference (e.g., due to uneven EP load), setting `--gpu-memory-utilization` too high may lead to OOM (Out of Memory) issues during actual inference. The default value is `0.9`.
- `--enable-expert-parallel` indicates that EP is enabled. Note that vLLM does not support a mixed approach of ETP and EP; that is, MoE can either use pure EP or pure TP.
- `--no-enable-prefix-caching` indicates that prefix caching is disabled. To enable it, for mamba-like models Qwen3.5, set `--enable-prefix-caching` and `--mamba-cache-mode align`. Notice the current implementation of hybrid kv cache might result in a very large block_size when scheduling. For example, the block_size may be adjusted to 2048, which means that any prefix shorter than 2048 will never be cached.
- `--quantization` "ascend" indicates that quantization is used. To disable quantization, remove this option.
- `--compilation-config` contains configurations related to the aclgraph graph mode. The most significant configurations are "cudagraph_mode" and "cudagraph_capture_sizes", which have the following meanings:
"cudagraph_mode": represents the specific graph mode. Currently, "PIECEWISE" and "FULL_DECODE_ONLY" are supported. The graph mode is mainly used to reduce the cost of operator dispatch. Currently, "FULL_DECODE_ONLY" is recommended.
- "cudagraph_capture_sizes": represents different levels of graph modes. The default value is [1, 2, 4, 8, 16, 24, 32, 40,..., `--max-num-seqs`]. In the graph mode, the input for graphs at different levels is fixed, and inputs between levels are automatically padded to the next level. Currently, the default setting is recommended. Only in some scenarios is it necessary to set this separately to achieve optimal performance.

### Multi-node Deployment with MP (Recommended)

Assume you have 2 Atlas 800 A2 nodes, and want to deploy the `Qwen3.5-397B-A17B` model across multiple nodes.

Node 0

```shell
#!/bin/sh
# Load model from ModelScope to speed up download
export VLLM_USE_MODELSCOPE=true
# To reduce memory fragmentation and avoid out of memory
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
# this obtained through ifconfig
# nic_name is the network interface name corresponding to local_ip of the current node
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
--async-scheduling \
--gpu-memory-utilization 0.9 \
--no-enable-prefix-caching \
--speculative_config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": true}' \
--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
--additional-config '{"enable_cpu_binding":true, "multistream_overlap_shared_expert": true}'
```

Node1

```shell
#!/bin/sh
# Load model from ModelScope to speed up download
export VLLM_USE_MODELSCOPE=true
# To reduce memory fragmentation and avoid out of memory
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
# this obtained through ifconfig
# nic_name is the network interface name corresponding to local_ip of the current node
nic_name="xxxx"
local_ip="xxxx"

# The value of node0_ip must be consistent with the value of local_ip set in node0 (master node)
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
--async-scheduling \
--gpu-memory-utilization 0.9 \
--no-enable-prefix-caching \
--speculative_config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": true}' \
--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
--additional-config '{"enable_cpu_binding":true, "multistream_overlap_shared_expert": true}'
```

If the service starts successfully, the following information will be displayed on node 0:

```shell
INFO:     Started server process [44610]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Started server process [44611]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

### Multi-node Deployment with Ray

- refer to [Ray Distributed (Qwen/Qwen3-235B-A22B)](../features/ray.md).

### Prefill-Decode Disaggregation

We recommend using Mooncake for deployment: [Mooncake](../features/pd_disaggregation_mooncake_multi_node.md).

Take Atlas 800 A3 (64G × 16) for example, we recommend to deploy 1P1D (3 nodes) to run Qwen3.5-397B-A17B.

- `Qwen3.5-397B-A17B-w8a8-mtp 1P1D` require 3 Atlas 800 A3 (64G × 16).

To run the vllm-ascend `Prefill-Decode Disaggregation` service, you need to deploy `run_p.sh` 、`run_d0.sh` and `run_d1.sh` script on each node and deploy a `proxy.sh` script on prefill master node to forward requests.

1. Prefill Node 0 `run_p.sh` script

```shell
unset ftp_proxy
unset https_proxy
unset http_proxy
# this obtained through ifconfig
# nic_name is the network interface name corresponding to local_ip of the current node
nic_name="xxx"
local_ip="xxx"

# [Optional] jemalloc
# jemalloc is for better performance, if `libjemalloc.so` is installed on your machine, you can turn it on.
# export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export VLLM_ENGINE_READY_TIMEOUT_S=30000
export VLLM_NIXL_ABORT_REQUEST_TIMEOUT=30000
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

export VLLM_ASCEND_ENABLE_FUSED_MC2=1
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
  --max-num_seqs 64 \
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
  --speculative_config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": true}' \
  --additional-config '{"recompute_scheduler_enable": true, "enable_cpu_binding": true}' \
  --gpu-memory-utilization 0.9 \
  --enforce-eager \
  --kv-transfer-config \
  '{"kv_connector": "MooncakeLayerwiseConnector",
  "kv_role": "kv_producer",
  "kv_port": "23010",
  "engine_id": "0",
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

3. Decode Node 0 `run_d0.sh` script

```shell
unset ftp_proxy
unset https_proxy
unset http_proxy
#!/bin/bash
# this obtained through ifconfig
# nic_name is the network interface name corresponding to local_ip of the current node
nic_name="xxx"
local_ip="xxx"
# The value of node0_ip must be consistent with the value of local_ip set in node0 (master node)
node0_ip="xxxx"

export VLLM_ENGINE_READY_TIMEOUT_S=30000
export VLLM_NIXL_ABORT_REQUEST_TIMEOUT=30000
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

export VLLM_ASCEND_ENABLE_FUSED_MC2=1
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
  --max-num_seqs 32 \
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
  --speculative_config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": true}' \
  --additional-config '{"recompute_scheduler_enable": true, "enable_cpu_binding": true}' \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --gpu-memory-utilization 0.96 \
  --kv-transfer-config \
  '{"kv_connector": "MooncakeLayerwiseConnector",
  "kv_buffer_device": "npu",
  "kv_role": "kv_consumer",
  "kv_port": "36010",
  "engine_id": "1",
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

5. Decode Node 1 `run_d1.sh` script

```shell
unset ftp_proxy
unset https_proxy
unset http_proxy
#!/bin/bash
# this obtained through ifconfig
# nic_name is the network interface name corresponding to local_ip of the current node
nic_name="xxx"
local_ip="xxx"
# The value of node0_ip must be consistent with the value of local_ip set in node0 (master node)
node0_ip="xxxx"

export VLLM_ENGINE_READY_TIMEOUT_S=30000
export VLLM_NIXL_ABORT_REQUEST_TIMEOUT=30000
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

export VLLM_ASCEND_ENABLE_FUSED_MC2=1
export HCCL_OP_EXPANSION_MODE="AIV"
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
  --max-num_seqs 32 \
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
  --speculative_config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": true}' \
  --additional-config '{"recompute_scheduler_enable": true, "enable_cpu_binding": true}' \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --gpu-memory-utilization 0.96 \
  --kv-transfer-config \
  '{"kv_connector": "MooncakeLayerwiseConnector",
  "kv_buffer_device": "npu",
  "kv_role": "kv_consumer",
  "kv_port": "36010",
  "engine_id": "2",
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

**Notice:**
The parameters are explained as follows:

- `--async-scheduling`: enables the asynchronous scheduling function. When Multi-Token Prediction (MTP) is enabled, asynchronous scheduling of operator delivery can be implemented to overlap the operator delivery latency.
- `cudagraph_capture_sizes`: The recommended value is `n x (mtp + 1)`. And the min is `n = 1` and the max is `n = max-num-seqs`. For other values, it is recommended to set them to the number of frequently occurring requests on the Decode (D) node.
- `recompute_scheduler_enable: true`: enables the recomputation scheduler. When the Key-Value Cache (KV Cache) of the decode node is insufficient, requests will be sent to the prefill node to recompute the KV Cache. In the PD separation scenario, it is recommended to enable this configuration on both prefill and decode nodes simultaneously.
- `no-enable-prefix-caching`: The prefix-cache feature is enabled by default. You can use the `--no-enable-prefix-caching` parameter to disable this feature. Notice: for Prefill-Decode disaggregation feature, known issue on D node: [#7944](https://github.com/vllm-project/vllm-ascend/issues/7944)

7. Run the `proxy.sh` script on the prefill master node

Run a proxy server on the same node with the prefiller service instance. You can get the proxy program in the repository's examples: [load\_balance\_proxy\_server\_example.py](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py)

```shell
unset ftp_proxy
unset https_proxy
unset http_proxy
#/bin/bash

if [[ "$offset" == "" ]]; then
    offset=0
fi

python3 load_balance_proxy_layerwise_server_example.py \
    --prefiller-hosts 141.xx.xx.1 \
    --prefiller-ports 30060 \
    --decoder-hosts 141.xx.xx.2 \
    --decoder-ports 30050 \
    --host 141.xx.xx.1 \
    --port 8010
```

```shell
cd vllm-ascend/examples/disaggregated_prefill_v1/
bash proxy.sh
```

## Functional Verification

Once your server is started, you can query the model with input prompts:

```shell
curl http://<node0_ip>:<port>/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen3.5",
        "prompt": "The future of AI is",
        "max_completion_tokens": 50,
        "temperature": 0
    }'
```

## Accuracy Evaluation

Here are two accuracy evaluation methods.

### Using AISBench

1. Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) for details.

2. After execution, you can get the result, here is the result of `Qwen3.5-397B-A17B-w8a8` in `vllm-ascend:v0.17.0rc1` for reference only.

| dataset | version | metric | mode | vllm-api-general-chat |
|----- | ----- | ----- | ----- | -----|
| gsm8k | - | accuracy | gen | 96.74 |

## Performance

### Using AISBench

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

### Using vLLM Benchmark

Run performance evaluation of `Qwen3.5-397B-A17B-w8a8` as an example.

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/contributing/benchmarks.html) for more details.

There are three `vllm bench` subcommands:

- `latency`: Benchmark the latency of a single batch of requests.
- `serve`: Benchmark the online serving throughput.
- `throughput`: Benchmark offline inference throughput.

Take the `serve` as an example. Run the code as follows.

```shell
export VLLM_USE_MODELSCOPE=true
vllm bench serve --model Eco-Tech/Qwen3.5-397B-A17B-w8a8-mtp --dataset-name random --random-input 200 --num-prompts 200 --request-rate 1 --save-result --result-dir ./
```

After about several minutes, you can get the performance evaluation result.
