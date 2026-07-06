# DeepSeek-V3/3.1

## 1 Introduction

DeepSeek-V3.1 is a hybrid model that supports both thinking mode and non-thinking mode. Compared to the previous version, this upgrade brings improvements in multiple aspects:

- Hybrid thinking mode: One model supports both thinking mode and non-thinking mode by changing the chat template.

- Smarter tool calling: Through post-training optimization, the model's performance in tool usage and agent tasks has significantly improved.

- Higher thinking efficiency: DeepSeek-V3.1-Think achieves comparable answer quality to DeepSeek-R1-0528, while responding more quickly.

This document will show the main verification steps of the model, including supported features, feature configuration, environment preparation, single-node and multi-node deployment, accuracy and performance evaluation.

This document is validated and written based on **vLLM-Ascend v0.9.1rc3**. The current model (DeepSeek-V3.1) is first supported in this version.

## 2 Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [feature guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## 3 Prerequisites

### 3.1 Model Weight

- `DeepSeek-V3.1`(BF16 version): [Download model weight](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-V3.1).
- `DeepSeek-V3.1-w8a8-mtp-QuaRot`(Quantized version with mix mtp): [Download model weight](https://www.modelscope.cn/models/Eco-Tech/DeepSeek-V3.1-w8a8-mtp-QuaRot).
- `DeepSeek-V3.1-Terminus-w4a8-mtp-QuaRot`(Quantized version with mix mtp): [Download model weight](https://www.modelscope.cn/models/Eco-Tech/DeepSeek-V3.1-Terminus-w4a8-mtp-QuaRot).
- `Quantization method`: [msmodelslim](https://gitcode.com/Ascend/msit/blob/master/msmodelslim/example/DeepSeek/README.md#deepseek-v31-w8a8-%E6%B7%B7%E5%90%88%E9%87%8F%E5%8C%96-mtp-%E9%87%8F%E5%8C%96). You can use this method to quantize the model.

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`.

### 3.2 Verify Multi-node Communication (Optional)

If you want to deploy multi-node environment, you need to verify multi-node communication according to [verify multi-node communication environment](../../installation.md#verify-multi-node-communication).

## 4 Installation

### 4.1 Docker Image Installation

You can use our official docker image to run `DeepSeek-V3.1` directly.

Select an image based on your machine type and start the docker image on your node, refer to [using docker](../../installation.md#set-up-using-docker).

:::::{tab-set}
:sync-group: install

::::{tab-item} A3 series
:sync: A3

Start the docker image on your each node.

```{code-block} bash
   :substitutions:

export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|-a3
docker run --rm \
    --name vllm-ascend \
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
    -it $IMAGE bash
```

::::
::::{tab-item} A2 series
:sync: A2

Start the docker image on your each node.

```{code-block} bash
   :substitutions:

export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
docker run --rm \
    --name vllm-ascend \
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

::::
:::::

After a successful docker run, you can verify the running container service by executing the `docker ps` command.

### 4.2 Source Code Installation

If you don't want to use the docker image as above, you can also build all from source:

- Install `vllm-ascend` from source, refer to [installation](../../installation.md).

If you want to deploy multi-node environment, you need to set up environment on each node.

## 5 Online Service Deployment

### 5.1 Single-Node Online Deployment

Single-node deployment completes both Prefill and Decode within the same node. The quantized model `DeepSeek-V3.1-w8a8-mtp-QuaRot` can be deployed on 1 Atlas 800 A3 (64G × 16).

Startup Command:

```shell
#!/bin/sh
# this obtained through ifconfig
# nic_name is the network interface name corresponding to local_ip of the current node
nic_name="xxxx"
local_ip="xxxx"

# [Optional] jemalloc
# jemalloc is for better performance, if `libjemalloc.so` is installed on your machine, you can turn it on.
# export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD

# AIV
export HCCL_OP_EXPANSION_MODE="AIV"

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export VLLM_ASCEND_BALANCE_SCHEDULING=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

vllm serve /weights/DeepSeek-V3.1-w8a8-mtp-QuaRot \
--host 0.0.0.0 \
--port 8015 \
--data-parallel-size 4 \
--tensor-parallel-size 4 \
--quantization ascend \
--seed 1024 \
--served-model-name deepseek_v3 \
--enable-expert-parallel \
--max-num-seqs 16 \
--max-model-len 16384 \
--max-num-batched-tokens 4096 \
--trust-remote-code \
--no-enable-prefix-caching \
--gpu-memory-utilization 0.92 \
--speculative-config '{"num_speculative_tokens": 3, "method": "mtp"}' \
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
```

Key Parameter Descriptions:

- Setting the environment variable `VLLM_ASCEND_BALANCE_SCHEDULING=1` enables balance scheduling. This may help increase output throughput and reduce TPOT in v1 scheduler. However, TTFT may degrade in some scenarios. Furthermore, enabling this feature is not recommended in scenarios where PD is separated.
- For single-node deployment, we recommend using `dp4tp4` instead of `dp2tp8`.
- `--max-model-len` specifies the maximum context length - that is, the sum of input and output tokens for a single request. For performance testing with an input length of 3.5K and output length of 1.5K, a value of `16384` is sufficient, however, for precision testing, please set it at least `35000`.
- `--no-enable-prefix-caching` indicates that prefix caching is disabled. To enable it, remove this option.
- If you use the w4a8 weight, more memory will be allocated to kvcache, and you can try to increase system throughput to achieve greater throughput.

Common Issues Tip: If you encounter issues, please refer to the [Public FAQ](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html) for troubleshooting.

Service Verification:

```shell
curl http://<node_ip>:8015/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "deepseek_v3",
        "messages": [{
            "role": "user",
            "content": [
            {
                "type": "text",
                "text": "The future of AI is"
            }]
        }],
        "max_tokens": 1024,
        "temperature": 1.0,
        "top_p": 0.95
    }'
```

Expected Result:

The service returns HTTP 200 OK with a JSON response containing the `choices` field. Example output:

```json
{
    "id": "chatcmpl-xxxxxxxxxxxxx",
    "object": "chat.completion",
    "model": "deepseek_v3",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Of course. The future of AI is not a single..."
            },
            "finish_reason": "length"
        }
    ],
    "usage": {
        "prompt_tokens": 9,
        "total_tokens": 1033,
        "completion_tokens": 1024
    }
}
```

### 5.2 Multi-Node Data Parallel Deployment

- `DeepSeek-V3.1-w8a8-mtp-QuaRot`: require at least 2 Atlas 800 A2 (64G × 8).

Run the following scripts on two nodes respectively.

:::::{tab-set}
:sync-group: Deployment

::::{tab-item} Node 0
:sync: Node 0

Startup Command:

```{code-block} bash
   :substitutions:
#!/bin/sh

# this obtained through ifconfig
# nic_name is the network interface name corresponding to local_ip of the current node
nic_name="xxxx"
local_ip="xxxx"

# The value of node0_ip must be consistent with the value of local_ip set in node0 (master node)
node0_ip="xxxx"

# [Optional] jemalloc
# jemalloc is for better performance, if `libjemalloc.so` is installed on your machine, you can turn it on.
# export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_BALANCE_SCHEDULING=1
export HCCL_INTRA_PCIE_ENABLE=1
export HCCL_INTRA_ROCE_ENABLE=0

vllm serve /weights/DeepSeek-V3.1-w8a8-mtp-QuaRot \
--host 0.0.0.0 \
--port 8004 \
--data-parallel-size 4 \
--data-parallel-size-local 2 \
--data-parallel-address $node0_ip \
--data-parallel-rpc-port 13389 \
--tensor-parallel-size 4 \
--quantization ascend \
--seed 1024 \
--served-model-name deepseek_v3 \
--enable-expert-parallel \
--max-num-seqs 16 \
--max-model-len 16384 \
--max-num-batched-tokens 4096 \
--trust-remote-code \
--no-enable-prefix-caching \
--gpu-memory-utilization 0.92 \
--speculative-config '{"num_speculative_tokens": 3, "method": "mtp"}' \
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
```

::::
::::{tab-item} Node 1
:sync: Node 1

Startup Command:

```{code-block} bash
   :substitutions:
#!/bin/sh

# this obtained through ifconfig
# nic_name is the network interface name corresponding to local_ip of the current node
nic_name="xxx"
local_ip="xxx"

# The value of node0_ip must be consistent with the value of local_ip set in node0 (master node)
node0_ip="xxxx"

# [Optional] jemalloc
# jemalloc is for better performance, if `libjemalloc.so` is installed on your machine, you can turn it on.
# export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_BALANCE_SCHEDULING=1
export HCCL_INTRA_PCIE_ENABLE=1
export HCCL_INTRA_ROCE_ENABLE=0

vllm serve /weights/DeepSeek-V3.1-w8a8-mtp-QuaRot \
--host 0.0.0.0 \
--port 8004 \
--headless \
--data-parallel-size 4 \
--data-parallel-size-local 2 \
--data-parallel-start-rank 2 \
--data-parallel-address $node0_ip \
--data-parallel-rpc-port 13389 \
--tensor-parallel-size 4 \
--quantization ascend \
--seed 1024 \
--served-model-name deepseek_v3 \
--enable-expert-parallel \
--max-num-seqs 16 \
--max-model-len 16384 \
--max-num-batched-tokens 4096 \
--trust-remote-code \
--no-enable-prefix-caching \
--gpu-memory-utilization 0.92 \
--speculative-config '{"num_speculative_tokens": 3, "method": "mtp"}' \
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
```

::::
:::::

Key Parameter Descriptions:

- `--data-parallel-size`: total number of data parallel ranks across all nodes. In this example, `4` means the model is split across 4 DP ranks total (2 per node).
- `--data-parallel-size-local`: number of data parallel ranks running on the current node. In this example, each node runs 2 DP ranks.
- `--data-parallel-start-rank`: starting rank offset for data parallel ranks on this node. Node 0 starts at rank 0 (default), Node 1 starts at rank 2. This ensures each node's DP ranks occupy distinct positions in the overall rank space.
- `--data-parallel-address`: IP address of the data parallel master node (Node 0). This value must be consistent with `local_ip` set on Node 0.
- `--data-parallel-rpc-port`: RPC port for data parallel master communication. Must be the same across all nodes.
- `--headless`: indicates that this vLLM instance is not the master service node. Only set on non-master nodes (Node 1). The master node (Node 0) should NOT set this flag.
- For single-node deployment, we recommend using `dp4 tp4` instead of `dp2 tp8`.

Common Issues Tip: If you encounter issues, please refer to the [Public FAQ](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html) for troubleshooting.

Service Verification:

```shell
curl http://<node_ip>:8015/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "deepseek_v3",
        "messages": [{
            "role": "user",
            "content": [
            {
                "type": "text",
                "text": "The future of AI is"
            }]
        }],
        "max_tokens": 1024,
        "temperature": 1.0,
        "top_p": 0.95
    }'
```

Expected Result:

The service returns HTTP 200 OK. The JSON response contains the `choices` field with the generated text.

### 5.3 Multi-Node PD Separation Deployment

We recommend using Mooncake for deployment: [Mooncake](../features/pd_disaggregation_mooncake_multi_node.md).

In the standard single-node deployment mode, Prefill (prompt processing) and Decode (token generation) tasks run on the same set of NPUs. PD (Prefill-Decode) separation addresses this by running Prefill and Decode on dedicated node groups, each configured independently:

- **Prefill nodes** focus on high-throughput prompt processing, optimized for compute and communication.
- **Decode nodes** focus on low-latency token generation, optimized for memory bandwidth.

This architecture is recommended for production deployments with concurrent multi-user workloads, where stable latency and high throughput are both required.

Take Atlas 800 A3 (64G × 16) for example, we recommend to deploy 2P1D (4 nodes) rather than 1P1D (2 nodes), because there is no enough NPU memory to serve high concurrency in 1P1D case.

- `DeepSeek-V3.1-w8a8-mtp-QuaRot 2P1D Layerwise` require 4 Atlas 800 A3 (64G × 16).

To run the vllm-ascend `Prefill-Decode Disaggregation` service, you need to deploy a `launch_online_dp.py` script and a `run_dp_template.sh` script on each node and deploy a `proxy.sh` script on prefill master node to forward requests.

[launch_online_dp.py](https://github.com/vllm-project/vllm-ascend/blob/main/examples/external_online_dp/launch_online_dp.py)

Parameter descriptions:

|Parameter|Type|Required|Default|Description|
|---------|----|--------|-------|-----------|
|`--dp-size`|int|Yes|-|Data parallel size (total number of DP ranks across all nodes).|
|`--tp-size`|int|No|1|Tensor parallel size within each DP rank.|
|`--dp-size-local`|int|No|(same as `--dp-size`)|Number of DP ranks on the current node. If not set, defaults to `--dp-size`.|
|`--dp-rank-start`|int|No|0|Starting rank offset for data parallel ranks on this node.|
|`--dp-address`|str|Yes|-|IP address of the data parallel master node (node 0).|
|`--dp-rpc-port`|str|No|12345|RPC port for data parallel master communication.|
|`--vllm-start-port`|int|No|9000|Starting port for each vLLM engine instance on this node. Each DP rank's engine port = `vllm_start_port` + local rank index.|

1. `run_dp_template.sh` script

:::::{tab-set}
:sync-group: script

::::{tab-item} Node 0(Prefill)
:sync: Node 0(Prefill)

```{code-block} bash
    :substitutions:
# this obtained through ifconfig
# nic_name is the network interface name corresponding to local_ip of the current node
nic_name="xxx"
local_ip="141.xx.xx.1"

# The value of node0_ip must be consistent with the value of local_ip set in node0 (master node)
node0_ip="xxxx"

# [Optional] jemalloc
# jemalloc is for better performance, if `libjemalloc.so` is installed on your machine, you can turn it on.
# export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name

export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export HCCL_EXEC_TIMEOUT=204
export HCCL_CONNECT_TIMEOUT=120

export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=256
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_USE_V1=1
export ASCEND_RT_VISIBLE_DEVICES=$1
export ASCEND_BUFFER_POOL=4:8
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH

export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

vllm serve /weights/DeepSeek-V3.1-w8a8-mtp-QuaRot \
    --host 0.0.0.0 \
    --port $2 \
    --data-parallel-size $3 \
    --data-parallel-rank $4 \
    --data-parallel-address $5 \
    --data-parallel-rpc-port $6 \
    --tensor-parallel-size $7 \
    --enable-expert-parallel \
    --seed 1024 \
    --served-model-name deepseek_v3 \
    --max-model-len 65536 \
    --max-num-batched-tokens 16384 \
    --max-num-seqs 8 \
    --enforce-eager \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --quantization ascend \
    --no-enable-prefix-caching \
    --speculative-config '{"num_speculative_tokens": 1, "method": "mtp"}' \
    --kv-transfer-config \
    '{"kv_connector": "MooncakeConnectorV1",
    "kv_role": "kv_producer",
    "kv_port": "30000",
    "kv_connector_extra_config": {
            "prefill": {
                    "dp_size": 2,
                    "tp_size": 8
            },
            "decode": {
                    "dp_size": 32,
                    "tp_size": 1
            }
        }
    }'
```

::::
::::{tab-item} Node 1(Prefill)
:sync: Node 1(Prefill)

```{code-block} bash
    :substitutions:
# this obtained through ifconfig
# nic_name is the network interface name corresponding to local_ip of the current node
nic_name="xxx"
local_ip="141.xx.xx.2"

# The value of node0_ip must be consistent with the value of local_ip set in node0 (master node)
node0_ip="xxxx"

# [Optional] jemalloc
# jemalloc is for better performance, if `libjemalloc.so` is installed on your machine, you can turn it on.
# export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name

export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export HCCL_EXEC_TIMEOUT=204
export HCCL_CONNECT_TIMEOUT=120

export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=256
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_USE_V1=1
export ASCEND_RT_VISIBLE_DEVICES=$1
export ASCEND_BUFFER_POOL=4:8
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH

export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

vllm serve /weights/DeepSeek-V3.1-w8a8-mtp-QuaRot \
    --host 0.0.0.0 \
    --port $2 \
    --data-parallel-size $3 \
    --data-parallel-rank $4 \
    --data-parallel-address $5 \
    --data-parallel-rpc-port $6 \
    --tensor-parallel-size $7 \
    --enable-expert-parallel \
    --seed 1024 \
    --served-model-name deepseek_v3 \
    --max-model-len 65536 \
    --max-num-batched-tokens 16384 \
    --max-num-seqs 8 \
    --enforce-eager \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --quantization ascend \
    --no-enable-prefix-caching \
    --speculative-config '{"num_speculative_tokens": 1, "method": "mtp"}' \
    --kv-transfer-config \
    '{"kv_connector": "MooncakeConnectorV1",
    "kv_role": "kv_producer",
    "kv_port": "30100",
    "kv_connector_extra_config": {
            "prefill": {
                    "dp_size": 2,
                    "tp_size": 8
            },
            "decode": {
                    "dp_size": 32,
                    "tp_size": 1
            }
        }
    }'
```

::::
::::{tab-item} Node 0(Decode)
:sync: Node 0(Decode)

```{code-block} bash
    :substitutions:
# this obtained through ifconfig
# nic_name is the network interface name corresponding to local_ip of the current node
nic_name="xxx"
local_ip="141.xx.xx.3"

# The value of node0_ip must be consistent with the value of local_ip set in node0 (master node)
node0_ip="xxxx"

# [Optional] jemalloc
# jemalloc is for better performance, if `libjemalloc.so` is installed on your machine, you can turn it on.
# export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name

export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export HCCL_EXEC_TIMEOUT=204
export HCCL_CONNECT_TIMEOUT=120

export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1100
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_USE_V1=1
export ASCEND_RT_VISIBLE_DEVICES=$1
export ASCEND_BUFFER_POOL=4:8
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH

vllm serve /weights/DeepSeek-V3.1-w8a8-mtp-QuaRot \
    --host 0.0.0.0 \
    --port $2 \
    --data-parallel-size $3 \
    --data-parallel-rank $4 \
    --data-parallel-address $5 \
    --data-parallel-rpc-port $6 \
    --tensor-parallel-size $7 \
    --enable-expert-parallel \
    --seed 1024 \
    --served-model-name deepseek_v3 \
    --max-model-len 65536 \
    --max-num-batched-tokens 256 \
    --max-num-seqs 28 \
    --trust-remote-code \
    --gpu-memory-utilization 0.92 \
    --quantization ascend \
    --no-enable-prefix-caching \
    --speculative-config '{"num_speculative_tokens": 1, "method": "mtp"}' \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --additional-config '{"recompute_scheduler_enable":true,"multistream_overlap_shared_expert": true,"finegrained_tp_config": {"lmhead_tensor_parallel_size":16}}' \
    --kv-transfer-config \
    '{"kv_connector": "MooncakeConnectorV1",
    "kv_role": "kv_consumer",
    "kv_port": "30200",
    "kv_connector_extra_config": {
            "prefill": {
                    "dp_size": 2,
                    "tp_size": 8
            },
            "decode": {
                    "dp_size": 32,
                    "tp_size": 1
            }
        }
    }'
```

::::
::::{tab-item} Node 1(Decode)
:sync: Node 1(Decode)

```{code-block} bash
   :substitutions:
# this obtained through ifconfig
# nic_name is the network interface name corresponding to local_ip of the current node
nic_name="xxx"
local_ip="141.xx.xx.4"

# The value of node0_ip must be consistent with the value of local_ip set in node0 (master node)
node0_ip="xxxx"

# [Optional] jemalloc
# jemalloc is for better performance, if `libjemalloc.so` is installed on your machine, you can turn it on.
# export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name

export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export HCCL_EXEC_TIMEOUT=204
export HCCL_CONNECT_TIMEOUT=120

export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1100
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_USE_V1=1
export ASCEND_RT_VISIBLE_DEVICES=$1
export ASCEND_BUFFER_POOL=4:8
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH

vllm serve /weights/DeepSeek-V3.1-w8a8-mtp-QuaRot \
    --host 0.0.0.0 \
    --port $2 \
    --data-parallel-size $3 \
    --data-parallel-rank $4 \
    --data-parallel-address $5 \
    --data-parallel-rpc-port $6 \
    --tensor-parallel-size $7 \
    --enable-expert-parallel \
    --seed 1024 \
    --served-model-name deepseek_v3 \
    --max-model-len 65536 \
    --max-num-batched-tokens 256 \
    --max-num-seqs 28 \
    --trust-remote-code \
    --gpu-memory-utilization 0.92 \
    --quantization ascend \
    --no-enable-prefix-caching \
    --speculative-config '{"num_speculative_tokens": 1, "method": "mtp"}' \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --additional-config '{"recompute_scheduler_enable":true,"multistream_overlap_shared_expert": true,"finegrained_tp_config": {"lmhead_tensor_parallel_size":16}}' \
    --kv-transfer-config \
    '{"kv_connector": "MooncakeConnectorV1",
    "kv_role": "kv_consumer",
    "kv_port": "30200",
    "kv_connector_extra_config": {
            "prefill": {
                    "dp_size": 2,
                    "tp_size": 8
            },
            "decode": {
                    "dp_size": 32,
                    "tp_size": 1
            }
        }
    }'
```

::::
:::::

Key Parameter Descriptions:

- `VLLM_ASCEND_ENABLE_FLASHCOMM1=1`: enables the communication optimization function on the prefill nodes.
- `VLLM_ASCEND_ENABLE_MLAPO=1`: enables the fusion operator, which can significantly improve performance but consumes more NPU memory. In the Prefill-Decode (PD) separation scenario, enable MLAPO only on decode nodes.
- `recompute_scheduler_enable: true`: enables the recomputation scheduler. When the Key-Value Cache (KV Cache) of the decode node is insufficient, requests will be sent to the prefill node to recompute the KV Cache. In the PD separation scenario, enable this configuration only on decode nodes.
- `multistream_overlap_shared_expert: true`: When the Tensor Parallelism (TP) size is 1 or `enable_shared_expert_dp: true`, an additional stream is enabled to overlap the computation process of shared experts for improved efficiency.
- `lmhead_tensor_parallel_size: 16`: When the Tensor Parallelism (TP) size of the decode node is 1, this parameter allows the TP size of the LMHead embedding layer to be greater than 1, which is used to reduce the computational load of each card on the LMHead embedding layer.

2. run server for each node:

    ```shell
    # p0
    python launch_online_dp.py --dp-size 2 --tp-size 8 --dp-size-local 2 --dp-rank-start 0 --dp-address 141.xx.xx.1 --dp-rpc-port 12321 --vllm-start-port 7100
    # p1
    python launch_online_dp.py --dp-size 2 --tp-size 8 --dp-size-local 2 --dp-rank-start 0 --dp-address 141.xx.xx.2 --dp-rpc-port 12321 --vllm-start-port 7100
    # d0
    python launch_online_dp.py --dp-size 32 --tp-size 1 --dp-size-local 16 --dp-rank-start 0 --dp-address 141.xx.xx.3 --dp-rpc-port 12321 --vllm-start-port 7100
    # d1
    python launch_online_dp.py --dp-size 32 --tp-size 1 --dp-size-local 16 --dp-rank-start 16 --dp-address 141.xx.xx.3 --dp-rpc-port 12321 --vllm-start-port 7100
    ```

3. Run the `proxy.sh` script on the prefill master node

    Run a proxy server on the same node with the prefiller service instance. You can get the proxy program in the repository's examples: [load\_balance\_proxy\_server\_example.py](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py)

    ```shell
    python load_balance_proxy_server_example.py \
      --port 1999 \
      --host 141.xx.xx.1 \
      --prefiller-hosts \
        141.xx.xx.1 \
        141.xx.xx.1 \
        141.xx.xx.2 \
        141.xx.xx.2 \
      --prefiller-ports \
        7100 7101 7100 7101 \
      --decoder-hosts \
        141.xx.xx.3 \
        141.xx.xx.3 \
        141.xx.xx.3 \
        141.xx.xx.3 \
        141.xx.xx.3 \
        141.xx.xx.3 \
        141.xx.xx.3 \
        141.xx.xx.3 \
        141.xx.xx.3 \
        141.xx.xx.3 \
        141.xx.xx.3 \
        141.xx.xx.3 \
        141.xx.xx.3 \
        141.xx.xx.3 \
        141.xx.xx.3 \
        141.xx.xx.3 \
        141.xx.xx.4 \
        141.xx.xx.4 \
        141.xx.xx.4 \
        141.xx.xx.4 \
        141.xx.xx.4 \
        141.xx.xx.4 \
        141.xx.xx.4 \
        141.xx.xx.4 \
        141.xx.xx.4 \
        141.xx.xx.4 \
        141.xx.xx.4 \
        141.xx.xx.4 \
        141.xx.xx.4 \
        141.xx.xx.4 \
        141.xx.xx.4 \
        141.xx.xx.4 \
      --decoder-ports \
        7100 7101 7102 7103 7104 7105 7106 7107 7108 7109 7110 7111 7112 7113 7114 7115 \
        7100 7101 7102 7103 7104 7105 7106 7107 7108 7109 7110 7111 7112 7113 7114 7115 \
    ```

    ```shell
    cd vllm-ascend/examples/disaggregated_prefill_v1/
    bash proxy.sh
    ```

Deployment Verification:

After the PD separation service is fully started, send a request through the proxy port on the prefill master node to verify that Prefill and Decode nodes are working correctly together:

```shell
curl http://141.xx.xx.1:1999/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "deepseek_v3",
        "messages": [{
            "role": "user",
            "content": [
            {
                "type": "text",
                "text": "The future of AI is"
            }]
        }],
        "max_tokens": 1024,
        "temperature": 1.0,
        "top_p": 0.95
    }'
```

Expected Result:

The proxy returns HTTP 200 OK. The JSON response contains the `choices` field with the generated text, confirming that Prefill nodes have successfully processed the prompt and Decode nodes have generated the response:

```json
{
    "id": "chatcmpl-xxxxxxxxxxxxx",
    "object": "chat.completion",
    "model": "deepseek_v3",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "The future of AI is not a destination we are passively approaching...",
                "finish_reason": "length"
            }
        }
    ],
    "usage": {
        "prompt_tokens": 13,
        "total_tokens": 1037,
        "completion_tokens": 1024
    }
}
```

Common Issues Tip: If you encounter issues with PD separation deployment, please refer to the [Public FAQ](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html) for troubleshooting.

## 6 Functional Verification

Once your server is started, you can query the model with input prompts:

```shell
curl http://<node0_ip>:<port>/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "deepseek_v3",
        "prompt": "The future of AI is",
        "max_completion_tokens": 50,
        "temperature": 0
    }'
```

## 7 Accuracy Evaluation

Here is one accuracy evaluation method.

### Using AISBench

1. Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) for details.

2. After execution, you can get the result, here is the result of `DeepSeek-V3.1-w8a8-mtp-QuaRot` in `vllm-ascend:0.18.0` for reference only.

| dataset | version | metric | mode | vllm-api-general-chat | note |
|----- | ----- | ----- | ----- | -----| ----- |
| ceval | - | accuracy | gen | 90.94 | 1 Atlas 800 A3 (64G × 16) |
| gsm8k | - | accuracy | gen | 96.28 | 1 Atlas 800 A3 (64G × 16) |

### Using Language Model Evaluation Harness

Not test yet.

## 8 Performance Evaluation

### Using AISBench

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

The performance result is:  

**Hardware**: A3-752T, 4 node

**Deployment**: 2P1D, Prefill node: DP2+TP8, Decode Node: DP32+TP1

**Input/Output**: 3.5k/1.5k

**Performance**: TTFT = 6.16s, TPOT = 48.82ms, Average performance of each card is 478 TPS (Token Per Second).

### Using vLLM Benchmark

Run performance evaluation of `DeepSeek-V3.1-w8a8-mtp-QuaRot` as an example.

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/benchmarking/) for more details.

There are three `vllm bench` subcommands:

- `latency`: Benchmark the latency of a single batch of requests.
- `serve`: Benchmark the online serving throughput.
- `throughput`: Benchmark offline inference throughput.

Take the `serve` as an example. Run the code as follows.

```shell
vllm bench serve --model /weights/DeepSeek-V3.1-w8a8-mtp-QuaRot  --dataset-name random --random-input 1024 --num-prompts 200 --request-rate 1 --save-result --result-dir ./
```

After about several minutes, you can get the performance evaluation result.

## 9 Performance Tuning

### 9.1 Recommended Configurations

> **Note**: The following configurations are validated in specific test environments and are for reference only. The optimal configuration depends on factors such as maximum input/output length, prefix cache hit rate, precision requirements, and deployment machine ratios. It is recommended to refer to Section 9.2 for tuning based on actual conditions.

#### Table 1: Scenario Overview

> `*Total NPUs` indicates the total number of NPUs used across all nodes. 1 node = 1 Atlas 800 A3 server (64G × 16 NPUs).

|Scenario|Deployment Mode|*Total NPUs|Weight Version|Key Considerations|
|--------|---------------|-----------|--------------|------------------|
|High Throughput<br>(3.5K/16K input)|Single-Node Mixed|16 (A3)|DeepSeek-V3.1-w4a8-perchannle|Use dp4 tp4 to balance memory capacity and compute efficiency|
|Low Latency<br>(3.5K/16K input)|Single-Node Mixed|16 (A3)|DeepSeek-V3.1-w4a8-perchannle|Use dp2 tp8 to balance memory capacity and compute efficiency|
|High Throughput / Low Latency<br>(64K input)|Single-Node Mixed|16 (A3)|DeepSeek-V3.1-w4a8-perchannle|Use dp2 tp8 to balance memory capacity and compute efficiency|
|High Throughput / Low Latency<br>(3.5K input)|2P1D deployment|64 (A3)|DeepSeek-V3.1-w4a8-perchannle|Use dp2 tp8 to balance memory capacity and compute efficiency|
|High Throughput / Low Latency<br>(16K input)|2P1D deployment|64 (A3)|DeepSeek-V3.1-w4a8-perchannle|Use dp2 tp8 to balance memory capacity and compute efficiency|
|Long Context<br>(64K input, no prefix cache)|2P1D deployment|64 (A3)|DeepSeek-V3.1-w4a8-perchannle|Use dp1 tp8 to balance memory capacity and compute efficiency|

#### Table 2: Detailed Node Configuration

|Scenario|Configuration|NPUs|TP|DP|Max Model Len|MTP Speculation Num|
|--------|-------------|-----|--|--|-------------------|--------------------|
|High Throughput (3.5K)|Server / Single Machine|16|4|4|39K|3|
|High Throughput (16K)|Server / Single Machine|16|4|4|36K|3|
|Low Latency (3.5K)|Server / Single Machine|16|8|2|36K|3|
|Low Latency (16K)|Server / Single Machine|16|8|2|36K|3|
|High Throughput / Low Latency (64K)|Server / Single Machine|16|8|2|132K|3|

|High Throughput (16K)|Server-P Node|16|8|2|36K|1|
|High Throughput (16K)|Server-D Node|16|4|8|36K|1|
|Low Latency (16K)|Server-P Node|16|8|2|36K|3|
|Low Latency (16K)|Server-D Node|16|4|8|36K|3|
|Long Context (64K)|Server-P Node|16|8|1(PCP2DCP8)|36K|3|
|Long Context (64K)|Server-D Node|16|4|8|36K|3|

> For complete startup commands and parameter descriptions, please refer to the deployment examples in [Chapter 5](#5-online-service-deployment).

**Notice:**
`max-model-len` and `max-num-seqs` need to be set according to the actual usage scenario. For other settings, please refer to the **[Deployment](#5-online-service-deployment)** chapter.

### 9.2 Tuning Guidelines

Please refer to the [Public Performance Tuning Documentation](../../developer_guide/performance_and_debug/optimization_and_tuning.md) for tuning methods.

Please refer to the [Feature Guide](../../user_guide/support_matrix/feature_matrix.md) for detailed feature descriptions.

## 10 FAQ

For common environment, installation, and general parameter issues, please refer to the [Public FAQ](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html).
