# Kimi-K2.5

## 1 Introduction

Kimi K2.5 is an open-source, native multimodal agentic model built through continual pretraining on approximately 15 trillion mixed visual and text tokens atop Kimi-K2-Base. It seamlessly integrates vision and language understanding with advanced agentic capabilities, instant and thinking modes, as well as conversational and agentic paradigms.

This document will show the main verification steps of the model, including supported features, feature configuration, environment preparation, single-node and multi-node deployment, accuracy and performance evaluation.

This document is validated and written based on **vLLM-Ascend v0.17.0rc1**. The current model (Kimi-K2.5) is first supported in this version, and **v0.17.0rc1 and later versions** can run stably.

## 2 Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [feature guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## 3 Prerequisites

### 3.1 Model Weight

- `Kimi-K2.5-w4a8` (Quantized version for w4a8): requires 1 Atlas 800 A3 (64G × 16) node or 2 Atlas 800 A2 (64G × 8) nodes. [Download model weight](https://modelscope.cn/models/Eco-Tech/Kimi-K2.5-W4A8).
- `kimi-k2.5-eagle3` (Eagle3 MTP draft model for accelerating inference of Kimi-K2.5): [Download model weight](https://huggingface.co/lightseekorg/kimi-k2.5-eagle3)

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`.

### 3.2 Verify Multi-node Communication (Optional)

If you want to deploy multi-node environment, you need to verify multi-node communication according to [verify multi-node communication environment](../../installation.md#verify-multi-node-communication).

## 4 Installation

### 4.1 Docker Image Installation

Select an image based on your machine type and start the docker image on your node, refer to [using docker](../../installation.md#set-up-using-docker).

=== "A3 series"

    Start the docker image on your each node.

    ```bash

    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-a3
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

=== "A2 series"

    Start the docker image on your each node.

    ```bash

    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}
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

After a successful docker run, you can verify the running container service by executing the `docker ps` command.

### 4.2 Source Code Installation

If you don't want to use the docker image as above, you can also build all from source:

- Install `vllm-ascend` from source, refer to [installation](../../installation.md).

If you want to deploy multi-node environment, you need to set up environment on each node.

## 5 Online Service Deployment

### 5.1 Single-Node Online Deployment

Single-node deployment completes both Prefill and Decode within the same node. The quantized model `Kimi-K2.5-w4a8` can be deployed on 1 Atlas 800 A3 (64G × 16).

Run the following script to execute online inference.

Startup Command:

```bash
#!/bin/sh
# [Optional] jemalloc
# jemalloc is for better performance, if `libjemalloc.so` is installed on your machine, you can turn it on.
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD

echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000
export HCCL_OP_EXPANSION_MODE="AIV"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1

export HCCL_BUFFSIZE=800
export VLLM_ASCEND_ENABLE_MLAPO=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_BALANCE_SCHEDULING=1

vllm serve Eco-Tech/Kimi-K2.5-W4A8 \
  --host 0.0.0.0 \
  --port 8088 \
  --quantization ascend \
  --served-model-name kimi_k25 \
  --allowed-local-media-path / \
  --trust-remote-code \
  --no-enable-prefix-caching \
  --seed 1024 \
  --tensor-parallel-size 4 \
  --data-parallel-size 4 \
  --enable-expert-parallel \
  --max-num-seqs 64 \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.9 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --speculative-config '{"method":"eagle3", "model":"lightseekorg/kimi-k2.5-eagle3", "num_speculative_tokens":3}' \
  --mm-encoder-tp-mode data
```

Key Parameter Descriptions:

- Setting the environment variable `VLLM_ASCEND_BALANCE_SCHEDULING=1` enables balance scheduling. This may help increase output throughput and reduce TPOT in v1 scheduler. However, TTFT may degrade in some scenarios. Furthermore, enabling this feature is not recommended in scenarios where PD is separated.
- For single-node deployment, we recommend using `dp4 tp4` instead of `dp2 tp8`.
- `--max-model-len` specifies the maximum context length - that is, the sum of input and output tokens for a single request. For performance testing with an input length of 3.5K and output length of 1.5K, a value of `16384` is sufficient, however, for precision testing, please set it at least `35000`.
- `--no-enable-prefix-caching` indicates that prefix caching is disabled. To enable it, remove this option.
- `--mm-encoder-tp-mode` indicates how to optimize multi-modal encoder inference using tensor parallelism (TP). If you want to test the multimodal inputs, we recommend using `data`.
- If you use the w4a8 weight, more memory will be allocated to kvcache, and you can try to increase system throughput to achieve greater throughput.

Common Issues Tip: If you encounter issues, please refer to the [Public FAQ](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html) for troubleshooting.

Service Verification:

```shell
curl http://<node_ip>:8088/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "kimi_k25",
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
    "model": "kimi_k25",
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

### 5.2 Multi-Node Data Parallel Deployment

`Kimi-K2.5-w4a8` can be deployed across multiple nodes using data parallelism. This deployment mode requires at least 2 Atlas 800 A2 (64G × 8) nodes.

Run the following scripts on two nodes respectively.

**Node 0**

Startup Command:

```bash
#!/bin/sh

# this obtained through ifconfig
# nic_name is the network interface name corresponding to local_ip of the current node
nic_name="xxxx"
local_ip="141.xx.xx.1"

# The value of node0_ip must be consistent with the value of local_ip set in node0 (master node)
node0_ip="xxxx"

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export HCCL_INTRA_PCIE_ENABLE=1
export HCCL_INTRA_ROCE_ENABLE=0

# [Optional] jemalloc
# jemalloc is for better performance, if `libjemalloc.so` is installed on your machine, you can turn it on.
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD

echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000
export HCCL_OP_EXPANSION_MODE="AIV"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1

export HCCL_BUFFSIZE=1024
export VLLM_ASCEND_ENABLE_MLAPO=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_BALANCE_SCHEDULING=1

vllm serve Eco-Tech/Kimi-K2.5-W4A8 \
  --host 0.0.0.0 \
  --port 8088 \
  --quantization ascend \
  --served-model-name kimi_k25 \
  --allowed-local-media-path / \
  --trust-remote-code \
  --no-enable-prefix-caching \
  --seed 1024 \
  --data-parallel-size 4 \
  --data-parallel-size-local 2 \
  --data-parallel-address $node0_ip \
  --data-parallel-rpc-port 13389 \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --max-num-seqs 16 \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.9 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --speculative-config '{"method":"eagle3", "model":"lightseekorg/kimi-k2.5-eagle3", "num_speculative_tokens":3}' \
  --mm-encoder-tp-mode data
```

**Node 1**

Startup Command:

```bash
#!/bin/sh

# this obtained through ifconfig
# nic_name is the network interface name corresponding to local_ip of the current node
nic_name="xxx"
local_ip="141.xx.xx.2"

# The value of node0_ip must be consistent with the value of local_ip set in node0 (master node)
node0_ip="xxxx"

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export HCCL_INTRA_PCIE_ENABLE=1
export HCCL_INTRA_ROCE_ENABLE=0

# [Optional] jemalloc
# jemalloc is for better performance, if `libjemalloc.so` is installed on your machine, you can turn it on.
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD

echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000
export HCCL_OP_EXPANSION_MODE="AIV"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1

export HCCL_BUFFSIZE=1024
export VLLM_ASCEND_ENABLE_MLAPO=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_BALANCE_SCHEDULING=1

vllm serve Eco-Tech/Kimi-K2.5-W4A8 \
  --host 0.0.0.0 \
  --port 8088 \
  --quantization ascend \
  --served-model-name kimi_k25 \
  --allowed-local-media-path / \
  --trust-remote-code \
  --no-enable-prefix-caching \
  --seed 1024 \
  --headless \
  --data-parallel-size 4 \
  --data-parallel-size-local 2 \
  --data-parallel-start-rank 2 \
  --data-parallel-address $node0_ip \
  --data-parallel-rpc-port 13389 \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --max-num-seqs 16 \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.9 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --speculative-config '{"method":"eagle3", "model":"lightseekorg/kimi-k2.5-eagle3", "num_speculative_tokens":3}' \
  --mm-encoder-tp-mode data
```

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
curl http://<node0_ip>:8088/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "kimi_k25",
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

- **Prefill nodes** focus on high-throughput prompt processing, optimized for compute and communication (e.g., enabling FlashComm for Allreduce acceleration).
- **Decode nodes** focus on low-latency token generation, optimized for memory bandwidth (e.g., enabling MLAPO fusion operators).

This architecture is recommended for production deployments with concurrent multi-user workloads, where stable latency and high throughput are both required.

Take Atlas 800 A3 (64G × 16) for example, we recommend to deploy 2P1D (4 nodes) rather than 1P1D (2 nodes), because there is not enough NPU memory to serve high concurrency in 1P1D case.

- `Kimi-K2.5-w4a8 2P1D`: requires 4 Atlas 800 A3 (64G × 16) nodes.

To run the vllm-ascend `Prefill-Decode Disaggregation` service, you need to deploy a `launch_online_dp.py` script and a `run_dp_template.sh` script on each node and deploy a `proxy.sh` script on prefill master node to forward requests.

1. `launch_online_dp.py` to launch external dp vllm servers.
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

2. Prefill Node 0 `run_dp_template.sh` script

    ```shell
    # this obtained through ifconfig
    # nic_name is the network interface name corresponding to local_ip of the current node
    nic_name="xxx"
    local_ip="141.xx.xx.1"

    # The value of node0_ip must be consistent with the value of local_ip set in node0 (master node)
    node0_ip="xxxx"

    export HCCL_IF_IP=$local_ip
    export GLOO_SOCKET_IFNAME=$nic_name
    export TP_SOCKET_IFNAME=$nic_name
    export HCCL_SOCKET_IFNAME=$nic_name

    # [Optional] jemalloc
    # jemalloc is for better performance, if `libjemalloc.so` is installed on your machine, you can turn it on.
    export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
    echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
    sysctl -w vm.swappiness=0
    sysctl -w kernel.numa_balancing=0
    sysctl kernel.sched_migration_cost_ns=50000
    export VLLM_RPC_TIMEOUT=3600000
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000

    export HCCL_OP_EXPANSION_MODE="AIV"
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=1
    export TASK_QUEUE_ENABLE=1
    export ASCEND_BUFFER_POOL=4:8
    export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH

    export HCCL_BUFFSIZE=256
    export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
    export ASCEND_RT_VISIBLE_DEVICES=$1

    vllm serve Eco-Tech/Kimi-K2.5-W4A8 \
      --host 0.0.0.0 \
      --port $2 \
      --data-parallel-size $3 \
      --data-parallel-rank $4 \
      --data-parallel-address $5 \
      --data-parallel-rpc-port $6 \
      --tensor-parallel-size $7 \
      --enable-expert-parallel \
      --seed 1024 \
      --quantization ascend \
      --served-model-name kimi_k25 \
      --trust-remote-code \
      --max-num-seqs 8 \
      --max-model-len 32768 \
      --max-num-batched-tokens 16384 \
      --no-enable-prefix-caching \
      --gpu-memory-utilization 0.8 \
      --enforce-eager \
      --speculative-config '{"method": "eagle3", "model":"lightseekorg/kimi-k2.5-eagle3", "num_speculative_tokens": 3}' \
      --mm-encoder-tp-mode data \
      --kv-transfer-config \
      '{"kv_connector": "MooncakeConnectorV1",
      "kv_role": "kv_producer",
      "kv_port": "30000",
      "engine_id": "0",
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

3. Prefill Node 1 `run_dp_template.sh` script

    ```shell
    # this obtained through ifconfig
    # nic_name is the network interface name corresponding to local_ip of the current node
    nic_name="xxx"
    local_ip="141.xx.xx.2"

    # The value of node0_ip must be consistent with the value of local_ip set in node0 (master node)
    node0_ip="xxxx"

    export HCCL_IF_IP=$local_ip
    export GLOO_SOCKET_IFNAME=$nic_name
    export TP_SOCKET_IFNAME=$nic_name
    export HCCL_SOCKET_IFNAME=$nic_name

    # [Optional] jemalloc
    # jemalloc is for better performance, if `libjemalloc.so` is installed on your machine, you can turn it on.
    export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
    echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
    sysctl -w vm.swappiness=0
    sysctl -w kernel.numa_balancing=0
    sysctl kernel.sched_migration_cost_ns=50000
    export VLLM_RPC_TIMEOUT=3600000
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000

    export HCCL_OP_EXPANSION_MODE="AIV"
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=1
    export TASK_QUEUE_ENABLE=1
    export ASCEND_BUFFER_POOL=4:8
    export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH

    export HCCL_BUFFSIZE=256
    export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
    export ASCEND_RT_VISIBLE_DEVICES=$1

    vllm serve Eco-Tech/Kimi-K2.5-W4A8 \
      --host 0.0.0.0 \
      --port $2 \
      --data-parallel-size $3 \
      --data-parallel-rank $4 \
      --data-parallel-address $5 \
      --data-parallel-rpc-port $6 \
      --tensor-parallel-size $7 \
      --enable-expert-parallel \
      --seed 1024 \
      --quantization ascend \
      --served-model-name kimi_k25 \
      --trust-remote-code \
      --max-num-seqs 8 \
      --max-model-len 32768 \
      --max-num-batched-tokens 16384 \
      --no-enable-prefix-caching \
      --gpu-memory-utilization 0.8 \
      --enforce-eager \
      --speculative-config '{"method": "eagle3", "model":"lightseekorg/kimi-k2.5-eagle3", "num_speculative_tokens": 3}' \
      --mm-encoder-tp-mode data \
      --kv-transfer-config \
      '{"kv_connector": "MooncakeConnectorV1",
      "kv_role": "kv_producer",
      "kv_port": "30100",
      "engine_id": "1",
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

4. Decode Node 0 `run_dp_template.sh` script

    ```shell
    # this obtained through ifconfig
    # nic_name is the network interface name corresponding to local_ip of the current node
    nic_name="xxx"
    local_ip="141.xx.xx.3"

    # The value of node0_ip must be consistent with the value of local_ip set in node0 (master node)
    node0_ip="xxxx"

    export HCCL_IF_IP=$local_ip
    export GLOO_SOCKET_IFNAME=$nic_name
    export TP_SOCKET_IFNAME=$nic_name
    export HCCL_SOCKET_IFNAME=$nic_name

    # [Optional] jemalloc
    # jemalloc is for better performance, if `libjemalloc.so` is installed on your machine, you can turn it on.
    export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
    echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
    sysctl -w vm.swappiness=0
    sysctl -w kernel.numa_balancing=0
    sysctl kernel.sched_migration_cost_ns=50000
    export VLLM_RPC_TIMEOUT=3600000
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000

    export HCCL_OP_EXPANSION_MODE="AIV"
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=1
    export TASK_QUEUE_ENABLE=1
    export ASCEND_BUFFER_POOL=4:8
    export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH

    export HCCL_BUFFSIZE=1100
    export VLLM_ASCEND_ENABLE_MLAPO=1
    export ASCEND_RT_VISIBLE_DEVICES=$1

    vllm serve Eco-Tech/Kimi-K2.5-W4A8 \
      --host 0.0.0.0 \
      --port $2 \
      --data-parallel-size $3 \
      --data-parallel-rank $4 \
      --data-parallel-address $5 \
      --data-parallel-rpc-port $6 \
      --tensor-parallel-size $7 \
      --enable-expert-parallel \
      --seed 1024 \
      --quantization ascend \
      --served-model-name kimi_k25 \
      --trust-remote-code \
      --max-num-seqs 48 \
      --max-model-len 32768 \
      --max-num-batched-tokens 256 \
      --no-enable-prefix-caching \
      --gpu-memory-utilization 0.95 \
      --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
      --additional-config '{"recompute_scheduler_enable":true,"multistream_overlap_shared_expert": false}' \
      --speculative-config '{"method": "eagle3", "model":"lightseekorg/kimi-k2.5-eagle3", "num_speculative_tokens": 3}' \
      --kv-transfer-config \
      '{"kv_connector": "MooncakeConnectorV1",
      "kv_role": "kv_consumer",
      "kv_port": "30200",
      "engine_id": "2",
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

5. Decode Node 1 `run_dp_template.sh` script

    ```shell
    # this obtained through ifconfig
    # nic_name is the network interface name corresponding to local_ip of the current node
    nic_name="xxx"
    local_ip="141.xx.xx.4"

    # The value of node0_ip must be consistent with the value of local_ip set in node0 (master node)
    node0_ip="xxxx"

    export HCCL_IF_IP=$local_ip
    export GLOO_SOCKET_IFNAME=$nic_name
    export TP_SOCKET_IFNAME=$nic_name
    export HCCL_SOCKET_IFNAME=$nic_name

    # [Optional] jemalloc
    # jemalloc is for better performance, if `libjemalloc.so` is installed on your machine, you can turn it on.
    export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
    echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
    sysctl -w vm.swappiness=0
    sysctl -w kernel.numa_balancing=0
    sysctl kernel.sched_migration_cost_ns=50000
    export VLLM_RPC_TIMEOUT=3600000
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000

    export HCCL_OP_EXPANSION_MODE="AIV"
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=1
    export TASK_QUEUE_ENABLE=1
    export ASCEND_BUFFER_POOL=4:8
    export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH

    export HCCL_BUFFSIZE=1100
    export VLLM_ASCEND_ENABLE_MLAPO=1
    export ASCEND_RT_VISIBLE_DEVICES=$1

    vllm serve Eco-Tech/Kimi-K2.5-W4A8 \
      --host 0.0.0.0 \
      --port $2 \
      --data-parallel-size $3 \
      --data-parallel-rank $4 \
      --data-parallel-address $5 \
      --data-parallel-rpc-port $6 \
      --tensor-parallel-size $7 \
      --enable-expert-parallel \
      --seed 1024 \
      --quantization ascend \
      --served-model-name kimi_k25 \
      --trust-remote-code \
      --max-num-seqs 48 \
      --max-model-len 32768 \
      --max-num-batched-tokens 256 \
      --no-enable-prefix-caching \
      --gpu-memory-utilization 0.95 \
      --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
      --additional-config '{"recompute_scheduler_enable":true,"multistream_overlap_shared_expert": false}' \
      --speculative-config '{"method": "eagle3", "model":"lightseekorg/kimi-k2.5-eagle3", "num_speculative_tokens": 3}' \
      --kv-transfer-config \
      '{"kv_connector": "MooncakeConnectorV1",
      "kv_role": "kv_consumer",
      "kv_port": "30200",
      "engine_id": "2",
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

Key Parameter Descriptions:

- `VLLM_ASCEND_ENABLE_FLASHCOMM1=1`: enables the communication optimization function on the prefill nodes.
- `VLLM_ASCEND_ENABLE_MLAPO=1`: enables the fusion operator, which can significantly improve performance but consumes more NPU memory. In the Prefill-Decode (PD) separation scenario, enable MLAPO only on decode nodes.
- `recompute_scheduler_enable: true`: enables the recomputation scheduler. When the Key-Value Cache (KV Cache) of the decode node is insufficient, requests will be sent to the prefill node to recompute the KV Cache. In the PD separation scenario, enable this configuration only on decode nodes.
- `multistream_overlap_shared_expert: true`: When the Tensor Parallelism (TP) size is 1 or `enable_shared_expert_dp: true`, an additional stream is enabled to overlap the computation process of shared experts for improved efficiency.

The `run_dp_template.sh` scripts use positional parameters (`$1`-`$7`) to receive configuration values from `launch_online_dp.py`:

- `$1` (`ASCEND_RT_VISIBLE_DEVICES`): the NPU devices assigned to this DP instance, e.g., `0,1,2,3` or `4,5,6,7`.
- `$2` (`--port`): the vLLM server port for this DP instance, auto-assigned starting from `--vllm-start-port` (e.g., `7100`, `7101`).
- `$3` (`--data-parallel-size`): total number of DP ranks.
- `$4` (`--data-parallel-rank`): the rank index of this DP instance.
- `$5` (`--data-parallel-address`): IP address of the DP master node.
- `$6` (`--data-parallel-rpc-port`): RPC port for DP master communication.
- `$7` (`--tensor-parallel-size`): TP size within each DP rank.

6. Run server for each node:

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

7. Run the `proxy.sh` script on the prefill master node

    Run a proxy server on the same node with the prefiller service instance. You can get the proxy program in the repository's examples: [load_balance_proxy_server_example.py](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py)

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
        "model": "kimi_k25",
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
    "model": "kimi_k25",
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
curl http://<node0_ip>:8088/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "kimi_k25",
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

The service returns HTTP 200 OK. The JSON response contains the `choices` field with the generated text, along with usage statistics:

```json
{
    "id": "chatcmpl-xxxxxxxxxxxxx",
    "object": "chat.completion",
    "model": "kimi_k25",
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

## 7 Accuracy Evaluation

Here is one accuracy evaluation method.

### Using AISBench

1. Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) for details.

2. After execution, you can get the result. Here is the result of `Kimi-K2.5-w4a8` in `vllm-ascend:v0.17.0rc1` for reference only.

| dataset | version | metric | mode | vllm-api-general-chat | note |
| ----- | ----- | ----- | ----- | ----- | ----- |
| GSM8K | - | accuracy | gen | 96.07 | 1 Atlas 800 A3 (64G × 16) |
| AIME2025 | - | accuracy | gen | 90.00 | 1 Atlas 800 A3 (64G × 16) |
| GPQA | - | accuracy | gen | 84.85 | 1 Atlas 800 A3 (64G × 16) |
| TextVQA | - | accuracy | gen | 80.29 | 1 Atlas 800 A3 (64G × 16) |

## 8 Performance Evaluation

### Using AISBench

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

### Using vLLM Benchmark

Run performance evaluation of `Kimi-K2.5-w4a8` as an example.

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/benchmarking/) for more details.

There are three `vllm bench` subcommands:

- `latency`: Benchmark the latency of a single batch of requests.
- `serve`: Benchmark the online serving throughput.
- `throughput`: Benchmark offline inference throughput.

Take the `serve` as an example. Run the code as follows.

```shell
vllm bench serve --model Eco-Tech/Kimi-K2.5-w4a8 --dataset-name random --random-input 1024 --num-prompts 200 --request-rate 1 --save-result --result-dir ./
```

After about several minutes, you can get the performance evaluation result.

## 9 Performance Tuning

### 9.1 Recommended Configurations

> **Note**: The following configurations are validated in specific test environments and are for reference only. The optimal configuration depends on factors such as maximum input/output length, prefix cache hit rate, precision requirements, and deployment machine ratios. It is recommended to refer to Section 9.2 for tuning based on actual conditions.

#### Table 1: Scenario Overview

> `*Total NPUs` indicates the total number of NPUs used across all nodes. 1 node = 1 Atlas 800 A3 server (64G × 16 NPUs).

|Scenario|Deployment Mode|*Total NPUs|Weight Version|Key Considerations|
|--------|---------------|-----------|--------------|------------------|
|High Throughput / Low Latency<br>(16K context)|Single-Node Mixed|16 (A3)|kimi-k2.5-w4a8|Use dp4 tp4 for optimal throughput and low latency|
|High Throughput / Low Latency<br>(16K context)|2-Node Data Parallel|16 (A2)|kimi-k2.5-w4a8|dp4 tp4 across 2 nodes; balanced latency and throughput|
|High Throughput / Low Latency<br>(16K context)|2P2D deployment|64 (A3)|kimi-k2.5-w4a8|Prefill: dp2 tp8; Decode: dp32 tp1 for high concurrency|
|Long Context<br>(128K, low concurrency ≤4)|Single-Node Mixed|16 (A3)|kimi-k2.5-w4a8|dp1 tp16 to maximize TP, accommodate extreme context lengths|
|Long Context<br>(128K, high concurrency >4)|Single-Node Mixed|16 (A3)|kimi-k2.5-w4a8|dp2 tp8 to optimize memory bandwidth and support higher concurrency|

#### Table 2: Detailed Node Configuration

|Scenario|Configuration|NPUs|TP|DP|Max Model Len|MTP Speculation Num|
|--------|-------------|-----|--|--|-------------|--------------------|
|High Throughput / Low Latency (16K)|Server / Single Machine|16|4|4|~16K|3|
|High Throughput / Low Latency (16K)|Server / 2-Node DP|8|4|2|~16K|3|
|High Throughput / Low Latency (16K)|Server-P Node|16|8|2|~16K|3|
|High Throughput / Low Latency (16K)|Server-D Node|16|1|32|~16K|3|
|Long Context (128K, low concurrency ≤4)|Server / Single Machine|16|16|1|128K|3|
|Long Context (128K, high concurrency >4)|Server / Single Machine|16|8|2|128K|3|

> For complete startup commands and parameter descriptions, please refer to the deployment examples in [Chapter 5](#5-online-service-deployment).

**Notice:**
`max-model-len` and `max-num-seqs` need to be set according to the actual usage scenario. For other settings, please refer to the **[Deployment](#5-online-service-deployment)** chapter.

### 9.2 Tuning Guidelines

Please refer to the [Public Performance Tuning Documentation](../../developer_guide/performance_and_debug/optimization_and_tuning.md) for tuning methods.

Please refer to the [Feature Guide](../../user_guide/support_matrix/feature_matrix.md) for detailed feature descriptions.

## 10 FAQ

For common environment, installation, and general parameter issues, please refer to the [Public FAQ](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html); this chapter only covers model-specific issues.

- **Q: What is the recommended TP/DP configuration for single-node deployment?**

  A: For single-node deployment, we recommend using `dp4 tp4` instead of `dp2 tp8`.
