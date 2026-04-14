# Kimi-K2.5

## Introduction

Kimi K2.5 is an open-source, native multimodal agentic model built through continual pretraining on approximately 15 trillion mixed visual and text tokens atop Kimi-K2-Base. It seamlessly integrates vision and language understanding with advanced agentic capabilities, instant and thinking modes, as well as conversational and agentic paradigms.

The `Kimi-K2.5` model is first supported in `vllm-ascend:v0.17.0rc1`.

This document will show the main verification steps of the model, including supported features, feature configuration, environment preparation, single-node and multi-node deployment, accuracy and performance evaluation.

## Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [feature guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## Environment Preparation

### Model Weight

- `Kimi-K2.5-w4a8`(Quantized version for w4a8): [Download model weight](https://modelscope.cn/models/Eco-Tech/Kimi-K2.5-W4A8).
- `kimi-k2.5-eagle3`(Eagle3 MTP draft model for accelerating inference of Kimi-K2.5): [Download model weight](https://huggingface.co/lightseekorg/kimi-k2.5-eagle3)

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`.

### Verify Multi-node Communication(Optional)

If you want to deploy multi-node environment, you need to verify multi-node communication according to [verify multi-node communication environment](../../installation.md#verify-multi-node-communication).

### Installation

You can use our official docker image to run `Kimi-K2.5` directly.

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

In addition, if you don't want to use the docker image as above, you can also build all from source:

- Install `vllm-ascend` from source, refer to [installation](../../installation.md).

If you want to deploy multi-node environment, you need to set up environment on each node.

## Deployment

### Single-node Deployment

- Quantized model `Kimi-K2.5-w4a8` can be deployed on 1 Atlas 800 A3 (64G × 16).

Run the following script to execute online inference.

```shell
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
  --async-scheduling \
  --max-num-seqs 64 \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.9 \
  --compilation-config '{"cudagraph_capture_sizes":[4,8,16,32,64,128,256], "cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --speculative-config '{"method":"eagle3", "model":"lightseekorg/kimi-k2.5-eagle3", "num_speculative_tokens":3}' \
  --mm-encoder-tp-mode data
```

**Notice:**
The parameters are explained as follows:

- Setting the environment variable `VLLM_ASCEND_BALANCE_SCHEDULING=1` enables balance scheduling. This may help increase output throughput and reduce TPOT in v1 scheduler. However, TTFT may degrade in some scenarios. Furthermore, enabling this feature is not recommended in scenarios where PD is separated.
- For single-node deployment, we recommend using `dp4tp4` instead of `dp2tp8`.
- `--max-model-len` specifies the maximum context length - that is, the sum of input and output tokens for a single request. For performance testing with an input length of 3.5K and output length of 1.5K, a value of `16384` is sufficient, however, for precision testing, please set it at least `35000`.
- `--no-enable-prefix-caching` indicates that prefix caching is disabled. To enable it, remove this option.
- `--mm-encoder-tp-mode` indicates how to optimize multi-modal encoder inference using tensor parallelism (TP). If you want to test the multimodal inputs,  we recommend using `data`.
- If you use the w4a8 weight, more memory will be allocated to kvcache, and you can try to increase system throughput to achieve greater throughput.

### Multi-node Deployment

- `Kimi-K2.5-w4a8`: require at least 2 Atlas 800 A2 (64G × 8).

Run the following scripts on two nodes respectively.

**Node 0**

```shell
#!/bin/sh

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
  --async-scheduling \
  --max-num-seqs 16 \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.9 \
  --compilation-config '{"cudagraph_capture_sizes":[4,8,16,32,64], "cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --speculative-config '{"method":"eagle3", "model":"lightseekorg/kimi-k2.5-eagle3", "num_speculative_tokens":3}' \
  --mm-encoder-tp-mode data
```

**Node 1**

```shell
#!/bin/sh

# this obtained through ifconfig
# nic_name is the network interface name corresponding to local_ip of the current node
nic_name="xxx"
local_ip="xxx"

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
  --async-scheduling \
  --max-num-seqs 16 \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.9 \
  --compilation-config '{"cudagraph_capture_sizes":[4,8,16,32,64], "cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --speculative-config '{"method":"eagle3", "model":"lightseekorg/kimi-k2.5-eagle3", "num_speculative_tokens":3}' \
  --mm-encoder-tp-mode data
```

### Prefill-Decode Disaggregation

We recommend using Mooncake for deployment: [Mooncake](../features/pd_disaggregation_mooncake_multi_node.md).

Take Atlas 800 A3 (64G × 16) for example, we recommend to deploy 2P1D (4 nodes) rather than 1P1D (2 nodes), because there is not enough NPU memory to serve high concurrency in 1P1D case.

- `Kimi-K2.5-w4a8 2P1D` require 4 Atlas 800 A3 (64G × 16).

To run the vllm-ascend `Prefill-Decode Disaggregation` service, you need to deploy a `launch_dp_program.py` script and a `run_dp_template.sh` script on each node and deploy a `proxy.sh` script on prefill master node to forward requests.

1. `launch_online_dp.py` to launch external dp vllm servers.
    [launch\_online\_dp.py](https://github.com/vllm-project/vllm-ascend/blob/main/examples/external_online_dp/launch_online_dp.py)

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
      --additional-config '{"recompute_scheduler_enable":true}' \
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
      --additional-config '{"recompute_scheduler_enable":true}' \
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
      --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes":[4,8,16,32,48,64,80,96,112,128,144,160]}' \
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
      --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes":[4,8,16,32,48,64,80,96,112,128,144,160]}' \
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

**Notice:**
The parameters are explained as follows:

- `VLLM_ASCEND_ENABLE_FLASHCOMM1=1`: enables the communication optimization function on the prefill nodes.
- `VLLM_ASCEND_ENABLE_MLAPO=1`: enables the fusion operator, which can significantly improve performance but consumes more NPU memory. In the Prefill-Decode (PD) separation scenario, enable MLAPO only on decode nodes.
- `--async-scheduling`: enables the asynchronous scheduling function. When Multi-Token Prediction (MTP) is enabled, asynchronous scheduling of operator delivery can be implemented to overlap the operator delivery latency.
- `cudagraph_capture_sizes`: The recommended value is `n x (mtp + 1)`. And the min is `n = 1` and the max is `n = max-num-seqs`. For other values, it is recommended to set them to the number of frequently occurring requests on the Decode (D) node.
- `recompute_scheduler_enable: true`: enables the recomputation scheduler. When the Key-Value Cache (KV Cache) of the decode node is insufficient, requests will be sent to the prefill node to recompute the KV Cache. In the PD separation scenario, it is recommended to enable this configuration on both prefill and decode nodes simultaneously.
- `multistream_overlap_shared_expert: true`: When the Tensor Parallelism (TP) size is 1 or `enable_shared_expert_dp: true`, an additional stream is enabled to overlap the computation process of shared experts for improved efficiency.

1. run server for each node:

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

2. Run the `proxy.sh` script on the prefill master node

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

## Functional Verification

Once your server is started, you can query the model with input prompts:

```shell
curl http://<node0_ip>:<port>/v1/chat/completions \
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

## Accuracy Evaluation

Here are two accuracy evaluation methods.

### Using AISBench

1. Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) for details.

2. After execution, you can get the result, here is the result of `Kimi-K2.5-w4a8` in `vllm-ascend:v0.17.0rc1` for reference only.

| dataset | version | metric | mode | vllm-api-general-chat | note |
|----- | ----- | ----- | ----- | -----| ----- |
| GSM8K | - | accuracy | gen | 96.07 | 1 Atlas 800 A3 (64G × 16) |
| AIME2025 | - | accuracy | gen | 90.00 | 1 Atlas 800 A3 (64G × 16) |
| GPQA | - | accuracy | gen | 84.85 | 1 Atlas 800 A3 (64G × 16) |
| TextVQA | - | accuracy | gen | 80.29 | 1 Atlas 800 A3 (64G × 16) |

## Performance

### Using AISBench

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

### Using vLLM Benchmark

Run performance evaluation of `Kimi-K2.5-w4a8` as an example.

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/contributing/benchmarks.html) for more details.

There are three `vllm bench` subcommands:

- `latency`: Benchmark the latency of a single batch of requests.
- `serve`: Benchmark the online serving throughput.
- `throughput`: Benchmark offline inference throughput.

Take the `serve` as an example. Run the code as follows.

```shell
export VLLM_USE_MODELSCOPE=true
vllm bench serve --model Eco-Tech/Kimi-K2.5-w4a8 --dataset-name random --random-input 1024 --num-prompts 200 --request-rate 1 --save-result --result-dir ./
```

After about several minutes, you can get the performance evaluation result.

## Best Practices

In this chapter, we recommend best practices for three scenarios:

- Long-context: For long sequences with low concurrency (≤ 4): set `dp1 tp16`; For long sequences with high concurrency (> 4): set `dp2 tp8`
- Low-latency: For short sequences with low latency: we recommend setting `dp2 tp8`
- High-throughput: For short sequences with high throughput: we also recommend setting `dp4 tp4`

**Notice:**
`max-model-len` and `max-num-seqs` need to be set according to the actual usage scenario. For other settings, please refer to the **[Deployment](#deployment)** chapter.

## FAQ

- **Q: Why is the TPOT performance poor in Long-context test?**

  A: Please ensure that the FIA operator replacement script has been executed successfully to complete the replacement of FIA operators. Here is the script: [A2](../../../../tools/install_flash_infer_attention_score_ops_a2.sh) and [A3](../../../../tools/install_flash_infer_attention_score_ops_a3.sh)

- **Q: Startup fails with HCCL port conflicts (address already bound). What should I do?**

  A: Clean up old processes and restart: `pkill -f VLLM*`.

- **Q: How to handle OOM or unstable startup?**

  A: Reduce `--max-num-seqs` and `--max-model-len` first. If needed, reduce concurrency and load-testing pressure (e.g., `max-concurrency` / `num-prompts`).
