# Qwen3.8-27B

## 1 Introduction

Qwen3.8-27B is the 27-billion-parameter dense member of the Qwen3.8 family, the most capable generation in the Qwen open-model family to date. Built on the architectural foundation of Qwen3.5, it shares the same hybrid-attention backbone as the 2.4T MoE flagship: of its 64 layers, only 16 run full (gated) attention (`full_attention_interval: 4`) while the other 48 run linear attention (Gated DeltaNet) with a constant recurrent state. It is a native vision-language model — the architecture is `Qwen3_5ForConditionalGeneration` and `config.json` carries a `vision_config` — that understands images and videos, and it ships with a built-in MTP (Multi-Token Prediction) draft head and a native 262,144-token context window extensible up to 1,000,000 tokens.

Delivering substantial gains over Qwen3.5/Qwen3.6 across coding, professional work, research, and long-horizon agentic tasks, Qwen3.8-27B features stronger autonomous planning, more reliable end-to-end task completion, and broader downstream compatibility with popular harnesses and development tools. Thinking mode is on by default and can be disabled per request; reasoning depth is tunable via `reasoning_effort` (`xhigh`/`medium`/`low`), and reasoning context from historical messages is retained via `preserve_thinking`.

This document focuses on text serving on Ascend NPUs. It describes the main validation steps for the model, including supported features, prerequisites, installation, multi-node deployment, functional verification, accuracy and performance evaluation, performance tuning, and FAQs.

This document is validated and written based on **vLLM-Ascend 0.23.0**. The current model (Qwen3.8-27B) is first supported in this version.

## 2 Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_features.md) to get the model's supported feature matrix.

Refer to [feature guide](../../user_guide/feature_guide/index.md) to get feature configuration details.

:::{note}
The support matrix records the maximum verified capability for this model. Adjust `--max-model-len`, `--max-num-seqs`, and `--max-num-batched-tokens` based on your service workload and available KV cache.
:::

## 3 Prerequisites

### 3.1 Model Weight

The following model weights are available:
- `Qwen3.8-27B` (BF16 version): requires 1 Ascend950DT series (96GB × 8) node or 1 Atlas 800 A3 (64GB × 16) node. [Download model weight](https://www.modelscope.cn/models/Qwen/Qwen3.8-27B)
- `Qwen3.8-27B-w8a8` (Quantized version): requires 1 Atlas 800 A3 (64GB × 16) node. [Download model weight](https://www.modelscope.cn/models/Eco-Tech/Qwen3.8-27B-w8a8)
- `Qwen3.8-27B-w8a8-mxfp8` (Quantized version): requires 1 Ascend950DT series (96GB × 8) node. [Download model weight](https://www.modelscope.cn/models/Eco-Tech/Qwen3.8-27B-w8a8-mxfp8)

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`.

### 3.2 Verify Multi-node Communication (Optional)

If you want to deploy the model in a multi-node environment, verify the communication environment according to [verify multi-node communication environment](../../installation.md#verify-multi-node-communication).

## 4 Installation

### 4.1 Docker Image Installation

Select an image based on your machine type and start the docker image on your node, refer to [using docker](../../installation.md#set-up-using-docker).

:::::{tab-set}
:sync-group: install

::::{tab-item} A3 series
:sync: A3

Start the docker image on each node.

```{code-block} bash
  :substitutions:
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|-a3
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

::::{tab-item} Ascend950DT series
:sync: 950dt

Start the docker image on each node.

```{code-block} bash
  :substitutions:
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|-a5
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
    --device /dev/hisi_hdc \
    --device /dev/ummu \
    --device /dev/uburma \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /etc/hccl_rootinfo.json:/etc/hccl_rootinfo.json \
    -v /etc/hixlep/:/etc/hixlep/ \
    -v /root/.cache:/root/.cache \
    -v /usr/local/sbin:/usr/local/sbin \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
    -v /usr/lib64:/usr/lib64 \
    -it $IMAGE bash
```

::::

:::::

After entering the container, verify that vLLM and vLLM-Ascend can be imported:

```shell
python -c "import vllm, vllm_ascend; print('vllm and vllm_ascend are ready')"
```

### 4.2 Source Code Installation

You can also build and install `vllm-ascend` from source. Refer to [set up using Python](../../installation.md#set-up-using-python).

If you want to deploy a multi-node service, install the same version of vLLM and vLLM-Ascend on each node.

## 5 Online Service Deployment

### 5.1 Single-Node Online Deployment

Single-node deployment completes both Prefill and Decode within the same node, suitable for development, testing, and medium-scale inference scenarios.

Before starting the service:

- Replace the model path, parallel sizes and service port with values from the target environment.

:::::{tab-set}

::::{tab-item} Atlas 800 A3

The following example is for Atlas 800 A3. Quantized versions need `--quantization ascend`.

```shell
#!/bin/sh
# Load model from ModelScope to speed up download
export VLLM_USE_MODELSCOPE=True
# To reduce memory fragmentation and avoid out of memory
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
# Size of the shared buffer (in MB) used by HCCL for NPU-to-NPU collective communication
export HCCL_BUFFSIZE=512
# Whether OpenMP threads are bound to specific CPU cores
export OMP_PROC_BIND=false
# Number of OpenMP threads available for parallel regions
export OMP_NUM_THREADS=1
# Enables the Ascend task queue for asynchronous operator dispatch
export TASK_QUEUE_ENABLE=1

# Model weight path; can be a ModelScope model id (e.g., Eco-Tech/Qwen3.8-27B-w8a8) or a local directory path
export MODEL_PATH=Eco-Tech/Qwen3.8-27B-w8a8

vllm serve $MODEL_PATH \
    --host 0.0.0.0 \
    --port 8000 \
    --data-parallel-size 1 \
    --tensor-parallel-size 2 \
    --seed 1024 \
    --quantization ascend \
    --served-model-name qwen3.8 \
    --max-num-seqs 32 \
    --max-model-len 131072 \
    --max-num-batched-tokens 16384 \
    --trust-remote-code \
    --gpu-memory-utilization 0.85 \
    --no-enable-prefix-caching \
    --speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": true}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --additional-config '{"enable_cpu_binding":true}'
```

Key Parameter Descriptions:

- `--data-parallel-size 1` and `--tensor-parallel-size 2` are common settings for data parallelism (DP) and tensor parallelism (TP) sizes.
- `--max-model-len` represents the context length, which is the maximum value of the input plus output for a single request.
- `--max-num-seqs` indicates the maximum number of requests that each DP group is allowed to process. If the number of requests sent to the service exceeds this limit, the excess requests will remain in a waiting state and will not be scheduled. Note that the time spent in the waiting state is also counted in metrics such as TTFT and TPOT. Therefore, when testing performance, it is generally recommended that `--max-num-seqs` * `--data-parallel-size` >= the actual total concurrency.
- `--max-num-batched-tokens` represents the maximum number of tokens that the model can process in a single step. Currently, vLLM v1 scheduling enables ChunkPrefill/SplitFuse by default, which means:
    - (1) If the input length of a request is greater than `--max-num-batched-tokens`, it will be divided into multiple rounds of computation according to `--max-num-batched-tokens`;
    - (2) Decode requests are prioritized for scheduling, and prefill requests are scheduled only if there is available capacity.
    - Generally, if `--max-num-batched-tokens` is set to a larger value, the overall latency will be lower, but the pressure on HBM memory (activation value usage) will be greater.
- `--gpu-memory-utilization` represents the proportion of HBM that vLLM will use for actual inference. Its essential function is to calculate the available kv_cache size. During the warm-up phase (referred to as profile run in vLLM), vLLM records the peak HBM memory usage during an inference process with an input size of `--max-num-batched-tokens`. The available kv_cache size is then calculated as: `--gpu-memory-utilization` * HBM size - peak HBM memory usage. Therefore, the larger the value of `--gpu-memory-utilization`, the more kv_cache can be used. However, since the HBM memory usage during the warm-up phase may differ from that during actual inference (e.g., due to uneven EP load), setting `--gpu-memory-utilization` too high may lead to OOM (Out of Memory) issues during actual inference. The default value is `0.9`.
- `--no-enable-prefix-caching` indicates that prefix caching is disabled. The current implementation of hybrid kv cache for Qwen3.8-27B may result in a very large effective `block_size` when prefix caching is enabled (e.g., 2048), which means any prefix shorter than `block_size` will never be cached. If your workload has many short repeated prefixes, consider keeping prefix caching disabled. For related issues, see the [Public FAQ](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html).
- `--quantization ascend` indicates that quantization is used. To disable quantization, remove this option.
- `--speculative-config` uses `qwen3_5_mtp` for `Qwen3.8-27B` because it shares the same MTP head design as `Qwen3.5-27B`.
- `--compilation-config` contains configurations related to the aclgraph graph mode. The most significant configurations are `"cudagraph_mode"` and `"cudagraph_capture_sizes"`, which have the following meanings:
    - `"cudagraph_mode"`: represents the specific graph mode. Currently, `"PIECEWISE"` and `"FULL_DECODE_ONLY"` are supported. The graph mode is mainly used to reduce the cost of operator dispatch. Currently, `"FULL_DECODE_ONLY"` is recommended.
    - `"cudagraph_capture_sizes"`: represents different levels of graph modes. The default value is `[1, 2, 4, 8, 16, 24, 32, 40,..., --max-num-seqs]`. In the graph mode, the input for graphs at different levels is fixed, and inputs between levels are automatically padded to the next level. Currently, the default setting is recommended. Only in some scenarios is it necessary to set this separately to achieve optimal performance.

::::

::::{tab-item} Ascend950DT series

The following example is for Ascend950DT series. Quantized versions need `--quantization ascend`.

```bash
#!/bin/sh
# Load model from ModelScope to speed up download
export VLLM_USE_MODELSCOPE=True
# To reduce memory fragmentation and avoid out of memory
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
# Size of the shared buffer (in MB) used by HCCL for NPU-to-NPU collective communication
export HCCL_BUFFSIZE=512
# Whether OpenMP threads are bound to specific CPU cores
export OMP_PROC_BIND=false
# Number of OpenMP threads available for parallel regions
export OMP_NUM_THREADS=1
# Enables the Ascend task queue for asynchronous operator dispatch
export TASK_QUEUE_ENABLE=1

# Model weight path; can be a ModelScope model id (e.g., Eco-Tech/Qwen3.8-27B-w8a8-mxfp8) or a local directory path
export MODEL_PATH=Eco-Tech/Qwen3.8-27B-w8a8-mxfp8

vllm serve $MODEL_PATH \
    --host 0.0.0.0 \
    --port 8000 \
    --data-parallel-size 1 \
    --tensor-parallel-size 1 \
    --seed 1024 \
    --quantization ascend \
    --served-model-name qwen3.8 \
    --max-num-seqs 32 \
    --max-model-len 131072 \
    --max-num-batched-tokens 16384 \
    --trust-remote-code \
    --gpu-memory-utilization 0.85 \
    --no-enable-prefix-caching \
    --speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": true}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --additional-config '{"enable_cpu_binding":true}'
```

Key Parameter Descriptions:

- `--data-parallel-size 1` and `--tensor-parallel-size 1` are common settings for data parallelism (DP) and tensor parallelism (TP) sizes.
- `--max-model-len` represents the context length, which is the maximum value of the input plus output for a single request.
- `--max-num-seqs` indicates the maximum number of requests that each DP group is allowed to process. If the number of requests sent to the service exceeds this limit, the excess requests will remain in a waiting state and will not be scheduled. Note that the time spent in the waiting state is also counted in metrics such as TTFT and TPOT. Therefore, when testing performance, it is generally recommended that `--max-num-seqs` * `--data-parallel-size` >= the actual total concurrency.
- `--max-num-batched-tokens` represents the maximum number of tokens that the model can process in a single step. Currently, vLLM v1 scheduling enables ChunkPrefill/SplitFuse by default, which means:
    - (1) If the input length of a request is greater than `--max-num-batched-tokens`, it will be divided into multiple rounds of computation according to `--max-num-batched-tokens`;
    - (2) Decode requests are prioritized for scheduling, and prefill requests are scheduled only if there is available capacity.
    - Generally, if `--max-num-batched-tokens` is set to a larger value, the overall latency will be lower, but the pressure on HBM memory (activation value usage) will be greater.
- `--gpu-memory-utilization` represents the proportion of HBM that vLLM will use for actual inference. Its essential function is to calculate the available kv_cache size. During the warm-up phase (referred to as profile run in vLLM), vLLM records the peak HBM memory usage during an inference process with an input size of `--max-num-batched-tokens`. The available kv_cache size is then calculated as: `--gpu-memory-utilization` * HBM size - peak HBM memory usage. Therefore, the larger the value of `--gpu-memory-utilization`, the more kv_cache can be used. However, since the HBM memory usage during the warm-up phase may differ from that during actual inference (e.g., due to uneven EP load), setting `--gpu-memory-utilization` too high may lead to OOM (Out of Memory) issues during actual inference. The default value is `0.9`.
- `--no-enable-prefix-caching` indicates that prefix caching is disabled. The current implementation of hybrid kv cache for Qwen3.8-27B may result in a very large effective `block_size` when prefix caching is enabled (e.g., 2048), which means any prefix shorter than `block_size` will never be cached. If your workload has many short repeated prefixes, consider keeping prefix caching disabled. For related issues, see the [Public FAQ](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html).
- `--quantization ascend` indicates that quantization is used. To disable quantization, remove this option.
- `--speculative-config` uses `qwen3_5_mtp` for `Qwen3.8-27B` because it shares the same MTP head design as `Qwen3.5-27B`.
- `--compilation-config` contains configurations related to the aclgraph graph mode. The most significant configurations are `"cudagraph_mode"` and `"cudagraph_capture_sizes"`, which have the following meanings:
    - `"cudagraph_mode"`: represents the specific graph mode. Currently, `"PIECEWISE"` and `"FULL_DECODE_ONLY"` are supported. The graph mode is mainly used to reduce the cost of operator dispatch. Currently, `"FULL_DECODE_ONLY"` is recommended.
    - `"cudagraph_capture_sizes"`: represents different levels of graph modes. The default value is `[1, 2, 4, 8, 16, 24, 32, 40,..., --max-num-seqs]`. In the graph mode, the input for graphs at different levels is fixed, and inputs between levels are automatically padded to the next level. Currently, the default setting is recommended. Only in some scenarios is it necessary to set this separately to achieve optimal performance.

::::

:::::

### 5.2 Multi-Node PD Separation Deployment

For high-concurrency production scenarios, multi-node PD (Prefill-Decode) separation can be used to scale the service. The recommended approach is to use Mooncake for deployment: [Mooncake Multi-Node PD Disaggregation Guide](../features/pd_disaggregation_mooncake_multi_node.md).

In the standard single-node deployment mode, Prefill (prompt processing) and Decode (token generation) tasks run on the same set of NPUs. This can lead to two issues:

1. **Prefill preemption interrupts Decode**: Prefill is a compute-intensive task that processes the entire input context at once, while Decode generates tokens one by one. When a new user request arrives, its Prefill phase can preempt and interrupt ongoing Decode tasks, causing jitter and higher time-per-output-token (TPOT) latency.
2. **Inflexible resource allocation**: Prefill and Decode have fundamentally different computational characteristics — Prefill is compute-bound and memory-bandwidth-intensive, while Decode is memory-bandwidth-bound. Running them on the same hardware forces a compromise that satisfies neither optimally.

PD (Prefill-Decode) separation addresses these issues by running Prefill and Decode on dedicated node groups, each configured independently:

- **Prefill nodes** focus on high-throughput prompt processing, optimized for compute and communication (e.g., enabling FlashComm for Allreduce acceleration).
- **Decode nodes** focus on low-latency token generation, optimized for memory bandwidth (e.g., enabling async-scheduling and full-decode aclgraph).

For `Qwen3.8-27B-w8a8`, a typical **1P1D** configuration requires **2 Atlas 800 A3 (64GB × 16) nodes** (1 Prefill node + 1 Decode node), with **TP=2** and **DP=8** on each node, which fully utilizes all 16 NPUs of an Atlas A3. The example below uses `Qwen3.8-27B-w8a8`.

> **Note**: Since `Qwen3.8-27B` fits in a single node, multi-node PD separation is only recommended for high-concurrency production deployments. For the Mooncake deployment specifics, please refer to the [Mooncake Multi-Node PD Disaggregation Guide](../features/pd_disaggregation_mooncake_multi_node.md).

To run the vllm-ascend Prefill-Decode Disaggregation service, you need to:

- Deploy a `launch_online_dp.py` script and a `run_dp_template.sh` script on each node;
- Deploy a `load_balance_proxy_server_example.py` script on the prefill master node to forward requests.

1. `launch_online_dp.py` is used to launch external dp vllm servers.
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

2. Prefill Node 0 `run_dp_template.sh` script. You can get the template in the repository's examples: [run_dp_template.sh](https://github.com/vllm-project/vllm-ascend/blob/main/examples/external_online_dp/run_dp_template.sh).

    ```shell
    # nic_name is the network interface name corresponding to local_ip of the current node
    nic_name="xxx"
    local_ip="141.xx.xx.1"

    export HCCL_IF_IP=$local_ip
    export GLOO_SOCKET_IFNAME=$nic_name
    export TP_SOCKET_IFNAME=$nic_name
    export HCCL_SOCKET_IFNAME=$nic_name

    # [Optional] jemalloc
    # jemalloc is for better performance, if `libjemalloc.so` is installed on your machine, you can turn it on.
    # export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD

    export HCCL_OP_EXPANSION_MODE="AIV"
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=1
    export TASK_QUEUE_ENABLE=1
    export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH

    export HCCL_BUFFSIZE=1024
    export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
    export ASCEND_RT_VISIBLE_DEVICES=$1

    vllm serve Eco-Tech/Qwen3.8-27B-w8a8 \
      --host 0.0.0.0 \
      --port $2 \
      --data-parallel-size $3 \
      --data-parallel-rank $4 \
      --data-parallel-address $5 \
      --data-parallel-rpc-port $6 \
      --tensor-parallel-size $7 \
      --seed 1024 \
      --quantization ascend \
      --served-model-name qwen3.8 \
      --trust-remote-code \
      --max-num-seqs 4 \
      --max-model-len 32768 \
      --max-num-batched-tokens 16384 \
      --no-enable-prefix-caching \
      --gpu-memory-utilization 0.95 \
      --enforce-eager \
      --speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": true}' \
      --additional-config '{"enable_cpu_binding":true}' \
      --kv-transfer-config \
      '{"kv_connector": "MooncakeConnectorV1",
      "kv_role": "kv_producer",
      "kv_port": "30000",
      "engine_id": "0",
      "kv_connector_extra_config": {
                "prefill": {
                        "dp_size": 8,
                        "tp_size": 2
                },
                "decode": {
                        "dp_size": 8,
                        "tp_size": 2
            }
        }
      }'
    ```

3. Decode Node 0 `run_dp_template.sh` script. You can get the template in the repository's examples: [run_dp_template.sh](https://github.com/vllm-project/vllm-ascend/blob/main/examples/external_online_dp/run_dp_template.sh).

    ```shell
    # nic_name is the network interface name corresponding to local_ip of the current node
    nic_name="xxx"
    local_ip="141.xx.xx.2"

    export HCCL_IF_IP=$local_ip
    export GLOO_SOCKET_IFNAME=$nic_name
    export TP_SOCKET_IFNAME=$nic_name
    export HCCL_SOCKET_IFNAME=$nic_name

    # [Optional] jemalloc
    # jemalloc is for better performance, if `libjemalloc.so` is installed on your machine, you can turn it on.
    # export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD

    export HCCL_OP_EXPANSION_MODE="AIV"
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=1
    export TASK_QUEUE_ENABLE=1
    export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH

    export HCCL_BUFFSIZE=1024
    export ASCEND_RT_VISIBLE_DEVICES=$1

    vllm serve Eco-Tech/Qwen3.8-27B-w8a8 \
      --host 0.0.0.0 \
      --port $2 \
      --data-parallel-size $3 \
      --data-parallel-rank $4 \
      --data-parallel-address $5 \
      --data-parallel-rpc-port $6 \
      --tensor-parallel-size $7 \
      --seed 1024 \
      --quantization ascend \
      --served-model-name qwen3.8 \
      --trust-remote-code \
      --max-num-seqs 16 \
      --max-model-len 32768 \
      --max-num-batched-tokens 2048 \
      --no-enable-prefix-caching \
      --gpu-memory-utilization 0.91 \
      --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
      --additional-config '{"recompute_scheduler_enable":true,"enable_cpu_binding":true}' \
      --async-scheduling \
      --speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": true}' \
      --kv-transfer-config \
      '{"kv_connector": "MooncakeConnectorV1",
      "kv_role": "kv_consumer",
      "kv_port": "30200",
      "engine_id": "1",
      "kv_connector_extra_config": {
                "prefill": {
                        "dp_size": 8,
                        "tp_size": 2
                },
                "decode": {
                        "dp_size": 8,
                        "tp_size": 2
            }
        }
      }'
    ```

   Key Parameter Descriptions:

   - `VLLM_ASCEND_ENABLE_FLASHCOMM1=1`: enables the Allreduce communication optimization on prefill nodes, which reduces the communication overhead of long-context prefill.
   - `recompute_scheduler_enable: true`: enables the recomputation scheduler. When the KV Cache of the decode node is insufficient, requests will be sent to the prefill node to recompute the KV Cache. In the PD separation scenario, enable this configuration only on decode nodes.
   - `--async-scheduling` (on decode nodes): enables asynchronous scheduling, which can reduce TPOT for high-concurrency decode workloads.
   - `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'` (on decode nodes): enables the full-decode aclgraph mode, which significantly reduces scheduling latency on the decode side.

4. Run server for each node:

    ```shell
    # p0 (Prefill node 0)
    python launch_online_dp.py --dp-size 8 --tp-size 2 --dp-size-local 8 --dp-rank-start 0 --dp-address 141.xx.xx.1 --dp-rpc-port 12321 --vllm-start-port 7100
    # d0 (Decode node 0)
    python launch_online_dp.py --dp-size 8 --tp-size 2 --dp-size-local 8 --dp-rank-start 0 --dp-address 141.xx.xx.2 --dp-rpc-port 12321 --vllm-start-port 7100
    ```

5. Run the proxy server on the prefill master node.

    You can get the proxy program in the repository's examples: [load_balance_proxy_server_example.py](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py).

    Note: Since each node has 8 DP ranks (with `--vllm-start-port 7100` + local rank index, occupying ports 7100-7107), you need to list all 8 ports for each node in the proxy command:

    ```shell
    python load_balance_proxy_server_example.py \
      --port 1999 \
      --host 141.xx.xx.1 \
      --prefiller-hosts \
        141.xx.xx.1 \
        141.xx.xx.1 \
        141.xx.xx.1 \
        141.xx.xx.1 \
        141.xx.xx.1 \
        141.xx.xx.1 \
        141.xx.xx.1 \
        141.xx.xx.1 \
      --prefiller-ports \
        7100 7101 7102 7103 7104 7105 7106 7107 \
      --decoder-hosts \
        141.xx.xx.2 \
        141.xx.xx.2 \
        141.xx.xx.2 \
        141.xx.xx.2 \
        141.xx.xx.2 \
        141.xx.xx.2 \
        141.xx.xx.2 \
        141.xx.xx.2 \
      --decoder-ports \
        7100 7101 7102 7103 7104 7105 7106 7107 \
    ```

Deployment Verification:

After the PD separation service is fully started, send a request through the proxy port on the prefill master node to verify that Prefill and Decode nodes are working correctly together:

```bash
curl http://<proxy_node0_ip>:1999/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen3.8",
        "messages": [
            {"role": "user", "content": "The future of AI is"}
        ],
        "max_tokens": 1024,
        "temperature": 1.0,
        "top_p": 0.95
    }'
```

Expected Result: The proxy returns HTTP 200 OK. The JSON response contains the `choices` field with the generated text, confirming that Prefill nodes have successfully processed the prompt and Decode nodes have generated the response.

Common Issues Tip: If you encounter issues with PD separation deployment, please refer to the [Public FAQ](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html) for troubleshooting.

## 6 Functional Verification

After the service is started, the model can be invoked by sending a prompt. Two API interfaces are supported: `completions` and `chat/completions`. Use the `--served-model-name` you configured (`qwen3.8` for `Qwen3.8-27B`).

**Completions API:**

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen3.8",
        "prompt": "The future of AI is",
        "max_tokens": 50,
        "temperature": 0.7
    }'
```

**Chat Completions API:**

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen3.8",
        "messages": [
            {"role": "user", "content": "The future of AI is"}
        ],
        "max_completion_tokens": 1024,
        "temperature": 1.0,
        "top_p": 0.95
    }'
```

Expected Result: The service returns HTTP 200 OK. The JSON response contains the `choices` field with generated text. Example output for the completions API (content truncated for brevity):

```json
{
    "id": "cmpl-xxxxxxxxxxxxx",
    "object": "text_completion",
    "created": 1780971952,
    "model": "qwen3.8",
    "choices": [
        {
            "index": 0,
            "text": "The future of AI is a rapidly evolving landscape with breakthroughs in natural language understanding, multimodal reasoning, and autonomous agents. As models grow more capable and efficient...",
            "logprobs": null,
            "finish_reason": "length"
        }
    ],
    "usage": {
        "prompt_tokens": 4,
        "total_tokens": 54,
        "completion_tokens": 50
    }
}
```

## 7 Accuracy Evaluation

Here are two accuracy evaluation methods.

### Using AISBench

1. Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) for details.

2. After execution, you can get the result. Here are the results of `Qwen3.8-27B`, `Qwen3.8-27B-w8a8` and `Qwen3.8-27B-w8a8-mxfp8` in `vllm-ascend:v0.23.0rc1` for reference only.

| dataset | model | metric | mode | vllm-api-general-chat |
|----- | ----- | ----- | ----- | -----|
| GPQA Diamond | Qwen3.8-27B | accuracy | gen | 90.40 |
| GPQA Diamond | Qwen3.8-27B-w8a8 | accuracy | gen | 89.90 |
| GPQA Diamond | Qwen3.8-27B-w8a8-mxfp8 | accuracy | gen | 89.39 |

### Using Language Model Evaluation Harness

Using the `gsm8k` dataset as an example test dataset, run the accuracy evaluation for `Qwen3.8-27B-w8a8` in online mode.

1. For `lm_eval` installation, please refer to [Using lm_eval](../../developer_guide/evaluation/using_lm_eval.md).
2. Run `lm_eval` to execute the accuracy evaluation.

```shell
# For Qwen3.8-27B-w8a8
export VLLM_USE_MODELSCOPE=True
vllm serve Eco-Tech/Qwen3.8-27B-w8a8 \
    --served-model-name qwen3.8 \
    --trust-remote-code \
    --quantization ascend \
    --tensor-parallel-size 2 \
    --max-model-len 13312 \
    --max-num-seqs 32 \
    --gpu-memory-utilization 0.90 \
    --no-enable-prefix-caching

# Run lm_eval in another terminal
lm_eval \
  --model local-completions \
  --model_args model=qwen3.8,base_url=http://127.0.0.1:8000/v1/completions,tokenized_requests=False,trust_remote_code=True \
  --tasks gsm8k \
  --output_path ./
```

## 8 Performance Evaluation

### 8.1 Install AISBench

Run AISBench in a separate environment or container so that the load generator does not affect the serving processes. Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation).

### 8.2 Performance Service Configuration

Use the deployment in Section 5 as the baseline. Change the following values on the node:

| Parameter | Standard deployment | Performance test |
| --- | ---: | ---: |
| `--max-model-len` | 131072 | 250000 |
| `--max-num-batched-tokens` | 16384 | 8192 |
| `--gpu-memory-utilization` | 0.85 | 0.95 |

### 8.3 Run the Tests

Run performance evaluation of `Qwen3.8-27B` as an example. Refer to [vLLM benchmark](https://docs.vllm.ai/en/latest/benchmarking/) for more details.

```shell
vllm bench serve \
  --model Qwen/Qwen3.8-27B \
  --served-model-name qwen3.8 \
  --base-url http://<server_ip>:8000 \
  --dataset-name random \
  --random-input-len 8192 \
  --random-output-len 1024 \
  --num-prompts 16 \
  --max-concurrency 4 \
  --save-result \
  --result-dir ./
```

Record the node count, DP/TP topology, context length, concurrency, reasoning effort, and weight revision together with the result.

### 8.4 Enabled Optimizations

| Feature | Description |
| --- | --- |
| Chunked Prefill | Splits long prefill inputs into chunks to reduce per-step memory peaks. |
| W8A8 | Uses Ascend quantization for the validated checkpoint. |
| Lazy Safetensors | Avoids prefetching the complete NFS checkpoint. |
| MTP | Uses three speculative tokens with the `qwen3_5_mtp` method. |
| ACL Graph | Uses `FULL_DECODE_ONLY` replay. |
| CPU Binding | Reduces cross-core scheduling overhead. |

## 9 Performance Tuning

Use the deployment values above as a baseline. Adjust `max-model-len`, `max-num-seqs`, `max-num-batched-tokens`, and `gpu-memory-utilization` together for the target workload.

Refer to the [performance tuning guide](../../developer_guide/performance_and_debug/optimization_and_tuning.md) and the [feature matrix](../../user_guide/support_matrix/feature_matrix.md) for additional guidance.

## 10 FAQ

For common environment, installation, and general parameter issues, refer to the [Public FAQ](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html).