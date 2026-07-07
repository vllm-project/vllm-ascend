# Qwen3.5-27B/Qwen3.6-27B

## 1 Introduction

Qwen3.5-27B and Qwen3.6-27B are dense hybrid Mamba-Transformer language models in the Qwen3.5/Qwen3.6 family, integrating breakthroughs in architectural efficiency, reinforcement learning scale, and global accessibility. They share the same hybrid attention design (GDN + full attention), so deployment on Ascend NPUs follows the same pattern for both models. They are suitable for general-purpose text generation tasks such as dialogue, content creation, and code generation running on Ascend NPUs.

This document will demonstrate the main validation steps for the models, including supported features, feature configuration, environment preparation, single-node and multi-node deployment, as well as accuracy and performance evaluation.

It is **strongly recommended to use the latest release candidate (rc) version or the latest official version** of `vllm-ascend`. As a minimum-version requirement, `Qwen3.5-27B` is first supported in `vllm-ascend:v0.17.0rc1`, and `Qwen3.6-27B` is first supported in `vllm-ascend:v0.18.0rc1`.

## 2 Supported Features

Please refer to the [Supported Features List](../../user_guide/support_matrix/supported_models.md) for the model support matrix.

Please refer to the [Feature Guide](../../user_guide/feature_guide/index.md) for feature configuration information.

## 3 Prerequisites

### 3.1 Model Weight

**Qwen3.5-27B**

- `Qwen3.5-27B` (BF16 version): requires 1 Atlas 800 A3 (64G × 16) node or 1 Atlas 800 A2 (64G × 8) node. [Download model weight](https://modelscope.cn/models/Qwen/Qwen3.5-27B)
- `Qwen3.5-27B-w8a8` (Quantized version): requires 1 Atlas 800 A3 (64G × 16) node or 1 Atlas 800 A2 (64G × 8) node. [Download model weight](https://www.modelscope.cn/models/Eco-Tech/Qwen3.5-27B-w8a8-mtp)

**Qwen3.6-27B**

- `Qwen3.6-27B` (BF16 version): requires 1 Atlas 800 A3 (64G × 16) node or 1 Atlas 800 A2 (64G × 8) node. [Download model weight](https://modelscope.cn/models/Qwen/Qwen3.6-27B)
- `Qwen3.6-27B-w8a8` (Quantized version): requires 1 Atlas 800 A3 (64G × 16) node or 1 Atlas 800 A2 (64G × 8) node. [Download model weight](https://www.modelscope.cn/models/Eco-Tech/Qwen3.6-27B-w8a8)

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`.

### 3.2 Verify Multi-node Communication

If you want to deploy multi-node environment, you need to verify multi-node communication according to [verify multi-node communication environment](../../installation.md#verify-multi-node-communication).

## 4 Installation

### 4.1 Docker Image Installation

Select an image based on your machine type and start the docker image on your node, refer to [using docker](../../installation.md#set-up-using-docker).

It is **recommended to use the latest release candidate (rc) version or the latest official version** of the `vllm-ascend` image to ensure the best compatibility and access to the latest features. As a minimum-version requirement, use `vllm-ascend:v0.17.0rc1` (or a later version) for `Qwen3.5-27B`, and `vllm-ascend:v0.18.0rc1` (or a later version) for `Qwen3.6-27B`. For `Qwen3.6-27B` on Atlas 800 A3, please use the matching `v0.18.0rc1-a3` (or a later `-a3`) image.

=== "A3 series"

    Start the docker image on your each node.

    ```bash
    export IMAGE=m.daocloud.io/quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-a3
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

=== "A2 series"

    Start the docker image on your each node.

    ```bash
    export IMAGE=m.daocloud.io/quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}
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

After a successful docker run, you can verify the running container service by executing the `docker ps` command. The expected result is that the container `vllm-ascend` is listed with status `Up`, confirming the docker installation is successful.

### 4.2 Source Code Installation

If you don't want to use the docker image as above, you can also build all from source:

1. Clone the repository and install `vllm-ascend` from source:

    ```bash
    git clone https://github.com/vllm-project/vllm-ascend.git
    cd vllm-ascend
    pip install -e .
    ```

    For the complete installation steps, refer to [installation](../../installation.md).

2. If you want to deploy a multi-node environment, you need to set up the environment on each node.

To verify the source code installation, run the following command and confirm the displayed version matches the one you installed:

```bash
pip show vllm-ascend
```

Expected result: The version information of `vllm-ascend` is displayed, confirming a successful installation.

## 5 Online Service Deployment

### 5.1 Single-Node Online Deployment

Single-node deployment completes both Prefill and Decode within the same node, suitable for development, testing, and medium-scale inference scenarios. The `Qwen3.5-27B`, `Qwen3.5-27B-w8a8`, `Qwen3.6-27B`, and `Qwen3.6-27B-w8a8` models can all be deployed on 1 Atlas 800 A3 (64G × 16) or 1 Atlas 800 A2 (64G × 8). The quantized versions need to start with the `--quantization ascend` parameter.

Both `Qwen3.5-27B` and `Qwen3.6-27B` share the same MTP head design, so the `qwen3_5_mtp` speculative decoding method can be used for both.

**Qwen3.5-27B-w8a8**

Startup Command:

```bash
#!/bin/sh
# Load model from ModelScope to speed up download
export VLLM_USE_MODELSCOPE=True
# To reduce memory fragmentation and avoid out of memory
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=512
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1

vllm serve Eco-Tech/Qwen3.5-27B-w8a8-mtp \
--host 0.0.0.0 \
--port 8000 \
--data-parallel-size 1 \
--tensor-parallel-size 2 \
--seed 1024 \
--quantization ascend \
--served-model-name qwen3.5 \
--max-num-seqs 32 \
--max-model-len 133000 \
--max-num-batched-tokens 8096 \
--trust-remote-code \
--gpu-memory-utilization 0.90 \
--no-enable-prefix-caching \
--speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": true}' \
--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
--additional-config '{"enable_cpu_binding":true}' \
--async-scheduling
```

**Qwen3.6-27B-w8a8**

Startup Command (supports up to 262144 context length):

```bash
#!/bin/sh
# Load model from ModelScope to speed up download
export VLLM_USE_MODELSCOPE=True
# To reduce memory fragmentation and avoid out of memory
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=512
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1

vllm serve Eco-Tech/Qwen3.6-27B-w8a8 \
--host 0.0.0.0 \
--port 8000 \
--data-parallel-size 1 \
--tensor-parallel-size 2 \
--seed 1024 \
--quantization ascend \
--served-model-name qwen3.6 \
--max-num-seqs 32 \
--max-model-len 262144 \
--max-num-batched-tokens 8096 \
--trust-remote-code \
--gpu-memory-utilization 0.90 \
--no-enable-prefix-caching \
--speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": true}' \
--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
--additional-config '{"enable_cpu_binding":true}' \
--async-scheduling
```

Key Parameter Descriptions:

- `--data-parallel-size 1` and `--tensor-parallel-size 2` are common settings for data parallelism (DP) and tensor parallelism (TP) sizes.
- `--max-model-len` represents the context length, which is the maximum value of the input plus output for a single request. The Qwen3.6-27B model supports up to 262144.
- `--max-num-seqs` indicates the maximum number of requests that each DP group is allowed to process. If the number of requests sent to the service exceeds this limit, the excess requests will remain in a waiting state and will not be scheduled. Note that the time spent in the waiting state is also counted in metrics such as TTFT and TPOT. Therefore, when testing performance, it is generally recommended that `--max-num-seqs` * `--data-parallel-size` >= the actual total concurrency.
- `--max-num-batched-tokens` represents the maximum number of tokens that the model can process in a single step. Currently, vLLM v1 scheduling enables ChunkPrefill/SplitFuse by default, which means:
    - (1) If the input length of a request is greater than `--max-num-batched-tokens`, it will be divided into multiple rounds of computation according to `--max-num-batched-tokens`;
    - (2) Decode requests are prioritized for scheduling, and prefill requests are scheduled only if there is available capacity.
    - Generally, if `--max-num-batched-tokens` is set to a larger value, the overall latency will be lower, but the pressure on HBM memory (activation value usage) will be greater.
- `--gpu-memory-utilization` represents the proportion of HBM that vLLM will use for actual inference. Its essential function is to calculate the available kv_cache size. During the warm-up phase (referred to as profile run in vLLM), vLLM records the peak HBM memory usage during an inference process with an input size of `--max-num-batched-tokens`. The available kv_cache size is then calculated as: `--gpu-memory-utilization` * HBM size - peak HBM memory usage. Therefore, the larger the value of `--gpu-memory-utilization`, the more kv_cache can be used. However, since the HBM memory usage during the warm-up phase may differ from that during actual inference (e.g., due to uneven EP load), setting `--gpu-memory-utilization` too high may lead to OOM (Out of Memory) issues during actual inference. The default value is `0.9`.
- `--no-enable-prefix-caching` indicates that prefix caching is disabled. The current implementation of hybrid kv cache for Qwen3.5-27B / Qwen3.6-27B may result in a very large effective `block_size` when prefix caching is enabled (e.g., 2048), which means any prefix shorter than `block_size` will never be cached. If your workload has many short repeated prefixes, consider keeping prefix caching disabled. For related issues, see the [Public FAQ](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html).
- `--quantization ascend` indicates that quantization is used. To disable quantization, remove this option.
- `--speculative-config` uses `qwen3_5_mtp` for both `Qwen3.5-27B` and `Qwen3.6-27B` because they share the same MTP head design.
- `--compilation-config` contains configurations related to the aclgraph graph mode. The most significant configurations are `"cudagraph_mode"` and `"cudagraph_capture_sizes"`, which have the following meanings:
    - `"cudagraph_mode"`: represents the specific graph mode. Currently, `"PIECEWISE"` and `"FULL_DECODE_ONLY"` are supported. The graph mode is mainly used to reduce the cost of operator dispatch. Currently, `"FULL_DECODE_ONLY"` is recommended.
    - `"cudagraph_capture_sizes"`: represents different levels of graph modes. The default value is `[1, 2, 4, 8, 16, 24, 32, 40,..., --max-num-seqs]`. In the graph mode, the input for graphs at different levels is fixed, and inputs between levels are automatically padded to the next level. Currently, the default setting is recommended. Only in some scenarios is it necessary to set this separately to achieve optimal performance.

Common Issues Tip: If you encounter issues, please refer to the [Public FAQ](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html) for troubleshooting.

Service Verification:

If the service starts successfully, the following startup log will be displayed:

```text
(APIServer pid=<pid>) INFO:     Started server process [<pid>]
(APIServer pid=<pid>) INFO:     Waiting for application startup.
(APIServer pid=<pid>) INFO:     Application startup complete.
```

For functional testing (e.g., `completions` and `chat.completions` curl examples with expected responses), please refer to [Section 6](#6-functional-verification).

### 5.2 Multi-Node PD Separation Deployment

For high-concurrency production scenarios, multi-node PD (Prefill-Decode) separation can be used to scale the service. The recommended approach is to use Mooncake for deployment: [Mooncake Multi-Node PD Disaggregation Guide](../features/pd_disaggregation_mooncake_multi_node.md).

In the standard single-node deployment mode, Prefill (prompt processing) and Decode (token generation) tasks run on the same set of NPUs. This can lead to two issues:

1. **Prefill preemption interrupts Decode**: Prefill is a compute-intensive task that processes the entire input context at once, while Decode generates tokens one by one. When a new user request arrives, its Prefill phase can preempt and interrupt ongoing Decode tasks, causing jitter and higher time-per-output-token (TPOT) latency.
2. **Inflexible resource allocation**: Prefill and Decode have fundamentally different computational characteristics — Prefill is compute-bound and memory-bandwidth-intensive, while Decode is memory-bandwidth-bound. Running them on the same hardware forces a compromise that satisfies neither optimally.

PD (Prefill-Decode) separation addresses these issues by running Prefill and Decode on dedicated node groups, each configured independently:

- **Prefill nodes** focus on high-throughput prompt processing, optimized for compute and communication (e.g., enabling FlashComm for Allreduce acceleration).
- **Decode nodes** focus on low-latency token generation, optimized for memory bandwidth (e.g., enabling async-scheduling and full-decode aclgraph).

For `Qwen3.5-27B-w8a8` and `Qwen3.6-27B-w8a8`, a typical **1P1D** configuration requires **2 Atlas 800 A3 (64G × 16) nodes** (1 Prefill node + 1 Decode node), with **TP=2** and **DP=8** on each node, which fully utilizes all 16 NPUs of an Atlas A3. The example below uses `Qwen3.5-27B-w8a8`; for `Qwen3.6-27B-w8a8`, replace the model path with `Eco-Tech/Qwen3.6-27B-w8a8` and adjust `--served-model-name` to `qwen3.6` (and `--max-model-len` to 262144 if needed).

> **Why TP=2 + DP=8 (DP-first strategy)?** The `Qwen3.5-27B-w8a8` (and `Qwen3.6-27B-w8a8`) model is only ~30 GB, which easily fits in a single NPU (each NPU has 64 GB HBM). **TP > 1 is mainly needed for models that do not fit in one NPU.** For a 27 B model, `TP=2` is sufficient to balance operator-dispatch overhead across NPUs, while **maximizing DP** keeps all 16 NPUs of an Atlas A3 busy with independent request batches, fully utilizing the hardware. This **DP-first parallelism strategy** is the standard practice for small dense models (e.g., Qwen3.5-27B, Qwen3.6-27B, Llama-3-8B) and has been validated by the [Qwen3.5-27B B200 benchmark](https://thenextgentechinsider.com/pulse/qwen-35-27b-delivers-11m-tokenssecond-on-nvidia-b200-cluster), where switching from TP=8 to DP=8 lifted per-node throughput from 9.5k to 95k tokens/s.
>
> **Note**: Since `Qwen3.5-27B` and `Qwen3.6-27B` fit in a single node, multi-node PD separation is only recommended for high-concurrency production deployments. For the Mooncake deployment specifics, please refer to the [Mooncake Multi-Node PD Disaggregation Guide](../features/pd_disaggregation_mooncake_multi_node.md).

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

    vllm serve Eco-Tech/Qwen3.5-27B-w8a8-mtp \
      --host 0.0.0.0 \
      --port $2 \
      --data-parallel-size $3 \
      --data-parallel-rank $4 \
      --data-parallel-address $5 \
      --data-parallel-rpc-port $6 \
      --tensor-parallel-size $7 \
      --seed 1024 \
      --quantization ascend \
      --served-model-name qwen3.5 \
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

    vllm serve Eco-Tech/Qwen3.5-27B-w8a8-mtp \
      --host 0.0.0.0 \
      --port $2 \
      --data-parallel-size $3 \
      --data-parallel-rank $4 \
      --data-parallel-address $5 \
      --data-parallel-rpc-port $6 \
      --tensor-parallel-size $7 \
      --seed 1024 \
      --quantization ascend \
      --served-model-name qwen3.5 \
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
        "model": "qwen3.5",
        "messages": [
            {"role": "user", "content": "The future of AI is"}
        ],
        "max_tokens": 1024,
        "temperature": 1.0,
        "top_p": 0.95
    }'
```

> **Note**: For `Qwen3.6-27B-w8a8`, change the `model` field above to `"qwen3.6"` and the `--served-model-name` of the Prefill/Decode nodes to `qwen3.6`.

Expected Result: The proxy returns HTTP 200 OK. The JSON response contains the `choices` field with the generated text, confirming that Prefill nodes have successfully processed the prompt and Decode nodes have generated the response.

Common Issues Tip: If you encounter issues with PD separation deployment, please refer to the [Public FAQ](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html) for troubleshooting.

## 6 Functional Verification

After the service is started, the model can be invoked by sending a prompt. Two API interfaces are supported: `completions` and `chat.completions`. Use the `--served-model-name` you configured (`qwen3.5` for `Qwen3.5-27B` or `qwen3.6` for `Qwen3.6-27B`).

**Completions API:**

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen3.5",
        "prompt": "The future of AI is",
        "max_completion_tokens": 50,
        "temperature": 0
    }'
```

> **Note**: For `Qwen3.6-27B`, set `"model": "qwen3.6"` in the request body.

**Chat Completions API:**

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen3.5",
        "messages": [
            {"role": "user", "content": "The future of AI is"}
        ],
        "max_completion_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.95
    }'
```

> **Note**: For `Qwen3.6-27B`, set `"model": "qwen3.6"` in the request body.

Expected Result: The service returns HTTP 200 OK. The JSON response contains the `choices` field with generated text. Example output for the completions API (content truncated for brevity):

```json
{
    "id": "cmpl-xxxxxxxxxxxxx",
    "object": "text_completion",
    "created": 1780971952,
    "model": "qwen3.5",
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

2. After execution, you can get the result. Here is the result of `Qwen3.5-27B-w8a8` in `vllm-ascend:v0.17.0rc1` for reference only. The accuracy result of `Qwen3.6-27B-w8a8` can be obtained in the same way and is not listed here.

| dataset | version | metric | mode | vllm-api-general-chat |
|----- | ----- | ----- | ----- | -----|
| gsm8k | - | accuracy | gen | 96.74 |

### Using Language Model Evaluation Harness

Using the `gsm8k` dataset as an example test dataset, run the accuracy evaluation for `Qwen3.5-27B-w8a8` in online mode.

1. For `lm_eval` installation, please refer to [Using lm_eval](../../developer_guide/evaluation/using_lm_eval.md).
2. Run `lm_eval` to execute the accuracy evaluation.

```shell
# For Qwen3.5-27B-w8a8
export VLLM_USE_MODELSCOPE=True
vllm serve Eco-Tech/Qwen3.5-27B-w8a8-mtp \
    --served-model-name qwen3.5 \
    --trust-remote-code \
    --quantization ascend \
    --tensor-parallel-size 2 \
    --max-model-len 133000 \
    --max-num-seqs 32 \
    --gpu-memory-utilization 0.90 \
    --no-enable-prefix-caching

# Run lm_eval in another terminal
lm_eval \
  --model local-completions \
  --model_args model=qwen3.5,base_url=http://127.0.0.1:8000/v1/completions,tokenized_requests=False,trust_remote_code=True \
  --tasks gsm8k \
  --output_path ./
```

## 8 Performance Evaluation

### Using AISBench

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

### Using vLLM Benchmark

Run performance evaluation of `Qwen3.5-27B-w8a8` or `Qwen3.6-27B-w8a8` as an example.

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/benchmarking/) for more details.

There are three `vllm bench` subcommands:

- `latency`: Benchmark the latency of a single batch of requests.
- `serve`: Benchmark the online serving throughput.
- `throughput`: Benchmark offline inference throughput.

Take the `serve` as an example. Run the code as follows.

```bash
export VLLM_USE_MODELSCOPE=True
# For Qwen3.5-27B-w8a8:
vllm bench serve --model Eco-Tech/Qwen3.5-27B-w8a8-mtp --dataset-name random --random-input 200 --num-prompts 200 --request-rate 1 --save-result --result-dir ./
# For Qwen3.6-27B-w8a8:
vllm bench serve --model Eco-Tech/Qwen3.6-27B-w8a8 --dataset-name random --random-input 200 --num-prompts 200 --request-rate 1 --save-result --result-dir ./
```

After about several minutes, you can get the performance evaluation result.

## 9 Performance Tuning

### 9.1 Recommended Configurations

> **Note**: The following configurations are validated in specific test environments and are for reference only. The optimal configuration depends on factors such as maximum input/output length, prefix cache hit rate, precision requirements, and deployment machine ratios. It is recommended to refer to [Section 9.2](#92-tuning-guidelines) for tuning based on actual conditions.
>
> **Parallelism Strategy**: `Qwen3.5-27B-w8a8` and `Qwen3.6-27B-w8a8` are only ~30 GB and easily fit in a single NPU (64 GB HBM per NPU). Following the **DP-first** principle, **TP=2 is the recommended default** for most scenarios, and the remaining NPUs should be allocated to DP for parallel request batches. **TP=8 is only recommended for ultra-long context (256K+) scenarios**, where it shards the KV cache across 8 NPUs to maximize the available context window per rank. For `Qwen3.6-27B-w8a8`, you can also raise `--max-model-len` up to 262144 in the same TP/DP layout.

#### Table 1: Scenario Overview

| Scenario | Deployment Mode | *Total NPUs | Weight Version | Key Considerations |
|----------|----------------|-------------|----------------|---------------------|
| High Throughput<br>(128K context) | Single-Node (A2) | 8 (A2) | Qwen3.5-27B-w8a8 / Qwen3.6-27B-w8a8 | TP=2 + DP=4 fully utilizes all 8 NPUs for parallel request batches |
| High Throughput<br>(128K context) | Single-Node (A3) | 16 (A3) | Qwen3.5-27B-w8a8 / Qwen3.6-27B-w8a8 | TP=2 + DP=8 fully utilizes all 16 NPUs for parallel request batches |
| Low Latency<br>(128K context) | Single-Node (A3) | 16 (A3) | Qwen3.5-27B-w8a8 / Qwen3.6-27B-w8a8 | TP=2 + DP=8 reduces per-layer Allreduce overhead for small interactive batches |
| Long Context<br>(256K+ context) | Single-Node (A3) | 16 (A3) | Qwen3.5-27B-w8a8 / Qwen3.6-27B-w8a8 | TP=8 + DP=2 shards the KV cache across 8 NPUs to maximize the available context window |

> `*Total NPUs` indicates the total number of NPUs used across all nodes. 1 Atlas 800 A3 node = 16 NPUs, 1 Atlas 800 A2 node = 8 NPUs.

#### Table 2: Detailed Node Configuration

| Scenario | Configuration | NPUs | TP | DP | Max Num Seqs | Max Num Batched Tokens | Max Model Len | MTP Speculation Num | Async Scheduling |
|----------|---------------|-------|----|----|----|-------------|--------------------|---------------------|------------------|
| High Throughput (128K) | Single-Node (A2) | 8 | 2 | 4 | 32 | 16384 | 133000 | 3 | On |
| High Throughput (128K) | Single-Node (A3) | 16 | 2 | 8 | 32 | 16384 | 133000 | 3 | On |
| Low Latency (128K) | Single-Node (A3) | 16 | 2 | 8 | 4 | 4096 | 133000 | 3 | On |
| Long Context (256K+) | Single-Node (A3) | 16 | 8 | 2 | 8 | 8192 | 266000 | 3 | On |

> For complete startup commands and parameter descriptions, please refer to the deployment examples in [Chapter 5](#5-online-service-deployment).

### 9.2 Tuning Guidelines

#### 9.2.1 General Tuning Reference

Please refer to the [Public Performance Tuning Documentation](../../developer_guide/performance_and_debug/optimization_and_tuning.md) for tuning methods.
Please refer to the [Feature Guide](../../user_guide/support_matrix/feature_matrix.md) for detailed feature descriptions.

## 10 FAQ

For common environment, installation, and general parameter issues, please refer to the [vLLM-Ascend Public FAQ](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html).
