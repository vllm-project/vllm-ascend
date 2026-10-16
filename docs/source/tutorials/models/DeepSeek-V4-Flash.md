# DeepSeek-V4-Flash

## 1 Introduction

!!! note

    The "PD Separation Deployment" mentioned in this document is provided only as an example/reference design to demonstrate the basic process of deploying Prefill and Decode in separated mode. It is not recommended for use in the production environment.
    
    **For the production environment on the customer side, [MindIE Motor](https://www.hiascend.com/document/detail/en/mindie/latest/quickstart/docs/en/user_guide/quick_start.md)  + vLLM-Ascend is recommended.**

DeepSeek-V4 introduces several key upgrades over DeepSeek-V3:

- The Manifold-Constrained Hyper-Connections (mHC) to strengthen conventional residual connections.
- A hybrid attention architecture, which greatly improves long-context efficiency through Compress-4-Attention and Compress-128-Attention. For the Mixture-of-Experts (MoE) components, it still adopts the DeepSeekMoE architecture, with only minor adjustments.

DeepSeek-V4-Flash is the lightweight variant of the DeepSeek-V4 family, suitable for high-throughput and low-latency serving scenarios.

This document will show the main verification steps of the model, including supported features, feature configuration, environment preparation, single-node and multi-node deployment, accuracy and performance evaluation.

## 2 Supported Features

Refer to [Supported Features List](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [Feature Guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## 3 Prerequisites

### 3.1 Model Weight

|  Weight Version                              | Hardware Requirements                                                      | Download Links |
|----------------------------------------------|----------------------------------------------------------------------------|----------------|
| `DeepSeek-V4-Flash-w8a8-mtp` (Quantized version) | 1 Atlas 800 A3 (128GB × 8) node or 1 Atlas 800 A2 (64GB × 8) node | [ModelScope](https://www.modelscope.cn/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp) |
| `DeepSeek-V4-Flash-0731-w8a8` | | [ModelScope](https://www.modelscope.cn/models/Eco-Tech/DeepSeek-V4-Flash-0731-w8a8) |
| `DeepSeek-V4-Flash-0731-w4a4c8` | 950PR&950DT Products | Quantize locally with [msmodelslim](https://gitcode.com/Ascend/msmodelslim/blob/master/lab_practice/deepseek_v4/deepseek_v4_flash_w4a4c8.yaml) |
| `DeepSeek-V4-Flash-0731` | 950DT products (96GB × 8); examples cover four-NPU PD-colocated and 4-Prefill/4-Decode PD-separated deployments | [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) |

- `DeepSeek-V4-Flash-0731-w4a4c8` (W4A4 MXFP4 mixed-precision quantization): For how the ultra-low-bit quantization preserves accuracy, see the [W4A4C8 FAQ](#q-how-do-the-w4a4c8-ultra-low-bit-quantized-weights-preserve-accuracy).

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`.

>**Path description**: Download the model weights to a directory of your choice and record it. Ensure the model path in the subsequent deployment command matches this directory.

### 3.2 Verify Multi-node Communication (Optional)

If you want to deploy a multi-node environment, you need to verify multi-node communication according to [verify multi-node communication environment](../../getting_started/installation.md#installation-multi-node-interconnect).

## 4 Installation

### 4.1 Docker Image Installation

Select an image based on your machine type and start the docker image on your node, refer to [using docker](../../getting_started/installation.md#installation-prebuilt-image).

**Attention**: DSpark is supported on A2 and A3 in vLLM Ascend `v0.25.0` and later. Use `quay.io/ascend/vllm-ascend:DeepSeekV4-flash-0731` for A2, `quay.io/ascend/vllm-ascend:DeepSeekV4-flash-0731-a3` for A3, and the 950DT image shown below for 950DT products.

=== "Ascend 950DT series"

    Start the Docker container on each server.

    ```bash
    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-950dt
    export NAME=vllm-ascend

    docker run --rm \
        --name $NAME \
        --net=host \
        --shm-size=512g \
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
        -itd $IMAGE bash
    ```

=== "A3 series"

    Start the docker image on each node.

    ```bash

    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-a3
    docker run --rm \
        --name vllm-ascend \
        --shm-size=512g \
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
        -v /etc/hccn.conf:/etc/hccn.conf \
        -v /root/.cache:/root/.cache \
        -it $IMAGE bash
    ```

=== "A2 series"

    Start the docker image on each node.

    ```bash
    # deepseek-v4-flash uses the following image
    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}

    # deepseek-v4-flash-dspark uses the following image
    export IMAGE=quay.io/ascend/vllm-ascend:nightly-main

    docker run --rm \
        --name vllm-ascend \
        --shm-size=512g \
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
        -v /etc/hccn.conf:/etc/hccn.conf \
        -v /root/.cache:/root/.cache \
        -it $IMAGE bash
    ```

After a successful docker run, you can verify the running container service by executing the `docker ps` command.

### 4.2 Source Code Installation

If you don't want to use the docker image as above, you can also build all from source:

- Install `vllm-ascend` from source, refer to [installation](../../getting_started/installation.md).

If you want to deploy a multi-node environment, you need to set up the environment on each node.

## 5 Online Service Deployment {: #5-online-service-deployment }

!!! note

    In this tutorial, we suppose you downloaded the model weight to `/root/.cache/modelscope/hub/models/vllm-ascend/`. Feel free to change it to your own path.

    It is recommended that the following service code be encapsulated in a .sh script file and executed in Bash mode.

### 5.1 Single-Node Online Deployment

Single-node deployment completes both Prefill and Decode within the same node. The quantized model `DeepSeek-V4-Flash-w8a8-mtp` can be deployed on 1 Atlas 800 A3 (128GB × 8) or 1 Atlas 800 A2 (64GB × 8).

=== "A2 series"

    Run the following script to execute online inference.

    ```shell
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=10
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export HCCL_BUFFSIZE=1024
    export TASK_QUEUE_ENABLE=1
    export HCCL_OP_EXPANSION_MODE="AIV"

    # Ensure the model path matches the directory recorded during download
    vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V4-Flash-w8a8-mtp \
        --max-model-len 133120 \
        --max-num-batched-tokens 8192 \
        --served-model-name dsv4 \
        --gpu-memory-utilization 0.9 \
        --max-num-seqs 32 \
        --data-parallel-size 1 \
        --tensor-parallel-size 8 \
        --enable-expert-parallel \
        --tokenizer-mode deepseek_v4 \
        --tool-call-parser deepseek_v4 \
        --enable-auto-tool-choice \
        --reasoning-parser deepseek_v4 \
        --no-enable-prefix-caching \
        --model-loader-extra-config='{"enable_multithread_load": true, "num_threads": 128}' \
        --quantization ascend \
        --port 8000 \
        --block-size 128 \
        --attention_config.indexer_kv_dtype int8 \
        --speculative-config '{"num_speculative_tokens": 1,"method": "mtp","enforce_eager": true}' \
        --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
        --additional-config '
        {"ascend_compilation_config":{
            "enable_npugraph_ex":true,
            "enable_static_kernel":false
            },
        "enable_cpu_binding": true,
        "enable_dsa_cp": true,
        "multistream_overlap_shared_expert": true}'
    ```

=== "A2 series with dspark"

    Run the following script to execute online inference.

    ```shell
    export OMP_NUM_THREADS=10
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export HCCL_BUFFSIZE=1024
    export TASK_QUEUE_ENABLE=1
    export HCCL_OP_EXPANSION_MODE=AIV
    
    # Ensure the model path matches the directory recorded during download
    vllm serve /root/.cache/modelscope/hub/models/UploadWeight/DeepSeek-V4-Flash-DSpark-w4a8-test \
        --max-model-len 800000 \
        --max-num-batched-tokens 8192 \
        --served-model-name dsv4 \
        --gpu-memory-utilization 0.9 \
        --max-num-seqs 32 \
        --data-parallel-size 1 \
        --tensor-parallel-size 8 \
        --enable-expert-parallel \
        --tokenizer-mode deepseek_v4 \
        --tool-call-parser deepseek_v4 \
        --enable-auto-tool-choice \
        --reasoning-parser deepseek_v4 \
        --no-disable-hybrid-kv-cache-manager \
        --model-loader-extra-config='{"enable_multithread_load": true, "num_threads": 128}' \
        --quantization ascend \
        --port 8000 \
        --block-size 128 \
        --attention_config.indexer_kv_dtype int8 \
        --speculative-config '{"method": "dspark", "num_speculative_tokens": 7, "enforce_eager": true}'  \
        --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
    ```
    tps more than 50+ ,its reach  2X speed of dsv4f with mtp

=== "A3 series"

    Run the following script to execute online inference.

    ```shell
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=10
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export HCCL_BUFFSIZE=1024
    export TASK_QUEUE_ENABLE=1
    export HCCL_OP_EXPANSION_MODE="AIV"
    export VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096

    # Ensure the model path matches the directory recorded during download
    vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V4-Flash-w8a8-mtp \
        --max-model-len 1048576 \
        --max-num-batched-tokens 10240 \
        --served-model-name dsv4 \
        --gpu-memory-utilization 0.9 \
        --api-server-count 1 \
        --max-num-seqs 64 \
        --data-parallel-size 4 \
        --tensor-parallel-size 4 \
        --enable-expert-parallel \
        --tokenizer-mode deepseek_v4 \
        --tool-call-parser deepseek_v4 \
        --enable-auto-tool-choice \
        --reasoning-parser deepseek_v4 \
        --model-loader-extra-config='{"enable_multithread_load": true, "num_threads": 128}' \
        --quantization ascend \
        --port 8000 \
        --block-size 32 \
        --attention_config.indexer_kv_dtype int8 \
        --speculative-config '{"num_speculative_tokens": 1,"method": "mtp","enforce_eager": true}' \
        --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
        --additional-config '
        {"ascend_compilation_config":{
            "enable_npugraph_ex": true,
            "enable_static_kernel": false
            },
        "enable_cpu_binding": true,
        "multistream_overlap_shared_expert": true}'
    ```

=== "A3 series with DSpark"

    Run the following script to execute online inference.

    ```shell
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=10
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export HCCL_BUFFSIZE=1024
    export TASK_QUEUE_ENABLE=1
    export HCCL_OP_EXPANSION_MODE="AIV"
    export VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096

    # Ensure the model path matches the directory recorded during download
    vllm serve /path/to/DeepSeek-V4-Flash-0731-w8a8 \
        --max-model-len 1048576 \
        --max-num-batched-tokens 10240 \
        --served-model-name dsv4 \
        --gpu-memory-utilization 0.9 \
        --max-num-seqs 64 \
        --data-parallel-size 4 \
        --tensor-parallel-size 4 \
        --enable-expert-parallel \
        --tokenizer-mode deepseek_v4 \
        --tool-call-parser deepseek_v4 \
        --enable-auto-tool-choice \
        --reasoning-parser deepseek_v4 \
        --model-loader-extra-config='{"enable_multithread_load": true, "num_threads": 128}' \
        --quantization ascend \
        --port 8000 \
        --block-size 32 \
        --attention_config.indexer_kv_dtype int8 \
        --speculative-config '{"method":"dspark","num_speculative_tokens":7,"enforce_eager":true}' \
        --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
        --additional-config '{
            "ascend_compilation_config": {
                "enable_npugraph_ex": true,
                "enable_static_kernel": false
            },
            "enable_cpu_binding": true,
            "enable_dsa_cp": true,
            "multistream_overlap_shared_expert": true
        }'
    ```

=== "950DT series with DSpark"

    Run the following script to execute PD-colocated online inference.

    ```shell
    #!/usr/bin/env bash

    source /root/.bashrc
    source /usr/local/Ascend/ascend-toolkit/set_env.sh

    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/usr/local/lib/"
    export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'
    export VLLM_USE_V2_MODEL_RUNNER=1

    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=10
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
    export TASK_QUEUE_ENABLE=1
    export HCCL_OP_EXPANSION_MODE=AIV
    export HCCL_BUFFSIZE=1024
    export ASCEND_PROCESS_LOG_PATH="${RUNTIME_DIR}/logs"

  
    vllm serve /root/.cache/DeepSeek-V4-Flash-0731  \
        --port 8000 \
        --served-model-name dsv4 \
        --data-parallel-size 2 \
        --prefill-context-parallel-size 2 \
        --distributed-executor-backend mp \
        --enable-expert-parallel \
        --all2all-backend allgather_reducescatter \
        --enable-chunked-prefill \
        --max-model-len 1048576 \
        --max-num-batched-tokens 16384 \
        --max-num-seqs 32 \
        --gpu-memory-utilization 0.95 \
        --block-size 128 \
        --safetensors-load-strategy prefetch \
        --tokenizer-mode deepseek_v4 \
        --tool-call-parser deepseek_v4 \
        --enable-auto-tool-choice \
        --reasoning-parser deepseek_v4 \
        --api-server-count 1 \
        --kv-cache-dtype fp8 \
        --attention_config.indexer_kv_dtype fp8 \
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
        --no-disable-hybrid-kv-cache-manager \
        --profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${PROFILE_DIR}\",\"torch_profiler_with_stack\":false}" \
        --additional-config '{"enable_cpu_binding":"True","multistream_overlap_shared_expert":true,"enable_flashcomm1":true,"enable_shared_expert_dp":true}' \
        --speculative-config '{"num_speculative_tokens":5,"method":"dspark"}'
    ```

Key Parameter Descriptions:

- `--max-model-len` specifies the maximum context length - that is, the sum of input and output tokens for a single request. Adjust it according to your actual scenario.
- `--max-num-seqs` indicates the maximum number of requests that each DP group is allowed to process. If the number of requests sent to the service exceeds this limit, the excess requests will remain in a waiting state and will not be scheduled. Note that the time spent in the waiting state is also counted in metrics such as TTFT and TPOT. Therefore, when testing performance, it is generally recommended that `--max-num-seqs` * `--data-parallel-size` >= the actual total concurrency.
- `--max-num-batched-tokens` is the maximum number of tokens processed in one scheduler step. A larger value can improve prefill efficiency but consumes more activation memory.
- `--data-parallel-size` sets the global number of data parallel ranks, while `--tensor-parallel-size` sets the tensor parallel size within each DP rank. Configure them together according to the deployment topology and available NPUs.
- `--enable-expert-parallel` enables expert parallelism for MoE layers. Do not mix MoE tensor parallelism and expert parallelism in the same MoE layer.
- `--no-enable-prefix-caching` indicates that prefix caching is disabled. To enable it, remove this option.
- `--block-size` sets the KV cache block size. To enable the experimental 4k prefix cache hit support, change it from `128` to `32`.
- `--quantization ascend` enables Ascend quantization for the W8A8 model.
- `--kv-cache-dtype turboquant_4bit_nc` enables 4-bit TurboQuant quantization for DeepSeek-V4's Compress-4 attention KV cache to reduce memory usage. It is suited to ultra-long-context workloads such as AI coding, especially when KV cache memory limits concurrency. It is independent of weight and indexer cache quantization and requires compatible TurboQuant and fused attention operators. P/D disaggregated deployments are not currently supported.
- `--speculative-config` configures speculative decoding to accelerate inference. Use `mtp` for Multi-Token Prediction (MTP) and `dspark` for DSpark models. When using DSpark, `num_speculative_tokens` must be at least 5 (check the checkpoint's `config.json`).
- `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'` enables full ACL graph execution in the decode phase to reduce scheduling latency.
- `--additional-config` enables Ascend-specific optimizations. `enable_npugraph_ex` enables enhanced ACL graph execution, `enable_static_kernel: false` keeps static-kernel compilation disabled, `enable_cpu_binding` enables Ascend-native CPU binding, `enable_dsa_cp` enables DSA context parallelism, and `multistream_overlap_shared_expert` overlaps shared expert computation for better MoE throughput.
- `VLLM_PREFIX_CACHE_RETENTION_INTERVAL`: Controls the retention interval, in tokens, for prefix-cache checkpoints of hybrid attention layers. It is applicable to DeepSeek-V4 and takes effect only when prefix caching is enabled. Under KV-cache pressure, it can improve the effective prefix-cache hit rate for reusable long prefixes. The value must be a non-negative multiple of `--block-size`; for DeepSeek-V4-Flash, 128 times `--block-size` is recommended. Set it to `4096` when `--block-size` is `32`, or `16384` when `--block-size` is `128`.

Common Issues Tip: If you encounter issues, please refer to the [Public FAQs](../../faqs.md) for troubleshooting.

Service Verification:

```shell
curl http://<node0_ip>:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "dsv4",
        "messages": [
            {
                "role": "user",
                "content": "Who are you?"
            }
        ],
        "max_tokens": 256,
        "temperature": 0
    }'
```

Expected Result:

The service returns HTTP 200 OK with a JSON response containing the `choices` field.

### 5.2 Multi-Node PD Separation Deployment

For standard PD disaggregation, we recommend using [Mooncake](../features/pd_disaggregation_mooncake_multi_node.md). Section 5.2.3 describes the Ascend 950DT deployment, and Section 5.3 extends that topology with a Memcache-backed KV Cache Pool.

In the standard single-node deployment mode, Prefill (prompt processing) and Decode (token generation) tasks run on the same set of NPUs. This can lead to two issues:

1. **Prefill preemption interrupts Decode**: Prefill is a compute-intensive task that processes the entire input context at once, while Decode generates tokens one by one. When a new user request arrives, its Prefill phase can preempt and interrupt ongoing Decode tasks, causing jitter and higher time-per-output-token (TPOT) latency.
2. **Inflexible resource allocation**: Prefill and Decode have fundamentally different computational characteristics — Prefill is compute-bound and memory-bandwidth-intensive, while Decode is memory-bandwidth-bound. Running them on the same hardware forces a compromise that satisfies neither optimally.

PD (Prefill-Decode) separation addresses these issues by running Prefill and Decode on dedicated node groups, each configured independently. This architecture is recommended for production deployments with concurrent multi-user workloads, where stable latency and high throughput are both required.

The following sections describe PD separation deployment on Atlas 800 A3 (128GB × 8), Atlas 800 A2 (64GB × 8), and Ascend 950DT (96GB × 8) multi-node environments.

#### 5.2.1 A3 Series PD Separation Deployment

This section shows the deployment guide of DeepSeek-V4-Flash on Atlas 800 A3 (128GB × 8) multi-node environment with 1P1D for better performance.

Before you start, please:

1. Prepare the script `launch_online_dp.py` on each node.

    ```python
    import argparse
    import multiprocessing
    import os
    import subprocess
    import sys

    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--dp-size",
            type=int,
            required=True,
            help="Data parallel size."
        )
        parser.add_argument(
            "--tp-size",
            type=int,
            default=1,
            help="Tensor parallel size."
        )
        parser.add_argument(
            "--dp-size-local",
            type=int,
            default=-1,
            help="Local data parallel size."
        )
        parser.add_argument(
            "--dp-rank-start",
            type=int,
            default=0,
            help="Starting rank for data parallel."
        )
        parser.add_argument(
            "--dp-address",
            type=str,
            required=True,
            help="IP address for data parallel master node."
        )
        parser.add_argument(
            "--dp-rpc-port",
            type=str,
            default=12345,
            help="Port for data parallel master node."
        )
        parser.add_argument(
            "--vllm-start-port",
            type=int,
            default=9000,
            help="Starting port for the engine."
        )
        return parser.parse_args()

    args = parse_args()
    dp_size = args.dp_size
    tp_size = args.tp_size
    dp_size_local = args.dp_size_local
    if dp_size_local == -1:
        dp_size_local = dp_size
    dp_rank_start = args.dp_rank_start
    dp_address = args.dp_address
    dp_rpc_port = args.dp_rpc_port
    vllm_start_port = args.vllm_start_port

    def run_command(visible_devices, dp_rank, vllm_engine_port):
        command = [
            "bash",
            "./run_dp_template.sh",
            visible_devices,
            str(vllm_engine_port),
            str(dp_size),
            str(dp_rank),
            dp_address,
            dp_rpc_port,
            str(tp_size),
        ]
        subprocess.run(command, check=True)

    if __name__ == "__main__":
        template_path = "./run_dp_template.sh"
        if not os.path.exists(template_path):
            print(f"Template file {template_path} does not exist.")
            sys.exit(1)

        processes = []
        num_cards = dp_size_local * tp_size
        for i in range(dp_size_local):
            dp_rank = dp_rank_start + i
            vllm_engine_port = vllm_start_port + i
            visible_devices = ",".join(str(x) for x in range(i * tp_size, (i + 1) * tp_size))
            process = multiprocessing.Process(target=run_command,
                                            args=(visible_devices, dp_rank,
                                                    vllm_engine_port))
            processes.append(process)
            process.start()

        for process in processes:
            process.join()
    ```

    Parameter descriptions:

    |Parameter|Type|Required|Default|Description|
    |---------|----|--------|-------|-----------|
    |`--dp-size`|int|Yes|-|Data parallel size (total number of DP ranks across all nodes).|
    |`--tp-size`|int|No|1|Tensor parallel size within each DP rank.|
    |`--dp-size-local`|int|No|(same as `--dp-size`)|Number of DP ranks on the current node. If not set, defaults to `--dp-size`.|
    |`--dp-rank-start`|int|No|0|Starting rank offset for data parallel ranks on this node.|
    |`--dp-address`|str|Yes|-|IP address of the data parallel master node.|
    |`--dp-rpc-port`|str|No|12345|RPC port for data parallel master communication.|
    |`--vllm-start-port`|int|No|9000|Starting port for each vLLM engine instance on this node.|

2. Prepare the script `run_dp_template.sh` on each node.

=== "A3 series"

    1. Prefill node

        ```shell
        nic_name="xxxx" # change to your own nic name
        local_ip=xx.xx.xx.1 # change to your own ip

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
        export HCCL_BUFFSIZE=2560
        export TASK_QUEUE_ENABLE=1
        export HCCL_OP_EXPANSION_MODE="AIV"
        export ASCEND_RT_VISIBLE_DEVICES=$1
        export VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096

        # Ensure the model path matches the directory recorded during download
        vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V4-Flash-w8a8-mtp \
            --host 0.0.0.0 \
            --port $2 \
            --data-parallel-size $3 \
            --data-parallel-rank $4 \
            --data-parallel-address $5 \
            --data-parallel-rpc-port $6 \
            --tensor-parallel-size $7 \
            --enable-expert-parallel \
            --seed 1024 \
            --served-model-name dsv4 \
            --max-model-len 1048576 \
            --max-num-batched-tokens 8192 \
            --max-num-seqs 16 \
            --no-disable-hybrid-kv-cache-manager \
            --model-loader-extra-config='{"enable_multithread_load": true, "num_threads": 128}' \
            --speculative-config '{"num_speculative_tokens": 1,"method": "mtp","enforce_eager": true}' \
            --trust-remote-code \
            --block-size 32 \
            --attention_config.indexer_kv_dtype int8 \
            --tokenizer-mode deepseek_v4 \
            --tool-call-parser deepseek_v4 \
            --enable-auto-tool-choice \
            --reasoning-parser deepseek_v4 \
            --gpu-memory-utilization 0.9 \
            --quantization ascend \
            --enforce-eager \
            --additional-config '{"enable_cpu_binding": true, "enable_shared_expert_dp": true,  "enable_dsa_cp": true}' \
            --kv-transfer-config \
            '{"kv_connector": "MooncakeHybridConnector",
            "kv_role": "kv_producer",
            "kv_port": "30000",
            "engine_id": "0",
            "kv_connector_extra_config": {
                        "prefill": {
                                "dp_size": 4,
                                "tp_size": 4
                        },
                        "decode": {
                                "dp_size": 16,
                                "tp_size": 1
                        }
                }
            }'
        ```

    2. Decode node

        ```shell
        nic_name="xxxx" # change to your own nic name
        local_ip=xx.xx.xx.2 # change to your own ip

        export HCCL_OP_EXPANSION_MODE="AIV"
        export TASK_QUEUE_ENABLE=1
        export VLLM_RPC_TIMEOUT=3600000
        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
        export HCCL_EXEC_TIMEOUT=204
        export HCCL_CONNECT_TIMEOUT=1200
        export HCCL_IF_IP=$local_ip
        export GLOO_SOCKET_IFNAME=$nic_name
        export TP_SOCKET_IFNAME=$nic_name
        export HCCL_SOCKET_IFNAME=$nic_name
        export OMP_PROC_BIND=false
        export OMP_NUM_THREADS=10
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export HCCL_BUFFSIZE=1024
        export ASCEND_RT_VISIBLE_DEVICES=$1
        
        # Ensure the model path matches the directory recorded during download
        vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V4-Flash-w8a8-mtp \
            --host 0.0.0.0 \
            --port $2 \
            --data-parallel-size $3 \
            --data-parallel-rank $4 \
            --data-parallel-address $5 \
            --data-parallel-rpc-port $6 \
            --tensor-parallel-size $7 \
            --enable-expert-parallel \
            --seed 1024 \
            --served-model-name dsv4 \
            --max-model-len 1048576 \
            --max-num-batched-tokens 120 \
            --max-num-seqs 60 \
            --block-size 32 \
            --attention_config.indexer_kv_dtype int8 \
            --no-disable-hybrid-kv-cache-manager \
            --no-enable-prefix-caching \
            --trust-remote-code \
            --tokenizer-mode deepseek_v4 \
            --model-loader-extra-config='{"enable_multithread_load": true, "num_threads": 128}' \
            --tool-call-parser deepseek_v4 \
            --enable-auto-tool-choice \
            --reasoning-parser deepseek_v4 \
            --gpu-memory-utilization 0.9 \
            --quantization ascend \
            --speculative-config '{"num_speculative_tokens": 1,"method": "mtp","enforce_eager": true}' \
            --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
            --kv-transfer-config \
            '{"kv_connector": "MooncakeHybridConnector",
            "kv_role": "kv_consumer",
            "kv_port": "30100",
            "engine_id": "1",
            "kv_connector_extra_config": {
                        "prefill": {
                                "dp_size": 4,
                                "tp_size": 4
                        },
                        "decode": {
                                "dp_size": 16,
                                "tp_size": 1
                        }
                }
            }' \
            --additional-config '{
                "ascend_compilation_config":{
                    "enable_npugraph_ex":true,
                    "enable_static_kernel":false
                },
                "enable_cpu_binding":true,
                "multistream_overlap_shared_expert":true,
                "recompute_scheduler_enable":true
            }'
        ```

=== "A3 series with dspark"

    1. Prefill node

        ```shell
        nic_name="xxxx" # change to your own nic name
        local_ip=xx.xx.xx.1 # change to your own ip

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
        export HCCL_BUFFSIZE=1800
        export TASK_QUEUE_ENABLE=1
        export HCCL_OP_EXPANSION_MODE="AIV"
        export ASCEND_RT_VISIBLE_DEVICES=$1
        export VLLM_ASCEND_ENABLE_FUSED_MC2=1
        export VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096

        # Ensure the model path matches the directory recorded during download
        vllm serve /path/to/DeepSeek-V4-Flash-0731-w8a8 \
            --host 0.0.0.0 \
            --port $2 \
            --data-parallel-size $3 \
            --data-parallel-rank $4 \
            --data-parallel-address $5 \
            --data-parallel-rpc-port $6 \
            --tensor-parallel-size $7 \
            --enable-expert-parallel \
            --seed 1024 \
            --served-model-name dsv4 \
            --max-model-len 1048576 \
            --max-num-batched-tokens 8192 \
            --max-num-seqs 16 \
            --no-disable-hybrid-kv-cache-manager \
            --model-loader-extra-config='{"enable_multithread_load": true, "num_threads": 128}' \
            --speculative-config '{"num_speculative_tokens": 5,"method": "dspark","enforce_eager": true}' \
            --trust-remote-code \
            --block-size 32 \
            --attention_config.indexer_kv_dtype int8 \
            --tokenizer-mode deepseek_v4 \
            --tool-call-parser deepseek_v4 \
            --enable-auto-tool-choice \
            --reasoning-parser deepseek_v4 \
            --gpu-memory-utilization 0.9 \
            --quantization ascend \
            --enforce-eager \
            --additional-config '{"enable_cpu_binding": true, "enable_dsa_cp": true, "enable_flashcomm1":true}' \
            --kv-transfer-config \
            '{"kv_connector": "MooncakeHybridConnector",
            "kv_role": "kv_producer",
            "kv_port": "30000",
            "engine_id": "0",
            "kv_connector_extra_config": {
                        "prefill": {
                                "dp_size": 4,
                                "tp_size": 4
                        },
                        "decode": {
                                "dp_size": 16,
                                "tp_size": 1
                        }
                }
            }'
        ```

    2. Decode node

        ```shell
        nic_name="xxxx" # change to your own nic name
        local_ip=xx.xx.xx.2 # change to your own ip

        export HCCL_OP_EXPANSION_MODE="AIV"
        export TASK_QUEUE_ENABLE=1
        export VLLM_RPC_TIMEOUT=3600000
        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
        export HCCL_EXEC_TIMEOUT=204
        export HCCL_CONNECT_TIMEOUT=120
        export HCCL_IF_IP=$local_ip
        export GLOO_SOCKET_IFNAME=$nic_name
        export TP_SOCKET_IFNAME=$nic_name
        export HCCL_SOCKET_IFNAME=$nic_name
        export OMP_PROC_BIND=false
        export OMP_NUM_THREADS=10
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export HCCL_BUFFSIZE=2400
        export ASCEND_RT_VISIBLE_DEVICES=$1
        
        # Ensure the model path matches the directory recorded during download
        vllm serve /path/to/DeepSeek-V4-Flash-0731-w8a8 \
            --host 0.0.0.0 \
            --port $2 \
            --data-parallel-size $3 \
            --data-parallel-rank $4 \
            --data-parallel-address $5 \
            --data-parallel-rpc-port $6 \
            --tensor-parallel-size $7 \
            --enable-expert-parallel \
            --seed 1024 \
            --served-model-name dsv4 \
            --max-model-len 1048576 \
            --max-num-batched-tokens 480 \
            --max-num-seqs 60 \
            --async-scheduling \
            --block-size 32 \
            --attention_config.indexer_kv_dtype int8 \
            --no-disable-hybrid-kv-cache-manager \
            --no-enable-prefix-caching \
            --trust-remote-code \
            --tokenizer-mode deepseek_v4 \
            --model-loader-extra-config='{"enable_multithread_load": true, "num_threads": 128}' \
            --tool-call-parser deepseek_v4 \
            --enable-auto-tool-choice \
            --reasoning-parser deepseek_v4 \
            --gpu-memory-utilization 0.95 \
            --quantization ascend \
            --speculative-config '{"num_speculative_tokens": 5,"method": "dspark","enforce_eager": true}' \
            --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY","cudagraph_capture_sizes": [12, 24, 48, 96, 192, 256, 288, 360]}' \
            --kv-transfer-config \
            '{"kv_connector": "MooncakeHybridConnector",
            "kv_role": "kv_consumer",
            "kv_port": "30100",
            "engine_id": "1",
            "kv_connector_extra_config": {
                        "prefill": {
                                "dp_size": 4,
                                "tp_size": 4
                        },
                        "decode": {
                                "dp_size": 16,
                                "tp_size": 1
                        }
                }
            }' \
            --additional-config '{
                "ascend_compilation_config":{
                    "enable_npugraph_ex":true,
                    "enable_static_kernel":false
                },
                "enable_cpu_binding":true,
                "multistream_overlap_shared_expert":true,
                "recompute_scheduler_enable":true
            }'
        ```

3. Start the server with the following command on each node.

    1. Prefill node

        ```shell
        # change ip to your own
        python launch_online_dp.py --dp-size 4 --tp-size 4 --dp-size-local 4 --dp-rank-start 0 --dp-address xx.xx.xx.1 --dp-rpc-port 12321 --vllm-start-port 7100
        ```

    2. Decode node

        ```shell
        # change ip to your own
        python launch_online_dp.py --dp-size 16 --tp-size 1 --dp-size-local 16 --dp-rank-start 0 --dp-address xx.xx.xx.2 --dp-rpc-port 12321 --vllm-start-port 7100
        ```

4. Deploy the P-D disaggregation proxy.

    Refer to [Prefill-Decode Disaggregation (Deepseek)](../features/pd_disaggregation_mooncake_multi_node.md) to deploy the P-D disaggregation proxy.

#### 5.2.2 A2 Series PD Separation Deployment

This section shows the deployment guide of DeepSeek-V4-Flash on Atlas 800 A2 (64GB × 8) multi-node environment with 4\*1P 1\*4D for better performance.

Before you start, please:

1. Prepare the script `launch_online_dp.py` on each node.

    ```python
    import argparse
    import multiprocessing
    import os
    import subprocess
    import sys

    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--dp-size", type=int, required=True, help="Data parallel size.")
        parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size.")
        parser.add_argument("--dp-size-local", type=int, default=-1, help="Local data parallel size.")
        parser.add_argument("--dp-rank-start", type=int, default=0, help="Starting rank for data parallel.")
        parser.add_argument("--dp-address", type=str, required=True, help="IP address for data parallel master node.")
        parser.add_argument("--dp-rpc-port", type=str, default=12345, help="Port for data parallel master node.")
        parser.add_argument("--vllm-start-port", type=int, default=9000, help="Starting port for the engine.")
        return parser.parse_args()

    args = parse_args()
    dp_size = args.dp_size
    tp_size = args.tp_size
    dp_size_local = args.dp_size_local
    if dp_size_local == -1:
        dp_size_local = dp_size
    dp_rank_start = args.dp_rank_start
    dp_address = args.dp_address
    dp_rpc_port = args.dp_rpc_port
    vllm_start_port = args.vllm_start_port

    def run_command(visible_devices, dp_rank, vllm_engine_port):
        command = [
            "bash",
            "./run_dp_template.sh",
            visible_devices,
            str(vllm_engine_port),
            str(dp_size),
            str(dp_rank),
            dp_address,
            dp_rpc_port,
            str(tp_size),
        ]
        subprocess.run(command, check=True)

    if __name__ == "__main__":
        template_path = "./run_dp_template.sh"
        if not os.path.exists(template_path):
            print(f"Template file {template_path} does not exist.")
            sys.exit(1)

        processes = []
        num_cards = dp_size_local * tp_size
        for i in range(dp_size_local):
            dp_rank = dp_rank_start + i
            vllm_engine_port = vllm_start_port + i
            visible_devices = ",".join(str(x) for x in range(i * tp_size, (i + 1) * tp_size))
            process = multiprocessing.Process(target=run_command, args=(visible_devices, dp_rank, vllm_engine_port))
            processes.append(process)
            process.start()

        for process in processes:
            process.join()
    ```

2. Prepare the script `run_dp_template.sh` on each node.

    1. Prefill node (4 P nodes share the same script)

        For each P instance, only these two configuration values need to be modified: `kv_port` and `engine_id`. The `engine_id` should start from 0 and increment sequentially, while the `kv_port` (e.g., `30100`) must be unique for each P instance, such as 30000, 30100, etc.

        ```shell
        unset ftp_proxy
        unset https_proxy
        unset http_proxy
        rm -rf ~/ascend/log

        nic_name="xxxxxx" #eg."enp67s0f0np0"
        local_ip=`hostname -I|awk -F " " '{print$1}'`

        export HCCL_OP_EXPANSION_MODE="AIV"
        export TASK_QUEUE_ENABLE=1
        export VLLM_RPC_TIMEOUT=3600000
        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
        export HCCL_EXEC_TIMEOUT=204
        export HCCL_CONNECT_TIMEOUT=1200

        export HCCL_IF_IP=$local_ip
        export GLOO_SOCKET_IFNAME=$nic_name
        export TP_SOCKET_IFNAME=$nic_name
        export HCCL_SOCKET_IFNAME=$nic_name
        export OMP_PROC_BIND=false
        export OMP_NUM_THREADS=10
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export HCCL_BUFFSIZE=1024

        export ASCEND_RT_VISIBLE_DEVICES=$1
        export TASK_QUEUE_ENABLE=1

        # Ensure the model path matches the directory recorded during download
        vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V4-Flash-w8a8-mtp \
            --host 0.0.0.0 \
            --port $2 \
            --data-parallel-size $3 \
            --data-parallel-rank $4 \
            --data-parallel-address $5 \
            --data-parallel-rpc-port $6 \
            --tensor-parallel-size $7 \
            --enable-expert-parallel \
            --seed 1024 \
            --served-model-name dsv4 \
            --max-model-len 135000 \
            --max-num-batched-tokens 4096 \
            --max-num-seqs 16 \
            --block-size 128 \
            --enforce-eager \
            --no-disable-hybrid-kv-cache-manager \
            --trust-remote-code \
            --gpu-memory-utilization 0.9 \
            --quantization ascend \
            --model-loader-extra-config='{"enable_multithread_load": true, "num_threads": 128}' \
            --tokenizer-mode deepseek_v4 \
            --tool-call-parser deepseek_v4 \
            --enable-auto-tool-choice \
            --reasoning-parser deepseek_v4 \
            --attention_config.indexer_kv_dtype int8 \
            --additional-config '{"enable_cpu_binding": true, "enable_shared_expert_dp": true}' \
            --speculative-config '{"num_speculative_tokens": 1, "method": "mtp","enforce_eager": true}' \
            --kv-transfer-config \
            '{"kv_connector": "MooncakeHybridConnector",
            "kv_role": "kv_producer",
            "kv_port": "30000",
            "engine_id": "0",
            "kv_connector_extra_config": {
                        "prefill": {
                            "dp_size": 8,
                            "tp_size": 1
                        },
                        "decode": {
                            "dp_size": 32,
                            "tp_size": 1
                        }
                }
            }'
        ```

    2. Decode node (4 D nodes share the same script)

        ```shell
        unset ftp_proxy
        unset https_proxy
        unset http_proxy
        rm -rf ~/ascend/log

        nic_name="xxxxxx" #eg."enp67s0f0np0"
        local_ip=`hostname -I|awk -F " " '{print$1}'`

        export HCCL_OP_EXPANSION_MODE="AIV"
        export TASK_QUEUE_ENABLE=1
        export VLLM_RPC_TIMEOUT=3600000
        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
        export HCCL_EXEC_TIMEOUT=204
        export HCCL_CONNECT_TIMEOUT=1200

        export HCCL_IF_IP=$local_ip
        export GLOO_SOCKET_IFNAME=$nic_name
        export TP_SOCKET_IFNAME=$nic_name
        export HCCL_SOCKET_IFNAME=$nic_name
        export OMP_PROC_BIND=false
        export OMP_NUM_THREADS=10
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export HCCL_BUFFSIZE=1024

        export ASCEND_RT_VISIBLE_DEVICES=$1

        # Ensure the model path matches the directory recorded during download
        vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V4-Flash-w8a8-mtp \
            --host 0.0.0.0 \
            --port $2 \
            --data-parallel-size $3 \
            --data-parallel-rank $4 \
            --data-parallel-address $5 \
            --data-parallel-rpc-port $6 \
            --tensor-parallel-size $7 \
            --enable-expert-parallel \
            --seed 1024 \
            --served-model-name dsv4 \
            --max-model-len 135000 \
            --max-num-batched-tokens 60 \
            --max-num-seqs 30 \
            --block-size 128 \
            --no-disable-hybrid-kv-cache-manager \
            --no-enable-prefix-caching \
            --trust-remote-code \
            --gpu-memory-utilization 0.9 \
            --quantization ascend \
            --model-loader-extra-config='{"enable_multithread_load": true, "num_threads": 128}' \
            --tokenizer-mode deepseek_v4 \
            --tool-call-parser deepseek_v4 \
            --enable-auto-tool-choice \
            --reasoning-parser deepseek_v4 \
            --attention_config.indexer_kv_dtype int8 \
            --speculative-config '{"num_speculative_tokens": 1, "method": "mtp","enforce_eager": true}' \
            --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
            --kv-transfer-config \
            '{"kv_connector": "MooncakeHybridConnector",
            "kv_role": "kv_consumer",
            "kv_port": "30400",
            "engine_id": "4",
            "kv_connector_extra_config": {
                        "prefill": {
                            "dp_size": 8,
                            "tp_size": 1
                        },
                        "decode": {
                            "dp_size": 32,
                            "tp_size": 1
                        }
                }
            }' \
            --additional-config '{
                "ascend_compilation_config":{
                      "enable_npugraph_ex":true,
                      "enable_static_kernel":false
                },
               "enable_cpu_binding":true,
               "multistream_overlap_shared_expert":true,
               "recompute_scheduler_enable":true
            }'
        ```

3. Start the server with the following command on each node.

    1. Prefill node

        ```shell
        # change ip to your own
        python launch_online_dp.py --dp-size 8 --tp-size 1 --dp-size-local 8 --dp-rank-start 0 --dp-address x.x.x.x --dp-rpc-port 12321 --vllm-start-port 7100
        ```

        For each P instance, only the `--dp-address` parameter differs and must be configured as the IP address of the service within the same subnet as the other instances.

    2. Decode node

        ```shell
        # change ip to your own
        python launch_online_dp.py --dp-size 32 --tp-size 1 --dp-size-local 8 --dp-rank-start x --dp-address x.x.x.x --dp-rpc-port 12321 --vllm-start-port 7100
        ```

        For each D instance, only the `--dp-rank-start` parameter differs, which should be configured as 0, 8, 16, and 24 respectively. Each instance's `--dp-address` must be set to the IP address of the main D node, which is the IP of the Decode instance with `--dp-rank-start` set to 0.

4. Deploy the P-D disaggregation proxy.

    The proxy is also implemented by referring to [Prefill-Decode Disaggregation (Deepseek)](../features/pd_disaggregation_mooncake_multi_node.md).

Key Parameter Descriptions:

- `--no-disable-hybrid-kv-cache-manager` keeps the hybrid KV cache manager enabled. DeepSeek-V4 KV Pool deployments require this flag; otherwise, the service may OOM during startup.
- `--kv-cache-dtype turboquant_4bit_nc` is not currently supported in P/D disaggregated deployments.
- `--enforce-eager` forces eager execution on prefill nodes instead of graph compilation.
- The `prefill.dp_size` and `decode.dp_size` fields in `--kv-transfer-config` describe only the DP topology, while `prefill.tp_size` and `decode.tp_size` describe the TP topology. PCP is configured separately through `--prefill-context-parallel-size` and must not be included in the connector metadata.
- `recompute_scheduler_enable: true`: enables the recomputation scheduler. When the KV Cache of the decode node is insufficient, requests will be sent to the prefill node to recompute the KV Cache. In the PD separation scenario, enable this configuration only on decode nodes.
- `speculative-config`: When DSpark is enabled, Prefill and Decode must use the same number of speculative tokens, and `num_speculative_tokens` must be at least 5 (check the checkpoint's `config.json`). For MTP, we recommend setting Prefill to 1 and Decode to the actual number of speculative tokens.
- `MooncakeHybridConnector` and `MooncakeConnectorV2` transfer KV Cache from Prefill to Decode. The 950DT example in Section 5.2.3 uses `MooncakeConnectorV2`.
- `enable_shared_expert_dp: true`: enables data parallelism for shared experts, applicable to MoE models.
- `VLLM_PREFIX_CACHE_RETENTION_INTERVAL`: Controls the retention interval, in tokens, for prefix-cache checkpoints of hybrid attention layers. It is applicable to DeepSeek-V4 and takes effect only when prefix caching is enabled. Under KV-cache pressure, it can improve the effective prefix-cache hit rate for reusable long prefixes. The value must be a non-negative multiple of `--block-size`; for DeepSeek-V4-Flash, 128 times `--block-size` is recommended. Set it to `4096` when `--block-size` is `32`, or `16384` when `--block-size` is `128`.
- `VLLM_ASCEND_ENABLE_FUSED_MC2=1`: Enables the fused MoE computation (MC2) optimization, which consolidates shared expert and routed expert computation into fused kernels to reduce launch overhead and improve MoE inference throughput.
- `cudagraph_capture_sizes` (inside `--compilation-config`): Batch size tiers for ACL graph capture in Decode phase. vLLM pre-compiles graphs for these batch sizes to avoid runtime compilation overhead. In DSpark mode (`num_speculative_tokens=5`), each sequence processes 6 tokens per step (1 real + 5 speculative), so actual concurrency ≈ `batch_size / 6` (e.g., 360 → 60 concurrency). If unset, vLLM auto-captures on demand; set explicitly when concurrency patterns are predictable.

Deployment Verification:

After the PD separation service is fully started, send a request through the proxy port on the prefill master node to verify that Prefill and Decode nodes are working correctly together. Refer to [Prefill-Decode Disaggregation (Deepseek)](../features/pd_disaggregation_mooncake_multi_node.md) for the proxy verification method.

Common Issues Tip: If you encounter issues with PD separation deployment, please refer to the [Public FAQs](../../faqs.md) for troubleshooting.

#### 5.2.3 950DT Series PD Separation Deployment

This example deploys DeepSeek-V4-Flash on two 950DT nodes with the following topology:

- Prefill: DP1 × TP1 × PCP4, using four NPUs and exposing one API endpoint
- Decode: DP4 × TP1 × PCP1, using four NPUs and exposing four API endpoints
- P-to-D KV transfer: `MooncakeConnectorV2`

The `prefill.dp_size` and `decode.dp_size` fields in `--kv-transfer-config` describe only the DP topology.
PCP is configured through `--prefill-context-parallel-size` and must not be included in the connector metadata.

Replace `<nic_name>`, `<prefill_ip>`, and `<decode_ip>` with values from your environment. This example uses API port `7100` for Prefill and ports`7200`–`7203` for Decode.

Before you start, please:

1. Prepare run_prefill.sh
        unset https_proxy
        unset http_proxy
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/

        # 自动获取配置
        nic_name="<nic_name>"
        local_ip="<prefill_ip>"

        export HCCL_IF_IP=$local_ip
        export GLOO_SOCKET_IFNAME=$nic_name
        export TP_SOCKET_IFNAME=$nic_name
        export HCCL_SOCKET_IFNAME=$nic_name

        export VLLM_USE_V2_MODEL_RUNNER=1
        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
        export HCCL_EXEC_TIMEOUT=204
        export HCCL_CONNECT_TIMEOUT=120

        export OMP_PROC_BIND=false
        export OMP_NUM_THREADS=10
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export HCCL_BUFFSIZE=1024
        export TASK_QUEUE_ENABLE=1
        export HCCL_OP_EXPANSION_MODE=CCU_SCHED
        export VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096

        export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
        export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'

        vllm serve /root/.cache/DeepSeek-V4-Flash-0731  \
        --host $local_ip \
        --port 7100 \
        --served-model-name dsv4 \
        --data-parallel-size 1 \
        --data-parallel-rank 0 \
        --data-parallel-address $local_ip \
        --data-parallel-rpc-port 12325 \
        --tensor-parallel-size 1 \
        --prefill-context-parallel-size 4 \
        --enable-expert-parallel \
        --seed 1024 \
        --max-model-len 1048576 \
        --max-num-batched-tokens 8192 \
        --max-num-seqs 16 \
        --no-disable-hybrid-kv-cache-manager \
        --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":128}' \
        --speculative-config '{"num_speculative_tokens":5,"method":"dspark","enforce_eager":true}' \
        --trust-remote-code \
        --block-size 32 \
        --kv-cache-dtype fp8 \
        --attention_config.indexer_kv_dtype fp8 \
        --tokenizer-mode deepseek_v4 \
        --tool-call-parser deepseek_v4 \
        --enable-auto-tool-choice \
        --reasoning-parser deepseek_v4 \
        --gpu-memory-utilization 0.9 \
        --enforce-eager \
        --profiler-config '{"profiler": "torch", "torch_profiler_dir": "/path/to/profiling_output", "torch_profiler_with_stack": false}' \
        --additional-config '{"enable_cpu_binding":true,"enable_fused_mc2":0, "multistream_overlap_shared_expert": true, "enable_shared_expert_dp":true}' \
        --kv-transfer-config '{
            "kv_connector":"MooncakeConnectorV2",
            "kv_role":"kv_producer",
            "kv_port":"30000",
            "engine_id":"0",
            "kv_connector_extra_config":{
            "prefill":{
                "dp_size":1,
                "tp_size":1
            },
            "decode":{
                "dp_size":4,
                "tp_size":1
            }
            }
        }' \
        2>&1 | tee v4_flash_p.log

3. Prepare `run_decode.sh`.
        unset https_proxy
        unset http_proxy
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/

        # 自动获取配置
        nic_name="<nic_name>"
        local_ip="<prefill_ip>"

        export HCCL_IF_IP=$local_ip
        export GLOO_SOCKET_IFNAME=$nic_name
        export TP_SOCKET_IFNAME=$nic_name
        export HCCL_SOCKET_IFNAME=$nic_name

        export VLLM_USE_V2_MODEL_RUNNER=1
        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
        export HCCL_EXEC_TIMEOUT=204
        export HCCL_CONNECT_TIMEOUT=120

        export OMP_PROC_BIND=false
        export OMP_NUM_THREADS=10
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export HCCL_BUFFSIZE=1024
        export TASK_QUEUE_ENABLE=1

        export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
        export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'

        vllm serve /root/.cache/DeepSeek-V4-Flash-0731 \
        --host $local_ip \
        --port "${api_port}" \
        --served-model-name dsv4 \  
        --data-parallel-size 4 \
        --data-parallel-rank "${dp_rank}" \
        --data-parallel-address $local_ip \
        --data-parallel-rpc-port 12325 \
        --tensor-parallel-size 1 \
        --prefill-context-parallel-size 1 \
        --enable-expert-parallel \
        --seed 1024 \
        --max-model-len 1048576 \
        --max-num-batched-tokens 1024 \
        --max-num-seqs 56 \
        --async-scheduling \
        --block-size 32 \
        --kv-cache-dtype fp8 \
        --attention_config.indexer_kv_dtype fp8 \
        --no-enable-prefix-caching \
        --no-disable-hybrid-kv-cache-manager \
        --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":128}' \
        --trust-remote-code \
        --tokenizer-mode deepseek_v4 \
        --tool-call-parser deepseek_v4 \
        --enable-auto-tool-choice \
        --reasoning-parser deepseek_v4 \
        --gpu-memory-utilization 0.95 \
        --speculative-config '{"num_speculative_tokens":5,"method":"dspark","enforce_eager":false}' \
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
        --profiler-config '{"profiler": "torch", "torch_profiler_dir": "/path/to/profiling_output", "torch_profiler_with_stack": false}' \
        --kv-transfer-config '{
            "kv_connector":"MooncakeConnectorV2",
            "kv_role":"kv_consumer",
            "kv_port":"30100",
            "engine_id":"1",
            "kv_connector_extra_config":{
            "prefill":{
                "dp_size":1,
                "tp_size":1
            },
            "decode":{
                "dp_size":4,
                "tp_size":1
            }
            }
        }' \
        --additional-config '{
            "ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},"enable_cpu_binding":true,"multistream_overlap_shared_expert":true,
            "recompute_scheduler_enable":true, "enable_shared_expert_dp":true,"combine_quant_mode":4}' \2>&1 | tee v4_flash_d.log
5. Start the Prefill and Decode services in separate terminals.

    ```bash
    # Prefill terminal
    bash run_prefill.sh

    # Decode terminal
    bash run_decode.sh
    ```
6. Deploy the P-D disaggregation proxy.

   Refer to [Prefill-Decode Disaggregation (Deepseek)](../features/pd_disaggregation_mooncake_multi_node.md) to deploy the proxy. Configure the Prefill endpoint as `<prefill_node_ip>:7100`. Because the Decode node uses `DP4` with external DP load balancing, configure all four Decode endpoints as `<decode_node_ip>:7200` through `<decode_node_ip>:7203`.

   Configure the proxy to listen on `<proxy_node_ip>:5228`. Ensure that all Prefill and Decode endpoints are reachable from the proxy. Send inference requests to the proxy endpoint instead of directly accessing the Prefill or Decode endpoints.

#### 5.3 Multi-Node PD Separation Deployment with Memcache KV Cache Pool

This section builds on [Section 5.2.3](#523-ascend-950dt-series-pd-separation-deployment). Reuse the same topology, `launch_online_dp.py`, proxy mapping, and DeepSeek-V4 serving flags. Prefill and Decode switch to `MultiConnector` so live P-to-D KV transfer and the Memcache KV Cache Pool work together:

- `MooncakeConnectorV2` transfers KV Cache from Prefill to Decode in real time.
- `AscendStoreConnector` stores KV Cache in the Memcache KV Cache Pool so later Prefills with the same prefix can reuse it instead of recomputing.
- The Prefill node performs pool lookup, load, and write in this example. Decode continues to receive live KV Cache from Prefill.

For backend configuration and parameter descriptions, refer to [KV Cache Pool (Ascend Store)](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md#scenario-2-memcache-backend).

Compared with normal PD in Section 5.2.3, pay attention to these pooling-only requirements:

| Item | Normal PD (Section 5.2.3) | Pooled PD (this section) |
| ---- | ------------------------- | ------------------------ |
| Connector | `MooncakeConnectorV2` | `MultiConnector` wrapping `MooncakeConnectorV2` and `AscendStoreConnector` |
| KV pool service | Not required | Start one Memcache MetaService before Decode and Prefill |
| Additional environment | HCCL and Ascend direct-transfer variables | Also configure `MMC_LOCAL_CONFIG_PATH`, the Memcache library path, and `PYTHONHASHSEED` |
| Startup order | Decode → Prefill → Proxy | MetaService → Decode → Prefill → Proxy |
| Verification | Confirm P-to-D KV transfer | Also confirm an external KV Cache Pool hit with a repeated prefix |

#### 5.3.1 Prerequisites

INSTALL_PATH="<Location from the output of pip show memcache_hybrid>"
MEMCACHE_CONFIG_DIR="<directory containing mmc-local.conf>"

export MMC_LOCAL_CONFIG_PATH="${MEMCACHE_CONFIG_DIR}/mmc-local.conf"
export LD_LIBRARY_PATH="${INSTALL_PATH}/memcache_hybrid/lib:${PYTHON_LIB_DIR:-}:${LD_LIBRARY_PATH:-}:/usr/local/lib"

Configure one `mmc-meta.conf` for the MetaService:

```ini
ock.mmc.meta_service_url = tcp://<metaservice_ip>:5000
ock.mmc.meta_service.config_store_url = tcp://<metaservice_ip>:6000
ock.mmc.meta_service.metrics_url = http://<metaservice_ip>:8000
ock.mmc.log_level = info
```

Configure `mmc-local.conf` on both nodes:

```ini
ock.mmc.meta_service_url = tcp://<metaservice_ip>:5000
ock.mmc.local_service.config_store_url = tcp://<metaservice_ip>:6000
ock.mmc.log_level = info
ock.mmc.local_service.world_size = 256
ock.mmc.local_service.protocol = device_urma
ock.mmc.local_service.dram.size = 2GB
ock.mmc.local_service.max.dram.size = 2049GB
```

Use `device_urma` for Ascend 950DT Device UB. For UBOE, replace it with `device_uboe`. Both nodes must connect to the same MetaService and ConfigStore. `world_size` is the maximum number of LocalServices supported by the pool; it is not the number of NPUs used by this example.

This example uses the embedded LocalService mode. Each vLLM process initializes a LocalService by loading `mmc-local.conf` from `MMC_LOCAL_CONFIG_PATH`. Therefore, `ock.mmc.local_service.dram.size` must be greater than `0GB`. Do not start a separate standalone LocalService when using this configuration.

In this example, the Prefill instance uses PCP4 together with `AscendStoreConnector` in layerwise mode. The tested configuration sets `use_layerwise` to `true` and `layerwise_prefetch_layers` to `3`.

#### 5.3.2 Prefill Script

Prepare `run_prefill_memcache.sh` on the Prefill node.

```bash

        unset https_proxy
        unset http_proxy
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/
        
        # 自动获取配置
        nic_name="<nic_name>"
        local_ip="<prefill_ip>"
        
        # 池化环境变量
        export MMC_LOCAL_CONFIG_PATH="${memcache_config_dir}/mmc-local.conf"
        export LD_LIBRARY_PATH="${python_site_packages}/memcache_hybrid/lib:${PYTHON_LIB_DIR}:${LD_LIBRARY_PATH}:/usr/local/lib"
        
        export HCCL_IF_IP=$local_ip
        export GLOO_SOCKET_IFNAME=$nic_name
        export TP_SOCKET_IFNAME=$nic_name
        export HCCL_SOCKET_IFNAME=$nic_name
        
        export VLLM_USE_V2_MODEL_RUNNER=1
        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
        export HCCL_EXEC_TIMEOUT=204
        export HCCL_CONNECT_TIMEOUT=120
        
        export OMP_PROC_BIND=false
        export OMP_NUM_THREADS=10
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export HCCL_BUFFSIZE=1024
        export TASK_QUEUE_ENABLE=1
        export HCCL_OP_EXPANSION_MODE=CCU_SCHED
        export VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096
        
        export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
        export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'
        
        vllm serve /root/.cache/DeepSeek-V4-Flash-0731  \
        --host 0.0.0.0 \
        --port 7100 \
        --served-model-name dsv4 \ 
        --data-parallel-size 1 \
        --data-parallel-rank 0 \
        --data-parallel-address $local_ip \
        --data-parallel-rpc-port 12325 \
        --tensor-parallel-size 1 \
        --prefill-context-parallel-size 4 \
        --enable-expert-parallel \
        --seed 1024 \
        --max-model-len 1048576 \
        --max-num-batched-tokens 8192 \
        --max-num-seqs 16 \
        --no-disable-hybrid-kv-cache-manager \
        --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":128}' \
        --speculative-config '{"num_speculative_tokens":5,"method":"dspark","enforce_eager":true}' \
        --trust-remote-code \
        --block-size 32 \
        --kv-cache-dtype fp8 \
        --attention_config.indexer_kv_dtype fp8 \
        --tokenizer-mode deepseek_v4 \
        --tool-call-parser deepseek_v4 \
        --enable-auto-tool-choice \
        --reasoning-parser deepseek_v4 \
        --gpu-memory-utilization 0.9 \
        --enforce-eager \
        --profiler-config '{"profiler": "torch", "torch_profiler_dir": "/path/to/profiling_output", "torch_profiler_with_stack": false}' \
        --additional-config '{"enable_cpu_binding":true,"enable_fused_mc2":0, "multistream_overlap_shared_expert": true, "enable_shared_expert_dp":true}' \
        --kv-transfer-config '{
          "kv_connector": "MultiConnector",
          "kv_role": "kv_producer",
          "kv_load_failure_policy": "fail",
          "engine_id": "0",
          "kv_connector_extra_config": {
            "connectors": [
              {
                "kv_connector": "MooncakeConnectorV2",
                "kv_role": "kv_producer",
                "kv_port": "30000",
                "kv_connector_extra_config": {
                  "prefill": {
                    "dp_size": 1,
                    "tp_size": 1
                  },
                  "decode": {
                    "dp_size": 4,
                    "tp_size": 1
                  }
                }
              },
              {
                "kv_connector": "AscendStoreConnector",
                "kv_role": "kv_producer",
                "kv_connector_extra_config": {
                  "backend": "memcache",
                  "use_layerwise": true,
                  "layerwise_prefetch_layers": 3
                }
              }
            ]
          }
        }' \
        2>&1 | tee v4_flash_p_memcache.log
```

#### 5.3.3 Decode Script

Prepare `run_decode_memcache.sh` on the Decode node.

        ```bash
        #!/usr/bin/env bash
        source /root/.bashrc
        set -euo pipefail

        unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/
                
        nic_name="<nic_name>"
        local_ip="<prefill_ip>"
        
        export LD_LIBRARY_PATH="/usr/local/openssl-3.4/lib:${LD_LIBRARY_PATH}"
        export MMC_LOCAL_CONFIG_PATH="${memcache_config_dir}/mmc-local.conf"
        
        export HCCL_IF_IP=$local_ip
        export GLOO_SOCKET_IFNAME=$nic_name
        export TP_SOCKET_IFNAME=$nic_name
        export HCCL_SOCKET_IFNAME=$nic_name
        
        export VLLM_USE_V2_MODEL_RUNNER=1
        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
        export HCCL_EXEC_TIMEOUT=204
        export HCCL_CONNECT_TIMEOUT=120
        
        export OMP_PROC_BIND=false
        export OMP_NUM_THREADS=10
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export HCCL_BUFFSIZE=1024
        export TASK_QUEUE_ENABLE=1
        
        export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
        export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'
        
        vllm serve  /root/.cache/DeepSeek-V4-Flash-0731 \
        --host $local_ip \
        --port 7200 \
        --served-model-name dsv4 \ 
        --data-parallel-size 4 \
        --data-parallel-rank "${dp_rank}" \
        --data-parallel-address $local_ip  \
        --data-parallel-rpc-port 12325 \
        --tensor-parallel-size 1 \
        --prefill-context-parallel-size 1 \
        --enable-expert-parallel \
        --seed 1024 \
        --max-model-len 1048576 \
        --max-num-batched-tokens 1024 \
        --max-num-seqs 56 \
        --async-scheduling \
        --block-size 32 \
        --kv-cache-dtype fp8 \
        --attention_config.indexer_kv_dtype fp8 \
        --no-enable-prefix-caching \
        --no-disable-hybrid-kv-cache-manager \
        --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":128}' \
        --trust-remote-code \
        --tokenizer-mode deepseek_v4 \
        --tool-call-parser deepseek_v4 \
        --enable-auto-tool-choice \
        --reasoning-parser deepseek_v4 \
        --gpu-memory-utilization 0.95 \
        --speculative-config '{"num_speculative_tokens":5,"method":"dspark","enforce_eager":false}' \
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
        --profiler-config '{"profiler": "torch", "torch_profiler_dir": "/path/to/profiling_output", "torch_profiler_with_stack": false}' \
        --kv-transfer-config '{
        "kv_connector": "MultiConnector",
        "kv_role": "kv_consumer",
        "engine_id": "1",
        "kv_load_failure_policy": "fail",
        "kv_connector_extra_config": {
            "connectors": [
            {
                "kv_connector": "MooncakeConnectorV2",
                "kv_role": "kv_consumer",
                "kv_port": "30100",
                "kv_connector_extra_config": {
                "prefill": {
                    "dp_size": 1,
                    "tp_size": 1
                },
                "decode": {
                    "dp_size": 4,
                    "tp_size": 1
                }
                }
            },
            {
                "kv_connector": "AscendStoreConnector",
                "kv_role": "kv_consumer",
                "kv_connector_extra_config": {
                "backend": "memcache"
                }
            }
            ]
        }
        }' \
        --additional-config '{
            "ascend_compilation_config":{
            "enable_npugraph_ex":true,
            "enable_static_kernel":false
            },
            "enable_cpu_binding":true,
            "multistream_overlap_shared_expert":true,
            "recompute_scheduler_enable":true, 
            "enable_shared_expert_dp":true,
            "combine_quant_mode":4
        }' \
            2>&1 | tee "v4_flash_d_memcache_dp${dp_rank}.log"
        ```

#### 5.3.4 Start and Verify the Services

Start one MetaService and verify that ports 5000, 6000, and 8000 are listening:

```bash
export MMC_META_CONFIG_PATH="<CONFIG_DIRECTORY>/mmc-meta.conf"
python -c "from memcache_hybrid import MetaService; MetaService.main()"
```

Start the remaining services in this order:

```text
Memcache MetaService
        ↓
All Decode ranks
        ↓
Prefill
        ↓
P-D proxy
```

```bash
# Decode node: wait until all four Decode endpoints are ready
decode_ip="<decode_ip>"

for port in 7200 7201 7202 7203; do
    until curl --noproxy '*' -fsS \
        "http://${decode_ip}:${port}/health" >/dev/null; do
        sleep 2
    done

    echo "Decode endpoint ${port} is ready"
done
bash run_prefill_memcache.sh
# Prefill node: Start after all Decode ranks are ready
bash run_prefill_memcache.sh

# Prefill node, after all Decode ranks are ready
bash run_prefill_memcache.sh
```

Use the same proxy mapping as Section 5.2.3. Send a real generation request to `<proxy_ip>:5228` and confirm that the JSON response contains a non-empty `choices` field. Then send two requests with the same reusable prefix and confirm from Prefill logs or metrics that the second request hits the external KV Cache Pool. An HTTP 200 response alone does not prove that pooling is working.

Use the same proxy mapping as Section 5.2.3. First, send a generation request through the P-D proxy and confirm that the response contains a non-empty `choices` field:

```bash
curl --noproxy '*' -sS \
    http://<proxy_ip>:5228/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "dsv4",
        "messages": [
            {
                "role": "user",
                "content": "Who are you?"
            }
        ],
        "max_tokens": 64,
        "temperature": 0,
        "stream": false
    }'
```

A successful generation request only confirms that the P-D service is working. It does not prove that the external KV Cache Pool is being reused.

To verify KV Cache Pool reuse, first create two requests that share the same sufficiently long prefix:

```bash
python3 - <<'PY'
import json
import time

prefix_id = time.time_ns()
shared_prefix = (
    f"External KV Cache Pool verification prefix {prefix_id}. " * 4096
)

common = {
    "model": "dsv4",
    "max_tokens": 64,
    "temperature": 0,
    "stream": False,
}

suffixes = [
    "Summarize the preceding text in one sentence.",
    "Count the main topics in the preceding text.",
]

for index, suffix in enumerate(suffixes, start=1):
    request = {
        **common,
        "messages": [
            {
                "role": "user",
                "content": shared_prefix + suffix,
            }
        ],
    }

    with open(
        f"/tmp/dsv4_pool_request_{index}.json",
        "w",
        encoding="utf-8",
    ) as output:
        json.dump(request, output, ensure_ascii=False)
PY
```

Record the Prefill metrics before sending the requests:

```bash
curl --noproxy '*' -fsS \
    http://<prefill_node_ip>:7100/metrics \
    > /tmp/dsv4_prefill_metrics_before.txt

grep -E \
    'vllm:external_prefix_cache_(queries|hits)_total' \
    /tmp/dsv4_prefill_metrics_before.txt || true
```

Send the first request through the proxy to populate the external KV Cache Pool:

```bash
curl --noproxy '*' -sS \
    http://<proxy_ip>:5228/v1/chat/completions \
    -H "Content-Type: application/json" \
    --data-binary @/tmp/dsv4_pool_request_1.json \
    > /tmp/dsv4_pool_response_1.json
```

Wait for the KV Cache write to complete, and then send the second request with the same reusable prefix:

```bash
sleep 5

curl --noproxy '*' -sS \
    http://<proxy_ip>:5228/v1/chat/completions \
    -H "Content-Type: application/json" \
    --data-binary @/tmp/dsv4_pool_request_2.json \
    > /tmp/dsv4_pool_response_2.json
```

Confirm that both responses contain a non-empty `choices` field:

```bash
python3 - <<'PY'
import json

for index in (1, 2):
    path = f"/tmp/dsv4_pool_response_{index}.json"
    with open(path, encoding="utf-8") as response_file:
        response = json.load(response_file)

    if not response.get("choices"):
        raise SystemExit(f"Request {index} failed: choices is empty")

    print(f"Request {index}: choices is not empty")
PY
```

Query the Prefill metrics again:

```bash
curl --noproxy '*' -fsS \
    http://<prefill_node_ip>:7100/metrics \
    > /tmp/dsv4_prefill_metrics_after.txt

grep -E \
    'vllm:external_prefix_cache_(queries|hits)_total' \
    /tmp/dsv4_prefill_metrics_after.txt || true
```

Compare the value of `vllm:external_prefix_cache_hits_total` before and after the requests. The second request proves external KV Cache Pool reuse only when the metric increases and is greater than zero. Also retain the Prefill logs showing that the first request wrote KV Cache to the external pool.

An HTTP 200 response, a healthy process, or lower request latency alone does not prove that KV Cache Pool reuse is working.

## 6 Functional Verification

This section verifies basic inference functionality for all deployment modes. For the Memcache pooled PD deployment, also complete the KV Cache Pool reuse verification in Section 5.3.4.

Once your server is started, you can query the model with input prompts:

In <node0_ip>:<port>, use the IP address and port number of the primary node. If the primary and standby nodes are separated, use the IP address and port number of the proxy node.

```shell
curl http://<node0_ip>:<port>/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "dsv4",
        "messages": [
            {
                "role": "user",
                "content": "Who are you?"
            }
        ],
        "max_tokens": 256,
        "temperature": 0
    }'
```

Expected Result:

The service returns HTTP 200 OK with a JSON response containing the `choices` field.

## 7 Accuracy Evaluation

Here is the accuracy evaluation method using AISBench.

### 7.1 Using AISBench

1. Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) for details.

2. After execution, you can get the result.

| dataset | version | metric | mode | vllm-api-general-chat | note |
| ----- | ----- | ----- | ----- | ----- | ----- |
| GPQA | - | accuracy | gen | 88.17 | 1 Atlas 800 A3 (128GB × 8) |
| GSM8K | - | accuracy | gen | 96.30 | 1 Atlas 800 A3 (128GB × 8) |
| GPQA | v0.25.1rc | accuracy | gen | 90.40 | A3 1P1D DSpark w8a8 |
| SWE Multilingual | v0.25.1rc | accuracy | gen | 68.33 | A3 1P1D DSpark w8a8 |
| GPQA | - | accuracy | gen | 90.40 | 950PR&950DT Products, w4a4c8 |

> **Note**: Dataset evaluation results fluctuate between runs due to sampling and runtime non-determinism. When a dataset is evaluated multiple times, the average score is reported.

## 8 Performance Evaluation

### 8.1 Using AISBench

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

### 8.2 Using vLLM Benchmark

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/benchmarking/) for more details.

## 9 Performance Tuning

### 9.1 Recommended Configurations

> For complete startup commands and parameter descriptions, please refer to the deployment examples in [Chapter 5](#5-online-service-deployment).

### 9.2 Tuning Guidelines

#### 9.2.1 General Tuning Reference

Please refer to the [Public Performance Tuning Documentation](../../developer_guide/performance_and_debug/optimization_and_tuning.md) for tuning methods.

Please refer to the [Feature Matrix](../../user_guide/support_matrix/feature_matrix.md) for detailed feature descriptions.

## 10 FAQ

For common environment, installation, and general parameter issues, please refer to the [Public FAQs](../../faqs.md); this chapter only covers model-specific issues.

### Q: How do the W4A4C8 ultra-low-bit quantized weights preserve accuracy?

A: For 950PR&950DT Products, the W4A4C8 weights of `DeepSeek-V4-Flash-0731` are quantized with [msmodelslim](https://gitcode.com/Ascend/msmodelslim/blob/master/lab_practice/deepseek_v4/deepseek_v4_flash_w4a4c8.yaml). To preserve accuracy at such a low bit-width, a mixed-precision strategy is used: only the modules that dominate the parameter count are quantized to 4 bit, while precision-sensitive modules fall back to higher precision or are not quantized at all.

| Module | Quantization configuration |
| ------ | -------------------------- |
| MoE FFN routed experts | W4A4 MXFP4 |
| Attention linear layers | W8A8 MXFP8 |
| FFN shared experts | W8A8 MXFP8 |
| Router gates, compressor projections (`wgate`/`wkv`), indexer projections, and MTP layers | Not quantized (kept in high precision) |

- **Quantization algorithm**: The weights and activations use the MXFP4/MXFP8 microscaling formats. Each tensor is divided into blocks of 32 elements, and each block shares a single scaling factor obtained with symmetric min-max calibration on a mixed calibration dataset. Block-wise scaling adapts to the local dynamic range of each tensor, so the quantization error is far lower than that of uniform INT4 quantization.
- **C8 (FP8 KV cache)**: C8 is a dynamic quantization applied to the KV cache during inference. On 950PR&950DT Products, the KV cache is quantized to FP8 (not INT8). It is unrelated to the parameters stored in the weight checkpoint and is controlled only by the serving configuration. DeepSeek-V4-Flash and DeepSeek-V4-Pro share exactly the same C8 scheme:
    - **Attention KV cache** (FP8 by default on 950PR&950DT Products; switch back to BF16 with `--kv-cache-dtype bfloat16`): the compressed KV latent, which serves as both K and V, is dynamically quantized per 64-element tile to FP8 (E4M3) with a shared E8M0 scale per tile; the RoPE part of the KV cache stays in BF16, and Q is not quantized.
    - **Indexer cache** (enabled with `--attention_config.indexer_kv_dtype fp8`): both indexer Q and K are first rotated by a Hadamard matrix to spread outliers, and then dynamically quantized per token to FP8 (E4M3) with per-token FP32 scales.

For W4A4C8 accuracy scores, see [Accuracy Evaluation](#7-accuracy-evaluation).
