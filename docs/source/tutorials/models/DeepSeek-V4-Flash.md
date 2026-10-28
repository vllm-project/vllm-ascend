# DeepSeek-V4-Flash

## 1 Introduction

!!! note

    The "PD Separation Deployment" mentioned in this document is provided only as an example/reference design to demonstrate the basic process of deploying Prefill and Decode in separated mode. It is not recommended for use in the production environment.

    **For the production environment on the customer side, [MindIE Motor](https://www.hiascend.com/document/detail/en/mindie/latest/quickstart/docs/en/user_guide/quick_start.md) + vLLM-Ascend is recommended.**

DeepSeek-V4 introduces several key upgrades over DeepSeek-V3:

- The Manifold-Constrained Hyper-Connections (mHC) to strengthen conventional residual connections.
- A hybrid attention architecture, which greatly improves long-context efficiency through Compress-4-Attention and Compress-128-Attention. For the Mixture-of-Experts (MoE) components, it still adopts the DeepSeekMoE architecture, with only minor adjustments.

DeepSeek-V4-Flash is the lightweight variant of the DeepSeek-V4 family, suitable for high-throughput and low-latency serving scenarios.

This document will show the main verification steps of the model, including supported features, feature configuration, environment preparation, single-node and multi-node deployment, accuracy and performance evaluation.

DeepSeek-V4-Flash support is largely feature-complete starting from **vLLM-Ascend v0.26.0**. This document is validated and written based on **vLLM-Ascend v0.30.0**. To use the latest features, it is recommended to use the latest release candidate or official version.

## 2 Supported Features

Refer to [Supported Features List](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [Feature Guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## 3 Prerequisites

### 3.1 Model Weight

|  Weight Version                              | Hardware Requirements                                                      | Download Links |
|----------------------------------------------|----------------------------------------------------------------------------|----------------|
| `DeepSeek-V4-Flash-w8a8-mtp` (Quantized version) | 1 Atlas 800I A3 (128GB × 8) node or 1 Atlas 800I A2 (64GB × 8) node | [ModelScope](https://www.modelscope.cn/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp) |
| `DeepSeek-V4-Flash-0731-w8a8` | 1 Atlas 800I A3 (128GB × 8) node or 1 Atlas 800I A2 (64GB × 8) node | [ModelScope](https://www.modelscope.cn/models/Eco-Tech/DeepSeek-V4-Flash-0731-w8a8) |
| `DeepSeek-V4-Flash-0731` | 1 950DT node (96GB × 8)| [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) |
| `DeepSeek-V4-Flash-0731-w4a4c8` | 950PR&950DT products | Quantize locally with [msmodelslim](https://gitcode.com/Ascend/msmodelslim/blob/master/lab_practice/deepseek_v4/deepseek_v4_flash_w4a4c8.yaml) |

- `DeepSeek-V4-Flash-0731-w4a4c8` (W4A4 MXFP4 mixed-precision quantization): For how the ultra-low-bit quantization preserves accuracy, see the [W4A4C8 FAQ](#q-how-do-the-w4a4c8-ultra-low-bit-quantized-weights-preserve-accuracy).

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`.

>**Path description**: Download the model weights to a directory of your choice and record it. Ensure the model path in the subsequent deployment command matches this directory.

### 3.2 Verify Multi-node Communication (Optional)

If you want to deploy a multi-node environment, you need to verify multi-node communication according to [verify multi-node communication environment](../../getting_started/installation.md#installation-multi-node-interconnect).

## 4 Installation

### 4.1 Docker Image Installation

Select an image based on your machine type and start the docker image on your node, refer to [using docker](../../getting_started/installation.md#installation-prebuilt-image).

=== "950DT products"

    Start the Docker container on each server.

    ```bash
    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-a5
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

Single-node deployment completes both Prefill and Decode within the same node. The quantized model `DeepSeek-V4-Flash-w8a8-mtp` can be deployed on 1 Atlas 800I A3 (128GB × 8) or 1 Atlas 800I A2 (64GB × 8).

=== "950DT products with DSpark"

    Run the following script to execute PD-colocated online inference.

    ```shell
    export VLLM_USE_V2_MODEL_RUNNER=1
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=10
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export TASK_QUEUE_ENABLE=1
    export HCCL_OP_EXPANSION_MODE=CCU_SCHED
    export HCCL_BUFFSIZE=1024
    export VLLM_USE_FASTOKENS=1

    vllm serve /root/.cache/DeepSeek-V4-Flash-0731  \
        --port 8000 \
        --served-model-name dsv4 \
        --data-parallel-size 8 \
        --prefill-context-parallel-size 1 \
        --enable-expert-parallel \
        --enable-chunked-prefill \
        --max-model-len 1048576 \
        --max-num-batched-tokens 16384 \
        --max-num-seqs 32 \
        --gpu-memory-utilization 0.9 \
        --block-size 32 \
        --safetensors-load-strategy prefetch \
        --tokenizer-mode deepseek_v4 \
        --tool-call-parser deepseek_v4 \
        --enable-auto-tool-choice \
        --reasoning-parser deepseek_v4 \
        --kv-cache-dtype fp8 \
        --attention_config.indexer_kv_dtype fp8 \
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
        --no-disable-hybrid-kv-cache-manager \
        --speculative-config '{"num_speculative_tokens":5,"method":"dspark"}' \
        --additional-config '{"enable_cpu_binding":"True","multistream_overlap_shared_expert":true,"enable_shared_expert_dp":true}' \
    ```

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

=== "A2 series with DSpark"

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

    TPS is more than 50, reaching 2X the speed of DeepSeek-V4-Flash with MTP.

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
- `--additional-config` enables Ascend-specific optimizations. `enable_npugraph_ex` enables enhanced ACL graph execution, `enable_static_kernel: false` keeps static-kernel compilation disabled, `enable_cpu_binding` enables Ascend-native CPU binding, and `multistream_overlap_shared_expert` overlaps shared expert computation for better MoE throughput.
- `VLLM_PREFIX_CACHE_RETENTION_INTERVAL`: Controls the retention interval, in tokens, for prefix-cache checkpoints of hybrid attention layers. It is applicable to DeepSeek-V4 and takes effect only when prefix caching is enabled. Under KV-cache pressure, it can improve the effective prefix-cache hit rate for reusable long prefixes. The value must be a non-negative multiple of `--block-size`; for DeepSeek-V4-Flash, 128 times `--block-size` is recommended. Set it to `4096` when `--block-size` is `32`, or `16384` when `--block-size` is `128`.

Common Issues Tip: If you encounter issues, please refer to the [Public FAQs](../../faqs.md) for troubleshooting.

Service Verification:
Use the API port configured for the selected deployment. For the 950DT example, use port `8000`.

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

### 5.2 PD Separation Deployment

We recommend using Mooncake for deployment: [Mooncake](../features/pd_disaggregation_mooncake_multi_node.md).

In the standard single-node deployment mode, Prefill (prompt processing) and Decode (token generation) tasks run on the same set of NPUs. This can lead to two issues:

1. **Prefill preemption interrupts Decode**: Prefill is a compute-intensive task that processes the entire input context at once, while Decode generates tokens one by one. When a new user request arrives, its Prefill phase can preempt and interrupt ongoing Decode tasks, causing jitter and higher time-per-output-token (TPOT) latency.
2. **Inflexible resource allocation**: Prefill and Decode have fundamentally different computational characteristics — Prefill is compute-bound and memory-bandwidth-intensive, while Decode is memory-bandwidth-bound. Running them on the same hardware forces a compromise that satisfies neither optimally.

PD (Prefill-Decode) separation addresses these issues by running Prefill and Decode on dedicated NPU groups, each configured independently, on one or more nodes. This architecture is recommended for production deployments with concurrent multi-user workloads, where stable latency and high throughput are both required.

The following sections describe multi-node PD separation deployment on Atlas 800I A3 (128GB × 8) and Atlas 800I A2 (64GB × 8), and single-node PD separation on 950DT products (96GB × 8).

=== "950DT products with DSpark"

    #### 950DT products PD Separation Deployment with DSpark {: #524-950dt-series-pd-separation-deployment }

    This section describes PD separation on a single 950DT products node (96GB × 8). The Prefill instance uses DP1/TP1 with PCP4 on NPUs 0–3, while the Decode instances use DP4/TP1 with PCP1 on NPUs 4–7. KV Cache is transferred from Prefill to Decode using MooncakeConnectorV2.

    You can adjust the Prefill and Decode parallelism according to your workload and service requirements. The prefill and decode parallelism settings in --kv-transfer-config must match the actual engine configurations.

    Before starting the services, set VLLM_USE_V2_MODEL_RUNNER=1 in both templates to enable Model Runner V2. Replace nic_name, local_ip, and the model path with the appropriate values for your environment.

    Both Prefill and Decode use the same node IP (`<node_ip>`). The Prefill DP group uses RPC port `12321`, while the Decode DP group uses RPC port `12325`.

    Before you start, please:

    1. Prepare one shared `launch_online_dp.py` launcher for Prefill and Decode.

        ```python
        import argparse
        import multiprocessing
        import os
        import subprocess
        import sys

        parser = argparse.ArgumentParser()
        parser.add_argument("--script", required=True, help="Prefill or Decode startup script.")
        parser.add_argument("--dp-size", type=int, required=True)
        parser.add_argument("--tp-size", type=int, default=1)
        parser.add_argument("--pcp-size", type=int, default=1)
        parser.add_argument("--dp-size-local", type=int, default=-1)
        parser.add_argument("--dp-rank-start", type=int, default=0)
        parser.add_argument("--dp-address", required=True)
        parser.add_argument("--dp-rpc-port", default="12345")
        parser.add_argument("--vllm-start-port", type=int, default=9000)
        args = parser.parse_args()

        if args.dp_size_local == -1:
            args.dp_size_local = args.dp_size


        def run_command(visible_devices, dp_rank, port):
            script_args = [
                visible_devices,
                str(port),
                str(args.dp_size),
                str(dp_rank),
                args.dp_address,
                args.dp_rpc_port,
                str(args.tp_size),
                str(args.pcp_size),
            ]

            if args.dp_size == 1:
                with open(args.script, encoding="utf-8") as f:
                    script = f.read()
                script = "\n".join(
                    line for line in script.splitlines()
                    if "--data-parallel-rank" not in line
                    and "--data-parallel-external-lb" not in line
                )
                command = ["bash", "-o", "pipefail", "-c", script, args.script, *script_args]
            else:
                command = ["bash", "-o", "pipefail", args.script, *script_args]

            subprocess.run(command, check=True)


        if __name__ == "__main__":
            if not os.path.isfile(args.script):
                sys.exit(f"Cannot find {args.script}")

            processes = []
            devices_per_rank = args.tp_size * args.pcp_size

            for i in range(args.dp_size_local):
                visible_devices = ",".join(
                    str(x)
                    for x in range(i * devices_per_rank, (i + 1) * devices_per_rank)
                )
                process = multiprocessing.Process(
                    target=run_command,
                    args=(
                        visible_devices,
                        args.dp_rank_start + i,
                        args.vllm_start_port + i,
                    ),
                )
                processes.append(process)
                process.start()

            for process in processes:
                process.join()

            sys.exit(1 if any(p.exitcode != 0 for p in processes) else 0)
        ```

        Select the startup script with `--script`. When `--dp-size` is `1`, the launcher filters DP rank and external/hybrid load-balancing flags before running the script. It also propagates pipeline failures through `pipefail`.

        Parameter descriptions:

        | Parameter | Type | Required | Default | Description |
        |---|---|---|---|---|
        | `--script` | str | Yes | — | Path to the Prefill or Decode startup script. |
        | `--dp-size` | int | Yes | — | Total number of DP ranks across all nodes. |
        | `--tp-size` | int | No | `1` | Tensor parallel size for each DP rank. |
        | `--pcp-size` | int | No | `1` | Prefill context parallel size. Each local DP rank is assigned `--tp-size` × `--pcp-size` devices. |
        | `--dp-size-local` | int | No | `--dp-size` | Number of DP ranks launched on the current node. |
        | `--dp-rank-start` | int | No | `0` | Starting DP rank for the current node. |
        | `--dp-address` | str | Yes | — | IP address of the DP master node. |
        | `--dp-rpc-port` | str | No | `12345` | RPC port for DP coordination. |
        | `--vllm-start-port` | int | No | `9000` | Starting API port for local vLLM instances. Each subsequent instance uses the next port. |

    2. Prepare the P/D startup scripts.

        Save the following templates as `run_prefill.sh` and `run_decode.sh` in the same directory as `launch_online_dp.py`. If you use different filenames or directories, pass their paths through `--script`; no launcher changes are required.

        1. Prefill script: `run_prefill.sh`

            ```shell
            nic_name="xxxx" # change to your own nic name
            local_ip=xx.xx.xx.1 # change to your own ip

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
            export VLLM_USE_FASTOKENS=1

            # With --pcp-size 4, the launcher passes 0,1,2,3 for this Prefill instance.
            export ASCEND_RT_VISIBLE_DEVICES="$1"
            export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'

            vllm serve /root/.cache/DeepSeek-V4-Flash-0731 \
                --host "$local_ip" \
                --port "$2" \
                --served-model-name dsv4 \
                --data-parallel-size "$3" \
                --data-parallel-rank "$4" \
                --data-parallel-address "$5" \
                --data-parallel-rpc-port "$6" \
                --tensor-parallel-size "$7" \
                --prefill-context-parallel-size "$8" \
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
                --additional-config '{"enable_cpu_binding":true,"enable_fused_mc2":0, "multistream_overlap_shared_expert": true, "enable_shared_expert_dp":true}' \
                --kv-transfer-config '{
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
                }' \
                2>&1 | tee v4_flash_p.log
            ```

        2. Decode script: `run_decode.sh`

            ```shell
            nic_name="xxxx" # change to your own nic name
            local_ip=xx.xx.xx.1 # change to your own ip
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
            export VLLM_USE_FASTOKENS=1

            # The Decode group uses NPUs 4-7; each DP rank sees one of those NPUs.
            export ASCEND_RT_VISIBLE_DEVICES="$(( $1 + 4 ))"
            export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'

            vllm serve /root/.cache/DeepSeek-V4-Flash-0731 \
                --host "$local_ip" \
                --served-model-name dsv4 \
                --port "$2" \
                --data-parallel-size "$3" \
                --data-parallel-rank "$4" \
                --data-parallel-address "$5" \
                --data-parallel-rpc-port "$6" \
                --tensor-parallel-size "$7" \
                --prefill-context-parallel-size "$8" \
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
                --gpu-memory-utilization 0.9 \
                --speculative-config '{"num_speculative_tokens":5,"method":"dspark","enforce_eager":false}' \
                --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
                --kv-transfer-config '{
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
                }' \
                --additional-config '{"ascend_compilation_config": {"enable_npugraph_ex": true,"enable_static_kernel": false},
                    "enable_cpu_binding": true,"multistream_overlap_shared_expert": true,"recompute_scheduler_enable": true,"enable_shared_expert_dp": true,"combine_quant_mode": 4
                }' \
                2>&1 | tee "v4_flash_d_dp${4}.log"
            ```

    3. Start the services.

        Open separate terminals in the same container and run the commands from the directory containing `launch_online_dp.py` and the startup scripts. Start Prefill and Decode in separate terminals. The startup sequence is Prefill and Decode → Proxy. Both roles use the same node IP but different DP RPC ports.

        - Decode (NPUs 4–7)

            ```shell
            # Change the IP address to your own.
            python ./launch_online_dp.py --script ./run_decode.sh --dp-size 4 --tp-size 1 --pcp-size 1 --dp-size-local 4 --dp-rank-start 0 --dp-address xx.xx.xx.1 --dp-rpc-port 12325 --vllm-start-port 7200
            ```

        - Prefill (NPUs 0–3)

            ```shell
            # Change the IP address to your own.
            python ./launch_online_dp.py --script ./run_prefill.sh --dp-size 1 --tp-size 1 --pcp-size 4 --dp-size-local 1 --dp-rank-start 0 --dp-address xx.xx.xx.1 --dp-rpc-port 12321 --vllm-start-port 7100
            ```

    4. Deploy the P-D disaggregation proxy.

        Wait until the Prefill API endpoint and all four Decode API endpoints are healthy, then deploy the P-D disaggregation proxy as described in [Prefill-Decode Disaggregation (DeepSeek)](../features/pd_disaggregation_mooncake_multi_node.md).

    Key Parameter Descriptions:

    - `--no-disable-hybrid-kv-cache-manager` keeps the hybrid KV cache manager enabled. DeepSeek-V4 KV Pool deployments require this flag; otherwise, the service may OOM during startup.

    - `--enforce-eager` forces eager execution on Prefill nodes instead of graph compilation.

    - `--async-scheduling` enables asynchronous scheduling to overlap CPU scheduling work with model execution.

    - `--trust-remote-code` allows the model repository's custom code to be loaded. Only use trusted model repositories.

    - `VLLM_USE_V2_MODEL_RUNNER=1`: Keep this environment variable exported in both the Prefill and Decode pooling scripts. PCP requires Model Runner V2, which does not support DSA-CP. Therefore, do not set `enable_dsa_cp` in `--additional-config`.

    - `--prefill-context-parallel-size` configures prefill context parallelism (PCP) on Model Runner V2 (MRv2), replacing the DSA-CP configuration used on MRv1. Besides speeding up long-context prefill, enabling PCP also shards the `o_proj` weights across the PCP group, providing partial tensor parallelism even when `--tensor-parallel-size` is `1` and relieving memory pressure.

    - `--kv-transfer-config` configures KV cache transfer between the Prefill producer and Decode consumer in PD separation.

    - `MooncakeConnectorV2` transfers KV cache from Prefill to Decode in PD separation. Configure Prefill with `kv_role: kv_producer` and Decode with `kv_role: kv_consumer`, allowing Decode to reuse the KV cache computed by Prefill for subsequent token generation.

    - `kv_connector_extra_config.prefill.dp_size/tp_size` and `decode.dp_size/tp_size` must match the actual global DP and TP layout on the Prefill and Decode sides, respectively.

    - `scheduler_config.recompute_scheduler_enable: true` enables the recomputation scheduler. When the Decode node has insufficient KV cache space, requests are sent to Prefill for KV cache recomputation. Enable it only on PD Decode nodes; the legacy top-level key is deprecated.

    - `--speculative-config`: Prefill and Decode must use the same DSpark `num_speculative_tokens` value. Use the value declared by the checkpoint; this example uses `5`. The nested `enforce_eager` applies to DSpark draft execution, while the top-level `--enforce-eager` keeps the Prefill target model in eager mode.

    - `--kv-cache-dtype fp8` and `--attention_config.indexer_kv_dtype fp8` use FP8 KV cache. The P and D sides must use the same KV data types so that the transferred KV cache is directly reusable.

    - `enable_fused_mc2: 0` disables the Fused MC2 fusion operator on Prefill; `multistream_overlap_shared_expert: true` overlaps shared-expert computation with MoE communication instead, and `enable_shared_expert_dp: true` places shared experts on DP ranks.

    - `ascend_compilation_config.enable_npugraph_ex: true` enables the extended NPU graph mode on Decode for graph compilation, while Prefill runs in eager mode. `enable_static_kernel: false` uses dynamic-shape kernels instead of static-shape kernels for ACL Graph execution.

    - `combine_quant_mode: 4` forces the MoE MC2 combine stage on Decode nodes to use FP8 (MXFP float8_e4m3) communication quantization.

=== "A3 series"

    #### A3 Series PD Separation Deployment {: #521-a3-series-pd-separation-deployment }

    This section shows the deployment guide of DeepSeek-V4-Flash on Atlas 800I A3 (128GB × 8) multi-node environment with 1P1D for better performance.

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
                "--pcp-size",
                type=int,
                default=1,
                help="Prefill context parallel size."
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
        pcp_size = args.pcp_size
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
                str(pcp_size),
            ]
            subprocess.run(command, check=True)

        if __name__ == "__main__":
            template_path = "./run_dp_template.sh"
            if not os.path.exists(template_path):
                print(f"Template file {template_path} does not exist.")
                sys.exit(1)

            processes = []
            devices_per_rank = tp_size * pcp_size
            for i in range(dp_size_local):
                dp_rank = dp_rank_start + i
                vllm_engine_port = vllm_start_port + i
                visible_devices = ",".join(str(x) for x in range(i * devices_per_rank, (i + 1) * devices_per_rank))
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
        |`--pcp-size`|int|No|1|Prefill context parallel size within each DP rank. Each rank uses `--tp-size` × `--pcp-size` devices.|
        |`--dp-size-local`|int|No|(same as `--dp-size`)|Number of DP ranks on the current node. If not set, defaults to `--dp-size`.|
        |`--dp-rank-start`|int|No|0|Starting rank offset for data parallel ranks on this node.|
        |`--dp-address`|str|Yes|-|IP address of the data parallel master node.|
        |`--dp-rpc-port`|str|No|12345|RPC port for data parallel master communication.|
        |`--vllm-start-port`|int|No|9000|Starting port for each vLLM engine instance on this node.|

    2. Prepare the script `run_dp_template.sh` on each node.

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
                --kv-transfer-config '{"kv_connector": "MooncakeHybridConnector",
                "kv_role": "kv_producer",
                "kv_port": "30000",
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
                --kv-transfer-config '{"kv_connector": "MooncakeHybridConnector",
                "kv_role": "kv_consumer",
                "kv_port": "30100",
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

            For MTP (`A3 series`), keep DP4 and TP4:

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

        Refer to [Prefill-Decode Disaggregation (DeepSeek)](../features/pd_disaggregation_mooncake_multi_node.md) to deploy the P-D disaggregation proxy.

=== "A3 series with DSpark"

    #### A3 Series PD Separation Deployment with DSpark

    This section shows the deployment guide of DeepSeek-V4-Flash on Atlas 800I A3 (128GB × 8) multi-node environment with 1P1D for better performance.

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
                "--pcp-size",
                type=int,
                default=1,
                help="Prefill context parallel size."
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
        pcp_size = args.pcp_size
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
                str(pcp_size),
            ]
            subprocess.run(command, check=True)

        if __name__ == "__main__":
            template_path = "./run_dp_template.sh"
            if not os.path.exists(template_path):
                print(f"Template file {template_path} does not exist.")
                sys.exit(1)

            processes = []
            devices_per_rank = tp_size * pcp_size
            for i in range(dp_size_local):
                dp_rank = dp_rank_start + i
                vllm_engine_port = vllm_start_port + i
                visible_devices = ",".join(str(x) for x in range(i * devices_per_rank, (i + 1) * devices_per_rank))
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
        |`--pcp-size`|int|No|1|Prefill context parallel size within each DP rank. Each rank uses `--tp-size` × `--pcp-size` devices.|
        |`--dp-size-local`|int|No|(same as `--dp-size`)|Number of DP ranks on the current node. If not set, defaults to `--dp-size`.|
        |`--dp-rank-start`|int|No|0|Starting rank offset for data parallel ranks on this node.|
        |`--dp-address`|str|Yes|-|IP address of the data parallel master node.|
        |`--dp-rpc-port`|str|No|12345|RPC port for data parallel master communication.|
        |`--vllm-start-port`|int|No|9000|Starting port for each vLLM engine instance on this node.|

    2. Prepare the script `run_dp_template.sh` on each node.

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
            export VLLM_USE_V2_MODEL_RUNNER=1
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
                --prefill-context-parallel-size $8 \
                --enable-expert-parallel \
                --seed 1024 \
                --served-model-name dsv4 \
                --max-model-len 1048576 \
                --max-num-batched-tokens 16384 \
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
                --gpu-memory-utilization 0.94 \
                --quantization ascend \
                --enforce-eager \
                --additional-config '{"enable_cpu_binding": true, "enable_fused_mc2": 1}' \
                --kv-transfer-config '{"kv_connector": "MooncakeHybridConnector",
                "kv_role": "kv_producer",
                "kv_port": "30000",
                "kv_connector_extra_config": {
                            "prefill": {
                                    "dp_size": 2,
                                    "tp_size": 1,
                                    "pcp_size": 8
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
            export VLLM_USE_V2_MODEL_RUNNER=1

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
                --speculative-config '{"num_speculative_tokens": 5,"method": "dspark"}' \
                --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
                --kv-transfer-config '{"kv_connector": "MooncakeHybridConnector",
                "kv_role": "kv_consumer",
                "kv_port": "30100",
                "kv_connector_extra_config": {
                            "prefill": {
                                    "dp_size": 2,
                                    "tp_size": 1,
                                    "pcp_size": 8
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

            For DSpark (`A3 series with DSpark`), use PCP8, DP2, and TP1:

            ```shell
            # change ip to your own
            python launch_online_dp.py --dp-size 2 --tp-size 1 --pcp-size 8 --dp-size-local 2 --dp-rank-start 0 --dp-address xx.xx.xx.1 --dp-rpc-port 12321 --vllm-start-port 7100
            ```

        2. Decode node

            ```shell
            # change ip to your own
            python launch_online_dp.py --dp-size 16 --tp-size 1 --dp-size-local 16 --dp-rank-start 0 --dp-address xx.xx.xx.2 --dp-rpc-port 12321 --vllm-start-port 7100
            ```

    4. Deploy the P-D disaggregation proxy.

        Refer to [Prefill-Decode Disaggregation (DeepSeek)](../features/pd_disaggregation_mooncake_multi_node.md) to deploy the P-D disaggregation proxy.

=== "A2 series"

    #### A2 Series PD Separation Deployment {: #522-a2-series-pd-separation-deployment }

    This section shows the deployment guide of DeepSeek-V4-Flash on Atlas 800I A2 (64GB × 8) multi-node environment with 4\*1P 1\*4D for better performance.

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

            For each P instance, set a unique `kv_port` (e.g., 30000, 30100, etc.).

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
                --kv-transfer-config '{"kv_connector": "MooncakeHybridConnector",
                "kv_role": "kv_producer",
                "kv_port": "30000",
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
                --kv-transfer-config '{"kv_connector": "MooncakeHybridConnector",
                "kv_role": "kv_consumer",
                "kv_port": "30400",
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

            For each P instance, only the `--dp-address` parameter differs and must be configured as the node IP address in the same subnet as the other instances.

        2. Decode node

            ```shell
            # change ip to your own
            python launch_online_dp.py --dp-size 32 --tp-size 1 --dp-size-local 8 --dp-rank-start x --dp-address x.x.x.x --dp-rpc-port 12321 --vllm-start-port 7100
            ```

            For each D instance, only the `--dp-rank-start` parameter differs, which should be configured as 0, 8, 16, and 24 respectively. Each instance's `--dp-address` must be set to the IP address of the main D node, which is the IP of the Decode instance with `--dp-rank-start` set to 0.

    4. Deploy the P-D disaggregation proxy.

        The proxy is also implemented by referring to [Prefill-Decode Disaggregation (DeepSeek)](../features/pd_disaggregation_mooncake_multi_node.md).

    Key Parameter Descriptions:

    - `--no-disable-hybrid-kv-cache-manager` keeps the hybrid KV cache manager enabled. DeepSeek-V4 KV Pool deployments require this flag; otherwise, the service may OOM during startup.
    - `--kv-cache-dtype turboquant_4bit_nc` is not currently supported in P/D disaggregated deployments.
    - `--enforce-eager` forces eager execution on prefill nodes instead of graph compilation.
    - `VLLM_USE_V2_MODEL_RUNNER=1`: Enables Model Runner V2. Set it with `export VLLM_USE_V2_MODEL_RUNNER=1` on both DSpark nodes in the A3 example. PCP requires Model Runner V2; DSA-CP is unsupported with V2, so do not set `enable_dsa_cp` in `--additional-config`.
    - `kv_connector_extra_config.prefill.dp_size/tp_size` and `decode.dp_size/tp_size` must match the actual global DP and TP layout on the prefill and decode sides.
    - `recompute_scheduler_enable: true`: enables the recomputation scheduler. When the KV Cache of the decode node is insufficient, requests will be sent to the prefill node to recompute the KV Cache. In the PD separation scenario, enable this configuration only on decode nodes.
    - `speculative-config`: When DSpark is enabled, Prefill and Decode must use the same number of speculative tokens, and `num_speculative_tokens` must be at least 5 (check the checkpoint's `config.json`). For MTP, we recommend setting Prefill to 1 and Decode to the actual number of speculative tokens.
    - `MooncakeHybridConnector`: the KV transfer connector used for PD separation, transferring KV Cache between prefill and decode nodes.
    - `enable_shared_expert_dp: true`: enables data parallelism for shared experts, applicable to MoE models.
    - `VLLM_PREFIX_CACHE_RETENTION_INTERVAL`: Controls the retention interval, in tokens, for prefix-cache checkpoints of hybrid attention layers. It is applicable to DeepSeek-V4 and takes effect only when prefix caching is enabled. Under KV-cache pressure, it can improve the effective prefix-cache hit rate for reusable long prefixes. The value must be a non-negative multiple of `--block-size`; for DeepSeek-V4-Flash, 128 times `--block-size` is recommended. Set it to `4096` when `--block-size` is `32`, or `16384` when `--block-size` is `128`.
    - `VLLM_ASCEND_ENABLE_FUSED_MC2=1`: Enables the fused MoE computation (MC2) optimization, which consolidates shared expert and routed expert computation into fused kernels to reduce launch overhead and improve MoE inference throughput.
    - `cudagraph_capture_sizes` (inside `--compilation-config`): Batch size tiers for ACL graph capture in Decode phase. vLLM pre-compiles graphs for these batch sizes to avoid runtime compilation overhead. In DSpark mode (`num_speculative_tokens=5`), each sequence processes 6 tokens per step (1 real + 5 speculative), so actual concurrency ≈ `batch_size / 6` (e.g., 360 → 60 concurrency). If unset, vLLM auto-captures on demand; set explicitly when concurrency patterns are predictable.

    Deployment Verification:

    After the PD separation service is fully started, send a request through the proxy port on the prefill master node to verify that Prefill and Decode nodes are working correctly together. Refer to [Prefill-Decode Disaggregation (DeepSeek)](../features/pd_disaggregation_mooncake_multi_node.md) for the proxy verification method.

    Common Issues Tip: If you encounter issues with PD separation deployment, please refer to the [Public FAQs](../../faqs.md) for troubleshooting.

    ##### Ultra-Long Sequence Deployment {: #523-ultra-long-sequence-deployment }

    For ultra-long sequence scenarios, support can be achieved by adjusting the PD (Prefill/Decode) ratio and the model parallelism strategy. For example, in a 1M sequence scenario, a 1\*4P-1\*4D ratio can be used, with the model parallelism set to DP4TP8 mode.

=== "A2 series with DSpark"

    #### A2 Series PD Separation Deployment with DSpark (2P2D, 32 cards) {: #525-a2-series-pd-separation-deployment-with-dspark-2p2d-32-cards }

    Use `DeepSeek-V4-Flash-0731-w8a8` on four Atlas 800I A2 servers (8 NPUs per server). Choose the workload settings from the table before starting either role. The settings below retain the configurations selected in completed A2 measurements; they are starting points for the listed workloads, not a universal optimum across workloads or software versions.

    | Role | Nodes / NPUs | Parallelism | HTTP endpoints |
    |---|---|---|---|
    | Prefill | 2 / 16 | ExternalDP2 × TP1 × PP1 × PCP8 | node A:7100, node B:7101 |
    | Decode | 2 / 16 | DP16 × TP1, eight processes per node | 7100–7107 on each node |

    !!! note "Measurement and version scope"
        The measured setup used the A2 `dev-26.2.0.day20261021` image, MRv2, P eager and D target/draft graph mode. It also included runtime changes for UVA dirty tracking, exact request-hash reuse and RoPE capacity; those changes are not delivered by this documentation update. Pin the image, model and loaded code when reproducing measurements. Revalidate these settings on a different build, including the `nightly-main` A2 image described in Section 4.1.

    !!! warning "Topology and dispatch limits"
        Keep `PP1 × PCP8` with `MooncakeHybridConnector`. The validated implementation rejects simultaneous PP>1 and PCP>1. Its A2 dispatch path also requires `D max-num-seqs × (1 + draft tokens) <= 256`: the measured pairs are 5/42, 2/85 and 1/128. A different draft length changes the legal request capacity; 42 is not a universal maximum.

    **Choose the server settings and client load.** D uses `--max-num-batched-tokens 720` and `--gpu-memory-utilization 0.97` in every configuration. P uses `--max-num-seqs 16`, `--api-server-count 4`, `--gpu-memory-utilization 0.96` and draft5/eager.

    | Configuration | Workload / native window | P token budget / prefix cache | D draft / max-num-seqs | Proxy workers |
    |---|---|---|---|---:|
    | `decode-4k-total` | 4K input / 2K output, pure D, total | 4096 / on | 2 / 85 | 4 |
    | `decode-4k-stable` | 4K / 2K, pure D, native stable | 4096 / on | 1 / 128 | 4 |
    | `decode-16k` | 16K / 1K, pure D | 4096 / on | 1 / 128 | 4 |
    | `decode-long` | 64K / 1K or 128K / 1K, pure D | 4096 / on | 5 / 42 | 1 |
    | `cold-pd` | PC0 pure P or full PD | 8192 / off | 1 / 128 | 4 |
    | `pc90` | Distinct-suffix, multiple-prefix pooling; see [A2 KV Pooling](#526-a2-kv-pooling-mooncake-ssd-profile-and-memcache-integration) | 8192 / on | 1 / 128 | 4 |

    The recorded client loads below are separate from the server profiles. Input counts include the measured client's 84-token template offset for these ordinary-length cases. Keep the payload definition and tokenizer unchanged when comparing them.

    | Profile / window | Measured input / output tokens | Requests / concurrency / rate (req/s) |
    |---|---|---|
    | `decode-4k-total` / total | 4180 / 2048 | 4096 / 1536 / 0 |
    | `decode-4k-stable` / native stable | 4180 / 2048 | 8192 / 2304 / 0; stable subset chosen by AISBench |
    | `decode-16k` / total, TTFT trial | 16468 / 1024 | 8192 / 2048 / 0 |
    | `decode-16k` / recorded native stable | 16468 / 1024 | 8192 / 2304 / 0; do not substitute the CC2048 total result |
    | `decode-long` / 64K total or native stable | 65620 / 1024 | 3840 / 576 / 0 |
    | `decode-long` / 128K total | 131156 / 1024 | 3920 / 960 / 0 |
    | `decode-long` / 128K native stable | 131156 / 1024 | 2960 / 960 / 0 |
    | `cold-pd` / full PD total | 16468 / 1024 | 240 / 60 / 0 |

    The complete scripts below use the measured 16K pure-D setting: P token budget4096/APC on, D draft1/mns128/GMU0.97. For other workloads, replace the explicit values listed in the table; for `cold-pd`, replace the P cache switch with `--no-enable-prefix-caching`. No external helper or profile file is required.

    The measured context budgets are intentionally asymmetric: `--max-model-len 150000` on P and `508576` on D. The larger D budget does not extend the prompt length that P can prefill. All ordinary-length workloads listed here fit within the P budget; do not advertise the D value as the end-to-end prompt limit. If a deployment requires equal P/D limits, change both to the chosen service limit and revalidate KV capacity and throughput before reusing these measurements.

    Server configuration and client load must be recorded together. For the 16K/1K total-window latency trial, N8192/CC2048/rate0 was tested twice against a current-deployment N8192/CC2304/rate0 control. The last repeat reduced legacy TTFT mean/P90 from 21.30/50.32 seconds to 11.32/42.51 seconds while output throughput changed by -0.54%. TPOT was 93.6 ms, not a 20 ms SLO pass. This is a workload-specific result; lower concurrency does not guarantee lower latency elsewhere. Do not substitute total results for native stable results. The 1M cases require a separately validated long-context deployment, not a larger `--max-model-len` in these scripts.

    **Prefill: run this complete script on each P node.** Set `p_rank=0` on node A and `p_rank=1` on node B; keep `head_ip` pointing at P node A. Start both ranks before waiting for readiness.

    ```shell
    set -euo pipefail
    nic_name="<NETWORK_INTERFACE>"
    local_ip="<THIS_PREFILL_NODE_IP>"
    head_ip="<PREFILL_NODE_A_IP>"
    p_rank=0  # Set to 1 on Prefill node B.
    model_path="/path/to/DeepSeek-V4-Flash-0731-w8a8"

    export ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.protocol_desc":["roce:device","hccs:device"]}'
    export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'
    export VLLM_USE_V2_MODEL_RUNNER=1
    export HCCL_INTRA_ROCE_ENABLE=1
    export HCCL_CONNECT_TIMEOUT=6000 HCCL_EXEC_TIMEOUT=204 HCCL_BUFFSIZE=1200
    export GLOO_SOCKET_IFNAME="$nic_name" HCCL_SOCKET_IFNAME="$nic_name" TP_SOCKET_IFNAME="$nic_name"
    export HCCL_IF_IP="$local_ip"
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000 VLLM_RPC_TIMEOUT=3600000
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    kv_config='{
      "kv_connector": "MooncakeHybridConnector",
      "kv_role": "kv_producer",
      "kv_port": "30000",
      "kv_connector_extra_config": {
        "prefill": {
          "dp_size": 2,
          "tp_size": 1,
          "pp_size": 1,
          "pcp_size": 8
        },
        "decode": {
          "dp_size": 16,
          "tp_size": 1,
          "pp_size": 1,
          "pcp_size": 1
        }
      },
      "kv_load_failure_policy": "fail"
    }'

    vllm serve "$model_path" \
      --host 0.0.0.0 --port "$((7100 + p_rank))" \
      --data-parallel-size 2 --data-parallel-rank "$p_rank" \
      --data-parallel-address "$head_ip" --data-parallel-rpc-port 12321 \
      --tensor-parallel-size 1 --pipeline-parallel-size 1 \
      --prefill-context-parallel-size 8 --decode-context-parallel-size 1 \
      --served-model-name dsv4 --enable-expert-parallel \
      --max-model-len 150000 --max-num-batched-tokens 4096 \
      --max-num-seqs 16 --api-server-count 4 \
      --block-size 32 --gpu-memory-utilization 0.96 \
      --enable-chunked-prefill --no-disable-hybrid-kv-cache-manager \
      --enable-prefix-caching --enforce-eager \
      --quantization ascend --tokenizer-mode deepseek_v4 \
      --attention_config.indexer_kv_dtype int8 \
      --tool-call-parser deepseek_v4 --enable-auto-tool-choice \
      --reasoning-parser deepseek_v4 --trust-remote-code --seed 1024 \
      --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":128}' \
      --speculative-config '{"method":"dspark","num_speculative_tokens":5,"enforce_eager":true}' \
      --additional-config '{"enable_cpu_binding":true,"enable_flashcomm1":true,"enable_fused_mc2":1}' \
      --kv-transfer-config "$kv_config"
    ```

    **Decode: run this complete script on each D node.** Set `rank_offset=0` on node A and `rank_offset=8` on node B; keep `head_ip` pointing at D node A. Launch all ranks before waiting for readiness.

    ```shell
    set -euo pipefail
    nic_name="<NETWORK_INTERFACE>"
    local_ip="<THIS_DECODE_NODE_IP>"
    head_ip="<DECODE_NODE_A_IP>"
    model_path="/path/to/DeepSeek-V4-Flash-0731-w8a8"
    rank_offset=0  # Set to 8 on Decode node B.

    export ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.protocol_desc":["roce:device","hccs:device"]}'
    export VLLM_USE_V2_MODEL_RUNNER=1
    export HCCL_INTRA_ROCE_ENABLE=1
    export HCCL_CONNECT_TIMEOUT=120 HCCL_EXEC_TIMEOUT=204 HCCL_BUFFSIZE=1800
    export GLOO_SOCKET_IFNAME="$nic_name" HCCL_SOCKET_IFNAME="$nic_name" TP_SOCKET_IFNAME="$nic_name"
    export HCCL_IF_IP="$local_ip"
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000 VLLM_RPC_TIMEOUT=3600000
    export OMP_NUM_THREADS=10 OMP_PROC_BIND=false

    for local_rank in 0 1 2 3 4 5 6 7; do
      global_rank=$((local_rank + rank_offset))
      export ASCEND_RT_VISIBLE_DEVICES="$local_rank"
      kv_config='{
        "kv_connector": "MooncakeHybridConnector",
        "kv_role": "kv_consumer",
        "kv_port": "30100",
        "kv_connector_extra_config": {
          "prefill": {
            "dp_size": 2,
            "tp_size": 1,
            "pp_size": 1,
            "pcp_size": 8
          },
          "decode": {
            "dp_size": 16,
            "tp_size": 1,
            "pp_size": 1,
            "pcp_size": 1
          }
        }
      }'
      vllm serve "$model_path" \
        --host 0.0.0.0 --port "$((7100 + local_rank))" \
        --data-parallel-size 16 --data-parallel-rank "$global_rank" \
        --data-parallel-address "$head_ip" --data-parallel-rpc-port 12321 \
        --tensor-parallel-size 1 --enable-expert-parallel --seed 1024 \
        --max-model-len 508576 --max-num-batched-tokens 720 \
        --max-num-seqs 128 --async-scheduling --block-size 32 \
        --served-model-name dsv4 --attention_config.indexer_kv_dtype int8 \
        --no-enable-prefix-caching --no-disable-hybrid-kv-cache-manager \
        --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":128}' \
        --trust-remote-code --tokenizer-mode deepseek_v4 \
        --tool-call-parser deepseek_v4 --enable-auto-tool-choice \
        --reasoning-parser deepseek_v4 --gpu-memory-utilization 0.97 \
        --quantization ascend \
        --speculative-config "{\"num_speculative_tokens\":1,\"method\":\"dspark\"}" \
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
        --kv-transfer-config "$kv_config" \
        --additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},"enable_cpu_binding":true,"multistream_overlap_shared_expert":true,"recompute_scheduler_enable":true}' &
      sleep 3
    done
    wait
    ```

    **Start the proxy after all endpoints are healthy.** From the repository root, run the following complete command block. Use four workers for the short/medium and pooling configurations, or one for the recorded long-input configuration:

    ```shell
    set -euo pipefail
    proxy_workers=4
    ulimit -n 65536
    p_a="<PREFILL_NODE_A_IP>"
    p_b="<PREFILL_NODE_B_IP>"
    d_a="<DECODE_NODE_A_IP>"
    d_b="<DECODE_NODE_B_IP>"
    decode_hosts=()
    decode_ports=()
    for node in "$d_a" "$d_b"; do
      for port in 7100 7101 7102 7103 7104 7105 7106 7107; do
        decode_hosts+=("$node")
        decode_ports+=("$port")
      done
    done
    python3 examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
      --host 0.0.0.0 --port 18590 --workers "$proxy_workers" \
      --prefiller-hosts "$p_a" "$p_b" --prefiller-ports 7100 7101 \
      --decoder-hosts "${decode_hosts[@]}" --decoder-ports "${decode_ports[@]}"
    ```

    Verify `/healthcheck` reports two P and sixteen D endpoints and inspect the proxy/worker file-descriptor limits. Use the same `ulimit -n 65536` on the benchmark client. Both P endpoints must receive requests; their separate APIs do not imply independent compute capacity, because expert/MC2 collectives can span both DP groups. Measure aggregate P throughput in one shared wall-clock window.

    P explicitly selects `ASCEND_LOCAL_COMM_RES` version 1.3 in the measured image. Verify the actual transport engine on both roles when changing images; an environment variable alone is not proof of matching engines. The historical `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` environment variable was not read by that image and is omitted here. Check the resolved `--prefix-cache-retention-interval` before changing retention; zero does not disable all prefix reuse. D stays in graph mode for target and draft. Recheck dispatch capacity, KV residency and throughput together when adjusting budgets; a low KV usage gauge does not exclude other memory or compute bottlenecks.

### 5.3 KV Cache Pool Deployment {: #53-memcache-kv-cache-pool-deployment }

This section provides PD separation deployment examples with a KV Cache Pool for cross-engine KV cache reuse. The 950DT and A3 examples use the MemCache backend, while the A2 examples cover both Mooncake and MemCache. Each hardware tab describes backend and storage configuration, Prefill and Decode startup steps, and KV cache reuse verification.

=== "950DT products"

    #### 950DT Single-Node PD Separation Deployment with Memcache KV Cache Pool {: #54-950dt-single-node-pd-separation-deployment }

    This section extends [the 950DT PD deployment example](#524-950dt-series-pd-separation-deployment) with a Memcache-backed KV Cache Pool on a single 950DT products node. Prefill uses NPUs `0,1,2,3`, while Decode uses NPUs `4,5,6,7`. The existing external DP load balancing, API ports, DP RPC ports, and proxy configuration remain unchanged.

    Reuse the shared `launch_online_dp.py` from [the 950DT PD deployment example](#524-950dt-series-pd-separation-deployment). Save the pooled templates below as `run_prefill_memcache.sh` and `run_decode_memcache.sh`, and select them with `--script`. These templates use `MultiConnector` and the required Memcache environment variables.

    For backend configuration and parameter descriptions, refer to [KV Cache Pool (Ascend Store)](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md#scenario-2-memcache-backend).

    1. <a id="5321-prerequisites-and-memcache-configuration"></a>**Prerequisites and MemCache Configuration**

        Configure the Memcache environment and configuration files inside the container where both Prefill and Decode run. Then, open separate terminals in the same container to launch Prefill on NPUs 0–3 and Decode on NPUs 4–7.

        Replace the paths below with the actual values in your environment. `INSTALL_PATH` is the `Location` reported by `pip show memcache_hybrid`. Set `MMC_LOCAL_CONFIG_PATH` to the absolute path of `mmc-local.conf`. Export the environment variables so they are inherited by the launcher and its child processes. Run these exports in each Prefill and Decode terminal before starting the corresponding launcher.

        ```bash
        # Set these variables to the actual paths in your environment.
        export MEMCACHE_CONFIG_DIR="/absolute/path/to/memcache-config"
        export INSTALL_PATH="<Location from pip show memcache_hybrid>"
        export MMC_LOCAL_CONFIG_PATH="${MEMCACHE_CONFIG_DIR}/mmc-local.conf"
        export PYTHONHASHSEED=0
        ```

        Configure one `mmc-meta.conf` for the MetaService:

        ```ini
        ock.mmc.meta_service_url = tcp://<metaservice_ip>:5000
        ock.mmc.meta_service.config_store_url = tcp://<metaservice_ip>:6000
        ock.mmc.meta_service.metrics_url = http://<metaservice_ip>:8000
        ock.mmc.log_level = info
        ```

        Configure `mmc-local.conf` for both roles on this node:

        ```ini
        ock.mmc.meta_service_url = tcp://<metaservice_ip>:5000
        ock.mmc.local_service.config_store_url = tcp://<metaservice_ip>:6000
        ock.mmc.log_level = info
        ock.mmc.local_service.world_size = 256
        ock.mmc.local_service.protocol = device_urma
        ock.mmc.local_service.dram.size = 40GB
        ock.mmc.local_service.max.dram.size = 2048GB
        ```

        Use `device_urma` for 950DT products Device UB. For UBOE, replace it with `device_uboe`. Both roles must connect to the same MetaService and ConfigStore. `world_size` is the maximum number of LocalServices supported by the pool; it is not the number of NPUs used by this example.

        This example uses the embedded LocalService mode. Each vLLM process initializes a LocalService by loading `mmc-local.conf` from `MMC_LOCAL_CONFIG_PATH`. Therefore, `ock.mmc.local_service.dram.size` must be greater than `0GB`. Do not start a separate standalone LocalService when using this configuration.

        In this example, the Prefill instance uses PCP4 together with `AscendStoreConnector` in layerwise mode. This example sets `use_layerwise` to `true` and `layerwise_prefetch_layers` to `3`.

    2. <a id="5322-prepare-the-prefill-startup-script"></a>**Prepare the Prefill Startup Script**

        Save the following template as `run_prefill_memcache.sh` in the same directory as the shared `launch_online_dp.py`.

        ```bash
        #!/usr/bin/env bash
        source /root/.bashrc
        nic_name="xxxx" # change to your own nic name
        local_ip=xx.xx.xx.1 # change to your own ip

        : "${MEMCACHE_CONFIG_DIR:?Set MEMCACHE_CONFIG_DIR before starting the launcher}"
        export MMC_LOCAL_CONFIG_PATH="${MEMCACHE_CONFIG_DIR}/mmc-local.conf"
        export HCCL_IF_IP=$local_ip
        export GLOO_SOCKET_IFNAME=$nic_name
        export TP_SOCKET_IFNAME=$nic_name
        export HCCL_SOCKET_IFNAME=$nic_name

        export VLLM_USE_V2_MODEL_RUNNER=1
        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
        export HCCL_EXEC_TIMEOUT=204
        export HCCL_CONNECT_TIMEOUT=120
        export PYTHONHASHSEED=0
        export OMP_PROC_BIND=false
        export OMP_NUM_THREADS=10
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export HCCL_BUFFSIZE=1024
        export TASK_QUEUE_ENABLE=1
        export HCCL_OP_EXPANSION_MODE=CCU_SCHED
        export VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096
        export VLLM_USE_FASTOKENS=1

        # With --pcp-size 4, the launcher passes 0,1,2,3 for this Prefill instance.
        export ASCEND_RT_VISIBLE_DEVICES="$1"
        export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'

        vllm serve /root/.cache/DeepSeek-V4-Flash-0731 \
            --host "$local_ip" \
            --port "$2" \
            --served-model-name dsv4 \
            --data-parallel-size "$3" \
            --data-parallel-rank "$4" \
            --data-parallel-address "$5" \
            --data-parallel-rpc-port "$6" \
            --tensor-parallel-size "$7" \
            --prefill-context-parallel-size "$8" \
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
            --additional-config '{"enable_cpu_binding":true,"enable_fused_mc2":0, "multistream_overlap_shared_expert": true, "enable_shared_expert_dp":true}' \
            --kv-transfer-config '{
                "kv_connector": "MultiConnector",
                "kv_role": "kv_producer",
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

    3. <a id="5323-prepare-the-decode-startup-script"></a>**Prepare the Decode Startup Script**

        Save the following template as `run_decode_memcache.sh` in the same directory as the shared `launch_online_dp.py`. It accepts the same eight arguments, maps device indices to NPUs `4`–`7`, and writes a separate log for each DP rank.

        ```bash
        #!/usr/bin/env bash
        source /root/.bashrc
        nic_name="xxxx" # change to your own nic name
        local_ip=xx.xx.xx.1 # change to your own ip

        : "${MEMCACHE_CONFIG_DIR:?Set MEMCACHE_CONFIG_DIR before starting the launcher}"
        export MMC_LOCAL_CONFIG_PATH="${MEMCACHE_CONFIG_DIR}/mmc-local.conf"
        export HCCL_IF_IP=$local_ip
        export GLOO_SOCKET_IFNAME=$nic_name
        export TP_SOCKET_IFNAME=$nic_name
        export HCCL_SOCKET_IFNAME=$nic_name

        export VLLM_USE_V2_MODEL_RUNNER=1
        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
        export HCCL_EXEC_TIMEOUT=204
        export HCCL_CONNECT_TIMEOUT=120
        export PYTHONHASHSEED=0
        export OMP_PROC_BIND=false
        export OMP_NUM_THREADS=10
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export HCCL_BUFFSIZE=1024
        export TASK_QUEUE_ENABLE=1
        export VLLM_USE_FASTOKENS=1

        # The Decode group uses NPUs 4-7; each DP rank sees one of those NPUs.
        export ASCEND_RT_VISIBLE_DEVICES="$(( $1 + 4 ))"
        export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'

        vllm serve /root/.cache/DeepSeek-V4-Flash-0731 \
            --host "$local_ip" \
            --served-model-name dsv4 \
            --port "$2" \
            --data-parallel-size "$3" \
            --data-parallel-rank "$4" \
            --data-parallel-address "$5" \
            --data-parallel-rpc-port "$6" \
            --tensor-parallel-size "$7" \
            --prefill-context-parallel-size "$8" \
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
            --gpu-memory-utilization 0.9 \
            --speculative-config '{"num_speculative_tokens":5,"method":"dspark","enforce_eager":false}' \
            --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
            --kv-transfer-config '{
                "kv_connector": "MultiConnector",
                "kv_role": "kv_consumer",
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
                "ascend_compilation_config": {"enable_npugraph_ex": true,"enable_static_kernel": false},
                "enable_cpu_binding": true,"multistream_overlap_shared_expert": true,"recompute_scheduler_enable": true,"enable_shared_expert_dp": true,"combine_quant_mode": 4
            }' \
            2>&1 | tee "v4_flash_d_memcache_dp${4}.log"
        ```

    4. <a id="5324-startup-and-verification"></a>**Startup and Verification**

        1. Start one MetaService on the node containing `mmc-meta.conf`:

            ```bash
            export MMC_META_CONFIG_PATH="<directory containing mmc-meta.conf>/mmc-meta.conf"
            python -c "from memcache_hybrid import MetaService; MetaService.main()"
            ```

            After the MetaService is ready, start Prefill and Decode in separate terminals in the same container. The startup sequence is MetaService → Prefill and Decode → Proxy.

        2. Start Prefill and Decode in separate terminals in the same container. Run both commands from the directory containing the shared launcher and the pooled scripts. Replace `MEMCACHE_CONFIG_DIR` with the directory prepared above. Both roles use the same node IP but different DP RPC ports.

            - Decode (NPUs 4–7)

                ```shell
                # Change the IP address to your own.
                python ./launch_online_dp.py --script ./run_decode_memcache.sh --dp-size 4 --tp-size 1 --pcp-size 1 --dp-size-local 4 --dp-rank-start 0 --dp-address xx.xx.xx.1 --dp-rpc-port 12325 --vllm-start-port 7200
                ```

            - Prefill (NPUs 0–3)

                ```shell
                # Change the IP address to your own.
                python ./launch_online_dp.py --script ./run_prefill_memcache.sh --dp-size 1 --tp-size 1 --pcp-size 4 --dp-size-local 1 --dp-rank-start 0 --dp-address xx.xx.xx.1 --dp-rpc-port 12321 --vllm-start-port 7100
                ```

        3. Start the Proxy and Verify KV Cache Pool Reuse

            Wait until the Prefill API endpoint and all four Decode API endpoints are healthy, then start the proxy using the configuration from [the 950DT PD deployment example](#524-950dt-series-pd-separation-deployment). Send a generation request to `<proxy_ip>:1999` as described in [Section 6](#6-functional-verification). Confirm that the response contains generated text.

            To verify KV Cache Pool reuse, send an initial request to warm up the cache, then send another request with the same identical token prefix. Check the Prefill logs for KV Pool lookup/get/put operations and cache-hit information. An HTTP 200 response alone does not confirm KV Cache Pool reuse. For details,refer to [KV Cache Pool (Ascend Store)](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md#scenario-2-memcache-backend).

    Key pooling parameters:

    - `AscendStoreConnector` integrates with the MemCache KV Cache Pool to support external KV cache storage and reuse. In this example, Prefill handles pool lookups, loading, and storage, while Decode receives KV cache from Prefill.

    - `backend: "memcache"` selects MemCache as the pool storage backend. On Prefill, `use_layerwise: true` enables layer-by-layer KV cache storage and loading, while `layerwise_prefetch_layers: 3` configures prefetching to load KV cache ahead of layer execution.

=== "A3 series"

    #### A3 1P1D Deployment with Memcache KV Cache Pool {: #53-a3-1p1d-deployment-with-memcache-kv-cache-pool }

    This example deploys the DSpark checkpoint `DeepSeek-V4-Flash-0731-w8a8` on
    two A3 servers (16 NPU dies per server), using a Memcache KV Cache Pool:

    | Role | DP | TP | PCP | NPU dies | HTTP ports |
    | --- | --- | --- | --- | --- | --- |
    | Prefill | 2 | 1 | 8 | 16 | 7100-7101 |
    | Decode | 16 | 1 | 1 | 16 | 7100-7115 |

    The deployment follows the same Memcache service setup as
    [DeepSeek-V4.1-Flash](DeepSeek-V4.1-Flash.md#531-prepare-the-memcache-services),
    while all non-pooling options and the parallel layout follow the
    [A3 series with DSpark](#a3-series-pd-separation-deployment-with-dspark) tab in Section 5.2. `MooncakeHybridConnector`
    continues to transfer KV
    cache from Prefill to Decode, while `AscendStoreConnector` stores reusable KV
    cache in Memcache. Both connectors are configured as children of
    `MultiConnector`.

    Before starting vLLM, follow the
    [Memcache backend guide](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md#scenario-2-memcache-backend)
    to check the HDK/CANN and A3 memory prerequisites, install `memfabric-hybrid`
    and `memcache-hybrid`, configure
    `mmc-meta.conf`, `mmc-local.conf`, and `mmc-local-standalone.conf`, and prepare
    the services below. The Prefill and Decode nodes must use the same MetaService endpoint. Keep the
    `/etc/hccn.conf` container mount from Section 4.1.

    This example uses separated Memcache deployment so that the pool reserves host
    memory before vLLM loads the model. Prepare one MetaService configuration and
    two LocalService configurations as described below.

    1. <a id="531-prepare-the-memcache-services"></a>**Prepare the Memcache Services**

        Create `mmc-meta.conf` on the node that runs MetaService:

        ```ini
        ock.mmc.log_level = info
        ock.mmc.meta_service_url = tcp://<META_SERVICE_IP>:15000
        ock.mmc.meta_service.config_store_url = tcp://<META_SERVICE_IP>:16000
        ock.mmc.meta_service.metrics_url = http://<META_SERVICE_IP>:18000
        ```

        Create `mmc-local.conf` on both Prefill and Decode nodes for the LocalService
        embedded in vLLM:

        ```ini
        ock.mmc.meta_service_url = tcp://<META_SERVICE_IP>:15000
        ock.mmc.local_service.config_store_url = tcp://<META_SERVICE_IP>:16000
        ock.mmc.log_level = info
        ock.mmc.local_service.world_size = 256
        ock.mmc.local_service.protocol = device_sdma
        ock.mmc.local_service.dram.size = 0GB
        ock.mmc.local_service.max.dram.size = 1024GB
        ock.mmc.local_service.storage.enabled = false
        ```

        Create `mmc-local-standalone.conf` on both nodes for the standalone Memcache
        process. It uses the same endpoints, world size, and protocol, but contributes
        the host DRAM used by the pool:

        ```ini
        ock.mmc.meta_service_url = tcp://<META_SERVICE_IP>:15000
        ock.mmc.local_service.config_store_url = tcp://<META_SERVICE_IP>:16000
        ock.mmc.log_level = info
        ock.mmc.local_service.world_size = 256
        ock.mmc.local_service.protocol = device_sdma
        ock.mmc.local_service.dram.size = 20GB
        ock.mmc.local_service.max.dram.size = 1024GB
        ock.mmc.local_service.storage.enabled = false
        ```

        `world_size` is the maximum number of LocalServices that can join the pool, not
        the DP size. Changing it after LocalServices connect requires restarting
        MetaService. `device_sdma` is recommended for A3 HCCS and requires LingQu
        Computing Network 1.5 or later. The `20GB` standalone DRAM value is an example;
        set it according to the host memory available after reserving capacity for the
        operating system and vLLM. `max.dram.size` must be at least the largest
        `dram.size` contributed by any LocalService.

        Start MetaService with `mmc-meta.conf`:

        ```shell
        export MMC_META_CONFIG_PATH="<MEMCACHE_CONFIG_DIR>/mmc-meta.conf"
        python3 -c 'from memcache_hybrid import MetaService; MetaService.main()'
        ```

        Save the following script as `memcache_standalone.py` on both Prefill and
        Decode nodes:

        ```python
        import time

        from memcache_hybrid import DistributedObjectStore


        store = DistributedObjectStore()
        result = store.init(0)
        if result != 0:
            raise RuntimeError(f"Failed to initialize Memcache: {result}")
        print("Successfully initialized Memcache")

        while True:
            time.sleep(1)
        ```

        After MetaService is ready, start the standalone Memcache process on both
        Prefill and Decode nodes and wait for successful initialization:

        ```shell
        export MMC_LOCAL_CONFIG_PATH="<MEMCACHE_CONFIG_DIR>/mmc-local-standalone.conf"
        python3 memcache_standalone.py
        ```

        For additional parameters and capacity constraints, refer to
        [Memcache and vLLM separated deployment](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md#Memcache-and-vLLM-Separated-Deployment).

        Reuse `launch_online_dp.py` from [the A3 PD deployment example](#521-a3-series-pd-separation-deployment). Set `MEMCACHE_INSTALL_PATH` to
        the `Location` reported by `pip show memcache_hybrid`, and set
        `MEMCACHE_CONFIG_DIR` to the absolute directory containing the configuration
        files above on both nodes. Use the same checkpoint on both nodes. Keep
        `/etc/hccn.conf` mounted, and verify both the NPU HCCS fabric and the host TCP
        network as described in Section 3.2. The MetaService, config store, DP RPC,
        KV transfer, and HTTP ports must be reachable between the nodes.

        The configuration above uses a DRAM-only pool. To enable SSD in this A3 separated deployment, keep `storage.enabled=false` in the LocalService configuration embedded in vLLM and use the following alternative `mmc-local-standalone-ssd.conf` for the standalone process on each node:

        ```ini
        ock.mmc.meta_service_url = tcp://<META_SERVICE_IP>:15000
        ock.mmc.local_service.config_store_url = tcp://<META_SERVICE_IP>:16000
        ock.mmc.log_level = info
        ock.mmc.local_service.world_size = 256
        ock.mmc.local_service.protocol = device_sdma
        ock.mmc.local_service.dram.size = 20GB
        ock.mmc.local_service.max.dram.size = 1024GB
        ock.mmc.local_service.storage.enabled = true
        ubsio.disk.path = <DEDICATED_SSD_BLOCK_DEVICE>
        ubsio.mem.size_in_gb = 50
        ubsio.standalone.device_count = 1
        ubsio.standalone.force_new_disk = true
        ```

        Use a dedicated, unmounted block device without a filesystem signature. `force_new_disk=true` reinitializes the SSD cache layout at startup and invalidates previous cache contents; do not point it at a device containing data to keep. The 20 GiB DRAM contribution and 50 GiB UBS IO allocation are separate per-process host-memory budgets. Size both for the actual prefix working set and available host memory. With one standalone process per host, use `device_count=1`; it is not the number of model ranks.

        After MetaService is ready, start one standalone process on each node with the selected SSD configuration, then all Decode ranks, both Prefill ranks and the proxy:

        ```shell
        export MMC_LOCAL_CONFIG_PATH="<MEMCACHE_CONFIG_DIR>/mmc-local-standalone-ssd.conf"
        python3 memcache_standalone.py
        ```

        Wait for `Successfully initialized Memcache` before starting vLLM. vLLM processes continue to use `mmc-local.conf`, not the standalone configuration. Follow [Enable Memcache SSD Cache](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md#step-7-enable-memcache-ssd-cache) for SDK requirements, storage ownership and capacity constraints. Verify real cache restore and backend hit activity; enabling an SSD flag or seeing HTTP 200 alone does not prove three-level cache reuse.

    2. <a id="532-start-the-prefill-node-with-pooling"></a>**Start the Prefill Node with Pooling**

        On the Prefill node, save the following script as `run_dp_template.sh` beside
        `launch_online_dp.py` from [the A3 PD deployment example](#521-a3-series-pd-separation-deployment). Replace the network, model, and
        Memcache path placeholders for this node.

        ```shell
        #!/usr/bin/env bash
        set -e

        unset https_proxy http_proxy

        nic_name="<NETWORK_INTERFACE>"
        local_ip="<PREFILL_NODE_IP>"

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
        export VLLM_USE_V2_MODEL_RUNNER=1
        export VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096

        # Parameters required by the Memcache KV Cache Pool.
        MEMCACHE_INSTALL_PATH="<MEMCACHE_INSTALL_PATH>"
        MEMCACHE_CONFIG_DIR="<MEMCACHE_CONFIG_DIR>"
        export PYTHONHASHSEED=0
        export MMC_LOCAL_CONFIG_PATH="$MEMCACHE_CONFIG_DIR/mmc-local.conf"
        export LD_LIBRARY_PATH="$MEMCACHE_INSTALL_PATH/memcache_hybrid/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}:/usr/local/lib/"

        # Ensure the model path matches the directory recorded during download
        vllm serve "<MODEL_PATH>/DeepSeek-V4-Flash-0731-w8a8" \
            --host 0.0.0.0 \
            --port $2 \
            --data-parallel-size $3 \
            --data-parallel-rank $4 \
            --data-parallel-address $5 \
            --data-parallel-rpc-port $6 \
            --tensor-parallel-size $7 \
            --prefill-context-parallel-size $8 \
            --enable-expert-parallel \
            --seed 1024 \
            --served-model-name dsv4 \
            --max-model-len 1048576 \
            --max-num-batched-tokens 16384 \
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
            --gpu-memory-utilization 0.94 \
            --quantization ascend \
            --enforce-eager \
            --enable-prefix-caching \
            --additional-config '{"enable_cpu_binding": true, "enable_fused_mc2": 1}' \
            --kv-transfer-config '{
            "kv_connector": "MultiConnector",
            "kv_role": "kv_producer",
            "kv_port": "30000",
            "kv_load_failure_policy": "fail",
            "kv_connector_extra_config": {
                "connectors": [
                    {
                        "kv_connector": "MooncakeHybridConnector",
                        "kv_role": "kv_producer",
                        "kv_port": "30000",
                        "kv_connector_extra_config": {
                            "prefill": {
                                "dp_size": 2,
                                "tp_size": 1,
                                "pcp_size": 8
                            },
                            "decode": {
                                "dp_size": 16,
                                "tp_size": 1
                            }
                        }
                    },
                    {
                        "kv_connector": "AscendStoreConnector",
                        "kv_role": "kv_producer",
                        "kv_connector_extra_config": {
                            "lookup_rpc_port": "0",
                            "backend": "memcache",
                            "use_layerwise": false
                        }
                    }
                ]
            }
        }'
        ```

    3. <a id="533-start-the-decode-node-with-pooling"></a>**Start the Decode Node with Pooling**

        On the Decode node, save the following script as `run_dp_template.sh` beside
        `launch_online_dp.py` from [the A3 PD deployment example](#521-a3-series-pd-separation-deployment). Replace the network, model, and
        Memcache path placeholders for this node.

        ```shell
        #!/usr/bin/env bash
        set -e

        unset https_proxy http_proxy

        nic_name="<NETWORK_INTERFACE>"
        local_ip="<DECODE_NODE_IP>"

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
        export VLLM_USE_V2_MODEL_RUNNER=1

        # Parameters required by the Memcache KV Cache Pool.
        MEMCACHE_INSTALL_PATH="<MEMCACHE_INSTALL_PATH>"
        MEMCACHE_CONFIG_DIR="<MEMCACHE_CONFIG_DIR>"
        export PYTHONHASHSEED=0
        export MMC_LOCAL_CONFIG_PATH="$MEMCACHE_CONFIG_DIR/mmc-local.conf"
        export LD_LIBRARY_PATH="$MEMCACHE_INSTALL_PATH/memcache_hybrid/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}:/usr/local/lib/"

        # Ensure the model path matches the directory recorded during download
        vllm serve "<MODEL_PATH>/DeepSeek-V4-Flash-0731-w8a8" \
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
            --speculative-config '{"num_speculative_tokens": 5,"method": "dspark"}' \
            --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
            --kv-transfer-config '{
            "kv_connector": "MultiConnector",
            "kv_role": "kv_consumer",
            "kv_port": "30100",
            "kv_load_failure_policy": "fail",
            "kv_connector_extra_config": {
                "connectors": [
                    {
                        "kv_connector": "MooncakeHybridConnector",
                        "kv_role": "kv_consumer",
                        "kv_port": "30100",
                        "kv_connector_extra_config": {
                            "prefill": {
                                "dp_size": 2,
                                "tp_size": 1,
                                "pcp_size": 8
                            },
                            "decode": {
                                "dp_size": 16,
                                "tp_size": 1
                            }
                        }
                    },
                    {
                        "kv_connector": "AscendStoreConnector",
                        "kv_role": "kv_consumer",
                        "kv_connector_extra_config": {
                            "lookup_rpc_port": "0",
                            "backend": "memcache",
                            "use_layerwise": false
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
                "recompute_scheduler_enable":true
            }'
        ```

    4. <a id="534-startup-and-verification"></a>**Startup and Verification**

        Start MetaService first, then the standalone Memcache process on each node.
        Wait for both standalone processes to report successful initialization.
        Keep all three processes running while vLLM is serving requests.

        Start Decode on the Decode node in a separate terminal:

        ```shell
        python launch_online_dp.py \
            --dp-size 16 --tp-size 1 --dp-size-local 16 --dp-rank-start 0 \
            --dp-address "<DECODE_NODE_IP>" --dp-rpc-port 12321 --vllm-start-port 7100
        ```

        Wait until all Decode engines report `Application startup complete`, then
        start Prefill on the Prefill node:

        ```shell
        python launch_online_dp.py \
            --dp-size 2 --tp-size 1 --pcp-size 8 --dp-size-local 2 --dp-rank-start 0 \
            --dp-address "<PREFILL_NODE_IP>" --dp-rpc-port 12321 --vllm-start-port 7100
        ```

        Each Prefill DP rank uses eight consecutive NPU dies: 0-7 and 8-15.
        Each Decode DP rank uses one die. `--dp-address` is the master IP of its own
        role; Prefill and Decode use separate DP groups.

        After every Prefill engine is ready, start the proxy from
        [Prefill-Decode Disaggregation (DeepSeek)](../features/pd_disaggregation_mooncake_multi_node.md)
        with both Prefill endpoints and all sixteen Decode endpoints:

        ```shell
        PREFILL_IP="<PREFILL_NODE_IP>"
        DECODE_IP="<DECODE_NODE_IP>"
        prefiller_hosts=()
        decoder_hosts=()
        for ((i=0; i<2; i++)); do prefiller_hosts+=("$PREFILL_IP"); done
        for ((i=0; i<16; i++)); do decoder_hosts+=("$DECODE_IP"); done

        python load_balance_proxy_server_example.py \
            --host "$PREFILL_IP" --port 1999 \
            --prefiller-hosts "${prefiller_hosts[@]}" \
            --prefiller-ports {7100..7101} \
            --decoder-hosts "${decoder_hosts[@]}" \
            --decoder-ports {7100..7115}
        ```

        Send requests to the proxy at ``http://<PREFILL_NODE_IP>:1999``, using the
        `dsv4` served model name from Section 6. Verify both the response and the
        P-to-D transfer in engine logs. To check pool reuse, send the same long prompt
        (at least 4096 tokens for this example) twice, allowing the first request's
        KV save to complete before sending the second. Confirm Memcache put activity
        on the first request and lookup/hit/get activity on a subsequent request in
        the Prefill logs. A request served entirely by a Prefill engine's local prefix
        cache may not load from Memcache; verify reuse on an engine whose local prefix
        cache is cold. A successful response alone does not confirm a pool hit.

        Key pooling parameters:

        - `MultiConnector` combines live P-to-D transfer through
          `MooncakeHybridConnector` with reusable prefix storage through
          `AscendStoreConnector`. In this example, both children and the parent use
          `kv_producer` on Prefill and `kv_consumer` on Decode. Decode does not put or
          load pool entries unless `consumer_is_to_put` or `consumer_is_to_load` is
          explicitly enabled.
        - `PYTHONHASHSEED=0` must match across all engines. Prefill enables prefix
          caching; Decode disables it. With `--block-size 32`, use
          `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096` on Prefill to retain hybrid
          attention checkpoints for reusable long prefixes.
        - Keep `--no-disable-hybrid-kv-cache-manager` on both roles and the top-level
          `kv_load_failure_policy` as `fail`; the `recompute` load-failure policy does
          not support DeepSeek-V4 hybrid attention.
        - `VLLM_USE_V2_MODEL_RUNNER=1` is required for PCP. Do not enable `enable_dsa_cp`
          with Model Runner V2. The two roles use the same DSpark checkpoint and five
          speculative tokens. Keep the connector's DP/TP/PCP sizes aligned with the
          launcher commands.
        - `lookup_rpc_port` selects the local scheduler-worker IPC endpoint. The
          implementation adds the DP rank, so this example uses `"0"` for each DP
          rank. Use a different value for another deployment sharing the same IPC
          namespace.

        For capacity tuning, optional SSD configuration, and troubleshooting, refer to
        [KV Cache Pool (Ascend Store)](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md).

=== "A2 series"

    #### A2 KV Pooling: Mooncake SSD Profile and Memcache Integration {: #526-a2-kv-pooling-mooncake-ssd-profile-and-memcache-integration }

    Pooling adds `AscendStoreConnector` to the Hybrid P-to-D transport through `MultiConnector`. A PC90 or third-level-pooling scenario name does not select a backend or prove that SSD offload is active. Select and prepare one backend, then copy its complete P/D scripts below. They use P token budget8192, P/D prefix caching enabled and D draft1/mns128/GMU0.97.

    | Backend | Preparation | Validation scope |
    |---|---|---|
    | `mooncake` | Embedded Store clients, a Mooncake master, DRAM and SSD configuration below | Backend used for the completed A2 PC90/SSD measurements; matching Mooncake client/native-library ABI 0.3.13 |
    | `memcache` | MetaService and embedded LocalServices, using the A2 RoCE protocol below | Integration recipe based on the Memcache backend guide; do not attribute the Mooncake profile's measured throughput to this backend |

    1. **Prepare the KV Cache Pool Backend**

        **Mooncake: prepare the Store and SSD layer.** Follow the [Mooncake backend prerequisites](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md#scenario-1-mooncake-backend) and keep the client wheel, native libraries and master compatible. The following is the measured configuration structure, with deployment addresses and paths replaced. Save `mooncake.json` on all four nodes:

        ```json
        {
          "metadata_server": "P2PHANDSHAKE",
          "protocol": "ascend",
          "device_name": "",
          "master_server_address": "<MOONCAKE_MASTER_IP>:50088",
          "global_segment_size": "1GB",
          "local_buffer_size": "1GB",
          "preferred_segment": false,
          "prefer_alloc_in_same_node": true,
          "enable_ssd_offload": true,
          "ssd_offload_path": "/nvme/a2_flash_store",
          "tenant_id": "a2-flash-pc90"
        }
        ```

        Use the same tenant and master endpoint on P and D. Create the SSD directory on each host and ensure it is writable. Confirm the installed backend assigns distinct worker storage paths; otherwise provide distinct paths per worker. Independent services with overlapping global ranks must not claim the same storage root. DRAM registration, local buffers and SSD capacity are per client/rank budgets, not a total-node memory cap.

        Start one master before starting vLLM. These offload-on-eviction settings follow the [SSD backend guide](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md#step-5-mooncakestore-ssd-offload-embedded-real-client-mode); verify flag support in the installed master:

        ```shell
        mooncake_master \
          --port 50088 --eviction_high_watermark_ratio 0.5 --eviction_ratio 0.1 \
          --default_kv_lease_ttl 11000 --enable_offload=true \
          --offload_on_evict=true --offload_force_evict=false --client_ttl=120
        ```

        After the Store is ready, use these complete P/D scripts. The Mooncake environment and the full `MultiConnector` JSON are included in each block.

        **Memcache: prepare the A2 embedded-client alternative.** Install compatible `memfabric-hybrid` and `memcache-hybrid` packages using the [Memcache backend guide](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md#scenario-2-memcache-backend). Use `device_rdma` for A2 RoCE; do not copy the A3 `device_sdma` protocol. The guide currently limits separated Memcache/vLLM deployment to A3, so the A2 template here uses embedded LocalServices.

        Save `mmc-meta.conf` on the MetaService host:

        ```ini
        ock.mmc.log_level = info
        ock.mmc.meta_service_url = tcp://<META_SERVICE_IP>:15000
        ock.mmc.meta_service.config_store_url = tcp://<META_SERVICE_IP>:16000
        ock.mmc.meta_service.metrics_url = http://<META_SERVICE_IP>:18000
        ```

        Save `mmc-local.conf` on all four nodes. This example contributes 1 GiB per LocalService and enables DRAM pooling:

        ```ini
        ock.mmc.meta_service_url = tcp://<META_SERVICE_IP>:15000
        ock.mmc.local_service.config_store_url = tcp://<META_SERVICE_IP>:16000
        ock.mmc.log_level = info
        ock.mmc.local_service.world_size = 256
        ock.mmc.local_service.protocol = device_rdma
        ock.mmc.local_service.dram.size = 1GB
        ock.mmc.local_service.max.dram.size = 1024GB
        ock.mmc.local_service.storage.enabled = false
        ```

        `world_size` is the maximum number of LocalServices, not the model DP size. Size DRAM for the prefix working set and all processes before launching. This template does **not** enable SSD; use the [Memcache SSD instructions](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md#step-7-enable-memcache-ssd-cache) with a supported hardware/deployment combination if SSD is required. The A3 separated SSD example is described in the [A3 KV Pooling tab](#53-a3-1p1d-deployment-with-memcache-kv-cache-pool).

        Start MetaService and wait for it to report readiness:

        ```shell
        export MMC_META_CONFIG_PATH="<CONFIG_DIR>/mmc-meta.conf"
        python3 -c 'from memcache_hybrid import MetaService; MetaService.main()'
        ```

        After MetaService is ready, use these complete Memcache P/D scripts. Set `MEMCACHE_INSTALL_PATH` to the `Location` reported by `pip show memcache_hybrid` and edit the configuration path.

    2. **Prepare the Prefill Startup Scripts**

        **Mooncake backend**

        **Prefill: run this complete script on each P node.** Set `p_rank=0` on node A and `p_rank=1` on node B; keep `head_ip` pointing at P node A. Start both ranks before waiting for readiness.

        ```shell
        set -euo pipefail
        nic_name="<NETWORK_INTERFACE>"
        local_ip="<THIS_PREFILL_NODE_IP>"
        head_ip="<PREFILL_NODE_A_IP>"
        p_rank=0  # Set to 1 on Prefill node B.
        model_path="/path/to/DeepSeek-V4-Flash-0731-w8a8"

        export ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.protocol_desc":["roce:device","hccs:device"]}'
        export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'
        export VLLM_USE_V2_MODEL_RUNNER=1
        export HCCL_INTRA_ROCE_ENABLE=1
        export HCCL_CONNECT_TIMEOUT=6000 HCCL_EXEC_TIMEOUT=204 HCCL_BUFFSIZE=1200
        export GLOO_SOCKET_IFNAME="$nic_name" HCCL_SOCKET_IFNAME="$nic_name" TP_SOCKET_IFNAME="$nic_name"
        export HCCL_IF_IP="$local_ip"
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000 VLLM_RPC_TIMEOUT=3600000
        export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        export MOONCAKE_CONFIG_PATH="<CONFIG_DIR>/mooncake.json"
        export MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES=34359738368
        export MOONCAKE_OFFLOAD_BUCKET_MAX_TOTAL_SIZE=34359738368
        export MOONCAKE_OFFLOAD_BUCKET_EVICTION_POLICY=lru
        export MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES=1073741824
        # Compatibility settings used by the measured client/native-library build.
        export MOONCAKE_OFFLOAD_FILE_STORAGE_PATH=/nvme/a2_flash_store
        export MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR=bucket_storage_backend
        export ASCEND_GLOBAL_RESOURCE_CONFIG=""
        kv_config='{
          "kv_connector": "MultiConnector",
          "kv_role": "kv_producer",
          "kv_load_failure_policy": "fail",
          "kv_connector_extra_config": {
            "connectors": [
              {
                "kv_connector": "MooncakeHybridConnector",
                "kv_role": "kv_producer",
                "kv_port": "30000",
                "kv_connector_extra_config": {
                  "prefill": {
                    "dp_size": 2,
                    "tp_size": 1,
                    "pp_size": 1,
                    "pcp_size": 8
                  },
                  "decode": {
                    "dp_size": 16,
                    "tp_size": 1,
                    "pp_size": 1,
                    "pcp_size": 1
                  }
                },
                "kv_load_failure_policy": "fail"
              },
              {
                "kv_connector": "AscendStoreConnector",
                "kv_role": "kv_producer",
                "kv_connector_extra_config": {
                  "backend": "mooncake",
                  "lookup_rpc_port": "'"$p_rank"'",
                  "use_layerwise": false
                }
              }
            ]
          }
        }'

        vllm serve "$model_path" \
          --host 0.0.0.0 --port "$((7100 + p_rank))" \
          --data-parallel-size 2 --data-parallel-rank "$p_rank" \
          --data-parallel-address "$head_ip" --data-parallel-rpc-port 12321 \
          --tensor-parallel-size 1 --pipeline-parallel-size 1 \
          --prefill-context-parallel-size 8 --decode-context-parallel-size 1 \
          --served-model-name dsv4 --enable-expert-parallel \
          --max-model-len 150000 --max-num-batched-tokens 8192 \
          --max-num-seqs 16 --api-server-count 4 \
          --block-size 32 --gpu-memory-utilization 0.96 \
          --enable-chunked-prefill --no-disable-hybrid-kv-cache-manager \
          --enable-prefix-caching --enforce-eager \
          --quantization ascend --tokenizer-mode deepseek_v4 \
          --attention_config.indexer_kv_dtype int8 \
          --tool-call-parser deepseek_v4 --enable-auto-tool-choice \
          --reasoning-parser deepseek_v4 --trust-remote-code --seed 1024 \
          --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":128}' \
          --speculative-config '{"method":"dspark","num_speculative_tokens":5,"enforce_eager":true}' \
          --additional-config '{"enable_cpu_binding":true,"enable_flashcomm1":true,"enable_fused_mc2":1}' \
          --kv-transfer-config "$kv_config"
        ```

        **Memcache backend**

        **Prefill: run this complete script on each P node.** Set `p_rank=0` on node A and `p_rank=1` on node B; keep `head_ip` pointing at P node A. Start both ranks before waiting for readiness.

        ```shell
        set -euo pipefail
        nic_name="<NETWORK_INTERFACE>"
        local_ip="<THIS_PREFILL_NODE_IP>"
        head_ip="<PREFILL_NODE_A_IP>"
        p_rank=0  # Set to 1 on Prefill node B.
        model_path="/path/to/DeepSeek-V4-Flash-0731-w8a8"

        export ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.protocol_desc":["roce:device","hccs:device"]}'
        export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'
        export VLLM_USE_V2_MODEL_RUNNER=1
        export HCCL_INTRA_ROCE_ENABLE=1
        export HCCL_CONNECT_TIMEOUT=6000 HCCL_EXEC_TIMEOUT=204 HCCL_BUFFSIZE=1200
        export GLOO_SOCKET_IFNAME="$nic_name" HCCL_SOCKET_IFNAME="$nic_name" TP_SOCKET_IFNAME="$nic_name"
        export HCCL_IF_IP="$local_ip"
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000 VLLM_RPC_TIMEOUT=3600000
        export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        MEMCACHE_INSTALL_PATH="<MEMCACHE_INSTALL_PATH>"
        export LD_LIBRARY_PATH="$MEMCACHE_INSTALL_PATH/memcache_hybrid/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        export MMC_LOCAL_CONFIG_PATH="<CONFIG_DIR>/mmc-local.conf"
        export PYTHONHASHSEED=0
        kv_config='{
          "kv_connector": "MultiConnector",
          "kv_role": "kv_producer",
          "kv_load_failure_policy": "fail",
          "kv_connector_extra_config": {
            "connectors": [
              {
                "kv_connector": "MooncakeHybridConnector",
                "kv_role": "kv_producer",
                "kv_port": "30000",
                "kv_connector_extra_config": {
                  "prefill": {
                    "dp_size": 2,
                    "tp_size": 1,
                    "pp_size": 1,
                    "pcp_size": 8
                  },
                  "decode": {
                    "dp_size": 16,
                    "tp_size": 1,
                    "pp_size": 1,
                    "pcp_size": 1
                  }
                },
                "kv_load_failure_policy": "fail"
              },
              {
                "kv_connector": "AscendStoreConnector",
                "kv_role": "kv_producer",
                "kv_connector_extra_config": {
                  "backend": "memcache",
                  "lookup_rpc_port": "'"$p_rank"'",
                  "use_layerwise": false
                }
              }
            ]
          }
        }'

        vllm serve "$model_path" \
          --host 0.0.0.0 --port "$((7100 + p_rank))" \
          --data-parallel-size 2 --data-parallel-rank "$p_rank" \
          --data-parallel-address "$head_ip" --data-parallel-rpc-port 12321 \
          --tensor-parallel-size 1 --pipeline-parallel-size 1 \
          --prefill-context-parallel-size 8 --decode-context-parallel-size 1 \
          --served-model-name dsv4 --enable-expert-parallel \
          --max-model-len 150000 --max-num-batched-tokens 8192 \
          --max-num-seqs 16 --api-server-count 4 \
          --block-size 32 --gpu-memory-utilization 0.96 \
          --enable-chunked-prefill --no-disable-hybrid-kv-cache-manager \
          --enable-prefix-caching --enforce-eager \
          --quantization ascend --tokenizer-mode deepseek_v4 \
          --attention_config.indexer_kv_dtype int8 \
          --tool-call-parser deepseek_v4 --enable-auto-tool-choice \
          --reasoning-parser deepseek_v4 --trust-remote-code --seed 1024 \
          --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":128}' \
          --speculative-config '{"method":"dspark","num_speculative_tokens":5,"enforce_eager":true}' \
          --additional-config '{"enable_cpu_binding":true,"enable_flashcomm1":true,"enable_fused_mc2":1}' \
          --kv-transfer-config "$kv_config"
        ```

    3. **Prepare the Decode Startup Scripts**

        **Mooncake backend**

        **Decode: run this complete script on each D node.** Set `rank_offset=0` on node A and `rank_offset=8` on node B; keep `head_ip` pointing at D node A. Launch all ranks before waiting for readiness.

        ```shell
        set -euo pipefail
        nic_name="<NETWORK_INTERFACE>"
        local_ip="<THIS_DECODE_NODE_IP>"
        head_ip="<DECODE_NODE_A_IP>"
        model_path="/path/to/DeepSeek-V4-Flash-0731-w8a8"
        rank_offset=0  # Set to 8 on Decode node B.

        export ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.protocol_desc":["roce:device","hccs:device"]}'
        export VLLM_USE_V2_MODEL_RUNNER=1
        export HCCL_INTRA_ROCE_ENABLE=1
        export HCCL_CONNECT_TIMEOUT=120 HCCL_EXEC_TIMEOUT=204 HCCL_BUFFSIZE=1800
        export GLOO_SOCKET_IFNAME="$nic_name" HCCL_SOCKET_IFNAME="$nic_name" TP_SOCKET_IFNAME="$nic_name"
        export HCCL_IF_IP="$local_ip"
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000 VLLM_RPC_TIMEOUT=3600000
        export OMP_NUM_THREADS=10 OMP_PROC_BIND=false
        export MOONCAKE_CONFIG_PATH="<CONFIG_DIR>/mooncake.json"
        export MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES=34359738368
        export MOONCAKE_OFFLOAD_BUCKET_MAX_TOTAL_SIZE=34359738368
        export MOONCAKE_OFFLOAD_BUCKET_EVICTION_POLICY=lru
        export MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES=1073741824
        # Compatibility settings used by the measured client/native-library build.
        export MOONCAKE_OFFLOAD_FILE_STORAGE_PATH=/nvme/a2_flash_store
        export MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR=bucket_storage_backend
        export ASCEND_GLOBAL_RESOURCE_CONFIG=""

        for local_rank in 0 1 2 3 4 5 6 7; do
          global_rank=$((local_rank + rank_offset))
          export ASCEND_RT_VISIBLE_DEVICES="$local_rank"
          kv_config='{
            "kv_connector": "MultiConnector",
            "kv_role": "kv_consumer",
            "kv_load_failure_policy": "fail",
            "kv_connector_extra_config": {
              "connectors": [
                {
                  "kv_connector": "MooncakeHybridConnector",
                  "kv_role": "kv_consumer",
                  "kv_port": "30100",
                  "kv_connector_extra_config": {
                    "prefill": {
                      "dp_size": 2,
                      "tp_size": 1,
                      "pp_size": 1,
                      "pcp_size": 8
                    },
                    "decode": {
                      "dp_size": 16,
                      "tp_size": 1,
                      "pp_size": 1,
                      "pcp_size": 1
                    }
                  }
                },
                {
                  "kv_connector": "AscendStoreConnector",
                  "kv_role": "kv_consumer",
                  "kv_connector_extra_config": {
                    "backend": "mooncake",
                    "lookup_rpc_port": "'"$((global_rank + 2))"'",
                    "use_layerwise": false,
                    "consumer_is_to_load": true
                  }
                }
              ]
            }
          }'
          vllm serve "$model_path" \
            --host 0.0.0.0 --port "$((7100 + local_rank))" \
            --data-parallel-size 16 --data-parallel-rank "$global_rank" \
            --data-parallel-address "$head_ip" --data-parallel-rpc-port 12321 \
            --tensor-parallel-size 1 --enable-expert-parallel --seed 1024 \
            --max-model-len 508576 --max-num-batched-tokens 720 \
            --max-num-seqs 128 --async-scheduling --block-size 32 \
            --served-model-name dsv4 --attention_config.indexer_kv_dtype int8 \
            --enable-prefix-caching --no-disable-hybrid-kv-cache-manager \
            --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":128}' \
            --trust-remote-code --tokenizer-mode deepseek_v4 \
            --tool-call-parser deepseek_v4 --enable-auto-tool-choice \
            --reasoning-parser deepseek_v4 --gpu-memory-utilization 0.97 \
            --quantization ascend \
            --speculative-config "{\"num_speculative_tokens\":1,\"method\":\"dspark\"}" \
            --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
            --kv-transfer-config "$kv_config" \
            --additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},"enable_cpu_binding":true,"multistream_overlap_shared_expert":true,"recompute_scheduler_enable":true}' &
          sleep 3
        done
        wait
        ```

        The two disk limits are 32 GiB per participating client/rank and the offload buffer is 1 GiB; budget their sum across the node. Clearing `ASCEND_GLOBAL_RESOURCE_CONFIG` belongs to this measured Mooncake Store path, not to pooling in general. The environment variables alone do not enable SSD: the JSON switch and master offload settings are also required. In this profile P **and D** use `--enable-prefix-caching`; the pooled scripts replace the ordinary D `--no-enable-prefix-caching` flag. D's Store child sets `consumer_is_to_load=true`.

        **Memcache backend**

        **Decode: run this complete script on each D node.** Set `rank_offset=0` on node A and `rank_offset=8` on node B; keep `head_ip` pointing at D node A. Launch all ranks before waiting for readiness.

        ```shell
        set -euo pipefail
        nic_name="<NETWORK_INTERFACE>"
        local_ip="<THIS_DECODE_NODE_IP>"
        head_ip="<DECODE_NODE_A_IP>"
        model_path="/path/to/DeepSeek-V4-Flash-0731-w8a8"
        rank_offset=0  # Set to 8 on Decode node B.

        export ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.protocol_desc":["roce:device","hccs:device"]}'
        export VLLM_USE_V2_MODEL_RUNNER=1
        export HCCL_INTRA_ROCE_ENABLE=1
        export HCCL_CONNECT_TIMEOUT=120 HCCL_EXEC_TIMEOUT=204 HCCL_BUFFSIZE=1800
        export GLOO_SOCKET_IFNAME="$nic_name" HCCL_SOCKET_IFNAME="$nic_name" TP_SOCKET_IFNAME="$nic_name"
        export HCCL_IF_IP="$local_ip"
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000 VLLM_RPC_TIMEOUT=3600000
        export OMP_NUM_THREADS=10 OMP_PROC_BIND=false
        MEMCACHE_INSTALL_PATH="<MEMCACHE_INSTALL_PATH>"
        export LD_LIBRARY_PATH="$MEMCACHE_INSTALL_PATH/memcache_hybrid/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        export MMC_LOCAL_CONFIG_PATH="<CONFIG_DIR>/mmc-local.conf"
        export PYTHONHASHSEED=0

        for local_rank in 0 1 2 3 4 5 6 7; do
          global_rank=$((local_rank + rank_offset))
          export ASCEND_RT_VISIBLE_DEVICES="$local_rank"
          kv_config='{
            "kv_connector": "MultiConnector",
            "kv_role": "kv_consumer",
            "kv_load_failure_policy": "fail",
            "kv_connector_extra_config": {
              "connectors": [
                {
                  "kv_connector": "MooncakeHybridConnector",
                  "kv_role": "kv_consumer",
                  "kv_port": "30100",
                  "kv_connector_extra_config": {
                    "prefill": {
                      "dp_size": 2,
                      "tp_size": 1,
                      "pp_size": 1,
                      "pcp_size": 8
                    },
                    "decode": {
                      "dp_size": 16,
                      "tp_size": 1,
                      "pp_size": 1,
                      "pcp_size": 1
                    }
                  }
                },
                {
                  "kv_connector": "AscendStoreConnector",
                  "kv_role": "kv_consumer",
                  "kv_connector_extra_config": {
                    "backend": "memcache",
                    "lookup_rpc_port": "'"$((global_rank + 2))"'",
                    "use_layerwise": false,
                    "consumer_is_to_load": true
                  }
                }
              ]
            }
          }'
          vllm serve "$model_path" \
            --host 0.0.0.0 --port "$((7100 + local_rank))" \
            --data-parallel-size 16 --data-parallel-rank "$global_rank" \
            --data-parallel-address "$head_ip" --data-parallel-rpc-port 12321 \
            --tensor-parallel-size 1 --enable-expert-parallel --seed 1024 \
            --max-model-len 508576 --max-num-batched-tokens 720 \
            --max-num-seqs 128 --async-scheduling --block-size 32 \
            --served-model-name dsv4 --attention_config.indexer_kv_dtype int8 \
            --enable-prefix-caching --no-disable-hybrid-kv-cache-manager \
            --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":128}' \
            --trust-remote-code --tokenizer-mode deepseek_v4 \
            --tool-call-parser deepseek_v4 --enable-auto-tool-choice \
            --reasoning-parser deepseek_v4 --gpu-memory-utilization 0.97 \
            --quantization ascend \
            --speculative-config "{\"num_speculative_tokens\":1,\"method\":\"dspark\"}" \
            --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
            --kv-transfer-config "$kv_config" \
            --additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},"enable_cpu_binding":true,"multistream_overlap_shared_expert":true,"recompute_scheduler_enable":true}' &
          sleep 3
        done
        wait
        ```

    4. **Startup and Verification**

        **Start and verify the selected pool.** Bring up the backend first, then all D ranks and both P ranks using the same backend, wait for their readiness, and start the proxy. For the Memcache alternative, revalidate model initialization and capacity before adopting any throughput setting. For Mooncake SSD, verify worker offload activation, actual disk writes and successful same-key restore. For Memcache, inspect lookup/get/put and hit activity. HTTP 200 and an installed SDK do not establish pool reuse.

        For PC90, warm multiple distinct prefix groups, then send suffixes not used during warmup. Repeating whole warmed prompts measures near-PC100. Keep the requested prefix ratio separate from observed P local-cache and D external-KV hit rates. Warmup is excluded from formal timing. Run diagnostic probes and formal loads serially, drain all 18 endpoints between rounds, and retain full success/token counts, native timing windows, launch configuration and source identity. Do not claim every request used SSD, or transfer a throughput result from one backend to another.

## 6 Functional Verification

This section verifies basic inference functionality for all deployment modes. For Memcache pooled PD deployments, also verify external KV Cache Pool reuse using the [A3 startup and verification steps](#534-startup-and-verification) or the “Start the Proxy and Verify KV Cache Pool Reuse” step in the [950DT example](#54-950dt-single-node-pd-separation-deployment).

Once your server is started, you can query the model with input prompts:

Replace `<node0_ip>` and `<port>` with the IP address and port of the API server for the single-node deployment in Section 5.1. For the PD separation deployment in Section 5.2 or the KV Cache Pool deployment in Section 5.3, use the proxy's IP address and port, since requests are routed through the proxy to the Prefill and Decode instances.

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
| GPQA | - | accuracy | gen | 88.17 | 1 Atlas 800I A3 (128GB × 8) |
| GSM8K | - | accuracy | gen | 96.30 | 1 Atlas 800I A3 (128GB × 8) |
| GPQA | - | accuracy | gen | 90.40 | A3 1P1D DSpark w8a8 |
| SWE Multilingual | - | accuracy | gen | 68.33 | A3 1P1D DSpark w8a8 |
| GPQA | - | accuracy | gen | 90.40 | 950PR&950DT products, w4a4c8 |
| GPQA | - | accuracy | gen | 90.91 | 950DT products DSpark, w4a8c8_mxfp |

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

A: For 950PR&950DT products, the W4A4C8 weights of `DeepSeek-V4-Flash-0731` are quantized with [msmodelslim](https://gitcode.com/Ascend/msmodelslim/blob/master/lab_practice/deepseek_v4/deepseek_v4_flash_w4a4c8.yaml). To preserve accuracy at such a low bit-width, a mixed-precision strategy is used: only the modules that dominate the parameter count are quantized to 4 bit, while precision-sensitive modules fall back to higher precision or are not quantized at all.

| Module | Quantization configuration |
| ------ | -------------------------- |
| MoE FFN routed experts | W4A4 MXFP4 |
| Attention linear layers | W8A8 MXFP8 |
| FFN shared experts | W8A8 MXFP8 |
| Router gates, compressor projections (`wgate`/`wkv`), indexer projections, and MTP layers | Not quantized (kept in high precision) |

- **Quantization algorithm**: The weights and activations use the MXFP4/MXFP8 microscaling formats. Each tensor is divided into blocks of 32 elements, and each block shares a single scaling factor obtained with symmetric min-max calibration on a mixed calibration dataset. Block-wise scaling adapts to the local dynamic range of each tensor, so the quantization error is far lower than that of uniform INT4 quantization.
- **C8 (FP8 KV cache)**: C8 is a dynamic quantization applied to the KV cache during inference. On 950PR&950DT products, the KV cache is quantized to FP8 (not INT8). It is unrelated to the parameters stored in the weight checkpoint and is controlled only by the serving configuration. DeepSeek-V4-Flash and DeepSeek-V4-Pro share exactly the same C8 scheme:
    - **Attention KV cache** (FP8 by default on 950PR&950DT products; switch back to BF16 with `--kv-cache-dtype bfloat16`): the compressed KV latent, which serves as both K and V, is dynamically quantized per 64-element tile to FP8 (E4M3) with a shared E8M0 scale per tile; the RoPE part of the KV cache stays in BF16, and Q is not quantized.
    - **Indexer cache** (enabled with `--attention_config.indexer_kv_dtype fp8`): both indexer Q and K are first rotated by a Hadamard matrix to spread outliers, and then dynamically quantized per token to FP8 (E4M3) with per-token FP32 scales.

For W4A4C8 accuracy scores, see [Accuracy Evaluation](#7-accuracy-evaluation).
