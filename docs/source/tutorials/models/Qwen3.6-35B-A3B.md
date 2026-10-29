# Qwen3.6-35B-A3B

## 1 Introduction

!!! note

    The "PD Separation Deployment" mentioned in this document is provided only as an example/reference design to demonstrate the basic process of deploying Prefill and Decode in separated mode. It is not recommended for use in the production environment.

    **For the production environment on the customer side, [MindIE Motor](https://www.hiascend.com/document/detail/en/mindie/latest/quickstart/docs/en/user_guide/quick_start.md) + vLLM-Ascend is recommended.**

Qwen3.6-35B-A3B is a sparse MoE model in the Qwen3.6 family, with 35B total parameters and about 3B activated parameters per token. It uses the hybrid attention architecture used by Qwen3.5-style models, and is suitable for long-context online serving on Ascend hardware.

This document describes the main validation steps for the model, including supported features, prerequisites, installation, single-node online deployment, functional verification, accuracy and performance evaluation, performance tuning, and FAQs.

The `Qwen3.6-35B-A3B` model has been supported since vLLM-Ascend `v0.18.0rc1`. For 950DT Products, use `v0.30.0` or later. This guide's deployment examples target vLLM-Ascend `v0.30.0`; when using a newer version, check its release notes for configuration changes. The examples below use the version placeholder configured by the documentation build system.

## 2 Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_features.md) to get the model's supported feature matrix, including BF16, W8A8 quantization, chunked prefill, automatic prefix caching, asynchronous scheduling, tensor parallelism, expert parallelism, and ACLGraph support.

Refer to [Feature Guide](../../user_guide/feature_guide/index.md) to get feature configuration details.

## 3 Prerequisites

### 3.1 Model Weight

|  Weight Version  | Hardware Requirements | Download Links |
|------------------|-----------------------|----------------|
| `Qwen3.6-35B-A3B` (BF16 version) | 1 950DT Products (96GB x 8) node, 1 A3 series (64GB x 16) node, 1 A2 series (64GB x 8) node, or Atlas 300I DUO | [ModelScope](https://www.modelscope.cn/models/Qwen/Qwen3.6-35B-A3B) |
| `Qwen3.6-35B-A3B-w8a8` (quantized version) | 1 A3 series (64GB x 16) node, 1 A2 series (64GB x 8) node, or Atlas 300I DUO | [ModelScope](https://www.modelscope.cn/models/Eco-Tech/Qwen3.6-35B-A3B-w8a8) |
| `Qwen3.6-35B-A3B-w8a8-mxfp8` (quantized version) | 1 950DT Products (96GB x 8) node | [ModelScope](https://www.modelscope.cn/models/Eco-Tech/Qwen3.6-35B-A3B-w8a8-mxfp8) |
| `Qwen3.6-35B-A3B-w8a8c8-mxfp8` (quantized version) | 1 950DT Products (96GB x 8) node | [ModelScope](https://www.modelscope.cn/models/Eco-Tech/Qwen3.6-35B-A3B-w8a8c8-mxfp8) |
| `Qwen3.6-35B-A3B-w4a8-mxfp` (quantized version) | 1 950DT Products (96GB x 8) node | [ModelScope](https://www.modelscope.cn/models/Eco-Tech/Qwen3.6-35B-A3B-w4a8-mxfp) |

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`.

>**Path description**: Download the model weights to a directory of your choice and record it. Ensure the model path in the subsequent deployment command matches this directory.

## 4 Installation

### 4.1 Docker Image Installation

Refer to [using docker](../../getting_started/installation.md#installation-prebuilt-image) for the complete installation guide.

=== "950DT Products"

    ```bash
    # Download the model weight to /root/.cache in advance.
    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-a5
    export NAME=vllm-ascend

    docker run --rm \
      --name $NAME \
      --net=host \
      --shm-size=500g \
      --device /dev/davinci_manager \
      --device /dev/hisi_hdc \
      --device /dev/ummu \
      --device /dev/uburma \
      --device /dev/davinci0 \
      --device /dev/davinci1 \
      --device /dev/davinci2 \
      --device /dev/davinci3 \
      --device /dev/davinci4 \
      --device /dev/davinci5 \
      --device /dev/davinci6 \
      --device /dev/davinci7 \
      -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
      -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
      -v /root/host:/root/host  \
      -v /usr/local/sbin:/usr/local/sbin \
      -v /usr/local/dcmi:/usr/local/dcmi \
      -v /var/log/npu:/usr/slog \
      -v /etc/ascend_install.info:/etc/ascend_install.info \
      -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
      -v /etc/hixlep:/etc/hixlep \
      -v /root/.cache:/root/.cache \
      -it $IMAGE bash
    ```

=== "A3 Series"

    ```bash

    # Download the model weight to /root/.cache in advance.
    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-a3
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

=== "A2 Series"

    ```bash

    # Download the model weight to /root/.cache in advance.
    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}
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

=== "Atlas 300I DUO"

    #### Standard container

    ```bash

    # Download the model weight to /root/.cache in advance.
    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-310p

    docker run --rm \
    --name vllm-ascend \
    --shm-size=10g \
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
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /root/.cache:/root/.cache \
    -p 8080:8080 \
    -it $IMAGE bash
    ```

After entering the container, verify that vLLM and vLLM-Ascend can be imported:

```shell
python -c "import vllm, vllm_ascend; print('vllm and vllm_ascend are ready')"
```

### 4.2 Source Code Installation

You can also build and install `vllm-ascend` from source. Refer to [set up using python](../../getting_started/installation.md#installation-existing-cann-install).

!!! note

    For Atlas 300I DUO, source installation may pull in `triton` and `triton-ascend`. Uninstall them before running vLLM-Ascend on Atlas 300I DUO:

    ```bash
    pip uninstall -y triton-ascend triton
    ```

## 5 Online Service Deployment {: #5-online-service-deployment }

### 5.1 Single-Node Online Deployment

Single-node deployment runs both Prefill and Decode on the same node. `Qwen3.6-35B-A3B-w8a8` can be deployed on 1 A3 series (64GB x 16) or 1 A2 series (64GB x 8), or Atlas 300I DUO. `Qwen3.6-35B-A3B-w8a8-mxfp8` can be deployed on a single NPU of a 950DT Products (96GB x 8) node. The quantized versions need `--quantization ascend`.

=== "950DT Products"

    Run the following script to execute online inference with up to 135168 context length on a single NPU of a 950DT Products (96GB x 8) node using `Qwen3.6-35B-A3B-w8a8-mxfp8`.

    ```shell
    export VLLM_USE_MODELSCOPE=True
    export ASCEND_RT_VISIBLE_DEVICES=0
    export VLLM_USE_V2_MODEL_RUNNER=1
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export TASK_QUEUE_ENABLE=1

    vllm serve Eco-Tech/Qwen3.6-35B-A3B-w8a8-mxfp8 \
      --host 0.0.0.0 \
      --port 8000 \
      --served-model-name qwen3.6 \
      --trust-remote-code \
      --quantization ascend \
      --safetensors-load-strategy lazy \
      --tensor-parallel-size 1 \
      --data-parallel-size 1 \
      --max-model-len 135168 \
      --max-num-seqs 128 \
      --max-num-batched-tokens 4096 \
      --gpu-memory-utilization 0.90 \
      --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}' \
      --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
      --additional-config '{"enable_cpu_binding":true,"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":true,"enable_super_kernel":false}}' \
      --enable-prefix-caching
    ```

    !!! note

        When using the C8 weights (`Qwen3.6-35B-A3B-w8a8c8-mxfp8`), add the following option to the `vllm serve` deployment command:

        ```shell
        --kv-cache-dtype mxfp8
        ```

        Enabling static kernel generation can increase compilation time during startup. Wait for the service to finish graph capture and report that it is ready before sending requests.

    **Key parameters:**

    - `ASCEND_RT_VISIBLE_DEVICES=0`, `--tensor-parallel-size 1`, and `--data-parallel-size 1` run a single model instance on the selected NPU without tensor or data parallelism. Adjust the visible device according to the available NPUs.
    - `VLLM_USE_V2_MODEL_RUNNER=1` selects Model Runner V2 for this deployment example.
    - `--quantization ascend` enables Ascend quantization for the W8A8 MXFP8 model.
    - `--safetensors-load-strategy lazy` uses lazy loading for Safetensors model weights.
    - `--max-model-len 135168` sets the maximum input plus output length for a single request. Ensure enough KV cache is available for the required context length.
    - `--max-num-seqs 128` sets the maximum number of active requests scheduled concurrently. Reduce it if KV cache or graph capture memory is insufficient.
    - `--max-num-batched-tokens 4096` limits the number of tokens processed in one scheduler step. Increase it only after checking prefill throughput and activation memory usage.
    - `--gpu-memory-utilization 0.90` sets the fraction of device memory available to the model executor when calculating KV cache capacity. Reduce it if startup or runtime requests report OOM.
    - `--speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}'` enables Qwen3.5-style MTP speculative decoding with three draft tokens per step. Tune the number of speculative tokens based on acceptance rate, request concurrency, and measured latency and throughput.
    - `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'` enables full decode ACLGraph replay to reduce dispatch overhead.
    - `--additional-config` enables Ascend-native CPU binding with `enable_cpu_binding`. Its `ascend_compilation_config` enables npugraph_ex and static kernel generation; `enable_super_kernel:false` keeps Super Kernel disabled instead of inheriting the static kernel setting.
    - `--enable-prefix-caching` enables reuse of cached KV blocks for requests with shared prefixes. Monitor KV cache usage for long-context workloads.
    - For the C8 weight version, `--kv-cache-dtype mxfp8` selects MXFP8 KV cache storage.

=== "A2 Series / A3 Series"

    ```shell
    # Load model from ModelScope to speed up download.
    export VLLM_USE_MODELSCOPE=True
    export HCCL_BUFFSIZE=1024
    export ASCEND_RT_VISIBLE_DEVICES=0,1
    export VLLM_USE_V2_MODEL_RUNNER=1
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
    export HCCL_OP_EXPANSION_MODE="AIV"
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export TASK_QUEUE_ENABLE=1

    # Reduce memory fragmentation and avoid out-of-memory errors.
    echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
    sysctl -w vm.swappiness=0
    sysctl -w kernel.numa_balancing=0
    sysctl kernel.sched_migration_cost_ns=50000

    # Ensure the model path matches the directory recorded during download
    vllm serve Eco-Tech/Qwen3.6-35B-A3B-w8a8 \
      --host 0.0.0.0 \
      --port 8000 \
      --data-parallel-size 1 \
      --tensor-parallel-size 2 \
      --enable-expert-parallel \
      --quantization ascend \
      --served-model-name qwen3.6 \
      --max-num-seqs 128 \
      --max-model-len 135168 \
      --max-num-batched-tokens 4096 \
      --trust-remote-code \
      --safetensors-load-strategy lazy \
      --gpu-memory-utilization 0.90 \
      --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}' \
      --dtype float16 \
      --enable-prefix-caching \
      --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
      --additional-config '{"enable_cpu_binding":true, "multistream_overlap_shared_expert": true}'
    ```

    **Key parameters:**

    - `ASCEND_RT_VISIBLE_DEVICES=0,1`, `--tensor-parallel-size 2`, and `--data-parallel-size 1` run one model instance across the two selected NPUs with tensor parallelism and one data-parallel group.
    - `VLLM_USE_V2_MODEL_RUNNER=1` selects Model Runner V2 for this deployment example.
    - `HCCL_BUFFSIZE=1024` sets the HCCL communication buffer size for this example.
    - `--enable-expert-parallel` enables expert parallelism for MoE layers.
    - `--safetensors-load-strategy lazy` uses lazy loading for Safetensors model weights.
    - `--dtype float16` selects FP16 for this deployment example.
    - `--max-model-len 135168` sets the maximum input plus output length for a single request. Ensure enough KV cache is available for the required context length.
    - `--max-num-seqs 128` sets the maximum number of active requests scheduled concurrently. Reduce it if KV cache or graph capture memory is insufficient.
    - `--max-num-batched-tokens 4096` limits the number of tokens processed in one scheduler step. Increase it only after checking prefill throughput and activation memory usage.
    - `--gpu-memory-utilization 0.90` sets the fraction of device memory available to the model executor when calculating KV cache capacity. Reduce it if startup or runtime requests report OOM.
    - `--speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}'` enables Qwen3.5-style MTP speculative decoding with three draft tokens per step.
    - `--additional-config '{"enable_cpu_binding":true, "multistream_overlap_shared_expert": true}'` enables Ascend-native CPU binding and shared-expert computation overlap.
    - `--enable-prefix-caching` enables prefix caching. For long-context serving, monitor memory usage because prefix caching can increase KV cache pressure.
    - `--quantization ascend` enables Ascend quantization for the W8A8 model used in this example.
    - `--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'` enables the hybrid full and piecewise ACLGraph mode.

=== "Atlas 300I DUO"

    ```shell
    export VLLM_USE_MODELSCOPE=True
    export ASCEND_RT_VISIBLE_DEVICES=0,1

    # Ensure the model path matches the directory recorded during download
    vllm serve Eco-Tech/Qwen3.6-35B-A3B-w8a8 \
      --host 127.0.0.1 \
      --port 8080 \
      --tensor-parallel-size 2 \
      --gpu-memory-utilization 0.90 \
      --max-num-seqs 16 \
      --served-model-name qwen3.6 \
      --dtype float16 \
      --additional-config '{"ascend_compilation_config": {"enable_npugraph_ex":false}}' \
      --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1,8]}' \
      --quantization ascend \
      --max-model-len 20480 \
      --no-enable-prefix-caching
    ```

    **Key parameters:**

    - `--tensor-parallel-size 2` maps the model across two Atlas inference devices. Adjust it together with `ASCEND_RT_VISIBLE_DEVICES` according to the available devices and memory.
    - `--dtype float16` is used for Atlas 300I DUO to match the Atlas inference execution path.
    - `--max-num-seqs 16` limits concurrent active requests to reduce KV cache and graph capture pressure on Atlas 300I DUO.
    - `--gpu-memory-utilization` controls KV cache capacity. Reduce it if startup or runtime requests report OOM.
    - `--additional-config` with `"ascend_compilation_config": {"enable_npugraph_ex": false}` is required because `enable_npugraph_ex` is not supported on Atlas 300I DUO.
    - `--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1,8]}'` enables decode ACLGraph replay and explicitly limits capture sizes for Atlas 300I DUO.
    - `--no-enable-prefix-caching` is the default recommendation for this Atlas 300I DUO example to reduce memory pressure.
    - `--quantization ascend` enables Ascend quantization for the W8A8 model. Remove this option when deploying the BF16 model.
    - To enable MTP speculative decoding, use `--speculative_config '{"method": "mtp", "num_speculative_tokens": 1}'`. We recommend setting num_speculative_tokens to 1. If your usage scenario involves fewer than two concurrent requests, it is recommended to enable MTP. Otherwise, it is recommended not to enable MTP.

Common Issues Tip: If the service fails to start, HBM is insufficient, or requests are not scheduled as expected, refer to [Public FAQs](../../faqs.md) first, and then check the model-specific FAQ in [Section 10](#10-faq).

**Service Verification:**

Loading the model weights and capturing graphs can take time, especially when static kernel generation is enabled in the 950DT example. Startup duration depends on storage speed and the selected configuration. Use the health check below to confirm that the service is ready.

Use the API port configured in the selected deployment command. For example, check the 950DT deployment from its serving node or container on port `8000`:

```shell
curl -fsS -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/health
```

For other deployments, use their configured API port. The Atlas 300I DUO example listens on `127.0.0.1:8080`, so run its check on the serving node or container. A healthy service returns HTTP `200`. If the check fails, inspect the startup log for errors and wait for initialization to finish. After the health check passes, use [Section 6](#6-functional-verification) to verify that the model generates a response.

## 6 Functional Verification

After the server is started, send a request to verify basic model functionality.

```shell
curl http://<server_ip>:<port>/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6",
    "prompt": "The future of AI is",
    "max_tokens": 50,
    "temperature": 0
  }'
```

Expected result: the HTTP status is 200 and the JSON response contains a `choices` field with generated text.

## 7 Accuracy Evaluation

### 7.1 Using AISBench

Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) for details. After execution, you can get the accuracy result of `Qwen3.6-35B-A3B-w8a8`.

For MMMU, follow the [AISBench MMMU dataset guide](https://github.com/AISBench/benchmark/blob/master/ais_bench/benchmark/configs/datasets/mmmu/README_en.md)
to download [AI-ModelScope/MMMU](https://modelscope.cn/datasets/AI-ModelScope/MMMU/summary) into `ais_bench/datasets/mmmu`.

| dataset | metric | mode | hardware | vllm-api-general-chat |
| ------- | ------ | ---- | -------- | --------------------- |
| mmmu | accuracy | gen | 1 A3 series (64GB x 16) node | 83.3 |
| gpqa | accuracy | gen | 1 A3 series (64GB x 16) node | 83.3 |

### 7.2 Using Language Model Evaluation Harness

Refer to [Using lm_eval](../../developer_guide/evaluation/using_lm_eval.md) for installation and usage details. When using online serving, set `base_url` to the endpoint started in [Section 5](#5-online-service-deployment).

```shell
lm_eval \
  --model local-completions \
  --model_args model=qwen3.6,base_url=http://127.0.0.1:8000/v1/completions,tokenized_requests=False,trust_remote_code=True \
  --tasks gsm8k \
  --output_path ./
```

## 8 Performance Evaluation

### 8.1 Using AISBench

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

### 8.2 Using vLLM Benchmark

Refer to [vLLM benchmark](https://docs.vllm.ai/en/latest/benchmarking/) for more details.

## 9 Performance Tuning

### 9.1 Recommended Configurations

The following configurations are for reference only. The optimal configuration depends on hardware type, maximum input/output length, request concurrency, prefix cache hit rate, and quantization. Tune the parameters in [Section 9.2](#92-tuning-guidelines) based on your actual workload.

The 950DT Products and A2/A3 Series configurations below follow their deployment examples in [Section 5.1](#51-single-node-online-deployment). The NPU count is the number used by the service, rather than the total number available on the node.

| Scenario | Deployment Mode | Total NPUs | Weight Version | Key Considerations |
| -------- | --------------- | ---------- | -------------- | ------------------ |
| Long context | Single-node online serving | 2 or more NPUs | W8A8 | Use larger `--max-model-len` and reserve enough KV cache. Lower `--max-num-seqs` if OOM occurs. |
| Long context | Single-node online serving | 1 950DT Products NPU | W8A8 MXFP8 | [Section 5.1](#51-single-node-online-deployment) uses TP=1, DP=1, `--max-model-len 135168`, `--max-num-batched-tokens 4096`, static kernel generation, and MTP with 3 speculative tokens. |
| Long context | Single-node online serving | 2 A2/A3 Series NPUs | W8A8 | [Section 5.1](#51-single-node-online-deployment) uses TP=2, DP=1, `--max-model-len 135168`, `--max-num-batched-tokens 4096`, and MTP with 3 speculative tokens. |
| High throughput | Single-node online serving | 8 or more NPUs | W8A8 | Increase local DP groups within one node and tune `--max-num-batched-tokens`. |
| Low latency | Single-node online serving | 2 or more NPUs | W8A8 | Use smaller `--max-num-batched-tokens`, full decode ACLGraph, and disable speculative decoding by default. |

| Scenario | Node Role | NPUs | TP | DP | Max Num Seqs | Max Model Len | Max Num Batched Tokens | Prefix Cache | Main Optimizations |
| -------- | --------- | ---- | -- | -- | ------------ | ------------- | ---------------------- | ------------ | ------------------ |
| Long context | Single node | 2 or more | 2 | 1 | 128 | 262144 | 16384 | On | FullGraph, sequence parallelism, shared expert overlap, CPU binding |
| Long context (950DT Products) | Single node | 1 | 1 | 1 | 128 | 135168 | 4096 | On | Model Runner V2, full decode ACLGraph, static kernel, MTP, CPU binding |
| Long context (A2/A3 Series) | Single node | 2 | 2 | 1 | 128 | 135168 | 4096 | On | Model Runner V2, `FULL_AND_PIECEWISE` ACLGraph, expert parallelism, shared expert overlap, MTP, CPU binding |
| High throughput | Single node | 8 or more | 2 | 4 or more | 32 per DP | 65536 | 8192 | On | FullGraph, sequence parallelism, async scheduling, shared expert overlap |
| Low latency | Single node | 2 or more | 2 | 1 | Tune by concurrency | 32768 or 65536 | 1024 to 4096 | Workload dependent | FullGraph, CPU binding, speculative decoding disabled |

### 9.2 Tuning Guidelines

Refer to [Public Performance Tuning Documentation](../../developer_guide/performance_and_debug/optimization_and_tuning.md) for general tuning methods, and refer to [Feature Matrix](../../user_guide/support_matrix/feature_matrix.md) for feature descriptions.

Recommended tuning order:

1. Use single-node deployment. If more throughput is required, increase local DP groups within the same node.
2. Choose the maximum context length with `--max-model-len`. Long context increases KV cache usage, so reduce `--max-num-seqs` or `--gpu-memory-utilization` if OOM occurs.
3. Tune `--max-num-batched-tokens`. Larger values usually improve prefill throughput but increase activation memory. Decode-heavy workloads usually need smaller values.
4. Tune `--max-num-seqs` according to service concurrency. Requests above this value wait in the queue and the waiting time is counted in TTFT and TPOT.
5. Tune `--gpu-memory-utilization`. Increase it to provide more KV cache, but leave headroom for runtime memory fluctuation and expert imbalance.
6. Tune ACLGraph capture. `FULL_DECODE_ONLY` is recommended for decode. If you set `cudagraph_capture_sizes` manually, include common decode batch sizes. With sequence parallelism, use capture sizes that are multiples of TP size.

### 9.3 Model-Specific Optimizations

| Optimization | Enablement | Benefit | Notes |
| ------------ | ---------- | ------- | ----- |
| Hybrid attention support | Enabled by model implementation | Supports Qwen3.6 long-context inference. | Tune context length based on KV cache capacity. |
| Full decode ACLGraph | `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'` | Reduces operator dispatch overhead and stabilizes decode performance. | Recommended for decode-heavy serving. |
| Shared expert overlap | `--additional-config '{"multistream_overlap_shared_expert": true}'` | Overlaps shared expert computation in MoE workloads. | Recommended for throughput scenarios. |
| Prefix caching | `--enable-prefix-caching` | Improves repeated-prefix workloads. | Monitor HBM usage for long-context workloads; keep prefix caching disabled on Atlas 300I DUO (see [Section 5.1](#51-single-node-online-deployment)).  |
| Qwen3.6 MTP speculative decoding | `--speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": true}'` | Can improve decode throughput when stable and accepted tokens are high. | In [Section 5.1](#51-single-node-online-deployment), 950DT Products and A2/A3 Series use 3 speculative tokens; the optional Atlas 300I DUO configuration uses 1. Validate stability, TTFT, TPOT, and throughput for your workload. |

## 10 FAQ

For common environment, installation, and general parameter issues, refer to [Public FAQs](../../faqs.md). This section only covers model-specific issues for Qwen3.6-35B-A3B.

### 10.1 Why does the service report OOM during startup or soon after accepting requests?

**Phenomenon:** The service fails during profile run, or it starts successfully but reports OOM when real traffic arrives.

**Cause:** Qwen3.6 long-context serving consumes a large KV cache. Large `--max-model-len`, large `--max-num-seqs`, large `--max-num-batched-tokens`, or high `--gpu-memory-utilization` can leave insufficient HBM headroom.

**Solution:** Use the W8A8 model with `--quantization ascend` when possible, lower `--max-model-len`, lower `--max-num-seqs`, lower `--max-num-batched-tokens`, or reduce `--gpu-memory-utilization`. Keep `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`.

### 10.2 Why does enabling prefix caching not improve performance?

**Phenomenon:** Prefix caching is enabled, but throughput or latency does not improve.

**Cause:** Prefix caching only helps when requests share reusable prefixes. For random prompts or low cache hit rates, it may add memory pressure without visible gains.

**Solution:** Enable prefix caching for repeated-prefix workloads. For random benchmark datasets or memory-constrained long-context workloads, compare with `--no-enable-prefix-caching`.
