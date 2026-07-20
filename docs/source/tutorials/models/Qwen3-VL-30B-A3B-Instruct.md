# Qwen3-VL-30B-A3B-Instruct

## 1 Introduction

Qwen3-VL-30B-A3B-Instruct is a sparse MoE vision-language model in the Qwen3-VL family, with about 30B total parameters and about 3B activated parameters per token. It is suitable for image understanding, video understanding, multimodal dialogue, and long-context online serving on Ascend hardware.

This document describes the main validation steps for the model, including supported features, prerequisites, installation, image and video online deployment, offline inference, functional verification, accuracy and performance evaluation, performance tuning, and FAQs.

The `Qwen3-VL-30B-A3B-Instruct` tutorial was introduced for the `vllm-ascend` `v0.13.0` validation cycle. Use `v0.13.0` or later for this model. The examples below use the version placeholder configured by the documentation build system.

## 2 Supported Features

Refer to [Supported Features List](../../user_guide/support_matrix/supported_features.md) to get the model's supported feature matrix.

Refer to [Feature Guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## 3 Prerequisites

### 3.1 Model Weight

- `Qwen3-VL-30B-A3B-Instruct` (BF16 version): requires 1 Atlas 800 A3 (64G x 16) node or 1 Atlas 800 A2 (64G x 8) node. [Model Weight](https://modelscope.cn/models/Qwen/Qwen3-VL-30B-A3B-Instruct).
- `Qwen3-VL-30B-A3B-Instruct-w8a8-mxfp8` (quantized version): requires 1 Ascend950DT (96G x 8) node. [Model Weight]#TODO

It is recommended to download the model weight to a shared directory across multiple nodes.

## 4 Installation

### 4.1 Docker Image Installation

Select an image based on your machine type and start the docker image on your node, refer to [using docker](../../installation.md#set-up-using-docker).

:::::{tab-set}
:sync-group: install

::::{tab-item} Ascend950DT series
:sync: 950dt

Start the docker image on your each node.

```{code-block} bash
  :substitutions:
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|-#TODO
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
  -itd $IMAGE bash
```

::::
::::{tab-item} A3 series
:sync: A3

Start the docker image on each node.

```{code-block} bash
   :substitutions:

export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|-a3
docker run --rm \
    --name vllm-ascend \
    --shm-size=512g \
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
    -v /etc/hccn.conf:/etc/hccn.conf \
    -it $IMAGE bash
```

::::
::::{tab-item} A2 series
:sync: A2

Start the docker image on each node.

```{code-block} bash
   :substitutions:

export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
docker run --rm \
    --name vllm-ascend \
    --shm-size=512g \
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
    -v /etc/hccn.conf:/etc/hccn.conf \
    -it $IMAGE bash
```

::::
:::::

After a successful docker run, you can verify the running container service by executing the `docker ps` command.

### 4.2 Source Code Installation

If you prefer not to use the Docker image, you can build from source. Install vLLM from source first:

1. Clone and install vLLM:

   ```bash
   git clone https://github.com/vllm-project/vllm.git
   cd vllm
   pip install -e .
   ```

2. Clone and install the vLLM-Ascend repository:

   ```bash
   git clone https://github.com/vllm-project/vllm-ascend.git
   cd vllm-ascend
   pip install -e .
   ```

**Installation Verification:**

```bash
pip show vllm vllm-ascend
```

Expected result: The version information for both packages is displayed, confirming a successful installation.

:::{note}
If deploying a multi-node environment, set up the environment on each node.
:::

For more details, please refer to the [Installation Guide](../../installation.md).

## 5 Online Service Deployment

### 5.1 Single-Node Online Deployment

Single-node deployment runs both Prefill and Decode on the same node. The following examples are suitable for image-only online serving.

:::::{tab-set}
:sync-group: deploy

::::{tab-item} Ascend950DT series
:sync: 950dt

Run the following script to execute online inference on 1 Ascend950DT (96G x 8). The quantized version (`Qwen3-VL-30B-A3B-Instruct-w8a8-mxfp8`) can be deployed on a single Ascend950DT node. The W8A8 version needs `--quantization ascend`.

```shell
#!/bin/sh

# Load model from ModelScope to speed up download.
export VLLM_USE_MODELSCOPE=True

# Reduce memory fragmentation and avoid out-of-memory errors.
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=400
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=100
export TASK_QUEUE_ENABLE=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

vllm serve Eco-Tech/Qwen3-VL-30B-A3B-Instruct-w8a8-mxfp8 \
  --host 0.0.0.0 \
  --port 8000 \
  --distributed-executor-backend mp \
  --data-parallel-size 1 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --seed 1024 \
  --quantization ascend \
  --served-model-name qwen3-vl-30b \
  --max-num-seqs 32 \
  --max-model-len 32768 \
  --max-num-batched-tokens 8192 \
  --trust-remote-code \
  --no-enable-prefix-caching \
  --mm-processor-cache-gb 0 \
  --limit-mm-per-prompt.image 1 \
  --limit-mm-per-prompt.video 0 \
  --gpu-memory-utilization 0.9 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
```

::::

::::{tab-item} A3 series
:sync: A3

Run the following script to start image-only serving on 1 Atlas 800 A3 (64G x 16) node.

```shell
#!/bin/sh

# Load model from ModelScope to speed up download.
export VLLM_USE_MODELSCOPE=True

# Reduce memory fragmentation and avoid out-of-memory errors.
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=1024
export OMP_NUM_THREADS=1
export OMP_PROC_BIND=false
export TASK_QUEUE_ENABLE=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_ENABLE_FUSED_MC2=1

vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name qwen3-vl-30b \
  --data-parallel-size 1 \
  --tensor-parallel-size 2 \
  --enable-expert-parallel \
  --seed 1024 \
  --max-num-seqs 32 \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.9 \
  --no-enable-prefix-caching \
  --mm-processor-cache-gb 0 \
  --limit-mm-per-prompt.image 1 \
  --limit-mm-per-prompt.video 0 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,24,32]}'
```

::::

::::{tab-item} A2 series
:sync: A2

The A3 series script above also works on 1 Atlas 800 A2 (64G x 8) node.

::::
:::::

Key Parameter Descriptions:

- `--tensor-parallel-size 2` maps the model to two NPUs. Increase TP only after validating memory, communication, and throughput on your hardware.
- `--enable-expert-parallel` enables expert parallelism for MoE layers. Do not mix MoE tensor parallelism and expert parallelism in the same MoE layer.
- `--max-model-len` is the maximum input plus output length for a single request. By default, the model can support long context, but `128000` is a practical validation value for many image/video workloads.
- `--max-num-seqs` is the maximum number of active requests scheduled by each DP group. Video requests consume more memory, so the video example uses a smaller value.
- `--max-num-batched-tokens` is the maximum number of tokens processed in one scheduler step. A larger value can improve prefill efficiency but consumes more activation memory.
- `--gpu-memory-utilization` controls how much HBM vLLM can use to calculate KV cache capacity. Increase it only after confirming the service is stable.
- `--limit-mm-per-prompt.video 0` disables video inputs and saves memory for image-only serving.
- `--allowed-local-media-path /media` allows requests to use local files such as `file:///media/test.mp4`.
- `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'` enables full decode ACLGraph replay to reduce dispatch overhead.

Common Issues Tip: If you encounter issues, please refer to the [Public FAQ](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html) for troubleshooting.

Service Verification:

```shell
curl http://<server_ip>:<port>/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen3-vl-30b",
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

## 6 Functional Verification

After the server is started, send a request to verify basic multimodal functionality.

```shell
curl http://<server_ip>:<port>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-vl-30b",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://modelscope.oss-cn-beijing.aliyuncs.com/resource/qwen.png"}},
        {"type": "text", "text": "What is the text in the illustration?"}
      ]}
    ],
    "max_completion_tokens": 100,
    "temperature": 0
  }'
```

Expected result: the HTTP status is 200 and the JSON response contains a `choices` field with generated text, for example text similar to `TONGYI Qwen`.

## 7 Accuracy Evaluation

### Using AISBench

1. Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) for details.

2. After execution, you can get the result.

| dataset | version | metric | mode | result |
| ------- | ------- | ------ | ---- | ------ |
| mmmu_val | - | acc,none | gen | 0.58 |

## 8 Performance Evaluation

### 8.1 Using AISBench

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details. For image or video performance, use a dataset with real multimodal payloads instead of random text-only prompts.

### 8.2 Using vLLM Benchmark

Run performance evaluation of `Qwen3-VL-30B-A3B-Instruct` as an example. Refer to [vLLM benchmark](https://docs.vllm.ai/en/latest/benchmarking/) for more details.

There are three `vllm bench` subcommands:

- `latency`: benchmark the latency of a single batch of requests.
- `serve`: benchmark online serving throughput.
- `throughput`: benchmark offline inference throughput.

Take `serve` as an example:

```shell
export VLLM_USE_MODELSCOPE=True

vllm bench serve \
  --model Qwen/Qwen3-VL-30B-A3B-Instruct \
  --served-model-name qwen3-vl-30b \
  --dataset-name random \
  --random-input 200 \
  --num-prompts 200 \
  --request-rate 1 \
  --save-result \
  --result-dir ./
```

After several minutes, you can get the performance evaluation result. This random benchmark is useful for serving pipeline validation; use AISBench or a custom multimodal dataset for image/video-token performance.

## 9 Performance Tuning

### 9.1 Recommended Configurations

> **Note**: The following configurations are validated in specific test environments and are for reference only. The optimal configuration depends on hardware type, image resolution, video length, maximum input/output length, request concurrency, prefix cache hit rate, and prefill/decode ratio. Tune the parameters in Section 9.2 based on your actual workload.

#### Table 1: Scenario Overview

| Scenario | Deployment Mode | *Total NPUs | Weight Version | Key Considerations |
| -------- | --------------- | ---------- | -------------- | ------------------ |
| Image-only serving | Single-node online serving | 2 or more NPUs | BF16 | Disable video, tune context length, and keep enough KV cache for visual tokens. |
| Video serving | Single-node online serving | 2 or more NPUs | BF16 | Use local media paths, lower concurrency, and reduce video length or frame sampling if OOM occurs. |
| Functional graph validation | Single-node PP | 2 NPUs | BF16 | Use shorter context and explicit capture sizes to validate full decode ACLGraph behavior. |

> `*Total NPUs` indicates the total number of NPUs used across all nodes. 1 node = 1 Atlas 800 A3 server (64G × 16 NPUs).

#### Table 2: Detailed Node Configuration

| Scenario | Node Role | NPUs | TP | PP | Max Num Seqs | Max Model Len | Max Num Batched Tokens | Prefix Cache | Main Optimizations |
| -------- | --------- | ---- | -- | -- | ------------ | ------------- | ---------------------- | ------------ | ------------------ |
| Image-only serving | Single node | 2 or more | 2 | 1 | 16 | 128000 | 4096 | Workload dependent | FullGraph, EP, video disabled |
| Video serving | Single node | 2 or more | 2 | 1 | 8 | 128000 | 4096 | Workload dependent | FullGraph, EP, local media path |
| Graph validation | Single node | 2 | 1 | 2 | Tune by test | 4096 | 1024 | Off | FullGraph capture sizes |

> For complete startup commands and parameter descriptions, please refer to the deployment examples in [Chapter 5](#5-online-service-deployment).

### 9.2 Tuning Guidelines

#### 9.2.1 General Tuning Reference

Please refer to the [Public Performance Tuning Documentation](../../developer_guide/performance_and_debug/optimization_and_tuning.md) for tuning methods.

Please refer to the [Feature Guide](../../user_guide/support_matrix/feature_matrix.md) for detailed feature descriptions.

#### 9.2.2 Recommended tuning order

1. Start from image-only serving. Add video only after the image path is stable.
2. Choose the maximum context length with `--max-model-len`. Multimodal requests consume KV cache for both text tokens and visual tokens, so reduce image resolution, video length, request concurrency, or context length if OOM occurs.
3. Tune multimodal limits. Use `--limit-mm-per-prompt.image` and `--limit-mm-per-prompt.video` to match your request shape.
4. Tune `--max-num-batched-tokens`. Larger values usually improve prefill throughput but increase activation memory. Video-heavy workloads usually need conservative values.
5. Tune `--max-num-seqs` according to service concurrency. Video requests are more memory intensive than image requests, so start with a smaller value.
6. Tune `--gpu-memory-utilization`. Increase it to provide more KV cache, but leave headroom for runtime memory fluctuation and media preprocessing.
7. Tune ACLGraph capture. `FULL_DECODE_ONLY` is recommended for decode. If you set `cudagraph_capture_sizes` manually, include common decode batch sizes.

### 9.3 Model-Specific Optimizations

| Optimization | Enablement | Benefit | Notes |
| ------------ | ---------- | ------- | ----- |
| Multimodal prompt limits | `--limit-mm-per-prompt.image`, `--limit-mm-per-prompt.video` | Avoids reserving memory for unused media types. | Disable video for image-only serving. |
| Local media access | `--allowed-local-media-path /media` | Avoids slow network video downloads during serving. | Use `file:///media/...` in requests. |
| Full decode ACLGraph | `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'` | Reduces operator dispatch overhead and stabilizes decode performance. | Recommended for decode-heavy serving. |
| Expert parallelism | `--enable-expert-parallel` | Improves MoE serving throughput. | Do not mix MoE tensor parallelism and expert parallelism in the same MoE layer. |
| Prefix caching | `--enable-prefix-caching` | Improves repeated-prefix workloads. | Random prompts or unique media may not benefit. |
| Asynchronous scheduling | `--async-scheduling` | Can improve high-concurrency throughput. | Disable and compare for latency-sensitive workloads. |
| Pipeline parallel validation | `--pipeline-parallel-size 2` | Provides another two-card validation layout. | Use shorter context and lower batch tokens for functional tests. |

## 10 FAQ

For common environment, installation, and general parameter issues, refer to [Public FAQs](../../faqs.md). This section only covers model-specific issues for Qwen3-VL-30B-A3B-Instruct.

### Q1: Why does the service report OOM during startup?

**Phenomenon:** The service fails during profile run or exits before accepting requests.

**Cause:** Long context, high image resolution, video inputs, large `--max-num-seqs`, large `--max-num-batched-tokens`, or high `--gpu-memory-utilization` can leave insufficient HBM headroom.

**Solution:** Start with image-only serving, set `--limit-mm-per-prompt.video 0`, reduce `--max-model-len`, lower `--max-num-seqs`, lower `--max-num-batched-tokens`, or reduce `--gpu-memory-utilization`. Keep `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`.

### Q2: Why is video disabled in the image-only command?

**Phenomenon:** The service reserves more memory than expected even when requests only contain images.

**Cause:** Allowing video inputs can reserve memory for long visual embeddings and preprocessing paths.

**Solution:** Use `--limit-mm-per-prompt.video 0` for image-only serving. Enable video only when the workload needs it.

### Q3: Why does the video request fail with a local file path?

**Phenomenon:** The request reports that the file is not allowed or cannot be found.

**Cause:** The server can only access local media paths that are mounted into the container and allowed by `--allowed-local-media-path`.

**Solution:** Mount the host media directory to `/media`, start the server with `--allowed-local-media-path /media`, and use a request URL like `file:///media/test.mp4`.

### Q4: Why does enabling prefix caching not improve performance?

**Phenomenon:** Prefix caching is enabled, but throughput or latency does not improve.

**Cause:** Prefix caching only helps when requests share reusable prefixes. Unique images, unique videos, or random prompts may add memory pressure without visible gains.

**Solution:** Enable prefix caching for repeated-prefix workloads. For random benchmarks or memory-constrained video workloads, compare with prefix caching disabled.

### Q5: Why does multimodal accuracy evaluation fail to insert image tokens?

**Phenomenon:** Evaluation fails because image placeholders cannot be found in the prompt.

**Cause:** Qwen3-VL multimodal tasks rely on the model chat template to insert image placeholder tokens before multimodal processing.

**Solution:** Enable chat template application in the evaluation configuration. For lm_eval-based multimodal tasks, set `apply_chat_template` to true.
