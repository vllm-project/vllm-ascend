# GLM-4.6V

## 1 Introduction

GLM-4.6V uses a Mixture-of-Experts (MoE) architecture with 106B total parameters and 12B activated parameters. It supports image, video, and document understanding.

This document will show the main verification steps of the model, including supported features, feature configuration, environment preparation, single-node deployment, and functional verification.

The deployment examples use the released vLLM Ascend image selected by the documentation's `vllm_ascend_version` setting.

## 2 Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [feature guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## 3 Prerequisites

### 3.1 Model Weight

The deployment example requires an A3 node with eight available NPU devices and sufficient device memory for the model weights and KV cache. For software requirements, refer to [installation](../../getting_started/installation.md).

- `GLM-4.6V` (BF16 version): [Download model weight](https://www.modelscope.cn/models/ZhipuAI/GLM-4.6V).

It is recommended to download the model weight to a shared directory, such as `/root/.cache/`.

```bash
git lfs install
git clone https://www.modelscope.cn/ZhipuAI/GLM-4.6V.git /root/.cache/GLM-4.6V
git -C /root/.cache/GLM-4.6V lfs pull
git -C /root/.cache/GLM-4.6V rev-parse HEAD
```

## 4 Installation

### 4.1 Docker Image Installation

You can use our official docker image to run `GLM-4.6V` directly.

=== "A3 series"

    Start the docker image on your node.

    ```bash
    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-a3
    docker pull "$IMAGE"
    docker run --rm \
        --name vllm-ascend \
        --shm-size=8g \
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
        -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 \
        -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
        -v /etc/ascend_install.info:/etc/ascend_install.info \
        -v /root/.cache:/root/.cache \
        -it "$IMAGE" bash
    ```

The model weight directory must be accessible inside the container. The example mounts `/root/.cache` at the same path and uses 8 GiB of shared memory for TP workers and the multimodal processor cache.

In another terminal on the host, check that the container is running:

```bash
docker ps --filter name=vllm-ascend
```

The output should list the `vllm-ascend` container with an `Up` status.

### 4.2 Source Code Installation

- Install `vllm-ascend` from source, refer to [installation](../../getting_started/installation.md).

## 5 Online Service Deployment

### 5.1 Single-Node Online Deployment

The following example uses eight NPU devices on an A3 node with TP=8 and expert parallelism.

Run the following script to execute online inference.

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ASCEND_GLOBAL_WORK_SIZE=32
export ASCEND_LAUNCH_BLOCKING=0

vllm serve /root/.cache/GLM-4.6V \
    --tensor-parallel-size 8 \
    --mm-encoder-tp-mode data \
    --mm-processor-cache-type shm \
    --enable-expert-parallel \
    --max-model-len 65536 \
    --max-num-seqs 256 \
    --max-num-batched-tokens 8192 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.95 \
    --enable-auto-tool-choice \
    --tool-call-parser glm45 \
    --reasoning-parser glm45 \
    --served-model-name glm \
    --host 0.0.0.0 \
    --port 8000
```

**Notice:**
The parameters are explained as follows:

- `--mm-encoder-tp-mode data`: Use data parallelism for the vision encoder.
- `--mm-processor-cache-type shm`: Use shared memory for the multimodal processor cache.
- `--tool-call-parser glm45` / `--reasoning-parser glm45` / `--enable-auto-tool-choice`: Enable tool-call and reasoning parsing.
- `--max-model-len` and `--max-num-seqs` need to be set according to the actual usage scenario.

After model loading completes, check the service status and model list from another terminal:

```bash
curl --fail-with-body http://127.0.0.1:8000/health
curl --fail-with-body http://127.0.0.1:8000/v1/models
```

The health check should return HTTP 200, and the model list should contain `glm`. For startup and memory issues, refer to the [Public FAQs](../../faqs.md).

## 6 Functional Verification

Once your server is started, you can query the model with input prompts.

### 6.1 Text Input

Send a text request:

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"glm","messages":[{"role":"user","content":"Hello, please introduce yourself."}],"max_tokens":256,"temperature":0.0}'
```

The request should return HTTP 200 and a JSON response containing `choices` and the generated message.

### 6.2 Image Input

Set `IMAGE_PATH` to a local JPEG file and send an image request:

```bash
IMAGE_PATH=/root/.cache/example.jpg
IMAGE_BASE64=$(base64 -w 0 "$IMAGE_PATH")

curl --fail-with-body http://127.0.0.1:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    --data-binary @- <<EOF
{
    "model": "glm",
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe the main content of this image."},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,${IMAGE_BASE64}"}}
        ]
    }],
    "max_tokens": 1024,
    "temperature": 0.0
}
EOF
```

The request should return HTTP 200 and a JSON response containing `choices`. Check that the generated description matches the image content.

## 7 Accuracy Evaluation

### 7.1 Evaluation Results

The following GLM-4.6V accuracy results were provided by the contributor:

| Dataset | Score |
| --- | --- |
| MMBench | 88.62 |
| MMStar | 76.13 |

### 7.2 Using AISBench

For evaluating MMStar with AISBench, refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) and the [MMStar dataset configuration](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/mmstar).

## 8 Performance Evaluation

### Using AISBench

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

## 9 Performance Tuning

Refer to [Performance Tuning](../../developer_guide/performance_and_debug/optimization_and_tuning.md) for general tuning methods. Adjust `--max-model-len` and `--max-num-seqs` for your workload and available device memory.

## 10 FAQ

For common environment, installation, and general parameter issues, refer to the [Public FAQs](../../faqs.md).
