# GLM-4.6V

## 1 Introduction

GLM-4.6V uses a Mixture-of-Experts (MoE) architecture with 106B total parameters and 12B activated parameters. It supports image, video, and document understanding.

This document will show the main verification steps of the model, including supported features, feature configuration, environment preparation, single-node deployment, and functional verification.

This document is based on vLLM Ascend v0.23.0. GLM-4.6V (BF16) has experimental support on A3 in this version.

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

You can use our official Docker image to run `GLM-4.6V` directly.

=== "A3 series"

    Start the Docker image on your node.

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

The following example uses eight NPU devices on an A3 node with TP=8 and expert parallelism. Tune the limits for your workload and available device memory.

Run the following command to start the online service:

```bash
vllm serve /root/.cache/GLM-4.6V \
    --tensor-parallel-size 8 \
    --mm-encoder-tp-mode data \
    --mm-processor-cache-type shm \
    --enable-expert-parallel \
    --max-num-seqs 32 \
    --max-num-batched-tokens 32768 \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --gpu-memory-utilization 0.90 \
    --enable-auto-tool-choice \
    --tool-call-parser glm45 \
    --reasoning-parser glm45 \
    --served-model-name glm
```

**Notice:**

- `--mm-encoder-tp-mode data`: Required for this TP=8 deployment. Use data parallelism for the vision encoder; omitting this option causes engine initialization to fail.
- `--mm-processor-cache-type shm`: Use shared memory for the multimodal processor cache.
- `--enable-auto-tool-choice`, `--tool-call-parser glm45`, and `--reasoning-parser glm45`: Enable automatic tool choice, tool-call parsing, and reasoning parsing.

After model loading completes, check the service status and model list from another terminal:

```bash
curl --fail-with-body http://127.0.0.1:8000/health
curl --fail-with-body http://127.0.0.1:8000/v1/models
```

The health check should return HTTP 200, and the model list should contain `glm`. For startup and memory issues, refer to the [Public FAQs](../../faqs.md).

## 6 Functional Verification

Once your server is started, you can query the model with input prompts.

### 6.1 Text Request

Send a text request:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"glm","messages":[{"role":"user","content":"Hello, please introduce yourself."}],"max_tokens":256,"temperature":0.0}'
```

Expected result: the HTTP status is 200 and the JSON response contains a `choices` field with the generated reply in `.choices[0].message.content`.

### 6.2 Image Request

Set `IMAGE_PATH` to a local JPEG file and send an image request:

```bash
IMAGE_PATH=/root/.cache/example.jpg
IMAGE_BASE64=$(base64 -w 0 "$IMAGE_PATH")

curl http://127.0.0.1:8000/v1/chat/completions \
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

Expected result: the HTTP status is 200 and the JSON response contains a `choices` field whose `.choices[0].message.content` describes the image.

## 7 Accuracy Evaluation

### 7.1 Using VLMEvalKit

1. Refer to the [VLMEvalKit documentation](https://github.com/open-compass/VLMEvalKit) for installation and evaluation instructions.

2. After execution, you can get the result.

| dataset | version | metric | mode | result |
| ------- | ------- | ------ | ---- | ------ |
| MMBench | - | accuracy | gen | 88.62 |
| MMStar | - | accuracy | gen | 76.13 |

## 8 Performance Evaluation

### 8.1 Using AISBench

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

## 9 FAQ

For common environment, installation, and general parameter issues, refer to the [Public FAQs](../../faqs.md).
