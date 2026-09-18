# Step-3.7-Flash

## 1 Introduction

Step 3.7 Flash is a 198B-parameter sparse Mixture-of-Experts (MoE) vision-language model that combines a 196B-parameter language backbone with a 1.8B-parameter vision encoder for native image understanding.

This document covers supported features, environment and model preparation, single-node deployment, thinking and parser configuration, functional verification, accuracy evaluation, and troubleshooting.

This document is written based on the vLLM-Ascend v0.28.0 release. This model is supported in this release.

## 2 Supported Features

Refer to [Supported Features List](../../user_guide/support_matrix/supported_models.md) for the model support matrix.

Refer to the [Feature Guide](../../user_guide/feature_guide/index.md) for feature configuration instructions.

## 3 Prerequisites

### 3.1 Model Weight

- `Step-3.7-Flash` (BF16): requires 8 × 96 GB 950DT NPU chips. [Download the model weights](https://www.modelscope.cn/models/stepfun-ai/Step-3.7-Flash).
- `Step-3.7-Flash-MXFP8` (MXFP8): used for 950DT products (96GB × 8). [Download the model weights](https://huggingface.co/MiniMaxAI/MiniMax-M3-MXFP8).

It is recommended to place the model weight in a shared cache directory.

## 4 Installation

### 4.1 Docker Image Installation

You can use the official all-in-one Docker image. For the available image tags and published versions, refer to [Using Docker](../../getting_started/installation.md#installation-prebuilt-image).

- Step 1: Download the latest Docker image

  ```bash
  docker pull quay.io/ascend/vllm-ascend:{tag}
  ```

- Step 2: Start Docker container

  ```bash
  # Set the vLLM Ascend image name.
  export IMAGE=quay.io/ascend/vllm-ascend:{tag}
  export NAME=minimax-m3-dev

  # Start the container with the variables defined above.
  # Update --device for your hardware (Atlas A3: /dev/davinci[0-15]; Atlas A2: /dev/davinci[0-7]).
  # If you use a Docker bridge network, open the ports required for multi-node communication in advance.
  docker run --rm \
  --name $NAME \
  --net=host \
  --shm-size=100g \
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

  Expected result: The container is listed with status `Up`. You can also verify the vllm-ascend version inside the container:

  ```bash
  pip show vllm-ascend
  ```

  Expected result: The version information is displayed, matching the pulled image version.

## 5 Online Service Deployment {: #5-online-service-deployment }

Start the online serving service with the following command:

For descriptions of the standard `vllm serve` arguments used in the deployment examples, refer to the [vLLM Serving Arguments documentation](https://docs.vllm.ai/en/latest/cli/serve/#arguments). For Ascend-specific options passed through `--additional-config`, refer to [Additional Configuration](../../user_guide/configuration/additional_config.md). For Ascend-specific environment variables, refer to [Environment Variables](../../user_guide/configuration/env_vars.md).

### 5.1 Single-Node Deployment

Single-node deployment completes both Prefill and Decode within the same node. Both the bfloat(Step-3.7-Flash) and quantized(MXFP8) model can be deployed on 1 950DT products (96GB × 8).

=== "950DT products"

    ```bash
    export HCCL_BUFFSIZE=1024
    export HCCL_OP_EXPANSION_MODE="CCU_SCHED"
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export TRITON_ALL_BLOCKS_PARALLEL=1

    vllm serve ${WEIGHT_PATH} \
      --host 0.0.0.0 \
      --port 11223 \
      --served-model-name step \
      --trust-remote-code \
      --tensor-parallel-size 8 \
      --enable-expert-parallel \
      --max-num-batched-tokens 16384 \
      --max-num-seqs 64 \
      --enable-prefix-caching \
      --limit-mm-per-prompt '{"image":1}' \
      --gpu-memory-utilization 0.92 \
      --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
      --additional-config '{"multistream_overlap_shared_expert":true, "enable_shared_expert_dp":true}' \
      --speculative-config '{"method": "mtp", "num_speculative_tokens": 3} \
      --safetensors-load-strategy prefetch \
      --reasoning-parser step3p5 \
      --enable-auto-tool-choice \
      --tool-call-parser step3p5 \
    ```

**Note**: In the script above, `max-num-seqs` represents the maximum number of sequences the scheduler can process in a single iteration. Adjust the `max-num-seqs` parameter dynamically based on actual business.

For text-only deployment, `--limit-mm-per-prompt` can be omitted. For multimodal deployment, configure this parameter according to the actual request shape. For example, use `--limit-mm-per-prompt '{"image":2}'` for two-image requests.

### 5.4 Multimodal and ViT DP (Optional)

Step-3.7-Flash supports image inputs on Ascend. The deployment examples above keep `--limit-mm-per-prompt '{"image":1}'` as the default multimodal capacity assumption because the other serving parameters are tuned for the single-image path.

For the ViT / multimodal encoder part, data parallel execution is supported and can be enabled with:

```bash
--mm-encoder-tp-mode data
```

This option is not enabled in the default deployment examples because it can increase per-card memory usage. When enabling ViT DP, re-evaluate memory-related parameters such as `--max-model-len`, `--max-num-seqs`, and `--gpu-memory-utilization` for the target workload.

When using local media paths in requests, such as `file:///path/to/image.png`, add an explicit allowlist path:

```bash
--allowed-local-media-path /
```

## 6 Parser Configuration

### 6.1 Reasoning Parser

The Step-3.7-Flash reasoning parser (`--reasoning-parser step3p5`) extracts the thinking block `<mm:think>...</mm:think>` from model output and exposes it as the `reasoning` field. The remaining text is returned as `content`.

### 6.2 Tool Call Parser

Step-3.7-Flash uses a namespace-delimited XML format for tool calls. Enable it with `--tool-parser step3p5`.

## 7 Functional Verification

### 7.1 Text

  ```bash
  curl http://{ip}:{port}/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d @- <<EOF
  {
    "model": "step",
    "messages": [
      {
        "role": "user",
        "content": "Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.\n\nA student regrets that he fell asleep during a lecture in electrochemistry, facing the following incomplete statement in a test:\nThermodynamically, oxygen is a …oxidant in basic solutions. Kinetically, oxygen reacts …in acidic solutions.\nWhich combination of weaker/stronger and faster/slower is correct?\n\nA) weaker —faster\nB) stronger —faster\nC) weaker - slower\nD) stronger —slower"
      }
    ],
    "max_tokens": 8000,
    "temperature": 1.0
  }
  EOF
  ```

  Expected result: the answer is C.

### 7.2 Single Image

  Start the service with image input enabled, for example `--limit-mm-per-prompt '{"image":1,"video":0}'`. Replace `${IMAGE_PATH}` with a local image path on the client side.

  ```bash
  IMAGE_PATH=/path/to/image.jpg
  IMAGE_BASE64="$(base64 -w 0 "${IMAGE_PATH}")"

  curl http://{ip}:{port}/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d @- <<EOF
  {
    "model": "step",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,${IMAGE_BASE64}"}},
          {"type": "text", "text": "Briefly describe this image."}
        ]
      }
    ],
    "max_tokens": 512,
    "temperature": 0
  }
  EOF
  ```

  Expected result: HTTP 200 response with a JSON body containing non-empty `choices` and generated text describing the image.

## 8 Accuracy Evaluation

### 8.1 Using AISBench

For detailed instructions, refer to [Using AISBench for accuracy evaluation](../../developer_guide/evaluation/using_ais_bench.md).

### 8.2 Text Evaluation

| Dataset | Hardware | Score | max-model-len | max-num-seqs | max_out_len | batch_size | generation_kwargs |
|---------|----------|-------|---------------|--------------|-------------|------------|-------------------|
| AIME2026 | 8 950DT A5 (96GB × 8)      | 93.3    | 262144        | 32         | 250000           | 16         | temperature=0.6, top_p=0.95 |

## 9 Performance Tuning

> **Note**: The following configurations are validated in specific test environments and are for reference only. The optimal configuration depends on factors such as maximum input/output length, prefix cache hit rate, precision requirements, and deployment machine ratios. It is recommended to refer to Section 9.2 for tuning based on actual conditions.

### 9.1 Recommended Configurations

The recommended configurations are the same as those specified in Chapter 5, "Online Service Deployment."

### 9.2 Tuning Guidelines

#### 9.2.1 General Tuning Reference

Please refer to the [Public Performance Tuning Documentation](../../developer_guide/performance_and_debug/optimization_and_tuning.md) for general tuning methods.

Please refer to the [Feature Matrix](../../user_guide/support_matrix/feature_matrix.md) for detailed feature descriptions.

