# DeepSeek-OCR-2

## Introduction

DeepSeekOCR2 is a model to investigate the role of vision encoders from an LLM-centric viewpoint.

The `DeepSeek-OCR-2` model is first supported in `vllm-ascend:v0.16.0` and can stably run in v0.16.0 and later version.

This document will show the main verification steps of the model, including supported features, feature configuration, environment preparation, single-node deployment, accuracy and performance evaluation.

## Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [feature guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## Environment Preparation

### Model Weight

- `DeepSeek-OCR-2`: [Download model weight](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2).

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`.

### Verify Multi-node Communication(Optional)

If you want to deploy multi-node environment, you need to verify multi-node communication according to [verify multi-node communication environment](../../installation.md#verify-multi-node-communication).

### Installation

You can use our official docker image to run `DeepSeek-OCR-2` directly.

Select an image based on your machine type and start the docker image on your node, refer to [using docker](../../installation.md#set-up-using-docker).

```{code-block} bash
   :substitutions:
# Update --device according to your device (Atlas A2: /dev/davinci[0-7] Atlas A3:/dev/davinci[0-15]).
# Update the vllm-ascend image according to your environment.
# Note you should download the weight to /root/.cache in advance.
# Update the vllm-ascend image
export IMAGE=m.daocloud.io/quay.io/ascend/vllm-ascend:|vllm_ascend_version|
export NAME=vllm-ascend

# Run the container using the defined variables
# Note: If you are running bridge network with docker, please expose available ports for multiple nodes communication in advance.
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

If you want to deploy multi-node environment, you need to set up environment on each node.

## Deployment

### Single-node Deployment

- `DeepSeek-OCR-2` can be deployed on 1 Atlas 800 A2.

Run the following script to execute online inference.

```shell
#!/bin/sh

export VLLM_USE_V1=1
export VLLM_ASCEND_ENABLE_NZ=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export TASK_QUEUE_ENABLE=1
export TOKENIZERS_PARALLELISM=false

vllm serve /root/.cache/DeepSeek-OCR-2 \
    --served-model-name deepseekocr2 \
    --trust-remote-code \
    --tensor-parallel-size 1  \
    --port 1055 \
    --max_model_len 8192 \
    --no-enable-prefix-caching \
    --gpu-memory-utilization 0.8 \
    --allowed-local-media-path / \
    --async-scheduling \
    --additional-config '{
      "enable_cpu_binding": true,
      "multistream_overlap_shared_expert": true,
      "ascend_compilation_config": {"fuse_qknorm_rope": false}
    }' \
    --mm-processor-cache-gb 0
```

**Notice:**
The parameters are explained as follows:

- `--max-model-len` specifies the maximum context length - that is, the sum of input and output tokens for a single request.
- `--no-enable-prefix-caching` indicates that prefix caching is disabled. To enable it, remove this option.
- `--gpu-memory-utilization` represents the proportion of HBM that vLLM will use for actual inference. Its essential function is to calculate the available kv_cache size. During the warm-up phase (referred to as profile run in vLLM), vLLM records the peak GPU memory usage during an inference process with an input size of `--max-num-batched-tokens`. The available kv_cache size is then calculated as: `--gpu-memory-utilization` * HBM size - peak GPU memory usage. Therefore, the larger the value of `--gpu-memory-utilization`, the more kv_cache can be used. However, since the GPU memory usage during the warm-up phase may differ from that during actual inference (e.g., due to uneven EP load), setting `--gpu-memory-utilization` too high may lead to OOM (Out of Memory) issues during actual inference. The default value is `0.9`.
- `--async-scheduling` Asynchronous scheduling is a technique used to optimize inference efficiency. It allows non-blocking task scheduling to improve concurrency and throughput, especially when processing large-scale models.

### Multi-node Deployment

Single-node deployment is recommended.

### Prefill-Decode Disaggregation

We don't need to Prefill-Decode disaggregation

## Functional Verification

If your service start successfully, you can see the info shown below:

```bash
INFO:     Started server process [87471]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

Once your server is started, you can query the model with input prompts:

```shell
curl http://<node0_ip>:<port>/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "deepseekocr2",
        "prompt": "The future of AI is",
        "max_completion_tokens": 50,
        "temperature": 0
    }'
```

## Accuracy Evaluation

Here are two accuracy evaluation methods.

### Using AISBench

1. Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) for details.

2. After execution, you can get the result, here is the result of `DeepSeek-OCR-2` for reference only.

| dataset | version | metric | mode | vllm-api-general-chat | note |
|----- | ----- | ----- | ----- | -----| ----- |
| textvqa | - | accuracy | gen | 50.28 | 1 Atlas 800 A2 |
| ominidocbench | - | accuracy | gen | 66.86 | 1 Atlas 800 A2 |

## Performance

### Using AISBench

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

The performance result is:  

**Hardware**: A2-313T, 1 node

**Input/Output**: 1080P/256

**Performance**: TTFT = 2s, TPOT = 200ms, Average performance of each card is 864 TPS (Token Per Second).

## Best Practices

In this chapter, we recommend best practices. for details about best practices, see the "Single-node Deployment" section.

## FAQ

- **Q: Startup fails with HCCL port conflicts (address already bound). What should I do?**

  A: Clean up old processes and restart: `pkill -f vLLM*`.

- **Q: How to handle OOM or unstable startup?**

  A: Reduce `--max-num-seqs` and `--max-model-len` first. If needed, reduce concurrency and load-testing pressure (e.g., `max-concurrency` / `num-prompts`).
