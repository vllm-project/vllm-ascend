# Multi-NPU (Qwen2.5-VL-32B-Instruct)

## Introduction

Key Enhancements:
- Understand things visually: Qwen2.5-VL is not only proficient in recognizing common objects such as flowers, birds, fish, and insects, but it is highly capable of analyzing texts, charts, icons, graphics, and layouts within images.

- Being agentic: Qwen2.5-VL directly plays as a visual agent that can reason and dynamically direct tools, which is capable of computer use and phone use.

- Understanding long videos and capturing events: Qwen2.5-VL can comprehend videos of over 1 hour, and this time it has a new ability of cpaturing event by pinpointing the relevant video segments.

- Capable of visual localization in different formats: Qwen2.5-VL can accurately localize objects in an image by generating bounding boxes or points, and it can provide stable JSON outputs for coordinates and attributes.

- Generating structured outputs: for data like scans of invoices, forms, tables, etc. Qwen2.5-VL supports structured outputs of their contents, benefiting usages in finance, commerce, etc.

This document will demonstrate the main validation steps of the model, including supported features, feature configuration, environment preparation, single-node deployment, as well as accuracy and performance evaluation.

## Supported Features

Refer to [supported features](../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [feature guide](../user_guide/feature_guide/index.md) to get the feature's configuration.

## Environment Preparation

### Model Weight

- A sample Qwen2.5-VL quantization script can be found in the modelslim code repository. [Qwen2.5-VL Quantization Script Example](https://gitcode.com/Ascend/msit/blob/master/msmodelslim/example/multimodal_vlm/Qwen2.5-VL/README.md)

- `Qwen2.5-VL-32B-Instruct-w8a8`(Quantized version): require 1 Atlas 800 A2 (64G × 8) node. 

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`

### Verify Multi-node Communication(Optional)

If you want to deploy multi-node environment, you need to verify multi-node communication according to [verify multi-node communication environment](../installation.md#verify-multi-node-communication).


## Deployment
### Run docker container
```shell
export IMAGE=quay.io/ascend/vllm-ascend:0.11.0rc1
docker run --rm \
--shm-size=1g \
--net=host \
--name vllm-ascend-qwen25_VL \
--device /dev/davinci0 \
--device /dev/davinci1 \
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--device /dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /root/.cache:/root/.cache \
-v /data:/data \
-it $IMAGE bash
```

### Single-node Deployment

Run the following script to execute online inference. Recommend two NPU cards for deploying the Qwen2.5-VL-32B-Instruct-w8a8 model.

```shell
#!/bin/sh
# apt install libjemalloc2 or yum install jemalloc
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_USE_V1=1

# Comment out a line in the code. "MRotaryEmbedding": AscendMRotaryEmbedding
vim /vllm-workspace/vllm-ascend/vllm_ascend/utils.py +538

vllm serve /data/Qwen2.5-VL-32B-Instruct-w8a8 \
    --host 0.0.0.0 \
    --port 8888 \
    --served-model-name qwen25_vl \
    --max-num-batched-tokens 2048 \
    --quantization ascend \
    --async-scheduling \
    --tensor-parallel-size 2 \
    --max_model_len 30000 \
    --max-num-batched-tokens 50000 \
    --max-num-seqs 30 \
    --no-enable-prefix-caching \
    --trust-remote-code \
    --additional-config '{"enable_weight_nz_layout":true}'

```


### Prefill-Decode Disaggregation

Not supported yet.

## Functional Verification

Once your server is started, you can query the model with input prompts:

```shell
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
    "model": "qwen25_vl",
    "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://modelscope.oss-cn-beijing.aliyuncs.com/resource/qwen.png"}},
        {"type": "text", "text": "What is the text in the illustrate?"}
    ]}
    ]
    }'
```

## Accuracy Evaluation

Here are two accuracy evaluation methods.

### Using AISBench

1. Refer to [Using AISBench](../developer_guide/evaluation/using_ais_bench.md) for details.


## Performance

### Using AISBench

Refer to [Using AISBench for performance evaluation](../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

### Using vLLM Benchmark

Run performance evaluation of `Qwen2.5-VL-32B-Instruct-w8a8` as an example.

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/contributing/benchmarks.html) for more details.

There are three `vllm bench` subcommand:
- `latency`: Benchmark the latency of a single batch of requests.
- `serve`: Benchmark the online serving throughput.
- `throughput`: Benchmark offline inference throughput.

Take the `serve` as an example. Run the code as follows.

```shell
export VLLM_USE_MODELSCOPE=true
vllm bench serve --model /data/Qwen2.5-VL-32B-Instruct-w8a8  --dataset-name random --random-input 200 --num-prompt 200 --request-rate 1 --save-result --result-dir ./
```

After about several minutes, you can get the performance evaluation result.
