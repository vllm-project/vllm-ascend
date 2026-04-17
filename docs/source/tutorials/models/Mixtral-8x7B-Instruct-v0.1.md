# Mixtral-8x7B-Instruct-v0.1

## Introduction

Mixtral-8x7B-Instruct-v0.1 is a state-of-the-art mixture-of-experts (MoE) language model developed by Mistral AI. It features 8 expert models, each with 7B parameters, and is specifically fine-tuned for instruction following tasks.

Key features of Mixtral-8x7B-Instruct-v0.1 include:

- 8x7B parameters with sparse activation (only 2 experts activated per token)
- Strong performance across various NLP tasks
- Support for extended context length
- High-quality instruction following capabilities

This document will show the main verification steps of the model, including supported features, feature configuration, environment preparation, single-node deployment, accuracy and performance evaluation.

The `Mixtral-8x7B-Instruct-v0.1` model is supported in vllm-ascend.

## Environment Preparation

### Model Weight

- `Mixtral-8x7B-Instruct-v0.1`(BF16 version): [Download model weight](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)
- Quantized versions may be available from third-party providers.

It is recommended to download the model weight to a local directory, such as `/data/models/`.

### Installation

You can use our official docker image to run `Mixtral-8x7B-Instruct-v0.1` directly.

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

## Deployment

### Single-node Deployment

- `Mixtral-8x7B-Instruct-v0.1` can be deployed on 1 Atlas 800 A3 (64G × 16) or 1 Atlas 800 A2 (64G × 8).

Run the following script to execute online inference.

```shell
export VLLM_USE_MODELSCOPE=true
export HCCL_OP_EXPANSION_MODE="AIV"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export VLLM_USE_V1=1
export HCCL_BUFFSIZE=200
export VLLM_ASCEND_BALANCE_SCHEDULING=1
export VLLM_ASCEND_ENABLE_MLAPO=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

vllm serve /root/.cache/modelscope/hub/models/mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --host 0.0.0.0 \
    --port 8000 \
    --data-parallel-size 1 \
    --tensor-parallel-size 4 \
    --seed 1024 \
    --served-model-name mixtral_8x7b_instruct_v0_1 \
    --enable-expert-parallel \
    --max-num-seqs 16 \
    --max-model-len 4096 \
    --max-num-batched-tokens 4096 \
    --trust-remote-code \
    --gpu-memory-utilization 0.92 \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
```

**Notice:**
The parameters are explained as follows:

- Setting the environment variable `VLLM_ASCEND_BALANCE_SCHEDULING=1` enables balance scheduling. This may help increase output throughput and reduce TPOT in v1 scheduler. However, TTFT may degrade in some scenarios.
- `--served-model-name mixtral_8x7b_instruct_v0_1` sets the model name exposed by the service API.
- `--enable-expert-parallel` enables expert parallelism for the MoE model.
- `--max-num-seqs` specifies the maximum number of sequences processed in one scheduling step. A value of `16` is used here as a balanced starting point.
- `--max-model-len` specifies the maximum context length - that is, the sum of input and output tokens for a single request. For testing purposes, a value of `4096` is used here.
- `--max-num-batched-tokens` sets the upper bound of batched tokens in one iteration. A value of `4096` is used here.
- `--trust-remote-code` allows loading models with custom code.
- `--gpu-memory-utilization` sets the proportion of NPU memory to use for the model, with a value of `0.92` used here.
- `--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'` enables the recommended graph compilation mode for decoding.

## Functional Verification

Once your server is started, you can query the model with input prompts. Mixtral-8x7B-Instruct-v0.1 uses a specific prompt format with [INST] and [/INST] tags:

```shell
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "mixtral_8x7b_instruct_v0_1",
        "messages": [
            {"role": "user", "content": "你好，介绍一下你自己"}
        ],
        "max_tokens": 100,
        "temperature": 0.7
    }'
```

For instruction following tasks, you can use prompts like:

```shell
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "mixtral_8x7b_instruct_v0_1",
        "messages": [
            {"role": "user", "content": "扮演一位资深架构师，评价一下在昇腾 Atlas A2 上部署 vLLM 的优势。"}
        ],
        "max_tokens": 100,
        "temperature": 0.7
    }'
```

For MoE-related questions:

```shell
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "mixtral_8x7b_instruct_v0_1",
        "messages": [
            {"role": "user", "content": "简单解释一下为什么 Mixtral 模型被称为\"混合专家模型\"(MoE)？"}
        ],
        "max_tokens": 100,
        "temperature": 0.7
    }'
```

### Using AISBench

1. Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) for details.

2. After execution, you can get the result. For reference, Mixtral-8x7B-Instruct-v0.1 typically performs well on various benchmarks including reasoning, comprehension, and instruction following tasks.

## Performance

### Using AISBench

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

### Using vLLM Benchmark

Run performance evaluation of `Mixtral-8x7B-Instruct-v0.1` as an example.

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/benchmarking/) for more details.

There are three `vllm bench` subcommands:

- `latency`: Benchmark the latency of a single batch of requests.
- `serve`: Benchmark the online serving throughput.
- `throughput`: Benchmark offline inference throughput.

Take the `serve` as an example. First, start the server:

```shell
vllm serve /root/.cache/modelscope/hub/models/mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --host 0.0.0.0 \
    --port 8000 \
    --data-parallel-size 1 \
    --tensor-parallel-size 4 \
    --seed 1024 \
    --served-model-name mixtral_8x7b_instruct_v0_1 \
    --enable-expert-parallel \
    --max-num-seqs 16 \
    --max-model-len 512 \
    --max-num-batched-tokens 4096 \
    --trust-remote-code \
    --gpu-memory-utilization 0.92 \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
```

## Conclusion

Mixtral-8x7B-Instruct-v0.1 is a powerful MoE model that offers excellent performance for instruction following tasks. With proper deployment on Ascend hardware using vllm-ascend, you can achieve high throughput and low latency for your AI applications.

For more details about model capabilities and best practices, refer to the official Mixtral documentation and vllm-ascend user guide.
