# Kimi-Linear-48B-A3B-Instruct

## Introduction

Kimi-Linear-48B-A3B-Instruct is a hybrid-architecture language model developed by Moonshot AI. It combines Multi-head Latent Attention (MLA) with Kimi Delta Attention (KDA) linear attention mechanism, achieving efficient long-context inference with reduced memory requirements.

Key features of Kimi-Linear-48B-A3B-Instruct include:

- ~48B total parameters with ~3B activated (Mixture-of-Experts)
- Hybrid architecture: 7 MLA layers + 20 KDA linear attention layers
- Support for up to 1M context length
- KDA linear attention for efficient sequence processing

This document will show the main verification steps of the model, including supported features, feature configuration, environment preparation, single-node deployment, accuracy and performance evaluation.

The `Kimi-Linear-48B-A3B-Instruct` model is experimentally supported in vllm-ascend.

## Environment Preparation

### Model Weight

- `Kimi-Linear-48B-A3B-Instruct`(BF16 version): [Download model weight](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct)

It is recommended to download the model weight to a local directory, such as `/data/models/`.

### Installation

You can use our official docker image to run `Kimi-Linear-48B-A3B-Instruct` directly.

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

- `Kimi-Linear-48B-A3B-Instruct` can be deployed on 1 Atlas 800 A2 (64G x 2) or above.

Run the following script to execute online inference.

``` bash

export HCCL_OP_EXPANSION_MODE="AIV"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export VLLM_USE_V1=1
export HCCL_BUFFSIZE=200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
```

``` bash

vllm serve "moonshotai/Kimi-Linear-48B-A3B-Instruct" \
  --tensor-parallel-size 2 \
  --max-model-len 4096 \
  --dtype bfloat16 \
  --trust-remote-code \
  --enforce-eager \
  --block-size 128 \
  --gpu-memory-utilization 0.7
```

**Notice:**
The parameters are explained as follows:

- `--max-model-len` specifies the maximum context length - that is, the sum of input and output tokens for a single request. For testing purposes, a value of `4096` is used here.
- `--dtype bfloat16` specifies the data type for model weights and computations.
- `--trust-remote-code` allows loading models with custom code.
- `--enforce-eager` forces the use of eager execution mode instead of graph compilation, which is more stable for experimental models.
- `--block-size` specifies the block size for KV cache management, with a value of `128` used here. The actual block size may be adjusted at runtime to align MLA and KDA layer page sizes.
- `--gpu-memory-utilization` sets the proportion of NPU memory to use for the model, with a value of `0.7` used here to reduce memory usage.

## Functional Verification

Once your server is started, you can query the model with input prompts.

```shell
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "moonshotai/Kimi-Linear-48B-A3B-Instruct",
        "messages": [
            {"role": "user", "content": "你好，介绍一下你自己"}
        ],
        "max_tokens": 100,
        "temperature": 0.7
    }'
```

### Using AISBench

1. Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) for details.

2. After execution, you can get the result.

## Performance

### Using AISBench

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

### Using vLLM Benchmark

Run performance evaluation of `Kimi-Linear-48B-A3B-Instruct` as an example.

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/benchmarking/) for more details.

There are three `vllm bench` subcommands:

- `latency`: Benchmark the latency of a single batch of requests.
- `serve`: Benchmark the online serving throughput.
- `throughput`: Benchmark offline inference throughput.

Take the `serve` as an example. First, start the server:

```shell
python -m vllm.entrypoints.openai.api_server \
    --model moonshotai/Kimi-Linear-48B-A3B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 2 \
    --max-model-len 512 \
    --dtype bfloat16 \
    --trust-remote-code \
    --enforce-eager \
    --block-size 128 \
    --gpu-memory-utilization 0.7
```

## Known Limitations

- This model is experimentally supported. The KDA (Kimi Delta Attention) linear attention layers use a PyTorch fallback for the chunk_kda operation, which may impact prefill performance.
- The mixed KV cache (MLA + KDA) page size alignment is handled automatically at runtime.

## Conclusion

Kimi-Linear-48B-A3B-Instruct is an innovative hybrid-attention MoE model that combines the strengths of standard MLA attention and KDA linear attention. With proper deployment on Ascend hardware using vllm-ascend, you can leverage its efficient long-context capabilities.

For more details about model capabilities and best practices, refer to the official Moonshot AI documentation and vllm-ascend user guide.
