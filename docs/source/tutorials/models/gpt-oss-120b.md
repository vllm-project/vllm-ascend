# gpt-oss-120b

## Introduction

gpt-oss-120b and gpt-oss-20b are two open-weight reasoning models that push the frontier of accuracy and inference cost. The models use an efficient mixture-of-expert transformer architecture and are trained using large-scale distillation and reinforcement learning. We optimize the models to have strong agentic capabilities (deep research browsing, python tool use, and support for developer-provided functions), all while using a rendered chat format that enables clear instruction following and role delineation. Both models achieve strong results on benchmarks ranging from mathematics, coding, and safety. We release the model weights, inference implementations, tool environments, and tokenizers under an Apache 2.0 license to enable broad use and further research

## Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [feature guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## Environment Preparation

### Model Weight

- `gpt-oss-120b`(bf16 version): require 1 Atlas 800 A3 (64G × 16) nodes or 1 Atlas 800 A2 (64G × 8) nodes. [Download model weight](https://huggingface.co/unsloth/gpt-oss-120b-BF16)

### Installation

You can use our official docker image for supporting gpt-oss-120b-bf16 models.
Currently, we provide the all-in-one images.[Download images](https://quay.io/repository/ascend/vllm-ascend?tab=tags)

#### Docker Pull (by tag)

```{code-block} bash
   :substitutions:

docker pull quay.io/ascend/vllm-ascend:|vllm_ascend_version|

```

#### Docker run

```{code-block} bash
   :substitutions:

# Update --device according to your device (Atlas A2: /dev/davinci[0-7] Atlas A3:/dev/davinci[0-15]).
# Update the vllm-ascend image according to your environment.
# Note you should download the weight to /root/.cache in advance.
# For Atlas A2 machines:
# export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
# For Atlas A3 machines:
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|-a3
docker run --rm \
    --name vllm-ascend-env \
    --shm-size=1g \
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
    -v /root/.cache:/root/.cache \
    -it $IMAGE bash
```

The default workdir is `/workspace`, vLLM and vLLM Ascend code are placed in `/vllm-workspace` and installed in [development mode](https://setuptools.pypa.io/en/latest/userguide/development_mode.html) (`pip install -e`) to help developer immediately take place changes without requiring a new installation.

In addition, if you don't want to use the docker image as above, you can also build all from source:

- Install `vllm-ascend` from source, refer to [installation](../../installation.md).

## Deployment

### Troubleshooting

Run into

"openai_harmony.HarmonyError: error downloading or loading vocab file: failed to download or load vocab error"

Solution: This is caused by a bug in openai_harmony code. This can be worked around by downloading the tiktoken encoding files in advance and setting the TIKTOKEN_ENCODINGS_BASE environment variable. See this [GitHub](https://github.com/openai/harmony/pull/41) issue for more information.

```bash
mkdir -p tiktoken_encodings
wget -O tiktoken_encodings/o200k_base.tiktoken "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
wget -O tiktoken_encodings/cl100k_base.tiktoken "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
export TIKTOKEN_ENCODINGS_BASE=${PWD}/tiktoken_encodings
```

### Single-node Deployment

`gpt-oss-120b` can both be deployed on 1 Atlas 800 A3(64G × 16), 1 Atlas 800 A2(64G × 8).

Run the following script to execute online inference.

```shell
#!/bin/sh
# Load model from ModelScope to speed up download
export VLLM_USE_MODELSCOPE=True
# To reduce memory fragmentation and avoid out of memory
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=512
export NPU_MEMORY_FRACTION=0.95
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export OMP_PROC_BIND=false
export VLLM_USE_V1=1
export TASK_QUEUE_ENABLE=1
export OMP_NUM_THREADS=1
export TIKTOKEN_ENCODINGS_BASE=/${PWD}/tiktoken_encodings

vllm serve unsloth/gpt-oss-120b-BF16 \
--served-model-name gpt-oss-120b-bf16 \
--port 8000 \
--trust-remote-code \
--async-scheduling \
--max-num-seqs 4 \
--gpu-memory-utilization 0.90 \
--tensor-parallel-size 4 \
--max-model-len 4096 \
--max-num-batched-tokens 4096 \
--enable-expert-parallel \
--compilation_config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes":[1,2,3,4]}'
```

The parameters are explained as follows:

- `--tensor-parallel-size` are common settings for tensor parallelism (TP) sizes.
- `--max-model-len` represents the context length, which is the maximum value of the input plus output for a single request.
- `--max-num-seqs` indicates the maximum number of requests that each DP group is allowed to process. If the number of requests sent to the service exceeds this limit, the excess requests will remain in a waiting state and will not be scheduled. Note that the time spent in the waiting state is also counted in metrics such as TTFT and TPOT. Therefore, when testing performance, it is generally recommended that `--max-num-seqs` * `--data-parallel-size` >= the actual total concurrency.
- `--max-num-batched-tokens` represents the maximum number of tokens that the model can process in a single step. Currently, vLLM v1 scheduling enables ChunkPrefill/SplitFuse by default, which means:
    - (1) If the input length of a request is greater than `--max-num-batched-tokens`, it will be divided into multiple rounds of computation according to `--max-num-batched-tokens`;
    - (2) Decode requests are prioritized for scheduling, and prefill requests are scheduled only if there is available capacity.
    - Generally, if `--max-num-batched-tokens` is set to a larger value, the overall latency will be lower, but the pressure on GPU memory (activation value usage) will be greater.
- `--gpu-memory-utilization` represents the proportion of HBM that vLLM will use for actual inference. Its essential function is to calculate the available kv_cache size. During the warm-up phase (referred to as profile run in vLLM), vLLM records the peak GPU memory usage during an inference process with an input size of `--max-num-batched-tokens`. The available kv_cache size is then calculated as: `--gpu-memory-utilization` * HBM size - peak GPU memory usage. Therefore, the larger the value of `--gpu-memory-utilization`, the more kv_cache can be used. However, since the GPU memory usage during the warm-up phase may differ from that during actual inference (e.g., due to uneven EP load), setting `--gpu-memory-utilization` too high may lead to OOM (Out of Memory) issues during actual inference. The default value is `0.9`.
- `--compilation-config` contains configurations related to the aclgraph graph mode. The most significant configurations are "cudagraph_mode" and "cudagraph_capture_sizes", which have the following meanings:
"cudagraph_mode": represents the specific graph mode. Currently, "PIECEWISE" and "FULL_DECODE_ONLY" are supported. The graph mode is mainly used to reduce the cost of operator dispatch. Currently, "FULL_DECODE_ONLY" is recommended.
- "cudagraph_capture_sizes": represents different levels of graph modes. The default value is [1, 2, 4, 8, 16, 24, 32, 40,..., `--max-num-seqs`]. In the graph mode, the input for graphs at different levels is fixed, and inputs between levels are automatically padded to the next level. Currently, the default setting is recommended. Only in some scenarios is it necessary to set this separately to achieve optimal performance.

## Functional Verification

Once your server is started, you can query the model with input prompts:

```shell
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "gpt-oss-120b-bf16",
        "messages": [{"role":"user", "content":"who are you"}]
    }'
```

## Accuracy Evaluation

Here are two accuracy evaluation methods.

### Using AISBench

1. Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) for details.

2. After execution, you can get the result, here is the result of `gpt-oss-120b-bf16`  for reference only.

| dataset | version | metric | mode | vllm-api-general-chat |
|----- | ----- | ----- | ----- | -----|
| mmlu | - | accuracy | gen | 89.50 |

## Performance

### Using AISBench

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

### Using vLLM Benchmark

Run performance evaluation of `gpt-oss-120b-BF16` as an example.

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/benchmarking/) for more details.

There are three `vllm bench` subcommands:

- `latency`: Benchmark the latency of a single batch of requests.
- `serve`: Benchmark the online serving throughput.
- `throughput`: Benchmark offline inference throughput.

Take the `serve` as an example. Run the code as follows.

```shell
export VLLM_USE_MODELSCOPE=True
vllm bench serve --model unsloth/gpt-oss-120b-BF16 --dataset-name random --random-input 200 --num-prompts 200 --request-rate 1 --save-result --result-dir ./
```

After about several minutes, you can get the performance evaluation result.
