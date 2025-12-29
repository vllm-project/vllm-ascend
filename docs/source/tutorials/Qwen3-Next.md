# Qwen3-Next

## Introduction

The Qwen3-Next model is a sparse MoE (Mixture of Experts) model with high sparsity. Compared to the MoE architecture of Qwen3, it has introduced key improvements in aspects such as the hybrid attention mechanism and multi-token prediction mechanism, enhancing the training and inference efficiency of the model under long contexts and large total parameter scales.

This document will present the core verification steps of the model, including supported features, environment preparation, as well as accuracy and performance evaluation. Qwen3 Next is currently using Triton Ascend, which is in the experimental phase. In subsequent versions, its performance related to stability and accuracy may change, and performance will be continuously optimized.

The `Qwen3-Next` model is first supported in `vllm-ascend:v0.10.2rc1`.

## Supported Features

Refer to [supported features](../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [feature guide](../user_guide/feature_guide/index.md) to get the feature's configuration.

## Weight Preparation

 Download Link for the `Qwen3-Next-80B-A3B-Instruct` Model Weights: [Download model weight](https://modelers.cn/models/Modelers_Park/Qwen3-Next-80B-A3B-Instruct/tree/main)

## Deployment

If the machine environment is an Atlas 800I A3(64G*16), the deployment approach stays identical.

### Run docker container

```{code-block} bash
   :substitutions:
# Update the vllm-ascend image
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
docker run --rm \
--shm-size=1g \
--name vllm-ascend-qwen3 \
--device /dev/davinci0 \
--device /dev/davinci1 \
--device /dev/davinci2 \
--device /dev/davinci3 \
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--device /dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /root/.cache:/root/.cache \
-p 8000:8000 \
-it $IMAGE bash
```

The Qwen3 Next is using [Triton Ascend](https://gitee.com/ascend/triton-ascend) which is currently experimental. In future versions, there may be behavioral changes related to stability, accuracy, and performance improvement.

### Install Triton Ascend

:::::{tab-set}
::::{tab-item} Linux (AArch64)

The [Triton Ascend](https://gitee.com/ascend/triton-ascend) is required when you run Qwen3 Next, please follow the instructions below to install it and its dependency.

Source the Ascend BiSheng toolkit, execute the command:

```bash
source /usr/local/Ascend/ascend-toolkit/8.3.RC2/bisheng_toolkit/set_env.sh
```

Install Triton Ascend:

```bash
wget https://vllm-ascend.obs.cn-north-4.myhuaweicloud.com/vllm-ascend/triton_ascend-3.2.0.dev2025110717-cp311-cp311-manylinux_2_27_aarch64.whl
pip install triton_ascend-3.2.0.dev2025110717-cp311-cp311-manylinux_2_27_aarch64.whl
```

::::

::::{tab-item} Linux (x86_64)

Coming soon ...

::::
:::::

### Inference

Please make sure you have already executed the command:

```bash
source /usr/local/Ascend/ascend-toolkit/8.3.RC2/bisheng_toolkit/set_env.sh
```

:::::{tab-set}
::::{tab-item} Online Inference

Run the following script to start the vLLM server on multi-NPU:

```bash
vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct --tensor-parallel-size 4 --max-model-len 32768 --gpu-memory-utilization 0.8 --max-num-batched-tokens 4096 --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
```

Once your server is started, you can query the model with input prompts.

```bash
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "Qwen/Qwen3-Next-80B-A3B-Instruct",
  "messages": [
    {"role": "user", "content": "Who are you?"}
  ],
  "temperature": 0.6,
  "top_p": 0.95,
  "top_k": 20,
  "max_tokens": 32
}'
```

::::

::::{tab-item} Offline Inference

Run the following script to execute offline inference on multi-NPU:

```python
import gc
import torch

from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import (destroy_distributed_environment,
                                             destroy_model_parallel)

def clean_up():
    destroy_model_parallel()
    destroy_distributed_environment()
    gc.collect()
    torch.npu.empty_cache()

if __name__ == '__main__':
    prompts = [
        "Who are you?",
    ]
    sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=40, max_tokens=32)
    llm = LLM(model="Qwen/Qwen3-Next-80B-A3B-Instruct",
              tensor_parallel_size=4,
              enforce_eager=True,
              distributed_executor_backend="mp",
              gpu_memory_utilization=0.7,
              max_model_len=4096)

    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    del llm
    clean_up()
```

If you run this script successfully, you can see the info shown below:

```bash
Prompt: 'Who are you?', Generated text: ' What do you know about me?\n\nHello! I am Qwen, a large-scale language model independently developed by the Tongyi Lab under Alibaba Group. I am'
```

::::
:::::

## Accuracy Evaluation

### Using AISBench

1. Refer to [Using AISBench](../developer_guide/evaluation/using_ais_bench.md) for details.

2. After execution, you can get the result, here is the result of `Qwen3-Next-80B-A3B-Instruct` in `vllm-ascend:0.13.0rc1` for reference only.

| dataset | version | metric | mode | vllm-api-general-chat |
|----- | ----- | ----- | ----- | -----|
| gsm8k | - | accuracy | gen | 95.53 |

## Performance

### Using AISBench

Refer to [Using AISBench for performance evaluation](../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

### Using vLLM Benchmark

Run performance evaluation of `Qwen3-Next` as an example.

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/contributing/benchmarks.html) for more details.

There are three `vllm bench` subcommand:
- `latency`: Benchmark the latency of a single batch of requests.
- `serve`: Benchmark the online serving throughput.
- `throughput`: Benchmark offline inference throughput.

Take the `serve` as an example. Run the code as follows.

```shell
export VLLM_USE_MODELSCOPE=true
vllm bench serve --model Qwen/Qwen3-Next-80B-A3B-Instruct  --dataset-name random --random-input 200 --num-prompt 200 --request-rate 1 --save-result --result-dir ./
```

After about several minutes, you can get the performance evaluation result.

The performance result is:  

**Hardware**: A3-752T, 2 node

**Deployment**: TP4 + Full Decode Only

**Input/Output**: 2k/2k

**Concurrency**: 32

**Performance**: 580tps, TPOT 54ms
