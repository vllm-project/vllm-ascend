# Qwen3.5-0.8B

## Introduction

Qwen3.5 represents a significant leap forward, integrating breakthroughs in multimodal learning, architectural efficiency, reinforcement learning scale, and global accessibility to empower developers and enterprises with unprecedented capability and efficiency.

Qwen3.5-0.8B is the ultra-compact model in the Qwen3.5 family, featuring a Hybrid Attention + Mamba (Gated DeltaNet) architecture that combines full attention and linear attention layers for efficient inference. With only 0.8B parameters and ~1.73GB weight size, it is designed for edge deployment and resource-constrained scenarios, while supporting multimodal capabilities (text + image + video).

This document will show the main verification steps of the model, including supported features, environment preparation, deployment, accuracy and performance evaluation.

The `Qwen3.5-0.8B` model is first supported in `vllm-ascend:v0.17.0rc1`.

## Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [feature guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## Weight Preparation

`Qwen3.5-0.8B` (BF16 version) requires 1 Atlas 800I A2/A3 (64G x 1) NPU card.

Download model weight from [HuggingFace](https://huggingface.co/Qwen/Qwen3.5-0.8B) or [ModelScope](https://modelscope.cn/models/Qwen/Qwen3.5-0.8B).

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`.

## Deployment

### Run docker container

```{code-block} bash
   :substitutions:
# Update --device according to your device (Atlas A2: /dev/davinci[0-7] Atlas A3:/dev/davinci[0-15]).
# Update the vllm-ascend image according to your environment.
# Note you should download the weight to /root/.cache in advance.
# Update the vllm-ascend image
export IMAGE=m.daocloud.io/quay.io/ascend/vllm-ascend:|vllm_ascend_version|
export NAME=vllm-ascend

# Run the container using the defined variables
docker run --rm \
--name $NAME \
--net=host \
--shm-size=1g \
--device /dev/davinci0 \
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--device /dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-it $IMAGE bash
```

### Inference

:::::{tab-set}
::::{tab-item} Online Inference

Run the following script to start the vLLM server:

```bash
# Load model from ModelScope to speed up download
export VLLM_USE_MODELSCOPE=true
# To reduce memory fragmentation and avoid out of memory
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=512
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1

vllm serve Qwen/Qwen3.5-0.8B \
--host 0.0.0.0 \
--port 8000 \
--tensor-parallel-size 1 \
--seed 1024 \
--served-model-name qwen3.5 \
--max-num-seqs 256 \
--max-model-len 4096 \
--max-num-batched-tokens 8192 \
--trust-remote-code \
--gpu-memory-utilization 0.90 \
--no-enable-prefix-caching \
--compilation-config '{"aclgraph_mode":"FULL_DECODE_ONLY"}' \
--additional-config '{"enable_cpu_binding":true}' \
--async-scheduling
```

Once your server is started, you can query the model with input prompts:

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen3.5",
        "prompt": "The future of AI is",
        "max_completion_tokens": 50,
        "temperature": 0
    }'
```

::::

::::{tab-item} Offline Inference

Run the following script to execute offline inference:

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
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0, max_tokens=50)
    llm = LLM(model="Qwen/Qwen3.5-0.8B",
              enforce_eager=True,
              trust_remote_code=True,
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
Prompt: 'The future of AI is', Generated text: ' a fascinating topic that has been widely discussed...'
```

::::
:::::

:::{note}
For mamba-like models such as Qwen3.5, if you want to enable prefix caching, set `--enable-prefix-caching` and `--mamba-cache-mode align`. Note that the current implementation of hybrid kv cache might result in a very large block_size when scheduling. For example, the block_size may be adjusted to 2048, which means that any prefix shorter than 2048 will never be cached.
:::

## Accuracy Evaluation

### Using AISBench

1. Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) for details.

2. After execution, you can get the result, here is the result of `Qwen3.5-0.8B` in `vllm-ascend:v0.17.0rc1` for reference only.

| dataset | version | metric | mode | vllm-api-general-chat |
| --- | --- | --- | --- | --- |
| gsm8k | - | accuracy | gen | 55.42 |

## Performance

### Using AISBench

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

### Using vLLM Benchmark

Run performance evaluation of `Qwen3.5-0.8B` as an example.

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/contributing/benchmarks.html) for more details.

There are three `vllm bench` subcommands:

- `latency`: Benchmark the latency of a single batch of requests.
- `serve`: Benchmark the online serving throughput.
- `throughput`: Benchmark offline inference throughput.

Take the `serve` as an example. Run the code as follows.

```shell
export VLLM_USE_MODELSCOPE=true
vllm bench serve --model Qwen/Qwen3.5-0.8B --served-model-name qwen3.5 --dataset-name random --random-input 512 --random-output 128 --num-prompts 64 --request-rate inf --save-result --result-dir ./
```

After about several minutes, you can get the performance evaluation result.

The performance result is:

**Hardware**: Atlas 800I A3 (64G x 1)

**Deployment**: TP1 + Full Decode Only + Async Scheduling

**Input/Output**: 512/128

**Performance**: Output tok/s ~994 (peak), TPOT ~41ms
