# Multi-NPU (Qwen3-Next)

```{note}
The Qwen3 Next is using [Triton Ascend](https://gitee.com/ascend/triton-ascend) which is currently experimental. In future versions, there may be behavioral changes related to stability, accuracy, and performance improvement.
```

## Run vllm-ascend on Multi-NPU with Qwen3 Next

Run docker container:

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

Set up environment variables:

```bash
# Load model from ModelScope to speed up download
export VLLM_USE_MODELSCOPE=True
```

### Install Triton Ascend

:::::{tab-set}
::::{tab-item} Linux (AArch64)

The [Triton Ascend](https://gitee.com/ascend/triton-ascend) is required when you run Qwen3 Next, please follow the instructions below to install it and its dependency.

Install the Ascend BiSheng toolkit:

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

### Inference on Multi-NPU

Please make sure you have already executed the command:

```bash
source /usr/local/Ascend/ascend-toolkit/8.3.RC2/bisheng_toolkit/set_env.sh
```

:::::{tab-set}
::::{tab-item} Online Inference

Run the following script to start the vLLM server on multi-NPU:

For an Atlas A2 with 64 GB of NPU card memory, tensor-parallel-size should be at least 4, and for 32 GB of memory, tensor-parallel-size should be at least 8.

```bash
vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct --tensor-parallel-size 4 --max-model-len 4096 --gpu-memory-utilization 0.7 --enforce-eager
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
