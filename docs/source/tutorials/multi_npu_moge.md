# Multi-NPU (Pangu Pro MoE)

## Run vllm-ascend on Multi-NPU

Run container:

```{code-block} bash
   :substitutions:
# Update the vllm-ascend image
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
docker run --rm \
--name vllm-ascend \
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

Setup environment variables:

```bash
# Set `max_split_size_mb` to reduce memory fragmentation and avoid out of memory
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256
```

Download the model:

```bash
git lfs install
git clone https://gitcode.com/ascend-tribe/pangu-pro-moe-model.git
```

### Online Inference on Multi-NPU

Run the following script to start the vLLM server on Multi-NPU:

```bash
vllm serve /path/to/pangu-pro-moe-model \
--tensor-parallel-size 4 \
--enable-expert-parallel \
--trust-remote-code \
--enforce-eager
```

Once your server is started, you can query the model with input prompts:

:::::{tab-set}
::::{tab-item} v1/completions

```{code-block} bash
   :substitutions:
export question="你是谁？"
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "[unused9]系统：[unused10][unused9]用户：'${question}'[unused10][unused9]助手：",
    "max_tokens": 64,
    "top_p": 0.95,
    "top_k": 50,
    "temperature": 0.6
  }'
```

::::

::::{tab-item} v1/chat/completions

```{code-block} bash
   :substitutions:
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "messages": [
      {"role": "system", "content": ""},
      {"role": "user", "content": "你是谁？"}
    ],
        "max_tokens": "64",
        "top_p": "0.95",
        "top_k": "50",
        "temperature": "0.6",
        "add_special_tokens" : true
    }'
```

::::
:::::

If you run this successfully, you can see the info shown below:

```json
{"id":"cmpl-2cd4223228ab4be9a91f65b882e65b32","object":"text_completion","created":1751255067,"model":"/root/.cache/pangu-pro-moe-model","choices":[{"index":0,"text":" [unused16] 好的，用户问我是谁，我需要根据之前的设定来回答。用户提到我是华为开发的“盘古Reasoner”，属于盘古大模型系列，作为智能助手帮助解答问题和提供 信息支持。现在用户再次询问，可能是在确认我的身份或者测试我的回答是否一致。\n\n首先，我要确保","logprobs":null,"finish_reason":"length","stop_reason":null,"prompt_logprobs":null}],"usage":{"prompt_tokens":15,"total_tokens":79,"completion_tokens":64,"prompt_tokens_details":null},"kv_transfer_params":null}
```

### Offline Inference on Multi-NPU

Run the following script to execute offline inference on multi-NPU:

:::::{tab-set}
::::{tab-item} Graph Mode

```{code-block} python
   :substitutions:
import gc
from transformers import AutoTokenizer
import torch
import os

from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import (destroy_distributed_environment,
                                             destroy_model_parallel)

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
def clean_up():
    destroy_model_parallel()
    destroy_distributed_environment()
    gc.collect()
    torch.npu.empty_cache()


if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained("/path/to/pangu-pro-moe-model", trust_remote_code=True)
    tests = [
        "Hello, my name is",
        "The future of AI is",
    ]
    prompts = []
    for text in tests:
        messages = [
        {"role": "system", "content": ""},    # Optionally customize system content
        {"role": "user", "content": text}
    ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)

    sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=40)

    llm = LLM(model="/path/to/pangu-pro-moe-model",
            tensor_parallel_size=4,
            enable_expert_parallel=True,
            distributed_executor_backend="mp",
            max_model_len=1024,
            trust_remote_code=True,
            additional_config={
            'torchair_graph_config': {
            'enabled': True,
            },
            'ascend_scheduler_config':{
            'enabled': True,
            'enable_chunked_prefill' : False,
            'chunked_prefill_enabled': False
            },
            })

    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    del llm
    clean_up()
```

::::

::::{tab-item} Eager Mode

```{code-block} python
   :substitutions:
import gc
from transformers import AutoTokenizer
import torch
import os

from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import (destroy_distributed_environment,
                                             destroy_model_parallel)

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
def clean_up():
    destroy_model_parallel()
    destroy_distributed_environment()
    gc.collect()
    torch.npu.empty_cache()


if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained("/path/to/pangu-pro-moe-model", trust_remote_code=True)
    tests = [
        "Hello, my name is",
        "The future of AI is",
    ]
    prompts = []
    for text in tests:
        messages = [
        {"role": "system", "content": ""},    # Optionally customize system content
        {"role": "user", "content": text}
    ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)

    sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=40)

    llm = LLM(model="/path/to/pangu-pro-moe-model",
            tensor_parallel_size=4,
            distributed_executor_backend="mp",
            max_model_len=1024,
            trust_remote_code=True,
            enforce_eager=True)

    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    del llm
    clean_up()
```

::::
:::::

If you run this script successfully, you can see the info shown below:

```bash
Prompt: 'Hello, my name is', Generated text: ' Daniel and I am an 8th grade student at York Middle School. I'
Prompt: 'The future of AI is', Generated text: ' following you. As the technology advances, a new report from the Institute for the'
```
