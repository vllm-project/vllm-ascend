# Single Node (Atlas 300I series)

```{note}
1. This Atlas 300I series is currently experimental. In future versions, there may be behavioral changes around model coverage, performance improvement.
2. Currently, the 310I series only supports eager mode and the data type is float16.
```

## Run vLLM on Altlas 300I series

Run docker container:

```{code-block} bash
   :substitutions:
# Update the vllm-ascend image
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|-310p
docker run --rm \
--name vllm-ascend \
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
# Load model from ModelScope to speed up download
export VLLM_USE_MODELSCOPE=True

# Set `max_split_size_mb` to reduce memory fragmentation and avoid out of memory
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256
```

### Online Inference on NPU

Run the following script to start the vLLM server on NPU(Qwen3-0.6B:1 card, Qwen2.5-7B-Instruct:2 cards, Pangu-Pro-MoE-72B: 8 cards):

:::::{tab-set}
:sync-group: inference

::::{tab-item} Qwen3-0.6B
:selected:
:sync: qwen0.6

Run the following command to start the vLLM server:

```{code-block} bash
   :substitutions:
vllm serve Qwen/Qwen3-0.6B \
    --tensor-parallel-size 1 \
    --enforce-eager \
    --dtype float16 \
    --compilation-config '{"custom_ops":["none", "+rms_norm", "+rotary_embedding"]}'
```

Once your server is started, you can query the model with input prompts

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The future of AI is",
    "max_tokens": 64,
    "top_p": 0.95,
    "top_k": 50,
    "temperature": 0.6
  }'
```

::::

::::{tab-item} Qwen2.5-7B-Instruct
:sync: qwen7b

Run the following command to start the vLLM server:

```{code-block} bash
   :substitutions:
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --tensor-parallel-size 2 \
    --enforce-eager \
    --dtype float16 \
    --compilation-config '{"custom_ops":["none", "+rms_norm", "+rotary_embedding"]}'
```

Once your server is started, you can query the model with input prompts

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The future of AI is",
    "max_tokens": 64,
    "top_p": 0.95,
    "top_k": 50,
    "temperature": 0.6
  }'
```

::::

::::{tab-item} Qwen2.5-VL-3B-Instruct
:sync: qwen-vl-2.5-3b

Run the following command to start the vLLM server:

```{code-block} bash
   :substitutions:
vllm serve Qwen/Qwen2.5-VL-3B-Instruct \
    --tensor-parallel-size 1 \
    --enforce-eager \
    --dtype float16 \
    --compilation-config '{"custom_ops":["none", "+rms_norm", "+rotary_embedding"]}'
```

Once your server is started, you can query the model with input prompts

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The future of AI is",
    "max_tokens": 64,
    "top_p": 0.95,
    "top_k": 50,
    "temperature": 0.6
  }'
```

::::

::::{tab-item} Pangu-Pro-MoE-72B
:sync: pangu

Download the model:

```bash
git lfs install
git clone https://gitcode.com/ascend-tribe/pangu-pro-moe-model.git
```

Run the following command to start the vLLM server:

```{code-block} bash
   :substitutions:

vllm serve /home/pangu-pro-moe-mode/ \
--tensor-parallel-size 4 \
--enable-expert-parallel \
--dtype "float16" \
--trust-remote-code \
--enforce-eager

```

Once your server is started, you can query the model with input prompts

```bash
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
:::::

If you run this script successfully, you can see the results.

### Offline Inference

Run the following script (`example.py`) to execute offline inference on NPU:

:::::{tab-set}
:sync-group: inference

::::{tab-item} Qwen3-0.6B
:selected:
:sync: qwen0.6

```{code-block} python
   :substitutions:
from vllm import LLM, SamplingParams
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
prompts = [
    "Hello, my name is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(max_tokens=100, temperature=0.0)
# Create an LLM.
llm = LLM(
    model="Qwen/Qwen3-0.6B",
    tensor_parallel_size=1,
    enforce_eager=True, # For 300I series, only eager mode is supported.
    dtype="float16", # IMPORTANT cause some ATB ops cannot support bf16 on 300I series
    compilation_config={"custom_ops":["none", "+rms_norm", "+rotary_embedding"]}, # High performance for 300I series
)
# Generate texts from the prompts.
outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
del llm
clean_up()
```

::::

::::{tab-item} Qwen2.5-7B-Instruct
:sync: qwen7b

```{code-block} python
   :substitutions:
from vllm import LLM, SamplingParams
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
prompts = [
    "Hello, my name is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(max_tokens=100, temperature=0.0)
# Create an LLM.
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    tensor_parallel_size=2,
    enforce_eager=True, # For 300I series, only eager mode is supported.
    dtype="float16", # IMPORTANT cause some ATB ops cannot support bf16 on 300I series
    compilation_config={"custom_ops":["none", "+rms_norm", "+rotary_embedding"]}, # High performance for 300I series
)
# Generate texts from the prompts.
outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
del llm
clean_up()
```

::::

::::{tab-item} Qwen2.5-VL-3B-Instruct
:sync: qwen-vl-2.5-3b

```{code-block} python
   :substitutions:
from vllm import LLM, SamplingParams
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
prompts = [
    "Hello, my name is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(max_tokens=100, top_p=0.95, top_k=50, temperature=0.6)
# Create an LLM.
llm = LLM(
    model="Qwen/Qwen2.5-VL-3B-Instruct",
    tensor_parallel_size=1,
    enforce_eager=True, # For 300I series, only eager mode is supported.
    dtype="float16", # IMPORTANT cause some ATB ops cannot support bf16 on 300I series
    compilation_config={"custom_ops":["none", "+rms_norm", "+rotary_embedding"]}, # High performance for 300I series
)
# Generate texts from the prompts.
outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
del llm
clean_up()
```

::::

::::{tab-item} Pangu-Pro-MoE-72B
:sync: pangu

Download the model:

```bash
git lfs install
git clone https://gitcode.com/ascend-tribe/pangu-pro-moe-model.git
```

```{code-block} python
   :substitutions:

import gc
from transformers import AutoTokenizer
import torch

from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import (destroy_distributed_environment,
                                             destroy_model_parallel)

def clean_up():
    destroy_model_parallel()
    destroy_distributed_environment()
    gc.collect()
    torch.npu.empty_cache()


if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained("/home/pangu-pro-moe-mode/", trust_remote_code=True)
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
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)        # 推荐使用官方的template
        prompts.append(prompt)
    sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=40)

    llm = LLM(model="/home/pangu-pro-moe-mode/",
            tensor_parallel_size=8,
            distributed_executor_backend="mp",
            enable_expert_parallel=True,
            dtype="float16",
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

Run script:

```bash
python example.py
```

If you run this script successfully, you can see the info shown below:

```bash
Prompt: 'Hello, my name is', Generated text: " Lina. I'm a 22-year-old student from China. I'm interested in studying in the US. I'm looking for a job in the US. I want to know if there are any opportunities in the US for me to work. I'm also interested in the culture and lifestyle in the US. I want to know if there are any opportunities for me to work in the US. I'm also interested in the culture and lifestyle in the US. I'm interested in the culture"
Prompt: 'The future of AI is', Generated text: " not just about the technology itself, but about how we use it to solve real-world problems. As AI continues to evolve, it's important to consider the ethical implications of its use. AI has the potential to bring about significant changes in society, but it also has the power to create new challenges. Therefore, it's crucial to develop a comprehensive approach to AI that takes into account both the benefits and the risks associated with its use. This includes addressing issues such as bias, privacy, and accountability."
```
