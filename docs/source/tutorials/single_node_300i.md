# Single Node (Atlas 300I series)

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

Run the following script to start the vLLM server on NPU(Qwen3-0.6B:1 card, Qwen2.5-7B-Instruct:2 cards):

:::::{tab-set}
::::{tab-item} Qwen3-0.6B

```{code-block} bash
   :substitutions:
export VLLM_USE_V1=1
export MODEL="Qwen/Qwen3-0.6B"
python -m vllm.entrypoints.api_server \
    --model $MODEL \
    --tensor-parallel-size 1 \
    --max-num-batched-tokens 2048 \
    --gpu-memory-utilization 0.5 \
    --max-num-seqs 4 \
    --enforce-eager \
    --trust-remote-code \
    --max-model-len 1024 \
    --disable-custom-all-reduce \
    --dtype float16 \
    --port 8000 \
    --compilation-config '{"custom_ops":["+rms_norm", "+rotary_embedding"]}' 
```
::::

::::{tab-item} Qwen/Qwen2.5-7B-Instruct

```{code-block} bash
   :substitutions:
export VLLM_USE_V1=1
export MODEL="Qwen/Qwen2.5-7B-Instruct"
python -m vllm.entrypoints.api_server \
    --model $MODEL \
    --tensor-parallel-size 2 \
    --max-num-batched-tokens 2048 \
    --gpu-memory-utilization 0.5 \
    --max-num-seqs 4 \
    --enforce-eager \
    --trust-remote-code \
    --max-model-len 1024 \
    --disable-custom-all-reduce \
    --dtype float16 \
    --port 8000 \
    --compilation-config '{"custom_ops":["+rms_norm", "+rotary_embedding"]}' 
```
::::

:::::

Once your server is started, you can query the model with input prompts

```bash
curl http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello, my name is ？",
    "max_tokens": 20,
    "temperature": 0
  }'
```

If you run this script successfully, you can see the info shown below:

```bash
{"text":["The future of AI is ？  \nA. 充满希望的  \nB. 不确定的  \nC. 危险的  \nD. 无法预测的  \n答案：A  \n解析："]}
```

### Offline Inference

Run the following script to execute offline inference on NPU:

:::::{tab-set}
::::{tab-item} Qwen3-0.6B

```{code-block} python
   :substitutions:
from vllm import LLM, SamplingParams
import gc
import os
import torch
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import (destroy_distributed_environment,
                                             destroy_model_parallel)
os.environ["VLLM_USE_V1"] = "1"
def clean_up():
    destroy_model_parallel()
    destroy_distributed_environment()
    gc.collect()
    torch.npu.empty_cache()
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(max_tokens=100, temperature=0.0)
# Create an LLM.
llm = LLM(
    model="Qwen/Qwen3-0.6B",
    max_model_len=4096,
    max_num_seqs=4,
    trust_remote_code=True,
    tensor_parallel_size=1,
    enforce_eager=True, # For 300I series, only eager mode is supported.
    dtype="float16", # IMPORTANT cause some ATB ops cannot support bf16 on 300I series
    disable_custom_all_reduce=True, # IMPORTANT cause 300I series needed
    compilation_config={"custom_ops":["+rms_norm", "+rotary_embedding"]}, # IMPORTANT cause 300I series needed custom ops
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

```{code-block} python
   :substitutions:
from vllm import LLM, SamplingParams
import gc
import os
import torch
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import (destroy_distributed_environment,
                                             destroy_model_parallel)
os.environ["VLLM_USE_V1"] = "1"
def clean_up():
    destroy_model_parallel()
    destroy_distributed_environment()
    gc.collect()
    torch.npu.empty_cache()
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(max_tokens=100, temperature=0.0)
# Create an LLM.
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    max_model_len=4096,
    max_num_seqs=4,
    trust_remote_code=True,
    tensor_parallel_size=2,
    enforce_eager=True, # For 300I series, only eager mode is supported.
    dtype="float16", # IMPORTANT cause some ATB ops cannot support bf16 on 300I series
    disable_custom_all_reduce=True, # IMPORTANT cause 300I series needed
    compilation_config={"custom_ops":["+rms_norm", "+rotary_embedding"]}, # IMPORTANT cause 300I series needed custom ops
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

:::::

If you run this script successfully, you can see the info shown below:

```bash
Prompt: 'Hello, my name is', Generated text: " Lina. I'm a 22-year-old student from China. I'm interested in studying in the US. I'm looking for a job in the US. I want to know if there are any opportunities in the US for me to work. I'm also interested in the culture and lifestyle in the US. I want to know if there are any opportunities for me to work in the US. I'm also interested in the culture and lifestyle in the US. I'm interested in the culture"
Prompt: 'The president of the United States is', Generated text: ' the same as the president of the United Nations. This is because the president of the United States is the same as the president of the United Nations. The president of the United States is the same as the president of the United Nations. The president of the United States is the same as the president of the United Nations. The president of the United States is the same as the president of the United Nations. The president of the United States is the same as the president of the United Nations. The president'
Prompt: 'The capital of France is', Generated text: ' Paris. The capital of Italy is Rome. The capital of Spain is Madrid. The capital of China is Beijing. The capital of Japan is Tokyo. The capital of India is New Delhi. The capital of Brazil is Brasilia. The capital of Egypt is Cairo. The capital of South Africa is Cape Town. The capital of Nigeria is Abuja. The capital of Lebanon is Beirut. The capital of Morocco is Rabat. The capital of Indonesia is Jakarta. The capital of Peru is Lima. The'
Prompt: 'The future of AI is', Generated text: " not just about the technology itself, but about how we use it to solve real-world problems. As AI continues to evolve, it's important to consider the ethical implications of its use. AI has the potential to bring about significant changes in society, but it also has the power to create new challenges. Therefore, it's crucial to develop a comprehensive approach to AI that takes into account both the benefits and the risks associated with its use. This includes addressing issues such as bias, privacy, and accountability."
```
