# Single NPU (Qwen2.5 7B)

## Run vllm-ascend on Single NPU

### Offline Inference on Single NPU

Run docker container:

```{code-block} bash
   :substitutions:
# Update the vllm-ascend image
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
docker run --rm \
--name vllm-ascend \
--device /dev/davinci0 \
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

(Optional) Install MindIE Turbo for performance acceleration:

```{code-block} bash
   :substitutions:

pip install mindie_turbo==|pip_mindie_turbo_version|
```

Setup environment variables:

```bash
# Load model from ModelScope to speed up download
export VLLM_USE_MODELSCOPE=True

# Set `max_split_size_mb` to reduce memory fragmentation and avoid out of memory
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256
```

:::{note}
`max_split_size_mb` prevents the native allocator from splitting blocks larger than this size (in MB). This can reduce fragmentation and may allow some borderline workloads to complete without running out of memory. You can find more details [<u>here</u>](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/envref/envref_07_0061.html).
:::

Run the following script to execute offline inference on a single NPU:

```python
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
llm = LLM(model="Qwen/Qwen2.5-7B-Instruct", max_model_len=26240)

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

If you run this script successfully, you can see the info shown below:

```bash
Prompt: 'Hello, my name is', Generated text: ' Daniel and I am an 8th grade student at York Middle School. I'
Prompt: 'The future of AI is', Generated text: ' following you. As the technology advances, a new report from the Institute for the'
```

### Online Serving on Single NPU

Run docker container to start the vLLM server on a single NPU:

```{code-block} bash
   :substitutions:

# Update the vllm-ascend image
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
docker run --rm \
--name vllm-ascend \
--device /dev/davinci0 \
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
-e VLLM_USE_MODELSCOPE=True \
-e PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256 \
-it $IMAGE bash
```

(Optional) Install MindIE Turbo for performance acceleration:

```{code-block} bash
   :substitutions:

pip install mindie_turbo==|pip_mindie_turbo_version|
```

Run the following script to start the vLLM server:

```
vllm serve Qwen/Qwen2.5-7B-Instruct --max_model_len 26240
```

:::{note}
Add `--max_model_len` option to avoid ValueError that the Qwen2.5-7B model's max seq len (32768) is larger than the maximum number of tokens that can be stored in KV cache (26240). This will differ with different NPU series base on the HBM size. Please modify the value according to a suitable value for your NPU series.
:::

If your service start successfully, you can see the info shown below:

```bash
INFO:     Started server process [6873]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

Once your server is started, you can query the model with input prompts:

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "prompt": "The future of AI is",
        "max_tokens": 7,
        "temperature": 0
    }'
```

If you query the server successfully, you can see the info shown below (client):

```bash
{"id":"cmpl-b25a59a2f985459781ce7098aeddfda7","object":"text_completion","created":1739523925,"model":"Qwen/Qwen2.5-7B-Instruct","choices":[{"index":0,"text":" here. It’s not just a","logprobs":null,"finish_reason":"length","stop_reason":null,"prompt_logprobs":null}],"usage":{"prompt_tokens":5,"total_tokens":12,"completion_tokens":7,"prompt_tokens_details":null}}
```

Logs of the vllm server:

```bash
INFO:     172.17.0.1:49518 - "POST /v1/completions HTTP/1.1" 200 OK
INFO 02-13 08:34:35 logger.py:39] Received request cmpl-574f00e342904692a73fb6c1c986c521-0: prompt: 'San Francisco is a', params: SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=0.0, top_p=1.0, top_k=-1, min_p=0.0, seed=None, stop=[], stop_token_ids=[], bad_words=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=7, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None, guided_decoding=None), prompt_token_ids: [23729, 12879, 374, 264], lora_request: None, prompt_adapter_request: None.
```
