# Single NPU (Pangu 7B)

## Run vllm-ascend on Single NPU

### Offline Inference on Single NPU

Run docker container:

```{code-block} bash
   :substitutions:
# Update the vllm-ascend image
export IMAGE=quay.io/ascend/vllm-ascend:v0.9.2rc1
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

Setup environment variables:

```bash
# Load model from ModelScope to speed up download
export VLLM_USE_MODELSCOPE=True
```


Run the following script to execute offline inference on a single NPU:

:::::{tab-set}
::::{tab-item} Graph Mode

```{code-block} python
   :substitutions:
import os
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
llm = LLM(
        model="pangu_7b",
        max_model_len=26240,
        trust_remote_code=True
)

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

::::

::::{tab-item} Eager Mode

```{code-block} python
   :substitutions:
import os
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
llm = LLM(
        model="pangu_7b",
        max_model_len=26240,
        trust_remote_code=True,
        enforce_eager=True
)

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

::::
:::::

If you run this script successfully, you can see the info shown below:

```bash
Prompt: 'Hello, my name is', Generated text: ' Leif and I am a Medical student in the UK. I have been studying'
Prompt: 'The future of AI is', Generated text: ' about more than just technology. It’s about creating ethical frameworks, understanding human values'
```

### Online Serving on Single NPU

Run docker container to start the vLLM server on a single NPU:

:::::{tab-set}
::::{tab-item} Graph Mode

```{code-block} bash
   :substitutions:
# Update the vllm-ascend image
export IMAGE=quay.io/ascend/vllm-ascend:v0.9.2rc1
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
-it $IMAGE \
vllm serve pangu_7b --max_model_len 26240
```

::::

::::{tab-item} Eager Mode

```{code-block} bash
   :substitutions:
export IMAGE=quay.io/ascend/vllm-ascend:v0.9.2rc1
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
-it $IMAGE \
vllm serve pangu_7b --max_model_len 26240 --enforce-eager
```

::::
:::::

:::{note}
Add `--max_model_len` option to avoid ValueError that the pangu_7b model's max seq len (32768) is larger than the maximum number of tokens that can be stored in KV cache (26240). This will differ with different NPU series base on the HBM size. Please modify the value according to a suitable value for your NPU series.
:::

If your service start successfully, you can see the info shown below:

```bash
INFO:     Started server process [6392]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

Once your server is started, you can query the model with input prompts:

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "pangu_7b",
        "prompt": "The future of AI is",
        "max_tokens": 7,
        "temperature": 0
    }'
```

If you query the server successfully, you can see the info shown below (client):

```bash
{"id":"cmpl-225dde793d7040c9ab13fc2119039f81","object":"text_completion","created":1753858678,"model":"pangu_7b","choices":[{"index":0,"text":" here, and it's in your","logprobs":null,"finish_reason":"length","stop_reason":null,"prompt_logprobs":null}],"usage":{"prompt_tokens":6,"total_tokens":13,"completion_tokens":7,"prompt_tokens_details":null},"kv_transfer_params":null}
```

Logs of the vllm server:

```bash
INFO:     172.17.0.1:49518 - "POST /v1/completions HTTP/1.1" 200 OK
INFO 07-30 14:57:58 [logger.py:43] Received request cmpl-225dde793d7040c9ab13fc2119039f81-0: prompt: 'The future of AI is', params: SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=0.0, top_p=1.0, top_k=0, min_p=0.0, seed=None, stop=[], stop_token_ids=[], bad_words=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=7, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None, guided_decoding=None, extra_args=None), prompt_token_ids: [1, 43615, 15807, 42323, 41598, 41824], prompt_embeds shape: None, lora_request: None, prompt_adapter_request: None.
```
