# Single NPU (Qwen2.5-VL 7B)

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

```bash
pip install torchvision==0.20.1 qwen_vl_utils --extra-index-url https://download.pytorch.org/whl/cpu/
```

```python
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"

llm = LLM(
    model=MODEL_PATH,
    max_model_len=16384,
    limit_mm_per_prompt={"image": 10},
)

sampling_params = SamplingParams(
    max_tokens=512
)

image_messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://modelscope.oss-cn-beijing.aliyuncs.com/resource/qwen.png",
                "min_pixels": 224 * 224,
                "max_pixels": 1280 * 28 * 28,
            },
            {"type": "text", "text": "Please provide a detailed description of this image"},
        ],
    },
]

messages = image_messages

processor = AutoProcessor.from_pretrained(MODEL_PATH)
prompt = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

image_inputs, _, _ = process_vision_info(messages, return_video_kwargs=True)

mm_data = {}
if image_inputs is not None:
    mm_data["image"] = image_inputs

llm_inputs = {
    "prompt": prompt,
    "multi_modal_data": mm_data,
}

outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
generated_text = outputs[0].outputs[0].text

print(generated_text)

```

If you run this script successfully, you can see the info shown below:

```bash
Processed prompts: 100%|███████████████| 1/1 [00:11<00:00, 11.29s/it, est. speed input: 9.48 toks/s, output: 20.55 toks/s]
The image displays a logo consisting of two main elements: a stylized geometric design and a pair of text elements.

1. **Geometric Design**: On the left side of the image, there is a blue geometric design that appears to be made up of interconnected shapes. These shapes resemble a network or a complex polygonal structure, possibly hinting at a technological or interconnected theme. The design is monochromatic and uses only blue as its color, which could be indicative of a specific brand or company.

2. **Text Elements**: To the right of the geometric design, there are two lines of text. The first line reads "TONGYI" in a sans-serif font, with the "YI" part possibly being capitalized. The second line reads "Qwen" in a similar sans-serif font, but in a smaller size.

The overall design is modern and minimalist, with a clear contrast between the geometric and textual elements. The use of blue for the geometric design could suggest themes of technology, connectivity, or innovation, which are common associations with the color blue in branding. The simplicity of the design makes it easily recognizable and memorable.
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
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --dtype bfloat16 --max_model_len 16384 --max-num-batched-tokens 16384
```

:::{note}
Add `--max_model_len` option to avoid ValueError that the Qwen2.5-VL-7B-Instruct model's max seq len (128000) is larger than the maximum number of tokens that can be stored in KV cache. This will differ with different NPU series base on the HBM size. Please modify the value according to a suitable value for your NPU series.
:::

If your service start successfully, you can see the info shown below:

```bash
INFO:     Started server process [2736]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

Once your server is started, you can query the model with input prompts:

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
    "model": "Qwen/Qwen2.5-VL-7B-Instruct",
    "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://modelscope.oss-cn-beijing.aliyuncs.com/resource/qwen.png"}},
        {"type": "text", "text": "What is the text in the illustrate?"}
    ]}
    ]
    }'
```

If you query the server successfully, you can see the info shown below (client):

```bash
{"id":"chatcmpl-f04fb20e79bb40b39b8ed7fdf5bd613a","object":"chat.completion","created":1741749149,"model":"Qwen/Qwen2.5-VL-7B-Instruct","choices":[{"index":0,"message":{"role":"assistant","reasoning_content":null,"content":"The text in the illustration reads \"TONGYI Qwen.\"","tool_calls":[]},"logprobs":null,"finish_reason":"stop","stop_reason":null}],"usage":{"prompt_tokens":74,"total_tokens":89,"completion_tokens":15,"prompt_tokens_details":null},"prompt_logprobs":null}
```

Logs of the vllm server:

```bash
INFO 03-12 11:16:50 logger.py:39] Received request chatcmpl-92148a41eca64b6d82d3d7cfa5723aeb: prompt: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\nWhat is the text in the illustrate?<|im_end|>\n<|im_start|>assistant\n', params: SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=1.0, top_p=1.0, top_k=-1, min_p=0.0, seed=None, stop=[], stop_token_ids=[], bad_words=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=16353, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None, guided_decoding=None), prompt_token_ids: None, lora_request: None, prompt_adapter_request: None.
INFO 03-12 11:16:50 engine.py:280] Added request chatcmpl-92148a41eca64b6d82d3d7cfa5723aeb.
INFO:     127.0.0.1:54004 - "POST /v1/chat/completions HTTP/1.1" 200 OK
```