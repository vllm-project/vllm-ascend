# Qwen3-VL-Dense

## Introduction

The Qwen-VL(Vision-Language)series from Alibaba Cloud comprises a family of powerful Large Vision-Language Models (LVLMs) designed for comprehensive multimodal understanding. They accept images, text, and bounding boxes as input, and output text and detection boxes, enabling advanced functions like image detection, multi-modal dialogue, and multi-image reasoning.

This document will show the main verification steps of the model, including supported features, feature configuration, environment preparation, NPU deployment, accuracy and performance evaluation.

This tutorial uses the vLLM-Ascend `v0.11.0rc2` version for demonstration, showcasing the `Qwen3-VL-8B-Instruct model` as an example for single NPU deployment and the `Qwen2.5-VL-32B-Instruct` model to illustrate multi-NPU deployment.

## Supported Features

Refer to [supported features](../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [feature guide](../user_guide/feature_guide/index.md) to get the feature's configuration.

## Environment Preparation

### Model Weight

require 1 Atlas 800I A2 (64G × 8) node or 1 Atlas 800 A3 (64G × 16) node:
- `Qwen2.5-VL-3B-Instruct`: [Download model weight](https://modelscope.cn/models/Qwen/Qwen2.5-VL-3B-Instruct)
- `Qwen2.5-VL-7B-Instruct`: [Download model weight](https://modelscope.cn/models/Qwen/Qwen2.5-VL-7B-Instruct)
- `Qwen2.5-VL-32B-Instruct`:[Download model weight](https://modelscope.cn/models/Qwen/Qwen2.5-VL-32B-Instruct)
- `Qwen2.5-VL-72B-Instruct`:[Download model weight](https://modelscope.cn/models/Qwen/Qwen2.5-VL-72B-Instruct)
- `Qwen3-VL-2B-Instruct`:   [Download model weight](https://modelscope.cn/models/Qwen/Qwen3-VL-2B-Instruct)
- `Qwen3-VL-4B-Instruct`:   [Download model weight](https://modelscope.cn/models/Qwen/Qwen3-VL-4B-Instruct)
- `Qwen3-VL-8B-Instruct`:   [Download model weight](https://modelscope.cn/models/Qwen/Qwen3-VL-8B-Instruct)
- `Qwen3-VL-32B-Instruct`:  [Download model weight](https://modelscope.cn/models/Qwen/Qwen3-VL-32B-Instruct)

A sample Qwen2.5-VL quantization script can be found in the modelslim code repository. [Qwen2.5-VL Quantization Script Example](https://gitcode.com/Ascend/msit/blob/master/msmodelslim/example/multimodal_vlm/Qwen2.5-VL/README.md)

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`

### Installation

Run docker container:

:::::{tab-set}
:sync-group: install

::::{tab-item} single-NPU
:sync: single

```{code-block} bash
   :substitutions:
# Update the vllm-ascend image
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|

docker run --rm \
--name vllm-ascend \
--shm-size=1g \
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

# Set `max_split_size_mb` to reduce memory fragmentation and avoid out of memory
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256
```

::::
::::{tab-item} multi-NPU
:sync: multi

```{code-block} bash
   :substitutions:
# Update the vllm-ascend image
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
docker run --rm \
--shm-size=1g \
--net=host \
--name vllm-ascend \
--device /dev/davinci0 \
--device /dev/davinci1 \
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--device /dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /root/.cache:/root/.cache \
-v /data:/data \
-it $IMAGE bash
```

Setup environment variables:

```bash
# Load model from ModelScope to speed up download
export VLLM_USE_MODELSCOPE=True

# Set `max_split_size_mb` to reduce memory fragmentation and avoid out of memory
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256
```

::::
:::::

:::{note}
`max_split_size_mb` prevents the native allocator from splitting blocks larger than this size (in MB). This can reduce fragmentation and may allow some borderline workloads to complete without running out of memory. You can find more details [<u>here</u>](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/envref/envref_07_0061.html).
:::

## Deployment

### Offline Inference

:::::{tab-set}
:sync-group: install

::::{tab-item} Qwen3-VL-8B-Instruct
:sync: single

Run the following script to execute offline inference on single-NPU:

```bash
pip install qwen_vl_utils --extra-index-url https://download.pytorch.org/whl/cpu/
```

```python
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

MODEL_PATH = "Qwen/Qwen3-VL-8B-Instruct"

llm = LLM(
    model=MODEL_PATH,
    tensor_parallel_size=1,
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
The image displays a logo consisting of two main elements: a stylized geometric design and a pair of text elements.

1. **Geometric Design**: On the left side of the image, there is a blue geometric design that appears to be made up of interconnected shapes. These shapes resemble a network or a complex polygonal structure, possibly hinting at a technological or interconnected theme. The design is monochromatic and uses only blue as its color, which could be indicative of a specific brand or company.

2. **Text Elements**: To the right of the geometric design, there are two lines of text. The first line reads "TONGYI" in a sans-serif font, with the "YI" part possibly being capitalized. The second line reads "Qwen" in a similar sans-serif font, but in a smaller size.

The overall design is modern and minimalist, with a clear contrast between the geometric and textual elements. The use of blue for the geometric design could suggest themes of technology, connectivity, or innovation, which are common associations with the color blue in branding. The simplicity of the design makes it easily recognizable and memorable.
```

::::
::::{tab-item} Qwen2.5-VL-32B-Instruct
:sync: multi

Run the following script to execute offline inference on multi-NPU:

```bash
pip install qwen_vl_utils --extra-index-url https://download.pytorch.org/whl/cpu/
```

```python
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

MODEL_PATH = "Qwen/Qwen2.5-VL-32B-Instruct"

llm = LLM(
    model=MODEL_PATH,
    tensor_parallel_size=2,
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
### **1. Logo:**
- **Shape:** The logo on the left side of the image features a stylized geometric design composed of interconnected shapes that resemble interconnected hexagons or other polygonal forms.
- **Color:**
  - The logo is primarily composed of **blue lines**.
  - The design appears to be constructed from a network-like or abstract geometric pattern, suggesting themes of connection, integration, or technology.
  - The shapes are rendered using clean, crisp lines and angles, giving it a modern and tech-oriented aesthetic.
  - The geometric design appears interconnected and dynamic, indicating concepts like networking, intelligence, and complexity.
  - The lines are minimalist and symmetrical, with darker and lighter shades of blue, giving it a modular and abstract appearance.
  - The design has a **three-dimensional or engineered feel**, with overlapping elements, possibly symbolizing interconnectedness, scalability, or a tech-forward, futuristic look.
  - The shapes are simple yet intricate, implying advanced technology or AI-related innovation.
  - The use of blue gives it a futuristic and professional tone.

### **2. Text:**
- **TONGYI:**
  - Written in ** uppercase letters**.
  - The font is clean, sharp, and modern, aligning with themes of innovation and digital technology.
  - The interconnected hexagons or polygons give a sense of sophistication and precision.
  
- **Qwen:**
  - Below "TONGYI," the word "Qwen" is written in a sans-serif font, which is typical of brand logos aimed at conveying simplicity, reliability, and digital symmetry.
  - The font of "Qwen" is in a thinner, sleek typeface, aligning with branding associated with AI and technology.
  - The blue color reinforces the technological and digital theme, often used to convey trust, stability, and intelligence, aligning with the design elements, reinforcing the overall tech-savvy and modern appearance.
  - "Qwen" is placed below "TONGYI," further emphasizing the company or model's focus on precision and precision.
  - The font
```

::::
:::::

### Online Serving

:::::{tab-set}
:sync-group: install

::::{tab-item} Qwen3-VL-8B-Instruct
:sync: single

Run docker container to start the vLLM server on single-NPU:

```{code-block} bash
   :substitutions:
vllm serve Qwen/Qwen3-VL-8B-Instruct \
--tensor-parallel-size 1 \
--dtype bfloat16 \
--max_model_len 16384 \
--max-num-batched-tokens 16384
```

:::{note}
Add `--max_model_len` option to avoid ValueError that the Qwen3-VL-8B-Instruct model's max seq len (256000) is larger than the maximum number of tokens that can be stored in KV cache. This will differ with different NPU series base on the HBM size. Please modify the value according to a suitable value for your NPU series.
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
    "model": "Qwen/Qwen3-VL-8B-Instruct",
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
{"id":"chatcmpl-f04fb20e79bb40b39b8ed7fdf5bd613a","object":"chat.completion","created":1741749149,"model":"Qwen/Qwen3-VL-8B-Instruct","choices":[{"index":0,"message":{"role":"assistant","reasoning_content":null,"content":"The text in the illustration reads \"TONGYI Qwen.\"","tool_calls":[]},"logprobs":null,"finish_reason":"stop","stop_reason":null}],"usage":{"prompt_tokens":74,"total_tokens":89,"completion_tokens":15,"prompt_tokens_details":null},"prompt_logprobs":null}
```

Logs of the vllm server:

```bash
INFO 03-12 11:16:50 logger.py:39] Received request chatcmpl-92148a41eca64b6d82d3d7cfa5723aeb: prompt: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\nWhat is the text in the illustrate?<|im_end|>\n<|im_start|>assistant\n', params: SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=1.0, top_p=1.0, top_k=-1, min_p=0.0, seed=None, stop=[], stop_token_ids=[], bad_words=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=16353, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None, guided_decoding=None), prompt_token_ids: None, lora_request: None, prompt_adapter_request: None.
INFO 03-12 11:16:50 engine.py:280] Added request chatcmpl-92148a41eca64b6d82d3d7cfa5723aeb.
INFO:     127.0.0.1:54004 - "POST /v1/chat/completions HTTP/1.1" 200 OK
```

::::
::::{tab-item} Qwen2.5-VL-32B-Instruct
:sync: multi

Run docker container to start the vLLM server on multi-NPU:

```shell
#!/bin/sh
# if os is Ubuntu
apt update
apt install libjemalloc2 
# if os is openEuler
yum update
yum install jemalloc
# Add the LD_PRELOAD environment variable
if [ -f /usr/lib/aarch64-linux-gnu/libjemalloc.so.2 ]; then
    # On Ubuntu, first install with `apt install libjemalloc2`
    export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
elif [ -f /usr/lib64/libjemalloc.so.2 ]; then
    # On openEuler, first install with `yum install jemalloc`
    export LD_PRELOAD=/usr/lib64/libjemalloc.so.2:$LD_PRELOAD
fi
# Enable the AIVector core to directly schedule ROCE communication
export HCCL_OP_EXPANSION_MODE="AIV"
# Set vLLM to Engine V1
export VLLM_USE_V1=1

vllm serve Qwen/Qwen2.5-VL-32B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --async-scheduling \
    --tensor-parallel-size 2 \
    --max-model-len 30000 \
    --max-num-batched-tokens 50000 \
    --max-num-seqs 30 \
    --no-enable-prefix-caching \
    --trust-remote-code \
    --dtype bfloat16

```

:::{note}
Add `--max_model_len` option to avoid ValueError that the Qwen2.5-VL-32B-Instruct model's max seq len (128000) is larger than the maximum number of tokens that can be stored in KV cache. This will differ with different NPU series base on the HBM size. Please modify the value according to a suitable value for your NPU series.
:::

If your service start successfully, you can see the info shown below:

```bash
INFO:     Started server process [14431]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

Once your server is started, you can query the model with input prompts:

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
    "model": "Qwen/Qwen2.5-VL-32B-Instruct",
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
{"id":"chatcmpl-c07088bf992a4b77a89d79480122a483","object":"chat.completion","created":1764905884,"model":"Qwen/Qwen2.5-VL-32B-Instruct","choices":[{"index":0,"message":{"role":"assistant","content":"The text in the illustration is:\n\n**TONGYI Qwen**","refusal":null,"annotations":null,"audio":null,"function_call":null,"tool_calls":[],"reasoning":null,"reasoning_content":null},"logprobs":null,"finish_reason":"stop","stop_reason":null,"token_ids":null}],"service_tier":null,"system_fingerprint":null,"usage":{"prompt_tokens":73,"total_tokens":89,"completion_tokens":16,"prompt_tokens_details":null},"prompt_logprobs":null,"prompt_token_ids":null,"kv_transfer_params":null}
```

Logs of the vllm server:

```bash
INFO 03-12 11:16:50 logger.py:39] Received request chatcmpl-92148a41eca64b6d82d3d7cfa5723aeb: prompt: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\nWhat is the text in the illustrate?<|im_end|>\n<|im_start|>assistant\n', params: SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=1.0, top_p=1.0, top_k=-1, min_p=0.0, seed=None, stop=[], stop_token_ids=[], bad_words=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=16353, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None, guided_decoding=None), prompt_token_ids: None, lora_request: None, prompt_adapter_request: None.
INFO 03-12 11:16:50 engine.py:280] Added request chatcmpl-92148a41eca64b6d82d3d7cfa5723aeb.
INFO:     127.0.0.1:54004 - "POST /v1/chat/completions HTTP/1.1" 200 OK
```

::::
:::::

## Accuracy Evaluation

### Using Language Model Evaluation Harness

The accuracy of some models is already within our CI monitoring scope, including:
- `Qwen2.5-VL-7B-Instruct`
- `Qwen3-VL-8B-Instruct`

You can refer to the [monitoring configuration](https://github.com/vllm-project/vllm-ascend/blob/main/.github/workflows/vllm_ascend_test_nightly_a2.yaml).

:::::{tab-set}
:sync-group: install

::::{tab-item} Qwen3-VL-8B-Instruct
:sync: single

As an example, take the `mmmu_val` dataset as a test dataset, and run accuracy evaluation of `Qwen3-VL-8B-Instruct` in offline mode.

1. Refer to [Using lm_eval](../developer_guide/evaluation/using_lm_eval.md) for `lm_eval` installation.

2. Run `lm_eval` to execute the accuracy evaluation.

```shell
lm_eval \
    --model vllm-vlm \
    --model_args pretrained=Qwen/Qwen3-VL-8B-Instruct,max_model_len=8192,gpu_memory_utilization=0.7,tensor_parallel_size=1 \
    --tasks mmmu_val \
    --batch_size 32 \
    --apply_chat_template \
    --trust_remote_code \
    --output_path ./results
```

3. After execution, you can get the result, here is the result of `Qwen3-VL-8B-Instruct` in `vllm-ascend:0.11.0rc2` for reference only.

|  Tasks  |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
|---------|------:|------|-----:|------|---|-----:|---|-----:|
|mmmu_val |      0|none  |      |acc   |↑  |0.5389|±  |0.0159|

::::
::::{tab-item} Qwen2.5-VL-32B-Instruct
:sync: multi

As an example, take the `mmmu_val` dataset as a test dataset, and run accuracy evaluation of `Qwen2.5-VL-32B-Instruct` in offline mode.

1. Refer to [Using lm_eval](../developer_guide/evaluation/using_lm_eval.md) for `lm_eval` installation.

2. Run `lm_eval` to execute the accuracy evaluation.

```shell
lm_eval \
    --model vllm-vlm \
    --model_args pretrained=Qwen/Qwen2.5-VL-32B-Instruct,max_model_len=8192,tensor_parallel_size=2 \
    --tasks mmmu_val \
    --batch_size 32 \
    --apply_chat_template \
    --trust_remote_code \
    --output_path ./results
```

3. After execution, you can get the result, here is the result of `Qwen3-VL-8B-Instruct` in `vllm-ascend:0.11.0rc2` for reference only.

|  Tasks  |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
|---------|------:|------|-----:|------|---|-----:|---|-----:|
|mmmu_val |      0|none  |      |acc   |↑  |0.5839|±  |0.0159|

::::
:::::

## Performance

### Using vLLM Benchmark

The performance evaluation must be conducted in an online mode.

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/contributing/benchmarks.html) for more details.

There are three `vllm bench` subcommand:
- `latency`: Benchmark the latency of a single batch of requests.
- `serve`: Benchmark the online serving throughput.
- `throughput`: Benchmark offline inference throughput.

Take the `serve` as an example. Run the code as follows.

:::::{tab-set}
:sync-group: install

::::{tab-item} Qwen3-VL-8B-Instruct
:sync: single

```shell
vllm bench serve --model Qwen/Qwen3-VL-8B-Instruct  --dataset-name random --random-input 200 --num-prompt 200 --request-rate 1 --save-result --result-dir ./
```

::::
::::{tab-item} Qwen2.5-VL-32B-Instruct
:sync: multi

```shell
vllm bench serve --model Qwen/Qwen2.5-VL-32B-Instruct  --dataset-name random --random-input 200 --num-prompt 200 --request-rate 1 --save-result --result-dir ./
```

::::
:::::

After about several minutes, you can get the performance evaluation result.
