# Qwen-VL-Dense(Qwen2.5VL-3B/7B, Qwen3-VL-2B/4B/8B/32B)

## Introduction

The Qwen-VL(Vision-Language)series from Alibaba Cloud comprises a family of powerful Large Vision-Language Models (LVLMs) designed for comprehensive multimodal understanding. They accept images, text, and bounding boxes as input, and output text and detection boxes, enabling advanced functions like image detection, multi-modal dialogue, and multi-image reasoning.

This document will show the main verification steps of the model, including supported features, feature configuration, environment preparation, NPU deployment, accuracy and performance evaluation.

This tutorial uses the vLLM-Ascend `v0.11.0rc3-a3` version for demonstration, showcasing the `Qwen3-VL-8B-Instruct` model as an example for single NPU deployment and the `Qwen2.5-VL-32B-Instruct` model as an example for multi-NPU deployment.

## Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [feature guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

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

:::::{tab-set}
:sync-group: install

::::{tab-item} single-NPU
:sync: single

Run docker container:

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

::::
::::{tab-item} multi-NPU
:sync: multi

Run docker container:

```{code-block} bash
   :substitutions:
# Update the vllm-ascend image
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|

docker run --rm \
--name vllm-ascend \
--shm-size=1g \
--net=host \
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

::::
:::::

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
    max_model_len=16384,
    limit_mm_per_prompt={"image": 10},
)

sampling_params = SamplingParams(
    max_completion_tokens=512
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

```shell
**Visual Components:**

1.  **Abstract Geometric Icon (Left Side):**
    *   The logo features a stylized, abstract icon on the left.
    *   It is composed of interconnected lines and angular shapes, forming a complex, hexagonal-like structure.
    *   The icon is rendered in a solid, thin blue line, giving it a modern, technological, and clean appearance.

2.  **Text (Right Side):**
    *   To the right of the icon, the name "TONGYI Qwen" is written.
    *   **"TONGYI"** is written in uppercase letters in a bold, modern sans-serif font. The color is a medium blue, matching the icon's color.
    *   **"Qwen"** is written below "TONGYI" in a slightly larger, bold, sans-serif font. The color of "Qwen" is a dark gray or black, creating a strong contrast with the blue text above it.
    *   The text is aligned and spaced neatly, with "Qwen" appearing slightly larger and bolder than "TONGYI," emphasizing the proper noun.

**Overall Design and Aesthetics:**

*   The logo has a clean, contemporary, and professional feel, suitable for a technology and AI product.
*   The use of blue conveys trust, innovation, and intelligence, while the dark gray adds stability and clarity.
*   The overall layout is balanced and symmetrical, with the icon and text arranged horizontally for easy recognition and memorability.
*   The design effectively communicates the product's high-tech nature while remaining brand-identifiable and straightforward.

The logo is designed to be easily recognizable across various media and scales, from digital screens to printed materials.
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
    max_completion_tokens=512
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

```shell
The image displays a logo and text related to the Qwen model, which is an artificial intelligence (AI) language model developed by Alibaba Cloud. Here is a detailed description of the elements in the image:

### **1. Logo:**
- The logo on the left side of the image consists of a stylized, abstract geometric design.
- The logo is primarily composed of interconnected lines and shapes that resemble a combination of arrows, lines, and geometric forms.
- The lines are arranged in a triangular pattern, giving it a dynamic and modern appearance.
- The lines are rendered in a dark blue color, and they form a three-dimensional, arrow-like structure. This conveys a sense of movement, forward momentum, or direction, which is often symbolic of progress and integration.
- The design appears to be complex yet minimalistic, with clean and sharp lines.
- The triangular and square-like structure suggests precision, connectivity, and innovation, which are often associated with technology and advanced systems.
- This abstract, arrow-like design implies a sense of flow, direction, and connectivity, which aligns with themes of progress and technological advancement.

### **2. Text:**
- **"TONGYI" (on the top right side):
  - The text is in dark blue, which is a color often associated with technology, stability, and trustworthiness.
  - The name "Tongyi" is written in a bold, sans-serif font, giving it a modern and professional look.
- **"Qwen" (below "Tongyi"):
  - The font for "Qwen" is in a bold, uppercase format.
  - The style
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
{"id":"chatcmpl-d3270d4a16cb4b98936f71ee3016451f","object":"chat.completion","created":1764924127,"model":"Qwen/Qwen3-VL-8B-Instruct","choices":[{"index":0,"message":{"role":"assistant","content":"The text in the illustration is: **TONGYI Qwen**","refusal":null,"annotations":null,"audio":null,"function_call":null,"tool_calls":[],"reasoning_content":null},"logprobs":null,"finish_reason":"stop","stop_reason":null,"token_ids":null}],"service_tier":null,"system_fingerprint":null,"usage":{"prompt_tokens":107,"total_tokens":123,"completion_tokens":16,"prompt_tokens_details":null},"prompt_logprobs":null,"prompt_token_ids":null,"kv_transfer_params":null}
```

Logs of the vllm server:

```bash
INFO 12-05 08:42:07 [chat_utils.py:560] Detected the chat template content format to be 'openai'. You can set `--chat-template-content-format` to override this.
Downloading Model from https://www.modelscope.cn to directory: /root/.cache/modelscope/hub/models/Qwen/Qwen3-VL-8B-Instruct
INFO 12-05 08:42:11 [acl_graph.py:187] Replaying aclgraph
INFO:     127.0.0.1:60988 - "POST /v1/chat/completions HTTP/1.1" 200 OK
INFO 12-05 08:42:13 [loggers.py:127] Engine 000: Avg prompt throughput: 10.7 tokens/s, Avg generation throughput: 1.6 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
INFO 12-05 08:42:23 [loggers.py:127] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
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
Add `--max_model_len` option to avoid ValueError that the Qwen2.5-VL-32B-Instruct model's max_model_len (128000) is larger than the maximum number of tokens that can be stored in KV cache. This will differ with different NPU series base on the HBM size. Please modify the value according to a suitable value for your NPU series.
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
The image processor of type `Qwen2VLImageProcessor` is now loaded as a fast processor by default, even if the model checkpoint was saved with a slow processor. This is a breaking change and may produce slightly different outputs. To continue using the slow processor, instantiate this class with `use_fast=False`. Note that this behavior will be extended to all models in a future release.
INFO 12-05 08:50:57 [chat_utils.py:560] Detected the chat template content format to be 'openai'. You can set `--chat-template-content-format` to override this.
Downloading Model from https://www.modelscope.cn to directory: /root/.cache/modelscope/hub/models/Qwen/Qwen2.5-VL-32B-Instruct
2025-12-05 08:50:58,913 - modelscope - INFO - Target directory already exists, skipping creation.
INFO 12-05 08:51:00 [acl_graph.py:187] Replaying aclgraph
INFO:     127.0.0.1:50720 - "POST /v1/chat/completions HTTP/1.1" 200 OK
INFO 12-05 08:51:10 [loggers.py:127] Engine 000: Avg prompt throughput: 7.3 tokens/s, Avg generation throughput: 1.6 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
INFO 12-05 08:51:20 [loggers.py:127] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
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

1. Refer to [Using lm_eval](../../developer_guide/evaluation/using_lm_eval.md) for more details on `lm_eval` installation.

```shell
pip install lm_eval
```

2. Run `lm_eval` to execute the accuracy evaluation.

```shell
lm_eval \
    --model vllm-vlm \
    --model_args pretrained=Qwen/Qwen3-VL-8B-Instruct,max_model_len=8192,gpu_memory_utilization=0.7 \
    --tasks mmmu_val \
    --batch_size 32 \
    --apply_chat_template \
    --trust_remote_code \
    --output_path ./results
```

3. After execution, you can get the result, here is the result of `Qwen3-VL-8B-Instruct` in `vllm-ascend:0.11.0rc3` for reference only.

|  Tasks  |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
|---------|------:|------|-----:|------|---|-----:|---|-----:|
|mmmu_val |      0|none  |      |acc   |↑  |0.5389|±  |0.0159|

::::
::::{tab-item} Qwen2.5-VL-32B-Instruct
:sync: multi

As an example, take the `mmmu_val` dataset as a test dataset, and run accuracy evaluation of `Qwen2.5-VL-32B-Instruct` in offline mode.

1. Refer to [Using lm_eval](../../developer_guide/evaluation/using_lm_eval.md) for more details on `lm_eval` installation.

```shell
pip install lm_eval
```

2. Run `lm_eval` to execute the accuracy evaluation.

```shell
lm_eval \
    --model vllm-vlm \
    --model_args pretrained=Qwen/Qwen2.5-VL-32B-Instruct,max_model_len=8192,tensor_parallel_size=2 \
    --tasks mmmu_val \
    --apply_chat_template \
    --trust_remote_code \
    --output_path ./results
```

3. After execution, you can get the result, here is the result of `Qwen2.5-VL-32B-Instruct` in `vllm-ascend:0.11.0rc3` for reference only.

|  Tasks  |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
|---------|------:|------|-----:|------|---|-----:|---|-----:|
|mmmu_val |      0|none  |      |acc   |↑  |0.5744|±  |0.0158|

::::
:::::

## Performance

### Using vLLM Benchmark

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/contributing/benchmarks.html) for more details.

There are three `vllm bench` subcommands:

- `latency`: Benchmark the latency of a single batch of requests.
- `serve`: Benchmark the online serving throughput.
- `throughput`: Benchmark offline inference throughput.

The performance evaluation must be conducted in an online mode. Take the `serve` as an example. Run the code as follows.

:::::{tab-set}
:sync-group: install

::::{tab-item} Qwen3-VL-8B-Instruct
:sync: single

```shell
vllm bench serve --model Qwen/Qwen3-VL-8B-Instruct  --dataset-name random --random-input 200 --num-prompts 200 --request-rate 1 --save-result --result-dir ./
```

::::
::::{tab-item} Qwen2.5-VL-32B-Instruct
:sync: multi

```shell
vllm bench serve --model Qwen/Qwen2.5-VL-32B-Instruct  --dataset-name random --random-input 200 --num-prompts 200 --request-rate 1 --save-result --result-dir ./
```

::::
:::::

After about several minutes, you can get the performance evaluation result.
