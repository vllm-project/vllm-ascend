# Single NPU (InternVL Series)

## Overview

InternVL is a series of open-source multimodal large language models developed by OpenGVLab. The InternVL series support matrix is as follows:

| Model | Supported | Recommended Configuration |
|-------|-----------|---------------------------|
| InternVL2-8B | ✅ | max_model_len=8192, dtype=bfloat16 |
| InternVL2.5-8B | ✅ | max_model_len=8192, dtype=bfloat16 |
| InternVL3-8B | ✅ | max_model_len=8192, dtype=bfloat16 |

## Run vllm-ascend on Single NPU

### Offline Inference on Single NPU

Run the docker container:

```{code-block} bash
   :substitutions:
# Update the vllm-ascend image
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
docker run --rm \
--name vllm-ascend-internvl \
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

# Set `max_split_size_mb` to reduce memory fragmentation and avoid out-of-memory errors
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256
```

:::{note}
`max_split_size_mb` prevents the native allocator from splitting blocks larger than this size (in MB). This can reduce fragmentation and may allow some borderline workloads to complete without running out of memory. You can find more details [<u>here</u>](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/envref/envref_07_0061.html).
:::


Run the following script to execute offline inference on a single NPU:

#### Single Image Inference

```python
from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset

# Model configuration
MODEL_PATH = "OpenGVLab/InternVL3-8B"

# Initialize the LLM
llm = LLM(
    model=MODEL_PATH,
    trust_remote_code=True,
    max_model_len=8192,
    limit_mm_per_prompt={"image": 4},
    dtype="bfloat16",
)

# Sampling parameters
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512,
    stop_token_ids=[2, 92542],  # 2 = </s>, 92542 = <|im_end|>
)

# Load test image
image = ImageAsset("cherry_blossom").pil_image

# InternVL uses chat template format
# Format: <|im_start|>user\n<image>\nQUESTION<|im_end|>\n<|im_start|>assistant\n
prompt = "<|im_start|>user\n<image>\nDescribe this image in detail.<|im_end|>\n<|im_start|>assistant\n"

# Generate output using multi_modal_data
outputs = llm.generate(
    {
        "prompt": prompt,
        "multi_modal_data": {"image": image},
    },
    sampling_params=sampling_params,
)

# Print the result
for output in outputs:
    print(output.outputs[0].text)
```

Then, the model outputs:

```
The image features a picturesque scene with cherry blossoms in full bloom, creating a natural frame around a tall tower in the background. The cherry blossom branches are filled with clusters of delicate pink flowers, extending across the foreground. The blossoms are vibrant, with soft hues of pink against the clear blue sky. In the background, the Tokyo Skytree is prominently visible, towering above the cherry blossoms. Its sleek, modern design contrasts beautifully with the organic shapes and colors of the flowers. The image captures the essence of spring, blending nature's beauty with a prominent urban landmark.
```

#### Multiple Images in Single Prompt

InternVL supports multiple images in a single prompt. Here's how to use this feature:

```python
from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset

MODEL_PATH = "OpenGVLab/InternVL3-8B"

llm = LLM(
    model=MODEL_PATH,
    max_model_len=8192,
    trust_remote_code=True,
    limit_mm_per_prompt={"image": 4},  # Allow up to 4 images per prompt
    dtype="bfloat16",
)

sampling_params = SamplingParams(max_tokens=256)

# Load multiple images
image1 = ImageAsset("cherry_blossom").pil_image.convert("RGB")
image2 = ImageAsset("stop_sign").pil_image.convert("RGB")

# Create prompt with multiple <image> tags
prompt = (
    "<|im_start|>user\n"
    "<image>\n"
    "Describe the first image.\n"
    "<image>\n"
    "Describe the second image.\n"
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
)
outputs = llm.generate(
    {
        "prompt": prompt,
        "multi_modal_data": {
            "image": [image1, image2]
        },
    },
    sampling_params=sampling_params,
)

print(outputs[0].outputs[0].text)
```

The model outputs:

```
The first image showcases a beautiful scene where cherry blossoms are in full bloom, with a dense cluster of pink flowers in the foreground. These flowers frame the view, creating a picturesque, natural border. In the background, a prominent tower rises against a clear blue sky. The tower has a sleek, modern design with a metallic structure, and its upper part is cylindrical with observation decks visible. The branches of the cherry blossom tree, adorned with numerous blossoms, intersect and partially obscure the tower, adding depth to the photograph. The combination of natural beauty and architectural marvel creates a striking contrast that highlights both the vibrant springtime flora and the impressive man-made structure.

The second image depicts a vibrant street scene in what appears to be a Chinatown district, identifiable by the traditional Chinese architectural elements. At the center of the image is a red and gold Chinese archway adorned with intricate designs and Chinese characters. Beneath the arch, there are two white stone lion statues placed symmetrically on either side, which are common guardians in Chinese culture. In the foreground, there is a red octagonal stop sign mounted on a post, juxtaposing the traditional elements with modern traffic regulations. A black car is captured in motion, driving through the street, while various shop signs are visible in the backgroun
```

#### Batch Inference

Process multiple prompts efficiently with batch inference:

```python
from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset

MODEL_PATH = "OpenGVLab/InternVL3-8B"

llm = LLM(
    model=MODEL_PATH,
    max_model_len=8192,
    trust_remote_code=True,
    limit_mm_per_prompt={"image": 4},
    dtype="bfloat16",
)

sampling_params = SamplingParams(max_tokens=128)

# Load test image
image = ImageAsset("cherry_blossom").pil_image.convert("RGB")

# Create multiple prompts
questions = [
    "What is the content of this image?",
    "Describe this image in detail.",
    "What colors are prominent in this image?",
]

prompts = [
    f"<|im_start|>user\n<image>\n{q}<|im_end|>\n<|im_start|>assistant\n"
    for q in questions
]

# Same image for all prompts
images = [image] * len(prompts)

# Batch generate with one image
outputs = llm.generate(
    {
        "prompt": prompts,
        "multi_modal_data": {"image": image},
    },
    sampling_params=sampling_params,
)


# Print results
for i, output in enumerate(outputs):
    print(f"\nQuestion {i+1}: {questions[i]}")
    print(f"Answer: {output.outputs[0].text}")
```

The model outputs:

```
Question 1: What is the content of this image?
Answer: The image features a beautiful pink cherry blossom tree in full bloom, with flowers in the foreground. Behind the blossoms, part of a tall tower is visible. This tower resembles Tokyo Skytree, a communications and observation tower in Tokyo, Japan. The background is a clear blue sky.

Question 2: Describe this image in detail.
Answer: The image depicts a view of a tall, modern tower seen through a foreground of blooming cherry blossoms. The sky is a clear, vibrant blue, providing a striking contrast to the soft pink flowers. The cherry blossoms are in full bloom, with delicate petals forming clusters that partially obscure the tower. The tower itself is sleek and metallic, with a cylindrical observation deck and antenna. Its structure includes intricate frameworks, typical of contemporary architecture. The photo is taken from a low angle, emphasizing the height of the tower while the cherry blossoms frame it beautifully, suggesting a picturesque scene reminiscent of springtime in Japan.

Question 3: What colors are prominent in this image?
Answer: The image prominently features shades of pink from the cherry blossoms and a bright blue sky. There are some white and grey tones in the tower.
```

### Online Serving on Single NPU

Run a docker container to start the vLLM server on a single NPU:

```{code-block} bash
   :substitutions:

# Update the vllm-ascend image
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
docker run --rm \
--name vllm-ascend-internvl \
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
-it $IMAGE \
vllm serve OpenGVLab/InternVL3-8B \
--dtype bfloat16 \
--max-model-len 8192 \
--limit-mm-per-prompt '{"image":4}' \
--trust-remote-code
```

:::{note}
The `--max-model-len` option prevents a ValueError when the model's max sequence length exceeds available KV cache. Adjust this value based on your NPU's HBM size.
:::

If your service starts successfully, you should see output similar to the following:

```bash
INFO:     Started server process [2736]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

#### Query the Server

Once your server is started, you can query the model with input prompts:

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
    "model": "OpenGVLab/InternVL3-8B",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://modelscope.oss-cn-beijing.aliyuncs.com/resource/qwen.png"
                    }
                },
                {
                    "type": "text",
                    "text": "What is in this image?"
                }
            ]
        }
    ],
    "max_tokens": 128
    }'
```

Then, you can get the output like:

```json
{"id":"chatcmpl-975f938a783944de8917e5b0ffc10108","object":"chat.completion","created":1761203983,"model":"OpenGVLab/InternVL3-8B","choices":[{"index":0,"message":{"role":"assistant","content":"The image shows the logo for TONGYI Qwen. The logo features a stylized design resembling interconnected arrows pointing inward, along with the text \"TONGYI\" in purple and \"Qwen\" in gray.","refusal":null,"annotations":null,"audio":null,"function_call":null,"tool_calls":[],"reasoning_content":null},"logprobs":null,"finish_reason":"stop","stop_reason":null,"token_ids":null}],"service_tier":null,"system_fingerprint":null,"usage":{"prompt_tokens":2870,"total_tokens":2917,"completion_tokens":47,"prompt_tokens_details":null},"prompt_logprobs":null,"prompt_token_ids":null,"kv_transfer_params":null}
```