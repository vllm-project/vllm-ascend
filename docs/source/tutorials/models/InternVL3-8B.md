# InternVL3-8B

## Model Overview

InternVL3-8B is a powerful vision-language model from the InternVL series, designed for multimodal understanding tasks. It combines a vision encoder with a language model to process both images and text, enabling capabilities such as image captioning, visual question answering, and multimodal dialogue.

Key features:
- 8B parameter vision-language model
- Supports high-resolution image understanding
- Efficient multimodal processing
- Compatible with OpenAI API format

This tutorial demonstrates how to deploy and use InternVL3-8B with vLLM-Ascend.

## Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [feature guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## Quick Start

### Basic Usage

Here's a simple example to get started with InternVL3-8B for offline inference:

```python
from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset

# Initialize the model
llm = LLM(
    model="OpenGVLab/InternVL3-8B",
    trust_remote_code=True,
    max_model_len=4096,
    limit_mm_per_prompt={"image": 1}
)

# Prepare sampling parameters
sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=512
)

# Create a prompt with an image
image = ImageAsset("cherry_blossom").pil_image
prompt = "USER: <image>\nDescribe this image in detail.\nASSISTANT:"

# Generate response
outputs = llm.generate(
    {
        "prompt": prompt,
        "multi_modal_data": {"image": image}
    },
    sampling_params=sampling_params
)

print(outputs[0].outputs[0].text)
```

### OpenAI-Compatible Server

Deploy InternVL3-8B as an OpenAI-compatible API server:

```bash
#!/bin/bash
export ASCEND_RT_VISIBLE_DEVICES=0
export MODEL_PATH="OpenGVLab/InternVL3-8B"

vllm serve ${MODEL_PATH} \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name internvl3-8b \
    --trust-remote-code \
    --max-model-len 4096 \
    --limit-mm-per-prompt '{"image": 1}'
```

Then query the server using curl:

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "internvl3-8b",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
                ]
            }
        ],
        "max_tokens": 512
    }'
```

Or use the OpenAI Python client:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

response = client.chat.completions.create(
    model="internvl3-8b",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image"},
                {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
            ]
        }
    ],
    max_tokens=512
)

print(response.choices[0].message.content)
```

## Configuration

### Key Parameters

- `max_model_len`: Maximum sequence length (default: 4096). Adjust based on your use case and available memory.
- `limit_mm_per_prompt`: Limits the number of images per prompt. For InternVL3-8B, typically set to `{"image": 1}` for single-image tasks.
- `trust_remote_code`: Required for InternVL models to load custom model code.
- `tensor_parallel_size`: Number of NPUs for tensor parallelism (for multi-NPU deployment).

### Hardware Requirements

**Single NPU Deployment:**
- Minimum: 1x Atlas 910B (32GB) or 1x Atlas 800I A2 (64GB)
- Recommended: 1x Atlas 800I A2 (64GB) for better performance

**Multi-NPU Deployment:**
- For larger batch sizes or longer sequences: 2-4 NPUs with tensor parallelism

### Environment Setup

Before running InternVL3-8B, set up the environment:

```bash
# Use ModelScope for faster model downloads (optional)
export VLLM_USE_MODELSCOPE=True

# Configure memory allocation to reduce fragmentation
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256

# Specify visible NPU devices
export ASCEND_RT_VISIBLE_DEVICES=0
```

## Performance Tips

1. **Optimize Batch Size**: Start with smaller batch sizes and gradually increase to find the optimal throughput for your hardware.

2. **Adjust Max Model Length**: Use the minimum `max_model_len` needed for your use case to save memory.

3. **Image Preprocessing**: Ensure images are appropriately sized. Very large images may impact performance.

4. **Use Tensor Parallelism**: For higher throughput, deploy across multiple NPUs:

```bash
vllm serve OpenGVLab/InternVL3-8B \
    --tensor-parallel-size 2 \
    --trust-remote-code \
    --max-model-len 4096
```

5. **Enable KV Cache Optimization**: vLLM automatically optimizes KV cache usage, but you can tune it:

```bash
vllm serve OpenGVLab/InternVL3-8B \
    --trust-remote-code \
    --gpu-memory-utilization 0.9
```

## Known Limitations

- **Image Count**: Currently optimized for single-image inputs. Multi-image support may require additional configuration.
- **Context Length**: Very long contexts (>8K tokens) may require careful memory management.
- **Video Input**: Video understanding is not currently supported in this version.

## Troubleshooting

### Out of Memory Errors

If you encounter OOM errors:

1. Reduce `max_model_len`:
```bash
vllm serve OpenGVLab/InternVL3-8B --max-model-len 2048 --trust-remote-code
```

2. Set memory allocation configuration:
```bash
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:128
```

3. Use tensor parallelism to distribute memory across NPUs.

### Slow Image Processing

If image processing is slow:

1. Ensure images are reasonably sized (e.g., max 1024x1024)
2. Use batch processing when possible
3. Check NPU utilization with `npu-smi info`

### Model Loading Issues

If the model fails to load:

1. Verify `trust_remote_code=True` is set
2. Check model weights are correctly downloaded
3. Ensure sufficient disk space in cache directory (`~/.cache/huggingface/`)

### Connection Errors

If you cannot connect to the server:

1. Verify the server is running: `curl http://localhost:8000/health`
2. Check firewall settings
3. Ensure the correct host and port are specified

## References

- [InternVL GitHub Repository](https://github.com/OpenGVLab/InternVL)
- [InternVL3 Model Card](https://huggingface.co/OpenGVLab/InternVL3-8B)
- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM-Ascend User Guide](../../user_guide/index.md)
