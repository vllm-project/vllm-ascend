# InternVL3-78B

## Model Overview

InternVL3-78B is a large-scale vision-language model from the InternVL series, designed for advanced multimodal understanding tasks. With 78 billion parameters, it offers superior performance in complex visual reasoning, detailed image analysis, and sophisticated multimodal dialogue compared to smaller variants.

Key features:
- 78B parameter vision-language model
- Enhanced reasoning and understanding capabilities
- Supports high-resolution image understanding
- Requires multi-NPU deployment with tensor parallelism
- Compatible with OpenAI API format

This tutorial demonstrates how to deploy and use InternVL3-78B with vLLM-Ascend.

## Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [feature guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## Quick Start

### Basic Usage

Here's a simple example to get started with InternVL3-78B for offline inference:

```python
from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset

# Initialize the model with tensor parallelism
llm = LLM(
    model="OpenGVLab/InternVL3-78B",
    trust_remote_code=True,
    max_model_len=8192,
    tensor_parallel_size=4,  # Use 4 NPUs
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

Deploy InternVL3-78B as an OpenAI-compatible API server:

```bash
#!/bin/bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export MODEL_PATH="OpenGVLab/InternVL3-78B"

vllm serve ${MODEL_PATH} \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name internvl3-78b \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --max-model-len 8192 \
    --limit-mm-per-prompt '{"image": 1}' \
    --enforce-eager
```

Then query the server using curl:

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "internvl3-78b",
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
    model="internvl3-78b",
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

- `tensor_parallel_size`: Number of NPUs for tensor parallelism. For InternVL3-78B, recommended values are 4 or 8.
- `max_model_len`: Maximum sequence length (default: 8192). Adjust based on your use case and available memory.
- `limit_mm_per_prompt`: Limits the number of images per prompt. For InternVL3-78B, typically set to `{"image": 1}` for single-image tasks.
- `trust_remote_code`: Required for InternVL models to load custom model code.
- `enforce_eager`: Recommended to avoid CANN compilation issues with kernel_meta directory.

### Hardware Requirements

**Multi-NPU Deployment (Required):**
- Minimum: 4x Atlas 910B (32GB each) or 4x Atlas 800I A2 (64GB each)
- Recommended: 8x Atlas 800I A2 (64GB each) for better performance and longer sequences
- Total memory requirement: ~150GB for model weights plus inference overhead

**Note:** InternVL3-78B cannot run on a single NPU due to its size. Multi-NPU deployment with tensor parallelism is mandatory.

### Environment Setup

Before running InternVL3-78B, set up the environment:

```bash
# Use ModelScope for faster model downloads (optional)
export VLLM_USE_MODELSCOPE=True

# Configure memory allocation to reduce fragmentation
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256

# Specify visible NPU devices (for 4-NPU setup)
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

# Set kernel_meta directory to avoid permission issues
export KERNEL_META_TEMP_DIR=~/kernel_meta
mkdir -p ~/kernel_meta
```

## Performance Tips

1. **Optimize Tensor Parallelism**: Use 4 or 8 NPUs depending on availability. More NPUs can improve throughput:

```bash
vllm serve OpenGVLab/InternVL3-78B \
    --tensor-parallel-size 8 \
    --trust-remote-code \
    --max-model-len 8192 \
    --enforce-eager
```

2. **Adjust Max Model Length**: Use the minimum `max_model_len` needed for your use case to save memory:

```bash
# For shorter contexts
vllm serve OpenGVLab/InternVL3-78B \
    --tensor-parallel-size 4 \
    --max-model-len 4096 \
    --trust-remote-code \
    --enforce-eager
```

3. **Batch Size Tuning**: Start with smaller batch sizes (e.g., `--max-num-seqs 2`) and gradually increase:

```bash
vllm serve OpenGVLab/InternVL3-78B \
    --tensor-parallel-size 4 \
    --max-num-seqs 2 \
    --trust-remote-code \
    --enforce-eager
```

4. **Memory Utilization**: Adjust GPU memory utilization based on available memory:

```bash
vllm serve OpenGVLab/InternVL3-78B \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.85 \
    --trust-remote-code \
    --enforce-eager
```

5. **Image Preprocessing**: Ensure images are appropriately sized. Very large images may impact performance.

## Known Limitations

- **Multi-NPU Required**: Cannot run on a single NPU due to model size.
- **Compilation Issues**: May encounter kernel_meta permission errors. Use `--enforce-eager` to bypass.
- **Image Count**: Currently optimized for single-image inputs. Multi-image support may require additional configuration.
- **Context Length**: Very long contexts (>16K tokens) may require careful memory management across NPUs.
- **Inference Speed**: Slower than smaller models (InternVL3-8B) due to increased parameter count.

## Troubleshooting

### Out of Memory Errors

If you encounter OOM errors:

1. Reduce `max_model_len`:
```bash
vllm serve OpenGVLab/InternVL3-78B \
    --tensor-parallel-size 4 \
    --max-model-len 4096 \
    --trust-remote-code \
    --enforce-eager
```

2. Reduce batch size:
```bash
vllm serve OpenGVLab/InternVL3-78B \
    --tensor-parallel-size 4 \
    --max-num-seqs 2 \
    --trust-remote-code \
    --enforce-eager
```

3. Increase tensor parallelism (use more NPUs):
```bash
vllm serve OpenGVLab/InternVL3-78B \
    --tensor-parallel-size 8 \
    --trust-remote-code \
    --enforce-eager
```

4. Set memory allocation configuration:
```bash
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:128
```

### kernel_meta Permission Errors

If you encounter permission errors related to kernel_meta directory:

1. Set the kernel_meta directory and use enforce-eager mode:
```bash
export KERNEL_META_TEMP_DIR=~/kernel_meta
mkdir -p ~/kernel_meta

vllm serve OpenGVLab/InternVL3-78B \
    --tensor-parallel-size 4 \
    --trust-remote-code \
    --enforce-eager \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.75
```

### Slow Image Processing

If image processing is slow:

1. Ensure images are reasonably sized (e.g., max 1024x1024)
2. Use batch processing when possible
3. Check NPU utilization with `npu-smi info`
4. Verify all NPUs are being utilized in tensor parallel mode

### Model Loading Issues

If the model fails to load:

1. Verify `trust_remote_code=True` is set
2. Check model weights are correctly downloaded
3. Ensure sufficient disk space in cache directory (`~/.cache/huggingface/`) - approximately 150GB needed
4. Verify all NPUs are visible: `npu-smi info`

### Multi-NPU Communication Issues

If you encounter issues with multi-NPU communication:

1. Verify all NPUs are available:
```bash
npu-smi info
```

2. Check HCCL (Huawei Collective Communication Library) configuration:
```bash
export HCCL_OP_EXPANSION_MODE=AIV
```

3. Ensure NPU devices are properly specified:
```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
```

### Connection Errors

If you cannot connect to the server:

1. Verify the server is running: `curl http://localhost:8000/health`
2. Check firewall settings
3. Ensure the correct host and port are specified
4. Check server logs for startup errors

## Performance Comparison

| Model | Parameters | NPUs Required | Inference Speed | Use Case |
|-------|-----------|---------------|-----------------|----------|
| InternVL3-8B | 8B | 1 | Fast | General tasks |
| InternVL3-78B | 78B | 4-8 | Slower | Complex reasoning |

## References

- [InternVL GitHub Repository](https://github.com/OpenGVLab/InternVL)
- [InternVL3-78B Model Card](https://huggingface.co/OpenGVLab/InternVL3-78B)
- [InternVL3-8B Tutorial](./InternVL3-8B.md)
- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM-Ascend User Guide](../../user_guide/index.md)
