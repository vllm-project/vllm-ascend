# Meta Chameleon-7B Deployment Guide

## Overview

[Chameleon-7B](https://huggingface.co/facebook/chameleon-7b) is an early-fusion multimodal model from Meta that unifies text and image understanding through a shared tokenization mechanism. This guide demonstrates how to deploy and run `facebook/chameleon-7b` efficiently on Huawei Ascend NPUs using vLLM Ascend.

### Key Characteristics

- **Early Fusion**: Text and images are processed through a unified tokenizer, enabling seamless interleaving of tokens
- **Mixed Modality**: Supports text-only, image-only, and image+text combinations in the same input
- **7B Parameters**: Efficient size suitable for single-card Ascend deployment
- **Instruction Following**: Fine-tuned for chat and instruction-following scenarios

## Prerequisites

### Hardware Requirements
- **Recommended**: Huawei Ascend 910B/C (for optimal performance)
- **Minimum**: Ascend 910A or Atlas A2 Series
- **Memory**: ≥ 32GB HBM for fp32, ≥ 16GB for bfloat16

### Software Requirements
- vLLM Ascend runtime installed
- Python 3.9+
- vLLM >= 0.6.0
- PyTorch with Ascend backend

### Environment Setup

```bash
# Activate Ascend environment
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Install vLLM Ascend
pip install vllm-ascend

# Verify installation
python -c "import vllm; print(vllm.__version__)"
```

## Quick Start: Text-Only Generation

### 1. Basic Text Generation

```python
from vllm import LLM, SamplingParams

# Initialize the model
model_path = "facebook/chameleon-7b"

llm = LLM(
    model=model_path,
    trust_remote_code=True,
    tensor_parallel_size=1,
    dtype="bfloat16",
    enforce_eager=True,
    max_model_len=2048,
)

# Define prompts
prompts = [
    "What is the capital of France?",
    "Explain the water cycle in simple terms.",
    "Write a Python function to check if a number is prime.",
]

# Sampling configuration
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=150,
    frequency_penalty=0.0,
)

# Generate
outputs = llm.generate(prompts, sampling_params)

# Print results
for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Generated: {output.outputs[0].text}")
    print("-" * 50)
```

## Advanced: Multimodal Image+Text Generation

### 2. Loading and Processing Images

```python
from vllm import LLM, SamplingParams
from vllm.multi_modal_data import ImageBase64, ImageUrl
import base64
from pathlib import Path

llm = LLM(
    model="facebook/chameleon-7b",
    trust_remote_code=True,
    tensor_parallel_size=1,
    dtype="bfloat16",
    enforce_eager=True,
    max_model_len=4096,  # Larger context for images
    gpu_memory_utilization=0.75,  # Account for image token expansion
)

# Method 1: Load local image as base64
def image_to_base64(image_path: str) -> str:
    """Convert image file to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# Method 2: Use URL-based images
def create_image_prompt(image_input, prompt_text: str) -> dict:
    """Create prompt with image for multimodal inference."""
    return {
        "prompt": prompt_text,
        "multi_modal_data": {
            "image": image_input,
        }
    }

# Example: Image understanding task
image_base64 = image_to_base64("./sample_image.jpg")
image_data = ImageBase64(image_base64=image_base64, image_type="jpeg")

prompt = "Describe the contents of this image in detail."
prompt_dict = create_image_prompt(image_data, prompt)

sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=300,
)

# Generate multimodal response
outputs = llm.generate([prompt_dict], sampling_params)
for output in outputs:
    print("Model Response:")
    print(output.outputs[0].text)
```

### 3. Batch Processing Images

```python
from vllm import LLM, SamplingParams
from vllm.multi_modal_data import ImageUrl
from typing import List

llm = LLM(
    model="facebook/chameleon-7b",
    trust_remote_code=True,
    dtype="bfloat16",
    enforce_eager=True,
    max_model_len=4096,
)

# Batch of image URLs
image_urls = [
    "https://example.com/image1.jpg",
    "https://example.com/image2.jpg",
    "https://example.com/image3.jpg",
]

# Create batch prompts
prompts = []
for url in image_urls:
    prompts.append({
        "prompt": "What is the main subject in this image?",
        "multi_modal_data": {
            "image": ImageUrl(image_url=url),
        }
    })

sampling_params = SamplingParams(temperature=0.7, max_tokens=128)
outputs = llm.generate(prompts, sampling_params)

for i, output in enumerate(outputs):
    print(f"Image {i+1}: {output.outputs[0].text}")
```

## OpenAI-Compatible API

### 4. Running as an OpenAI-Compatible Server

```bash
# Start vLLM server with Chameleon
vllm serve facebook/chameleon-7b \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --enforce-eager \
    --gpu-memory-utilization 0.75 \
    --max-model-len 2048 \
    --trust-remote-code
```

### 5. Client Usage (OpenAI SDK)

```python
from openai import OpenAI

# Initialize client (assuming server runs on localhost:8000)
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy",  # vLLM doesn't require real API key
)

# Text-only request
response = client.chat.completions.create(
    model="facebook/chameleon-7b",
    messages=[
        {
            "role": "user",
            "content": "Explain quantum computing in 2 sentences.",
        }
    ],
    temperature=0.7,
    max_tokens=150,
)

print(response.choices[0].message.content)

# Image+text request
response = client.chat.completions.create(
    model="facebook/chameleon-7b",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What's in this image?"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://example.com/image.jpg",
                    }
                }
            ]
        }
    ],
    max_tokens=300,
)

print(response.choices[0].message.content)
```

## Performance Tuning for Ascend NPUs

### 6. Optimization Configuration

```python
from vllm import LLM, SamplingParams

# Optimized configuration for Ascend
llm = LLM(
    model="facebook/chameleon-7b",
    tensor_parallel_size=1,
    dtype="bfloat16",              # Reduces memory vs. fp32
    enforce_eager=True,             # Avoids graph compilation on Ascend
    max_model_len=2048,            # Adjust based on available memory
    gpu_memory_utilization=0.75,   # Leave headroom for image tokens
    trust_remote_code=True,
    # Ascend-specific optimizations
    enable_prefix_caching=False,    # Keep False for stability
    num_scheduler_steps=1,          # Conservative scheduling
)

# Sampling for quality vs. speed trade-off
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    presence_penalty=0.0,
    frequency_penalty=0.0,
    max_tokens=256,
    skip_special_tokens=True,
)

outputs = llm.generate(["Prompt"], sampling_params)
```

## Important Notes for Ascend Deployment

### Memory Considerations
- **Image Expansion**: Each image can expand to 500-2000 tokens depending on resolution
- **Batch Processing**: Monitor total tokens (text + image tokens) not exceeding `max_model_len`
- **Mixed Precision**: Use bfloat16 to balance performance and memory usage

### Known Limitations
1. **enforce_eager=True required**: Graph compilation is not fully supported for Chameleon on Ascend
2. **Single-card inference**: Currently optimized for single-card; tensor parallelism experimental
3. **Image formats**: Supports JPEG, PNG, WebP (32KB-10MB recommended)

### Troubleshooting

**Issue: "CUDA out of memory"**
```python
# Solution: Reduce max_model_len or batch_size
llm = LLM(
    model="facebook/chameleon-7b",
    max_model_len=1024,  # Reduce from 2048
    gpu_memory_utilization=0.6,  # Further reduce
)
```

**Issue: Image token limit exceeded**
```python
# Resize images before inference
from PIL import Image

def resize_image(image_path: str, max_size: int = 1024) -> str:
    """Resize image to reduce token expansion."""
    img = Image.open(image_path)
    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    img.save("resized_image.jpg")
    return "resized_image.jpg"
```

**Issue: Slow generation speed**
```python
# Check Ascend utilization and adjust batch size
import os
os.environ["ASCEND_DEVICE_ID"] = "0"
os.environ["VLLM_ASCEND_ENABLE_NZ"] = "1"  # Enable memory optimization

llm = LLM(
    model="facebook/chameleon-7b",
    enforce_eager=True,
    dtype="bfloat16",
)
```

## Example: Complete End-to-End Workflow

```python
from vllm import LLM, SamplingParams
from vllm.multi_modal_data import ImageUrl
import json
from datetime import datetime

def inference_pipeline(model_name: str = "facebook/chameleon-7b"):
    """Complete inference pipeline for Chameleon on Ascend."""
    
    # Initialize model
    print(f"[{datetime.now()}] Loading {model_name}...")
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=2048,
    )
    
    # Define test cases
    test_cases = [
        {
            "type": "text",
            "prompt": "What is machine learning?",
            "description": "Text-only generation"
        },
        {
            "type": "image+text",
            "prompt": "Analyze this image",
            "image_url": "https://example.com/sample.jpg",
            "description": "Multimodal inference"
        },
    ]
    
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=200,
    )
    
    results = []
    
    for test in test_cases:
        print(f"\nRunning: {test['description']}")
        
        if test["type"] == "text":
            outputs = llm.generate([test["prompt"]], sampling_params)
        else:
            prompt_dict = {
                "prompt": test["prompt"],
                "multi_modal_data": {
                    "image": ImageUrl(image_url=test["image_url"]),
                }
            }
            outputs = llm.generate([prompt_dict], sampling_params)
        
        result = {
            "test": test["description"],
            "prompt": test["prompt"],
            "output": outputs[0].outputs[0].text,
            "tokens": len(outputs[0].outputs[0].token_ids),
        }
        results.append(result)
        print(f"Generated: {result['output'][:100]}...")
    
    return results

# Run pipeline
if __name__ == "__main__":
    results = inference_pipeline()
    print("\n" + "="*50)
    print("Pipeline Results:")
    print(json.dumps(results, indent=2, ensure_ascii=False))
```

## Additional Resources

- **HuggingFace Model Card**: https://huggingface.co/facebook/chameleon-7b
- **vLLM Documentation**: https://docs.vllm.ai/
- **vLLM Ascend Guide**: https://docs.vllm.ai/projects/ascend/
- **Ascend CANN Toolkit**: https://www.hiascend.com/
- **Chameleon Paper**: https://arxiv.org/abs/2405.09818

## Support & Feedback

For issues or questions:
- **GitHub Issues**: https://github.com/vllm-project/vllm-ascend/issues
- **vLLM Community**: https://github.com/vllm-project/vllm/discussions

