# GLM-4.1V-9B-Thinking

## Introduction

GLM-4.1V is an advanced multimodal model based on the GLM architecture specifically designed for agent applications.

This document will show the main verification steps of the model, including environment preparation, single-node deployment, and testing.

## Environment Preparation

### Model Weight

- `GLM-4.1V-9B-Thinking`: [Download model weight](https://huggingface.co/zai-org/GLM-4.1V-9B-Thinking).

## Deployment

### Single-node Deployment

Run the following script to execute online inference.

```shell
vllm serve zai-org/GLM-4.1V-9B-Thinking \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9 \
  --served-model-name glm4v
```

## Functional Verification

Once your server is started, you can query the model with input prompts:

```shell
curl -H "Accept: application/json" \
    -H "Content-type: application/json" \
    -X POST \
    -d '{
        "model": "glm4v", 
        "messages": [{ 
            "role": "user", 
            "content": "The future of AI is" 
        }], 
        "stream": false, 
        "ignore_eos": false, 
        "temperature": 0, 
        "max_tokens": 200 
    }' http://localhost:8000/v1/chat/completions
```

## Accuracy Evaluation

### Using E2E Multimodal Tests

You can run automated end-to-end multimodal tests via pytest:

```shell
pytest tests/e2e/
```

## Special Notes for GLM-4.1V

### 2D RoPE Intelligent Fallback
Due to the GLM-4.1V multimodal vision markers utilizing 2D coordinates (`positions.dim() > 1`), and the underlying highly fused operator `_npu_rotary_embedding` on Ascend only accepting 1D continuous text streams, we implemented an intelligent fallback mechanism at the operator layer. When 2D coordinates are detected, the operations automatically fall back to pure PyTorch computations to prevent Core Dump.

### Vision Tensor Memory Continuity
To resolve the crash of underlying NPU operators caused by `pixel_values` (which typically become non-contiguous in memory due to multidimensional Slice and Concat operations during preprocessing), an implicit `.contiguous()` patch is applied to ensure the memory continuity of vision tensors.
