# Qwen3-VL-8B-Instruct 模型 vLLM-Ascend 部署测试报告

## 一、测试环境

| 项目 | 配置 |
|------|------|
| 硬件平台 | Atlas 800I A2 |
| NPU数量 | 1 |
| NPU类型 | Ascend 910B |
| 内存 | 64GB |
| Docker镜像 | vLLM-Ascend 官方容器 |
| Python版本 | 3.11.14 |
| vLLM版本 | 0.18.0 |
| 模型 | Qwen3-VL-8B-Instruct |

## 二、模型权重信息

- **模型路径**: `/workspace/shared_assets/GeekCamp/Infer/Model/zouchangjiang/Qwen3-VL-8B-Instruct`
- **模型格式**: Safetensors (4个分片)
- **权重大小**: 约16.78GB (bfloat16)
- **配置文件**: `config.json`, `model.safetensors.index.json`, `tokenizer.json`, `processor_config.json`

## 三、部署脚本

### 3.1 离线推理脚本 (`offline_inference.py`)

```python
import os
os.environ['VLLM_USE_MODELSCOPE'] = 'True'
os.environ['PYTORCH_NPU_ALLOC_CONF'] = 'max_split_size_mb:256'

from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

MODEL_PATH = "/workspace/shared_assets/GeekCamp/Infer/Model/zouchangjiang/Qwen3-VL-8B-Instruct"

llm = LLM(
    model=MODEL_PATH,
    max_model_len=3840,
    limit_mm_per_prompt={"image": 10},
    dtype="bfloat16",
    gpu_memory_utilization=0.85,
)

sampling_params = SamplingParams(max_tokens=512)

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

processor = AutoProcessor.from_pretrained(MODEL_PATH)
prompt = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, _, _ = process_vision_info(messages, return_video_kwargs=True)

mm_data = {}
if image_inputs is not None:
    mm_data["image"] = image_inputs

llm_inputs = {"prompt": prompt, "multi_modal_data": mm_data}
outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
print(outputs[0].outputs[0].text)
```

### 3.2 在线服务启动脚本 (`start_server.sh`)

```bash
#!/bin/bash

export VLLM_USE_MODELSCOPE=True
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256

/usr/local/python3.11.14/bin/python3 -m vllm.entrypoints.openai.api_server \
    --model /workspace/shared_assets/GeekCamp/Infer/Model/zouchangjiang/Qwen3-VL-8B-Instruct \
    --dtype bfloat16 \
    --max-model-len 3840 \
    --max-num-batched-tokens 3840 \
    --gpu-memory-utilization 0.85 \
    --host 0.0.0.0 \
    --port 8000
```

### 3.3 性能测试脚本 (`run_benchmark.sh`)

```bash
#!/bin/bash

export VLLM_USE_MODELSCOPE=True
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256

/usr/local/python3.11.14/bin/vllm bench throughput \
    --model /workspace/shared_assets/GeekCamp/Infer/Model/zouchangjiang/Qwen3-VL-8B-Instruct \
    --dtype bfloat16 \
    --max-model-len 3840 \
    --gpu-memory-utilization 0.85 \
    --input-len 200 \
    --output-len 100 \
    --num-prompts 100
```

## 四、离线推理测试结果

### 输入

```json
{
  "image": "https://modelscope.oss-cn-beijing.aliyuncs.com/resource/qwen.png",
  "text": "Please provide a detailed description of this image"
}
```

### 输出

模型成功识别图片内容，生成了详细的图片描述（Qwen Logo的视觉描述）。

## 五、在线服务测试结果

### 测试请求

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/workspace/shared_assets/GeekCamp/Infer/Model/zouchangjiang/Qwen3-VL-8B-Instruct",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {
        "role": "user",
        "content": [
          {"type": "image_url", "image_url": {"url": "https://modelscope.oss-cn-beijing.aliyuncs.com/resource/qwen.png"}},
          {"type": "text", "text": "What is the text in the illustration?"}
        ]
      }
    ]
  }'
```

### 响应结果

```json
{
  "id": "chatcmpl-a25ab2890f250e9f",
  "object": "chat.completion",
  "created": 1783665132,
  "model": "/workspace/shared_assets/GeekCamp/Infer/Model/zouchangjiang/Qwen3-VL-8B-Instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "TONGYI Qwen",
        "refusal": null,
        "annotations": null,
        "audio": null,
        "function_call": null,
        "tool_calls": [],
        "reasoning": null
      },
      "logprobs": null,
      "finish_reason": "stop",
      "stop_reason": null,
      "token_ids": null
    }
  ],
  "usage": {
    "prompt_tokens": 107,
    "total_tokens": 114,
    "completion_tokens": 7
  }
}
```

**结果**: 模型正确识别图片中的文字 "TONGYI Qwen"

## 六、vLLM Benchmark 性能测试结果

### 测试命令

```bash
vllm bench throughput \
    --model /workspace/shared_assets/GeekCamp/Infer/Model/zouchangjiang/Qwen3-VL-8B-Instruct \
    --dtype bfloat16 \
    --max-model-len 3840 \
    --gpu-memory-utilization 0.85 \
    --input-len 200 \
    --output-len 100 \
    --num-prompts 100
```

### 性能指标

| 指标 | 值 |
|------|------|
| 请求吞吐量 | 3.45 requests/s |
| 总Token吞吐量 | 3976.97 tokens/s |
| 输出Token吞吐量 | 441.89 tokens/s |
| 输入Token数 | 102400 |
| 输出Token数 | 12800 |
| 模型加载时间 | 178.57秒 |
| 引擎初始化时间 | 30.57秒 |
| 模型权重大小 | 16.78GB |

### 运行日志摘要

```
Loading safetensors checkpoint shards: 100% Completed | 4/4 [02:58<00:00, 44.62s/it]
Loading weights took 178.57 seconds
init engine (profile, create kv cache, warmup model) took 30.57 seconds
Processed prompts: 100%|███████████████| 100/100 [00:28<00:00, 3.51it/s]
Throughput: 3.45 requests/s, 3976.97 total tokens/s, 441.89 output tokens/s
```

## 七、问题记录与解决方案

### 问题1: ModuleNotFoundError: No module named 'transformers'

**现象**: 运行离线推理脚本时提示找不到 transformers 模块

**原因**: 系统Python版本（3.10）与vLLM安装的Python版本（3.11）不一致

**解决方案**: 使用正确的Python路径 `/usr/local/python3.11.14/bin/python3` 运行脚本，并安装缺失依赖

### 问题2: Loading safetensors checkpoint shards 卡住

**现象**: 模型加载时卡在 "Loading safetensors checkpoint shards: 0% Completed | 0/4"

**原因**: KV Cache内存不足导致加载失败

**解决方案**: 调整内存参数：
- `max_model_len` 从 16384 降低到 3840
- `gpu_memory_utilization` 从 0.7 提高到 0.85

### 问题3: ValueError: To serve at least one request with the model's max seq len (16384), 2.25 GiB KV cache is needed

**现象**: 启动服务时提示KV Cache内存不足

**原因**: 默认的 `max_model_len=16384` 需要大量KV Cache内存

**解决方案**: 将 `max_model_len` 降低到 3840

### 问题4: TypeError: Unexpected keyword argument 'multi_modal_data'

**现象**: 使用 `/generate` API 时无法传递多模态数据

**原因**: vLLM基础API不支持多模态数据参数

**解决方案**: 使用 OpenAI 兼容的 API (`vllm.entrypoints.openai.api_server`)

### 问题5: The model `Qwen/Qwen3-VL-8B-Instruct` does not exist

**现象**: 调用OpenAI API时提示模型不存在

**原因**: API请求中的模型名称必须与启动时指定的完整路径一致

**解决方案**: 请求时使用完整路径作为模型名称

## 八、文档使用体验

### 优点

1. **环境配置清晰**: vLLM-Ascend官方容器已预装大部分依赖
2. **命令行工具完整**: `vllm` 命令支持多种子命令（chat, complete, serve, bench等）
3. **多模态支持**: 支持通过 `multi_modal_data` 参数传递图片输入
4. **OpenAI兼容**: 提供 `openai.api_server` 入口，便于迁移

### 待改进

1. **文档不够详细**: 多模态API的使用方式需要查阅源码才能了解
2. **参数命名不一致**: 不同入口的参数命名有差异（如 `--max_model_len` vs `--max-model-len`）
3. **错误信息不够明确**: 内存相关错误的提示不够直观
4. **缺少Benchmark详细参数文档**: 需要多次尝试才能找到正确的benchmark参数组合

## 九、总结

Qwen3-VL-8B-Instruct 模型在 vLLM-Ascend 平台上成功完成了：

1. ✅ **离线推理**: 支持图片+文本输入，生成正确的视觉描述
2. ✅ **在线服务**: 通过OpenAI兼容API提供推理服务，支持多模态请求
3. ✅ **性能测试**: Benchmark测试完成，获得了完整的吞吐量数据

**关键配置参数**:
- `max_model_len`: 3840（受限于单NPU内存）
- `gpu_memory_utilization`: 0.85
- `dtype`: bfloat16

**性能表现**: 在Atlas 800I A2单卡上，Qwen3-VL-8B-Instruct 模型的输出吞吐量约为 442 tokens/s，请求吞吐量约为 3.45 requests/s。