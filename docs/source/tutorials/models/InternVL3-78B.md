# InternVL3-78B

## 模型概述

InternVL3-78B是OpenGVLab推出的大规模多模态语言模型，支持图像和文本输入。相比8B版本，78B模型具有更强的理解和推理能力。

- **模型页面：** https://huggingface.co/OpenGVLab/InternVL3-78B
- **架构：** InternVLChatModel
- **支持硬件：** Atlas A2/A3系列（需要多卡）
- **参数量：** 78B

## 快速开始

### 基础用法

```python
from vllm import LLM, SamplingParams

# 初始化模型（需要多卡）
llm = LLM(
    model="OpenGVLab/InternVL3-78B",
    trust_remote_code=True,
    max_model_len=40960,
    tensor_parallel_size=4,  # 使用4张NPU
    dtype="bfloat16",
)

# 创建采样参数
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512,
)

# 运行多模态推理
prompts = [
    {
        "prompt": "<image>\n这张图片里有什么？",
        "multi_modal_data": {
            "image": "path/to/image.jpg"
        },
    }
]

outputs = llm.generate(prompts, sampling_params=sampling_params)
for output in outputs:
    print(output.outputs[0].text)
```

### OpenAI兼容服务器

```bash
# 使用enforce-eager模式（推荐）
vllm serve OpenGVLab/InternVL3-78B \
    --served-model-name InternVL3-78B \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 4 \
    --dtype bfloat16 \
    --enforce-eager \
    --trust-remote-code \
    --max-model-len 8192 \
    --max-num-seqs 2 \
    --gpu-memory-utilization 0.85
```

### 测试脚本

```python
#!/usr/bin/env python3
import requests
import json

API_URL = "http://localhost:8000/v1/chat/completions"

def test_text():
    """测试纯文本对话"""
    payload = {
        "model": "InternVL3-78B",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "请用一句话介绍InternVL3-78B模型的特点。"
                    }
                ]
            }
        ],
        "temperature": 0.7,
        "max_tokens": 256
    }

    response = requests.post(API_URL, json=payload)
    result = response.json()

    if "choices" in result:
        print("✅ 文本对话成功！")
        print(f"回复: {result['choices'][0]['message']['content']}")
    else:
        print("❌ 失败！")
        print(json.dumps(result, indent=2, ensure_ascii=False))

def test_multimodal():
    """测试多模态对话"""
    payload = {
        "model": "InternVL3-78B",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "请详细描述这张图片。"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg"
                        }
                    }
                ]
            }
        ],
        "temperature": 0.7,
        "max_tokens": 512
    }

    response = requests.post(API_URL, json=payload)
    result = response.json()

    if "choices" in result:
        print("✅ 多模态对话成功！")
        print(f"回复: {result['choices'][0]['message']['content']}")
    else:
        print("❌ 失败！")
        print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    print("🚀 测试InternVL3-78B模型\n")
    test_text()
    print()
    test_multimodal()
```

## 配置说明

### 关键参数

- `tensor_parallel_size`: 张量并行度，78B模型建议使用4-8张NPU
- `max_model_len`: 最大序列长度（默认：40960，可根据内存调整）
- `trust_remote_code`: 必须设置为True
- `enforce_eager`: 推荐使用，避免CANN编译问题
- `dtype`: 推荐使用bfloat16

### 硬件要求

- **最低配置：** 4x Atlas A2 NPU（每张32GB显存）
- **推荐配置：** 8x Atlas A2 NPU（更好的性能和更长的序列支持）
- **内存需求：** 模型权重约150GB，推理时需要额外内存

### 与InternVL3-8B的区别

| 特性 | InternVL3-8B | InternVL3-78B |
|------|--------------|---------------|
| 参数量 | 8B | 78B |
| NPU数量 | 1张 | 4-8张 |
| 推理速度 | 快 | 较慢 |
| 理解能力 | 良好 | 优秀 |
| 推理能力 | 良好 | 优秀 |
| 适用场景 | 通用任务 | 复杂任务 |

## 性能优化

### 1. 批处理

```bash
# 增加批处理大小以提高吞吐量
--max-num-seqs 4
```

### 2. 序列长度

```bash
# 根据实际需求调整，较短的序列可以节省内存
--max-model-len 4096  # 或 8192
```

### 3. 内存利用率

```bash
# 根据NPU内存情况调整
--gpu-memory-utilization 0.85  # 或 0.9
```

### 4. 张量并行

```bash
# 使用更多NPU可以提高性能
--tensor-parallel-size 8  # 如果有8张NPU
```

## 已知限制

1. **需要多卡：** 78B模型无法在单张NPU上运行
2. **编译问题：** 建议使用`--enforce-eager`模式避免kernel_meta权限错误
3. **内存需求：** 需要充足的NPU内存，建议使用32GB或更大显存的NPU
4. **推理速度：** 相比8B版本，推理速度较慢

## 故障排除

### 内存不足（OOM）

如果遇到OOM错误：
- 减少`--max-model-len`（例如：8192 → 4096）
- 减少`--max-num-seqs`（例如：4 → 2）
- 增加`--tensor-parallel-size`（使用更多NPU）
- 降低`--gpu-memory-utilization`（例如：0.85 → 0.75）

### kernel_meta权限错误

```bash
# 使用enforce-eager模式
export KERNEL_META_TEMP_DIR=~/kernel_meta
mkdir -p ~/kernel_meta

vllm serve OpenGVLab/InternVL3-78B \
    --enforce-eager \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.75
```

### 模型加载失败

确保：
- 已安装最新版本的vLLM-Ascend
- 设置了`trust_remote_code=True`参数
- 有足够的磁盘空间存储模型权重（约150GB）
- NPU驱动和CANN版本兼容

### 多卡通信问题

```bash
# 检查NPU状态
npu-smi info

# 确保所有NPU可用
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
```

## 性能基准

基于Atlas A2 NPU的性能参考（4卡配置）：

- **首token延迟：** ~2-3秒
- **生成速度：** ~10-15 tokens/秒
- **最大批处理：** 2-4个并发请求
- **最大序列长度：** 8192 tokens（取决于内存）

*注：实际性能取决于硬件配置、序列长度和批处理大小*

## 参考资料

- [InternVL3-78B模型卡](https://huggingface.co/OpenGVLab/InternVL3-78B)
- [InternVL GitHub](https://github.com/OpenGVLab/InternVL)
- [InternVL3-8B教程](./InternVL3-8B.md)
- [Issue #1362](https://github.com/vllm-project/vllm-ascend/issues/1362)
