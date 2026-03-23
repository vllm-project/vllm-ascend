# Qwen2.5-Math-RM-72B 模型部署教程

## 概述

本教程介绍如何在 vLLM Ascend 平台上部署 Qwen2.5-Math-RM-72B 奖励模型，包括环境准备、模型下载、适配开发和性能优化。

## 模型简介

**Qwen2.5-Math-RM-72B** 是阿里巴巴通义千问团队开发的 72B 参数数学奖励模型，用于评估数学问题的答案质量。

| 属性 | 值 |
|-----|-----|
| 模型名称 | Qwen2.5-Math-RM-72B |
| 参数量 | 72B |
| 模型类型 | 奖励模型 (Reward Model) |
| 输入 | 完整对话 (问题 + 答案) |
| 输出 | 标量分数 (reward score) |
| 上下文长度 | 4096 tokens |
| 权重格式 | safetensors |

## 环境准备

### 1. 硬件要求

| 组件 | 最低配置 | 推荐配置 |
|-----|---------|---------|
| NPU | 1x Ascend 910B | 2x Ascend 910B |
| 内存 | 256GB | 512GB |
| 存储 | 200GB SSD | 500GB NVMe SSD |

### 2. 软件环境

```bash
# 操作系统
Ubuntu 20.04/22.04 LTS

# Python 版本
Python 3.11

# 关键依赖
- transformers >= 4.55.2
- torch >= 2.0.0
- vllm >= 0.11.0
- vllm-ascend >= 0.11.0
```

### 3. 安装依赖

```bash
# 设置 Python 路径
export PATH=/usr/local/python3.11.13/bin:$PATH

# 安装 transformers
pip install transformers==4.55.2

# 安装 PyTorch
pip install torch==2.7.1

# 安装 vLLM
pip install vllm==0.11.0

# 安装 vLLM Ascend 插件
pip install vllm-ascend
```

## 模型下载

### 方式 1: 从 Hugging Face 下载

```bash
# 使用 huggingface-cli
huggingface-cli download Qwen/Qwen2.5-Math-RM-72B \
  --local-dir /data/Qwen2.5-Math-RM-72B \
  --local-dir-use-symlinks False
```

### 方式 2: 使用镜像加速

```bash
# 设置镜像
export HF_ENDPOINT=https://hf-mirror.com

# 使用下载脚本
python download_with_mirror.py
```

### 方式 3: Git LFS 克隆

```bash
# 安装 Git LFS
git lfs install

# 克隆仓库
git clone https://huggingface.co/Qwen/Qwen2.5-Math-RM-72B
```

## 部署方式

### 方式 1: 原生 Transformers 部署

#### 快速开始

```python
from transformers import AutoModel, AutoTokenizer
import torch

# 加载模型
model_path = "/data/Qwen2.5-Math-RM-72B"

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
    local_files_only=True
)

model = AutoModel.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    local_files_only=True
).eval()

# 准备对话
messages = [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "2+2 equals 4."}
]

# 应用对话模板
conversation_str = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False
)

# 编码输入
input_ids = tokenizer.encode(
    conversation_str,
    return_tensors="pt",
    add_special_tokens=False
).to(model.device)

# 推理
with torch.no_grad():
    outputs = model(input_ids=input_ids)

reward_score = outputs[0].item()
print(f"Reward Score: {reward_score}")
```

#### 启动 API 服务

```bash
# 使用 api_server.py
python api_server.py

# 服务启动后，默认地址为 http://localhost:8000
```

### 方式 2: vLLM Ascend 适配部署

#### 使用适配器

```python
from vllm_reward_adapter import VLLMRewardAdapter

# 创建适配器
adapter = VLLMRewardAdapter(
    model_path="/data/Qwen2.5-Math-RM-72B",
    device="npu"
)

# 加载模型（自动选择 vLLM 或 Transformers）
adapter.load()

# 获取后端信息
backend_info = adapter.get_backend_info()
print(f"Backend: {backend_info['backend']}")

# 推理
messages = [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "2+2=4"}
]

score, inference_time = adapter.get_reward(messages)
print(f"Score: {score:.4f}, Time: {inference_time:.2f}ms")
```

#### 启动 vLLM 适配服务

```bash
# 使用 vLLM 适配版本
python vllm_reward_adapter.py

# 或使用 vllm_ascend_server.py
python vllm_ascend_server.py
```

## API 接口说明

### 1. 健康检查

```bash
GET /health
```

**响应示例**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "backend": "vLLM",
  "device": "npu"
}
```

### 2. 单条奖励评分

```bash
POST /reward
Content-Type: application/json

{
  "messages": [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "2+2=4"}
  ]
}
```

**响应示例**:
```json
{
  "reward_score": 0.85,
  "status": "success",
  "message": "推理成功",
  "backend": "vLLM",
  "inference_time_ms": 45.2
}
```

### 3. 批量奖励评分

```bash
POST /reward/batch
Content-Type: application/json

{
  "conversations": [
    [
      {"role": "user", "content": "What is 2+2?"},
      {"role": "assistant", "content": "2+2=4"}
    ],
    [
      {"role": "user", "content": "What is 3+3?"},
      {"role": "assistant", "content": "3+3=6"}
    ]
  ]
}
```

**响应示例**:
```json
{
  "scores": [0.85, 0.82],
  "status": "success",
  "backend": "vLLM",
  "batch_size": 2,
  "total_time_ms": 89.5,
  "throughput": 22.35
}
```

## 性能优化

### 1. 批处理优化

```python
# 使用批量推理提高吞吐量
conversations = [
    [messages1],
    [messages2],
    # ...
]

scores, total_time = adapter.get_rewards_batch(conversations)
```

### 2. 内存优化

```python
# 使用 bfloat16 减少内存占用
model = AutoModel.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 使用梯度检查点（训练时）
model.gradient_checkpointing_enable()
```

### 3. vLLM 优化

```python
# 配置 vLLM 参数
engine_args = AsyncEngineArgs(
    model=model_path,
    tensor_parallel_size=2,  # 张量并行
    dtype="bfloat16",
    max_model_len=4096,
)
```

## 测试验证

### 运行测试用例

```bash
# 运行所有测试
pytest tests/e2e/models/configs/test_qwen2_5_math_rm_72b.py -v

# 运行特定测试类
pytest test_qwen2_5_math_rm_72b.py::TestQwen2_5_Math_RM_72B_RewardScenarios -v

# 运行性能测试
pytest test_qwen2_5_math_rm_72b.py::TestQwen2_5_Math_RM_72B_Performance -v
```

### 测试覆盖范围

1. **基础功能测试**: 模型加载、tokenizer、对话模板
2. **奖励模型场景**: 数学问题评分、答案质量评估
3. **边界条件**: 空输入、超长输入、特殊字符
4. **性能测试**: 延迟、吞吐量、内存使用
5. **vLLM 适配**: 双后端切换、插件验证

## 故障排除

### 问题 1: 模型加载失败

**症状**: `ModuleNotFoundError: No module named 'transformers'`

**解决方案**:
```bash
# 检查 Python 路径
export PATH=/usr/local/python3.11.13/bin:$PATH

# 重新安装依赖
pip install transformers==4.55.2
```

### 问题 2: 内存不足

**症状**: `CUDA out of memory` 或 `NPU out of memory`

**解决方案**:
```python
# 使用 CPU 加载
model = AutoModel.from_pretrained(
    model_path,
    device_map="cpu"
)

# 或使用更低精度
model = AutoModel.from_pretrained(
    model_path,
    torch_dtype=torch.float16
)
```

### 问题 3: vLLM 导入失败

**症状**: `ImportError: cannot import name 'AutoVideoProcessor'`

**解决方案**:
```bash
# 升级 transformers
pip install transformers>=4.55.2
```

### 问题 4: API 服务无法启动

**症状**: `Address already in use`

**解决方案**:
```bash
# 查找占用端口的进程
lsof -i :8000

# 杀死进程
kill -9 <PID>

# 或使用不同端口
python api_server.py --port 8001
```

## 最佳实践

### 1. 生产环境部署

- 使用 Docker 容器化部署
- 配置负载均衡
- 设置健康检查和自动重启
- 监控性能和资源使用

### 2. 安全性

- 使用 HTTPS 加密通信
- 配置访问控制和认证
- 限制请求频率
- 记录审计日志

### 3. 性能监控

```python
# 添加性能监控
import time
import logging

logger = logging.getLogger(__name__)

start_time = time.time()
score = get_reward(messages)
latency = time.time() - start_time

logger.info(f"Inference latency: {latency*1000:.2f}ms")
```

## 参考资料

- [Qwen2.5-Math 官方文档](https://huggingface.co/Qwen/Qwen2.5-Math-RM-72B)
- [vLLM 文档](https://docs.vllm.ai/)
- [Transformers 文档](https://huggingface.co/docs/transformers/)
- [Ascend 文档](https://www.hiascend.com/document)

## 版本历史

| 版本 | 日期 | 更新内容 |
|-----|------|---------|
| 1.0.0 | 2026-03-22 | 初始版本，支持 vLLM Ascend 适配 |

---

**作者**: AI Assistant  
**更新日期**: 2026-03-22  
**文档版本**: 1.0.0
