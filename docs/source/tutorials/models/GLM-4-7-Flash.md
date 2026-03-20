```markdown
# GLM-4-7-Flash 模型部署教程

## 模型介绍
GLM-4-7-Flash 是由 zai-org 发布的 4.7B 参数高效大语言模型，具备低资源占用、快速推理与长文本支持能力，可广泛应用于对话生成、文本理解、代码辅助等场景。

## 环境准备

### 硬件要求
- Atlas A2 系列 NPU
- 显存 ≥ 16GB

### 软件依赖
```bash
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
pip install -r requirements.txt 
 
###环境变量
```bash
export VLLM_ASCEND_DISABLE_RING_MLA=1 
 

##启动命令
###单卡运行
```bash
python -m vllm.entrypoints.api_server \
  --model zai-org/GLM-4-7-Flash \
  --device npu \
  --trust-remote-code \
  --max-model-len 8192 \
  --dtype float16 
 
###多卡张量并行
```bash
python -m vllm.entrypoints.api_server \
  --model zai-org/GLM-4-7-Flash \
  --device npu \
  --trust-remote-code \
  --max-model-len 8192 \
  --tensor-parallel-size 2 
 
##推理示例
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "zai-org/GLM-4-7-Flash",
    "prompt": "Hello, how are you?",
    "max_tokens": 256,
    "temperature": 0.7
  }' 
 




