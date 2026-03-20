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
 