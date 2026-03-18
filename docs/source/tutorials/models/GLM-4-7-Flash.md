\# GLM-4.7-Flash 模型部署指南（vLLM Ascend）



\## 1. 环境准备

\### 1.1 硬件要求

\- Ascend Atlas A2 系列及以上 NPU

\- 显存 ≥ 16GB

\- 内存 ≥ 32GB



\### 1.2 环境配置

```bash

\# 克隆仓库

git clone https://github.com/vllm-project/vllm-ascend.git

cd vllm-ascend



\# 安装依赖

pip install -r requirements.txt



\# 设置环境变量

export VLLM\_ASCEND\_DISABLE\_RING\_MLA=1

