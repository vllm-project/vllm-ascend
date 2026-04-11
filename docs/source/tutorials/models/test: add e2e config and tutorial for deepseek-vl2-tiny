# DeepSeek-VL2-Tiny 模型验证报告

# 基本信息

模型名称: deepseek-ai/deepseek-vl2-tiny

模型类型: 多模态视觉语言模型

发布时间: 2024年12月

验证环境: 昇腾 910B + vLLM 0.11.0 + PyTorch 2.7.1

# 验证结论

模型无法直接运行，需要适配才能支持。

# 问题描述

## 错误现象

执行模型加载时出现以下错误:

pydantic_core.\_pydantic_core.ValidationError: 1 validation error for ModelConfig

Value error, No model architectures are specified

## 根因分析

1\. 模型配置格式特殊

DeepSeek-VL2 的 config.json 采用嵌套结构:

\- 顶层没有 architectures 字段

\- 架构信息位于 language_config.architectures 中

2\. vLLM 未适配嵌套格式

当前 vLLM 版本(0.11.0)在读取模型配置时,只在顶层查找 architectures 字段,没有递归查找嵌套字段的逻辑。

## 验证方法

`cat config.json | grep -A 5 "language_config"`

输出显示:

```
"language_config": {
"architectures": ["DeepseekV2ForCausalLM"]
}
```


# 适配差距(GAP)分析

昇腾 NPU 环境: 正常

vLLM 基础依赖: 正常

模型架构识别: 缺失

# 适配建议

## 代码修改方向

修改 vllm/engine/arg_utils.py 中的 create_model_config 方法:

def get_architectures(config):

\# 先在顶层查找

if "architectures" in config:

return config\["architectures"\]

\# 再在 language_config 中查找

elif "language_config" in config and "architectures" in config\["language_config"\]:

return config\["language_config"\]\["architectures"\]

else:

raise ValueError("No model architectures are specified")

## 环境限制说明

\- 昇腾 NPU 需要 torch-npu,最高支持 PyTorch 2.9.0

\- vLLM 0.18.0 需要 PyTorch 2.10.0

\- 两者无法同时满足,适配时需基于 vLLM 0.11.0 进行

测试步骤

## 1\. 环境准备

创建昇腾 910B 容器实例

镜像: PyTorch 2.7.1 + CANN 8.3.RC2

## 2\. 依赖检查

npu-smi info

python -c "import torch_npu; print('NPU OK')"

## 3\. 模型加载测试

from vllm import LLM

model = LLM(model="deepseek-ai/deepseek-vl2-tiny")

相关资源

模型 HuggingFace 主页: https://huggingface.co/deepseek-ai/deepseek-vl2-tiny

vLLM 文档: https://docs.vllm.ai/

vLLM Ascend 文档: https://docs.vllm.ai/projects/ascend/en/latest/

任务 Issue #7319: https://github.com/vllm-project/vllm-ascend/issues/7319

# 报告日期

2026年3月21日
