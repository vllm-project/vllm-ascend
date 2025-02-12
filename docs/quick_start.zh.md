# 快速开始

## 前提
### 支持的设备
- Atlas A2 训练系列 (Atlas 800T A2, Atlas 900 A2 PoD, Atlas 200T A2 Box16, Atlas 300T A2)
- Atlas 800I A2 推理系列 (Atlas 800I A2)

### Dependencies
| 依赖  | 支持版本 | 推荐版本 | 请注意 |
| ------------ | ------- | ----------- | ----------- | 
| vLLM        | main              |       main          |安装vllm-ascend 必要  
| Python | >= 3.9 | [3.10](https://www.python.org/downloads/) | 安装vllm必要 |
| CANN         | >= 8.0.RC2 | [8.0.RC3](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.0.0.beta1) | 安装vllm-ascend 及 torch-npu必要 |
| torch-npu    | >= 2.4.0   | [2.5.1rc1](https://gitee.com/ascend/pytorch/releases/tag/v6.0.0.alpha001-pytorch2.5.1)    | 安装vllm-ascend必要|
| torch        | >= 2.4.0   | [2.5.1](https://github.com/pytorch/pytorch/releases/tag/v2.5.1)      | 安装torch-npu 和 vllm必要 |

点击[此处](./installation.zh.md)了解更多环境安装信息。