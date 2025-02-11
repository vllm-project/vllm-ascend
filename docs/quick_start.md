# Quick Start

## Prerequisites
### Support Devices
- Atlas A2 Training series (Atlas 800T A2, Atlas 900 A2 PoD, Atlas 200T A2 Box16, Atlas 300T A2)
- Atlas 800I A2 Inference series (Atlas 800I A2)

### Dependencies
| Requirement | Supported version | Recommended version | Note                                     |
|-------------|-------------------| ----------- |------------------------------------------|
| vLLM        | main              | main | Required for vllm-ascend                 |
| Python      | >= 3.9            | [3.10](https://www.python.org/downloads/) | Required for vllm                        |
| CANN        | >= 8.0.RC2        | [8.0.RC3](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.0.0.beta1) | Required for vllm-ascend and torch-npu   |
| torch-npu   | >= 2.4.0          | [2.5.1rc1](https://gitee.com/ascend/pytorch/releases/tag/v6.0.0.alpha001-pytorch2.5.1)    | Required for vllm-ascend                 |
| torch       | >= 2.4.0          | [2.5.1](https://github.com/pytorch/pytorch/releases/tag/v2.5.1)      | Required for torch-npu and vllm |

Find more about how to setup your environment in [here](docs/environment.md).