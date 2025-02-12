<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/docs/source/logos/vllm-ascend-logo-text-dark.png">
    <img alt="vllm-ascend" src="https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/docs/source/logos/vllm-ascend-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
vLLM Ascend Plugin
</h3>

<p align="center">
| <a href="https://www.hiascend.com/en/"><b>关于昇腾</b></a> | <a href="https://slack.vllm.ai"><b>开发者 Slack (#sig-ascend)</b></a> |
</p>

<p align="center">
<a href="README.md"><b>English</b></a> | <a><b>中文</b></a>
</p>

---
*最新消息* 🔥

- [2024/12] 我们正在与 vLLM 社区合作，以支持 [[RFC]: Hardware pluggable](https://github.com/vllm-project/vllm/issues/11162).
---
## 总览

vLLM 昇腾插件 (`vllm-ascend`) 是一个让vLLM在Ascend NPU无缝运行的后端插件。

此插件是 vLLM 社区中支持昇腾后端的推荐方式。它遵循[[RFC]: Hardware pluggable](https://github.com/vllm-project/vllm/issues/11162)所述原则：通过解耦的方式提供了vLLM对Ascend NPU的支持。

使用 vLLM 昇腾插件，可以让类Transformer、混合专家(MOE)、嵌入、多模态等流行的大语言模型在 Ascend NPU 上无缝运行。

## 准备

- 硬件：Atlas 800I A2 Inference系列、Atlas A2 Training系列
- 软件：
  * Python >= 3.9
  * CANN >= 8.0.RC2
  * PyTorch >= 2.4.0, torch-npu >= 2.4.0
  * vLLM (与vllm-ascend版本一致)

在[此处](docs/source/installation.md)，您可以了解如何逐步准备环境。

## 开始使用

> [!NOTE]
> 目前，我们正在积极与 vLLM 社区合作以支持 Ascend 后端插件，一旦支持，您可以使用一行命令: `pip install vllm vllm-ascend` 来完成安装。

通过源码安装:
```bash
# 安装vllm main 分支参考文档:
# https://docs.vllm.ai/en/latest/getting_started/installation/cpu/index.html#build-wheel-from-source
git clone --depth 1 https://github.com/vllm-project/vllm.git
cd vllm
pip install -r requirements-build.txt
VLLM_TARGET_DEVICE=empty pip install .

# 安装vllm-ascend main 分支
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
pip install -e .
```

运行如下命令使用 [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) 模型启动服务:

```bash
# 设置环境变量 VLLM_USE_MODELSCOPE=true 加速下载
vllm serve Qwen/Qwen2.5-0.5B-Instruct
curl http://localhost:8000/v1/models
```

**请参阅 [官方文档](https://vllm-ascend.readthedocs.io/en/latest/)以获取更多详细信息**

## 贡献
有关更多详细信息，请参阅 [CONTRIBUTING](docs/source/developer_guide/contributing.zh.md)，可以更详细的帮助您部署开发环境、构建和测试。

我们欢迎并重视任何形式的贡献与合作：
- 您可以在[这里](https://github.com/vllm-project/vllm-ascend/issues/19)反馈您的使用体验。
- 请通过[提交问题](https://github.com/vllm-project/vllm-ascend/issues)来告知我们您遇到的任何错误。

## 许可证

Apache 许可证 2.0，如 [LICENSE](./LICENSE) 文件中所示。