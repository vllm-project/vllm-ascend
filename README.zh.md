<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/docs/logos/vllm-ascend-logo-text-dark.png">
    <img alt="vllm-ascend" src="https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/docs/logos/vllm-ascend-logo-text-light.png" width=55%>
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

## 前提
### 支持的设备
- Atlas A2 训练系列 (Atlas 800T A2, Atlas 900 A2 PoD, Atlas 200T A2 Box16, Atlas 300T A2)
- Atlas 800I A2 推理系列 (Atlas 800I A2)

### 依赖
| 需求 | 支持的版本 | 推荐版本 | 注意                                     |
|-------------|-------------------| ----------- |------------------------------------------|
| vLLM        | main              | main |  vllm-ascend 依赖                 |
| Python      | >= 3.9            | [3.10](https://www.python.org/downloads/) |  vllm 依赖                       |
| CANN        | >= 8.0.RC2        | [8.0.RC3](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.0.0.beta1) |  vllm-ascend and torch-npu 依赖  |
| torch-npu   | >= 2.4.0          | [2.5.1rc1](https://gitee.com/ascend/pytorch/releases/tag/v6.0.0.alpha001-pytorch2.5.1)    | vllm-ascend 依赖                |
| torch       | >= 2.4.0          | [2.5.1](https://github.com/pytorch/pytorch/releases/tag/v2.5.1)      |  torch-npu and vllm 依赖 |

在[此处](docs/environment.zh.md)了解更多如何配置您环境的信息。

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

请参阅 [vLLM 快速入门](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)以获取更多详细信息。

## 构建

#### 从源码构建Python包

```bash
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
pip install -e .
```

#### 构建容器镜像
```bash
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
docker build -t vllm-ascend-dev-image -f ./Dockerfile .
```

查看[构建和测试](./CONTRIBUTING.zh.md)以获取更多详细信息，其中包含逐步指南，帮助您设置开发环境、构建和测试。

## 特性支持矩阵
| Feature | Supported | Note |
|---------|-----------|------|
| Chunked Prefill | ✗ | Plan in 2025 Q1 |
| Automatic Prefix Caching | ✅ | Imporve performance in 2025 Q1 |
| LoRA | ✗ | Plan in 2025 Q1 |
| Prompt adapter | ✅ ||
| Speculative decoding | ✅ | Impore accuracy in 2025 Q1|
| Pooling | ✗ | Plan in 2025 Q1 |
| Enc-dec | ✗ | Plan in 2025 Q1 |
| Multi Modality | ✅ (LLaVA/Qwen2-vl/Qwen2-audio/internVL)| Add more model support in 2025 Q1 |
| LogProbs | ✅ ||
| Prompt logProbs | ✅ ||
| Async output | ✅ ||
| Multi step scheduler | ✅ ||
| Best of | ✅ ||
| Beam search | ✅ ||
| Guided Decoding | ✗ | Plan in 2025 Q1 |

## 模型支持矩阵

此处展示了部分受支持的模型。有关更多详细信息，请参阅 [supported_models](docs/supported_models.md)：
| Model | Supported | Note |
|---------|-----------|------|
| Qwen 2.5 | ✅ ||
| Mistral |  | Need test |
| DeepSeek v2.5 | |Need test |
| LLama3.1/3.2 | ✅ ||
| Gemma-2 |  |Need test|
| baichuan |  |Need test|
| minicpm |  |Need test|
| internlm | ✅ ||
| ChatGLM | ✅ ||
| InternVL 2.5 | ✅ ||
| Qwen2-VL | ✅ ||
| GLM-4v |  |Need test|
| Molomo | ✅ ||
| LLaVA 1.5 | ✅ ||
| Mllama |  |Need test|
| LLaVA-Next |  |Need test|
| LLaVA-Next-Video |  |Need test|
| Phi-3-Vison/Phi-3.5-Vison |  |Need test|
| Ultravox |  |Need test|
| Qwen2-Audio | ✅ ||


## 贡献
我们欢迎并重视任何形式的贡献与合作：
- 您可以在[这里](https://github.com/vllm-project/vllm-ascend/issues/19)反馈您的使用体验。
- 请通过[提交问题](https://github.com/vllm-project/vllm-ascend/issues)来告知我们您遇到的任何错误。
- 请参阅 [CONTRIBUTING.zh.md](./CONTRIBUTING.zh.md) 中的贡献指南。

## 许可证

Apache 许可证 2.0，如 [LICENSE](./LICENSE) 文件中所示。