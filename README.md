<p align="center">
  <picture>
    <!-- TODO: Replace tmp link to logo url after vllm-projects/vllm-ascend ready -->
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/4a958093-58b5-4772-a942-638b51ced646">
    <img alt="vllm-ascend" src="https://github.com/user-attachments/assets/838afe2f-9a1d-42df-9758-d79b31556de0" width=55%>
  </picture>
</p>

<h3 align="center">
vLLM Ascend Plugin
</h3>

<p align="center">
| <a href="https://www.hiascend.com/en/"><b>About Ascend</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack (#sig-ascend)</b></a> |
</p>

<p align="center">
<a ><b>English</b></a> | <a href="README.zh.md"><b>中文</b></a>
</p>

---
*Latest News* 🔥

- [2024/12] We are working with the vLLM community to support [[RFC]: Hardware pluggable](https://github.com/vllm-project/vllm/issues/11162).
---
## Overview

vLLM Ascend plugin (`vllm-ascend`) is a backend plugin for running vLLM on the Ascend NPU.

This plugin is the recommended approach for supporting the Ascend backend within the vLLM community. It adheres to the principles outlined in the [[RFC]: Hardware pluggable](https://github.com/vllm-project/vllm/issues/11162), providing a hardware-pluggable interface that decouples the integration of the Ascend NPU with vLLM.

By using vLLM Ascend plugin, popular open-source models, including Transformer-like, Mixture-of-Expert, Embedding, Multi-modal LLMs can run seamlessly on the Ascend NPU.

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

## Getting Started

> [!NOTE]
> Currently, we are actively collaborating with the vLLM community to support the Ascend backend plugin, once supported you can use one line command `pip install vllm vllm-ascend` to compelete installation.

Installation from source code:
```bash
# Install vllm main branch according:
# https://docs.vllm.ai/en/latest/getting_started/installation/cpu/index.html#build-wheel-from-source
git clone --depth 1 https://github.com/vllm-project/vllm.git
cd vllm
pip install -r requirements-build.txt
VLLM_TARGET_DEVICE=empty pip install .

# Install vllm-ascend main branch
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
pip install -e .
```

Run the following command to start the vLLM server with the [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) model:

```bash
# export VLLM_USE_MODELSCOPE=true to speed up download
vllm serve Qwen/Qwen2.5-0.5B-Instruct
curl http://localhost:8000/v1/models
```

Please refer to [vLLM Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html) for more details.

## Building

#### Build Python package from source

```bash
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
pip install -e .
```

#### Build container image from source
```bash
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
docker build -t vllm-ascend-dev-image -f ./Dockerfile .
```

See [Building and Testing](./CONTRIBUTING.md) for more details, which is a step-by-step guide to help you set up development environment, build and test.

## Feature Support Matrix
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

## Model Support Matrix

The list here is a subset of the supported models. See [supported_models](docs/supported_models.md) for more details:
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

## Contributing
We welcome and value any contributions and collaborations:
- Please let us know if you encounter a bug by [filing an issue](https://github.com/vllm-project/vllm-ascend/issues).
- Please see the guidance on how to contribute in [CONTRIBUTING.md](./CONTRIBUTING.md).

## License

Apache License 2.0, as found in the [LICENSE](./LICENSE) file.
