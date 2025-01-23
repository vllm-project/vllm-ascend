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

---
*Latest News* 🔥

- [2024/12] We are working with the vLLM community to support [[RFC]: Hardware pluggable](https://github.com/vllm-project/vllm/issues/11162).
---
## Overview

`vllm-ascend` is a backend plugin for running vLLM on the Ascend NPU.

This plugin is the recommended approach for supporting the Ascend backend within the vLLM community. It adheres to the principles outlined in the [[RFC]: Hardware pluggable](https://github.com/vllm-project/vllm/issues/11162), providing a hardware-pluggable interface that decouples the integration of the Ascend NPU with vLLM.

By using `vllm-ascend`, popular open-source models, including Transformer-like, Mixture-of-Expert, Embedding, Multi-modal LLMs can run seamlessly on the Ascend NPU.

## Prerequisites
### Support Devices
- Atlas A2 Training series (Atlas 800T A2, Atlas 900 A2 PoD, Atlas 200T A2 Box16, Atlas 300T A2)
- Atlas 800I A2 Inference series (Atlas 800I A2)

### Dependencies
| Requirement  | Supported version | Recommended version | Note |
| ------------ | ------- | ----------- | ----------- | 
| Python | >= 3.9 | [3.10](https://www.python.org/downloads/) | Required for vllm |
| CANN         | >= 8.0.RC2 | [8.0.RC3](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.0.0.beta1) | Required for vllm-ascend and torch-npu |
| torch-npu    | >= 2.4.0   | [2.5.1rc1](https://gitee.com/ascend/pytorch/releases/tag/v6.0.0.alpha001-pytorch2.5.1)    | Required for vllm-ascend |
| torch        | >= 2.4.0   | [2.5.1](https://github.com/pytorch/pytorch/releases/tag/v2.5.1)      | Required for torch-npu and vllm required |

Find more about how to setup your environment in [here](docs/environment.md).

## Getting Started  

> [!NOTE]
> Currently, we are actively collaborating with the vLLM community to support the Ascend backend plugin, once supported we use one line command `pip install vllm vllm-ascend` to compelete installation.

Installation from source code:
```bash
# Install vllm main branch according:
# https://docs.vllm.ai/en/latest/getting_started/installation/cpu/index.html#build-wheel-from-source
git clone https://github.com/vllm-project/vllm.git
cd vllm
sudo apt-get update  -y
sudo apt-get install -y gcc-12 g++-12 libnuma-dev
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 --slave /usr/bin/g++ g++ /usr/bin/g++-12
pip install cmake>=3.26 wheel packaging ninja "setuptools-scm>=8" numpy
pip install -r requirements-cpu.txt
VLLM_TARGET_DEVICE=cpu python setup.py install

# Install vllm-ascend main branch
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
pip install -e .
```

Run the following command to start the vLLM server with the [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) model:

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct
curl http://localhost:8000/v1/models
```

Find more details in the [vLLM Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html).

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
docker build -t vllm-ascend-dev -f ./Dockerfile .
```

## Contributing

We welcome and value any contributions and collaborations, here is a quick note before you submit a PR:

```
# Downloading and install dev requirements
git clone https://github.com/vllm-project/vllm-ascend
pip install -r requirements-dev.txt

# Linting and formatting
bash format.sh
```

Find more details in the [CONTRIBUTING.md](./CONTRIBUTING.md).
