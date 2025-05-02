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
| <a href="https://www.hiascend.com/en/"><b>About Ascend</b></a> | <a href="https://vllm-ascend.readthedocs.io/en/latest/"><b>Documentation</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack (#sig-ascend)</b></a> |
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

- Hardware: Atlas 800I A2 Inference series, Atlas A2 Training series
- Software:
  * Python >= 3.9, < 3.12
  * CANN >= 8.1.rc1
  * PyTorch >= 2.5.1, torch-npu >= 2.5.1
  * vLLM (the same version as vllm-ascend)

Find more about how to setup your environment step by step in [here](docs/source/installation.md).

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

Please refer to [QuickStart](https://vllm-ascend.readthedocs.io/en/latest/quick_start.html) and [Installation](https://vllm-ascend.readthedocs.io/en/latest/installation.html) for more details.

## Contributing
See [CONTRIBUTING](docs/source/developer_guide/contributing.md) for more details, which is a step-by-step guide to help you set up development environment, build and test.

We welcome and value any contributions and collaborations:
- Please feel free comments [here](https://github.com/vllm-project/vllm-ascend/issues/19) about your usage of vLLM Ascend Plugin.
- Please let us know if you encounter a bug by [filing an issue](https://github.com/vllm-project/vllm-ascend/issues).

## Branch

vllm-ascend has main branch and dev branch.

- **main**: main branch，corresponds to the vLLM main branch, and is continuously monitored for quality through Ascend CI.
- **vX.Y.Z-dev**: development branch, created with part of new releases of vLLM. For example, `v0.7.3-dev` is the dev branch for vLLM `v0.7.3` version.

Below is maintained branches:

| Branch     | Status       | Note                                 |
|------------|--------------|--------------------------------------|
| main       | Maintained   | CI commitment for vLLM main branch   |
| v0.7.1-dev | Unmaintained | Only doc fixed is allowed |
| v0.7.3-dev | Maintained   | CI commitment for vLLM 0.7.3 version |

Please refer to [Versioning policy](docs/source/developer_guide/versioning_policy.md) for more details.

## License

Apache License 2.0, as found in the [LICENSE](./LICENSE) file.
