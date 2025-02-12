# vLLM昇腾插件
vLLM Ascend plugin (vllm-ascend) is a community maintained hardware plugin for running vLLM on the Ascend NPU.
昇腾插件（vllm-ascend）是一个社区维护的硬件插件，用于在 NPU 上运行 vLLM。

此插件是 vLLM 社区中支持昇腾后端的推荐方式。它遵循[[RFC]: Hardware pluggable](https://github.com/vllm-project/vllm/issues/11162)所述原则：通过解耦的方式提供了vLLM对Ascend NPU的支持。

使用 vLLM 昇腾插件，可以让类Transformer、混合专家(MOE)、嵌入、多模态等流行的大语言模型在 Ascend NPU 上无缝运行。

## 内容

- [快速开始](./quick_start.md)
- [安装](./installation.md)
- Usage
  - [在昇腾运行vLLM](./usage/running_vllm_with_ascend.md)
  - [特性支持](./usage/feature_support.md)
  - [模型支持](./usage/supported_models.md)
