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
| <a href="https://www.hiascend.com/en/"><b>关于昇腾</b></a> | <a href="https://docs.vllm.ai/projects/ascend/en/latest/"><b>官方文档</b></a> | <a href="https://slack.vllm.ai"><b>#sig-ascend</b></a> | <a href="https://discuss.vllm.ai/c/hardware-support/vllm-ascend-support"><b>用户论坛</b></a> | <a href="https://tinyurl.com/vllm-ascend-meeting"><b>社区例会</b></a> |
</p>

---
*消息* 🔥

- [2026/10] 正式创建了[v0.30.0版本](https://github.com/vllm-router/vllm-ascend/tree/main)，可以部署vLLM边云协同功能。
- [2026/10] 正式创建了[v0.23.0云侧复用版本](https://github.com/vllm-data-comm/vllm-ascend/tree/release/v0.23.0_share-cloud_T)，可以部署vLLM边云协同的云侧复用功能。
- [2026/10] 正式创建了[v0.23.0多实例版本](https://github.com/vllm-data-comm/vllm-ascend/tree/release/v0.23.0_share-edge_T)，可以部署vLLM边云协同的多实例功能。
- [2026/10] 正式创建了[v0.23.0基础版本](https://github.com/vllm-data-comm/vllm-ascend/tree/release/v0.23.0)，可以部署vLLM一边一云的边云协同基础功能。
- [2026/09] 正式创建了[vllm-data-comm/vllm-ascend](https://github.com/vllm-data-comm/vllm-ascend)仓库，让vLLM边云协同可以无缝运行在Ascend NPU。

<details>
<summary>更多内容</summary>

- [2026/10] 正式创建了[v0.30.0版本](https://github.com/vllm-router/vllm-ascend/tree/main)，可以部署vLLM边云协同功能。
- [2026/10] 正式创建了[v0.23.0云侧复用版本](https://github.com/vllm-data-comm/vllm-ascend/tree/release/v0.23.0_share-cloud_T)，可以部署vLLM边云协同的云侧复用功能。
- [2026/10] 正式创建了[v0.23.0多实例版本](https://github.com/vllm-data-comm/vllm-ascend/tree/release/v0.23.0_share-edge_T)，可以部署vLLM边云协同的多实例功能。
- [2026/10] 正式创建了[v0.23.0基础版本](https://github.com/vllm-data-comm/vllm-ascend/tree/release/v0.23.0)，可以部署vLLM一边一云的边云协同基础功能。
- [2026/09] 正式创建了[vllm-data-comm/vllm-ascend](https://github.com/vllm-data-comm/vllm-ascend)仓库，让vLLM边云协同可以无缝运行在Ascend NPU。
- 
</details>

---

## 总览

vLLM 昇腾插件 (`vllm-ascend`) 是一个由社区维护的让vLLM在Ascend NPU无缝运行的后端插件。

此插件是 vLLM 社区中支持昇腾后端的推荐方式。它遵循[[RFC]: Hardware pluggable](https://github.com/vllm-project/vllm/issues/11162)所述原则：通过解耦的方式提供了vLLM对Ascend NPU的支持。

使用 vLLM 昇腾插件，可以让类Transformer、混合专家(MOE)、嵌入、多模态等流行的大语言模型在 Ascend NPU 上无缝运行。


## 准备

- 硬件：Atlas 800I A2系列、Atlas 800I A3 I系列
- 操作系统：Linux
- 软件：
    - Python == 3.12 
    - CANN == 9.2.1 (Ascend HDK 版本详见 版本说明)
    - PyTorch == 2.10.0, TorchNPU == 2.10.0.post7 
    - vLLM (与vllm-ascend版本一致)

## 开始使用

推荐您使用以下版本快速开始使用：

| Version    | Release type | Doc                                                    |
|------------|--------------|--------------------------------------------------------|
| v0.30.0 | 最新RC版本   | 请查看[快速入门](docs/cloud-edge/quick_start.md)和[安装指南](docs/cloud-edge/install.md)了解更多 |

## 支持模型
图例说明
- ✅ = 支持的模型/功能
- 🔵 = 实验性支持的模型/功能
- ❌ = 不支持的模型/功能
- 🟡 = 未经测试或验证

| 模型              | 支持情况 | 备注 | BF16 | 支持的硬件 | W8A8 | 分<br/>块<br/>预<br/>填<br/>充 | 自<br/>动<br/>前<br/>缀<br/>缓<br/>存 | LoRA | 投<br/>机<br/>解<br/>码 | 异<br/>步<br/>调<br/>度 | 张<br/>量<br/>并<br/>行 | 流<br/>水<br/>线<br/>并<br/>行 | 专<br/>家<br/>并<br/>行 | 数<br/>据<br/>并<br/>行 | 与填充<br/>-<br/>解码分离 | 分段<br/>AclGraph | 全图<br/>AclGraph | max-model-len | MLP<br/>权重<br/>预取 |
|-------------------|----------|------|------|------------|------|--------------------------------|---------------------------------------|------|-------------------------|-------------------------|-------------------------|--------------------------------|-------------------------|-------------------------|---------------------------|-------------------|-------------------|---------------|-----------------------|
| DeepSeek V4-Flash | ✅       |      | ✅   | A2/A3      | ✅   | ✅                             | ✅                                    |      | ✅                      | ✅                      | ✅                      |                                | ✅                      | ✅                      | ✅                        |                   | ✅                | 1M            |                       |
| GLM-5.2           | 🔵       |      | ✅   | A2/A3      | ✅   | ✅                             | ✅                                    |      | ✅                      | ✅                      | ✅                      | ✅                             | ✅                      | ✅                      | ✅                        | ✅                | ✅                | 200k          |                       |
[更多](docs/cloud-edge/model_support_lists.md)

## 贡献

请参考[CONTRIBUTING](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/contribution/index.html)文档了解更多关于开发环境搭建、功能测试以及 PR 提交规范的信息。

我们欢迎并重视任何形式的贡献与合作：

- 请通过[Issue](https://github.com/vllm-project/vllm-ascend/issues)来告知我们您遇到的任何Bug。
- 请通过[用户论坛](https://discuss.vllm.ai/c/hardware-support/vllm-ascend-support)来交流使用问题和寻求帮助。

## 社区例会

- vLLM Ascend 每周社区例会: <https://tinyurl.com/vllm-ascend-meeting>
- 每周三下午，15:00 - 16:00 (UTC+8, [查看您的时区](https://dateful.com/convert/gmt8?t=15))

## 许可证

Apache 许可证 2.0，如 [LICENSE](LICENSE) 文件中所示。
