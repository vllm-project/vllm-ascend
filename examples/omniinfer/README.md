Reference implementation for distributed inference with vLLM + Omni-Infer
-----------

This reference implementation is still **EXPERIMENTAL**. Its interface, functionality, and performance may change during the final integration into vLLM and vLLM Ascend.

We are actively working with `Omni-infer` team to contribute `Omni-Infer` features to vLLM and vLLM Ascend (Ascend Plugin) upstream:

- [vLLM](https://github.com/vllm-project/vllm): A high-throughput and memory-efficient inference and serving engine for LLMs.
- [vLLM Ascend](https://github.com/vllm-project/vllm-ascend)：vLLM Ascend plugin (vllm-ascend) is a community maintained hardware plugin for running vLLM seamlessly on the Ascend NPU, include vLLM Ascend backend implementation.
- [Omni-Infer](https://gitee.com/omniai/omniinfer): Inference accelerators for distributed inference (PD Disaggregated) With Large Scale Expert Parallelism. Includes v0.9.0 vLLM + Omni-Infer deployment tool/acceleration library. Omni-infer uses `omni-cli` and `ansible` tools to simplify cluster deployment.

This example demonstrates best practice for efficient inference of large language models on Ascend NPUs. We will gradually submit `Omni-Infer` code pull requests (PRs) to upstream (vLLM and vLLM Ascend) to provide a preliminary usable version for users and developers.

You can find the progress in the following issues: [vllm-ascend/issues/3165](https://github.com/vllm-project/vllm-ascend/issues/3165), below are the initial steps:

- Integrate `Omni-Infer` to vLLM and vLLM Ascend v0.9.1-dev branch under example path as a reference.
- Follow the architectural principles of vLLM and vLLM Ascend and gradually migrate `Omni-Infer` features main branch.

---------------

此参考样例仍然在**实验阶段**，尚未合并到主分支。其接口、功能和性能可能会在最终集成到vLLM和vLLM Ascend过程中发生变化。

我们正在与Omni-infer团队积极合作，将`Omni-Infer`特性合入vLLM和vLLM Ascend（Ascend插件）上游：

- [vLLM](https://github.com/vllm-project/vllm)：一个高吞吐量和内存高效的LLM推理和服务引擎。
- [vLLM Ascend](https://github.com/vllm-project/vllm-ascend)：vLLM Ascend插件（vllm-ascend）是一个社区维护的硬件插件，用于在昇腾NPU上运行vLLM，包括vLLM Ascend后端实现。
- [Omni-Infer](https://gitee.com/omniai/omniinfer)：用于分布式(PD分离)推理的大规模专家并行推理加速器，包括v0.9.0 vLLM + Omni-Infer部署工具/加速库。Omni-infer 使用 `omni-cli` 和 `ansible` 工具来简化集群部署，最终使用vLLM进行推理。

本参考样例展示了如何在昇腾NPU上高效地进行大语言模型推理。可以预期的是，我们将逐步将`Omni-Infer`代码提交PR至上游 (vLLM、vLLM Ascend)，为用户和开发者提供一个初步可用版本。 

您可以在以下问题中找到最新进展：[vllm-ascend/issues/3165](https://github.com/vllm-project/vllm-ascend/issues/3165),下面是初始步骤：

- 第一步，将Omni-Infer合入到vLLM和vLLM Ascend v0.9.1-dev 开发分支的example路径下作为参考。
- 第二步，遵循vLLM和vLLM Ascend的架构原则，逐步将Omni-Infer合入，提交PR合入上游主干 (vLLM、vLLM Ascend)，由社区的开发者Review后，合入代码到主干分支。