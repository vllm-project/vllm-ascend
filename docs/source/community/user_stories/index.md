# User stories

Read case studies on how users and developers solve real, everyday problems with vLLM Ascend

- [LLaMA-Factory](./llamafactory.md) is an easy-to-use and efficient platform for training and fine-tuning large language models. It supports vLLM Ascend to speed up inference since [LLaMA-Factory#7739](https://github.com/hiyouga/LLaMA-Factory/pull/7739), gaining 2x performance enhancement in inference.

- [Huggingface/trl](https://github.com/huggingface/trl) is a cutting-edge library designed for post-training foundation models using advanced techniques like SFT, PPO and DPO. It uses vLLM Ascend since [v0.17.0](https://github.com/huggingface/trl/releases/tag/v0.17.0) to support RLHF on Ascend NPUs.

- [MindIE Turbo](https://pypi.org/project/mindie-turbo) is an LLM inference engine acceleration plugin library developed by Huawei on Ascend hardware, which includes self-developed LLM optimization algorithms and optimizations related to the inference engine framework. It supports vLLM Ascend since [2.0rc1](https://www.hiascend.com/document/detail/zh/mindie/20RC1/AcceleratePlugin/turbodev/mindie-turbo-0001.html).

- [GPUStack](https://github.com/gpustack/gpustack) is an open-source GPU cluster manager for running AI models. It supports vLLM Ascend since [v0.6.2](https://github.com/gpustack/gpustack/releases/tag/v0.6.2). See more GPUStack performance evaluation information at [this link](https://mp.weixin.qq.com/s/pkytJVjcH9_OnffnsFGaew).

- [verl](https://github.com/volcengine/verl) is a flexible, efficient, and production-ready RL training library for LLMs. It uses vLLM Ascend since [v0.4.0](https://github.com/volcengine/verl/releases/tag/v0.4.0). See more information on [verl x Ascend Quickstart](https://verl.readthedocs.io/en/latest/ascend_tutorial/ascend_quick_start.html).

:::{toctree}
:caption: More details
:maxdepth: 1
llamafactory
:::
