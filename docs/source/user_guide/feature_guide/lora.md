# LoRA Adapters Guide

## Overview
Like vLLM, vllm-ascend supports LoRA as well. The usage and more details can be found in [vLLM official document](https://docs.vllm.ai/en/latest/features/lora.html).

You can refer to [Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html#list-of-text-only-language-models) to find which models support LoRA in vLLM.

You can run LoRA with ACLGraph mode now. Please refer to [Graph Mode Guide](./graph_mode.md) for a better LoRA performance.

## Example
We provide a simple LoRA example here, which enables the ACLGraph mode by default.

```shell
vllm serve meta-llama/Llama-2-7b \
    --enable-lora \
    --lora-modules '{"name": "sql-lora", "path": "/path/to/lora", "base_model_name": "meta-llama/Llama-2-7b"}'
```

## Custom LoRA Operators

We have implemented LoRA-related AscendC operators, such as bgmv_shrink, bgmv_expand, sgmv_shrink and sgmv_expand. You can find them under the "csrc/kernels" directory of [vllm-ascend repo](https://github.com/vllm-project/vllm-ascend.git).

When you install vllm and vllm-ascend, those operators mentioned above will be compiled and installed automatically. If you do not want to use AscendC operators when you run vllm-ascend, you should set `COMPILE_CUSTOM_KERNELS=0` and reinstall vllm-ascend. To require more instructions about installation and compilation, you can refer to [installation guide](../../installation.md).
