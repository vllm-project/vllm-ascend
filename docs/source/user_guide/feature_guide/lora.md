# LoRA Adapters Guide

## Overview

Like vLLM, vllm-ascend supports LoRA as well. The usage and more details can be found in [vLLM official document](https://docs.vllm.ai/en/latest/features/lora.html).

You can refer to [Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html#list-of-text-only-language-models) to find which models support LoRA in vLLM.

You can run LoRA with ACLGraph mode now. Please refer to [Graph Mode Guide](./graph_mode.md) for a better LoRA performance.

Address for downloading models:

- base model: <https://www.modelscope.cn/models/vllm-ascend/Llama-2-7b-hf/files>
- lora model: <https://www.modelscope.cn/models/vllm-ascend/llama-2-7b-sql-lora-test/files>

## Example

We provide a simple LoRA example here, which enables the ACLGraph mode by default.

```shell
vllm serve meta-llama/Llama-2-7b \
    --enable-lora \
    --lora-modules '{"name": "sql-lora", "path": "/path/to/lora", "base_model_name": "meta-llama/Llama-2-7b"}'
```

## Custom LoRA Operators

We have implemented LoRA-related AscendC operators, such as bgmv_shrink, bgmv_expand, sgmv_shrink and sgmv_expand. You can find them under the "csrc/kernels" directory of [vllm-ascend repo](https://github.com/vllm-project/vllm-ascend.git).
