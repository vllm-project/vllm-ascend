# LoRA Adapters

Like vLLM, vllm-scend supports LoRA as well. The usage and more details can be found in [vLLM official document](https://docs.vllm.ai/en/latest/features/lora.html).

You can also refer to [this](https://docs.vllm.ai/en/latest/models/supported_models.html#list-of-text-only-language-models) to find which models support LoRA in vLLM.

## Tips
If you fail to run vllm-ascend with LoRA, you may follow [this instruction](https://vllm-ascend.readthedocs.io/en/latest/user_guide/graph_mode.html#fallback-to-eager-mode) to disable graph mode and try again.
