from vllm import ModelRegistry

import vllm_ascend.envs as envs


def register_model():
    from .llama_eagle3_vwn import CustomQwen3ForCausalLM  # noqa: F401

    ModelRegistry.register_model(
        "LlamaForCausalLMVwnEagle3", "vllm_ascend.models.llama_eagle3_vwn:Eagle3VwnLlamaForCausalLM")