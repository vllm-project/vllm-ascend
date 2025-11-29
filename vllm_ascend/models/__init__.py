from vllm import ModelRegistry


def register_model():
    # There is no PanguProMoEForCausalLM in vLLM, so we should register it before vLLM config initialization
    # to make sure the model can be loaded correctly. This register step can be removed once vLLM support PanguProMoEForCausalLM.
    ModelRegistry.register_model(
        "PanguProMoEForCausalLM",
        "vllm_ascend.torchair.models.torchair_pangu_moe:PanguProMoEForCausalLM"
    )
