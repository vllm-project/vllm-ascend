from vllm import ModelRegistry


def register_model():
    from .qwen2_vl import CustomQwen2VLForConditionalGeneration  # noqa: F401

    ModelRegistry.register_model(
        "Qwen2VLForConditionalGeneration",
        "vllm_ascend.models.qwen2_vl:CustomQwen2VLForConditionalGeneration")
