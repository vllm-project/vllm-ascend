from vllm_ascend.eplb.adaptor.deepseek_moe_adaptor import DeepSeekMoeAdaptor
from vllm_ascend.eplb.adaptor.qwen_moe_adaptor import QwenMoeAdaptor
from vllm_ascend.eplb.adaptor.vllm_adaptor import VllmEplbAdaptor

class EplbAdaptorFactory:

    _ADAPTOR_MAP = {
        "deepseek_v3": DeepSeekMoeAdaptor,
        "deepseek_v32": DeepSeekMoeAdaptor,
        "deepseek_v2": DeepSeekMoeAdaptor,
        "qwen3_moe": QwenMoeAdaptor,
    }

    @classmethod
    def get_eplb_adapator(cls, model_config) -> VllmEplbAdaptor:
        model_type = getattr(model_config.hf_config, "model_type",
                             None)
        if model_type is None:
            raise ValueError(
                "model_type not found in model_config.hf_config")

        adaptor_class = cls._ADAPTOR_MAP.get(model_type)
        if adaptor_class:
            return adaptor_class
        raise ValueError(f"Unknown eplb moe model type: {model_type}")
