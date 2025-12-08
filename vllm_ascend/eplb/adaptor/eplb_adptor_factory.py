from vllm_ascend.eplb.adaptor.deepseek_moe_adaptor import DeepSeekMoeAdaptor
from vllm_ascend.eplb.adaptor.qwen_moe_adaptor import QwenMoeAdaptor


class EplbAdaptorFactory:

    adaptor_map = {
        "deepseek_v3,deepseek_v32,deepseek_v2": DeepSeekMoeAdaptor,
        "qwen3_moe": QwenMoeAdaptor,
    }

    @classmethod
    def get_eplb_adapator(cls, vllm_config):
        model_type = getattr(vllm_config.model_config.hf_config, "model_type", None)
        assert model_type is not None
        for key, value in cls.adaptor_map.items():
            support_list = [s.strip() for s in key.split(",")]
            if model_type in support_list:
                return value
        raise ValueError(f"Unknown eplb moe model type: {model_type}")