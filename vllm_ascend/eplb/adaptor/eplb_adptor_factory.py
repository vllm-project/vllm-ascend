from vllm_ascend.eplb.adaptor.deepseek_moe_adaptor import DeepSeekMoeAdaptor
from vllm_ascend.eplb.adaptor.qwen_moe_adaptor import QwenMoeAdaptor


class EplbAdaptorFactory:

    adaptor_map = {
        "deepseek_v3,deepseek_v32,deepseek_v2": DeepSeekMoeAdaptor,
        "qwen3_moe": QwenMoeAdaptor
    }

    @classmethod
    def get_eplb_adapator(cls, model):
        model_class = model.model_type
        for key, value in enumerate(cls.adaptor_map):
            support_list = key.strip().split(",")
            if model_class in support_list:
                return value
        raise ValueError(f"Unknown eplb moe model class: {model_class}")