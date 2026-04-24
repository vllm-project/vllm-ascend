# mypy: ignore-errors
import vllm
import vllm.model_executor.models.config


@property
def is_deepseek_mla(self) -> bool:
    if not hasattr(self.hf_text_config, "model_type"):
        return False
    elif self.hf_text_config.model_type in (
            "deepseek_v2",
            "deepseek_v3",
            "deepseek_v32",
            "deepseek_v4",
            "deepseek_mtp",
            "kimi_k2",
            "kimi_linear",
            "longcat_flash",
            "pangu_ultra_moe",
            "pangu_ultra_moe_mtp",
    ):
        return self.hf_text_config.kv_lora_rank is not None
    elif self.hf_text_config.model_type == "eagle":
        # if the model is an EAGLE module, check for the
        # underlying architecture
        return (self.hf_text_config.model.model_type
                in ("deepseek_v2", "deepseek_v3", "deepseek_v32",
                    "deepseek_v4")
                and self.hf_text_config.kv_lora_rank is not None)
    return False


vllm.config.model.ModelConfig.is_deepseek_mla = is_deepseek_mla
