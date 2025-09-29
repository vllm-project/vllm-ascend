from vllm.config import ModelConfig


# mypy: ignore-errors
@property
def is_deepseek_mla(self: ModelConfig):
    if not hasattr(self.hf_text_config, "model_type"):
        return False
    elif self.hf_text_config.model_type in \
        ('deepseek_v2', 'deepseek_v3', 'deepseek_mtp',
            'kimi_k2', 'longcat_flash', 'deepseek_v32'):
        return self.hf_text_config.kv_lora_rank is not None
    elif self.hf_text_config.model_type == 'eagle':
        # if the model is an EAGLE module, check for the
        # underlying architecture
        return self.hf_text_config.model.model_type in \
                ('deepseek_v2', 'deepseek_v3', 'deepseek_v32') \
            and self.hf_text_config.kv_lora_rank is not None
    return False


ModelConfig.is_deepseek_mla = is_deepseek_mla
