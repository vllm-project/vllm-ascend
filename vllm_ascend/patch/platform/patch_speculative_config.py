from vllm.config.speculative import SpeculativeConfig

_orig_post_init = SpeculativeConfig.__post_init__


def _dspark_post_init(self):
    _orig_post_init(self)
    if self.use_dspark():
        draft_model_config = getattr(self, "draft_model_config", None)
        draft_hf_config = getattr(draft_model_config, "hf_config", None)
        if (
            draft_model_config is not None
            and draft_hf_config is not None
            and getattr(draft_hf_config, "model_type", None) == "glm5_dspark"
        ):
            draft_hf_config.architectures = ["Glm5DSparkForCausalLM"]
            self.update_arch_()
            draft_model_config.model_arch_config.is_deepseek_mla = True
        # deepseek v4 dspark
        if getattr(draft_hf_config, "ptd_token_id", None) is None:  # type: ignore
            draft_hf_config.ptd_token_id = getattr(draft_hf_config, "dspark_noise_token_id", None)  # type: ignore
        # gqa backend dspark
        if getattr(draft_hf_config, "ptd_token_id", None) is None:  # type: ignore
            draft_hf_config.ptd_token_id = getattr(draft_hf_config, "mask_token_id", None)  # type: ignore


SpeculativeConfig.__post_init__ = _dspark_post_init
