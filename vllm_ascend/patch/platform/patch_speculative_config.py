from vllm.config.speculative import SpeculativeConfig
from vllm.transformers_utils.configs.speculators.algos import SUPPORTED_SPECULATORS_TYPES

_orig_post_init = SpeculativeConfig.__post_init__
_orig_update_dspark = SUPPORTED_SPECULATORS_TYPES["dspark"]


def _ascend_update_dspark(config_dict: dict, pre_trained_config: dict) -> None:
    _orig_update_dspark(config_dict, pre_trained_config)
    mla_fields = (
        "q_lora_rank",
        "kv_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
    )
    is_glm5_dspark = pre_trained_config.get(
        "model_type"
    ) == "glm5_dspark" or "Glm5DSparkForCausalLM" in config_dict.get("architectures", [])
    if is_glm5_dspark and all(pre_trained_config.get(name) is not None for name in mla_fields):
        pre_trained_config["architectures"] = ["Glm5DSparkForCausalLM"]
        pre_trained_config["sliding_window_non_causal"] = config_dict.get("sliding_window_non_causal", True)


def _dspark_post_init(self):
    _orig_post_init(self)
    if self.use_dspark():
        draft_model_config = getattr(self, "draft_model_config", None)
        draft_hf_config = getattr(draft_model_config, "hf_config", None)
        # deepseek v4 dspark
        if getattr(draft_hf_config, "ptd_token_id", None) is None:  # type: ignore
            draft_hf_config.ptd_token_id = getattr(draft_hf_config, "dspark_noise_token_id", None)  # type: ignore
        # gqa backend dspark
        if getattr(draft_hf_config, "ptd_token_id", None) is None:  # type: ignore
            draft_hf_config.ptd_token_id = getattr(draft_hf_config, "mask_token_id", None)  # type: ignore


SUPPORTED_SPECULATORS_TYPES["dspark"] = _ascend_update_dspark
SpeculativeConfig.__post_init__ = _dspark_post_init
