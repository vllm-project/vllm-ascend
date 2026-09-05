from transformers import DeepseekV2Config, PretrainedConfig
from vllm.config.speculative import SpeculativeConfig

_orig_post_init = SpeculativeConfig.__post_init__
_orig_hf_config_override = SpeculativeConfig.hf_config_override

_UPSTREAM_K3_DSPARK_DCP_ERROR_FRAGMENT = (
    "MLA DSpark does not currently support decode context parallelism"
)


# Transformers 5.14 inherited a hidden_size % num_heads check from Llama in
# DeepseekV2Config. K3 MLA has independent projection/head dimensions (e.g.
# hidden_size=7168, num_heads=96), so that MHA constraint does not apply.
# strict stores unbound validators; patch that entry, not all config validation.
if hasattr(DeepseekV2Config, "__class_validators__"):
    _orig_validate_architecture = DeepseekV2Config.validate_architecture

    def _validate_dspark_architecture(config):
        if config.model_type != "k3_dspark":
            _orig_validate_architecture(config)

    DeepseekV2Config.__class_validators__ = [
        _validate_dspark_architecture if validator is _orig_validate_architecture else validator
        for validator in DeepseekV2Config.__class_validators__
    ]


def _normalize_legacy_qwen3_dspark_config(hf_config: PretrainedConfig) -> PretrainedConfig:
    hf_config = _orig_hf_config_override(hf_config)
    architectures = hf_config.architectures or ()
    if hf_config.model_type == "qwen3" and "DSparkDraftModel" in architectures:
        dflash_config = hf_config.dflash_config
        hf_config.update(
            {
                "architectures": ["Qwen3DSparkModel"],
                "mask_token_id": dflash_config["mask_token_id"],
                "target_layer_ids": dflash_config["target_layer_ids"],
            }
        )
    return hf_config


def _is_ascend_k3_dspark_dcp(self: SpeculativeConfig) -> bool:
    """Return whether this is the K3 DSpark+DCP case supported by Ascend."""
    target_parallel_config = getattr(self, "target_parallel_config", None)
    draft_model_config = getattr(self, "draft_model_config", None)
    architectures = getattr(draft_model_config, "architectures", None) or ()
    return (
        getattr(self, "method", None) == "dspark"
        and "K3DSparkModel" in architectures
        and getattr(target_parallel_config, "decode_context_parallel_size", 1) > 1
    )


def _dspark_post_init(self):
    try:
        _orig_post_init(self)
    except ValueError as error:
        # Upstream rejects K3 MLA DSpark+DCP for GPU backends. Ascend has its
        # own DCP metadata and MLA implementation, so bypass only the matching
        # upstream guard. Keep all other SpeculativeConfig validation intact.
        if (
            _UPSTREAM_K3_DSPARK_DCP_ERROR_FRAGMENT not in str(error)
            or not _is_ascend_k3_dspark_dcp(self)
        ):
            raise
    if self.use_dspark():
        draft_model_config = getattr(self, "draft_model_config", None)
        draft_hf_config = getattr(draft_model_config, "hf_config", None)
        # deepseek v4 dspark
        if getattr(draft_hf_config, "ptd_token_id", None) is None:  # type: ignore
            draft_hf_config.ptd_token_id = getattr(draft_hf_config, "dspark_noise_token_id", None)  # type: ignore
        # gqa backend dspark
        if getattr(draft_hf_config, "ptd_token_id", None) is None:  # type: ignore
            draft_hf_config.ptd_token_id = getattr(draft_hf_config, "mask_token_id", None)  # type: ignore


SpeculativeConfig.hf_config_override = staticmethod(_normalize_legacy_qwen3_dspark_config)
SpeculativeConfig.__post_init__ = _dspark_post_init
