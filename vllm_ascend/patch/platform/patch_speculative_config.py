import functools

from transformers import DeepseekV2Config, PretrainedConfig
from vllm.config.speculative import SpeculativeConfig
from vllm.v1.core.sched.scheduler import Scheduler

_orig_post_init = SpeculativeConfig.__post_init__
_orig_hf_config_override = SpeculativeConfig.hf_config_override
_orig_scheduler_init = Scheduler.__init__


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


def _num_drafter_query_tokens(self: SpeculativeConfig) -> int:
    """Return the configured per-request query width of the drafter."""
    num_query_tokens = self.num_speculative_tokens
    assert num_query_tokens is not None
    if self.use_dflash():
        return num_query_tokens + 1
    if not self.use_dspark():
        return num_query_tokens

    assert self.draft_model_config is not None
    sample_from_anchor = getattr(
        self.draft_model_config.hf_config,
        "sample_from_anchor",
        True,
    )
    return num_query_tokens + int(not sample_from_anchor)


@functools.wraps(_orig_scheduler_init)
def _scheduler_init(self: Scheduler, *args, **kwargs) -> None:
    _orig_scheduler_init(self, *args, **kwargs)
    speculative_config = self.vllm_config.speculative_config
    if speculative_config is not None and speculative_config.use_dspark():
        # This remains the configured maximum even when the per-batch K or the
        # per-request verification length is reduced dynamically. Those values
        # trim consumed drafts after the full parallel query has written KV.
        self.num_lookahead_tokens = _num_drafter_query_tokens(speculative_config)


SpeculativeConfig.hf_config_override = staticmethod(_normalize_legacy_qwen3_dspark_config)
SpeculativeConfig.__post_init__ = _dspark_post_init
SpeculativeConfig.num_drafter_query_tokens = property(  # type: ignore[attr-defined]
    _num_drafter_query_tokens
)
Scheduler.__init__ = _scheduler_init
