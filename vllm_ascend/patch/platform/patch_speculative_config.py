from dataclasses import replace

from transformers import DeepseekV2Config, PretrainedConfig
from vllm.config import VllmConfig
from vllm.config.speculative import SpeculativeConfig

_orig_post_init = SpeculativeConfig.__post_init__
_orig_hf_config_override = SpeculativeConfig.hf_config_override
_orig_validate_uno_config = getattr(VllmConfig, "_validate_uno_config", None)


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


def _normalize_deepseek_v4_dspark_draft(draft_model_config) -> None:
    """Restore the DSpark draft architecture after VL config conversion.

    DeepSeek-V4-Vision uses the same checkpoint for the target and DSpark
    drafter.  vLLM first rewrites that checkpoint to ``DSparkDraftModel``, but
    rebuilding ``model_arch_config`` with multimodal detection can restore the
    top-level ``*ForConditionalGeneration`` architecture.  The drafter would
    then instantiate a second full VL target and register duplicate attention
    layer names.  Update both config representations without re-running the
    multimodal architecture conversion.
    """
    hf_config = getattr(draft_model_config, "hf_config", None)
    if (
        hf_config is None
        or getattr(hf_config, "model_type", None) != "deepseek_v4"
        or getattr(hf_config, "dspark_target_layer_ids", None) is None
    ):
        return

    hf_config.update({"architectures": ["DSparkDraftModel"]})
    draft_model_config.model_arch_config = replace(
        draft_model_config.model_arch_config,
        architectures=["DSparkDraftModel"],
        model_type="deepseek_v4",
        is_mm_prefix_lm=False,
    )
    model_info, architecture = draft_model_config.registry.inspect_model_cls(
        draft_model_config.architectures,
        draft_model_config,
    )
    draft_model_config._model_info = model_info
    draft_model_config._architecture = architecture


def _dspark_post_init(self):
    _orig_post_init(self)
    if self.use_dspark():
        draft_model_config = getattr(self, "draft_model_config", None)
        draft_hf_config = getattr(draft_model_config, "hf_config", None)
        _normalize_deepseek_v4_dspark_draft(draft_model_config)
        # deepseek v4 dspark
        if getattr(draft_hf_config, "ptd_token_id", None) is None:  # type: ignore
            draft_hf_config.ptd_token_id = getattr(draft_hf_config, "dspark_noise_token_id", None)  # type: ignore
        # gqa backend dspark
        if getattr(draft_hf_config, "ptd_token_id", None) is None:  # type: ignore
            draft_hf_config.ptd_token_id = getattr(draft_hf_config, "mask_token_id", None)  # type: ignore


SpeculativeConfig.hf_config_override = staticmethod(_normalize_legacy_qwen3_dspark_config)
SpeculativeConfig.__post_init__ = _dspark_post_init


def _validate_ascend_uno_config(self: VllmConfig) -> None:
    """Allow UNO on Ascend MRV2 while retaining the upstream safeguards."""
    speculative_config = self.speculative_config
    if speculative_config is None or not speculative_config.use_uno():
        if _orig_validate_uno_config is not None:
            _orig_validate_uno_config(self)
        return

    if self.scheduler_config.async_scheduling:
        raise ValueError("Uno requires synchronous scheduling (--no-async-scheduling)")
    if self.lora_config is None:
        raise ValueError(
            "Uno requires --enable-lora and --max-lora-rank large enough "
            "for the Uno adapter"
        )
    if self.parallel_config.use_ubatching:
        raise ValueError("Uno does not support dual batch overlap")
    if self.kv_transfer_config is not None:
        raise ValueError("Uno does not support KV cache transfer")
    required_tokens = (
        self.scheduler_config.max_num_seqs
        * speculative_config.num_speculative_tokens
    )
    if self.scheduler_config.max_num_batched_tokens < required_tokens:
        raise ValueError(
            "Uno requires max_num_batched_tokens >= max_num_seqs * "
            f"num_speculative_tokens ({required_tokens}) for draft LoRA routing"
        )


if _orig_validate_uno_config is not None:
    VllmConfig._validate_uno_config = _validate_ascend_uno_config
