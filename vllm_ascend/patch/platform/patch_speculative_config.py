from dataclasses import dataclass

from pydantic.dataclasses import rebuild_dataclass
from transformers import DeepseekV2Config, PretrainedConfig
from vllm.config import VllmConfig
from vllm.config.speculative import SpeculativeConfig
from vllm.logger import logger

_orig_post_init = SpeculativeConfig.__post_init__
_orig_hf_config_override = SpeculativeConfig.hf_config_override


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
    if getattr(self, "skip_parallel_drafting_seq_lens_override", False):
        logger.warning_once(
            "skip_parallel_drafting_seq_lens_override is enabled: parallel "
            "drafting attention will preserve host-side sequence lengths "
            "instead of using the device-side sequence-length buffers. Enable "
            "this only when the draft model requires host-side metadata."
        )
    if self.use_dspark():
        draft_model_config = getattr(self, "draft_model_config", None)
        draft_hf_config = getattr(draft_model_config, "hf_config", None)
        # deepseek v4 dspark
        if getattr(draft_hf_config, "ptd_token_id", None) is None:  # type: ignore
            draft_hf_config.ptd_token_id = getattr(draft_hf_config, "dspark_noise_token_id", None)  # type: ignore
        # gqa backend dspark
        if getattr(draft_hf_config, "ptd_token_id", None) is None:  # type: ignore
            draft_hf_config.ptd_token_id = getattr(draft_hf_config, "mask_token_id", None)  # type: ignore


@dataclass
class _AscendSpeculativeConfigFields:
    skip_parallel_drafting_seq_lens_override: bool = False


def _add_ascend_speculative_config_fields() -> None:
    """Add plugin-owned fields before speculative config validation runs."""
    field_name = "skip_parallel_drafting_seq_lens_override"
    if field_name in SpeculativeConfig.__dataclass_fields__:
        return

    SpeculativeConfig.__annotations__[field_name] = bool
    SpeculativeConfig.__dataclass_fields__[field_name] = _AscendSpeculativeConfigFields.__dataclass_fields__[field_name]
    setattr(SpeculativeConfig, field_name, False)
    rebuild_dataclass(SpeculativeConfig, force=True)
    # VllmConfig may already have cached the nested SpeculativeConfig schema.
    rebuild_dataclass(VllmConfig, force=True)


SpeculativeConfig.hf_config_override = staticmethod(_normalize_legacy_qwen3_dspark_config)
SpeculativeConfig.__post_init__ = _dspark_post_init
_add_ascend_speculative_config_fields()
