import json
from pathlib import Path

from safetensors import safe_open
from transformers import DeepseekV2Config, PretrainedConfig
from vllm.config.speculative import SpeculativeConfig

_SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"
_QWEN3_5_MTP_WEIGHT_PREFIX = "mtp."

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


def _checkpoint_has_qwen3_5_mtp_weights(model_path: str | Path | None) -> bool | None:
    """Inspect a local safetensors checkpoint without loading tensor data."""
    if not model_path:
        return None

    checkpoint_dir = Path(model_path)
    if not checkpoint_dir.is_dir():
        return None

    index_path = checkpoint_dir / _SAFE_WEIGHTS_INDEX_NAME
    if index_path.is_file():
        try:
            with index_path.open(encoding="utf-8") as index_file:
                index = json.load(index_file)
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(index, dict):
            return None
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            return None
        return any(name.startswith(_QWEN3_5_MTP_WEIGHT_PREFIX) for name in weight_map)

    weight_files = sorted(checkpoint_dir.glob("*.safetensors"))
    if not weight_files:
        return None

    try:
        for weight_file in weight_files:
            with safe_open(
                str(weight_file),
                framework="pt",
                device="cpu",
            ) as weights:
                tensor_names = weights.keys()
                if any(name.startswith(_QWEN3_5_MTP_WEIGHT_PREFIX) for name in tensor_names):
                    return True
    except Exception:
        return None
    return False


def _validate_qwen3_5_mtp_checkpoint(speculative_config: SpeculativeConfig) -> None:
    if speculative_config.method != "qwen3_5_mtp":
        return

    draft_model_config = getattr(speculative_config, "draft_model_config", None)
    model_path = getattr(draft_model_config, "model", None)
    if _checkpoint_has_qwen3_5_mtp_weights(model_path) is False:
        raise ValueError(
            "qwen3_5_mtp speculative decoding was requested, but the local "
            f"checkpoint {model_path!r} does not contain any 'mtp.*' tensors. "
            "SFT exports often save only the language-model backbone and drop "
            "the pretrained MTP head. Remove the qwen3_5_mtp speculative config, "
            "or export/merge the original MTP tensors before serving."
        )


def _speculative_config_post_init(self):
    _orig_post_init(self)
    _validate_qwen3_5_mtp_checkpoint(self)
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
SpeculativeConfig.__post_init__ = _speculative_config_post_init
