from typing import TYPE_CHECKING, Any

from vllm.config.speculative import SpeculativeConfig
from vllm.transformers_utils.configs.speculators import algos as speculator_algos
from vllm.utils.import_utils import LazyLoader

from vllm_ascend.transformers_utils.configs.kimi_k3 import (
    K3DSparkConfig,
    register_k3_dspark_config,
)

_orig_post_init = SpeculativeConfig.__post_init__

if TYPE_CHECKING:
    import vllm.model_executor.layers.quantization as me_quant
    from transformers import PretrainedConfig
else:
    PretrainedConfig = Any

    me_quant = LazyLoader("model_executor", globals(), "vllm.model_executor.layers.quantization")


# This patch may be imported before the model plugin entry point runs. Register
# the lightweight config here as well so ModelConfig can parse the draft before
# speculative post-init normalizes its architecture.
register_k3_dspark_config()


# Backport vLLM #48639. v0.26 unconditionally rewrites Speculators DSpark
# checkpoints to the legacy bonus-anchor layout and drops the checkpoint's
# sample_from_anchor field before hf_config_override() runs.
_orig_update_dspark = speculator_algos.SUPPORTED_SPECULATORS_TYPES["dspark"]


def _update_dspark(config_dict: dict, pre_trained_config: dict) -> None:
    _orig_update_dspark(config_dict, pre_trained_config)
    aux_layer_ids = config_dict["aux_hidden_state_layer_ids"]
    pre_trained_config["dflash_config"] = {
        "mask_token_id": config_dict["mask_token_id"],
        "target_layer_ids": [i - 1 for i in aux_layer_ids],
    }
    pre_trained_config["sample_from_anchor"] = config_dict.get("sample_from_anchor", False)


speculator_algos.SUPPORTED_SPECULATORS_TYPES["dspark"] = _update_dspark


def hf_config_override(hf_config: PretrainedConfig) -> PretrainedConfig:
    initial_architecture = hf_config.architectures[0]
    if initial_architecture == "DSparkDraftModel" and hf_config.model_type == "qwen3":
        # Legacy Qwen3/GQA DSpark checkpoints keep the inference-only fields
        # under dflash_config and use the training-time architecture name.
        # Normalize those values before vLLM inspects the model registry.
        dflash_config = getattr(hf_config, "dflash_config", None) or {}

        def get_dflash_value(name: str) -> Any:
            if isinstance(dflash_config, dict):
                return dflash_config.get(name)
            return getattr(dflash_config, name, None)

        updates: dict[str, Any] = {"architectures": ["Qwen3DSparkModel"]}
        for name in ("mask_token_id", "target_layer_ids"):
            if (value := get_dflash_value(name)) is not None:
                updates[name] = value
        hf_config.update(updates)

    if hf_config.model_type in ("deepseek_v3", "deepseek_v32", "deepseek_v4", "glm_moe_dsa"):
        target_model_type = hf_config.model_type
        hf_config.model_type = "deepseek_mtp"
    if hf_config.model_type == "deepseek_mtp":
        if target_model_type == "deepseek_v4":
            hf_config.update({"architectures": ["DeepSeekV4MTPModel"]})
        else:
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update({"n_predict": n_predict, "architectures": ["DeepSeekMTPModel"]})
    if hf_config.model_type in ("pangu_ultra_moe"):
        hf_config.model_type = "pangu_ultra_moe_mtp"
    if hf_config.model_type == "pangu_ultra_moe_mtp":
        n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
        hf_config.update({"n_predict": n_predict, "architectures": ["OpenPanguMTPModel"]})

    if hf_config.architectures[0] == "MiMoForCausalLM":
        hf_config.model_type = "mimo_mtp"
        n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
        hf_config.update(
            {
                "num_hidden_layers": 0,
                "n_predict": n_predict,
                "architectures": ["MiMoMTPModel"],
            }
        )

    if hf_config.architectures[0] == "Glm4MoeForCausalLM":
        hf_config.model_type = "glm4_moe_mtp"
        n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
        hf_config.update(
            {
                "n_predict": n_predict,
                "architectures": ["Glm4MoeMTPModel"],
            }
        )

    if hf_config.architectures[0] == "Glm4MoeLiteForCausalLM":
        hf_config.model_type = "glm4_moe_lite_mtp"
        n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
        hf_config.update(
            {
                "num_hidden_layers": 0,
                "n_predict": n_predict,
                "architectures": ["Glm4MoeLiteMTPModel"],
            }
        )

    if hf_config.architectures[0] == "GlmOcrForConditionalGeneration":
        hf_config.model_type = "glm_ocr_mtp"
        n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
        hf_config.update(
            {
                "num_hidden_layers": 0,
                "n_predict": n_predict,
                "architectures": ["GlmOcrMTPModel"],
            }
        )

    if hf_config.model_type == "ernie4_5_moe":
        hf_config.model_type = "ernie_mtp"
    if hf_config.model_type == "ernie_mtp":
        n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
        hf_config.update({"n_predict": n_predict, "architectures": ["ErnieMTPModel"]})

    if (
        hf_config.model_type == "nemotron_h"
        and hasattr(hf_config, "num_nextn_predict_layers")
        and hf_config.num_nextn_predict_layers > 0
    ):
        # Check if this is an MTP variant
        hf_config.model_type = "nemotron_h_mtp"
    if hf_config.model_type == "nemotron_h_mtp":
        n_predict = getattr(hf_config, "num_nextn_predict_layers", 1)
        hf_config.update({"n_predict": n_predict, "architectures": ["NemotronHMTPModel"]})

    if hf_config.model_type == "qwen3_next":
        hf_config.model_type = "qwen3_next_mtp"
    if hf_config.model_type == "qwen3_next_mtp":
        n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
        hf_config.update({"n_predict": n_predict, "architectures": ["Qwen3NextMTP"]})

    if hf_config.model_type == "exaone_moe":
        hf_config.model_type = "exaone_moe_mtp"
    if hf_config.model_type == "exaone_moe_mtp":
        n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
        hf_config.update({"n_predict": n_predict, "architectures": ["ExaoneMoeMTP"]})

    if hf_config.model_type in ("qwen3_5", "qwen3_5_moe"):
        is_moe = hf_config.model_type == "qwen3_5_moe"
        hf_config.model_type = "qwen3_5_mtp"
        n_predict = getattr(hf_config, "mtp_num_hidden_layers", None)
        hf_config.update(
            {
                "n_predict": n_predict,
                "architectures": ["Qwen3_5MoeMTP" if is_moe else "Qwen3_5MTP"],
            }
        )
    if hf_config.model_type in ("qwen4_exp", "qwen4_exp_text"):
        text_config = getattr(hf_config, "text_config", hf_config)
        text_config.model_type = "qwen4_exp_mtp"
        n_predict = getattr(text_config, "mtp_num_hidden_layers", 1)
        text_config.update(
            {
                "n_predict": n_predict,
                "architectures": ["Qwen4ExpMTP"],
                "index_share_for_mtp_iteration": getattr(
                    text_config, "index_share_for_mtp_iteration", False
                ),
            }
        )
        hf_config = text_config
    if hf_config.model_type in ("longcat_flash", "longcat_flash_ngram"):
        hf_config.model_type = "longcat_flash_mtp"
        n_predict = getattr(hf_config, "num_nextn_predict_layers", 1)
        hf_config.update({"n_predict": n_predict, "architectures": ["LongCatFlashMTPModel"]})

    if hf_config.model_type in ("step3p5", "step3p7") or hf_config.architectures[0] in (
        "Step3p5ForCausalLM",
        "Step3p7ForConditionalGeneration",
    ):
        quantization_config = getattr(hf_config, "quantization_config", None)
        hf_config = getattr(hf_config, "text_config", hf_config)
        if quantization_config is not None and getattr(hf_config, "quantization_config", None) is None:
            hf_config.update({"quantization_config": quantization_config})
        hf_config.model_type = "step3p5_mtp"
        n_predict = getattr(hf_config, "num_nextn_predict_layers", 1)
        hf_config.update({"n_predict": n_predict, "architectures": ["Step3p5MTP"]})

    if initial_architecture == "MistralLarge3ForCausalLM":
        hf_config.update({"architectures": ["EagleMistralLarge3ForCausalLM"]})

    return hf_config


def _dspark_post_init(self):
    _orig_post_init(self)
    if self.use_dspark():
        draft_model_config = self.draft_model_config
        if draft_model_config is None:
            raise ValueError("DSpark requires a draft model config.")
        draft_hf_config = draft_model_config.hf_config
        # deepseek v4 dspark
        if getattr(draft_hf_config, "ptd_token_id", None) is None:  # type: ignore
            draft_hf_config.ptd_token_id = getattr(draft_hf_config, "dspark_noise_token_id", None)  # type: ignore
        # gqa backend dspark
        if getattr(draft_hf_config, "ptd_token_id", None) is None:  # type: ignore
            draft_hf_config.ptd_token_id = getattr(draft_hf_config, "mask_token_id", None)  # type: ignore
        # Upstream DSpark normalization rewrites the config's model_type and
        # architecture in place. The concrete config class remains intact, so
        # restore K3 from that explicit type contract instead of probing for
        # optional sentinel attributes shared by other draft families.
        if isinstance(draft_hf_config, K3DSparkConfig):
            draft_hf_config.model_type = K3DSparkConfig.model_type
            draft_hf_config.architectures = ["K3DSparkModel"]
            self.update_arch_()


SpeculativeConfig.hf_config_override = hf_config_override
SpeculativeConfig.__post_init__ = _dspark_post_init
