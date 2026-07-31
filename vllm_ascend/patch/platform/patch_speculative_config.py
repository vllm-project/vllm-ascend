from typing import TYPE_CHECKING, Any

from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig as _HFPretrainedConfig
from vllm.config.speculative import SpeculativeConfig
from vllm.transformers_utils import config as _vllm_config_module
from vllm.utils.import_utils import LazyLoader

_orig_post_init = SpeculativeConfig.__post_init__

if TYPE_CHECKING:
    import vllm.model_executor.layers.quantization as me_quant
    from transformers import PretrainedConfig
else:
    PretrainedConfig = Any

    me_quant = LazyLoader("model_executor", globals(), "vllm.model_executor.layers.quantization")


# Kimi-K3 (MLA) DSpark draft config registration.
# The K3 dspark checkpoint ships model_type="k3_dspark" with no auto_map, so
# neither vLLM's config parser nor transformers AutoConfig can load it unless
# the model_type is registered with both. PretrainedConfig accepts the K3 MLA
# / dspark fields (q_lora_rank, kv_lora_rank, target_layer_ids, markov_rank,
# rope_parameters, ...) as plain attributes.
class K3DSparkConfig(_HFPretrainedConfig):
    model_type = "k3_dspark"


_vllm_config_module._CONFIG_REGISTRY["k3_dspark"] = K3DSparkConfig
AutoConfig.register("k3_dspark", K3DSparkConfig, exist_ok=True)


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
        draft_model_config = getattr(self, "draft_model_config", None)
        draft_hf_config = getattr(draft_model_config, "hf_config", None)
        # deepseek v4 dspark
        if getattr(draft_hf_config, "ptd_token_id", None) is None:  # type: ignore
            draft_hf_config.ptd_token_id = getattr(draft_hf_config, "dspark_noise_token_id", None)  # type: ignore
        # gqa backend dspark
        if getattr(draft_hf_config, "ptd_token_id", None) is None:  # type: ignore
            draft_hf_config.ptd_token_id = getattr(draft_hf_config, "mask_token_id", None)  # type: ignore
        architectures = getattr(draft_hf_config, "architectures", ()) or ()
        if getattr(draft_hf_config, "model_type", None) == "qwen3" and "Qwen3DSparkModel" in architectures:
            block_size = getattr(draft_hf_config, "block_size", None)
            if not isinstance(block_size, int) or isinstance(block_size, bool) or block_size <= 0:
                raise ValueError("Qwen3/GQA DSpark requires a positive integer block_size in the draft config.")
            if self.num_speculative_tokens != block_size:
                raise ValueError(
                    "Qwen3/GQA DSpark requires num_speculative_tokens to match "
                    f"the trained block_size ({block_size}); got "
                    f"{self.num_speculative_tokens}."
                )

        # Kimi-K3 (MLA) dspark: upstream __post_init__ rewrites any non-
        # Qwen3/Gemma4 dspark draft to DSparkDraftModel (deepseek_v4), which
        # would clobber K3DSparkModel. The rewrite only changes model_type and
        # architectures, so K3-unique fields (markov_head_type /
        # target_num_hidden_layers) survive and let us restore the K3 arch so
        # the K3 dspark draft model is loaded. DSV4 dspark carries neither
        # field, so it is left untouched.
        if getattr(draft_hf_config, "model_type", None) == "deepseek_v4" and (
            getattr(draft_hf_config, "markov_head_type", None) is not None
            or getattr(draft_hf_config, "target_num_hidden_layers", None) is not None
        ):
            draft_hf_config.model_type = "k3_dspark"  # type: ignore
            draft_hf_config.architectures = ["K3DSparkModel"]  # type: ignore
            self.update_arch_()
            # fast-fail (no fallback): the K3 draft is trained with
            # block_size=7, so num_speculative_tokens must be exactly 7
            # (any other value yields garbled output, not just lower
            # acceptance).
            if self.num_speculative_tokens != 7:
                raise ValueError(
                    "K3 dspark requires num_speculative_tokens=7 "
                    f"(block_size=7); got {self.num_speculative_tokens}."
                )


SpeculativeConfig.hf_config_override = hf_config_override
SpeculativeConfig.__post_init__ = _dspark_post_init
