import typing
from typing import TYPE_CHECKING, Any, Optional

from vllm.config.speculative import SpeculativeConfig
from vllm.logger import logger
from vllm.utils.import_utils import LazyLoader

if TYPE_CHECKING:
    import vllm.model_executor.layers.quantization as me_quant
    from transformers import PretrainedConfig
else:
    PretrainedConfig = Any

    me_quant = LazyLoader("model_executor", globals(), "vllm.model_executor.layers.quantization")


def _register_dspark_speculative_method() -> None:
    """Teach ``SpeculativeConfig`` that ``method="dspark"`` is legal.

    ``SpeculativeConfig`` is a pydantic dataclass whose ``method`` field is a
    ``Literal[...]`` (``SpeculativeMethod``). pydantic validates ``method``
    against that literal set at construction time — BEFORE any of our dispatch /
    hf_config_override runs — so ``SpeculativeConfig(method="dspark")`` fails
    with a pydantic ``literal_error`` unless we extend the literal.

    We read the runtime literal (it differs between vLLM forks — the A3 image's
    fork already carries ``dflash``), append ``"dspark"``, re-point the module's
    ``SpeculativeMethod`` and the field annotation, then ``rebuild_dataclass`` so
    pydantic recompiles the schema. ``VllmConfig`` is rebuilt too because its
    cached nested schema for ``speculative_config`` may embed the pre-patch
    validator.
    """
    from vllm.config import speculative as spec_mod

    existing = typing.get_args(spec_mod.SpeculativeMethod)  # flat tuple of str
    if "dspark" in existing:
        return
    new_literal = typing.Literal[existing + ("dspark",)]  # type: ignore[valid-type]
    spec_mod.SpeculativeMethod = new_literal
    # speculative.py does NOT use `from __future__ import annotations`, so the
    # stored annotation is the real `SpeculativeMethod | None` object.
    SpeculativeConfig.__annotations__["method"] = Optional[new_literal]

    try:
        from pydantic.dataclasses import rebuild_dataclass
    except Exception as e:  # pragma: no cover — pydantic always present in vLLM
        logger.warning("Cannot import rebuild_dataclass (%s); dspark method may not validate.", e)
        return
    try:
        rebuild_dataclass(SpeculativeConfig, force=True)  # type: ignore[arg-type]
    except Exception as e:
        logger.warning("rebuild_dataclass(SpeculativeConfig) failed (%s); dspark method may not validate.", e)
        return
    try:
        from vllm.config.vllm import VllmConfig

        rebuild_dataclass(VllmConfig, force=True)  # type: ignore[arg-type]
    except Exception as e:
        logger.debug("rebuild_dataclass(VllmConfig) failed (%s); nested spec validation may be stale.", e)


def _is_dspark_v4_checkpoint(hf_config: PretrainedConfig) -> bool:
    """Detect DSpark V4-Flash / V4-Pro checkpoint by config signature.

    DSpark ckpts inherit the DSv4 ``model_type`` and ``DeepseekV4ForCausalLM``
    architecture, so we have to look at the DSpark-specific fields that
    ``inference/config.json`` carries (the HF root ``config.json`` keeps
    ``num_nextn_predict_layers=1`` for transformers compat). The cheapest
    signature is ``dspark_block_size`` + ``dspark_target_layer_ids``.
    """
    if hf_config.model_type != "deepseek_v4":
        return False
    return bool(getattr(hf_config, "dspark_block_size", 0)) and bool(
        getattr(hf_config, "dspark_target_layer_ids", None)
    )


def hf_config_override(hf_config: PretrainedConfig) -> PretrainedConfig:
    initial_architecture = hf_config.architectures[0]
    # DSpark routes to a separate draft architecture *before* the generic
    # deepseek_v4 → deepseek_mtp rewrite, since DSpark is not the serial MTP
    # path even though weights live under ``mtp.*`` (mirrors upstream PR
    # vllm#46995). The runtime selects this branch when method="dspark".
    if hf_config.model_type == "deepseek_v4" and _is_dspark_v4_checkpoint(hf_config):
        n_predict = (
            getattr(hf_config, "n_mtp_layers", None) or getattr(hf_config, "num_nextn_predict_layers", None) or 3
        )
        hf_config.update(
            {
                "model_type": "deepseek_v4_dspark",
                "n_predict": n_predict,
                "architectures": ["DSparkDeepseekV4ForCausalLM"],
            }
        )
        return hf_config
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
    if hf_config.model_type == "longcat_flash":
        hf_config.model_type = "longcat_flash_mtp"
        n_predict = getattr(hf_config, "num_nextn_predict_layers", 1)
        hf_config.update({"n_predict": n_predict, "architectures": ["LongCatFlashMTPModel"]})

    if hf_config.model_type == "step3p5":
        hf_config.model_type = "step3p5_mtp"
        n_predict = getattr(hf_config, "num_nextn_predict_layers", 1)
        hf_config.update({"n_predict": n_predict, "architectures": ["Step3p5MTP"]})

    if initial_architecture == "MistralLarge3ForCausalLM":
        hf_config.update({"architectures": ["EagleMistralLarge3ForCausalLM"]})

    return hf_config


SpeculativeConfig.hf_config_override = hf_config_override

# Extend the pydantic method literal so `method="dspark"` validates. Must run at
# import (patch application) time, before EngineArgs builds SpeculativeConfig.
_register_dspark_speculative_method()
