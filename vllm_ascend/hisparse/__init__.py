# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Shared HiSparse cache operations, layouts and Core backend integration."""

from vllm.config import VllmConfig

from vllm_ascend.device.hardware_profile import DeviceAdaptorFamily, HardwareCapability, get_current_hardware_profile

FLASH_MTP_QUERY_LEN = 6


def is_glm5_next_hisparse(vllm_config: VllmConfig) -> bool:
    text_config = getattr(vllm_config.model_config, "hf_text_config", None)
    return (
        vllm_config.attention_config.hisparse_config is not None
        and getattr(text_config, "model_type", None) == "glm5_next_text"
    )


def _supports_local_flash_geometry(vllm_config: VllmConfig) -> bool:
    if not is_glm5_next_hisparse(vllm_config):
        return False
    parallel = vllm_config.parallel_config
    text_config = vllm_config.model_config.hf_text_config
    return (
        getattr(text_config, "index_kpool", None) == 4
        and getattr(text_config, "qk_rope_head_dim", None) == 0
        and parallel.data_parallel_size == 1
        and parallel.pipeline_parallel_size == 1
        and parallel.prefill_context_parallel_size == 1
        and parallel.decode_context_parallel_size == 1
    )


def supports_glm5_next_hisparse(vllm_config: VllmConfig) -> bool:
    return (
        _supports_local_flash_geometry(vllm_config)
        and vllm_config.cache_config.cache_dtype == "fp8"
        and vllm_config.attention_config.indexer_kv_dtype in ("fp8", "fp8_e4m3")
        and get_current_hardware_profile().device_adaptor_family is DeviceAdaptorFamily.FP8_OPTIMIZED
    )


def supports_glm5_next_int8_hisparse(vllm_config: VllmConfig) -> bool:
    return (
        _supports_local_flash_geometry(vllm_config)
        and vllm_config.cache_config.cache_dtype == "int8"
        and vllm_config.attention_config.indexer_kv_dtype in ("auto", "bf16")
        and get_current_hardware_profile().supports(HardwareCapability.NATIVE_INT8_SPARSE)
    )


def validate_glm5_next_hisparse(vllm_config: VllmConfig) -> None:
    if not (supports_glm5_next_hisparse(vllm_config) or supports_glm5_next_int8_hisparse(vllm_config)):
        raise ValueError(
            "GLM-5.3-Flash HiSparse requires A3 INT8 main KV with BF16 indexer or A5 FP8 main and indexer KV, "
            "with DP, PP, PCP and DCP set to 1."
        )
