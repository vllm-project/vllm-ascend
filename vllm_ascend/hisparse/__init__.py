# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Shared HiSparse cache operations, layouts and Core backend integration."""

from contextvars import ContextVar
from importlib.util import find_spec

from vllm.config import VllmConfig

from vllm_ascend.device.hardware_profile import DeviceAdaptorFamily, get_current_hardware_profile

HISPARSE_GROUPED_QUERY_LEN = 6

# True only while building the separate target refill graph in a worker.
hisparse_refill_capture: ContextVar[bool] = ContextVar("hisparse_refill_capture", default=False)


def supports_hybrid_hisparse_core() -> bool:
    """Require Core support for separate indexer pages and fitted recurrent state."""
    if find_spec("vllm.v1.hisparse") is None:
        return False
    from vllm.v1.hisparse import binding

    return callable(getattr(binding, "fit_mamba_blocks_to_attention", None))


def require_hybrid_hisparse_core() -> None:
    if not supports_hybrid_hisparse_core():
        raise ValueError("Hybrid HiSparse requires the verified vLLM main pin with Mamba cache fitting.")


def uses_pooled_hisparse(vllm_config: VllmConfig) -> bool:
    """Detect the pooled indexer configuration before worker cache specs exist."""
    if getattr(getattr(vllm_config, "attention_config", None), "hisparse_config", None) is None:
        return False
    text_config = getattr(getattr(vllm_config, "model_config", None), "hf_text_config", None)
    pool_size = getattr(text_config, "index_kpool", None)
    return isinstance(pool_size, int) and pool_size > 1


def supports_pooled_hisparse(vllm_config: VllmConfig) -> bool:
    """Gate the qualified A5 packed NoPE and FP8 pooled-indexer geometry."""
    if not uses_pooled_hisparse(vllm_config):
        return False
    parallel = vllm_config.parallel_config
    text_config = vllm_config.model_config.hf_text_config
    return (
        vllm_config.cache_config.cache_dtype == "fp8"
        and vllm_config.attention_config.indexer_kv_dtype in ("fp8", "fp8_e4m3")
        and getattr(text_config, "index_kpool", None) == 4
        and getattr(text_config, "qk_rope_head_dim", None) == 0
        and parallel.data_parallel_size == 1
        and parallel.pipeline_parallel_size == 1
        and parallel.prefill_context_parallel_size == 1
        and parallel.decode_context_parallel_size == 1
        and get_current_hardware_profile().device_adaptor_family is DeviceAdaptorFamily.FP8_OPTIMIZED
    )
