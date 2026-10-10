# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes

import torch

from vllm_ascend.ascend_config import KVPPConfig, validate_additional_config_bool
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.utils import model_uses_sfa_sparse

from . import TURBOQUANT_CACHE_DTYPE, is_turboquant


def _validate_turboquant_common(vllm_config) -> None:
    """Constraints that hold for every TurboQuant entry point."""
    model = vllm_config.model_config
    if model.dtype != torch.bfloat16:
        raise ValueError(f"Ascend {TURBOQUANT_CACHE_DTYPE} requires a BF16 (bfloat16) model dtype, got {model.dtype}")
    if not get_current_hardware_profile().supports(HardwareCapability.TURBOQUANT_4BIT_NC_CACHE):
        raise ValueError(f"Ascend {TURBOQUANT_CACHE_DTYPE} is only supported on Ascend A2/A3")


def _validate_turboquant_sfa(vllm_config) -> None:
    """Checks for the packed SFA main cache.

    Unlike the compressed-cache path this one supports context parallelism, so
    the DCP/PCP restriction stays out of here.
    """
    additional_config = vllm_config.additional_config or {}
    enable_sparse_sfa_c8 = validate_additional_config_bool(
        additional_config.get("enable_sparse_sfa_c8", False),
        "additional_config.enable_sparse_sfa_c8",
    )
    if enable_sparse_sfa_c8:
        raise ValueError(f"{TURBOQUANT_CACHE_DTYPE} and enable_sparse_sfa_c8 cannot be enabled together")

    xlite_graph_config = additional_config.get("xlite_graph_config", {})
    enable_xlite = validate_additional_config_bool(
        xlite_graph_config.get("enabled", False),
        "additional_config.xlite_graph_config.enabled",
    )
    if enable_xlite:
        raise ValueError(f"{TURBOQUANT_CACHE_DTYPE} does not support xLite graph mode")

    hf_text_config = vllm_config.model_config.hf_text_config
    kv_lora_rank = getattr(hf_text_config, "kv_lora_rank", None)
    if kv_lora_rank != 512:
        raise ValueError(f"{TURBOQUANT_CACHE_DTYPE} requires kv_lora_rank=512, but got {kv_lora_rank}")

    rope_head_dim = getattr(hf_text_config, "qk_rope_head_dim", None)
    if rope_head_dim != 64:
        raise ValueError(f"{TURBOQUANT_CACHE_DTYPE} requires qk_rope_head_dim=64, but got {rope_head_dim}")


def _validate_turboquant_dsa(vllm_config) -> None:
    """Checks for the compressed-cache (DeepSeek V4) path."""
    model = vllm_config.model_config
    hf = getattr(model, "hf_text_config", None)
    if hf is None:
        hf = getattr(model, "hf_config", None)
    if hf is None or getattr(hf, "model_type", None) != "deepseek_v4":
        raise ValueError(f"Ascend {TURBOQUANT_CACHE_DTYPE} currently requires DeepSeek V4")
    if not vllm_config.use_v2_model_runner:
        raise ValueError("DeepSeek V4 TurboQuant requires VLLM_USE_V2_MODEL_RUNNER=1")
    head_dims = (getattr(hf, "head_dim", None), getattr(hf, "qk_rope_head_dim", None))
    if head_dims != (512, 64):
        raise ValueError("DeepSeek V4 TurboQuant requires head_dim=512 and qk_rope_head_dim=64")
    if getattr(hf, "index_topk", None) not in (512, 1024):
        raise ValueError("MixedQuantSparseFlashMla TurboQuant requires index_topk=512 or 1024")
    if getattr(vllm_config.attention_config, "indexer_kv_dtype", None) != "int8":
        raise ValueError(
            "DeepSeek V4 TurboQuant on Ascend A2/A3 requires "
            "attention_config.indexer_kv_dtype='int8' because the "
            "Lightning Indexer cache writer produces INT8 keys"
        )
    parallel = vllm_config.parallel_config
    if hf.num_attention_heads % (4 * parallel.tensor_parallel_size):
        raise ValueError("MixedQuantSparseFlashMla requires the per-rank query head count to be a multiple of 4")
    if parallel.decode_context_parallel_size != 1 or parallel.prefill_context_parallel_size != 1:
        raise ValueError("DeepSeek V4 TurboQuant does not yet support context parallelism")
    if KVPPConfig.from_vllm_config(vllm_config).size > 1:
        raise ValueError("DeepSeek V4 TurboQuant packed cache does not yet support KV layer parallelism")
    if vllm_config.kv_transfer_config is not None:
        raise ValueError("DeepSeek V4 TurboQuant packed cache does not yet support KV transfer")
    # Packed cache views must reach the fused operator without materializing
    # a full contiguous BF16/uint8 cache on every attention invocation.
    for library in (None, "libnnopbase.so"):
        try:
            _ = ctypes.CDLL(library).NnopbaseSupportTensorV2
            break
        except (OSError, AttributeError):
            continue
    else:
        raise ValueError("DeepSeek V4 TurboQuant packed cache requires opbase with NnopbaseSupportTensorV2")


def validate_turboquant(vllm_config) -> None:
    """Single entry point for every TurboQuant cache-dtype constraint.

    The two cache layouts are gated by the model family: DeepSeek V4 drives the
    compressed-cache path, other sparse models drive the packed SFA main cache.
    """
    if not is_turboquant(vllm_config):
        return
    _validate_turboquant_common(vllm_config)
    if model_uses_sfa_sparse(vllm_config.model_config):
        _validate_turboquant_sfa(vllm_config)
    else:
        _validate_turboquant_dsa(vllm_config)
