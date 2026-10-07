# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes

import torch

from vllm_ascend.ascend_config import KVPPConfig
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

from . import TURBOQUANT_CACHE_DTYPE, is_turboquant


def validate_turboquant(vllm_config):
    if not is_turboquant(vllm_config):
        return
    model = vllm_config.model_config
    hf = getattr(model, "hf_text_config", None)
    if hf is None:
        hf = getattr(model, "hf_config", None)
    if hf is None or getattr(hf, "model_type", None) != "deepseek_v4":
        raise ValueError(f"Ascend {TURBOQUANT_CACHE_DTYPE} currently requires DeepSeek V4")
    if not vllm_config.use_v2_model_runner:
        raise ValueError("DeepSeek V4 TurboQuant requires VLLM_USE_V2_MODEL_RUNNER=1")
    head_dims = (getattr(hf, "head_dim", None), getattr(hf, "qk_rope_head_dim", None))
    if model.dtype != torch.bfloat16 or head_dims != (512, 64):
        raise ValueError("DeepSeek V4 TurboQuant requires BF16, head_dim=512 and qk_rope_head_dim=64")
    if getattr(hf, "index_topk", None) not in (512, 1024):
        raise ValueError("MixedQuantSparseFlashMla TurboQuant requires index_topk=512 or 1024")
    if get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3):
        raise ValueError("DeepSeek V4 TurboQuant currently supports Ascend A2/A3")
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
