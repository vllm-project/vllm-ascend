# SPDX-License-Identifier: Apache-2.0
"""Optional package entry point for Kimi's local, unsharded A5 DSA projections."""

from vllm_ascend import envs
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.device.hardware_profile import DeviceAdaptorFamily, get_current_hardware_profile


def prepare_mega_dsa(layer):
    if not envs.VLLM_ASCEND_ENABLE_MEGA_DSA or not layer.is_kimi_k31:
        return None
    if get_current_hardware_profile().device_adaptor_family != DeviceAdaptorFamily.FP8_OPTIMIZED:
        return None
    pc = layer.vllm_config.parallel_config
    fine = get_ascend_config().finegrained_tp_config
    if (
        layer.tp_size != 1
        or pc.decode_context_parallel_size != 1
        or pc.prefill_context_parallel_size != 1
        or max(getattr(fine, name + "_tensor_parallel_size", 1) for name in ("qkv_iproj", "oproj", "gproj")) > 1
        or layer._is_mtp_layer
        or layer.use_rope
        or layer.use_index_cache
        or layer.rl_weight_update_enabled
        or getattr(layer.vllm_config.model_config, "enable_sleep_mode", False)
        or not layer.enable_sparse_sfa_c8
        or not layer.runtime_has_indexer
        or not layer.indexer.enable_sparse_li_c8
        or layer.g_proj is None
        or not hasattr(layer, "weight_dq_scale")
        or layer.indexer.qk_rope_head_dim != 0
        or layer.indexer.topk_tokens != 2048
        or layer.indexer.head_dim != 128
        or layer.indexer.n_head != 32
        or (layer.local_num_heads, layer.qk_nope_head_dim, layer.qk_rope_head_dim) != (96, 128, 64)
    ):
        return None
    from megadsa import prepare

    return prepare(layer)
