# SPDX-License-Identifier: Apache-2.0
"""Optional package entry point for Kimi's A5 DSA projections."""

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
        or getattr(fine, "qkv_iproj_tensor_parallel_size", 1) > 1
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
    g_size = getattr(fine, "gproj_tensor_parallel_size", 1)
    o_size = getattr(fine, "oproj_tensor_parallel_size", 1)
    if g_size == o_size == 1:
        from megadsa import prepare

        return prepare(layer)

    import megadsa

    prepare_sharded = getattr(megadsa, "prepare_localtp", None)
    if prepare_sharded is None:
        return None
    gate_group = getattr(layer.g_proj, "tp_group", None) if g_size > 1 else None
    if g_size > 1 and (getattr(gate_group, "world_size", 0) != g_size or getattr(layer, "begin_gate", None) is None):
        return None
    output_group = None
    if o_size > 1:
        if layer.o_proj.weight.shape != (7168, 12288 // o_size):
            return None
        from vllm_ascend.distributed.parallel_state import get_otp_group

        output_group = get_otp_group()
        if output_group.world_size != o_size:
            return None

    def capacities(rows):
        from vllm_ascend.ascend_forward_context import _EXTRA_CTX

        dp_rows = max(rows, _EXTRA_CTX.max_tokens_across_dp or 0)
        gate_rows = dp_rows if gate_group is not None else rows
        output_rows = dp_rows if output_group is not None else rows
        return gate_rows, output_rows

    return prepare_sharded(layer, gate_group=gate_group, output_group=output_group, capacity=capacities)
