# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

def patch_dp_attention():

    from sglang.srt.layers import dp_attention
    from omni.adaptors.sglang.layers.dp_attention import (
        dp_scatter,
        DPPaddingMode,
        dp_gather_partial,
        attn_tp_all_gather,
        dp_gather_replicate,
        get_attention_tp_size,
        get_attention_dp_rank,
        get_attention_tp_rank,
        get_attention_dp_size,
        get_attention_tp_group,
        initialize_dp_attention,
        dp_reduce_scatter_tensor,
        get_local_attention_dp_size,
        attn_tp_reduce_scatter_tensor,
        attn_tp_all_gather_into_tensor,
        compute_dp_attention_world_info,
    )

    dp_attention.dp_scatter = dp_scatter
    dp_attention.DPPaddingMode = DPPaddingMode
    dp_attention.dp_gather_partial = dp_gather_partial
    dp_attention.attn_tp_all_gather = attn_tp_all_gather
    dp_attention.dp_gather_replicate = dp_gather_replicate
    dp_attention.get_attention_tp_size = get_attention_tp_size
    dp_attention.get_attention_dp_rank = get_attention_dp_rank
    dp_attention.get_attention_tp_rank = get_attention_tp_rank
    dp_attention.get_attention_dp_size = get_attention_dp_size
    dp_attention.get_attention_tp_group = get_attention_tp_group
    dp_attention.initialize_dp_attention = initialize_dp_attention
    dp_attention.dp_reduce_scatter_tensor = dp_reduce_scatter_tensor
    dp_attention.get_local_attention_dp_size = get_local_attention_dp_size
    dp_attention.attn_tp_reduce_scatter_tensor = attn_tp_reduce_scatter_tensor
    dp_attention.attn_tp_all_gather_into_tensor = attn_tp_all_gather_into_tensor
    dp_attention.compute_dp_attention_world_info = compute_dp_attention_world_info

    print("+++++++++++++++++++++++++ patch_dp_attention ++++++++++++++++++++++++++++++++++")


def patch_token_dispatcher():

    from sglang.srt.layers.moe import token_dispatcher
    from omni.adaptors.sglang.layers.moe.token_dispatcher import (
        DeepEPDispatcher,
    )

    token_dispatcher.DeepEPDispatcher = DeepEPDispatcher

    print("+++++++++++++++++++++++++ patch_token_dispatcher ++++++++++++++++++++++++++++++++++")


_patch_done = False

def patch_all():
    global _patch_done
    if _patch_done:
        return
    patch_dp_attention()
    patch_token_dispatcher()
    _patch_done = True

patch_all()
