# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
"""A5/CANN validation of QFA's strided K/V and six-dimensional scales."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch_npu
from cann_ops_transformer.ops import quant_flash_attn, quant_flash_attn_metadata
from vllm.v1.kv_cache_interface import FullAttentionSpec

from vllm_ascend.attention.attention_c8_mxfp import (
    fill_mxfp_v_scale_cache,
    mxfp_cache_spec,
    mxfp_cache_views_for_spec,
    mxfp_k_scale_slot_index,
    scatter_mxfp_k_scale_cache,
    scatter_mxfp_pa_nz_kv_cache,
)
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.worker.utils import AscendKVBlockZeroer


@pytest.mark.parametrize("query_tokens", [1, 64])
@pytest.mark.parametrize("ratio", [1, 2])
def test_qfa_strided_scales_match_contiguous_reference(query_tokens, ratio):
    init_device_properties_triton()
    torch.manual_seed(42)
    device = torch.device("npu")
    spec = mxfp_cache_spec(
        FullAttentionSpec(
            block_size=512 * ratio,
            num_kv_heads=1,
            head_size=256,
            dtype=torch.float8_e4m3fn,
        )
    )
    # Exercise both an aligned allocation slice and trailing page padding.
    page_size = spec.page_size_bytes + ratio * 256
    spec = replace(spec, page_size_padded=page_size)
    backing = torch.full((128 + 3 * page_size + 256,), 91, dtype=torch.int8, device=device)
    caches = mxfp_cache_views_for_spec(backing[128:-256], spec, 512)
    num_tokens = 529  # cross a kernel-block boundary
    slots = torch.arange(num_tokens, dtype=torch.int64, device=device)
    key, key_scale = torch_npu.npu_dynamic_mx_quant(
        torch.randn(num_tokens, 1, 256, dtype=torch.bfloat16, device=device),
        dst_type=torch.float8_e4m3fn,
    )
    value = torch.randint(0, 112, key.shape, dtype=torch.int8, device=device).view(torch.float8_e4m3fn)
    scatter_mxfp_pa_nz_kv_cache(key, value, caches[0], caches[1], slots)
    scatter_mxfp_k_scale_cache(key_scale, caches[2], mxfp_k_scale_slot_index(slots, 512))
    fill_mxfp_v_scale_cache(torch.full((256,), 127, dtype=torch.uint8, device=device), caches[3])
    query, query_scale = torch_npu.npu_dynamic_mx_quant(
        torch.randn(query_tokens, 2, 256, dtype=torch.bfloat16, device=device),
        dst_type=torch.float8_e4m3fn,
    )
    layout_q_descale = "N2TGD" if query_tokens == 1 else "TND"
    if query_tokens == 1:
        query_scale = query_scale.view(torch.uint8).view(query_tokens, 1, 2, 4, 2).permute(1, 0, 2, 3, 4).contiguous()
    query_scale = query_scale.view(torch.float8_e8m0fnu)
    mask_mode = 0 if query_tokens == 1 else 3
    lengths = dict(
        cu_seqlens_q=torch.tensor([0, query_tokens], dtype=torch.int32, device=device),
        cu_seqlens_kv=None,
        seqused_q=None,
        seqused_kv=torch.tensor([num_tokens], dtype=torch.int32, device=device),
    )
    attrs = dict(
        max_seqlen_q=query_tokens,
        max_seqlen_kv=-1,
        mask_mode=mask_mode,
        win_left=-1,
        win_right=-1,
        layout_q="TND",
        layout_q_descale=layout_q_descale,
        layout_kv="PA_NZ",
        layout_out="TND",
    )
    metadata = quant_flash_attn_metadata(2, 1, 256, 1, **lengths, **attrs)
    mask = None if mask_mode == 0 else torch.triu(torch.ones(2048, 2048, dtype=torch.int8, device=device), diagonal=1)

    def run(views):
        result = quant_flash_attn(
            query,
            views[0],
            views[1],
            query_scale,
            views[2].view(torch.float8_e8m0fnu),
            views[3].view(torch.float8_e8m0fnu),
            1,
            block_table=torch.tensor([[0, 1]], dtype=torch.int32, device=device),
            p_scale=None,
            sinks=None,
            attn_mask=mask,
            metadata=metadata,
            softmax_scale=256**-0.5,
            return_softmax_lse=False,
            **lengths,
            **attrs,
        )
        return result[0] if isinstance(result, tuple) else result

    contiguous = tuple(view.view(torch.uint8).contiguous().view(view.dtype) for view in caches)
    torch.testing.assert_close(run(caches), run(contiguous), rtol=1e-3, atol=1e-3)
    static_scale_before = caches[3].clone()
    zeroer = AscendKVBlockZeroer(device, pin_memory=False)
    group = SimpleNamespace(kv_cache_spec=spec, kv_cache_group_id=0, layer_names=["attn"])
    zeroer.init_meta([group], [512], "mxfp8", set(), {"attn": SimpleNamespace(kv_cache=caches)})
    zeroer.zero_block_ids([2])
    torch.npu.synchronize()
    assert torch.equal(caches[3], static_scale_before)
    for cache in caches[:3]:
        assert (cache[2 * ratio :].view(torch.uint8) == 0).all()
    assert (backing[:128] == 91).all() and (backing[-256:] == 91).all()
