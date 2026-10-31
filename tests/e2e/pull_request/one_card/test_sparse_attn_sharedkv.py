# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.utils import bootstrap_custom_op_env

bootstrap_custom_op_env(include_vendor_lib=True)
import vllm_ascend.vllm_ascend_C  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3),
    reason="SparseAttnSharedkv is the A2/A3 BF16/FP16 attention backend",
)


def _layout_outputs(dtype, num_tokens, shuffled_indices):
    generator = torch.Generator().manual_seed(17)
    page_size, heads, dim, topk = 128, 16, 512, 512
    compress_ratio, window_size = 4, 128
    num_cmp = num_tokens // compress_ratio
    q = torch.randn(num_tokens, heads, dim, generator=generator).to(dtype).npu()
    ori = torch.randn(math.ceil(num_tokens / page_size), page_size, 1, dim, generator=generator).to(dtype).npu()
    logical_cmp = torch.randn(math.ceil(num_cmp / page_size), page_size, 1, dim, generator=generator).to(dtype).npu()
    sinks = torch.randn(heads, generator=generator).npu()
    order = torch.randperm(num_cmp, generator=generator) if shuffled_indices else torch.arange(num_cmp)
    indices = torch.full((num_tokens, 1, topk), -1, dtype=torch.int32)
    for token in range(num_tokens):
        visible = order[order < (token + 1) // compress_ratio][:topk]
        indices[token, 0, : len(visible)] = visible.to(torch.int32)
    indices = indices.npu()
    starts = torch.tensor([0, num_tokens], dtype=torch.int32, device="npu")
    lengths = torch.tensor([num_tokens], dtype=torch.int32, device="npu")
    metadata = torch.ops._C_ascend.npu_sparse_attn_sharedkv_metadata(
        num_heads_q=heads,
        num_heads_kv=1,
        head_dim=dim,
        cu_seqlens_q=starts,
        seqused_kv=lengths,
        batch_size=1,
        max_seqlen_q=num_tokens,
        max_seqlen_kv=num_tokens,
        cmp_topk=topk,
        cmp_ratio=compress_ratio,
        ori_win_left=window_size - 1,
        layout_q="TND",
        layout_kv="PA_ND",
    )
    pages = logical_cmp.shape[0]
    layouts = (
        torch.arange(pages),
        torch.arange(pages - 1, -1, -1),
        torch.randperm(pages, generator=generator),
    )
    outputs = []
    for permutation in layouts:
        physical_cmp = torch.empty_like(logical_cmp)
        physical_cmp[permutation.npu()] = logical_cmp
        output = torch.ops._C_ascend.npu_sparse_attn_sharedkv(
            q,
            ori_kv=ori,
            cmp_kv=physical_cmp,
            ori_block_table=torch.arange(ori.shape[0], dtype=torch.int32, device="npu").unsqueeze(0),
            cmp_block_table=permutation.to(dtype=torch.int32, device="npu").unsqueeze(0),
            cmp_sparse_indices=indices,
            cu_seqlens_q=starts,
            seqused_kv=lengths,
            sinks=sinks,
            metadata=metadata,
            softmax_scale=dim**-0.5,
            cmp_ratio=compress_ratio,
            ori_win_left=window_size - 1,
            layout_q="TND",
            layout_kv="PA_ND",
        )[0]
        outputs.append(output.cpu())
    return outputs


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("num_tokens", [1024, 3072])
@pytest.mark.parametrize("shuffled_indices", [False, True])
def test_sparse_attention_is_independent_of_physical_page_order(dtype, num_tokens, shuffled_indices):
    # All layouts contain exactly the same logical KV and TopK sequence. The
    # allocator must not change attention's reduction order or rounded output.
    reference, reversed_pages, shuffled_pages = _layout_outputs(dtype, num_tokens, shuffled_indices)
    torch.testing.assert_close(reversed_pages, reference, rtol=0, atol=0)
    torch.testing.assert_close(shuffled_pages, reference, rtol=0, atol=0)
