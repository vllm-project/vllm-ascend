# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.utils import enable_custom_op


def test_band_visible_lengths_require_one_row_per_query():
    if get_ascend_device_type() == AscendDeviceType.A5:
        pytest.skip("Continuous band visible-length replay requires the A2/A3 custom op")
    enable_custom_op()
    num_tokens, num_heads, head_dim, block_size = 2, 16, 512, 128
    band_mask_mode = 4
    query = torch.zeros((num_tokens, num_heads, head_dim), dtype=torch.bfloat16, device="npu")
    kv = torch.ones((1, block_size, 1, head_dim), dtype=torch.bfloat16, device="npu")
    starts = torch.tensor([0, num_tokens], dtype=torch.int32, device="npu")
    lengths = torch.tensor([num_tokens], dtype=torch.int32, device="npu")
    visible = torch.ones((num_tokens, 1), dtype=torch.int32, device="npu")
    metadata = torch.ops._C_ascend.npu_sparse_flash_mla_metadata(
        num_heads,
        1,
        head_dim,
        cu_seqlens_q=starts,
        seqused_ori_kv=lengths,
        batch_size=1,
        max_seqlen_q=num_tokens,
        max_seqlen_ori_kv=num_tokens,
        ori_mask_mode=band_mask_mode,
        ori_win_left=block_size - 1,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_BBND",
        has_ori_kv=True,
        has_cmp_kv=False,
    )
    kwargs = dict(
        ori_kv=kv,
        ori_block_table=torch.tensor([[0]], dtype=torch.int32, device="npu"),
        cu_seqlens_q=starts,
        seqused_ori_kv=lengths,
        sinks=torch.zeros(num_heads, dtype=torch.float32, device="npu"),
        metadata=metadata,
        softmax_scale=head_dim**-0.5,
        ori_mask_mode=band_mask_mode,
        ori_win_left=block_size - 1,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_BBND",
    )
    output, _ = torch.ops._C_ascend.npu_sparse_flash_mla(query, ori_topk_length=visible, **kwargs)
    torch.npu.synchronize()
    assert output.shape == query.shape
    assert torch.isfinite(output).all()
    # Zero queries give equal weight to each visible KV row and the zero
    # attention sink. A one-row replay window must yield 1 / (1 + 1), even
    # for the second query; ignoring visible lengths would yield 2 / 3 there.
    torch.testing.assert_close(output, torch.full_like(output, 0.5))

    with pytest.raises(RuntimeError, match="TND band ori_topk_length shape must be"):
        torch.ops._C_ascend.npu_sparse_flash_mla(query, ori_topk_length=visible[:1], **kwargs)
        torch.npu.synchronize()


@pytest.mark.parametrize("num_heads", [8, 64])
@pytest.mark.parametrize("table_width", [8, 1025], ids=["address-cache", "scalar-addresses"])
def test_csa_full_and_partial_gathers(num_heads, table_width):
    if get_ascend_device_type() == AscendDeviceType.A5:
        pytest.skip("This gather schedule is specific to arch22")
    enable_custom_op()
    valid_counts = (512, 255, 256, 257, 511, 512)
    topk, block_size, head_dim, num_pages = 512, 128, 512, 8
    context = num_pages * block_size
    n = len(valid_counts)
    rng = torch.Generator().manual_seed(913)
    q = torch.randn(n, num_heads, head_dim, generator=rng, dtype=torch.bfloat16)
    ori = torch.randn(context, head_dim, generator=rng, dtype=torch.bfloat16)
    cmp = torch.randn(context, head_dim, generator=rng, dtype=torch.bfloat16)
    sinks = torch.linspace(-1.0, 1.0, num_heads)
    indices = torch.full((n, 1, topk), -1, dtype=torch.int32)
    expected = []
    for i, count in enumerate(valid_counts):
        end = context - n + i + 1
        rows = torch.randperm(end, generator=rng)[:count]
        indices[i, 0, :count] = rows.to(torch.int32)
        kv = torch.cat((ori[end - block_size : end], cmp[rows])).float()
        scores = q[i].float() @ kv.T * head_dim**-0.5
        probabilities = torch.cat((scores, sinks[:, None]), dim=1).softmax(dim=1)
        expected.append(probabilities[:, :-1] @ kv)

    starts = torch.tensor([0, n], dtype=torch.int32, device="npu")
    lengths = torch.tensor([context], dtype=torch.int32, device="npu")
    kwargs = dict(
        cu_seqlens_q=starts,
        seqused_ori_kv=lengths,
        seqused_cmp_kv=lengths,
        cmp_ratio=1,
        cmp_mask_mode=3,
        ori_mask_mode=4,
        ori_win_left=block_size - 1,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_BBND",
    )
    metadata = torch.ops._C_ascend.npu_sparse_flash_mla_metadata(
        num_heads,
        1,
        head_dim,
        batch_size=1,
        max_seqlen_q=n,
        max_seqlen_ori_kv=context,
        max_seqlen_cmp_kv=context,
        cmp_topk=topk,
        has_ori_kv=True,
        has_cmp_kv=True,
        **kwargs,
    )
    pages = torch.arange(num_pages - 1, -1, -1, dtype=torch.int32).view(1, -1)
    # Extending the table past the 1024-page tiling limit exercises scalar translation with identical data.
    cmp_pages = torch.nn.functional.pad(pages, (0, table_width - num_pages))
    shape = (num_pages, block_size, 1, head_dim)
    output, _ = torch.ops._C_ascend.npu_sparse_flash_mla(
        q.npu(),
        ori_kv=ori.view(shape).flip(0).npu(),
        cmp_kv=cmp.view(shape).flip(0).npu(),
        ori_block_table=pages.npu(),
        cmp_block_table=cmp_pages.npu(),
        cmp_sparse_indices=indices.npu(),
        sinks=sinks.npu(),
        metadata=metadata,
        softmax_scale=head_dim**-0.5,
        **kwargs,
    )
    torch.testing.assert_close(output.cpu().float(), torch.stack(expected), atol=0.004, rtol=0.025)
