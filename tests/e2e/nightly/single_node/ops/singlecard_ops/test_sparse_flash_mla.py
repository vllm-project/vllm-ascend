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


# Cover padded-index boundaries, including the 257-valid-index case.
@pytest.mark.parametrize(
    "block_size,page_stride,query_tokens,heads,valid_topk,table_capacity,graph",
    [
        (64, 32768, 1, 4, 512, 16, False),
        (128, 65536, 6, 4, 512, 16, False),
        (64, 65536, 6, 4, 512, 16, False),
        (128, 131072, 1, 4, 512, 16, False),
        (64, 32769, 1, 4, 512, 16, False),
        (64, 32768, 6, 1, 512, 16, False),
        (128, 65536, 1, 64, 512, 16, False),
        (64, 32768, 1, 128, 512, 16, False),
        (64, 32768, 6, 4, 127, 16, False),
        (128, 65536, 6, 4, 128, 16, False),
        (64, 32768, 6, 4, 257, 16, False),
        (128, 65536, 6, 1, 511, 16, False),
        (64, 32768, 1, 4, 512, 1024, False),
        (64, 32768, 1, 4, 512, 1025, False),
        (64, 32768, 1, 4, 256, 16, False),
        (64, 32768, 6, 4, 512, 16, True),
    ],
)
def test_csa_v0_paged_address_cache_against_fp32(
    block_size, page_stride, query_tokens, heads, valid_topk, table_capacity, graph
):
    if get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3):
        pytest.skip("CSA V0 UB address caching targets A2/A3")
    enable_custom_op()
    generator = torch.Generator().manual_seed(15596)
    head_dim, physical_blocks, ori_length, cmp_length = 512, 32, 2048, 1024
    query_cpu = (torch.randn(query_tokens, heads, head_dim, generator=generator) / 8).bfloat16()
    ori_cpu = (torch.randn(physical_blocks, 128, 1, head_dim, generator=generator) / 8).bfloat16()
    cmp_cpu = (torch.randn(physical_blocks, block_size, 1, head_dim, generator=generator) / 8).bfloat16()
    ori_table_cpu = torch.randperm(physical_blocks, generator=generator)[:16].to(torch.int32)
    cmp_table_cpu = torch.zeros(table_capacity, dtype=torch.int32)
    cmp_pages = cmp_length // block_size
    cmp_table_cpu[:cmp_pages] = torch.randperm(physical_blocks, generator=generator)[:cmp_pages].to(torch.int32)
    topk = 512
    indices_cpu = torch.full((query_tokens, 1, topk), -1, dtype=torch.int32)
    for row in range(query_tokens):
        indices_cpu[row, 0, :valid_topk] = torch.randperm(1020, generator=generator)[:valid_topk].to(torch.int32)
    # Exercise duplicate indices as well as physical pages in arbitrary order.
    indices_cpu[:, 0, 1] = indices_cpu[:, 0, 0]
    sinks_cpu = torch.randn(heads, generator=generator) / 8
    expected = torch.empty_like(query_cpu, dtype=torch.float32)
    for row in range(query_tokens):
        position = ori_length - query_tokens + row
        ori_ids = torch.arange(position - 127, position + 1)
        ori_rows = ori_cpu[ori_table_cpu[ori_ids // 128].long(), ori_ids % 128, 0].float()
        cmp_ids = indices_cpu[row, 0, :valid_topk].long()
        cmp_rows = cmp_cpu[cmp_table_cpu[cmp_ids // block_size].long(), cmp_ids % block_size, 0].float()
        keys = torch.cat((ori_rows, cmp_rows))
        scores = query_cpu[row].float() @ keys.T * head_dim**-0.5
        probabilities = torch.softmax(torch.cat((scores, sinks_cpu[:, None]), dim=-1), dim=-1)
        expected[row] = probabilities[:, :-1] @ keys

    query = query_cpu.npu()
    ori = ori_cpu.npu()
    # Only dimension zero is strided; padding between physical pages is real.
    storage = torch.empty(
        (physical_blocks - 1) * page_stride + block_size * head_dim,
        dtype=torch.bfloat16,
        device="npu",
    )
    compressed = storage.as_strided((physical_blocks, block_size, 1, head_dim), (page_stride, head_dim, head_dim, 1))
    compressed.copy_(cmp_cpu.npu())
    starts = torch.tensor([0, query_tokens], dtype=torch.int32, device="npu")
    ori_used = torch.tensor([ori_length], dtype=torch.int32, device="npu")
    cmp_used = torch.tensor([cmp_length], dtype=torch.int32, device="npu")
    residual = torch.zeros(1, dtype=torch.int32, device="npu")
    metadata = torch.ops._C_ascend.npu_sparse_flash_mla_metadata(
        heads,
        1,
        head_dim,
        cu_seqlens_q=starts,
        seqused_ori_kv=ori_used,
        seqused_cmp_kv=cmp_used,
        cmp_residual_kv=residual,
        batch_size=1,
        max_seqlen_q=query_tokens,
        max_seqlen_ori_kv=ori_length,
        max_seqlen_cmp_kv=cmp_length,
        cmp_topk=topk,
        cmp_ratio=2,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        ori_win_left=127,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_BBND",
        has_ori_kv=True,
        has_cmp_kv=True,
    )
    kwargs = dict(
        ori_kv=ori,
        cmp_kv=compressed,
        cmp_sparse_indices=indices_cpu.npu(),
        ori_block_table=ori_table_cpu[None].npu(),
        cmp_block_table=cmp_table_cpu[None].npu(),
        cu_seqlens_q=starts,
        seqused_ori_kv=ori_used,
        seqused_cmp_kv=cmp_used,
        cmp_residual_kv=residual,
        sinks=sinks_cpu.npu(),
        metadata=metadata,
        softmax_scale=head_dim**-0.5,
        cmp_ratio=2,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        ori_win_left=127,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_BBND",
    )
    for _ in range(3):
        output, _ = torch.ops._C_ascend.npu_sparse_flash_mla(query, **kwargs)
    torch.npu.synchronize()
    if graph:
        captured = torch.npu.NPUGraph()
        with torch.npu.graph(captured):
            output, _ = torch.ops._C_ascend.npu_sparse_flash_mla(query, **kwargs)
        captured.replay()
        torch.npu.synchronize()
    torch.testing.assert_close(output.cpu().float(), expected, atol=0.002, rtol=0.01)
