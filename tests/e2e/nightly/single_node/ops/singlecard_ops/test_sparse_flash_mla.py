# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.utils import enable_custom_op


@pytest.mark.parametrize("graph", [False, True])
def test_a5_nope_sparse_indices_and_metadata_replay(graph):
    if get_ascend_device_type() != AscendDeviceType.A5:
        pytest.skip("SparseFlashMla NoPE uses the A5 CANN operator")
    from vllm_ascend.attention.sfa_v1 import SMLA_METADATA_SIZE, sparse_mla

    enable_custom_op()
    query = torch.zeros((1, 64, 512), dtype=torch.bfloat16, device="npu")
    cache = torch.ones((1, 128, 1, 512), dtype=torch.bfloat16, device="npu")
    cache[:, 64:] = 3
    indices = torch.arange(64, dtype=torch.int32, device="npu").view(1, 1, 64)
    metadata = SimpleNamespace(
        block_size=128,
        block_table=torch.tensor([[0]], dtype=torch.int32, device="npu"),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32, device="npu"),
        seq_lens=torch.tensor([128], dtype=torch.int32, device="npu"),
        max_query_len=1,
        max_seq_len=128,
        smla_metadata=torch.empty(SMLA_METADATA_SIZE, dtype=torch.int32, device="npu"),
        smla_topk_length=torch.empty((1, 1), dtype=torch.int32, device="npu"),
        smla_valid_mask=torch.ones((1, 1), dtype=torch.bool, device="npu"),
        smla_sinks=torch.zeros(64, dtype=torch.float32, device="npu"),
    )
    address = metadata.smla_metadata.data_ptr()
    output = sparse_mla(query, cache, indices, metadata, 512**-0.5)
    torch.npu.synchronize()
    # Equal logits over 64 selected values and the zero sink. Dropping the
    # indices would include the other half of the cache and change the result.
    torch.testing.assert_close(output, torch.full_like(output, 64 / 65))
    if graph:
        captured = torch.npu.NPUGraph()
        with torch.npu.graph(captured):
            output = sparse_mla(query, cache, indices, metadata, 512**-0.5)
    for valid, value in ((64, 3), (32, 1)):
        indices.copy_(torch.arange(64, dtype=torch.int32, device="npu").view_as(indices))
        if value == 3:
            indices.add_(64)
        indices[..., valid:] = -1
        if graph:
            captured.replay()
        else:
            output = sparse_mla(query, cache, indices, metadata, 512**-0.5)
        torch.npu.synchronize()
        torch.testing.assert_close(output, torch.full_like(output, value * valid / (valid + 1)))
        assert metadata.smla_metadata.data_ptr() == address


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
