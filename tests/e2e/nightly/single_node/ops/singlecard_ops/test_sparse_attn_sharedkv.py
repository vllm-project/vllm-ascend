# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import AscendDeviceType, bootstrap_custom_op_env, get_ascend_device_type

bootstrap_custom_op_env(include_vendor_lib=True)
import vllm_ascend.vllm_ascend_C  # type: ignore[import-untyped] # noqa: E402,F401

from vllm_ascend.ops.triton.sparse_index_remap import mask_invalid_sparse_attention_rows  # noqa: E402


@pytest.mark.parametrize("has_ori_kv", [False, True])
@torch.inference_mode()
def test_sparse_attn_sharedkv_c4_only_multiple_query_rows(has_ori_kv: bool):
    if get_ascend_device_type() != AscendDeviceType.A2:
        pytest.skip("C4-only SCFA device-loop regression requires Ascend A2")
    torch.npu.set_device(0)
    # Enough query groups for each core to process multiple rows and reset
    # s2Start. Zero Q gives uniform weights and an exact count-based reference.
    tokens, heads, head_dim, block_size, topk = 1024, 8, 512, 128, 512
    seq_len, window_size = 4096, 128
    q = torch.zeros((tokens, heads, head_dim), dtype=torch.bfloat16, device="npu")
    values = torch.arange(2 * block_size * head_dim, dtype=torch.float32).reshape(2, block_size, 1, head_dim)
    cmp_kv_cpu = ((values % 37 - 18) / 16).to(torch.bfloat16)
    cmp_kv = cmp_kv_cpu.npu()
    cmp_block_table = torch.tensor([[1, 0]], dtype=torch.int32, device="npu")
    indices = torch.full((tokens, 1, topk), -1, dtype=torch.int32)
    selected_indices = [0, 127, 128, 255]
    indices[:, 0, : len(selected_indices)] = torch.tensor(selected_indices, dtype=torch.int32)
    indices[::7] = -1  # Some tokens have no C4 row owned by this rank.
    indices_npu = indices.npu()
    query_start_loc = torch.tensor([0, tokens], dtype=torch.int32, device="npu")
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device="npu")
    metadata = torch.ops._C_ascend.npu_sparse_attn_sharedkv_metadata(
        heads,
        1,
        head_dim,
        cu_seqlens_q=query_start_loc,
        seqused_q=torch.tensor([tokens], dtype=torch.int32, device="npu"),
        seqused_kv=seq_lens,
        batch_size=1,
        max_seqlen_q=tokens,
        max_seqlen_kv=seq_len,
        cmp_topk=topk,
        cmp_ratio=4,
        ori_win_left=window_size - 1,
        layout_q="TND",
        layout_kv="PA_ND",
        has_ori_kv=has_ori_kv,
        has_cmp_kv=True,
    )
    # Resolve the deliberately permuted physical pages independently.
    selected_kv = torch.stack(
        (cmp_kv_cpu[1, 0, 0], cmp_kv_cpu[1, 127, 0], cmp_kv_cpu[0, 0, 0], cmp_kv_cpu[0, 127, 0])
    ).float()
    empty = indices[:, 0, 0] < 0
    counts = torch.full((tokens,), len(selected_indices), dtype=torch.float32)
    counts[empty] = 0
    numerator = selected_kv.sum(0).expand(tokens, head_dim).clone()
    numerator[empty] = 0

    for ori_value in (8.0, -8.0):
        ori_kv = torch.full((seq_len // block_size, block_size, 1, head_dim), ori_value, dtype=q.dtype, device="npu")
        output, lse = torch.ops._C_ascend.npu_sparse_attn_sharedkv(
            q,
            ori_kv=ori_kv,
            cmp_kv=cmp_kv,
            cmp_sparse_indices=indices_npu,
            ori_block_table=torch.arange(seq_len // block_size, dtype=torch.int32, device="npu").unsqueeze(0),
            cmp_block_table=cmp_block_table,
            cu_seqlens_q=query_start_loc,
            seqused_kv=seq_lens,
            sinks=torch.full((heads,), -1e30, dtype=torch.float32, device="npu"),
            metadata=metadata,
            softmax_scale=1 / math.sqrt(head_dim),
            cmp_ratio=4,
            ori_win_left=window_size - 1,
            layout_q="TND",
            layout_kv="PA_ND",
            return_softmax_lse=True,
        )
        if has_ori_kv:
            expected = (numerator + window_size * ori_value) / (counts + window_size).unsqueeze(1)
            expected_lse = (counts + window_size).log()
        else:
            mask_invalid_sparse_attention_rows(indices_npu, output, lse)
            expected = numerator / counts.clamp_min(1).unsqueeze(1)
            expected_lse = counts.log()
        torch.testing.assert_close(
            output.cpu().float(), expected.unsqueeze(1).expand(-1, heads, -1), rtol=1e-2, atol=2e-3
        )
        torch.testing.assert_close(lse.cpu(), expected_lse[:, None, None].expand(-1, heads, 1), rtol=1e-3, atol=1e-3)
