# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

HEAD_DIM = 512
NUM_Q_HEADS = 64
NUM_KV_HEADS = 1
INDEX_WIDTH = 512
BLOCK_SIZE = 16


def _reference_attention(
    query: torch.Tensor,
    rows_per_query: list[torch.Tensor],
    sinks: torch.Tensor,
) -> torch.Tensor:
    output = torch.empty_like(query)
    scale = 1.0 / math.sqrt(HEAD_DIM)
    for token_idx, kv_rows in enumerate(rows_per_query):
        key = kv_rows.expand(-1, NUM_Q_HEADS, -1).float()
        scores = torch.einsum("hd,khd->hk", query[token_idx].float(), key) * scale
        sink = sinks.float().view(NUM_Q_HEADS, 1)
        scores_max = torch.maximum(scores.max(dim=-1, keepdim=True).values, sink)
        exp_scores = torch.exp(scores - scores_max)
        probabilities = exp_scores / (exp_scores.sum(dim=-1, keepdim=True) + torch.exp(sink - scores_max))
        output[token_idx] = torch.einsum("hk,khd->hd", probabilities, key).to(query.dtype)
    return output


def _make_kv_cache(
    block_count: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    cache = torch.empty(block_count, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, device=device)
    flat_cache = cache.view(-1, NUM_KV_HEADS, HEAD_DIM)
    slot_ids = torch.arange(flat_cache.shape[0], device=device, dtype=torch.float32).view(-1, 1, 1)
    feature_ids = torch.arange(HEAD_DIM, device=device, dtype=torch.float32).view(1, 1, -1)
    # The first half drives MM1 scores while the second half makes MM2 values
    # slot-specific, so either gather using the wrong slot is independently visible.
    flat_cache[..., : HEAD_DIM // 2] = torch.sin(slot_ids * 0.013 + feature_ids[..., : HEAD_DIM // 2] * 0.017)
    flat_cache[..., HEAD_DIM // 2 :] = torch.cos(slot_ids * 0.019 + feature_ids[..., HEAD_DIM // 2 :] * 0.011)
    return (cache * 0.02).to(dtype)


def _run_scfa(
    query: torch.Tensor,
    ori_kv: torch.Tensor,
    cmp_kv: torch.Tensor,
    ori_sparse_indices: torch.Tensor | None,
    cmp_sparse_indices: torch.Tensor,
    ori_block_table: torch.Tensor,
    cmp_block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_kv: torch.Tensor,
    sinks: torch.Tensor,
    cmp_ratio: int,
) -> torch.Tensor:
    batch_size = seqused_kv.shape[0]
    metadata = torch.ops._C_ascend.npu_sparse_attn_sharedkv_metadata(
        num_heads_q=NUM_Q_HEADS,
        num_heads_kv=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        cu_seqlens_q=cu_seqlens_q,
        seqused_kv=seqused_kv,
        batch_size=batch_size,
        max_seqlen_q=int((cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item()),
        max_seqlen_kv=int(seqused_kv.max().item()),
        cmp_topk=cmp_sparse_indices.shape[-1],
        cmp_ratio=cmp_ratio,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        ori_win_left=127,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_ND",
        has_ori_kv=True,
        has_cmp_kv=True,
        device=str(query.device),
    )
    return torch.ops._C_ascend.npu_sparse_attn_sharedkv(
        query,
        ori_kv=ori_kv,
        cmp_kv=cmp_kv,
        ori_sparse_indices=ori_sparse_indices,
        cmp_sparse_indices=cmp_sparse_indices,
        ori_block_table=ori_block_table,
        cmp_block_table=cmp_block_table,
        cu_seqlens_q=cu_seqlens_q,
        seqused_kv=seqused_kv,
        sinks=sinks,
        metadata=metadata,
        softmax_scale=1.0 / math.sqrt(HEAD_DIM),
        cmp_ratio=cmp_ratio,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        ori_win_left=127,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_ND",
    )[0]


def _run_swa(
    query: torch.Tensor,
    ori_kv: torch.Tensor,
    ori_sparse_indices: torch.Tensor,
    ori_block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_kv: torch.Tensor,
    sinks: torch.Tensor,
) -> torch.Tensor:
    batch_size = seqused_kv.shape[0]
    metadata = torch.ops._C_ascend.npu_sparse_attn_sharedkv_metadata(
        num_heads_q=NUM_Q_HEADS,
        num_heads_kv=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        cu_seqlens_q=cu_seqlens_q,
        seqused_kv=seqused_kv,
        batch_size=batch_size,
        max_seqlen_q=int((cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item()),
        max_seqlen_kv=int(seqused_kv.max().item()),
        cmp_topk=0,
        cmp_ratio=1,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        ori_win_left=127,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_ND",
        has_ori_kv=True,
        has_cmp_kv=False,
        device=str(query.device),
    )
    return torch.ops._C_ascend.npu_sparse_attn_sharedkv(
        query,
        ori_kv=ori_kv,
        ori_sparse_indices=ori_sparse_indices,
        ori_block_table=ori_block_table,
        cu_seqlens_q=cu_seqlens_q,
        seqused_kv=seqused_kv,
        sinks=sinks,
        metadata=metadata,
        softmax_scale=1.0 / math.sqrt(HEAD_DIM),
        cmp_ratio=1,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        ori_win_left=127,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_ND",
    )[0]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("cmp_ratio", [4, 128])
@torch.inference_mode()
def test_scfa_original_slots_match_dense_reference(dtype: torch.dtype, cmp_ratio: int):
    torch.manual_seed(20260902 + cmp_ratio)
    device = torch.device("npu:0")
    q_lens = (2, 3)
    actual_kv_len = 512
    total_q = sum(q_lens)

    query = (torch.randn(total_q, NUM_Q_HEADS, HEAD_DIM, device=device) * 0.02).to(dtype)
    query[..., HEAD_DIM // 2 :] = 0
    ori_kv = _make_kv_cache(80, dtype, device)
    cmp_kv = _make_kv_cache(24, dtype, device).roll(shifts=31, dims=-1)
    ori_block_table = torch.tensor(
        [list(range(63, 31, -1)), list(range(31, -1, -1))],
        dtype=torch.int32,
        device=device,
    )
    cmp_block_table = torch.tensor(
        [[7, 1, 9, 3, 11, 5, 10, 0], [8, 2, 12, 4, 14, 6, 13, 15]],
        dtype=torch.int32,
        device=device,
    )
    cu_seqlens_q = torch.tensor([0, q_lens[0], total_q], dtype=torch.int32, device=device)
    seqused_kv = torch.full((2,), actual_kv_len, dtype=torch.int32, device=device)
    sinks = torch.linspace(-0.3, 0.3, NUM_Q_HEADS, dtype=torch.float32, device=device)

    ori_sparse_indices = torch.full((total_q, NUM_KV_HEADS, INDEX_WIDTH), -1, dtype=torch.int32, device=device)
    slot_rows: list[torch.Tensor] = []
    visible_counts = (0, 140, 257, 389, 511)
    for token_idx, visible_count in enumerate(visible_counts):
        slots = ((torch.arange(visible_count, device=device) * 37 + token_idx * 83) % 1280).to(torch.int32)
        ori_sparse_indices[token_idx, 0, :visible_count] = slots
        slot_rows.append(slots)

    cmp_sparse_indices = torch.arange(512, dtype=torch.int32, device=device).remainder(actual_kv_len // cmp_ratio)
    cmp_sparse_indices = cmp_sparse_indices.view(1, 1, -1).expand(total_q, NUM_KV_HEADS, -1).contiguous()
    actual = _run_scfa(
        query,
        ori_kv,
        cmp_kv,
        ori_sparse_indices,
        cmp_sparse_indices,
        ori_block_table,
        cmp_block_table,
        cu_seqlens_q,
        seqused_kv,
        sinks,
        cmp_ratio,
    )

    flat_ori_kv = ori_kv.reshape(-1, NUM_KV_HEADS, HEAD_DIM)
    rows_per_query = []
    token_idx = 0
    for batch_idx, q_len in enumerate(q_lens):
        for query_idx in range(q_len):
            cmp_count = min(512, (actual_kv_len - q_len + query_idx + 1) // cmp_ratio)
            logical_cmp = cmp_sparse_indices[token_idx, 0, :cmp_count].long()
            physical_cmp_blocks = cmp_block_table[batch_idx, logical_cmp // BLOCK_SIZE].long()
            cmp_rows = cmp_kv[physical_cmp_blocks, logical_cmp % BLOCK_SIZE]
            rows_per_query.append(torch.cat((flat_ori_kv[slot_rows[token_idx].long()], cmp_rows)))
            token_idx += 1
    expected = _reference_attention(query, rows_per_query, sinks)

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=3e-2, rtol=3e-2)


@torch.inference_mode()
def test_scfa_without_original_slots_preserves_contiguous_window():
    torch.manual_seed(20260903)
    device = torch.device("npu:0")
    dtype = torch.bfloat16
    actual_kv_len = 256
    query = (torch.randn(1, NUM_Q_HEADS, HEAD_DIM, device=device) * 0.02).to(dtype)
    ori_kv = (torch.randn(16, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, device=device) * 0.02).to(dtype)
    cmp_kv = (torch.randn(4, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, device=device) * 0.02).to(dtype)
    ori_block_table = torch.tensor(
        [[6, 1, 5, 0, 7, 2, 4, 3, 8, 9, 10, 11, 12, 13, 14, 15]],
        dtype=torch.int32,
        device=device,
    )
    cmp_block_table = torch.tensor([[3]], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    seqused_kv = torch.tensor([actual_kv_len], dtype=torch.int32, device=device)
    sinks = torch.zeros(NUM_Q_HEADS, dtype=torch.float32, device=device)
    cmp_sparse_indices = torch.zeros(1, NUM_KV_HEADS, 512, dtype=torch.int32, device=device)

    actual = _run_scfa(
        query,
        ori_kv,
        cmp_kv,
        None,
        cmp_sparse_indices,
        ori_block_table,
        cmp_block_table,
        cu_seqlens_q,
        seqused_kv,
        sinks,
        4,
    )
    logical_ori = torch.arange(actual_kv_len, device=device)
    physical_ori_blocks = ori_block_table[0, logical_ori // BLOCK_SIZE].long()
    ori_rows = ori_kv[physical_ori_blocks, logical_ori % BLOCK_SIZE]
    cmp_rows = cmp_kv[cmp_block_table[0, 0].long(), 0].unsqueeze(0).expand(16, -1, -1)
    expected = _reference_attention(query, [torch.cat((ori_rows, cmp_rows))], sinks)

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=3e-2, rtol=3e-2)


@torch.inference_mode()
def test_swa_ratio_one_and_scfa_match_for_same_original_slots():
    torch.manual_seed(20260904)
    device = torch.device("npu:0")
    dtype = torch.bfloat16
    actual_kv_len = 512
    query = torch.ones(1, NUM_Q_HEADS, HEAD_DIM, dtype=dtype, device=device)
    query[..., HEAD_DIM // 2 :] = 0
    ori_kv = _make_kv_cache(40, dtype, device)
    cmp_kv = torch.zeros(1, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=dtype, device=device)
    cmp_kv[..., : HEAD_DIM // 2] = -1000
    ori_block_table = torch.arange(31, -1, -1, dtype=torch.int32, device=device).view(1, -1)
    cmp_block_table = torch.zeros(1, 1, dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    seqused_kv = torch.tensor([actual_kv_len], dtype=torch.int32, device=device)
    sinks = torch.full((NUM_Q_HEADS,), -10000.0, dtype=torch.float32, device=device)
    slots = ((torch.arange(509, device=device) * 29 + 17) % (40 * BLOCK_SIZE)).to(torch.int32)
    ori_sparse_indices = torch.full((1, NUM_KV_HEADS, INDEX_WIDTH), -1, dtype=torch.int32, device=device)
    ori_sparse_indices[0, 0, : slots.numel()] = slots

    swa_output = _run_swa(
        query,
        ori_kv,
        ori_sparse_indices,
        ori_block_table,
        cu_seqlens_q,
        seqused_kv,
        sinks,
    )
    expected = _reference_attention(query, [ori_kv.reshape(-1, NUM_KV_HEADS, HEAD_DIM)[slots.long()]], sinks)
    cmp_sparse_indices = torch.zeros((1, NUM_KV_HEADS, 512), dtype=torch.int32, device=device)
    scfa_output = _run_scfa(
        query,
        ori_kv,
        cmp_kv,
        ori_sparse_indices,
        cmp_sparse_indices,
        ori_block_table,
        cmp_block_table,
        cu_seqlens_q,
        seqused_kv,
        sinks,
        128,
    )

    assert torch.isfinite(swa_output).all()
    assert torch.isfinite(scfa_output).all()
    torch.testing.assert_close(swa_output.cpu(), expected.cpu(), atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(scfa_output.cpu(), swa_output.cpu(), atol=3e-2, rtol=3e-2)


@torch.inference_mode()
def test_swa_service_geometry_block32_matches_dense_reference():
    torch.manual_seed(20260905)
    device = torch.device("npu:0")
    dtype = torch.bfloat16
    block_size = 32
    total_q = 237

    query = (torch.randn(total_q, NUM_Q_HEADS, HEAD_DIM, device=device) * 0.02).to(dtype)
    query[..., HEAD_DIM // 2 :] = 0
    ori_kv = torch.empty(224, block_size, NUM_KV_HEADS, HEAD_DIM, dtype=dtype, device=device)
    flat_ori_kv = ori_kv.view(-1, NUM_KV_HEADS, HEAD_DIM)
    slot_ids = torch.arange(flat_ori_kv.shape[0], device=device, dtype=torch.float32).view(-1, 1, 1)
    feature_ids = torch.arange(HEAD_DIM, device=device, dtype=torch.float32).view(1, 1, -1)
    flat_ori_kv[..., : HEAD_DIM // 2] = (
        torch.sin(slot_ids * 0.013 + feature_ids[..., : HEAD_DIM // 2] * 0.017) * 0.02
    ).to(dtype)
    flat_ori_kv[..., HEAD_DIM // 2 :] = (
        torch.cos(slot_ids * 0.019 + feature_ids[..., HEAD_DIM // 2 :] * 0.011) * 0.02
    ).to(dtype)

    ori_block_table = torch.tensor([[190, 3, 141, 72, 201, 18, 155, 44]], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor([0, total_q], dtype=torch.int32, device=device)
    seqused_kv = torch.tensor([total_q], dtype=torch.int32, device=device)
    sinks = torch.linspace(-0.3, 0.3, NUM_Q_HEADS, dtype=torch.float32, device=device)
    ori_sparse_indices = torch.full((total_q, NUM_KV_HEADS, INDEX_WIDTH), -1, dtype=torch.int32, device=device)
    slot_rows: list[torch.Tensor] = []
    cols = torch.arange(INDEX_WIDTH, device=device)
    for token_idx in range(total_q):
        start = max(token_idx - 127, 0)
        end = token_idx
        if 81 <= token_idx <= 216:
            start = min(start, 81)
            end = 216
        logical_slots = start + cols
        logical_slots = logical_slots[logical_slots <= end]
        physical_blocks = ori_block_table[0, logical_slots // block_size].long()
        slots = (physical_blocks * block_size + logical_slots % block_size).to(torch.int32)
        ori_sparse_indices[token_idx, 0, : slots.numel()] = slots
        slot_rows.append(slots)

    actual = _run_swa(
        query,
        ori_kv,
        ori_sparse_indices,
        ori_block_table,
        cu_seqlens_q,
        seqused_kv,
        sinks,
    )
    selected = [0, 80, 81, 128, 216, 217, 236]
    expected = _reference_attention(
        query[selected],
        [flat_ori_kv[slot_rows[token_idx].long()] for token_idx in selected],
        sinks,
    )

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual[selected].cpu(), expected.cpu(), atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("cmp_ratio", [4, 128])
@torch.inference_mode()
def test_scfa_service_geometry_block32_matches_dense_reference(cmp_ratio: int):
    torch.manual_seed(20260905 + cmp_ratio)
    device = torch.device("npu:0")
    dtype = torch.bfloat16
    block_size = 32
    total_q = 237
    actual_kv_len = total_q

    query = (torch.randn(total_q, NUM_Q_HEADS, HEAD_DIM, device=device) * 0.02).to(dtype)
    query[..., HEAD_DIM // 2 :] = 0

    ori_kv = torch.empty(224, block_size, NUM_KV_HEADS, HEAD_DIM, dtype=dtype, device=device)
    flat_ori_kv = ori_kv.view(-1, NUM_KV_HEADS, HEAD_DIM)
    slot_ids = torch.arange(flat_ori_kv.shape[0], device=device, dtype=torch.float32).view(-1, 1, 1)
    feature_ids = torch.arange(HEAD_DIM, device=device, dtype=torch.float32).view(1, 1, -1)
    flat_ori_kv[..., : HEAD_DIM // 2] = (
        torch.sin(slot_ids * 0.013 + feature_ids[..., : HEAD_DIM // 2] * 0.017) * 0.02
    ).to(dtype)
    flat_ori_kv[..., HEAD_DIM // 2 :] = (
        torch.cos(slot_ids * 0.019 + feature_ids[..., HEAD_DIM // 2 :] * 0.011) * 0.02
    ).to(dtype)

    ori_block_table = torch.tensor([[190, 3, 141, 72, 201, 18, 155, 44]], dtype=torch.int32, device=device)
    cmp_kv = _make_kv_cache(40, dtype, device).reshape(20, block_size, NUM_KV_HEADS, HEAD_DIM)
    cmp_block_table = torch.tensor([[17, 2, 19, 5, 13, 1, 11, 7]], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor([0, total_q], dtype=torch.int32, device=device)
    seqused_kv = torch.tensor([actual_kv_len], dtype=torch.int32, device=device)
    sinks = torch.linspace(-0.3, 0.3, NUM_Q_HEADS, dtype=torch.float32, device=device)

    ori_sparse_indices = torch.full((total_q, NUM_KV_HEADS, INDEX_WIDTH), -1, dtype=torch.int32, device=device)
    slot_rows: list[torch.Tensor] = []
    cols = torch.arange(INDEX_WIDTH, device=device)
    for token_idx in range(total_q):
        start = max(token_idx - 127, 0)
        end = token_idx
        if 81 <= token_idx <= 216:
            start = min(start, 81)
            end = 216
        logical_slots = start + cols
        valid = logical_slots <= end
        logical_slots = logical_slots[valid]
        physical_blocks = ori_block_table[0, logical_slots // block_size].long()
        slots = (physical_blocks * block_size + logical_slots % block_size).to(torch.int32)
        ori_sparse_indices[token_idx, 0, : slots.numel()] = slots
        slot_rows.append(slots)

    max_cmp_tokens = max((actual_kv_len + cmp_ratio - 1) // cmp_ratio, 1)
    cmp_sparse_indices = torch.arange(INDEX_WIDTH, dtype=torch.int32, device=device).remainder(max_cmp_tokens)
    cmp_sparse_indices = cmp_sparse_indices.view(1, 1, -1).expand(total_q, NUM_KV_HEADS, -1).contiguous()

    actual = _run_scfa(
        query,
        ori_kv,
        cmp_kv,
        ori_sparse_indices,
        cmp_sparse_indices,
        ori_block_table,
        cmp_block_table,
        cu_seqlens_q,
        seqused_kv,
        sinks,
        cmp_ratio,
    )

    selected = [0, 80, 81, 128, 216, 217, 236]
    rows_per_query = []
    for token_idx in selected:
        cmp_count = min(INDEX_WIDTH, (token_idx + 1) // cmp_ratio)
        logical_cmp = cmp_sparse_indices[token_idx, 0, :cmp_count].long()
        physical_cmp_blocks = cmp_block_table[0, logical_cmp // block_size].long()
        cmp_rows = cmp_kv[physical_cmp_blocks, logical_cmp % block_size]
        rows_per_query.append(torch.cat((flat_ori_kv[slot_rows[token_idx].long()], cmp_rows)))
    expected = _reference_attention(query[selected], rows_per_query, sinks)

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual[selected].cpu(), expected.cpu(), atol=3e-2, rtol=3e-2)
