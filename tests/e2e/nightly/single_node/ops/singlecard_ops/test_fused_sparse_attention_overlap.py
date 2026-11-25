import gc
import math
from dataclasses import dataclass

import pytest
import torch

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

BLOCK_SIZE = 128
KV_HEAD_DIM = 512
ROPE_HEAD_DIM = 64
QUERY_HEAD_COUNT = 4
KV_HEAD_COUNT = 1
SELECTION_TOPK_BLOCK_SIZE = 1
STATUS_ALIGNMENT = 8
MEMBERSHIP_MAX_TOKEN_COUNT = 16376
MEMBERSHIP_CONTROL_OFFSET = 16384
MEMBERSHIP_CONTROL_COUNT = 8
MEMBERSHIP_STORAGE_COUNT = 16400
MEMBERSHIP_READY_MARKER = 0x5A4D
SELECTION_PLAN_READY_MARKER = 0x5A50
SELECTION_EXTERNAL_PLAN_READY_MARKER = 0x5A45
DEFAULT_RTOL = 1e-2
DEFAULT_ATOL = 1e-2


@dataclass
class FusedSparseAttentionOverlapCase:
    query: torch.Tensor
    selection_k_rope: torch.Tensor
    selection_kv_cache: torch.Tensor
    selection_block_table: torch.Tensor
    selection_block_status: torch.Tensor
    selection_membership_map: torch.Tensor
    topk_indices: torch.Tensor
    full_k_rope: torch.Tensor
    full_kv_cache: torch.Tensor
    full_block_table: torch.Tensor
    full_actual_seq: torch.Tensor
    query_actual_seq: torch.Tensor
    reference_output: torch.Tensor
    topk: int


def _align_up(value: int, alignment: int) -> int:
    return math.ceil(value / alignment) * alignment


def _gather_paged_tokens(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    logical_blocks = token_ids.to(torch.int64) // BLOCK_SIZE
    block_offsets = token_ids.to(torch.int64) % BLOCK_SIZE
    physical_blocks = block_table[0, logical_blocks].to(torch.int64)
    return cache[physical_blocks, block_offsets]


def _reference_sparse_attention(
    query_nope: torch.Tensor,
    query_rope: torch.Tensor,
    full_kv_cache: torch.Tensor,
    full_k_rope: torch.Tensor,
    full_block_table: torch.Tensor,
    topk_indices: torch.Tensor,
    scale_value: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    token_ids = topk_indices[0, 0]
    selected_kv = _gather_paged_tokens(full_kv_cache, full_block_table, token_ids)
    selected_rope = _gather_paged_tokens(full_k_rope, full_block_table, token_ids)

    query = torch.cat([query_nope[0], query_rope[0]], dim=-1).float()
    key = torch.cat([selected_kv, selected_rope], dim=-1).float()
    scores = torch.matmul(query, key.transpose(0, 1)) * scale_value
    attention = torch.softmax(scores, dim=-1)
    return torch.matmul(attention, selected_kv.float()).to(dtype).unsqueeze(0)


def _make_case(dtype: torch.dtype, topk: int) -> FusedSparseAttentionOverlapCase:
    max_seq_len = max(topk * 2, BLOCK_SIZE * 2)
    logical_full_block_count = math.ceil(max_seq_len / BLOCK_SIZE)
    physical_full_block_count = logical_full_block_count + 1
    selection_block_count = math.ceil(topk / BLOCK_SIZE)
    physical_selection_block_count = selection_block_count + 1
    status_stride = _align_up(topk + 1, STATUS_ALIGNMENT)
    scale_value = 1.0 / math.sqrt(KV_HEAD_DIM + ROPE_HEAD_DIM)

    torch.manual_seed(2026 + topk)
    query_nope = torch.randn(1, QUERY_HEAD_COUNT, KV_HEAD_DIM, dtype=torch.float32).to(dtype)
    query_rope = torch.randn(1, QUERY_HEAD_COUNT, ROPE_HEAD_DIM, dtype=torch.float32).to(dtype)
    full_kv_cache = torch.randn(
        physical_full_block_count,
        BLOCK_SIZE,
        KV_HEAD_DIM,
        dtype=torch.float32,
    ).to(dtype)
    full_k_rope = torch.randn(
        physical_full_block_count,
        BLOCK_SIZE,
        ROPE_HEAD_DIM,
        dtype=torch.float32,
    ).to(dtype)
    full_block_table = torch.arange(
        physical_full_block_count - 1,
        0,
        -1,
        dtype=torch.int32,
    ).unsqueeze(0)
    topk_indices = torch.randperm(max_seq_len, dtype=torch.int64)[:topk].sort().values
    topk_indices = topk_indices.to(torch.int32).reshape(1, KV_HEAD_COUNT, topk)

    reference_output = _reference_sparse_attention(
        query_nope,
        query_rope,
        full_kv_cache,
        full_k_rope,
        full_block_table,
        topk_indices,
        scale_value,
        dtype,
    )

    selection_block_table = torch.arange(
        1,
        physical_selection_block_count,
        dtype=torch.int32,
    ).reshape(1, selection_block_count)
    selection_kv_cache = torch.full(
        (physical_selection_block_count, BLOCK_SIZE, KV_HEAD_DIM),
        float("nan"),
        dtype=dtype,
    )
    selection_k_rope = torch.full(
        (physical_selection_block_count, BLOCK_SIZE, ROPE_HEAD_DIM),
        float("nan"),
        dtype=dtype,
    )
    selection_block_status = torch.full(
        (1, KV_HEAD_COUNT, status_stride),
        -1,
        dtype=torch.int32,
    )
    selection_membership_map = torch.full(
        (1, KV_HEAD_COUNT, MEMBERSHIP_STORAGE_COUNT),
        -1,
        dtype=torch.int16,
    )

    return FusedSparseAttentionOverlapCase(
        query=torch.cat([query_nope, query_rope], dim=-1).contiguous().to("npu"),
        selection_k_rope=selection_k_rope.to("npu"),
        selection_kv_cache=selection_kv_cache.to("npu"),
        selection_block_table=selection_block_table.to("npu"),
        selection_block_status=selection_block_status.to("npu"),
        selection_membership_map=selection_membership_map.to("npu"),
        topk_indices=topk_indices.to("npu"),
        full_k_rope=full_k_rope.to("npu"),
        full_kv_cache=full_kv_cache.to("npu"),
        full_block_table=full_block_table.to("npu"),
        full_actual_seq=torch.tensor([max_seq_len], dtype=torch.int32, device="npu"),
        query_actual_seq=torch.tensor([1], dtype=torch.int32, device="npu"),
        reference_output=reference_output,
        topk=topk,
    )


def _call_fused(case: FusedSparseAttentionOverlapCase) -> torch.Tensor:
    return torch.ops._C_ascend.npu_fused_sparse_attention_overlap(
        query=case.query,
        selection_k_rope=case.selection_k_rope,
        selection_kv_cache=case.selection_kv_cache,
        selection_kv_block_table=case.selection_block_table,
        selection_kv_block_status=case.selection_block_status,
        selection_membership_map=case.selection_membership_map,
        selection_topk_indices=case.topk_indices,
        full_k_rope=case.full_k_rope,
        full_kv_cache=case.full_kv_cache,
        full_kv_block_table=case.full_block_table,
        full_kv_actual_seq=case.full_actual_seq,
        full_q_actual_seq=case.query_actual_seq,
        scale_value=1.0 / math.sqrt(KV_HEAD_DIM + ROPE_HEAD_DIM),
        sparse_block_size=1,
        selection_topk_block_size=SELECTION_TOPK_BLOCK_SIZE,
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=3,
    )


def _assert_attention_matches_reference(
    case: FusedSparseAttentionOverlapCase,
    output: torch.Tensor,
) -> None:
    actual = output.cpu()
    assert actual.shape == case.reference_output.shape
    assert actual.dtype == case.reference_output.dtype
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(
        actual,
        case.reference_output,
        rtol=DEFAULT_RTOL,
        atol=DEFAULT_ATOL,
    )


def _assert_selection_state_matches_full_cache(case: FusedSparseAttentionOverlapCase) -> None:
    status = case.selection_block_status.cpu()[0, 0]
    membership = case.selection_membership_map.cpu()[0, 0]
    topk_indices = case.topk_indices.cpu()[0, 0]
    selection_block_table = case.selection_block_table.cpu()
    full_block_table = case.full_block_table.cpu()
    selection_kv_cache = case.selection_kv_cache.cpu()
    selection_k_rope = case.selection_k_rope.cpu()
    full_kv_cache = case.full_kv_cache.cpu()
    full_k_rope = case.full_k_rope.cpu()

    valid_count = int(status[case.topk])
    resident_ids = status[:valid_count]
    assert valid_count == case.topk
    torch.testing.assert_close(
        resident_ids.sort().values,
        topk_indices.sort().values,
        rtol=0,
        atol=0,
    )

    slots = torch.arange(valid_count, dtype=torch.int64)
    selection_blocks = selection_block_table[0, slots // BLOCK_SIZE].to(torch.int64)
    selection_offsets = slots % BLOCK_SIZE
    expected_kv = _gather_paged_tokens(full_kv_cache, full_block_table, resident_ids)
    expected_rope = _gather_paged_tokens(full_k_rope, full_block_table, resident_ids)
    torch.testing.assert_close(
        selection_kv_cache[selection_blocks, selection_offsets],
        expected_kv,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        selection_k_rope[selection_blocks, selection_offsets],
        expected_rope,
        rtol=0,
        atol=0,
    )

    control = membership[MEMBERSHIP_CONTROL_OFFSET : MEMBERSHIP_CONTROL_OFFSET + MEMBERSHIP_CONTROL_COUNT]
    assert int(control[1]) != SELECTION_EXTERNAL_PLAN_READY_MARKER
    if int(control[0]) == MEMBERSHIP_READY_MARKER:
        expected_slots = torch.arange(1, valid_count + 1, dtype=torch.int16)
        torch.testing.assert_close(
            membership[resident_ids.to(torch.int64)],
            expected_slots,
            rtol=0,
            atol=0,
        )
        return

    assert int(control[1]) == SELECTION_PLAN_READY_MARKER
    plan_offset = int(control[3])
    assert 0 <= plan_offset <= MEMBERSHIP_MAX_TOKEN_COUNT - case.topk
    compact_plan = membership[plan_offset : plan_offset + case.topk].to(torch.int32)
    assert torch.all(compact_plan != 0)
    plan_slots = compact_plan.abs().to(torch.int64) - 1
    torch.testing.assert_close(
        resident_ids[plan_slots],
        topk_indices,
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("topk", [32, 128])
@torch.inference_mode()
def test_fused_sparse_attention_overlap_internal_planner_miss_then_all_hit_precision(
    dtype: torch.dtype,
    topk: int,
) -> None:
    assert hasattr(torch.ops._C_ascend, "npu_fused_sparse_attention_overlap")
    case = _make_case(dtype, topk)

    miss_output = _call_fused(case)
    torch.npu.synchronize()
    _assert_attention_matches_reference(case, miss_output)
    _assert_selection_state_matches_full_cache(case)

    status_after_miss = case.selection_block_status.clone()
    kv_after_miss = case.selection_kv_cache.clone()
    rope_after_miss = case.selection_k_rope.clone()

    all_hit_output = _call_fused(case)
    torch.npu.synchronize()
    _assert_attention_matches_reference(case, all_hit_output)
    _assert_selection_state_matches_full_cache(case)
    torch.testing.assert_close(case.selection_block_status, status_after_miss, rtol=0, atol=0)
    torch.testing.assert_close(case.selection_kv_cache, kv_after_miss, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(case.selection_k_rope, rope_after_miss, rtol=0, atol=0, equal_nan=True)

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
