# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
from dataclasses import dataclass, replace

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

from vllm_ascend.utils import enable_custom_op  # noqa: E402

enable_custom_op()

BLOCK_SIZE = 128
HEAD_DIM = 128
TOPK = 2048
MISS_CAPACITY = 32768
INVALID_SLOT = -(1 << 31)
PADDING_ID = -1
MAX_ROUTES = 14


@dataclass
class ManageCase:
    q_values: list[int]
    states: list[int]
    actual_key: list[int]
    offload_key: list[int]
    cache_tokens: list[int]
    req_entries: list[int]
    index_weights: torch.Tensor
    query_dequant_scale: torch.Tensor
    query: torch.Tensor
    index_key_dequant_scale: torch.Tensor
    index_key_cache: torch.Tensor
    index_block_table: torch.Tensor
    route_block_table: torch.Tensor
    actual_seq_lengths_query: torch.Tensor
    actual_seq_lengths_key: torch.Tensor
    offload_seq_lengths_key: torch.Tensor
    num_cache_tokens: torch.Tensor
    request_state: torch.Tensor
    req_pool_entries: torch.Tensor
    cache_seed: torch.Tensor


def _cumulative(values: list[int]) -> list[int]:
    result = []
    total = 0
    for value in values:
        total += value
        result.append(total)
    return result


def _build_case(
    *,
    q_values: list[int],
    states: list[int],
    offload_len: int,
    cache_tokens: int,
    dtype: torch.dtype = torch.bfloat16,
    heads: int = 32,
    seed: int = 7,
    random_block_table: bool = True,
    validate_routes: bool = True,
) -> ManageCase:
    assert len(q_values) == len(states)
    if validate_routes:
        assert all(1 <= q <= MAX_ROUTES for q in q_values)
    assert offload_len % BLOCK_SIZE == 0

    batch_size = len(q_values)
    total_queries = sum(q_values)
    actual_len = offload_len + BLOCK_SIZE
    source_capacity = actual_len
    block_count = source_capacity // BLOCK_SIZE
    pool_size = batch_size * 2 + 1
    req_entries = [request * 2 + 1 for request in range(batch_size)]

    torch.manual_seed(seed)
    query = torch.randn(total_queries, heads, HEAD_DIM, dtype=dtype, device="npu")
    index_weights = torch.randn(total_queries, heads, dtype=dtype, device="npu")
    index_key_cache = torch.randn(
        block_count,
        BLOCK_SIZE,
        1,
        HEAD_DIM,
        dtype=dtype,
        device="npu",
    )

    if random_block_table:
        generator = torch.Generator().manual_seed(seed + 1009)
        table_cpu = torch.stack(
            [torch.randperm(block_count, generator=generator, dtype=torch.int64) for _ in range(batch_size)]
        ).to(torch.int32)
    else:
        table_cpu = torch.arange(block_count, dtype=torch.int32).repeat(batch_size, 1)
    index_block_table = table_cpu.to("npu")

    query_to_request = [request for request, q in enumerate(q_values) for _ in range(q)]
    route_block_table = index_block_table[
        torch.tensor(query_to_request, dtype=torch.int64, device="npu")
    ].contiguous()

    cache_cpu = torch.full((pool_size, source_capacity), INVALID_SLOT, dtype=torch.int32)
    for request, state in enumerate(states):
        if state == -1:
            row = req_entries[request]
            cache_cpu[row, :cache_tokens] = torch.arange(cache_tokens, dtype=torch.int32)

    query_ends = _cumulative(q_values)
    actual_key = [actual_len] * batch_size
    offload_key = [offload_len] * batch_size
    cache_sizes = [cache_tokens] * batch_size

    def int_tensor(values: list[int]) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.int32, device="npu")

    return ManageCase(
        q_values=q_values,
        states=states,
        actual_key=actual_key,
        offload_key=offload_key,
        cache_tokens=cache_sizes,
        req_entries=req_entries,
        index_weights=index_weights,
        query_dequant_scale=torch.zeros(total_queries, heads, dtype=torch.float32, device="npu"),
        query=query,
        index_key_dequant_scale=torch.zeros(
            block_count,
            BLOCK_SIZE,
            1,
            dtype=torch.float32,
            device="npu",
        ),
        index_key_cache=index_key_cache,
        index_block_table=index_block_table,
        route_block_table=route_block_table,
        actual_seq_lengths_query=int_tensor(query_ends),
        actual_seq_lengths_key=int_tensor(actual_key),
        offload_seq_lengths_key=int_tensor(offload_key),
        num_cache_tokens=int_tensor(cache_sizes),
        request_state=int_tensor(states),
        req_pool_entries=int_tensor(req_entries),
        cache_seed=cache_cpu.to("npu"),
    )


def _make_outputs(case: ManageCase) -> tuple[torch.Tensor, ...]:
    total_queries = case.query.size(0)
    batch_size = len(case.q_values)
    device = case.query.device
    return (
        torch.full((total_queries, 1, TOPK), -313, dtype=torch.int32, device=device),
        torch.full((total_queries, 1, TOPK), -313, dtype=torch.int32, device=device),
        torch.full((total_queries,), -313, dtype=torch.int32, device=device),
        torch.full((batch_size, MISS_CAPACITY), -313, dtype=torch.int32, device=device),
        torch.full((batch_size, MISS_CAPACITY), -313, dtype=torch.int32, device=device),
        torch.full((batch_size,), -313, dtype=torch.int32, device=device),
    )


def _call_op(case: ManageCase, cache: torch.Tensor, outputs: tuple[torch.Tensor, ...]) -> None:
    assert hasattr(torch.ops, "_C_ascend")
    assert hasattr(torch.ops._C_ascend, "npu_fused_lightning_indexer_manage")
    result = torch.ops._C_ascend.npu_fused_lightning_indexer_manage(
        index_weights=case.index_weights,
        query_dequant_scale=case.query_dequant_scale,
        query=case.query,
        index_key_dequant_scale=case.index_key_dequant_scale,
        index_key_cache=case.index_key_cache,
        index_block_table=case.index_block_table,
        actual_seq_lengths_query=case.actual_seq_lengths_query,
        actual_seq_lengths_key=case.actual_seq_lengths_key,
        offload_seq_lengths_key=case.offload_seq_lengths_key,
        num_cache_tokens=case.num_cache_tokens,
        request_state=case.request_state,
        req_pool_entries=case.req_pool_entries,
        cache_slots_pool=cache,
        topk_src_ids=outputs[0],
        topk_dst_slots=outputs[1],
        topk_miss_counts=outputs[2],
        miss_src_ids=outputs[3],
        miss_dst_slots=outputs[4],
        miss_counts=outputs[5],
    )
    assert result is None


def _visible_lengths(case: ManageCase) -> list[int]:
    result = []
    for request, q in enumerate(case.q_values):
        for route in range(q):
            if case.states[request] == -3:
                result.append(case.actual_key[request] - (q - 1 - route))
            else:
                result.append(case.offload_key[request])
    return result


def _native_topk(case: ManageCase) -> torch.Tensor:
    rows = []
    for route, visible_len in enumerate(_visible_lengths(case)):
        result = torch_npu.npu_lightning_indexer(
            query=case.query[route : route + 1],
            key=case.index_key_cache,
            weights=case.index_weights[route : route + 1],
            actual_seq_lengths_query=torch.tensor([1], dtype=torch.int32, device="npu"),
            actual_seq_lengths_key=torch.tensor([visible_len], dtype=torch.int32, device="npu"),
            block_table=case.route_block_table[route : route + 1],
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=TOPK,
            sparse_mode=0,
        )
        output = result[0] if isinstance(result, (tuple, list)) else result
        rows.append(output.reshape(-1)[:TOPK])
    return torch.stack(rows)


def _assert_resident_bijection(cache_row: torch.Tensor, length: int, capacity: int) -> None:
    resident_slots = cache_row[:length]
    resident_slots = resident_slots[resident_slots >= 0]
    assert resident_slots.numel() == capacity
    torch.testing.assert_close(
        resident_slots.sort().values,
        torch.arange(capacity, dtype=torch.int32),
        rtol=0,
        atol=0,
    )


def _run_and_assert(case: ManageCase) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    cache = case.cache_seed.clone()
    old_cache = cache.cpu()
    reference = _native_topk(case).cpu()
    outputs = _make_outputs(case)

    _call_op(case, cache, outputs)
    torch.npu.synchronize()

    src, dst, route_miss, miss_src, miss_dst, miss_count = [tensor.cpu() for tensor in outputs]
    cache_cpu = cache.cpu()
    query_start = 0

    for request, q in enumerate(case.q_values):
        query_end = query_start + q
        state = case.states[request]
        row = case.req_entries[request]
        length = case.actual_key[request] if state == -3 else case.offload_key[request]
        valid = min(length, TOPK)

        for route in range(query_start, query_end):
            actual_topk = src[route, 0, :valid]
            expected_topk = reference[route, :valid]
            if state == -3:
                torch.testing.assert_close(actual_topk, expected_topk, rtol=0, atol=0)
                torch.testing.assert_close(dst[route], src[route], rtol=0, atol=0)
                assert int(route_miss[route]) == 0
            else:
                torch.testing.assert_close(
                    actual_topk.sort().values,
                    expected_topk.sort().values,
                    rtol=0,
                    atol=0,
                )
                for position in range(valid):
                    source = int(src[route, 0, position])
                    assert int(dst[route, 0, position]) == int(cache_cpu[row, source])

            if valid < TOPK:
                assert torch.all(src[route, 0, valid:] == PADDING_ID)
                assert torch.all(dst[route, 0, valid:] == PADDING_ID)

        if state == -3:
            assert int(miss_count[request]) == 0
            torch.testing.assert_close(
                cache_cpu[row],
                torch.arange(cache_cpu.size(1), dtype=torch.int32),
                rtol=0,
                atol=0,
            )
        elif state == -2:
            capacity = case.cache_tokens[request]
            assert int(miss_count[request]) == capacity
            assert torch.all(route_miss[query_start:query_end] == TOPK)
            torch.testing.assert_close(
                miss_dst[request, :capacity],
                torch.arange(capacity, dtype=torch.int32),
                rtol=0,
                atol=0,
            )
            union = torch.unique(src[query_start:query_end].reshape(-1), sorted=True)
            union = union[union >= 0]
            selected = torch.zeros(length, dtype=torch.bool)
            selected[union.to(torch.int64)] = True
            remainder = torch.arange(length, dtype=torch.int32)[~selected]
            expected_miss_src = torch.cat((union, remainder))[:capacity]
            torch.testing.assert_close(miss_src[request, :capacity], expected_miss_src, rtol=0, atol=0)
            _assert_resident_bijection(cache_cpu[row], length, capacity)
        else:
            expected_union = torch.unique(reference[query_start:query_end].reshape(-1), sorted=True)
            expected_misses = expected_union[
                old_cache[row, expected_union.to(torch.int64)] == INVALID_SLOT
            ]
            count = int(miss_count[request])
            assert count == expected_misses.numel()
            torch.testing.assert_close(miss_src[request, :count], expected_misses, rtol=0, atol=0)

            for route in range(query_start, query_end):
                route_sources = reference[route]
                expected_route_misses = int(
                    (old_cache[row, route_sources.to(torch.int64)] == INVALID_SLOT).sum()
                )
                assert int(route_miss[route]) == expected_route_misses

            _assert_resident_bijection(cache_cpu[row], length, case.cache_tokens[request])

        count = int(miss_count[request])
        for position in range(count):
            source = int(miss_src[request, position])
            assert int(miss_dst[request, position]) == int(cache_cpu[row, source])

        query_start = query_end

    active_rows = set(case.req_entries)
    for row in range(cache_cpu.size(0)):
        if row not in active_rows:
            torch.testing.assert_close(cache_cpu[row], old_cache[row], rtol=0, atol=0)

    return cache, outputs


@pytest.mark.parametrize(
    "dtype,heads",
    [(torch.bfloat16, 32), (torch.float16, 64)],
)
@torch.inference_mode()
def test_fused_lightning_indexer_manage_non_offload_matches_native(dtype, heads):
    case = _build_case(
        q_values=[1],
        states=[-3],
        offload_len=896,
        cache_tokens=896,
        dtype=dtype,
        heads=heads,
    )
    _run_and_assert(case)


@torch.inference_mode()
def test_fused_lightning_indexer_manage_first_decode_initializes_cache():
    case = _build_case(q_values=[1], states=[-2], offload_len=2176, cache_tokens=2048)
    _run_and_assert(case)


@torch.inference_mode()
def test_fused_lightning_indexer_manage_steady_replacement_then_all_hit():
    case = _build_case(q_values=[1], states=[-1], offload_len=2176, cache_tokens=2048)
    updated_cache, outputs = _run_and_assert(case)
    assert int(outputs[5][0]) > 0

    repeated = replace(case, cache_seed=updated_cache.clone())
    repeated_cache, repeated_outputs = _run_and_assert(repeated)
    assert torch.all(repeated_outputs[2] == 0)
    assert torch.all(repeated_outputs[5] == 0)
    torch.testing.assert_close(repeated_cache, updated_cache, rtol=0, atol=0)


@torch.inference_mode()
def test_fused_lightning_indexer_manage_mixed_state_mtp():
    case = _build_case(
        q_values=[1, 4, 8],
        states=[-3, -2, -1],
        offload_len=16512,
        cache_tokens=16384,
        dtype=torch.float16,
        heads=64,
    )
    _run_and_assert(case)


@torch.inference_mode()
def test_fused_lightning_indexer_manage_lifecycle():
    common = dict(q_values=[1, 4], offload_len=8320, cache_tokens=8192)

    for sequence in ((-2, -1, -1), (-3, -2, -1)):
        cache = None
        last_outputs = None
        for state in sequence:
            case = _build_case(states=[state, state], **common)
            if cache is not None:
                case.cache_seed = cache
            cache, last_outputs = _run_and_assert(case)
        assert cache is not None and last_outputs is not None
        if sequence[-2:] == (-1, -1):
            assert torch.all(last_outputs[2] == 0)
            assert torch.all(last_outputs[5] == 0)

    standard = _build_case(states=[-3, -3], **common)
    identity_cache, _ = _run_and_assert(standard)
    transition = _build_case(states=[-1, -1], **common)
    transition.cache_seed = identity_cache
    transition_outputs = _make_outputs(transition)
    _call_op(transition, identity_cache, transition_outputs)
    torch.npu.synchronize()

    for request, length in enumerate(transition.offload_key):
        row = transition.req_entries[request]
        _assert_resident_bijection(identity_cache.cpu()[row], length, transition.cache_tokens[request])

    stable = replace(transition, cache_seed=identity_cache.clone())
    stable_cache, stable_outputs = _run_and_assert(stable)
    assert torch.all(stable_outputs[2] == 0)
    assert torch.all(stable_outputs[5] == 0)
    torch.testing.assert_close(stable_cache, identity_cache, rtol=0, atol=0)


@torch.inference_mode()
def test_fused_lightning_indexer_manage_q14_boundary():
    case = _build_case(
        q_values=[14],
        states=[-1],
        offload_len=32768,
        cache_tokens=32640,
    )
    _run_and_assert(case)


@torch.inference_mode()
def test_fused_lightning_indexer_manage_long_source_id():
    case = _build_case(
        q_values=[1],
        states=[-1],
        offload_len=131200,
        cache_tokens=8192,
        random_block_table=False,
    )
    case.query.fill_(1)
    case.index_weights.fill_(1)
    # Make every source in the first block above 2^17 an unambiguous TopK
    # member while retaining random scores elsewhere to avoid a tied cutoff.
    case.index_key_cache[1024].fill_(4)

    reference = _native_topk(case)
    assert torch.any(reference >= (1 << 17))
    _run_and_assert(case)


@torch.inference_mode()
def test_fused_lightning_indexer_manage_rejects_invalid_contract():
    case = _build_case(q_values=[4], states=[-1], offload_len=8320, cache_tokens=8192)

    bad_scale = replace(case, query_dequant_scale=case.query_dequant_scale.to(torch.float16))
    with pytest.raises(RuntimeError, match="dequant scales must be fp32"):
        _call_op(bad_scale, bad_scale.cache_seed.clone(), _make_outputs(bad_scale))

    bad_outputs = list(_make_outputs(case))
    bad_outputs[3] = torch.empty((1, 16384), dtype=torch.int32, device="npu")
    bad_outputs[4] = torch.empty_like(bad_outputs[3])
    with pytest.raises(RuntimeError, match="miss outputs must be"):
        _call_op(case, case.cache_seed.clone(), tuple(bad_outputs))

    noncontiguous_query = torch.randn(
        case.query.size(0),
        case.query.size(1),
        HEAD_DIM * 2,
        dtype=case.query.dtype,
        device="npu",
    )[..., ::2]
    assert not noncontiguous_query.is_contiguous()
    noncontiguous = replace(case, query=noncontiguous_query)
    with pytest.raises(RuntimeError, match="must be contiguous"):
        _call_op(noncontiguous, noncontiguous.cache_seed.clone(), _make_outputs(noncontiguous))

    q15 = _build_case(
        q_values=[15],
        states=[-3],
        offload_len=32768,
        cache_tokens=32640,
        validate_routes=False,
    )
    with pytest.raises(RuntimeError):
        _call_op(q15, q15.cache_seed.clone(), _make_outputs(q15))


def teardown_module():
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
