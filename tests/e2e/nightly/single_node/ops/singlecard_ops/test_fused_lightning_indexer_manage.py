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
MAX_CACHE_TOKENS = 32640
INVALID_SLOT = -(1 << 31)
PADDING_ID = -1
MAX_ROUTES = 14
SMALL_HEAD_COUNT = 32
LARGE_HEAD_COUNT = 64
REQUEST_STATE_NON_OFFLOAD = -3
REQUEST_STATE_FIRST_DECODE = -2
REQUEST_STATE_STEADY = -1
OUTPUT_SENTINEL = -313
POOL_ENTRY_STRIDE = 2
FIRST_POOL_ENTRY = 1
BLOCK_TABLE_SEED_OFFSET = 1009
TOPK_SOURCE_OUTPUT = 0
TOPK_SLOT_OUTPUT = 1
TOPK_MISS_COUNT_OUTPUT = 2
MISS_SOURCE_OUTPUT = 3
MISS_SLOT_OUTPUT = 4
MISS_COUNT_OUTPUT = 5

FIRST_DECODE_SCENARIOS = (
    pytest.param([1], 8320, 8192, id="q1"),
    pytest.param([1, 2, 3], 8320, 8192, id="mixed-q1-q2-q3"),
    pytest.param([4], 16256, 12288, id="q4"),
    pytest.param([7], 16256, 14336, id="q7"),
)

STEADY_REPLACEMENT_SCENARIOS = (
    pytest.param([1], 8320, 8192, id="q1"),
    pytest.param([1, 2, 3], 8320, 8192, id="mixed-q1-q2-q3"),
    pytest.param([4], 16256, 12288, id="q4"),
    pytest.param([5], 16256, 12288, id="q5"),
    pytest.param([6], 16256, 12288, id="q6"),
    pytest.param([7], 16256, 14336, id="q7"),
)

ALL_ROUTE_SCENARIOS = (
    (1, 8320, 8192),
    (2, 8320, 8192),
    (3, 8320, 8192),
    (4, 16256, 12288),
    (5, 16256, 12288),
    (6, 16256, 12288),
    (7, 16256, 14336),
    (8, 16512, 16384),
    (9, 18560, 18432),
    (10, 20608, 20480),
    (11, 22656, 22528),
    (12, 24704, 24576),
    (13, 26752, 26624),
    (14, 32768, MAX_CACHE_TOKENS),
)

ROUTE_STATE_SCENARIOS = tuple(
    pytest.param(q, state, offload_len, cache_tokens, id=f"q{q}-state{state}")
    for q, offload_len, cache_tokens in ALL_ROUTE_SCENARIOS
    for state in (REQUEST_STATE_NON_OFFLOAD, REQUEST_STATE_FIRST_DECODE, REQUEST_STATE_STEADY)
)

DTYPE_HEAD_ROUTE_SCENARIOS = tuple(
    pytest.param(
        q,
        state,
        offload_len,
        cache_tokens,
        dtype,
        heads,
        id=f"q{q}-state{state}-{dtype_name}-h{heads}",
    )
    for dtype, dtype_name, heads in (
        (torch.bfloat16, "bf16", SMALL_HEAD_COUNT),
        (torch.bfloat16, "bf16", LARGE_HEAD_COUNT),
        (torch.float16, "fp16", SMALL_HEAD_COUNT),
        (torch.float16, "fp16", LARGE_HEAD_COUNT),
    )
    for q, state, offload_len, cache_tokens in (
        (1, REQUEST_STATE_NON_OFFLOAD, 8320, 8192),
        (4, REQUEST_STATE_FIRST_DECODE, 16256, 12288),
        (7, REQUEST_STATE_STEADY, 16256, 14336),
        (8, REQUEST_STATE_NON_OFFLOAD, 16512, 16384),
        (12, REQUEST_STATE_FIRST_DECODE, 24704, 24576),
        (14, REQUEST_STATE_STEADY, 32768, MAX_CACHE_TOKENS),
    )
)

MIXED_BATCH_SCENARIOS = (
    pytest.param(
        [1, 4, 7],
        [REQUEST_STATE_NON_OFFLOAD] * 3,
        16256,
        14336,
        id="local-all-non-offload",
    ),
    pytest.param(
        [1, 4, 7],
        [REQUEST_STATE_FIRST_DECODE] * 3,
        16256,
        14336,
        id="local-all-first-decode",
    ),
    pytest.param([1, 4, 7], [REQUEST_STATE_STEADY] * 3, 16256, 14336, id="local-all-steady"),
    pytest.param(
        [1, 4, 7],
        [REQUEST_STATE_NON_OFFLOAD, REQUEST_STATE_FIRST_DECODE, REQUEST_STATE_STEADY],
        16256,
        14336,
        id="local-mixed-state",
    ),
    pytest.param(
        [8, 12, 14],
        [REQUEST_STATE_NON_OFFLOAD] * 3,
        32768,
        MAX_CACHE_TOKENS,
        id="wide-all-non-offload",
    ),
    pytest.param(
        [8, 12, 14],
        [REQUEST_STATE_FIRST_DECODE] * 3,
        32768,
        MAX_CACHE_TOKENS,
        id="wide-all-first-decode",
    ),
    pytest.param(
        [8, 12, 14],
        [REQUEST_STATE_STEADY] * 3,
        32768,
        MAX_CACHE_TOKENS,
        id="wide-all-steady",
    ),
    pytest.param(
        [8, 12, 14],
        [REQUEST_STATE_NON_OFFLOAD, REQUEST_STATE_FIRST_DECODE, REQUEST_STATE_STEADY],
        32768,
        MAX_CACHE_TOKENS,
        id="wide-mixed-state",
    ),
    pytest.param(
        [1, 4, 7, 8, 12, 14],
        [
            REQUEST_STATE_NON_OFFLOAD,
            REQUEST_STATE_FIRST_DECODE,
            REQUEST_STATE_STEADY,
        ]
        * 2,
        32768,
        MAX_CACHE_TOKENS,
        id="mixed-local-wide-state",
    ),
)

LONG_SEQUENCE_SCENARIOS = (
    pytest.param(
        [1], [REQUEST_STATE_STEADY], 131200, 8192, 1 << 17, torch.bfloat16, SMALL_HEAD_COUNT, id="cross-2pow17"
    ),
    pytest.param(
        [4], [REQUEST_STATE_STEADY], 262272, 12288, 1 << 18, torch.bfloat16, SMALL_HEAD_COUNT, id="cross-2pow18"
    ),
    pytest.param(
        [7], [REQUEST_STATE_FIRST_DECODE], 524416, 14336, 1 << 19, torch.bfloat16, LARGE_HEAD_COUNT, id="cross-2pow19"
    ),
    pytest.param(
        [14],
        [REQUEST_STATE_STEADY],
        1048704,
        MAX_CACHE_TOKENS,
        1 << 20,
        torch.float16,
        LARGE_HEAD_COUNT,
        id="cross-2pow20",
    ),
    pytest.param(
        [1],
        [REQUEST_STATE_STEADY],
        2097024,
        8192,
        (1 << 21) - 256,
        torch.bfloat16,
        SMALL_HEAD_COUNT,
        id="near-2pow21-limit",
    ),
    pytest.param(
        [1, 4, 7],
        [REQUEST_STATE_NON_OFFLOAD, REQUEST_STATE_FIRST_DECODE, REQUEST_STATE_STEADY],
        131200,
        14336,
        1 << 17,
        torch.float16,
        LARGE_HEAD_COUNT,
        id="long-mixed-state-mtp",
    ),
)


@pytest.fixture(autouse=True)
def _show_test_progress(request: pytest.FixtureRequest):
    """Print this module's collection progress after every pytest case."""
    yield
    module_items = [item for item in request.session.items if item.path == request.node.path]
    position = module_items.index(request.node) + 1
    percent = (position * 100 + len(module_items) - 1) // len(module_items)
    print(f" [{position}/{len(module_items)}] [{percent}%]", flush=True)


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
    heads: int = SMALL_HEAD_COUNT,
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
    pool_size = batch_size * POOL_ENTRY_STRIDE + FIRST_POOL_ENTRY
    req_entries = [request * POOL_ENTRY_STRIDE + FIRST_POOL_ENTRY for request in range(batch_size)]

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
        generator = torch.Generator().manual_seed(seed + BLOCK_TABLE_SEED_OFFSET)
        table_cpu = torch.stack(
            [torch.randperm(block_count, generator=generator, dtype=torch.int64) for _ in range(batch_size)]
        ).to(torch.int32)
    else:
        table_cpu = torch.arange(block_count, dtype=torch.int32).repeat(batch_size, 1)
    index_block_table = table_cpu.to("npu")

    query_to_request = [request for request, q in enumerate(q_values) for _ in range(q)]
    route_block_table = index_block_table[torch.tensor(query_to_request, dtype=torch.int64, device="npu")].contiguous()

    cache_cpu = torch.full((pool_size, source_capacity), INVALID_SLOT, dtype=torch.int32)
    for request, state in enumerate(states):
        if state == REQUEST_STATE_STEADY:
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
        torch.full((total_queries, 1, TOPK), OUTPUT_SENTINEL, dtype=torch.int32, device=device),
        torch.full((total_queries, 1, TOPK), OUTPUT_SENTINEL, dtype=torch.int32, device=device),
        torch.full((total_queries,), OUTPUT_SENTINEL, dtype=torch.int32, device=device),
        torch.full((batch_size, MISS_CAPACITY), OUTPUT_SENTINEL, dtype=torch.int32, device=device),
        torch.full((batch_size, MISS_CAPACITY), OUTPUT_SENTINEL, dtype=torch.int32, device=device),
        torch.full((batch_size,), OUTPUT_SENTINEL, dtype=torch.int32, device=device),
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
        topk_src_ids=outputs[TOPK_SOURCE_OUTPUT],
        topk_dst_slots=outputs[TOPK_SLOT_OUTPUT],
        topk_miss_counts=outputs[TOPK_MISS_COUNT_OUTPUT],
        miss_src_ids=outputs[MISS_SOURCE_OUTPUT],
        miss_dst_slots=outputs[MISS_SLOT_OUTPUT],
        miss_counts=outputs[MISS_COUNT_OUTPUT],
    )
    assert result is None


def _visible_lengths(case: ManageCase) -> list[int]:
    result = []
    for request, q in enumerate(case.q_values):
        for route in range(q):
            if case.states[request] == REQUEST_STATE_NON_OFFLOAD:
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


def _build_occurrence_boundary_case(q: int, occurrence_count: int) -> ManageCase:
    cache_tokens = q * TOPK
    extra = ((occurrence_count + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    case = _build_case(
        q_values=[q],
        states=[REQUEST_STATE_STEADY],
        offload_len=cache_tokens + extra,
        cache_tokens=cache_tokens,
        random_block_table=False,
    )

    # Give every route one disjoint TOPK-sized positive-score interval. This
    # makes the requested miss-occurrence boundary deterministic.
    case.query.zero_()
    case.index_weights.fill_(1)
    case.index_key_cache.zero_()
    key_by_source = case.index_key_cache.reshape(-1, 1, HEAD_DIM)
    for route in range(q):
        case.query[route, :, route].fill_(1)
        key_by_source[route * TOPK : (route + 1) * TOPK, 0, route].fill_(1)

    reference = _native_topk(case).cpu().to(torch.int64)
    expected_union = torch.arange(cache_tokens, dtype=torch.int64)
    torch.testing.assert_close(
        torch.unique(reference.reshape(-1), sorted=True),
        expected_union,
        rtol=0,
        atol=0,
    )

    route_misses = [occurrence_count // q] * q
    for route in range(occurrence_count % q):
        route_misses[route] += 1
    missing = torch.cat([reference[route, : route_misses[route]] for route in range(q)])
    hits = torch.cat([reference[route, route_misses[route] :] for route in range(q)])
    assert missing.numel() == occurrence_count

    fillers = torch.arange(cache_tokens, cache_tokens + occurrence_count, dtype=torch.int64)
    cached = torch.cat((hits, fillers))
    assert cached.numel() == cache_tokens

    cache = torch.full_like(case.cache_seed.cpu(), INVALID_SLOT)
    row = case.req_entries[0]
    generator = torch.Generator().manual_seed(4096 + occurrence_count)
    cache[row, cached] = torch.randperm(cache_tokens, generator=generator, dtype=torch.int64).to(torch.int32)
    case.cache_seed = cache.to("npu")
    return case


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
        length = case.actual_key[request] if state == REQUEST_STATE_NON_OFFLOAD else case.offload_key[request]
        valid = min(length, TOPK)

        for route in range(query_start, query_end):
            actual_topk = src[route, 0, :valid]
            expected_topk = reference[route, :valid]
            if state == REQUEST_STATE_NON_OFFLOAD:
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

        if state == REQUEST_STATE_NON_OFFLOAD:
            assert int(miss_count[request]) == 0
            torch.testing.assert_close(
                cache_cpu[row],
                torch.arange(cache_cpu.size(1), dtype=torch.int32),
                rtol=0,
                atol=0,
            )
        elif state == REQUEST_STATE_FIRST_DECODE:
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
            expected_misses = expected_union[old_cache[row, expected_union.to(torch.int64)] == INVALID_SLOT]
            count = int(miss_count[request])
            assert count == expected_misses.numel()
            torch.testing.assert_close(miss_src[request, :count], expected_misses, rtol=0, atol=0)

            for route in range(query_start, query_end):
                route_sources = reference[route]
                expected_route_misses = int((old_cache[row, route_sources.to(torch.int64)] == INVALID_SLOT).sum())
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
    [(torch.bfloat16, SMALL_HEAD_COUNT), (torch.float16, LARGE_HEAD_COUNT)],
)
@torch.inference_mode()
def test_fused_lightning_indexer_manage_non_offload_matches_native(dtype, heads):
    case = _build_case(
        q_values=[1],
        states=[REQUEST_STATE_NON_OFFLOAD],
        offload_len=896,
        cache_tokens=896,
        dtype=dtype,
        heads=heads,
    )
    _run_and_assert(case)


@pytest.mark.parametrize("q_values,offload_len,cache_tokens", FIRST_DECODE_SCENARIOS)
@torch.inference_mode()
def test_fused_lightning_indexer_manage_first_decode_initializes_cache(q_values, offload_len, cache_tokens):
    case = _build_case(
        q_values=q_values,
        states=[REQUEST_STATE_FIRST_DECODE] * len(q_values),
        offload_len=offload_len,
        cache_tokens=cache_tokens,
    )
    _run_and_assert(case)


@pytest.mark.parametrize("q_values,offload_len,cache_tokens", STEADY_REPLACEMENT_SCENARIOS)
@torch.inference_mode()
def test_fused_lightning_indexer_manage_steady_replacement_then_all_hit(q_values, offload_len, cache_tokens):
    case = _build_case(
        q_values=q_values,
        states=[REQUEST_STATE_STEADY] * len(q_values),
        offload_len=offload_len,
        cache_tokens=cache_tokens,
    )
    updated_cache, outputs = _run_and_assert(case)
    assert torch.any(outputs[MISS_COUNT_OUTPUT] > 0)

    repeated = replace(case, cache_seed=updated_cache.clone())
    repeated_cache, repeated_outputs = _run_and_assert(repeated)
    assert torch.all(repeated_outputs[TOPK_MISS_COUNT_OUTPUT] == 0)
    assert torch.all(repeated_outputs[MISS_COUNT_OUTPUT] == 0)
    torch.testing.assert_close(repeated_cache, updated_cache, rtol=0, atol=0)


@pytest.mark.parametrize(
    "states",
    [
        pytest.param([REQUEST_STATE_NON_OFFLOAD] * 3, id="all-non-offload"),
        pytest.param([REQUEST_STATE_FIRST_DECODE] * 3, id="all-first-decode"),
        pytest.param([REQUEST_STATE_STEADY] * 3, id="all-steady"),
        pytest.param(
            [REQUEST_STATE_NON_OFFLOAD, REQUEST_STATE_FIRST_DECODE, REQUEST_STATE_STEADY],
            id="mixed-state",
        ),
    ],
)
@torch.inference_mode()
def test_fused_lightning_indexer_manage_state_patterns(states):
    case = _build_case(
        q_values=[1, 2, 3],
        states=states,
        offload_len=8320,
        cache_tokens=8192,
    )
    _run_and_assert(case)


@torch.inference_mode()
def test_fused_lightning_indexer_manage_mixed_state_mtp():
    case = _build_case(
        q_values=[1, 4, 8],
        states=[REQUEST_STATE_NON_OFFLOAD, REQUEST_STATE_FIRST_DECODE, REQUEST_STATE_STEADY],
        offload_len=16512,
        cache_tokens=16384,
        dtype=torch.float16,
        heads=LARGE_HEAD_COUNT,
    )
    _run_and_assert(case)


@torch.inference_mode()
def test_fused_lightning_indexer_manage_lifecycle():
    q_values = [1, 4]
    offload_len = 8320
    cache_tokens = 8192

    for sequence in (
        (REQUEST_STATE_FIRST_DECODE, REQUEST_STATE_STEADY, REQUEST_STATE_STEADY),
        (REQUEST_STATE_NON_OFFLOAD, REQUEST_STATE_FIRST_DECODE, REQUEST_STATE_STEADY),
    ):
        cache = None
        last_outputs = None
        for state in sequence:
            case = _build_case(
                q_values=q_values,
                states=[state, state],
                offload_len=offload_len,
                cache_tokens=cache_tokens,
            )
            if cache is not None:
                case.cache_seed = cache
            cache, last_outputs = _run_and_assert(case)
        assert cache is not None and last_outputs is not None
        if sequence[-2:] == (REQUEST_STATE_STEADY, REQUEST_STATE_STEADY):
            assert torch.all(last_outputs[TOPK_MISS_COUNT_OUTPUT] == 0)
            assert torch.all(last_outputs[MISS_COUNT_OUTPUT] == 0)

    standard = _build_case(
        q_values=q_values,
        states=[REQUEST_STATE_NON_OFFLOAD] * 2,
        offload_len=offload_len,
        cache_tokens=cache_tokens,
    )
    identity_cache, _ = _run_and_assert(standard)
    transition = _build_case(
        q_values=q_values,
        states=[REQUEST_STATE_STEADY] * 2,
        offload_len=offload_len,
        cache_tokens=cache_tokens,
    )
    transition.cache_seed = identity_cache
    transition_outputs = _make_outputs(transition)
    _call_op(transition, identity_cache, transition_outputs)
    torch.npu.synchronize()

    for request, length in enumerate(transition.offload_key):
        row = transition.req_entries[request]
        _assert_resident_bijection(identity_cache.cpu()[row], length, transition.cache_tokens[request])

    stable = replace(transition, cache_seed=identity_cache.clone())
    stable_cache, stable_outputs = _run_and_assert(stable)
    assert torch.all(stable_outputs[TOPK_MISS_COUNT_OUTPUT] == 0)
    assert torch.all(stable_outputs[MISS_COUNT_OUTPUT] == 0)
    torch.testing.assert_close(stable_cache, identity_cache, rtol=0, atol=0)


@pytest.mark.parametrize(
    "q,state,offload_len,cache_tokens",
    ROUTE_STATE_SCENARIOS,
)
@torch.inference_mode()
def test_fused_lightning_indexer_manage_route_state_matrix(q, state, offload_len, cache_tokens):
    case = _build_case(
        q_values=[q],
        states=[state],
        offload_len=offload_len,
        cache_tokens=cache_tokens,
    )
    _run_and_assert(case)


@pytest.mark.parametrize(
    "q,state,offload_len,cache_tokens,dtype,heads",
    DTYPE_HEAD_ROUTE_SCENARIOS,
)
@torch.inference_mode()
def test_fused_lightning_indexer_manage_dtype_head_matrix(q, state, offload_len, cache_tokens, dtype, heads):
    case = _build_case(
        q_values=[q],
        states=[state],
        offload_len=offload_len,
        cache_tokens=cache_tokens,
        dtype=dtype,
        heads=heads,
    )
    _run_and_assert(case)


@pytest.mark.parametrize(
    "q_values,states,offload_len,cache_tokens",
    MIXED_BATCH_SCENARIOS,
)
@torch.inference_mode()
def test_fused_lightning_indexer_manage_mixed_batch_matrix(q_values, states, offload_len, cache_tokens):
    case = _build_case(
        q_values=q_values,
        states=states,
        offload_len=offload_len,
        cache_tokens=cache_tokens,
    )
    _run_and_assert(case)


@pytest.mark.parametrize(
    "q,occurrence_count",
    [
        pytest.param(5, 2048, id="q5-occurrence-2048"),
        pytest.param(5, 2049, id="q5-occurrence-2049"),
        pytest.param(6, 4095, id="q6-occurrence-4095"),
        pytest.param(7, 4096, id="q7-occurrence-4096"),
        pytest.param(7, 4097, id="q7-occurrence-4097"),
    ],
)
@torch.inference_mode()
def test_fused_lightning_indexer_manage_occurrence_sort_boundaries(q, occurrence_count):
    case = _build_occurrence_boundary_case(q, occurrence_count)
    _, outputs = _run_and_assert(case)
    assert int(outputs[TOPK_MISS_COUNT_OUTPUT].sum()) == occurrence_count
    assert int(outputs[MISS_COUNT_OUTPUT][0]) == occurrence_count


@pytest.mark.parametrize(
    "q_values,states,offload_len,cache_tokens,min_source,dtype,heads",
    LONG_SEQUENCE_SCENARIOS,
)
@torch.inference_mode()
def test_fused_lightning_indexer_manage_long_sequence_source_ids(
    q_values, states, offload_len, cache_tokens, min_source, dtype, heads
):
    case = _build_case(
        q_values=q_values,
        states=states,
        offload_len=offload_len,
        cache_tokens=cache_tokens,
        dtype=dtype,
        heads=heads,
        random_block_table=False,
    )
    case.query.fill_(1)
    case.index_weights.fill_(1)

    # Give the target high-ID block distinct dominant scores so every route
    # selects long-sequence sources without introducing a tied TopK cutoff.
    target_block = min_source // BLOCK_SIZE
    for offset in range(BLOCK_SIZE):
        case.index_key_cache[target_block, offset].fill_(4.0 + offset * 0.125)

    reference = _native_topk(case)
    assert torch.all(torch.any(reference >= min_source, dim=1))
    _run_and_assert(case)


@torch.inference_mode()
def test_fused_lightning_indexer_manage_rejects_invalid_contract():
    case = _build_case(
        q_values=[4],
        states=[REQUEST_STATE_STEADY],
        offload_len=8320,
        cache_tokens=8192,
    )

    bad_scale = replace(case, query_dequant_scale=case.query_dequant_scale.to(torch.float16))
    with pytest.raises(RuntimeError, match="dequant scales must be fp32"):
        _call_op(bad_scale, bad_scale.cache_seed.clone(), _make_outputs(bad_scale))

    bad_outputs = list(_make_outputs(case))
    bad_outputs[MISS_SOURCE_OUTPUT] = torch.empty((1, MISS_CAPACITY // 2), dtype=torch.int32, device="npu")
    bad_outputs[MISS_SLOT_OUTPUT] = torch.empty_like(bad_outputs[MISS_SOURCE_OUTPUT])
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
        states=[REQUEST_STATE_NON_OFFLOAD],
        offload_len=32768,
        cache_tokens=MAX_CACHE_TOKENS,
        validate_routes=False,
    )
    with pytest.raises(RuntimeError):
        _call_op(q15, q15.cache_seed.clone(), _make_outputs(q15))


def teardown_module():
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
