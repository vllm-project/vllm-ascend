"""Generalized LIM contracts from upstream 012962af, executed on Ascend."""

from __future__ import annotations

import argparse

import pytest
import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

TOPK = 2048
BLOCK = 128
HEAD_DIM = 128
MISS_CAPACITY = 32768
INVALID_SLOT = -(1 << 31)
MAX_SOURCE_CAPACITY = 1 << 21


def cumulative(values: list[int]) -> list[int]:
    result: list[int] = []
    total = 0
    for value in values:
        total += value
        result.append(total)
    return result


def validate_dynamic_inputs(
    *,
    query_ends: list[int],
    actual_key: list[int],
    offload_key: list[int],
    cache_tokens: list[int],
    request_state: list[int],
    req_pool_entries: list[int],
    total_queries: int,
    source_capacity: int,
    pool_size: int,
) -> None:
    """Reference for checks performed by the scheduler before tensor creation."""

    if source_capacity <= 0 or source_capacity % BLOCK or source_capacity > MAX_SOURCE_CAPACITY:
        raise ValueError("source capacity must be 128-aligned and <=2^21")
    if pool_size <= 0:
        raise ValueError("pool_size must be positive")
    batch = len(query_ends)
    fields = (actual_key, offload_key, cache_tokens, request_state, req_pool_entries)
    if not batch or any(len(field) != batch for field in fields):
        raise ValueError("all request metadata must be non-empty and have length B")
    previous = 0
    for request in range(batch):
        end = query_ends[request]
        q = end - previous
        if not 1 <= q <= 14:
            raise ValueError("Q must be in [1,14]")
        if not q <= actual_key[request] <= source_capacity:
            raise ValueError("actual_seq_lengths_key is out of range")
        if not 0 <= req_pool_entries[request] < pool_size:
            raise ValueError("req_pool_entries is out of range")
        state = request_state[request]
        if state not in (-3, -2, -1):
            raise ValueError("request_state must be -3, -2 or -1")
        if state != -3:
            length = offload_key[request]
            capacity = cache_tokens[request]
            if not capacity <= length <= actual_key[request]:
                raise ValueError("requires C <= L <= actual_seq_lengths_key")
            if length < TOPK or length % BLOCK or capacity % BLOCK:
                raise ValueError("L/C alignment or minimum is invalid")
            causal_limit = ((actual_key[request] - q) // BLOCK) * BLOCK
            if length > causal_limit:
                raise ValueError("offload prefix is not visible to every query")
            if length <= q * TOPK:
                if capacity != length:
                    raise ValueError("C must equal L for a small offload prefix")
            elif not q * TOPK <= capacity <= 32640:
                raise ValueError("C is outside the multi-route cache budget")
        previous = end
    if previous != total_queries:
        raise ValueError("last actual_seq_lengths_query value must equal T")
    if len(set(req_pool_entries)) != batch:
        raise ValueError("active requests must use distinct pool rows")


def native_topk(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    block_table: torch.Tensor,
    visible_lengths: list[int],
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for row, visible in enumerate(visible_lengths):
        result = torch_npu.npu_lightning_indexer(
            query=query[row : row + 1],
            key=key,
            weights=weights[row : row + 1],
            actual_seq_lengths_query=torch.tensor([1], dtype=torch.int32, device=query.device),
            actual_seq_lengths_key=torch.tensor([visible], dtype=torch.int32, device=query.device),
            block_table=block_table[row : row + 1],
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=TOPK,
            sparse_mode=0,
        )
        output = result[0] if isinstance(result, (tuple, list)) else result
        rows.append(output.reshape(-1)[:TOPK])
    return torch.stack(rows)


def build_case(
    args: argparse.Namespace,
    *,
    q_values: list[int],
    states: list[int],
) -> dict[str, object]:
    if len(q_values) != len(states):
        raise ValueError("q_values and states must have equal length")
    if args.source_capacity % BLOCK or args.source_capacity > MAX_SOURCE_CAPACITY:
        raise ValueError("source capacity must be 128-aligned and <=2^21")
    batch = len(q_values)
    total_queries = sum(q_values)
    query_ends = cumulative(q_values)
    # The final route sees this length; adding one block makes the aligned
    # offload prefix causally visible even for Q=14.
    actual_key = [args.offload_len + BLOCK for _ in q_values]
    offload_key = [args.offload_len for _ in q_values]
    cache_tokens = [args.offload_len if args.offload_len <= q * TOPK else args.cache_tokens for q in q_values]
    # Active requests must own distinct pool rows.  Keep them deliberately
    # non-contiguous without wrapping: modulo (B + 3) aliases rows once B=7
    # (for example requests 0/5 and 1/6), creating invalid concurrent writes.
    pool_size = batch * 2 + 1
    req_entries = [request * 2 + 1 for request in range(batch)]
    validate_dynamic_inputs(
        query_ends=query_ends,
        actual_key=actual_key,
        offload_key=offload_key,
        cache_tokens=cache_tokens,
        request_state=states,
        req_pool_entries=req_entries,
        total_queries=total_queries,
        source_capacity=args.source_capacity,
        pool_size=pool_size,
    )
    torch.manual_seed(args.seed)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    device = torch.device(args.device)
    blocks = args.source_capacity // BLOCK
    query = torch.randn(total_queries, args.heads, HEAD_DIM, dtype=dtype, device=device)
    weights = torch.randn(total_queries, args.heads, dtype=dtype, device=device)
    key = torch.randn(blocks, BLOCK, 1, HEAD_DIM, dtype=dtype, device=device)
    block_table = torch.arange(blocks, dtype=torch.int32, device=device).repeat(batch, 1)
    # native_topk takes one block-table row per query.
    query_to_request = [request for request, q in enumerate(q_values) for _ in range(q)]
    route_table = block_table[torch.tensor(query_to_request, dtype=torch.int64, device=device)].contiguous()
    cache_cpu = torch.full((pool_size, args.source_capacity), INVALID_SLOT, dtype=torch.int32)
    for request, state in enumerate(states):
        if state != -1:
            continue
        row = req_entries[request]
        count = cache_tokens[request]
        cache_cpu[row, :count] = torch.arange(count, dtype=torch.int32)
    return {
        "q_values": q_values,
        "states": states,
        "query_ends": query_ends,
        "actual_key": actual_key,
        "offload_key": offload_key,
        "cache_tokens": cache_tokens,
        "req_entries": req_entries,
        "query": query,
        "weights": weights,
        "query_scale": torch.zeros(total_queries, args.heads, dtype=torch.float32, device=device),
        "key": key,
        "key_scale": torch.zeros(blocks, BLOCK, 1, dtype=torch.float32, device=device),
        "block_table": block_table,
        "route_table": route_table,
        "cache_seed": cache_cpu.to(device),
        "metadata": tuple(
            torch.tensor(values, dtype=torch.int32, device=device)
            for values in (
                query_ends,
                actual_key,
                offload_key,
                cache_tokens,
                states,
                req_entries,
            )
        ),
    }


def make_outputs(case: dict[str, object]) -> tuple[torch.Tensor, ...]:
    query = case["query"]
    assert isinstance(query, torch.Tensor)
    total_queries = query.size(0)
    batch = len(case["q_values"])
    device = query.device
    return (
        torch.full((total_queries, 1, TOPK), -313, dtype=torch.int32, device=device),
        torch.full((total_queries, 1, TOPK), -313, dtype=torch.int32, device=device),
        torch.full((total_queries,), -313, dtype=torch.int32, device=device),
        torch.full((batch, MISS_CAPACITY), -313, dtype=torch.int32, device=device),
        torch.full((batch, MISS_CAPACITY), -313, dtype=torch.int32, device=device),
        torch.full((batch,), -313, dtype=torch.int32, device=device),
    )


def call_custom(
    case: dict[str, object],
    cache: torch.Tensor,
    outputs: tuple[torch.Tensor, ...],
) -> None:
    query_ends, actual_key, offload_key, cache_tokens, states, req_entries = case["metadata"]
    torch.ops._C_ascend.npu_fused_li_manage_mtp.default(
        case["weights"],
        case["query_scale"],
        case["query"],
        case["key_scale"],
        case["key"],
        case["block_table"],
        query_ends,
        actual_key,
        offload_key,
        cache_tokens,
        states,
        req_entries,
        cache,
        *outputs,
    )


def visible_lengths(case: dict[str, object]) -> list[int]:
    result: list[int] = []
    for request, q in enumerate(case["q_values"]):
        state = case["states"][request]
        for route in range(q):
            result.append(
                case["actual_key"][request] - (q - 1 - route) if state == -3 else case["offload_key"][request]
            )
    return result


def assert_correctness(case: dict[str, object]) -> None:
    """Check causal selections, copy-list unions, and persistent slot maps."""
    cache = case["cache_seed"].clone()
    old_cache = cache.clone()
    outputs = make_outputs(case)
    reference = native_topk(
        case["query"], case["key"], case["weights"], case["route_table"], visible_lengths(case)
    ).cpu()
    call_custom(case, cache, outputs)
    torch.npu.synchronize()
    src, dst, route_miss, miss_src, miss_dst, miss_count = [x.cpu() for x in outputs]
    cache_cpu = cache.cpu()
    query_start = 0
    for request, q in enumerate(case["q_values"]):
        query_end = query_start + q
        state = case["states"][request]
        row = case["req_entries"][request]
        length = case["actual_key"][request] if state == -3 else case["offload_key"][request]
        for route in range(query_start, query_end):
            valid = min(length, TOPK)
            actual_topk = src[route, 0, :valid]
            expected_topk = reference[route, :valid]
            if state == -3:
                if not torch.equal(actual_topk, expected_topk):
                    mismatch_positions = torch.nonzero(actual_topk != expected_topk, as_tuple=False).flatten()
                    first_mismatch = int(mismatch_positions[0])
                    window_start = max(0, first_mismatch - 8)
                    window_end = min(valid, first_mismatch + 9)
                    same_members = torch.equal(
                        torch.sort(actual_topk).values,
                        torch.sort(expected_topk).values,
                    )
                    raise AssertionError(
                        "ordered TopK mismatch: "
                        f"request={request}, q={q}, route={route}, "
                        f"route_in_request={route - query_start}, valid={valid}, "
                        f"first_position={first_mismatch}, "
                        f"same_members_after_sort={same_members}, "
                        f"actual_window[{window_start}:{window_end}]="
                        f"{actual_topk[window_start:window_end].tolist()}, "
                        f"expected_window[{window_start}:{window_end}]="
                        f"{expected_topk[window_start:window_end].tolist()}"
                    )
            else:
                assert torch.equal(
                    torch.sort(actual_topk).values,
                    torch.sort(expected_topk).values,
                ), f"TopK mismatch at route={route}"
            if valid < TOPK:
                assert torch.all(src[route, 0, valid:] == -1)
            if state == -3:
                assert route_miss[route].item() == 0, (
                    f"non-offload route miss count mismatch: request={request}, "
                    f"route={route}, value={route_miss[route].item()}, "
                    f"all={route_miss.tolist()}, states={case['states']}"
                )
                assert torch.equal(dst[route], src[route])
            else:
                for position in range(TOPK):
                    source = int(src[route, 0, position])
                    if source >= 0:
                        assert int(dst[route, 0, position]) == int(cache_cpu[row, source])
        if state == -3:
            assert miss_count[request].item() == 0, (
                f"non-offload request miss count mismatch: request={request}, "
                f"value={miss_count[request].item()}, all={miss_count.tolist()}, "
                f"q={case['q_values']}, states={case['states']}"
            )
            assert torch.equal(cache_cpu[row], torch.arange(cache_cpu.size(1), dtype=torch.int32))
        elif state == -2:
            count = case["cache_tokens"][request]
            assert miss_count[request].item() == count
            assert torch.equal(miss_dst[request, :count], torch.arange(count, dtype=torch.int32))
            union = torch.unique(src[query_start:query_end].reshape(-1), sorted=True)
            union = union[union >= 0]
            selected = torch.zeros(length, dtype=torch.bool)
            selected[union.to(torch.int64)] = True
            remainder = torch.arange(length, dtype=torch.int32)[~selected]
            expected = torch.cat((union, remainder))[:count]
            assert torch.equal(miss_src[request, :count], expected)
            assert torch.all(route_miss[query_start:query_end] == TOPK)
        else:
            union = torch.unique(src[query_start:query_end].reshape(-1), sorted=True)
            expected = union[old_cache.cpu()[row, union.to(torch.int64)] < 0]
            count = int(miss_count[request])
            assert count == expected.numel()
            assert torch.equal(miss_src[request, :count], expected)
        if state in (-2, -1):
            resident_slots = cache_cpu[row, :length]
            resident_slots = resident_slots[resident_slots >= 0]
            capacity = case["cache_tokens"][request]
            if resident_slots.numel() != capacity:
                valid_slots = resident_slots[(resident_slots >= 0) & (resident_slots < capacity)]
                slot_counts = torch.bincount(valid_slots.to(torch.int64), minlength=capacity)
                missing = torch.nonzero(slot_counts == 0, as_tuple=False).flatten()
                duplicate = torch.nonzero(slot_counts > 1, as_tuple=False).flatten()
                raise AssertionError(
                    "cache resident count mismatch: "
                    f"request={request}, q={q}, state={state}, L={length}, "
                    f"C={capacity}, resident={resident_slots.numel()}, "
                    f"miss_count={int(miss_count[request])}, "
                    f"missing_slots={missing[:16].tolist()}, "
                    f"duplicate_slots={duplicate[:16].tolist()}"
                )
            assert torch.equal(
                torch.sort(resident_slots).values,
                torch.arange(capacity, dtype=torch.int32),
            ), f"cache slot mapping is not a bijection for request={request}"
        query_start = query_end


@pytest.mark.parametrize("heads,dtype", [(32, "bf16"), (64, "fp16")])
def test_generalized_lim_mixed_states_and_query_counts(heads, dtype):
    args = argparse.Namespace(
        source_capacity=16384, offload_len=8192, cache_tokens=8192, seed=7, dtype=dtype, heads=heads, device="npu:0"
    )
    # One launch mixes ordinary LI, first fill, steady offload and Q=1..7.
    case = build_case(args, q_values=[1, 2, 3, 4, 5, 6, 7], states=[-3, -2, -1, -3, -2, -1, -2])
    assert_correctness(case)


def test_generalized_lim_first_fill_then_repeat_is_all_hits():
    args = argparse.Namespace(
        source_capacity=16384, offload_len=8192, cache_tokens=6144, seed=17, dtype="bf16", heads=32, device="npu:0"
    )
    case = build_case(args, q_values=[1, 3], states=[-2, -2])
    cache = case["cache_seed"].clone()
    output = make_outputs(case)
    call_custom(case, cache, output)
    torch.npu.synchronize()
    for row, capacity in enumerate(case["cache_tokens"]):
        assert output[-1][row].item() == capacity
    first_slots = output[1].clone()
    case["metadata"][4].fill_(-1)
    call_custom(case, cache, output)
    torch.npu.synchronize()
    assert torch.count_nonzero(output[2]).item() == 0
    assert torch.count_nonzero(output[-1]).item() == 0
    # Compare sets because first-fill and steady output ordering may differ.
    torch.testing.assert_close(first_slots.sort(-1).values, output[1].sort(-1).values, rtol=0, atol=0)


@pytest.mark.parametrize("state", [-3, -2, -1])
def test_generalized_lim_source_ids_cross_128k(state):
    # The updated slot/source codec uses 17 source bits plus high-bit tags.
    # Force TopK sources above 2^17 in addition to checking the native LI set.
    args = argparse.Namespace(
        source_capacity=262144, offload_len=147456, cache_tokens=8192, seed=43, dtype="bf16", heads=32, device="npu:0"
    )
    case = build_case(args, q_values=[1, 4], states=[state, state])
    case["query"].abs_()
    case["weights"].abs_()
    case["key"][131072 // BLOCK :].abs_().add_(1)
    assert_correctness(case)
    outputs = make_outputs(case)
    call_custom(case, case["cache_seed"].clone(), outputs)
    torch.npu.synchronize()
    assert (outputs[0] >= 131072).any().item()


def test_generalized_lim_extended_routes_and_cache_budget():
    # Exercise the upstream workspace/slot-codec extension independently
    # from serving, whose currently supported MTP range stays unchanged.
    args = argparse.Namespace(
        source_capacity=65536, offload_len=49152, cache_tokens=32640, seed=47, dtype="bf16", heads=32, device="npu:0"
    )
    case = build_case(args, q_values=[8, 14], states=[-2, -1])
    assert_correctness(case)


def test_generalized_lim_copy_sfa_first_fill_and_replacement_chain():
    """Share the pinned miss buffers and verify attention plus persistent KV contents."""
    args = argparse.Namespace(
        source_capacity=16384, offload_len=8192, cache_tokens=6144, seed=27, dtype="bf16", heads=32, device="npu:0"
    )
    query_counts = [1, 3]
    case = build_case(args, q_values=query_counts, states=[-2, -2])
    batch, tokens, attention_heads = 2, sum(query_counts), 8
    capacity, prefix, tail = args.cache_tokens, args.offload_len, BLOCK
    source_blocks = args.source_capacity // BLOCK
    hbm_blocks_per_request = (capacity + tail) // BLOCK
    dtype, device = torch.bfloat16, torch.device(args.device)
    torch.manual_seed(31)
    dram_ckv = torch.randn(batch, args.source_capacity, 512, dtype=dtype)
    dram_kpe = torch.randn(batch, args.source_capacity, 64, dtype=dtype)
    initial_ckv = torch.zeros(batch, capacity + tail, 512, dtype=dtype)
    initial_kpe = torch.zeros(batch, capacity + tail, 64, dtype=dtype)
    initial_ckv[:, capacity:] = dram_ckv[:, prefix : prefix + tail]
    initial_kpe[:, capacity:] = dram_kpe[:, prefix : prefix + tail]
    hbm_ckv = initial_ckv.reshape(-1, BLOCK, 1, 512).to(device)
    hbm_kpe = initial_kpe.reshape(-1, BLOCK, 1, 64).to(device)
    dram_ckv_device = dram_ckv.reshape(-1, BLOCK, 512).to(device)
    dram_kpe_device = dram_kpe.reshape(-1, BLOCK, 64).to(device)
    dram_table = torch.arange(batch * source_blocks, dtype=torch.int32, device=device).reshape(batch, source_blocks)
    hbm_table = torch.arange(batch * hbm_blocks_per_request, dtype=torch.int32, device=device).reshape(
        batch, hbm_blocks_per_request
    )
    q = torch.randn(tokens, attention_heads, 512, dtype=dtype, device=device)
    q_rope = torch.randn(tokens, attention_heads, 64, dtype=dtype, device=device)
    actual_kv = torch.full((batch,), capacity + tail, dtype=torch.int32, device=device)
    attention_out = torch.empty_like(q)
    cache = case["cache_seed"].clone()
    outputs = make_outputs(case)
    # LIM writes directly to copy-SFA's contiguous [B,32768] miss buffers.
    scale = 576**-0.5

    for state in (-2, -1):
        case["metadata"][4].fill_(state)
        if state == -1:
            # Change selection to exercise misses and eviction after initialization.
            case["query"].neg_()
        call_custom(case, cache, outputs)
        src, dst, route_misses, miss_src, miss_dst, misses = outputs
        torch.ops._C_ascend.npu_fused_copy_sfa_mtp(
            q_rope,
            q,
            case["metadata"][0],
            actual_kv,
            case["metadata"][3],
            dst,
            src,
            route_misses,
            miss_src,
            miss_dst,
            misses,
            hbm_table,
            dram_table,
            hbm_kpe,
            hbm_ckv,
            dram_kpe_device,
            dram_ckv_device,
            scale,
            attention_out,
        )
        torch.npu.synchronize()
        sources = src.cpu().reshape(tokens, TOPK).long()
        query_cpu, rope_cpu = q.cpu().float(), q_rope.cpu().float()
        expected = []
        row = 0
        for request, count in enumerate(query_counts):
            for route in range(count):
                visible_tail = tail - (count - 1 - route)
                selected = torch.cat((sources[row], torch.arange(prefix, prefix + visible_tail)))
                ckv = dram_ckv[request, selected].float()
                kpe = dram_kpe[request, selected].float()
                scores = (query_cpu[row] @ ckv.T + rope_cpu[row] @ kpe.T) * scale
                expected.append(scores.softmax(-1) @ ckv)
                row += 1
        torch.testing.assert_close(attention_out.cpu().float(), torch.stack(expected), rtol=0.08, atol=0.08)
        resident_ckv = hbm_ckv.cpu().reshape(batch, capacity + tail, 512)
        resident_kpe = hbm_kpe.cpu().reshape(batch, capacity + tail, 64)
        mapping = cache.cpu()
        for request, pool in enumerate(case["req_entries"]):
            resident_sources = torch.nonzero(mapping[pool, :prefix] >= 0).flatten()
            resident_slots = mapping[pool, resident_sources].long()
            assert resident_sources.numel() == capacity
            torch.testing.assert_close(
                resident_ckv[request, resident_slots], dram_ckv[request, resident_sources], rtol=0, atol=0
            )
            torch.testing.assert_close(
                resident_kpe[request, resident_slots], dram_kpe[request, resident_sources], rtol=0, atol=0
            )
        torch.testing.assert_close(resident_ckv[:, capacity:], initial_ckv[:, capacity:], rtol=0, atol=0)
        torch.testing.assert_close(resident_kpe[:, capacity:], initial_kpe[:, capacity:], rtol=0, atol=0)


@pytest.mark.parametrize("queries", [1, 4, 7])
@pytest.mark.parametrize("use_graph", [False, True])
def test_generalized_lim_padding_overwrites_old_selection_without_misses(queries, use_graph):
    """A former active row becomes a short causal dummy, with a private map."""
    args = argparse.Namespace(
        source_capacity=16384,
        offload_len=8192,
        cache_tokens=8192,
        seed=41,
        dtype="bf16",
        heads=32,
        device="npu:0",
    )
    case = build_case(args, q_values=[queries], states=[-2])
    cache = case["cache_seed"].clone()
    outputs = make_outputs(case)
    call_custom(case, cache, outputs)
    graph = None
    if use_graph:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            call_custom(case, cache, outputs)
        graph.replay()
    active_map = cache[1].clone()
    query_ends, actual_key, offload_key, cache_tokens, states, pool = case["metadata"]
    actual_key.fill_(queries)
    offload_key.zero_()
    cache_tokens.fill_(2048)
    states.fill_(-3)
    pool.zero_()  # private dummy pool, distinct from the active request's row 1
    if graph is None:
        call_custom(case, cache, outputs)
    else:
        graph.replay()
    src, dst, route_misses, _, _, misses = outputs
    assert misses.item() == 0
    assert route_misses.count_nonzero().item() == 0
    for route in range(queries):
        valid = src[route, 0][src[route, 0] >= 0].cpu()
        assert torch.equal(valid.sort().values, torch.arange(route + 1, dtype=torch.int32))
        assert torch.equal(dst[route], src[route])
    assert torch.equal(cache[1], active_map)
