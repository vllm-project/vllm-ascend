"""A5 FP8 LIM contracts, ported from nanovllm-DSA-offload's A5 validators."""

from __future__ import annotations

import gc
import importlib

import pytest
import torch
import torch_npu

from vllm_ascend.device.device_config import is_950
from vllm_ascend.utils import bootstrap_custom_op_env

BLOCK = 128


HEAD_DIM = 128


TOPK = 2048


MISS_CAPACITY = 32768


INVALID_POOL_SLOT = -(1 << 31)


PACKED_SOURCE_CAPACITY = 1 << 17


MAX_SOURCE_CAPACITY = 1 << 21


def official_li(query, key, weights, query_scale, key_scale, block_table, length):
    op = getattr(torch_npu, "npu_quant_lightning_indexer", None)
    if op is None:
        raise RuntimeError("official A5 npu_quant_lightning_indexer is unavailable")
    out = op(
        query=query,
        key=key,
        weights=weights,
        query_dequant_scale=query_scale,
        key_dequant_scale=key_scale,
        actual_seq_lengths_query=torch.ones((1,), dtype=torch.int32, device=query.device),
        actual_seq_lengths_key=torch.tensor([length], dtype=torch.int32, device=query.device),
        block_table=block_table,
        query_quant_mode=0,
        key_quant_mode=0,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=TOPK,
        sparse_mode=3,
    )
    if isinstance(out, torch.Tensor):
        return out.reshape(1, 1, TOPK)
    return next(x for x in out if isinstance(x, torch.Tensor) and x.dtype == torch.int32 and x.numel() == TOPK).reshape(
        1, 1, TOPK
    )


def _official_batch_rows(case):
    """Use the same batch/Q dispatch as the fused call for its TopK oracle."""
    query, key, weights, query_scale, key_scale, block_table = case[:6]
    op = getattr(torch_npu, "npu_quant_lightning_indexer", None)
    if op is None:
        raise RuntimeError("official A5 npu_quant_lightning_indexer is unavailable")
    out = op(
        query=query,
        key=key,
        weights=weights,
        query_dequant_scale=query_scale,
        key_dequant_scale=key_scale,
        actual_seq_lengths_query=case[6],
        actual_seq_lengths_key=case[7],
        block_table=block_table,
        query_quant_mode=0,
        key_quant_mode=0,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=TOPK,
        sparse_mode=3,
    )
    route_count = sum(case[14])
    if not isinstance(out, torch.Tensor):
        out = next(
            (
                tensor
                for tensor in out
                if isinstance(tensor, torch.Tensor)
                and tensor.dtype == torch.int32
                and tensor.numel() == route_count * TOPK
            ),
            None,
        )
    if out is None:
        raise RuntimeError("official LI returned no batched int32 TopK")
    return out.reshape(route_count, TOPK).cpu()


def _needs_batched_oracle(case):
    # A5 changes tile/scheduling when T > 64 or many requests share cores.
    # Keep the established single-route oracle for smaller and long cases.
    return sum(case[14]) > 64 or len(case[14]) > 12


def make_case(
    device,
    heads,
    source_capacity,
    seed,
    *,
    key_length=None,
    query_counts=None,
):
    torch.manual_seed(seed)
    query_counts = [1, 4] if query_counts is None else list(query_counts)
    query_ends_cpu = []
    query_total = 0
    for count in query_counts:
        query_total += count
        query_ends_cpu.append(query_total)
    batch = len(query_counts)
    total_queries = sum(query_counts)
    blocks_per_request = source_capacity // BLOCK
    physical_blocks = batch * blocks_per_request
    query_bf16 = torch.randn(total_queries, heads, HEAD_DIM, dtype=torch.bfloat16).to(device)
    key_bf16 = torch.randn(physical_blocks, BLOCK, 1, HEAD_DIM, dtype=torch.bfloat16).to(device)
    weights_storage = torch.randn(total_queries, heads + 128, dtype=torch.bfloat16).to(device)
    weights = weights_storage[:, 128:]
    query, query_scale = torch_npu.npu_dynamic_quant(query_bf16, dst_type=torch.float8_e4m3fn)
    key_contiguous, key_scale_contiguous = torch_npu.npu_dynamic_quant(key_bf16, dst_type=torch.float8_e4m3fn)
    query = query.reshape(total_queries, heads, HEAD_DIM).contiguous()
    query_scale = query_scale.reshape(total_queries, heads).float().contiguous()
    key_block_stride = BLOCK * HEAD_DIM + 256
    key_storage = torch.empty(
        physical_blocks * key_block_stride,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    key = torch.as_strided(
        key_storage,
        size=(physical_blocks, BLOCK, 1, HEAD_DIM),
        stride=(key_block_stride, HEAD_DIM, HEAD_DIM, 1),
    )
    key.copy_(key_contiguous)
    scale_block_stride = BLOCK + 8
    scale_storage = torch.empty(physical_blocks * scale_block_stride, dtype=torch.float32, device=device)
    key_scale = torch.as_strided(
        scale_storage,
        size=(physical_blocks, BLOCK, 1),
        stride=(scale_block_stride, 1, 1),
    )
    key_scale.copy_(key_scale_contiguous.reshape(physical_blocks, BLOCK, 1).float())
    block_table = torch.arange(physical_blocks, dtype=torch.int32, device=device).reshape(batch, blocks_per_request)
    query_ends = torch.tensor(query_ends_cpu, dtype=torch.int32, device=device)
    key_length = source_capacity if key_length is None else key_length
    actual_key = torch.full((batch,), key_length, dtype=torch.int32, device=device)
    offload_key = actual_key.clone()
    cache_tokens = torch.full((batch,), 8192, dtype=torch.int32, device=device)
    states = torch.full((batch,), -3, dtype=torch.int32, device=device)
    default_entries = [2, 0, 1] + list(range(3, batch))
    entries = torch.tensor(default_entries[:batch], dtype=torch.int32, device=device)
    pool = torch.full(
        (max(4, batch), source_capacity),
        INVALID_POOL_SLOT,
        dtype=torch.int32,
        device=device,
    )
    outputs = (
        torch.full((total_queries, 1, TOPK), -7, dtype=torch.int32, device=device),
        torch.full((total_queries, 1, TOPK), -7, dtype=torch.int32, device=device),
        torch.full((total_queries,), -7, dtype=torch.int32, device=device),
        torch.full((batch, MISS_CAPACITY), -7, dtype=torch.int32, device=device),
        torch.full((batch, MISS_CAPACITY), -7, dtype=torch.int32, device=device),
        torch.full((batch,), -7, dtype=torch.int32, device=device),
    )
    return (
        query,
        key,
        weights,
        query_scale,
        key_scale,
        block_table,
        query_ends,
        actual_key,
        offload_key,
        cache_tokens,
        states,
        entries,
        pool,
        outputs,
        query_counts,
    )


def launch(case):
    (
        query,
        key,
        weights,
        query_scale,
        key_scale,
        block_table,
        query_ends,
        actual_key,
        offload_key,
        cache_tokens,
        states,
        entries,
        pool,
        outputs,
        _,
    ) = case
    result = torch.ops._C_ascend.npu_fused_quant_lightning_indexer_manage(
        weights,
        query_scale,
        query,
        key_scale,
        key,
        block_table,
        query_ends,
        actual_key,
        offload_key,
        cache_tokens,
        states,
        entries,
        pool,
        *outputs,
    )
    if result is not None:
        raise AssertionError("caller-owned operator must return None")


def _assert_offload_source_order(case):
    states = case[10].cpu().tolist()
    query_counts = case[14]
    src, _, route_miss, miss_src, miss_dst, request_miss = case[13]
    start = 0
    for request, routes in enumerate(query_counts):
        end = start + routes
        state = states[request]
        if state not in (-2, -1):
            start = end
            continue
        selected_union = set()
        for route in range(start, end):
            count = int(route_miss[route])
            values = src[route, 0].cpu().tolist()
            misses, hits = values[:count], values[count:]
            if misses != sorted(misses) or hits != sorted(hits):
                raise AssertionError(f"route {route} miss/hit source IDs are not ordered")
            selected_union.update(misses)
        copy_count = int(request_miss[request])
        copied = miss_src[request, :copy_count].cpu().tolist()
        if state == -1:
            if copied != sorted(copied):
                raise AssertionError(f"request {request} COPY union is not source ordered")
        else:
            union_count = len(selected_union)
            if copied[:union_count] != sorted(selected_union) or copied[union_count:] != sorted(copied[union_count:]):
                raise AssertionError(f"request {request} first-fill union/filler order is wrong")
            if miss_dst[request, :copy_count].cpu().tolist() != list(range(copy_count)):
                raise AssertionError(f"request {request} first-fill slot order is wrong")
        start = end


def _steady_pool_changes(old_slots, prefix, budget, selected, copied, slots):
    """Reconstruct only legal A5 prefix replacements from pre-call slots.

    Tail winners are direct hits at C + token % 256; they neither own a
    resident slot nor protect a prefix victim through that tail address.
    """
    selected_prefix = {int(token) for token in selected if 0 <= token < prefix}
    expected_misses = sorted(token for token in selected_prefix if old_slots[token] == INVALID_POOL_SLOT)
    if copied != expected_misses:
        raise AssertionError("steady COPY is not the exact pre-call prefix miss union")
    if len(slots) != len(copied) or len(set(slots)) != len(slots) or any(slot < 0 or slot >= budget for slot in slots):
        raise AssertionError("steady COPY destinations are not unique resident slots")
    owners = {}
    for source, value in enumerate(old_slots[:prefix]):
        if 0 <= value < budget:
            if value in owners:
                raise AssertionError("steady pre-call pool has a duplicate slot")
            owners[value] = source
    if len(owners) != budget:
        raise AssertionError("steady pre-call pool is not a C-slot bijection")
    changes = {}
    for incoming, slot in zip(copied, slots):
        victim = owners[slot]
        if victim in selected_prefix:
            raise AssertionError("steady COPY evicts a selected prefix hit")
        changes[victim] = INVALID_POOL_SLOT
        changes[incoming] = slot
    return changes


def _first_fill_sources(selected, prefix, budget):
    """A5 installs sorted prefix winners, then the lowest nonwinner fillers."""
    union = {int(token) for token in selected if 0 <= token < prefix}
    if len(union) > budget or budget > prefix:
        raise AssertionError("first-fill fixture cannot fit the selected prefix union")
    copied = sorted(union)
    for token in range(prefix):
        if len(copied) == budget:
            break
        if token not in union:
            copied.append(token)
    return copied


def _assert_first_fill_pool(actual_row, prefix, budget, selected, copied, slots):
    expected_sources = _first_fill_sources(selected, prefix, budget)
    if copied != expected_sources or slots != list(range(budget)):
        raise AssertionError("first-fill COPY does not match prefix union/filler/slot order")
    actual = actual_row.cpu()
    expected = torch.full_like(actual, INVALID_POOL_SLOT)
    expected[torch.tensor(expected_sources, dtype=torch.long)] = torch.arange(
        budget,
        dtype=actual.dtype,
    )
    if not torch.equal(actual, expected):
        raise AssertionError("first-fill complete pool differs from expected COPY mapping")


def _assert_steady_pool_delta(old_row, actual_row, prefix, budget, selected, copied, slots):
    """Compare the complete row, including untouched residents and tail."""
    old_row = old_row.cpu()
    _assert_sparse_pool(old_row, prefix, budget)
    changes = _steady_pool_changes(old_row.tolist(), prefix, budget, selected, copied, slots)
    expected = old_row.clone()
    if changes:
        indices = torch.tensor(list(changes), dtype=torch.long)
        expected[indices] = torch.tensor(list(changes.values()), dtype=expected.dtype)
    actual = actual_row.cpu()
    if not torch.equal(actual, expected):
        wrong = torch.nonzero(actual != expected).flatten()
        first = int(wrong[0])
        raise AssertionError(
            f"steady full pool delta mismatch at source {first}: "
            f"actual={int(actual[first])}, expected={int(expected[first])}"
        )


def validate_non_offload(case, *, ordered=False):
    actual_key, entries, pool, outputs, query_counts = (case[7], case[11], case[12], case[13], case[14])
    pointers = [pool.data_ptr(), *(x.data_ptr() for x in outputs)]
    launch(case)
    torch.npu.synchronize()
    if pointers != [pool.data_ptr(), *(x.data_ptr() for x in outputs)]:
        raise AssertionError("caller-owned addresses changed")
    topk_src, topk_dst, route_miss, _, _, request_miss = outputs
    if not torch.equal(topk_src, topk_dst):
        raise AssertionError("-3 requires topk_dst_slots == topk_src_ids")
    if bool((route_miss != 0).any()) or bool((request_miss != 0).any()):
        raise AssertionError(
            "-3 must publish zero miss counts; "
            f"route={route_miss.cpu().tolist()}, "
            f"request={request_miss.cpu().tolist()}"
        )
    # Exact order checks need the same B/Q input shape as the fused call;
    # slicing a Q4/Q14 request into Q1 calls can change C8 gather order.
    golden_rows = _official_batch_rows(case) if ordered or _needs_batched_oracle(case) else None
    start = 0
    for request, routes in enumerate(query_counts):
        end = start + routes
        for row in range(start, end):
            visible = int(actual_key[request]) - (end - 1 - row)
            golden = golden_rows[row] if golden_rows is not None else _route_golden(case, request, row, end)
            actual = topk_src[row, 0].cpu()
            if visible <= TOPK or ordered:
                if not torch.equal(actual, golden):
                    raise AssertionError(f"ordered route {row} differs from official A5 C8 LI")
                if visible < TOPK:
                    values = actual.tolist()
                    valid = [token for token in values if token >= 0]
                    if len(valid) != visible or set(valid) != set(range(visible)) or values.count(-1) != TOPK - visible:
                        raise AssertionError(f"short route {row} has incorrect source IDs or -1 padding")
            else:
                actual_ids = set(actual.tolist())
                golden_ids = set(golden.tolist())
                if actual_ids != golden_ids:
                    raise AssertionError(
                        f"route {row} differs from official A5 C8 LI: "
                        f"unique={len(actual_ids)}, "
                        f"missing={sorted(golden_ids - actual_ids)[:16]}, "
                        f"extra={sorted(actual_ids - golden_ids)[:16]}"
                    )
        start = end
    identity = torch.arange(pool.shape[1], dtype=torch.int32)
    for row in entries.cpu().tolist():
        if not torch.equal(pool[row].cpu(), identity):
            raise AssertionError(f"pool row {row} is not identity")
    for row in set(range(pool.shape[0])) - set(entries.cpu().tolist()):
        if bool((pool[row] != INVALID_POOL_SLOT).any()):
            raise AssertionError(f"unused pool row {row} changed")


def validate_tight_causal_boundary(device, heads, seed, route_counts=(1, 2, 3, 4)):
    """L=2048 is legal when the earliest Q route sees exactly L tokens."""
    prefix = budget = TOPK
    for routes in route_counts:
        final_length = TOPK + routes - 1
        case = make_case(
            device,
            heads,
            2304,
            seed + routes,
            key_length=final_length,
            query_counts=[routes],
        )
        case[8].fill_(prefix)
        case[9].fill_(budget)
        case[10].fill_(-2)
        case[12].fill_(INVALID_POOL_SLOT)
        launch(case)
        torch.npu.synchronize()
        _assert_offload_source_order(case)
        src, dst, route_miss, miss_src, miss_dst, request_miss = case[13]
        if int(request_miss[0]) != budget:
            raise AssertionError(f"tight Q{routes} first fill did not copy C sources")
        expected_prefix = torch.arange(prefix, dtype=torch.int32)
        row = case[12][int(case[11][0])].cpu()
        if (
            not torch.equal(row[:prefix], expected_prefix)
            or bool((row[prefix:] != INVALID_POOL_SLOT).any())
            or not torch.equal(miss_src[0, :budget].cpu(), expected_prefix)
            or not torch.equal(miss_dst[0, :budget].cpu(), expected_prefix)
        ):
            raise AssertionError(f"tight Q{routes} first fill pool or COPY list is wrong")
        for route in range(routes):
            golden = _route_golden(case, 0, route, routes)
            sources = src[route, 0].cpu().tolist()
            slots = dst[route, 0].cpu().tolist()
            if set(sources) != set(golden.tolist()):
                raise AssertionError(f"tight Q{routes} route {route} differs from official LI")
            expected_misses = sum(token < prefix for token in sources)
            if int(route_miss[route]) != expected_misses:
                raise AssertionError(f"tight Q{routes} route {route} miss count is wrong")
            for token, slot in zip(sources, slots):
                expected_slot = token if token < prefix else budget + token % 256
                if slot != expected_slot:
                    raise AssertionError(f"tight Q{routes} route {route} slot is wrong")

        case[10].fill_(-1)
        for output in case[13]:
            output.fill_(-7)
        launch(case)
        torch.npu.synchronize()
        _assert_offload_source_order(case)
        src, dst, route_miss, _, _, request_miss = case[13]
        if int(request_miss[0]) != 0 or bool((route_miss != 0).any()):
            raise AssertionError(f"tight Q{routes} steady call produced a miss")
        if not torch.equal(case[12][int(case[11][0])].cpu(), row):
            raise AssertionError(f"tight Q{routes} steady call changed the pool")
        for route in range(routes):
            golden = _route_golden(case, 0, route, routes)
            sources = src[route, 0].cpu().tolist()
            slots = dst[route, 0].cpu().tolist()
            if set(sources) != set(golden.tolist()):
                raise AssertionError(f"tight Q{routes} steady route differs from official LI")
            for token, slot in zip(sources, slots):
                expected_slot = token if token < prefix else budget + token % 256
                if slot != expected_slot:
                    raise AssertionError(f"tight Q{routes} steady slot is wrong")


def validate_steady_union_against_oracle(case, budget):
    """Check the selected miss union, pool update, and slots independently."""
    validate_first_fill(case, budget)
    states, entries, pool, outputs, routes = case[10:]
    states.fill_(-1)
    target_row = int(entries[0])
    route_count = routes[0]
    prefix = int(case[8][0])
    selected_counts: dict[int, int] = {}
    for route in range(route_count):
        for token in outputs[0][route, 0].cpu().tolist():
            if 0 <= int(token) < prefix:
                selected_counts[int(token)] = selected_counts.get(int(token), 0) + 1
    selected = set(selected_counts)
    repeated = sorted(token for token, count in selected_counts.items() if count > 1)
    if route_count > 1 and len(repeated) < 4:
        raise AssertionError("steady union fixture has too few cross-route duplicates")
    missed = set(repeated[:4])
    for token in sorted(selected):
        if len(missed) == 8:
            break
        missed.add(token)
    original_pool = pool.cpu().clone()
    replacements = [
        token
        for token in range(prefix)
        if int(original_pool[target_row, token]) == INVALID_POOL_SLOT and token not in selected
    ]
    if len(missed) != 8 or len(replacements) < 8:
        raise AssertionError("cannot construct steady miss fixture")
    for source, replacement in zip(sorted(missed), replacements):
        slot = int(original_pool[target_row, source])
        original_pool[target_row, source] = INVALID_POOL_SLOT
        original_pool[target_row, replacement] = slot

    pool.copy_(original_pool.to(pool.device))
    for output in outputs:
        output.fill_(-7)
    launch(case)
    torch.npu.synchronize()
    _assert_offload_source_order(case)
    sources, slots, route_misses, miss_sources, miss_slots, request_misses = outputs
    count = int(request_misses[0])
    if count != len(missed):
        raise AssertionError("steady request miss count differs from pre-call pool")
    copied = miss_sources[0, :count].cpu().tolist()
    copied_slots = miss_slots[0, :count].cpu().tolist()
    if copied != sorted(missed) or len(set(copied_slots)) != count:
        raise AssertionError("steady COPY union or destinations are incorrect")
    row_pool = pool[target_row].cpu()
    _assert_sparse_pool(row_pool, prefix, budget)
    _assert_steady_pool_delta(original_pool[target_row], row_pool, prefix, budget, selected, copied, copied_slots)
    for token, slot in zip(copied, copied_slots):
        if int(original_pool[target_row, token]) != INVALID_POOL_SLOT or int(row_pool[token]) != slot:
            raise AssertionError("steady COPY pair disagrees with pre-call/final pool")
    for route in range(route_count):
        golden = _route_golden(case, 0, route, route_count)
        actual_sources = sources[route, 0].cpu().tolist()
        actual_slots = slots[route, 0].cpu().tolist()
        if set(actual_sources) != set(golden.tolist()):
            raise AssertionError("steady TopK differs from official A5 C8 LI")
        miss_count = int(route_misses[route])
        if miss_count != sum(int(token) in missed for token in golden.tolist()):
            raise AssertionError("steady route miss count differs from pre-call pool")
        for index, (token, slot) in enumerate(zip(actual_sources, actual_slots)):
            if (index < miss_count) != (token in missed):
                raise AssertionError("steady miss/hit partition is incorrect")
            expected = int(row_pool[token]) if token < prefix else budget + token % 256
            if slot != expected:
                raise AssertionError("steady destination disagrees with final pool")
    # The peer request has no newly missing sources and must not be affected.
    peer_row = int(entries[1])
    if int(request_misses[1]) != 0 or not torch.equal(pool[peer_row].cpu(), original_pool[peer_row]):
        raise AssertionError("steady miss update affected the independent peer")
    peer_start = route_count
    peer_end = peer_start + routes[1]
    for route in range(peer_start, peer_end):
        golden = _route_golden(case, 1, route, peer_end)
        if int(route_misses[route]) != 0 or set(sources[route, 0].cpu().tolist()) != set(golden.tolist()):
            raise AssertionError("steady peer differs from official LI or old pool")


def validate_first_fill_ignores_old_pool(case, budget):
    """First fill follows official LI and does not depend on old pool values."""
    validate_first_fill(case, budget)
    states, entries, pool, outputs, routes = case[10:]
    baseline_pool = pool.cpu().clone()
    baseline_outputs = tuple(output.cpu().clone() for output in outputs)
    pool.fill_(37)
    for output in outputs:
        output.fill_(-7)
    launch(case)
    torch.npu.synchronize()
    _assert_offload_source_order(case)
    for request, row in enumerate(entries.cpu().tolist()):
        if int(outputs[5][request]) != budget:
            raise AssertionError("first-fill COPY count changed with old pool values")
        if not torch.equal(pool[row].cpu(), baseline_pool[row]):
            raise AssertionError("first-fill pool depends on old pool values")
        for index in (3, 4):
            if not torch.equal(
                outputs[index][request, :budget].cpu(),
                baseline_outputs[index][request, :budget],
            ):
                raise AssertionError("first-fill COPY list depends on old pool values")
        _assert_sparse_pool(pool[row].cpu(), int(case[8][request]), budget)
    for index in (0, 1, 2, 5):
        if not torch.equal(outputs[index].cpu(), baseline_outputs[index]):
            raise AssertionError("first-fill TopK or count depends on old pool values")
    start = 0
    for request, route_count in enumerate(routes):
        end = start + route_count
        for route in range(start, end):
            golden = _route_golden(case, request, route, end)
            if set(outputs[0][route, 0].cpu().tolist()) != set(golden.tolist()):
                raise AssertionError("first-fill TopK differs from official A5 C8 LI")
        start = end


def _route_golden(case, request, row, end):
    (query, key, weights, query_scale, key_scale, block_table, _, actual_key, *_rest) = case
    visible = int(actual_key[request]) - (end - 1 - row)
    return official_li(
        query[row : row + 1],
        key,
        weights[row : row + 1],
        query_scale[row : row + 1],
        key_scale,
        block_table[request : request + 1],
        visible,
    )[0, 0].cpu()


def validate_first_fill(case, budget):
    (query, _, _, _, _, _, _, actual_key, offload_key, cache_tokens, states, entries, pool, outputs, query_counts) = (
        case
    )
    del query
    prefix = (pool.shape[1] - 256) // BLOCK * BLOCK
    if prefix < budget:
        raise ValueError("source capacity is too small for the requested first-fill budget")
    states.fill_(-2)
    offload_key.fill_(prefix)
    cache_tokens.fill_(budget)
    actual_key.fill_(prefix + 256)
    pool.fill_(INVALID_POOL_SLOT)
    entry_rows = entries.cpu().tolist()
    identity = torch.arange(pool.shape[1], dtype=torch.int32, device=pool.device)
    pool[entry_rows[0]] = identity
    pool[entry_rows[1]] = torch.arange(pool.shape[1], dtype=torch.int32, device=pool.device).remainder(max(1, budget))
    before_unused = pool.clone()
    for output in outputs:
        output.fill_(-7)

    launch(case)
    torch.npu.synchronize()
    _assert_offload_source_order(case)
    golden_rows = _official_batch_rows(case) if _needs_batched_oracle(case) else None
    topk_src, topk_dst, route_miss, miss_src, miss_dst, request_miss = outputs
    if bool((request_miss != budget).any()):
        raise AssertionError(f"-2 request miss counts must equal C={budget}: {request_miss.cpu().tolist()}")
    start = 0
    selected_prefix_union: list[set[int]] = [set() for _ in query_counts]
    for request, routes in enumerate(query_counts):
        end = start + routes
        row_pool = pool[entry_rows[request]].cpu()
        for row in range(start, end):
            golden = golden_rows[row] if golden_rows is not None else _route_golden(case, request, row, end)
            actual_sources = topk_src[row, 0].cpu()
            actual_slots = topk_dst[row, 0].cpu()
            if set(actual_sources.tolist()) != set(golden.tolist()):
                raise AssertionError(f"-2 route {row} differs from full-visible A5 LI")
            expected_misses = sum(int(token) < prefix for token in golden.tolist())
            actual_misses = int(route_miss[row])
            if actual_misses != expected_misses:
                raise AssertionError(f"route {row} miss count {actual_misses} != {expected_misses}")
            prefix_sources = actual_sources[:actual_misses]
            tail_sources = actual_sources[actual_misses:]
            if bool((prefix_sources >= prefix).any()) or bool((tail_sources < prefix).any()):
                raise AssertionError("-2 output is not prefix-miss followed by tail-hit")
            for token, slot in zip(actual_sources.tolist(), actual_slots.tolist()):
                if token < prefix:
                    selected_prefix_union[request].add(token)
                    if slot != int(row_pool[token]):
                        raise AssertionError("prefix TopK slot disagrees with cache_slots_pool")
                elif slot != budget + token % 256:
                    raise AssertionError("selected tail did not use C + token % 256")
        start = end

        valid_sources = miss_src[request, :budget].cpu().tolist()
        valid_slots = miss_dst[request, :budget].cpu().tolist()
        if len(set(valid_sources)) != budget or set(valid_slots) != set(range(budget)):
            raise AssertionError("first-fill miss list is not a C-token slot bijection")
        if valid_slots != list(range(budget)):
            raise AssertionError("first-fill COPY destinations are not 0..C-1 in order")
        if not selected_prefix_union[request].issubset(set(valid_sources)):
            raise AssertionError("selected prefix union was not prioritized into resident cache")
        for source, slot in zip(valid_sources, valid_slots):
            if not 0 <= source < prefix or int(row_pool[source]) != slot:
                raise AssertionError("miss pair disagrees with rebuilt pool")
        resident_mask = row_pool >= 0
        if int(resident_mask.sum()) != budget:
            raise AssertionError("rebuilt pool does not contain exactly C resident sources")
        if set(row_pool[resident_mask].tolist()) != set(range(budget)):
            raise AssertionError("rebuilt pool slots are not exactly 0..C-1")
        if bool((row_pool[prefix:] != INVALID_POOL_SLOT).any()):
            raise AssertionError("tail was written into cache_slots_pool")

    used = set(entry_rows)
    for row in range(pool.shape[0]):
        if row not in used and not torch.equal(pool[row], before_unused[row]):
            raise AssertionError(f"unused pool row {row} changed")
    if bool((states != -2).any()):
        raise AssertionError("operator modified caller-owned request_state")


def validate_steady_zero_miss(case, budget):
    validate_first_fill(case, budget)
    golden_rows = _official_batch_rows(case) if _needs_batched_oracle(case) else None
    *_, states, entries, pool, outputs, query_counts = case
    states.fill_(-1)
    before = pool.clone()
    for output in outputs:
        output.fill_(-7)
    launch(case)
    torch.npu.synchronize()
    _assert_offload_source_order(case)
    topk_src, topk_dst, route_miss, _, _, request_miss = outputs
    if bool((route_miss != 0).any()) or bool((request_miss != 0).any()):
        raise AssertionError(
            "unchanged first-fill pool must produce zero steady misses: "
            f"route={route_miss.cpu().tolist()}, "
            f"request={request_miss.cpu().tolist()}"
        )
    if not torch.equal(pool, before):
        raise AssertionError("zero-miss steady call modified cache_slots_pool")
    prefix = int(case[8][0].cpu())
    entry_rows = entries.cpu().tolist()
    start = 0
    for request, routes in enumerate(query_counts):
        end = start + routes
        row_pool = pool[entry_rows[request]].cpu()
        for row in range(start, end):
            golden = golden_rows[row] if golden_rows is not None else _route_golden(case, request, row, end)
            sources = topk_src[row, 0].cpu()
            slots = topk_dst[row, 0].cpu()
            if set(sources.tolist()) != set(golden.tolist()):
                raise AssertionError("zero-miss steady TopK differs from A5 LI")
            for token, slot in zip(sources.tolist(), slots.tolist()):
                expected = int(row_pool[token]) if token < prefix else budget + token % 256
                if slot != expected:
                    raise AssertionError("zero-miss steady slot mapping is incorrect")
        start = end
    if bool((states != -1).any()):
        raise AssertionError("steady operator modified request_state")

    first_outputs = tuple(tensor.clone() for tensor in (topk_src, topk_dst, route_miss, request_miss))
    repeated_pool = pool.clone()
    for output in outputs:
        output.fill_(-7)
    launch(case)
    torch.npu.synchronize()
    repeated_outputs = (outputs[0], outputs[1], outputs[2], outputs[5])
    if any(not torch.equal(actual, expected) for actual, expected in zip(repeated_outputs, first_outputs)):
        raise AssertionError("repeated zero-miss steady call changed its outputs")
    if not torch.equal(pool, repeated_pool):
        raise AssertionError("repeated zero-miss steady call changed the pool")


def validate_steady_miss_and_atomicity(case, budget, misses_per_request=8):
    validate_first_fill(case, budget)
    golden_rows = _official_batch_rows(case) if _needs_batched_oracle(case) else None
    *_, states, entries, pool, outputs, query_counts = case
    states.fill_(-1)
    prefix = int(case[8][0].cpu())
    entry_rows = entries.cpu().tolist()
    first_topk = outputs[0].cpu()
    pool_cpu = pool.cpu().clone()
    chosen_by_request = []
    protected_slots = []
    start = 0
    for request, routes in enumerate(query_counts):
        end = start + routes
        selected = {
            int(token) for row in range(start, end) for token in first_topk[row, 0].tolist() if 0 <= int(token) < prefix
        }
        chosen = set(sorted(selected)[:misses_per_request])
        replacements = [
            source
            for source in range(prefix)
            if int(pool_cpu[entry_rows[request], source]) == INVALID_POOL_SLOT and source not in selected
        ]
        if len(chosen) != misses_per_request or len(replacements) < len(chosen):
            raise AssertionError("test fixture cannot construct steady misses")
        row_pool = pool_cpu[entry_rows[request]]
        for source, replacement in zip(sorted(chosen), replacements):
            slot = int(row_pool[source])
            row_pool[source] = INVALID_POOL_SLOT
            row_pool[replacement] = slot
        chosen_by_request.append(chosen)
        protected_slots.append({source: int(row_pool[source]) for source in selected - chosen})
        start = end
    pool.copy_(pool_cpu.to(pool.device))
    for output in outputs:
        output.fill_(-7)
    launch(case)
    torch.npu.synchronize()
    _assert_offload_source_order(case)
    topk_src, topk_dst, route_miss, miss_src, miss_dst, request_miss = outputs
    start = 0
    for request, routes in enumerate(query_counts):
        end = start + routes
        chosen = chosen_by_request[request]
        if int(request_miss[request]) != len(chosen):
            raise AssertionError("steady request miss count is not the unique union size")
        actual_union = set(miss_src[request, : len(chosen)].cpu().tolist())
        if actual_union != chosen:
            raise AssertionError("steady miss list differs from expected prefix union")
        miss_slots = miss_dst[request, : len(chosen)].cpu().tolist()
        if len(set(miss_slots)) != len(chosen) or not all(0 <= x < budget for x in miss_slots):
            raise AssertionError("steady miss destinations are not unique cache slots")
        row_pool = pool[entry_rows[request]].cpu()
        selected = set().union(*(set(first_topk[row, 0].tolist()) for row in range(start, end)))
        _assert_steady_pool_delta(
            pool_cpu[entry_rows[request]],
            row_pool,
            prefix,
            budget,
            selected,
            miss_src[request, : len(chosen)].cpu().tolist(),
            miss_slots,
        )
        for source, old_slot in protected_slots[request].items():
            if int(row_pool[source]) != old_slot:
                raise AssertionError("steady evicted or remapped a protected TopK hit")
        for source in chosen:
            if int(row_pool[source]) < 0:
                raise AssertionError("steady miss was not installed into cache")
        resident = row_pool >= 0
        if int(resident.sum()) != budget or set(row_pool[resident].tolist()) != set(range(budget)):
            raise AssertionError("steady update broke the C-slot pool bijection")
        if bool((row_pool[prefix:] != INVALID_POOL_SLOT).any()):
            raise AssertionError("steady update wrote selected tail into pool")
        for row in range(start, end):
            golden = golden_rows[row] if golden_rows is not None else _route_golden(case, request, row, end)
            expected_route_misses = sum(int(token) in chosen for token in golden.tolist())
            actual_route_misses = int(route_miss[row])
            if actual_route_misses != expected_route_misses:
                raise AssertionError("steady per-route miss count is incorrect")
            sources = topk_src[row, 0].cpu()
            slots = topk_dst[row, 0].cpu()
            if set(sources.tolist()) != set(golden.tolist()):
                raise AssertionError("steady TopK differs from full-visible A5 LI")
            if any(int(token) not in chosen for token in sources[:actual_route_misses].tolist()):
                raise AssertionError("steady miss prefix contains a hit")
            if any(int(token) in chosen for token in sources[actual_route_misses:].tolist()):
                raise AssertionError("steady hit suffix contains a miss")
            for token, slot in zip(sources.tolist(), slots.tolist()):
                expected = int(row_pool[token]) if token < prefix else budget + token % 256
                if slot != expected:
                    raise AssertionError("steady TopK destination is incorrect")
        start = end

    # The steady union does not scan arbitrary unrelated pool entries for
    # duplicate slots. A row with no residents is rejected by victim
    # planning before any update and must not affect the other request.
    bad_pool = pool.cpu().clone()
    bad_pool_row = bad_pool[entry_rows[0]]
    bad_pool_row.fill_(INVALID_POOL_SLOT)
    pool.copy_(bad_pool.to(pool.device))
    before = pool.clone()
    for output in outputs:
        output.fill_(-7)
    launch(case)
    torch.npu.synchronize()
    if int(outputs[5][0]) != 0 or not bool((outputs[2][: query_counts[0]] == 0).all()):
        raise AssertionError("invalid steady pool was not rejected")
    if not torch.equal(pool, before):
        raise AssertionError("invalid steady pool modified caller-owned state")


def _assert_sparse_pool(row_pool, prefix, budget):
    resident = row_pool >= 0
    resident_count = int(resident.sum())
    if resident_count != budget:
        raise AssertionError(f"sparse pool residents={resident_count}, expected C={budget}")
    if set(row_pool[resident].tolist()) != set(range(budget)):
        raise AssertionError("sparse pool slots are not a 0..C-1 bijection")
    if bool((row_pool[:prefix][~resident[:prefix]] != INVALID_POOL_SLOT).any()):
        raise AssertionError("sparse pool nonresidents must be INT32_MIN")
    if bool((row_pool[prefix:] != INVALID_POOL_SLOT).any()):
        raise AssertionError("sparse pool contains a resident outside prefix")


def validate_identity_to_steady(case, budget, *, final_length=None):
    validate_non_offload(case)
    *_, actual_key, offload_key, cache_tokens, states, entries, pool, outputs, query_counts = case
    prefix = (pool.shape[1] - 256) // BLOCK * BLOCK
    if prefix < budget:
        raise ValueError("source capacity is too small for identity normalization")
    actual_key.fill_(prefix + 256 if final_length is None else final_length)
    offload_key.fill_(prefix)
    cache_tokens.fill_(budget)
    states.fill_(-1)
    for output in outputs:
        output.fill_(-7)
    launch(case)
    torch.npu.synchronize()
    _assert_offload_source_order(case)

    topk_src, topk_dst, route_miss, miss_src, miss_dst, request_miss = outputs
    entry_rows = entries.cpu().tolist()
    start = 0
    for request, routes in enumerate(query_counts):
        end = start + routes
        expected_union: set[int] = set()
        golden_rows = []
        for row in range(start, end):
            golden = _route_golden(case, request, row, end)
            golden_rows.append(golden)
            expected_union.update(int(token) for token in golden.tolist() if budget <= int(token) < prefix)
        if int(request_miss[request]) != len(expected_union):
            raise AssertionError("identity-to-steady request miss count is incorrect")
        actual_union = set(miss_src[request, : len(expected_union)].cpu().tolist())
        if actual_union != expected_union:
            raise AssertionError("identity-to-steady miss union is incorrect")
        row_pool = pool[entry_rows[request]].cpu()
        _assert_sparse_pool(row_pool, prefix, budget)
        normalized = torch.full_like(row_pool, INVALID_POOL_SLOT)
        normalized[:budget] = torch.arange(budget, dtype=torch.int32)
        _assert_steady_pool_delta(
            normalized,
            row_pool,
            prefix,
            budget,
            set().union(*(set(row.tolist()) for row in golden_rows)),
            miss_src[request, : len(expected_union)].cpu().tolist(),
            miss_dst[request, : len(expected_union)].cpu().tolist(),
        )
        for token in set().union(*(set(row.tolist()) for row in golden_rows)):
            if token < budget and int(row_pool[token]) != token:
                raise AssertionError("identity normalization remapped a protected hit")
        for offset, row in enumerate(range(start, end)):
            golden = golden_rows[offset]
            expected_route_misses = sum(budget <= int(token) < prefix for token in golden.tolist())
            if int(route_miss[row]) != expected_route_misses:
                raise AssertionError("identity-to-steady route miss count is incorrect")
            sources = topk_src[row, 0].cpu()
            slots = topk_dst[row, 0].cpu()
            if set(sources.tolist()) != set(golden.tolist()):
                raise AssertionError("identity-to-steady TopK differs from A5 LI")
            if any(not (budget <= int(token) < prefix) for token in sources[:expected_route_misses].tolist()):
                raise AssertionError("identity-to-steady miss prefix contains a hit")
            for token, slot in zip(sources.tolist(), slots.tolist()):
                expected = int(row_pool[token]) if token < prefix else budget + token % 256
                if slot != expected:
                    raise AssertionError("identity-to-steady slot mapping is incorrect")
        start = end
    if bool((states != -1).any()):
        raise AssertionError("identity normalization modified request_state")


def validate_dynamic_failure_matrix(device, heads, source_capacity, seed):
    budget = 8192
    prefix = (source_capacity - 384) // BLOCK * BLOCK
    if prefix < budget or prefix + 256 > source_capacity:
        raise ValueError("source capacity is too small for failure-matrix coverage")

    def invalid_state(case):
        case[10][0] = 0

    def negative_state(case):
        case[10][0] = -4

    def invalid_pool_entry(case):
        case[11][0] = case[12].shape[0]

    def negative_pool_entry(case):
        case[11][0] = -1

    def negative_final_length(case):
        case[7][0] = -1

    def short_prefix(case):
        case[8][0] = TOPK - BLOCK

    def unaligned_prefix(case):
        case[8][0] = prefix - 1

    def short_final_length(case):
        case[7][0] = 1

    def oversized_final_length(case):
        case[7][0] = source_capacity + 1

    def causal_prefix_violation(case):
        case[7][0] = prefix

    def oversized_tail(case):
        case[7][0] = prefix + 257

    def unsupported_budget(case):
        case[9][0] = 9000

    def budget_exceeds_prefix(case):
        case[8][0] = 8192
        case[7][0] = 8192 + 256
        case[9][0] = 10240

    mutations = (
        ("invalid state", invalid_state),
        ("unsupported negative state", negative_state),
        ("out-of-range pool entry", invalid_pool_entry),
        ("negative pool entry", negative_pool_entry),
        ("negative final length", negative_final_length),
        ("prefix shorter than TopK", short_prefix),
        ("unaligned prefix", unaligned_prefix),
        ("final length shorter than Q", short_final_length),
        ("final length beyond source capacity", oversized_final_length),
        ("prefix beyond earliest visible length", causal_prefix_violation),
        ("tail longer than 256", oversized_tail),
        ("unsupported cache budget", unsupported_budget),
        ("cache budget larger than prefix", budget_exceeds_prefix),
    )

    case = make_case(
        device,
        heads,
        source_capacity,
        seed,
        query_counts=[2, 1],
    )
    (*_, actual_key, offload_key, cache_tokens, states, entries, pool, outputs, query_counts) = case
    baseline_entries = entries.clone()
    pool_pattern = torch.arange(pool.numel(), dtype=torch.int32, device=pool.device).reshape_as(pool)
    for name, mutate in mutations:
        actual_key.fill_(prefix + 256)
        offload_key.fill_(prefix)
        cache_tokens.fill_(budget)
        states.fill_(-2)
        entries.copy_(baseline_entries)
        pool.copy_(pool_pattern)
        bad_row = int(entries[0])
        legal_row = int(entries[1])
        bad_before = pool[bad_row].clone()
        mutate(case)
        for output in outputs:
            output.fill_(-7)
        launch(case)
        torch.npu.synchronize()
        bad_end = query_counts[0]
        if int(outputs[5][0]) != 0 or not bool((outputs[2][:bad_end] == 0).all()):
            raise AssertionError(f"{name} did not publish request failure")
        if bool((outputs[0][:bad_end] != -1).any()) or bool((outputs[1][:bad_end] != -1).any()):
            raise AssertionError(f"{name} did not invalidate TopK outputs")
        if not torch.equal(pool[bad_row], bad_before):
            raise AssertionError(f"{name} modified the invalid request pool")
        if int(outputs[5][1]) != budget:
            raise AssertionError(f"{name} poisoned the legal peer request")
        _assert_sparse_pool(pool[legal_row].cpu(), prefix, budget)
        legal_end = sum(query_counts)
        golden = _route_golden(case, 1, bad_end, legal_end)
        if set(outputs[0][bad_end, 0].cpu().tolist()) != set(golden.tolist()):
            raise AssertionError(f"{name} changed the legal peer's A5 C8 LI TopK")

    # Exercise the same isolation guarantee for a malformed steady pool.
    actual_key.fill_(prefix + 256)
    offload_key.fill_(prefix)
    cache_tokens.fill_(budget)
    entries.copy_(baseline_entries)
    states.copy_(torch.tensor([-1, -2], dtype=torch.int32, device=states.device))
    pool.fill_(INVALID_POOL_SLOT)
    bad_row = int(entries[0])
    legal_row = int(entries[1])
    # Match A5's detectable failure mode: no resident victim exists.
    # Unrelated duplicate mappings are outside steady union validation.
    pool[legal_row] = torch.arange(source_capacity, dtype=torch.int32, device=pool.device)
    bad_before = pool[bad_row].clone()
    for output in outputs:
        output.fill_(-7)
    launch(case)
    torch.npu.synchronize()
    bad_end = query_counts[0]
    if int(outputs[5][0]) != 0 or not bool((outputs[2][:bad_end] == 0).all()):
        raise AssertionError("malformed steady pool did not publish request failure")
    if not torch.equal(pool[bad_row], bad_before):
        raise AssertionError("malformed steady pool was modified")
    if int(outputs[5][1]) != budget:
        raise AssertionError("malformed steady pool poisoned the legal peer request")
    _assert_sparse_pool(pool[legal_row].cpu(), prefix, budget)


def validate_cache_budget_extension(device, heads, seed):
    """Cover dynamic C bounds with A5's full-visible tail slots."""
    scenarios = (
        # L <= Q*2048 requires C=L, including the lower legal bound.
        (2048, 2048, [1, 4]),
        (4096, 4096, [4, 1]),
        # L > Q*2048 permits C=Q*2048 and 128-aligned intermediate C.
        (8320, 8192, [4, 1]),
        # Long rows and mixed Q1-4 exercise scalar first-fill stores after
        # DMA clearing, including rebuilding a previously populated row.
        (65536, 12288, [1, 2, 3, 4]),
        (33536, 12416, [1, 4]),
        (33536, 32256, [1, 4]),
        (33536, 32384, [4, 1]),
    )
    for index, (prefix, budget, routes) in enumerate(scenarios):
        case = make_case(device, heads, prefix + 256, seed + index, query_counts=routes)
        validate_first_fill(case, budget)
        if budget == 12288:
            validate_first_fill_ignores_old_pool(case, budget)
        if budget in (2048, 12288, 12416, 32384):
            case = make_case(device, heads, prefix + 256, seed + 20 + index, query_counts=routes)
            validate_steady_zero_miss(case, budget)
        if budget in (12416, 32384):
            case = make_case(device, heads, prefix + 256, seed + 40 + index, query_counts=routes)
            validate_steady_miss_and_atomicity(case, budget)
        if budget == 32384:
            case = make_case(device, heads, prefix + 256, seed + 60 + index, query_counts=routes)
            validate_identity_to_steady(case, budget)
            case = make_case(device, heads, prefix + 256, seed + 70 + index, query_counts=routes)
            validate_first_fill_ignores_old_pool(case, budget)
            case = make_case(device, heads, prefix + 256, seed + 75 + index, query_counts=routes)
            validate_steady_union_against_oracle(case, budget)
    prefix, budget = 33536, 32384
    selected = [list(range(TOPK - 1)) + [prefix + 255]]
    case = _controlled_case(device, heads, seed + 80, prefix, 256, selected)
    case[9].fill_(budget)
    case[10].fill_(-2)
    launch(case)
    torch.npu.synchronize()
    _check_controlled_routes(case, selected, prefix, budget, -2)
    case[10].fill_(-1)
    old_pool = case[12][int(case[11][0])].cpu().clone()
    launch(case)
    torch.npu.synchronize()
    _check_controlled_routes(case, selected, prefix, budget, -1, old_pool=old_pool)

    case = make_case(device, heads, 33536 + 256, seed + 100, query_counts=[4, 1])
    states, entries, pool, outputs = case[10:14]
    states.copy_(torch.tensor([-2, -3], dtype=torch.int32, device=device))
    case[7].fill_(33536 + 256)
    case[8].fill_(33536)
    invalid_row = int(entries[0])
    peer_row = int(entries[1])
    for bad_c in (0, 8064, 12417, 32512, 32640, 32768, 33536):
        case[9][0] = bad_c
        pool[invalid_row] = torch.arange(pool.shape[1], dtype=torch.int32, device=device)
        pool[peer_row].fill_(INVALID_POOL_SLOT)
        before = pool[invalid_row].clone()
        for output in outputs:
            output.fill_(-7)
        launch(case)
        torch.npu.synchronize()
        if int(outputs[5][0]) != 0 or int(outputs[5][1]) != 0:
            raise AssertionError(f"invalid C={bad_c} did not isolate its request")
        if not bool((outputs[2][:4] == 0).all()) or not bool((outputs[0][:4] == -1).all()):
            raise AssertionError(f"invalid C={bad_c} published a valid TopK")
        if not torch.equal(pool[invalid_row], before):
            raise AssertionError(f"invalid C={bad_c} modified its pool row")
        if not torch.equal(pool[peer_row], torch.arange(pool.shape[1], dtype=torch.int32, device=device)):
            raise AssertionError(f"invalid C={bad_c} affected the legal peer")

    case = make_case(device, heads, 4096 + 256, seed + 101, query_counts=[4, 1])
    case[7].fill_(4096 + 256)
    case[8].fill_(4096)
    case[9][0] = 2048  # L <= Q*2048 requires C=L.
    case[10].fill_(-2)
    before = case[12].clone()
    launch(case)
    torch.npu.synchronize()
    if int(case[13][5][0]) != 0 or not torch.equal(case[12][int(case[11][0])], before[int(case[11][0])]):
        raise AssertionError("C<L at the short-prefix boundary was accepted")


def _controlled_case(device, heads, seed, prefix, tail, selected_by_route, future_by_route=None, ranked=False):
    """Orthogonal features give every route an exact Top2048.

    With ranked=True, the final selected token is the unique 2048th score.
    """
    routes = len(selected_by_route)
    capacity = ((prefix + max(tail, 1) + BLOCK - 1) // BLOCK) * BLOCK
    case = make_case(device, heads, capacity, seed, key_length=prefix + tail, query_counts=[routes])
    query, key, weights, query_scale, key_scale = case[:5]
    query_data = torch.zeros(query.shape, dtype=torch.bfloat16)
    key_data = torch.zeros(key.shape, dtype=torch.bfloat16)
    future_by_route = future_by_route or [[] for _ in range(routes)]
    for route, selected in enumerate(selected_by_route):
        if len(selected) != TOPK or len(set(selected)) != TOPK:
            raise ValueError("controlled route must select 2048 distinct tokens")
        query_data[route, 0, route] = 1
        for index, token in enumerate(selected):
            key_data[token // BLOCK, token % BLOCK, 0, route] = 1 if not ranked or index == TOPK - 1 else 2
        for token in future_by_route[route]:
            key_data[token // BLOCK, token % BLOCK, 0, route] = 3 if ranked else 2
    query.copy_(query_data.to(device=device, dtype=torch.float8_e4m3fn))
    key.copy_(key_data.to(device=device, dtype=torch.float8_e4m3fn))
    weights.zero_()
    weights[:, 0] = 1
    query_scale.fill_(1)
    key_scale.fill_(1)
    case[8].fill_(prefix)
    case[9].fill_(8192)
    return case


def _check_controlled_routes(case, selected_by_route, prefix, budget, state, old_pool=None):
    _assert_offload_source_order(case)
    pool = case[12][int(case[11][0])].cpu()
    src, dst, route_miss, miss_src, miss_dst, request_miss = case[13]
    expected_union = set()
    for route, selected in enumerate(selected_by_route):
        actual_src = src[route, 0].cpu().tolist()
        actual_dst = dst[route, 0].cpu().tolist()
        golden = _route_golden(case, 0, route, len(selected_by_route))
        golden_set = set(golden.tolist())
        selected_set = set(selected)
        if golden_set != selected_set:
            raise AssertionError(
                f"controlled fixture differs from official LI: state={state}, "
                f"route={route}, missing={sorted(selected_set - golden_set)[:8]}, "
                f"extra={sorted(golden_set - selected_set)[:8]}"
            )
        if set(actual_src) != set(selected) or len(set(actual_src)) != TOPK:
            actual_set = set(actual_src)
            raise AssertionError(
                f"controlled TopK differs from official LI: state={state}, "
                f"route={route}, request_miss={int(request_miss[0])}, "
                f"route_miss={int(route_miss[route])}, "
                f"missing={sorted(golden_set - actual_set)[:8]}, "
                f"extra={sorted(actual_set - golden_set)[:8]}, "
                f"invalid={sum(token < 0 for token in actual_src)}"
            )
        selected_prefix = {token for token in selected if token < prefix}
        if state == -3:
            expected_miss = 0
        elif state == -2:
            expected_miss = len(selected_prefix)
            expected_union.update(selected_prefix)
        else:
            expected = {token for token in selected_prefix if int(old_pool[token]) == INVALID_POOL_SLOT}
            expected_miss = len(expected)
            expected_union.update(expected)
        count = int(route_miss[route])
        if count != expected_miss:
            raise AssertionError(f"controlled route {route} miss count {count} != {expected_miss}")
        if state != -3:
            expected_route_misses = selected_prefix if state == -2 else expected
            if set(actual_src[:count]) != expected_route_misses:
                raise AssertionError("controlled miss-prefix differs from pre-call pool")
            if any(token in expected_route_misses for token in actual_src[count:]):
                raise AssertionError("controlled hit-suffix contains a miss")
        for token, slot in zip(actual_src, actual_dst):
            expected_slot = token if state == -3 else (int(pool[token]) if token < prefix else budget + token % 256)
            if slot != expected_slot:
                raise AssertionError("controlled TopK destination is incorrect")
    expected_request_miss = 0 if state == -3 else budget if state == -2 else len(expected_union)
    count = int(request_miss[0])
    if count != expected_request_miss:
        raise AssertionError(
            f"controlled request miss count {count} != {expected_request_miss}: "
            f"state={state}, route_counts={route_miss.cpu().tolist()}, "
            f"resident_after={int((pool >= 0).sum())}, "
            f"pool_valid={set(pool[pool >= 0].tolist()) == set(range(budget))}"
        )
    if state == -3:
        if not torch.equal(pool, torch.arange(pool.numel(), dtype=torch.int32)):
            raise AssertionError("controlled -3 did not fill identity pool")
    else:
        _assert_sparse_pool(pool, prefix, budget)
        valid_src = miss_src[0, :count].cpu().tolist()
        valid_dst = miss_dst[0, :count].cpu().tolist()
        if len(set(valid_src)) != count or len(set(valid_dst)) != count:
            raise AssertionError("controlled copy list is not unique")
        if state == -1 and set(valid_src) != expected_union:
            raise AssertionError("controlled request-level union differs")
        if state == -2 and not expected_union.issubset(set(valid_src)):
            raise AssertionError("controlled first-fill did not prioritize selected union")
        if any(source >= prefix or int(pool[source]) != slot for source, slot in zip(valid_src, valid_dst)):
            raise AssertionError("controlled copy list disagrees with final pool")
        if state == -2:
            _assert_first_fill_pool(
                pool,
                prefix,
                budget,
                set().union(*map(set, selected_by_route)),
                valid_src,
                valid_dst,
            )
        if state == -1 and old_pool is not None:
            _assert_steady_pool_delta(
                old_pool,
                pool,
                prefix,
                budget,
                set().union(*map(set, selected_by_route)),
                valid_src,
                valid_dst,
            )
            protected = {
                token
                for selected in selected_by_route
                for token in selected
                if token < prefix and int(old_pool[token]) >= 0
            }
            if any(int(pool[token]) != int(old_pool[token]) for token in protected):
                raise AssertionError("controlled steady evicted a selected prefix hit")


def validate_controlled_tail_and_future(device, heads, seed):
    # Both A5 tail parity alignments and all boundary lengths. The exact-score
    # construction makes nonselection/partial/all selection unambiguous.
    for parity, prefix in enumerate((12160, 12288)):
        for tail in (0, 1, 127, 128, 129, 255, 256):
            for selected_tail_count in (0, min(tail, 128), tail):
                selected = list(range(TOPK - selected_tail_count)) + list(range(prefix, prefix + selected_tail_count))
                case = _controlled_case(device, heads, seed + parity * 100 + tail, prefix, tail, [selected])
                for state in (-3, -2):
                    case[10].fill_(state)
                    case[12].fill_(37)
                    launch(case)
                    torch.npu.synchronize()
                    _check_controlled_routes(case, [selected], prefix, 8192, state)
                # Explicitly scrambled source->slot mapping: selected prefix
                # includes both resident hits and misses; tail remains a hit.
                case[10].fill_(-1)
                row = torch.full((case[12].shape[1],), INVALID_POOL_SLOT, dtype=torch.int32)
                resident = list(range(32, 8192 + 32))
                for slot, token in enumerate(resident):
                    row[token] = 8191 - slot
                case[12][int(case[11][0])].copy_(row.to(device))
                before = row.clone()
                launch(case)
                torch.npu.synchronize()
                _check_controlled_routes(case, [selected], prefix, 8192, -1, before)
    # Future positions score higher than the 2048 legal positives but are
    # causally invisible to early MTP routes.
    prefix, tail = 12288, 256
    final = prefix + tail
    selected_by_route = [list(range(route * TOPK, (route + 1) * TOPK)) for route in range(4)]
    future = [[final - (3 - route) + i for i in range(3 - route)] for route in range(4)]
    case = _controlled_case(device, heads, seed + 300, prefix, tail, selected_by_route, future)
    case[10].fill_(-3)
    launch(case)
    torch.npu.synchronize()
    _check_controlled_routes(case, selected_by_route, prefix, 8192, -3)

    for route, forbidden in enumerate(future):
        if set(forbidden) & set(case[13][0][route, 0].cpu().tolist()):
            raise AssertionError("future high-score token crossed causal boundary")


def _install_controlled_sparse_pool(case, prefix, budget, selected_by_route, missing):
    # Only durable-prefix sources belong to the sparse pool. Selected tail
    # sources are direct hits at C + token % 256 and must never be residents.
    selected_union = {token for route in selected_by_route for token in route if token < prefix}
    if not missing.issubset(selected_union):
        raise ValueError("controlled misses must belong to selected TopK")
    resident = sorted(selected_union - missing)
    resident.extend(token for token in range(prefix) if token not in selected_union and len(resident) < budget)
    if len(resident) != budget:
        raise ValueError("controlled sparse pool cannot fill the requested budget")
    row = torch.full((case[12].shape[1],), INVALID_POOL_SLOT, dtype=torch.int32)
    # Deliberately non-identity and reverse slot order, as in the A5 pool UT.
    row[resident] = torch.arange(budget - 1, -1, -1, dtype=torch.int32)
    case[12][int(case[11][0])].copy_(row.to(case[12].device))
    return row


def validate_union_vector_boundaries(device, heads, seed):
    """Port A5's real-QLI union boundaries, not its obsolete pool ABI."""
    prefix, tail, budget = 24576, 129, 12288
    thresholds = (0, 1, 31, 32, 33, 63, 64, 65, 255, 256, 257, 511, 512, 513, 2048)
    single = [list(range(TOPK))]
    for index, count in enumerate(thresholds):
        case = _controlled_case(device, heads, seed + index, prefix, tail, single)
        case[9].fill_(budget)
        case[10].fill_(-1)
        missing = set(range(count))
        before = _install_controlled_sparse_pool(case, prefix, budget, single, missing)
        launch(case)
        torch.npu.synchronize()
        try:
            _check_controlled_routes(case, single, prefix, budget, -1, before)
        except AssertionError as error:
            raise AssertionError(f"union boundary single_miss={count}: {error}") from error

    # 1024 common tokens are selected by every route; private blocks make the
    # pre-dedup route count differ from the request-level unique union count.
    shared = set(range(1024))
    routes = [sorted(shared | set(range(1024 + r * 1024, 2048 + r * 1024))) for r in range(4)]
    profiles = (
        (32, (1, 0, 0, 0)),
        (63, (0, 0, 0, 0)),
        (64, (1, 0, 0, 0)),
        (511, (1, 0, 0, 0)),
        (512, (1, 0, 0, 0)),
        (1024, (1024, 1024, 1024, 1024)),
        (0, (0, 1, 32, 65)),
        (0, (0, 0, 513, 0)),
        (0, (0, 0, 0, 1)),
    )
    for index, (common, private) in enumerate(profiles):
        missing = set(range(common))
        for route, count in enumerate(private):
            missing.update(range(1024 + route * 1024, 1024 + route * 1024 + count))
        case = _controlled_case(device, heads, seed + 100 + index, prefix, tail, routes)
        case[9].fill_(budget)
        case[10].fill_(-1)
        before = _install_controlled_sparse_pool(case, prefix, budget, routes, missing)
        launch(case)
        torch.npu.synchronize()
        try:
            _check_controlled_routes(case, routes, prefix, budget, -1, before)
        except AssertionError as error:
            raise AssertionError(f"union boundary common={common}, private={private}: {error}") from error
        if int(case[13][5][0]) != len(missing):
            raise AssertionError("cross-route union cardinality is incorrect")


def validate_q4_full_visible_tail(device, heads, seed):
    """A5 full-visible Q4 selection, including causal future above L."""
    prefix, tail, budget = 12288, 200, 8192
    final = prefix + tail
    for index, selection in enumerate(("none", "partial", "all")):
        selected = []
        future = []
        for route in range(4):
            visible = final - (3 - route)
            tail_ids = list(range(prefix, visible))
            chosen_tail = [] if selection == "none" else (tail_ids[::2] if selection == "partial" else tail_ids)
            chosen = list(range(route * 2048, route * 2048 + TOPK - len(chosen_tail)))
            selected.append(chosen + chosen_tail)
            future.append(list(range(visible, final)))
        case = _controlled_case(device, heads, seed + index, prefix, tail, selected, future)
        for state in (-2, -1):
            case[10].fill_(state)
            if state == -1:
                selected_prefix = {token for route in selected for token in route if token < prefix}
                missing = set(sorted(selected_prefix)[:300])
                before = _install_controlled_sparse_pool(
                    case,
                    prefix,
                    budget,
                    selected,
                    missing,
                )
            else:
                case[12].fill_(49)
                before = None
            launch(case)
            torch.npu.synchronize()
            try:
                _check_controlled_routes(case, selected, prefix, budget, state, before)
            except AssertionError as error:
                raise AssertionError(f"Q4 tail selection={selection}, state={state}: {error}") from error
            for route, forbidden in enumerate(future):
                if set(forbidden) & set(case[13][0][route, 0].cpu().tolist()):
                    raise AssertionError("Q4 future token crossed causal boundary")


def _scalar_score_case(device, heads, seed, prefix, tail, routes, scores):
    """Positive FP8 ones yield exactly heads*128*k_scale before BF16 sorting."""
    capacity = ((prefix + tail + BLOCK - 1) // BLOCK) * BLOCK
    case = make_case(device, heads, capacity, seed, key_length=prefix + tail, query_counts=[routes])
    query, key, weights, query_scale, key_scale = case[:5]
    query.copy_(torch.ones(query.shape, dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn))
    key.copy_(torch.ones(key.shape, dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn))
    weights.fill_(1)
    query_scale.fill_(1)
    scales = torch.zeros(key_scale.shape, dtype=torch.float32)
    scales.view(-1)[: len(scores)] = scores / float(heads * HEAD_DIM)
    key_scale.copy_(scales.to(device))
    case[8].fill_(prefix)
    return case


def validate_non_offload_ordered_scores(device, heads, seed, *, long=False):
    """Check exact Official output order and independent unique-score members.

    C8 histogram TopK gathers survivors; its public order is not required to
    equal a globally score-sorted argsort. Only use argsort for membership.
    """
    prefix = 131200 if long else 8320
    for offset, routes in enumerate((1, 4, 8, 14)):
        generator = torch.Generator().manual_seed(seed + offset)
        candidates = torch.randperm(prefix, generator=generator)[: TOPK + BLOCK]
        scores = torch.zeros(prefix + BLOCK, dtype=torch.float32)
        bits = torch.arange(0x3000, 0x3000 + len(candidates), dtype=torch.int32).to(torch.int16)
        scores[candidates] = bits.view(torch.bfloat16).float()
        # Two strictly ranked tail winners are visible even to Q14's first
        # route. They must remain in the full-visible LI; offload budget
        # checks and pool classification are irrelevant for identity -3.
        scores[prefix + 7], scores[prefix + 64] = 2, 4
        case = _scalar_score_case(device, heads, seed + offset, prefix, BLOCK, routes, scores)
        validate_non_offload(case, ordered=True)
        for route in range(routes):
            visible = prefix + BLOCK - (routes - 1 - route)
            expected = torch.argsort(scores[:visible], descending=True)[:TOPK]
            actual = case[13][0][route, 0].cpu().to(torch.int64)
            actual_ids, expected_ids = set(actual.tolist()), set(expected.tolist())
            if len(actual_ids) != TOPK or actual_ids != expected_ids:
                raise AssertionError(
                    f"unique-score Q{routes} route {route} members are wrong: "
                    f"missing={sorted(expected_ids - actual_ids)[:8]}, "
                    f"extra={sorted(actual_ids - expected_ids)[:8]}"
                )


def validate_static_abi_rejections(device, heads, seed):
    """Only the A3 host-validation ideas that apply to the A5 ABI."""
    case = make_case(device, heads, 16384, seed, query_counts=[1])
    variants = (
        ("fp16 query scale", 3, case[3].to(torch.float16)),
        ("fp16 key scale", 4, case[4].to(torch.float16)),
        ("narrow block table", 5, case[5][:, :-1].contiguous()),
        ("wrong pool width", 12, case[12][:, :-1].contiguous()),
        ("wrong TopK width", 13, (*case[13][:1], case[13][1][:, :, :-1].contiguous(), *case[13][2:])),
    )
    for name, index, replacement in variants:
        malformed = list(case)
        malformed[index] = replacement
        try:
            launch(tuple(malformed))
        except RuntimeError:
            continue
        raise AssertionError(f"host accepted {name}")

    # With B=1, T=15 violates the host's static T <= 14*B bound.
    case = make_case(device, heads, 33536, seed + 1, query_counts=[15])
    before_pool = case[12].clone()
    before_outputs = tuple(output.clone() for output in case[13])
    try:
        launch(case)
    except RuntimeError as error:
        if "1..14 query rows per request" not in str(error):
            raise AssertionError(f"Q=15 failed for an unexpected reason: {error}") from error
    else:
        raise AssertionError("host accepted B=1,T=15")
    if not torch.equal(case[12], before_pool) or any(
        not torch.equal(output, before) for output, before in zip(case[13], before_outputs)
    ):
        raise AssertionError("host-rejected Q=15 modified caller-owned buffers")

    # B=2,T=16 passes that static bound, but request 0 still has Q=15.
    # The device must reject only that request and preserve its pool row.
    case = make_case(device, heads, 33536, seed + 2, query_counts=[15, 1])
    before_pool = case[12].clone()
    launch(case)
    torch.npu.synchronize()
    src, dst, route_miss, _, _, request_miss = case[13]
    if (
        int(request_miss[0]) != 0
        or bool((route_miss[:15] != 0).any())
        or bool((src[:15] != -1).any())
        or bool((dst[:15] != -1).any())
        or not torch.equal(case[12][int(case[11][0])], before_pool[int(case[11][0])])
    ):
        raise AssertionError("device accepted Q=15 or changed its pool row")
    if int(request_miss[1]) != 0 or int(route_miss[15]) != 0:
        raise AssertionError("Q=15 invalid request poisoned a valid peer")
    if set(src[15, 0].cpu().tolist()) != set(_route_golden(case, 1, 15, 16).tolist()):
        raise AssertionError("valid peer after Q=15 differs from official LI")


def _assert_extended_requests(case, goldens, before_pool):
    """Check Q1--14 publications against LI and the pre-call sparse pool."""
    _assert_offload_source_order(case)
    src, dst, route_miss, miss_src, miss_dst, request_miss = case[13]
    src = src[:, 0].cpu()
    dst = dst[:, 0].cpu()
    route_miss = route_miss.cpu().tolist()
    request_miss = request_miss.cpu().tolist()
    miss_src = miss_src.cpu()
    miss_dst = miss_dst.cpu()
    after_pool = case[12].cpu()
    states = case[10].cpu().tolist()
    entries = case[11].cpu().tolist()
    prefixes = case[8].cpu().tolist()
    budgets = case[9].cpu().tolist()
    start = 0
    for request, routes in enumerate(case[14]):
        end = start + routes
        state, prefix, budget = states[request], prefixes[request], budgets[request]
        row = entries[request]
        current = after_pool[row]
        old = before_pool[row]
        expected_union = set()
        for route in range(start, end):
            sources = src[route].tolist()
            slots = dst[route].tolist()
            if len(set(sources)) != TOPK or set(sources) != goldens[route]:
                raise AssertionError(f"extended Q{routes} route {route} differs from A5 C8 LI")
            missing = (
                set()
                if state == -3
                else {
                    token
                    for token in goldens[route]
                    if token < prefix and (state == -2 or int(old[token]) == INVALID_POOL_SLOT)
                }
            )
            expected_union.update(missing)
            count = route_miss[route]
            if count != len(missing):
                raise AssertionError(
                    f"extended Q{routes} state={state} route={route} miss count {count} != {len(missing)}"
                )
            if set(sources[:count]) != missing or set(sources[count:]) != goldens[route] - missing:
                raise AssertionError("extended route miss/hit partition is wrong")
            for token, slot in zip(sources, slots):
                expected_slot = (
                    token if state == -3 else (int(current[token]) if token < prefix else budget + token % 256)
                )
                if slot != expected_slot:
                    raise AssertionError(
                        f"extended Q{routes} state={state} token={token} slot {slot} != {expected_slot}"
                    )
                if state == -1 and token < prefix and int(old[token]) >= 0:
                    if int(current[token]) != int(old[token]):
                        raise AssertionError("extended steady call remapped a protected hit")
        count = request_miss[request]
        expected_count = 0 if state == -3 else budget if state == -2 else len(expected_union)
        if count != expected_count:
            raise AssertionError(f"extended Q{routes} state={state} request miss count {count} != {expected_count}")
        if state == -3:
            identity = torch.arange(current.numel(), dtype=torch.int32)
            if not torch.equal(current, identity):
                wrong = torch.nonzero(current != identity).flatten()
                examples = [(int(index), int(current[index])) for index in wrong[:8]]
                raise AssertionError(
                    f"extended -3 identity pool is incorrect: request={request} "
                    f"Q={routes} row={row} mismatches={wrong.numel()} "
                    f"first(index,actual)={examples}"
                )
        else:
            try:
                _assert_sparse_pool(current, prefix, budget)
            except AssertionError as error:
                raise AssertionError(f"extended request={request} Q={routes} state={state}: {error}") from error
            copied = miss_src[request, :count].tolist()
            copied_slots = miss_dst[request, :count].tolist()
            if (
                len(set(copied)) != count
                or len(set(copied_slots)) != count
                or any(token >= prefix or int(current[token]) != slot for token, slot in zip(copied, copied_slots))
            ):
                raise AssertionError("extended COPY list disagrees with final pool")
            if state == -2:
                if not expected_union.issubset(set(copied)) or copied_slots != list(range(budget)):
                    raise AssertionError("extended first-fill union or slot order is wrong")
                _assert_first_fill_pool(current, prefix, budget, set().union(*goldens[start:end]), copied, copied_slots)
            elif copied != sorted(expected_union):
                raise AssertionError("extended steady COPY union is not source ordered")
            if state == -1:
                selected = set().union(*goldens[start:end])
                _assert_steady_pool_delta(old, current, prefix, budget, selected, copied, copied_slots)
        start = end
    used = set(entries)
    for row in range(after_pool.shape[0]):
        if row not in used and not torch.equal(after_pool[row], before_pool[row]):
            raise AssertionError("extended call changed an inactive pool row")


def validate_extended_all_route_counts(device, heads, seed, *, query_counts=None):
    """Run a mixed-Q batch through -3, -2 and repeatable -1 (Q1--14 by default)."""
    counts = list(range(1, 15) if query_counts is None else query_counts)
    if not counts or any(q < 1 or q > 14 for q in counts):
        raise ValueError("query_counts must contain Q values in 1..14")
    prefix = 32768
    case = make_case(
        device,
        heads,
        prefix + 256,
        seed,
        key_length=prefix + 256,
        query_counts=counts,
    )
    case[8].fill_(prefix)
    budgets = [32384 if q == 14 else q * TOPK + (BLOCK if q in (5, 8) else 0) for q in counts]
    case[9].copy_(torch.tensor(budgets, dtype=torch.int32, device=device))
    # Exercise the logical-to-physical block indirection used by A3's
    # semantic regressions.  The official A5 LI remains the TopK oracle.
    generator = torch.Generator().manual_seed(seed + 271828)
    table = case[5].cpu()
    for row in range(table.shape[0]):
        order = torch.randperm(table.shape[1], generator=generator)
        table[row] = table[row, order].clone()
    case[5].copy_(table.to(device))
    validate_non_offload(case)
    goldens = [set(row) for row in case[13][0][:, 0].cpu().tolist()]
    case[10].fill_(-2)
    # First decode from an identity row is a separate A3 lifecycle edge.
    before = case[12].cpu().clone()
    for output in case[13]:
        output.fill_(-7)
    launch(case)
    torch.npu.synchronize()
    _assert_extended_requests(case, goldens, before)

    case[12].fill_(37)  # First fill must not consume an old identity row.
    before = case[12].cpu().clone()
    for output in case[13]:
        output.fill_(-7)
    launch(case)
    torch.npu.synchronize()
    _assert_extended_requests(case, goldens, before)

    case[10].fill_(-1)
    before = case[12].cpu().clone()
    for output in case[13]:
        output.fill_(-7)
    launch(case)
    torch.npu.synchronize()
    _assert_extended_requests(case, goldens, before)
    if bool((case[13][5] != 0).any()) or not torch.equal(case[12].cpu(), before):
        raise AssertionError("extended first-fill pool did not converge at -1")
    for output in case[13]:
        output.fill_(-7)
    launch(case)
    torch.npu.synchronize()
    _assert_extended_requests(case, goldens, before)
    if bool((case[13][5] != 0).any()) or not torch.equal(case[12].cpu(), before):
        raise AssertionError("extended repeated -1 changed the resident pool")

    # Every request has eight selected resident sources displaced by eight
    # nonselected nonresidents.  The pool still has a C-slot bijection.
    modified = before.clone()
    entries = case[11].cpu().tolist()
    start = 0
    for request, routes in enumerate(counts):
        end = start + routes
        selected = set().union(*goldens[start:end])
        pool_row = modified[entries[request]]
        displaced = sorted(token for token in selected if token < prefix and int(pool_row[token]) >= 0)[:8]
        replacements = [
            token for token in range(prefix) if token not in selected and int(pool_row[token]) == INVALID_POOL_SLOT
        ][:8]
        if len(displaced) != 8 or len(replacements) != 8:
            raise AssertionError(f"extended Q{routes} miss fixture is too small")
        for source, replacement in zip(displaced, replacements):
            slot = int(pool_row[source])
            pool_row[source] = INVALID_POOL_SLOT
            pool_row[replacement] = slot
        start = end
    case[12].copy_(modified.to(device))
    for output in case[13]:
        output.fill_(-7)
    launch(case)
    torch.npu.synchronize()
    _assert_extended_requests(case, goldens, modified)
    if case[13][5].cpu().tolist() != [8] * len(counts):
        raise AssertionError(f"extended Q={counts} miss union did not contain eight IDs each")

    # Check the independent -3 -> -1 lifecycle for every Q, rather than
    # assuming that first decode covered identity-pool normalization.
    case[10].fill_(-3)
    before = case[12].cpu().clone()
    for output in case[13]:
        output.fill_(-7)
    launch(case)
    torch.npu.synchronize()
    _assert_extended_requests(case, goldens, before)
    case[10].fill_(-1)
    normalized = case[12].cpu().clone()
    for request, row in enumerate(entries):
        normalized[row, budgets[request] :] = INVALID_POOL_SLOT
    for output in case[13]:
        output.fill_(-7)
    launch(case)
    torch.npu.synchronize()
    _assert_extended_requests(case, goldens, normalized)


def validate_extended_tail_and_long(device, heads, seed, *, include_long=False):
    """Check Q14 causal tail, high source tags and the 21-bit endpoint."""
    routes, budget = 14, 32384
    prefix, tail = 33536, 256
    selected = [
        list(range(route * (TOPK - 1), (route + 1) * (TOPK - 1))) + [prefix + (255 if route == routes - 1 else 120)]
        for route in range(routes)
    ]
    final = prefix + tail
    future = [list(range(final - (routes - 1 - route), final)) for route in range(routes)]
    case = _controlled_case(device, heads, seed, prefix, tail, selected, future)
    case[9].fill_(budget)
    for state in (-3, -2):
        case[10].fill_(state)
        case[12].fill_(37)
        launch(case)
        torch.npu.synchronize()
        _check_controlled_routes(case, selected, prefix, budget, state)
    case[10].fill_(-1)
    missing = set(selected[0][:16])
    before = _install_controlled_sparse_pool(case, prefix, budget, selected, missing)
    launch(case)
    torch.npu.synchronize()
    _check_controlled_routes(case, selected, prefix, budget, -1, before)
    for route, forbidden in enumerate(future):
        if set(forbidden) & set(case[13][0][route, 0].cpu().tolist()):
            raise AssertionError("Q14 selected a causally invisible future token")

    if not include_long:
        return
    # The 2^21 endpoint uses all four high source tag bits, including tag 15;
    # Q14 also exercises a full-width route workspace at that boundary.
    prefix, tail = MAX_SOURCE_CAPACITY - BLOCK, BLOCK
    selected = []
    for route in range(routes):
        high = (route + 1) << 17
        low = list(range(TOPK - (3 if route == routes - 1 else 2)))
        values = low + [high, prefix - 1 - route]
        if route == routes - 1:
            values.append(MAX_SOURCE_CAPACITY - 1)
        selected.append(values)
    case = _controlled_case(device, heads, seed + 1, prefix, tail, selected)
    case[9].fill_(budget)
    case[10].fill_(-1)
    selected_prefix = {token for row in selected for token in row if token < prefix}
    high_missing = {token for token in selected_prefix if token >= PACKED_SOURCE_CAPACITY}
    before = _install_controlled_sparse_pool(
        case,
        prefix,
        budget,
        selected,
        high_missing,
    )
    launch(case)
    torch.npu.synchronize()
    _check_controlled_routes(case, selected, prefix, budget, -1, before)
    if MAX_SOURCE_CAPACITY - 1 not in case[13][0][-1, 0].cpu().tolist():
        raise AssertionError("Q14 lost the highest 21-bit tail source ID")


def _snapshot_outputs(case):
    return tuple(output.cpu().clone() for output in case[13]), case[12].cpu().clone()


def _compare_graph_eager(graph_result, eager_result):
    graph_outputs, graph_pool = graph_result
    eager_outputs, eager_pool = eager_result
    if not torch.equal(graph_pool, eager_pool):
        raise AssertionError("graph replay pool differs from eager")
    for index in (2, 5):
        if not torch.equal(graph_outputs[index], eager_outputs[index]):
            raise AssertionError(f"graph replay count output {index} differs from eager")
    for route, count in enumerate(graph_outputs[2].tolist()):
        graph_pairs = list(zip(graph_outputs[0][route, 0].tolist(), graph_outputs[1][route, 0].tolist()))
        eager_pairs = list(zip(eager_outputs[0][route, 0].tolist(), eager_outputs[1][route, 0].tolist()))
        if sorted(graph_pairs) != sorted(eager_pairs):
            raise AssertionError(f"graph replay route {route} differs from eager")
    for request, count in enumerate(graph_outputs[5].tolist()):
        if count < 0:
            continue
        graph_pairs = list(zip(graph_outputs[3][request, :count].tolist(), graph_outputs[4][request, :count].tolist()))
        eager_pairs = list(zip(eager_outputs[3][request, :count].tolist(), eager_outputs[4][request, :count].tolist()))
        if sorted(graph_pairs) != sorted(eager_pairs):
            raise AssertionError(f"graph replay request {request} copy list differs")


def validate_graph_replay(device, heads, seed):
    # Capture only LIM at fixed addresses. Replays change state and metadata;
    # eager sees exactly the same pre-call pool and tensor contents.
    case = make_case(device, heads, 16384, seed, query_counts=[1, 2, 3, 4])
    inputs = [*case[:13], *case[13]]
    pointers = tuple(t.data_ptr() for t in inputs)
    case[7].fill_(12544)
    case[8].fill_(12288)
    case[9].fill_(8192)
    case[10].fill_(-3)
    for _ in range(2):
        launch(case)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        launch(case)
    torch.npu.synchronize()
    scenarios = (
        ((-3, -3, -3, -3), 12416, (1, 2, 3, 4)),
        ((-2, -2, -2, -2), 12544, (4, 3, 2, 1)),
        ((-1, -1, -1, -1), 12544, (4, 3, 2, 1)),
        ((-3, -1, -2, -3), 12544, (2, 4, 1, 3)),
        ((-1, -1, -1, -1), 12544, (1, 2, 3, 4)),
    )
    for states, actual_length, route_counts in scenarios:
        case[10].copy_(torch.tensor(states, dtype=torch.int32, device=device))
        case[7].fill_(actual_length)
        ends: list[int] = []
        for count in route_counts:
            ends.append((ends[-1] if ends else 0) + count)
        case[6].copy_(torch.tensor(ends, dtype=torch.int32, device=device))
        before = case[12].clone()
        for output in case[13]:
            output.fill_(-777)
        graph.replay()
        torch.npu.synchronize()
        graph_result = _snapshot_outputs(case)
        case[12].copy_(before)
        for output in case[13]:
            output.fill_(-777)
        launch(case)
        torch.npu.synchronize()
        _compare_graph_eager(graph_result, _snapshot_outputs(case))
        if tuple(t.data_ptr() for t in inputs) != pointers:
            raise AssertionError("NPUGraph changed a caller-owned tensor address")


def validate_short_non_offload_cases(device, heads, source_capacity, seed):
    """Compare every short -3 route and its padding with official A5 C8 LI."""
    for offset, length in enumerate((1, 1024, 2047, 2048, 2049)):
        case = make_case(
            device,
            heads,
            source_capacity,
            seed + offset,
            key_length=length,
            query_counts=[1],
        )
        validate_non_offload(case)

    # One Q4 request crosses the 2048 boundary across its routes.
    # Route visibility is 2046, 2047, 2048, 2049 respectively.
    case = make_case(
        device,
        heads,
        source_capacity,
        seed + 10,
        key_length=2049,
        query_counts=[4],
    )
    validate_non_offload(case)

    # Independent requests can mix short and long routes in the same batch.
    case = make_case(
        device,
        heads,
        source_capacity,
        seed + 11,
        query_counts=[1, 4, 2, 3],
    )
    case[7].copy_(
        torch.tensor(
            [1, 2049, 1024, 2050],
            dtype=torch.int32,
            device=device,
        )
    )
    case[8].copy_(case[7])
    validate_non_offload(case)

    # An invalid metadata peer must not suppress the short and boundary routes.
    case[10][2] = 0
    case[12].fill_(INVALID_POOL_SLOT)
    for output in case[13]:
        output.fill_(-7)
    launch(case)
    torch.npu.synchronize()
    src, dst, route_miss, _, _, request_miss = case[13]
    if (
        int(request_miss[2]) != 0
        or bool((route_miss[5:7] != 0).any())
        or bool((src[5:7] != -1).any())
        or bool((dst[5:7] != -1).any())
    ):
        raise AssertionError("invalid -3 peer did not publish A3-style zero counts")
    bad_row = int(case[11][2])
    if bool((case[12][bad_row] != INVALID_POOL_SLOT).any()):
        raise AssertionError("invalid -3 peer modified its pool row")
    start = 0
    for request, routes in enumerate(case[14]):
        end = start + routes
        if request != 2:
            if int(request_miss[request]) != 0 or bool((route_miss[start:end] != 0).any()):
                raise AssertionError("invalid peer affected a valid -3 count")
            row = int(case[11][request])
            identity = torch.arange(case[12].shape[1], dtype=torch.int32)
            if not torch.equal(case[12][row].cpu(), identity):
                raise AssertionError("invalid peer affected a valid identity pool")
            for route in range(start, end):
                golden = _route_golden(case, request, route, end)
                actual = src[route, 0].cpu()
                if not torch.equal(actual, dst[route, 0].cpu()):
                    raise AssertionError("valid -3 route lost identity destinations")
                if int(case[7][request]) - (end - 1 - route) <= TOPK:
                    if not torch.equal(actual, golden):
                        raise AssertionError("valid short -3 route differs from official LI")
                elif set(actual.tolist()) != set(golden.tolist()):
                    raise AssertionError("valid long -3 route differs from official LI")
        start = end


@pytest.fixture(params=[32, 64])
def heads(request):
    return request.param


@pytest.fixture(scope="module")
def a5_custom_operator():
    # enable_custom_op() deliberately disables runtime custom ops on A5.
    # As in test_add_rms_norm_bias_a5, load the compiled extension explicitly
    # for this operator test without changing the production capability gate.
    bootstrap_custom_op_env()
    extension = importlib.import_module("vllm_ascend.vllm_ascend_C")
    op_name = "npu_fused_quant_lightning_indexer_manage"
    assert hasattr(torch.ops._C_ascend, op_name), (
        f"{extension.__file__} does not register _C_ascend::{op_name}. "
        "Rebuild and install vllm-ascend from the A3/A5 LIM branch, and check "
        "that pytest imports the newly built extension."
    )
    for dispatch_key in ("PrivateUse1", "Meta"):
        assert torch._C._dispatch_has_kernel_for_dispatch_key(f"_C_ascend::{op_name}", dispatch_key), (
            f"{extension.__file__} is missing {dispatch_key} registration for {op_name}"
        )


@pytest.fixture
def a5_device(a5_custom_operator):
    if not is_950():
        pytest.skip("Requires Ascend 950 (A5)")

    yield torch.device("npu")
    torch.npu.synchronize()
    gc.collect()
    torch.npu.empty_cache()


@pytest.mark.parametrize(
    "validate",
    [
        validate_tight_causal_boundary,
        validate_controlled_tail_and_future,
        validate_q4_full_visible_tail,
        validate_cache_budget_extension,
        validate_static_abi_rejections,
        validate_extended_all_route_counts,
        validate_non_offload_ordered_scores,
        validate_graph_replay,
        validate_union_vector_boundaries,
    ],
)
def test_a5_contracts(a5_device, heads, validate):
    validate(a5_device, heads, 20261010)


def test_a5_tail_and_long_source(a5_device, heads):
    validate_extended_tail_and_long(a5_device, heads, 20261010, include_long=True)


def test_a5_invalid_request_isolation(a5_device, heads):
    validate_dynamic_failure_matrix(a5_device, heads, 16384, 20261010)


@pytest.mark.parametrize("invalid", [None, "scale", "weight_stride"])
def test_a5_meta_contract(heads, invalid, a5_custom_operator):
    total_queries, batch, blocks = 4, 1, 128
    weights = torch.empty((total_queries, heads), dtype=torch.bfloat16, device="meta")
    query_scale = torch.empty_like(weights, dtype=torch.float32)
    query = torch.empty((total_queries, heads, HEAD_DIM), dtype=torch.float8_e4m3fn, device="meta")
    key_scale = torch.empty((blocks, BLOCK, 1), dtype=torch.float32, device="meta")
    key = torch.empty((blocks, BLOCK, 1, HEAD_DIM), dtype=torch.float8_e4m3fn, device="meta")
    table = torch.empty((batch, blocks), dtype=torch.int32, device="meta")
    metadata = [torch.empty((batch,), dtype=torch.int32, device="meta") for _ in range(6)]
    pool = torch.empty((batch, blocks * BLOCK), dtype=torch.int32, device="meta")
    outputs = [
        torch.empty((total_queries, 1, TOPK), dtype=torch.int32, device="meta"),
        torch.empty((total_queries, 1, TOPK), dtype=torch.int32, device="meta"),
        torch.empty((total_queries,), dtype=torch.int32, device="meta"),
        torch.empty((batch, MISS_CAPACITY), dtype=torch.int32, device="meta"),
        torch.empty((batch, MISS_CAPACITY), dtype=torch.int32, device="meta"),
        torch.empty((batch,), dtype=torch.int32, device="meta"),
    ]
    if invalid == "scale":
        query_scale = query_scale.to(torch.float16)
    elif invalid == "weight_stride":
        weights = torch.empty((heads, total_queries), dtype=torch.bfloat16, device="meta").t()
    args = (weights, query_scale, query, key_scale, key, table, *metadata, pool, *outputs)
    if invalid is None:
        assert torch.ops._C_ascend.npu_fused_quant_lightning_indexer_manage(*args) is None
        assert all(tensor.is_meta for tensor in outputs)
    else:
        with pytest.raises(RuntimeError):
            torch.ops._C_ascend.npu_fused_quant_lightning_indexer_manage(*args)


@pytest.mark.parametrize("routes", range(1, 15), ids=[f"q{q:02d}" for q in range(1, 15)])
@pytest.mark.parametrize("length_mode", ["minimum", "block128", "topk_minus1"])
def test_a5_short_non_offload(a5_device, heads, routes, length_mode):
    """Every Q=1..14 route stays below 2048; verify identity and -1 padding."""
    key_length = {
        "minimum": routes,
        "block128": BLOCK,
        "topk_minus1": TOPK - 1,
    }[length_mode]
    case = make_case(
        a5_device,
        heads,
        2 * TOPK,
        20261012 + routes * 100 + key_length,
        key_length=key_length,
        query_counts=[routes],
    )
    # Reuse the original official A5 oracle, identity-pool, zero-miss and
    # caller-owned buffer assertions. Every selected token is causally valid.
    validate_non_offload(case)
    sources, destinations = case[13][:2]
    for route in range(routes):
        visible = key_length - (routes - 1 - route)
        assert 0 < visible < TOPK
        for result in (sources[route, 0], destinations[route, 0]):
            assert bool((result[:visible] >= 0).all())
            assert bool((result[:visible] < visible).all())
            assert bool((result[visible:] == -1).all())


def test_a5_short_mixed_batch_and_invalid_peer(a5_device, heads):
    # Original source suite: lengths 1/1024/2047/2048/2049, a Q4 request
    # crossing the TopK boundary, a mixed-length batch and invalid-peer isolation.
    validate_short_non_offload_cases(a5_device, heads, 2 * TOPK, 20261012)
