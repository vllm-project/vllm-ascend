# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HY4 Flash indexer."""

from functools import lru_cache

import torch
import torch.distributed as dist
from vllm.distributed import get_tp_group
from vllm.models.hy_v4_flash.reference import FP8_MAX, INDEX_DIM
from vllm.triton_utils import HAS_TRITON, tl, triton

from vllm_ascend.device.device_config import is_950

_GROUPS_PER_PROGRAM = 16


@triton.jit
def pool_key_addresses(
    block_table,
    group,
    lane,
    live,
    ROW_WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    RATIO: tl.constexpr,
    LANES_PER_ROW: tl.constexpr,
):
    token = group * RATIO
    block = tl.load(block_table + token // BLOCK_SIZE, mask=live, other=0).to(tl.int64)
    slot = block * BLOCK_SIZE + token % BLOCK_SIZE
    # Spread each key across the spare lanes of its group's token rows.
    return (
        (slot[:, None] + lane[None, :] // LANES_PER_ROW) * ROW_WIDTH
        + ROW_WIDTH
        - LANES_PER_ROW
        + lane[None, :] % LANES_PER_ROW
    )


@triton.jit(do_not_specialize=["first_group", "num_groups"])
def _copy_pool_keys(
    cache,
    block_table,
    keys,
    first_group,
    num_groups,
    end_group,
    ROW_WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    RATIO: tl.constexpr,
    LANES_PER_ROW: tl.constexpr,
    WIDTH: tl.constexpr,
    GROUPS: tl.constexpr,
    WRITE: tl.constexpr,
    DEVICE_START: tl.constexpr,
    DEVICE_END: tl.constexpr,
):
    row = tl.program_id(0) * GROUPS + tl.arange(0, GROUPS)
    lane = tl.arange(0, WIDTH)
    start = tl.load(first_group) if DEVICE_START else first_group
    group = row + start
    live = row < num_groups
    if DEVICE_END:
        live = live & (group < tl.load(end_group))
    address = pool_key_addresses(block_table, group, lane, live, ROW_WIDTH, BLOCK_SIZE, RATIO, LANES_PER_ROW)
    key_address = row[:, None] * WIDTH + lane[None, :]
    mask = live[:, None]
    if WRITE:
        values = tl.load(keys + key_address, mask=mask, other=0)
        tl.store(cache + address, values, mask=mask)
    else:
        values = tl.load(cache + address, mask=mask, other=0).to(tl.float32)
        tl.store(keys + key_address, values, mask=row[:, None] < num_groups)


def copy_pool_keys(cache, block_table, keys, first_group, block_size, ratio, *, write, end_group=None):
    # The rounded E4M3 values have power-of-two row scales and fit exactly in BF16.
    cache = cache.view(torch.bfloat16)
    _copy_pool_keys[(triton.cdiv(keys.shape[0], _GROUPS_PER_PROGRAM),)](
        cache,
        block_table,
        keys,
        first_group,
        keys.shape[0],
        end_group,
        cache.shape[-1],
        block_size,
        ratio,
        INDEX_DIM // ratio,
        INDEX_DIM,
        _GROUPS_PER_PROGRAM,
        write,
        isinstance(first_group, torch.Tensor),
        end_group is not None,
        num_warps=1,
    )


_MIN_AMAX = 1e-4
_E4M3_MIN_NORMAL = tl.constexpr(2.0**-6)
_E4M3_FRACTION_BITS = tl.constexpr(3)
_FP32_FRACTION_BITS = tl.constexpr(23)
_FP32_EXPONENT_BIAS = tl.constexpr(127)


@triton.jit
def _indexpool_hadamard(
    vectors,
    transformed,
    maxima,
    WIDTH: tl.constexpr,
    LEVELS: tl.constexpr,
    NORMALIZE: tl.constexpr,
):
    row = tl.program_id(0)
    lane = tl.arange(0, WIDTH)
    offset = row * WIDTH + lane
    values = tl.load(vectors + offset).to(tl.float32)
    for level in tl.static_range(LEVELS):
        stride = 1 << level
        partner = tl.gather(values, lane ^ stride, 0)
        values = tl.where((lane & stride) == 0, values + partner, partner - values)
    values = values * NORMALIZE

    magnitude = tl.abs(values)
    nonfinite = tl.max(tl.where(magnitude < float("inf"), 0, 1), 0) != 0
    amax = tl.where(nonfinite, float("nan"), tl.max(magnitude, 0))
    tl.store(transformed + offset, values)
    tl.store(maxima + row, amax)


@triton.jit
def _indexpool_round_e4m3(
    transformed,
    normalized,
    scales,
    WIDTH: tl.constexpr,
    QUANT_MAX: tl.constexpr,
):
    row = tl.program_id(0)
    offset = row * WIDTH + tl.arange(0, WIDTH)
    values = tl.load(transformed + offset)
    scale = tl.load(scales + row)
    magnitude = tl.minimum(tl.abs(tl.load(normalized + offset)), QUANT_MAX)
    # The FP32 exponent gives the normal E4M3 spacing. Clamping at the
    # smallest normal magnitude also gives the fixed subnormal spacing.
    magnitude_bits = tl.maximum(magnitude, _E4M3_MIN_NORMAL).to(tl.int32, bitcast=True)
    exponent = (magnitude_bits >> _FP32_FRACTION_BITS) - _FP32_EXPONENT_BIAS - _E4M3_FRACTION_BITS
    unit = ((exponent + _FP32_EXPONENT_BIAS) << _FP32_FRACTION_BITS).to(tl.float32, bitcast=True)
    inverse_unit = ((_FP32_EXPONENT_BIAS - exponent) << _FP32_FRACTION_BITS).to(tl.float32, bitcast=True)
    rounded = tl.extra.cann.libdevice.nearbyint(magnitude * inverse_unit)
    sign = tl.where(values > 0, 1.0, tl.where(values < 0, -1.0, 0.0))
    result = tl.where(scale == scale, rounded * unit * sign * scale, float("nan"))
    tl.store(normalized + offset, result)


def encode_indexpool_fp8(vectors: torch.Tensor) -> torch.Tensor:
    """Return the FP32, E4M3-rounded IndexPool rows used by the scorer."""
    rows = vectors.numel() // INDEX_DIM
    transformed = torch.empty_like(vectors, dtype=torch.float32)
    maxima = vectors.new_empty((*vectors.shape[:-1], 1), dtype=torch.float32)
    _indexpool_hadamard[(rows,)](
        vectors,
        transformed,
        maxima,
        INDEX_DIM,
        INDEX_DIM.bit_length() - 1,
        INDEX_DIM**-0.5,
        num_warps=1,
        enable_fp_fusion=False,
    )
    # Keep CANN's scale and division rounding at exponent and E4M3 bin boundaries.
    scale = torch.exp2(torch.ceil(torch.log2(maxima.clamp_min(_MIN_AMAX) / FP8_MAX)))
    normalized = transformed / scale
    _indexpool_round_e4m3[(rows,)](
        transformed, normalized, scale, INDEX_DIM, FP8_MAX, num_warps=1, enable_fp_fusion=False
    )
    return normalized


@triton.jit
def _indexpool_encode_fused(
    vectors, output, width: tl.constexpr, levels: tl.constexpr, quant_max: tl.constexpr, min_amax: tl.constexpr
):
    row = tl.program_id(0)
    lane = tl.arange(0, width)
    value = tl.load(vectors + row * width + lane).to(tl.float32)
    for level in tl.static_range(levels):
        stride = 1 << level
        partner = tl.gather(value, lane ^ stride, 0)
        value = tl.where((lane & stride) == 0, value + partner, partner - value)
    value = value * (width**-0.5)
    maximum = tl.maximum(tl.max(tl.abs(value), 0), min_amax)
    scale = tl.exp2(tl.ceil(tl.log2(maximum / quant_max)))
    magnitude = tl.minimum(tl.abs(value / scale), quant_max)
    bits = tl.maximum(magnitude, _E4M3_MIN_NORMAL).to(tl.int32, bitcast=True)
    exponent = (bits >> _FP32_FRACTION_BITS) - _FP32_EXPONENT_BIAS - _E4M3_FRACTION_BITS
    unit = ((exponent + _FP32_EXPONENT_BIAS) << _FP32_FRACTION_BITS).to(tl.float32, bitcast=True)
    inverse_unit = ((_FP32_EXPONENT_BIAS - exponent) << _FP32_FRACTION_BITS).to(tl.float32, bitcast=True)
    rounded = tl.extra.cann.libdevice.nearbyint(magnitude * inverse_unit)
    sign = tl.where(value > 0, 1.0, tl.where(value < 0, -1.0, 0.0))
    tl.store(output + row * width + lane, rounded * unit * sign * scale)


def encode_indexpool_fp8_fused(vectors):
    output = torch.empty_like(vectors, dtype=torch.float32)
    _indexpool_encode_fused[(vectors.numel() // INDEX_DIM,)](
        vectors.contiguous(),
        output,
        INDEX_DIM,
        INDEX_DIM.bit_length() - 1,
        FP8_MAX,
        _MIN_AMAX,
        num_warps=1,
        enable_fp_fusion=False,
    )
    return output


@triton.jit
def _pool_norm_kernel(projected, ape, gamma, output, ratio: tl.constexpr, width: tl.constexpr, epsilon: tl.constexpr):
    group = tl.program_id(0)
    token = tl.arange(0, ratio)
    col = tl.arange(0, width)
    offset = (group * ratio + token[:, None]) * (2 * width) + col[None, :]
    value = tl.load(projected + offset)
    gate = tl.load(projected + offset + width) + tl.load(ape + token[:, None] * width + col[None, :])
    probability = tl.exp(gate - tl.max(gate, 0)[None, :])
    probability = probability / tl.sum(probability, 0)[None, :]
    pooled = tl.sum(value * probability, 0)
    inverse = tl.rsqrt(tl.sum(pooled * pooled, 0) / width + epsilon)
    gain = tl.load(gamma + col)
    tl.store(output + group * width + col, (pooled * inverse) * gain)


def pool_norm(raw, projection, ape, gamma, epsilon):
    groups, ratio, width = raw.shape
    projected = torch.nn.functional.linear(raw.flatten(0, 1).float(), projection)
    output = torch.empty((groups, width), device=raw.device, dtype=raw.dtype)
    _pool_norm_kernel[(groups,)](
        projected,
        ape,
        gamma,
        output,
        ratio,
        width,
        epsilon,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output


_QUERIES = 6
_GROUP_TILE = 128
_GROUP_MAJOR_TILE = 64
_GROUP_MAJOR_HEADS = 16
_GROUP_MAJOR_INDEX_DIM = 128
_GROUP_MAJOR_RATIO = 4
_GROUP_MAJOR_BUDGET = 511


@triton.jit
def _score_query_block(
    query,
    head_weight,
    visible,
    scores,
    key,
    group,
    QUERY_START: tl.constexpr,
    QUERY_BLOCK: tl.constexpr,
    MAX_GROUPS: tl.constexpr,
    WIDTH: tl.constexpr,
    HEADS: tl.constexpr,
    RATIO: tl.constexpr,
    BUDGET: tl.constexpr,
    GROUP_TILE: tl.constexpr,
    GROUP_MAJOR: tl.constexpr,
):
    qid = QUERY_START + tl.arange(0, QUERY_BLOCK)
    q_head = QUERY_START * HEADS + tl.arange(0, QUERY_BLOCK * HEADS)
    lane = tl.arange(0, WIDTH)
    # IndexPool encodes E4M3 values with power-of-two row scales. The
    # reconstructed query values, like the pooled keys, fit in BF16.
    q = tl.load(query + q_head[:, None] * WIDTH + lane[None, :]).to(tl.bfloat16)
    weight = tl.load(head_weight + q_head)
    if GROUP_MAJOR:
        # Keep the head axis contiguous through the vector epilogue. Only the
        # reduced [group, query] tensor is transposed for query-major stores.
        qk = tl.dot(key, tl.trans(q), out_dtype=tl.float32)
        weighted = tl.maximum(qk, 0.0) * weight[None, :]
        score = tl.trans(tl.sum(weighted.reshape((GROUP_TILE, QUERY_BLOCK, HEADS)), axis=2))
    else:
        qk = tl.dot(q, tl.trans(key), out_dtype=tl.float32)
        weighted = tl.maximum(qk, 0.0) * weight[:, None]
        score = tl.sum(weighted.reshape((QUERY_BLOCK, HEADS, GROUP_TILE)), axis=1)
    end_group = tl.load(visible + qid) // RATIO
    recall = (end_group[:, None] > BUDGET) & (group[None, :] < end_group[:, None])
    tl.store(
        scores + qid[:, None] * MAX_GROUPS + group[None, :],
        tl.where(recall, score, float("-inf")),
        group[None, :] < MAX_GROUPS,
    )


@triton.jit
def _score_paged_pool(
    query,
    head_weight,
    cache,
    block_table,
    visible,
    scores,
    MAX_GROUPS: tl.constexpr,
    WIDTH: tl.constexpr,
    HEADS: tl.constexpr,
    ROW_WIDTH: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    RATIO: tl.constexpr,
    BUDGET: tl.constexpr,
    GROUP_TILE: tl.constexpr,
    QUERIES: tl.constexpr,
    QUERY_TILE: tl.constexpr,
    GROUP_MAJOR: tl.constexpr,
):
    start = tl.program_id(0) * GROUP_TILE
    group = start + tl.arange(0, GROUP_TILE)
    qid = tl.arange(0, QUERY_TILE)
    end_group = tl.load(visible + qid, qid < QUERIES, -1) // RATIO
    recall = (end_group > BUDGET) & (qid < QUERIES)
    max_visible = tl.max(tl.where(recall, end_group, 0), axis=0)
    if start < max_visible:
        lane = tl.arange(0, WIDTH)
        live = (group < max_visible) & (group < MAX_GROUPS)
        address = pool_key_addresses(block_table, group, lane, live, ROW_WIDTH, PAGE_SIZE, RATIO, WIDTH // RATIO)
        key = tl.load(cache + address, live[:, None], 0)
        # Both layouts share one key tile across the same 4+2 query split.
        _score_query_block(
            query,
            head_weight,
            visible,
            scores,
            key,
            group,
            0,
            4,
            MAX_GROUPS,
            WIDTH,
            HEADS,
            RATIO,
            BUDGET,
            GROUP_TILE,
            GROUP_MAJOR,
        )
        _score_query_block(
            query,
            head_weight,
            visible,
            scores,
            key,
            group,
            4,
            2,
            MAX_GROUPS,
            WIDTH,
            HEADS,
            RATIO,
            BUDGET,
            GROUP_TILE,
            GROUP_MAJOR,
        )
    else:
        tl.store(
            scores + qid[:, None] * MAX_GROUPS + group[None, :],
            float("-inf"),
            (qid[:, None] < QUERIES) & (group[None, :] < MAX_GROUPS),
        )


def write_pool_scores(query, weights, cache, table, visible, page_size, max_groups, ratio, budget, scores):
    """Write one request's scores into a contiguous FP32 destination."""
    cache = cache.view(torch.bfloat16)
    # Specialize the measured six-query/16-head path. Other shapes retain the
    # original query-major layout and 128-group tile. The smaller group-major
    # tile limits live key/accumulator storage without padding extra queries.
    group_major = (
        query.device.type == "npu"
        and query.shape == (_QUERIES, _GROUP_MAJOR_HEADS, INDEX_DIM)
        and weights.shape == (_QUERIES, _GROUP_MAJOR_HEADS)
        and visible.shape == (_QUERIES,)
        and query.is_contiguous()
        and weights.is_contiguous()
        and visible.is_contiguous()
        and scores.is_contiguous()
        and INDEX_DIM == _GROUP_MAJOR_INDEX_DIM
        and ratio == _GROUP_MAJOR_RATIO
        and budget == _GROUP_MAJOR_BUDGET
    )
    group_tile = _GROUP_MAJOR_TILE if group_major else _GROUP_TILE
    _score_paged_pool[(triton.cdiv(max_groups, group_tile),)](
        query,
        weights,
        cache,
        table,
        visible,
        scores,
        max_groups,
        INDEX_DIM,
        query.shape[1],
        cache.shape[-1],
        page_size,
        ratio,
        budget,
        group_tile,
        _QUERIES,
        triton.next_power_of_2(_QUERIES),
        group_major,
        num_warps=1,
        enable_fp_fusion=False,
        multibuffer=False,
    )


def select_pool_candidates(scores, visible, max_groups, ratio, budget):
    cutoff = visible // ratio
    offsets = torch.arange(ratio, device=visible.device)
    values, selected = scores.topk(budget, dim=-1)
    selected = selected.masked_fill(values == -torch.inf, max_groups).sort(dim=-1).values.unsqueeze(-1)
    history = selected * ratio + offsets
    history = history.masked_fill(selected >= cutoff[:, None, None], -1).flatten(1)
    tail = cutoff[:, None] * ratio + offsets
    sparse = torch.cat((history, tail.masked_fill(tail > visible[:, None], -1)), dim=-1)
    # Short rows have no sparse scores, but still need their complete history.
    dense = torch.arange(sparse.shape[-1], device=visible.device).expand_as(sparse)
    dense = dense.masked_fill(dense > visible[:, None], -1)
    return torch.where((cutoff > budget)[:, None], sparse, dense)


WIDTH = 128
RATIO = 4
BUDGET = 511
COPY_GROUPS = 16


@triton.jit
def _refresh_pages(
    cache,
    pages,
    table,
    END: tl.constexpr,
    ROW_WIDTH: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    COPY_GROUPS: tl.constexpr = 16,
    WIDTH: tl.constexpr = 128,
    RATIO: tl.constexpr = 4,
):
    group = tl.program_id(0) * COPY_GROUPS + tl.arange(0, COPY_GROUPS)
    lane = tl.arange(0, WIDTH)
    live = group < END
    address = pool_key_addresses(table, group, lane, live, ROW_WIDTH, PAGE_SIZE, RATIO, WIDTH // RATIO)
    block = tl.load(table + group // (PAGE_SIZE // RATIO), live, 0).to(tl.int64)
    row = block * (PAGE_SIZE // RATIO) + group % (PAGE_SIZE // RATIO)
    key = tl.load(cache + address, live[:, None], 0)
    tl.store(pages + row[:, None] * WIDTH + lane[None, :], key, live[:, None])


@triton.jit
def _write_pages(
    keys,
    pages,
    table,
    first,
    end,
    COUNT: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    COPY_GROUPS: tl.constexpr = 16,
    WIDTH: tl.constexpr = 128,
    RATIO: tl.constexpr = 4,
):
    row = tl.arange(0, COPY_GROUPS)
    group = tl.load(first) + row
    live = (row < COUNT) & (group < tl.load(end))
    lane = tl.arange(0, WIDTH)
    block = tl.load(table + group // (PAGE_SIZE // RATIO), live, 0).to(tl.int64)
    target = block * (PAGE_SIZE // RATIO) + group % (PAGE_SIZE // RATIO)
    value = tl.load(keys + row[:, None] * WIDTH + lane[None, :], live[:, None], 0)
    tl.store(pages + target[:, None] * WIDTH + lane[None, :], value, live[:, None])


def refresh_pages(cache, pages, table, block_size, sequence_length):
    groups = sequence_length // RATIO
    if groups:
        _refresh_pages[(triton.cdiv(groups, COPY_GROUPS),)](
            cache.view(torch.bfloat16), pages, table, groups, cache.shape[-1] // 2, block_size, num_warps=1
        )


def write_pages(keys, pages, table, first, end, block_size):
    _write_pages[(1,)](keys, pages, table, first, end, keys.shape[0], block_size, num_warps=1)


def select_pages(query, weights, pages, table, visible, max_groups):
    import cann_ops_transformer  # noqa: F401 - register native operators lazily

    batch, queries = visible.shape
    lengths = visible[:, -1].clamp_min(0).div(RATIO, rounding_mode="floor").to(torch.int32)
    residual = visible[:, -1].clamp_min(0).remainder(RATIO).to(torch.int32)
    lengths = lengths.clamp_min(BUDGET + 1)
    metadata = torch.ops.cann_ops_transformer.lightning_indexer_metadata(
        16,
        1,
        WIDTH,
        BUDGET,
        seqused_k=lengths,
        cmp_residual_k=residual,
        batch_size=batch,
        max_seqlen_q=queries,
        max_seqlen_k=max_groups,
        layout_q="BSND",
        layout_k="PA_BBND",
        mask_mode=3,
        cmp_ratio=RATIO,
    )
    groups, _ = torch.ops.cann_ops_transformer.lightning_indexer(
        query.to(torch.bfloat16).view(batch, queries, 16, WIDTH),
        pages,
        weights.view(batch, queries, 16),
        BUDGET,
        seqused_k=lengths,
        cmp_residual_k=residual,
        block_table=table.contiguous(),
        metadata=metadata,
        max_seqlen_q=queries,
        layout_q="BSND",
        layout_k="PA_BBND",
        mask_mode=3,
        cmp_ratio=RATIO,
        return_value=0,
    )
    groups = groups.view(batch, queries, BUDGET)
    cutoff = visible // RATIO
    ordered = groups.masked_fill((groups < 0) | (groups >= cutoff[..., None]), max_groups).sort(dim=-1).values
    offsets = torch.arange(RATIO, device=query.device)
    history = ordered[..., None] * RATIO + offsets
    history.masked_fill_(ordered[..., None] == max_groups, -1)
    tail = cutoff[..., None] * RATIO + offsets
    tail.masked_fill_(tail > visible[..., None], -1)
    sparse = torch.cat((history.flatten(-2), tail), dim=-1)
    # Keep the latest PR's dense short-context and padding contract.
    dense = torch.arange((BUDGET + 1) * RATIO, device=query.device).expand(batch, queries, -1)
    dense = dense.masked_fill(dense > visible[..., None], -1)
    return torch.where((cutoff > BUDGET)[..., None], sparse, dense)


def expand_blocks(groups, offsets, ratio):
    return torch.add(offsets, groups[..., None], alpha=ratio)


def current_tail(positions, offsets, ratio):
    positions = positions.to(torch.int32)
    tail = expand_blocks(positions // ratio, offsets, ratio)
    return tail.masked_fill_(tail > positions[:, None], -1)


def merge_group_rows(groups, sentinel, offsets, tail, ratio, *, consume_groups=False):
    use_fused = (
        HAS_TRITON
        and groups.device.type == "npu"
        and is_950()
        and groups.dtype == tail.dtype == torch.int32
        and groups.ndim == 2
        and groups.shape[1] == 511
        and (16 <= groups.shape[0] <= 512 or (consume_groups and 768 <= groups.shape[0] <= 4096))
        and groups.is_contiguous()
        and tail.device == groups.device
        and tail.shape == (groups.shape[0], 4)
        and tail.is_contiguous()
        and ratio == 4
    )
    use_float_sort = (
        groups.device.type == "npu"
        and is_950()
        and groups.dtype == torch.int32
        and groups.ndim == 2
        and groups.shape[1] == 511
        and groups.shape[0] >= 1872
        and 0 <= sentinel <= 2**24
    )
    if use_float_sort:
        # Legal LI group IDs are below sentinel and remain exact in FP32.
        if HAS_TRITON and groups.is_contiguous():
            from .prefill_ops.indexer import mask_group_ids

            sortable = mask_group_ids(groups, sentinel, torch.float32)
        else:
            sortable = groups.masked_fill(groups < 0, sentinel).float()
        ordered = sortable.sort(dim=-1).values
    elif consume_groups and groups.device.type == "npu" and is_950() and 768 <= groups.shape[0] < 1872:
        # LI returns private group IDs. The out-of-place mask first clones them.
        ordered = groups.masked_fill_(groups < 0, sentinel).sort(dim=-1).values
    else:
        ordered = groups.masked_fill(groups < 0, sentinel).sort(dim=-1).values
    if use_fused:
        from .prefill_ops.indexer import expand_sorted_group_ids

        return expand_sorted_group_ids(ordered, sentinel, tail)
    history = expand_blocks(ordered.to(groups.dtype), offsets, ratio)
    history.masked_fill_(ordered[..., None] == sentinel, -1)
    return torch.cat((history.flatten(1), tail), dim=-1)


BATCH_ROWS = 4096
SCORE_ROWS = 64
WIDE_SCORE_ROWS = 512


HEADS = 16
DIM = 128
LAYOUT = "BSND"


def eligible(query, weights, pool, visible_len, chunk, ratio, budget):
    return (
        ratio == RATIO
        and budget == BUDGET
        and chunk == 16
        and visible_len - chunk >= (BUDGET + 1) * RATIO
        and tuple(query.shape) == (chunk, HEADS, DIM)
        and weights.dtype == torch.float32
        and weights.shape == (chunk, HEADS)
        and pool.ndim == 2
        and pool.shape[1] == DIM
        and 0 < (visible_len - 1) // RATIO <= pool.shape[0]
    )


@lru_cache(maxsize=2048)
def metadata_for_shape(device, stream_handle, query_rows, used_k, remainder):
    import cann_ops_transformer  # noqa: F401 - registers the paired ACLNN V2 operators

    scalars = torch.tensor([used_k, remainder], dtype=torch.int32, device=device)
    lens, residual = scalars[:1], scalars[1:]
    offsets = offsets_for_stream(device, stream_handle)
    metadata = torch.ops.cann_ops_transformer.lightning_indexer_metadata(
        HEADS,
        1,
        DIM,
        BUDGET,
        seqused_k=lens,
        cmp_residual_k=residual,
        batch_size=1,
        max_seqlen_q=query_rows,
        max_seqlen_k=used_k,
        layout_q=LAYOUT,
        layout_k=LAYOUT,
        mask_mode=3,
        cmp_ratio=RATIO,
    )
    return lens, residual, metadata, offsets


@lru_cache(maxsize=32)
def offsets_for_stream(device, stream_handle):
    return torch.arange(RATIO, dtype=torch.int32, device=device)


def prepare_tail(positions):
    if (
        HAS_TRITON
        and is_950()
        and positions.numel()
        and positions.device.type == "npu"
        and positions.ndim == 1
        and positions.is_contiguous()
        and positions.dtype in (torch.int32, torch.int64)
    ):
        from .prefill_ops.indexer import build_li_tail

        return build_li_tail(positions)
    stream = torch.npu.current_stream(positions.device)
    offsets = offsets_for_stream(positions.device, stream.npu_stream)
    return current_tail(positions, offsets, RATIO)


def _score_chunk(query_bf16, weights, pool_bf16, visible_len, stream):
    query_rows = query_bf16.shape[0]
    used_k, remainder = divmod(visible_len - 1, RATIO)
    lens, residual, metadata, offsets = metadata_for_shape(
        query_bf16.device, stream.npu_stream, query_rows, used_k, remainder
    )
    group_ids, _ = torch.ops.cann_ops_transformer.lightning_indexer(
        query_bf16.unsqueeze(0),
        pool_bf16[:used_k].view(1, used_k, 1, DIM),
        weights.contiguous().unsqueeze(0),
        BUDGET,
        seqused_k=lens,
        cmp_residual_k=residual,
        metadata=metadata,
        max_seqlen_q=query_rows,
        layout_q=LAYOUT,
        layout_k=LAYOUT,
        mask_mode=3,
        cmp_ratio=RATIO,
        return_value=0,
    )
    return group_ids.view(query_rows, BUDGET), used_k, offsets


def select_many(query, weights, pool_bf16, first_visible, *, tail, tp_shard=False):
    """Score Prefill rows, with TP sharding enabled by the caller for all-Prefill batches."""
    stream = torch.npu.current_stream(query.device)
    query_bf16 = query.to(torch.bfloat16).contiguous()
    weights = weights.contiguous()
    query_rows = query_bf16.shape[0]
    minimum_keys = (first_visible - 1) // RATIO
    if tp_shard and query_rows >= 2048 and query_rows % 64 == 0 and minimum_keys >= 8192:
        tp = get_tp_group()
        if tp.world_size == 4 and tp.device_group is not None:
            shard_rows = query_rows // tp.world_size
            lo = shard_rows * tp.rank_in_group
            hi = lo + shard_rows
            local_ids, _, offsets = _score_chunk(
                query_bf16[lo:hi], weights[lo:hi], pool_bf16, first_visible + hi - 16, stream
            )
            group_ids = torch.empty((query_rows, BUDGET), dtype=torch.int32, device=query.device)
            dist.all_gather_into_tensor(group_ids, local_ids.contiguous(), group=tp.device_group)
            last_visible = first_visible + query_rows - 16
            used_k = (last_visible - 1) // RATIO
            return merge_group_rows(group_ids, used_k, offsets, tail, RATIO, consume_groups=True)
    chunks = []
    lo = 0
    while lo < query.shape[0]:
        remaining = query.shape[0] - lo
        if remaining >= 2 * WIDE_SCORE_ROWS:
            # This range already satisfies the full-group causal mask. Reuse
            # the wide CANN launch for short histories as well.
            rows = remaining // WIDE_SCORE_ROWS * WIDE_SCORE_ROWS
        elif remaining >= WIDE_SCORE_ROWS:
            rows = WIDE_SCORE_ROWS
        else:
            rows = SCORE_ROWS if remaining >= SCORE_ROWS else 16
        hi = lo + rows
        # The right-down causal mask aligns the block's last query to its visible length.
        visible = first_visible + hi - 16
        groups, used_k, offsets = _score_chunk(query_bf16[lo:hi], weights[lo:hi], pool_bf16, visible, stream)
        chunks.append(groups)
        lo = hi
    # Each legal ID is smaller than its chunk's used_k, hence also smaller than the last used_k.
    group_ids = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0)
    return merge_group_rows(group_ids, used_k, offsets, tail, RATIO, consume_groups=True)


def select(query, weights, pool_bf16, visible_len, *, tail):
    group_ids, used_k, offsets = _score_chunk(
        query.to(torch.bfloat16).contiguous(),
        weights,
        pool_bf16,
        visible_len,
        torch.npu.current_stream(query.device),
    )
    return merge_group_rows(group_ids, used_k, offsets, tail, RATIO, consume_groups=True)
