"""A3 grouped Cube decode for a batch with a known uniform physical query length."""

from functools import lru_cache

import torch
from vllm.triton_utils import tl, triton

from .flash_mla_a3 import _LIBRARY_DIR


@lru_cache(maxsize=1)
def _kernel():
    # Register the NPU dispatcher and current stream before loading the operator.
    import torch_npu  # noqa: F401

    library = _LIBRARY_DIR / "libflash_mla_a3_grouped.so"
    if not library.is_file():
        raise RuntimeError(f"Missing packaged grouped FlashMLA library {library}")
    torch.ops.load_library(str(library))
    return torch.ops.flash_mla_a3_grouped.run


@triton.jit
def _combine(
    PARTIAL,
    STATS,
    OUT,
    OUT_STATS,
    B: tl.constexpr,
    S: tl.constexpr,
    SPLITS: tl.constexpr,
    GROUPS: tl.constexpr,
    TILE: tl.constexpr,
    DIM: tl.constexpr,
    NATIVE_INPUT: tl.constexpr,
    FINAL: tl.constexpr,
    NTD: tl.constexpr,
):
    # A row is one query position and one head. Intermediate buffers are
    # row-major so a second aggregation reads adjacent numerator blocks.
    job = tl.program_id(0)
    row_job = job // GROUPS
    block = job % GROUPS
    batch = row_job // (S * 4)
    row = row_job % (S * 4)
    split = block * TILE + tl.arange(0, TILE)
    columns = tl.program_id(1) * DIM + tl.arange(0, DIM)
    if NATIVE_INPUT:
        slot = batch * SPLITS + split
        item = slot.to(tl.int64) * 16 + row
        maximum_ptr = STATS + slot.to(tl.int64) * 32 + row
        total_ptr = maximum_ptr + 16
    else:
        item = row_job.to(tl.int64) * SPLITS + split
        maximum_ptr = STATS + item * 8
        total_ptr = maximum_ptr + 1

    maximum = tl.load(maximum_ptr, split < SPLITS, other=-float("inf"))
    total = tl.load(total_ptr, split < SPLITS, other=0.0)
    largest = tl.max(maximum, 0)
    factor = tl.exp(maximum - tl.where(largest == -float("inf"), 0.0, largest))
    denominator = tl.sum(total * factor, 0)
    invalid = tl.sum((total < 0.0).to(tl.int32), 0) > 0
    value = tl.load(PARTIAL + item[:, None] * 512 + columns[None, :], split[:, None] < SPLITS, other=0.0)
    numerator = tl.sum(value * factor[:, None], 0)
    if FINAL:
        answer = numerator / tl.where(denominator > 0.0, denominator, 1.0)
        dst = row % 4 * (B * S) + row_job // 4 if NTD else row_job
        tl.store(OUT + dst.to(tl.int64) * 512 + columns, tl.where((denominator > 0.0) & (~invalid), answer, 0.0))
    else:
        tl.store(OUT + job.to(tl.int64) * 512 + columns, numerator)
        if tl.program_id(1) == 0:
            # Each writer owns one complete, aligned 32-byte record.
            lane = tl.arange(0, 8)
            record = tl.where(lane == 0, largest, tl.where(lane == 1, tl.where(invalid, -1.0, denominator), 0.0))
            tl.store(OUT_STATS + job.to(tl.int64) * 8 + lane, record)


@triton.jit
def _finish_heads(
    PARTIAL,
    STATS,
    MAP,
    OUT,
    T: tl.constexpr,
    S: tl.constexpr,
    GROUPS: tl.constexpr,
    TILE: tl.constexpr,
    HEADS: tl.constexpr,
    NTD: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1) * HEADS + tl.arange(0, HEADS)
    column = tl.arange(0, 512)
    tile = token // S
    if S > 1:
        mapped = tl.load(MAP + tile * 8 + token % S)
        slot, row = mapped // 16, mapped % 16
    else:
        slot, row = tile, 0

    split = tl.arange(0, TILE)
    base = (slot * GROUPS + split).to(tl.int64)
    stats = STATS + base[:, None] * 32 + row + head[None, :]
    maximum = tl.load(stats, split[:, None] < GROUPS, other=-float("inf"))
    total = tl.load(stats + 16, split[:, None] < GROUPS, other=0.0)
    peak = tl.max(maximum, 0)
    factor = tl.exp(maximum - tl.where(peak == -float("inf"), 0.0, peak)[None, :])
    denominator = tl.sum(total * factor, 0)
    valid = (denominator > 0.0) & (tl.sum((total < 0.0).to(tl.int32), 0) == 0)

    address = (base[:, None] * 16 + row + head[None, :]) * 512
    partial = tl.load(PARTIAL + address[:, :, None] + column[None, None, :], split[:, None, None] < GROUPS, other=0.0)
    numerator = tl.sum(partial * factor[:, :, None], 0)
    value = numerator / tl.where(valid, denominator, 1.0)[:, None]
    dest = head * T + token if NTD else token * 4 + head
    tl.store(OUT + dest[:, None].to(tl.int64) * 512 + column[None, :], tl.where(valid[:, None], value, 0.0))


def group_decode(
    q,
    cache3,
    table,
    lengths,
    query_length,
    scale,
    causal,
    ntd,
    *,
    plan=None,
    cu_seqlens_q=None,
    seqused_q=None,
    packed=False,
    ragged=False,
):
    """Return BF16 paged MLA for uniformly packed queries, in TND or NTD order.

    Q is [B*S,4,576], cache3 is [P,128,576], and each row of the INT32
    block table maps a sequence's logical pages to its physical cache pages.
    lengths is an INT32 NPU tensor of live KV lengths. S must be in [1,4].
    B=1 also accepts GPU-resident packed offsets, effective Q length, or a
    metadata plan. For multi-batch ragged decode, S is the physical tile size,
    T=B*S, and the device-resident offsets determine each request's true length.
    The kernel reads these tensors on each invocation and graph replay.
    """
    if q.ndim != 3 or table.ndim != 2 or not 1 <= query_length <= 4:
        raise ValueError("Expected a [B*S,4,576] query, a [B,W] table and query_length in [1,4]")
    batch, width = table.shape
    if batch <= 0 or width <= 0 or q.shape != (batch * query_length, 4, 576):
        raise ValueError("The query shape must match the batch and its uniform query_length")

    from .flash_mla_a3 import _plain

    q, cache3, table, lengths = (_plain(x) for x in (q, cache3, table, lengths))
    # One page per AIC maximizes parallelism for a small grid. The 2/3-page
    # variants share one softmax maximum and keep partial PV sums in L0C.
    pages_per_group = 1 if batch * width <= 16 else (2 if batch * width <= 48 else 3)
    pages_per_group = min(width, pages_per_group)
    groups = triton.cdiv(width, pages_per_group)
    if ragged and groups > 16:
        raise ValueError("Multi-batch ragged grouped decode supports at most 16 page groups")
    slots = batch * query_length if ragged else batch
    mapping = lengths.new_empty((batch, 8)) if ragged else None
    sizes = (batch, width, 16, 128)
    logits = torch.empty(sizes, dtype=torch.float32, device=q.device)
    weights = q.new_empty(sizes)
    partial = logits.new_empty((slots, groups, 16, 512))
    stats = logits.new_empty((slots, groups, 2, 16))
    _kernel()(
        q,
        cache3,
        table,
        lengths,
        logits,
        weights,
        partial,
        stats,
        query_length,
        float(scale),
        bool(causal),
        pages_per_group,
        plan,
        cu_seqlens_q,
        seqused_q,
        bool(packed),
        mapping,
    )

    rows = batch * query_length * 4
    out_shape = (4, batch * query_length, 512) if ntd else (batch * query_length, 4, 512)
    out = q.new_empty(out_shape)
    if ragged:
        heads = 4 if groups <= 8 else 2
        tokens = batch * query_length
        _finish_heads[(tokens, 4 // heads)](
            partial, stats, mapping, out, tokens, query_length, groups, triton.next_power_of_2(groups), heads, bool(ntd)
        )
        return out
    native_input = True
    while groups > 32:
        count = triton.cdiv(groups, 16)
        next_partial = partial.new_empty((rows, count, 512))
        next_stats = stats.new_empty((rows, count, 8))
        _combine[(rows * count, 1)](
            partial,
            stats,
            next_partial,
            next_stats,
            batch,
            query_length,
            groups,
            count,
            16,
            512,
            native_input,
            False,
            False,
        )
        partial, stats, groups = next_partial, next_stats, count
        native_input = False

    tile = triton.next_power_of_2(groups)
    dim = 512 if groups <= 16 else 256
    _combine[(rows, 512 // dim)](
        partial, stats, out, stats, batch, query_length, groups, 1, tile, dim, native_input, True, bool(ntd)
    )
    return out
