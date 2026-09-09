# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Triton counterpart of the AscendC categorical sampler.

This is an explicit, opt-in operator entry point; importing it does not change
the model runner's backend. The RNG and fixed-point mode follow the native
operator, not the Gumbel-max implementation in worker/v2/sample/gumbel.py.
"""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _philox(seed, position):
    c0 = position.to(tl.uint32)
    c1 = (position.to(tl.uint64) >> 32).to(tl.uint32)
    c2 = tl.full((), 0, tl.uint32)
    c3 = tl.full((), 0, tl.uint32)
    k0 = seed.to(tl.uint32)
    k1 = (seed.to(tl.uint64) >> 32).to(tl.uint32)
    for _ in tl.static_range(10):
        hi0 = tl.umulhi(c0, 0xD2511F53)
        hi1 = tl.umulhi(c2, 0xCD9E8D57)
        lo0 = c0 * 0xD2511F53
        lo1 = c2 * 0xCD9E8D57
        c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
        k0 += 0x9E3779B9
        k1 += 0xBB67AE85
    return c0, c1


@triton.jit
def _mul_high64(lhs, rhs):
    lhs_low = lhs & 0xFFFFFFFF
    rhs_low = rhs & 0xFFFFFFFF
    lhs_high = lhs >> 32
    rhs_high = rhs >> 32
    low_product = lhs_low * rhs_low
    middle = lhs_high * rhs_low + (low_product >> 32)
    upper_middle = (middle & 0xFFFFFFFF) + lhs_low * rhs_high
    return lhs_high * rhs_high + (middle >> 32) + (upper_middle >> 32)


@triton.jit
def _fixed_mass_words(weight):
    # exp(logit - max) is in [0, 1]. Scale its exact FP32 significand
    # by 2^42 and round half up, as FloatToFixedMass does in AscendC.
    bits = weight.to(tl.uint32, bitcast=True)
    exponent = (bits >> 23) & 255
    mantissa = (bits & 0x7FFFFF) | tl.where(exponent == 0, 0, 0x800000)
    shift = tl.where(exponent == 0, -107, exponent.to(tl.int32) - 108)
    right = tl.minimum(tl.maximum(-shift, 1), 24)
    rounded = (mantissa + (1 << (right - 1))) >> right
    low = tl.where(shift >= 0, mantissa << tl.maximum(shift, 0), rounded)
    low = tl.where((weight > 0) & (shift >= -24), low, 0).to(tl.uint32)
    high = tl.where(shift > 0, mantissa >> tl.minimum(32 - shift, 31), 0).to(tl.uint32)
    return low, high


@triton.jit
def _fixed_mass(weight):
    low, high = _fixed_mass_words(weight)
    return low.to(tl.uint64) | (high.to(tl.uint64) << 32)


@triton.jit
def _element(values, index):
    return tl.sum(tl.gather(values, tl.full((1,), index, tl.int32), axis=0), axis=0)


@triton.jit
def _load_logits(logits, row_offset, offsets, vocab, temperature, APPLY: tl.constexpr):
    values = tl.load(logits + row_offset + offsets, offsets < vocab, other=-float("inf")).to(tl.float32)
    if logits.dtype.element_ty == tl.float32:
        # Native FP32 LoadTile uses Adds(0); preserve its signed-zero behavior.
        values = values + 0.0
    if APPLY:
        if temperature != 0:
            values = values / temperature
    return values


@triton.jit
def _categorical_kernel(
    logits,
    mapping,
    temperatures,
    seeds,
    positions,
    cache,
    cache_col,
    output,
    lse,
    row_stride: tl.constexpr,
    cache_stride: tl.constexpr,
    col_stride: tl.constexpr,
    VOCAB: tl.constexpr,
    REQUESTS: tl.constexpr,
    COLS: tl.constexpr,
    PER_ROW_COL: tl.constexpr,
    APPLY: tl.constexpr,
    RETURN_LSE: tl.constexpr,
    FP64: tl.constexpr,
    TILE: tl.constexpr,
    BLOCK: tl.constexpr,
    TILES: tl.constexpr,
    SUM_BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    request = tl.load(mapping + row)
    if request == -1:
        tl.store(output + row, 0)
        if RETURN_LSE:
            tl.store(lse + row, 0.0)
        return
    # Keep assertion machinery off the valid path: unconditional scalar
    # device_assert calls break ACLGraph/native-op interleaving on C220.
    if (request < 0) | (request >= REQUESTS):
        tl.device_assert(False, "CategoricalSample expanded index mapping is outside request state")
    temperature = tl.load(temperatures + request)
    column = tl.full((), 0, tl.int32)
    if cache is not None:
        if cache_col is not None:
            column = tl.load(cache_col + tl.where(PER_ROW_COL, row, 0))
        if (column < 0) | (column >= COLS):
            tl.device_assert(False, "CategoricalSample output processed logits column is outside cache bounds")
    offsets = tl.arange(0, BLOCK)
    row_offset = row * row_stride
    row_max = tl.full((), -float("inf"), tl.float32)
    first_max = tl.full((), 0, tl.int32)
    # Validate the entire row before writing any raw-logit cache entries.
    for tile in range(TILES):
        indices = tile * TILE + offsets
        values = _load_logits(logits, row_offset, indices, VOCAB, temperature, APPLY)
        valid = (offsets < TILE) & (indices < VOCAB)
        if tl.sum((valid & (values != values)).to(tl.int32), 0) != 0:
            tl.device_assert(False, "CategoricalSample processed logits must not contain NaN")
        values = tl.where(valid, values, -float("inf"))
        maximum = tl.max(values, 0)
        index = tl.min(tl.where(valid & (values == maximum), indices, VOCAB), 0)
        first_max = tl.where(maximum > row_max, index, first_max)
        row_max = tl.maximum(row_max, maximum)
    if row_max == -float("inf"):
        tl.device_assert(False, "CategoricalSample processed logits row must not be all -inf")

    total = tl.full((), 0.0, tl.float32)
    total_mass = tl.full((), 0, tl.uint64)
    infinity_count = tl.full((), 0, tl.int32)
    sums = tl.full((SUM_BLOCK,), 0.0, tl.float32)
    masses_low = tl.full((SUM_BLOCK,), 0, tl.uint32)
    masses_high = tl.full((SUM_BLOCK,), 0, tl.uint32)
    tile_ids = tl.arange(0, SUM_BLOCK)
    for tile in range(TILES):
        indices = tile * TILE + offsets
        valid = (offsets < TILE) & (indices < VOCAB)
        raw = tl.load(logits + row_offset + indices, valid, other=-float("inf")).to(tl.float32)
        if logits.dtype.element_ty == tl.float32:
            raw = raw + 0.0
        if cache is not None:
            tl.store(
                cache + request.to(tl.int64) * cache_stride + column.to(tl.int64) * col_stride + indices, raw, valid
            )
        values = raw
        if APPLY:
            if temperature != 0:
                values = values / temperature
        if row_max == float("inf"):
            infinity_count += tl.sum((valid & (values == float("inf"))).to(tl.int32), 0)
        elif (temperature != 0) | RETURN_LSE:
            weights = tl.where(valid, tl.exp(values - row_max), 0.0)
            if (not FP64) or RETURN_LSE:
                length = tl.minimum(TILE, VOCAB - tile * TILE)
                tile_sum = tl.full((), 0.0, tl.float32)
                if length % 256 != 0:
                    # Native unaligned tails and selected CDFs accumulate in
                    # token order. A parallel cumsum changes FP32 boundaries.
                    for index in range(length):
                        tile_sum += _element(weights, index)
                else:
                    # C220 calcount ReduceSum reduces each 64-float repeat,
                    # then accumulates repeat results in order (get_acc_val).
                    # A single tl.sum uses a different floating-point tree.
                    repeats = tl.sum(tl.reshape(weights, (BLOCK // 64, 64)), 1)
                    for repeat in range(length // 64):
                        tile_sum += _element(repeats, repeat)
                sums = tl.where(tile_ids == tile, tile_sum, sums)
                total += tile_sum
            if FP64:
                # C220 has no vector uint64 select. Sum 16-bit limbs in
                # uint32 (4096 * 65535 fits), then combine scalar uint64s.
                # Bound conversion temporaries without changing native's
                # 4096-element FP32 tiles or their accumulation order.
                tile_mass = tl.full((), 0, tl.uint64)
                for start in range(0, BLOCK, 256):
                    chunk = tl.gather(weights, start + tl.arange(0, 256), axis=0)
                    low, high = _fixed_mass_words(chunk)
                    tile_mass += (
                        tl.sum(low & 0xFFFF, 0).to(tl.uint64)
                        + (tl.sum((low >> 16) & 0xFFFF, 0).to(tl.uint64) << 16)
                        + (tl.sum(high, 0).to(tl.uint64) << 32)
                    )
                masses_low = tl.where(tile_ids == tile, tile_mass.to(tl.uint32), masses_low)
                masses_high = tl.where(tile_ids == tile, (tile_mass >> 32).to(tl.uint32), masses_high)
                total_mass += tile_mass

    if RETURN_LSE:
        tl.store(lse + row, tl.where(row_max == float("inf"), row_max, row_max + tl.log(total)))
    if temperature == 0:
        tl.store(output + row, first_max)
        return

    random0, random1 = _philox(tl.load(seeds + request), tl.load(positions + row))
    uniform = ((random0 >> 8).to(tl.float32) + 0.5) * (1.0 / 16777216.0)
    random64 = (random1.to(tl.uint64) << 32) | random0.to(tl.uint64)
    if row_max == float("inf"):
        if FP64:
            rank = _mul_high64(random64, infinity_count.to(tl.uint64)).to(tl.int32)
        else:
            rank = (uniform * infinity_count.to(tl.float32)).to(tl.int32)
            rank = tl.minimum(rank, infinity_count - 1)
        seen = tl.full((), 0, tl.int32)
        chosen = tl.full((), 0, tl.int32)
        for tile in range(TILES):
            indices = tile * TILE + offsets
            values = _load_logits(logits, row_offset, indices, VOCAB, temperature, APPLY)
            support = (offsets < TILE) & (indices < VOCAB) & (values == float("inf"))
            cumulative = tl.cumsum(support.to(tl.int32), 0) + seen
            candidate = tl.min(tl.where(support & (cumulative == rank + 1), indices, VOCAB), 0)
            chosen = tl.where(candidate < VOCAB, candidate, chosen)
            seen += tl.sum(support.to(tl.int32), 0)
        tl.store(output + row, chosen)
        return

    if FP64:
        target = _mul_high64(random64, total_mass)
        prefix = tl.full((), 0, tl.uint64)
    else:
        target = uniform * total
        prefix = tl.full((), 0.0, tl.float32)
    selected_tile = tl.full((), TILES - 1, tl.int32)
    tile = tl.full((), 0, tl.int32)
    selected = tl.full((), False, tl.int1)
    while (tile < TILES) & ~selected:
        if FP64:
            low = tl.sum(tl.where(tile_ids == tile, masses_low, 0), 0).to(tl.uint64)
            high = tl.sum(tl.where(tile_ids == tile, masses_high, 0), 0).to(tl.uint64)
            next_prefix = prefix + (low | (high << 32))
            selected = target < next_prefix
        else:
            next_prefix = prefix + _element(sums, tile)
            selected = target <= next_prefix
        if selected | (tile + 1 == TILES):
            selected_tile = tile
            selected = tl.full((), True, tl.int1)
        else:
            prefix = next_prefix
        tile += 1
    indices = selected_tile * TILE + offsets
    values = _load_logits(logits, row_offset, indices, VOCAB, temperature, APPLY)
    weights = tl.exp(values - row_max)
    length = tl.minimum(TILE, VOCAB - selected_tile * TILE)
    chosen = selected_tile * TILE
    index = tl.full((), 0, tl.int32)
    selected = tl.full((), False, tl.int1)
    while (index < length) & ~selected:
        weight = _element(weights, index)
        if FP64:
            mass = _fixed_mass(weight)
            chosen = tl.where(mass > 0, selected_tile * TILE + index, chosen)
            prefix += mass
            selected = prefix > target
        else:
            chosen = tl.where(weight > 0, selected_tile * TILE + index, chosen)
            prefix += weight
            selected = prefix >= target
        if selected:
            chosen = selected_tile * TILE + index
        index += 1
    tl.store(output + row, chosen)


def categorical_sample(
    processed_logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    seed: torch.Tensor,
    pos: torch.Tensor,
    return_lse: bool,
    apply_temperature: bool,
    logits_cache: torch.Tensor | None = None,
    logits_cache_col: torch.Tensor | None = None,
    use_fp64: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample with the native categorical interface, without calling native code.

    ``use_fp64`` selects the native 42-fractional-bit integer-mass algorithm,
    not floating-point double precision. FP32 sums preserve the C220 native
    tile/repeat/CDF ordering. Cache entries contain raw, unscaled logits.

    Invalid device values trigger Triton's device assertion; its runtime
    diagnostic text differs from the AscendC assertion. No device values are
    copied to the host for validation, so the call can be captured in ACLGraph.
    """
    logits = processed_logits
    float_dtypes = (torch.float16, torch.bfloat16, torch.float32)
    if logits.device.type != "npu" or logits.ndim != 2 or logits.dtype not in float_dtypes:
        raise RuntimeError("processed_logits must be a 2D NPU float16, bfloat16, or float32 tensor")
    rows, vocab = logits.shape
    if not 0 < rows <= 0xFFFFFFFF or not 0 < vocab <= 1_048_576:
        raise RuntimeError("invalid categorical logits shape")
    if logits.stride(1) != 1 or not (logits.stride(0) == 0 or vocab <= logits.stride(0) <= 0xFFFFFFFF):
        raise RuntimeError("invalid categorical logits strides")
    for name, tensor, dtype in (
        ("expanded_idx_mapping", expanded_idx_mapping, torch.int32),
        ("temperature", temperature, torch.float32),
        ("seed", seed, torch.int64),
        ("pos", pos, torch.int64),
    ):
        if tensor.device != logits.device or tensor.ndim != 1 or tensor.dtype != dtype or not tensor.is_contiguous():
            raise RuntimeError(f"{name} must be contiguous 1D {dtype} on the logits device")
    if expanded_idx_mapping.numel() != rows or pos.numel() != rows:
        raise RuntimeError("mapping and pos must have one entry per row")
    requests = temperature.numel()
    if requests == 0 or seed.numel() != requests:
        raise RuntimeError("seed and temperature must have the same nonzero length")
    cache_stride, col_stride, cols = 0, 0, 1
    if logits_cache is not None:
        cache = logits_cache
        if cache.device != logits.device or cache.ndim not in (2, 3) or cache.dtype not in float_dtypes:
            raise RuntimeError("logits_cache must be 2D or 3D floating point on the logits device")
        if cache.shape[0] != requests or cache.shape[-1] < vocab or cache.stride(-1) != 1:
            raise RuntimeError("invalid logits_cache shape or vocabulary stride")
        cols = cache.shape[1] if cache.ndim == 3 else 1
        col_stride = cache.stride(1) if cache.ndim == 3 else cache.shape[-1]
        cache_stride = cache.stride(0)
        if cols == 0 or not vocab <= col_stride <= 0xFFFFFFFF:
            raise RuntimeError("invalid logits_cache column stride")
        if not (cols - 1) * col_stride + vocab <= cache_stride <= 0xFFFFFFFF:
            raise RuntimeError("invalid logits_cache request stride")
    if logits_cache_col is not None:
        col = logits_cache_col
        if logits_cache is None:
            raise RuntimeError("logits_cache_col requires logits_cache")
        if col.device != logits.device or col.dtype != torch.int32 or not col.is_contiguous():
            raise RuntimeError("logits_cache_col must be contiguous int32 on the logits device")
        if col.ndim != 0 and (col.ndim != 1 or col.numel() != rows):
            raise RuntimeError("logits_cache_col must be scalar or have one entry per row")
    sampled = torch.empty(rows, dtype=torch.int64, device=logits.device)
    lse = torch.empty(rows if return_lse else 0, dtype=torch.float32, device=logits.device)
    tile = min(4096, triton.cdiv(vocab, 256) * 256)
    tiles = triton.cdiv(vocab, tile)
    _categorical_kernel[(rows,)](
        logits,
        expanded_idx_mapping,
        temperature,
        seed,
        pos,
        logits_cache,
        logits_cache_col,
        sampled,
        lse,
        logits.stride(0),
        cache_stride,
        col_stride,
        vocab,
        requests,
        cols,
        logits_cache_col is not None and logits_cache_col.ndim == 1,
        apply_temperature,
        return_lse,
        use_fp64,
        tile,
        triton.next_power_of_2(tile),
        tiles,
        triton.next_power_of_2(tiles),
        enable_fp_fusion=False,
        debug=True,
        multibuffer=False,
    )
    return sampled, lse
