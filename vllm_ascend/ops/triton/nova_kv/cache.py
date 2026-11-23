# SPDX-License-Identifier: Apache-2.0
"""NOVA-KV compressed page views, cache writes and bounded exact windows."""

import torch
from vllm.triton_utils import tl, triton


def planar_cache_views(cache):
    """Reinterpret allocator-owned pages, preserving padding between pages."""
    if cache.dtype != torch.uint8 or cache.ndim != 4 or cache.stride(-1) != 1:
        raise ValueError("NOVA-KV requires a four-dimensional packed byte cache")
    pages, page, heads, packet = cache.shape
    payload = (packet - 6) // 2
    if packet != 2 * payload + 6 or payload != 64 or page % 32:
        raise ValueError("compressed kernel requires dim=256 and page divisible by 32")
    stride = cache.stride(0)
    content = page * heads * packet
    if stride < content or stride % 2 or cache.storage_offset() % 2:
        raise ValueError("cache page stride must fit content and BF16 alignment")
    code_bytes = heads * payload * page
    shape = (pages, heads, payload, page)
    strides = (stride, payload * page, page, 1)
    base = cache.storage_offset()
    key = torch.as_strided(cache, shape, strides, base)
    value = torch.as_strided(cache, shape, strides, base + code_bytes)
    scale_bytes = torch.as_strided(cache, (pages, heads * 6 * page), (stride, 1), base + 2 * code_bytes)
    scales = scale_bytes.view(torch.bfloat16).view(pages, heads, 3, page)
    return key, value, scales


@triton.jit
def _write_planar(
    K,
    V,
    S,
    SLOTS,
    KC,
    VC,
    SC,
    H: tl.constexpr,
    PAYLOAD: tl.constexpr,
    PAGE: tl.constexpr,
    CODE_STRIDE: tl.constexpr,
    SCALE_STRIDE: tl.constexpr,
):
    row, head = tl.program_id(0), tl.program_id(1)
    slot = tl.load(SLOTS + row).to(tl.int64)
    page, token = slot // PAGE, slot % PAGE
    group = tl.arange(0, PAYLOAD)
    src = (row * H + head) * PAYLOAD + group
    dst = page * CODE_STRIDE + head * PAYLOAD * PAGE + group * PAGE + token
    tl.store(KC + dst, tl.load(K + src), slot >= 0)
    tl.store(VC + dst, tl.load(V + src), slot >= 0)
    component = tl.arange(0, 4)
    scale = tl.load(S + (row * H + head) * 3 + component, component < 3, 0)
    tl.store(
        SC + page * SCALE_STRIDE + head * 3 * PAGE + component * PAGE + token, scale, (slot >= 0) & (component < 3)
    )


def write_planar_packets(views, slots, key_codes, value_codes, scales):
    key, value, scale_cache = views
    rows, heads, payload = key_codes.shape
    if not all(x.is_contiguous() for x in (slots, key_codes, value_codes, scales)):
        raise ValueError("packet writer requires contiguous encoded input")
    if value_codes.shape != key_codes.shape or scales.shape != (rows, heads, 3):
        raise ValueError("invalid packet component geometry")
    if slots.numel() != rows or key.shape[1:3] != (heads, payload):
        raise ValueError("slots or cache geometry does not match packets")
    if rows:
        _write_planar[(rows, heads)](
            key_codes,
            value_codes,
            scales,
            slots,
            key,
            value,
            scale_cache,
            heads,
            payload,
            key.shape[-1],
            key.stride(0),
            scale_cache.stride(0),
        )


def merge_attention_segments(cold, cold_lse, exact, exact_lse):
    """Combine normalized segment outputs, including empty/padded requests."""
    maximum = torch.maximum(cold_lse, exact_lse)
    finite = torch.isfinite(maximum)
    safe_maximum = torch.where(finite, maximum, torch.zeros_like(maximum))
    cold_weight = torch.exp(cold_lse - safe_maximum)
    exact_weight = torch.exp(exact_lse - safe_maximum)
    total = (cold_weight + exact_weight).clamp_min(torch.finfo(torch.float32).tiny)
    out = (cold * cold_weight[..., None] + exact * exact_weight[..., None]) / total[..., None]
    lse = torch.where(finite, safe_maximum + total.log(), -torch.inf)
    return out, lse


@triton.jit
def _write_exact_rows(
    K,
    V,
    REQUEST_SLOTS,
    POSITIONS,
    WINDOW_K,
    WINDOW_V,
    H: tl.constexpr,
    D: tl.constexpr,
    SINK: tl.constexpr,
    RING: tl.constexpr,
    CAPACITY: tl.constexpr,
    LIMIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row, head = tl.program_id(0), tl.program_id(1)
    request = tl.load(REQUEST_SLOTS + row)
    position = tl.load(POSITIONS + row)
    offset = tl.where(position < SINK, position, SINK + position % RING)
    dims = tl.arange(0, BLOCK)
    valid = (request >= 0) & (request < LIMIT) & (position >= 0) & (dims < D)
    source = (row * H + head) * D + dims
    destination = ((request * CAPACITY + offset) * H + head) * D + dims
    tl.store(WINDOW_K + destination, tl.load(K + source, dims < D, 0), valid)
    tl.store(WINDOW_V + destination, tl.load(V + source, dims < D, 0), valid)


class ExactWindowPool:
    """Lossless sink and recent rows, bounded independently of context length.

    The caller owns request-slot assignment. Bulk writes must contain unique
    ring destinations per request; seed_chunk trims long chunks accordingly.
    """

    def __init__(self, limit, heads, dim, sink, recent, device, dtype):
        if limit < 1 or sink < 0 or recent < 1:
            raise ValueError("invalid exact-window capacity")
        self.limit, self.sink, self.ring = limit, sink, recent + 1
        self.capacity = sink + self.ring
        shape = (limit + 1, self.capacity, heads, dim)
        self.key = torch.zeros(shape, device=device, dtype=dtype)
        self.value = torch.zeros_like(self.key)
        self.offsets = torch.arange(self.capacity, device=device, dtype=torch.int64)

    @property
    def nbytes(self):
        return (self.key.numel() + self.value.numel()) * self.key.element_size()

    def write_rows(self, key, value, request_slots, positions):
        rows, heads, dim = key.shape
        if value.shape != key.shape or key.shape[1:] != self.key.shape[2:]:
            raise ValueError("exact-window input geometry mismatch")
        if request_slots.numel() != rows or positions.numel() != rows:
            raise ValueError("exact-window row metadata mismatch")
        if not all(x.is_contiguous() for x in (key, value, request_slots, positions)):
            raise ValueError("exact-window writer requires contiguous input")
        if rows:
            _write_exact_rows[(rows, heads)](
                key,
                value,
                request_slots,
                positions,
                self.key,
                self.value,
                heads,
                dim,
                self.sink,
                self.ring,
                self.capacity,
                self.limit,
                triton.next_power_of_2(dim),
            )

    def seed_chunk(self, key, value, request_slot, old_length):
        """Retain at most sink+recent+1 rows from an eager prefill chunk."""
        count = key.shape[0]
        sink_count = min(count, max(0, self.sink - old_length))
        ring_start = max(sink_count, count - self.ring)
        indices = torch.cat(
            (torch.arange(sink_count, device=key.device), torch.arange(ring_start, count, device=key.device))
        )
        slots = torch.full_like(indices, request_slot)
        self.write_rows(
            key.index_select(0, indices).contiguous(),
            value.index_select(0, indices).contiguous(),
            slots,
            indices + old_length,
        )

    def attend(self, query, request_slots, lengths, scale):
        """Return exact-segment output/LSE using only fixed-size tensors."""
        lengths = lengths.to(torch.int64)
        slots = request_slots.to(torch.int64)
        valid_request = (slots >= 0) & (slots < self.limit) & (lengths > 0)
        safe_slots = torch.where(valid_request, slots, self.limit)
        recent_start = (lengths - self.ring).clamp_min(self.sink)
        offsets = self.offsets[None, :]
        positions = torch.where(offsets < self.sink, offsets, recent_start[:, None] + offsets - self.sink)
        valid = valid_request[:, None] & (positions < lengths[:, None])
        storage_offsets = torch.where(offsets < self.sink, offsets, self.sink + positions % self.ring)
        key = self.key[safe_slots[:, None], storage_offsets]
        value = self.value[safe_slots[:, None], storage_offsets]
        logits = torch.einsum("bhgd,blhd->bhgl", query.float(), key.float()) * scale
        logits = logits.masked_fill(~valid[:, None, None, :], -torch.inf)
        lse = logits.logsumexp(-1)
        normalizer = torch.where(torch.isfinite(lse), lse, torch.zeros_like(lse))
        probabilities = torch.exp(logits - normalizer[..., None])
        output = torch.einsum("bhgl,blhd->bhgd", probabilities, value.float())
        return output, lse

    def attend_and_merge(self, query, request_slots, lengths, scale, cold, cold_lse, buffers=None):
        if buffers is None:
            buffers = torch.empty_like(query), torch.empty_like(cold_lse)
        output, lse = buffers
        batch, heads, groups, dim = query.shape
        _exact_and_cold[(batch * heads, groups)](
            query.contiguous(),
            request_slots.contiguous(),
            lengths.contiguous(),
            self.key,
            self.value,
            cold.contiguous(),
            cold_lse.contiguous(),
            output,
            lse,
            heads,
            groups,
            dim,
            self.sink,
            self.ring,
            self.capacity,
            self.limit,
            scale,
            64,
        )
        return output, lse, buffers


@triton.jit
def _exact_and_cold(
    Q,
    SLOTS,
    LENGTHS,
    KEY,
    VALUE,
    COLD,
    COLD_LSE,
    OUT,
    LSE,
    H: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
    SINK: tl.constexpr,
    RING: tl.constexpr,
    CAPACITY: tl.constexpr,
    LIMIT: tl.constexpr,
    SCALE: tl.constexpr,
    TILE: tl.constexpr,
):
    bh, group = tl.program_id(0), tl.program_id(1)
    request, head = bh // H, bh % H
    slot = tl.load(SLOTS + request)
    length = tl.load(LENGTHS + request)
    valid_request = (slot >= 0) & (slot < LIMIT) & (length > 0)
    slot = tl.where(valid_request, slot, LIMIT)
    dims = tl.arange(0, D)
    q = tl.load(Q + (bh * G + group) * D + dims).to(tl.float32)
    maximum = tl.load(COLD_LSE + bh * G + group)
    denominator = tl.where(maximum > -float("inf"), 1.0, 0.0)
    accum = tl.load(COLD + (bh * G + group) * D + dims).to(tl.float32) * denominator
    # Once the exact window is full, the sink plus ring is a permutation of
    # the complete retained set. Reading physical rows directly avoids the
    # per-lane logical-position arithmetic in the hot path. Short requests
    # still need the mapping to exclude stale ring rows.
    full_window = length >= CAPACITY
    for begin in range(0, CAPACITY, TILE):
        offsets = begin + tl.arange(0, TILE)
        if full_window:
            valid = valid_request & (offsets < CAPACITY)
            index = offsets
        else:
            recent_start = tl.maximum(SINK, length - RING)
            position = tl.where(offsets < SINK, offsets, recent_start + offsets - SINK)
            valid = valid_request & (offsets < CAPACITY) & (position < length)
            index = tl.where(offsets < SINK, offsets, SINK + position % RING)
        addresses = ((slot * CAPACITY + index) * H + head) * D
        key = tl.load(KEY + addresses[:, None] + dims[None, :], valid[:, None], 0).to(tl.float32)
        scores = tl.sum(key * q[None, :], 1) * SCALE
        scores = tl.where(valid, scores, -float("inf"))
        next_maximum = tl.maximum(maximum, tl.max(scores, 0))
        safe_maximum = tl.where(next_maximum > -float("inf"), next_maximum, 0.0)
        correction = tl.exp(maximum - safe_maximum)
        probabilities = tl.exp(scores - safe_maximum)
        value = tl.load(VALUE + addresses[:, None] + dims[None, :], valid[:, None], 0).to(tl.float32)
        accum = accum * correction + tl.sum(value * probabilities[:, None], 0)
        denominator = denominator * correction + tl.sum(probabilities, 0)
        maximum = next_maximum
    tl.store(OUT + (bh * G + group) * D + dims, accum / tl.maximum(denominator, 1.0e-30))
    tl.store(LSE + bh * G + group, tl.where(denominator > 0, maximum + tl.log(denominator), -float("inf")))
