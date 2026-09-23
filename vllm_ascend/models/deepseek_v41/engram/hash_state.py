# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend KV-slot adapter for the upstream Engram hash state.

Upstream ``NgramHashState`` reads exactly two things from its
``swa_cache_module``: ``block_size`` once at construction, and ``kv_cache`` on
every ``ensure_cache()`` to size (and re-size) the slot-keyed history. The
Ascend SWA cache differs in ways the adapter absorbs:

* the storage block size comes from the Ascend block-size table
  (``DSV4_BLOCK_SIZES``) and is not the logical block size the scheduler uses;
* ``kv_cache`` is a one-element list and is absent until the KV allocator binds
  it, so ``numel()``/``shape[0]`` must not be called on the raw attribute.

The subclass retains upstream history allocation and lifecycle. Its hash
launch is split into request search and hashing for the Ascend compiler.
"""

import torch
from vllm.triton_utils import tl, triton  # type: ignore[import-not-found]

from .._compat import (
    DEAD_ID,
    EngramLayout,
    NgramHashState,
    _write_hash_cache_kernel,
)


class AscendEngramSlotCache:
    """``swa_cache_module`` view for upstream ``NgramHashState``."""

    def __init__(self, swa_cache_layer) -> None:
        self._layer = swa_cache_layer
        self.block_size = int(swa_cache_layer.block_size)

    @property
    def kv_cache(self) -> torch.Tensor:
        """The SWA KV cache as a single tensor, or an empty one when unbound."""
        cache = getattr(self._layer, "kv_cache", None)
        while isinstance(cache, (list, tuple)) and len(cache) == 1:
            cache = cache[0]
        if cache is None or isinstance(cache, (list, tuple)):
            # Unbound (profiling pass) or a multi-plane cache we do not index.
            return torch.empty(0, dtype=torch.int32)
        return cache


def engram_dead_mask(
    token_ids: torch.Tensor,
    image_token_id: int,
    image_pad_token_id: int,
) -> torch.Tensor:
    """Positions that must not take part in an n-gram.

    Covers both image sentinels, for the current chunk and for the lookback
    window alike. A ``-1`` lookback padding entry is not an image: callers pass
    the window through the same sentinel check and rely on the negative value
    staying a non-match.
    """
    return (token_ids == image_token_id) | (token_ids == image_pad_token_id)


def create_engram_hash_state(vllm_config, config, swa_cache_layer) -> NgramHashState:
    """Bind the upstream hash state to the Ascend SWA cache."""
    layout = EngramLayout.from_config(config)
    assert layout is not None, "Engram hash state needs at least one Engram layer"
    return AscendNgramHashState(vllm_config, layout, AscendEngramSlotCache(swa_cache_layer))


@triton.jit(do_not_specialize=["num_tokens", "num_query_rows"])
def _engram_req_index_kernel(
    query_start_loc,
    req_ids,
    num_tokens,
    num_query_rows,
    query_stride,
    BLOCK_T: tl.constexpr,
):
    """Map each token to the request row whose chunk contains it.

    Split out of ``_hash_ids_kernel``: the Ascend bishengir backend aborts on
    the per-token search when it shares a kernel with the hash body
    ("LLVM ERROR: The buffer memory has been released"), while both halves
    compile on their own. Hoisting also runs the search once per token instead
    of once per (token, layer) program.
    """
    token = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    valid = token < num_tokens
    lo = tl.full((BLOCK_T,), 0, tl.int32)
    hi = tl.full((BLOCK_T,), num_query_rows, tl.int32)
    while tl.sum((lo < hi).to(tl.int32), 0) > 0:
        mid = (lo + hi) // 2
        end = tl.load(query_start_loc + (mid + 1) * query_stride, lo < hi, other=0)
        right = token >= end
        active = lo < hi
        lo = tl.where(active & right, mid + 1, lo)
        hi = tl.where(active & ~right, mid, hi)
    tl.store(req_ids + token, tl.minimum(lo, num_query_rows - 1), mask=valid)


@triton.jit(
    do_not_specialize=[
        "num_tokens",
        "num_slots",
        "num_table_rows",
        "max_blocks",
    ]
)
def _hash_ids_kernel(
    input_ids,
    token_map,
    dead_mask,
    positions,
    block_table,
    query_start_loc,
    req_ids,
    multipliers,
    primes,
    offsets,
    cache,
    lookback_token_ids,
    lookback_dead_mask,
    output,
    num_tokens,
    num_slots,
    pad_id,
    input_stride,
    mask_stride,
    position_stride,
    table_stride,
    table_col_stride,
    query_stride,
    num_table_rows,
    max_blocks,
    cache_block_size,
    MAX_NGRAM: tl.constexpr,
    num_heads,
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
    dead_id,
    lookback_depth,
    lookback_row_stride,
    lookback_col_stride,
    lookback_mask_row_stride,
    lookback_mask_col_stride,
):
    token = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    layer = tl.program_id(1)
    num_layers = tl.num_programs(1)
    valid = token < num_tokens
    # The request search lives in _engram_req_index_kernel (see its docstring).
    req = tl.load(req_ids + token, valid, other=0).to(tl.int64)
    chunk_idx = tl.load(query_start_loc + req * query_stride)
    chunk_idx = tl.minimum(chunk_idx, num_tokens - 1).to(tl.int64)
    chunk_start = tl.load(positions + chunk_idx * position_stride)
    position = tl.load(positions + token * position_stride, valid, other=0).to(tl.int64)
    head = tl.arange(0, BLOCK_H)
    blocked = tl.full((BLOCK_T,), False, tl.int1)
    rolling = tl.full((BLOCK_T,), 0, tl.int64)
    for shift in tl.static_range(MAX_NGRAM):
        lookback = position - shift
        in_batch = lookback >= chunk_start
        batch_idx = tl.maximum(token - shift, 0)
        batch_token = tl.load(input_ids + batch_idx * input_stride, valid & in_batch, other=0)
        batch_source = tl.load(token_map + batch_token, valid & in_batch, other=0)
        batch_dead = tl.load(dead_mask + batch_idx * mask_stride, valid & in_batch, other=False)
        batch_source = tl.where(batch_dead, dead_id, batch_source)

        col = chunk_start - 1 - lookback
        in_window = valid & ~in_batch & (col >= 0) & (col < lookback_depth)
        col = tl.minimum(tl.maximum(col, 0), lookback_depth - 1)
        window_token = tl.load(
            lookback_token_ids + req * lookback_row_stride + col * lookback_col_stride,
            in_window,
            other=-1,
        )
        known = in_window & (window_token >= 0)
        window_source = tl.load(token_map + window_token, known, other=0)
        window_dead = tl.load(
            lookback_dead_mask + req * lookback_mask_row_stride + col * lookback_mask_col_stride,
            known,
            other=False,
        )
        window_source = tl.where(window_dead, dead_id, window_source)

        if cache is not None:
            clamped = tl.minimum(tl.maximum(lookback, 0), max_blocks * cache_block_size - 1)
            block_row = tl.minimum(req, num_table_rows - 1)
            needs_cache = valid & ~in_batch & ~known
            block = tl.load(
                block_table + block_row * table_stride + (clamped // cache_block_size) * table_col_stride,
                needs_cache,
                other=0,
            ).to(tl.int64)
            slot = tl.minimum(
                tl.maximum(block * cache_block_size + clamped % cache_block_size, 0),
                num_slots - 1,
            )
            fallback = tl.load(cache + slot, needs_cache, other=0)
        else:
            fallback = tl.full((BLOCK_T,), pad_id, tl.int32)
        source = tl.where(in_batch, batch_source, tl.where(known, window_source, fallback)).to(tl.int64)
        blocked |= (lookback < 0) | (source == dead_id)
        value = tl.where(blocked, pad_id, source)
        multiplier = tl.load(multipliers + layer * MAX_NGRAM + shift)
        rolling ^= value * multiplier
        if shift > 0:
            col = (shift - 1) * num_heads + head
            param_offset = layer * (MAX_NGRAM - 1) * num_heads + col
            prime = tl.load(primes + param_offset, head < num_heads, other=1)
            offset = tl.load(offsets + param_offset, head < num_heads, other=0)
            hashed = rolling[:, None] % prime[None, :] + offset[None, :]
            out_offset = (token.to(tl.int64) * num_layers + layer)[:, None] * ((MAX_NGRAM - 1) * num_heads) + col[
                None, :
            ]
            tl.store(output + out_offset, hashed, valid[:, None] & (head < num_heads))


class AscendNgramHashState(NgramHashState):
    """Reuse upstream history lifecycle, splitting only the NPU hash launch.

    Triton-Ascend cannot compile the combined request-search/hash kernel.
    Keep that compiler workaround in this subclass, without changing vLLM.
    """

    def dummy_hashes(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Participate in DP lookups without valid rows or hash-cache updates."""
        num_tokens = input_ids.shape[0]
        num_layers, max_ngram = self.multipliers.shape
        num_heads = self.primes.shape[-1]
        hashes = input_ids.new_full(
            (num_tokens, num_layers, (max_ngram - 1) * num_heads),
            DEAD_ID,
            dtype=torch.int32,
        )
        keep = torch.zeros(num_tokens, dtype=torch.bool, device=input_ids.device)
        return hashes, keep

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        query_start_loc: torch.Tensor,
        dead_mask: torch.Tensor,
        lookback_token_ids: torch.Tensor,
        lookback_dead_mask: torch.Tensor,
        slot_mapping: torch.Tensor | None,
        block_table: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute [tokens, layers, hash columns] int32 n-gram hashes.

        History comes from the current chunk, then the runner's lookback
        window, then the optional V1 slot cache.
        """
        cache = self._cache if self.use_slot_cache else None
        num_tokens = input_ids.shape[0]
        num_layers, max_ngram = self.multipliers.shape
        num_heads = self.primes.shape[-1]
        output = input_ids.new_empty((num_tokens, num_layers, (max_ngram - 1) * num_heads), dtype=torch.int32)
        if num_tokens == 0:
            return output
        # Ascend needs the request search out of the hash body (E3); one launch
        # still covers every layer.
        req_ids = input_ids.new_empty(num_tokens, dtype=torch.int32)
        _engram_req_index_kernel[(triton.cdiv(num_tokens, 32),)](
            query_start_loc,
            req_ids,
            num_tokens,
            query_start_loc.numel() - 1,
            query_start_loc.stride(0),
            BLOCK_T=32,
        )
        if self.use_slot_cache:
            assert cache is not None and slot_mapping is not None
            assert block_table is not None
            # Finish writes before other thread blocks read fallback history.
            _write_hash_cache_kernel[(triton.cdiv(num_tokens, 256),)](
                input_ids,
                self.token_map,
                dead_mask,
                slot_mapping,
                cache,
                num_tokens,
                input_ids.stride(0),
                dead_mask.stride(0),
                slot_mapping.stride(0),
                256,
                DEAD_ID,
            )
        _hash_ids_kernel[(triton.cdiv(num_tokens, 32), num_layers)](
            input_ids,
            self.token_map,
            dead_mask,
            positions,
            block_table,
            query_start_loc,
            req_ids,
            self.multipliers,
            self.primes,
            self.offsets,
            cache,
            lookback_token_ids,
            lookback_dead_mask,
            output,
            num_tokens,
            cache.shape[0] if cache is not None else 0,
            self.pad_id,
            input_stride=input_ids.stride(0),
            mask_stride=dead_mask.stride(0),
            position_stride=positions.stride(0),
            table_stride=block_table.stride(0) if block_table is not None else 0,
            table_col_stride=block_table.stride(1) if block_table is not None else 0,
            query_stride=query_start_loc.stride(0),
            num_table_rows=block_table.shape[0] if block_table is not None else 0,
            max_blocks=block_table.shape[1] if block_table is not None else 0,
            cache_block_size=self.block_size,
            MAX_NGRAM=max_ngram,
            num_heads=num_heads,
            BLOCK_T=32,
            BLOCK_H=triton.next_power_of_2(num_heads),
            dead_id=DEAD_ID,
            lookback_depth=lookback_token_ids.shape[1],
            lookback_row_stride=lookback_token_ids.stride(0),
            lookback_col_stride=lookback_token_ids.stride(1),
            lookback_mask_row_stride=lookback_dead_mask.stride(0),
            lookback_mask_col_stride=lookback_dead_mask.stride(1),
            num_warps=4,
        )
        return output
