# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Index-K and tail cache writes for the GLM-5.3-Flash kpool indexer.

The indexer scores pools, not tokens: every ``pool_size`` consecutive keys
collapse into one cached vector, so ``index_kpool`` times fewer entries are
scored and ``npu_lightning_indexer`` returns pool ids directly. Producing those
entries is what this module does.

Two caches are involved. The index-K cache holds one compressed vector per
*complete* pool and is what the indexer scores against. The tail cache is a
per-request ring of ``pool_size`` slots holding the raw keys and gate scores of
the pool still being filled: that pool cannot be compressed yet, but its tokens
must not be lost, because the pool completes on a later step -- possibly after a
PD transfer. Prefill seeds the ring, decode advances it and compresses each pool
as it completes.

Ascend departs from the CUDA path in one way, and it simplifies everything
downstream: the cached entry stays bf16 instead of being Hadamard-rotated and
quantized to FP8. ``npu_lightning_indexer`` scores bf16 keys, and the rotation
only ever existed to spread energy before quantization -- it is a normalized
orthogonal transform, so ``(Hq).(Hk) == q.k`` and dropping it from both sides
changes nothing. The result is *more* accurate than the FP8 path, at 2 bytes per
dimension per pool.

Nothing here reads a device tensor on the host. Writes that must be skipped --
padded requests, tokens whose pool has not completed -- are aimed at slot 0
instead of being masked out: block 0 is vLLM's reserved null block, which
``BlockPool`` never hands to a request, so those writes land somewhere nothing
reads. That keeps every scatter a plain ``index_copy_`` with in-range indices.
"""

import torch

from vllm_ascend.models.glm5next.ops.kpool_compress import compress_pool


def _pool_pages(index_cache: torch.Tensor, head_dim: int, pools_per_block: int) -> torch.Tensor:
    """The pooled cache as ``[pages, pools_per_block, head_dim]``.

    vLLM numbers a slot ``page * block_states + offset`` using the spec's own
    block size, never the allocated page's width, so a page that page-size
    unification padded out to the MLA one still has to be indexed by its
    leading axis. Flattening the cache instead would walk pages at the
    unpadded width and scatter the writes across pages nothing reads.
    """
    pages = index_cache.shape[0]
    flat = index_cache.reshape(pages, -1, head_dim)
    if flat.shape[1] < pools_per_block:
        raise RuntimeError(
            f"kpool indexer page holds {flat.shape[1]} pools, fewer than the "
            f"{pools_per_block} one scheduler block addresses."
        )
    # The padding follows the block's pools, so narrowing keeps the page stride.
    return flat[:, :pools_per_block]


def _scatter_pools(
    index_cache: torch.Tensor,
    pooled: torch.Tensor,
    pool_slots: torch.Tensor,
    pools_per_block: int,
) -> None:
    """Write pool entries at ``pool_slots``, sending negative slots to slot 0."""
    head_dim = pooled.shape[-1]
    entries = _pool_pages(index_cache, head_dim, pools_per_block)
    destinations = pool_slots.reshape(-1).clamp_min(0).to(torch.int64)
    entries[destinations // pools_per_block, destinations % pools_per_block] = pooled.reshape(-1, head_dim).to(
        entries.dtype
    )


def _tail_ring(tail_cache: torch.Tensor, pool_size: int, head_dim: int) -> torch.Tensor:
    """The ring as ``[pages, 2, pool_size, head_dim]``.

    Same story as ``_pool_pages``: the ring slots are numbered by the spec's
    block size, so the page axis of the allocation is what a slot's block
    refers to, however wide unification made the page.
    """
    if tail_cache.ndim != 4 or tail_cache.shape[1] != 2 or tail_cache.shape[-1] != head_dim:
        raise RuntimeError(f"kpool tail cache must be [pages, 2, states, {head_dim}], got {tuple(tail_cache.shape)}")
    states = tail_cache.shape[2]
    if states < pool_size:
        raise RuntimeError(
            f"kpool tail cache page holds {states} states, fewer than the {pool_size} a ring block needs."
        )
    return tail_cache[:, :, :pool_size]


def _stash_tail(
    tail_cache: torch.Tensor,
    keys: torch.Tensor,
    gate_scores: torch.Tensor,
    tail_slots: torch.Tensor,
    pool_size: int,
) -> None:
    """Write raw keys and gate scores into the ring at ``tail_slots``.

    ``tail_slots`` is token-granular, ``block * pool_size + position %
    pool_size``, and the ring block holds the keys and the gate scores as its
    two heads.
    """
    head_dim = keys.shape[-1]
    ring = _tail_ring(tail_cache, pool_size, head_dim)
    slots = tail_slots.reshape(-1).clamp_min(0).to(torch.int64)
    blocks = slots // pool_size
    offsets = slots % pool_size
    ring[blocks, 0, offsets] = keys.reshape(-1, head_dim).to(ring.dtype)
    ring[blocks, 1, offsets] = gate_scores.reshape(-1, head_dim).to(ring.dtype)


def write_prefill(
    index_cache: torch.Tensor,
    tail_cache: torch.Tensor | None,
    keys: torch.Tensor,
    gate_scores: torch.Tensor,
    position_bias: torch.Tensor,
    *,
    pool_slots: torch.Tensor,
    tail_slots: torch.Tensor | None,
    pool_size: int,
    pools_per_block: int,
) -> None:
    """Compress a prefill batch's complete pools and seed each request's ring.

    Args:
        index_cache: pool-granular cache, any shape whose last axis is
            ``head_dim`` and whose entries are ordered by slot.
        tail_cache: ring cache, or None to skip seeding.
        keys: ``[num_tokens, head_dim]`` bf16 -- raw indexer K.
        gate_scores: ``[num_tokens, head_dim]``.
        position_bias: ``[pool_size, head_dim]``.
        pool_slots: ``[num_tokens]`` -- pool-granular slot mapping. Only the
            last token of a complete pool carries a slot; the tokens inside a
            pool, and padding, carry negatives.
        tail_slots: ``[num_tokens]`` -- token-granular ring slot mapping.
        pool_size: the checkpoint's ``index_kpool``.
        pools_per_block: pools one scheduler block addresses, which is the
            stride the slot mappings above are numbered with.

    Chunk starts are assumed pool-aligned, which is what makes every token a
    pool-completion candidate whose members are the ``pool_size`` tokens ending
    at it.
    """
    num_tokens = keys.shape[0]
    # No pool can complete in a batch shorter than one pool, and the clamped
    # member gather below would read past the batch.
    if num_tokens < pool_size:
        return

    batch_positions = torch.arange(num_tokens, device=keys.device)
    offsets = torch.arange(pool_size, device=keys.device)
    members = (batch_positions - (pool_size - 1)).clamp_min(0).unsqueeze(1) + offsets.unsqueeze(0)
    pooled = compress_pool(keys[members], gate_scores[members], position_bias)

    # A pool whose start falls before the batch was gathered from clamped
    # indices, so drop it however its slot reads.
    completed = torch.where(batch_positions >= pool_size - 1, pool_slots, torch.full_like(pool_slots, -1))
    _scatter_pools(index_cache, pooled, completed, pools_per_block)

    if tail_cache is not None and tail_slots is not None:
        _stash_tail(tail_cache, keys, gate_scores, _tail_only(tail_slots, pool_size), pool_size)


def _tail_only(tail_slots: torch.Tensor, pool_size: int) -> torch.Tensor:
    """Negate every ring slot except those of each request's trailing pool.

    Only the last ``pool_size`` tokens of a request need seeding -- earlier
    tokens were already compressed into the index cache, and writing them would
    just churn the ring. A token belongs to its request's trailing pool when the
    token ``pool_size`` further along the batch sits in a different ring block,
    or runs past the batch.
    """
    num_tokens = tail_slots.shape[0]
    blocks = tail_slots.clamp_min(0) // pool_size
    ahead = torch.cat(
        [
            tail_slots[pool_size:],
            torch.full((min(pool_size, num_tokens),), -1, dtype=tail_slots.dtype, device=tail_slots.device),
        ]
    )[:num_tokens]
    # A negative slot ahead means the batch ended or padding began, so the
    # current token is in the tail either way.
    same_block = (ahead >= 0) & (ahead.clamp_min(0) // pool_size == blocks)
    keep = (tail_slots >= 0) & ~same_block
    return torch.where(keep, tail_slots, torch.full_like(tail_slots, -1))


def write_decode(
    index_cache: torch.Tensor,
    tail_cache: torch.Tensor,
    keys: torch.Tensor,
    gate_scores: torch.Tensor,
    position_bias: torch.Tensor,
    *,
    pool_slots: torch.Tensor,
    tail_slots: torch.Tensor,
    positions: torch.Tensor,
    pool_size: int,
    pools_per_block: int,
) -> None:
    """Advance each request's ring and compress the pools that complete.

    Args:
        index_cache: pool-granular cache.
        tail_cache: ring cache.
        keys: ``[num_requests, tokens_per_request, head_dim]`` bf16.
        gate_scores: ``[num_requests, tokens_per_request, head_dim]``.
        position_bias: ``[pool_size, head_dim]``.
        pool_slots: ``[num_requests, tokens_per_request]`` -- pool-granular.
        tail_slots: ``[num_requests, tokens_per_request]`` -- token-granular.
        positions: ``[num_requests, tokens_per_request]`` -- token-granular
            positions, which is what the pool phase is derived from.
        pool_size: the checkpoint's ``index_kpool``.
        pools_per_block: pools one scheduler block addresses, which is the
            stride ``pool_slots`` is numbered with.

    A request's tokens must be applied in position order, because a pool
    completing at one token needs the tokens before it. The loop below runs over
    ``tokens_per_request`` -- a host-side shape, one on a plain decode step and
    ``num_spec + 1`` on a draft-verify step -- and vectorizes across requests,
    so the order holds without a Python loop over the batch or a host read.
    """
    num_requests, tokens_per_request, head_dim = keys.shape
    if num_requests == 0 or tokens_per_request == 0:
        return

    ring = _tail_ring(tail_cache, pool_size, head_dim)
    slot_offsets = torch.arange(pool_size, device=keys.device)
    # Every request's tokens are consecutive positions, so a pool member's
    # index within this call is its distance from the request's first token.
    first_position = positions[:, 0].unsqueeze(1)

    for token in range(tokens_per_request):
        position = positions[:, token]
        pool_slot = pool_slots[:, token]
        tail_slot = tail_slots[:, token]
        phase = position.clamp_min(0) % pool_size
        block = tail_slot.clamp_min(0).to(torch.int64) // pool_size

        # --- compress the pool this token completes, if any ------------------
        # The pool starts pool_size - 1 positions back, and a pool start is
        # aligned, so pool slot s is also ring slot s.
        pool_start = (position - (pool_size - 1)).unsqueeze(1)
        member_index = pool_start + slot_offsets.unsqueeze(0) - first_position
        # Members at or before this token come from this call; earlier ones were
        # stashed in the ring on a previous step. Reading the in-call ones from
        # `keys` rather than the ring keeps this independent of stash order.
        from_call = (member_index >= 0) & (member_index <= token)
        gather = member_index.clamp(0, tokens_per_request - 1).unsqueeze(-1).expand(-1, -1, head_dim)
        stashed = ring.index_select(0, block)
        member_keys = torch.where(from_call.unsqueeze(-1), keys.gather(1, gather), stashed[:, 0])
        member_gates = torch.where(from_call.unsqueeze(-1), gate_scores.gather(1, gather), stashed[:, 1])

        completes = (pool_slot >= 0) & (position >= 0) & (phase == pool_size - 1)
        _scatter_pools(
            index_cache,
            compress_pool(member_keys, member_gates, position_bias),
            torch.where(completes, pool_slot, torch.full_like(pool_slot, -1)),
            pools_per_block,
        )

        # --- stash this token, after the read above --------------------------
        # Gated on the token-granular ring slot, not on the pool-granular
        # `pool_slot`, which is only set on a pool's last token: gating on that
        # would drop every token inside a pool and leave the completion reading
        # stale ring entries.
        stash = (position >= 0) & (tail_slot >= 0)
        _stash_tail(
            tail_cache,
            keys[:, token],
            gate_scores[:, token],
            torch.where(stash, tail_slot, torch.full_like(tail_slot, -1)),
            pool_size,
        )
