# SPDX-License-Identifier: Apache-2.0
"""Bit-preserving native-DCP output/LSE exchange for small decode batches."""

import torch
import torch.distributed as dist

from vllm_ascend.ops.triton.sfa_dcp_merge import fused_merge

try:
    import triton  # type: ignore[import-untyped]
    import triton.language as tl  # type: ignore[import-untyped]
except ImportError:
    triton = None
    tl = None


_SUPPORTED_RANKS = (2, 4, 8)
_HEAD_DIM = 512
_MAX_LOCAL_HEADS = 32
# Manually chosen conservative cap, not a demonstrated kernel limit.
# Inputs above 192 have not been tested and retain the upstream path.
_MAX_DECODE_TOKENS = 192


if triton is not None:

    @triton.jit
    def _pack_t1_peer(output_words, lse, send, LOCAL_HEADS: tl.constexpr):
        peer = tl.program_id(0)
        OUTPUT_WORDS: tl.constexpr = LOCAL_HEADS * 256
        PEER_WORDS: tl.constexpr = OUTPUT_WORDS + LOCAL_HEADS
        word = tl.arange(0, OUTPUT_WORDS)
        values = tl.load(output_words + peer * OUTPUT_WORDS + word, mask=word < OUTPUT_WORDS, other=0)
        tl.store(send + peer * PEER_WORDS + word, values, mask=word < OUTPUT_WORDS)
        local_head = tl.arange(0, LOCAL_HEADS)
        stats = tl.load(
            lse + peer * LOCAL_HEADS + local_head,
            mask=local_head < LOCAL_HEADS,
            other=0.0,
        )
        tl.store(
            send + peer * PEER_WORDS + OUTPUT_WORDS + local_head,
            stats.to(tl.int32, bitcast=True),
            mask=local_head < LOCAL_HEADS,
        )

    @triton.jit(do_not_specialize=["tokens"])
    def _pack_token_peer(
        output,
        lse,
        send,
        output_s0,
        output_s1,
        lse_s0,
        lse_s1,
        tokens,
        LOCAL_HEADS: tl.constexpr,
        HEAD_WORDS: tl.constexpr,
        STATS_TILE: tl.constexpr,
    ):
        peer = tl.program_id(0)
        OUTPUT_WORDS: tl.constexpr = LOCAL_HEADS * HEAD_WORDS
        peer_words = tokens * (OUTPUT_WORDS + LOCAL_HEADS)
        word = tl.arange(0, OUTPUT_WORDS)
        local_head = word // HEAD_WORDS
        dim_word = word % HEAD_WORDS
        peer_base = peer * peer_words
        for token in range(tokens):
            values = tl.load(output + token * output_s0 + (peer * LOCAL_HEADS + local_head) * output_s1 + dim_word)
            tl.store(send + peer_base + token * OUTPUT_WORDS + word, values)
        index = tl.arange(0, STATS_TILE)
        token, head = index // LOCAL_HEADS, index % LOCAL_HEADS
        stats = tl.load(
            lse + token * lse_s0 + (peer * LOCAL_HEADS + head) * lse_s1,
            mask=index < tokens * LOCAL_HEADS,
            other=0.0,
        )
        tl.store(
            send + peer_base + tokens * OUTPUT_WORDS + index,
            stats.to(tl.int32, bitcast=True),
            mask=index < tokens * LOCAL_HEADS,
        )


def can_exchange(output: torch.Tensor, lse: torch.Tensor, ranks: int = 8) -> bool:
    # These properties are uniform within a valid DCP group. Do not choose
    # different collective counts based on rank-local allocation/stride state.
    return (
        triton is not None
        and output.device.type == "npu"
        and output.dtype == torch.bfloat16
        and output.ndim == 3
        and ranks in _SUPPORTED_RANKS
        and 0 < output.shape[0] <= _MAX_DECODE_TOKENS
        and output.shape[2] == _HEAD_DIM
        and output.shape[1] % ranks == 0
        and 0 < output.shape[1] // ranks <= _MAX_LOCAL_HEADS
        # Full vectors preserve the unmasked copy lowering. Other head counts
        # retain upstream packing rather than scalarizing a masked token loop.
        and ((output.shape[1] // ranks) & ((output.shape[1] // ranks) - 1)) == 0
        and lse.device == output.device
        and lse.dtype == torch.float32
        and tuple(lse.shape) == (output.shape[0], output.shape[1], 1)
    )


def _exchange_tokens(output: torch.Tensor, lse: torch.Tensor, group, ranks: int) -> torch.Tensor:
    tokens = output.shape[0]
    local_heads = output.shape[1] // ranks
    output_words = local_heads * (_HEAD_DIM // 2)
    # Stride-dependent normalization stays inside the one-collective path.
    if output.stride(-1) != 1 or any(stride % 2 for stride in output.stride()[:2]):
        output = output.contiguous()
    if output.storage_offset() % 2:
        output = output.clone()
    words = output.view(torch.int32)
    peer_words = tokens * (output_words + local_heads)
    send = torch.empty((ranks, peer_words), dtype=torch.int32, device=output.device)
    _pack_token_peer[(ranks,)](
        words,
        lse,
        send,
        *words.stride()[:2],
        *lse.stride()[:2],
        tokens,
        local_heads,
        _HEAD_DIM // 2,
        triton.next_power_of_2(tokens * local_heads),
    )
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)
    # Final receiver layout is token-major: no transpose or receive-side copy.
    parts = recv.view(output.dtype).as_strided(
        (ranks, tokens, local_heads, _HEAD_DIM),
        (peer_words * 2, local_heads * _HEAD_DIM, _HEAD_DIM, 1),
    )
    stats = recv.view(torch.float32).as_strided(
        (ranks, tokens, local_heads),
        (peer_words, local_heads, 1),
        storage_offset=tokens * output_words,
    )
    return fused_merge(parts, stats, token_dim=1)


def exchange(output: torch.Tensor, lse: torch.Tensor, group) -> torch.Tensor:
    """Exchange head shards, retaining FP32 LSE bits and merge math."""
    ranks = dist.get_world_size(group)
    if not can_exchange(output, lse, ranks):
        raise ValueError("Unsupported native-DCP output/LSE exchange")
    if output.shape[0] != 1:
        return _exchange_tokens(output, lse, group, ranks)
    local_heads = output.shape[1] // ranks
    output_words = local_heads * (_HEAD_DIM // 2)
    peer_words = output_words + local_heads
    output = output.contiguous()
    if output.storage_offset() % 2:
        output = output.clone()
    lse = lse.contiguous()
    words = output.view(torch.int32)
    send = torch.empty((ranks, peer_words), dtype=torch.int32, device=output.device)
    # T1 normalizes LSE to contiguous [1, heads, 1], so the head stride is one.
    _pack_t1_peer[(ranks,)](words, lse, send, local_heads)
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)
    parts = recv.view(output.dtype).as_strided(
        (ranks, local_heads, 1, _HEAD_DIM),
        (peer_words * 2, _HEAD_DIM, _HEAD_DIM, 1),
    )
    stats = recv.view(torch.float32).as_strided(
        (ranks, local_heads, 1),
        (peer_words, 1, 1),
        storage_offset=output_words,
    )
    return fused_merge(parts, stats, token_dim=2)
