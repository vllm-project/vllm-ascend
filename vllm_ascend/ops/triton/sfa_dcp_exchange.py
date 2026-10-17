# SPDX-License-Identifier: Apache-2.0
"""Bit-preserving native-DCP8 output/LSE exchange for small decode batches."""

import torch
import torch.distributed as dist

from vllm_ascend.ops.triton.sfa_dcp_merge import fused_merge

try:
    import triton  # type: ignore[import-untyped]
    import triton.language as tl  # type: ignore[import-untyped]
except ImportError:
    triton = None
    tl = None


_RANKS = 8
_HEADS = 64
_HEAD_DIM = 512
_OUTPUT_WORDS_PER_PEER = 2048
_WORDS_PER_PEER = 2056
# Manually chosen conservative cap, not a demonstrated kernel limit.
# Inputs above 192 have not been tested and retain the upstream path.
_MAX_DECODE_TOKENS = 192


if triton is not None:

    @triton.jit
    def _pack_t1_peer(output_words, lse, send):
        peer = tl.program_id(0)
        word = tl.arange(0, 2048)
        values = tl.load(output_words + peer * 2048 + word, mask=word < 2048, other=0)
        tl.store(send + peer * 2056 + word, values, mask=word < 2048)
        local_head = tl.arange(0, 8)
        stats = tl.load(
            lse + peer * 8 + local_head,
            mask=local_head < 8,
            other=0.0,
        )
        tl.store(
            send + peer * 2056 + 2048 + local_head,
            stats.to(tl.int32, bitcast=True),
            mask=local_head < 8,
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


def can_exchange(output: torch.Tensor, lse: torch.Tensor) -> bool:
    # These properties are uniform within a valid DCP group. Do not choose
    # different collective counts based on rank-local allocation/stride state.
    return (
        triton is not None
        and output.device.type == "npu"
        and output.dtype == torch.bfloat16
        and output.ndim == 3
        and 0 < output.shape[0] <= _MAX_DECODE_TOKENS
        and tuple(output.shape[1:]) == (_HEADS, _HEAD_DIM)
        and lse.device == output.device
        and lse.dtype == torch.float32
        and tuple(lse.shape) == (output.shape[0], _HEADS, 1)
    )


def _exchange_tokens(output: torch.Tensor, lse: torch.Tensor, group) -> torch.Tensor:
    if not can_exchange(output, lse) or dist.get_world_size(group) != _RANKS:
        raise ValueError("Unsupported native-DCP8 output/LSE exchange")
    tokens = output.shape[0]
    local_heads = _HEADS // _RANKS
    # Stride-dependent normalization stays inside the one-collective path.
    if output.stride(-1) != 1 or any(stride % 2 for stride in output.stride()[:2]):
        output = output.contiguous()
    if output.storage_offset() % 2:
        output = output.clone()
    words = output.view(torch.int32)
    peer_words = tokens * _WORDS_PER_PEER
    send = torch.empty((_RANKS, peer_words), dtype=torch.int32, device=output.device)
    _pack_token_peer[(_RANKS,)](
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
        (_RANKS, tokens, local_heads, _HEAD_DIM),
        (peer_words * 2, local_heads * _HEAD_DIM, _HEAD_DIM, 1),
    )
    stats = recv.view(torch.float32).as_strided(
        (_RANKS, tokens, local_heads),
        (peer_words, local_heads, 1),
        storage_offset=tokens * _OUTPUT_WORDS_PER_PEER,
    )
    return fused_merge(parts, stats, token_dim=1)


def exchange(output: torch.Tensor, lse: torch.Tensor, group) -> torch.Tensor:
    """Exchange eight head shards, retaining FP32 LSE bits and merge math."""
    if output.shape[0] != 1:
        return _exchange_tokens(output, lse, group)
    output = output.contiguous()
    if output.storage_offset() % 2:
        output = output.clone()
    lse = lse.contiguous()
    words = output.view(torch.int32)
    send = torch.empty((_RANKS, _WORDS_PER_PEER), dtype=torch.int32, device=output.device)
    # T1 normalizes LSE to contiguous [1, 64, 1], so the head stride is one.
    _pack_t1_peer[(_RANKS,)](words, lse, send)
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)
    parts = recv.view(output.dtype).as_strided(
        (_RANKS, _HEADS // _RANKS, 1, _HEAD_DIM),
        (_WORDS_PER_PEER * 2, _HEAD_DIM, _HEAD_DIM, 1),
    )
    stats = recv.view(torch.float32).as_strided(
        (_RANKS, _HEADS // _RANKS, 1),
        (_WORDS_PER_PEER, 1, 1),
        storage_offset=_OUTPUT_WORDS_PER_PEER,
    )
    return fused_merge(parts, stats, token_dim=2)
