# SPDX-License-Identifier: Apache-2.0
"""Kimi MLA port of PR #16349's bit-preserving output/LSE exchange.

Retain native FP32 normalization and accumulation order while removing
receive-side materialization for qualified single-row DCP16 calls.
"""

import torch
import torch.distributed as dist
import torch_npu

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

try:
    from triton.language.extra.cann.extension import get_element
except ImportError:
    get_element = None


_LOCAL_HEADS = 6
_HEAD_DIM = 512
_MAX_DECODE_TOKENS = 12


if triton is not None:

    @triton.jit
    def _pack(
        output,
        lse,
        send,
        os0: tl.constexpr,
        os1: tl.constexpr,
        ls0: tl.constexpr,
        ls1: tl.constexpr,
        tokens: tl.constexpr,
        heads: tl.constexpr,
        dim_words: tl.constexpr,
        block: tl.constexpr,
    ):
        peer = tl.program_id(0)
        token = tl.program_id(1)
        words: tl.constexpr = heads * dim_words
        peer_words: tl.constexpr = tokens * (words + heads)
        i = tl.arange(0, block)
        head, d = i // dim_words, i % dim_words
        values = tl.load(output + token * os0 + (peer * heads + head) * os1 + d, mask=i < words, other=0)
        tl.store(send + peer * peer_words + token * words + i, values, mask=i < words)
        h = tl.arange(0, 8)
        stats = tl.load(lse + token * ls0 + (peer * heads + h) * ls1, mask=h < heads, other=0)
        tl.store(
            send + peer * peer_words + tokens * words + token * heads + h,
            stats.to(tl.int32, bitcast=True),
            mask=h < heads,
        )

    @triton.jit
    def _unpack(
        recv, recv_bf16, output, lse, tokens: tl.constexpr, heads: tl.constexpr, dim: tl.constexpr, block: tl.constexpr
    ):
        peer = tl.program_id(0)
        row = tl.program_id(1)
        rows: tl.constexpr = tokens * heads
        peer_words: tl.constexpr = rows * (dim // 2 + 1)
        d = tl.arange(0, block)
        values = tl.load(recv_bf16 + peer * peer_words * 2 + row * dim + d, mask=d < dim, other=0).to(tl.float32)
        tl.store(output + (peer * rows + row) * dim + d, values, mask=d < dim)
        stat = tl.load(recv + peer * peer_words + rows * (dim // 2) + row)
        stat = stat.to(tl.float32, bitcast=True)
        # Same empty-shard identity as the native transport, without a launch.
        stat = tl.where(stat == float("-inf"), -3.4028234663852886e38, stat)
        tl.store(lse + peer * rows + row, stat)

    @triton.jit
    def _merge_packed(
        recv,
        recv_bf16,
        output,
        rows: tl.constexpr,
        ranks: tl.constexpr,
        programs: tl.constexpr,
        block: tl.constexpr,
    ):
        rank = tl.arange(0, ranks)
        dim = tl.arange(0, block)
        peer_words: tl.constexpr = rows * 257
        for tile in tl.range(tl.program_id(0), rows * (512 // block), programs):
            row = tile // (512 // block)
            d = (tile % (512 // block)) * block + dim
            raw = tl.load(recv + rank * peer_words + rows * 256 + row).to(tl.float32, bitcast=True)
            lse = tl.where(raw == float("-inf"), -3.4028234663852886e38, raw)
            lse = tl.where(raw == float("inf"), float("-inf"), lse)
            maximum = tl.max(lse, 0)
            exp_lse = tl.exp(lse - maximum)
            denominator = get_element(exp_lse, (0,))
            for peer in tl.static_range(1, ranks):
                denominator = denominator + get_element(exp_lse, (peer,))
            # Match native normalization and accumulation order, rather than
            # replacing log-sum-exp with an algebraically equivalent division.
            combined_lse = maximum + tl.log(denominator)
            weights = tl.exp(lse - combined_lse)
            result = tl.zeros((block,), tl.float32)
            for peer in tl.static_range(0, ranks):
                values = tl.load(recv_bf16 + peer * peer_words * 2 + row * 512 + d).to(tl.float32)
                term = values * get_element(weights, (peer,))
                if peer == 0:
                    result = term
                else:
                    result = result + term
            tl.store(output + row * 512 + d, result)


def can_exchange(output: torch.Tensor, lse: torch.Tensor, dcp_size: int) -> bool:
    # Only group-uniform shape/dtype properties select the collective path.
    # Rank-local stride differences are normalized inside exchange().
    return (
        triton is not None
        and dcp_size in (2, 8, 16)
        and output.device.type == "npu"
        and output.dtype == torch.bfloat16
        and output.ndim == 3
        and 0 < output.shape[0] <= _MAX_DECODE_TOKENS
        and tuple(output.shape[1:]) == (dcp_size * _LOCAL_HEADS, _HEAD_DIM)
        and lse.device == output.device
        and lse.dtype == torch.float32
        and tuple(lse.shape) == (output.shape[0], output.shape[1], 1)
    )


def _exchange_buffers(output: torch.Tensor, lse: torch.Tensor, group):
    ranks = dist.get_world_size(group)
    if not can_exchange(output, lse, ranks):
        raise ValueError("Unsupported Kimi MLA DCP output/LSE exchange")
    tokens = output.shape[0]
    if output.stride(-1) != 1 or any(s % 2 for s in output.stride()[:2]):
        output = output.contiguous()
    if output.storage_offset() % 2:
        output = output.clone()
    words = output.view(torch.int32)
    rows = tokens * _LOCAL_HEADS
    peer_words = rows * (_HEAD_DIM // 2 + 1)
    if ranks == 16 and tokens == 1 and words.is_contiguous() and lse.is_contiguous():
        # Native concatenation handles the six-word LSE tail more efficiently
        # than the generic pack kernel. Both inputs are views of raw bits.
        send = torch.cat((words.reshape(ranks, -1), lse.view(torch.int32).reshape(ranks, -1)), dim=1)
    else:
        send = torch.empty((ranks, peer_words), dtype=torch.int32, device=output.device)
        _pack[(ranks, tokens)](
            words,
            lse,
            send,
            *words.stride()[:2],
            *lse.stride()[:2],
            tokens,
            _LOCAL_HEADS,
            _HEAD_DIM // 2,
            triton.next_power_of_2(_LOCAL_HEADS * _HEAD_DIM // 2),
        )
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)
    return recv, ranks, tokens


def exchange(output: torch.Tensor, lse: torch.Tensor, group) -> torch.Tensor:
    if get_element is not None and dist.get_world_size(group) == 16 and output.ndim == 3 and output.shape[0] == 1:
        return exchange_fused(output, lse, group)
    recv, ranks, tokens = _exchange_buffers(output, lse, group)
    rows = tokens * _LOCAL_HEADS
    device = output.device
    parts = torch.empty((ranks, rows, _HEAD_DIM), dtype=torch.float32, device=device)
    stats = torch.empty((ranks, rows), dtype=torch.float32, device=device)
    _unpack[(ranks, rows)](recv, recv.view(torch.bfloat16), parts, stats, tokens, _LOCAL_HEADS, _HEAD_DIM, _HEAD_DIM)
    merged, _ = torch_npu.npu_attention_update(stats.unbind(0), parts.unbind(0), 0)
    return merged.view(tokens, _LOCAL_HEADS, _HEAD_DIM)


def exchange_fused(output: torch.Tensor, lse: torch.Tensor, group) -> torch.Tensor:
    """Fuse packed receive and native-order FP32 merge without intermediate GM buffers."""
    recv, ranks, tokens = _exchange_buffers(output, lse, group)
    rows = tokens * _LOCAL_HEADS
    result = torch.empty((tokens, _LOCAL_HEADS, _HEAD_DIM), device=output.device, dtype=torch.float32)
    cores = triton.runtime.driver.active.utils.get_device_properties(output.device.index)["num_vectorcore"]
    programs = min(rows * 4, cores)
    _merge_packed[(programs,)](
        recv, recv.view(torch.bfloat16), result, rows, ranks, programs, 128, enable_fp_fusion=False
    )
    return result
