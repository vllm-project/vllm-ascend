from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class KimiC8DCPQueryPlan:
    """Request-level TND inputs for native quant MLA metadata."""

    cache_seqlens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    seqused_q: torch.Tensor
    block_table: torch.Tensor
    global_seqlens: torch.Tensor
    cp_world: int
    cp_rank: int
    interleave_size: int
    max_seqlen_kv: int = -1


def local_cache_seqlens(lengths, world_size, rank, interleave_size):
    cycle = world_size * interleave_size
    return (
        torch.div(lengths, cycle, rounding_mode="floor") * interleave_size
        + (lengths.remainder(cycle) - rank * interleave_size).clamp(0, interleave_size)
    ).to(torch.int32)


def _host_int_vector(value: Sequence[int] | torch.Tensor, name: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor) and value.device.type != "cpu":
        raise ValueError(f"{name} must use the existing host metadata, without a device synchronization")
    result = torch.as_tensor(value, dtype=torch.int64)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return result


def build_c8_dcp_query_plan(
    global_seqlens: Sequence[int] | torch.Tensor,
    cu_seqlens_q: Sequence[int] | torch.Tensor,
    block_table: torch.Tensor,
    *,
    world_size: int,
    rank: int,
    interleave_size: int,
) -> KimiC8DCPQueryPlan:
    """Keep one row per request; the operator derives DCP causal boundaries."""
    lengths = _host_int_vector(global_seqlens, "global_seqlens")
    cu = _host_int_vector(cu_seqlens_q, "cu_seqlens_q")
    if cu.numel() != lengths.numel() + 1 or cu[0] != 0:
        raise ValueError("cu_seqlens_q must contain B+1 offsets with a leading zero")
    query_lens = cu[1:] - cu[:-1]
    if bool((query_lens < 0).any()):
        raise ValueError("C8 query offsets must be nondecreasing")
    device = block_table.device
    lengths = lengths.to(device=device, dtype=torch.int32)
    return KimiC8DCPQueryPlan(
        cache_seqlens=local_cache_seqlens(lengths, world_size, rank, interleave_size),
        cu_seqlens_q=cu.to(device=device, dtype=torch.int32),
        seqused_q=torch.where(lengths > 0, query_lens.to(device=device, dtype=torch.int32), 0),
        block_table=block_table[: lengths.numel()],
        global_seqlens=lengths,
        cp_world=world_size,
        cp_rank=rank,
        interleave_size=interleave_size,
    )
