import os
from dataclasses import dataclass

import torch
from vllm.triton_utils import triton

from .utils import prepare_chunk_indices, prepare_chunk_offsets

GDN_PREFILL_CHUNK_SIZE = 64
SOLVE_TRIL_LARGE_BLOCK_T = 608 * 2
VALIDATE_GDN_PREFILL_PRECOMPUTE_ENV = "VLLM_ASCEND_VALIDATE_GDN_PREFILL_PRECOMPUTE"


@dataclass(frozen=True)
class GDNPrefillPrecomputed:
    chunk_size_64_indices: torch.LongTensor
    chunk_size_64_offsets: torch.LongTensor
    solve_tril_large_block_t: int
    solve_tril_large_block_indices: torch.LongTensor
    cumsum_optim_block_size: int
    cumsum_block_indices: torch.LongTensor


def get_chunk_local_cumsum_optim_block_size(
    num_heads: int,
    chunk_size: int = GDN_PREFILL_CHUNK_SIZE,
) -> int:
    return triton.next_power_of_2((2**18) // (num_heads * chunk_size))


def _validate_precomputed_tensor(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    if not torch.equal(actual, expected):
        raise RuntimeError(f"GDN prefill precompute mismatch for {name}")


def validate_gdn_prefill_precomputed(
    cu_seqlens: torch.LongTensor,
    num_heads: int,
    precomputed: GDNPrefillPrecomputed,
) -> None:
    if os.getenv(VALIDATE_GDN_PREFILL_PRECOMPUTE_ENV) != "1":
        return

    expected_chunk_size_64_indices = prepare_chunk_indices(
        cu_seqlens, GDN_PREFILL_CHUNK_SIZE
    )
    expected_chunk_size_64_offsets = prepare_chunk_offsets(
        cu_seqlens, GDN_PREFILL_CHUNK_SIZE
    )
    expected_solve_tril_large_block_indices = prepare_chunk_indices(
        cu_seqlens, SOLVE_TRIL_LARGE_BLOCK_T
    )
    expected_cumsum_optim_block_size = get_chunk_local_cumsum_optim_block_size(
        num_heads, GDN_PREFILL_CHUNK_SIZE
    )
    expected_cumsum_block_indices = prepare_chunk_indices(
        cu_seqlens, expected_cumsum_optim_block_size
    )

    _validate_precomputed_tensor(
        "chunk_size_64_indices",
        precomputed.chunk_size_64_indices,
        expected_chunk_size_64_indices,
    )
    _validate_precomputed_tensor(
        "chunk_size_64_offsets",
        precomputed.chunk_size_64_offsets,
        expected_chunk_size_64_offsets,
    )
    _validate_precomputed_tensor(
        "solve_tril_large_block_indices",
        precomputed.solve_tril_large_block_indices,
        expected_solve_tril_large_block_indices,
    )
    if precomputed.cumsum_optim_block_size != expected_cumsum_optim_block_size:
        raise RuntimeError(
            "GDN prefill precompute mismatch for cumsum_optim_block_size"
        )
    _validate_precomputed_tensor(
        "cumsum_block_indices",
        precomputed.cumsum_block_indices,
        expected_cumsum_block_indices,
    )


@torch.compiler.disable
def build_gdn_prefill_precomputed(
    cu_seqlens: torch.LongTensor,
    num_heads: int,
) -> GDNPrefillPrecomputed:
    cumsum_optim_block_size = get_chunk_local_cumsum_optim_block_size(num_heads)
    precomputed = GDNPrefillPrecomputed(
        chunk_size_64_indices=prepare_chunk_indices(
            cu_seqlens, GDN_PREFILL_CHUNK_SIZE
        ),
        chunk_size_64_offsets=prepare_chunk_offsets(
            cu_seqlens, GDN_PREFILL_CHUNK_SIZE
        ),
        solve_tril_large_block_t=SOLVE_TRIL_LARGE_BLOCK_T,
        solve_tril_large_block_indices=prepare_chunk_indices(
            cu_seqlens, SOLVE_TRIL_LARGE_BLOCK_T
        ),
        cumsum_optim_block_size=cumsum_optim_block_size,
        cumsum_block_indices=prepare_chunk_indices(cu_seqlens, cumsum_optim_block_size),
    )
    validate_gdn_prefill_precomputed(cu_seqlens, num_heads, precomputed)
    return precomputed
