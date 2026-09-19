# SPDX-License-Identifier: Apache-2.0


def encode_worker_rank(
    pp_rank: int,
    pcp_rank: int,
    tp_rank: int,
    pcp_size: int,
    tp_size: int,
) -> int:
    """Flatten a Mooncake worker's ``(PP, PCP, TP)`` coordinates."""
    return (pp_rank * pcp_size + pcp_rank) * tp_size + tp_rank


def decode_worker_rank(worker_rank: int, pcp_size: int, tp_size: int) -> tuple[int, int, int]:
    """Restore ``(PP, PCP, TP)`` coordinates from a flattened worker rank."""
    pp_rank, rank_within_pp = divmod(worker_rank, pcp_size * tp_size)
    pcp_rank, tp_rank = divmod(rank_within_pp, tp_size)
    return pp_rank, pcp_rank, tp_rank


def insert_pcp_rank(pp_tp_rank: int, pcp_rank: int, pcp_size: int, tp_size: int) -> int:
    """Insert a PCP coordinate into a rank flattened as ``(PP, TP)``."""
    pp_rank, tp_rank = divmod(pp_tp_rank, tp_size)
    return encode_worker_rank(pp_rank, pcp_rank, tp_rank, pcp_size, tp_size)


def remove_pcp_rank(worker_rank: int, pcp_size: int, tp_size: int) -> int:
    """Collapse a flattened ``(PP, PCP, TP)`` rank back to ``(PP, TP)``."""
    pp_rank, _, tp_rank = decode_worker_rank(worker_rank, pcp_size, tp_size)
    return pp_rank * tp_size + tp_rank
