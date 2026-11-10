# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Bound wire-key construction shared by writers and scheduler lookups."""

from collections.abc import Callable
from dataclasses import dataclass, field


def layerwise_dcp_rank(tp_rank: int, pcp_rank: int, pcp_size: int, dcp_size: int) -> int:
    """Return the DCP rank that owns a layerwise worker's context shard.

    vLLM builds DCP groups over the (TP, PCP) rank plane after transposing the
    last two dimensions and slicing every ``dcp_size`` entries. Flattening that
    plane row-major gives ``tp_rank * pcp_size + pcp_rank``, so the rank inside
    its DCP group is that flat index modulo ``dcp_size``. With the standalone
    layout accepted by Mooncake layerwise pooling (``pcp_size == 1``) this
    reduces to ``tp_rank % dcp_size``.

    This is the single source of truth for the worker's shard coordinate. The
    scheduler-side hit check requires an object for every DCP rank, and the
    worker validates its own rank against this mapping at startup, so a key
    can never be written under a coordinate that the scheduler does not query.
    """
    return (tp_rank * pcp_size + pcp_rank) % dcp_size


def is_layerwise_dcp_save_leader(tp_rank: int, put_step: int, dcp_size: int, dcp_rank: int) -> bool:
    """Whether ``tp_rank`` is the unique writer of its (head, DCP shard) object.

    Without DCP all TP ranks in a ``put_step`` saving-head group hold identical
    KV, so the head leader is simply the lowest rank of the group. With DCP the
    context is sharded: every DCP rank owns a different shard and only one rank
    per ``(head, dcp_rank)`` content class may write it. The class members are
    the ranks inside the same saving-head group that fall on the same DCP rank;
    the leader is the smallest of them.

    Callers must guarantee ``put_step >= dcp_size`` (``bind_layerwise_keys``
    enforces it). Otherwise a saving-head group of consecutive TP ranks does
    not cover every DCP shard, the peer set can be empty for some classes, and
    the fallback below would make several replicas write the same object.
    """
    if dcp_size <= 1:
        return tp_rank % put_step == 0
    head = tp_rank // put_step
    peers = [rank for rank in range(head * put_step, (head + 1) * put_step) if rank % dcp_size == dcp_rank]
    if not peers:
        return True
    return tp_rank == min(peers)


@dataclass(frozen=True)
class LayerwiseKeyBuilder:
    # The backend binds model/layout identity; callers supply only coordinates.
    make_base_key: Callable[[int, str, int, int], str]
    pp_size: int
    pcp_size: int = 1
    pcp_sharded_group_ids: set[int] = field(default_factory=set)
    dcp_size: int = 1

    def make_full_key(
        self,
        group_id: int,
        block_hash: str,
        rank: int,
        stage: int,
        pcp_rank: int = 0,
        dcp_rank: int | None = None,
    ) -> str:
        key = self.make_base_key(group_id, block_hash, rank, stage)
        if self.pcp_size > 1 and group_id in self.pcp_sharded_group_ids:
            # State head partitions depend on the PCP topology.
            key = f"{key}@pcp_size:{self.pcp_size}@pcp_rank:{pcp_rank}"
        if self.dcp_size > 1:
            if dcp_rank is None:
                # Require the shard coordinate explicitly: a default of 0 would
                # silently put every worker on shard 0 when a caller forgets it.
                raise ValueError("dcp_rank must be passed explicitly when dcp_size > 1")
            # Each DCP rank stores a distinct context shard of the same
            # logical block, so it needs its own objects and sessions.
            key = f"{key}@dcp_size:{self.dcp_size}@dcp_rank:{dcp_rank}"
        return key

    def make_hit_check_keys(self, group_id: int, block_hash: str, num_ranks: int) -> list[str]:
        """Require every saving head on every PP stage, PCP rank and DCP rank."""
        pcp_size = self.pcp_size if group_id in self.pcp_sharded_group_ids else 1
        return [
            self.make_full_key(group_id, block_hash, rank, stage, pcp_rank, dcp_rank)
            for stage in range(self.pp_size)
            for pcp_rank in range(pcp_size)
            for dcp_rank in range(self.dcp_size)
            for rank in range(num_ranks)
        ]
