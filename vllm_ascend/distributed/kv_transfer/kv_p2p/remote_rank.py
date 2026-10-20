# SPDX-License-Identifier: Apache-2.0
"""Choose a Mooncake remote-rank group for asymmetric prefill/decode TP."""


def select_remote_rank_group(
    remote_ranks: list[list[int]],
    *,
    kv_role: str,
    tp_rank: int,
) -> list[int]:
    """Select the remote rank group for this worker.

    ``remote_ranks`` is indexed by decode TP rank. Consumers, and producers
    whose rank is not a member of any group and is still in range, keep that
    index. A prefill producer first uses the group that contains ``tp_rank``.
    If that rank is outside every group and outside the decode-TP index range,
    use ``remote_ranks[tp_rank % len(remote_ranks)]``.
    """
    if not remote_ranks:
        raise ValueError("remote_ranks is empty; cannot select a Mooncake remote rank group")
    if kv_role == "kv_producer":
        for ranks in remote_ranks:
            if tp_rank in ranks:
                return ranks
        if tp_rank >= len(remote_ranks):
            return remote_ranks[tp_rank % len(remote_ranks)]
    return remote_ranks[tp_rank]
