# SPDX-License-Identifier: Apache-2.0
"""Pure transfer port/topology helpers of the pull mooncake v1 connector.

These three functions used to be nested in
``MooncakeConnectorWorker._get_kv_split_metadata``. They only do arithmetic on
parallel topology and head layout, so they can be unit-tested without a worker.

They live in ``pull/`` rather than ``common/`` on purpose: the transfer planner
that follows consumes ``pull.metadata`` types (``ReqMeta``/``GroupPull``), so
keeping the planner helpers in the pull layer avoids a ``common`` -> ``pull``
dependency; these helpers themselves only belong to the pull protocol family.

This module is pure Python and must stay dependency-free (only ``math``).
"""

import math


def kv_head_groups(*, num_key_value_heads: int, use_mla: bool, use_sparse: bool, tp_size: int) -> list[tuple[int, ...]]:
    """Split KV heads into per-TP-rank groups.

    Note: when neither ``num_key_value_heads // tp_size >= 1`` nor
    ``tp_size // num_key_value_heads > 1`` holds (e.g. 2 heads over 3 TP ranks)
    the legacy implementation falls through without a return and implicitly
    yields ``None``; that behaviour is preserved on purpose.
    """
    if use_mla or use_sparse:
        kv_head_groups = []
        kv_head_ids = [0]
        kv_head_groups.append(tuple(kv_head_ids))
        return kv_head_groups
    if num_key_value_heads // tp_size >= 1:
        kv_head_groups = []
        for tp_rank in range(tp_size):
            kv_head_ids = [
                head_idx + tp_rank * (num_key_value_heads // tp_size)
                for head_idx in range(num_key_value_heads // tp_size)
            ]
            kv_head_groups.append(tuple(kv_head_ids))
        return kv_head_groups
    if tp_size // num_key_value_heads > 1:
        kv_head_groups = []
        for kv_head_ids_ in range(num_key_value_heads):
            kv_head_groups.append(tuple([kv_head_ids_]))
        return kv_head_groups


def context_parallel_parameters_check(
    *,
    remote_cp_size: int,
    local_cp_size: int,
    use_mla: bool,
    use_sparse: bool,
    num_key_value_heads: int,
    prefill_tp_size: int,
    decode_tp_size: int,
) -> None:
    """Validate P/D context-parallel sizes and the resulting head partition."""
    assert remote_cp_size % local_cp_size == 0 or local_cp_size % remote_cp_size == 0, (
        f"P/D CP sizes must be divisible in either direction, got P={remote_cp_size}, D={local_cp_size}."
    )
    if not (use_mla or use_sparse):
        p_node_heads_per_rank = math.ceil(num_key_value_heads / prefill_tp_size)
        d_node_heads_per_rank = math.ceil(num_key_value_heads / decode_tp_size)
        assert d_node_heads_per_rank % p_node_heads_per_rank == 0


def cp_group_meta(
    *,
    num_key_value_heads: int,
    use_mla: bool,
    use_sparse: bool,
    tp_size: int,
    pcp_size: int,
    dcp_size: int,
    port_base: int,
) -> dict[tuple[int, ...], dict]:
    """Map each KV head group to its CP rank groups and handshake ports."""
    # key is kv_head_group, value is cp_groups and which cp_groups to select
    cp_group_meta: dict = {}
    kv_head_group_list = kv_head_groups(
        num_key_value_heads=num_key_value_heads,
        use_mla=use_mla,
        use_sparse=use_sparse,
        tp_size=tp_size,
    )
    dcp_repeat_num = tp_size * pcp_size // len(kv_head_group_list) // dcp_size

    for kv_head_group_idx, kv_head_group in enumerate(kv_head_group_list):
        if kv_head_group not in cp_group_meta:
            cp_group_meta[kv_head_group] = {}
            cp_group_meta[kv_head_group]["cp_groups"] = []
            cp_group_meta[kv_head_group]["select_cp_groups_id"] = 0
        kv_head_group_offset = tp_size // len(kv_head_group_list) * kv_head_group_idx
        for dcp_repeat_idx in range(dcp_repeat_num):
            # MRV2 orders DCP ranks along PCP first, then TP.
            cp_group = []
            dcp_repeat_offset = dcp_size * dcp_repeat_idx
            for dcp_rank in range(dcp_size):
                flat_rank = dcp_repeat_offset + dcp_rank
                pcp_rank = flat_rank % pcp_size
                tp_rank = flat_rank // pcp_size + kv_head_group_offset
                cp_group.append(port_base + pcp_rank * tp_size + tp_rank)
            cp_group_meta[kv_head_group]["cp_groups"].append(cp_group)

    return cp_group_meta


__all__ = [
    "context_parallel_parameters_check",
    "cp_group_meta",
    "kv_head_groups",
]
