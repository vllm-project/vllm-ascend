# SPDX-License-Identifier: Apache-2.0
"""Pure transfer port/topology helpers of the pull mooncake v1 connector.

These functions used to be nested in
``MooncakeConnectorWorker._get_kv_split_metadata``. They only do arithmetic on
parallel topology, head layout and handshake ports, so they can be unit-tested
without a worker. The port-mapping/HMA helpers are pure address computation too;
their worker-side logging stays in the connector delegate.

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


def local_remote_block_port_mappings(
    *,
    num_key_value_heads: int,
    use_mla: bool,
    use_sparse: bool,
    prefill_tp_size: int,
    remote_cp_size: int,
    local_cp_size: int,
    remote_pcp_size: int,
    remote_dcp_size: int,
    remote_port: int,
    decode_tp_size: int,
    decode_pcp_size: int,
    decode_dcp_size: int,
    decode_side_channel_port: int,
    r_blk: int,
) -> tuple[dict[tuple[int, ...], dict], dict[tuple[int, ...], dict], dict[int, list[list[int]]]]:
    """Build the D-side -> P-side handshake port mapping for one CP geometry.

    Returns the P-side cp-group metadata, the D-side cp-group metadata and the
    final mapping keyed by D-side handshake port. Logging stays in the worker
    delegate so this module keeps no vLLM/logger dependency.
    """
    context_parallel_parameters_check(
        remote_cp_size=remote_cp_size,
        local_cp_size=local_cp_size,
        use_mla=use_mla,
        use_sparse=use_sparse,
        num_key_value_heads=num_key_value_heads,
        prefill_tp_size=prefill_tp_size,
        decode_tp_size=decode_tp_size,
    )
    p_node_cp_group_meta = cp_group_meta(
        num_key_value_heads=num_key_value_heads,
        use_mla=use_mla,
        use_sparse=use_sparse,
        tp_size=prefill_tp_size,
        pcp_size=remote_pcp_size,
        dcp_size=remote_dcp_size,
        port_base=remote_port,
    )
    d_node_cp_group_meta = cp_group_meta(
        num_key_value_heads=num_key_value_heads,
        use_mla=use_mla,
        use_sparse=use_sparse,
        tp_size=decode_tp_size,
        pcp_size=decode_pcp_size,
        dcp_size=decode_dcp_size,
        port_base=decode_side_channel_port,
    )
    local_remote_block_port_mappings: dict[int, list[list[int]]] = {}
    for d_node_head_key in d_node_cp_group_meta:
        for p_node_head_key in p_node_cp_group_meta:
            if not set(p_node_head_key).issubset(set(d_node_head_key)):
                continue
            d_node_head_group = d_node_cp_group_meta[d_node_head_key]
            p_node_head_group = p_node_cp_group_meta[p_node_head_key]
            for d_cp_group in d_node_head_group["cp_groups"]:
                select_cp_groups_id = p_node_head_group["select_cp_groups_id"]
                p_cp_groups = p_node_head_group["cp_groups"]
                p_cp_group = p_cp_groups[select_cp_groups_id]
                p_node_head_group["select_cp_groups_id"] = (
                    select_cp_groups_id + 1 if select_cp_groups_id + 1 < len(p_cp_groups) else 0
                )
                for d_idx, d_port in enumerate(d_cp_group):
                    if d_port not in local_remote_block_port_mappings:
                        local_remote_block_port_mappings[d_port] = []
                    p_port_remote_list = []
                    for p_idx, p_port in enumerate(p_cp_group):
                        # When Bd == Bp, r_blk = 1, which degenerates to the original `p_idx % Lcp` rule.
                        # When Bd = r * Bp, all blocks of P CP rank q are mapped to D rank `(q // r) % Lcp`.
                        if (p_idx // r_blk - d_idx) % min(len(d_cp_group), len(p_cp_group)) == 0:
                            p_port_remote_list.append(p_port)
                    local_remote_block_port_mappings[d_port].append(p_port_remote_list)

    return p_node_cp_group_meta, d_node_cp_group_meta, local_remote_block_port_mappings


def set_hma_shared_port(
    *,
    is_hma_required: bool,
    use_mla: bool,
    use_sparse: bool,
    num_key_value_heads: int,
    prefill_tp_size: int,
    remote_port: int,
    remote_dcp_size: int,
    decode_tp_size: int,
    decode_tp_rank: int,
    chosen_tp_list: list[int],
    remote_handshake_port_list: list[list[int]],
) -> list[list[int]]:
    """Rewrite remote attention ports for HMA load balancing and append Mamba ports.

    Only applies to HMA (hybrid) non-MLA/non-sparse models. It does two things:

    1. Attention replica balancing. A remote attention port offset decomposes as
       ``kv_head_group_offset + dcp_repeat_offset + dcp_rank``. Within the same head
       group, only the TP replicas that share the same ``dcp_rank`` but differ in
       ``dcp_repeat`` hold the exact same attention KV shard. So substitution is
       restricted to the dcp_repeat (replica) dimension - the head group and dcp_rank
       parts are preserved - otherwise different DCP shards would fetch duplicated KV.
       The replica is picked from the request's random rank choice to spread load.
    2. Mamba port append. The Mamba state lives on a different set of P ranks than the
       attention shards, so the matching Mamba ports are appended to the final shard
       (which carries the Mamba transfer); duplicates are skipped.
    """
    if is_hma_required and not (use_mla or use_sparse):
        remote_dcp = max(remote_dcp_size, 1)
        group_span = prefill_tp_size // len(
            kv_head_groups(
                num_key_value_heads=num_key_value_heads,
                use_mla=use_mla,
                use_sparse=use_sparse,
                tp_size=prefill_tp_size,
            )
        )
        n_replica = max(group_span // remote_dcp, 1)
        if n_replica > 1:
            for shard_ports in remote_handshake_port_list:
                for i in range(len(shard_ports)):
                    # Decompose the port offset into pcp segment + head-group offset +
                    # dcp part, keeping all of them and only swapping the replica.
                    tp_off = (shard_ports[i] - remote_port) % prefill_tp_size
                    pcp_seg = (shard_ports[i] - remote_port) - tp_off
                    group_off = tp_off // group_span * group_span
                    dcp_part = (tp_off - group_off) % remote_dcp
                    # Determine replica ID using random choice of current request to maintain load balancing.
                    replica = (chosen_tp_list[i % len(chosen_tp_list)] // remote_dcp) % n_replica
                    shard_ports[i] = remote_port + pcp_seg + group_off + replica * remote_dcp + dcp_part
        # Append this D rank's matching Mamba ports to the final shard (the one that
        # carries the Mamba state); k = prefill_tp / decode_tp ports per D rank.
        k = prefill_tp_size // decode_tp_size
        final_ports = remote_handshake_port_list[-1]
        pcp_seg = (final_ports[0] - remote_port) // prefill_tp_size * prefill_tp_size
        for j in range(k):
            p = remote_port + pcp_seg + decode_tp_rank * k + j
            if p not in final_ports:
                final_ports.append(p)
    return remote_handshake_port_list


__all__ = [
    "context_parallel_parameters_check",
    "cp_group_meta",
    "kv_head_groups",
    "local_remote_block_port_mappings",
    "set_hma_shared_port",
]
