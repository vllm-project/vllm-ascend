# SPDX-License-Identifier: Apache-2.0
"""Pure transfer port/topology helpers of the pull mooncake v1 connector.

These functions used to be methods/nested functions of the pull worker
``MooncakeConnectorWorker._get_kv_split_metadata``. They only do arithmetic on
parallel topology, head layout and handshake ports, so they can be unit-tested
without a worker. The port-mapping/HMA helpers are pure address computation too;
their worker-side logging stays in the connector delegate.

They live in ``pull/`` rather than ``common/`` on purpose: the transfer planner
that follows consumes ``pull.metadata`` types (``ReqMeta``/``GroupPull``), so
keeping the planner helpers in the pull layer avoids a ``common`` -> ``pull``
dependency; these helpers themselves only belong to the pull protocol family.

This module stays pure Python. Runtime imports are limited to the standard library
(``math``/``random``/``collections``/``typing``), ``numpy`` for the rank-split grid,
``GroupPull``/``string_to_int64_hash`` from the same pull family (the algorithms build
``GroupPull`` descriptors and need request-seeded routing), and ``vllm.logger`` for the
SFA debug line. ``ReqMeta``/``BlockIds``/``RemotePortInfo`` are type-only imports.
"""

import math
import random
from collections import OrderedDict
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
from vllm.logger import logger

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.common.transport import string_to_int64_hash
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.pull.metadata import GroupPull

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds

    from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.common.protocol import RemotePortInfo
    from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.pull.metadata import ReqMeta


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


def remote_host_info_by_port(
    *,
    base_port: int,
    remote_handshake_port: int,
    remote_host: str,
    remote_engine_id: str,
    remote_multi_nodes_meta_mapping: dict | None,
    kv_port: int,
) -> tuple[str, str]:
    """Resolve the host/engine id owning a remote handshake port.

    Looks the port offset up in the multi-node handshake mapping, first against
    the serving kv_port and then against the request-level base port.
    """
    if remote_multi_nodes_meta_mapping is None:
        return remote_host, remote_engine_id

    rank = str(remote_handshake_port - kv_port)
    info = remote_multi_nodes_meta_mapping.get(rank)
    if info is None:
        rank = str(remote_handshake_port - base_port)
        info = remote_multi_nodes_meta_mapping.get(rank)
    if info is None:
        return remote_host, remote_engine_id
    return info.get("host", remote_host), info.get("engine_id", remote_engine_id)


def remote_port_send_num(
    *,
    remote_port: int,
    remote_pcp_size: int,
    prefill_tp_size: int,
    kv_port: int,
    remote_multi_nodes_meta_mapping: dict,
    remote_host: str,
    remote_engine_id: str,
    local_remote_block_port_mappings: dict[int, list[list[int]]],
) -> dict[int, "RemotePortInfo"]:
    """Count how many transfer tasks each remote handshake port receives."""
    remote_port_send_num: dict[int, RemotePortInfo] = {}
    remote_ports: set[int] = set(range(remote_port, remote_port + prefill_tp_size * remote_pcp_size))
    for key, remote_host_info in remote_multi_nodes_meta_mapping.items():
        remote_ports.add(int(remote_host_info.get("handshake_port", kv_port + int(key))))
    for remote_port_head_list in local_remote_block_port_mappings.values():
        for remote_port_list in remote_port_head_list:
            for target_port in remote_port_list:
                remote_ports.add(target_port)

    for target_port in remote_ports:
        target_remote_host, _ = remote_host_info_by_port(
            base_port=remote_port,
            remote_handshake_port=target_port,
            remote_host=remote_host,
            remote_engine_id=remote_engine_id,
            remote_multi_nodes_meta_mapping=remote_multi_nodes_meta_mapping,
            kv_port=kv_port,
        )
        remote_port_send_num[target_port] = {"num": 0, "host": target_remote_host}

    for remote_port_head_list in local_remote_block_port_mappings.values():
        for remote_port_list in remote_port_head_list:
            for target_port in remote_port_list:
                remote_port_send_num[target_port]["num"] += 1
    return remote_port_send_num


def local_remote_cp_params(
    *,
    remote_block_size: int | None,
    block_size: int,
    dcp_rank: int,
    dcp_size: int,
    remote_dcp_size: int,
) -> tuple[int, int, int, int, int]:
    """Resolve CP geometry: (remote_block_size, local_cp_rank, local_cp_size,
    remote_cp_size, r_blk), where r_blk = Bd/Bp (>=1) is the D/P block-size ratio.
    Also validates that P/D block sizes are compatible under D-side CP.
    """
    remote_block_size = remote_block_size or block_size
    # MRV2's DCP group already spans PCP; PCP is not another KV shard axis.
    local_cp_rank = dcp_rank
    local_cp_size = dcp_size
    remote_cp_size = remote_dcp_size

    if remote_block_size != block_size:
        assert block_size % remote_block_size == 0 or remote_block_size % block_size == 0, (
            f"Block sizes of P ({remote_block_size}) and D ({block_size}) must be divisible by each other."
        )
        if local_cp_size > 1:
            assert block_size % remote_block_size == 0, (
                f"D node DCP not support P node block_size({remote_block_size}) > D block_size({block_size})"
            )
            # Ensure that the blocks of each P cp rank belong to the same D rank.
            assert (remote_cp_size // local_cp_size) % (block_size // remote_block_size) == 0, (
                f"remote_cp_size({remote_cp_size}) must be an integer multiple of "
                f"r({block_size // remote_block_size}) * local_cp_size({local_cp_size})"
            )

    r_blk = block_size // remote_block_size if block_size > remote_block_size else 1
    return remote_block_size, local_cp_rank, local_cp_size, remote_cp_size, r_blk


def kernel_block_scale(*, layer_indices: list[int], block_size_scale: list[list[int]]) -> int:
    """Kernel block scale for logical-to-tensor block expansion."""
    if layer_indices and layer_indices[0] < len(block_size_scale) and block_size_scale[layer_indices[0]]:
        return block_size_scale[layer_indices[0]][0]
    return 1


def group_kernel_params(
    *,
    kv_group2layeridx: dict[int, tuple[dict, list[int]]],
    block_size: int,
    block_size_scale: list[list[int]],
    remote_block_size: int,
) -> dict[int, tuple[int, int, int]]:
    """Per attention group kernel-expansion params: (local_scale, remote_scale, kernel_size).

    The kernel size is shared by both sides, so remote_scale is derived locally
    from it (no remote handshake scale needed). Mamba groups are not
    block-sharded and are skipped.
    """
    group_kernel_params: dict[int, tuple[int, int, int]] = {}
    for group_idx, (group_spec, layer_indices) in kv_group2layeridx.items():
        if group_spec["kv_cache_spec_type"] == "MambaSpec":
            continue
        local_scale = kernel_block_scale(layer_indices=layer_indices, block_size_scale=block_size_scale)
        kernel_size = block_size // local_scale
        assert remote_block_size % kernel_size == 0, (
            f"remote_block_size({remote_block_size}) not divisible by kernel_size({kernel_size})"
        )
        remote_scale = remote_block_size // kernel_size
        group_kernel_params[group_idx] = (local_scale, remote_scale, kernel_size)
    return group_kernel_params


def dcp_shard_pulls(
    *,
    remote_handshake_port_list: list[list[int]],
    prefill_tp_size: int,
    remote_base_port: int,
    remote_pcp_size: int,
    tp_size: int,
    tp_rank: int,
    kv_group2layeridx: dict[int, tuple[dict[str, Any], list[int]]],
    get_tp_num_need_pulls: Callable[[int | None], int],
) -> list[list[list[GroupPull]]]:
    """Build group pulls from the selected ports of each DCP shard."""
    mamba_num = prefill_tp_size // tp_size
    attn_num = get_tp_num_need_pulls(prefill_tp_size)
    attn_gids = [g for g, (spec, li) in kv_group2layeridx.items() if li and spec["kv_cache_spec_type"] != "MambaSpec"]
    mamba_gids = [g for g, (spec, li) in kv_group2layeridx.items() if li and spec["kv_cache_spec_type"] == "MambaSpec"]
    num_shards = len(remote_handshake_port_list)
    result = []
    for shard_idx, ports in enumerate(remote_handshake_port_list):
        is_final = shard_idx == num_shards - 1
        shard_pulls = []
        for port_idx, port in enumerate(ports):
            pulls = []
            port_tp = (port - remote_base_port) % prefill_tp_size
            pp_rank = (port - remote_base_port) // (prefill_tp_size * remote_pcp_size)
            # Attention uses the leading ports selected for each DCP shard.
            if port_idx < attn_num:
                pulls += [
                    GroupPull(
                        group_id=g,
                        remote_tp_offset=port_idx,
                        num_group_pulls=attn_num,
                        prefill_pp_rank=pp_rank,
                        is_group_transfer_end=port_idx == attn_num - 1,
                    )
                    for g in attn_gids
                ]
            # Transfer Mamba state only on the final DCP shard, using the port's TP rank to derive its offset.
            if is_final:
                m_off = port_tp - tp_rank * mamba_num
                if 0 <= m_off < mamba_num:
                    pulls += [
                        GroupPull(
                            group_id=g,
                            remote_tp_offset=m_off,
                            num_group_pulls=mamba_num,
                            prefill_pp_rank=pp_rank,
                            is_group_transfer_end=m_off == mamba_num - 1,
                        )
                        for g in mamba_gids
                    ]
            shard_pulls.append(pulls)
        result.append(shard_pulls)
    return result


def group_pulls_metadata(
    *,
    req_id: str,
    remote_handshake_port_list: list[list[int]],
    prefill_tp_size: int,
    remote_base_port: int,
    is_hma_required: bool,
    prefill_pp_size: int,
    kv_group2layeridx: dict[int, tuple[dict[str, Any], list[int]]],
    get_tp_num_need_pulls: Callable[[int | None], int],
    get_selected_pcp_rank: Callable[[str, int], int],
    get_hybrid_remote_rank_group_pulls: Callable[..., tuple[list[int], dict[int, list[GroupPull]]]],
    get_dcp_shard_pulls: Callable[..., list[list[list[GroupPull]]]],
    remote_pcp_size: int = 1,
    remote_dcp_size: int = 1,
) -> list[list[list[GroupPull]]]:
    """Build per-port KV cache group pull descriptors.

    Args:
        req_id: Remote request id used to reproduce hybrid-attention rank
            selection for the same request.
        remote_handshake_port_list: Output from ``_get_kv_split_metadata``.
            Each outer item is one transfer shard; each inner item is a
            remote P worker handshake port.
        prefill_tp_size: Effective remote prefill TP size. This may come
            from ``meta.remote_ptp_size`` when P and D use different TP
            sizes.
        remote_base_port: Remote P-side handshake base port. A remote
            worker rank is ``remote_handshake_port - remote_base_port``.

    Returns:
        A three-level list aligned with ``remote_handshake_port_list``:
        ``result[shard_idx][remote_port_idx]`` is the list of ``GroupPull``
        entries for that remote port. Each ``GroupPull`` identifies the KV
        cache group, the remote TP offset to read, the number of pulls
        needed to assemble that group, the prefill PP rank, and whether
        this pull is the final pull for the group. The final-pull flag is
        used by the receiver to decide when group reformatting can run.
    """
    if is_hma_required:
        if remote_dcp_size == 1:
            # The table uses TP/PP ranks without PCP replica offsets.
            # Undo the offset added by _get_kv_split_metadata, keeping the PP stage.
            # E.g. TP2/PP1, PCP rank 1: port base + 3 maps back to rank 3 - 2 = 1.
            _, rank_group_pulls = get_hybrid_remote_rank_group_pulls(req_id, prefill_tp_size)
            pcp_offset = get_selected_pcp_rank(req_id, remote_pcp_size) * prefill_tp_size
            return [
                [rank_group_pulls[p - remote_base_port - pcp_offset] for p in ports]
                for ports in remote_handshake_port_list
            ]

        # The DCP path has already selected the source ports for each shard.
        return get_dcp_shard_pulls(remote_handshake_port_list, prefill_tp_size, remote_base_port, remote_pcp_size)

    tp_num_need_pulls = get_tp_num_need_pulls(prefill_tp_size)
    group_ids = [group_id for group_id, (_, layer_indices) in kv_group2layeridx.items() if layer_indices]

    def make_group_pulls(remote_tp_offset: int, prefill_pp_rank: int) -> list[GroupPull]:
        return [
            GroupPull(
                group_id=group_id,
                remote_tp_offset=remote_tp_offset,
                num_group_pulls=tp_num_need_pulls,
                prefill_pp_rank=prefill_pp_rank,
                is_group_transfer_end=remote_tp_offset == tp_num_need_pulls - 1,
            )
            for group_id in group_ids
        ]

    group_pulls_list = []
    for shard_idx, remote_ports in enumerate(remote_handshake_port_list):
        if len(remote_ports) == 1:
            remote_tp_offsets = [shard_idx % tp_num_need_pulls]
            prefill_pp_ranks = [
                ((remote_ports[0] - remote_base_port) % (prefill_tp_size * prefill_pp_size)) // prefill_tp_size
            ]
        else:
            assert len(remote_ports) % tp_num_need_pulls == 0, (
                f"tp_num_need_pulls: {tp_num_need_pulls}, remote_ports: {remote_ports}"
            )
            remote_tp_offsets = [rank_idx % tp_num_need_pulls for rank_idx in range(len(remote_ports))]
            prefill_pp_ranks = [
                ((remote_port - remote_base_port) % (prefill_tp_size * prefill_pp_size)) // prefill_tp_size
                for remote_port in remote_ports
            ]
        group_pulls_list.append(
            [
                make_group_pulls(remote_tp_offset, prefill_pp_rank)
                for remote_tp_offset, prefill_pp_rank in zip(remote_tp_offsets, prefill_pp_ranks)
            ]
        )
    return group_pulls_list


def hybrid_remote_rank_group_pulls(
    *,
    req_id: str,
    prefill_tp_size: int,
    tp_rank: int | None,
    default_tp_rank: int,
    tp_size: int,
    prefill_pp_size: int,
    kv_group2layeridx: dict[int, tuple[dict[str, Any], list[int]]],
    get_attention_group_num_need_pulls: Callable[..., int],
    get_attention_group_num_key_value_heads: Callable[..., int],
    get_attention_group_remote_rank: Callable[..., list[int]],
) -> tuple[list[int], dict[int, list[GroupPull]]]:
    if tp_rank is None:
        tp_rank = default_tp_rank

    rank_group_pulls: OrderedDict[int, list[GroupPull]] = OrderedDict()
    has_mamba_group = any(
        layer_indices and group_spec["kv_cache_spec_type"] == "MambaSpec"
        for group_spec, layer_indices in kv_group2layeridx.values()
    )
    mamba_num_group_pulls = 0
    if has_mamba_group:
        if prefill_tp_size % tp_size != 0:
            raise ValueError(
                f"Hybrid Mamba prefill tp size({prefill_tp_size}) must be divisible by decode tp size({tp_size})."
            )
        mamba_num_group_pulls = prefill_tp_size // tp_size

    def add_group_pull(remote_rank: int, group_pull: GroupPull) -> None:
        rank_group_pulls.setdefault(remote_rank, []).append(group_pull)

    for group_id, (group_spec, layer_indices) in kv_group2layeridx.items():
        if not layer_indices:
            continue

        if group_spec["kv_cache_spec_type"] == "MambaSpec":
            num_group_pulls = mamba_num_group_pulls
            for pp_rank in range(prefill_pp_size):
                pp_rank_offset = pp_rank * prefill_tp_size
                local_tp_offset = tp_rank * num_group_pulls
                for remote_tp_offset in range(num_group_pulls):
                    remote_rank = pp_rank_offset + local_tp_offset + remote_tp_offset
                    add_group_pull(
                        remote_rank,
                        GroupPull(
                            group_id=group_id,
                            remote_tp_offset=remote_tp_offset,
                            num_group_pulls=num_group_pulls,
                            prefill_pp_rank=pp_rank,
                            is_group_transfer_end=remote_tp_offset == num_group_pulls - 1,
                        ),
                    )
            continue

        num_group_pulls = get_attention_group_num_need_pulls(group_spec, prefill_tp_size)
        num_key_value_heads = get_attention_group_num_key_value_heads(group_spec)
        if has_mamba_group and num_key_value_heads == 1 and num_group_pulls == 1:
            # Keep a replicated single-head attention cache inside this
            # D rank's Mamba owner group so only its owner can signal that
            # all request state has finished transferring.
            replica_offset = random.Random(string_to_int64_hash(req_id)).randrange(mamba_num_group_pulls)
            chosen_rank_list = [
                pp_rank * prefill_tp_size + tp_rank * mamba_num_group_pulls + replica_offset
                for pp_rank in range(prefill_pp_size)
            ]
        else:
            chosen_rank_list = get_attention_group_remote_rank(
                req_id,
                group_spec,
                prefill_tp_size,
                tp_rank,
            )
        assert len(chosen_rank_list) == num_group_pulls * prefill_pp_size, (
            f"chosen_rank_list({chosen_rank_list}) does not match num_group_pulls({num_group_pulls}) "
            f"and prefill pp size({prefill_pp_size})."
        )
        for rank_idx, remote_rank in enumerate(chosen_rank_list):
            prefill_pp_rank = rank_idx // num_group_pulls
            add_group_pull(
                remote_rank,
                GroupPull(
                    group_id=group_id,
                    remote_tp_offset=rank_idx % num_group_pulls,
                    num_group_pulls=num_group_pulls,
                    prefill_pp_rank=prefill_pp_rank,
                    is_group_transfer_end=rank_idx % num_group_pulls == num_group_pulls - 1,
                ),
            )

    return list(rank_group_pulls), dict(rank_group_pulls)


def sfa_replicate_k_block_ids(
    *,
    meta: "ReqMeta",
    use_sfa_sparse: bool,
    enable_sfa_dcp_replicated_indexer: bool,
    dcp_size: int,
    block_size: int,
) -> tuple["BlockIds", "BlockIds"]:
    remote_uses_replicated_indexer = use_sfa_sparse and meta.remote_dcp_size > 1
    if not (enable_sfa_dcp_replicated_indexer or remote_uses_replicated_indexer):
        return tuple(), tuple()
    if meta.num_external_tokens <= 0 or not meta.remote_block_ids or not meta.local_block_ids:
        return tuple(), tuple()

    if len(meta.remote_block_ids) != 1 or len(meta.local_block_ids) != 1:
        raise AssertionError(
            "SFA replicate-K currently expects exactly one KV cache group. "
            f"Got remote groups={len(meta.remote_block_ids)}, local groups={len(meta.local_block_ids)}."
        )

    remote_cp_size = meta.remote_dcp_size
    local_cp_size = dcp_size
    if (
        local_cp_size <= 0
        or remote_cp_size <= 0
        or (remote_cp_size % local_cp_size != 0 and local_cp_size % remote_cp_size != 0)
    ):
        raise AssertionError(
            f"SFA replicate-K requires positive P/D CP sizes with one divisible by the other, "
            f"got P CP={remote_cp_size}, D CP={local_cp_size}."
        )
    num_prefix_cached_blocks = min(meta.num_computed_tokens // block_size, meta.num_prompt_blocks)
    num_external_blocks = meta.num_prompt_blocks - num_prefix_cached_blocks
    num_external_blocks_from_tokens = math.ceil(meta.num_external_tokens / block_size)
    if num_external_blocks < num_external_blocks_from_tokens:
        raise AssertionError(
            f"num_external_blocks({num_external_blocks}) derived from num_computed_tokens "
            f"must cover num_external_blocks_from_tokens({num_external_blocks_from_tokens})."
        )

    if num_prefix_cached_blocks > 0 and not meta.local_full_block_ids:
        raise AssertionError("SFA replicate-K requires full local block ids when prefix cache is used.")

    remote_blocks = list(meta.remote_block_ids[0])
    local_full_blocks = list((meta.local_full_block_ids or meta.local_block_ids)[0])
    if not local_full_blocks:
        return tuple(), tuple()

    local_block_ids: list[int] = []
    remote_block_ids: list[int] = []
    for global_block_idx in range(num_prefix_cached_blocks, meta.num_prompt_blocks):
        remote_local_idx = global_block_idx // remote_cp_size
        local_local_idx = global_block_idx // local_cp_size
        if remote_local_idx >= len(remote_blocks) or local_local_idx >= len(local_full_blocks):
            break
        remote_block_ids.append(
            int(remote_blocks[remote_local_idx]) * remote_cp_size + global_block_idx % remote_cp_size
        )
        local_block_ids.append(
            int(local_full_blocks[local_local_idx]) * local_cp_size + global_block_idx % local_cp_size
        )

    local_block_ids = local_block_ids[:num_external_blocks]
    remote_block_ids = remote_block_ids[: len(local_block_ids)]
    local_block_ids = local_block_ids[: len(remote_block_ids)]

    logger.debug(
        "Mooncake SFA replicate-K block ids prepared from aligned full blocks. "
        "remote_cp_size=%s local_cp_size=%s num_prompt_blocks=%s num_computed_tokens=%s "
        "num_external_blocks=%s local_len=%s remote_len=%s",
        remote_cp_size,
        local_cp_size,
        meta.num_prompt_blocks,
        meta.num_computed_tokens,
        num_external_blocks,
        len(local_block_ids),
        len(remote_block_ids),
    )

    return (local_block_ids,), (remote_block_ids,)


def remote_ranks_for_req(
    *,
    req_id: str,
    default_prefill_tp_size: int,
    decode_tp_size: int,
    prefill_pp_size: int,
    default_num_key_value_heads: int,
    use_sparse: bool,
    model_is_deepseek_mla: bool,
    get_tp_num_need_pulls: Callable[[int | None], int],
    get_remote_tp_ranks: Callable[..., list[list[int]]],
    prefill_tp_size: int | None = None,
    num_key_value_heads: int | None = None,
    tp_num_need_pulls: int | None = None,
    use_mla: bool | None = None,
) -> list[list[int]]:
    if prefill_tp_size is None:
        prefill_tp_size = default_prefill_tp_size
    if num_key_value_heads is None:
        if model_is_deepseek_mla or use_sparse:
            num_key_value_heads = 1
        else:
            num_key_value_heads = default_num_key_value_heads
    if tp_num_need_pulls is None:
        tp_num_need_pulls = get_tp_num_need_pulls(prefill_tp_size)
    if use_mla is None:
        use_mla = model_is_deepseek_mla

    # Divide the ports according to the TP within the PP
    sampled_nums = []
    if prefill_tp_size == decode_tp_size:
        sampled_nums = list(
            map(
                lambda tp: [tp + pp * prefill_tp_size for pp in range(prefill_pp_size)],
                range(prefill_tp_size),
            )
        )
        return sampled_nums
    num_kv_head = num_key_value_heads
    ori_data = np.arange(prefill_tp_size * prefill_pp_size)
    seed = string_to_int64_hash(req_id)
    rand = random.Random(seed)
    # random split prefill tp list
    ori_data_2d = ori_data.reshape(prefill_pp_size, -1)
    num_groups = max(
        1, len(ori_data_2d[0]) // num_kv_head
    )  # The number of redundant copies for each KV head within the PP stage
    rand_group_index = rand.sample(range(num_groups), (max(decode_tp_size // num_kv_head, 1)))  # random choose a group
    all_results = [
        get_remote_tp_ranks(
            ori_data_2d[pp_index],
            rand_group_index,
            num_groups,
            prefill_tp_size,
            num_key_value_heads,
            tp_num_need_pulls,
            use_mla,
        )
        for pp_index in range(prefill_pp_size)
    ]
    for group_index in range(len(all_results[0])):
        group = []
        for pp_index in range(prefill_pp_size):
            group.extend(all_results[pp_index][group_index])
        sampled_nums.append(group)
    return sampled_nums


__all__ = [
    "context_parallel_parameters_check",
    "cp_group_meta",
    "dcp_shard_pulls",
    "group_kernel_params",
    "group_pulls_metadata",
    "hybrid_remote_rank_group_pulls",
    "kernel_block_scale",
    "kv_head_groups",
    "local_remote_block_port_mappings",
    "local_remote_cp_params",
    "remote_host_info_by_port",
    "remote_port_send_num",
    "remote_ranks_for_req",
    "set_hma_shared_port",
    "sfa_replicate_k_block_ids",
]
