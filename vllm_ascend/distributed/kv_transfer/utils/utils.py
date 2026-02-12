import math
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from vllm.logger import logger

from vllm_ascend.distributed.parallel_state import get_p_tp_group


def kv_alltoall_and_rearrange(pd_tp_ratio: int, key: torch.Tensor, value: torch.TensorType):
    if pd_tp_ratio <= 1:
        return None, None
    elif key is None or value is None:
        raise ValueError("key or value is None")
    k_output = alltoall_and_rearrange(pd_tp_ratio, key)
    v_output = alltoall_and_rearrange(pd_tp_ratio, value)
    return k_output, v_output


def alltoall_and_rearrange(tp_ratio: int, input_tensor: torch.Tensor):
    num_kv_heads = input_tensor.size(1)
    output_tensor = torch.zeros_like(input_tensor)
    dist.all_to_all_single(output_tensor, input_tensor, group=get_p_tp_group().device_group)
    input_tensor = 0
    result = rearrange_output(output_tensor, tp_ratio, num_kv_heads)
    output_tensor = 0
    return result


def rearrange_output(base_output: torch.Tensor, cut_num: int, num_kv_heads: int):
    size_0 = base_output.size(0)
    if size_0 % cut_num != 0:
        raise ValueError(f"The size of dim 0 [{size_0}] must be divisible by the cut_num [{cut_num}]")
    chunk_size = size_0 // cut_num
    reshaped = base_output.view(cut_num, chunk_size, -1)
    transposed = reshaped.transpose(0, 1)
    return transposed.contiguous().view(size_0, num_kv_heads, -1)


def align_memory(tensor: torch.Tensor, alignment: int) -> torch.Tensor:
    data_ptr = tensor.data_ptr()
    aligned_addr = (data_ptr + alignment - 1) // alignment * alignment
    offset = (aligned_addr - data_ptr) // tensor.element_size()
    return tensor[int(offset) :]


def get_transfer_timeout_value():
    ascend_transfer_timeout = os.getenv("ASCEND_TRANSFER_TIMEOUT", "")
    if len(ascend_transfer_timeout) > 0:
        return int(ascend_transfer_timeout)
    hccl_rdma_timeout = int(os.getenv("HCCL_RDMA_TIMEOUT", "20"))  # type: ignore
    hccl_rdma_retry_cnt = int(os.getenv("HCCL_RDMA_RETRY_CNT", "7"))  # type: ignore
    return int((4.096 * (2**hccl_rdma_timeout)) * hccl_rdma_retry_cnt // 1000 + 3000)


@dataclass
class parallel_info:
    tp_size: int
    pcp_size: int
    dcp_size: int
    use_mla: bool
    pd_head_ratio: int


def get_cp_group(tp: int, heads: int, dcp: int):
    # Partition the second dimension of [pcp][head_group][dcp] to obtain a complete head group
    # head_group is all blocks for request in the same head
    # tp8 dcp2 heads4 return[[0,1,2,3]]
    # tp8 dcp1 heads4 return[[0,2,4,6],[1,3,5,7]]
    step = tp // heads
    if step == 0:
        return [[i for i in range(tp // dcp)]]
    else:
        return [
            set([k // dcp for h in range(heads) for k in range(h * step + i * dcp, h * step + (i + 1) * dcp)])
            for i in range(step // dcp)
        ]


def context_parallel_parameters_check(
    remote_pcp_size: int,
    remote_dcp_size: int,
    p_parallel_info: parallel_info,
    d_parallel_info: parallel_info,
    total_num_kv_heads: int,
):
    # Check whether the pcp–dcp ratio is supported
    assert (p_parallel_info.pcp_size * p_parallel_info.dcp_size) % (remote_pcp_size * remote_dcp_size) == 0
    if not p_parallel_info.use_mla:
        p_node_heads_per_rank = math.ceil(total_num_kv_heads / p_parallel_info.tp_size)
        d_node_heads_per_rank = math.ceil(total_num_kv_heads / d_parallel_info.dcp_size)
        assert d_node_heads_per_rank % p_node_heads_per_rank == 0


def get_tp_rank_head_mapping(num_key_value_heads: int, tp_size: int):
    # Get the head_idx corresponding to the tp_rank, {tp_rank:[head_indx]}
    mapping = {}
    if tp_size <= num_key_value_heads:
        if num_key_value_heads % tp_size != 0:
            raise ValueError(f"Number of heads ({num_key_value_heads}) cannot be evenly divided by TP ({tp_size}).")

        heads_per_rank = num_key_value_heads // tp_size

        for rank in range(tp_size):
            start_idx = rank * heads_per_rank
            end_idx = start_idx + heads_per_rank
            mapping[rank] = list(range(start_idx, end_idx))
    else:
        if tp_size % num_key_value_heads != 0:
            raise ValueError(f"Number of heads ({num_key_value_heads}) cannot be evenly divided by TP ({tp_size}).")
        ranks_per_head = tp_size // num_key_value_heads
        for rank in range(tp_size):
            head_idx = rank // ranks_per_head
            mapping[rank] = [head_idx]
    return mapping


def get_head_group_mapping(num_key_value_heads: int, tp_size: int, num_groups: int, select_cp_group: list[int]):
    # Get the mapping dictionary, where the key is head_group_rank and the value is head_idx
    if tp_size % num_groups != 0:
        raise ValueError(
            f"Total number of devices ({tp_size}) cannot be divided by the number of groups ({num_groups})."
        )
    ranks_per_group = tp_size // num_groups
    tp_mapping = get_tp_rank_head_mapping(num_key_value_heads, tp_size)
    group_mapping = {}
    for group_rank in range(num_groups):
        if group_rank in select_cp_group:
            start_rank = group_rank * ranks_per_group
            end_rank = start_rank + ranks_per_group
            heads_set = set()

            for rank in range(start_rank, end_rank):
                heads_set.update(tp_mapping[rank])
            group_mapping[group_rank] = sorted(list(heads_set))
    return group_mapping


def get_local_remote_block_port_mappings(
    to_trans_idx: int,
    p_parallel_info: parallel_info,
    d_parallel_info: parallel_info,
    d_hosts: list[str],
    d_port: int,
    selected_p_cp_group: list[int],
    selected_d_cp_group: list[int],
    prompt_len: int,
    block_size: int,
    req_meta,
    total_num_kv_heads: int,
    req_id: str,
):
    p_head_group_size = p_parallel_info.tp_size // p_parallel_info.dcp_size
    d_head_group_size = d_parallel_info.tp_size // d_parallel_info.dcp_size
    world_size = d_parallel_info.pcp_size * d_head_group_size * d_parallel_info.dcp_size
    # Compute which logic_block_idx corresponds to each tp_rank
    p_rank_block_mapping: list[list[list[list[int]]]] = [
        [[[] for _ in range(p_parallel_info.dcp_size)] for _ in range(p_head_group_size)]
        for _ in range(p_parallel_info.pcp_size)
    ]
    for logic_block_idx in range(to_trans_idx):
        pcp_rank = (logic_block_idx // p_parallel_info.dcp_size) % p_parallel_info.pcp_size
        dcp_rank = logic_block_idx % p_parallel_info.dcp_size
        for p_head_group_rank in range(p_head_group_size):
            if p_head_group_rank in selected_p_cp_group:
                p_rank_block_mapping[pcp_rank][p_head_group_rank][dcp_rank].append(logic_block_idx)

    # Find the remote device that holds the logic_block_idx
    d_block_rank_mapping: dict[int, dict[int, dict[str, Any]]] = defaultdict(lambda: defaultdict(dict))
    for logic_block_idx in range(to_trans_idx):
        pcp_rank = (logic_block_idx // d_parallel_info.dcp_size) % d_parallel_info.pcp_size
        for d_head_group_rank in range(d_head_group_size):
            if d_head_group_rank in selected_d_cp_group:
                dcp_rank = logic_block_idx % d_parallel_info.dcp_size
                world_rank = (
                    pcp_rank * d_head_group_size * d_parallel_info.dcp_size
                    + d_head_group_rank * d_parallel_info.dcp_size
                    + dcp_rank
                )
                world_size = d_parallel_info.pcp_size * d_head_group_size * d_parallel_info.dcp_size
                host = d_hosts[(len(d_hosts) * world_rank) // world_size]
                port = d_port + world_rank
                block_idx = (logic_block_idx - (pcp_rank * d_parallel_info.pcp_size + dcp_rank)) // (
                    d_parallel_info.pcp_size * d_parallel_info.dcp_size
                )
                d_block_rank_mapping[logic_block_idx][d_head_group_rank] = {
                    "pcp_rank": pcp_rank,
                    "dcp_rank": dcp_rank,
                    "host": host,
                    "port": port,
                    "block_idx": block_idx,
                }
    # Get how many times each device should receive done_single for this request
    d_trans_count_mapping = {}
    trans_block_size = math.ceil(prompt_len / block_size)  # Total number of blocks
    transed_block_size = math.ceil(req_meta.remote_cache_tokens / block_size)  # Number of prefix cache hit blocks
    d_cp_size = d_parallel_info.pcp_size * d_parallel_info.dcp_size
    for d_pcp_rank in range(d_parallel_info.pcp_size):
        for d_head_group_rank in range(d_head_group_size):
            for d_dcp_rank in range(d_parallel_info.dcp_size):
                if trans_block_size >= (p_parallel_info.pcp_size * p_parallel_info.dcp_size):
                    trans_count = (p_parallel_info.pcp_size * p_parallel_info.dcp_size) // d_cp_size
                else:
                    current_rank_idx = d_pcp_rank * d_parallel_info.dcp_size + d_dcp_rank
                    total_global_blocks = transed_block_size + trans_block_size

                    target_total_count = total_global_blocks // d_cp_size
                    if current_rank_idx < (total_global_blocks % d_cp_size):
                        target_total_count += 1

                    prev_processed_count = transed_block_size // d_cp_size
                    if current_rank_idx < (transed_block_size % d_cp_size):
                        prev_processed_count += 1

                    trans_count = target_total_count - prev_processed_count
                world_rank = (
                    d_pcp_rank * d_head_group_size * d_parallel_info.dcp_size
                    + d_head_group_rank * d_parallel_info.dcp_size
                    + d_dcp_rank
                )
                host = d_hosts[(len(d_hosts) * world_rank) // world_size]
                port = d_port + world_rank
                d_trans_count_mapping[(host, port)] = trans_count * p_parallel_info.pd_head_ratio

    # Compute the mapping between local and remote head_group_rank
    p_tp_rank_head_mapping = get_head_group_mapping(
        total_num_kv_heads, p_parallel_info.tp_size, p_head_group_size, selected_p_cp_group
    )
    d_tp_rank_head_mapping = get_head_group_mapping(
        total_num_kv_heads, d_parallel_info.tp_size, d_head_group_size, selected_d_cp_group
    )
    head_to_d_groups = defaultdict(set)
    for d_rank, heads in d_tp_rank_head_mapping.items():
        for head in heads:
            head_to_d_groups[head].add(d_rank)
    pd_head_mapping = {}
    for p_rank, p_heads in p_tp_rank_head_mapping.items():
        target_d_ranks = set()
        for head in p_heads:
            if head in head_to_d_groups:
                target_d_ranks.update(head_to_d_groups[head])
            else:
                logger.info(f"Warning: Head {head} exists in P but not in D mapping.")
        pd_head_mapping[p_rank] = sorted(list(target_d_ranks))
    logger.debug(
        f"MooncakeLayerwiseConnector _get_kv_split_metadata {req_id=} "
        f"P-side logic_block to rank mapping: {p_rank_block_mapping}, "
        f"D-side logic_block to rank mapping: {d_block_rank_mapping}, "
        f"P&D head_group_rank mapping: {pd_head_mapping}"
    )
    return p_rank_block_mapping, d_block_rank_mapping, pd_head_mapping, d_trans_count_mapping


def get_transfer_mappings(
    p_rank_block_mapping: list[list[list[list[int]]]],
    d_block_rank_mapping: dict[int, dict[int, dict[str, Any]]],
    pd_head_mapping: dict[int, set],
    d_trans_count_mapping: dict[tuple[str, int], int],
    req_meta,
    p_parallel_info: parallel_info,
    req_id: str,
    transed_idx: int,
    to_trans_idx: int,
    tp_rank: int,
    pcp_rank: int,
    dcp_rank: int,
):
    transfer_mappings: dict[tuple[str, int], dict[str, Any]] = {}
    p_head_group_rank = (tp_rank - dcp_rank) // p_parallel_info.dcp_size
    p_block_idxs: list[int] = p_rank_block_mapping[pcp_rank][p_head_group_rank][dcp_rank]
    for p_block_idx, logic_block_idx in enumerate(p_block_idxs):
        if logic_block_idx < transed_idx or logic_block_idx >= to_trans_idx:
            continue
        for d_head_group_rank in pd_head_mapping[p_head_group_rank]:
            p_block_id = req_meta.local_block_ids[p_block_idx]
            remote_host = d_block_rank_mapping[logic_block_idx][d_head_group_rank]["host"]
            remote_port = d_block_rank_mapping[logic_block_idx][d_head_group_rank]["port"]
            d_block_idx = d_block_rank_mapping[logic_block_idx][d_head_group_rank]["block_idx"]
            d_block_id = req_meta.remote_block_ids[d_block_idx]
            if (remote_host, remote_port) not in transfer_mappings:
                transfer_mappings[(remote_host, remote_port)] = {
                    "local_block_ids": [],
                    "remote_block_ids": [],
                    "trans_count": 0,
                }
            transfer_mappings[(remote_host, remote_port)]["local_block_ids"].append(p_block_id)
            transfer_mappings[(remote_host, remote_port)]["remote_block_ids"].append(d_block_id)
    for (host, port), block_dict in transfer_mappings.items():
        block_dict["trans_count"] = d_trans_count_mapping[(host, port)]
    logger.debug(f"MooncakeLayerwiseConnector Request {req_id} transfer tasks: {transfer_mappings}")
    return transfer_mappings
