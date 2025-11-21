# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import ctypes
import re
import time
from collections import defaultdict

import torch

from . import omni_placement
from .distributed_ops import distribution_warmup
from .utils import (calculate_time, convert_param_dict_to_list,
                    convert_param_to_ctype, filter_dict_keys)


def deepseek_filter_func(key, first_k_dense_replace=3):
    pattern = r".*\.layers\.(\d+)\..*\.experts(?:\..*)?$"
    match = re.match(pattern, key)

    if match:
        layer = int(match.group(1))  # 提取layer数字
        return layer >= first_k_dense_replace
    return False


def deepseek_get_layer_idx_func(key, first_k_dense_replace=3):
    pattern = r"^.*\.layers\.(\d+)\..*\.(.+)$"
    match = re.match(pattern, key)
    assert match, f"current key is {key},  layer.layer_idx "
    layer = int(match.group(1))  # 提取layer数字
    return layer - first_k_dense_replace


@calculate_time
def init_dram_weights(moe_weights, param_dict, first_k_dense_replace,
                      init_shm):
    """
    Args:
        moeweights: omni_placement.MoEWeights
        param_dict: 权重信息 Dict(name: torch.Tensor)
        local_rank_pattern (torch.Tensor): pattern, dtype:bool, shape: [num_layers, num_experts]
    """
    # Type checking
    if not isinstance(moe_weights, omni_placement.MoEWeights):
        raise TypeError(
            "moe_weights must be an instance of omni_placement.MoEWeights")
    if not isinstance(param_dict, dict):
        raise TypeError("param_dict must be a dictionary")

    filter_func_params = {"first_k_dense_replace": first_k_dense_replace}
    param_dict = filter_dict_keys(param_dict, deepseek_filter_func,
                                  filter_func_params)  # 传入过滤函数， 过滤出专家权重
    get_layer_func_params = {"first_k_dense_replace": first_k_dense_replace}
    param_list = convert_param_dict_to_list(
        param_dict, deepseek_get_layer_idx_func,
        get_layer_func_params)  # 传入layer识别函数， 权重从 Dict转化为list
    ctype_param_list = convert_param_to_ctype(param_list)  # 取权重地址，转化为c++接收类型

    # 调用C++端的init_weights方法
    moe_weights.init_weights(ctype_param_list, init_shm)


def create_placement_manager(rank,
                             world_size,
                             hccl_comm_world_size,
                             num_devices_per_host,
                             cluster_activation=None,
                             expert_mapping=None,
                             enable_dynamic=False):
    """
    Creates a Placement manage.

    Args:
        rank (int): Rank of the current process in the distributed system.
        world_size (int): Total number of processes in the distributed system.
        num_devices_per_host (int): Number of devices per host machine.
        cluster_activation (optional): Cluster activation object; defaults to None.
        expert_mapping (optional): Expert mapping object containing placement data; defaults to None.

    Returns:
        omni_placement.Placement: A Placement object managing MoE expert placement.
    """
    placement_mapping = expert_mapping.placement_mapping

    src_rank_offset = hccl_comm_world_size - world_size
    rootinfo = get_hccl_root_info(rank, src_rank_offset=src_rank_offset)

    # Instantiate Placement object
    placement = omni_placement.Placement(rank, world_size,
                                         hccl_comm_world_size,
                                         num_devices_per_host,
                                         cluster_activation, placement_mapping,
                                         rootinfo, enable_dynamic)

    return placement


def create_cluster_activation(rank, world_size, hccl_comm_world_size,
                              num_layers, num_deploy_experts_per_rank,
                              count_activation, max_activation_count):
    """
    Creates a ClusterActivation object for managing cluster-level activations.

    Args:
        rank (int): Rank of the current process in the distributed system.
        world_size (int): Total number of processes in the distributed system.
        expert_mapping: Expert mapping object providing layer and expert information.
        count_activation (torch.Tensor): Tensor containing activation count data.

    Returns:
        omni_placement.ClusterActivation: A ClusterActivation object for tracking activations.
    """
    # Extract shape information from expert_mapping
    activation_window_size = 10  # Default activation window size

    length = count_activation.numel()
    element_size = count_activation.element_size()
    address = count_activation.data_ptr()
    dtype = str(count_activation.dtype)[len('torch.'):]

    tensor = omni_placement.Tensor(data_ptr=address,
                                   length=length,
                                   element_size=element_size,
                                   dtype=dtype,
                                   name="")

    # Instantiate ClusterActivation object
    cluster_activation = omni_placement.ClusterActivation(
        tensor, max_activation_count, num_layers, num_deploy_experts_per_rank,
        activation_window_size, world_size, hccl_comm_world_size, rank)
    return cluster_activation


def do_placement_optimizer(placement_manager, layer_id: int):
    omni_placement.do_placement_optimizer(placement_manager, layer_id)


def get_hccl_root_info(rank, src_rank_offset=0):
    torch.distributed.barrier(
    )  # Avoid other ranks accessing dist.backend during warmup processing on this rank.
    distribution_warmup()  # must be warmup before get_hccl_root_info
    if rank == 0:
        root_info = omni_placement.get_pd_rootinfo()
        tensor_to_broadcast = torch.tensor(list(root_info),
                                           dtype=torch.uint8,
                                           device="npu")
        shape = torch.tensor(tensor_to_broadcast.shape,
                             dtype=torch.int64,
                             device="npu")
    else:
        shape = torch.zeros(1, dtype=torch.int64, device="npu")

    # step1. broadcast shape
    torch.distributed.broadcast(shape, src=src_rank_offset)
    if rank != 0:
        tensor_to_broadcast = torch.zeros(shape,
                                          dtype=torch.uint8,
                                          device="npu")

    # Step2. broadcast rootinfo
    torch.distributed.broadcast(tensor_to_broadcast, src=src_rank_offset)
    torch.npu.synchronize()
    root_info = bytes(tensor_to_broadcast.cpu().numpy())
    if rank == 0:
        print(
            f"Rank {rank}: Broadcasted rootinfo = {root_info[:16]}... (length={len(root_info)})",
            flush=True)
    return root_info
