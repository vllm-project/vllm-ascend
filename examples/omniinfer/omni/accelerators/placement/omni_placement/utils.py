# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import ctypes
import time
from collections import defaultdict
from functools import wraps

import torch
import torch_npu

from . import omni_placement


def get_expert_ids(local_rank_pattern):
    """
    临时提供
    将字典转换为列表，列表索引由layer_func提取的layer_idx决定

    Args:
        local_rank_pattern (torch.Tensor): pattern, dtype:bool, shape: [num_layers, num_experts]
    Returns:
        list: 转换后的列表[list[list]]，索引为layer_idx对应的整数, local_expert_idx
    """
    if not isinstance(local_rank_pattern, torch.Tensor):
        raise TypeError(
            "placement_pattern_current_rank must be a torch.Tensor")
    if local_rank_pattern.dtype != torch.bool:
        raise ValueError(
            "placement_pattern_current_rank must have dtype torch.bool")
    if local_rank_pattern.dim() != 2:
        raise ValueError("placement_pattern_current_rank must be a 2D tensor")

    layer_expert_ids_list = []
    for layer_id, experts in enumerate(local_rank_pattern):
        global_expert_idxs = torch.where(experts)[0].sort()[0].tolist()
        layer_expert_ids_list.append(global_expert_idxs)
    return layer_expert_ids_list


def filter_dict_keys(param_dict, filter_func, filter_param={}):
    """
    根据过滤函数对字典的key进行过滤

    Args:
        param_dict (dict): 输入的字典
        filter_func (callable): 过滤函数，接受key和额外参数，返回布尔值
        filter_param: 传递给filter_func的额外参数，默认为{}

    Returns:
        dict: 过滤后的新字典
    """
    if not isinstance(param_dict, dict):
        raise TypeError("param_dict must be a dictionary")
    if not callable(filter_func):
        raise TypeError("filter_func must be callable")

    return {
        k: v
        for k, v in param_dict.items() if filter_func(k, **filter_param)
    }


def convert_param_dict_to_list(param_dict, layer_func, layer_func_param={}):
    """
    将字典转换为列表，列表索引由layer_func提取的layer_idx决定

    Args:
        param_dict (dict): 输入字典，其键为字符串
        layer_func (callable): 提取layer索引的函数，接受key并返回整数
        layer_func_param: 传递给layer_func的额外参数，默认为{}

    Returns:
        list: 转换后的列表[list[list[list]]]，索引为layer_idx对应的整数

    Raises:
        TypeError: 如果param_dict不是字典或layer_func不可调用
        ValueError/IndexError: 来自layer_func的异常
    """

    def combine_tensors_to_nested_list(tensor_list):
        """
        将一个包含多个[N, M]张量的列表逐元素组合成嵌套列表结构

        Args:
            tensor_list (list): 包含多个torch.Tensor的列表，每个张量形状为[N, M]

        Returns:
            list: 嵌套列表，结构为[[tensor1_0, tensor2_0, ...], [tensor1_1, tensor2_1, ...], ...]

        Raises:
            ValueError: 如果tensor_list为空，或张量形状不匹配
            TypeError: 如果tensor_list不是列表或包含非张量元素
        """
        # 类型检查
        if not isinstance(tensor_list, list):
            raise TypeError("tensor_list must be a list")
        if not tensor_list:
            raise ValueError("tensor_list cannot be empty")

        # 检查所有元素是否为张量并形状一致
        for t in tensor_list:
            if not isinstance(t, torch.Tensor):
                raise TypeError(
                    "All elements in tensor_list must be torch.Tensor")

        unbound_tensors = [torch.unbind(t, dim=0) for t in tensor_list]
        return [list(group) for group in zip(*unbound_tensors)]

    if not isinstance(param_dict, dict):
        raise TypeError("param_dict must be a dictionary")
    if not callable(layer_func):
        raise TypeError("layer_func must be callable")

    # 确定最大索引
    max_idx = -1
    for key in param_dict.keys():
        layer_idx = layer_func(key, **layer_func_param)
        max_idx = max(max_idx, layer_idx)

    # 初始化列表，每个元素为独立的defaultdict(list)
    result_list = [[] for _ in range(max_idx + 1)]

    # 填充列表
    for key, value in param_dict.items():
        layer_idx = layer_func(key, **layer_func_param)
        result_list[layer_idx].append(value)

    for layer_idx, tensor_list in enumerate(result_list):
        if len(tensor_list) == 0: continue
        result_list[layer_idx] = combine_tensors_to_nested_list(tensor_list)
    return result_list


def convert_param_to_ctype(param_list):
    """
    获取list中tensor的地址

    Args:
        list: 转换后的列表[list[list[list]]]，索引Dim:0 为layer_idx对应的整数, Dim:1 为local_expert_idx对应的整数, Dim:2 为 expert 对应的 params数量 (Bias, Weight, ...), element 为 torch.Tensor

    Returns:
        list: 转换后的列表[list[list[list]]]，索引Dim:0 为layer_idx对应的整数, Dim:1 为local_expert_idx对应的整数, Dim:2 为 expert 对应的 params数量 (Bias, Weight, ...), element 为 omni.Tensor
    """
    if not isinstance(param_list, list):
        raise TypeError("param_list must be a list")

    def tensor_to_omni_tensor(tensor, tensor_name):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("All elements must be torch.Tensor")
        length = tensor.numel()
        element_size = tensor.element_size()
        address = tensor.data_ptr()
        dtype = str(tensor.dtype)[len('torch.'):]
        weight = omni_placement.Tensor(data_ptr=address,
                                       length=length,
                                       element_size=element_size,
                                       dtype=dtype,
                                       name=tensor_name)
        return weight

    # 递归处理三维嵌套列表
    return [[[
        tensor_to_omni_tensor(
            tensor,
            f"layer.{layer_idx}.expert.{expert_idx}.weight.{tensor_idx}")
        for tensor_idx, tensor in enumerate(expert_params)
    ] for expert_idx, expert_params in enumerate(local_expert_params)]
            for layer_idx, local_expert_params in enumerate(param_list)]


def calculate_time(func):

    @wraps(func)  # 保留原始函数的元信息
    def wrapper(*args, **kwargs):
        prefix = kwargs.pop('prefix', "")
        start_time = time.perf_counter()  # 记录开始时间
        result = func(*args, **kwargs)  # 执行被装饰的函数
        torch.npu.synchronize()
        end_time = time.perf_counter()  # 记录结束时间
        elapsed_time = end_time - start_time  # 计算耗时
        try:
            rank = torch.distributed.get_rank()
        except:
            rank = 0
        if rank == 0:
            print(
                f"rank: {rank}:, {prefix}Function '{func.__name__}' took {elapsed_time:.6f} seconds to execute",
                flush=True)
        return result  # 返回原函数的结果

    return wrapper
