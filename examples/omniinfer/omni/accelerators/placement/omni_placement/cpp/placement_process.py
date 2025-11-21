# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

"""
创建CPP类,完成当前进程下NPU专家权重拷贝至CPU当中，
并且记录当前NPU专家权重地址data_ptr于CPP类对象中， 用于执行CPU专家到NPU专家的替换
note: 该CPP类初始化需放在vllm_npu warmup完成后， 因为warmup阶段experts权重地址会发生改变
"""

import ctypes

import torch
import torch_npu

from .. import omni_placement

# 配置PyCapsule的C API接口
ctypes.pythonapi.PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
ctypes.pythonapi.PyCapsule_New.restype = ctypes.py_object

NAMES = ["w13_weight","w2_weight","w2_weight_offset","w2_weight_scale","w13_weight_offset","w13_weight_scale"]

def generate_name(layer_idx,weight_name):
    return f"model.layers.{layer_idx}.mlp.experts.{weight_name}"

def init_moe_weight(tensor,global_expert_idx,weight_name):

    length = tensor.numel()
    element_size = tensor.element_size()
    address = tensor.data_ptr()
    # 创建胶囊对象（关键步骤）
    capsule = ctypes.pythonapi.PyCapsule_New(
        ctypes.c_void_p(address),
        None,  # 无析构函数名
        None    # 无析构参数
    )
    weight = omni_placement.Weight(
        data_ptr=capsule,
        length=length,
        element_size=element_size,
        expert_id = global_expert_idx,
        name=weight_name
    )
    return weight

class MoEWeights:
    moe_weights = None
    @classmethod
    def init_moe_weights(cls,param_dict,layers:tuple,num_expert_per_rank,rank_id,world_size=4):
        # Only Support EP
        layer_weights_list = []
        for layer_id in range(*layers):
            expert_weights_list=[]
            for expert_id in range(num_expert_per_rank):
                weights_list = [] # Gate, bias, up_down, weight
                for _idx,weight_name in  enumerate(NAMES):
                    weight_name = generate_name(layer_id,weight_name)
                    global_expert_idx = expert_id + num_expert_per_rank*rank_id
                    tensor = param_dict[weight_name][expert_id]
                    # tensor = torch.ones([10],dtype=torch.float32,device="npu:0")*(expert_id+1)
                    # print("Tensor values:", tensor)
                    # print("Tensor dtype:", tensor.dtype)
                    # print("Tensor device:", tensor.device)
                    # print("Tensor size in bytes:", tensor.numel() * tensor.element_size())
                    # print("Tensor data pointer address:", hex(tensor.data_ptr()))
                    # print(tensor.is_contiguous())
                    # # print(tensor)
                    weight = init_moe_weight(tensor,global_expert_idx,weight_name)
                    weights_list.append(weight)
                expert_weights_list.append(weights_list)
            layer_weights_list.append(expert_weights_list)
        # Initial
        cls.moe_weights = omni_placement.MoEWeights(
            npu_weights = layer_weights_list,
            world_size = world_size
        )