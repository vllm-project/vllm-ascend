# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import ctypes
import multiprocessing as mp
import sys
import threading
import time

import numpy as np
import torch
import torch_npu

from .. import omni_placement


def worker(moe, layer, expert):
    # Simulate expert activation
    for _ in range(200):
        moe.collect_activation(layer, expert)
        print(f"Worker {layer},{expert} activations: {moe.get_activations()}")
        time.sleep(5)

def simulate_npu(moe):
    # Simulate NPU memory (in reality, use NPU API)
    npu_size = 4 * 2 * 3 * 4  # num_layers * experts_per_layer * weights_per_expert * sizeof(float)
    npu_memory = torch.zeros((16,7192,1024),dtype=torch.float,device="npu:0")  # Shared array as NPU proxy
    print(f"NPU memory init: {list(npu_memory)[:12]}")  # First layer

    # 获取NPU内存指针地址
    address = npu_memory.data_ptr()

    # 创建PyCapsule封装指针
    import ctypes

    # 配置PyCapsule的C API接口
    ctypes.pythonapi.PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    ctypes.pythonapi.PyCapsule_New.restype = ctypes.py_object

    # 创建胶囊对象（关键步骤）
    capsule = ctypes.pythonapi.PyCapsule_New(
        ctypes.c_void_p(address),
        None,  # 无析构函数名
        None    # 无析构参数
    )

    # 调用C++函数传递胶囊
    moe.replicate_weight(capsule)

    # moe.replicate_weight(npu_ptr)
    print(f"NPU memory replicated: {list(npu_memory)[:12]}")  # First layer for brevity

def create_cluster_activation(placement_pattern):
    # 将expert_mapping的形状信息传递给C++，创建ClusterActivation对象
    num_layers = placement_pattern.shape[1]
    experts_per_layer = int(torch.sum(placement_pattern[:,0,:]).item() + 0.5)
    activation_window_size = 10  # 假设激活窗口大小为10

    cluster_activation = omni_placement.ClusterActivation(num_layers, experts_per_layer, activation_window_size, True)
    return cluster_activation

def create_moe_weights(rank, world_size, cluster_activation, expert_mapping, placement_pattern):
    # 将 torch.dtype 映射到 c10::ScalarType 的整数值
    dtype_map = {
        torch.int32: 3,     # c10::ScalarType::Int
        # 根据需要添加更多类型
    }
    scalar_type = dtype_map[torch.int32]

    expert_shape = list(expert_mapping.size())
    placement_shape = list(placement_pattern.size())

    # 调用 MoEWeights
    moe_weights = omni_placement.MoEWeights(
        rank,
        world_size,
        cluster_activation,
        expert_mapping.data_ptr(),
        expert_shape,
        scalar_type,
        placement_pattern.data_ptr(),
        placement_shape,
        scalar_type
    )
    return moe_weights
"""
# 创建 NPU 张量
device = torch.device("npu")
expert_to_index = torch.tensor([[0, 1], [1, 0], [1, 1]], dtype=torch.int32, device=device).contiguous() #.to("npu:0")
placement_pattern = torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.int32, device=device).contiguous() #.to("npu:0")


cluster_activation = create_cluster_activation(expert_to_index)
moe_weights = create_moe_weights(cluster_activation, expert_to_index, placement_pattern)
print(expert_to_index)
print(placement_pattern)
print("omni_placement is successfully initialized.")
"""
"""
rank_id = "0"
# 检查是否提供了足够的参数
if len(sys.argv) != 2:
    print("Usage: python sample_planner.py <rank_id>")
else:
    # 打印每个参数
    print("Number of arguments:", len(sys.argv) - 1)
    print("Argument List:", str(sys.argv))

    # 获取第一个参数（不包括脚本名）
    rank_id = sys.argv[1]
    print("rank_id is:", rank_id)

# Initialize MoE weights
moe = moe_weights.MoEWeights(
    num_layers=4,
    experts_per_layer=2,
    weights_per_expert=3,

    shm_name="/moe_shm"
)

# Initialize some weights
moe.init_cpu_weight(0, 0, [1.0, 2.0, 3.0])
moe.init_cpu_weight(0, 1, [4.0, 5.0, 6.0])
print(f"Initial weights: {moe.get_cpu_weights()}")

threads = []
# Start threads to simulate activations
if rank_id == "0":
    threads = [
        threading.Thread(target=worker, args=(moe, 0, 0))
        # threading.Thread(target=worker, args=(moe, 0, 1))
        # threading.Thread(target=simulate_npu, args=(moe,))
        # threading.Thread(target=worker, args=(moe, 0, 1))
    ]
elif rank_id == "1":
    threads = [
        threading.Thread(target=worker, args=(moe, 0, 1))
    ]
else:
    print("rank id error ", rank_id)

for t in threads:
    t.start()

for t in threads:
    t.join()

print(f"Final activations: {moe.get_activations()}")"
"""