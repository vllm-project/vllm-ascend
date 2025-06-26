#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# Adapted from vllm/tests/basic_correctness/test_basic_correctness.py
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import multiprocessing
import os

import torch
from vllm.distributed.parallel_state import (get_world_group,
                                             init_distributed_environment)
from vllm.utils import update_environment_variables

from vllm_ascend.distributed.device_communicators.pyhccl import \
    PyHcclCommunicator


def distributed_run(fn, world_size):
    number_of_processes = world_size
    processes: list[multiprocessing.Process] = []
    for i in range(number_of_processes):
        env: dict[str, str] = {}
        env['RANK'] = str(i)
        env['LOCAL_RANK'] = str(i)
        env['WORLD_SIZE'] = str(number_of_processes)
        env['LOCAL_WORLD_SIZE'] = str(number_of_processes)
        env['MASTER_ADDR'] = 'localhost'
        env['MASTER_PORT'] = '12345'
        p = multiprocessing.Process(target=fn, args=(env, ))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    for p in processes:
        assert p.exitcode == 0


def worker_fn_wrapper(fn):
    # `multiprocessing.Process` cannot accept environment variables directly
    # so we need to pass the environment variables as arguments
    # and update the environment variables in the function
    def wrapped_fn(env):
        update_environment_variables(env)
        local_rank = os.environ['LOCAL_RANK']
        device = torch.device(f"npu:{local_rank}")
        torch.npu.set_device(device)
        init_distributed_environment(backend="hccl")
        fn()

    return wrapped_fn


@worker_fn_wrapper
def worker_fn():
    pynccl_comm = PyHcclCommunicator(get_world_group().cpu_group,
                                     device=get_world_group().device)
    tensor = torch.ones(16, 1024, 1024,
                        dtype=torch.float32).npu(pynccl_comm.rank)
    tensor = pynccl_comm.all_reduce(tensor)
    torch.npu.synchronize()
    assert torch.all(tensor == pynccl_comm.world_size).cpu().item()


# def test_pyhccl():
#     distributed_run(worker_fn, 2)


@worker_fn_wrapper
def broadcast_worker_fn():
    # Test broadcast for every root rank.
    # Essentially this is an all-gather operation.
    pyhccl_comm = PyHcclCommunicator(get_world_group().cpu_group,
                                     device=get_world_group().device)
    recv_tensors = [
        torch.empty(16,
                    1024,
                    1024,
                    dtype=torch.float32,
                    device=pyhccl_comm.device)
        for i in range(pyhccl_comm.world_size)
    ]
    recv_tensors[pyhccl_comm.rank] = torch.ones(
        16, 1024, 1024, dtype=torch.float32,
        device=pyhccl_comm.device) * pyhccl_comm.rank

    for i in range(pyhccl_comm.world_size):
        pyhccl_comm.broadcast(recv_tensors[i], src=i)
        # the broadcast op might be launched in a different stream
        # need to synchronize to make sure the tensor is ready
        torch.npu.synchronize()
        assert torch.all(recv_tensors[i] == i).cpu().item()


# def test_pyhccl_broadcast():
#     distributed_run(broadcast_worker_fn, 4)
