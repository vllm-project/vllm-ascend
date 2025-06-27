#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
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
# Adapted from vllm/model_executor/models/qwen2_vl.py
# This file is a part of the vllm-ascend project.

import torch
import vllm
import vllm.distributed
import vllm.envs as envs
from vllm.config import ParallelConfig

from vllm_ascend.utils import is_310p


def ascend_destroy_model_parallel():
    """Set the groups to none and destroy them."""
    from vllm.distributed.parallel_state import _DP, _PP, _TP
    if _TP:
        _TP.destroy()
    _TP = None

    if _PP:
        _PP.destroy()
    _PP = None

    if _DP:
        _DP.destroy()
    _DP = None
    from vllm_ascend.distributed.parallel_state import \
        destory_ascend_model_parallel
    destory_ascend_model_parallel()


def parallel_config_get_dp_port(self) -> int:
    """
    We might need to initialize process groups in multiple
    processes that is related to data parallelism,
    e.g. both in the worker and in the engine, which
    can live in different processes. To avoid port conflicts, we
    increment the port number each time we need to initialize a
    new process group related to data parallelism.
    """
    answer = self.data_parallel_master_port
    self.data_parallel_master_port += 1

    # NOTE: Get port from envs directly when using torchrun
    port = envs.VLLM_DP_MASTER_PORT if envs.VLLM_DP_MASTER_PORT else answer
    return port


vllm.distributed.parallel_state.destroy_model_parallel = ascend_destroy_model_parallel
ParallelConfig.get_next_dp_init_port = parallel_config_get_dp_port


class NullHandle:

    def __init__(self):
        pass

    def wait(self):
        pass


def communication_adaptation_310p():

    def broadcast310p(tensor, src, group=None, async_op=False):
        rank = torch.distributed.get_rank(group)
        world_size = torch.distributed.get_world_size(group)
        tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
        tensor_list[rank] = tensor
        torch.distributed.all_gather(tensor_list, tensor, group=group)
        tensor[...] = tensor_list[src]
        if async_op:
            return NullHandle()
        else:
            return None

    torch.distributed.broadcast = broadcast310p
    torch.distributed.distributed_c10d.broadcast = broadcast310p

    def all_reduce_wrapper_310p(fn):

        def all_reduce(
            tensor,
            op=torch.distributed.ReduceOp.SUM,
            group=None,
            async_op=False,
        ):
            if tensor.dtype != torch.int64:
                return fn(tensor, op, group, async_op)
            rank = torch.distributed.get_rank(group)
            world_size = torch.distributed.get_world_size(group)
            tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
            tensor_list[rank] = tensor
            torch.distributed.all_gather(tensor_list, tensor, group=group)
            if op == torch.distributed.ReduceOp.SUM:
                return torch.stack(tensor_list).sum(0)
            elif op == torch.distributed.ReduceOp.MAX:
                return torch.tensor(
                    torch.stack(tensor_list).cpu().numpy().max(0),
                    device=tensor.device,
                )
            else:
                raise RuntimeError(f"not implement op {op}")

        return all_reduce

    torch.distributed.all_reduce = all_reduce_wrapper_310p(
        torch.distributed.all_reduce)
    torch.distributed.distributed_c10d.all_reduce = all_reduce_wrapper_310p(
        torch.distributed.distributed_c10d.all_reduce)

    def reduce_scatter_310p(output_tensor, input_tensor, group=None):
        rank = torch.distributed.get_rank(group)
        world_size = torch.distributed.get_world_size(group)
        torch.distributed.all_reduce(input_tensor,
                                     torch.distributed.ReduceOp.SUM,
                                     group,
                                     async_op=False)
        interval = input_tensor.shape[0] // world_size
        output_tensor[:] = input_tensor[rank * interval:(rank + 1) * interval]

    torch.distributed._reduce_scatter_base = reduce_scatter_310p
    torch.distributed.distributed_c10d._reduce_scatter_base = reduce_scatter_310p


if is_310p():
    communication_adaptation_310p()
