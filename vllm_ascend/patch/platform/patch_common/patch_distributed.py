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
import vllm.envs as envs_vllm
from vllm.config import ParallelConfig

from vllm_ascend.utils import is_310p


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
    port = envs_vllm.VLLM_DP_MASTER_PORT if envs_vllm.VLLM_DP_MASTER_PORT else answer
    return port


ParallelConfig.get_next_dp_init_port = parallel_config_get_dp_port


class NullHandle:

    def __init__(self):
        pass

    def wait(self):
        pass


def communication_adaptation_310p():

    def broadcast310p_wrapper(fn):

        def broadcast310p(tensor, src, group=None, async_op=False):
            if tensor.device == torch.device('cpu'):
                return fn(tensor, src, group, async_op)
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

        return broadcast310p

    torch.distributed.broadcast = broadcast310p_wrapper(
        torch.distributed.broadcast)
    torch.distributed.distributed_c10d.broadcast = broadcast310p_wrapper(
        torch.distributed.distributed_c10d.broadcast)

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


if is_310p():
    communication_adaptation_310p()
