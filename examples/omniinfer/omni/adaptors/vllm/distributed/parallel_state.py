# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

import os
from typing import Any, List, Optional

import torch
import torch.distributed
from omni.models.common.config.model_config import model_extra_config
from vllm.config import get_current_vllm_config
from vllm.distributed import GroupCoordinator as GroupCoordinatorGPU
from vllm.distributed import (get_ep_group, get_world_group,
                              init_model_parallel_group, parallel_state)
from vllm.logger import logger

initialize_model_parallel_default = parallel_state.initialize_model_parallel

_DIE_PER_NODE_910C = 16
_DIE_PER_NODE_910B = 8

is_device_a2 = os.getenv("ASCEND_PLATFORM", "A3") == "A2"


def get_npu_device_count():
    if is_device_a2:
        return _DIE_PER_NODE_910B
    else:
        return _DIE_PER_NODE_910C


class GroupCoordinator(GroupCoordinatorGPU):

    def all_to_all(
        self,
        input_: torch.Tensor,
        scatter_dim: int = 0,
        gather_dim: int = -1,
        scatter_sizes: Optional[List[int]] = None,
        gather_sizes: Optional[List[int]] = None,
    ) -> torch.Tensor:
        if self.world_size == 1:
            return input_
        return self.device_communicator.all_to_all(input_, scatter_dim, gather_dim, scatter_sizes, gather_sizes)

    def swap(self, input: torch.Tensor, method="all2allv") -> torch.Tensor:
        if len(self.ranks) != 2:
            return input

        if method == "all2allv":
            rank_0 = self.ranks[0]
            rank_1 = self.ranks[1]
            input_shape = input.shape
            input = input.view(-1)
            output = torch.empty_like(input, dtype=input.dtype, device=input.device)

            if self.rank == rank_0:
                split_sizes = [0, input.shape[0]]
            elif self.rank == rank_1:
                split_sizes = [input.shape[0], 0]

            torch.distributed.all_to_all_single(output, input,
                                                output_split_sizes=split_sizes,
                                                input_split_sizes=split_sizes,
                                                group=self.device_group)
            return output.view(input_shape)

        if method == "allgather":
            rank_0 = self.ranks[0]
            rank_1 = self.ranks[1]
            output = torch.empty_like(input, dtype=input.dtype, device=input.device)
            input_size = input.size()
            output_size= (input_size[0] * 2, ) + input_size[1:]
            output_tensor = torch.empty(output_size, dtype=input.dtype, device=input.device)
            torch.distributed.all_gather_into_tensor(output_tensor, input, group=self.device_group)

            if self.rank == rank_1:
                output, _ = torch.split(output_tensor, output_tensor.shape[0] // 2, dim=0)
            elif self.rank == rank_0:
                _, output = torch.split(output_tensor, output_tensor.shape[0] // 2, dim=0)

            return output
        return input

    def all_reduce_async(self, input_: torch.Tensor):
        # Bypass the function if we are using only 1 GPU.
        if self.world_size == 1:
            return input_, None

        return self.device_communicator.all_reduce_async(input_)

    def all_gather_async(self, input_: torch.Tensor, dim: int = -1):
        world_size = self.world_size
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return input_
        # assert -input_.dim() <= dim < input_.dim(), (
        #     f"Invalid dim ({dim}) for input tensor with shape {input_.size()}")

        return self.device_communicator.all_gather_async(input_, dim)

    def reduce_scatter_async(self, input_: torch.Tensor) -> torch.Tensor:
        if self.world_size == 1:
            return input_
        return self.device_communicator.reduce_scatter_async(input_)

    def all_gather_v(self, output_list_: List[torch.Tensor], input_:torch.Tensor) -> None:
        self.device_communicator.all_gather_v(output_list_, input_)

    def reduce_scatter_v(self, output_:torch.Tensor, input_list_:List[torch.Tensor]) -> None:
        self.device_communicator.reduce_scatter_v(output_, input_list_)

_NUM_COMM_GROUP = 2
_LOCAL_COMM_LIST = None
_CROSS_COMM_LIST = None
_GLOBAL_COMM_LIST = None
_CROSS_FAR_COMM_LIST = None
_CROSS_NEAR_COMM_LIST = None
_CROSS_ROUND_COMM_LIST = None
# kept for backward compatibility
_LOCAL_WORLD: Optional[GroupCoordinator] = None
_MLP_TP: Optional[GroupCoordinator] = None
_STREAM1_ATTN_GROUP: Optional[GroupCoordinator] = None
_STREAM1_MLP_GROUP: Optional[GroupCoordinator] = None
_STREAM1_MOE_GROUP: Optional[GroupCoordinator] = None
_SCALE_PARALLEL_GROUP: Optional[GroupCoordinator] = None
GROUP_STREAM1_ATTN = "stream1_attn" # p侧使能双micro batch为第二个流创建 attention 层通信域
GROUP_STREAM1_MLP = "stream1_mlp" # p侧使能双micro batch为第二个流创建 mlp 层通信域
GROUP_STREAM1_MOE = "stream1_moe" # p侧使能双micro batch为第二个流创建 moe 层通信域
_O_PROJ_TP: Optional[GroupCoordinator] = None
_O_PROJ_DP: Optional[GroupCoordinator] = None


def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    enable_expert_parallel: bool = False,
    backend: Optional[str] = None,
) -> None:
    # TP、PP、EP、DP
    initialize_model_parallel_default(
        tensor_model_parallel_size,
        pipeline_model_parallel_size,
        backend,
    )
    initialize_mlp_tp_group(backend)
    initialize_local_world_group(backend)

    if model_extra_config.operator_opt_config.enable_prefill_micro_batch:
        initialize_stream1_attn_group(backend)
        initialize_stream1_mlp_group(backend)
        initialize_stream1_moe_group(backend)

    if model_extra_config.parall_config.o_proj_tp_size > 1:
        initialize_o_proj_tp_group(backend)
        initialize_o_proj_dp_group(backend)

    if is_device_a2:
        if model_extra_config.operator_opt_config.two_stage_comm:
            initialize_cross_comm_group_list(backend)
            initialize_local_comm_group_list(backend)
        else:
            initialize_world_comm_group_list(backend)
            initialize_local_comm_group_list(backend)
            initialize_cross_comm_group_list(backend)

        if model_extra_config.operator_opt_config.enable_round_pipeline_comm:
            num_nodes = torch.distributed.get_world_size() // get_npu_device_count()
            if num_nodes == 4:
                initialize_round_cross_comm_group_list(backend)
                model_extra_config.operator_opt_config.enable_pipeline_comm = 0
            else:
                model_extra_config.operator_opt_config.enable_pipeline_comm = 1
                model_extra_config.operator_opt_config.enable_round_pipeline_comm = 0

            if model_extra_config.operator_opt_config.enable_pipeline_comm:
                initialize_far_cross_comm_group_list(backend)
                initialize_near_cross_comm_group_list(backend)

    scale_parallel = os.environ.get('SCALE_PARALLEL','0') == '1'
    if scale_parallel:
        initial_scale_parallel_group()


def initial_scale_parallel_group():
    config = get_current_vllm_config()
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    data_parallel_size = config.parallel_config.data_parallel_size
    pipeline_model_parallel_size = config.parallel_config.pipeline_parallel_size
    tensor_model_parallel_size = config.parallel_config.tensor_parallel_size

    all_ranks = torch.arange(world_size).reshape(-1, data_parallel_size, pipeline_model_parallel_size,
                tensor_model_parallel_size)
    group_ranks = all_ranks.view(-1, tensor_model_parallel_size).unbind(0)
    group_ranks = [x.tolist() for x in group_ranks]
    global _SCALE_PARALLEL_GROUP
    _SCALE_PARALLEL_GROUP = init_model_parallel_group(group_ranks,
                                    parallel_state.get_world_group().local_rank,
                                    "hccl",
                                    use_message_queue_broadcaster=False,
                                    group_name="scale_parallel")

def _init_parallel_group_factory(
        group_name: str,
        local_size: int,
        backend: Optional[str] = None,
) -> Any:
    if not torch.distributed.is_initialized():
        raise RuntimeError("torch.distributed must be initialized")
    if local_size <= 0:
        raise RuntimeError(f"local_size must be positive, got {local_size}")
    world_size = torch.distributed.get_world_size()
    if world_size % local_size != 0:
        raise RuntimeError(f"world_size[{world_size}] must be divisible by local_size[{local_size}]")

    num_groups = world_size // local_size
    group_ranks = [
        list(range(i * local_size, (i + 1) * local_size))
        for i in range(num_groups)
    ]
    return init_model_parallel_group(
        group_ranks=group_ranks,
        local_rank=get_world_group().local_rank,
        backend=backend or torch.distributed.get_backend(),
        group_name=f'{group_name}'
    )


def  initialize_stream1_attn_group(backend: Optional[str] = None) -> None:
    global _STREAM1_ATTN_GROUP
    if _STREAM1_ATTN_GROUP is not None:
        raise RuntimeError("stream1 attn group already initialized")
    _STREAM1_ATTN_GROUP = _init_parallel_group_factory(
        group_name=GROUP_STREAM1_ATTN,
        local_size=torch.distributed.get_world_size(),
        backend=backend,
    )


def initialize_stream1_mlp_group(backend: Optional[str] = None) -> None:
    global _STREAM1_MLP_GROUP
    if _STREAM1_MLP_GROUP is not None:
        raise RuntimeError("stream1 mlp group already initialized")
    _STREAM1_MLP_GROUP = _init_parallel_group_factory(
        group_name=GROUP_STREAM1_MLP,
        local_size=16,
        backend=backend,
    )


def initialize_stream1_moe_group(backend: Optional[str] = None) -> None:
    global _STREAM1_MOE_GROUP
    if _STREAM1_MOE_GROUP is not None:
        raise RuntimeError("stream1 moe group already initialized")
    _STREAM1_MOE_GROUP = _init_parallel_group_factory(
        group_name=GROUP_STREAM1_MOE,
        local_size=get_ep_group().world_size,
        backend=backend,
    )


def get_stream1_attn_group() -> Any:
    global _STREAM1_ATTN_GROUP
    if _STREAM1_ATTN_GROUP is None:
        raise RuntimeError("stream1 attn group not initialized")
    return _STREAM1_ATTN_GROUP


def get_stream1_mlp_group() -> Any:
    global _STREAM1_MLP_GROUP
    if _STREAM1_MLP_GROUP is None:
        raise RuntimeError("stream1 mlp group not initialized")
    return _STREAM1_MLP_GROUP


def get_stream1_moe_group() -> Any:
    global _STREAM1_MOE_GROUP
    if _STREAM1_MOE_GROUP is None:
        raise RuntimeError("stream1 moe group not initialized")
    return _STREAM1_MOE_GROUP


def calculate_effective_local_size(local_size: int, world_size: int) -> int:
    """
    Calculate the effective local size based on available devices and world size.

    Args:
        local_size (int): Number of available NPU devices.
        world_size (int): Total number of processes in the distributed setup.

    Returns:
        int: The effective local size (minimum of local_size and world_size).

    Notes:
        - Logs a warning if not all devices are used.
        - Ensures world_size is divisible by the effective local size (raises AssertionError otherwise).
    """
    effective_local_size = min(local_size, world_size)
    if effective_local_size < local_size:
        logger.info(f"Note: Using only {effective_local_size} of {local_size} available NPU devices")

    if world_size % effective_local_size != 0:
        raise AssertionError(
            f"world_size ({world_size}) must be divisible by effective_local_size ({effective_local_size})"
        )
    return effective_local_size


def initialize_mlp_tp_group(backend) -> None:
    if not torch.distributed.is_initialized():
        raise RuntimeError("torch.distributed must be initialized")
    world_size: int = torch.distributed.get_world_size()
    mtp_tp_size = model_extra_config.parall_config.dense_mlp_tp_size
    backend = backend or torch.distributed.get_backend(get_world_group().device_group)
    if world_size % mtp_tp_size != 0:
        raise RuntimeError(f"Dense MLP TP Size ({mtp_tp_size}) should be divisible by world size ({world_size})")
    num_groups: int = world_size // mtp_tp_size
    global _MLP_TP
    if _MLP_TP is not None:
        raise RuntimeError("_MLP_TP must be None")
    group_ranks = []
    for i in range(num_groups):
        ranks = list(range(i * mtp_tp_size, (i + 1) * mtp_tp_size))
        group_ranks.append(ranks)

    # message queue broadcaster is only used in tensor model parallel group
    _MLP_TP = init_model_parallel_group(
        group_ranks,
        get_world_group().local_rank,
        backend,
        use_message_queue_broadcaster=True,
        group_name="mlp_tp_group",
    )


def initialize_o_proj_tp_group(backend) -> None:
    # Get world size and rank. Ensure some consistencies.
    if not torch.distributed.is_initialized():
        raise RuntimeError("torch.distributed must be initialized")
    world_size: int = torch.distributed.get_world_size()
    o_proj_tp_size = model_extra_config.parall_config.o_proj_tp_size
    if world_size % o_proj_tp_size != 0:
        raise RuntimeError(f"o_proj TP Size ({o_proj_tp_size}) should be divisible by world size ({world_size})")
    backend = backend or torch.distributed.get_backend(get_world_group().device_group)

    num_local_groups: int = world_size // o_proj_tp_size
    global _O_PROJ_TP
    if _O_PROJ_TP is not None:
        raise RuntimeError("_O_PROJ_TP must be None")
    group_ranks = []
    for i in range(num_local_groups):
        ranks = list(range(i * o_proj_tp_size, (i + 1) * o_proj_tp_size))
        group_ranks.append(ranks)

    # message queue broadcaster is only used in tensor model parallel group
    _O_PROJ_TP = init_model_parallel_group(
        group_ranks,
        get_world_group().local_rank,
        backend,
        use_message_queue_broadcaster=False,
        group_name="o_proj_tp_group",
    )


def initialize_o_proj_dp_group(backend) -> None:
    # Get world size and rank. Ensure some consistencies.
    if not torch.distributed.is_initialized():
        raise RuntimeError("torch.distributed must be initialized")
    world_size: int = torch.distributed.get_world_size()
    o_proj_tp_size = model_extra_config.parall_config.o_proj_tp_size
    backend = backend or torch.distributed.get_backend(get_world_group().device_group)

    dp_size: int = world_size // o_proj_tp_size
    global _O_PROJ_DP
    if _O_PROJ_DP is not None:
        raise RuntimeError("_O_PROJ_DP must be None")
    all_ranks = torch.arange(world_size).reshape(dp_size, o_proj_tp_size)
    group_ranks = all_ranks.transpose(0, 1)
    group_ranks = [x.tolist() for x in group_ranks]
    # message queue broadcaster is only used in tensor model parallel group
    _O_PROJ_DP = init_model_parallel_group(
        group_ranks,
        get_world_group().local_rank,
        backend,
        use_message_queue_broadcaster=False,
        group_name="o_proj_dp_group",
    )


def initialize_local_world_group(backend) -> None:
    # Get world size and rank. Ensure some consistencies.
    if not torch.distributed.is_initialized():
        raise RuntimeError("torch.distributed must be initialized")
    world_size: int = torch.distributed.get_world_size()
    local_size = calculate_effective_local_size(torch.npu.device_count() if not int(os.getenv("NO_NPU_MOCK", "0")) \
        else len(os.getenv("ASCEND_RT_VISIBLE_DEVICES").split(",")), world_size)

    backend = backend or torch.distributed.get_backend(get_world_group().device_group)

    num_local_groups: int = world_size // local_size
    global _LOCAL_WORLD
    if _LOCAL_WORLD is not None:
        raise RuntimeError("_LOCAL_WORLD must be None")
    group_ranks = []
    for i in range(num_local_groups):
        ranks = list(range(i * local_size, (i + 1) * local_size))
        group_ranks.append(ranks)

    # message queue broadcaster is only used in tensor model parallel group
    _LOCAL_WORLD = init_model_parallel_group(
        group_ranks,
        get_world_group().local_rank,
        backend,
        use_message_queue_broadcaster=True,
        group_name="world_local",
    )


def initialize_local_comm_group_list(backend) -> None:
    # Get world size and rank. Ensure some consistencies.
    if not torch.distributed.is_initialized():
        raise RuntimeError("torch.distributed must be initialized")
    world_size: int = torch.distributed.get_world_size()
    local_size = calculate_effective_local_size(torch.npu.device_count() if not int(os.getenv("NO_NPU_MOCK", "0")) \
        else len(os.getenv("ASCEND_RT_VISIBLE_DEVICES").split(",")), world_size)

    backend = backend or torch.distributed.get_backend(get_world_group().device_group)

    num_local_groups: int = world_size // local_size
    global _LOCAL_COMM_LIST
    if _LOCAL_COMM_LIST is not None:
        raise RuntimeError("_LOCAL_COMM_LIST must be None")
    _LOCAL_COMM_LIST = list()
    group_ranks = []
    for i in range(num_local_groups):
        ranks = list(range(i * local_size, (i + 1) * local_size))
        group_ranks.append(ranks)

    # message queue broadcaster is only used in tensor model parallel group
    comm_group_per_server = 1
    total_comm_groups = comm_group_per_server * num_local_groups + _NUM_COMM_GROUP # one group for topk and the other is redundant
    for i in range(total_comm_groups):
        _LOCAL_COMM_LIST.append(
            init_model_parallel_group(
                group_ranks,
                get_world_group().local_rank,
                backend,
                use_message_queue_broadcaster=True,
                group_name="world_local",
            )
        )


def initialize_cross_comm_group_list(backend) -> None:
    # Get world size and rank. Ensure some consistencies.
    if not torch.distributed.is_initialized():
        raise RuntimeError("torch.distributed must be initialized")
    world_size: int = torch.distributed.get_world_size()
    local_size = calculate_effective_local_size(torch.npu.device_count() if not int(os.getenv("NO_NPU_MOCK", "0")) \
        else len(os.getenv("ASCEND_RT_VISIBLE_DEVICES").split(",")), world_size)

    server_size = world_size // local_size

    backend = backend or torch.distributed.get_backend(get_world_group().device_group)

    # Build the pipeline model-parallel groups.
    num_cross_groups: int = world_size // server_size
    global _CROSS_COMM_LIST
    if _CROSS_COMM_LIST is not None:
        raise RuntimeError("pipeline model parallel group is already initialized")
    _CROSS_COMM_LIST = list()
    group_ranks = []
    for i in range(num_cross_groups):
        ranks = list(range(i, world_size, num_cross_groups))
        group_ranks.append(ranks)
    # pipeline parallel does not need custom allreduce

    for i in range(_NUM_COMM_GROUP):
        _CROSS_COMM_LIST.append(
            init_model_parallel_group(
                group_ranks,
                get_world_group().local_rank,
                backend,
                group_name="world_cross",
            )
        )


def initialize_world_comm_group_list(backend) -> None:
    # Get world size and rank. Ensure some consistencies.
    if not torch.distributed.is_initialized():
        raise RuntimeError("torch.distributed must be initialized")
    world_size: int = torch.distributed.get_world_size()

    backend = backend or torch.distributed.get_backend(get_world_group().device_group)

    global _GLOBAL_COMM_LIST
    if _GLOBAL_COMM_LIST is not None:
        raise RuntimeError("_GLOBAL_COMM_LIST must be None")
    _GLOBAL_COMM_LIST = list()
    group_ranks = [range(world_size)]
    for i in range(_NUM_COMM_GROUP):
        _GLOBAL_COMM_LIST.append(
            init_model_parallel_group(
                group_ranks,
                get_world_group().local_rank,
                backend,
                use_message_queue_broadcaster=True,
                group_name="world_local",
            )
        )


def get_mlp_tp_group() -> GroupCoordinator:
    return _MLP_TP


def get_o_proj_tp_group() -> GroupCoordinator:
    return _O_PROJ_TP


def get_o_proj_dp_group() -> GroupCoordinator:
    return _O_PROJ_DP


def get_local_world_group() -> GroupCoordinator:
    return _LOCAL_WORLD

def get_scale_parallel_group() -> GroupCoordinator:
    return _SCALE_PARALLEL_GROUP

def get_local_group_from_list(idx: int) -> GroupCoordinator:
    return _LOCAL_COMM_LIST[idx]


def get_cross_group_from_list(idx: int) -> GroupCoordinator:
    return _CROSS_COMM_LIST[idx]


def get_world_group_from_list(idx: int) -> GroupCoordinator:
    return _GLOBAL_COMM_LIST[idx]


def get_local_group_world_size_from_list(idx: int):
    return _LOCAL_COMM_LIST[idx].world_size


def get_local_group_rank_from_list(idx: int):
    return _LOCAL_COMM_LIST[idx].rank_in_group


def get_near_cross_group_from_list(idx: int) -> GroupCoordinator:
    return _CROSS_NEAR_COMM_LIST[idx]


def get_far_cross_group_from_list(idx: int) -> GroupCoordinator:
    return _CROSS_FAR_COMM_LIST[idx]


def get_local_group_size():
    return get_local_group_from_list(idx=0).world_size


def get_local_group_rank():
    return get_local_group_from_list(idx=0).rank_in_group


def initialize_round_cross_comm_group_list(backend) -> None:
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size: int = torch.distributed.get_world_size()

    local_size = get_npu_device_count()
    assert world_size % local_size == 0

    server_size = world_size // local_size

    backend = backend or torch.distributed.get_backend(
        get_world_group().device_group)

    num_cross_groups: int = (world_size // server_size)
    global _CROSS_ROUND_COMM_LIST
    assert _CROSS_ROUND_COMM_LIST is None, (
        "pipeline model parallel group is already initialized")
    _CROSS_ROUND_COMM_LIST = list()

    group_ranks_round0 = []
    group_ranks_round1 = []
    group_ranks_round2 = []
    for i in range(num_cross_groups):
        ranks = [[i + 0 * num_cross_groups, i + 1 * num_cross_groups], \
                [i + 2 * num_cross_groups, i + 3 * num_cross_groups]]
        group_ranks_round0.extend(ranks)

        ranks = [[i + 0 * num_cross_groups, i + 2 * num_cross_groups], \
                [i + 1 * num_cross_groups, i + 3 * num_cross_groups]]
        group_ranks_round1.extend(ranks)

        ranks = [[i + 0 * num_cross_groups, i + 3 * num_cross_groups], \
                [i + 1 * num_cross_groups, i + 2 * num_cross_groups]]
        group_ranks_round2.extend(ranks)


    _CROSS_ROUND_COMM_LIST.append(init_model_parallel_group(group_ranks_round0,
                                    get_world_group().local_rank,
                                    backend,
                                    group_name="world_round0_cross"))

    _CROSS_ROUND_COMM_LIST.append(init_model_parallel_group(group_ranks_round1,
                                    get_world_group().local_rank,
                                    backend,
                                    group_name="world_round1_cross"))

    _CROSS_ROUND_COMM_LIST.append(init_model_parallel_group(group_ranks_round2,
                                    get_world_group().local_rank,
                                    backend,
                                    group_name="world_round2_cross"))


def get_round_cross_group_from_list(round: int) -> GroupCoordinator:
    return _CROSS_ROUND_COMM_LIST[round]


def initialize_far_cross_comm_group_list(backend) -> None:
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size: int = torch.distributed.get_world_size()

    local_size  = get_npu_device_count()
    assert world_size % local_size == 0

    server_size = world_size // local_size

    backend = backend or torch.distributed.get_backend(
        get_world_group().device_group)

    num_cross_groups: int = (world_size // server_size)
    global _CROSS_FAR_COMM_LIST
    assert _CROSS_FAR_COMM_LIST is None, (
        "pipeline model parallel group is already initialized")
    _CROSS_FAR_COMM_LIST = list()
    group_ranks = []
    for i in range(num_cross_groups):
        for j in range(server_size // 2):
            ranks = list(range(i + j * num_cross_groups, world_size, world_size // 2))
            group_ranks.append(ranks)

    for i in range(_NUM_COMM_GROUP):
        _CROSS_FAR_COMM_LIST.append(init_model_parallel_group(group_ranks,
                                    get_world_group().local_rank,
                                    backend,
                                    group_name="world_far_cross"))


def initialize_near_cross_comm_group_list(backend) -> None:
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size: int = torch.distributed.get_world_size()

    local_size = get_npu_device_count()
    assert world_size % local_size == 0

    server_size = world_size // local_size

    backend = backend or torch.distributed.get_backend(
        get_world_group().device_group)

    num_cross_groups: int = (world_size // server_size)
    global _CROSS_NEAR_COMM_LIST
    assert _CROSS_NEAR_COMM_LIST is None, (
        "pipeline model parallel group is already initialized")
    _CROSS_NEAR_COMM_LIST = list()
    group_ranks = []
    for i in range(num_cross_groups):
        ranks = list(range(i, world_size // 2, num_cross_groups))
        group_ranks.append(ranks)

        ranks = list(range(world_size // 2 + i, world_size, num_cross_groups))
        group_ranks.append(ranks)

    for i in range(_NUM_COMM_GROUP):
        _CROSS_NEAR_COMM_LIST.append(init_model_parallel_group(group_ranks,
                                    get_world_group().local_rank,
                                    backend,
                                    group_name="world_near_cross"))