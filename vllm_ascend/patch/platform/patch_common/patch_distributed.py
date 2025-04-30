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
from torch.distributed import ProcessGroup
from torch.distributed.distributed_c10d import (Backend, PrefixStore,
                                                _get_default_timeout,
                                                is_nccl_available)
from torch.distributed.rendezvous import rendezvous
from vllm.config import ParallelConfig


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


def ascend_stateless_init_torch_distributed_process_group(
        host: str, port: int, rank: int, world_size: int,
        backend: str) -> ProcessGroup:
    """
    A replacement for `torch.distributed.init_process_group` that does not
    pollute the global state. The created ProcessGroup object can be used for
    some operations such as `allreduce`, because it does not depend on the
    global rank. However, some operations such as `broadcast` cannot be used
    because it depends on the global rank.

    # TODO: ask for help from PyTorch team if we need the `broadcast` operation.

    This function is useful when we are not sure about the total number of
    processes in the process group. For example, we may have process
    1, 2, ..., 8 who want to communicate, and process 9 might be the same
    process as process 1, or it might be a different process; process 10
    might be the same process as process 5, or it might be a different process.
    In this case, how can we reliably form a communication channel within
    process 9 and 10, without affecting the communication channel within
    process 1, 2, ..., 8?

    One possible solution is to figure out if process 9 and 10 are the same
    as process 1 and 5 beforehand, and then form a communication channel
    based on the information, adjusting the ranks and world_size etc. However,
    figuring out the information is not always easy, and it will interfere
    with the main communication channel.

    Our solution is to always form a communication channel with process 1, 2,
    ..., 8, and then use this function to form another communication channel
    with process 9 and 10. This way, regardless of whether process 9 and 10
    are the same as process 1 and 5, the main communication channel is
    always formed with process 1, 2, ..., 8, and the additional communication
    channel is formed with process 9 and 10.
    """
    init_method = f"tcp://{host}:{port}"
    backend = Backend(backend)  # it is basically string
    timeout = _get_default_timeout(backend)

    store, rank, world_size = next(
        rendezvous(init_method, rank, world_size, timeout=timeout))
    store.set_timeout(timeout)

    group_rank = rank
    group_size = world_size

    # Use a PrefixStore to avoid accidental overrides of keys used by
    # different systems (e.g. RPC) in case the store is multi-tenant.
    prefix_store = PrefixStore(init_method, store)

    pg: ProcessGroup = ProcessGroup(
        prefix_store,
        group_rank,
        group_size,
    )
    if backend == "gloo":
        from torch.distributed.distributed_c10d import ProcessGroupGloo
        backend_class = ProcessGroupGloo(prefix_store,
                                         group_rank,
                                         group_size,
                                         timeout=timeout)
        backend_type = ProcessGroup.BackendType.GLOO
        device = torch.device("cpu")
    elif backend == "nccl":
        assert is_nccl_available()
        from torch.distributed.distributed_c10d import ProcessGroupNCCL

        backend_options = ProcessGroupNCCL.Options()
        backend_options._timeout = timeout

        backend_class = ProcessGroupNCCL(prefix_store, group_rank, group_size,
                                         backend_options)
        backend_type = ProcessGroup.BackendType.NCCL
        device = torch.device("cuda")
    elif backend == "hccl":
        from torch.distributed import is_hccl_available
        assert is_hccl_available()
        from torch_npu._C._distributed_c10d import ProcessGroupHCCL
        backend_options = ProcessGroupHCCL.Options()
        backend_options._timeout = timeout
        backend_class = ProcessGroupHCCL(prefix_store, group_rank, group_size,
                                         backend_options)
        device = torch.device("npu")
        backend_class._set_sequence_number_for_group()
        backend_type = ProcessGroup.BackendType.CUSTOM
        pg._register_backend(device, backend_type, backend_class)
        return pg
    else:
        raise RuntimeError(f"Unsupported torch distributed backend: {backend}")

    pg._set_default_backend(backend_type)
    backend_class._set_sequence_number_for_group()

    pg._register_backend(device, backend_type, backend_class)

    return pg


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
    import os

    # NOTE: Get port from envs directly when using torchrun
    port = int(os.environ.get("MASTER_PORT", answer))  # type: ignore
    return port


def ascend_stateless_init_dp_group(self) -> "ProcessGroup":
    from vllm.distributed.utils import \
        stateless_init_torch_distributed_process_group

    dp_group = stateless_init_torch_distributed_process_group(
        self.data_parallel_master_ip,
        self.get_next_dp_init_port(),
        self.data_parallel_rank,
        self.data_parallel_size,
        backend="hccl")

    return dp_group


vllm.distributed.parallel_state.destroy_model_parallel = ascend_destroy_model_parallel
vllm.distributed.stateless_init_torch_distributed_process_group = ascend_stateless_init_torch_distributed_process_group
ParallelConfig.get_next_dp_init_port = parallel_config_get_dp_port
ParallelConfig.stateless_init_dp_group = ascend_stateless_init_dp_group
