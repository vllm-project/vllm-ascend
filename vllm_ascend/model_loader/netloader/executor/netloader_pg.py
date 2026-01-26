#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

import gc
import ipaddress
from datetime import timedelta
from typing import Any, Optional

import torch
import torch_npu
from torch._C._distributed_c10d import (_DEFAULT_PG_TIMEOUT,
                                        _register_process_group,
                                        _unregister_process_group)
from torch.distributed import ProcessGroup, is_hccl_available
from torch.distributed.distributed_c10d import (Backend, BackendConfig,
                                                PrefixStore, _world)
from torch.distributed.rendezvous import rendezvous
from torch_npu._C._distributed_c10d import ProcessGroupHCCL
from vllm.logger import logger


def stateless_init_process_group(
    host: str,
    port: int,
    world_size: int,
    rank: int,
    timeout: timedelta = _DEFAULT_PG_TIMEOUT,
    group_name: str = "",
    pg_options: Optional[Any] = None,
) -> ProcessGroup:
    """
    Initializes a stateless process group.

    Args:
    host: Hostname.
    port: Port number.
    world_size: Size of the process group.
    rank: Rank of the current process.
    timeout: Timeout duration, defaults to _DEFAULT_PG_TIMEOUT.
    group_name: Name of the process group, defaults to an empty string.
    pg_options: Options for the process group, defaults to None.

    Returns:
    ProcessGroup: The initialized process group.

    Raises:
    RuntimeError: If world_size is not positive, or if rank is not within [0, world_size - 1], or if HCCL is unavailable.
    TypeError: If timeout is not a timedelta type.
    ValueError: If group_name already exists.
    """

    # Check if world_size is positive
    if not world_size > 0:
        raise RuntimeError("world_size must be positive")
    # Check if rank is within [0, world_size - 1]
    if not (rank >= 0 and rank <= world_size - 1):
        raise RuntimeError(
            "rank should be a number between 0 and ``world_size``-1")
    # Check if HCCL is available
    if not is_hccl_available():
        raise RuntimeError("HCCL is not available")
    # Check if timeout is a timedelta type
    if not isinstance(timeout, timedelta):
        raise TypeError(
            f"Expected timeout argument to be of type datetime.timedelta, got {timeout}"
        )
    # Check if group_name already exists
    if group_name in _world.pg_names.values():
        raise ValueError(
            f"The specified group name {group_name} has already been "
            "created, please use a different group name")

    # Function to check if an IPv6 address is valid
    def is_valid_ipv6_address(address: str) -> bool:
        try:
            ipaddress.IPv6Address(address)
            return True
        except ValueError:
            return False

    # Function to get TCP URI
    def get_tcp_uri(ip: str, port: int) -> str:
        if is_valid_ipv6_address(ip):
            return f"tcp://[{ip}]:{port}"
        else:
            return f"tcp://{ip}:{port}"

    # Get initialization method
    init_method = get_tcp_uri(host, port)
    # Create Backend object
    backend = Backend('hccl')
    # Use rendezvous function to get store, rank, and world_size
    store, rank, world_size = next(
        rendezvous(init_method, rank, world_size, timeout=timeout))

    # Set timeout for store
    store.set_timeout(timeout)
    # Create PrefixStore object
    prefix_store = PrefixStore(f"{init_method}/{group_name}/", store)
    # Set group_rank and group_size
    group_rank = rank
    group_size = world_size
    # Create ProcessGroup object
    pg: ProcessGroup = ProcessGroup(
        prefix_store,
        group_rank,
        group_size,
    )
    # Create BackendConfig object
    backend_config = BackendConfig(backend)
    # Set default backend for ProcessGroup
    pg._set_default_backend(Backend.backend_type_map[backend])

    # Check if pg_options is None or not of type ProcessGroupHCCL.Options
    if pg_options is None or not isinstance(
            pg_options,
            torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options):
        pg_options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
    # Set attributes for pg_options
    pg_options.is_high_priority_stream = False
    pg_options._timeout = timeout
    pg_options.global_ranks_in_group = []
    pg_options.group_id = f"{init_method}/{group_name}/"
    # Create ProcessGroupHCCL object
    backend_class = ProcessGroupHCCL(prefix_store, group_rank, group_size,
                                     pg_options)
    # Set sequence number for backend_class
    backend_class._set_sequence_number_for_group()
    # Set backend_type
    backend_type = ProcessGroup.BackendType.CUSTOM
    # Register backend
    pg._register_backend(torch.device("npu"), backend_type, backend_class)

    # Set group_desc and pg_tag
    group_desc = "undefined"
    assert group_name is not None
    assert group_desc is not None
    pg._set_group_name(group_name)
    pg._set_group_desc(group_desc)

    # Update attributes in _world
    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}
    _world.pg_map[pg] = (backend, prefix_store)
    _world.pg_names[pg] = group_name
    _register_process_group(group_name, pg)
    _world.pg_backend_config[pg] = str(backend_config)
    return pg


def destroy_stateless_process_group(pg: ProcessGroup, manual_gc: bool = False):
    """
    Destroy a stateless process group.

    Args:
    pg: Process group to be destroyed.
    manual_gc: Whether to manually perform garbage collection, defaults to False.
    """
    # Shutdown the process group
    pg.shutdown()
    # Remove related attributes from _world
    _world.pg_map.pop(pg, None)
    _world.pg_names.pop(pg, None)
    _world.pg_group_ranks.pop(pg, None)
    _world.pg_backend_config.pop(pg, None)
    # Check if pg is in keys of _world.pg_coalesce_state
    if pg in _world.pg_coalesce_state.keys():
        logger.warning("Some coalesced collectives haven't been launched when "
                       "ProcessGroup is destroyed. They will be cleaned.")
        del _world.pg_coalesce_state[pg]
    # Unregister the process group
    _unregister_process_group(pg.group_name)

    # If manual_gc is True, perform garbage collection
    if manual_gc:
        gc.collect()
