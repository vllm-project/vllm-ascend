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

from datetime import timedelta
from typing import Any, Optional
import ipaddress
import gc

import torch
import torch_npu
from torch.distributed import ProcessGroup, is_hccl_available
from torch.distributed.rendezvous import rendezvous
from torch.distributed.distributed_c10d import (
    Backend, 
    PrefixStore, 
    _world,
    BackendConfig
)
from torch._C._distributed_c10d import (
    _DEFAULT_PG_TIMEOUT,
    _register_process_group,
    _unregister_process_group
)
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

    if not world_size > 0:
        raise RuntimeError("world_size must be positive")
    if not (rank >= 0 and rank <= world_size - 1):
        raise RuntimeError("rank should be a number between 0 and ``world_size``-1")
    if not is_hccl_available():
        raise RuntimeError("HCCL is not available")
    if not isinstance(timeout, timedelta):
        raise TypeError(
            f"Expected timeout argument to be of type datetime.timedelta, got {timeout}"
        )
    if group_name in _world.pg_names.values():
        raise ValueError(
            f"The specified group name {group_name} has already been "
            "created, please use a different group name"
        )

    def is_valid_ipv6_address(address: str) -> bool:
        try:
            ipaddress.IPv6Address(address)
            return True
        except ValueError:
            return False

    def get_tcp_uri(ip: str, port: int) -> str:
        if is_valid_ipv6_address(ip):
            return f"tcp://[{ip}]:{port}"
        else:
            return f"tcp://{ip}:{port}"


    init_method = get_tcp_uri(host, port)
    backend = Backend('hccl')
    store, rank, world_size = next(rendezvous(init_method, rank, world_size, timeout=timeout))

    store.set_timeout(timeout)
    prefix_store = PrefixStore(f"{init_method}/{group_name}/", store)
    group_rank = rank
    group_size = world_size
    pg: ProcessGroup = ProcessGroup(
            prefix_store,
            group_rank,
            group_size,
        )
    backend_config = BackendConfig(backend)
    pg._set_default_backend(Backend.backend_type_map[backend])

    if pg_options is None or not isinstance(pg_options, torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options):
        pg_options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
    pg_options.is_high_priority_stream = False
    pg_options._timeout = timeout
    pg_options.global_ranks_in_group = []
    pg_options.group_id = f"{init_method}/{group_name}/"
    backend_class = ProcessGroupHCCL(prefix_store, group_rank, group_size, pg_options)
    backend_class._set_sequence_number_for_group()
    backend_type = ProcessGroup.BackendType.CUSTOM
    pg._register_backend(torch.device("npu"), backend_type, backend_class)

    group_desc = "undefined"
    pg_tag = None
    assert group_name is not None
    assert group_desc is not None
    pg._set_group_name(group_name)
    pg._set_group_desc(group_desc)

    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}
    _world.pg_map[pg] = (backend, prefix_store)
    _world.pg_names[pg] = group_name
    _register_process_group(group_name, pg)
    _world.pg_backend_config[pg] = str(backend_config)
    if pg_tag in [None, ""]:
        pg_tag = f"ptd:{group_name}"
        _world.tags_to_pg.setdefault("", []).append(pg)
    else:
        pg_tag = f"user:{pg_tag}"

    _world.tags_to_pg.setdefault(pg_tag, []).append(pg)
    _world.pg_to_tag[pg] = pg_tag

    return pg

def destroy_stateless_process_group(pg: ProcessGroup, manual_gc: bool = False):
    pg.shutdown()
    _world.pg_map.pop(pg, None)
    _world.pg_names.pop(pg, None)
    _world.pg_group_ranks.pop(pg, None)
    _world.pg_backend_config.pop(pg, None)
    if pg in _world.pg_coalesce_state.keys():
        logger.warning(
            "Some coalesced collectives haven't been launched when "
            "ProcessGroup is destroyed. They will be cleaned."
        )
        del _world.pg_coalesce_state[pg]
    tag = _world.pg_to_tag.get(pg)
    if tag is not None:
        del _world.pg_to_tag[pg]
        try:
            _world.tags_to_pg[tag].remove(pg)
            if tag.startswith("ptd:"):
                _world.tags_to_pg[""].remove(pg)
        except Exception:
            pass
    _unregister_process_group(pg.group_name)
    
    if manual_gc:
        gc.collect()