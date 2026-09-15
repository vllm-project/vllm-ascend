# SPDX-License-Identifier: Apache-2.0
"""Coordinate stateless DP group ports for local multiprocessing.

vLLM 0.23.0 pre-allocates ports before spawning local DP EngineCore
processes. Another process can bind one of those ports in the meantime. Its
per-rank EADDRINUSE retry then sends rank 0 to a new port while the other ranks
keep waiting on the old one.
"""

from typing import Any

from vllm.distributed.utils import create_tcp_store
from vllm.v1.engine import utils as engine_utils

_ORIGINAL_CORE_ENGINE_PROC_MANAGER_INIT = engine_utils.CoreEngineProcManager.__init__


def _create_local_dp_coord_store(local_engine_count: int, vllm_config: Any):
    parallel_config = vllm_config.parallel_config
    if (
        parallel_config.data_parallel_size <= 1
        or local_engine_count != parallel_config.data_parallel_size
        or parallel_config._coord_store_port
    ):
        return None

    store = create_tcp_store(
        parallel_config.data_parallel_master_ip,
        0,
        is_master=True,
        world_size=-1,
        wait_for_workers=False,
    )
    parallel_config._coord_store_port = store.port
    return store


def _patched_core_engine_proc_manager_init(
    self,
    local_engine_count,
    start_index,
    local_start_index,
    vllm_config,
    local_client,
    handshake_address,
    executor_class,
    log_stats,
    client_handshake_address=None,
    tensor_queue=None,
):
    coord_store = _create_local_dp_coord_store(local_engine_count, vllm_config)
    if coord_store is not None:
        # Keep the server alive for stateless groups created after EngineCore
        # startup, including groups initialized by worker subprocesses.
        self._ascend_dp_coord_store = coord_store

    return _ORIGINAL_CORE_ENGINE_PROC_MANAGER_INIT(
        self,
        local_engine_count,
        start_index,
        local_start_index,
        vllm_config,
        local_client,
        handshake_address,
        executor_class,
        log_stats,
        client_handshake_address,
        tensor_queue,
    )


engine_utils.CoreEngineProcManager.__init__ = _patched_core_engine_proc_manager_init
