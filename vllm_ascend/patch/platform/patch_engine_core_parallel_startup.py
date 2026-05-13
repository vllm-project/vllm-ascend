#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Portions of `_AscendCoreEngineProcManagerBackport` mirror vLLM
# CoreEngineProcManager startup logic under Apache License 2.0.
#
"""Backport parallel EngineCore process startup when upstream lacks it.

Aligns behavior with vLLM commit 10189846215751d9c4eb1b8b94e86e9d2940f877
(CoreEngineProcManager.__init__, _run_async_startup, _start_processes_async).

If vllm.v1.engine.utils.CoreEngineProcManager already defines _run_async_startup,
this patch is a no-op.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
import weakref
from multiprocessing import connection
from multiprocessing.process import BaseProcess
from multiprocessing.queues import Queue
from typing import Any, cast

import vllm.v1.engine.utils as engine_utils
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils import numa_utils
from vllm.utils.system_utils import get_mp_context
from vllm.v1.executor import Executor
from vllm.v1.utils import shutdown as engine_shutdown_utils

logger = init_logger(__name__)

# Mirrors vllm v1/engine/utils.py
_ASYNC_STARTUP_ENGINE_THRESHOLD = 1


class _AscendCoreEngineProcManagerBackport:
    """
    Duplicate of upstream CoreEngineProcManager async-startup semantics for older vLLM.
    Keep synced with vllm.v1.engine.utils.CoreEngineProcManager when patching is active.
    """

    def __init__(
        self,
        local_engine_count: int,
        start_index: int,
        local_start_index: int,
        vllm_config: VllmConfig,
        local_client: bool,
        handshake_address: str,
        executor_class: type[Executor],
        log_stats: bool,
        client_handshake_address: str | None = None,
        tensor_queue: Queue | None = None,
    ):
        context = get_mp_context()
        common_kwargs: dict[str, Any] = {
            "vllm_config": vllm_config,
            "local_client": local_client,
            "handshake_address": handshake_address,
            "executor_class": executor_class,
            "log_stats": log_stats,
            "tensor_queue": tensor_queue,
        }

        if client_handshake_address:
            common_kwargs["client_handshake_address"] = client_handshake_address

        from vllm.v1.engine.core import EngineCoreProc

        data_parallel = vllm_config.parallel_config.data_parallel_size > 1
        need_env_control = data_parallel and (
            not current_platform.is_cuda_alike() or vllm_config.parallel_config.use_ray
        )
        if need_env_control:
            evar = current_platform.device_control_env_var
            world_size = vllm_config.parallel_config.world_size
            local_world_size = vllm_config.parallel_config.local_world_size

        get_device_indices = engine_utils.get_device_indices

        self.processes: list[BaseProcess] = []
        local_dp_ranks: list[int] = []
        for index in range(local_engine_count):
            local_index = local_start_index + index
            global_index = start_index + index

            local_dp_ranks.append(local_index)
            self.processes.append(
                context.Process(
                    target=(
                        engine_utils._enginecore_bootstrap
                        if need_env_control
                        else EngineCoreProc.run_engine_core
                    ),
                    name=(f"EngineCore_DP{global_index}" if data_parallel else "EngineCore"),
                    kwargs=(
                        {
                            "evar": evar,
                            "value": get_device_indices(
                                evar, local_index, world_size, local_world_size
                            ),
                            "target_fn": EngineCoreProc.run_engine_core,
                            "target_kwargs": common_kwargs
                            | {
                                "dp_rank": global_index,
                                "local_dp_rank": local_index,
                            },
                        }
                        if need_env_control
                        else common_kwargs
                        | {
                            "dp_rank": global_index,
                            "local_dp_rank": local_index,
                        }
                    ),
                )
            )

        self._finalizer = weakref.finalize(self, engine_shutdown_utils, self.processes)
        self.manager_stopped = threading.Event()
        self.failed_proc_name: str | None = None

        use_async_startup = local_engine_count > _ASYNC_STARTUP_ENGINE_THRESHOLD
        if use_async_startup:
            logger.info(
                "Using async parallel startup for %d EngineCore processes "
                "(Ascend patch backport).",
                local_engine_count,
            )
            try:
                self._run_async_startup(vllm_config, local_dp_ranks)
                logger.info(
                    "All %d EngineCore processes started successfully.",
                    local_engine_count,
                )
            finally:
                if self.finished_procs():
                    self.shutdown()
        else:
            try:
                for proc, local_dp_rank in zip(self.processes, local_dp_ranks):
                    with numa_utils.configure_subprocess(
                        vllm_config,
                        local_rank=0,
                        dp_local_rank=local_dp_rank,
                        process_kind="EngineCore",
                    ):
                        proc.start()
            finally:
                if self.finished_procs():
                    self.shutdown()

    def _run_async_startup(
        self,
        vllm_config: VllmConfig,
        local_dp_ranks: list[int],
    ) -> None:
        try:
            asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    asyncio.run,
                    self._start_processes_async(vllm_config, local_dp_ranks),
                )
                future.result()
        except RuntimeError:
            asyncio.run(self._start_processes_async(vllm_config, local_dp_ranks))

    async def _start_processes_async(
        self,
        vllm_config: VllmConfig,
        local_dp_ranks: list[int],
    ) -> None:
        async def _start_one(proc: BaseProcess, local_dp_rank: int) -> None:
            def _start_with_numa() -> None:
                with numa_utils.configure_subprocess(
                    vllm_config,
                    local_rank=0,
                    dp_local_rank=local_dp_rank,
                    process_kind="EngineCore",
                ):
                    proc.start()

            await asyncio.to_thread(_start_with_numa)

        await asyncio.gather(
            *(
                _start_one(proc, rank)
                for proc, rank in zip(self.processes, local_dp_ranks)
            )
        )

    def shutdown(self, timeout: float | None = None) -> None:
        self.manager_stopped.set()
        if self._finalizer.detach() is not None:
            engine_shutdown_utils(self.processes, timeout=timeout)

    def monitor_engine_liveness(self) -> None:
        sentinel_to_proc = {proc.sentinel: proc for proc in self.processes}
        sentinels = set(sentinel_to_proc.keys())

        while sentinels and not self.manager_stopped.is_set():
            died_sentinels = connection.wait(sentinels, timeout=1)

            for sentinel in died_sentinels:
                proc = sentinel_to_proc.pop(cast(int, sentinel))
                exitcode = proc.exitcode
                if exitcode != 0 and not self.manager_stopped.is_set():
                    self.failed_proc_name = proc.name
            if died_sentinels:
                break

        self.shutdown()

    def sentinels(self) -> list:
        return [proc.sentinel for proc in self.processes]

    def finished_procs(self) -> dict[str, int]:
        return {
            proc.name: proc.exitcode
            for proc in self.processes
            if proc.exitcode is not None
        }


def _apply_core_engine_proc_manager_patch() -> None:
    mgr = engine_utils.CoreEngineProcManager
    if hasattr(mgr, "_run_async_startup"):
        logger.debug(
            "CoreEngineProcManager already provides async startup; "
            "skipping Ascend backport."
        )
        return

    logger.warning(
        "vLLM CoreEngineProcManager has no _run_async_startup; "
        "applying Ascend backport from commit 10189846215751d9c4eb1b8b94e86e9d2940f877 "
        "semantics. Pin a newer vLLM to drop this shim."
    )
    engine_utils.CoreEngineProcManager = _AscendCoreEngineProcManagerBackport


_apply_core_engine_proc_manager_patch()
