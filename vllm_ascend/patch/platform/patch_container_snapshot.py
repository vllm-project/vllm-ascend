# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Container snapshot runtime monkey patches for vLLM 0.21.0.

This module keeps the snapshot logic inside vllm-ascend patch hooks instead
of mutating upstream vLLM source files at runtime.
"""

from __future__ import annotations

import asyncio
import gc
import os
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import regex as re
from vllm.logger import logger

_PATCHED = False


def _is_target_version() -> bool:
    try:
        import vllm

        version = getattr(vllm, "__version__", "")
        if version.startswith("0.21."):
            return True
        if version == "0.21.0" or version.startswith("0.21.0"):
            return True

        # Source-tree case: v0.21.0 checkout may report "dev" when
        # vllm/_version.py is not generated. Detect by git tag / commit.
        vllm_file = Path(vllm.__file__).resolve()
        repo_root = vllm_file.parent.parent
        if (repo_root / ".git").exists():
            exact_tag = subprocess.run(
                ["git", "-C", str(repo_root), "describe", "--tags", "--exact-match", "HEAD"],
                text=True,
                capture_output=True,
            )
            if exact_tag.returncode == 0 and exact_tag.stdout.strip() == "v0.21.0":
                return True

            head = subprocess.run(
                ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                text=True,
                capture_output=True,
            )
            if head.returncode == 0 and head.stdout.strip().startswith("ad7125a43"):
                return True

        return False
    except Exception:
        return False


def _patch_protocol() -> None:
    from vllm.engine.protocol import EngineClient

    if "suspend" not in EngineClient.__dict__:

        async def suspend(self, model_save_path=None) -> None:
            raise NotImplementedError

        EngineClient.suspend = suspend  # type: ignore[attr-defined]

    if "resume" not in EngineClient.__dict__:

        async def resume(self, data_parallel_master_ip=None, model_path=None) -> None:
            raise NotImplementedError

        EngineClient.resume = resume  # type: ignore[attr-defined]

    if "device_unlock" not in EngineClient.__dict__:

        async def device_unlock(self) -> None:
            raise NotImplementedError

        EngineClient.device_unlock = device_unlock  # type: ignore[attr-defined]


def _patch_executor_abstract() -> None:
    from vllm.v1.executor.abstract import Executor

    if "suspend" not in Executor.__dict__:

        def suspend(self, model_save_path=None):
            return None

        Executor.suspend = suspend  # type: ignore[attr-defined]

    if "resume" not in Executor.__dict__:

        def resume(self, data_parallel_master_ip: str | None = None, model_path=None):
            return None

        Executor.resume = resume  # type: ignore[attr-defined]


def _patch_async_llm() -> None:
    from vllm.v1.engine.async_llm import AsyncLLM

    if "suspend" not in AsyncLLM.__dict__:

        async def suspend(self, model_save_path=None) -> None:
            await self.engine_core.suspend_async(model_save_path=model_save_path)

        AsyncLLM.suspend = suspend  # type: ignore[attr-defined]

    if "resume" not in AsyncLLM.__dict__:

        async def resume(self, data_parallel_master_ip: str | None = None, model_path=None) -> None:
            await self.engine_core.resume_async(
                data_parallel_master_ip=data_parallel_master_ip,
                model_path=model_path,
            )

        AsyncLLM.resume = resume  # type: ignore[attr-defined]

    if "device_unlock" not in AsyncLLM.__dict__:

        async def device_unlock(self) -> None:
            await self.engine_core.device_unlock_async()

        AsyncLLM.device_unlock = device_unlock  # type: ignore[attr-defined]


def _patch_sleep_api_router() -> None:
    """Mirror vllm snapshot branch's module-level @router.post definitions.

    Routes are registered on the router at import time; attach_router() still
    gates mounting on VLLM_SERVER_DEV_MODE, matching the source branch.
    """
    from fastapi import HTTPException, Request, status
    from fastapi.responses import Response
    from vllm.entrypoints.serve.sleep import api_router  # type: ignore[import-not-found]

    # This module uses `from __future__ import annotations`, so the
    # `raw_request: Request` annotations below are stored as strings. FastAPI
    # resolves them via inspect.signature(..., eval_str=True) against each
    # route function's __globals__ (this module's namespace). Expose Request
    # there so the annotation can be evaluated; fastapi stays lazily imported.
    globals()["Request"] = Request

    router = api_router.router

    if not any(getattr(r, "path", None) == "/suspend" for r in router.routes):

        @router.post("/suspend")
        async def suspend(raw_request: Request):
            model_save_path = raw_request.query_params.get("model_save_path")
            if model_save_path is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Missing required parameter: model_save_path",
                )
            await api_router.engine_client(raw_request).suspend(model_save_path=model_save_path)
            return Response(status_code=200)

    if not any(getattr(r, "path", None) == "/resume" for r in router.routes):

        @router.post("/resume")
        async def resume(raw_request: Request):
            data_parallel_master_ip = raw_request.query_params.get("data_parallel_master_ip")
            model_path = raw_request.query_params.get("model_path")
            if data_parallel_master_ip is None or model_path is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Missing required parameter: data_parallel_master_ip and model_path",
                )
            await api_router.engine_client(raw_request).resume(
                data_parallel_master_ip=data_parallel_master_ip,
                model_path=model_path,
            )
            return Response(status_code=200)

    if not any(getattr(r, "path", None) == "/device_unlock" for r in router.routes):

        @router.post("/device_unlock")
        async def device_unlock(raw_request: Request):
            await api_router.engine_client(raw_request).device_unlock()
            return Response(status_code=200)


def _patch_parallel_state() -> None:
    import torch
    import vllm.distributed.parallel_state as ps

    if not hasattr(ps, "_reset_group_name_registry"):

        def _reset_group_name_registry() -> None:
            ps._group_name_counter.clear()
            ps._groups.clear()

        ps._reset_group_name_registry = _reset_group_name_registry  # type: ignore[attr-defined]

    if not hasattr(ps, "cleanup_dist_env_for_snapshot"):

        def cleanup_dist_env_for_snapshot(shutdown_ray: bool = False):
            ps.destroy_model_parallel()
            ps.logger.info("destroy_model_parallel() end")
            ps.destroy_distributed_environment()
            ps._reset_group_name_registry()
            if shutdown_ray:
                import ray  # type: ignore[import-not-found]

                ray.shutdown()

        ps.cleanup_dist_env_for_snapshot = cleanup_dist_env_for_snapshot  # type: ignore[attr-defined]

    # vllm snapshot branch re-exports cleanup_dist_env_for_snapshot via the
    # `from .parallel_state import *` in vllm/distributed/__init__.py. That
    # star-import already ran at import time, so mirror it manually onto the
    # live vllm.distributed package namespace.
    import vllm.distributed as dist_pkg

    dist_pkg.cleanup_dist_env_for_snapshot = ps.cleanup_dist_env_for_snapshot  # type: ignore[attr-defined]

    group_cls = ps.GroupCoordinator
    if getattr(group_cls, "_container_snapshot_destroy_patched", False):
        return

    def _patched_destroy(self):
        if hasattr(self, "device_group"):
            self.device_group._get_backend(torch.device("npu")).abort_hccl_comm("reinit")
            del self.device_group
        if hasattr(self, "cpu_group"):
            torch.distributed.destroy_process_group(self.cpu_group)
            del self.cpu_group
        if getattr(self, "device_communicator", None) is not None:
            self.device_communicator.destroy()
        if getattr(self, "mq_broadcaster", None) is not None:
            self.mq_broadcaster = None

    group_cls.destroy = _patched_destroy  # type: ignore[assignment]
    group_cls._container_snapshot_destroy_patched = True  # type: ignore[attr-defined]


def _patch_utils() -> None:
    import vllm.utils as vllm_utils

    from vllm_ascend.utils import is_restore

    if not hasattr(vllm_utils, "is_restore"):
        vllm_utils.is_restore = is_restore  # type: ignore[attr-defined]

    if not hasattr(vllm_utils, "get_local_ip"):

        def get_local_ip() -> str:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(0.1)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip

        vllm_utils.get_local_ip = get_local_ip  # type: ignore[attr-defined]


def _patch_engine_core_client() -> None:
    import vllm.v1.engine.core_client as core_client_mod

    if "suspend_async" not in core_client_mod.EngineCoreClient.__dict__:

        async def suspend_async(self, model_save_path=None) -> None:
            raise NotImplementedError

        core_client_mod.EngineCoreClient.suspend_async = suspend_async  # type: ignore[attr-defined]

    if "resume_async" not in core_client_mod.EngineCoreClient.__dict__:

        async def resume_async(self, data_parallel_master_ip: str | None = None, model_path=None) -> None:
            raise NotImplementedError

        core_client_mod.EngineCoreClient.resume_async = resume_async  # type: ignore[attr-defined]

    if "device_unlock_async" not in core_client_mod.EngineCoreClient.__dict__:

        async def device_unlock_async(self) -> None:
            raise NotImplementedError

        core_client_mod.EngineCoreClient.device_unlock_async = device_unlock_async  # type: ignore[attr-defined]

    if not getattr(core_client_mod.AsyncMPClient, "_container_snapshot_init_patched", False):
        original_async_init = core_client_mod.AsyncMPClient.__init__

        def _patched_async_init(
            self,
            vllm_config,
            executor_class,
            log_stats,
            client_addresses=None,
            client_count=1,
            client_index=0,
        ):
            original_async_init(
                self,
                vllm_config,
                executor_class,
                log_stats,
                client_addresses,
                client_count,
                client_index,
            )
            self.is_suspend = False
            self.is_resume = False

        core_client_mod.AsyncMPClient.__init__ = _patched_async_init  # type: ignore[assignment]
        core_client_mod.AsyncMPClient._container_snapshot_init_patched = True  # type: ignore[attr-defined]

    if "wait_for_engines_ready" not in core_client_mod.AsyncMPClient.__dict__:

        async def wait_for_engines_ready(self):
            identities = set(self.core_engines)
            sync_input_socket = core_client_mod.zmq.Socket.shadow(self.input_socket)
            while identities:
                if not sync_input_socket.poll(timeout=core_client_mod.VLLM_ENGINE_READY_TIMEOUT_S * 1000):
                    raise TimeoutError(
                        "[snapshot] Timed out waiting for engines to send initial message on input socket."
                    )
                identity, _ = sync_input_socket.recv_multipart()
                identities.remove(identity)
                logger.info("[snapshot] Engine %s ready. Remaining: %d", identity, len(identities))
            logger.info("[snapshot] api server wait for all engines ready!")

        core_client_mod.AsyncMPClient.wait_for_engines_ready = wait_for_engines_ready  # type: ignore[attr-defined]

    if "suspend_async" not in core_client_mod.AsyncMPClient.__dict__:

        async def suspend_async(self, model_save_path=None) -> None:
            if self.is_suspend:
                logger.warning("[snapshot] api server is already suspend.")
                return
            time_before_suspend = time.perf_counter()
            await self.call_utility_async("suspend", model_save_path)
            self.is_suspend = True
            logger.info(
                "It took %.6f seconds to fall suspend.",
                time.perf_counter() - time_before_suspend,
            )

        core_client_mod.AsyncMPClient.suspend_async = suspend_async  # type: ignore[attr-defined]

    if "device_unlock_async" not in core_client_mod.AsyncMPClient.__dict__:

        async def device_unlock_async(self) -> None:
            if not self.is_suspend:
                logger.warning("[snapshot] api server is not suspend, skip device_unlock.")
                return
            time_before_unlock = time.perf_counter()
            await self.call_utility_async("device_unlock")
            logger.info(
                "It took %.6f seconds to device_unlock.",
                time.perf_counter() - time_before_unlock,
            )

        core_client_mod.AsyncMPClient.device_unlock_async = device_unlock_async  # type: ignore[attr-defined]

    if "resume_async" not in core_client_mod.AsyncMPClient.__dict__:

        async def resume_async(self, data_parallel_master_ip: str | None = None, model_path=None) -> None:
            if not self.is_suspend:
                logger.warning("[snapshot] api server is not suspend.")
                return
            if self.is_resume:
                logger.warning("[snapshot] api server is resuming now.")
                return
            from vllm.utils import is_restore

            if not is_restore():
                logger.warning("[snapshot] api server resume fail, not find /root/.grusflag")
                return
            time_before_resume = time.perf_counter()
            self.is_resume = True
            logger.info("[snapshot] api server wait_for_engines_ready")
            task = asyncio.create_task(self.wait_for_engines_ready())
            await self.call_utility_async("resume", data_parallel_master_ip, model_path)
            await task
            self.is_suspend = False
            self.is_resume = False
            logger.info(
                "It took %.6f seconds to resume.",
                time.perf_counter() - time_before_resume,
            )

        core_client_mod.AsyncMPClient.resume_async = resume_async  # type: ignore[attr-defined]

    if "resume_async" not in core_client_mod.DPAsyncMPClient.__dict__:

        async def dp_resume_async(self, data_parallel_master_ip: str | None = None, model_path=None) -> None:
            if not self.is_suspend:
                logger.warning("[snapshot] api server is not suspend.")
                return
            if self.is_resume:
                logger.warning("[snapshot] api server is resuming now.")
                return
            from vllm.utils import is_restore

            if not is_restore():
                logger.warning("[snapshot] api server resume fail, not find /root/.grusflag")
                return

            time_before_resume = time.perf_counter()
            self.is_resume = True
            if not self.resources.stats_update_task.done():
                self.resources.stats_update_task.cancel()
                try:
                    await self.resources.stats_update_task
                except asyncio.CancelledError:
                    logger.info("[snapshot] api server stats_update_task cancelled successfully")
            self.resources.stats_update_task = None
            self.stats_update_address = re.sub(
                r"\d+\.\d+\.\d+\.\d+",
                data_parallel_master_ip,  # type: ignore[arg-type]
                self.stats_update_address,
            )
            self.first_req_sock_addr = core_client_mod.get_open_zmq_inproc_path()
            self.first_req_send_socket = self.resources.first_req_send_socket = core_client_mod.make_zmq_socket(
                self.ctx, self.first_req_sock_addr, core_client_mod.zmq.PAIR, bind=True
            )
            try:
                asyncio.get_running_loop()
                self._ensure_stats_update_task()
            except RuntimeError:
                logger.error("[snapshot] api server resume_async start stats_update_task failed")
                raise

            logger.info("[snapshot] api server wait_for_engines_ready")
            task = asyncio.create_task(self.wait_for_engines_ready())
            await self.call_utility_async("resume", data_parallel_master_ip, model_path)
            await task
            self.is_suspend = False
            self.is_resume = False
            logger.info(
                "It took %.6f seconds to resume.",
                time.perf_counter() - time_before_resume,
            )

        core_client_mod.DPAsyncMPClient.resume_async = dp_resume_async  # type: ignore[attr-defined]

    if not getattr(core_client_mod.DPAsyncMPClient, "_container_snapshot_stats_task_patched", False):

        def _patched_dp_ensure_stats_update_task(self):
            resources = self.resources
            if resources.stats_update_task is not None:
                return

            assert self.stats_update_address is not None
            stats_addr: str = self.stats_update_address
            assert len(self.engine_ranks_managed) > 0

            async def run_engine_stats_update_task():
                with (
                    core_client_mod.make_zmq_socket(self.ctx, stats_addr, core_client_mod.zmq.XSUB, linger=0) as socket,
                    core_client_mod.make_zmq_socket(
                        self.ctx, self.first_req_sock_addr, core_client_mod.zmq.PAIR, bind=False, linger=0
                    ) as first_req_rcv_socket,
                ):
                    assert isinstance(socket, core_client_mod.zmq.asyncio.Socket)
                    assert isinstance(first_req_rcv_socket, core_client_mod.zmq.asyncio.Socket)
                    self.resources.stats_update_socket = socket
                    self.resources.first_req_rcv_socket = first_req_rcv_socket
                    await socket.send(b"\x01")

                    poller = core_client_mod.zmq.asyncio.Poller()
                    poller.register(socket, core_client_mod.zmq.POLLIN)
                    poller.register(first_req_rcv_socket, core_client_mod.zmq.POLLIN)

                    try:
                        while True:
                            events = await poller.poll()
                            if not self.engines_running and len(events) == 2 or (events[0][0] == first_req_rcv_socket):
                                buf = first_req_rcv_socket.recv(flags=core_client_mod.zmq.NOBLOCK).result()
                                decoded = core_client_mod.msgspec.msgpack.decode(buf)
                                if (
                                    isinstance(decoded, (list, tuple))
                                    and len(decoded) == 2
                                    and decoded[0] == "SCALE_ELASTIC_EP"
                                ):
                                    parallel_config = self.vllm_config.parallel_config
                                    dp_size = parallel_config.data_parallel_size
                                    dp_rank = parallel_config.data_parallel_rank
                                    assert dp_rank == 0
                                    assert dp_size == decoded[1]
                                    assert not (
                                        parallel_config.data_parallel_hybrid_lb
                                        or parallel_config.data_parallel_external_lb
                                    )
                                    self.engine_ranks_managed = list(range(dp_rank, dp_rank + dp_size))
                                    new_engine_count = decoded[1]
                                    if len(self.lb_engines) < new_engine_count:
                                        self.lb_engines = self.lb_engines + [
                                            [0, 0] for _ in range(new_engine_count - len(self.lb_engines))
                                        ]
                                    else:
                                        self.lb_engines = self.lb_engines[:new_engine_count]
                                    scale_msg = core_client_mod.msgspec.msgpack.encode(
                                        ("SCALE_ELASTIC_EP", new_engine_count)
                                    )
                                    await socket.send(scale_msg)
                                    continue

                                assert decoded[0] == "FIRST_REQ"
                                target_eng_index = decoded[1]
                                self.engines_running = True
                                msg = core_client_mod.msgspec.msgpack.encode((target_eng_index, self.current_wave))
                                await socket.send(msg)

                            buf = None
                            while True:
                                future = socket.recv(flags=core_client_mod.zmq.NOBLOCK)
                                if isinstance(future.exception(), core_client_mod.zmq.Again):
                                    break
                                buf = future.result()
                            if buf is None:
                                continue

                            counts, wave, running = core_client_mod.msgspec.msgpack.decode(buf)
                            self.current_wave = wave
                            self.engines_running = running
                            if counts is not None:
                                ranks = self.engine_ranks_managed
                                count_slice = slice(ranks[0], ranks[-1] + 1)
                                sliced_counts = counts[count_slice]
                                self.lb_engines = sliced_counts
                                logger.debug("Received counts: %s (%s)", sliced_counts, count_slice)
                    except asyncio.CancelledError:
                        logger.info("[snapshot] api server stats_update_task raise cancelled")
                        raise

            resources.stats_update_task = asyncio.create_task(run_engine_stats_update_task())

        core_client_mod.DPAsyncMPClient._ensure_stats_update_task = _patched_dp_ensure_stats_update_task  # type: ignore[assignment]
        core_client_mod.DPAsyncMPClient._container_snapshot_stats_task_patched = True  # type: ignore[attr-defined]


def _patch_mp_client() -> None:
    from contextlib import contextmanager

    import vllm.v1.engine.core_client as core_client_mod
    from vllm.v1.engine.utils import launch_core_engines

    if getattr(core_client_mod.MPClient, "_container_snapshot_output_address_patched", False):
        return

    original_init = core_client_mod.MPClient.__init__
    original_launch = launch_core_engines

    class _LaunchCapture:
        output_address: str | None = None

    @contextmanager
    def _capturing_launch(*args, **kwargs):
        with original_launch(*args, **kwargs) as result:
            _LaunchCapture.output_address = result[2].outputs[0]
            yield result

    def _patched_init(self, asyncio_mode, vllm_config, executor_class, log_stats, client_addresses=None):
        import vllm.v1.engine.core_client as cm

        cm.launch_core_engines = _capturing_launch  # type: ignore[assignment]
        try:
            original_init(self, asyncio_mode, vllm_config, executor_class, log_stats, client_addresses)
        finally:
            cm.launch_core_engines = original_launch  # type: ignore[assignment]

        if hasattr(self, "output_address"):
            return
        if client_addresses:
            self.output_address = client_addresses["output_address"]  # type: ignore[attr-defined]
        elif _LaunchCapture.output_address is not None:
            self.output_address = _LaunchCapture.output_address  # type: ignore[attr-defined]

    core_client_mod.MPClient.__init__ = _patched_init  # type: ignore[assignment]
    core_client_mod.MPClient._container_snapshot_output_address_patched = True  # type: ignore[attr-defined]


def _patch_engine_core_init_threads() -> None:
    import vllm.v1.engine.core as core_mod

    if getattr(core_mod.EngineCoreProc, "_container_snapshot_input_thread_patched", False):
        return

    original_init = core_mod.EngineCoreProc.__init__
    original_thread_start = threading.Thread.start

    def _capturing_thread_start(thread_self: threading.Thread, *args, **kwargs):
        # Capture the target before start(): Thread.run() deletes _target once
        # the thread finishes, which could race with this hook.
        target = getattr(thread_self, "_target", None)
        original_thread_start(thread_self, *args, **kwargs)
        if target is None:
            return
        func = getattr(target, "__func__", target)
        if func is core_mod.EngineCoreProc.process_input_sockets:
            # target is the bound EngineCoreProc.process_input_sockets, so
            # target.__self__ is the owning EngineCoreProc instance.
            owner = getattr(target, "__self__", None)
            if owner is not None:
                owner.input_thread = thread_self  # type: ignore[attr-defined]

    def _patched_init(self, *args, **kwargs):
        self.addresses = None  # type: ignore[attr-defined]
        threading.Thread.start = _capturing_thread_start  # type: ignore[method-assign,assignment]
        try:
            original_init(self, *args, **kwargs)
        finally:
            threading.Thread.start = original_thread_start  # type: ignore[method-assign]
        if not hasattr(self, "identity"):
            self.identity = self.engine_index.to_bytes(length=2, byteorder="little")

    core_mod.EngineCoreProc.__init__ = _patched_init  # type: ignore[assignment]
    core_mod.EngineCoreProc._container_snapshot_input_thread_patched = True  # type: ignore[attr-defined]


def _patch_engine_core_socket_threads() -> None:
    from collections import deque
    from contextlib import ExitStack

    import msgspec
    import zmq
    from vllm.utils.network_utils import make_zmq_socket
    from vllm.v1.engine import EngineCoreReadyResponse, EngineCoreRequest, EngineCoreRequestType
    from vllm.v1.engine.core import EngineCoreProc
    from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

    if not hasattr(EngineCoreProc, "ENGINE_CORE_THREAD_FINISH"):
        EngineCoreProc.ENGINE_CORE_THREAD_FINISH = b"ENGINE_CORE_THREAD_FINISH"  # type: ignore[attr-defined]

    if getattr(EngineCoreProc, "_container_snapshot_socket_patched", False):
        return

    def process_input_sockets(
        self,
        input_addresses: list[str],
        coord_input_address: str | None,
        identity: bytes,
        ready_event: threading.Event,
    ):
        add_request_decoder = MsgpackDecoder(EngineCoreRequest, oob_tensor_provider=self.tensor_ipc_receiver)
        generic_decoder = MsgpackDecoder(oob_tensor_provider=self.tensor_ipc_receiver)

        with ExitStack() as stack, zmq.Context() as ctx:  # type: ignore[attr-defined]
            input_sockets = [
                stack.enter_context(
                    make_zmq_socket(ctx, input_address, zmq.DEALER, identity=identity, bind=False)  # type: ignore[attr-defined]
                )
                for input_address in input_addresses
            ]
            if coord_input_address is None:
                coord_socket = None
            else:
                coord_socket = stack.enter_context(
                    make_zmq_socket(
                        ctx,
                        coord_input_address,
                        zmq.XSUB,  # type: ignore[attr-defined]
                        identity=identity,
                        bind=False,
                    )
                )
                coord_socket.send(b"\x01")

            poller = zmq.Poller()  # type: ignore[attr-defined]
            ready_response = EngineCoreReadyResponse(
                max_model_len=self.vllm_config.model_config.max_model_len,
                num_gpu_blocks=self.vllm_config.cache_config.num_gpu_blocks or 0,
                dp_stats_address=self.frontend_stats_publish_address,
            )
            ready_payload = msgspec.msgpack.encode(ready_response)
            for input_socket in input_sockets:
                input_socket.send(ready_payload)
                poller.register(input_socket, zmq.POLLIN)  # type: ignore[attr-defined]

            if coord_socket is not None:
                assert coord_socket.recv() == b"READY"
                poller.register(coord_socket, zmq.POLLIN)  # type: ignore[attr-defined]

            ready_event.set()
            del ready_event
            flag = True
            while flag:
                for input_socket, _ in poller.poll():
                    parts = input_socket.recv_multipart(copy=False)
                    type_frame, *data_frames = parts
                    if type_frame.buffer == b"READY":
                        assert input_socket == coord_socket
                        continue
                    request_type = EngineCoreRequestType(bytes(type_frame.buffer))

                    request: Any
                    if request_type == EngineCoreRequestType.ADD:
                        req: EngineCoreRequest = add_request_decoder.decode(data_frames)
                        try:
                            request = self.preprocess_add_request(req)
                        except Exception:
                            self._handle_request_preproc_error(req)
                            continue
                    else:
                        request = generic_decoder.decode(data_frames)
                        if request_type == EngineCoreRequestType.ABORT:
                            self.aborts_queue.put_nowait(request)

                    self.input_queue.put_nowait((request_type, request))

                    if len(parts) == 2 and (b"resume" in bytes(parts[1].buffer)):
                        logger.info("[snapshot] engine core input thread received resume, stop input thread")
                        flag = False
                        break

    def process_output_sockets(self, output_paths: list[str], coord_output_path: str | None, engine_index: int):
        encoder = MsgpackEncoder()
        reuse_buffers: list[bytearray] = []
        pending = deque[tuple[zmq.MessageTracker, Any, bytearray]]()  # type: ignore[name-defined]

        with ExitStack() as stack, zmq.Context() as ctx:  # type: ignore[attr-defined]
            sockets = [
                stack.enter_context(make_zmq_socket(ctx, output_path, zmq.PUSH, linger=4000))  # type: ignore[attr-defined]
                for output_path in output_paths
            ]
            coord_socket = (
                stack.enter_context(make_zmq_socket(ctx, coord_output_path, zmq.PUSH, bind=False, linger=4000))  # type: ignore[attr-defined]
                if coord_output_path is not None
                else None
            )
            max_reuse_bufs = len(sockets) + 1

            flag = True
            while flag:
                output = self.output_queue.get()
                if output == EngineCoreProc.ENGINE_CORE_THREAD_FINISH:
                    logger.info(
                        "[snapshot] engine core output thread received ENGINE_CORE_THREAD_FINISH, stop output thread"
                    )
                    break
                if output == EngineCoreProc.ENGINE_CORE_DEAD:
                    for socket in sockets:
                        socket.send(output)
                    break
                assert not isinstance(output, bytes)
                client_index, outputs = output
                outputs.engine_index = engine_index

                if client_index == -1:
                    assert coord_socket is not None
                    coord_socket.send_multipart(encoder.encode(outputs))
                    continue

                while pending and pending[-1][0].done:
                    reuse_buffers.append(pending.pop()[2])

                buffer = reuse_buffers.pop() if reuse_buffers else bytearray()
                buffers = encoder.encode_into(outputs, buffer)
                tracker = sockets[client_index].send_multipart(buffers, copy=False, track=True)
                if not tracker.done:
                    ref = outputs if len(buffers) > 1 else None
                    pending.appendleft((tracker, ref, buffer))
                elif len(reuse_buffers) < max_reuse_bufs:
                    reuse_buffers.append(buffer)

    EngineCoreProc.process_input_sockets = process_input_sockets  # type: ignore[assignment]
    EngineCoreProc.process_output_sockets = process_output_sockets  # type: ignore[assignment]
    EngineCoreProc._container_snapshot_socket_patched = True  # type: ignore[attr-defined]


def _patch_engine_core() -> None:
    import vllm.v1.engine.core as core_mod
    from vllm.utils import get_local_ip

    def _snap_log(title: str) -> str:
        return "[snapshot] [engine] " + "-" * 20 + title + "-" * 20

    if not hasattr(core_mod.EngineCoreProc, "ENGINE_CORE_THREAD_FINISH"):
        core_mod.EngineCoreProc.ENGINE_CORE_THREAD_FINISH = b"ENGINE_CORE_THREAD_FINISH"  # type: ignore[attr-defined]

    if "suspend" not in core_mod.EngineCoreProc.__dict__:

        def suspend(self, model_save_path=None):
            logger.info(_snap_log("start dump model"))
            self.collective_rpc("dump_model", args=(model_save_path,))
            logger.info(_snap_log("gc.collect()"))
            gc.collect()
            logger.info(_snap_log("aclrt_snapshot_process_lock"))
            self.collective_rpc("aclrt_snapshot_process_lock")
            logger.info(_snap_log("aclrt_snapshot_process_backup"))
            self.collective_rpc("aclrt_snapshot_process_backup")

        core_mod.EngineCoreProc.suspend = suspend  # type: ignore[attr-defined]

    if "device_unlock" not in core_mod.EngineCoreProc.__dict__:

        def device_unlock(self) -> None:
            logger.info(_snap_log("aclrt_snapshot_process_unlock"))
            self.collective_rpc("aclrt_snapshot_process_unlock")

        core_mod.EngineCoreProc.device_unlock = device_unlock  # type: ignore[attr-defined]

    if "resume" not in core_mod.EngineCoreProc.__dict__:

        def resume(self, data_parallel_master_ip: str | None = None, model_path=None):
            logger.info(_snap_log("stop input and output thread"))
            self.output_queue.put(core_mod.EngineCoreProc.ENGINE_CORE_THREAD_FINISH)
            self.input_thread.join(timeout=9999)
            self.output_thread.join(timeout=9999)

            logger.info(_snap_log("start input and output thread"))
            self.vllm_config.parallel_config.data_parallel_master_ip = data_parallel_master_ip
            if self.addresses.coordinator_input is not None and data_parallel_master_ip is not None:
                self.addresses.coordinator_input = re.sub(
                    r"\d+\.\d+\.\d+\.\d+",
                    data_parallel_master_ip,
                    self.addresses.coordinator_input,
                )
            if self.addresses.coordinator_output is not None and data_parallel_master_ip is not None:
                self.addresses.coordinator_output = re.sub(
                    r"\d+\.\d+\.\d+\.\d+",
                    data_parallel_master_ip,
                    self.addresses.coordinator_output,
                )

            ready_event = threading.Event()
            self.input_thread = threading.Thread(
                target=self.process_input_sockets,
                args=(
                    self.addresses.inputs,
                    self.addresses.coordinator_input,
                    self.identity,
                    ready_event,
                ),
                daemon=True,
            )
            self.input_thread.start()

            self.output_thread = threading.Thread(
                target=self.process_output_sockets,
                args=(
                    self.addresses.outputs,
                    self.addresses.coordinator_output,
                    self.engine_index,
                ),
                daemon=True,
            )
            self.output_thread.start()

            while not ready_event.wait(timeout=10):
                if not self.input_thread.is_alive():
                    raise RuntimeError("Input socket thread died during startup")
                assert self.addresses.coordinator_input is not None
                logger.info("Waiting for READY message from DP Coordinator...")

            logger.info(_snap_log("aclrt_snapshot_process_restore"))
            self.collective_rpc("aclrt_snapshot_process_restore")
            logger.info(_snap_log("aclrt_snapshot_process_unlock"))
            self.collective_rpc("aclrt_snapshot_process_unlock")

            logger.info(_snap_log("update_worker_info_after_resume"))
            local_ip = get_local_ip()
            os.environ["HCCL_IF_IP"] = local_ip
            self.vllm_config.parallel_config.data_parallel_master_ip = data_parallel_master_ip
            self.collective_rpc(
                "update_worker_info_after_resume",
                args=(local_ip, data_parallel_master_ip),
            )

            logger.info(_snap_log("rebuild_parallel_group_after_resume"))
            self.collective_rpc("rebuild_parallel_group_after_resume")

            from vllm.distributed import stateless_destroy_torch_distributed_process_group

            dp_group = getattr(self, "dp_group", None)
            if dp_group is not None:
                logger.info(_snap_log("rebuild engie core dp_group"))
                stateless_destroy_torch_distributed_process_group(dp_group)
                self.vllm_config.parallel_config._data_parallel_master_port_list.clear()
                self.dp_group = self.vllm_config.parallel_config.stateless_init_dp_group()
            else:
                logger.info(
                    "[snapshot] [engine] skip engine-core dp_group rebuild "
                    "(data_parallel_size==1 or non-DPEngineCoreProc)"
                )

            logger.info(_snap_log("re_load_weights"))
            self.collective_rpc("re_load_weights", args=(model_path,))
            logger.info(_snap_log("recapture_graph"))
            self.collective_rpc("recapture_graph")

            # Refresh worker side_channel_host to new pod IP (P and D).
            logger.info(_snap_log("rebuild_kv_transfer_engine_after_resume"))
            self.collective_rpc("rebuild_kv_transfer_engine_after_resume", args=(local_ip,))

            # Refresh scheduler-side KV state in engine core (P and D).
            logger.info(_snap_log("snapshot_refresh_scheduler_after_resume"))
            _refresh_scheduler_after_resume(self, local_ip)

        core_mod.EngineCoreProc.resume = resume  # type: ignore[attr-defined]


def _refresh_scheduler_after_resume(engine_core, local_ip: str) -> None:
    """[snapshot] Refresh scheduler side_channel_host and rotate engine_id (P/D)."""
    kv_cfg = getattr(getattr(engine_core, "vllm_config", None), "kv_transfer_config", None)
    if kv_cfg is None:
        return
    connector_name = getattr(kv_cfg, "kv_connector", "") or ""
    if "Hybrid" in connector_name or not (
        getattr(kv_cfg, "is_kv_producer", False) or getattr(kv_cfg, "is_kv_consumer", False)
    ):
        return

    connector = getattr(getattr(engine_core, "scheduler", None), "connector", None)
    cs = getattr(connector, "connector_scheduler", None) if connector else None
    if cs is None:
        return

    if hasattr(cs, "side_channel_host"):
        old_host = cs.side_channel_host
        cs.side_channel_host = local_ip
        logger.info("[snapshot][rebuild] scheduler side_channel_host %s->%s", old_host, local_ip)

    if hasattr(cs, "engine_id"):
        old_id = str(cs.engine_id)
        new_id = _rotate_snapshot_engine_id(old_id)
        cs.engine_id = new_id
        if connector is not None and hasattr(connector, "engine_id"):
            connector.engine_id = new_id
        kv_cfg.engine_id = new_id
        logger.info("[snapshot][rebuild] scheduler engine_id %s->%s", old_id, new_id)


def _rotate_snapshot_engine_id(engine_id: str) -> str:
    """Replace the process-level uuid in runtime engine_id, keep instance_id prefix."""
    match = re.match(r"^(.+)-([0-9a-f]{32})(_dp\d+)?$", engine_id)
    if match is None:
        logger.warning(
            "[snapshot][rebuild] engine_id %s does not match expected format, append new uuid suffix",
            engine_id,
        )
        return f"{engine_id}-{uuid4().hex}"
    prefix = match.group(1)
    dp_suffix = match.group(3) or ""
    return f"{prefix}-{uuid4().hex}{dp_suffix}"


def _snapshot_run_coordinator(
    engine_count: int,
    front_publish_address: str,
    back_output_address: str,
    back_publish_address: str,
    zmq_addr_pipe=None,
    min_stats_update_interval_ms: int = 100,
    enable_wave_coordination: bool = True,
    advertise_host: str | None = None,
):
    # Defined at module level (not nested in _patch_coordinator) so it is
    # picklable by reference: the DP coordinator runs via the spawn start
    # method, which pickles this function as the Process target. Importing this
    # module in the spawned child triggers _apply_container_snapshot_patches(),
    # so DPCoordinatorProc is patched before the target is resolved/called.
    import vllm.v1.engine.coordinator as coordinator_mod

    if advertise_host is None:
        match = re.search(r"(\d+\.\d+\.\d+\.\d+)", front_publish_address)
        advertise_host = match.group(1) if match else None
    coordinator = coordinator_mod.DPCoordinatorProc(
        engine_count=engine_count,
        min_stats_update_interval_ms=min_stats_update_interval_ms,
        enable_wave_coordination=enable_wave_coordination,
    )
    try:
        coordinator.process_input_socket(
            front_publish_address,
            back_output_address,
            back_publish_address,
            zmq_addr_pipe,
            advertise_host=advertise_host,
        )
    except KeyboardInterrupt:
        logger.info("DP Coordinator process exiting")
    finally:
        if zmq_addr_pipe is not None:
            zmq_addr_pipe.close()


def _patch_coordinator() -> None:
    import vllm.v1.engine.coordinator as coordinator_mod
    from vllm.v1.engine.utils import get_mp_context

    if getattr(coordinator_mod.DPCoordinatorProc, "_container_snapshot_coordinator_patched", False):
        return

    if not hasattr(coordinator_mod.DPCoordinator, "_container_snapshot_init_patched"):
        original_dp_init = coordinator_mod.DPCoordinator.__init__

        def _patched_dp_coordinator_init(self, parallel_config, enable_wave_coordination=True):
            mp_context = get_mp_context()
            original_process = mp_context.Process
            host = parallel_config.data_parallel_master_ip

            # Mirror vllm snapshot branch, which simply adds "advertise_host": host
            # to the kwargs passed to run_coordinator. We can't edit the middle
            # of DPCoordinator.__init__, so temporarily wrap mp_context.Process
            # with a factory that injects advertise_host before delegating to
            # the real Process class. A factory (not a Process subclass) keeps
            # the created proc picklable for the spawn start method.
            def _process_with_advertise_host(*args, **kwargs):
                if kwargs.get("name") == "VLLM_DP_Coordinator":
                    proc_kwargs = dict(kwargs.get("kwargs") or {})
                    proc_kwargs.setdefault("advertise_host", host)
                    kwargs["kwargs"] = proc_kwargs
                return original_process(*args, **kwargs)

            mp_context.Process = _process_with_advertise_host  # type: ignore[assignment]
            try:
                original_dp_init(self, parallel_config, enable_wave_coordination)
            finally:
                mp_context.Process = original_process

        coordinator_mod.DPCoordinator.__init__ = _patched_dp_coordinator_init  # type: ignore[assignment]
        coordinator_mod.DPCoordinator._container_snapshot_init_patched = True  # type: ignore[attr-defined]

    if not hasattr(coordinator_mod.DPCoordinatorProc, "_advertise_zmq_endpoint"):

        def _advertise_zmq_endpoint(endpoint: str, advertise_host: str | None) -> str:
            if advertise_host is None or not endpoint.startswith("tcp://"):
                return endpoint
            return re.sub(r"\d+\.\d+\.\d+\.\d+", advertise_host, endpoint)

        coordinator_mod.DPCoordinatorProc._advertise_zmq_endpoint = staticmethod(  # type: ignore[attr-defined]
            _advertise_zmq_endpoint
        )

    original_process_input_socket = coordinator_mod.DPCoordinatorProc.process_input_socket

    def process_input_socket(
        self,
        front_publish_address: str,
        back_output_address: str,
        back_publish_address: str,
        zmq_addr_pipe=None,
        advertise_host: str | None = None,
    ):
        front_publish_address = re.sub(r"\d+\.\d+\.\d+\.\d+", "0.0.0.0", front_publish_address)
        back_output_address = re.sub(r"\d+\.\d+\.\d+\.\d+", "0.0.0.0", back_output_address)
        back_publish_address = re.sub(r"\d+\.\d+\.\d+\.\d+", "0.0.0.0", back_publish_address)
        logger.info(
            "[snapshot] coordinator bind on 0.0.0.0, advertise tcp as %s",
            advertise_host,
        )

        if zmq_addr_pipe is None:
            return original_process_input_socket(
                self,
                front_publish_address,
                back_output_address,
                back_publish_address,
                zmq_addr_pipe,
            )

        class _AdvertisedPipe:
            def __init__(self, pipe, host: str | None):
                self._pipe = pipe
                self._host = host

            def send(self, endpoints):
                if isinstance(endpoints, tuple) and len(endpoints) == 3:
                    endpoints = tuple(
                        coordinator_mod.DPCoordinatorProc._advertise_zmq_endpoint(ep, self._host) for ep in endpoints
                    )
                return self._pipe.send(endpoints)

            def close(self):
                return self._pipe.close()

        return original_process_input_socket(
            self,
            front_publish_address,
            back_output_address,
            back_publish_address,
            _AdvertisedPipe(zmq_addr_pipe, advertise_host),
        )

    coordinator_mod.DPCoordinatorProc.run_coordinator = staticmethod(_snapshot_run_coordinator)  # type: ignore[assignment]
    coordinator_mod.DPCoordinatorProc.process_input_socket = process_input_socket  # type: ignore[assignment]
    coordinator_mod.DPCoordinatorProc._container_snapshot_coordinator_patched = True  # type: ignore[attr-defined]


def _patch_mla_attention_snapshot_state() -> None:
    import torch
    from vllm.model_executor.layers.attention.mla_attention import MLAAttention

    if getattr(MLAAttention, "_container_snapshot_weights_patched", False):
        return

    original = MLAAttention.process_weights_after_loading

    def _patched(self, act_dtype: torch.dtype):
        original(self, act_dtype)
        for name in ("W_UV", "W_UK_T"):
            tensor = getattr(self, name, None)
            if tensor is None or isinstance(tensor, torch.nn.Parameter):
                continue
            param = torch.nn.Parameter(tensor, requires_grad=False)
            if name in self._parameters:
                self._parameters[name] = param
            else:
                # The unmodified vllm process_weights_after_loading assigns
                # self.W_UV / self.W_UK_T as plain attributes. register_parameter
                # raises KeyError("attribute ... already exists") when the name
                # is already a plain attribute (or buffer) outside _parameters,
                # so drop the existing attribute first. vllm snapshot branch avoids
                # this by assigning to a local var and registering directly.
                delattr(self, name)
                self.register_parameter(name, param)

    MLAAttention.process_weights_after_loading = _patched  # type: ignore[assignment]
    MLAAttention._container_snapshot_weights_patched = True  # type: ignore[attr-defined]


def _patch_rotary_embedding_snapshot_state() -> None:
    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbeddingBase

    if getattr(RotaryEmbeddingBase, "_container_snapshot_cache_patched", False):
        return

    original = RotaryEmbeddingBase.__init__

    def _patched(self, *args, **kwargs):
        original(self, *args, **kwargs)
        if hasattr(self, "cos_sin_cache"):
            cache = self.cos_sin_cache
            if "cos_sin_cache" in self._buffers:
                del self._buffers["cos_sin_cache"]
            self.register_buffer("cos_sin_cache", cache, persistent=True)

    RotaryEmbeddingBase.__init__ = _patched  # type: ignore[assignment]
    RotaryEmbeddingBase._container_snapshot_cache_patched = True  # type: ignore[attr-defined]


def _apply_container_snapshot_patches() -> None:
    global _PATCHED
    if _PATCHED or not _is_target_version():
        return

    _patch_protocol()
    _patch_executor_abstract()
    _patch_async_llm()
    _patch_sleep_api_router()
    _patch_parallel_state()
    _patch_utils()
    _patch_engine_core_socket_threads()
    _patch_engine_core_init_threads()
    _patch_engine_core_client()
    _patch_engine_core()
    _patch_mp_client()
    _patch_coordinator()
    _patch_mla_attention_snapshot_state()
    _patch_rotary_embedding_snapshot_state()

    _PATCHED = True
    logger.info("[snapshot] Applied container_snapshot monkey patches.")


_apply_container_snapshot_patches()
