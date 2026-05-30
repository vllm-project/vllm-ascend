# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import MagicMock, patch

import pytest
import vllm.v1.engine.utils as engine_utils

from vllm_ascend.patch.platform.patch_engine_core_parallel_startup import (
    _ascend_enginecore_bootstrap,
    _AscendCoreEngineProcManagerBackport,
    _resolve_enginecore_bootstrap,
)
from vllm_ascend.patch.platform.patch_multiproc_executor import AscendMultiprocExecutor


def test_ascend_enginecore_bootstrap_sets_env(monkeypatch):
    called: dict = {}

    def target_fn(**kwargs):
        called["kwargs"] = kwargs

    monkeypatch.delenv("TEST_DEVICE_VAR", raising=False)
    _ascend_enginecore_bootstrap(
        evar="TEST_DEVICE_VAR",
        value="0,1",
        target_fn=target_fn,
        target_kwargs={"dp_rank": 0},
    )
    assert os.environ["TEST_DEVICE_VAR"] == "0,1"
    assert called["kwargs"] == {"dp_rank": 0}


def test_resolve_enginecore_bootstrap_uses_upstream_when_present():
    bootstrap = _resolve_enginecore_bootstrap()
    if hasattr(engine_utils, "_enginecore_bootstrap"):
        assert bootstrap is engine_utils._enginecore_bootstrap
    else:
        assert bootstrap is _ascend_enginecore_bootstrap


@pytest.mark.asyncio
async def test_start_workers_async_appends_started_workers_before_failure():
    executor = MagicMock(spec=AscendMultiprocExecutor)
    executor.local_world_size = 3
    executor.vllm_config = MagicMock()
    executor._is_driver_worker = lambda rank: rank % 2 == 0

    def make_worker_process(**kwargs):
        rank = kwargs["rank"]
        if rank == 1:
            raise RuntimeError("simulated worker start failure")
        handle = MagicMock()
        handle.rank = rank
        return handle

    unready_workers: list = []

    with (
        patch(
            "vllm_ascend.patch.platform.patch_multiproc_executor.AscendWorkerProc.make_worker_process",
            side_effect=make_worker_process,
        ),
        pytest.raises(RuntimeError, match="simulated worker start failure"),
    ):
        await AscendMultiprocExecutor._start_workers_async(
            executor,
            global_start_rank=0,
            distributed_init_method="tcp://127.0.0.1:12345",
            scheduler_output_handle=None,
            shared_worker_lock=MagicMock(),
            unready_workers=unready_workers,
        )

    assert len(unready_workers) == 1
    assert unready_workers[0].rank == 0


@patch.object(_AscendCoreEngineProcManagerBackport, "shutdown")
@patch.object(_AscendCoreEngineProcManagerBackport, "_run_async_startup", side_effect=RuntimeError("boom"))
@patch("vllm_ascend.patch.platform.patch_engine_core_parallel_startup.get_mp_context")
@patch("vllm_ascend.patch.platform.patch_engine_core_parallel_startup.current_platform")
def test_backport_shutdown_when_async_startup_fails(
    mock_platform,
    mock_get_mp_context,
    mock_shutdown,
    mock_run_async_startup,
):
    mock_platform.is_cuda_alike.return_value = True
    mock_proc = MagicMock()
    mock_proc.exitcode = None
    mock_get_mp_context.return_value.Process.return_value = mock_proc

    mock_vllm_config = MagicMock()
    mock_vllm_config.parallel_config.data_parallel_size = 1

    with patch("vllm.v1.engine.core.EngineCoreProc"), pytest.raises(RuntimeError, match="boom"):
        _AscendCoreEngineProcManagerBackport(
            local_engine_count=2,
            start_index=0,
            local_start_index=0,
            vllm_config=mock_vllm_config,
            local_client=True,
            handshake_address="ipc:///tmp/test",
            executor_class=MagicMock(),
            log_stats=False,
        )

    mock_shutdown.assert_called_once()


def test_core_engine_patch_skipped_when_upstream_has_async_startup():
    if hasattr(engine_utils.CoreEngineProcManager, "_run_async_startup"):
        assert engine_utils.CoreEngineProcManager is not _AscendCoreEngineProcManagerBackport
    else:
        assert engine_utils.CoreEngineProcManager is _AscendCoreEngineProcManagerBackport
