# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import signal
from multiprocessing.connection import Connection
from typing import cast
from unittest.mock import MagicMock, call, patch

import pytest

from tests.e2e.conftest import (
    _DP_RUNNER_SHUTDOWN_TIMEOUT_SECONDS,
    DPVllmRunner,
    _shutdown_vllm_runner_engine,
)


def _make_runner(num_workers: int = 2) -> tuple[DPVllmRunner, list[MagicMock], list[MagicMock]]:
    runner = object.__new__(DPVllmRunner)
    conns = [MagicMock(name=f"conn_{rank}") for rank in range(num_workers)]
    processes = [MagicMock(name=f"proc_{rank}") for rank in range(num_workers)]
    for rank, proc in enumerate(processes):
        proc.pid = 1000 + rank
        proc.sentinel = 2000 + rank
    runner._dp_parent_conns = cast(list[Connection], conns)
    runner._dp_processes = processes
    return runner, conns, processes


def test_wait_for_data_parallel_workers_reports_later_rank_error_immediately():
    runner, conns, processes = _make_runner()
    conns[1].recv.return_value = {
        "status": "error",
        "traceback": "rank 1 failed",
    }

    with (
        patch("tests.e2e.conftest.wait", return_value=[conns[1]]) as mock_wait,
        pytest.raises(RuntimeError, match="Data parallel worker 1 failed") as exc_info,
    ):
        runner._wait_for_data_parallel_workers(
            expected_status="ok",
            timeout=900,
            operation="generate",
        )

    waitables = mock_wait.call_args.args[0]
    assert set(waitables) == set(conns) | {proc.sentinel for proc in processes}
    assert "rank 1 failed" in str(exc_info.value)
    conns[0].recv.assert_not_called()


def test_wait_for_data_parallel_workers_detects_process_exit():
    runner, _conns, processes = _make_runner()
    processes[1].exitcode = 7

    with (
        patch("tests.e2e.conftest.wait", return_value=[processes[1].sentinel]),
        pytest.raises(RuntimeError, match="worker 1 exited unexpectedly.*exit code: 7"),
    ):
        runner._wait_for_data_parallel_workers(
            expected_status="ok",
            timeout=900,
            operation="generate",
        )

    processes[1].join.assert_called_once_with(timeout=0)


def test_wait_for_data_parallel_workers_returns_rank_order():
    runner, conns, _processes = _make_runner()
    conns[0].recv.return_value = {"status": "ok", "rank": 0}
    conns[1].recv.return_value = {"status": "ok", "rank": 1}

    with patch("tests.e2e.conftest.wait", side_effect=[[conns[1]], [conns[0]]]):
        messages = runner._wait_for_data_parallel_workers(
            expected_status="ok",
            timeout=900,
            operation="generate",
        )

    assert [message["rank"] for message in messages] == [0, 1]


def test_stop_data_parallel_workers_signals_all_process_groups_together():
    runner, conns, processes = _make_runner()
    processes[0].is_alive.return_value = False
    processes[1].is_alive.return_value = True

    with (
        patch.object(runner, "_wait_for_data_parallel_processes"),
        patch("tests.e2e.conftest.os.killpg") as mock_killpg,
    ):
        runner._stop_data_parallel_workers()

    for conn in conns:
        conn.send.assert_called_once_with({"command": "shutdown"})
        conn.close.assert_called_once_with()
    assert mock_killpg.call_args_list == [
        call(processes[0].pid, signal.SIGTERM),
        call(processes[1].pid, signal.SIGTERM),
        call(processes[0].pid, signal.SIGKILL),
        call(processes[1].pid, signal.SIGKILL),
    ]
    assert runner._dp_parent_conns == []
    assert runner._dp_processes == []


def test_shutdown_vllm_runner_engine_is_explicit_and_bounded():
    llm = MagicMock()

    _shutdown_vllm_runner_engine(llm)

    llm.llm_engine.engine_core.shutdown.assert_called_once_with(timeout=_DP_RUNNER_SHUTDOWN_TIMEOUT_SECONDS)
