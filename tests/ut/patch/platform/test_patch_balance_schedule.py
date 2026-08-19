# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm_ascend.patch.platform.patch_balance_schedule import (
    BalanceDPEngineCoreProc,
    _coordinated_dummy_run_enabled,
)


class _StopBusyLoop(Exception):
    pass


def _make_engine_core(*, local_has_work: bool, global_has_work: bool, executed: bool):
    scheduler = MagicMock()
    scheduler.has_unfinished_requests.return_value = local_has_work
    scheduler.finished_req_ids = set()

    engine_core = SimpleNamespace(
        scheduler=scheduler,
        engines_running=False,
        dp_group=MagicMock(),
        _process_input_queue=MagicMock(),
        _process_engine_step=MagicMock(return_value=executed),
        _maybe_publish_request_counts=MagicMock(),
        execute_dummy_batch=MagicMock(),
        _complete_wave=MagicMock(),
    )
    engine_core._has_global_unfinished_reqs = MagicMock(side_effect=[global_has_work, _StopBusyLoop()])
    return engine_core


def test_idle_dp_rank_runs_dummy_when_another_rank_has_work():
    engine_core = _make_engine_core(local_has_work=False, global_has_work=True, executed=False)

    with pytest.raises(_StopBusyLoop):
        BalanceDPEngineCoreProc.run_busy_loop(engine_core)

    engine_core._has_global_unfinished_reqs.assert_any_call(False)
    engine_core._process_engine_step.assert_called_once_with()
    engine_core.execute_dummy_batch.assert_called_once_with()
    engine_core.scheduler.balance_gather.assert_called_once_with(engine_core.dp_group)


def test_active_dp_rank_does_not_run_dummy():
    engine_core = _make_engine_core(local_has_work=True, global_has_work=True, executed=True)

    with pytest.raises(_StopBusyLoop):
        BalanceDPEngineCoreProc.run_busy_loop(engine_core)

    engine_core._has_global_unfinished_reqs.assert_any_call(True)
    engine_core._process_engine_step.assert_called_once_with()
    engine_core.execute_dummy_batch.assert_not_called()
    engine_core.scheduler.balance_gather.assert_called_once_with(engine_core.dp_group)


def test_fused_moe_enables_coordinated_dummy_without_balance_scheduling():
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_size=4,
            enable_expert_parallel=True,
        ),
        additional_config={"enable_fused_mc2": 1},
    )

    assert _coordinated_dummy_run_enabled(vllm_config)
