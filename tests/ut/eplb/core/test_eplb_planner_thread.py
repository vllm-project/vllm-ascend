from unittest.mock import MagicMock, patch

import pytest

from vllm_ascend.eplb.core.eplb_worker import EplbPlannerThread


@pytest.fixture
def planner() -> EplbPlannerThread:
    ep_group = MagicMock(rank_in_group=0)
    with (
        patch(
            "vllm_ascend.eplb.core.eplb_worker.get_ep_group",
            return_value=ep_group,
        ),
        patch(
            "vllm_ascend.eplb.core.eplb_worker.PolicyFactory.generate_policy",
            return_value=MagicMock(),
        ),
    ):
        return EplbPlannerThread(shared_dict={}, policy_type=0)


def test_planner_runs_policy_in_daemon_thread(planner: EplbPlannerThread) -> None:
    expected = [("send", "recv", "map", "log2phy", 0)]
    planner.worker.do_update = MagicMock(return_value=expected)

    planner.start()
    planner.start()
    assert planner.is_alive()
    assert planner._thread is not None
    assert planner._thread.daemon

    planner.submit()
    assert planner.get_result() == expected
    planner.worker.do_update.assert_called_once_with()

    planner.shutdown()
    assert not planner.is_alive()


def test_planner_propagates_policy_failure(planner: EplbPlannerThread) -> None:
    planner.worker.do_update = MagicMock(side_effect=ValueError("invalid placement"))
    planner.start()
    planner.submit()

    with pytest.raises(RuntimeError, match="failed while calculating") as exc_info:
        planner.get_result()

    assert isinstance(exc_info.value.__cause__, ValueError)
    planner.shutdown()
    assert not planner.is_alive()


def test_planner_shutdown_is_idempotent(planner: EplbPlannerThread) -> None:
    planner.shutdown()
    planner.shutdown()

    with pytest.raises(RuntimeError, match="cannot be restarted"):
        planner.start()


def test_upstream_multiproc_executor_is_not_replaced() -> None:
    from vllm.v1.executor import multiproc_executor

    executor_cls = multiproc_executor.MultiprocExecutor
    assert executor_cls.__name__ == "MultiprocExecutor"
    assert executor_cls.__module__ == "vllm.v1.executor.multiproc_executor"
