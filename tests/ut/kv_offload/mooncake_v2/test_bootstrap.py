# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake import push_scheduler
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_worker import MooncakeBaseConnectorWorker
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_worker import MooncakePullConnectorWorker
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.push_scheduler import MooncakePushConnectorScheduler
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.push_worker import MooncakePushConnectorWorker


@pytest.fixture
def scheduler():
    scheduler = MooncakePushConnectorScheduler.__new__(MooncakePushConnectorScheduler)
    scheduler.kv_transfer_config = SimpleNamespace(
        is_kv_producer=True,
        get_from_extra_config=lambda key, default: default,
    )
    scheduler.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            local_engines_only=False,
            data_parallel_rank_local=0,
            data_parallel_index=0,
        )
    )
    scheduler.bootstrap_server = None
    return scheduler


def test_bootstrap_lifecycle_is_scheduler_owned_and_idempotent(scheduler, monkeypatch):
    monkeypatch.setattr(push_scheduler, "get_mooncake_bootstrap_addr", lambda config: ("127.0.0.1", 8998))
    factory = MagicMock()
    monkeypatch.setattr(push_scheduler, "MooncakeBootstrapServer", factory)
    scheduler.start_bootstrap_server()
    scheduler.start_bootstrap_server()
    factory.assert_called_once_with("0.0.0.0", 8998)
    factory.return_value.start.assert_called_once_with()
    scheduler.shutdown()
    scheduler.shutdown()
    factory.return_value.shutdown.assert_called_once_with()


def test_bootstrap_start_failure_cleans_up(scheduler, monkeypatch):
    monkeypatch.setattr(push_scheduler, "get_mooncake_bootstrap_addr", lambda config: ("127.0.0.1", 8998))
    factory = MagicMock()
    factory.return_value.start.side_effect = OSError("port occupied")
    monkeypatch.setattr(push_scheduler, "MooncakeBootstrapServer", factory)
    with pytest.raises(OSError, match="port occupied"):
        scheduler.start_bootstrap_server()
    assert scheduler.bootstrap_server is None
    factory.return_value.shutdown.assert_called_once_with()


@pytest.mark.parametrize(
    "producer, external, local_only, local_rank, index, expected",
    [
        (True, False, False, 0, 0, True),
        (True, False, False, 0, 1, False),
        (True, False, True, 0, 4, True),
        (True, False, True, 1, 4, False),
        (False, False, False, 0, 0, False),
        (True, True, False, 0, 0, False),
    ],
)
def test_bootstrap_scheduler_election(scheduler, producer, external, local_only, local_rank, index, expected):
    scheduler.kv_transfer_config.is_kv_producer = producer
    scheduler.kv_transfer_config.get_from_extra_config = lambda key, default: "host:8998" if external else default
    scheduler.vllm_config.parallel_config = SimpleNamespace(
        local_engines_only=local_only,
        data_parallel_rank_local=local_rank,
        data_parallel_index=index,
    )
    assert scheduler._should_launch_bootstrap_server() is expected


def test_push_scheduler_constructor_starts_service(monkeypatch):
    monkeypatch.setattr(push_scheduler.MooncakeBaseConnectorScheduler, "__init__", lambda *args: None)
    start = MagicMock()
    monkeypatch.setattr(MooncakePushConnectorScheduler, "start_bootstrap_server", start)
    scheduler = MooncakePushConnectorScheduler(MagicMock(), "engine-p", MagicMock())
    start.assert_called_once_with()
    assert scheduler.bootstrap_server is None


def test_bootstrap_is_not_part_of_pull_or_base_worker():
    assert not hasattr(MooncakePushConnectorWorker, "register_worker_with_bootstrap")
    for cls in (MooncakeBaseConnectorWorker, MooncakePullConnectorWorker):
        assert not hasattr(cls, "register_worker_with_bootstrap")
        assert not hasattr(cls, "get_remote_worker_addresses")
    for cls in (MooncakeBaseConnectorWorker, MooncakePullConnectorWorker, MooncakePushConnectorWorker):
        assert not hasattr(cls, "start_bootstrap_server")
