# SPDX-License-Identifier: Apache-2.0
"""Component selection happens before constructors can start background work."""

from unittest.mock import MagicMock, patch

import pytest
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake import connector as module
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake import pull_scheduler


def test_pull_subclass_constructs_only_its_selected_component():
    scheduler = MagicMock()
    worker = MagicMock()

    class SpecializedConnector(module.MooncakePullConnector):
        def _create_scheduler(self, config, caches):
            scheduler(config, caches)
            return scheduler

        def _create_worker(self, config, caches):
            worker(config, caches)
            return worker

    def initialize_base(self, *_args):
        self.engine_id = "engine"
        self.connector_scheduler = None
        self.connector_worker = None

    config, caches = MagicMock(), MagicMock()
    with (
        patch.object(module.MooncakeBaseConnector, "__init__", initialize_base),
        patch.object(module, "MooncakePullConnectorScheduler") as ordinary_scheduler,
        patch.object(module, "MooncakePullConnectorWorker") as ordinary_worker,
    ):
        selected_scheduler = SpecializedConnector(config, KVConnectorRole.SCHEDULER, caches)
        selected_worker = SpecializedConnector(config, KVConnectorRole.WORKER, caches)
    assert selected_scheduler.connector_scheduler is scheduler
    assert selected_worker.connector_worker is worker
    scheduler.assert_called_once_with(config, caches)
    worker.assert_called_once_with(config, caches)
    ordinary_scheduler.assert_not_called()
    ordinary_worker.assert_not_called()


def test_request_policy_is_initialized_before_scheduler_threads_start():
    class SpecializedScheduler(pull_scheduler.MooncakePullConnectorScheduler):
        def _initialize_request_state(self, config):
            raise ValueError("policy initialization failed")

    def initialize_base(self, *_args):
        self.kv_transfer_config = MagicMock()
        self.kv_transfer_config.get_from_extra_config.return_value = 30
        self.kv_transfer_config.is_kv_consumer = True

    with (
        patch.object(pull_scheduler.MooncakeBaseConnectorScheduler, "__init__", initialize_base),
        patch.object(pull_scheduler, "MooncakeSchedulerRecvingThread") as receiver,
        patch.object(pull_scheduler, "MooncakeHeartbeatThread") as heartbeat,
        pytest.raises(ValueError, match="policy initialization failed"),
    ):
        SpecializedScheduler(MagicMock(), "engine", MagicMock())
    receiver.assert_not_called()
    heartbeat.assert_not_called()
