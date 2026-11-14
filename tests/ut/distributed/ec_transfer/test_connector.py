# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorRole
from vllm.distributed.ec_transfer.ec_connector.tiering.connector import ECTieringConnector

from vllm_ascend.distributed.ec_transfer.ec_connector.cpu.connector import (
    AscendECCPUConnector,
)
from vllm_ascend.distributed.ec_transfer.ec_connector.tiering.connector import AscendECTieringConnector


def tiering_config(extra=None):
    return SimpleNamespace(
        use_v2_model_runner=True,
        ec_transfer_config=SimpleNamespace(
            ec_connector_extra_config={} if extra is None else extra,
            is_ec_producer=True,
            is_ec_consumer=True,
        ),
    )


def test_make_worker_uses_ascend_backend(monkeypatch):
    import vllm_ascend.distributed.ec_transfer.ec_connector.cpu.worker as worker_mod

    config = object()
    expected = object()
    monkeypatch.setattr(worker_mod, "AscendECCPUWorker", lambda cfg: expected)
    connector = AscendECCPUConnector.__new__(AscendECCPUConnector)

    assert connector._make_worker(config) is expected


def test_tiering_scheduler_role_reuses_upstream_without_worker(monkeypatch):
    import vllm.distributed.ec_transfer.ec_connector.tiering.scheduler as scheduler_mod

    expected = Mock()
    factory = Mock(return_value=expected)
    monkeypatch.setattr(scheduler_mod, "ECTieringScheduler", factory)
    worker = Mock(side_effect=AssertionError("scheduler constructed NPU worker"))
    monkeypatch.setattr(AscendECTieringConnector, "_make_worker", worker)
    config = tiering_config()
    connector = AscendECTieringConnector(config, ECConnectorRole.SCHEDULER)
    factory.assert_called_once_with(config)
    assert connector.connector_scheduler is expected
    assert connector.connector_worker is None
    request = object()
    connector.on_new_request(request)
    expected.on_new_request.assert_called_once_with(request)
    assert connector.get_ec_connector_stats() is None
    expected.get_stats.assert_not_called()


def test_tiering_worker_role_uses_existing_ascend_worker_without_timing_interface(monkeypatch):
    import vllm_ascend.distributed.ec_transfer.ec_connector.cpu.worker as worker_mod

    # Intentionally no get_stats(): this is the existing Ascend worker contract.
    expected = SimpleNamespace()
    factory = Mock(return_value=expected)
    monkeypatch.setattr(worker_mod, "AscendECCPUWorker", factory)
    monkeypatch.setattr(
        AscendECTieringConnector, "_make_scheduler", Mock(side_effect=AssertionError("worker built FS"))
    )
    config = tiering_config()
    connector = AscendECTieringConnector(config, ECConnectorRole.WORKER)
    factory.assert_called_once_with(config)
    assert connector.connector_worker is expected
    assert connector.connector_scheduler is None
    assert connector.get_ec_connector_stats() is None


@pytest.mark.parametrize("connector_cls", [AscendECCPUConnector, AscendECTieringConnector])
@pytest.mark.parametrize("value", [True, False])
@pytest.mark.parametrize("role", list(ECConnectorRole))
def test_ascend_offload_rejects_nixl_key_before_constructing_delegates(connector_cls, value, role):
    with pytest.raises(ValueError, match="does not support ec_enable_nixl"):
        connector_cls(tiering_config({"ec_enable_nixl": value}), role)


@pytest.mark.parametrize("role", list(ECConnectorRole))
def test_tiering_requires_v2_model_runner(role):
    config = tiering_config()
    config.use_v2_model_runner = False
    with pytest.raises(ValueError, match="V2 model runner"):
        AscendECTieringConnector(config, role)


def test_tiering_does_not_override_scheduler_or_metadata_protocol():
    for name in (
        "_make_scheduler",
        "on_new_request",
        "ensure_cache_available",
        "update_state_after_alloc",
        "build_connector_meta",
        "update_connector_output",
        "request_finished",
        "build_connector_worker_meta",
    ):
        assert getattr(AscendECTieringConnector, name) is getattr(ECTieringConnector, name)
