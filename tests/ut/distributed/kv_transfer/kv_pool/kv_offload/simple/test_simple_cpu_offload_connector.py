# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace

import pytest
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector import (
    SimpleCPUOffloadConnector,
)

from vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.simple import (
    simple_cpu_offload_connector as connector_module,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.simple.simple_cpu_offload_connector import (
    AscendSimpleCPUOffloadConnector,
)


def test_factory_registration_uses_consolidated_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_ascend.distributed.kv_transfer import register_connector

    registrations: dict[str, tuple[str, str]] = {}

    def capture_registration(cls, name: str, module_path: str, class_name: str) -> None:
        registrations[name] = (module_path, class_name)

    # Keep the test independent of whether the vLLM plugin was already loaded
    # by the current pytest environment.
    monkeypatch.setattr(KVConnectorFactory, "_registry", {})
    monkeypatch.setattr(
        KVConnectorFactory,
        "register_connector",
        classmethod(capture_registration),
    )
    register_connector()

    assert registrations["SimpleCPUOffloadConnector"] == (
        "vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.simple.simple_cpu_offload_connector",
        "AscendSimpleCPUOffloadConnector",
    )


@pytest.mark.parametrize(
    ("role", "has_upstream_worker", "expect_npu_worker"),
    [
        (KVConnectorRole.WORKER, True, True),
        (KVConnectorRole.WORKER, False, False),
        (KVConnectorRole.SCHEDULER, False, False),
    ],
)
def test_connector_only_replaces_enabled_worker(
    monkeypatch: pytest.MonkeyPatch,
    role: KVConnectorRole,
    has_upstream_worker: bool,
    expect_npu_worker: bool,
) -> None:
    upstream_worker = SimpleNamespace(cpu_capacity_bytes=512) if has_upstream_worker else None

    def fake_upstream_init(self, vllm_config, connector_role, kv_cache_config):
        self.worker_handler = upstream_worker

    created: list[tuple[object, object, int]] = []
    npu_worker = object()

    def fake_npu_worker(vllm_config, kv_cache_config, cpu_capacity):
        created.append((vllm_config, kv_cache_config, cpu_capacity))
        return npu_worker

    monkeypatch.setattr(SimpleCPUOffloadConnector, "__init__", fake_upstream_init)
    monkeypatch.setattr(
        connector_module,
        "SimpleCPUOffloadNPUWorker",
        fake_npu_worker,
    )

    config = object()
    kv_cache_config = object()
    connector = AscendSimpleCPUOffloadConnector(config, role, kv_cache_config)

    if expect_npu_worker:
        assert connector.worker_handler is npu_worker
        assert created == [(config, kv_cache_config, 512)]
    else:
        assert connector.worker_handler is upstream_worker
        assert not created
