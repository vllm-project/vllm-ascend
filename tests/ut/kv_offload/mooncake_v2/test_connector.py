# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from prometheus_client import Counter, Gauge, Histogram
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole, KVConnectorTransferResults

from vllm_ascend.distributed.kv_transfer import register_connector
from vllm_ascend.distributed.kv_transfer.ascend_multi_connector import AscendMultiConnector
from vllm_ascend.distributed.kv_transfer.kv_p2p import hisparse_pd
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake import connector as connector_module
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.connector import (
    MooncakeBaseConnector,
    MooncakeConnector,
    MooncakeConnectorV2,
    MooncakePullConnector,
    MooncakePushConnector,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.metadata import MooncakeConnectorMetadata
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.stats import MooncakeKVConnectorStats


def make_facade() -> MooncakeBaseConnector:
    connector = MooncakeBaseConnector.__new__(MooncakeBaseConnector)
    connector.connector_scheduler = MagicMock()
    connector.connector_worker = MagicMock()
    connector._connector_metadata = MooncakeConnectorMetadata()
    return connector


def test_mooncake_connector_aliases_pull_connector() -> None:
    assert MooncakeConnector is MooncakePullConnector


def test_loaded_group_ids_hook_is_version_gated() -> None:
    assert "get_loaded_kv_cache_group_ids" in MooncakeBaseConnector.__dict__


def test_loaded_group_ids_returns_transfer_groups() -> None:
    # A P/D transfer restores the request's own blocks, so the scheduler must
    # skip zeroing (and SWA replay) for every transfer group, not only the
    # prefix-cacheable ones.
    connector = make_facade()
    connector._kv_cache_config = SimpleNamespace(transfer_group_ids=(0, 2))  # type: ignore[attr-defined]
    assert connector.get_loaded_kv_cache_group_ids(MagicMock()) == (0, 2)


def test_required_layout_is_hnd_for_regular_attention_and_none_for_mla() -> None:
    config = SimpleNamespace(model_config=SimpleNamespace(use_mla=False))
    assert MooncakeBaseConnector.get_required_kvcache_layout(config) == "HND"  # type: ignore[arg-type]

    config.model_config.use_mla = True
    assert MooncakeBaseConnector.get_required_kvcache_layout(config) is None  # type: ignore[arg-type]

    config.model_config = None
    assert MooncakeBaseConnector.get_required_kvcache_layout(config) is None  # type: ignore[arg-type]


def test_facade_delegates_scheduler_methods() -> None:
    connector = make_facade()
    request = MagicMock()
    blocks = MagicMock()
    output = MagicMock()
    connector.connector_scheduler.get_num_new_matched_tokens.return_value = (8, True)
    connector.connector_scheduler.request_finished.return_value = (True, {"remote": True})

    assert connector.get_num_new_matched_tokens(request, 4) == (8, True)
    connector.update_state_after_alloc(request, blocks, 8)
    connector.update_connector_output(output)
    assert connector.request_finished(request, [1, 2]) == (True, {"remote": True})

    connector.connector_scheduler.update_state_after_alloc.assert_called_once_with(request, blocks, 8)
    connector.connector_scheduler.request_finished.assert_called_once_with(request, ([1, 2],))


def test_facade_delegates_worker_methods_and_metadata() -> None:
    connector = make_facade()
    connector.connector_worker.get_finished.return_value = ({"sent"}, {"received"})
    results = KVConnectorTransferResults(
        finished_sending={"sent"}, finished_recving={"received", "failed"}, failed_recving={"failed"}
    )
    connector.connector_worker.get_transfer_results.return_value = results
    connector.connector_worker.get_block_ids_with_load_errors.return_value = {10}
    connector.connector_worker.xfer_handshake_metadata = MagicMock()
    connector._connector_metadata.requests["request"] = MagicMock()

    connector.register_kv_caches({})
    connector.start_load_kv(MagicMock())

    assert connector.get_finished(set()) == ({"sent"}, {"received"})
    assert connector.get_transfer_results(set()) is results
    connector.connector_worker.get_transfer_results.assert_called_once_with()
    assert connector.get_block_ids_with_load_errors() == {10}
    assert connector.get_handshake_metadata() is connector.connector_worker.get_handshake_metadata.return_value
    connector.connector_worker.get_handshake_metadata.assert_called_once_with()
    connector.connector_worker.start_load_kv.assert_called_once_with(connector._connector_metadata)


@pytest.mark.parametrize(
    "connector_cls, prefix", [(MooncakePullConnector, "MooncakePull"), (MooncakePushConnector, "MooncakePush")]
)
@pytest.mark.parametrize("extra", [{}, {"transfer_backend": "memfabric"}])
def test_connector_selects_implementation_by_role(connector_cls, prefix, extra) -> None:
    config = MagicMock()
    config.kv_transfer_config.engine_id = "engine"
    config.kv_transfer_config.kv_role = "kv_producer"
    config.kv_transfer_config.kv_connector = connector_cls.__name__
    config.kv_transfer_config.kv_connector_extra_config = extra
    kv_cache_config = MagicMock()

    def initialize_base(self, *_args, **_kwargs) -> None:
        self.engine_id = "engine"
        self.connector_scheduler = None
        self.connector_worker = None

    with (
        patch.object(MooncakeBaseConnector, "__init__", initialize_base),
        patch.object(
            connector_module,
            f"{prefix}ConnectorScheduler",
            return_value=MagicMock(),
        ) as scheduler_cls,
        patch.object(
            connector_module,
            f"{prefix}ConnectorWorker",
            return_value=MagicMock(),
        ) as worker_cls,
    ):
        scheduler_connector = connector_cls(config, KVConnectorRole.SCHEDULER, kv_cache_config)
        worker_connector = connector_cls(config, KVConnectorRole.WORKER, kv_cache_config)

    assert scheduler_connector.connector_scheduler is scheduler_cls.return_value
    assert worker_connector.connector_worker is worker_cls.return_value

    assert scheduler_connector.connector_worker is None
    assert worker_connector.connector_scheduler is None
    scheduler_cls.assert_called_once_with(config, "engine", kv_cache_config)
    worker_cls.assert_called_once_with(config, "engine", kv_cache_config)
    scheduler_connector.shutdown()
    scheduler_cls.return_value.shutdown.assert_called_once_with()


@pytest.mark.parametrize("role", [KVConnectorRole.SCHEDULER, KVConnectorRole.WORKER])
def test_v2_memfabric_uses_private_pull_connector(monkeypatch, role):
    monkeypatch.setattr(KVConnectorFactory, "_registry", {})
    register_connector()
    connector_cls = KVConnectorFactory.get_connector_class_by_name("MooncakeConnectorV2")
    config = MagicMock()
    config.kv_transfer_config.kv_connector = "MooncakeConnectorV2"
    config.kv_transfer_config.kv_connector_extra_config = {"transfer_backend": "memfabric"}
    config.kv_transfer_config.is_kv_consumer = True
    cache = MagicMock()
    cache.hisparse_host_num_blocks = None

    def initialize_base(self, *_args, **_kwargs):
        self.engine_id = "engine"
        self.kv_transfer_config = config.kv_transfer_config
        self.connector_worker = self.connector_scheduler = None

    with (
        patch.object(MooncakeBaseConnector, "__init__", initialize_base),
        patch("vllm_ascend.distributed.kv_transfer.kv_p2p.hisparse_pd._HiSparsePDWorker") as worker,
        patch.object(connector_module, "MooncakePullConnectorScheduler") as scheduler,
    ):
        if role == KVConnectorRole.WORKER:
            with pytest.raises(ValueError, match="requires HiSparse Host Source"):
                connector_cls(config, role, cache)
            worker.assert_not_called()
            scheduler.assert_not_called()
            config.kv_transfer_config.is_kv_consumer = False
        connector = connector_cls(config, role, cache)
    assert type(connector) is hisparse_pd._HiSparseMooncakePullConnector
    assert hisparse_pd._HiSparseMooncakePullConnector.__name__ not in KVConnectorFactory._registry
    if role == KVConnectorRole.WORKER:
        assert connector.connector_worker is worker.return_value
        worker.assert_called_once_with(config, "engine", cache)
        scheduler.assert_not_called()
    else:
        assert connector.connector_scheduler is scheduler.return_value
        scheduler.assert_called_once_with(config, "engine", cache)
        worker.assert_not_called()


def test_multi_connector_uses_public_stats_key_for_hisparse(monkeypatch):
    monkeypatch.setattr(KVConnectorFactory, "_registry", {})
    register_connector()
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            engine_id="engine",
            kv_connector_extra_config={
                "connectors": [
                    {
                        "kv_connector": "MooncakePullConnector",
                        "kv_role": "kv_consumer",
                        "kv_connector_extra_config": {"transfer_backend": "memfabric"},
                    },
                    {
                        "kv_connector": "MooncakeConnectorV2",
                        "kv_role": "kv_consumer",
                        "kv_connector_extra_config": {"transfer_backend": "memfabric"},
                    },
                    {
                        "kv_connector": "MooncakeConnectorV2",
                        "kv_role": "kv_consumer",
                        "kv_connector_extra_config": {"use_layerwise": True},
                    },
                ]
            },
        ),
    )
    children = AscendMultiConnector._get_connector_classes_and_configs(config)  # type: ignore[arg-type]
    assert [connector_cls for connector_cls, _ in children] == [
        MooncakePullConnector,
        MooncakeConnectorV2,
        MooncakePushConnector,
    ]

    transfer_stats = MooncakeKVConnectorStats()
    transfer_stats.record_transfer(0.012, 1024)
    child = object.__new__(hisparse_pd._HiSparseMooncakePullConnector)
    child.connector_worker = MagicMock()
    child.connector_worker.get_kv_connector_stats.return_value = transfer_stats
    multi = object.__new__(AscendMultiConnector)
    multi._connectors = [child]
    stats = multi.get_kv_connector_stats()
    assert stats is not None
    payload = stats.to_dict()
    assert payload == {"MooncakeConnectorV2": transfer_stats.to_dict()}
    restored = AscendMultiConnector.build_kv_connector_stats(payload)
    assert restored is not None
    assert restored.to_dict() == payload

    recorder = MagicMock()
    with (
        patch.object(MooncakeBaseConnector, "build_prom_metrics", return_value=None),
        patch.object(MooncakeConnectorV2, "build_prom_metrics", return_value=recorder),
    ):
        metrics = AscendMultiConnector.build_prom_metrics(
            config,  # type: ignore[arg-type]
            {Gauge: Gauge, Counter: Counter, Histogram: Histogram},
            [],
            {},
        )
    metrics.observe(payload)
    recorder.observe.assert_called_once_with(transfer_stats.to_dict(), 0)
