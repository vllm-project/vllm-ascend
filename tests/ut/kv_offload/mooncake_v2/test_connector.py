# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from prometheus_client import Counter, Gauge, Histogram
from vllm.config import KVTransferConfig
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


@pytest.mark.parametrize("kv_role", ["kv_producer", "kv_consumer"])
@pytest.mark.parametrize("connector_cls", [MooncakeConnectorV2, MooncakePullConnector])
def test_memfabric_without_hisparse_uses_ordinary_pull(kv_role, connector_cls):
    config = MagicMock()
    config.kv_transfer_config = KVTransferConfig(
        kv_connector=connector_cls.__name__,
        kv_role=kv_role,
        kv_connector_extra_config={"transfer_backend": "memfabric"},
    )
    cache = MagicMock()

    def initialize_base(self, *_args, **_kwargs):
        self.engine_id = "engine"
        self.connector_worker = self.connector_scheduler = None

    with (
        patch.object(MooncakeBaseConnector, "__init__", initialize_base),
        patch.object(connector_module, "_MemFabricPullWorker") as memfabric_worker,
        patch.object(connector_module, "MooncakePullConnectorWorker") as default_worker,
    ):
        connector = connector_cls(config, KVConnectorRole.WORKER, cache)
    assert type(connector) is MooncakePullConnector
    if connector_cls is MooncakeConnectorV2:
        memfabric_worker.assert_called_once_with(config, "engine", cache)
        default_worker.assert_not_called()
    else:
        default_worker.assert_called_once_with(config, "engine", cache)
        memfabric_worker.assert_not_called()


def test_multi_connector_uses_public_stats_key_for_hisparse(monkeypatch):
    monkeypatch.setattr(KVConnectorFactory, "_registry", {})
    register_connector()
    monkeypatch.setitem(KVConnectorFactory._registry, "HiSparseConnector", lambda: MooncakeBaseConnector)
    config = SimpleNamespace(
        kv_transfer_config=KVTransferConfig(
            kv_connector="MultiConnector",
            kv_role="kv_consumer",
            engine_id="engine",
            kv_connector_extra_config={
                "connectors": [
                    {"kv_connector": "HiSparseConnector", "kv_role": "kv_consumer"},
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
        MooncakeBaseConnector,
        MooncakePullConnector,
        hisparse_pd._HiSparseMooncakePullConnector,
        MooncakePushConnector,
    ]
    config_without_hisparse = SimpleNamespace(
        kv_transfer_config=KVTransferConfig(
            kv_connector="MultiConnector",
            kv_role="kv_consumer",
            kv_connector_extra_config={
                "connectors": config.kv_transfer_config.kv_connector_extra_config["connectors"][1:]
            },
        )
    )
    children_without_hisparse = AscendMultiConnector._get_connector_classes_and_configs(config_without_hisparse)  # type: ignore[arg-type]
    assert [connector_cls for connector_cls, _ in children_without_hisparse] == [
        MooncakePullConnector,
        MooncakePullConnector,
        MooncakePushConnector,
    ]

    selected_cls, child_config = children[2]
    cache = MagicMock(hisparse_host_num_blocks=None)
    with (
        patch.object(MooncakeBaseConnector, "__init__", return_value=None),
        pytest.raises(ValueError, match="requires HiSparse Host Source"),
    ):
        selected_cls(child_config, KVConnectorRole.WORKER, cache)

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
        patch.object(hisparse_pd._HiSparseMooncakePullConnector, "build_prom_metrics", return_value=recorder),
    ):
        metrics = AscendMultiConnector.build_prom_metrics(
            config,  # type: ignore[arg-type]
            {Gauge: Gauge, Counter: Counter, Histogram: Histogram},
            [],
            {},
        )
    metrics.observe(payload)
    recorder.observe.assert_called_once_with(transfer_stats.to_dict(), 0)
