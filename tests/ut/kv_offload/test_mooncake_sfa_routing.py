# SPDX-License-Identifier: Apache-2.0
"""Unified Pull dispatch and local metadata preserve the SFA contract."""

import ast
from multiprocessing import Pipe
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

from vllm_ascend import ascend_config
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake import connector as ordinary
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.metadata import MooncakeConnectorMetadata
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_sfa import connector as compatible
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_sfa import scheduler as sparse_scheduler
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_sfa import worker as sparse_worker
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_sfa.metadata import MooncakeSFAConnectorMetadata

from .test_mooncake_sfa_config import config as make_config


@pytest.mark.parametrize("enabled", [None, False, True])
@pytest.mark.parametrize("consumer", [False, True])
@pytest.mark.parametrize("component", ["worker", "scheduler"])
def test_unified_entry_selects_exactly_one_implementation_before_initialization(
    monkeypatch, enabled, consumer, component
):
    config = SimpleNamespace(
        additional_config={},
        kv_transfer_config=SimpleNamespace(
            kv_connector="MooncakeConnectorV2",
            kv_connector_extra_config={},
            is_kv_consumer=consumer,
            has_connector=lambda _name: False,
        ),
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(index_topk=2048)),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1, decode_context_parallel_size=1, pipeline_parallel_size=1
        ),
        speculative_config=None,
        use_v2_model_runner=True,
    )
    monkeypatch.setattr(ascend_config, "is_950", lambda: False)

    def validated_config(cfg):
        settings = {} if enabled is None else {"enabled": enabled}
        resolved = ascend_config.SparseKVOffloadConfig.from_additional_config(cfg, settings)
        return SimpleNamespace(sparse_kv_offload_config=resolved)

    monkeypatch.setattr(compatible, "init_ascend_config", validated_config)
    suffix = "Worker" if component == "worker" else "Scheduler"
    original = MagicMock()
    specialized = MagicMock()
    monkeypatch.setattr(ordinary, "MooncakePullConnector" + suffix, original)
    sparse = sparse_worker if component == "worker" else sparse_scheduler
    monkeypatch.setattr(sparse, "MooncakeSFAConnector" + suffix, specialized)
    facade = object.__new__(compatible.MooncakeSFACompatibleConnector)
    facade.engine_id = "engine"
    caches = object()
    if enabled and not consumer:
        with pytest.raises(AssertionError, match="can only be used in D node"):
            getattr(facade, "_create_" + component)(config, caches)
        original.assert_not_called()
        specialized.assert_not_called()
        return
    selected = getattr(facade, "_create_" + component)(config, caches)
    chosen, unused = (specialized, original) if enabled else (original, specialized)
    assert selected is chosen.return_value
    chosen.assert_called_once_with(config, "engine", caches)
    unused.assert_not_called()
    assert isinstance(facade, ordinary.MooncakePullConnector)


@pytest.mark.parametrize("enabled", [None, False])
@pytest.mark.parametrize("use_layerwise", [False, True])
@pytest.mark.parametrize("role", [KVConnectorRole.WORKER, KVConnectorRole.SCHEDULER])
def test_v2_without_sparse_offload_preserves_upstream_pull_push_selection(monkeypatch, enabled, use_layerwise, role):
    config = make_config()
    config.kv_transfer_config.kv_connector_extra_config = {"use_layerwise": use_layerwise}
    sparse_config = None if enabled is None else SimpleNamespace(enabled=enabled)
    monkeypatch.setattr(
        compatible, "init_ascend_config", lambda _config: SimpleNamespace(sparse_kv_offload_config=sparse_config)
    )
    pull, push = MagicMock(), MagicMock()
    monkeypatch.setattr(ordinary, "MooncakePullConnector", pull)
    monkeypatch.setattr(ordinary, "MooncakePushConnector", push)
    caches = object()
    connector = compatible.MooncakeSFACompatibleConnector(config, role, caches)
    selected, unused = (push, pull) if use_layerwise else (pull, push)
    assert connector is selected.return_value
    selected.assert_called_once_with(config, role, caches)
    unused.assert_not_called()


@pytest.mark.parametrize("role", [KVConnectorRole.WORKER, KVConnectorRole.SCHEDULER])
def test_v2_sparse_layerwise_mode_is_rejected_before_component_construction(monkeypatch, role):
    config = make_config()
    config.kv_transfer_config.kv_connector_extra_config = {"use_layerwise": True}
    monkeypatch.setattr(
        compatible,
        "init_ascend_config",
        lambda _config: SimpleNamespace(
            sparse_kv_offload_config=SimpleNamespace(enabled=True, use_fused_overlap=False)
        ),
    )
    ordinary_constructor = MagicMock()
    worker, scheduler = MagicMock(), MagicMock()
    monkeypatch.setattr(compatible, "MooncakeConnectorV2", ordinary_constructor)
    monkeypatch.setattr(sparse_worker, "MooncakeSFAConnectorWorker", worker)
    monkeypatch.setattr(sparse_scheduler, "MooncakeSFAConnectorScheduler", scheduler)
    with pytest.raises(ValueError, match="requires pull mode"):
        compatible.MooncakeSFACompatibleConnector(config, role, object())
    ordinary_constructor.assert_not_called()
    worker.assert_not_called()
    scheduler.assert_not_called()


@pytest.mark.parametrize("role", [KVConnectorRole.WORKER, KVConnectorRole.SCHEDULER])
@pytest.mark.parametrize("mode", [None, "batch", "request"])
def test_v2_sparse_pull_constructs_only_the_specialized_component(monkeypatch, role, mode):
    extra = {} if mode is None else {"pull_transfer_mode": mode}
    config = make_config()
    config.kv_transfer_config.kv_connector_extra_config = extra
    monkeypatch.setattr(
        compatible,
        "init_ascend_config",
        lambda _config: SimpleNamespace(
            sparse_kv_offload_config=SimpleNamespace(enabled=True, use_fused_overlap=False)
        ),
    )
    monkeypatch.setattr(
        ordinary.MooncakeBaseConnector, "__init__", lambda self, *_args: setattr(self, "engine_id", "engine")
    )
    ordinary_constructor = MagicMock()
    worker, scheduler = MagicMock(), MagicMock()
    monkeypatch.setattr(compatible, "MooncakeConnectorV2", ordinary_constructor)
    monkeypatch.setattr(sparse_worker, "MooncakeSFAConnectorWorker", worker)
    monkeypatch.setattr(sparse_scheduler, "MooncakeSFAConnectorScheduler", scheduler)
    caches = object()
    connector = compatible.MooncakeSFACompatibleConnector(config, role, caches)
    assert isinstance(connector, compatible.MooncakeSFACompatibleConnector)
    selected, unused = (worker, scheduler) if role == KVConnectorRole.WORKER else (scheduler, worker)
    selected.assert_called_once_with(config, "engine", caches)
    unused.assert_not_called()
    ordinary_constructor.assert_not_called()


@pytest.mark.parametrize("role", [KVConnectorRole.WORKER, KVConnectorRole.SCHEDULER])
@pytest.mark.parametrize(
    "invalid,match",
    [("runner", "ModelRunner V2"), ("overlap", "LIM"), ("dspark", "DSpark"), ("recompute", "failure_policy")],
)
def test_invalid_sparse_combination_fails_before_component_construction(monkeypatch, role, invalid, match):
    config = make_config()
    config.use_v2_model_runner = invalid != "runner"
    config.speculative_config = SimpleNamespace(method="dspark") if invalid == "dspark" else None
    config.kv_transfer_config.kv_load_failure_policy = "recompute" if invalid == "recompute" else "fail"
    sparse = SimpleNamespace(enabled=True, use_fused_overlap=invalid == "overlap")
    monkeypatch.setattr(
        compatible, "init_ascend_config", lambda _config: SimpleNamespace(sparse_kv_offload_config=sparse)
    )
    worker, scheduler = MagicMock(), MagicMock()
    monkeypatch.setattr(sparse_worker, "MooncakeSFAConnectorWorker", worker)
    monkeypatch.setattr(sparse_scheduler, "MooncakeSFAConnectorScheduler", scheduler)
    with pytest.raises(ValueError, match=match):
        compatible.MooncakeSFACompatibleConnector(config, role, object())
    worker.assert_not_called()
    scheduler.assert_not_called()


@pytest.mark.parametrize("enabled", [None, False, True])
@pytest.mark.parametrize("use_layerwise", [False, True])
@pytest.mark.parametrize("role", [KVConnectorRole.WORKER, KVConnectorRole.SCHEDULER])
def test_registered_pull_alias_preserves_ordinary_type_or_rejects_sparse(monkeypatch, enabled, use_layerwise, role):
    from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

    from vllm_ascend.distributed.kv_transfer import register_connector

    config = make_config()
    config.kv_transfer_config.kv_connector = "MooncakePullConnector"
    config.kv_transfer_config.kv_connector_extra_config = {"use_layerwise": use_layerwise}
    sparse = None if enabled is None else SimpleNamespace(enabled=enabled)
    monkeypatch.setattr(
        compatible, "init_ascend_config", lambda _config: SimpleNamespace(sparse_kv_offload_config=sparse)
    )
    monkeypatch.setattr(
        ordinary.MooncakeBaseConnector, "__init__", lambda self, *_: setattr(self, "engine_id", "engine")
    )
    worker, scheduler = MagicMock(), MagicMock()
    monkeypatch.setattr(ordinary, "MooncakePullConnectorWorker", worker)
    monkeypatch.setattr(ordinary, "MooncakePullConnectorScheduler", scheduler)
    monkeypatch.setattr(KVConnectorFactory, "_registry", {})
    register_connector()
    entry = KVConnectorFactory.get_connector_class_by_name("MooncakePullConnector")
    caches = object()
    if enabled:
        with pytest.raises(ValueError, match="kv_connector='MooncakeConnectorV2'"):
            entry(config, role, caches)
        worker.assert_not_called()
        scheduler.assert_not_called()
    else:
        connector = entry(config, role, caches)
        assert type(connector) is ordinary.MooncakePullConnector
        selected, unused = (worker, scheduler) if role == KVConnectorRole.WORKER else (scheduler, worker)
        selected.assert_called_once_with(config, "engine", caches)
        unused.assert_not_called()


@pytest.mark.parametrize("connector", ["MooncakeConnectorV2", "MooncakePullConnector", None])
def test_multi_validates_outer_config_before_expanding_mooncake_children(monkeypatch, connector):
    from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiConnector

    from vllm_ascend.distributed.kv_transfer.ascend_multi_connector import AscendMultiConnector

    config = make_config()
    config.kv_transfer_config.kv_connector = "MultiConnector"
    config.kv_transfer_config.has_connector = lambda name: name == connector
    sparse = SimpleNamespace(enabled=True)
    init = MagicMock(return_value=SimpleNamespace(sparse_kv_offload_config=sparse))
    expand = MagicMock(return_value=[])
    monkeypatch.setattr(compatible, "init_ascend_config", init)
    monkeypatch.setattr(MultiConnector, "_get_connector_classes_and_configs", expand)
    if connector is not None:
        with pytest.raises(ValueError, match="direct connector"):
            AscendMultiConnector._get_connector_classes_and_configs(config)
        expand.assert_not_called()
    else:
        assert AscendMultiConnector._get_connector_classes_and_configs(config) == []
        init.assert_not_called()
        expand.assert_called_once_with(config)


def test_sparse_local_metadata_roundtrips_without_extending_the_wire_handshake():
    meta = MooncakeSFAConnectorMetadata({"r": 2, "local-hit": 3})
    meta.add_new_req(
        "r",
        ([1],),
        16,
        16,
        {
            "remote_block_ids": ([4],),
            "remote_host": "peer",
            "remote_port": 6000,
            "remote_engine_id": "p",
            "remote_request_id": "p-r",
            "remote_num_prompt_tokens": 16,
        },
    )
    receiver, sender = Pipe(duplex=False)
    try:
        sender.send(meta)
        restored = receiver.recv()
    finally:
        sender.close()
        receiver.close()
    assert isinstance(restored, MooncakeConnectorMetadata)
    assert restored.copy_sfa_slots == {"r": 2, "local-hit": 3}
    assert restored.requests["r"].copy_sfa_slot == 2
    assert restored.requests["r"].remote_block_ids == ([4],)
    assert not hasattr(MooncakeConnectorMetadata(), "copy_sfa_slots")


def test_common_mooncake_does_not_import_offload_components():
    package = Path(ordinary.__file__).parent
    for path in package.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert "sparse_kv_offload" not in (node.module or ""), path.name
                assert "mooncake_sfa" not in (node.module or ""), path.name


@pytest.mark.parametrize("producer", [False, True])
def test_pull_without_sparse_state_preserves_existing_preparation(producer):
    facade = object.__new__(compatible.MooncakeSFACompatibleConnector)
    facade.connector_worker = (
        sparse_worker.MooncakeSFAConnectorWorker.__new__(sparse_worker.MooncakeSFAConnectorWorker)
        if producer
        else SimpleNamespace()
    )
    assert facade.get_copy_sfa_slot_bindings() == {}
    facade.prepare_sparse_kv_offload(object())
    dense_fills = {0: (0, 32)}
    restore, remaining, complete = facade.prepare_copy_sfa_decode(
        object(), object(), (), dense_fills, restore_tails=True
    )
    assert restore and remaining is dense_fills and complete is None


def test_v2_factory_and_stats_class_alias_resolve_the_compatible_entry(monkeypatch):
    from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
    from vllm.v1.kv_offload.factory import OffloadingSpecFactory

    from vllm_ascend.distributed.kv_transfer import register_connector

    monkeypatch.setattr(KVConnectorFactory, "_registry", {})
    monkeypatch.setattr(OffloadingSpecFactory, "_registry", {})
    register_connector()
    for name in ["MooncakeConnectorV2", "MooncakeSFACompatibleConnector"]:
        assert KVConnectorFactory.get_connector_class_by_name(name) is compatible.MooncakeSFACompatibleConnector
    assert (
        KVConnectorFactory.get_connector_class_by_name("MooncakePullConnector")
        is compatible.MooncakeSFACompatibleConnector
    )
    assert KVConnectorFactory.get_connector_class_by_name("MooncakePushConnector") is ordinary.MooncakePushConnector
