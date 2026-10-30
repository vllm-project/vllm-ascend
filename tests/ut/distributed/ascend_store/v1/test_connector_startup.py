"""Registered Connector selection and the shared startup configuration contracts."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.model_executor.models.config import HybridAttentionMambaModelConfig
from vllm.platforms.interface import Platform
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.kv_offload.factory import OffloadingSpecFactory

from tests.ut.distributed.ascend_store.v1.construction_fixtures import create_scheduler, create_worker
from vllm_ascend.distributed.kv_transfer import register_connector
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import (
    get_layerwise_reuse_config,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1 import connector
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1 import connector as connector_module
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure import (
    configuration as configuration_module,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.configuration import (
    resolve_kv_pool_config,
)
from vllm_ascend.platform import NPUPlatform


@pytest.fixture
def registered_connectors(monkeypatch):
    monkeypatch.setattr(KVConnectorFactory, "_registry", {})
    monkeypatch.setattr(OffloadingSpecFactory, "_registry", OffloadingSpecFactory._registry.copy())
    register_connector()


def transfer_config(name, *, nested=False, extra=None):
    child = {"kv_connector": name, "kv_role": "kv_both", "kv_connector_extra_config": extra or {}}
    if nested:
        return KVTransferConfig(
            kv_connector="MultiConnector", kv_role="kv_both", kv_connector_extra_config={"connectors": [child]}
        )
    return KVTransferConfig(**child)


@pytest.mark.usefixtures("registered_connectors")
@pytest.mark.parametrize("role", (KVConnectorRole.SCHEDULER, KVConnectorRole.WORKER))
def test_registered_name_creates_v1_for_both_roles(configuration_case, monkeypatch, role):
    config, cache = configuration_case
    config.kv_transfer_config = transfer_config("AscendStoreV1Connector")
    config.parallel_config.data_parallel_rank = 0
    config.parallel_config.rank = 1
    scheduler_owner, worker_owner = (object(), object())
    create_scheduler = Mock(return_value=scheduler_owner)
    create_worker = Mock(return_value=worker_owner)
    monkeypatch.setattr(connector.AscendStoreV1Connector, "_create_scheduler", create_scheduler)
    monkeypatch.setattr(connector.AscendStoreV1Connector, "_create_worker", create_worker)
    instance = KVConnectorFactory.create_connector(config, role, cache)
    assert type(instance) is connector.AscendStoreV1Connector
    assert config.kv_transfer_config.kv_connector_module_path is None
    if role == KVConnectorRole.SCHEDULER:
        assert instance.scheduler is scheduler_owner
        create_scheduler.assert_called_once()
        assert create_scheduler.call_args.args[1] == instance._resolve_lookup_address(config)
        create_worker.assert_not_called()
    else:
        assert instance.worker is worker_owner
        create_worker.assert_called_once()
        assert create_worker.call_args.args[:2] == (config, cache)
        create_scheduler.assert_not_called()


@pytest.mark.usefixtures("registered_connectors")
@pytest.mark.parametrize("nested", (False, True))
@pytest.mark.parametrize("layerwise", (False, True))
@pytest.mark.parametrize("name", ("AscendStoreConnector", "AscendStoreV1Connector"))
def test_registered_connector_preserves_hma_and_layerwise_graph_requirements(name, nested, layerwise):
    transfer = transfer_config(name, nested=nested, extra={"use_layerwise": layerwise})
    connector_type = KVConnectorFactory.get_connector_class(transfer)
    assert KVConnectorFactory.supports_hma_config(transfer)
    assert connector_type.requires_piecewise_for_cudagraph(transfer.kv_connector_extra_config) is layerwise
    if not nested:
        assert connector_type.__name__ == name
        assert KVConnectorFactory.get_connector_class_by_name(name) is connector_type


@pytest.mark.parametrize("name", ("AscendStoreConnector", "AscendStoreV1Connector"))
@pytest.mark.parametrize("nested", (False, True))
@pytest.mark.parametrize("prefix_caching", (False, True))
def test_hybrid_startup_aligns_mamba_before_v1_admission(configuration_case, monkeypatch, name, nested, prefix_caching):
    config, cache = configuration_case
    config.kv_transfer_config = transfer_config(name, nested=nested)
    config.cache_config.enable_prefix_caching = prefix_caching
    config.cache_config.mamba_cache_mode = "none"
    config.cache_config.mamba_block_size = None
    config.model_config.architecture = "HybridStartupFixture"
    config.scheduler_config.enable_chunked_prefill = True
    HybridAttentionMambaModelConfig.verify_and_update_config(config)
    monkeypatch.setattr(Platform, "update_block_size_for_backend", classmethod(lambda cls, config: None))
    NPUPlatform.update_block_size_for_backend(config)
    assert config.cache_config.mamba_cache_mode == "align"
    assert config.cache_config.mamba_block_size == config.cache_config.block_size
    state = MambaSpec(
        block_size=config.cache_config.mamba_block_size,
        shapes=((1,),),
        dtypes=(torch.float32,),
        mamba_cache_mode=config.cache_config.mamba_cache_mode,
    )
    cache.kv_cache_groups.append(SimpleNamespace(kv_cache_spec=state, is_eagle_group=False))
    cache.transfer_group_ids = (0, 1)
    if nested:
        child = config.kv_transfer_config.kv_connector_extra_config["connectors"][0]
        config.kv_transfer_config = KVTransferConfig(**child)
    resolved = resolve_kv_pool_config(config, cache)
    configuration_module.validate_kv_pool_config(config, cache, resolved)
    assert resolved.hash_block_size == config.cache_config.block_size
    assert resolved.align_state_group_ids == frozenset({1})


@pytest.mark.parametrize("nested", (False, True))
@pytest.mark.parametrize("side", ("scheduler", "worker"))
def test_registered_v1_rejects_shared_layerwise_buffers_before_binding(configuration_case, monkeypatch, nested, side):
    config, cache = configuration_case
    extra = {"backend": "memcache", "use_layerwise": True, "layerwise_num_shared_buffers": 2}
    config.kv_transfer_config = transfer_config("AscendStoreV1Connector", nested=nested, extra=extra)
    assert get_layerwise_reuse_config(config.kv_transfer_config) == extra
    if nested:
        child = config.kv_transfer_config.kv_connector_extra_config["connectors"][0]
        config.kv_transfer_config = KVTransferConfig(**child)
    cache.kv_cache_groups[0].layer_names = [f"model.layers.{index}.attn" for index in range(4)]
    config.model_config.get_num_layers = lambda _parallel: 4
    bind_topology = Mock(side_effect=AssertionError("Topology must not be bound"))
    lookup_client = Mock(side_effect=AssertionError("Lookup must not be connected"))
    monkeypatch.setattr(connector_module, "resolve_worker_topology", bind_topology)
    monkeypatch.setattr(connector_module, "LookupClient", lookup_client)
    with pytest.raises(ValueError, match="Layerwise buffer reuse requires native partial-object support"):
        if side == "scheduler":
            create_scheduler(config, cache, "unused")
        else:
            create_worker(config, cache)
    bind_topology.assert_not_called()
    lookup_client.assert_not_called()
