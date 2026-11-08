# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
"""Native connector construction and registration with device boundaries mocked."""

import ctypes
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401

native_connector_module = pytest.importorskip("vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.connector")
if not hasattr(native_connector_module.MooncakeStoreConnector, "_create_worker"):
    pytest.skip("requires vLLM native Store device hooks (#66294)", allow_module_level=True)

# isort: split
from vllm.config import CacheConfig, KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store import worker as native_worker
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.layout import MambaStoreLayout, TokenMajorStoreLayout

from tests.ut.distributed.ascend_store.native_glm_fixture import make_glm_connectors
from tests.ut.distributed.ascend_store.test_native_cache_views import (
    assert_attention_registration,
    make_attention_connector,
)
from tests.ut.distributed.ascend_store.test_native_qwen_registration import (
    BLOCK,
    BLOCKS,
    CHUNK,
    KV_HEADS,
    STORE_TP,
    expected_payload,
    logical_views,
    make_qwen_registration_inputs,
    seed_source,
)
from vllm_ascend.distributed.kv_transfer import _register_with_class_name_alias
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import native_worker as ascend_worker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.native_connector import AscendNativeStoreConnector


def test_factory_constructs_native_worker_and_registers_real_glm_views(monkeypatch, tmp_path):
    worker_connector, scheduler_connector, cache_config, tensors, backend, _ = make_glm_connectors(
        monkeypatch, tmp_path
    )
    groups = cache_config.kv_cache_groups
    worker = worker_connector.connector_worker
    try:
        worker_connector.register_kv_caches(tensors)
        assert isinstance(worker.kv_send_thread, ascend_worker.AscendNativeStoreSendingThread)
        assert all(isinstance(t, ascend_worker.AscendNativeStoreRecvingThread) for t in worker.kv_recv_threads)
        selected = scheduler_connector.connector_scheduler._store_group_ids
        assert [g.layer_names for g in worker._kv_cache_groups] == [groups[i].layer_names for i in selected]
        assert len(worker.token_dbs) == len(selected)
        assert backend.register_buffer_checked.call_count == 3
    finally:
        worker_connector.shutdown()
        scheduler_connector.shutdown()


def test_factory_registers_ordinary_attention_without_model_dispatch(monkeypatch, tmp_path):
    connector, backend = make_attention_connector(monkeypatch, tmp_path)
    # Planar K/V share one allocation, with real FullAttentionSpec and no MTP.
    planes = torch.zeros((2, 3, 8, 2, 4), dtype=torch.bfloat16)
    try:
        planned = connector.connector_worker.token_dbs[0].store_layout
        connector.register_kv_caches({"attention": (planes[0], planes[1])})
        assert_attention_registration(connector, backend, planes)
        bound = connector.connector_worker.token_dbs[0].store_layout
        assert bound.metadata.store_namespace != planned.metadata.store_namespace
        assert len(bound.layer_specs) == 2
    finally:
        connector.shutdown()


def test_factory_constructs_native_worker_and_registers_real_qwen_views(monkeypatch, tmp_path):
    # This helper allocates only specs/tensors: no preconstructed worker or layouts.
    config, cache_config, tensors = make_qwen_registration_inputs(1, "NHD")
    config.parallel_config = SimpleNamespace(
        data_parallel_index=0,
        rank=0,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        tensor_parallel_size=1,
        world_size=1,
    )
    config.scheduler_config = SimpleNamespace(disable_hybrid_kv_cache_manager=False)
    config.cache_config = CacheConfig(block_size=BLOCK, prefix_match_unit=CHUNK, enable_prefix_caching=True)
    config.cache_config.kv_cache_layout = "LBNHC"
    config.kv_events_config = None
    config.speculative_config = None
    config.model_config.model = "/models/qwen-cpu"
    config.model_config.get_num_layers = lambda parallel: 2
    config.model_config.get_total_num_kv_heads = lambda: KV_HEADS
    config.model_config.use_mla = False
    config.kv_transfer_config = KVTransferConfig(
        kv_connector="MooncakeStoreConnector",
        kv_role="kv_both",
        kv_load_failure_policy="fail",
        kv_connector_extra_config={"store_tp_size": STORE_TP, "cache_prefix": "qwen-factory", "load_async": True},
    )
    store_config = tmp_path / "qwen-store.json"
    store_config.write_text(
        '{"master_server_address":"127.0.0.1:1","protocol":"ascend","global_segment_size":65536,"local_buffer_size":0}'
    )
    monkeypatch.setenv("MOONCAKE_CONFIG_PATH", str(store_config))
    monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", "NHD")
    backend = Mock()
    backend._completion_unknown = False
    backend.config.prefer_alloc_in_same_node = True
    backend.config.preferred_segment = False
    monkeypatch.setattr(ascend_worker, "MooncakeBackend", lambda *args, **kwargs: backend)
    monkeypatch.setattr(native_worker, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(native_worker, "get_tensor_model_parallel_world_size", lambda: 1)
    group = SimpleNamespace(world_size=1, rank_in_group=0)
    monkeypatch.setattr(native_worker, "get_pcp_group", lambda: group)
    monkeypatch.setattr(native_worker, "get_dcp_group", lambda: group)
    monkeypatch.setattr(native_worker, "LookupKeyServer", Mock())
    for cls in (ascend_worker.AscendNativeStoreSendingThread, ascend_worker.AscendNativeStoreRecvingThread):
        monkeypatch.setattr(cls, "start", lambda self: self.ready_event.set())
    monkeypatch.setattr(KVConnectorFactory, "_registry", dict(KVConnectorFactory._registry))
    _register_with_class_name_alias(
        "MooncakeStoreConnector",
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.native_connector",
        "AscendNativeStoreConnector",
        override=True,
    )
    connector = KVConnectorFactory.create_connector(config, KVConnectorRole.WORKER, cache_config)
    try:
        assert isinstance(connector, AscendNativeStoreConnector)
        worker = connector.connector_worker
        assert isinstance(worker, ascend_worker.AscendNativeStoreWorker)
        assert worker.store_tp_size == STORE_TP
        assert worker._group_tp_replication_factors == (1, 1)
        assert [group.layer_names for group in worker._kv_cache_groups] == [["attention"], ["gdn"]]
        planned = tuple(db.store_layout for db in worker.token_dbs)
        assert len(planned) == 2
        assert isinstance(planned[0], TokenMajorStoreLayout)
        assert isinstance(planned[1], MambaStoreLayout)
        assert planned[0].store_chunk_size == CHUNK
        assert planned[1].store_chunk_size == BLOCK
        backend.register_buffer_checked.assert_not_called()
        views = logical_views(tensors, "NHD")
        seed_source(views)
        connector.register_kv_caches(tensors)
        assert worker.num_blocks == BLOCKS
        regions = {
            (t.untyped_storage().data_ptr(), t.untyped_storage().nbytes())
            for values in tensors.values()
            for t in values
        }
        calls = [call.args for call in backend.register_buffer_checked.call_args_list]
        assert len(calls) == 3
        assert all(len(pointers) == len(lengths) == 1 for pointers, lengths in calls)
        assert {(pointers[0], lengths[0]) for pointers, lengths in calls} == regions
        assert worker._device_regions == dict(regions)
        assert isinstance(worker.kv_send_thread, ascend_worker.AscendNativeStoreSendingThread)
        assert worker.kv_recv_threads
        assert all(isinstance(t, ascend_worker.AscendNativeStoreRecvingThread) for t in worker.kv_recv_threads)
        for group_id, (before, db) in enumerate(zip(planned, worker.token_dbs)):
            after = db.store_layout
            assert after is not before and type(after) is type(before)
            if isinstance(after, MambaStoreLayout):
                assert after.metadata.store_namespace == before.metadata.store_namespace
            else:
                assert after.metadata.store_namespace == before.metadata.store_namespace + "@ascend_kv_planes:v1"
            assert after.local_shard_ids == before.local_shard_ids == tuple(range(STORE_TP))
            assert after.store_chunk_size == before.store_chunk_size
            # Registration must yield real payload descriptors from the exact
            # layouts selected by the constructor, including a nonzero block ID.
            for shard in after.local_shard_ids:
                addresses, sizes, selected = after.prepare_values([(0, CHUNK)], [2], [shard])
                assert selected == [2]
                payload = b"".join(ctypes.string_at(address, size) for address, size in zip(addresses[0], sizes[0]))
                assert payload == expected_payload(views, group_id, 2, shard, 0)
                assert after.key_for(shard, b"prefix") in [
                    prefix + "@" + b"prefix".hex() for prefix in worker._lookup_key_prefixes[group_id]
                ]
    finally:
        connector.shutdown()
