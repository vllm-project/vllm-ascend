# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
"""CPU coverage for ordinary attention through the actual native Store factory."""

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
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.layout import (
    MambaStoreLayout,
    RankLocalStoreLayout,
    TokenMajorStoreLayout,
)
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, MLAAttentionSpec

from tests.ut.distributed.ascend_store.native_glm_fixture import make_glm_connectors
from vllm_ascend.distributed.kv_transfer import _register_with_class_name_alias
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import native_worker as ascend_worker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.native_connector import AscendNativeStoreConnector


def make_attention_connector(monkeypatch, tmp_path, *, store_tp_size=2, spec=None):
    extra = {"cache_prefix": "generic-factory", "load_async": True}
    if store_tp_size is not None:
        extra["store_tp_size"] = store_tp_size
    if spec is None:
        spec = FullAttentionSpec(block_size=8, num_kv_heads=2, head_size=4, dtype=torch.bfloat16)
    cache_config = KVCacheConfig(
        num_blocks=3, kv_cache_tensors=[], kv_cache_groups=[KVCacheGroupSpec(["attention"], spec)]
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type="llama"),
            model="ordinary-attention-cpu",
            get_num_layers=lambda parallel: 1,
            get_total_num_kv_heads=lambda: spec.num_kv_heads,
            use_mla=isinstance(spec, MLAAttentionSpec),
        ),
        parallel_config=SimpleNamespace(
            data_parallel_index=0,
            rank=0,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            tensor_parallel_size=1,
            world_size=1,
        ),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        cache_config=CacheConfig(block_size=8, prefix_match_unit=4, enable_prefix_caching=True),
        compilation_config=SimpleNamespace(static_forward_context={}),
        kv_events_config=None,
        speculative_config=None,
        num_hash_lookahead_tokens=0,
        kv_transfer_config=KVTransferConfig(
            kv_connector="MooncakeStoreConnector",
            kv_role="kv_both",
            kv_load_failure_policy="fail",
            kv_connector_extra_config=extra,
        ),
    )
    config.cache_config.kv_cache_layout = "LBNHC"
    store_config = tmp_path / "store.json"
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
    assert isinstance(connector, AscendNativeStoreConnector)
    assert isinstance(connector.connector_worker, ascend_worker.AscendNativeStoreWorker)
    backend.register_buffer_checked.assert_not_called()
    return connector, backend


def assert_attention_registration(connector, backend, allocation):
    worker = connector.connector_worker
    assert worker.num_blocks == 3
    assert [group.layer_names for group in worker._kv_cache_groups] == [["attention"]]
    assert len(worker.token_dbs) == 1
    layout = worker.token_dbs[0].store_layout
    assert isinstance(layout, TokenMajorStoreLayout)
    assert layout.local_shard_ids == (0, 1)
    assert layout.store_chunk_size == 8
    storage = allocation.untyped_storage()
    pointer, length = storage.data_ptr(), storage.nbytes()
    backend.register_buffer_checked.assert_called_once_with([pointer], [length])
    assert worker._device_regions == {pointer: length}
    assert isinstance(worker.kv_send_thread, ascend_worker.AscendNativeStoreSendingThread)
    assert worker.kv_recv_threads
    assert all(isinstance(t, ascend_worker.AscendNativeStoreRecvingThread) for t in worker.kv_recv_threads)
    for shard in layout.local_shard_ids:
        addresses, sizes, selected = layout.prepare_values([(0, 8)], [2], [shard])
        assert selected == [2]
        assert sum(sizes[0]) == 2 * 8 * 4 * allocation.element_size()
        assert all(
            pointer <= address < address + size <= pointer + length for address, size in zip(addresses[0], sizes[0])
        )


def test_factory_accepts_core_native_packed_attention(monkeypatch, tmp_path):
    connector, backend = make_attention_connector(monkeypatch, tmp_path)
    # Native logical [block, head, token, packed K/V content], no planar adapter.
    packed = torch.zeros((3, 8, 2, 8), dtype=torch.bfloat16).transpose(1, 2)
    try:
        planned = connector.connector_worker.token_dbs[0].store_layout
        connector.register_kv_caches({"attention": packed})
        assert_attention_registration(connector, backend, packed)
        bound = connector.connector_worker.token_dbs[0].store_layout
        assert bound.metadata.store_namespace == planned.metadata.store_namespace
        assert bound.layer_specs == planned.layer_specs
    finally:
        connector.shutdown()


@pytest.mark.parametrize("use_recoverssm", [False, True], ids=["two-state", "recoverssm"])
def test_generic_glm_full_state_host_pages_and_private_tail(monkeypatch, tmp_path, use_recoverssm):
    # Real constructors select native groups/layouts before any views are bound.
    source, source_scheduler, _, src, source_backend, layer = make_glm_connectors(
        monkeypatch, tmp_path, use_recoverssm=use_recoverssm
    )
    target, target_scheduler, _, dst, target_backend, _ = make_glm_connectors(
        monkeypatch, tmp_path, use_recoverssm=use_recoverssm
    )
    try:
        conv, recurrent, *records = src["kda"]
        c8 = src["c8"][0]
        dst_conv, dst_recurrent, *dst_records = dst["kda"]
        dst_c8 = dst["c8"][0]
        assert c8.shape[0] == 3 and conv.shape[0] == recurrent.shape[0] == 5
        assert src["c8"][1].numel() == 0
        assert len(records) == (2 if use_recoverssm else 0)
        assert tuple(tuple(t.shape[1:]) for t in src["kda"]) == layer.get_state_shape()
        assert tuple(t.dtype for t in src["kda"]) == layer.get_state_dtype()
        if use_recoverssm:
            heads, steps, dim = layer.num_heads // layer.tp_size, layer.num_spec + 1, layer.head_dim
            for record, width in zip(records, (dim, 2 * dim), strict=True):
                assert record.shape == (5, heads, steps, width)
                assert record.stride() == (heads * steps * width, steps * width, width, 1)
                assert record.dtype == torch.float32
                assert record[4].data_ptr() - record[0].data_ptr() == 4 * record.stride(0) * record.element_size()
        for index, tensor in enumerate((c8, conv, recurrent, *records)):
            values = torch.arange(tensor.numel()).reshape(tensor.shape)
            tensor.copy_((values % 97 + 1 + index * 11).to(tensor.dtype))
        for tensor in (dst_c8, dst_conv, dst_recurrent, *dst_records):
            tensor.fill_(-7)
        source_tail, target_tail = src["tail"].clone(), dst["tail"].clone()
        expected = [tensor.clone() for tensor in (dst_c8, dst_conv, dst_recurrent, *dst_records)]
        # Highest host/device slots exercise each pool's actual capacity.
        expected[0][1, :4] = c8[2, :4]
        expected[1][1] = conv[4]
        expected[2][1] = recurrent[4]
        for oracle, record in zip(expected[3:], records, strict=True):
            oracle[1] = record[4]

        for connector, caches, backend in ((source, src, source_backend), (target, dst, target_backend)):
            worker = connector.connector_worker
            planned = tuple(db.store_layout for db in worker.token_dbs)
            connector.register_kv_caches(caches)
            assert [group.layer_names for group in worker._kv_cache_groups] == [["c8"], ["kda"]]
            assert type(worker.token_dbs[1].store_layout) is MambaStoreLayout
            assert worker.token_dbs[0].store_layout.layer_specs[0].num_heads == 1
            regions = {
                (tensor.untyped_storage().data_ptr(), tensor.untyped_storage().nbytes())
                for tensor in (caches["c8"][0], *caches["kda"])
            }
            calls = [call.args for call in backend.register_buffer_checked.call_args_list]
            assert len(calls) == 3 + len(records)
            assert all(len(pointers) == len(lengths) == 1 for pointers, lengths in calls)
            assert {(pointers[0], lengths[0]) for pointers, lengths in calls} == regions
            assert worker._device_regions == {
                tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes() for tensor in caches["kda"]
            }
            for group_id, (before, db) in enumerate(zip(planned, worker.token_dbs, strict=True)):
                assert type(db.store_layout) is type(before)
                assert db.metadata.store_namespace == before.metadata.store_namespace
                assert db.metadata == db.store_layout.metadata
                for shard in db.store_layout.local_shard_ids:
                    assert db.store_layout.key_for(shard, b"prefix") in [
                        prefix + "@" + b"prefix".hex() for prefix in worker._lookup_key_prefixes[group_id]
                    ]

        def as_bytes(tensor):
            return tensor.contiguous().view(torch.uint8).numpy().tobytes()

        # GLM has equal Q/K/V projections. Derive the independent oracle from
        # the loaded-layer geometry, without using Store split/layout helpers.
        projection = layer.num_heads // layer.tp_size * layer.head_dim
        logical_conv = conv.transpose(1, 2) if is_conv_state_dim_first() else conv
        assert logical_conv.shape[1] == layer.conv_size - 1 + layer.num_spec
        assert logical_conv.shape[2] == 3 * projection
        for group_id, (src_db, dst_db) in enumerate(
            zip(source.connector_worker.token_dbs, target.connector_worker.token_dbs, strict=True)
        ):
            src_layout, dst_layout = src_db.store_layout, dst_db.store_layout
            chunk = 4 if group_id == 0 else 8
            assert src_layout.store_chunk_size == dst_layout.store_chunk_size == chunk
            assert src_layout.local_shard_ids == dst_layout.local_shard_ids
            for local_shard, shard in enumerate(src_layout.local_shard_ids):
                assert src_layout.key_for(shard, b"prefix") == dst_layout.key_for(shard, b"prefix")
                if group_id == 0:
                    oracle = as_bytes(c8[2, :4])
                    source_block = 2
                    allowed_source, allowed_target = (c8,), (dst_c8,)
                else:
                    # Both local Store shards own half of EACH Q/K/V projection,
                    # all convolution rows (including speculative rows), and one SSM head.
                    assert len(src_layout.local_shard_ids) == 2
                    width = projection // 2
                    pieces = [
                        logical_conv[4, :, offset + local_shard * width : offset + (local_shard + 1) * width]
                        for offset in (0, projection, 2 * projection)
                    ]
                    joined = torch.cat(pieces, dim=1)
                    if is_conv_state_dim_first():
                        joined = joined.transpose(0, 1)
                    heads = recurrent.shape[1] // 2
                    oracle = as_bytes(joined) + as_bytes(recurrent[4, local_shard * heads : (local_shard + 1) * heads])
                    oracle += b"".join(
                        as_bytes(record[4, local_shard * heads : (local_shard + 1) * heads]) for record in records
                    )
                    source_block = 4
                    allowed_source, allowed_target = src["kda"], dst["kda"]
                addresses, sizes, selected = src_layout.prepare_values([(0, chunk)], [source_block], [shard])
                target_addresses, target_sizes, target_selected = dst_layout.prepare_values([(0, chunk)], [1], [shard])
                assert selected == [source_block] and target_selected == [1]
                for pointers, lengths, allocations in (
                    (addresses[0], sizes[0], allowed_source),
                    (target_addresses[0], target_sizes[0], allowed_target),
                ):
                    assert len(pointers) == len(lengths)
                    assert all(
                        any(
                            tensor.data_ptr()
                            <= address
                            < address + size
                            <= tensor.data_ptr() + tensor.numel() * tensor.element_size()
                            for tensor in allocations
                        )
                        for address, size in zip(pointers, lengths, strict=True)
                    )
                payload = b"".join(ctypes.string_at(address, size) for address, size in zip(addresses[0], sizes[0]))
                assert payload == oracle
                assert sum(target_sizes[0]) == len(oracle)
                cursor = 0
                for address, size in zip(target_addresses[0], target_sizes[0], strict=True):
                    ctypes.memmove(address, payload[cursor : cursor + size], size)
                    cursor += size
        for actual, oracle in zip((dst_c8, dst_conv, dst_recurrent, *dst_records), expected, strict=True):
            assert torch.equal(actual, oracle)  # Full state rows plus every untouched block.
        assert torch.equal(src["tail"], source_tail)
        assert torch.equal(dst["tail"], target_tail)
    finally:
        source.shutdown()
        target.shutdown()
        source_scheduler.shutdown()
        target_scheduler.shutdown()


def test_factory_accepts_packed_attention_with_default_rank_local_store(monkeypatch, tmp_path):
    connector, backend = make_attention_connector(monkeypatch, tmp_path, store_tp_size=None)
    allocation = torch.arange(3 * 8 * 2 * 8, dtype=torch.float32).to(torch.bfloat16).reshape(3, 8, 2, 8)
    packed = allocation.transpose(1, 2)
    try:
        worker = connector.connector_worker
        config = worker._ascend_vllm_config
        assert "store_tp_size" not in config.kv_transfer_config.kv_connector_extra_config
        assert config.speculative_config is None
        assert worker.store_tp_size is None
        planned = worker.token_dbs[0].store_layout
        assert type(planned) is RankLocalStoreLayout
        connector.register_kv_caches({"attention": packed})
        assert worker.num_blocks == 3
        assert len(worker.token_dbs) == 1
        bound = worker.token_dbs[0].store_layout
        assert type(bound) is RankLocalStoreLayout
        assert bound.metadata == planned.metadata
        assert bound.local_shard_ids == (0,)
        assert bound.key_for(0, b"prefix") == planned.key_for(0, b"prefix")
        assert bound.key_for(0, b"prefix") in [
            prefix + "@" + b"prefix".hex() for prefix in worker._lookup_key_prefixes[0]
        ]
        storage = allocation.untyped_storage()
        backend.register_buffer_checked.assert_called_once_with([storage.data_ptr()], [storage.nbytes()])
        assert worker._device_regions == {storage.data_ptr(): storage.nbytes()}
        assert isinstance(worker.kv_send_thread, ascend_worker.AscendNativeStoreSendingThread)
        assert worker.kv_recv_threads
        assert all(isinstance(t, ascend_worker.AscendNativeStoreRecvingThread) for t in worker.kv_recv_threads)
        addresses, sizes, selected = bound.prepare_values([(0, 8)], [2], [0])
        assert selected == [2]
        assert addresses == [[allocation[2].data_ptr()]]
        assert sizes == [[allocation[2].numel() * allocation.element_size()]]
        assert ctypes.string_at(addresses[0][0], sizes[0][0]) == allocation[2].view(torch.uint8).numpy().tobytes()
    finally:
        connector.shutdown()


def test_typed_shared_mamba2_pages_preserve_group_replication():
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.layout import KeyMetadata
    from vllm.v1.kv_cache_interface import MambaAttentionBackendEnum, MambaSpec

    # Four recurrent heads, head_dim=2, state_size=4, and only two B/C
    # groups. Each B/C group must be repeated on TWO of the four Store shards.
    # Scalar slicing would instead split each group into two smaller payloads.
    blocks, offset, page_stride = 3, 32, 320
    conv_shape = (24, 3) if is_conv_state_dim_first() else (3, 24)
    spec = MambaSpec(
        block_size=8,
        shapes=(conv_shape, (4, 2, 4)),
        dtypes=(torch.float16, torch.float32),
        mamba_type=MambaAttentionBackendEnum.MAMBA2,
        mamba_cache_mode="align",
        page_size_padded=page_stride,
    )
    content_bytes = 24 * 3 * 2 + 4 * 2 * 4 * 4

    def make_endpoint():
        raw = torch.full((offset + blocks * page_stride + 32,), 0xA5, dtype=torch.uint8)
        tensors = []
        start = offset
        for shape, dtype in zip(spec.shapes, spec.dtypes, strict=True):
            slot = torch.empty(shape, dtype=dtype)
            size = slot.element_size()
            tensors.append(
                torch.as_strided(
                    raw.view(dtype), (blocks, *shape), (page_stride // size, *slot.stride()), start // size
                )
            )
            start += slot.numel() * size
        worker = ascend_worker.AscendNativeStoreWorker.__new__(ascend_worker.AscendNativeStoreWorker)
        worker._capacity_only = False
        worker._ascend_vllm_config = SimpleNamespace(compilation_config=SimpleNamespace(static_forward_context={}))
        group = KVCacheGroupSpec(["mamba"], spec)
        worker._kv_cache_config = KVCacheConfig(num_blocks=blocks, kv_cache_tensors=[], kv_cache_groups=[group])
        worker._kv_cache_groups = [group]
        worker._group_tp_replication_factors = (1,)
        worker.tp_size = worker.pp_size = worker.pcp_size = worker.dcp_size = 1
        worker.tp_rank = worker.pp_rank = worker.pcp_rank = worker.dcp_rank = 0
        worker.hash_block_size = 8
        worker.requested_store_chunk_size = None
        worker.store_tp_size = 4
        worker._device_regions = {}
        worker.store = SimpleNamespace(register_buffer=Mock(return_value=0), raise_if_failed=Mock())
        worker._start_transfer_threads = Mock()
        plan = worker._make_store_group_plan(group, 4, TokenMajorStoreLayout, 1)
        assert plan is not None
        worker.token_dbs = worker._build_token_databases(KeyMetadata("mamba2-cpu", 0, 0, 0, 0), (plan,))
        planned = worker.token_dbs[0].store_layout
        worker.register_kv_caches({"mamba": tensors})
        layout = worker.token_dbs[0].store_layout
        assert type(layout) is MambaStoreLayout
        assert layout.metadata == planned.metadata
        assert layout.local_shard_ids == (0, 1, 2, 3)
        view_start = raw.data_ptr() + offset
        view_span = (blocks - 1) * page_stride + content_bytes
        worker.store.register_buffer.assert_called_once_with(view_start, view_span)
        assert worker._device_regions == {view_start: view_span}
        worker._start_transfer_threads.assert_called_once_with()
        return raw, tensors, layout

    source, (conv, recurrent), src = make_endpoint()
    target, _, dst = make_endpoint()
    for index, tensor in enumerate((conv, recurrent)):
        tensor.copy_((torch.arange(tensor.numel()).reshape(tensor.shape) + 1 + index * 1000).to(tensor.dtype))
    original = source.clone()
    expected = target.clone()
    expected[offset + page_stride : offset + page_stride + content_bytes] = source[
        offset + 2 * page_stride : offset + 2 * page_stride + content_bytes
    ]

    def as_bytes(tensor):
        return tensor.contiguous().view(torch.uint8).numpy().tobytes()

    logical_conv = conv.transpose(1, 2) if is_conv_state_dim_first() else conv
    for shard in src.local_shard_ids:
        group_start = (shard // 2) * 4
        # Independent logical-index oracle: x is split by head; B and C
        # each contribute one COMPLETE state_size=4 group, replicated in pairs.
        parts = [logical_conv[2, :, shard * 2 : (shard + 1) * 2]]
        parts.extend(logical_conv[2, :, base + group_start : base + group_start + 4] for base in (8, 16))
        selected_conv = torch.cat(parts, dim=1)
        if is_conv_state_dim_first():
            selected_conv = selected_conv.transpose(0, 1)
        oracle = as_bytes(selected_conv) + as_bytes(recurrent[2, shard : shard + 1])
        sa, ss, sb = src.prepare_values([(0, 8)], [2], [shard])
        da, ds, db = dst.prepare_values([(0, 8)], [1], [shard])
        assert sb == [2] and db == [1]
        assert src.key_for(shard, b"prefix") == dst.key_for(shard, b"prefix")
        for addresses, sizes, raw, block in ((sa[0], ss[0], source, 2), (da[0], ds[0], target, 1)):
            page = raw.data_ptr() + offset + block * page_stride
            assert all(
                page <= address < address + size <= page + content_bytes
                for address, size in zip(addresses, sizes, strict=True)
            )
        payload = b"".join(ctypes.string_at(address, size) for address, size in zip(sa[0], ss[0], strict=True))
        assert payload == oracle
        assert sum(ds[0]) == len(oracle)
        cursor = 0
        for address, size in zip(da[0], ds[0], strict=True):
            ctypes.memmove(address, payload[cursor : cursor + size], size)
            cursor += size
    assert torch.equal(source, original)
    # Also covers allocation prefix/suffix, every page's padding, and unselected pages.
    assert torch.equal(target, expected)


def test_aligned_offset_views_register_only_selected_strided_span(monkeypatch, tmp_path):
    from vllm_ascend.distributed.kv_transfer.utils.utils import collect_storage_merged_register_regions

    connector, backend = make_attention_connector(monkeypatch, tmp_path)
    raw = torch.full((4096,), 0xA5, dtype=torch.uint8)
    alignment = 256
    # Strictly advance from the storage origin, even if it is already aligned.
    offset = alignment - raw.data_ptr() % alignment
    start = raw.data_ptr() + offset
    blocks, page_stride, content_bytes = 3, 320, 256
    span = (blocks - 1) * page_stride + content_bytes
    packed = torch.empty(0, dtype=torch.bfloat16).set_(
        raw.untyped_storage(), offset // 2, (blocks, 2, 8, 8), (page_stride // 2, 8, 16, 1)
    )
    try:
        assert packed.storage_offset() > 0 and packed.data_ptr() == start
        assert start % alignment == 0
        assert span > packed.numel() * packed.element_size()
        connector.register_kv_caches(
            {
                "attention": packed,
                "omitted": raw[:offset],
                "private_tail": raw[offset + span :],
            }
        )
        backend.register_buffer_checked.assert_called_once_with([start], [span])
        worker = connector.connector_worker
        assert worker._device_regions == {start: span}
        assert [group.layer_names for group in worker._kv_cache_groups] == [["attention"]]

        # CPU registration needs no alignment. Exercise the same source collector's
        # optional bounded down-alignment explicitly with a nonaligned subview.
        shifted = packed[:, :, 1:, :]
        assert shifted.data_ptr() == start + 32
        unaligned = collect_storage_merged_register_regions({"attention": shifted})
        assert unaligned.ptrs == [start + 32] and unaligned.lengths == [span - 32]
        aligned = collect_storage_merged_register_regions({"attention": shifted}, alignment=alignment)
        assert aligned.ptrs == [start] and aligned.lengths == [span]
        assert raw.data_ptr() < aligned.ptrs[0]

        layout = worker.token_dbs[0].store_layout
        for shard in layout.local_shard_ids:
            addresses, sizes, selected = layout.prepare_values([(0, 8)], [2], [shard])
            assert selected == [2]
            actual = {
                byte
                for address, size in zip(addresses[0], sizes[0], strict=True)
                for byte in range(address, address + size)
            }
            # One head's packed K/V per token; no block padding, omitted prefix,
            # private suffix, or other head enters the transfer descriptors.
            expected = {
                start + 2 * page_stride + token * 32 + shard * 16 + byte for token in range(8) for byte in range(16)
            }
            assert actual == expected
    finally:
        connector.shutdown()


def test_factory_planar_attention_preserves_unequal_kv_width_bytes(monkeypatch, tmp_path):
    spec = FullAttentionSpec(block_size=8, num_kv_heads=2, head_size=4, head_size_v=6, dtype=torch.bfloat16)
    connector, backend = make_attention_connector(monkeypatch, tmp_path, spec=spec)
    k = (torch.arange(3 * 8 * 2 * 4).reshape(3, 8, 2, 4) % 101).to(torch.bfloat16)
    v = -(torch.arange(3 * 8 * 2 * 6).reshape(3, 8, 2, 6) % 103 + 1).to(torch.bfloat16)
    try:
        worker = connector.connector_worker
        planned = worker.token_dbs[0].store_layout
        connector.register_kv_caches({"attention": (k, v)})
        layout = worker.token_dbs[0].store_layout
        assert type(layout) is TokenMajorStoreLayout
        assert layout.local_shard_ids == (0, 1)
        assert layout.metadata.store_namespace == planned.metadata.store_namespace + "@ascend_kv_planes:v1"
        assert [plane.state_content_size_bytes for plane in layout.layer_specs] == [4 * 2, 6 * 2]
        calls = [call.args for call in backend.register_buffer_checked.call_args_list]
        assert len(calls) == 2
        assert {(pointers[0], lengths[0]) for pointers, lengths in calls} == {
            (tensor.data_ptr(), tensor.numel() * tensor.element_size()) for tensor in (k, v)
        }
        assert worker._device_regions == {
            tensor.data_ptr(): tensor.numel() * tensor.element_size() for tensor in (k, v)
        }
        for shard in layout.local_shard_ids:
            addresses, sizes, selected = layout.prepare_values([(0, 8)], [2], [shard])
            assert selected == [2]
            assert sum(sizes[0]) == 8 * (4 + 6) * 2
            payload = b"".join(
                ctypes.string_at(address, size) for address, size in zip(addresses[0], sizes[0], strict=True)
            )
            # Logical NHD indexing: each shard owns one head, with K then V.
            oracle = b"".join(
                tensor[2, :, shard, :].contiguous().view(torch.uint8).numpy().tobytes() for tensor in (k, v)
            )
            assert payload == oracle
    finally:
        connector.shutdown()


def test_factory_standard_mla_two_components_preserve_descriptor_bytes(monkeypatch, tmp_path):
    spec = MLAAttentionSpec(block_size=8, num_kv_heads=1, head_size=6, dtype=torch.bfloat16)
    connector, backend = make_attention_connector(monkeypatch, tmp_path, store_tp_size=1, spec=spec)
    compressed_kv = torch.arange(3 * 8 * 4, dtype=torch.bfloat16).reshape(3, 8, 1, 4)
    rope = -(torch.arange(3 * 8 * 2, dtype=torch.bfloat16) + 1).reshape(3, 8, 1, 2)
    try:
        worker = connector.connector_worker
        planned = worker.token_dbs[0].store_layout
        connector.register_kv_caches({"attention": (compressed_kv, rope)})
        layout = worker.token_dbs[0].store_layout
        assert type(layout) is TokenMajorStoreLayout
        assert layout.local_shard_ids == (0,)
        assert layout.metadata.store_namespace == planned.metadata.store_namespace + "@ascend_kv_planes:v1"
        assert len(layout.layer_specs) == 2
        assert all(isinstance(component, MLAAttentionSpec) for component in layout.layer_specs)
        assert [component.state_content_size_bytes for component in layout.layer_specs] == [8, 4]
        assert (
            sum(component.state_content_size_bytes for component in layout.layer_specs) == spec.state_content_size_bytes
        )
        regions = {tensor.data_ptr(): tensor.numel() * tensor.element_size() for tensor in (compressed_kv, rope)}
        calls = [call.args for call in backend.register_buffer_checked.call_args_list]
        assert len(calls) == 2
        assert {(pointers[0], lengths[0]) for pointers, lengths in calls} == set(regions.items())
        assert worker._device_regions == regions
        addresses, sizes, selected = layout.prepare_values([(0, 8)], [2], [0])
        assert selected == [2]
        assert sum(sizes[0]) == 8 * (4 + 2) * 2
        payload = b"".join(
            ctypes.string_at(address, size) for address, size in zip(addresses[0], sizes[0], strict=True)
        )
        # Native component order: complete compressed latent chunk, then RoPE.
        oracle = b"".join(
            tensor[2, :, 0, :].contiguous().view(torch.uint8).numpy().tobytes() for tensor in (compressed_kv, rope)
        )
        assert payload == oracle
    finally:
        connector.shutdown()
