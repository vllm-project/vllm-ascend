# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
import threading
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.config import CacheConfig, KVTransferConfig, SchedulerConfig
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MLAAttentionSpec,
)

from vllm_ascend.core.kv_cache_interface import AscendIndexerKPoolTailSpec, AscendMLAAttentionSpec
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pool_control import PoolControlServer
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pool_metadata import (
    PoolSaveRequest,
    PoolSourceProfile,
    PoolWorkerHandshake,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.attention_fence import record_attention_compute_start
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.cache_layout import describe_cache_bindings
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.key_codec import PoolKeyIdentity
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.kv_transfer import (
    KVCacheStoreLayerRecvingThread,
    KVCacheStoreLayerSendingThread,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.metadata import LayerLoadTask
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.object_adapter import LayerwiseObjectAdapter
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.pool_scheduler import LayerwiseKVPoolScheduler
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.pool_worker import LayerwiseKVPoolWorker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import AscendConnectorMetadata, LoadSpec, ReqMeta
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_cache_layout import PoolCacheLayout
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.prefix_lookup import probe_pool_prefix

from .test_pool_state_snapshot import kda


def memory_backend():
    """Native transport substitute; all layouts and pointer copies are real."""
    backend = MagicMock()
    backend.layerwise_data_plane = "gva"
    memory, committed = {}, set()

    def allocate(keys, sizes, ttl):
        for key, size in zip(keys, sizes, strict=True):
            assert key not in memory
            memory[key] = ctypes.create_string_buffer(size)
        return [ctypes.addressof(memory[key]) for key in keys]

    def copy(gvas, buffers, sizes, direction):
        for gva, buffer, size in zip(gvas, buffers, sizes, strict=True):
            ctypes.memmove(gva if direction == 0 else buffer, buffer if direction == 0 else gva, size)
        return 0

    def commit(keys, results):
        committed.update(key for key, code in zip(keys, results, strict=True) if code == 0)
        return [0] * len(keys)

    backend.batch_alloc.side_effect = allocate
    backend.batch_copy_gva.side_effect = copy
    backend.batch_write_finish.side_effect = commit
    backend.batch_is_readable.side_effect = lambda keys: [key in committed for key in keys]
    backend.batch_add_lease.side_effect = lambda keys, ttl: [0 if key in committed else -1 for key in keys]
    backend.batch_remove_lease.return_value = 0

    def revoke(keys):
        for key in keys:
            memory.pop(key, None)
            committed.discard(key)
        return [0] * len(keys)

    backend.batch_revoke.side_effect = revoke
    backend.batch_get_key_info.side_effect = lambda keys, **kwargs: [
        SimpleNamespace(size=lambda key=key: len(memory[key]), gva_list=lambda key=key: [ctypes.addressof(memory[key])])
        for key in keys
    ]
    return backend, memory, committed


def layout(caches, *, rank=0, tp=1, dcp=1, heads=2):
    spec = FullAttentionSpec(block_size=4, num_kv_heads=heads, head_size=2, dtype=torch.float16)
    config = SimpleNamespace(
        num_blocks=next(iter(caches.values())).shape[0],
        kv_cache_groups=[SimpleNamespace(layer_names=list(caches), kv_cache_spec=spec)],
    )
    return PoolCacheLayout(
        cache_config=config,
        bindings=describe_cache_bindings(config, caches),
        specs={name: spec for name in caches},
        tp_rank=rank,
        tp_size=tp,
        dcp_rank=rank % dcp,
        dcp_size=dcp,
        pcp_rank=0,
        pcp_size=1,
        total_num_kv_heads=2,
        pp_rank=0,
        base_layers=len(caches),
        pool_head_rank=rank,
        pool_layer_offset=0,
        data_plane="gva",
        worker_id=f"p{rank}",
    )


def sender(backend, layers=2):
    thread = KVCacheStoreLayerSendingThread(
        backend,
        MagicMock(),
        4,
        0,
        1,
        1,
        32,
        threading.Event(),
        layers,
        [threading.Event() for _ in range(layers)],
        [MagicMock() for _ in range(layers)],
        group_builders=[MagicMock()],
    )
    thread.start()
    assert thread.ready_event.wait(5)
    return thread


def profile():
    return PoolSourceProfile("source", "incarnation", PoolKeyIdentity("memcache-v1", "model", 1, 1))


def test_ordinary_full_worker_registration_and_bound_step_round_trip(monkeypatch):
    """Keep construction, registration and both original transfer queues real."""
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import pool_worker as base

    backend, _, committed = memory_backend()
    monkeypatch.setattr(base, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(base, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(base, "get_decode_context_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(base, "get_pcp_group", lambda: SimpleNamespace(world_size=1))
    # Replace only native client construction, not its capability/key class.
    monkeypatch.setattr(
        base,
        "importlib",
        SimpleNamespace(import_module=lambda path: SimpleNamespace(MemcacheBackend=lambda *a, **k: backend)),
    )
    monkeypatch.setattr(torch, "npu", SimpleNamespace(Event=MagicMock, current_stream=MagicMock), raising=False)
    names = ["model.layers.0.attn", "model.layers.1.attn"]
    caches = {name: torch.zeros((4, 2, 4, 2), dtype=torch.float16) for name in names}
    spec = FullAttentionSpec(block_size=4, num_kv_heads=2, head_size=2, dtype=torch.float16)
    cache = KVCacheConfig(
        4,
        [
            KVCacheTensor(
                size=t.untyped_storage().nbytes(),
                layers=[name],
                layer_stride=t.untyped_storage().nbytes(),
                block_stride=t.stride(0) * t.element_size(),
            )
            for name, t in caches.items()
        ],
        [KVCacheGroupSpec(names, spec)],
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            model="org/model",
            max_model_len=64,
            use_mla=False,
            hf_text_config=SimpleNamespace(num_hidden_layers=2),
            get_num_layers=lambda parallel: 2,
            get_total_num_hidden_layers=lambda: 2,
            get_layers_start_end_indices=lambda parallel: (0, 2),
            get_total_num_kv_heads=lambda: 2,
        ),
        parallel_config=SimpleNamespace(
            data_parallel_rank=0,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            rank=0,
        ),
        cache_config=CacheConfig(block_size=4, enable_prefix_caching=True),
        scheduler_config=SchedulerConfig(
            max_model_len=64,
            max_num_batched_tokens=32,
            max_num_seqs=4,
            disable_hybrid_kv_cache_manager=False,
            is_encoder_decoder=False,
        ),
        kv_transfer_config=KVTransferConfig(
            kv_connector="AscendStoreConnector",
            kv_role="kv_both",
            kv_connector_extra_config={"backend": "memcache", "control_host": "127.0.0.1"},
        ),
        speculative_config=None,
        additional_config={},
        kv_events_config=None,
    )
    worker = LayerwiseKVPoolWorker(config, True, cache)
    try:
        worker.register_kv_caches(caches)
        assert worker._object_adapter.writer is not None
        assert worker.kv_send_thread.is_alive() and worker.kv_recv_thread.is_alive()
        assert worker._object_handshake.profile.key_identity == worker.layerwise_keys.identity
        backend.register_buffer.assert_called_once()
        save = ReqMeta(
            "save",
            token_len_chunk=4,
            block_ids=[1],
            can_save=True,
            num_prompt_tokens=4,
            pool_request_digest="input",
            pool_group_hashes=(("hash",),),
        )
        metadata = AscendConnectorMetadata(set())
        metadata.add_request(save)
        worker.prepare_layerwise_step(metadata)
        for index, name in enumerate(names):
            caches[name][1].fill_(11 + index)
            worker.wait_for_layer_load()
            record_attention_compute_start()
            worker.save_kv_layer(metadata, name)
        assert len(committed) == 1
        for tensor in caches.values():
            tensor.fill_(-1)
        load = ReqMeta(
            "load",
            token_len_chunk=4,
            block_ids=[2],
            can_save=False,
            load_spec=LoadSpec(0, 4, True),
            pool_request_digest="other-input",
            pool_group_hashes=(("hash",),),
        )
        metadata = AscendConnectorMetadata(set())
        metadata.add_request(load)
        worker.prepare_layerwise_step(metadata)
        for index, name in enumerate(names):
            worker.wait_for_layer_load()
            record_attention_compute_start()
            assert torch.all(caches[name][2] == 11 + index)
            assert torch.all(caches[name][1] == -1)
            worker.save_kv_layer(metadata, name)
        assert worker.kv_recv_thread.get_and_clear_finished_requests() == {"load"}
        # Ordinary layerwise loads finish synchronously at their layer gate;
        # they do not use the bulk async scheduler completion notification.
        assert worker.get_finished(set(), metadata) == (set(), set())
    finally:
        worker.shutdown()
    assert not worker.kv_send_thread.is_alive() and not worker.kv_recv_thread.is_alive()
    backend.close.assert_called_once()


@pytest.mark.parametrize("kind", ["packed", "tail"])
def test_special_component_objects_reuse_across_attempts_and_restore_with_page_padding(kind):
    backend, _, committed = memory_backend()
    name = "model.layers.0.attn"
    if kind == "packed":
        spec = AscendMLAAttentionSpec(
            block_size=4, num_kv_heads=1, head_size=8, dtype=torch.uint8, cache_sparse_sfa_c8=True
        )
        source = torch.arange(64, dtype=torch.uint8).reshape(2, 32)
        storage = torch.full((2, 40), 255, dtype=torch.uint8)
        target, end, blocks = storage[:, :32], 6, (0, 1)
    else:
        spec = AscendIndexerKPoolTailSpec(
            block_size=7, num_kv_heads=1, head_size=2, dtype=torch.float32, sliding_window=4, compress_ratio=4
        )
        source = torch.arange(56, dtype=torch.float32).reshape(2, 7, 4)
        storage = torch.full((2, 10, 4), -1.0)
        target, end, blocks = storage[:, :7], 15, (1,)
    config = SimpleNamespace(num_blocks=2, kv_cache_groups=[SimpleNamespace(layer_names=[name], kv_cache_spec=spec)])

    def special(tensor):
        return replace(
            layout({name: tensor}),
            cache_config=config,
            specs={name: spec},
            bindings=describe_cache_bindings(config, {name: tensor}),
            total_num_kv_heads=1,
        )

    thread = sender(backend, 1)
    writer = LayerwiseObjectAdapter(backend, special(source), profile(), lambda name: 0, sender=thread)
    reader = LayerwiseObjectAdapter(backend, special(target), profile(), lambda name: 0)
    request = PoolSaveRequest("request", "first", "same-input", 0, end, (blocks,), (("hash",),))
    try:
        writer.prepare_save("request", request)
        writer.submit_save_layer(0, MagicMock(), MagicMock())
        assert writer.finish_step() == []
        sources = writer.snapshot_sources("request", end)
        # Retry and another engine lifetime find precisely the same objects.
        keys = set(committed)
        alternate = replace(profile(), profile_id="other-profile", engine_incarnation="other-engine")
        _, _, repeated = writer.layout.build_source_plan(replace(request, attempt_id="retry"), alternate)
        assert {ref.resolve(alternate.key_identity)[0].key for ref in repeated} == keys
        assert any(ref.object_id.startswith("pool-component-v1:") for ref in repeated)
        assert all(ref.object_id.startswith(("pool-full-v1:", "pool-component-v1:")) for ref in repeated)
        reader.prepare_load("restored", request, sources=sources)
        reader.load_layer(0, MagicMock())
        assert reader.finish_step() == []
        for block in blocks:
            assert torch.equal(target[block], source[block])
        if kind == "packed":
            assert torch.all(storage[:, 32:] == 255)
        else:
            assert torch.all(storage[:, 7:] == -1)
            assert torch.all(target[0] == -1)
    finally:
        reader.close()
        writer.close()
        thread.stop()
        thread.join(5)


def test_ordinary_writer_and_reader_preserve_rotating_hbm_layer_bytes():
    backend, memory, _ = memory_backend()
    cache = torch.zeros((4, 2, 4, 2), dtype=torch.float16)
    shared = layout({"layer.0": cache, "layer.1": cache})
    thread = sender(backend)
    adapter = LayerwiseObjectAdapter(
        backend, shared, profile(), lambda name: int(name.rsplit(".", 1)[1]), sender=thread
    )
    request = PoolSaveRequest("request", "attempt", "input", 0, 4, ((1,),), (("hash",),))
    finished = MagicMock()
    try:
        adapter.prepare_save("request", request)
        for index, value in enumerate((11, 29)):
            cache[1].fill_(value)
            done = threading.Event()
            adapter.submit_save_layer(index, MagicMock(), done.set)
            assert done.wait(5)  # Reuse only after this layer's actual copy.
        assert adapter.finish_step() == []
        assert thread.is_alive()
        key = next(iter(memory))
        expected = torch.cat((torch.full((16,), 11, dtype=torch.float16), torch.full((16,), 29, dtype=torch.float16)))
        assert ctypes.string_at(ctypes.addressof(memory[key]), 64) == expected.numpy().tobytes()
        cache.fill_(-1)
        adapter.prepare_load("request", request)
        for index, value in enumerate((11, 29)):
            adapter.load_layer(index, finished)
            assert torch.all(cache[1] == value)
            assert finished.call_count == index
        assert adapter.finish_step() == []
    finally:
        adapter.close()
        thread.stop()
        thread.join(5)
    assert not thread.is_alive()


def test_ordinary_cp_component_objects_restore_to_a_different_tp_layout():
    backend, _, _ = memory_backend()
    adapters, threads, sources = [], [], []
    try:
        request = PoolSaveRequest("request", "attempt", "input", 0, 8, ((1,),), (("hash",),))
        for rank in range(2):
            cache = torch.zeros((4, 2, 4, 2), dtype=torch.float16)
            cache[1].copy_(torch.arange(rank * 16, (rank + 1) * 16).reshape(4, 2, 2).permute(1, 0, 2))
            local = layout({"layer.0": cache}, rank=rank, tp=2, dcp=2)
            _, _, refs = local.build_source_plan(request, profile())
            assert all(ref.object_id.startswith("pool-component-v1:") for ref in refs)
            sources.extend(part for ref in refs for part in ref.resolve(profile().key_identity))
            thread = sender(backend, 1)
            threads.append(thread)
            adapter = LayerwiseObjectAdapter(backend, local, profile(), lambda name: 0, sender=thread)
            adapters.append(adapter)
            adapter.prepare_save("request", request)
            adapter.submit_save_layer(0, MagicMock(), MagicMock())
            assert adapter.finish_step() == []
        target = torch.full((4, 4, 1, 2), -1, dtype=torch.float16)
        local = layout({"layer.0": target}, tp=2, heads=1)
        reader = LayerwiseObjectAdapter(backend, local, profile(), lambda name: 0)
        adapters.append(reader)
        request = replace(request, block_ids=((1, 2),), group_hashes=(("h0", "h1"),))
        reader.prepare_load("request", request, sources=sources)
        done = MagicMock()
        reader.load_layer(0, done)
        expected = torch.arange(32, dtype=torch.float16).reshape(8, 2, 2)[:, :1]
        assert torch.equal(target[1:3].reshape(8, 1, 2), expected)
        done.assert_called_once_with("request")
        assert reader.finish_step() == []
    finally:
        for adapter in adapters:
            adapter.close()
        for thread in threads:
            thread.stop()
            thread.join(5)


def test_equal_size_hnd_nhd_objects_do_not_alias_and_can_be_restored_by_descriptors():
    backend, memory, _ = memory_backend()
    source = torch.arange(64, dtype=torch.float16).reshape(4, 2, 4, 2)
    nhd = source.permute(0, 2, 1, 3).contiguous()
    target = torch.full_like(source, -1)
    request = PoolSaveRequest("request", "attempt", "input", 0, 4, ((1,),), (("same-hash",),))
    adapters, threads = [], []
    try:
        for cache in (source, nhd):
            local = layout({"layer.0": cache})
            thread = sender(backend, 1)
            threads.append(thread)
            adapter = LayerwiseObjectAdapter(backend, local, profile(), lambda name: 0, sender=thread)
            adapters.append(adapter)
            adapter.prepare_save("request", request)
            adapter.submit_save_layer(0, MagicMock(), MagicMock())
            assert adapter.finish_step() == []
        assert len(memory) == 2  # Same byte count must not imply same format.
        _, _, refs = adapters[1].layout.build_source_plan(request, profile())
        sources = [part for ref in refs for part in ref.resolve(profile().key_identity)]
        reader = LayerwiseObjectAdapter(backend, layout({"layer.0": target}), profile(), lambda name: 0)
        adapters.append(reader)
        reader.prepare_load("request", request, sources=sources)
        reader.load_layer(0, MagicMock())
        assert reader.finish_step() == []
        assert torch.equal(target[1], source[1])
        assert torch.all(target[0] == -1)
    finally:
        for adapter in adapters:
            adapter.close()
        for thread in threads:
            thread.stop()
            thread.join(5)


@pytest.mark.parametrize("recover", [False, True])
def test_ordinary_kda_objects_restore_through_v2_projection_rules(recover):
    backend, _, _ = memory_backend()
    name = "model.layers.0.linear_attn"
    adapters, threads = [], []
    tensors = []
    try:
        for heads, rank, tp in ((2, 0, 1), (1, 1, 2)):
            spec, values, components = kda(heads, 3, recover=recover, mode="align")
            tensors.append(values)
            config = SimpleNamespace(
                num_blocks=4, kv_cache_groups=[SimpleNamespace(layer_names=[name], kv_cache_spec=spec)]
            )
            local = replace(
                layout({"layer.0": torch.zeros((4, 2, 4, 2), dtype=torch.float16)}),
                cache_config=config,
                bindings={name: components},
                specs={name: spec},
                tp_rank=rank,
                tp_size=tp,
                pool_head_rank=rank,
            )
            thread = sender(backend, 1) if tp == 1 else None
            if thread is not None:
                threads.append(thread)
            adapters.append(LayerwiseObjectAdapter(backend, local, profile(), lambda layer: 0, sender=thread))
        source, target = tensors
        for tensor in source:
            tensor[1].copy_(torch.arange(tensor[1].numel()).reshape_as(tensor[1]))
        request = PoolSaveRequest("request", "attempt", "input", 0, 4, ((1,),), (("hash",),))
        writer, reader = adapters
        _, _, refs = writer.layout.build_source_plan(request, profile())
        sources = [part for ref in refs for part in ref.resolve(profile().key_identity)]
        writer.prepare_save("request", request)
        writer.submit_save_layer(0, MagicMock(), MagicMock())
        assert writer.finish_step() == []
        reader.prepare_load("request", request, sources=sources)
        reader.load_layer(0, MagicMock())
        assert reader.finish_step() == []
        for projection in range(3):
            torch.testing.assert_close(
                target[0][1, :, projection * 2 : (projection + 1) * 2],
                source[0][1, :, projection * 4 + 2 : (projection + 1) * 4],
            )
        for p, d in zip(source[1:], target[1:], strict=True):
            torch.testing.assert_close(d[1, 0], p[1, 1])
            assert torch.all(d[0] == -1)
    finally:
        for adapter in adapters:
            adapter.close()
        for thread in threads:
            thread.stop()
            thread.join(5)


def test_component_load_obeys_existing_save_compute_and_external_slot_gates():
    thread = KVCacheStoreLayerRecvingThread.__new__(KVCacheStoreLayerRecvingThread)
    thread.load_disabled = threading.Event()
    thread.layer_save_finished_events = [threading.Event()]
    thread.sync_save_events = [MagicMock()]
    thread.layer_load_finished_events = [threading.Event()]
    order = []
    thread.save_failure_checker = lambda: order.append("check")
    thread.external_slot_release_waiter = lambda layer: order.append("release")
    thread._wait_for_event = (
        lambda event, *args: order.append("save" if event is thread.layer_save_finished_events[0] else "compute")
        or True
    )
    thread.sync_save_events[0].synchronize.side_effect = lambda: order.append("synchronize")
    task = LayerLoadTask(
        0, [], 0, attention_start_gate=threading.Event(), object_load=lambda: order.append("read") or True
    )
    assert thread._copy_gva_layers(task)
    assert order == ["save", "check", "synchronize", "compute", "release", "read"]
    order.clear()
    thread.load_disabled.set()
    assert not thread._copy_gva_layers(task)
    assert "read" not in order


def test_failed_component_read_never_finishes_a_partially_restored_request():
    backend, _, _ = memory_backend()
    cache = torch.zeros((4, 2, 4, 2), dtype=torch.float16)
    adapter = LayerwiseObjectAdapter(backend, layout({"layer.0": cache}), profile(), lambda name: 0)
    request = PoolSaveRequest("request", "attempt", "input", 0, 4, ((1,),), (("hash",),))
    adapter.prepare_load("request", request)
    backend.absent_read_lease_keys.return_value = {adapter._load_layers[0][0][1][0].key}
    finished, failed = MagicMock(), MagicMock()
    try:
        adapter.load_layer(0, finished, failed)
        finished.assert_not_called()
        assert failed.call_args.args[0] == "request"
        backend.batch_copy_gva.assert_not_called()
        assert adapter.finish_step() == []
    finally:
        adapter.close()


def test_ordinary_worker_hooks_publish_only_after_all_layer_copies():
    backend, memory, committed = memory_backend()
    cache = torch.zeros((4, 2, 4, 2), dtype=torch.float16)
    thread = sender(backend)
    adapter = LayerwiseObjectAdapter(
        backend,
        layout({"layer.0": cache, "layer.1": cache}),
        profile(),
        lambda name: int(name.rsplit(".", 1)[1]),
        sender=thread,
    )
    worker = LayerwiseKVPoolWorker.__new__(LayerwiseKVPoolWorker)
    worker._object_adapter = adapter
    worker._masked_runs_cache = {}
    worker.num_layers = 2
    worker.layerwise_offload = True
    worker.cacheable_group_ids = [0]
    worker.can_put = True
    worker.kv_role = "kv_producer"
    worker.consumer_is_to_put = False
    worker._is_layerwise_save_owner = lambda group: True
    worker._attention_saved_layers = set()
    worker.sync_save_events = thread.sync_save_events
    worker.layer_save_finished_events = thread.layer_save_finished_events
    worker.kv_send_thread = thread
    worker.block_key_hybrid = False
    worker.use_block_key_layerwise = False
    worker.prefetch_layer_map = {}
    worker._save_disabled = threading.Event()
    request = ReqMeta(
        "request",
        token_len_chunk=4,
        target_token_len=4,
        block_ids=[1],
        can_save=True,
        is_last_chunk=True,
        pool_request_digest="input",
        pool_group_hashes=(("hash",),),
    )
    try:
        worker.process_layer_data([request])
        cache[1].fill_(7)
        worker._submit_attention_save(0)
        assert worker.layer_save_finished_events[0].wait(5)
        assert not committed  # One complete object still lacks layer 1.
        cache[1].fill_(19)
        worker._submit_attention_save(1)
        worker._wait_for_final_layer_save(2, thread)
        assert len(committed) == 1 and "pool-full-v1:" in next(iter(committed))
        assert thread.get_and_clear_finished_requests() == {"request"}
        assert not adapter.writes
        expected = torch.cat((torch.full((16,), 7, dtype=torch.float16), torch.full((16,), 19, dtype=torch.float16)))
        assert ctypes.string_at(ctypes.addressof(memory[next(iter(committed))]), 64) == expected.numpy().tobytes()
    finally:
        adapter.close()
        thread.stop()
        thread.join(5)


def test_ordinary_scheduler_discovers_component_objects_through_shared_prefix_query():
    backend, _, _ = memory_backend()
    # Force the component recipe rather than a legacy whole-object key.
    # A strided physical page is semantically the same token/head cache.
    cache = torch.zeros((4, 2, 8, 2), dtype=torch.float16)[:, :, ::2, :]
    local = layout({"layer.0": cache})
    thread = sender(backend, 1)
    adapter = LayerwiseObjectAdapter(backend, local, profile(), lambda name: 0, sender=thread)
    source = PoolSaveRequest("old-request", "attempt", "old-input", 0, 4, ((1,),), ((b"a".hex(),),))
    adapter.prepare_save("old-request", source)
    adapter.submit_save_layer(0, MagicMock(), MagicMock())
    assert adapter.finish_step() == []
    assert backend.batch_write_finish.called
    server = PoolControlServer(
        "127.0.0.1", lambda query: probe_pool_prefix(local, backend, profile(), "pp0/pcp0/tp0", query)
    )
    scheduler = LayerwiseKVPoolScheduler.__new__(LayerwiseKVPoolScheduler)
    scheduler.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1, pipeline_parallel_size=1, prefill_context_parallel_size=1
        ),
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config={}),
    )
    scheduler.use_layerwise_transfer = True
    scheduler.original_block_size = [4]
    scheduler.cacheable_group_ids = [0]
    scheduler._build_cache_coordinator = lambda: None
    scheduler.hash_block_size = 4
    scheduler._object_inputs, scheduler._object_hits = {}, {}
    scheduler._object_queries = None
    scheduler.layerwise_offload = False
    scheduler.load_specs = {}
    scheduler.kv_role = "kv_producer"
    scheduler.use_eagle = False
    scheduler.use_layerwise = True
    scheduler.load_async = False
    request = SimpleNamespace(
        request_id="new-request",
        prompt_token_ids=list(range(9)),
        block_hashes=[b"a", b"b"],
        num_tokens=9,
        mm_features=[],
        lora_request=None,
        cache_salt=None,
    )
    try:
        worker = PoolWorkerHandshake("pp0/pcp0/tp0", replace(profile(), group_block_sizes=(4,)), server.endpoint)
        scheduler.set_object_handshakes({0: worker})
        scheduler.on_new_request(request)
        assert scheduler.get_num_new_matched_tokens(request, 0) == (4, False)
        assert scheduler.load_specs[request.request_id].can_load
        hit = scheduler._object_hits[request.request_id]
        assert hit.token_end == 4
        assert all(ref.object_id.startswith("pool-component-v1:") for ref in hit.objects)
        # A real object eviction after the query is still a read error; lookup
        # never promises that the later transfer cannot fail.
        backend.batch_is_readable.side_effect = lambda keys: [False] * len(keys)
        assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
        assert not scheduler.load_specs and not scheduler._object_hits
        with pytest.raises(ValueError, match="missing a PP/PCP/TP"):
            scheduler.set_object_handshakes({})
    finally:
        scheduler.close_object_queries()
        server.close()
        adapter.close()
        thread.stop()
        thread.join(5)


def test_ordinary_offload_uses_committed_snapshot_after_speculative_length_rollback():
    backend, _, _ = memory_backend()
    cache = torch.arange(64, dtype=torch.float16).reshape(4, 2, 4, 2)
    expected = cache[1:3].clone()
    thread = sender(backend, 1)
    adapter = LayerwiseObjectAdapter(backend, layout({"layer.0": cache}), profile(), lambda name: 0, sender=thread)
    worker = LayerwiseKVPoolWorker.__new__(LayerwiseKVPoolWorker)
    worker._object_adapter = adapter
    worker._masked_runs_cache = {}
    worker.num_layers = 1
    worker.layerwise_offload = True
    worker.independent_layers = []
    worker.prefetch_layer_map = {0: 0}
    worker.cacheable_group_ids = [0]
    worker.can_put = True
    worker._is_layerwise_save_owner = lambda group: True
    worker.kv_recv_thread = SimpleNamespace(set_finished_request=MagicMock(), _mark_failed_requests=MagicMock())
    request = PoolSaveRequest("request", "previous", "input", 0, 8, ((1, 2),), (("h0",),))
    try:
        # The last page has no committed hash yet. Its identity names the
        # actually saved end=8, not the accepted length of the next step.
        adapter.prepare_save("request", request)
        adapter.submit_save_layer(0, MagicMock(), MagicMock())
        assert adapter.finish_step() == []
        cache.fill_(-1)
        meta = ReqMeta(
            "request",
            token_len_chunk=6,
            target_token_len=6,
            block_ids=[2, 3],
            can_save=False,
            pool_request_digest="input",
            pool_group_hashes=(("h0",),),
            load_spec=LoadSpec(
                5,
                5,
                True,
                layerwise_previous_step=True,
                layerwise_snapshot_end_token=8,
                layerwise_snapshot_start_block=1,
                layerwise_snapshot_block_index=1,
            ),
        )
        worker.process_layer_data([meta])
        assert worker._load_object_layer(0)
        assert torch.equal(cache[2], expected[0])
        assert torch.equal(cache[3, :, :1], expected[1, :, :1])
        assert torch.all(cache[3, :, 1:] == -1)
        worker.kv_recv_thread.set_finished_request.assert_called_once_with("request")
        worker.kv_recv_thread._mark_failed_requests.assert_not_called()
        assert adapter.finish_step() == []
    finally:
        adapter.close()
        thread.stop()
        thread.join(5)


@pytest.mark.parametrize("failed_cleanup", [False, True])
def test_unaccepted_step_objects_are_private_and_cannot_create_prefix_hits(failed_cleanup):
    backend, _, committed = memory_backend()
    source = torch.zeros((4, 2, 4, 2), dtype=torch.float16)
    target = torch.full_like(source, -1)
    local = layout({"layer.0": source})
    thread = sender(backend, 1)
    adapter = LayerwiseObjectAdapter(backend, local, profile(), lambda name: 0, sender=thread)
    reader = LayerwiseObjectAdapter(backend, layout({"layer.0": target}), profile(), lambda name: 0)
    try:
        for request_id, tail in (("a", 11), ("b", 29)):
            source[1].fill_(7)  # Same accepted prefix.
            source[2].fill_(tail)  # Different unaccepted histories.
            request = PoolSaveRequest(request_id, "step", "same-input", 0, 8, ((1, 2),), (("h0",),))
            adapter.prepare_save(request_id, request, transient=True)
            adapter.submit_save_layer(0, MagicMock(), MagicMock())
            assert adapter.finish_step() == []
        a = {part.key for part in adapter.snapshot_sources("a", 5)}
        b = {part.key for part in adapter.snapshot_sources("b", 5)}
        assert len(a & b) == 1  # Only the accepted full prefix is shared.
        private = {key for key in a | b if "pool-step-v1:" in key}
        assert len(private) == 2
        for request_id, value in (("a", 11), ("b", 29)):
            target.fill_(-1)
            request = PoolSaveRequest(request_id, "next", "same-input", 0, 5, ((1, 2),), (("h0",),))
            reader.prepare_load(request_id, request, sources=adapter.snapshot_sources(request_id, 5))
            reader.load_layer(0, MagicMock())
            assert reader.finish_step() == []
            assert torch.all(target[1] == 7)
            assert torch.all(target[2, :, :1] == value)
            assert torch.all(target[2, :, 1:] == -1)
        query = {
            "op": "prefix",
            "incarnation": profile().engine_incarnation,
            "mode": "probe",
            "group_ids": [0],
            "granularity": 4,
            "request": {
                "transfer_id": "prefix",
                "attempt_id": "query",
                "request_digest": "same-input",
                "token_start": 0,
                "token_end": 8,
                "group_hashes": [["h0", "h1"]],
            },
        }
        assert probe_pool_prefix(local, backend, profile(), "p0", query)["ready_ends"] == [4]
        original_revoke = backend.batch_revoke.side_effect
        if failed_cleanup:
            backend.batch_revoke.side_effect = lambda keys: [-7] * len(keys)
            with pytest.raises(RuntimeError, match="snapshot cleanup failed"):
                adapter.release_requests(("a", "b"))
            assert adapter._retired_snapshot_keys == private
            assert private <= committed
            backend.batch_revoke.side_effect = original_revoke
            adapter.release_requests(())
        else:
            adapter.release_requests(("a", "b"))
        assert not private & committed
        assert a & b <= committed
        assert not adapter.snapshots and not adapter._retired_snapshot_keys
    finally:
        reader.close()
        adapter.close()
        thread.stop()
        thread.join(5)


def test_failed_prefix_read_cancels_planned_saves_before_shared_publication():
    backend, _, committed = memory_backend()
    cache = torch.zeros((4, 2, 4, 2), dtype=torch.float16)
    local = layout({"layer.0": cache})
    thread = sender(backend, 1)
    adapter = LayerwiseObjectAdapter(backend, local, profile(), lambda name: 0, sender=thread)
    seed = PoolSaveRequest("seed", "step", "input", 0, 4, ((1,),), (("h0",),))
    current = PoolSaveRequest("request", "step", "input", 0, 8, ((1, 2),), (("h0", "h1"),))
    worker = LayerwiseKVPoolWorker.__new__(LayerwiseKVPoolWorker)
    worker._object_adapter = adapter
    worker._object_requests = {"request": ReqMeta("request", block_ids=[1, 2])}
    worker.kv_recv_thread = SimpleNamespace(set_finished_request=MagicMock(), _mark_failed_requests=MagicMock())
    try:
        adapter.prepare_save("seed", seed)
        adapter.submit_save_layer(0, MagicMock(), MagicMock())
        assert adapter.finish_step() == []
        existing = set(committed)
        adapter.prepare_load("request", replace(seed, transfer_id="request"))
        adapter.prepare_save("request", current)
        backend.batch_copy_gva.side_effect = lambda *args: -7
        assert not worker._load_object_layer(0)
        done = MagicMock()
        adapter.submit_save_layer(0, MagicMock(), done)
        done.assert_called_once()
        assert adapter.finish_step() == []  # Drained cancellation permits framework recompute.
        assert committed == existing
        assert "request" not in adapter.snapshots
        worker.kv_recv_thread._mark_failed_requests.assert_called_once()
        worker.kv_recv_thread.set_finished_request.assert_not_called()
        backend.batch_revoke.assert_not_called()  # Never delete another request's prefix.
    finally:
        adapter.close()
        thread.stop()
        thread.join(5)


def test_mla_replica_restores_leader_step_objects_without_owning_or_rewriting_them():
    backend, _, committed = memory_backend()
    values = [torch.arange(64, dtype=torch.float16).reshape(4, 4, 1, 4), torch.full((4, 4, 1, 4), -1.0)]
    values[1] = values[1].to(torch.float16)
    spec = MLAAttentionSpec(block_size=4, num_kv_heads=1, head_size=4, dtype=torch.float16)
    config = SimpleNamespace(
        num_blocks=4, kv_cache_groups=[SimpleNamespace(layer_names=["layer.0"], kv_cache_spec=spec)]
    )
    adapters, threads = [], []
    request = PoolSaveRequest("request", "step", "input", 0, 5, ((1, 2),), (("h0",),))
    try:
        for rank, cache in enumerate(values):
            local = replace(
                layout({"layer.0": cache}),
                cache_config=config,
                specs={"layer.0": spec},
                bindings=describe_cache_bindings(config, {"layer.0": cache}),
                tp_size=2,
                tp_rank=rank,
                pool_head_rank=0,
                total_num_kv_heads=1,
            )
            thread = sender(backend, 1)
            threads.append(thread)
            adapter = LayerwiseObjectAdapter(
                backend,
                local,
                replace(profile(), profile_id=f"p{rank}", engine_incarnation=f"e{rank}"),
                lambda name: 0,
                sender=thread,
            )
            adapters.append(adapter)
            adapter.prepare_save(
                "request",
                request,
                group_ids={0} if rank == 0 else set(),
                transient=True,
                step_id="scheduler-request-step",
            )
            adapter.submit_save_layer(0, MagicMock(), MagicMock())
        owner, replica = adapters
        assert owner.finish_step() == replica.finish_step() == []
        owner_sources = owner.snapshot_sources("request", 5)
        replica_sources = replica.snapshot_sources("request", 5)
        assert {part.key for part in owner_sources} == {part.key for part in replica_sources}
        assert backend.batch_alloc.call_count == 1
        replica.prepare_load(
            "request",
            replace(request, block_ids=((2, 3),)),
            sources=replica_sources,
        )
        replica.load_layer(0, MagicMock())
        assert replica.finish_step() == []
        assert torch.equal(values[1][2], values[0][1])
        assert torch.equal(values[1][3, :1], values[0][2, :1])
        replica.release_requests(("request",))
        backend.batch_revoke.assert_not_called()
        owner.release_requests(("request",))
        assert backend.batch_revoke.call_count == 1
        assert all("pool-step-v1:" not in key for key in committed)
    finally:
        for adapter in adapters:
            adapter.close()
        for thread in threads:
            thread.stop()
            thread.join(5)
