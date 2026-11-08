# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real upstream layouts and ordinary ReqMeta, with only device I/O simulated."""

import threading
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.v1.kv_cache_interface import MambaSpec, MLAAttentionSpec

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    AscendConnectorMetadata,
    LoadSpec,
    ReqMeta,
    RequestTracker,
    block_hash_to_str,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.shared_transfer import (
    SharedPoolTransfer,
    validate_shared_pool,
)

from .test_shared_io import MemoryBackend
from .test_shared_layout import attention


@pytest.mark.parametrize("cancel", [False, True])
def test_pd_live_state_waits_for_copy_before_finish_or_cancel(monkeypatch, cancel):
    from vllm_ascend.distributed.kv_transfer.kv_p2p.pd_pool.worker import PoolPDWorker
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.shared_layout import PoolObject, PoolRange

    copying, proceed, returned = threading.Event(), threading.Event(), threading.Event()
    monkeypatch.setattr(torch, "npu", SimpleNamespace(Event=MagicMock), raising=False)
    backend = MemoryBackend()
    common, _ = transfer(backend, 2, 0, True, True)
    pd = PoolPDWorker(SimpleNamespace(shared_pool=common))
    # These are live source views, as in KDA conv/recurrent and circular tail.
    tensors = [torch.arange(8, dtype=torch.float32) + i * 10 for i in range(3)]
    owners = ("conv", "recurrent", "ring")
    common.layer_bundles["kda"] = owners
    plans = tuple(
        PoolObject(name, tensor.numel() * 4, 1, (PoolRange(name, tensor.data_ptr(), 0, tensor.numel() * 4),))
        for name, tensor in zip(owners, tensors, strict=True)
    )
    pd.tails.append(common.io.begin(plans, writing=True))
    native_copy = backend.batch_copy

    def blocked_copy(*args):
        copying.set()
        assert proceed.wait(5)
        return native_copy(*args)

    backend.batch_copy = blocked_copy
    errors = []

    def finish():
        try:
            pd.abort() if cancel else pd.finish()
        except Exception as error:
            errors.append(error)
        finally:
            returned.set()

    thread = threading.Thread(target=finish)
    try:
        pd.save_layer("kda")
        assert copying.wait(2)
        thread.start()
        assert not returned.wait(0.1)
        assert backend.leases == set(owners)
        assert not backend.readable
    finally:
        proceed.set()
        if thread.ident is not None:
            thread.join(5)
        common.close()
    assert returned.is_set() and not errors
    assert bool(backend.leases) is cancel
    for name, tensor in zip(owners, tensors, strict=True):
        saved = bytes(backend.objects[name])
        assert saved == tensor.numpy().tobytes()
        tensor.fill_(-1)  # The next step may overwrite only after drain.
        assert bytes(backend.objects[name]) == saved
    assert backend.readable == (set() if cancel else set(owners))


@pytest.mark.parametrize("abort", [False, True])
def test_layerwise_copy_overlaps_compute_but_drains_before_release(monkeypatch, abort):
    entered, allow_copy, drained = threading.Event(), threading.Event(), threading.Event()

    class ReadyEvent:
        def record(self):
            pass

        def synchronize(self):
            entered.set()
            assert allow_copy.wait(5)

    monkeypatch.setattr(torch, "npu", SimpleNamespace(Event=ReadyEvent), raising=False)
    backend = MemoryBackend()
    writer, caches = transfer(backend, 2, 0, True, True)
    request = saved_request()
    request.event_id = 7
    writer.prepare_step(step(request))
    failures = []

    def finish():
        try:
            writer.abort_step() if abort else writer.finish_step()
        except Exception as error:
            failures.append(error)
        finally:
            drained.set()

    thread = threading.Thread(target=finish)
    try:
        for owner in caches:
            writer.save_layer(owner)
        # The forward thread returned while the sender still waits for NPU
        # readiness. Neither publication nor source-release ACK may happen.
        assert entered.wait(2)
        assert backend.leases and not backend.readable
        assert not writer.completed_events
        thread.start()
        assert not drained.wait(0.1)
        assert backend.leases
    finally:
        allow_copy.set()
        if thread.ident is not None:
            thread.join(5)
        writer.close()
    assert drained.is_set() and not failures
    assert bool(backend.leases) is abort
    assert bool(backend.readable) is not abort
    assert writer.completed_events == ({} if abort else {7: 1})


def test_layerwise_failure_drains_pending_tasks_without_publication(monkeypatch):
    monkeypatch.setattr(torch, "npu", SimpleNamespace(Event=MagicMock), raising=False)
    backend = MemoryBackend()
    backend.fail_copy = True
    writer, caches = transfer(backend, 2, 0, True, True)
    writer.prepare_step(step(saved_request()))
    try:
        for owner in caches:
            writer.save_layer(owner)
        writer.finish_step()
    except RuntimeError:
        pass
    else:
        pytest.fail("Failed asynchronous copy was not reported")
    with pytest.raises(RuntimeError):
        writer.close()
    assert not backend.readable and backend.leases  # Failed writes expire by native TTL.
    assert not writer.completed_events
    assert not writer._sender.is_alive()
    assert writer._sender.request_queue.unfinished_tasks == 0


def transfer(backend, tp, rank, layerwise, source, *, enforce_eager=True):
    layout, caches = attention(tp, rank, source=source, chunk=4)
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(enforce_eager=enforce_eager),
            kv_transfer_config=SimpleNamespace(
                kv_connector_extra_config={
                    "store_tp_size": 4,
                    "use_layerwise": layerwise,
                    "cache_prefix": "weights-v1",
                }
            ),
        ),
        m_store=backend,
        can_put=True,
        cache_transfer_granularity=4,
        kv_cache_config=SimpleNamespace(
            num_blocks=3,
            kv_cache_groups=[
                SimpleNamespace(layer_names=list(caches), kv_cache_spec=next(iter(layout.specs.values()))),
            ],
        ),
        tp_size=tp,
        tp_rank=rank,
        model_name="test-model",
        hash_block_size=2,
        block_size=4,
        dcp_size=1,
        token_database=SimpleNamespace(group_block_len={}),
        _as_cache_tuple=tuple,
    )
    result = SharedPoolTransfer(worker)
    result.register(caches)
    return result, caches


def step(*requests):
    meta = AscendConnectorMetadata(set(), set())
    for request in requests:
        meta.add_request(request)
    return meta


def saved_request():
    tracker = RequestTracker(
        req_id="writer",
        token_len=4,
        allocated_block_ids_by_group=[[1]],
        num_prompt_tokens=12,
        prefill_end_tokens=12,
    )
    # This is the first chunk of a longer prefill, not its final chunk.
    return ReqMeta.from_request_tracker(
        tracker,
        4,
        block_hashes=[b"hash2", b"hash4"],
        hash_block_size=2,
        is_last_chunk=False,
    )


def test_unnamed_layer_save_uses_execution_wait_not_prefetch(monkeypatch):
    monkeypatch.setattr(torch, "npu", SimpleNamespace(Event=MagicMock), raising=False)
    writer, caches = transfer(MemoryBackend(), 2, 0, True, True)
    writer.prepare_step(step(saved_request()))
    for owner in caches:
        writer.load_layer(owner)
        writer.load_layer("prefetched.other.layer", execution=False)
        assert writer.save_layer("") == owner
    writer.finish_step()
    assert writer._executing_owner is None
    with pytest.raises(RuntimeError, match="preceding layer execution wait"):
        writer.save_layer("")
    writer.close()


def test_unnamed_draft_save_cannot_reuse_previous_target_owner():
    writer, caches = transfer(MemoryBackend(), 2, 0, True, True)
    target = next(iter(caches))
    writer.load_layer(target)
    assert writer.save_layer("") == target
    writer.load_layer("draft.owner")
    assert writer.save_layer("") == "draft.owner"
    with pytest.raises(RuntimeError, match="preceding layer execution wait"):
        writer.save_layer("")
    writer.load_layer(target)
    writer.abort_step()
    with pytest.raises(RuntimeError, match="preceding layer execution wait"):
        writer.save_layer("")
    writer.close()


def test_sparse_state_lookup_requires_an_exact_common_checkpoint():
    executor, _ = transfer(MemoryBackend(), 2, 0, False, False)
    executor.layouts = {
        0: SimpleNamespace(
            chunk_size=4, hash_block_size=4, store_tp=1, state=False, key=lambda shard, digest: "kv:" + digest
        ),
        1: SimpleNamespace(
            chunk_size=4, hash_block_size=4, store_tp=1, state=True, key=lambda shard, digest: "state:" + digest
        ),
    }
    hashes = [b"a", b"b", b"c"]
    keys = {"kv:" + block_hash_to_str(value) for value in hashes}
    keys.add("state:" + block_hash_to_str(hashes[1]))
    executor.io.batch_is_exist = lambda requested: [key in keys for key in requested]
    assert executor.lookup(12, hashes) == 8  # Missing state at 4 is legal.
    keys.remove("kv:" + block_hash_to_str(hashes[0]))
    assert executor.lookup(12, hashes) == 0  # Missing attention history is not.
    keys.add("kv:" + block_hash_to_str(hashes[0]))
    executor.worker.cache_transfer_granularity = 12
    assert executor.lookup(12, hashes) == 0  # Do not invent an aligned state.


def test_state_plans_skip_null_history_and_load_only_final_checkpoint():
    executor, _ = transfer(MemoryBackend(), 2, 0, False, False)
    layout = SimpleNamespace(
        chunk_size=4,
        hash_block_size=4,
        replicated=False,
        state=True,
        objects=lambda start, block, digest: [(start, block, digest)],
    )
    executor.layouts = {0: layout}
    request = SimpleNamespace(
        save_start_token=0,
        save_end_token=12,
        block_ids_by_group=[[0, 1, 2]],
        block_hashes=[b"a", b"b", b"c"],
        load_spec=SimpleNamespace(vllm_cached_tokens=0, kvpool_cached_tokens=12),
    )
    assert [p[:2] for p in executor._plans(request, writing=True)] == [(4, 1), (8, 2)]
    assert [p[:2] for p in executor._plans(request, writing=False)] == [(8, 2)]
    request.block_ids_by_group = [[0, 1, 0]]
    with pytest.raises(ValueError, match="null"):
        executor._plans(request, writing=False)


@pytest.mark.parametrize("writer_tp,reader_tp", [(2, 4), (4, 2)])
@pytest.mark.parametrize("write_layerwise,read_layerwise", [(False, False), (True, False), (False, True), (True, True)])
def test_step_metadata_prefix_lookup_and_partial_hit_restore(
    monkeypatch,
    writer_tp,
    reader_tp,
    write_layerwise,
    read_layerwise,
):
    # CPU tests mock only the device completion fence; data moves through
    # the real generated addresses and object offsets using ctypes.memmove.
    monkeypatch.setattr(torch, "npu", SimpleNamespace(Event=MagicMock), raising=False)
    backend = MemoryBackend()
    for rank in range(writer_tp):
        writer, caches = transfer(backend, writer_tp, rank, write_layerwise, True)
        meta = step(saved_request())
        writer.prepare_step(meta)
        writer.start_load(meta)
        if write_layerwise:
            for owner in caches:
                writer.save_layer(owner)
        writer.finish_step()
        writer.close()
    assert len(backend.readable) == 4
    for rank in range(reader_tp):
        reader, caches = transfer(backend, reader_tp, rank, read_layerwise, False)
        assert reader.lookup(4, [b"hash2", b"hash4"]) == 4
        missing = backend.readable.pop()
        assert reader.lookup(4, [b"hash2", b"hash4"]) == 0
        backend.readable.add(missing)
        load = ReqMeta(
            req_id="reader",
            block_ids_by_group=[[2]],
            block_hashes=[b"hash2", b"hash4"],
            can_save=False,
            load_spec=LoadSpec(0, 3, True),
        )
        meta = step(load)
        reader.prepare_step(meta)
        reader.start_load(meta)
        if read_layerwise:
            for owner in caches:
                reader.load_layer(owner)
                reader.load_layer(owner)  # Duplicate readiness hooks are idempotent.
        reader.finish_step()
        _, expected = attention(reader_tp, rank, source=True, chunk=4)
        for owner, tensors in caches.items():
            for actual, wanted in zip(tensors, expected[owner]):
                torch.testing.assert_close(actual[4:6], wanted[2:4], rtol=0, atol=0)
                assert torch.all(actual[:4] == -99)
        reader.close()
    assert not backend.leases
    assert not backend.registered


def test_graph_step_copies_and_publishes_without_layer_callbacks(monkeypatch):
    event = MagicMock()
    monkeypatch.setattr(torch, "npu", SimpleNamespace(Event=lambda: event), raising=False)
    backend = MemoryBackend()
    for rank in range(2):
        writer, _ = transfer(backend, 2, rank, True, True, enforce_eager=False)
        writer.prepare_step(step(saved_request()))
        writer.finish_step()
        writer.close()
    assert len(backend.readable) == 4
    assert event.synchronize.call_count == 2

    reader, caches = transfer(backend, 2, 0, True, False, enforce_eager=False)
    meta = step(
        ReqMeta(
            req_id="graph-reader",
            block_ids_by_group=[[2]],
            block_hashes=[b"hash2", b"hash4"],
            can_save=False,
            load_spec=LoadSpec(0, 3, True),
        )
    )
    reader.prepare_step(meta)
    reader.start_load(meta)
    # A replay consumes the restored bytes before finish_step runs.
    _, expected = attention(2, 0, source=True, chunk=4)
    for owner, tensors in caches.items():
        for actual, wanted in zip(tensors, expected[owner]):
            torch.testing.assert_close(actual[4:6], wanted[2:4], rtol=0, atol=0)
    reader.finish_step()
    reader.close()
    assert not backend.leases


def test_save_range_skips_previous_chunks_and_uncommitted_hashes():
    worker, _ = transfer(MemoryBackend(), 2, 0, False, True)
    request = ReqMeta(
        req_id="next-chunk",
        block_ids_by_group=[[1, 2]],
        block_hashes=[b"h2", b"h4", b"h6", b"h8"],
        can_save=True,
        save_start_token=4,
        save_end_token=8,
    )
    plans = worker._plans(request, writing=True)
    assert len(plans) == 2 and all(plan.block_id == 2 for plan in plans)
    request.block_hashes = request.block_hashes[:3]
    with pytest.raises(ValueError, match="committed prefix hash"):
        worker._plans(request, writing=True)


def test_layerwise_missing_owner_cannot_publish(monkeypatch):
    monkeypatch.setattr(torch, "npu", SimpleNamespace(Event=MagicMock), raising=False)
    backend = MemoryBackend()
    writer, caches = transfer(backend, 2, 0, True, True)
    writer.prepare_step(step(saved_request()))
    writer.save_layer(next(iter(caches)))
    with pytest.raises(RuntimeError, match="partially copied"):
        writer.finish_step()
    assert not backend.readable and backend.leases  # Failed writes expire by native TTL.


def test_supported_contract_and_fail_fast_on_unsupported_features():
    layout, _ = attention(2, 0, source=False, chunk=4)
    spec = next(iter(layout.specs.values()))
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config={
                "store_tp_size": 4,
                "cache_prefix": "weights-v1",
                "backend": "memcache",
            }
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        use_v2_model_runner=True,
        model_config=SimpleNamespace(enforce_eager=True),
        speculative_config=None,
        kv_events_config=None,
    )
    cache_config = SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec, layer_names=["layer.0"])])
    validate_shared_pool(config, cache_config)
    config.model_config.enforce_eager = False
    validate_shared_pool(config, cache_config)
    for changed in (replace(spec, non_causal=True), replace(spec, tokens_per_state=2), replace(spec, max_tp_shards=1)):
        cache_config.kv_cache_groups[0].kv_cache_spec = changed
        with pytest.raises(ValueError):
            validate_shared_pool(config, cache_config)
    cache_config.kv_cache_groups[0].kv_cache_spec = spec
    config.speculative_config = SimpleNamespace(use_eagle=lambda: True)
    # Adding a default-false field is insufficient: grouping may have lost
    # every draft marker, leaving draft caches silently treated as target KV.
    with pytest.raises(ValueError, match="ownership"):
        validate_shared_pool(config, cache_config)
    cache_config.kv_cache_groups.append(
        SimpleNamespace(kv_cache_spec=replace(spec, is_draft=True), layer_names=["draft.layer.0"])
    )
    validate_shared_pool(config, cache_config)
    cache_config.kv_cache_groups.pop()
    # Speculators without a draft model (e.g. n-gram) have no draft KV owners.
    config.speculative_config = SimpleNamespace(use_eagle=lambda: False)
    validate_shared_pool(config, cache_config)
    config.use_v2_model_runner = False
    with pytest.raises(ValueError, match="MRV2"):
        validate_shared_pool(config, cache_config)
    config.use_v2_model_runner = True
    config.speculative_config = None
    config.kv_transfer_config.kv_connector_extra_config["store_tp_size"] = 2
    state_spec = MambaSpec(
        block_size=640,
        shapes=((3, 3072), (8, 128, 128)),
        dtypes=(torch.bfloat16, torch.float32),
        mamba_cache_mode="align",
    )
    cache_config.kv_cache_groups[0].kv_cache_spec = state_spec
    validate_shared_pool(config, cache_config)
    for invalid in (replace(state_spec, mamba_cache_mode="none"), replace(state_spec, shapes=((3, 3072),))):
        cache_config.kv_cache_groups[0].kv_cache_spec = invalid
        with pytest.raises(ValueError, match="aligned committed"):
            validate_shared_pool(config, cache_config)
    cache_config.kv_cache_groups[0].kv_cache_spec = MLAAttentionSpec(
        block_size=640,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        tokens_per_state=4,
    )
    validate_shared_pool(config, cache_config)
