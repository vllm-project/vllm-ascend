from contextlib import nullcontext
from threading import Event, get_ident
from types import SimpleNamespace

import pytest
import torch

import vllm_ascend.worker.v2.kvpp as kvpp_module
from vllm_ascend.worker.v2.kvpp import (
    KVPPRuntime,
    KVPPScheduler,
    build_layer_cache_bundles,
    select_active_pages,
)


def _layer(index: int) -> str:
    return f"model.layers.{index}.self_attn.attn"


def _indexer(index: int) -> str:
    return f"model.layers.{index}.self_attn.indexer.k_cache"


def _scheduler(layer_count: int = 2) -> KVPPScheduler:
    owners = {_layer(index): 0 for index in range(layer_count)}
    transport = SimpleNamespace(initialize_transport=lambda _caches, _bundles, _max_pages: None)
    return KVPPScheduler(
        kvpp_group=SimpleNamespace(rank_in_group=0, world_size=1),
        layer_owner_ranks=owners,
        kv_caches={layer_name: object() for layer_name in owners},
        num_physical_blocks=10,
        tokens_per_block=4,
        max_active_pages=10,
        transport=transport,
    )


def _begin(scheduler: KVPPScheduler) -> None:
    scheduler.schedule_forward(
        torch.tensor([[7, 2, -1], [2, 8, 12]], dtype=torch.int32),
        [5, 9],
    )


def test_kvpp_06_builds_sparse_main_and_indexer_cache_bundles():
    owners = {
        _layer(0): 0,
        _indexer(0): 0,
        _layer(1): 1,
        _indexer(1): 1,
    }

    layer_cache_bundles = build_layer_cache_bundles(owners, (_layer(0), _layer(1)))

    assert layer_cache_bundles == {
        _layer(0): (_layer(0), _indexer(0)),
        _layer(1): (_layer(1), _indexer(1)),
    }


def test_kvpp_07_active_pages_are_fixed_shape_deduplicated_and_masked():
    table = torch.tensor([[7, 2, -1, 0], [2, 8, 12, 0]], dtype=torch.int32)
    original = table.clone()

    pages = select_active_pages(table, [5, 9], tokens_per_block=4, num_physical_blocks=10)

    assert pages.physical_page_ids.shape == (table.numel(),)
    assert pages.physical_page_ids.tolist() == [2, 2, 7, 8, 10, 10, 10, 10]
    assert pages.valid_page_mask.tolist() == [True, False, True, True, False, False, False, False]
    assert pages.staging_page_indices.tolist() == [0, 0, 1, 2, 2, 2, 2, 2]
    assert torch.equal(table, original)


def test_kvpp_first_chunk_has_no_computed_pages():
    table = torch.tensor([[7, 2, 1, 0]], dtype=torch.int32)

    pages = select_active_pages(table, [0], tokens_per_block=4, num_physical_blocks=10)

    assert pages.physical_page_ids.tolist() == [10, 10, 10, 10]
    assert pages.valid_page_mask.tolist() == [False, False, False, False]
    assert pages.staging_page_indices.tolist() == [-1, -1, -1, -1]


def test_kvpp_08_prefetch_starts_when_scheduled_and_advances_on_wait():
    scheduler = _scheduler()
    _begin(scheduler)
    assert scheduler._next_attention_layer_index == 0

    scheduler.wait_for_layer(_layer(0))
    assert scheduler._next_attention_layer_index == 1

    scheduler.wait_for_layer(_layer(1))
    assert scheduler._next_attention_layer_index == 2
    scheduler.complete_forward()
    assert scheduler._active_pages is None


@pytest.fixture
def parallel_scheduler(monkeypatch):
    calls = []
    recorded_events = []
    schedulers = []

    class FakeEvent:
        def record(self, stream):
            recorded_events.append(self)

    class FakeStream:
        def wait_event(self, event):
            calls.append(("wait", event))

    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            Event=FakeEvent,
            Stream=FakeStream,
            current_device=lambda: 0,
            current_stream=lambda: "compute",
            set_device=lambda _device: None,
            stream=lambda _stream: nullcontext(),
        ),
        raising=False,
    )
    monkeypatch.setattr(kvpp_module.dist, "send", lambda _token, dst, group, tag: calls.append(("send", dst, tag)))
    monkeypatch.setattr(kvpp_module.dist, "recv", lambda _token, src, group, tag: calls.append(("recv", src, tag)))

    def copy(direction):
        def enqueue(bundle, pages, stream):
            calls.append((direction, bundle, pages))
            return SimpleNamespace(synchronize=lambda: calls.append(("complete", direction)))

        return enqueue

    def make(rank):
        owners = {_layer(38): 0, _layer(39): 1, _layer(40): 0}
        scheduler = KVPPScheduler(
            kvpp_group=SimpleNamespace(rank_in_group=rank, world_size=3, ranks=[16, 17, 18], cpu_group="pp1"),
            layer_owner_ranks=owners,
            kv_caches={name: object() for name in owners},
            num_physical_blocks=10,
            tokens_per_block=4,
            max_active_pages=10,
            transport=SimpleNamespace(
                initialize_transport=lambda *_args: None,
                copy_active_pages_to_staging=copy("push"),
                copy_active_pages_from_staging=copy("pull"),
            ),
        )
        schedulers.append(scheduler)
        return scheduler, calls, recorded_events

    yield make
    for scheduler in schedulers:
        scheduler._prefetch_executor.shutdown(wait=True)


@pytest.mark.parametrize(("rank", "layer_index", "peers"), [(0, 38, [17, 18]), (1, 39, [16, 18])])
def test_owner_sends_after_forward_ready_without_waiting_for_scratch(parallel_scheduler, rank, layer_index, peers):
    scheduler, calls, _events = parallel_scheduler(rank)
    pages, forward_ready, scratch_ready = object(), object(), object()
    layer = _layer(layer_index)

    scheduler.run_layer_prefetch(layer, pages, forward_ready, scratch_ready)

    assert calls == (
        [("recv", peer, kvpp_module._KVPP_READY_TAG) for peer in peers]
        + [("wait", forward_ready), ("push", (layer,), pages), ("complete", "push")]
        + [("send", peer, kvpp_module._KVPP_DONE_TAG) for peer in peers]
    )


@pytest.mark.parametrize(("rank", "layer_index", "owner"), [(1, 38, 16), (0, 39, 17)])
def test_receiver_requires_send_completion_and_destination_ready(parallel_scheduler, rank, layer_index, owner):
    scheduler, calls, _events = parallel_scheduler(rank)
    pages, forward_ready, scratch_ready = object(), object(), object()
    layer = _layer(layer_index)

    scheduler.run_layer_prefetch(layer, pages, forward_ready, scratch_ready)

    assert calls == [
        ("send", owner, kvpp_module._KVPP_READY_TAG),
        ("recv", owner, kvpp_module._KVPP_DONE_TAG),
        ("wait", forward_ready),
        ("wait", scratch_ready),
        ("pull", (layer,), pages),
        ("complete", "pull"),
    ]


def test_background_receiver_publishes_staging_ready_before_waiting_for_done(parallel_scheduler, monkeypatch):
    scheduler, calls, events = parallel_scheduler(1)
    waiting_for_done, send_done = Event(), Event()
    caller_thread = get_ident()
    worker_threads = []

    def wait_for_sender(_token, src, group, tag):
        worker_threads.append(get_ident())
        assert calls == [("send", src, kvpp_module._KVPP_READY_TAG)]
        waiting_for_done.set()
        assert send_done.wait(timeout=5), "Test did not release the sender completion."
        calls.append(("recv", src, tag))

    monkeypatch.setattr(kvpp_module.dist, "recv", wait_for_sender)
    _begin(scheduler)
    try:
        assert waiting_for_done.wait(timeout=5)
        assert len(worker_threads) == 1
        assert worker_threads[0] != caller_thread
        assert not scheduler._prefetch_future.done()
        assert not any(call[0] in ("wait", "pull") for call in calls)
    finally:
        send_done.set()
    scheduler._prefetch_future.result(timeout=5)
    assert ("wait", events[0]) in calls
    assert ("wait", events[1]) in calls
    assert calls[-1] == ("complete", "pull")


def test_forward_event_is_reused_without_background_advancing_layers(parallel_scheduler):
    scheduler, calls, events = parallel_scheduler(0)
    _begin(scheduler)
    forward_ready = scheduler._forward_ready
    scheduler._prefetch_future.result(timeout=5)
    assert scheduler._next_attention_layer_index == 0
    assert len(events) == 2  # One forward event and the first scratch safe point.
    assert [call[0] for call in calls].count("push") == 1

    for layer in (38, 39, 40):
        scheduler.wait_for_layer(_layer(layer))
        assert scheduler._forward_ready is forward_ready
    assert len(events) == 4
    scheduler.complete_forward()
    assert scheduler._forward_ready is None
    _begin(scheduler)
    assert scheduler._forward_ready is not forward_ready
    scheduler._prefetch_future.result(timeout=5)


def test_explicit_scratch_event_is_passed_to_worker_without_an_extra_safe_point(parallel_scheduler):
    scheduler, calls, events = parallel_scheduler(1)
    scheduler._active_pages = pages = object()
    scheduler._forward_ready = forward_ready = object()
    scratch_ready = object()

    scheduler.start_layer_prefetch(_layer(38), scratch_ready=scratch_ready)
    scheduler._prefetch_future.result(timeout=5)

    assert events == []
    assert ("wait", forward_ready) in calls
    assert ("wait", scratch_ready) in calls
    assert ("pull", (_layer(38),), pages) in calls
    with pytest.raises(RuntimeError, match="previous prefetch must be consumed"):
        scheduler.start_layer_prefetch(_layer(39), scratch_ready=scratch_ready)


def test_kvpp_runtime_disabled_preparation_is_noop():
    runtime = KVPPRuntime()
    runtime.prepare_forward((torch.zeros(1, 1, dtype=torch.int32),), [1])
    runtime.complete_forward()
    assert runtime.scheduler is None


def test_kvpp_09_runtime_binds_cache_and_attention_hook(monkeypatch):
    target = _layer(0)
    indexer = _indexer(0)
    hook = SimpleNamespace(layerwise_kv_cache_hook=None)
    target_cache = object()
    indexer_cache = object()
    context = {
        target: SimpleNamespace(impl=hook, kv_cache=target_cache),
        indexer: SimpleNamespace(kv_cache=indexer_cache),
    }
    owners = {target: 0, indexer: 0}
    group = SimpleNamespace(rank_in_group=0, world_size=1)
    initialized_caches = {}

    class FakeTransport:
        def __init__(self, *_args):
            pass

        def initialize_transport(self, caches, _bundles, _max_pages):
            initialized_caches.update(caches)

    monkeypatch.setattr(
        kvpp_module.KVPPConfig,
        "from_vllm_config",
        lambda _config: SimpleNamespace(size=2),
    )
    monkeypatch.setattr(kvpp_module, "map_kvpp_layers_to_owners", lambda *_args: owners)
    monkeypatch.setattr(kvpp_module, "get_kvpp_group", lambda: group)
    monkeypatch.setattr(kvpp_module, "MemFabricMTEKVPPTransport", FakeTransport)

    runtime = KVPPRuntime.create_from_kv_cache(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(max_model_len=4096),
            scheduler_config=SimpleNamespace(max_num_seqs=4),
        ),
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[SimpleNamespace(layer_names=(target, indexer))],
            num_blocks=8,
        ),
        block_tables=SimpleNamespace(
            blocks_per_kv_block=(2,),
            kernel_block_sizes=(128,),
        ),
        static_forward_context=context,
    )

    assert initialized_caches == {target: target_cache, indexer: indexer_cache}
    assert runtime.managed_cache_group_index == 0
    assert runtime.scheduler.layer_cache_bundles == {target: (target, indexer)}
    assert hook.layerwise_kv_cache_hook is runtime.scheduler


def test_kvpp_10_rejects_managed_layers_from_multiple_cache_groups(monkeypatch):
    target = _layer(0)
    indexer = _indexer(0)
    owners = {target: 0, indexer: 0}
    monkeypatch.setattr(
        kvpp_module.KVPPConfig,
        "from_vllm_config",
        lambda _config: SimpleNamespace(size=2),
    )
    monkeypatch.setattr(kvpp_module, "map_kvpp_layers_to_owners", lambda *_args: owners)

    with pytest.raises(ValueError, match="must belong to one cache group"):
        KVPPRuntime.create_from_kv_cache(
            vllm_config=object(),
            kv_cache_config=SimpleNamespace(
                kv_cache_groups=[
                    SimpleNamespace(layer_names=(target,)),
                    SimpleNamespace(layer_names=(indexer,)),
                ]
            ),
            block_tables=SimpleNamespace(blocks_per_kv_block=(1, 1), kernel_block_sizes=(128, 128)),
            static_forward_context={},
        )
