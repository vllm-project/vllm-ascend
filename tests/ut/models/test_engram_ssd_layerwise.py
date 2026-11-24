# SPDX-License-Identifier: Apache-2.0
"""Layer readiness and shared cache lease lifetime on real Gloo ranks."""
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from datetime import timedelta
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import vllm_ascend.ops  # noqa: F401
from vllm_ascend.models.deepseek_v41.engram.ssd.cache import SSDRowCache
from vllm_ascend.models.deepseek_v41.engram.ssd.distributed import (
    EngramSSDProtocol,
    LeaderCacheHooks,
    SSDProtocolError,
)


def _rank(rank, init_file, plane, first_seen, inline_rows):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank,
                            world_size=3, timeout=timedelta(seconds=30))
    group = SimpleNamespace(cpu_group=dist.group.WORLD, world_size=3, rank_in_group=rank)
    cache = SSDRowCache(64, row_limit=1000) if rank == 0 else None
    fail = False

    class Hooks(LeaderCacheHooks):
        def fill(self, rows, slots):
            if rows[0] >= 100:
                assert np.all(rows >= 100)
                assert first_seen.wait(10), "layer 1 was held until layer 14 completed"
                if fail:
                    raise ValueError("injected layer 14 read failure")
            else:
                assert np.all(rows < 100)
            plane.numpy()[slots] = rows

        def abort(self, lease):
            assert retired.done()
            assert retired.result()
            super().abort(lease)

    hooks = Hooks(cache, None, None, None) if rank == 0 else None
    protocol = EngramSSDProtocol(group, epoch_fn=lambda: 1, leader_hooks=hooks,
                                 timeout_s=30, layered_inline_rows=inline_rows)
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            for index in range(3):
                fail = index == 1
                rows = np.array(([0, 1, 2][rank] + index * 10,
                                 [0, 1, 2][rank] + 100 + index * 10)
                                if rank != 2 else [], dtype=np.uint64)
                if rank == 1:
                    # This rank exceeds the inline limit when inline_rows is 1.
                    rows = np.insert(rows, 0, rows[0])
                first_count = np.count_nonzero(rows < 100)
                first, second = Future(), Future()
                retired = Future()
                done = executor.submit(protocol.step_layered, rows, first_count, 100,
                                       (first, second), retired, num_tokens=first_count)
                try:
                    first_slots = first.result(timeout=20)
                    np.testing.assert_array_equal(plane.numpy()[first_slots], rows[:first_count])
                    if rank == 1:
                        first_seen.set()
                    if fail:
                        with pytest.raises(SSDProtocolError, match="layer stage 2"):
                            second.result(timeout=20)
                    else:
                        np.testing.assert_array_equal(
                            plane.numpy()[second.result(timeout=20)], rows[first_count:])
                    assert not done.done()
                finally:
                    retired.set_result(True)
                if fail:
                    with pytest.raises(SSDProtocolError, match="layer stage 2"):
                        done.result(timeout=20)
                else:
                    done.result(timeout=20)
                if rank == 0:
                    first_seen.clear()
                    assert protocol._leader_lease is None
                dist.barrier()
        ready = (Future(), Future())
        retired = Future()
        rows = np.empty(0, np.uint64)
        with pytest.raises(SSDProtocolError, match="failed to stage input"):
            protocol.step_layered(rows, -1 if rank == 1 else 0, 100, ready, retired)
        for layer in ready:
            with pytest.raises(SSDProtocolError, match="failed to stage input"):
                layer.result()
        assert protocol.round == 3
        assert protocol.stats.rounds_failed == 2
        assert protocol._leader_lease is None
    finally:
        if cache is not None:
            cache.close()
        dist.destroy_process_group()


@pytest.mark.parametrize("inline_rows", [0, 1, 2], ids=["variable", "overflow", "packed"])
def test_layer_one_ready_before_layer_fourteen_and_failed_round_can_recover(tmp_path, inline_rows):
    plane = torch.zeros(64, dtype=torch.int64).share_memory_()
    first_seen = mp.get_context("spawn").Event()
    mp.spawn(_rank, args=(str(tmp_path / "init"), plane, first_seen, inline_rows), nprocs=3)


@pytest.mark.parametrize("graph_mode", ["eager", "full", "full_error"])
def test_runtime_submits_both_gathers_before_consumption(monkeypatch, graph_mode):
    from vllm_ascend.models.deepseek_v41.engram.ssd import runtime as module

    d2h_entered, d2h_ready, second_submitted = Event(), Event(), Event()
    waits, gathers, synced, published = [], [], [], []
    fake_stream = SimpleNamespace(synchronize=lambda: synced.append(True), wait_event=waits.append)

    class DeviceEvent:
        def record(self, stream=None):
            if len(gathers) == 2 and graph_mode == "eager":
                second_submitted.set()

        def synchronize(self):
            d2h_entered.set()
            assert d2h_ready.wait(5), "prepare waited for D2H on the caller"

    def protocol_step(rows, first_count, split, ready, retired, *, num_tokens, submit_layer):
        np.testing.assert_array_equal(rows, [7, 108])
        assert (first_count, split, num_tokens) == (1, 100, 2)
        ready[0].set_result(submit_layer(0, np.array([3], np.int32)))
        if graph_mode == "full_error":
            ready[1].set_exception(ValueError("read failed"))
            second_submitted.set()
        else:
            ready[1].set_result(submit_layer(1, np.array([4], np.int32)))
        assert retired.result(timeout=5)
        assert synced == [True]

    def gather(codes, scales, slots, *, output, **kwargs):
        gathers.append(slots.tolist())
        output.copy_(slots.view(-1, 1).expand(-1, 256))

    rt = module.EngramSSDRuntime.__new__(module.EngramSSDRuntime)
    rt._closed = False
    rt._lease = None
    rt.layerwise_overlap = True
    rt.device = torch.device("cpu")
    rt.max_tokens, rt.num_layers, rt.total_hash_cols = 2, 2, 1
    rt.col_start, rt.col_count, rt.capacity = 0, 1, 8
    rt.layer_ids, rt.layer_bases, rt.layer_rows = [1, 14], [0, 100], [100, 100]
    rt._d2h = torch.empty((2, 2, 1), dtype=torch.int32)
    rt._output = torch.full((2, 2, 1, 256), -99.0)
    rt._slot_host = torch.empty((2, 2, 1), dtype=torch.int32)
    rt._slot_dev = torch.empty_like(rt._slot_host)
    rt._layered_lookup = rt._layered_executor = None
    rt._step = 0
    rt.planes = SimpleNamespace(codes=None, scales=None)
    rt.protocol = SimpleNamespace(epoch_fn=lambda: 1, step_layered=protocol_step)
    monkeypatch.setattr(torch.npu, "Event", DeviceEvent)
    monkeypatch.setattr(torch.npu, "current_stream", lambda: fake_stream)
    monkeypatch.setattr(torch.npu, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.npu, "stream", lambda _: nullcontext())
    monkeypatch.setattr(module, "gather_dequantize_host_uva", gather)

    class GraphEvent:
        def __init__(self, layer):
            self.layer = layer

        def record(self, stream):
            assert stream is fake_stream
            published.append((self.layer, len(gathers)))
            if self.layer == 14:
                second_submitted.set()

    options = {}
    if graph_mode != "eager":
        options["graph_ready_events"] = {layer: GraphEvent(layer) for layer in (1, 14)}
    prepared = rt.prepare(torch.tensor([[[7], [-1]], [[-1], [8]]], dtype=torch.int32), None, **options)
    try:
        assert d2h_entered.wait(5)
        assert not gathers
        d2h_ready.set()
        assert second_submitted.wait(5)
        assert not waits
        if graph_mode == "eager":
            rt.consume_layer(1)
            rt.consume_layer(14)
            assert len(waits) == 2
            assert not published
        elif graph_mode == "full":
            assert published == [(1, 1), (14, 2)]
        else:
            assert published == [(1, 1), (14, 1)]
        assert prepared.layer_outputs[1][:, 0, 0].tolist() == [3.0, -1.0]
        expected = [0.0, 0.0] if graph_mode == "full_error" else [-1.0, 4.0]
        assert prepared.layer_outputs[14][:, 0, 0].tolist() == expected
    finally:
        d2h_ready.set()
        if graph_mode == "full_error":
            with pytest.raises(ValueError, match="read failed"):
                rt.finish_execution()
        else:
            rt.finish_execution()
        rt._layered_executor.shutdown()


def test_prefill_lookahead_runs_next_round_before_current_forward_returns(tmp_path):
    from vllm_ascend.models.deepseek_v41.engram.ssd.cache import SSDForcedReadCache

    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'lookahead'}",
                            rank=0, world_size=1, timeout=timedelta(seconds=15))
    group = SimpleNamespace(cpu_group=dist.group.WORLD, world_size=1, rank_in_group=0)
    cache = SSDForcedReadCache(2, row_limit=200, mode="ssd-no-dedup")
    reads = []
    current_forward_done = Event()

    class Hooks(LeaderCacheHooks):
        def fill(self, rows, slots):
            assert not current_forward_done.is_set()
            reads.extend(rows.tolist())

    protocol = EngramSSDProtocol(group, epoch_fn=lambda: 1,
                                 leader_hooks=Hooks(cache, None, None, None, base1=100))
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            def queue(rows):
                ready = (Future(), Future())
                done = executor.submit(
                    protocol.step_layered, np.array(rows, np.uint64), 1, 100, ready, None,
                    synchronize_gathers=lambda: all(event.done() for event in ready),
                )
                return done

            current = queue([3, 103])
            lookahead = queue([4, 104])
            current.result(timeout=10)
            lookahead.result(timeout=10)
            assert reads == [3, 103, 4, 104]
            assert not cache.is_active
            current_forward_done.set()
    finally:
        dist.destroy_process_group()
