# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of the production handler, including real multi-group Gloo jobs."""

from collections import deque
from concurrent.futures import Future
from contextlib import nullcontext
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm_ascend.worker.v2 import pp_feedback as feedback


@pytest.mark.parametrize(
    "device,depth,async_scheduling,mode,expected",
    [
        ("npu", 2, True, "AIV", True),
        ("cuda", 2, True, "AIV", False),
        ("cpu", 2, True, "AIV", False),
        ("npu", 1, True, "AIV", False),
        ("npu", 3, True, "AIV", False),
        ("npu", 2, False, "AIV", False),
        ("npu", 2, True, "AI_CPU", False),
        ("npu", 2, True, None, False),
    ],
)
def test_selection(monkeypatch, device, depth, async_scheduling, mode, expected):
    monkeypatch.setattr(feedback, "envs", SimpleNamespace(HCCL_OP_EXPANSION_MODE=mode))
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=depth),
        scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
    )
    assert feedback.use_host_pp_feedback(config, SimpleNamespace(type=device)) is expected


@pytest.fixture
def handler(monkeypatch):
    old_group, new_group = object(), object()
    monkeypatch.setattr(feedback, "get_pp_group", lambda: SimpleNamespace(group_ranks=[[0, 1]], rank=0, world_size=2))
    monkeypatch.setattr(dist, "new_group", Mock(return_value=new_group))
    monkeypatch.setattr(dist, "destroy_process_group", Mock())
    parent = SimpleNamespace(
        queue=deque([None, None]),
        sampled_tokens_broadcast=False,
        broadcast_group=old_group,
        is_last_rank=False,
        last_rank=1,
        device=torch.device("cpu"),
        max_sample_len=4,
        num_speculative_steps=3,
        req_idx_gen_np=np.zeros(8, dtype=np.int32),
        main_stream=Mock(),
        broadcast_stream=Mock(),
        upstream_aux_keys=("aux_0",),
    )
    result = feedback.AscendPPHandler(parent)
    assert result.upstream_aux_keys == parent.upstream_aux_keys
    dist.destroy_process_group.assert_called_once_with(old_group)
    dist.new_group.assert_called_once_with(
        [0, 1], backend="gloo", timeout=timedelta(seconds=120), group_desc="pp_host_feedback"
    )
    yield result
    result.shutdown()
    result.shutdown()
    assert dist.destroy_process_group.call_count == 2


def test_snapshot_stream_order_and_device_guard(handler, monkeypatch):
    events = []
    handler.is_last_rank = True
    handler.broadcast_stream.wait_stream.side_effect = lambda stream: events.append("wait main")
    handler.broadcast_stream.record_event.side_effect = lambda: events.append("record event") or SimpleNamespace(
        synchronize=lambda: events.append("event ready")
    )
    handler.main_stream.wait_event.side_effect = lambda event: events.append("wait local copy")
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(stream=lambda stream: nullcontext(), device=lambda d: nullcontext()),
        raising=False,
    )
    empty = torch.empty
    monkeypatch.setattr(
        torch, "empty", lambda *args, **kwargs: empty(*args, **{k: v for k, v in kwargs.items() if k != "pin_memory"})
    )
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda tensor, stream: events.append("record source"))
    monkeypatch.setattr(dist, "broadcast", lambda tensor, **kwargs: events.append("broadcast"))
    source = torch.tensor([[7, 8]])
    handler._enqueue_host_feedback((source,))
    handler._host_feedback.pending[-1].result(timeout=5)
    assert events.index("wait main") < events.index("record source") < events.index("record event")
    assert events.index("record event") < events.index("wait local copy")
    assert events.index("wait local copy") < events.index("event ready") < events.index("broadcast")


@pytest.mark.parametrize("freed", [[False, False, False], [True, False, True], [True, True, True]])
@pytest.mark.parametrize("drafts", [False, True])
def test_fifo_generation_filter(handler, monkeypatch, freed, drafts):
    monkeypatch.setattr(feedback, "async_tensor_h2d", lambda values, device: torch.as_tensor(values))
    idx = np.array([1, 3, 5], dtype=np.int32)
    payload = [torch.tensor([[11], [22], [33]]), torch.tensor([[1, 2, 3], [0, 1, 2]], dtype=torch.int32)]
    if drafts:
        payload.append(torch.tensor([[10, 11, 12], [20, 21, 22], [30, 31, 32]]))
    future = Future()
    future.set_result(payload)
    slot = feedback.PendingHostRecv(
        future, torch.as_tensor(idx), idx.copy(), np.array([True, True, False]), np.zeros(3)
    )
    handler.queue[-1] = slot
    for request, was_freed in zip(idx, freed):
        if was_freed:
            handler.on_req_idx_freed(request)
    assert handler.get_prev_sampled_outputs() is None
    result = handler.get_prev_sampled_outputs()
    if all(freed[:2]):
        assert result is None
    else:
        assert result["idx_mapping"].tolist() == [-1 if dead or i == 2 else int(idx[i]) for i, dead in enumerate(freed)]
        assert torch.equal(result["sampled_tokens"], payload[0])
        assert result["num_sampled"].tolist() == [1, 2, 3]
        assert result["num_rejected"].tolist() == [0, 1, 2]
        assert (result["draft_tokens"] is not None) is drafts
    assert handler.get_prev_sampled_outputs() is None


def test_receive_copies_metadata_and_skips_prefill(handler, monkeypatch):
    mask = np.array([True, False])
    monkeypatch.setattr(feedback, "compute_need_sampled_mask", lambda batch: mask)
    empty = torch.empty
    monkeypatch.setattr(
        torch, "empty", lambda *args, **kwargs: empty(*args, **{k: v for k, v in kwargs.items() if k != "pin_memory"})
    )
    monkeypatch.setattr(dist, "broadcast", lambda tensor, **kwargs: None)
    batch = SimpleNamespace(
        num_reqs=2, num_draft_tokens=3, idx_mapping=torch.tensor([1, 2]), idx_mapping_np=np.array([1, 2])
    )
    assert not handler.receive(batch, recv_drafts=True)
    slot = handler.queue[-1]
    batch.idx_mapping_np[:] = 7
    mask[:] = False
    assert slot.idx_mapping_np.tolist() == [1, 2]
    assert slot.need_sampled_mask.tolist() == [True, False]
    assert [tuple(t.shape) for t in slot.future.result(timeout=5)] == [(2, 4), (2, 2), (2, 3)]
    handler.queue[-1] = None
    monkeypatch.setattr(feedback, "compute_need_sampled_mask", lambda batch: None)
    assert not handler.receive(batch, recv_drafts=False)
    assert handler.queue[-1] is None


def test_skipped_broadcast_never_sends_drafts(handler, monkeypatch):
    handler.is_last_rank = True
    monkeypatch.setattr(feedback, "compute_need_sampled_mask", lambda batch: None)
    enqueue = Mock()
    monkeypatch.setattr(handler, "_enqueue_host_feedback", enqueue)
    handler.broadcast(torch.zeros((1, 1), dtype=torch.int64), torch.ones(1), torch.zeros(1), None)
    handler.broadcast_drafts(torch.zeros((1, 3), dtype=torch.int64))
    enqueue.assert_not_called()


def test_discarded_request_failure_is_observed(handler, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("peer exited")

    monkeypatch.setattr(dist, "broadcast", fail)
    future = handler._host_feedback.submit([torch.empty(1)])
    with pytest.raises(RuntimeError, match="peer exited"):
        future.result(timeout=5)
    with pytest.raises(RuntimeError, match="peer exited"):
        handler.get_prev_sampled_outputs()
    with pytest.raises(RuntimeError, match="previously failed"):
        handler.shutdown()


def test_backpressure_and_closed_transport(monkeypatch):
    transport = feedback.HostFeedbackTransport(None, 1, 1, torch.device("cpu"))
    pending = Mock()
    pending.done.return_value = False
    pending.result.side_effect = TimeoutError("bounded wait")
    transport.pending.append(pending)
    with pytest.raises(TimeoutError, match="bounded wait"):
        transport.submit([])
    pending.result.assert_called_once_with(timeout=feedback.HOST_FEEDBACK_TIMEOUT_SECONDS)
    pending.result.side_effect = None
    pending.done.return_value = True
    transport.close()
    with pytest.raises(RuntimeError, match="closed"):
        transport.submit([])
    transport.close()


def _gloo_worker(rank, store, fail_peer):
    world_size = 2 if fail_peer else 4
    dist.init_process_group(
        "gloo", init_method=f"file://{store}", rank=rank, world_size=world_size, timeout=timedelta(seconds=5)
    )
    groups = [[0, 1]] if fail_peer else [[0, 2], [1, 3]]
    own_group = None
    for ranks in groups:
        candidate = dist.new_group(ranks, backend="gloo", timeout=timedelta(seconds=2))
        if rank in ranks:
            own_group = candidate
            source = ranks[-1]
    transport = feedback.HostFeedbackTransport(own_group, source, 4, torch.device("cpu"))
    try:
        if fail_peer:
            if rank == source:
                dist.destroy_process_group(own_group)
            else:
                future = transport.submit([torch.empty(1)])
                with pytest.raises(RuntimeError):
                    future.result(timeout=10)
                with pytest.raises(RuntimeError):
                    transport.close()
            return
        received = []
        for step in range(8):
            width = 4 if step % 2 else 1
            if rank == source:
                transport.submit(
                    [
                        torch.full((3, width), source * 100 + step, dtype=torch.int64),
                        torch.full((2, 3), step, dtype=torch.int32),
                    ]
                )
                if step % 2:
                    transport.submit([torch.full((3, 3), step + 20, dtype=torch.int64)])
            else:
                payload = [torch.empty((3, width), dtype=torch.int64), torch.empty((2, 3), dtype=torch.int32)]
                if step % 2:
                    payload.append(torch.empty((3, 3), dtype=torch.int64))
                received.append(transport.submit(payload))
        for step, future in enumerate(received):
            payload = future.result(timeout=10)
            assert torch.all(payload[0] == source * 100 + step)
            assert torch.all(payload[1] == step)
            if step % 2:
                assert torch.all(payload[2] == step + 20)
        transport.close()
        dist.destroy_process_group(own_group)
    finally:
        transport.close()
        dist.destroy_process_group()


@pytest.mark.parametrize("fail_peer", [False, True])
def test_real_gloo_feedback(tmp_path, monkeypatch, fail_peer):
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    mp.spawn(_gloo_worker, args=(str(tmp_path / "store"), fail_peer), nprocs=2 if fail_peer else 4, join=True)
