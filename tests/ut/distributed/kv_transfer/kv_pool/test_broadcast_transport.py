# SPDX-License-Identifier: Apache-2.0
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tests.ut.kvpp_utils import ManualExecutor
from vllm_ascend.distributed.kv_transfer.kv_pool import broadcast_transport


@pytest.mark.parametrize("local_rank", [0, 1], ids=["receiver", "owner"])
def test_full_layer_broadcast_completes_before_future(monkeypatch, local_rank):
    events: list[object] = []
    group = SimpleNamespace(ranks=[4, 9], rank_in_group=local_rank, device_group=object())
    backing = torch.zeros(40, dtype=torch.int8)
    payload = backing[2:38]
    ready = object()
    transfer = Mock()
    transfer.wait_event.side_effect = lambda event: events.append(("wait_ready", event))
    work = Mock()
    work.wait.side_effect = lambda: events.append("work_wait")
    done = Mock()
    done.record.side_effect = lambda stream: events.append(("record_done", stream))
    executor = ManualExecutor()

    def broadcast(tensor, *, src, group, async_op):
        assert tensor is payload
        assert tensor.numel() == 36
        assert src == 9
        assert group is device_group
        assert async_op is True
        events.append("broadcast")
        return work

    def synchronize():
        assert not future.done()
        events.append("synchronize")
        payload.fill_(17)

    device_group = group.device_group
    done.synchronize.side_effect = synchronize

    def use_stream(stream):
        assert stream is transfer
        return nullcontext(stream)

    monkeypatch.setattr(broadcast_transport.torch.npu, "stream", use_stream)
    monkeypatch.setattr(broadcast_transport.torch.npu, "Event", lambda: done)
    monkeypatch.setattr(broadcast_transport.dist, "broadcast", broadcast)
    transport = broadcast_transport.BroadcastKVPPTransport(group, {"layer": 1}, {"layer": payload})
    future = executor.submit(transport.prefetch, "layer", ready, transfer)
    executor.run_next()
    assert future.done()
    assert future.result() is None
    assert events == [("wait_ready", ready), "broadcast", "work_wait", ("record_done", transfer), "synchronize"]
    assert torch.all(payload == 17)
    assert torch.count_nonzero(backing[:2]) == 0
    assert torch.count_nonzero(backing[38:]) == 0
