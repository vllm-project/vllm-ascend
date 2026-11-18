# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, call
from weakref import ref

import torch

from vllm_ascend.models.kimi_k3 import kimi_k3


def test_dsa_gate_reuses_stream_and_waits_at_each_consumer(monkeypatch):
    producer, project_stream, finish_stream, comm = (Mock() for _ in range(4))
    create_stream = Mock(return_value=comm)
    current = producer
    trace = []
    records = []
    group = SimpleNamespace(world_size=2, device_group=object())

    @contextmanager
    def switch(stream):
        nonlocal current
        previous, current = current, stream
        try:
            yield
        finally:
            current = previous

    def submit(name, received, send, process_group, async_op):
        assert current is comm and process_group is group.device_group and async_op
        trace.append(name)
        # The work double must not keep a temporary send alive on behalf of the caller.
        send_ref, received_ref = ref(send), ref(received)
        assert (id(send), comm) in records

        def wait():
            trace.append((name, current))
            assert send_ref() is not None
            received_ref().fill_(1)

        return SimpleNamespace(wait=wait)

    def gather(received, send, group, async_op):
        return submit("AG", received, send, group, async_op)

    def exchange(received, send, group, async_op):
        return submit("A2A", received, send, group, async_op)

    def project(gathered):
        assert current is project_stream and (id(gathered), current) in records
        trace.append("project")
        return (gathered + 1,)

    monkeypatch.setattr(torch.npu, "current_stream", lambda: current)
    monkeypatch.setattr(torch.npu, "Stream", create_stream)
    monkeypatch.setattr(kimi_k3, "npu_stream_switch", switch)
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda tensor, stream: records.append((id(tensor), stream)))
    monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", gather)
    monkeypatch.setattr(torch.distributed, "all_to_all_single", exchange)
    monkeypatch.setattr(kimi_k3.KDAInputProjectionTPLayout, "prepare_input", staticmethod(lambda x: (x, x.shape[0])))
    begin_gate = kimi_k3._make_dsa_gate_projection(project, group, 2, 2)
    create_stream.assert_not_called()
    for _ in range(2):
        current = producer
        project_gate = begin_gate(torch.zeros(2, 4)[:, ::2])
        assert trace == ["AG"]
        current = project_stream
        finish_gate = project_gate()
        assert trace == ["AG", ("AG", project_stream), "project", "A2A"]
        current = finish_stream
        gate = finish_gate()
        assert trace[-1] == ("A2A", finish_stream)
        assert [stream for _, stream in records] == [
            comm,
            project_stream,
            project_stream,
            comm,
            finish_stream,
            finish_stream,
        ]
        torch.testing.assert_close(gate, torch.ones(2, 2, 2))
        trace.clear()
        records.clear()
    create_stream.assert_called_once_with()
    assert comm.wait_stream.call_args_list == [call(producer), call(project_stream)] * 2
