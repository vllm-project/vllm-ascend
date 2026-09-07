# SPDX-License-Identifier: Apache-2.0
"""DP metadata must not synchronize draft graph replay before parameter updates."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.worker.v2.spec_decode.dflash import aclgraph


@pytest.mark.parametrize("dp_size, dp_rank", [(1, 0), (2, 0), (2, 1)])
def test_replay_updates_graph_with_host_dp_counts(monkeypatch, dp_size, dp_rank):
    manager = object.__new__(aclgraph.DFlashAclGraphManager)
    manager.device = torch.device("meta")
    manager.vllm_config = MagicMock()
    manager.update_stream = MagicMock()
    manager.speculator = MagicMock(dp_size=dp_size)
    manager.speculator.attn_backends = {"draft": "backend"}
    desc = SimpleNamespace(num_tokens=16, num_reqs=2, cg_mode=MagicMock())
    events = []
    output = object()

    def replay(self, desc):
        events.append("replay")
        return output

    @contextmanager
    def forward_context(*args, **kwargs):
        counts = kwargs["num_tokens_across_dp"]
        assert counts.device.type == "cpu", "DP validation must not wait on graph replay"
        assert counts.tolist() == [16] * dp_size
        assert counts[dp_rank] == kwargs["num_tokens"]
        events.append("host_metadata")
        yield

    monkeypatch.setattr(aclgraph.DFlashCudaGraphManager, "run_fullgraph", replay)
    monkeypatch.setattr(torch.npu, "current_stream", MagicMock())
    monkeypatch.setattr(aclgraph, "set_forward_context", forward_context)
    monkeypatch.setattr(aclgraph, "get_forward_context", MagicMock())
    monkeypatch.setattr(aclgraph, "_EXTRA_CTX", SimpleNamespace())
    monkeypatch.setattr(aclgraph, "update_full_graph_params", lambda *a, **kw: events.append("update"))

    assert manager.run_fullgraph(desc) is output
    assert events == ["replay", "host_metadata", "update"]
