# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.worker.v2 import aclgraph_utils


@pytest.mark.parametrize("release", [True, False], ids=["v028", "main"])
@pytest.mark.parametrize("use_pcp", [True, False], ids=["pcp", "no_pcp"])
def test_graph_capture_reads_runtime_input_buffers(monkeypatch, release, use_pcp):
    """Replay must observe PCP's current inputs and padding, not global rows."""

    def make_buffers():
        return SimpleNamespace(
            input_ids=torch.full((4,), -1, dtype=torch.int32),
            positions=torch.full((4,), -1, dtype=torch.int64),
            seq_lens=torch.full((4,), -1, dtype=torch.int32),
            is_padding=torch.ones(4, dtype=torch.bool),
        )

    global_buffers = make_buffers()
    local_buffers = make_buffers()
    pcp_manager = SimpleNamespace(_input_buffers=local_buffers) if use_pcp else None
    graph_manager = aclgraph_utils.ModelAclGraphManager.__new__(aclgraph_utils.ModelAclGraphManager)
    graph_manager.model_runner = SimpleNamespace(pcp_manager=pcp_manager)
    captured = {}

    def record_capture(self, model, model_state, input_buffers, *args, **kwargs):
        captured["inputs"] = vars(input_buffers).copy()
        captured["kwargs"] = kwargs

    monkeypatch.setattr(aclgraph_utils.ModelCudaGraphManager, "capture", record_capture)
    monkeypatch.setattr(aclgraph_utils, "vllm_version_is", lambda version: release)
    monkeypatch.setattr(aclgraph_utils, "communicator_switch", nullcontext)
    # Restore the capture helper after the production wrapper replaces it.
    monkeypatch.setattr(aclgraph_utils.cudagraph_utils, "prepare_inputs_to_capture", lambda *args, **kwargs: None)

    # Before #53515 the upstream runner passes global inputs even with PCP.
    caller_buffers = global_buffers if release or not use_pcp else local_buffers
    graph_manager.capture(torch.nn.Identity(), object(), caller_buffers, None, object(), [], object())

    runtime_buffers = local_buffers if use_pcp else global_buffers
    runtime_buffers.input_ids.copy_(torch.tensor([41, 42, 43, 0], dtype=torch.int32))
    runtime_buffers.positions.copy_(torch.tensor([11, 12, 13, 0]))
    runtime_buffers.seq_lens.copy_(torch.tensor([12, 13, 14, 0], dtype=torch.int32))
    runtime_buffers.is_padding.copy_(torch.tensor([False, False, False, True]))
    for name, runtime_tensor in vars(runtime_buffers).items():
        assert captured["inputs"][name].data_ptr() == runtime_tensor.data_ptr(), name
        torch.testing.assert_close(captured["inputs"][name], runtime_tensor)
    if release:
        assert "pcp_manager" not in captured["kwargs"]
    else:
        assert captured["kwargs"]["pcp_manager"] is pcp_manager
