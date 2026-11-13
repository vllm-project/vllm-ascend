# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from vllm.v1.core.sched.output import SchedulerOutput

from vllm_ascend.worker.v2 import draft_tokens


def test_draft_handoff_waits_only_for_selected_batch():
    first_event, later_event = Mock(), Mock()
    stream = Mock()
    npu = SimpleNamespace(
        Stream=Mock(return_value=stream),
        Event=Mock(side_effect=[first_event, later_event]),
        current_stream=Mock(),
        stream=lambda _: nullcontext(),
    )
    first = Mock(shape=(2, 2), **{"numpy.return_value": torch.tensor([[11, 12], [13, 14]]).numpy()})
    later = Mock(shape=(1, 2), **{"numpy.return_value": torch.tensor([[21, 22]]).numpy()})
    with (
        patch.object(torch, "npu", npu, create=True),
        patch.object(draft_tokens, "async_copy_to_np", side_effect=lambda tensor: tensor.numpy()),
    ):
        handler = draft_tokens.AscendDraftTokensHandler(torch.device("cpu"))
        handler.set_draft_tokens(
            SimpleNamespace(req_ids=["a", "b"], has_structured_output_reqs=True),
            first,
        )
        handler.set_draft_tokens(SimpleNamespace(req_ids=["later"], has_structured_output_reqs=True), later)
        handler.set_draft_tokens(
            SimpleNamespace(req_ids=["plain"], has_structured_output_reqs=False), torch.tensor([[31, 32]])
        )

    step = SchedulerOutput.make_empty()
    step.pending_structured_output_tokens = True
    if hasattr(step, "has_structured_output_spec_decode"):
        step.has_structured_output_spec_decode = True
    step.scheduled_spec_decode_tokens = {"b": [-1, -1], "a": [-1, -1], "plain": [-1, -1]}
    handler.begin_step(step)
    result = handler.get_draft_tokens()
    assert result.req_ids == ["b", "a", "plain"]
    assert result.draft_token_ids == [[13, 14], [11, 12], [-1, -1]]
    first_event.synchronize.assert_called_once()
    later_event.synchronize.assert_not_called()

    handler.discard("a")
    handler.begin_step(SchedulerOutput.make_empty())
    result = handler.get_draft_tokens()
    assert result.req_ids == ["b", "later", "plain"]
    assert result.draft_token_ids == [[13, 14], [21, 22], [-1, -1]]
    first_event.synchronize.assert_called_once()
    later_event.synchronize.assert_called_once()
