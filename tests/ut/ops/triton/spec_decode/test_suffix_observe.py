# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call

from vllm_ascend.ops.triton.spec_decode import suffix_observe as ops


def test_observer_launcher_consumes_previous_drafts_for_active_requests(monkeypatch):
    observe, clear = MagicMock(), MagicMock()
    monkeypatch.setattr(ops, "_suffix_observe_kernel", observe)
    monkeypatch.setattr(ops, "_suffix_clear_observed_kernel", clear)
    launches = Mock()
    launches.attach_mock(observe.__getitem__.return_value, "observe")
    launches.attach_mock(clear.__getitem__.return_value, "clear")
    states = [
        SimpleNamespace(shape=(7,)),
        object(),
        object(),
        SimpleNamespace(shape=(18, 3, 2)),
        SimpleNamespace(shape=(128,)),
        SimpleNamespace(shape=(128,)),
    ]

    ops.launch_suffix_observe(*states, K=3, K_PO2=4, num_warps=1)

    observe.__getitem__.assert_called_once_with((18,))
    clear.__getitem__.assert_called_once_with((1,))
    assert launches.mock_calls == [
        call.observe(*states, 7, K=3, K_PO2=4, REQS_PO2=8, num_warps=4),
        call.clear(states[0], states[1], states[4], 7, BLOCK=256, num_warps=4),
    ]
