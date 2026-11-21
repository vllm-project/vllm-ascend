# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace

from vllm_ascend.ops.triton.spec_decode import suffix_observe as ops


def test_observer_launcher_consumes_previous_drafts_for_active_requests(monkeypatch, recording_kernel):
    calls = []
    monkeypatch.setattr(ops, "_suffix_observe_kernel", recording_kernel("observe", calls))
    monkeypatch.setattr(ops, "_suffix_clear_observed_kernel", recording_kernel("clear", calls))
    states = [
        SimpleNamespace(shape=(7,)),
        object(),
        object(),
        SimpleNamespace(shape=(18, 3, 2)),
        SimpleNamespace(shape=(128,)),
        SimpleNamespace(shape=(128,)),
    ]
    ops.launch_suffix_observe(*states, K=3, K_PO2=4, num_warps=1)
    assert [call[0] for call in calls] == ["observe", "clear"]
    _, grid, args, options = calls[0]
    assert grid == (18,)
    assert args == (*states, 7)
    assert options == {"K": 3, "K_PO2": 4, "REQS_PO2": 8, "num_warps": 4}
    assert calls[1][1] == (1,)
    assert calls[1][2] == (states[0], states[1], states[4], 7)
