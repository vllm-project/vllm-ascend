# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace

from vllm_ascend.ops.triton.spec_decode import suffix as ops


class RecordingKernel:
    def __init__(self, name, calls):
        self.name = name
        self.calls = calls

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((self.name, grid, args, kwargs))

        return launch


def test_launch_adapter_forwards_grid_and_arguments():
    calls = []

    def launch(grid, *args, **kwargs):
        calls.append((grid, args, kwargs))

    adapter = ops.SuffixKernelLaunchAdapter(launch)
    token_ids = object()
    adapter[(7, 2)](token_ids, BLOCK=128)
    assert calls == [((7, 2), (token_ids,), {"BLOCK": 128})]


def test_observer_adapter_translates_grid_and_consumes_previous_drafts(monkeypatch):
    calls = []
    monkeypatch.setattr(ops, "_suffix_observe_kernel", RecordingKernel("observe", calls))
    monkeypatch.setattr(ops, "_suffix_clear_observed_kernel", RecordingKernel("clear", calls))
    adapter = ops.SuffixKernelLaunchAdapter(ops.launch_suffix_observe)
    states = [object(), object(), object(), SimpleNamespace(shape=(18, 3, 2)), object(), object()]
    adapter[(7,)](*states, K=3, K_PO2=4, num_warps=1)
    assert [call[0] for call in calls] == ["observe", "clear"]
    _, grid, args, options = calls[0]
    assert grid == (18,)
    assert args == (*states, 7)
    assert options == {"K": 3, "K_PO2": 4, "REQS_PO2": 8, "num_warps": 4}
    assert calls[1][2] == (states[0], states[1], states[4], 7)


def test_empty_observer_does_not_launch(monkeypatch):
    calls = []
    monkeypatch.setattr(ops, "_suffix_observe_kernel", RecordingKernel("observe", calls))
    monkeypatch.setattr(ops, "_suffix_clear_observed_kernel", RecordingKernel("clear", calls))
    ops.launch_suffix_observe((0,), None, None, None, None, None, None, K=3, K_PO2=4)
    assert calls == []


def test_ingestion_launcher_uses_tensor_stride_and_document_count(monkeypatch):
    calls = []
    monkeypatch.setattr(ops, "_suffix_ingest_kernel", RecordingKernel("ingest", calls))
    adapter = ops.SuffixKernelLaunchAdapter(ops.launch_suffix_ingest)
    token_ids = SimpleNamespace(shape=(4, 2048), stride=lambda axis: 2056)
    args = (token_ids, 2048, object(), 2, *[object() for _ in range(8)])
    adapter[(2, 1)](*args, BLOCK=1024, num_warps=4)
    assert calls[0][1] == (2, 2)
    assert calls[0][2][1] == 2056
