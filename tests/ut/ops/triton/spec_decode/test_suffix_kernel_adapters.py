# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace

import pytest

from vllm_ascend.ops.triton.spec_decode import suffix as ops
from vllm_ascend.ops.triton.spec_decode.suffix import (
    ascend_suffix_ingestion,
    ascend_suffix_proposal,
)


class RecordingKernel:
    def __init__(self, name, calls):
        self.name = name
        self.calls = calls

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((self.name, grid, args, kwargs))

        return launch


def _upstream_bindings():
    return SimpleNamespace(
        _suffix_local_scan_kernel=object(),
        _suffix_finalize_kernel=object(),
        _suffix_observe_kernel=object(),
        _suffix_ingest_kernel=object(),
    )


@pytest.mark.parametrize("fail", [False, True])
def test_proposal_bindings_restore_on_return_and_exception(fail):
    module = _upstream_bindings()
    originals = vars(module).copy()

    def invoke_parent():
        with ascend_suffix_proposal(module):
            assert module._suffix_ingest_kernel is originals["_suffix_ingest_kernel"]
            for name in ("_suffix_local_scan_kernel", "_suffix_finalize_kernel", "_suffix_observe_kernel"):
                assert module.__dict__[name] is not originals[name]
            if fail:
                raise RuntimeError("parent launch failed")

    if fail:
        with pytest.raises(RuntimeError, match="parent launch failed"):
            invoke_parent()
    else:
        invoke_parent()
    for name, original in originals.items():
        assert getattr(module, name) is original


def test_nested_proposal_and_ingestion_restore_outer_bindings():
    module = _upstream_bindings()
    originals = vars(module).copy()
    with ascend_suffix_proposal(module):
        outer_scan = module._suffix_local_scan_kernel
        with ascend_suffix_proposal(module), ascend_suffix_ingestion(module):
            assert module._suffix_local_scan_kernel is not outer_scan
            assert module._suffix_ingest_kernel is not originals["_suffix_ingest_kernel"]
        assert module._suffix_local_scan_kernel is outer_scan
        assert module._suffix_ingest_kernel is originals["_suffix_ingest_kernel"]
    for name, original in originals.items():
        assert getattr(module, name) is original


def test_ingestion_restores_binding_when_parent_raises():
    module = _upstream_bindings()
    original = module._suffix_ingest_kernel
    with pytest.raises(RuntimeError, match="ingestion failed"), ascend_suffix_ingestion(module):
        assert module._suffix_ingest_kernel is not original
        raise RuntimeError("ingestion failed")
    assert module._suffix_ingest_kernel is original


def test_observer_adapter_translates_grid_and_consumes_previous_drafts(monkeypatch):
    calls = []
    monkeypatch.setattr(ops, "_suffix_observe_kernel", RecordingKernel("observe", calls))
    monkeypatch.setattr(ops, "_suffix_clear_observed_kernel", RecordingKernel("clear", calls))
    module = _upstream_bindings()
    states = [object(), object(), object(), SimpleNamespace(shape=(18, 3, 2)), object(), object()]
    with ascend_suffix_proposal(module):
        module._suffix_observe_kernel[(7,)](*states, K=3, K_PO2=4, num_warps=1)
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
    module = _upstream_bindings()
    token_ids = SimpleNamespace(shape=(4, 2048), stride=lambda axis: 2056)
    args = (token_ids, 2048, object(), 2, *[object() for _ in range(8)])
    with ascend_suffix_ingestion(module):
        module._suffix_ingest_kernel[(2, 1)](*args, BLOCK=1024, num_warps=4)
    assert calls[0][1] == (2, 2)
    assert calls[0][2][1] == 2056
