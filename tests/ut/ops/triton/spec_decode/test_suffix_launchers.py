# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace

from vllm_ascend.ops.triton.spec_decode import suffix as ops

KernelCall = tuple[str, tuple[int, ...], tuple[object, ...], dict[str, object]]


class RecordingKernel:
    def __init__(self, name: str, calls: list[KernelCall]):
        self.name = name
        self.calls = calls

    def __getitem__(self, grid: tuple[int, ...]):
        def launch(*args: object, **kwargs: object) -> None:
            self.calls.append((self.name, grid, args, kwargs))

        return launch


def test_local_scan_uses_active_mapping_not_buffer_capacity(monkeypatch):
    calls: list[KernelCall] = []
    monkeypatch.setattr(ops, "_suffix_local_scan_kernel", RecordingKernel("scan", calls))
    token_ids = SimpleNamespace(shape=(128, 2048), stride=lambda axis: 2048)
    scratch = SimpleNamespace(shape=(128, 8), stride=lambda axis: 8)
    mapping = SimpleNamespace(shape=(7,))
    ops.launch_suffix_local_scan(token_ids, 2048, mapping, object(), object(), scratch, 8, 2049)
    assert calls[0][1] == (7, 8)


def test_finalize_uses_active_mapping_not_buffer_capacity(monkeypatch):
    calls: list[KernelCall] = []
    monkeypatch.setattr(ops, "_suffix_finalize_kernel", RecordingKernel("finalize", calls))
    token_ids = SimpleNamespace(shape=(128, 2048), stride=lambda axis: 2048)
    mapping = SimpleNamespace(shape=(7,))
    scratch = SimpleNamespace(shape=(128, 8), stride=lambda axis: 8)
    args = (token_ids, 2048, mapping, object(), object(), object(), scratch, 8, 8, 2049, *[object() for _ in range(9)])
    ops.launch_suffix_finalize(*args, K=1, MAX_DEPTH=3)
    assert calls[0][1] == (7,)


def test_observer_launcher_consumes_previous_drafts_for_active_requests(monkeypatch):
    calls: list[KernelCall] = []
    monkeypatch.setattr(ops, "_suffix_observe_kernel", RecordingKernel("observe", calls))
    monkeypatch.setattr(ops, "_suffix_clear_observed_kernel", RecordingKernel("clear", calls))
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


def test_ingestion_launcher_uses_tensor_stride_and_document_count(monkeypatch):
    calls: list[KernelCall] = []
    monkeypatch.setattr(ops, "_suffix_ingest_kernel", RecordingKernel("ingest", calls))
    token_ids = SimpleNamespace(shape=(4, 2048), stride=lambda axis: 2056)
    args = (token_ids, 2048, object(), 2, *[object() for _ in range(8)])
    ops.launch_suffix_ingest(*args, BLOCK=1024, num_warps=4)
    assert calls[0][1] == (2, 2)
    assert calls[0][2][1] == 2056
