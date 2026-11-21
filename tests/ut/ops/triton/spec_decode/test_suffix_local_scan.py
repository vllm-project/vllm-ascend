# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace

from vllm_ascend.ops.triton.spec_decode import suffix_local_scan as ops


def test_local_scan_uses_active_mapping_not_buffer_capacity(monkeypatch, recording_kernel):
    calls = []
    monkeypatch.setattr(ops, "_suffix_local_scan_kernel", recording_kernel("scan", calls))
    token_ids = SimpleNamespace(shape=(128, 2048), stride=lambda axis: 2048)
    scratch = SimpleNamespace(shape=(128, 8), stride=lambda axis: 8)
    mapping = SimpleNamespace(shape=(7,))
    ops.launch_suffix_local_scan(token_ids, 2048, mapping, object(), object(), scratch, 8, 2049)
    assert calls[0][1] == (7, 8)
