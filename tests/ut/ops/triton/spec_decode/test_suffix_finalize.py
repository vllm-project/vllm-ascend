# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace

from vllm_ascend.ops.triton.spec_decode import suffix_finalize as ops


def test_finalize_uses_active_mapping_not_buffer_capacity(monkeypatch, recording_kernel):
    calls = []
    monkeypatch.setattr(ops, "_suffix_finalize_kernel", recording_kernel("finalize", calls))
    token_ids = SimpleNamespace(shape=(128, 2048), stride=lambda axis: 2048)
    mapping = SimpleNamespace(shape=(7,))
    scratch = SimpleNamespace(shape=(128, 8), stride=lambda axis: 8)
    args = (token_ids, 2048, mapping, object(), object(), object(), scratch, 8, 8, 2049, *[object() for _ in range(9)])
    ops.launch_suffix_finalize(*args, K=1, MAX_DEPTH=3)
    assert calls[0][1] == (7,)
