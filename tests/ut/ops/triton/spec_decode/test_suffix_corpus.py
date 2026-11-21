# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace

from vllm_ascend.ops.triton.spec_decode import suffix_corpus as ops


def test_ingestion_launcher_uses_tensor_stride_and_document_count(monkeypatch, recording_kernel):
    calls = []
    monkeypatch.setattr(ops, "_suffix_ingest_kernel", recording_kernel("ingest", calls))
    token_ids = SimpleNamespace(shape=(4, 2048), stride=lambda axis: 2056)
    args = (token_ids, 2048, object(), 2, *[object() for _ in range(8)])
    ops.launch_suffix_ingest(*args, BLOCK=1024, num_warps=4)
    assert calls[0][1] == (2, 2)
    assert calls[0][2][1] == 2056
